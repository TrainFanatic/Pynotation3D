"""FuelDeep3D classification of a LAS point cloud, without going through R.

Ported from ``predict_new_scans_default_model.py``: it reproduces the
inference of ``FuelDeep3D::predict()`` for the 4-channel (xyz + height above
ground) model bundled with the package.
"""
from pathlib import Path

import laspy
import numpy as np

from . import _deps

NUM_CLASSES = 3
BLOCK_SIZE = 6.0  # m, square XY tile
STRIDE = 1.0  # m between tile origins
SAMPLE_N = 4096  # points fed to the model per tile
REPEAT_PER_TILE = 4  # random draws per tile when it has >= SAMPLE_N points
MIN_PTS_TILE = 512  # tiles with fewer points are skipped (filled by nearest neighbour)
CELL_SIZE = 0.25  # m, grid cell for the ground estimate
QUANTILE = 0.05  # z quantile taken as ground in each cell


def _load_model(python_dir, model_path, device):
    import torch

    model_mod = _deps.fueldeep3d_module(python_dir, "model")
    model = model_mod.HeightAwarePointNetTiny(
        in_ch=4, num_classes=NUM_CLASSES, k=16, z_idx=2, hag_idx=3, use_height_prior=True
    ).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise ValueError(
            f"{model_path} is not a 4-channel (xyz + HAG), {NUM_CLASSES}-class FuelDeep3D model; "
            "only that architecture is supported"
        ) from e
    model.eval()
    return model


def predict_classes(xyz, model, *, python_dir, device, rng):
    """Return one FuelDeep3D class per point of ``xyz`` (N, 3) by tiled voting."""
    import torch
    from sklearn.neighbors import NearestNeighbors

    utils_mod = _deps.fueldeep3d_module(python_dir, "utils")
    xyz = np.asarray(xyz, dtype=np.float32)
    n = xyz.shape[0]

    ground_z = utils_mod.estimate_ground_z(xyz, cell=CELL_SIZE, quantile=QUANTILE)
    hag = (xyz[:, 2] - ground_z).astype(np.float32)

    xyz_local = xyz.copy()
    xyz_local[:, :2] -= xyz_local[:, :2].min(0, keepdims=True)
    xy = xyz_local[:, :2]
    maxs = xy.max(0)
    xs = np.arange(0.0, maxs[0] + 1e-6, STRIDE, dtype=np.float32)
    ys = np.arange(0.0, maxs[1] + 1e-6, STRIDE, dtype=np.float32)

    votes = np.zeros((n, NUM_CLASSES), dtype=np.int32)
    with torch.no_grad():
        for x0 in xs:
            # Select the x strip once, then tile it along y.
            strip = np.where((xy[:, 0] >= x0) & (xy[:, 0] < x0 + BLOCK_SIZE))[0]
            if strip.size < MIN_PTS_TILE:
                continue
            strip_y = xy[strip, 1]
            for y0 in ys:
                sel = strip[(strip_y >= y0) & (strip_y < y0 + BLOCK_SIZE)]
                m = sel.size
                if m < MIN_PTS_TILE:
                    continue
                for _ in range(REPEAT_PER_TILE if m >= SAMPLE_N else 1):
                    idx = sel[rng.choice(m, SAMPLE_N, replace=m < SAMPLE_N)]
                    pts = xyz_local[idx]
                    pts = pts - pts.mean(0, keepdims=True)
                    feat = np.hstack([pts, hag[idx][:, None]]).astype(np.float32)
                    pred = model(torch.from_numpy(feat[None]).to(device)).argmax(-1).squeeze(0).cpu().numpy()
                    np.add.at(votes, (idx, pred), 1)

    has_vote = votes.sum(1) > 0
    if not has_vote.any():
        raise ValueError(f"No tile had at least {MIN_PTS_TILE} points; the point cloud is too sparse to classify")
    y_pred = np.empty(n, dtype=np.uint8)
    y_pred[has_vote] = votes[has_vote].argmax(1)

    # Points in skipped tiles take the class of their nearest voted neighbour in XY.
    left = np.where(~has_vote)[0]
    if left.size:
        right = np.where(has_vote)[0]
        _, j = NearestNeighbors(n_neighbors=1).fit(xyz[right, :2]).kneighbors(xyz[left, :2])
        y_pred[left] = y_pred[right[j[:, 0]]]
    return y_pred


def run_fueldeep3d(las_path, out_las, *, fueldeep3d_dir=None, model_path=None, device=None, seed=0):
    """Classify ``las_path`` with FuelDeep3D and write a copy with the classes in ``classification``.

    Every point is kept, in its original order.

    Args:
        las_path: Input LAS/LAZ point cloud.
        out_las: Output LAS path.
        fueldeep3d_dir: FuelDeep3D's ``extdata/python`` directory (default: the
            ``PYNOTATION3D_FUELDEEP3D_DIR`` environment variable).
        model_path: Model checkpoint (default: FuelDeep3D's bundled ``best_model.pth``).
        device: Torch device (default: ``"cuda"`` if available, else ``"cpu"``).
        seed: Seed for the random point sampling within tiles.

    Returns:
        Path of the written LAS.
    """
    import torch

    python_dir = _deps.fueldeep3d_python_dir(fueldeep3d_dir)
    model_path = Path(model_path) if model_path else _deps.fueldeep3d_default_model(python_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    model = _load_model(python_dir, model_path, device)
    las = laspy.read(las_path)
    y_pred = predict_classes(
        np.c_[las.x, las.y, las.z], model,
        python_dir=python_dir, device=device, rng=np.random.default_rng(seed),
    )

    las.classification = y_pred
    out_las = Path(out_las)
    out_las.parent.mkdir(parents=True, exist_ok=True)
    las.write(out_las)
    return out_las
