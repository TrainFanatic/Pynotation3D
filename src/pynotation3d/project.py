"""Function 1: project a 3D point cloud to 2D spherical images."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from . import _deps
from . import fast_projection
from .classify import run_fueldeep3d
from .export import CLASS_COLORS, write_projection_pngs

DATASET = "SEMANTIC3D"  # TLS-SphereMap's reader for LAS files with RGB + Classification
TABLE_KEY = "geom_feat"
METHODS = ("full", "fast")

# Geometric-feature parameters used for the LAS-all scans.
DEFAULT_GEOM_PARAMS = {
    "base_radius": 0.06,
    "max_neighbors": 20,
    "batching_method": "xyz",
    "pts_num_per_batch": 12000,
    "buffer_pts": 30,
    "export": True,
    "out_signature_str": TABLE_KEY,
}


@dataclass
class ProjectionOutputs:
    out_dir: Path
    classified_las: Path
    points_table: Path  # projected points, read back by backproject_annotation
    image_cube: Path
    image_cube_meta: Path
    rgb_png: Path
    irz_png: Path
    exists_png: Path
    classification_png: Path
    metadata: Path


def canvas_size(h_res, v_res, h_fov, v_fov):
    """Image (height, width) in pixels, as TLS-SphereMap computes it."""
    return int((v_fov[1] - v_fov[0]) / v_res), int((h_fov[1] - h_fov[0]) / h_res)


def recenter_points(points, preprocess):
    """Shift X/Y so their midpoint is at the origin and recompute the angle columns.

    Same as TLS-SphereMap's ``tools/recenter_geom_feat.py``, applied before
    projecting. Returns the (x, y) shift that was subtracted.
    """
    cx = float(points["X"].min() + points["X"].max()) / 2
    cy = float(points["Y"].min() + points["Y"].max()) / 2
    points["X"] -= cx
    points["Y"] -= cy
    points.drop(columns=["azimuth", "elevation", "zenith", "rangemeter"], inplace=True)
    preprocess.add_angle_range_to_df(points)
    return cx, cy


def _read_points(classified_las, cut_percent, clean_pc, recenter):
    preprocess = _deps.tls_module("tools.preprocess_point_cloud")
    points = preprocess.read_and_clean_pcd(
        classified_las, cut_percent=cut_percent, clean_pc=clean_pc, dataset_name=DATASET, flip_mangrove=False
    )
    xy_offset = recenter_points(points, preprocess) if recenter else (0.0, 0.0)
    return points, xy_offset


def _project_full(classified_las, out_dir, stem, size, angular_res, h_fov, v_fov, params,
                  cut_percent, clean_pc, recenter, overwrite, metadata_path):
    """TLS-SphereMap: GPU geometric features, then its spherical projection (24-channel cube)."""
    table = out_dir / "pcd" / f"{stem}_{TABLE_KEY}.txt"
    if overwrite or not table.exists():
        calc_geom_feature = _deps.tls_module("processing.calc_geom_feature")
        points, xy_offset = _read_points(classified_las, cut_percent, clean_pc, recenter)
        calc_geom_feature.generate_geom_feat_from_pcd(params, points, out_dir, stem)
    elif metadata_path.exists():
        # Reusing the table: keep the shift it was made with.
        xy_offset = tuple(json.loads(metadata_path.read_text())["xy_offset"])
    else:
        raise FileNotFoundError(
            f"{table} exists but {metadata_path} does not, so its recentring is unknown; rerun with overwrite=True"
        )

    cube = out_dir / "img" / f"{stem}_image_cube.npy"
    if overwrite or not cube.exists():
        spherical_projection = _deps.tls_module("processing.spherical_projection")
        spherical_projection.generate_2D_projection_images(
            output_dir=out_dir,
            canvas_size=size,
            angular_res=angular_res,
            dataset_name=DATASET,
            out_key_str=TABLE_KEY,
            input_file_stem=stem,
            color_map={str(k): tuple(c / 255 for c in rgb) for k, rgb in CLASS_COLORS.items()},
            saveflag=False,
            visualize=False,
            extra_maps=False,
            v_fov=tuple(v_fov),
            h_fov=tuple(h_fov),
            input_path=table,
        )
    return table, cube, xy_offset


def _project_fast(classified_las, out_dir, stem, size, angular_res, cut_percent, clean_pc, recenter):
    """CPU projection of only the channels the PNGs need (see ``fast_projection``)."""
    points, xy_offset = _read_points(classified_las, cut_percent, clean_pc, recenter)
    cube_data = fast_projection.project_points(points, size, angular_res)

    table = out_dir / "pcd" / f"{stem}_points.npz"
    np.savez(table, **{c: points[c].to_numpy() for c in fast_projection.TABLE_COLUMNS})
    cube = out_dir / "img" / f"{stem}_image_cube.npy"
    np.save(cube, cube_data)
    meta = {"channel_names": fast_projection.CHANNELS, "shape": list(cube_data.shape), "dtype": str(cube_data.dtype)}
    cube.with_name(f"{stem}_image_cube_meta.yaml").write_text(yaml.safe_dump(meta))
    return table, cube, xy_offset


def project_point_cloud(
    las_path,
    out_dir,
    *,
    method="full",
    fueldeep3d_dir=None,
    model_path=None,
    h_res=0.2,
    v_res=0.2,
    h_fov=(0, 360),
    v_fov=(0, 300),
    cut_percent=0.005,
    clean_pc=True,
    recenter=False,
    geom_params=None,
    overwrite=False,
):
    """Classify a point cloud with FuelDeep3D and project it to four 2D images.

    Writes into ``out_dir/img``: ``<stem>_rgb.png``, ``<stem>_irz.png``,
    ``<stem>_exists.png`` and ``<stem>_classification.png``. Keep the points
    table in ``out_dir/pcd`` and ``out_dir/<stem>_projection.json``:
    ``backproject_annotation`` needs both to map annotated pixels back to
    these exact points.

    Args:
        las_path: Input LAS point cloud. Pixels are placed by angle around the
            coordinate origin, which should be the scanner position (see ``recenter``).
        out_dir: Output directory (created if needed).
        method: ``"full"`` runs TLS-SphereMap's GPU geometric features and
            projection, writing its 24-channel image cube and
            ``pcd/<stem>_geom_feat.txt``. ``"fast"`` projects on CPU only the
            channels the four PNGs need, with the same pixel rules, writing
            ``pcd/<stem>_points.npz``; it takes minutes and needs no GPU.
        fueldeep3d_dir: FuelDeep3D's ``extdata/python`` directory (default: the
            ``PYNOTATION3D_FUELDEEP3D_DIR`` environment variable).
        model_path: FuelDeep3D model checkpoint (default: the bundled model).
        h_res, v_res: Horizontal and vertical angular resolution, in degrees per pixel.
        h_fov, v_fov: Horizontal and vertical field of view, in degrees.
        cut_percent: Fraction of extreme range/intensity points removed when ``clean_pc``.
        clean_pc: Remove outlying points before projecting.
        recenter: Project around the X/Y midpoint of the cloud instead of the
            origin, for crops where the scanner is off-centre. The points table
            is then in shifted coordinates; ``backproject_annotation`` shifts
            the points back.
        geom_params: Overrides for ``DEFAULT_GEOM_PARAMS`` (``"full"`` only).
        overwrite: Recompute steps whose outputs already exist. By default an
            existing FuelDeep3D output is reused, and with ``"full"`` also the
            feature table and image cube. ``"fast"`` always reprojects.

    Returns:
        ProjectionOutputs with the paths of everything written.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")
    las_path = Path(las_path)
    out_dir = Path(out_dir)
    stem = las_path.stem
    (out_dir / "pcd").mkdir(parents=True, exist_ok=True)
    (out_dir / "img").mkdir(parents=True, exist_ok=True)

    size = canvas_size(h_res, v_res, h_fov, v_fov)
    angular_res = (v_res, h_res)
    params = {**DEFAULT_GEOM_PARAMS, **(geom_params or {})}
    metadata_path = out_dir / f"{stem}_projection.json"

    classified_las = out_dir / "pcd" / f"{stem}_fueldeep3d.las"
    if overwrite or not classified_las.exists():
        run_fueldeep3d(las_path, classified_las, fueldeep3d_dir=fueldeep3d_dir, model_path=model_path)

    if method == "full":
        table, cube, xy_offset = _project_full(
            classified_las, out_dir, stem, size, angular_res, h_fov, v_fov, params,
            cut_percent, clean_pc, recenter, overwrite, metadata_path,
        )
    else:
        table, cube, xy_offset = _project_fast(
            classified_las, out_dir, stem, size, angular_res, cut_percent, clean_pc, recenter
        )

    cube_meta = cube.with_name(f"{stem}_image_cube_meta.yaml")
    pngs = write_projection_pngs(cube, cube_meta, out_dir / "img", stem)

    outputs = ProjectionOutputs(
        out_dir=out_dir,
        classified_las=classified_las,
        points_table=table,
        image_cube=cube,
        image_cube_meta=cube_meta,
        rgb_png=pngs["rgb"],
        irz_png=pngs["irz"],
        exists_png=pngs["exists"],
        classification_png=pngs["classification"],
        metadata=metadata_path,
    )
    metadata = {
        "stem": stem,
        "method": method,
        "input_las": str(las_path.resolve()),
        "model_path": str(model_path) if model_path else "FuelDeep3D bundled best_model.pth",
        "canvas_size": list(size),
        "angular_res": list(angular_res),
        "h_fov": list(h_fov),
        "v_fov": list(v_fov),
        "cut_percent": cut_percent,
        "clean_pc": clean_pc,
        "xy_offset": list(xy_offset),  # subtracted from X/Y in the points table
        "geom_params": params if method == "full" else None,
        "class_colors": {str(k): list(v) for k, v in CLASS_COLORS.items()},
        "outputs": {k: str(v) for k, v in asdict(outputs).items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return outputs
