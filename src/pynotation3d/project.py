"""Function 1: project a 3D point cloud to 2D spherical images."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from . import _deps
from .classify import run_fueldeep3d
from .export import CLASS_COLORS, write_projection_pngs

DATASET = "SEMANTIC3D"  # TLS-SphereMap's reader for LAS files with RGB + Classification
TABLE_KEY = "geom_feat"

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
    geom_feat: Path
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

    Same as TLS-SphereMap's ``tools/recenter_geom_feat.py``, applied before the
    features are computed. Returns the (x, y) shift that was subtracted.
    """
    cx = float(points["X"].min() + points["X"].max()) / 2
    cy = float(points["Y"].min() + points["Y"].max()) / 2
    points["X"] -= cx
    points["Y"] -= cy
    points.drop(columns=["azimuth", "elevation", "zenith", "rangemeter"], inplace=True)
    preprocess.add_angle_range_to_df(points)
    return cx, cy


def project_point_cloud(
    las_path,
    out_dir,
    *,
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

    Runs FuelDeep3D, then TLS-SphereMap's geometric features (needs a CUDA GPU)
    and spherical projection, then writes into ``out_dir/img``:
    ``<stem>_rgb.png``, ``<stem>_irz.png``, ``<stem>_exists.png`` and
    ``<stem>_classification.png``.

    The per-point table ``out_dir/pcd/<stem>_geom_feat.txt`` and
    ``out_dir/<stem>_projection.json`` are kept: ``backproject_annotation``
    needs both to map annotated pixels back to these exact points.

    Args:
        las_path: Input LAS point cloud. Pixels are placed by angle around the
            coordinate origin, which should be the scanner position (see ``recenter``).
        out_dir: Output directory (created if needed).
        fueldeep3d_dir: FuelDeep3D's ``extdata/python`` directory (default: the
            ``PYNOTATION3D_FUELDEEP3D_DIR`` environment variable).
        model_path: FuelDeep3D model checkpoint (default: the bundled model).
        h_res, v_res: Horizontal and vertical angular resolution, in degrees per pixel.
        h_fov, v_fov: Horizontal and vertical field of view, in degrees.
        cut_percent: Fraction of extreme range/intensity points removed when ``clean_pc``.
        clean_pc: Remove outlying points before projecting.
        recenter: Project around the X/Y midpoint of the cloud instead of the
            origin, for crops where the scanner is off-centre. The feature table
            is then in shifted coordinates; ``backproject_annotation`` shifts
            the points back.
        geom_params: Overrides for ``DEFAULT_GEOM_PARAMS``.
        overwrite: Recompute steps whose outputs already exist. By default,
            existing FuelDeep3D, feature-table and image-cube outputs are reused.

    Returns:
        ProjectionOutputs with the paths of everything written.
    """
    las_path = Path(las_path)
    out_dir = Path(out_dir)
    stem = las_path.stem
    pcd_dir, img_dir = out_dir / "pcd", out_dir / "img"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    size = canvas_size(h_res, v_res, h_fov, v_fov)
    angular_res = (v_res, h_res)
    params = {**DEFAULT_GEOM_PARAMS, **(geom_params or {})}

    classified_las = pcd_dir / f"{stem}_fueldeep3d.las"
    if overwrite or not classified_las.exists():
        run_fueldeep3d(las_path, classified_las, fueldeep3d_dir=fueldeep3d_dir, model_path=model_path)

    geom_feat = pcd_dir / f"{stem}_{TABLE_KEY}.txt"
    metadata_path = out_dir / f"{stem}_projection.json"
    if overwrite or not geom_feat.exists():
        preprocess = _deps.tls_module("tools.preprocess_point_cloud")
        calc_geom_feature = _deps.tls_module("processing.calc_geom_feature")
        points = preprocess.read_and_clean_pcd(
            classified_las, cut_percent=cut_percent, clean_pc=clean_pc, dataset_name=DATASET, flip_mangrove=False
        )
        xy_offset = recenter_points(points, preprocess) if recenter else (0.0, 0.0)
        calc_geom_feature.generate_geom_feat_from_pcd(params, points, out_dir, stem)
    elif metadata_path.exists():
        # Reusing the table: keep the shift it was made with.
        xy_offset = tuple(json.loads(metadata_path.read_text())["xy_offset"])
    else:
        raise FileNotFoundError(
            f"{geom_feat} exists but {metadata_path} does not, so its recentring is unknown; "
            "rerun with overwrite=True"
        )

    image_cube = img_dir / f"{stem}_image_cube.npy"
    image_cube_meta = img_dir / f"{stem}_image_cube_meta.yaml"
    if overwrite or not image_cube.exists():
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
            input_path=geom_feat,
        )

    pngs = write_projection_pngs(image_cube, image_cube_meta, img_dir, stem)

    outputs = ProjectionOutputs(
        out_dir=out_dir,
        classified_las=classified_las,
        geom_feat=geom_feat,
        image_cube=image_cube,
        image_cube_meta=image_cube_meta,
        rgb_png=pngs["rgb"],
        irz_png=pngs["irz"],
        exists_png=pngs["exists"],
        classification_png=pngs["classification"],
        metadata=metadata_path,
    )
    metadata = {
        "stem": stem,
        "input_las": str(las_path.resolve()),
        "model_path": str(model_path) if model_path else "FuelDeep3D bundled best_model.pth",
        "canvas_size": list(size),
        "angular_res": list(angular_res),
        "h_fov": list(h_fov),
        "v_fov": list(v_fov),
        "cut_percent": cut_percent,
        "clean_pc": clean_pc,
        "xy_offset": list(xy_offset),  # subtracted from X/Y in the feature table
        "geom_params": params,
        "class_colors": {str(k): list(v) for k, v in CLASS_COLORS.items()},
        "outputs": {k: str(v) for k, v in asdict(outputs).items()},
    }
    outputs.metadata.write_text(json.dumps(metadata, indent=2))
    return outputs
