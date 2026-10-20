"""Function 2: map an annotated 2D projection back onto the 3D points."""
import json
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
from PIL import Image

from . import _deps

VOID_CLASS = 0
_TABLE_COLUMNS = ["X", "Y", "Z", "Intensity", "r", "g", "b", "Classification", "azimuth", "elevation"]


@dataclass
class BackprojectionOutputs:
    las: Path
    mask_png: Path
    class_counts: dict


def load_projection_metadata(projection):
    """Load ``*_projection.json``, given the file or the ``project_point_cloud`` output directory."""
    path = Path(projection)
    if path.is_dir():
        matches = sorted(path.glob("*_projection.json"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one *_projection.json in {path}, found {len(matches)}")
        path = matches[0]
    return json.loads(path.read_text())


def load_points_table(meta):
    """Load the projected points: TLS-SphereMap's ``*_geom_feat.txt`` or the fast method's ``*_points.npz``."""
    outputs = meta["outputs"]
    path = Path(outputs.get("points_table") or outputs["geom_feat"])
    if path.suffix == ".npz":
        with np.load(path) as data:
            return pd.DataFrame({c: data[c] for c in _TABLE_COLUMNS})
    return pd.read_csv(path, usecols=_TABLE_COLUMNS)


def resolve_label_map(label_map):
    """Return ``{(r, g, b): class_id}`` from a TLS-SphereMap dataset name or a dict."""
    if isinstance(label_map, str):
        label_map_utils = _deps.tls_module("data_annotation.seg_map_tools.label_map_utils")
        color_to_index, _ = label_map_utils.load_dataset_label_map(label_map)
        return color_to_index
    return {tuple(int(c) for c in color): int(class_id) for color, class_id in label_map.items()}


def annotation_to_mask(image, color_to_index, tolerance=3):
    """Convert an annotated PNG to a class-ID mask.

    Colours match within ``tolerance`` per channel, as TLS-SphereMap's
    ``convert_color_to_mask``. Unmatched pixels are void (0), and so are fully
    transparent ones whatever their RGB: transparent means unclassified, but
    erased pixels often keep their old colour (TLS-SphereMap ignores alpha and
    would count them as labelled).
    """
    rgba = np.asarray(image.convert("RGBA"))
    rgb = rgba[:, :, :3].astype(np.int16)
    mask = np.full(rgb.shape[:2], VOID_CLASS, dtype=np.uint8)
    for color, class_id in color_to_index.items():
        mask[np.all(np.abs(rgb - np.array(color)) <= tolerance, axis=-1)] = class_id
    mask[rgba[:, :, 3] == 0] = VOID_CLASS
    return mask


def write_labelled_las(table, classes, out_las, xy_offset=(0.0, 0.0)):
    """Write points with annotated ``classification``, real RGB/intensity and the FuelDeep3D class.

    ``xy_offset`` is added back to X/Y, undoing ``project_point_cloud(recenter=True)``.
    """
    xyz = table[["X", "Y", "Z"]].to_numpy(dtype=np.float64)
    xyz[:, :2] += np.asarray(xy_offset, dtype=np.float64)
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = xyz.min(axis=0)
    header.add_extra_dim(laspy.ExtraBytesParams(name="fueldeep3d_class", type=np.uint8))

    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    # The table holds intensity rescaled to [0, 1] and the LAS's raw 16-bit colours.
    las.intensity = np.round(table["Intensity"].to_numpy() * 65535).clip(0, 65535).astype(np.uint16)
    las.red, las.green, las.blue = (table[c].to_numpy().clip(0, 65535).astype(np.uint16) for c in ("r", "g", "b"))
    las.classification = classes
    las.fueldeep3d_class = table["Classification"].to_numpy().astype(np.uint8)
    out_las = Path(out_las)
    out_las.parent.mkdir(parents=True, exist_ok=True)
    las.write(out_las)
    return out_las


def backproject_annotation(annotation_png, projection, label_map, out_las=None, *, tolerance=3, drop_void=False):
    """Assign each 3D point the class annotated at its pixel and write a LAS.

    Args:
        annotation_png: Annotated projection, the same size as the projection images.
        projection: ``*_projection.json`` written by ``project_point_cloud``, or its output directory.
        label_map: Dataset name in TLS-SphereMap's ``label_maps.json`` (e.g.
            ``"LAS-all-cropped-79"``) or a ``{(r, g, b): class_id}`` dict.
        out_las: Output LAS (default: ``<out_dir>/pcd/<annotation stem>_annotated.las``).
        tolerance: Per-channel colour tolerance when matching the label map.
        drop_void: Leave out points whose pixel is void (class 0).

    Returns:
        BackprojectionOutputs with the LAS path, the class-ID mask PNG and
        the number of points per class.
    """
    meta = load_projection_metadata(projection)
    canvas = tuple(meta["canvas_size"])
    out_dir = Path(meta["outputs"]["out_dir"])
    annotation_png = Path(annotation_png)

    image = Image.open(annotation_png)
    if image.size != (canvas[1], canvas[0]):
        raise ValueError(
            f"{annotation_png} is {image.size[0]}x{image.size[1]} (WxH) but the projection is "
            f"{canvas[1]}x{canvas[0]}; annotate the full, uncropped projection image"
        )
    mask = annotation_to_mask(image, resolve_label_map(label_map), tolerance)
    mask_png = out_dir / "img" / f"{annotation_png.stem}_mask.png"
    Image.fromarray(mask).save(mask_png)

    # Same table and pixel mapping as the forward projection, so every point lands on its pixel.
    preprocess = _deps.tls_module("tools.preprocess_point_cloud")
    table = load_points_table(meta)
    x_pix, y_pix = preprocess.map_angle_to_pixel(
        table["azimuth"].to_numpy(), table["elevation"].to_numpy(), canvas, tuple(meta["angular_res"])
    )
    classes = mask[y_pix, x_pix]

    if drop_void:
        keep = classes != VOID_CLASS
        table, classes = table[keep], classes[keep]

    out_las = Path(out_las) if out_las else out_dir / "pcd" / f"{annotation_png.stem}_annotated.las"
    write_labelled_las(table, classes, out_las, meta.get("xy_offset", (0.0, 0.0)))
    ids, counts = np.unique(classes, return_counts=True)
    return BackprojectionOutputs(
        las=out_las, mask_png=mask_png, class_counts={int(i): int(c) for i, c in zip(ids, counts)}
    )
