"""Split a large plot into square subplots, each centred on the origin."""
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np


@dataclass
class Subplot:
    las: laspy.LasData  # points of this subplot, X/Y shifted so the tile centre is (0, 0)
    index: tuple  # (column, row) of the tile in the grid, from the grid origin
    center: tuple  # (x, y) tile centre in the input's coordinates; subtracted from X/Y
    bounds: tuple  # (xmin, xmax, ymin, ymax) of the tile in the input's coordinates


def _centred_copy(las, mask, cx, cy):
    """Copy the masked points into a new LasData with X/Y shifted by (-cx, -cy)."""
    header = copy.deepcopy(las.header)
    # New X/Y offsets so the shifted coordinates fit the scaled integers whatever the input's offsets.
    header.offsets = np.array([0.0, 0.0, las.header.offsets[2]])
    x, y, z = np.asarray(las.x[mask]) - cx, np.asarray(las.y[mask]) - cy, np.asarray(las.z[mask])
    sub = laspy.LasData(header, points=las.points[mask].copy())
    sub.x, sub.y, sub.z = x, y, z
    sub.update_header()
    return sub


def split_into_subplots(las, size=10.0, *, origin=None, min_points=1):
    """Split a point cloud into a grid of square subplots, each centred on (0, 0).

    The grid covers the X/Y extent of the cloud; height is unlimited, and Z is
    left unchanged. Each point goes to exactly one tile (tiles are
    half-open, [xmin, xmax) x [ymin, ymax)). Edge tiles may be partly empty.

    Args:
        las: LAS/LAZ path or ``laspy.LasData``.
        size: Tile side in metres, or ``(size_x, size_y)``.
        origin: (x, y) corner of the grid (default: the cloud's X/Y minimum).
            Use it to align the grid to a plot layout or the scanner position.
        min_points: Tiles with fewer points are left out.

    Returns:
        List of ``Subplot``, ordered by row then column.
    """
    if not isinstance(las, laspy.LasData):
        las = laspy.read(las)
    size_x, size_y = (size, size) if np.isscalar(size) else size
    if size_x <= 0 or size_y <= 0:
        raise ValueError(f"size must be positive, got {size}")
    x, y = np.asarray(las.x), np.asarray(las.y)
    x0, y0 = (x.min(), y.min()) if origin is None else origin

    col = np.floor((x - x0) / size_x).astype(np.int64)
    row = np.floor((y - y0) / size_y).astype(np.int64)
    keys, inverse = np.unique(np.c_[row, col], axis=0, return_inverse=True)
    inverse = inverse.ravel()

    subplots = []
    for k, (r, c) in enumerate(keys):
        mask = inverse == k
        if mask.sum() < min_points:
            continue
        xmin, ymin = x0 + c * size_x, y0 + r * size_y
        cx, cy = xmin + size_x / 2, ymin + size_y / 2
        subplots.append(Subplot(
            las=_centred_copy(las, mask, cx, cy),
            index=(int(c), int(r)),
            center=(float(cx), float(cy)),
            bounds=(float(xmin), float(xmin + size_x), float(ymin), float(ymin + size_y)),
        ))
    return subplots


def write_subplots(subplots, out_dir, stem):
    """Write each subplot to ``out_dir/<stem>_c<col>_r<row>.las`` plus ``<stem>_subplots.json``.

    The JSON lists each file with its tile index, bounds, point count and the
    centre that was subtracted from X/Y (add it back to return to the input's
    coordinates). Returns the list of LAS paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, manifest = [], []
    for sp in subplots:
        path = out_dir / f"{stem}_c{sp.index[0]:03d}_r{sp.index[1]:03d}.las"
        sp.las.write(path)
        paths.append(path)
        manifest.append({
            "file": path.name,
            "index": list(sp.index),
            "center": list(sp.center),
            "bounds": list(sp.bounds),
            "n_points": len(sp.las.points),
        })
    (out_dir / f"{stem}_subplots.json").write_text(json.dumps(manifest, indent=2))
    return paths
