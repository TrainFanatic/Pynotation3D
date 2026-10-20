"""Build the 2D projection PNGs from a TLS-SphereMap image cube."""
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

# FuelDeep3D's three classes, as coloured in the earlier classification PNGs.
CLASS_COLORS = {0: (113, 170, 52), 1: (92, 64, 51), 2: (0, 0, 255)}


def load_image_cube(cube_path, meta_path):
    """Return ``(cube, channel_names)`` for an ``*_image_cube.npy`` and its meta yaml."""
    cube = np.load(cube_path, mmap_mode="r")
    with open(meta_path) as f:
        channel_names = yaml.safe_load(f)["channel_names"]
    return cube, channel_names


def channel(cube, channel_names, name):
    return np.asarray(cube[:, :, channel_names.index(name)], dtype=np.float64)


def valid_mask(cube, channel_names):
    """Pixels that received a point (TLS-SphereMap leaves range at 0 elsewhere)."""
    rng = channel(cube, channel_names, "range_raw")
    return np.isfinite(rng) & (rng > 0)


def minmax_to_uint8(values, valid):
    """Stretch ``values`` to 0-255 using the min/max of valid pixels; invalid pixels are 0."""
    out = np.zeros(values.shape, dtype=np.uint8)
    if not valid.any():
        return out
    lo, hi = values[valid].min(), values[valid].max()
    if hi > lo:
        out[valid] = np.round((values[valid] - lo) / (hi - lo) * 255).astype(np.uint8)
    return out


def composite(cube, channel_names, names, valid):
    """Stack three channels as an RGB uint8 image, each stretched separately."""
    return np.stack([minmax_to_uint8(channel(cube, channel_names, n), valid) for n in names], axis=-1)


def rgb_image(cube, channel_names, valid):
    return composite(cube, channel_names, ("True-R", "True-G", "True-B"), valid)


def irz_image(cube, channel_names, valid):
    """Intensity -> R, range -> G, Z -> B, from the raw (not histogram-equalized) channels."""
    return composite(cube, channel_names, ("intensity_raw", "range_raw", "z_raw"), valid)


def exists_image(valid):
    return np.where(valid, 255, 0).astype(np.uint8)


def classification_image(cube, channel_names, valid, colors=CLASS_COLORS):
    """Colour the ``seg_mask`` channel; void and unknown classes stay black."""
    seg = channel(cube, channel_names, "seg_mask")
    out = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        out[valid & (seg == class_id)] = color
    return out


def write_projection_pngs(cube_path, meta_path, img_dir, stem):
    """Write ``<stem>_{rgb,irz,exists,classification}.png`` into ``img_dir``; return their paths."""
    cube, names = load_image_cube(cube_path, meta_path)
    valid = valid_mask(cube, names)
    images = {
        "rgb": rgb_image(cube, names, valid),
        "irz": irz_image(cube, names, valid),
        "exists": exists_image(valid),
        "classification": classification_image(cube, names, valid),
    }
    paths = {}
    for kind, image in images.items():
        paths[kind] = Path(img_dir) / f"{stem}_{kind}.png"
        Image.fromarray(image).save(paths[kind])
    return paths
