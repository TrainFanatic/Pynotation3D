"""CPU spherical projection without geometric features.

Produces the channels that ``project_point_cloud`` turns into PNGs (true RGB,
intensity, Z, range and class) with the same rules as TLS-SphereMap's
``equirectangular_projection_multi``: each pixel takes the point with the
smallest range (the first in table order on ties), intensity and inverted Z
are min-max scaled to [0.01, 1], RGB is divided by 255, empty pixels get
TLS-SphereMap's fill colour and void class 9. The pixel loop is replaced by
one sort, so it takes seconds instead of hours and needs no GPU.
"""
import numpy as np

from . import _deps

CHANNELS = ["True-R", "True-G", "True-B", "intensity_raw", "z_raw", "range_raw", "seg_mask"]
EMPTY_RGB = (180, 170, 255)  # TLS-SphereMap's "Periwinkle Mist" fill for pixels with no point
VOID_CLASS = 9  # TLS-SphereMap's seg_mask value for pixels with no point
TABLE_COLUMNS = ["X", "Y", "Z", "Intensity", "r", "g", "b", "Classification", "azimuth", "elevation", "rangemeter"]


def scale_like_tls(values):
    """Min-max scale to [0.01, 1]; a constant column becomes 0.01."""
    lo, hi = values.min(), values.max()
    if hi == lo:
        return np.full(values.shape, 0.01)
    return 0.01 + 0.99 * (values - lo) / (hi - lo)


def nearest_point_per_pixel(x_pix, y_pix, rangemeter, width):
    """Return (point index, flat pixel index) of the smallest-range point in each occupied pixel."""
    pix = y_pix.astype(np.int64) * width + x_pix.astype(np.int64)
    order = np.lexsort((rangemeter, pix))  # stable, so ties keep table order like pandas idxmin
    pix_sorted = pix[order]
    first = np.r_[True, pix_sorted[1:] != pix_sorted[:-1]]
    return order[first], pix_sorted[first]


def project_points(points, canvas_size, angular_res):
    """Project a TLS-SphereMap point table to an (H, W, len(CHANNELS)) float32 cube.

    Args:
        points: DataFrame with at least ``TABLE_COLUMNS`` (as written by
            TLS-SphereMap's ``read_and_clean_pcd``).
        canvas_size: (height, width) in pixels.
        angular_res: (vertical, horizontal) degrees per pixel.
    """
    preprocess = _deps.tls_module("tools.preprocess_point_cloud")
    height, width = canvas_size
    x_pix, y_pix = preprocess.map_angle_to_pixel(
        points["azimuth"].to_numpy(), points["elevation"].to_numpy(), canvas_size, angular_res
    )
    winners, pixels = nearest_point_per_pixel(x_pix, y_pix, points["rangemeter"].to_numpy(), width)

    z = points["Z"].to_numpy(dtype=np.float64)
    values = {
        "True-R": points["r"].to_numpy(dtype=np.float64) / 255.0,
        "True-G": points["g"].to_numpy(dtype=np.float64) / 255.0,
        "True-B": points["b"].to_numpy(dtype=np.float64) / 255.0,
        "intensity_raw": scale_like_tls(points["Intensity"].to_numpy(dtype=np.float64)),
        "z_raw": scale_like_tls(-(z - z.min())),
        "range_raw": points["rangemeter"].to_numpy(dtype=np.float64),
        "seg_mask": points["Classification"].to_numpy(dtype=np.float64),
    }
    empty = {"True-R": EMPTY_RGB[0] / 255.0, "True-G": EMPTY_RGB[1] / 255.0, "True-B": EMPTY_RGB[2] / 255.0,
             "seg_mask": VOID_CLASS}

    cube = np.zeros((height * width, len(CHANNELS)), dtype=np.float32)
    for i, name in enumerate(CHANNELS):
        cube[:, i] = empty.get(name, 0.0)
        cube[pixels, i] = values[name][winners]
    return cube.reshape(height, width, len(CHANNELS))
