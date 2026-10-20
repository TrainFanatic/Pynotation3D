import numpy as np
import pandas as pd

from pynotation3d import fast_projection as fp

CANVAS = (3, 4)  # (height, width)
RES = (1.0, 1.0)


def make_points():
    """Points 0-2 share pixel (2, 0); point 1 is nearest, point 2 ties with it but comes later."""
    return pd.DataFrame({
        "X": [0.0] * 4, "Y": [0.0] * 4, "Z": [1.0, 2.0, 3.0, 5.0],
        "Intensity": [0.0, 0.2, 0.4, 1.0],
        "r": [255.0, 510.0, 765.0, 1020.0], "g": [0.0] * 4, "b": [0.0] * 4,
        "Classification": [0, 1, 2, 2],
        # el.min() = 0: y = 3 - int(el), clipped -> rows 2, 2, 2, 1.
        "azimuth": [0.5, 0.5, 0.5, 2.5], "elevation": [0.0, 0.0, 0.0, 2.0],
        "rangemeter": [3.0, 1.0, 1.0, 2.0],
    })


def test_nearest_point_wins_and_ties_keep_table_order():
    winners, pixels = fp.nearest_point_per_pixel(
        np.array([0, 0, 0, 2]), np.array([2, 2, 2, 1]), np.array([3.0, 1.0, 1.0, 2.0]), width=4
    )
    assert dict(zip(pixels.tolist(), winners.tolist())) == {8: 1, 6: 3}


def test_project_points_channels():
    cube = fp.project_points(make_points(), CANVAS, RES)
    ch = {name: cube[..., i] for i, name in enumerate(fp.CHANNELS)}
    assert cube.shape == (3, 4, len(fp.CHANNELS))
    assert ch["range_raw"][2, 0] == 1.0 and ch["range_raw"][1, 2] == 2.0
    assert ch["seg_mask"][2, 0] == 1 and ch["seg_mask"][1, 2] == 2
    assert np.isclose(ch["True-R"][2, 0], 2.0)  # 510 / 255
    # Intensity and inverted Z scaled to [0.01, 1] over all points, like TLS-SphereMap.
    assert np.isclose(ch["intensity_raw"][2, 0], 0.01 + 0.99 * 0.2)
    assert np.isclose(ch["z_raw"][2, 0], 0.01 + 0.99 * 0.75)  # -(2-1) on a 0..-4 scale
    assert np.isclose(ch["z_raw"][1, 2], 0.01)  # the highest point


def test_empty_pixels_match_tls_fill():
    cube = fp.project_points(make_points(), CANVAS, RES)
    ch = {name: cube[..., i] for i, name in enumerate(fp.CHANNELS)}
    assert ch["range_raw"][0, 0] == 0
    assert ch["seg_mask"][0, 0] == fp.VOID_CLASS
    assert np.isclose(ch["True-B"][0, 0], fp.EMPTY_RGB[2] / 255)
