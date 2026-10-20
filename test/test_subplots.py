import json

import laspy
import numpy as np
import pytest

from pynotation3d.subplots import split_into_subplots, write_subplots

# UTM-like coordinates, to check the shift survives large header offsets.
X0, Y0 = 500_000.0, 4_200_000.0


def make_las(n=2000, extent=25.0, seed=0):
    rng = np.random.default_rng(seed)
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([X0, Y0, 0.0])
    header.add_extra_dim(laspy.ExtraBytesParams(name="label", type=np.uint8))
    las = laspy.LasData(header)
    las.x = X0 + rng.uniform(0, extent, n)
    las.y = Y0 + rng.uniform(0, extent, n)
    las.z = rng.uniform(-2, 30, n)
    las.red = rng.integers(0, 65535, n, dtype=np.uint16)
    las.intensity = rng.integers(0, 65535, n, dtype=np.uint16)
    las.label = rng.integers(0, 5, n, dtype=np.uint8)
    return las


def test_every_point_in_exactly_one_tile():
    las = make_las()
    subplots = split_into_subplots(las, 10.0)
    assert len(subplots) == 9  # 25 m extent -> 3 x 3 tiles of 10 m
    assert sum(len(sp.las.points) for sp in subplots) == len(las.points)


def test_tiles_are_centred_and_shift_is_recorded():
    las = make_las()
    for sp in split_into_subplots(las, 10.0):
        x, y = np.asarray(sp.las.x), np.asarray(sp.las.y)
        assert x.min() >= -5 - 1e-6 and x.max() < 5 + 1e-6
        assert y.min() >= -5 - 1e-6 and y.max() < 5 + 1e-6
        xmin, xmax, ymin, ymax = sp.bounds
        assert sp.center == ((xmin + xmax) / 2, (ymin + ymax) / 2)
        # Adding the centre back gives the original coordinates.
        orig = (np.asarray(las.x) >= xmin) & (np.asarray(las.x) < xmax) & \
               (np.asarray(las.y) >= ymin) & (np.asarray(las.y) < ymax)
        np.testing.assert_allclose(np.sort(x + sp.center[0]), np.sort(np.asarray(las.x)[orig]), atol=1e-3)


def test_other_dimensions_and_height_unchanged():
    las = make_las()
    sp = split_into_subplots(las, 10.0)[0]
    xmin, xmax, ymin, ymax = sp.bounds
    orig = (np.asarray(las.x) >= xmin) & (np.asarray(las.x) < xmax) & \
           (np.asarray(las.y) >= ymin) & (np.asarray(las.y) < ymax)
    np.testing.assert_allclose(sp.las.z, np.asarray(las.z)[orig], atol=1e-3)
    assert np.array_equal(sp.las.red, np.asarray(las.red)[orig])
    assert np.array_equal(sp.las.label, np.asarray(las.label)[orig])


def test_rectangular_size_origin_and_min_points():
    las = make_las()
    subplots = split_into_subplots(las, (12.5, 25.0), origin=(X0, Y0))
    assert sorted(sp.index for sp in subplots) == [(0, 0), (1, 0)]
    assert split_into_subplots(las, 10.0, min_points=10**6) == []
    with pytest.raises(ValueError):
        split_into_subplots(las, 0)


def test_write_and_read_back(tmp_path):
    las = make_las()
    paths = write_subplots(split_into_subplots(las, 10.0), tmp_path, "plot")
    manifest = json.loads((tmp_path / "plot_subplots.json").read_text())
    assert [m["file"] for m in manifest] == [p.name for p in paths]
    total = 0
    for p, m in zip(paths, manifest):
        back = laspy.read(p)
        total += len(back.points)
        assert len(back.points) == m["n_points"]
        assert np.abs(np.asarray(back.x)).max() <= 5 + 1e-3
        assert "label" in back.point_format.dimension_names
    assert total == len(las.points)
