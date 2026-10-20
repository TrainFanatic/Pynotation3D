import json

import laspy
import numpy as np
import pandas as pd
from PIL import Image

from pynotation3d import backproject

CANVAS = (4, 6)  # (height, width)
RES = (1.0, 1.0)  # (vertical, horizontal) degrees per pixel
RED, BLUE = (255, 0, 0), (0, 0, 255)
LABELS = {RED: 1, BLUE: 2}


def make_projection(tmp_path):
    """Three points: two land on red pixels, one on a transparent pixel."""
    (tmp_path / "pcd").mkdir()
    (tmp_path / "img").mkdir()
    table = pd.DataFrame({
        "X": [1.0, 2.0, 3.0], "Y": [0.5, 0.25, 0.125], "Z": [0.1, 0.2, 0.3],
        "Intensity": [0.0, 0.5, 1.0],
        "r": [1000.0, 2000.0, 3000.0], "g": [10.0, 20.0, 30.0], "b": [7.0, 8.0, 9.0],
        "Classification": [0, 1, 2],
        # With el.min() = 0: y = 4 - int(el) -> rows 3 (clipped from 4), 3, 2.
        "azimuth": [0.5, 1.5, 4.5], "elevation": [0.0, 1.0, 2.0],
        "zenith": [90.0, 89.0, 88.0], "rangemeter": [1.0, 2.0, 3.0],
    })
    table_path = tmp_path / "pcd" / "s_geom_feat.txt"
    table.to_csv(table_path, index=False)
    meta = {
        "canvas_size": list(CANVAS), "angular_res": list(RES),
        "outputs": {"out_dir": str(tmp_path), "geom_feat": str(table_path)},
    }
    (tmp_path / "s_projection.json").write_text(json.dumps(meta))


def make_annotation(path):
    rgba = np.zeros((*CANVAS, 4), dtype=np.uint8)
    rgba[3, 0] = (*RED, 255)
    rgba[3, 1] = (253, 2, 1, 128)  # anti-aliased edge, within tolerance of red
    rgba[2, 4] = (*BLUE, 0)  # fully transparent: void
    Image.fromarray(rgba, "RGBA").save(path)


def test_annotation_to_mask_tolerance_and_transparency(tmp_path):
    make_annotation(tmp_path / "a.png")
    mask = backproject.annotation_to_mask(Image.open(tmp_path / "a.png"), LABELS)
    assert mask[3, 0] == 1 and mask[3, 1] == 1
    assert mask[2, 4] == 0
    assert mask.sum() == 2


def test_backproject_keeps_real_colour(tmp_path):
    make_projection(tmp_path)
    make_annotation(tmp_path / "a.png")
    result = backproject.backproject_annotation(tmp_path / "a.png", tmp_path, LABELS)

    assert result.class_counts == {0: 1, 1: 2}
    las = laspy.read(result.las)
    assert list(las.classification) == [1, 1, 0]
    assert list(las.fueldeep3d_class) == [0, 1, 2]
    assert list(las.red) == [1000, 2000, 3000]
    assert list(las.intensity) == [0, 32768, 65535]
    np.testing.assert_allclose(las.x, [1.0, 2.0, 3.0], atol=1e-3)
    assert result.mask_png.exists()


def test_backproject_drop_void(tmp_path):
    make_projection(tmp_path)
    make_annotation(tmp_path / "a.png")
    result = backproject.backproject_annotation(tmp_path / "a.png", tmp_path, LABELS, drop_void=True)
    assert len(laspy.read(result.las).points) == 2


def test_backproject_undoes_recentring(tmp_path):
    make_projection(tmp_path)
    meta_path = tmp_path / "s_projection.json"
    meta = json.loads(meta_path.read_text())
    meta["xy_offset"] = [100.0, -50.0]
    meta_path.write_text(json.dumps(meta))
    make_annotation(tmp_path / "a.png")
    las = laspy.read(backproject.backproject_annotation(tmp_path / "a.png", tmp_path, LABELS).las)
    np.testing.assert_allclose(las.x, [101.0, 102.0, 103.0], atol=1e-3)
    np.testing.assert_allclose(las.y, [-49.5, -49.75, -49.875], atol=1e-3)
    np.testing.assert_allclose(las.z, [0.1, 0.2, 0.3], atol=1e-3)


def test_backproject_rejects_wrong_size(tmp_path):
    make_projection(tmp_path)
    Image.new("RGB", (5, 5)).save(tmp_path / "small.png")
    try:
        backproject.backproject_annotation(tmp_path / "small.png", tmp_path, LABELS)
    except ValueError as e:
        assert "uncropped" in str(e)
    else:
        raise AssertionError("expected ValueError")
