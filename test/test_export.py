import numpy as np
import yaml

from pynotation3d import export

NAMES = ["True-R", "True-G", "True-B", "intensity_raw", "z_raw", "range_raw", "seg_mask"]


def make_cube():
    """2x3 cube; pixel (1, 2) received no point (range 0, seg_mask void 9)."""
    cube = np.zeros((2, 3, len(NAMES)), dtype=np.float32)
    cube[..., NAMES.index("True-R")] = [[10, 20, 30], [40, 50, 0.7]]  # 0.7 is the empty-pixel fill
    cube[..., NAMES.index("range_raw")] = [[1, 2, 3], [4, 5, 0]]
    cube[..., NAMES.index("seg_mask")] = [[0, 1, 2], [5, 0, 9]]
    return cube


def test_valid_mask_uses_range():
    valid = export.valid_mask(make_cube(), NAMES)
    assert valid.tolist() == [[True, True, True], [True, True, False]]


def test_minmax_ignores_invalid_pixels():
    cube = make_cube()
    valid = export.valid_mask(cube, NAMES)
    red = export.minmax_to_uint8(export.channel(cube, NAMES, "True-R"), valid)
    assert red[0, 0] == 0 and red[1, 1] == 255  # stretched over 10..50, not 0.7..50
    assert red[1, 2] == 0


def test_minmax_constant_channel_is_black():
    values = np.full((2, 2), 3.0)
    assert not export.minmax_to_uint8(values, np.ones((2, 2), bool)).any()


def test_classification_colors_and_void():
    cube = make_cube()
    image = export.classification_image(cube, NAMES, export.valid_mask(cube, NAMES))
    assert tuple(image[0, 0]) == export.CLASS_COLORS[0]
    assert tuple(image[0, 1]) == export.CLASS_COLORS[1]
    assert tuple(image[0, 2]) == export.CLASS_COLORS[2]
    assert tuple(image[1, 0]) == (0, 0, 0)  # class 5 has no colour
    assert tuple(image[1, 2]) == (0, 0, 0)  # void


def test_write_projection_pngs(tmp_path):
    cube_path, meta_path = tmp_path / "s_image_cube.npy", tmp_path / "s_image_cube_meta.yaml"
    np.save(cube_path, make_cube())
    meta_path.write_text(yaml.safe_dump({"channel_names": NAMES}))
    paths = export.write_projection_pngs(cube_path, meta_path, tmp_path, "s")
    assert sorted(paths) == ["classification", "exists", "irz", "rgb"]
    assert all(p.exists() for p in paths.values())
