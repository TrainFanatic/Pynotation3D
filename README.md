# Pynotation3D

A library pipeline for annotating, processing, and segmenting subcanopy fuels
in terrestrial lidar (TLS) point clouds.

0. **Subplots** (`split_into_subplots`, `write_subplots`): split a large plot
   into square subplots (10 x 10 m by default, height unlimited), each shifted
   so its centre is at (0, 0).
1. **Project** (`project_point_cloud`): classify a LAS point cloud with
   FuelDeep3D, then project it to 2D spherical images with
   [TLS-SphereMap](https://github.com/TrainFanatic/TLS-SphereMap): RGB, IRZ
   (intensity/range/Z false colour), a point/no-point mask and the FuelDeep3D
   classification.
2. **Backproject** (`backproject_annotation`): take one of those projections
   after it has been hand-annotated and write a LAS whose `classification` is
   the annotated class, keeping the real RGB and intensity.
3. 3DMASC classification (not implemented yet).

## Installation

```bash
pip install -e .
```

This also installs TLS-SphereMap (a work of Fei Zhang) from GitHub. Three things must be installed
separately:

- `torch`, a CUDA build for `method="full"` (TLS-SphereMap's geometric
  features need a GPU);
- `open3d`;
- the FuelDeep3D R package, whose PyTorch model is loaded from its
  `extdata/python` directory. Find that directory in R with
  `system.file("extdata", "python", package = "FuelDeep3D")` and pass it as
  `fueldeep3d_dir=` or set `PYNOTATION3D_FUELDEEP3D_DIR`.

Users on the Stanford Sherlock cluster should load the following modules as such:

```bash
ml devel python/3.12.1 math py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
   py-scikit-learn/1.5.1_py312 viz py-open3d/0.19.0_py312 devel py-pandas/2.2.1_py312
source $GROUP_HOME/envs/pynotation3d/bin/activate
```

## Usage

```python
from pynotation3d import split_into_subplots, write_subplots, project_point_cloud, backproject_annotation

subplots = split_into_subplots("55.las", size=10.0)  # or size=(10.0, 20.0)
paths = write_subplots(subplots, "subplots/55", "55")  # 55_c<col>_r<row>.las + 55_subplots.json

out = project_point_cloud("55-cropped.las", "outputs/55-cropped", recenter=True)
# out.rgb_png, out.irz_png, out.exists_png, out.classification_png

result = backproject_annotation(
    "55-cropped_annotated.png",  # same size as the projection images
    "outputs/55-cropped",        # project_point_cloud's output directory
    "LAS-all-cropped-55",        # label map name in TLS-SphereMap's label_maps.json
)
# result.las, result.class_counts
```

Or from the shell (e.g. in a Slurm job):

```bash
pynotation3d subplots 55.las subplots/55 --size 10                               # 10 x 10 m tiles
pynotation3d project 55-cropped.las outputs/55-cropped --recenter               # GPU, full cube
pynotation3d project 55-cropped.las outputs/55-cropped --recenter --method fast # CPU, PNG channels only
pynotation3d backproject 55-cropped_annotated.png outputs/55-cropped LAS-all-cropped-55
```

Subplots are already centred, so project them without `--recenter`. Each
point goes to exactly one subplot. The grid starts at the cloud's X/Y minimum
unless `origin=` / `--origin` is given. `<stem>_subplots.json` records each
subplot's bounds and the centre subtracted from X/Y; add it back to return to
the original coordinates.

`project` has two methods:

- `method="full"` (default, `--method full`): TLS-SphereMap's GPU geometric
  features and projection. Writes its full 24-channel image cube (geometric
  features, PCA/MNF/ICA) and `pcd/<stem>_geom_feat.txt`. Needs a GPU; can take
  hours on a large scan.
- `method="fast"` (`--method fast`): projects on CPU only the channels the four
  PNGs need, with the same pixel rules as TLS-SphereMap. Writes
  `pcd/<stem>_points.npz`. Takes minutes and needs no GPU (FuelDeep3D still
  uses one if available).

Either way, keep the points table and `<stem>_projection.json` next to the
images: `backproject` maps pixels back to exactly those points. Use
`--recenter` for crops where the scanner is not at the coordinate origin.

## Tests

```bash
pip install -e ".[test]"
pytest
```
