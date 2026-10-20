"""Locate and import the external codebases Pynotation3D builds on.

TLS-SphereMap is a regular pip dependency (its modules are imported as the
top-level packages ``tools``, ``processing`` and ``data_annotation``).

FuelDeep3D is an R package whose PyTorch model lives in its
``extdata/python`` directory. It is loaded from there at runtime rather
than copied into this package, because FuelDeep3D is GPL-3 licensed.
"""
import importlib
import importlib.util
import os
from pathlib import Path

FUELDEEP3D_ENV = "PYNOTATION3D_FUELDEEP3D_DIR"

_TLS_INSTALL_HINT = (
    "TLS-SphereMap is required: "
    "pip install git+https://github.com/TrainFanatic/TLS-SphereMap"
)


def tls_module(name):
    """Import a TLS-SphereMap module, e.g. ``tls_module("processing.spherical_projection")``."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise ImportError(f"{_TLS_INSTALL_HINT} (failed to import {name!r}: {e})") from e


def fueldeep3d_python_dir(fueldeep3d_dir=None):
    """Return FuelDeep3D's ``extdata/python`` directory.

    Uses ``fueldeep3d_dir`` if given, otherwise the ``PYNOTATION3D_FUELDEEP3D_DIR``
    environment variable.
    """
    path = fueldeep3d_dir or os.environ.get(FUELDEEP3D_ENV)
    if not path:
        raise FileNotFoundError(
            "FuelDeep3D's Python directory is not set. Pass fueldeep3d_dir= or set "
            f"{FUELDEEP3D_ENV}. Find it in R with: "
            'system.file("extdata", "python", package = "FuelDeep3D")'
        )
    path = Path(path).expanduser()
    missing = [f for f in ("model.py", "utils.py") if not (path / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{path} is not FuelDeep3D's extdata/python directory (missing {missing})")
    return path


def fueldeep3d_default_model(python_dir):
    """Return the model checkpoint bundled with FuelDeep3D (``extdata/model/best_model.pth``)."""
    model = Path(python_dir).parent / "model" / "best_model.pth"
    if not model.is_file():
        raise FileNotFoundError(f"FuelDeep3D's bundled model not found at {model}")
    return model


def fueldeep3d_module(python_dir, name):
    """Load ``<python_dir>/<name>.py`` under a private module name."""
    spec = importlib.util.spec_from_file_location(f"_pynotation3d_fueldeep3d_{name}", Path(python_dir) / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
