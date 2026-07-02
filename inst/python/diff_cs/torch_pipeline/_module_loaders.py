"""Lazy file-path imports of the sibling 03_circuit_solver.py / 04_diff_omniscape.py
modules, plus the path resolution they depend on.

This file lives in ``inst/python/diff_cs/torch_pipeline/`` — one directory
*below* 03_circuit_solver.py / 04_diff_omniscape.py / 05_torch_pipeline.py,
which all stay siblings in ``inst/python/diff_cs/``. ``_SCRIPT_DIR`` therefore
resolves via ``.parent.parent`` (not ``.parent``) so the lazy-import spec
paths still point at the right files.
"""
import importlib.util
from pathlib import Path

# inst/python/diff_cs/torch_pipeline/_module_loaders.py -> .parent is
# .../torch_pipeline/, .parent.parent is .../diff_cs/ (sibling of 03_/04_).
_SCRIPT_DIR = Path(__file__).resolve().parent.parent


def _get_circuit_module():
    """Lazy-import 03_circuit_solver.py."""
    if not hasattr(_get_circuit_module, "_mod"):
        spec = importlib.util.spec_from_file_location(
            "circuit_solver_03",
            str(_SCRIPT_DIR / "03_circuit_solver.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _get_circuit_module._mod = mod
    return _get_circuit_module._mod


def _get_diff_omniscape_module():
    """Lazy-import 04_diff_omniscape.py."""
    if not hasattr(_get_diff_omniscape_module, "_mod"):
        spec = importlib.util.spec_from_file_location(
            "diff_omniscape_04",
            str(_SCRIPT_DIR / "04_diff_omniscape.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _get_diff_omniscape_module._mod = mod
    return _get_diff_omniscape_module._mod
