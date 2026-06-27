"""DiffiScape JAX: JAX-based circuit theory connectivity package."""

# amjax 0.0.3 renamed AMJAXSolver → MultilevelSolver; jaxscape still
# imports the old name.  Patch before jaxscape.solvers is loaded.
try:
    import amjax as _amjax
    if not hasattr(_amjax, "AMJAXSolver") and hasattr(_amjax, "MultilevelSolver"):
        _amjax.AMJAXSolver = _amjax.MultilevelSolver
except ImportError:
    pass
