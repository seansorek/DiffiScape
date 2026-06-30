"""DiffiScape JAX: JAX-based circuit theory connectivity package."""

# Enable 64-bit precision in JAX.  Without this, jnp.array(..., dtype=float64)
# is silently downcast to float32, which degrades accuracy on the
# ill-conditioned Laplacian solves used throughout this package.
# Must run before any JAX arrays are created.
try:
    import jax as _jax
    _jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

# amjax 0.0.3 renamed AMJAXSolver → MultilevelSolver; jaxscape still
# imports the old name.  Patch before jaxscape.solvers is loaded.
try:
    import amjax as _amjax
    if not hasattr(_amjax, "AMJAXSolver") and hasattr(_amjax, "MultilevelSolver"):
        _amjax.AMJAXSolver = _amjax.MultilevelSolver
except ImportError:
    pass
