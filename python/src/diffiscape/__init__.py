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

from .api import connectivity, fit, predict, sample
from .window import cumulative_current
from .likelihoods import fit_intensity
from .surrogate import optimize_surrogate
from .diagnostics import posterior_predictive

__all__ = ["connectivity", "fit", "predict", "sample", "cumulative_current", "fit_intensity", "optimize_surrogate", "posterior_predictive"]
__version__ = "0.1.0"
