"""SciPy B-spline construction with a small JAX differentiation adapter."""
from functools import partial

import numpy as np
from scipy.interpolate import BSpline
import jax
import jax.numpy as jnp


def knots(n_internal=10, degree=3, xmin=0.0, xmax=1.0):
    if not isinstance(n_internal, int) or n_internal < 0 or not isinstance(degree, int) or degree < 1:
        raise ValueError("n_internal must be nonnegative and degree must be positive integers")
    if not np.isfinite([xmin, xmax]).all() or xmin >= xmax:
        raise ValueError("spline range must have finite increasing endpoints")
    return np.concatenate([np.repeat(xmin, degree + 1), np.linspace(xmin, xmax, n_internal + 2)[1:-1], np.repeat(xmax, degree + 1)])


def basis(values, knot_vector, degree=3, derivative=0):
    """Use SciPy's basis and derivatives, retaining legacy endpoint clamping."""
    t = np.asarray(knot_vector, dtype=np.float64)
    x = np.asarray(values, dtype=np.float64)
    n_basis = len(t) - degree - 1
    return BSpline(t, np.eye(n_basis), degree)(x, nu=derivative)


@partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def _evaluate(values, knot_tuple, degree, derivative):
    n_basis = len(knot_tuple) - degree - 1
    shape = jax.ShapeDtypeStruct(values.shape + (n_basis,), jnp.float64)
    return jax.pure_callback(lambda x: basis(x, knot_tuple, degree, derivative), shape, values, vmap_method="sequential")


@_evaluate.defjvp
def _evaluate_jvp(knot_tuple, degree, derivative, primals, tangents):
    (values,), (tangent,) = primals, tangents
    evaluated = _evaluate(values, knot_tuple, degree, derivative)
    slope = _evaluate(values, knot_tuple, degree, derivative + 1)
    return evaluated, slope * tangent[..., None]


def differentiable_basis(values, knot_vector, degree=3):
    """Differentiate SciPy evaluation with SciPy derivatives, including Hessians."""
    t = tuple(float(x) for x in knot_vector)
    # Legacy Torch basis pulls the upper endpoint inward by this amount.
    values = jnp.clip(jnp.asarray(values, dtype=jnp.float64), t[0], t[-1] - (t[-1] - t[0]) * 1e-10)
    return _evaluate(values, t, degree, 0)


def design_matrix(values, n_internal=10, degree=3, include_interactions=True):
    """Precompute covariate bases outside differentiated optimization."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("spline covariates must be a finite matrix")
    t = knots(n_internal, degree)
    clamped = np.clip(x, 0, 1 - 1e-10)
    marginal = np.stack([basis(clamped[:, k], t, degree) for k in range(x.shape[1])], axis=1)
    pairs = [(j, k) for j in range(x.shape[1]) for k in range(j + 1, x.shape[1])] if include_interactions else []
    tensor = np.stack([(marginal[:, j, :, None] * marginal[:, k, None, :]).reshape(len(x), -1) for j, k in pairs], axis=1) if pairs else np.empty((len(x), 0, marginal.shape[-1] ** 2))
    return marginal, tensor
