"""Core functions for DiffiScape JAX circuit theory connectivity."""
import time

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

try:
    from jaxscape import GridGraph, ResistanceDistance
except ImportError:
    GridGraph = None
    ResistanceDistance = None


DEFAULT_R_MIN = 1.0
DEFAULT_R_MAX = 5000.0
DEFAULT_P_MIN = 1.0 / 5000.0
DEFAULT_P_MAX = 1.0


def ppp_loglik(connectivity_valid, obs_counts, cell_area, alpha=0.0, gamma=1.0):
    """Poisson point process log-likelihood.

    Computes the log-likelihood under the intensity model

        log lambda = alpha + gamma * log(1 + C)

    where *C* is connectivity (resistance distance) and *alpha* / *gamma*
    are learnable parameters.

    Parameters
    ----------
    connectivity_valid : jnp.ndarray, shape (n_valid,)
        Connectivity values at valid cells.
    obs_counts : jnp.ndarray, shape (n_valid,)
        Observed counts per valid cell.
    cell_area : float
        Area of each cell (for intensity integration).
    alpha : float or jnp scalar
        Intensity intercept (learnable).
    gamma : float or jnp scalar
        Connectivity coefficient (learnable).

    Returns
    -------
    float
        Scalar log-likelihood.
    """
    log_lambda = alpha + gamma * jnp.log1p(connectivity_valid)
    lam = jnp.exp(log_lambda)
    return jnp.sum(obs_counts * log_lambda) - jnp.sum(lam * cell_area)


def prepare_permeability(
    surface,
    parameterization,
    r_min=DEFAULT_R_MIN,
    r_max=DEFAULT_R_MAX,
    p_min=DEFAULT_P_MIN,
    p_max=DEFAULT_P_MAX,
):
    """Convert resistance to permeability or clamp permeability values.

    Parameters
    ----------
    surface : array-like
        Input surface, either resistance values or permeability values.
    parameterization : str
        Either "resistance" or "permeability". If "resistance", converts to
        permeability by taking 1/R and clamping. If "permeability", returns
        the input clamped to [p_min, p_max].
    r_min : float, optional
        Minimum resistance value for clamping (default: 1.0).
    r_max : float, optional
        Maximum resistance value for clamping (default: 5000.0).
    p_min : float, optional
        Minimum permeability value for clamping (default: 1/5000).
    p_max : float, optional
        Maximum permeability value for clamping (default: 1.0).

    Returns
    -------
    jax.numpy.ndarray
        Permeability array, clamped to valid range.
    """
    if jnp is None:
        raise ImportError("JAX is not installed")

    surface = jnp.asarray(surface, dtype=jnp.float64)

    if parameterization == "resistance":
        # Clamp resistance values first, then invert to get permeability
        clamped = jnp.clip(surface, r_min, r_max)
        return 1.0 / clamped
    elif parameterization == "permeability":
        # Just clamp the permeability values
        return jnp.clip(surface, p_min, p_max)
    else:
        raise ValueError(
            f"parameterization must be 'resistance' or 'permeability', got '{parameterization}'"
        )


def _mean_weight(x, y):
    """Compute mean edge weight from two node permeabilities.

    Parameters
    ----------
    x : float
        Permeability of first node.
    y : float
        Permeability of second node.

    Returns
    -------
    float
        Mean of the two permeabilities.
    """
    return (x + y) / 2


def forward_solve(
    resistance_matrix,
    n_rows,
    n_cols,
    sources=None,
    parameterization="resistance",
    solver_type="pyamg",
):
    """Solve circuit using JAXScape GridGraph and ResistanceDistance.

    Parameters
    ----------
    resistance_matrix : array-like
        2D array of resistance or permeability values, shape (n_rows, n_cols).
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.
    sources : array-like, optional
        Indices of source nodes. If None, defaults to node (0,0).
    parameterization : str, optional
        Either "resistance" or "permeability" (default: "resistance").
    solver_type : str, optional
        Solver type, currently unused but reserved for future use (default: "pyamg").

    Returns
    -------
    dict
        Dictionary with keys:
        - "connectivity": np.ndarray of shape (n_rows, n_cols) containing resistance distances
        - "elapsed": float, time elapsed in seconds
    """
    if jnp is None:
        raise ImportError("JAX is not installed")
    if GridGraph is None or ResistanceDistance is None:
        raise ImportError("JAXScape is not installed")

    # Convert to JAX array and ensure float64
    surface = jnp.array(resistance_matrix, dtype=jnp.float64)

    # Convert to permeability space
    permeability = prepare_permeability(surface, parameterization)

    # Create GridGraph with mean edge weight function
    grid = GridGraph(grid=permeability, fun=_mean_weight)

    # Set default sources to (0, 0) if not provided
    if sources is None:
        sources = grid.coord_to_index(jnp.array([0]), jnp.array([0]))

    # Create resistance distance solver
    distance = ResistanceDistance()

    # Run the solve
    t0 = time.time()
    dist_values = distance(grid, sources)
    elapsed = time.time() - t0

    # Convert back to 2D array
    connectivity = np.array(grid.node_values_to_array(dist_values))

    return {
        "connectivity": connectivity,
        "elapsed": elapsed,
    }


def _apply_link(params, basis_values, link_fn):
    """Map linear predictor through a link function to get resistance values.

    Computes ``link_fn(r_0 + basis_values @ z)`` where ``params = [r_0, z_1, ..., z_k]``.

    Parameters
    ----------
    params : jax.numpy.ndarray
        Parameter vector of length k+1. First element is the intercept (r_0),
        remaining elements are basis coefficients.
    basis_values : jax.numpy.ndarray
        Basis matrix of shape (n_cells, k).
    link_fn : str
        Link function name: "exp", "softplus", or "identity".

    Returns
    -------
    jax.numpy.ndarray
        Resistance values of shape (n_cells,).
    """
    r_0 = params[0]
    z = params[1:]
    log_R = r_0 + basis_values @ z
    if link_fn == "exp":
        return jnp.exp(log_R)
    elif link_fn == "softplus":
        return jax.nn.softplus(log_R)
    elif link_fn == "identity":
        return log_R
    else:
        return jnp.exp(log_R)


def _connectivity_objective(params, basis_values, valid_mask, n_rows, n_cols,
                            cell_area, link_fn, radius, block_size,
                            parameterization, obs_counts=None):
    """Compute a scalar PPP log-likelihood for gradient-based optimization.

    Builds the full chain: params -> link function -> resistance surface ->
    permeability -> GridGraph -> ResistanceDistance -> connectivity ->
    PPP log-likelihood.

    The last two elements of *params* are the intensity parameters
    ``(alpha, gamma)`` for the PPP model
    ``log lambda = alpha + gamma * log(1 + C)``.
    The remaining elements are passed to :func:`_apply_link` as resistance
    parameters.

    Parameters
    ----------
    params : jax.numpy.ndarray
        Parameter vector ``[r_0, z_1, ..., z_K, alpha, gamma]``.
        The first ``len(params) - 2`` elements are resistance parameters
        (intercept + basis coefficients); the last two are the PPP intensity
        intercept and connectivity coefficient.
    basis_values : jax.numpy.ndarray
        Basis matrix of shape (n_valid_cells, n_basis).
    valid_mask : jax.numpy.ndarray
        Boolean mask of shape (n_rows * n_cols,) indicating valid cells.
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.
    cell_area : float
        Area of each grid cell (used for intensity integration).
    link_fn : str
        Link function name: "exp", "softplus", or "identity".
    radius : int
        Window radius for future windowed solving.
    block_size : int
        Block size for future windowed solving.
    parameterization : str
        Either "resistance" or "permeability".
    obs_counts : jax.numpy.ndarray, optional
        Observed counts per valid cell.

    Returns
    -------
    float
        Scalar log-likelihood value.
    """
    # Split params: resistance params and intensity params (alpha, gamma)
    resistance_params = params[:-2]
    alpha = params[-2]
    gamma = params[-1]

    resistance_flat = _apply_link(resistance_params, basis_values, link_fn)

    # Build the full resistance surface from valid cells
    full_surface = jnp.ones(n_rows * n_cols) * jnp.mean(resistance_flat)
    full_surface = full_surface.at[valid_mask].set(resistance_flat)
    surface_2d = full_surface.reshape((n_rows, n_cols))

    permeability = prepare_permeability(surface_2d, parameterization)
    grid = GridGraph(grid=permeability, fun=_mean_weight)
    distance = ResistanceDistance()
    source = grid.coord_to_index(jnp.array([0]), jnp.array([0]))
    connectivity = distance(grid, source)

    # PPP log-likelihood on valid cells
    conn_valid = connectivity.ravel()[valid_mask]
    return ppp_loglik(conn_valid, obs_counts, cell_area, alpha, gamma)
