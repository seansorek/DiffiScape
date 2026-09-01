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
    from jaxscape import GridGraph, ResistanceDistance, WindowOperation
    from jaxscape.solvers import AMJaxCGSolver, BCOOLinearOperator, linear_solve
    from jaxscape.solvers.amjaxcgsolver import build_amjax_solver, amjax_preconditioner_operator
    from jaxscape.utils import graph_laplacian
    from jaxscape.utils import padding as jaxscape_padding
    import lineax as lx
    from jax.experimental.sparse import BCOO
except ImportError:
    GridGraph = None
    ResistanceDistance = None
    WindowOperation = None
    AMJaxCGSolver = None
    BCOOLinearOperator = None
    graph_laplacian = None
    jaxscape_padding = None
    lx = None
    BCOO = None


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


_EPS = 1e-8


def _boundary_mask(n_rows, n_cols):
    """Boolean mask (n_nodes,) that is True for boundary pixels."""
    n = n_rows * n_cols
    mask = jnp.zeros(n, dtype=bool)
    # first/last row
    mask = mask.at[:n_cols].set(True)
    mask = mask.at[-n_cols:].set(True)
    # first/last column
    rows = jnp.arange(n_rows)
    mask = mask.at[rows * n_cols].set(True)
    mask = mask.at[rows * n_cols + n_cols - 1].set(True)
    return mask


def _grounded_laplacian(permeability, n_rows, n_cols, absorption, boundary):
    """Build the boundary-grounded Laplacian (and adjacency) for *permeability*."""
    grid = GridGraph(grid=permeability, fun=_mean_weight)
    A = grid.get_adjacency_matrix()
    L = graph_laplacian(A)

    boundary_idx = jnp.where(boundary)[0].astype(L.indices.dtype)
    abs_indices = jnp.stack([boundary_idx, boundary_idx], axis=-1)
    abs_data = jnp.full(boundary_idx.shape, absorption, dtype=L.data.dtype)
    absorption_diag = BCOO((abs_data, abs_indices), shape=L.shape)
    L_grounded = L + absorption_diag
    return A, L_grounded


def circuit_solve_init(permeability, n_rows, n_cols, absorption=0.01):
    """Build AMJax-preconditioned solver state for boundary absorption.

    Adds absorption only to boundary nodes: (L + alpha * I_boundary) v = 1_interior.
    Call once per grid geometry, then pass the returned state to
    :func:`circuit_solve` for each permeability update.

    Only the AMJax multigrid *preconditioner* built here is reused across
    calls -- it depends on the sparsity structure (grid geometry), not the
    permeability values. :func:`circuit_solve` rebuilds the Laplacian/operator
    from whatever permeability it is given and materializes a fresh
    operator-specific CG state against the reused preconditioner, so the
    cached state can legitimately be reused across a sequence of different
    permeability surfaces on the same geometry.

    Returns
    -------
    dict with keys 'solver', 'state', 'boundary', 'interior', 'n', 'n_rows',
    'n_cols', 'absorption'
    """
    boundary = _boundary_mask(n_rows, n_cols)
    A, L_grounded = _grounded_laplacian(permeability, n_rows, n_cols, absorption, boundary)

    solver = AMJaxCGSolver(rtol=1e-6, atol=1e-6)
    operator = BCOOLinearOperator(L_grounded)
    state = solver.init_preconditioner(operator, {})

    return {
        "solver": solver,
        "state": state,
        "boundary": boundary,
        "interior": ~boundary,
        "n": n_rows * n_cols,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "absorption": absorption,
    }


def circuit_solve_absorption_init(permeability, n_rows, n_cols, absorption=0.01):
    """Build AMJax-preconditioned solver state for uniform absorption.

    Adds absorption to ALL nodes: (L + alpha * I) v = 1.
    Call once per grid geometry, then pass the returned state to
    :func:`circuit_solve_absorption` for each permeability update.

    As with :func:`circuit_solve_init`, only the AMJax preconditioner is
    cached here; :func:`circuit_solve_absorption` rebuilds the Laplacian from
    whatever permeability it receives and materializes a fresh CG state
    against the reused preconditioner.

    Returns
    -------
    dict with keys 'solver', 'state', 'n', 'n_rows', 'n_cols', 'absorption'
    """
    n = n_rows * n_cols
    all_nodes = jnp.ones(n, dtype=bool)
    A, L_abs = _grounded_laplacian(permeability, n_rows, n_cols, absorption, all_nodes)

    solver = AMJaxCGSolver(rtol=1e-6, atol=1e-6)
    operator = BCOOLinearOperator(L_abs)
    state = solver.init_preconditioner(operator, {})

    return {
        "solver": solver,
        "state": state,
        "n": n,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "absorption": absorption,
    }


def circuit_solve(
    permeability,
    n_rows,
    n_cols,
    source_from_resistance=False,
    absorption=0.01,
    solver_state=None,
):
    """Single-solve circuit current density using AMJax-preconditioned CG.

    Mirrors the Torch pipeline's boundary-grounded Laplacian solve but runs
    entirely in JAX (JIT-compilable, differentiable, GPU-capable).

    Parameters
    ----------
    permeability : jax.Array, shape (n_rows, n_cols)
        Permeability surface (already converted from resistance).
    n_rows, n_cols : int
        Grid dimensions.
    source_from_resistance : bool
        If True, source strength is proportional to permeability.
    absorption : float
        Diagonal absorption added to boundary nodes for grounding.
    solver_state : dict, optional
        Pre-computed solver state from :func:`circuit_solve_init`.  When
        provided, skips AMG hierarchy construction (much faster) but the
        Laplacian/operator are still rebuilt from *this call's*
        ``permeability`` -- the cached state only supplies the reusable AMJax
        multigrid preconditioner, which depends on grid geometry, not on the
        permeability values.

    Returns
    -------
    current_density : jax.Array, shape (n_rows * n_cols,)
        Current density at each node.
    voltage : jax.Array, shape (n_rows * n_cols,)
        Voltage at each node.
    """
    if solver_state is None:
        solver_state = circuit_solve_init(
            permeability, n_rows, n_cols, absorption
        )

    solver = solver_state["solver"]
    boundary = solver_state["boundary"]
    interior = solver_state["interior"]
    n = solver_state["n"]

    A, L_grounded = _grounded_laplacian(
        permeability, n_rows, n_cols, absorption, boundary
    )
    operator = BCOOLinearOperator(L_grounded)
    state = solver.materialize_state(operator, {}, solver_state["state"])

    b = jnp.where(interior, 1.0, 0.0)
    if source_from_resistance:
        b = b * permeability.ravel()

    v = lx.linear_solve(operator, b, solver=solver, state=state).value

    rows = A.indices[:, 0]
    cols = A.indices[:, 1]
    edge_currents = A.data * jnp.sqrt((v[rows] - v[cols]) ** 2 + _EPS)
    current_density = jnp.zeros(n).at[rows].add(edge_currents) * 0.5

    return current_density, v


def circuit_solve_absorption(
    permeability,
    n_rows,
    n_cols,
    source_from_resistance=False,
    absorption=0.01,
    solver_state=None,
):
    """Uniform absorption solve: (L + alpha * I) v = 1.

    All nodes get equal absorption (well-conditioned system).
    Mirrors the Torch pipeline's ``solve_circuit_absorption``.

    Parameters
    ----------
    permeability : jax.Array, shape (n_rows, n_cols)
        Permeability surface.
    n_rows, n_cols : int
        Grid dimensions.
    source_from_resistance : bool
        If True, source strength is proportional to permeability.
    absorption : float
        Diagonal absorption added uniformly to all nodes.
    solver_state : dict, optional
        Pre-computed solver state from :func:`circuit_solve_absorption_init`.
        As with :func:`circuit_solve`, only the AMJax preconditioner is
        reused from this state -- the Laplacian/operator are rebuilt from
        this call's ``permeability``.

    Returns
    -------
    current_density : jax.Array, shape (n_rows * n_cols,)
    voltage : jax.Array, shape (n_rows * n_cols,)
    """
    if solver_state is None:
        solver_state = circuit_solve_absorption_init(
            permeability, n_rows, n_cols, absorption
        )

    solver = solver_state["solver"]
    n = solver_state["n"]

    all_nodes = jnp.ones(n, dtype=bool)
    A, L_abs = _grounded_laplacian(permeability, n_rows, n_cols, absorption, all_nodes)
    operator = BCOOLinearOperator(L_abs)
    state = solver.materialize_state(operator, {}, solver_state["state"])

    b = jnp.ones(n, dtype=jnp.float64)
    if source_from_resistance:
        b = b * permeability.ravel()

    v = lx.linear_solve(operator, b, solver=solver, state=state).value

    rows = A.indices[:, 0]
    cols = A.indices[:, 1]
    edge_currents = A.data * jnp.sqrt((v[rows] - v[cols]) ** 2 + _EPS)
    current_density = jnp.zeros(n).at[rows].add(edge_currents) * 0.5

    return current_density, v


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

    grid = GridGraph(grid=permeability, fun=_mean_weight)

    if sources is None:
        sources = grid.coord_to_index(jnp.array([0]), jnp.array([0]))

    distance = ResistanceDistance()

    t0 = time.time()
    dist_values = distance(grid, sources)
    elapsed = time.time() - t0

    # Multi-source: average distance over all sources before reshaping
    if dist_values.ndim > 1:
        dist_values = jnp.mean(dist_values, axis=0)
    connectivity = np.array(grid.node_values_to_array(dist_values))

    return {
        "connectivity": connectivity,
        "elapsed": elapsed,
    }


def cumulative_current_core(permeability, n_rows, n_cols, radius, block_size):
    """Differentiable moving-window (Omniscape-style) connectivity accumulation.

    This is the JAX-differentiable heart of
    :func:`diffiscape_jax.window.cumulative_current` -- the SAME operator
    used on the forward / evaluation path (``ds_jax_connectivity()`` ->
    ``omni$cum_current`` in R).  It operates directly on an already-JAX
    ``permeability`` array (no host round-trip via ``numpy``), so it can be
    embedded inside a larger differentiable objective and passed through
    :func:`jax.grad`.

    Tiles the landscape into overlapping windows (``block_size`` core pixels
    surrounded by a ``radius``-pixel buffer), places a single source at each
    window's centre, solves :class:`jaxscape.ResistanceDistance` within the
    window, and accumulates (sums) the resulting distance surfaces across all
    overlapping windows -- exactly mirroring
    :func:`diffiscape_jax.window.cumulative_current`.

    Parameters
    ----------
    permeability : jax.numpy.ndarray, shape (n_rows, n_cols)
        Permeability surface (already converted from resistance via
        :func:`prepare_permeability`).
    n_rows, n_cols : int
        Grid dimensions.
    radius : int
        Buffer radius around each core window (must match the forward path's
        ``radius`` for the two connectivity definitions to agree).
    block_size : int
        Core window size / source-block side length (must match the forward
        path's ``block_size``).

    Returns
    -------
    jax.numpy.ndarray, shape (n_rows, n_cols)
        Accumulated connectivity surface, in the same units and with the
        same sign convention as ``window.cumulative_current()``'s
        ``"current"`` output (NOT re-normalized or negated).
    """
    if WindowOperation is None or jaxscape_padding is None:
        raise ImportError("JAXScape is not installed")
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")

    # Pad with edge values so buffer zones reflect real landscape, then pad
    # further so dimensions satisfy WindowOperation's divisibility
    # constraint -- identical to window.cumulative_current().
    padded = jnp.pad(permeability, radius, mode='edge')
    padded = jaxscape_padding(padded, buffer_size=radius, window_size=block_size)

    window_op = WindowOperation(
        shape=padded.shape,
        window_size=block_size,
        buffer_size=radius,
    )

    distance_solver = ResistanceDistance()
    current_acc = jnp.zeros(padded.shape, dtype=jnp.float64)

    for xy, window in window_op.lazy_iterator(padded):
        grid = GridGraph(grid=window, fun=_mean_weight)
        center = window.shape[0] // 2
        source = grid.coord_to_index(
            jnp.array([center]), jnp.array([center])
        )
        dist_values = distance_solver(grid, source)
        dist_2d = grid.node_values_to_array(dist_values)
        current_acc = window_op.update_raster_with_window(
            xy, current_acc, dist_2d, fun=jnp.add,
        )

    # Strip padding to recover original extent. Stays a jax.Array (no numpy
    # conversion) so this remains differentiable.
    return current_acc[radius:radius + n_rows, radius:radius + n_cols]


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
    permeability -> GridGraph -> moving-window cumulative-current solve ->
    connectivity -> PPP log-likelihood.

    As of the fix for GH issue #105, this calls the SAME differentiable
    moving-window operator (:func:`cumulative_current_core`) used on the
    forward / evaluation path (``window.cumulative_current()`` ->
    ``ds_jax_connectivity()`` -> ``omni$cum_current`` in R, consumed by
    ``evaluate_full_model()``, ``diffiscape()`` step 6, and
    ``ds_posterior()``).  Previously this objective placed a single source
    at grid cell (0, 0) and used plain ``ResistanceDistance``, which (a)
    ignored the caller's ``radius``/``block_size`` entirely and (b) fit
    ``gamma`` against a connectivity definition with the opposite sign
    convention and a different (single, arbitrary) source topology than
    every downstream evaluation of the model. Sharing one operator for both
    training and evaluation eliminates both mismatches by construction.

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
        Moving-window buffer radius, honored and forwarded to
        :func:`cumulative_current_core` -- must match the ``radius`` used on
        the forward/evaluation path for the fitted ``gamma`` to be
        meaningful there.
    block_size : int
        Moving-window core / source-block size, honored and forwarded to
        :func:`cumulative_current_core` -- must match the ``block_size``
        used on the forward/evaluation path.
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
    connectivity = cumulative_current_core(
        permeability, n_rows, n_cols, radius, block_size
    )

    # PPP log-likelihood on valid cells
    conn_valid = connectivity.ravel()[valid_mask]
    return ppp_loglik(conn_valid, obs_counts, cell_area, alpha, gamma)
