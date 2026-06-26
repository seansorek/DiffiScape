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
    from jaxscape import GridGraph
    from jaxscape.distances import ResistanceDistance
except ImportError:
    GridGraph = None
    ResistanceDistance = None


DEFAULT_R_MIN = 1.0
DEFAULT_R_MAX = 5000.0
DEFAULT_P_MIN = 1.0 / 5000.0
DEFAULT_P_MAX = 1.0


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
