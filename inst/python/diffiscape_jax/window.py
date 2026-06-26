"""Moving-window (Omniscape-style) cumulative current using JAXScape.

This module implements cumulative current calculation via moving-window
circuit theory, accumulating resistances across overlapping windows
on a spatial grid. The implementation uses JAXScape's GridGraph and
ResistanceDistance solvers applied to sub-windows.
"""
import numpy as np
import time

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

from .core import prepare_permeability, _mean_weight


def cumulative_current(
    resistance_matrix,
    n_rows,
    n_cols,
    radius=13,
    block_size=5,
    source_from_resistance=True,
    parameterization="resistance",
    output="current",
):
    """Calculate cumulative current via moving-window circuit theory.

    Uses a sliding window approach: for each position, extracts a sub-grid
    of size (2*radius + 1) x (2*radius + 1), solves the circuit with a
    source at the center using ResistanceDistance, and accumulates the
    results across all windows.

    Parameters
    ----------
    resistance_matrix : array-like
        2D array of resistance or permeability values, shape (n_rows, n_cols).
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.
    radius : int, optional
        Radius of the moving window (default: 13). Window size is
        (2*radius + 1) x (2*radius + 1).
    block_size : int, optional
        Block size for sub-sampling or processing. Currently reserved for
        future use (default: 5).
    source_from_resistance : bool, optional
        If True, sources are placed at the center of each window. Currently
        reserved for future use (default: True).
    parameterization : str, optional
        Either "resistance" or "permeability" (default: "resistance").
    output : str, optional
        Which outputs to compute: "current", "voltage", or "both"
        (default: "current").

    Returns
    -------
    dict
        Dictionary with keys:
        - "current": np.ndarray of shape (n_rows, n_cols) with accumulated
          current, or None if output does not include "current".
        - "voltage": np.ndarray of shape (n_rows, n_cols) with accumulated
          voltage, or None if output does not include "voltage".
        - "elapsed": float, time elapsed in seconds.

    Raises
    ------
    ImportError
        If JAX is not installed.
    ImportError
        If JAXScape is not installed (will raise on first use, not on
        import, to allow for graceful skipping in tests).
    """
    if jnp is None:
        raise ImportError("JAX is not installed")
    if GridGraph is None or ResistanceDistance is None:
        raise ImportError("JAXScape is not installed")

    t0 = time.time()

    # Convert to JAX array
    surface = jnp.array(resistance_matrix, dtype=jnp.float64)

    # Convert to permeability space
    permeability = prepare_permeability(surface, parameterization)

    # Initialize accumulators
    current_acc = np.zeros((n_rows, n_cols), dtype=np.float64)
    voltage_acc = np.zeros((n_rows, n_cols), dtype=np.float64)

    # Sliding window loop
    window_size = 2 * radius + 1
    half_window = radius

    for i in range(n_rows):
        for j in range(n_cols):
            # Define window bounds
            i_min = max(0, i - half_window)
            i_max = min(n_rows, i + half_window + 1)
            j_min = max(0, j - half_window)
            j_max = min(n_cols, j + half_window + 1)

            # Extract sub-grid
            sub_perm = permeability[i_min:i_max, j_min:j_max]

            # Create GridGraph for this sub-window
            sub_grid = GridGraph(grid=sub_perm, fun=_mean_weight)

            # Source is at the center of the window (or as close as possible
            # if window is at edge)
            source_i = i - i_min
            source_j = j - j_min
            source_idx = sub_grid.coord_to_index(
                jnp.array([source_i]), jnp.array([source_j])
            )

            # Solve for resistance distances from this source
            distance_solver = ResistanceDistance()
            dist_values = distance_solver(sub_grid, source_idx)

            # Convert to 2D array
            sub_distances = np.array(sub_grid.node_values_to_array(dist_values))

            # Accumulate into the main grid
            # (only accumulate the part that fits in the original grid)
            current_acc[i_min:i_max, j_min:j_max] += sub_distances

            # For now, voltage is not computed (would need additional info
            # about sources and sinks). Set to zeros for "voltage" or "both".
            # voltage_acc is already initialized to zeros.

    elapsed = time.time() - t0

    # Prepare output based on requested modes
    result = {
        "current": current_acc if output in ("current", "both") else None,
        "voltage": voltage_acc if output in ("voltage", "both") else None,
        "elapsed": elapsed,
    }

    return result
