"""Moving-window (Omniscape-style) cumulative current using JAXScape.

This module implements cumulative current calculation via moving-window
circuit theory, accumulating resistances across overlapping windows
on a spatial grid. Uses JAXScape's WindowOperation for proper windowing
with edge-padded buffers.
"""
import numpy as np
import time

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

try:
    from jaxscape import GridGraph, ResistanceDistance, SpielmanApproximation, WindowOperation
    from jaxscape.solvers import AMJaxCGSolver
    from jaxscape.utils import padding as jaxscape_padding
except ImportError:
    GridGraph = None
    ResistanceDistance = None
    SpielmanApproximation = None
    WindowOperation = None
    AMJaxCGSolver = None
    jaxscape_padding = None

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

    Uses JAXScape's WindowOperation to tile the landscape into overlapping
    windows.  Each window has a core of ``block_size x block_size`` pixels
    surrounded by a buffer of ``radius`` pixels.  A single source is placed
    at the window centre, the circuit is solved via ResistanceDistance, and
    the resulting distance map is accumulated across all windows.

    Parameters
    ----------
    resistance_matrix : array-like
        2D array of resistance or permeability values, shape (n_rows, n_cols).
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.
    radius : int, optional
        Buffer radius around each core window (default: 13).
    block_size : int, optional
        Core window size controlling sub-sampling density (default: 5).
        Smaller values produce denser source placement (block_size=1
        places a source at every cell).
    source_from_resistance : bool, optional
        Reserved for future use (default: True).
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
    """
    if jnp is None:
        raise ImportError("JAX is not installed")
    if GridGraph is None or ResistanceDistance is None:
        raise ImportError("JAXScape is not installed")

    if resistance_matrix.shape != (n_rows, n_cols):
        raise ValueError(
            f"resistance_matrix.shape {resistance_matrix.shape} must match "
            f"(n_rows, n_cols) = ({n_rows}, {n_cols})"
        )
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")

    if output in ("voltage", "both"):
        raise NotImplementedError(
            f"output='{output}' is not yet implemented. "
            "Only output='current' is supported."
        )

    t0 = time.time()

    surface = jnp.array(resistance_matrix, dtype=jnp.float64)
    permeability = prepare_permeability(surface, parameterization)

    # Pad with edge values so buffer zones reflect real landscape
    padded = jnp.pad(permeability, radius, mode='edge')
    # Ensure dimensions satisfy WindowOperation divisibility constraint
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

    # Strip padding to recover original extent
    result_current = np.array(
        current_acc[radius:radius + n_rows, radius:radius + n_cols]
    )

    elapsed = time.time() - t0

    return {
        "current": result_current,
        "voltage": None,
        "elapsed": elapsed,
    }
