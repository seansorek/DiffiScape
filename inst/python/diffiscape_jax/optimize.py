"""Parametric optimization for DiffiScape connectivity surfaces.

Provides L-BFGS (via jaxopt) and Adam (via optax) optimizers that minimize
the negative log-likelihood of a connectivity-based point process model.
Both methods require JAX, jaxopt, and optax; these are imported lazily so the
module can be loaded (and tested for import errors) even when the dependencies
are absent.
"""
import time

import numpy as np

from .core import _connectivity_objective


def run_parametric_optimization(
    basis_values,
    obs_counts,
    valid_mask,
    n_rows,
    n_cols,
    cell_area=1.0,
    init_params=None,
    link_fn="exp",
    radius=13,
    block_size=5,
    parameterization="resistance",
    method="lbfgs",
    lr=0.01,
    n_epochs=300,
    patience=30,
    seed=42,
    verbose=True,
):
    """Run parametric optimization of a connectivity surface.

    Minimizes ``neg_loglik = -_connectivity_objective(params, ...)`` using
    either L-BFGS (second-order, via ``jaxopt.LBFGS``) or Adam (first-order,
    via ``optax.adam`` with early stopping).

    Parameters
    ----------
    basis_values : array-like
        Basis matrix of shape (n_cells, n_basis).
    obs_counts : array-like
        Observed counts per cell, shape (n_cells,).
    valid_mask : array-like
        Boolean mask of shape (n_rows * n_cols,) indicating valid cells.
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.
    cell_area : float, optional
        Area of each grid cell (default: 1.0).
    init_params : array-like, optional
        Initial parameter vector (intercept + basis coefficients).
        If None, defaults to zeros of length n_basis + 1.
    link_fn : str, optional
        Link function name: "exp", "softplus", or "identity" (default: "exp").
    radius : int, optional
        Window radius for windowed solving (default: 13).
    block_size : int, optional
        Block size for windowed solving (default: 5).
    parameterization : str, optional
        Either "resistance" or "permeability" (default: "resistance").
    method : str, optional
        Optimization method: "lbfgs" or "adam" (default: "lbfgs").
    lr : float, optional
        Learning rate for Adam optimizer (default: 0.01). Ignored for L-BFGS.
    n_epochs : int, optional
        Maximum number of iterations/epochs (default: 300).
    patience : int, optional
        Early-stopping patience for Adam: stop after this many epochs without
        improvement (default: 30). Ignored for L-BFGS.
    seed : int, optional
        Random seed (default: 42). Currently unused but reserved.
    verbose : bool, optional
        Whether to print progress messages (default: True).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"best_params"`` : np.ndarray -- optimized parameter vector.
        - ``"best_loglik"`` : float -- best log-likelihood (negated loss).
        - ``"loss_history"`` : list[float] -- per-iteration negative log-likelihood.
        - ``"n_epochs_run"`` : int -- number of iterations/epochs completed.
        - ``"elapsed"`` : float -- wall-clock seconds.
        - ``"converged"`` : bool -- whether the optimizer converged.

    Raises
    ------
    ImportError
        If jax, jaxopt, or optax are not installed.
    ValueError
        If *method* is not "lbfgs" or "adam".
    """
    # Lazy imports so the module is loadable without these packages.
    import jax
    import jax.numpy as jnp
    import jaxopt
    import optax

    if method not in ("lbfgs", "adam"):
        raise ValueError(
            f"method must be 'lbfgs' or 'adam', got '{method}'"
        )

    # Convert inputs to JAX arrays.
    params = jnp.array(init_params, dtype=jnp.float64)
    basis_jnp = jnp.array(basis_values, dtype=jnp.float64)
    mask_jnp = jnp.array(valid_mask)
    obs_jnp = jnp.array(obs_counts, dtype=jnp.float64)

    def neg_loglik(p):
        return -_connectivity_objective(
            p, basis_jnp, mask_jnp, n_rows, n_cols,
            cell_area, link_fn, radius, block_size,
            parameterization, obs_jnp,
        )

    t0 = time.time()
    loss_history = []

    if method == "lbfgs":
        solver = jaxopt.LBFGS(fun=neg_loglik, maxiter=n_epochs, tol=1e-6)
        result = solver.run(params)
        best_params = result.params
        best_loss = float(neg_loglik(best_params))
        loss_history = [best_loss]
        n_run = int(result.state.iter_num) if hasattr(result.state, "iter_num") else n_epochs
        converged = True

    else:  # adam
        schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_epochs)
        optimizer = optax.adam(learning_rate=schedule)
        opt_state = optimizer.init(params)
        grad_fn = jax.grad(neg_loglik)

        best_loss = float("inf")
        best_params = params
        stall = 0

        for epoch in range(n_epochs):
            g = grad_fn(params)
            updates, opt_state = optimizer.update(g, opt_state)
            params = optax.apply_updates(params, updates)
            loss = float(neg_loglik(params))
            loss_history.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_params = params
                stall = 0
            else:
                stall += 1

            if stall >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        n_run = len(loss_history)
        converged = stall < patience

    elapsed = time.time() - t0

    return {
        "best_params": np.array(best_params),
        "best_loglik": float(-best_loss),
        "loss_history": loss_history,
        "n_epochs_run": int(n_run),
        "elapsed": elapsed,
        "converged": converged,
    }
