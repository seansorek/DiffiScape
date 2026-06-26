"""Optimization for DiffiScape connectivity surfaces.

Provides parametric (L-BFGS / Adam) and neural (Flax-based) optimizers that
minimize the negative log-likelihood of a connectivity-based point process
model.  Both families require JAX and optax; neural optimization additionally
requires Flax.  Dependencies are imported lazily so the module can be loaded
(and tested for import errors) even when the packages are absent.
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

    The parameter vector passed to the objective has layout
    ``[r_0, z_1, ..., z_K, alpha, gamma]``.  The caller's *init_params*
    should contain only the resistance parameters (intercept + basis
    coefficients); ``alpha=0`` and ``gamma=1`` are appended automatically.

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
        Initial resistance parameter vector (intercept + basis coefficients).
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

        - ``"best_params"`` : np.ndarray -- optimized resistance parameter
          vector (without alpha/gamma).
        - ``"alpha"`` : float -- optimized PPP intensity intercept.
        - ``"gamma"`` : float -- optimized PPP connectivity coefficient.
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
    resistance_params = jnp.array(init_params, dtype=jnp.float64)
    basis_jnp = jnp.array(basis_values, dtype=jnp.float64)
    mask_jnp = jnp.array(valid_mask)
    obs_jnp = jnp.array(obs_counts, dtype=jnp.float64)

    # Append intensity params: alpha=0 (intercept), gamma=1 (coefficient)
    params = jnp.concatenate([resistance_params, jnp.array([0.0, 1.0])])

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

    # Split best_params back into resistance params and intensity params
    best_resistance_params = np.array(best_params[:-2])
    best_alpha = float(best_params[-2])
    best_gamma = float(best_params[-1])

    return {
        "best_params": best_resistance_params,
        "alpha": best_alpha,
        "gamma": best_gamma,
        "best_loglik": float(-best_loss),
        "loss_history": loss_history,
        "n_epochs_run": int(n_run),
        "elapsed": elapsed,
        "converged": converged,
    }


def run_neural_optimization(
    basis_values,
    obs_counts,
    valid_mask,
    n_rows,
    n_cols,
    cell_area=1.0,
    model_type="mlp",
    model_config=None,
    optim_config=None,
    parameterization="resistance",
    seed=42,
    verbose=True,
):
    """Run neural-network optimization of a connectivity surface.

    Creates a Flax resistance model (MLP, Conv, Spline-GAM, or IRL), defines
    a loss function that chains
    ``model -> exp(log_r) -> permeability -> GridGraph -> ResistanceDistance
    -> PPP log-likelihood``,
    and optimizes with Adam + cosine schedule + early stopping.

    Parameters
    ----------
    basis_values : array-like
        Covariate matrix of shape ``(n_valid_cells, n_covariates)`` for MLP /
        Spline / IRL, or ``(H, W, C)`` for Conv.  For non-Conv models the
        values correspond to valid cells identified by *valid_mask*.
    obs_counts : array-like
        Observed counts per valid cell, shape ``(n_valid_cells,)``.
    valid_mask : array-like
        Boolean mask of shape ``(n_rows * n_cols,)`` indicating valid cells.
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.
    cell_area : float, optional
        Area of each grid cell (default: 1.0).
    model_type : str, optional
        One of ``"mlp"``, ``"conv"``, ``"spline_gam"``, or ``"irl"``
        (default: ``"mlp"``).
    model_config : dict, optional
        Model-specific keyword arguments forwarded to the Flax module
        constructor.  Recognised keys depend on *model_type*:

        - **mlp**: ``hidden_dim`` (int, default 32), ``n_hidden_layers``
          (int, default 2).
        - **conv**: ``conv_channels`` (int, default 16),
          ``n_conv_layers`` (int, default 3).
        - **spline_gam**: ``n_knots`` (int, default 10).
        - **irl**: ``hidden_dim`` (int, default 32),
          ``n_hidden_layers`` (int, default 2).
    optim_config : dict, optional
        Optimizer settings: ``lr`` (float, default 0.01),
        ``n_epochs`` (int, default 300), ``patience`` (int, default 30).
    parameterization : str, optional
        Either ``"resistance"`` or ``"permeability"`` (default:
        ``"resistance"``).
    seed : int, optional
        Random seed for JAX PRNG (default: 42).
    verbose : bool, optional
        Whether to print progress messages (default: True).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"resistance"`` : np.ndarray, shape ``(n_valid_cells,)`` --
          optimized resistance values (natural scale).
        - ``"best_loglik"`` : float -- best log-likelihood (negated loss).
        - ``"loss_history"`` : list[float] -- per-epoch negative
          log-likelihood.
        - ``"n_epochs_run"`` : int -- number of epochs completed.
        - ``"elapsed"`` : float -- wall-clock seconds.
        - ``"model_type"`` : str -- echo of *model_type*.

    Raises
    ------
    ImportError
        If JAX, Flax, optax, or JAXScape are not installed.
    ValueError
        If *model_type* is not one of the recognised types.
    """
    # Validate early (before expensive imports) ----------------------------
    if model_config is None:
        model_config = {}
    if optim_config is None:
        optim_config = {}

    valid_types = ("mlp", "conv", "spline_gam", "irl")
    if model_type not in valid_types:
        raise ValueError(
            f"model_type must be one of {valid_types}, got '{model_type}'"
        )

    # Lazy imports -------------------------------------------------------
    import jax
    import jax.numpy as jnp
    import optax

    from .resistance import (
        ResistanceMLP,
        ResistanceConv,
        ResistanceSpline,
        ResistanceIRL,
    )
    from .core import prepare_permeability, _mean_weight, ppp_loglik

    from jaxscape import GridGraph, ResistanceDistance

    rng = jax.random.PRNGKey(seed)

    # --- Convert inputs to JAX arrays -----------------------------------
    basis_jnp = jnp.array(basis_values, dtype=jnp.float64)
    obs_jnp = jnp.array(obs_counts, dtype=jnp.float64)
    mask_jnp = jnp.array(valid_mask)

    # --- Build Flax model -----------------------------------------------
    if model_type == "mlp":
        model = ResistanceMLP(
            features=model_config.get("hidden_dim", 32),
            n_hidden=model_config.get("n_hidden_layers", 2),
        )
    elif model_type == "conv":
        model = ResistanceConv(
            channels=model_config.get("conv_channels", 16),
            n_layers=model_config.get("n_conv_layers", 3),
        )
    elif model_type == "spline_gam":
        n_cov = basis_jnp.shape[1] if basis_jnp.ndim > 1 else 1
        model = ResistanceSpline(
            n_knots=model_config.get("n_knots", 10),
            n_covariates=n_cov,
        )
    elif model_type == "irl":
        model = ResistanceIRL(
            hidden_dim=model_config.get("hidden_dim", 32),
            n_hidden=model_config.get("n_hidden_layers", 2),
        )

    # Initialise model parameters
    flax_params = model.init(rng, basis_jnp)

    # Intensity parameters: alpha (intercept), gamma (connectivity coeff)
    intensity_params = {"alpha": jnp.array(0.0), "gamma": jnp.array(1.0)}
    params = (flax_params, intensity_params)

    # --- Optimizer setup ------------------------------------------------
    lr = optim_config.get("lr", 0.01)
    n_epochs = optim_config.get("n_epochs", 300)
    patience = optim_config.get("patience", 30)

    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_epochs)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(params)

    # --- Loss function --------------------------------------------------
    # PPP negative log-likelihood with learnable alpha/gamma:
    #   log lambda = alpha + gamma * log(1 + C)
    # The Flax model outputs log-resistance; we exponentiate to get
    # resistance, embed into the full grid, convert to permeability, solve
    # for resistance distance, then evaluate the PPP objective.

    def loss_fn(all_params):
        flax_p, int_p = all_params
        log_r = model.apply(flax_p, basis_jnp)
        resistance = jnp.exp(log_r)

        # Embed valid-cell resistance into the full grid
        fill_val = jnp.mean(resistance)
        full_surface = jnp.ones(n_rows * n_cols) * fill_val
        full_surface = full_surface.at[mask_jnp].set(resistance)
        surface_2d = full_surface.reshape((n_rows, n_cols))

        # Permeability conversion and circuit solve
        perm = prepare_permeability(surface_2d, parameterization)
        grid = GridGraph(grid=perm, fun=_mean_weight)
        dist_solver = ResistanceDistance()
        source = grid.coord_to_index(jnp.array([0]), jnp.array([0]))
        connectivity = dist_solver(grid, source)

        # PPP log-likelihood on valid cells
        conn_valid = connectivity.ravel()[mask_jnp]
        loglik = ppp_loglik(
            conn_valid, obs_jnp, cell_area,
            int_p["alpha"], int_p["gamma"],
        )
        return -loglik  # minimize negative log-likelihood

    # --- Training loop --------------------------------------------------
    grad_fn = jax.grad(loss_fn)
    best_loss = float("inf")
    best_params = params
    loss_history = []
    stall = 0
    t0 = time.time()

    for epoch in range(n_epochs):
        g = grad_fn(params)
        updates, opt_state = optimizer.update(g, opt_state)
        params = optax.apply_updates(params, updates)
        loss = float(loss_fn(params))
        loss_history.append(loss)

        if loss < best_loss:
            best_loss = loss
            best_params = params
            stall = 0
        else:
            stall += 1

        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss={loss:.4f}")

        if stall >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - t0

    # --- Extract final resistance surface -------------------------------
    best_flax_params, best_intensity = best_params
    final_log_r = np.array(model.apply(best_flax_params, basis_jnp))
    resistance = np.exp(final_log_r)

    return {
        "resistance": resistance,
        "alpha": float(best_intensity["alpha"]),
        "gamma": float(best_intensity["gamma"]),
        "best_loglik": float(-best_loss),
        "loss_history": loss_history,
        "n_epochs_run": len(loss_history),
        "elapsed": elapsed,
        "model_type": model_type,
    }
