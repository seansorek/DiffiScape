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
    lower_bounds=None,
    upper_bounds=None,
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
    either bounded L-BFGS-B (via SciPy) or Adam (first-order,
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
    lower_bounds, upper_bounds : array-like, optional
        Per-resistance-parameter bounds (intercept + basis coefficients,
        same length as *init_params*). If either is None, the resistance
        parameters are left unconstrained. ``alpha``/``gamma`` (the
        intensity params appended internally) are always unconstrained
        regardless of these bounds.
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
        If JAX or Optax is not installed.
    ValueError
        If *method* is not "lbfgs" or "adam".
    """
    # Lazy imports so the module is loadable without these packages.
    import jax
    import jax.numpy as jnp
    import optax
    from scipy.optimize import minimize

    if method not in ("lbfgs", "adam"):
        raise ValueError(
            f"method must be 'lbfgs' or 'adam', got '{method}'"
        )

    # Convert inputs to JAX arrays.
    basis_jnp = jnp.array(basis_values, dtype=jnp.float64)
    if init_params is None:
        resistance_params = jnp.zeros(basis_jnp.shape[1] + 1, dtype=jnp.float64)
    else:
        resistance_params = jnp.array(init_params, dtype=jnp.float64)
    mask_jnp = jnp.array(valid_mask)
    obs_jnp = jnp.array(obs_counts, dtype=jnp.float64)

    # Append intensity params: alpha=0 (intercept), gamma=1 (coefficient)
    params = jnp.concatenate([resistance_params, jnp.array([0.0, 1.0])])

    # Pad bounds to the full [resistance..., alpha, gamma] layout, with
    # alpha/gamma always left unconstrained (+/-inf). Bounds are enforced
    # (LBFGSB / per-step clipping below) rather than just used to pick the
    # starting point, so a caller-supplied box actually constrains the fit
    # the way optimize_resistance()'s surrogate path does (GH #118).
    n_resistance = resistance_params.shape[0]
    if (lower_bounds is None) != (upper_bounds is None):
        raise ValueError("lower_bounds and upper_bounds must be supplied together")
    if lower_bounds is not None and upper_bounds is not None:
        lower_bounds, upper_bounds = np.asarray(lower_bounds), np.asarray(upper_bounds)
        if (lower_bounds.shape != (n_resistance,) or upper_bounds.shape != (n_resistance,)
                or np.isnan(lower_bounds).any() or np.isnan(upper_bounds).any()
                or np.any(lower_bounds > upper_bounds)):
            raise ValueError("bounds must match resistance parameters and lower must be <= upper")
        lo = jnp.concatenate([
            jnp.array(lower_bounds, dtype=jnp.float64),
            jnp.array([-jnp.inf, -jnp.inf]),
        ])
        hi = jnp.concatenate([
            jnp.array(upper_bounds, dtype=jnp.float64),
            jnp.array([jnp.inf, jnp.inf]),
        ])
    else:
        lo = jnp.full(n_resistance + 2, -jnp.inf, dtype=jnp.float64)
        hi = jnp.full(n_resistance + 2, jnp.inf, dtype=jnp.float64)

    def neg_loglik(p):
        return -_connectivity_objective(
            p, basis_jnp, mask_jnp, n_rows, n_cols,
            cell_area, link_fn, radius, block_size,
            parameterization, obs_jnp,
        )

    t0 = time.time()
    loss_history = []

    if method == "lbfgs":
        value_and_grad = jax.value_and_grad(neg_loglik)

        def scipy_objective(p):
            value, gradient = value_and_grad(jnp.asarray(p))
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise FloatingPointError("nonfinite objective or gradient during L-BFGS-B")
            return float(value), np.asarray(gradient, dtype=np.float64)

        result = minimize(scipy_objective, np.asarray(params), jac=True,
                          method="L-BFGS-B", bounds=list(zip(np.asarray(lo), np.asarray(hi))),
                          options={"maxiter": n_epochs, "gtol": 1e-6, "ftol": 1e-12})
        best_params = result.x
        best_loss = float(result.fun)
        loss_history = [best_loss]
        n_run = int(result.nit)
        converged = bool(result.success)

    else:  # adam
        params = jnp.clip(params, lo, hi)
        schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_epochs)
        optimizer = optax.adam(learning_rate=schedule)
        opt_state = optimizer.init(params)
        val_and_grad_fn = jax.value_and_grad(neg_loglik)

        best_loss = float("inf")
        best_params = params
        stall = 0

        for epoch in range(n_epochs):
            loss_val, g = val_and_grad_fn(params)
            loss = float(loss_val)
            loss_history.append(loss)

            # Snapshot best_params from the params that were actually
            # scored as `loss` -- *before* applying this step's update --
            # so best_params/best_loglik always describe the same point in
            # parameter space instead of one optimizer step apart.
            if loss < best_loss:
                best_loss = loss
                best_params = params
                stall = 0
            else:
                stall += 1

            updates, opt_state = optimizer.update(g, opt_state)
            params = jnp.clip(optax.apply_updates(params, updates), lo, hi)

            if stall >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        n_run = len(loss_history)
        # Converged means the loss plateaued (the same condition that
        # triggers early stopping above), not "ran out of patience budget
        # without stopping" -- which is what the inverted `stall < patience`
        # previously reported.
        converged = stall >= patience

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
    radius=13,
    block_size=5,
    seed=42,
    verbose=True,
):
    """Run neural-network optimization of a connectivity surface.

    Creates a Flax resistance model (MLP, Conv, or legacy RBF), defines
    a loss function that chains
    ``model -> exp(log_r) -> permeability -> GridGraph -> ResistanceDistance
    -> PPP log-likelihood``,
    and optimizes with Adam + cosine schedule + early stopping.

    Parameters
    ----------
    basis_values : array-like
        Covariate matrix of shape ``(n_valid_cells, n_covariates)`` for MLP /
        Spline, or ``(H, W, C)`` for Conv.  For non-Conv models the
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
        One of ``"mlp"``, ``"conv"``, ``"spline_gam"``
        (default: ``"mlp"``).
    model_config : dict, optional
        Model-specific keyword arguments forwarded to the Flax module
        constructor.  Recognised keys depend on *model_type*:

        - **mlp**: ``hidden_dim`` (int, default 32), ``n_hidden_layers``
          (int, default 2).
        - **conv**: ``conv_channels`` (int, default 16),
          ``n_conv_layers`` (int, default 3).
        - **spline_gam**: ``n_knots`` (int, default 10).
    optim_config : dict, optional
        Optimizer settings: ``lr`` (float, default 0.01),
        ``n_epochs`` (int, default 300), ``patience`` (int, default 30).
    parameterization : str, optional
        Either ``"resistance"`` or ``"permeability"`` (default:
        ``"resistance"``).
    radius : int, optional
        Moving-window buffer radius, forwarded to the same
        ``window.cumulative_current()``-based operator used on the
        forward/evaluation path (default: 13). Must match the radius used
        downstream (e.g. in ``ds_jax_connectivity()``) for the fitted
        ``gamma`` to be meaningful there -- see GH #105.
    block_size : int, optional
        Moving-window core / source-block size, forwarded to the same
        operator as *radius* above (default: 5).
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

    valid_types = ("mlp", "conv", "spline_gam")
    if model_type not in valid_types:
        raise ValueError(
            f"model_type must be one of {valid_types}, got '{model_type}'"
        )

    # Lazy imports -------------------------------------------------------
    import jax
    import jax.numpy as jnp
    import optax

    from .core import (
        prepare_permeability, ppp_loglik, cumulative_current_core,
        invalid_cell_fill_value,
    )

    rng = jax.random.PRNGKey(seed)

    # --- Convert inputs to JAX arrays -----------------------------------
    basis_jnp = jnp.array(basis_values, dtype=jnp.float64)
    obs_jnp = jnp.array(obs_counts, dtype=jnp.float64)
    mask_jnp = jnp.array(valid_mask)

    from .resistance import legacy_model_input
    model_input = legacy_model_input(basis_jnp, mask_jnp, n_rows, n_cols, model_type)
    model_args = ()

    # --- Build Flax model -----------------------------------------------
    from .resistance import create_model
    model = create_model(model_type, model_config, basis_jnp.shape[-1])

    # Initialise model parameters
    flax_params = model.init(rng, model_input, *model_args)

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
    # resistance, embed into the full grid, convert to permeability, and
    # solve via the SAME differentiable moving-window operator used on the
    # forward/evaluation path (window.cumulative_current(), honoring the
    # caller's radius/block_size) before evaluating the PPP objective --
    # see GH #105 for why this must match the forward path exactly rather
    # than a single-source ResistanceDistance call.

    # Convolutional models return one value per full-grid cell.
    is_full_grid = model_type == "conv"

    def loss_fn(all_params):
        flax_p, int_p = all_params
        log_r = model.apply(flax_p, model_input, *model_args)
        resistance = jnp.exp(log_r)

        if is_full_grid:
            # Conv outputs all cells (H*W,) from full-grid input.
            # Invalid cells get the barrier fill (GH #134), matching the
            # forward path's no-data convention -- not the model's learned
            # resistance for those cells.
            fill_val = invalid_cell_fill_value(parameterization)
            full_surface = jnp.where(mask_jnp, resistance, fill_val)
            surface_2d = full_surface.reshape((n_rows, n_cols))
        else:
            # Other models output valid cells only; embed into full grid.
            # Invalid cells get the barrier fill (GH #134), matching the
            # forward path's no-data convention -- not the mean.
            fill_val = invalid_cell_fill_value(parameterization)
            full_surface = jnp.ones(n_rows * n_cols) * fill_val
            full_surface = full_surface.at[mask_jnp].set(resistance)
            surface_2d = full_surface.reshape((n_rows, n_cols))

        # Permeability conversion and moving-window circuit solve
        perm = prepare_permeability(surface_2d, parameterization)
        connectivity = cumulative_current_core(
            perm, n_rows, n_cols, radius, block_size
        )

        # PPP log-likelihood on valid cells
        conn_valid = connectivity.ravel()[mask_jnp]
        loglik = ppp_loglik(
            conn_valid, obs_jnp, cell_area,
            int_p["alpha"], int_p["gamma"],
        )
        return -loglik  # minimize negative log-likelihood

    # --- Training loop --------------------------------------------------
    val_and_grad_fn = jax.value_and_grad(loss_fn)
    best_loss = float("inf")
    best_params = params
    loss_history = []
    stall = 0
    t0 = time.time()

    for epoch in range(n_epochs):
        loss_val, g = val_and_grad_fn(params)
        loss = float(loss_val)
        if not np.isfinite(loss) or not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(g)):
            raise FloatingPointError("nonfinite neural objective or gradient")
        loss_history.append(loss)

        # Snapshot best_params from the params that were actually scored as
        # `loss` -- *before* applying this step's update -- so best_params/
        # best_loglik always describe the same point in parameter space
        # (see run_optimization's Adam branch for the same fix).
        if loss < best_loss:
            best_loss = loss
            best_params = params
            stall = 0
        else:
            stall += 1

        updates, opt_state = optimizer.update(g, opt_state)
        params = optax.apply_updates(params, updates)

        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss={loss:.4f}")

        if stall >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - t0

    # --- Extract final resistance surface -------------------------------
    best_flax_params, best_intensity = best_params
    final_log_r = np.array(model.apply(best_flax_params, model_input, *model_args))
    resistance = np.exp(final_log_r)
    if is_full_grid:
        resistance = resistance[np.asarray(valid_mask)]

    return {
        "resistance": resistance,
        "model_state": jax.tree_util.tree_map(np.asarray, best_flax_params),
        "model_config": dict(model_config),
        "alpha": float(best_intensity["alpha"]),
        "gamma": float(best_intensity["gamma"]),
        "best_loglik": float(-best_loss),
        "loss_history": loss_history,
        "n_epochs_run": len(loss_history),
        "elapsed": elapsed,
        "model_type": model_type,
        "converged": stall >= patience,
    }
