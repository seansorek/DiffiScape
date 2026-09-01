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
    if lower_bounds is not None and upper_bounds is not None:
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
        lbfgs_tol = 1e-6
        solver = jaxopt.LBFGSB(fun=neg_loglik, maxiter=n_epochs, tol=lbfgs_tol, jit=False)
        result = solver.run(params, bounds=(lo, hi))
        best_params = result.params
        best_loss = float(neg_loglik(best_params))
        loss_history = [best_loss]
        n_run = int(result.state.iter_num) if hasattr(result.state, "iter_num") else n_epochs
        # jaxopt reports the optimality residual (gradient norm) in
        # state.error; the solver only actually converged if that residual
        # is at or below its own stopping tolerance, not unconditionally.
        converged = bool(result.state.error <= lbfgs_tol)

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
          ``n_hidden_layers`` (int, default 2), ``beta`` (float,
          default 1.0), ``gamma_d`` (float, default 0.9), and
          ``n_value_iter`` (int, default 60) -- the soft value-iteration
          temperature, discount factor, and sweep count.  A 4-neighbour
          grid adjacency is built internally and passed to the model so
          value iteration actually runs (see GH #127); without it these
          three keys would be silently dead.
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

    valid_types = ("mlp", "conv", "spline_gam", "irl")
    if model_type not in valid_types:
        raise ValueError(
            f"model_type must be one of {valid_types}, got '{model_type}'"
        )

    # Lazy imports -------------------------------------------------------
    import jax
    import jax.numpy as jnp
    import optax
    from jax.experimental.sparse import BCOO

    from .resistance import (
        ResistanceMLP,
        ResistanceConv,
        ResistanceSpline,
        ResistanceIRL,
    )
    from .core import (
        prepare_permeability, ppp_loglik, cumulative_current_core,
        GridGraph, _mean_weight,
    )

    rng = jax.random.PRNGKey(seed)

    # --- Convert inputs to JAX arrays -----------------------------------
    basis_jnp = jnp.array(basis_values, dtype=jnp.float64)
    obs_jnp = jnp.array(obs_counts, dtype=jnp.float64)
    mask_jnp = jnp.array(valid_mask)

    # IRL's soft value iteration needs a reward defined at every grid node
    # (not just valid cells) so it can propagate value across neighbours,
    # plus the 4-neighbour grid adjacency describing that neighbourhood
    # structure. Neither of these existed before GH #127: model.init /
    # model.apply were called with the valid-cells-only basis and no
    # adjacency at all, so ResistanceIRL's `adjacency is None` branch
    # always fired and value iteration never ran. Invalid cells are filled
    # with the per-covariate mean of the valid cells (same convention used
    # for the non-conv resistance embedding below).
    adjacency = None
    if model_type == "irl":
        n_cov = basis_jnp.shape[1] if basis_jnp.ndim > 1 else 1
        basis_2d = basis_jnp.reshape(basis_jnp.shape[0], n_cov)
        fill_val = jnp.mean(basis_2d, axis=0, keepdims=True)
        basis_full = jnp.tile(fill_val, (n_rows * n_cols, 1))
        basis_full = basis_full.at[mask_jnp].set(basis_2d)
        model_input = basis_full
        # 4-neighbour (rook contiguity) lattice adjacency -- structural
        # only, so a uniform grid is used regardless of the (not-yet-fit)
        # resistance surface. Row-normalised to a stochastic transition
        # matrix (each row sums to 1) so `adjacency @ value` in
        # ResistanceIRL's value iteration is an AVERAGE over neighbours,
        # matching the single "move to a neighbour" transition the model's
        # Bellman update assumes. An unnormalised 0/1 adjacency instead
        # sums 2-4 neighbour values per step, which combined with
        # `gamma_d` close to 1 diverges within a handful of sweeps.
        raw_adjacency = GridGraph(
            grid=jnp.ones((n_rows, n_cols)), fun=_mean_weight
        ).get_adjacency_matrix()
        degree = jnp.zeros(n_rows * n_cols).at[raw_adjacency.indices[:, 0]].add(
            raw_adjacency.data
        )
        degree = jnp.where(degree > 0, degree, 1.0)
        normalized_data = raw_adjacency.data / degree[raw_adjacency.indices[:, 0]]
        adjacency = BCOO(
            (normalized_data, raw_adjacency.indices), shape=raw_adjacency.shape
        )
    else:
        model_input = basis_jnp

    model_args = (adjacency,) if model_type == "irl" else ()

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
            beta=model_config.get("beta", 1.0),
            gamma_d=model_config.get("gamma_d", 0.9),
            n_value_iter=model_config.get("n_value_iter", 60),
        )

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

    # Conv operates on a full-grid (H, W, C) input and IRL needs reward
    # defined at every node for value iteration to propagate across
    # neighbours (see model_input/adjacency setup above) -- both therefore
    # output one value per grid cell already, unlike mlp/spline_gam which
    # output one value per *valid* cell only.
    is_full_grid = model_type in ("conv", "irl")

    def loss_fn(all_params):
        flax_p, int_p = all_params
        log_r = model.apply(flax_p, model_input, *model_args)
        resistance = jnp.exp(log_r)

        if is_full_grid:
            # Conv/IRL output all cells (H*W,) from full-grid input
            surface_2d = resistance.reshape((n_rows, n_cols))
        else:
            # Other models output valid cells only; embed into full grid
            fill_val = jnp.mean(resistance)
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

    return {
        "resistance": resistance,
        "alpha": float(best_intensity["alpha"]),
        "gamma": float(best_intensity["gamma"]),
        "best_loglik": float(-best_loss),
        "loss_history": loss_history,
        "n_epochs_run": len(loss_history),
        "elapsed": elapsed,
        "model_type": model_type,
        "converged": stall >= patience,
    }
