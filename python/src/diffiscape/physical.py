"""Complete fitting, prediction, and NumPyro operations for physical currents."""
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
from scipy.optimize import minimize

from .core import physical_solve_init, physical_solve, focal_current
from .physical_models import create_physical_model, smoothing_penalty, IntensityMLP
from .splines import design_matrix, knots, differentiable_basis


def _prepare(basis, mask, cfg):
    model = create_physical_model(cfg["model_type"], basis.shape[1], cfg["model_config"])
    basis_jax = jnp.asarray(basis, dtype=jnp.float64)
    args = (basis_jax,)
    if cfg["model_type"] == "spline_gam":
        settings = cfg["model_config"]
        marginal, tensor = design_matrix(basis, settings.get("n_knots", 10), settings.get("spline_degree", 3), settings.get("include_interactions", True))
        args += (jnp.asarray(marginal), jnp.asarray(tensor))
    elif cfg["model_type"] == "conv":
        grid = np.zeros((mask.size, basis.shape[1]))
        grid[mask] = basis
        args = (jnp.asarray(grid.reshape(cfg["n_rows"], cfg["n_cols"], -1)), mask, basis_jax)
    parameters = model.init(jax.random.PRNGKey(cfg["seed"]), *args)
    state = None if cfg["formulation"] == "focal_current" else physical_solve_init(
        jnp.full((cfg["n_rows"], cfg["n_cols"]), 0.05), cfg["formulation"], cfg["absorption"], cfg.get("cg_tol", 1e-10))
    return model, args, parameters, state


def _evaluate(parameters, model, args, mask, cfg, state, epoch=0, training=False):
    model_args = {}
    if cfg["model_type"] == "conv":
        model_args = dict(deterministic=not training, rngs={"dropout": jax.random.fold_in(jax.random.PRNGKey(cfg["seed"]), epoch)})
    log_r, raw = model.apply(parameters["resistance"], *args, return_raw=True, **model_args)
    resistance = jnp.exp(log_r)
    # Torch's masked cells are finite 1e6-resistance barriers.
    full = jnp.full(mask.size, 1e6).at[mask].set(resistance)
    permeability = 1 / full.reshape(cfg["n_rows"], cfg["n_cols"])
    if cfg["formulation"] == "focal_current":
        current = focal_current(permeability, cfg["radius"], cfg["block_size"], cfg["source_from_resistance"]).ravel()
        voltage = None
    else:
        current, voltage = physical_solve(permeability, cfg["formulation"], cfg["absorption"], cfg["source_spacing"], cfg["source_from_resistance"], state)
    gamma = jax.nn.softplus(parameters["raw_gamma"])
    log_connectivity = jnp.log1p(jnp.maximum(current[mask], 0))
    settings = cfg["model_config"]
    if settings.get("intensity_spline", False):
        t = knots(settings.get("intensity_n_knots", 5), settings.get("intensity_degree", 3), 0, settings.get("intensity_log1p_max", 10.0))
        contribution = differentiable_basis(log_connectivity, t, settings.get("intensity_degree", 3)) @ parameters["intensity_coefs"]
        log_intensity = parameters["alpha"] + contribution - jnp.mean(contribution)
    elif settings.get("intensity_hidden", 0):
        contribution = IntensityMLP(settings["intensity_hidden"]).apply(parameters["intensity"], log_connectivity[:, None])
        log_intensity = parameters["alpha"] + gamma * contribution
    else:
        log_intensity = parameters["alpha"] + gamma * log_connectivity
    penalty = mask.sum() * cfg["reg_mean"] * (jnp.mean(raw) - cfg["log_R_baseline"]) ** 2
    if cfg["model_type"] == "spline_gam":
        penalty += cfg["penalty_scale"] * smoothing_penalty(parameters["resistance"], cfg["model_config"].get("lambda_min", 0.0))
        if settings.get("intensity_spline", False):
            penalty += cfg["penalty_scale"] * (settings.get("lambda_min", 0.0) + jnp.exp(parameters["log_lambda_intensity"])) * jnp.sum(jnp.diff(parameters["intensity_coefs"], n=2) ** 2)
    else:
        variance = jnp.var(raw, ddof=1) if raw.size > 1 else jnp.array(0.0)
        penalty += mask.sum() * cfg["reg_var"] * jax.nn.softplus(jnp.log(cfg["target_logR_var"]) - jnp.log(variance + 0.01))
        skip = parameters["resistance"]["params"]["skip"]["kernel"]
        penalty += mask.sum() * cfg["reg_skip"] / (optax.safe_norm(skip, 0.0) + 0.01)
    return resistance, current, voltage, log_intensity, penalty


def fit_physical(basis, counts, mask, n_rows, n_cols, cell_area=1.0,
                 model_type="loglinear", model_config=None, formulation="global_absorption",
                 source_spacing=1, source_from_resistance=True, absorption=0.01,
                 radius=13, block_size=5, method="adam", lr=0.01, n_epochs=300,
                 patience=30, grad_clip=10.0, warmup_epochs=10, seed=42, verbose=True,
                 reg_mean=1.0, reg_var=0.1, target_logR_var=1.0, reg_skip=0.1,
                 log_R_baseline=3.0, penalty_scale=1.0, init_params=None,
                 lower_bounds=None, upper_bounds=None, cg_tol=1e-10):
    """Fit the physical-current PPP, preserving the default log-linear objective."""
    if formulation not in ("global_absorption", "boundary", "focal_current"):
        raise ValueError("unknown physical connectivity formulation")
    if method not in ("adam", "lbfgs"):
        raise ValueError("method must be adam or lbfgs")
    if any(not isinstance(n, (int, np.integer)) or n < 1 for n in (source_spacing, radius, block_size, n_epochs, patience)):
        raise ValueError("spacing, windows, n_epochs, and patience must be positive integers")
    if not np.isfinite(absorption) or absorption <= 0 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("absorption and lr must be positive and finite")
    if not isinstance(warmup_epochs, (int, np.integer)) or warmup_epochs < 0 or not np.isfinite(grad_clip) or grad_clip <= 0 or not np.isfinite(cg_tol) or cg_tol <= 0:
        raise ValueError("warmup_epochs must be a nonnegative integer; grad_clip and cg_tol must be positive")
    if any(not np.isfinite(x) or x < 0 for x in (reg_mean, reg_var, reg_skip, penalty_scale)) or target_logR_var <= 0:
        raise ValueError("regularization weights must be nonnegative; target variance must be positive")
    cfg = dict(model_type=model_type, model_config=dict(model_config or {}), formulation=formulation,
        n_rows=n_rows, n_cols=n_cols, valid_mask=mask.copy(), n_covariates=basis.shape[1], cell_area=cell_area,
        source_spacing=source_spacing, source_from_resistance=source_from_resistance, absorption=absorption,
        radius=radius, block_size=block_size, cg_tol=cg_tol, reg_mean=reg_mean, reg_var=reg_var, reg_skip=reg_skip,
        target_logR_var=target_logR_var, log_R_baseline=log_R_baseline, penalty_scale=penalty_scale, seed=seed)
    allowed = {
        "loglinear": set(),
        "mlp": {"hidden_dim", "n_hidden_layers"},
        "conv": {"hidden_dim", "n_hidden_layers", "conv_channels", "n_conv_layers", "conv_kernel_size", "use_dilated", "dropout", "intensity_hidden"},
        "spline_gam": {"n_knots", "spline_degree", "include_interactions", "lambda_init_marginal", "lambda_init_interaction", "lambda_min", "intensity_spline", "intensity_n_knots", "intensity_degree", "lambda_init_intensity", "intensity_log1p_max"},
    }
    if model_type not in allowed or (unknown := set(cfg["model_config"]) - allowed[model_type]):
        raise ValueError(f"unsupported model type or model_config keys: {model_type}, {sorted(unknown) if model_type in allowed else []}")
    model, args, resistance_params, state = _prepare(basis, mask, cfg)
    if init_params is not None:
        if model_type != "loglinear" or np.asarray(init_params).shape != (basis.shape[1] + 1,):
            raise ValueError("init_params is a loglinear [intercept, coefficients...] vector")
        resistance_params["params"]["skip"]["bias"] = jnp.asarray(init_params[:1])
        resistance_params["params"]["skip"]["kernel"] = jnp.asarray(init_params[1:]).reshape(-1, 1)
    params = dict(resistance=resistance_params, alpha=jnp.array(0.0), raw_gamma=jnp.array(np.log(np.expm1(1.0))))
    settings = cfg["model_config"]
    if settings.get("intensity_spline", False):
        n_basis = settings.get("intensity_n_knots", 5) + settings.get("intensity_degree", 3) + 1
        params["intensity_coefs"] = jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(seed), 1), (n_basis,), dtype=jnp.float64) * 0.01
        params["log_lambda_intensity"] = jnp.array(settings.get("lambda_init_intensity", 2.0))
    elif settings.get("intensity_hidden", 0):
        params["intensity"] = IntensityMLP(settings["intensity_hidden"]).init(jax.random.fold_in(jax.random.PRNGKey(seed), 1), jnp.zeros((basis.shape[0], 1)))
    flat, unflatten = ravel_pytree(params)
    lower, upper = np.full(flat.size, -np.inf), np.full(flat.size, np.inf)
    if lower_bounds is not None or upper_bounds is not None:
        if model_type != "loglinear" or lower_bounds is None or upper_bounds is None:
            raise ValueError("box bounds require a loglinear model and both lower_bounds and upper_bounds")
        lo, hi = np.asarray(lower_bounds), np.asarray(upper_bounds)
        if lo.shape != (basis.shape[1] + 1,) or hi.shape != lo.shape or np.isnan(lo).any() or np.isnan(hi).any() or (lo > hi).any():
            raise ValueError("bounds must match [intercept, coefficients...] and lower <= upper")
        # The flattened PyTree follows sorted dict keys: alpha, raw_gamma, resistance.
        lower[2:3], upper[2:3] = lo[:1], hi[:1]
        lower[3:], upper[3:] = lo[1:], hi[1:]
    obs = jnp.asarray(counts)
    def objective(p, epoch):
        _, _, _, log_lambda, penalty = _evaluate(unflatten(p), model, args, mask, cfg, state, epoch, method == "adam")
        return -jnp.sum(obs * log_lambda) + jnp.sum(jnp.exp(log_lambda) * cell_area) + penalty
    value_grad = jax.jit(jax.value_and_grad(objective))
    flat = jnp.clip(flat, lower, upper)
    best, best_loss, history, stall = flat, float("inf"), [], 0
    started = time.time()
    if method == "lbfgs":
        if cfg["model_config"].get("dropout", 0):
            raise ValueError("L-BFGS-B requires dropout=0")
        def evaluate(p):
            value, gradient = value_grad(jnp.asarray(p), 0)
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise FloatingPointError("nonfinite objective or gradient during physical L-BFGS-B fitting")
            return float(value), np.asarray(gradient, dtype=np.float64)
        fitted = minimize(evaluate, np.asarray(flat), jac=True, method="L-BFGS-B",
            bounds=list(zip(lower, upper)), options={"maxiter": n_epochs, "gtol": 1e-6, "ftol": 1e-12})
        best, best_loss, history, n_run, converged = fitted.x, float(fitted.fun), [float(fitted.fun)], int(fitted.nit), bool(fitted.success)
    else:
        warmup = min(warmup_epochs, n_epochs - 1)
        cosine = optax.cosine_decay_schedule(lr, max(n_epochs - warmup, 1), alpha=0.01)
        schedule = optax.join_schedules([optax.linear_schedule(lr * 0.1, lr, max(warmup, 1)), cosine], [warmup]) if warmup else cosine
        optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(schedule))
        optimizer_state = optimizer.init(flat)
        for epoch in range(n_epochs):
            value, gradient = value_grad(flat, epoch)
            loss = float(value)
            if not np.isfinite(loss) or not np.isfinite(gradient).all():
                raise FloatingPointError("nonfinite objective or gradient during physical fitting")
            history.append(loss)
            if loss < best_loss:
                best, best_loss, stall = flat, loss, 0
            else:
                stall += 1
            if stall >= patience:
                break
            updates, optimizer_state = optimizer.update(gradient, optimizer_state, flat)
            flat = jnp.clip(optax.apply_updates(flat, updates), lower, upper)
        n_run, converged = len(history), stall >= patience
    best_tree = unflatten(jnp.asarray(best))
    resistance, current, voltage, log_lambda, _ = _evaluate(best_tree, model, args, mask, cfg, state)
    result = dict(model_state=jax.tree_util.tree_map(np.asarray, best_tree), configuration=cfg,
        resistance=np.asarray(resistance), current=np.asarray(current), voltage=None if voltage is None else np.asarray(voltage),
        intensity=np.asarray(jnp.exp(log_lambda)), alpha=float(best_tree["alpha"]), gamma=float(jax.nn.softplus(best_tree["raw_gamma"])),
        best_loglik=float(jnp.sum(obs * log_lambda) - jnp.sum(jnp.exp(log_lambda) * cell_area)), best_loss=best_loss,
        loss_history=history, n_epochs_run=n_run, converged=converged, elapsed=time.time() - started, model_type=model_type,
        n_params=int(flat.size) - int(settings.get("intensity_spline", False)), convergence_message=("loss plateaued" if converged else "iteration budget exhausted") if method == "adam" else str(fitted.message))
    skip = best_tree["resistance"]["params"]["skip"]
    intercept = np.asarray(skip["bias"]).ravel()[0]
    if model_type == "spline_gam":
        intercept += float(best_tree["resistance"]["params"]["intercept"])
    result["effective_loglinear"] = np.concatenate([[intercept], np.asarray(skip["kernel"]).ravel()])
    if model_type == "spline_gam":
        from .physical_models import partial_effects
        effects = partial_effects(result)
        result.update(effects)
        endpoints, _ = design_matrix(np.array([[0.0], [1.0]]), settings.get("n_knots", 10), settings.get("spline_degree", 3), False)
        result["effective_loglinear"][1:] += np.asarray(best_tree["resistance"]["params"]["spline_coefs"]) @ (endpoints[1, 0] - endpoints[0, 0])
    if model_type == "loglinear":
        skip = best_tree["resistance"]["params"]["skip"]
        result["best_params"] = np.concatenate([np.asarray(skip["bias"]), np.asarray(skip["kernel"]).ravel()])
    return result


def predict_physical(result, basis):
    cfg, params = result["configuration"], result["model_state"]
    mask = np.asarray(cfg["valid_mask"], dtype=bool)
    model, args, _, state = _prepare(basis, mask, cfg)
    resistance, current, voltage, log_lambda, _ = _evaluate(params, model, args, mask, cfg, state)
    shape = (cfg["n_rows"], cfg["n_cols"])
    surface = np.full(mask.size, np.nan)
    surface[mask] = np.asarray(resistance)
    intensity = np.full(mask.size, np.nan)
    intensity[mask] = np.exp(np.asarray(log_lambda))
    return dict(surface=surface.reshape(shape), connectivity=np.where(mask, np.asarray(current), np.nan).reshape(shape),
                intensity=intensity.reshape(shape), voltage=None if voltage is None else np.where(mask, np.asarray(voltage), np.nan).reshape(shape))


def _posterior_model(result, basis, counts, fix_smoothing=True, fix_intensity=False):
    """Proper priors approved for migration; the penalized legacy target was improper."""
    import numpyro
    import numpyro.distributions as dist

    cfg = result["configuration"]
    mask = np.asarray(cfg["valid_mask"], dtype=bool)
    model, args, _, state = _prepare(basis, mask, cfg)
    params = jax.tree_util.tree_map(jnp.asarray, result["model_state"])
    paths_leaves, structure = jax.tree_util.tree_flatten_with_path(params)
    paths = [[entry.key for entry in path] for path, _ in paths_leaves]
    leaves = [leaf for _, leaf in paths_leaves]
    free = [not ((fix_smoothing and any(str(key).startswith("log_lambda") for key in path)) or
                 (fix_intensity and path[0] in ("alpha", "raw_gamma", "intensity", "intensity_coefs")) or
                 (cfg["model_config"].get("intensity_spline", False) and path[0] == "raw_gamma")) for path in paths]
    flat_init, unpack_free = ravel_pytree(tuple(leaf for leaf, active in zip(leaves, free) if active))
    means = [jnp.full_like(leaf, np.log(np.expm1(1.0)) if path[0] == "raw_gamma" else 0.0)
             for path, leaf, active in zip(paths, leaves, free) if active]
    scales = [jnp.full_like(leaf, 1.0 if path[0] == "raw_gamma" else 2.0)
              for path, leaf, active in zip(paths, leaves, free) if active]
    prior_mean, _ = ravel_pytree(tuple(means))
    prior_scale, _ = ravel_pytree(tuple(scales))
    def unpack(vector):
        active_values = iter(unpack_free(vector))
        return jax.tree_util.tree_unflatten(structure, [next(active_values) if active else leaf for leaf, active in zip(leaves, free)])
    def posterior():
        vector = numpyro.sample("parameters", dist.Normal(prior_mean, prior_scale).to_event(1))
        p = unpack(vector)
        _, _, _, log_lambda, penalty = _evaluate(p, model, args, mask, cfg, state)
        loglik = jnp.sum(jnp.asarray(counts) * log_lambda) - jnp.sum(jnp.exp(log_lambda) * cfg["cell_area"])
        numpyro.factor("loglik", loglik)
        numpyro.factor("regularization", -penalty)
        numpyro.deterministic("alpha", p["alpha"])
        numpyro.deterministic("gamma", jax.nn.softplus(p["raw_gamma"]))
    return posterior, flat_init, unpack


def sample_physical(result, basis, counts, method="nuts", n_samples=1000, warmup=1000,
                    max_treedepth=10, target_accept=0.8, max_iter=2000, lr=0.01,
                    seed=42, fix_smoothing=True, fix_intensity=False):
    """Sample the shared physical model with NumPyro, including proper priors."""
    import numpyro
    from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive, init_to_value
    from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
    from numpyro.diagnostics import summary
    from .sampling import _summarize_samples

    if method not in ("nuts", "advi"):
        raise ValueError("method must be nuts or advi")
    if any(not isinstance(n, (int, np.integer)) or n < 1 for n in (n_samples, warmup, max_treedepth, max_iter)):
        raise ValueError("sampling and iteration counts must be positive integers")
    if not 0 < target_accept < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("target_accept must be between 0 and 1 and lr must be positive")
    model, initial, unpack = _posterior_model(result, basis, counts, fix_smoothing, fix_intensity)
    rng = jax.random.PRNGKey(seed)
    started = time.time()
    if method == "nuts":
        kernel = NUTS(model, max_tree_depth=max_treedepth, target_accept_prob=target_accept,
                       init_strategy=init_to_value(values={"parameters": initial}))
        inference = MCMC(kernel, num_warmup=warmup, num_samples=n_samples, progress_bar=False)
        inference.run(rng, extra_fields=("diverging",))
        draws = inference.get_samples()
        diagnostics = summary(inference.get_samples(group_by_chain=True), group_by_chain=True) if n_samples >= 4 else None
        details = dict(n_divergences=int(np.asarray(inference.get_extra_fields()["diverging"]).sum()), diagnostics=diagnostics)
    else:
        guide = AutoLowRankMultivariateNormal(model, init_loc_fn=init_to_value(values={"parameters": initial}))
        inference = SVI(model, guide, numpyro.optim.Adam(lr), Trace_ELBO())
        fitted = inference.run(rng, max_iter, progress_bar=False)
        losses = np.asarray(fitted.losses)
        if not np.isfinite(losses).all():
            raise FloatingPointError("nonfinite variational loss")
        draws = Predictive(model, guide=guide, params=fitted.params, num_samples=n_samples,
                           return_sites=["parameters", "alpha", "gamma"])(jax.random.fold_in(rng, 1))
        window = losses[-min(50, len(losses)):]
        converged = len(window) >= 10 and np.std(window) < 1e-3 * max(abs(np.mean(window)), 1.0)
        details = dict(n_divergences=0, diagnostics=None, best_elbo=float(-losses[-1]), converged=bool(converged), loss_history=losses)
    arrays = {key: np.asarray(value) for key, value in draws.items() if key in ("parameters", "alpha", "gamma")}
    trees = jax.vmap(unpack)(jnp.asarray(arrays["parameters"]))
    return dict(samples=arrays, parameter_draws=jax.tree_util.tree_map(np.asarray, trees),
                summary=_summarize_samples(arrays), elapsed=time.time() - started,
                configuration=result["configuration"], **details)


def verify_gradient(basis, mask, n_rows, n_cols, model_type="loglinear", model_config=None,
                    formulation="global_absorption", absorption=0.01, radius=3, block_size=2,
                    source_spacing=1, seed=42, n_check=12, epsilon=1e-5):
    """Finite differences of the same full objective used by physical fitting."""
    mask = np.asarray(mask, dtype=bool)
    counts = np.linspace(0.5, 2, mask.sum())
    fitted = fit_physical(np.asarray(basis), counts, mask, n_rows, n_cols,
        model_type=model_type, model_config=model_config, formulation=formulation,
        absorption=absorption, radius=radius, block_size=block_size, source_spacing=source_spacing,
        n_epochs=1, seed=seed, verbose=False)
    cfg = fitted["configuration"]
    model, args, _, state = _prepare(np.asarray(basis), mask, cfg)
    flat, unpack = ravel_pytree(fitted["model_state"])
    def objective(vector):
        log_lambda, penalty = _evaluate(unpack(vector), model, args, mask, cfg, state)[3:]
        return -jnp.sum(counts * log_lambda) + jnp.sum(jnp.exp(log_lambda)) + penalty
    evaluate = jax.jit(objective)
    derivative = np.asarray(jax.jit(jax.grad(objective))(flat))
    indices = np.random.default_rng(seed).choice(len(flat), min(n_check, len(flat)), replace=False)
    differences = []
    for index in indices:
        step = jnp.zeros_like(flat).at[index].set(epsilon)
        differences.append(float((evaluate(flat + step) - evaluate(flat - step)) / (2 * epsilon)))
    analytic, finite = derivative[indices], np.asarray(differences)
    absolute = np.max(np.abs(analytic - finite))
    relative = np.max(np.abs(analytic - finite) / np.maximum(1.0, np.maximum(np.abs(analytic), np.abs(finite))))
    return {"pass": bool(np.isfinite(relative) and relative < 1e-3), "max_rel_error": float(relative),
            "max_abs_error": float(absolute), "analytic": analytic, "finite_difference": finite}
