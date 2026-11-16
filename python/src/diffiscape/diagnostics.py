"""Shared numerical summaries; plotting and spatial metadata stay in R."""
import numpy as np
import jax
import jax.numpy as jnp

from .likelihoods import deviance_residuals


def simulate_counts(expected, family="poisson", size=1.0, pi=0.0, seed=42, n_sim=1):
    """Aggregate PPP mixtures: one shared Gamma rate and zero state per replicate."""
    mu = np.asarray(expected, dtype=np.float64)
    if mu.ndim != 1 or not np.isfinite(mu).all() or (mu < 0).any():
        raise ValueError("expected must contain finite nonnegative cell means")
    if family not in ("poisson", "negbin", "zinb"):
        raise ValueError("predictive counts require poisson, negbin, or zinb")
    if not np.isfinite(size) or size <= 0 or not 0 <= pi <= 1 or not isinstance(n_sim, (int, np.integer)) or n_sim < 1:
        raise ValueError("size must be positive, pi in [0,1], and n_sim a positive integer")
    rng = np.random.default_rng(seed)
    rate = np.ones(n_sim) if family == "poisson" else rng.gamma(size, 1 / size, n_sim)
    if family == "zinb":
        rate *= rng.random(n_sim) >= pi
    return rng.poisson(rate[:, None] * mu)


def information_criteria(loglik, k, n):
    aic = -2 * loglik + 2 * k
    return dict(AIC=aic, BIC=-2 * loglik + np.log(n) * k,
                AICc=aic + 2 * k * (k + 1) / max(n - k - 1, 1))


def posterior_summary(samples, prob=0.95):
    if not 0 < prob < 1:
        raise ValueError("prob must be between zero and one")
    alpha = (1 - prob) / 2
    output = {name: [] for name in ("parameter", "mean", "sd", "lower", "median", "upper")}
    for name, values in samples.items():
        values = np.asarray(values, dtype=np.float64)
        if name == "loglik":
            continue
        if values.ndim != 1:
            raise ValueError("each posterior parameter must be a one-dimensional draw vector")
        values = values[~np.isnan(values)]
        output["parameter"].append(name)
        output["mean"].append(np.mean(values) if values.size else np.nan)
        output["sd"].append(np.std(values, ddof=1) if values.size > 1 else np.nan)
        quantiles = np.quantile(values, [alpha, 0.5, 1 - alpha]) if values.size else [np.nan] * 3
        for key, value in zip(("lower", "median", "upper"), quantiles):
            output[key].append(value)
    return {name: np.asarray(values) for name, values in output.items()}


def posterior_predictive(fitted, basis_values, posterior, seed=42):
    """PPP replicated cell counts from the same fitted physical model and mask."""
    from .physical import _prepare, _evaluate
    cfg = fitted["configuration"]
    mask = np.asarray(cfg["valid_mask"], dtype=bool)
    model, args, _, state = _prepare(np.asarray(basis_values), mask, cfg)
    parameters = jax.tree_util.tree_map(jnp.asarray, posterior["parameter_draws"])
    def mean_counts(p):
        log_lambda = _evaluate(p, model, args, mask, cfg, state)[3]
        return jnp.exp(log_lambda) * cfg["cell_area"]
    expected = jax.vmap(mean_counts)(parameters)
    replicated = jax.random.poisson(jax.random.PRNGKey(seed), expected)
    return dict(expected_counts=np.asarray(expected), replicated_counts=np.asarray(replicated), valid_mask=mask)


def diagnose_physical(fitted, observed_counts):
    counts = np.asarray(observed_counts, dtype=np.float64)
    expected = np.asarray(fitted["intensity"]) * fitted["configuration"]["cell_area"]
    if counts.shape != expected.shape or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("observed_counts must match valid fitted cells")
    residuals = np.asarray(deviance_residuals(counts, expected, family="poisson"))
    return dict(residuals=residuals, mean_deviance=float(np.mean(np.abs(residuals))),
                prop_large=float(np.mean(np.abs(residuals) > 2)),
                information_criteria=information_criteria(fitted["best_loglik"], fitted["n_params"], float(counts.sum())))


def count_test_quantities(counts, expected, names=("total_count", "vmi_ratio", "mean_deviance"), family="poisson", size=1.0):
    y, mu = np.asarray(counts), np.asarray(expected)
    output = {}
    for name in names:
        if name == "total_count":
            output[name] = float(y.sum())
        elif name == "vmi_ratio":
            output[name] = float(np.var(y, ddof=1) / max(float(y.mean()), 1e-10)) if y.size > 1 else np.nan
        elif name == "mean_deviance":
            valid = (mu > 0) & np.isfinite(mu)
            output[name] = float(np.mean(np.abs(np.asarray(deviance_residuals(y[valid], mu[valid], family, size))))) if valid.any() else np.nan
        else:
            raise ValueError(f"unsupported count test quantity: {name}")
    return output


def intensity_predictive_checks(z, covariates, observed_counts, map_parameters, draws,
                                cell_area=1.0, family="negbin", seed=42,
                                test_quantities=("total_count", "vmi_ratio", "mean_deviance")):
    """Predictive statistics for the shared built-in count intensity families."""
    from .likelihoods import compute_intensity
    z = np.asarray(z, dtype=np.float64)
    counts = np.asarray(observed_counts)
    if z.ndim != 1 or counts.shape != z.shape or not np.isfinite(z).all() or not np.isfinite(counts).all() or (counts < 0).any() or not np.isfinite(cell_area) or cell_area <= 0:
        raise ValueError("predictive checks require matching finite cell vectors and positive cell_area")
    def means(parameters):
        betas = {name: parameters.get("beta_" + name, 0) for name in covariates}
        return np.asarray(compute_intensity(z, parameters.get("alpha", 0), parameters["gamma"], covariates, betas)) * cell_area
    observed = count_test_quantities(observed_counts,
        means(map_parameters) * (1 - map_parameters.get("pi", 0)), test_quantities,
        family, map_parameters.get("size", 1))
    simulations = {name: [] for name in test_quantities}
    n_draws = len(np.atleast_1d(draws["gamma"]))
    if not n_draws or any(np.atleast_1d(values).shape != (n_draws,) for values in draws.values()):
        raise ValueError("posterior parameters must be matching nonempty draw vectors")
    rng = np.random.default_rng(seed)
    for index in range(n_draws):
        parameters = {name: np.atleast_1d(values)[index] for name, values in draws.items()}
        mu = means(parameters)
        pi = parameters.get("pi", 0) if family == "zinb" else 0
        size = parameters.get("size", 1)
        counts = simulate_counts(mu, family, size, pi, seed=int(rng.integers(2**31)))[0]
        values = count_test_quantities(counts, mu * (1 - pi), test_quantities, family, size)
        for name, value in values.items():
            simulations[name].append(value)
    probabilities = {}
    for name, values in simulations.items():
        finite = np.asarray(values)[np.isfinite(values)]
        probabilities[name] = min(1.0, 2 * min(np.mean(finite >= observed[name]), np.mean(finite <= observed[name]))) if finite.size and np.isfinite(observed[name]) else np.nan
    return dict(observed=observed, simulated=simulations, bayesian_p=probabilities,
                n_sim=n_draws, test_quantities=list(test_quantities))


def physical_predictive_checks(fitted, basis, posterior, observed_counts, n_sim=200,
                               test_quantities=("total_count", "vmi_ratio", "mean_deviance"), thin=1, seed=42):
    """Stream posterior draws so checking a large raster does not retain all replicas."""
    from .physical import _prepare, _evaluate
    if any(not isinstance(n, (int, np.integer)) or n < 1 for n in (n_sim, thin)):
        raise ValueError("n_sim and thin must be positive integers")
    cfg = fitted["configuration"]
    mask = np.asarray(cfg["valid_mask"], dtype=bool)
    model, args, _, state = _prepare(np.asarray(basis), mask, cfg)
    trees = posterior["parameter_draws"]
    n_draws = np.asarray(jax.tree_util.tree_leaves(trees)[0]).shape[0]
    indices = np.arange(0, n_draws, thin)
    if len(indices) > n_sim:
        indices = np.random.default_rng(seed).choice(indices, n_sim, replace=False)
    observed = count_test_quantities(observed_counts, np.asarray(fitted["intensity"]) * cfg["cell_area"], test_quantities)
    simulations = {name: [] for name in test_quantities}
    expected_counts = jax.jit(lambda p: jnp.exp(_evaluate(p, model, args, mask, cfg, state)[3]) * cfg["cell_area"])
    for index in indices:
        parameters = jax.tree_util.tree_map(lambda values: jnp.asarray(values[index]), trees)
        expected = expected_counts(parameters)
        counts = jax.random.poisson(jax.random.fold_in(jax.random.PRNGKey(seed), int(index)), expected)
        quantities = count_test_quantities(counts, expected, test_quantities)
        for name, value in quantities.items():
            simulations[name].append(value)
    p_values = {}
    for name, values in simulations.items():
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        p_values[name] = min(1.0, 2 * min(np.mean(finite >= observed[name]), np.mean(finite <= observed[name]))) if finite.size else np.nan
    return dict(observed=observed, simulated={k: np.array(v) for k, v in simulations.items()},
                bayesian_p=p_values, n_sim=len(indices), test_quantities=list(test_quantities))
