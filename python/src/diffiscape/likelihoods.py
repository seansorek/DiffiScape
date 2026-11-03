"""Weighted built-in intensity families shared by Python and R.

NB and ZINB mix the aggregate PPP count, not independent per-cell counts.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp
from scipy.optimize import minimize
from scipy.linalg import lstsq


def _vector(values):
    return jnp.asarray(values).reshape(-1)


def standardise_connectivity(C_obs, C_int, c_scale=None):
    observed, integration = np.maximum(np.asarray(C_obs), 0), np.maximum(np.asarray(C_int), 0)
    if observed.ndim != 1 or integration.ndim != 1 or observed.size < 2 or not np.isfinite(observed).all() or not np.isfinite(integration).all():
        raise ValueError("standardization requires finite vectors and at least two observed locations")
    positive = observed[observed > 0]
    if c_scale is None:
        c_scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(c_scale) or c_scale <= 0:
        raise ValueError("c_scale must be positive and finite")
    log_observed = np.log1p(observed / c_scale)
    mean, sd = float(np.mean(log_observed)), float(np.std(log_observed, ddof=1))
    sd = 1.0 if sd < 1e-10 else sd
    return dict(z_obs=(log_observed - mean) / sd, z_int=(np.log1p(integration / c_scale) - mean) / sd,
                c_scale=c_scale, mu=mean, sigma=sd)


def residualise_connectivity(z_obs, z_int, covariates_obs, covariates_int):
    names = list(covariates_obs)
    observed, integration = np.asarray(z_obs), np.asarray(z_int)
    X = np.column_stack([np.ones(len(observed))] + [np.asarray(covariates_obs[name]) for name in names])
    Z = np.column_stack([np.ones(len(integration))] + [np.asarray(covariates_int[name]) for name in names])
    if not np.isfinite(X).all() or not np.isfinite(Z).all() or not np.isfinite(observed).all() or not np.isfinite(integration).all():
        raise ValueError("residualization requires finite locations and covariates")
    coefficients, _, rank, _ = lstsq(X, observed)
    residuals = observed - X @ coefficients
    total = np.sum((observed - observed.mean()) ** 2)
    r2 = 1 - np.sum(residuals ** 2) / total if total > 0 else np.nan
    model = dict(coefficients=coefficients, names=names, rank=int(rank))
    return dict(z_obs_resid=residuals, z_int_resid=integration - Z @ coefficients,
                aux_coefs=coefficients, aux_r2=float(r2), aux_model=model)


def predict_residualisation(model, covariates):
    names = model["names"]
    if isinstance(names, str):
        names = [names]
    columns = [np.asarray(covariates[name]).reshape(-1) for name in names]
    X = np.column_stack([np.ones(len(columns[0]))] + columns)
    return X @ np.asarray(model["coefficients"])


def compute_intensity(z, alpha, gamma, covariates=None, betas=None):
    predictor = alpha + gamma * _vector(z)
    if covariates is not None and betas is not None:
        for name, values in covariates.items():
            beta = jnp.asarray(betas.get(name, 0.0))
            predictor = predictor + jnp.where(jnp.isnan(beta), 0, beta) * _vector(values)
    return jnp.exp(predictor)


def family_spec(family, n_cov=0, known_sd=None, background_weight=1000,
                stratum_ids_used=None, stratum_ids_avail=None):
    if family not in ("poisson", "negbin", "gaussian", "zinb", "rsf", "rsp", "clogit"):
        raise ValueError(f"Unknown intensity family: {family}")
    if not isinstance(n_cov, (int, np.integer)) or n_cov < 0:
        raise ValueError("n_cov must be a nonnegative integer")
    if not np.isfinite(background_weight) or background_weight <= 0:
        raise ValueError("background_weight must be positive and finite")
    if (stratum_ids_used is None) != (stratum_ids_avail is None):
        raise ValueError("used and available stratum IDs must be supplied together")
    relative = family in ("rsf", "clogit")
    start = ([0.0] if relative else [0.0, 0.0 if family == "rsp" else 1.0]) + [0.0] * n_cov
    lower, upper = [-10.0] * len(start), [10.0] * len(start)
    extras = []
    if family in ("negbin", "zinb"):
        extras.append("log_nb_theta")
        start += [0.0]; lower += [np.log(0.01)]; upper += [np.log(1e6)]
    if family == "zinb":
        extras.append("logit_pi")
        start += [0.0]; lower += [-5.0]; upper += [5.0]
    if family == "gaussian" and known_sd is None:
        extras.append("log_sd")
        start += [0.0]; lower += [np.log(0.001)]; upper += [np.log(100.0)]
    if known_sd is not None and (not np.isfinite(known_sd) or known_sd <= 0):
        raise ValueError("known_sd must be positive and finite")
    return dict(start=np.array(start), lower=np.array(lower), upper=np.array(upper), extra_param_names=extras)


def negative_loglik(theta, z_obs, z_int, int_weights, obs_weights,
                   cov_obs=None, cov_int=None, cov_names=(), y_obs=None,
                   family="poisson", known_sd=None, background_weight=1000,
                   stratum_ids_used=None, stratum_ids_avail=None):
    """Differentiable NLL on prepared arrays; theta uses optimizer-scale extras."""
    theta = _vector(theta)
    z_obs, z_int = _vector(z_obs), _vector(z_int)
    int_weights, obs_weights = _vector(int_weights), _vector(obs_weights)
    relative = family in ("rsf", "clogit")
    gamma = theta[0] if relative else theta[1]
    intercept = 0.0 if relative else theta[0]
    f_obs, f_int = intercept + gamma * z_obs, intercept + gamma * z_int
    for index, name in enumerate(cov_names):
        coefficient = theta[index + (1 if relative else 2)]
        f_obs = f_obs + coefficient * _vector(cov_obs[name])
        f_int = f_int + coefficient * _vector(cov_int[name])
    count = jnp.sum(obs_weights)
    if family == "clogit" and stratum_ids_used is not None:
        used, available = np.atleast_1d(stratum_ids_used), np.atleast_1d(stratum_ids_avail)
        nll = 0.0
        for stratum in np.unique(used):
            u, a = np.flatnonzero(used == stratum), np.flatnonzero(available == stratum)
            nll = nll + logsumexp(jnp.concatenate([f_obs[u], f_int[a]])) - f_obs[u[0]]
    elif relative:
        nll = -(jnp.sum(obs_weights * f_obs) - count * logsumexp(f_int, b=int_weights))
    elif family == "rsp":
        nll = -(jnp.sum(obs_weights * jax.nn.log_sigmoid(f_obs)) + background_weight * jnp.sum(int_weights * jax.nn.log_sigmoid(-f_int)))
    elif family == "gaussian":
        if y_obs is None:
            raise ValueError("family_gaussian requires an observed response vector (y_obs)")
        sd = jnp.exp(theta[-1]) if known_sd is None else known_sd
        nll = 0.5 * jnp.sum(obs_weights * ((jnp.exp(f_obs) - _vector(y_obs)) / sd) ** 2)
        nll = nll + count * (jnp.log(sd) + 0.5 * jnp.log(2 * jnp.pi))
    else:
        integral = jnp.sum(int_weights * jnp.exp(f_int))
        point_term = jnp.sum(obs_weights * jnp.log(jnp.maximum(jnp.exp(f_obs), 1e-300)))
        if family == "poisson":
            nll = integral - point_term
        elif family in ("negbin", "zinb"):
            size = jnp.exp(theta[-2] if family == "zinb" else theta[-1])
            # Algebraically cancel count*log(integral) before evaluation.
            loglik = point_term + gammaln(count + size) - gammaln(size)
            loglik = loglik + size * jnp.log(size) - (count + size) * jnp.log(size + integral)
            if family == "zinb":
                log_pi, log_real = jax.nn.log_sigmoid(theta[-1]), jax.nn.log_sigmoid(-theta[-1])
                loglik = jnp.where(count > 0, loglik + log_real,
                                   jnp.logaddexp(log_pi, log_real + size * (jnp.log(size) - jnp.log(size + integral))))
            nll = -loglik
        else:
            raise ValueError(f"Unknown intensity family: {family}")
    return jnp.where(jnp.isfinite(nll), nll, 1e15)


def fit_intensity(z_obs, z_int, int_weights, obs_weights, cov_obs=None, cov_int=None,
                  cov_names=(), y_obs=None, family="poisson", initial=None, **configuration):
    """Bounded intensity MLE and observed Hessian using the same density."""
    spec = family_spec(family, len(cov_names), **configuration)
    if initial is not None:
        initial = {key: np.asarray(initial[key], dtype=np.float64) for key in ("start", "lower", "upper")}
        if any(value.shape != np.asarray(spec["start"]).shape or np.isnan(value).any() for value in initial.values()) or not np.isfinite(initial["start"]).all() or (initial["lower"] > initial["upper"]).any():
            raise ValueError("initial parameters and bounds must match the family parameter layout")
        spec.update(initial)
    for values, weights in ((z_obs, obs_weights), (z_int, int_weights)):
        values, weights = np.atleast_1d(values), np.atleast_1d(weights)
        if values.shape != weights.shape or not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("locations and nonnegative weights must be finite matching vectors")
    if np.sum(int_weights) <= 0:
        raise ValueError("integration weights must have positive total weight")
    if family == "gaussian" and (y_obs is None or np.atleast_1d(y_obs).shape != np.atleast_1d(z_obs).shape or not np.isfinite(y_obs).all()):
        raise ValueError("Gaussian y_obs must be a finite response vector matching observations")
    for name in cov_names:
        for values, locations in ((cov_obs[name], z_obs), (cov_int[name], z_int)):
            if np.atleast_1d(values).shape != np.atleast_1d(locations).shape or not np.isfinite(values).all():
                raise ValueError("covariates must be finite vectors matching their locations")
    if (strata := configuration.get("stratum_ids_used")) is not None:
        available = configuration.get("stratum_ids_avail")
        u, a = np.atleast_1d(strata), np.atleast_1d(available)
        if u.shape != np.atleast_1d(z_obs).shape or a.shape != np.atleast_1d(z_int).shape or set(u) != set(a) or len(set(u)) != len(u):
            raise ValueError("strata must match locations, with one used location and available locations in each stratum")
    objective = lambda p: negative_loglik(p, z_obs, z_int, int_weights, obs_weights,
        cov_obs, cov_int, cov_names, y_obs, family, **configuration)
    vg = jax.value_and_grad(objective)
    def evaluate(p):
        value, gradient = vg(jnp.asarray(p, dtype=jnp.float64))
        return float(value), np.asarray(gradient, dtype=np.float64)
    result = minimize(evaluate, spec["start"], jac=True, method="L-BFGS-B",
                       bounds=list(zip(spec["lower"], spec["upper"])))
    return dict(par=result.x, value=float(result.fun), convergence=0 if result.success else int(result.status or 1),
                hessian=np.asarray(jax.hessian(objective)(jnp.asarray(result.x))), message=str(result.message))


def deviance_residuals(observed, fitted, family="negbin", size=1.0):
    y, mu = _vector(observed), jnp.maximum(_vector(fitted), 1e-10)
    if family == "gaussian":
        return y - _vector(fitted)
    if family in ("rsf", "rsp", "clogit"):
        return jnp.full(y.shape, jnp.nan)
    first = jnp.where(y > 0, y * jnp.log(jnp.maximum(y, 1e-300) / mu), 0)
    second = y - mu if family == "poisson" else (y + size) * jnp.log((y + size) / (mu + size))
    return jnp.sign(y - mu) * jnp.sqrt(jnp.maximum(2 * (first - second), 0))
