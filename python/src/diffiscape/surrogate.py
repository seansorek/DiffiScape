"""Ordinary Matérn kriging and surrogate optimization using SMT and SciPy."""
import base64
from functools import lru_cache
import pickle

import numpy as np
from scipy.stats import qmc, norm
from scipy.differentiate import hessian


def fit_surrogate(X, y, config=None):
    from smt.surrogate_models import KRG
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or len(X) != len(y) or len(X) < 2 or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("surrogate training requires a finite matrix and matching responses, with at least two rows")
    _, keep = np.unique(X, axis=0, return_index=True)
    X, y = X[np.sort(keep)], y[np.sort(keep)]
    if len(X) < 2:
        raise ValueError("surrogate training requires at least two distinct parameter vectors")
    options = dict(corr="matern52", poly="constant", eval_noise=True, noise0=[1.0], hyper_opt="Cobyla", print_global=False, seed=42)
    settings = dict(config or {})
    if "length_scale" in settings:
        lengths = np.asarray(settings.pop("length_scale"), dtype=np.float64)
        if not np.isfinite(lengths).all() or np.any(lengths <= 0):
            raise ValueError("length_scale must be positive and finite")
        options.update(theta0=np.std(X, axis=0, ddof=1) / lengths, hyper_opt="NoOp")
    options.update(settings)
    model = KRG(**options)
    model.set_training_values(X, y)
    try:
        model.train()
    except np.linalg.LinAlgError:
        # Match the old regularized retry, expressed in SMT's normalized noise.
        options.update(eval_noise=False, noise0=[0.01])
        model = KRG(**options)
        model.set_training_values(X, y)
        model.train()
    return dict(engine="smt", model_state=base64.b64encode(pickle.dumps(model)).decode("ascii"),
                n_features=X.shape[1], X=X, y=y)


@lru_cache(maxsize=8)
def _model(state):
    # Fits are trusted local artifacts, as with the package's pickle workflow.
    return pickle.loads(base64.b64decode(state))


def predict_surrogate(surrogate, newdata):
    X = np.atleast_2d(np.asarray(newdata, dtype=np.float64))
    if X.shape[1] != surrogate["n_features"] or not np.isfinite(X).all():
        raise ValueError("surrogate prediction requires finite rows in fitted parameter order")
    model = _model(surrogate["model_state"])
    return dict(mean=model.predict_values(X).ravel(), sd=np.sqrt(np.maximum(model.predict_variances(X).ravel(), 0)))


def expected_improvement(mean, sd, y_best, xi_initial=0.1, iteration=None, n_iter=None,
                         xi_min=0.02, decay_divisor=5, min_sd=1e-10):
    xi = max(xi_initial * np.exp(-iteration / (n_iter / decay_divisor)), xi_min) if iteration is not None and n_iter is not None and n_iter > 0 else xi_initial
    sd = np.asarray(sd)
    improvement = y_best - np.asarray(mean) - xi
    z = improvement / np.maximum(sd, min_sd)
    return np.where(sd < min_sd, 0.0, improvement * norm.cdf(z) + sd * norm.pdf(z))


def loo_cv(X, y, config=None):
    """Refit ordinary kriging, excluding failed observations and refits."""
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.shape != (len(X),) or not np.isfinite(X[np.isfinite(y)]).all():
        raise ValueError("LOO requires matching training rows and responses")
    X, y = X[np.isfinite(y)], y[np.isfinite(y)]
    predicted, sd = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for index in range(len(y)):
        keep = np.arange(len(y)) != index
        try:
            fitted = fit_surrogate(X[keep], y[keep], config)
            output = predict_surrogate(fitted, X[index:index + 1])
            predicted[index], sd[index] = output["mean"][0], output["sd"][0]
        except Exception:
            continue
    valid = np.isfinite(predicted)
    observed, predicted, sd = y[valid], predicted[valid], sd[valid]
    residual = observed - predicted
    total = np.sum((observed - observed.mean()) ** 2) if valid.any() else 0
    return dict(observed=observed, predicted=predicted, residuals=residual,
        rmse=float(np.sqrt(np.mean(residual ** 2))) if valid.any() else np.nan,
        r_squared=float(1 - np.sum(residual ** 2) / total) if total > 0 else np.nan,
        coverage_95=float(np.nanmean(np.abs(residual) <= norm.ppf(.975) * sd)) if valid.any() else np.nan,
        n_failed=int((~valid).sum()))


def failure_penalty(valid):
    valid = np.asarray(valid)
    if not valid.size:
        return 1e10
    worst, spread = np.max(valid), np.ptp(valid)
    margin = 0.1 * spread if spread > 0 else max(0.1 * abs(worst), 1e-6)
    return float(worst + margin)


def lhs_design(n, bounds, seed=42):
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("bounds must be finite increasing (n_parameters, 2) intervals")
    return qmc.scale(qmc.LatinHypercube(bounds.shape[0], seed=seed).random(n), bounds[:, 0], bounds[:, 1])


def candidates(n, bounds, best_point=None, sigma=None, local_frac=0.5, seed=42):
    bounds = np.asarray(bounds)
    n_local = int(np.floor(n * local_frac)) if best_point is not None else 0
    global_points = lhs_design(n - n_local, bounds, seed)
    if not n_local:
        return global_points
    sigma = np.full(len(bounds), 0.05) if sigma is None else np.asarray(sigma)
    local = np.random.default_rng(seed).normal(best_point, sigma * np.ptp(bounds, axis=1), (n_local, len(bounds)))
    return np.concatenate([global_points, np.clip(local, bounds[:, 0], bounds[:, 1])])


def dimension_scales(surrogate, best_point, max_ratio=5):
    x = np.asarray(best_point, dtype=np.float64)
    def mean(values):
        shape = values.shape[1:]
        return predict_surrogate(surrogate, values.reshape(len(x), -1).T)["mean"].reshape(shape)
    curvature = hessian(mean, x).ddf
    if not np.isfinite(curvature).all():
        return np.ones(len(x))
    scales = 1 / np.sqrt(np.maximum(np.abs(np.diag(curvature)), 1e-10))
    scales /= scales.mean()
    ratio = scales.max() / scales.min()
    if ratio > max_ratio:
        scales = np.exp(np.log(scales) * np.log(max_ratio) / np.log(ratio))
        scales /= scales.mean()
    return scales


def optimize_surrogate(objective, bounds, config=None, surrogate_config=None):
    """Derivative-free BO; Python callables work independently of R callbacks."""
    cfg = dict(n_init=20, n_iter=50, seed=42, acquisition="TS", n_candidates=3000,
        ts_min_sd=1e-6, ei_xi_scale_factor=0.1, ei_xi_min=0.02, ei_decay_rate_divisor=5,
        sigma_initial=0.15, sigma_min=0.02, stall_threshold=3, decay_factor=0.7,
        local_frac_initial=0.5, local_frac_max=0.8, local_frac_increment=0.05, restart_threshold=5)
    if set(config or {}) - set(cfg):
        raise ValueError("unsupported surrogate optimization options")
    cfg.update(config or {})
    if cfg["acquisition"] not in ("TS", "EI"):
        raise ValueError("acquisition must be TS or EI")
    if cfg["n_init"] < 2 or cfg["n_iter"] < 0 or cfg["n_candidates"] < 1:
        raise ValueError("surrogate optimization requires n_init>=2, n_iter>=0, and n_candidates>=1")
    bounds = np.asarray(bounds, dtype=np.float64)
    rng = np.random.default_rng(cfg["seed"])
    X = lhs_design(cfg["n_init"], bounds, cfg["seed"])
    y = np.array([float(objective(row)) for row in X])
    if not np.isfinite(y).any():
        raise ValueError("all initial objective evaluations failed")
    sigma = np.full(len(bounds), cfg["sigma_initial"])
    local_frac, best = cfg["local_frac_initial"], np.nanmin(np.where(np.isfinite(y), y, np.nan))
    stalls, stalls_at_min = 0, 0
    finite = y[np.isfinite(y)]
    xi_initial = np.ptp(finite) * cfg["ei_xi_scale_factor"]
    for iteration in range(1, cfg["n_iter"] + 1):
        scored = np.where(np.isfinite(y), y, failure_penalty(y[np.isfinite(y)]))
        surrogate = fit_surrogate(X, scored, surrogate_config)
        best_point = X[np.argmin(scored)]
        scales = dimension_scales(surrogate, best_point) if iteration > 1 else np.ones(len(bounds))
        proposed = candidates(cfg["n_candidates"], bounds, best_point, sigma * scales, local_frac, int(rng.integers(2**31)))
        prediction = predict_surrogate(surrogate, proposed)
        if cfg["acquisition"] == "EI":
            acquisition = expected_improvement(prediction["mean"], prediction["sd"], np.min(scored), xi_initial, iteration, cfg["n_iter"], cfg["ei_xi_min"], cfg["ei_decay_rate_divisor"])
            point = proposed[np.argmax(acquisition)]
        else:
            draw = rng.normal(prediction["mean"], np.maximum(prediction["sd"], cfg["ts_min_sd"]))
            point = proposed[np.argmin(draw)]
        value = float(objective(point))
        X, y = np.vstack([X, point]), np.append(y, value)
        if np.isfinite(value) and value < best:
            best, stalls, stalls_at_min = value, 0, 0
        else:
            stalls += 1
            if stalls >= cfg["stall_threshold"]:
                if sigma.mean() <= cfg["sigma_min"] * 1.01:
                    stalls_at_min += 1
                    if stalls_at_min >= cfg["restart_threshold"]:
                        sigma[:] = cfg["sigma_initial"]
                        local_frac, stalls_at_min = cfg["local_frac_initial"], 0
                else:
                    sigma = np.maximum(sigma * cfg["decay_factor"], cfg["sigma_min"])
                    local_frac = min(local_frac + cfg["local_frac_increment"], cfg["local_frac_max"])
                stalls = 0
    scored = np.where(np.isfinite(y), y, failure_penalty(y[np.isfinite(y)]))
    best_index = int(np.nanargmin(np.where(np.isfinite(y), y, np.nan)))
    return dict(best_params=X[best_index], best_loglik=float(-y[best_index]), best_idx=best_index,
        X_evaluated=X, y_evaluated=y, n_evaluations=len(y), surrogate=fit_surrogate(X, scored, surrogate_config),
        bounds=bounds, acquisition=cfg["acquisition"], xi_initial=float(xi_initial),
        decay_rate=cfg["n_iter"] / cfg["ei_decay_rate_divisor"] if cfg["acquisition"] == "EI" else None)
