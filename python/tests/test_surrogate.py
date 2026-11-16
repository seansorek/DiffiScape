import json
from pathlib import Path

import numpy as np


def test_loo_statistics_exclude_failed_refits(monkeypatch):
    from diffiscape import surrogate
    original = surrogate.fit_surrogate
    calls = []
    config = {"eval_noise": False, "noise0": [.01]}
    def refit(X, y, configuration):
        calls.append(configuration)
        if len(calls) in (1, 3):
            raise ValueError("failed refit")
        return original(X, y, configuration)
    monkeypatch.setattr(surrogate, "fit_surrogate", refit)
    X = np.random.default_rng(1).uniform(size=(7, 2))
    y = np.sum(X ** 2, axis=1)
    y[-1] = np.nan
    result = surrogate.loo_cv(X, y, config)
    assert result["n_failed"] == 2
    assert len(result["observed"]) == 4
    assert all(value == config for value in calls)
    np.testing.assert_allclose(result["residuals"], result["observed"] - result["predicted"])
    np.testing.assert_allclose(result["rmse"], np.sqrt(np.mean(result["residuals"] ** 2)))
import pytest

from diffiscape.surrogate import fit_surrogate, predict_surrogate, expected_improvement, failure_penalty, optimize_surrogate


def test_fixed_matern_matches_r_ordinary_kriging():
    reference = json.loads((Path(__file__).parent / "fixtures" / "r_kriging_reference.json").read_text())
    fitted = fit_surrogate(np.array(reference["x"])[:, None], reference["y"],
        dict(length_scale=reference["length_scale"], eval_noise=False, noise0=[0.0], nugget=0.0))
    prediction = predict_surrogate(fitted, np.array(reference["query"])[:, None])
    np.testing.assert_allclose(prediction["mean"], reference["mean"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(prediction["sd"], reference["sd"], rtol=2e-7, atol=1e-8)


def test_expected_improvement_and_failure_penalty():
    np.testing.assert_array_equal(expected_improvement([0, 0], [0, 0], -1), 0)
    assert expected_improvement([0], [1], 0, xi_initial=0)[0] > expected_improvement([0], [1], 0, xi_initial=1)[0]
    for values in ([-20, -10], [10, 20], [0, 0], []):
        assert failure_penalty(values) > max(values, default=0)


@pytest.mark.parametrize("acquisition", ["TS", "EI"])
def test_surrogate_optimization_preserves_failures_and_bounds(acquisition):
    calls = []
    def objective(x):
        calls.append(x.copy())
        return np.nan if len(calls) == 1 else float(np.sum((x - 0.2) ** 2))
    result = optimize_surrogate(objective, [[-1, 1]],
        dict(n_init=4, n_iter=2, n_candidates=20, acquisition=acquisition), dict(n_start=1))
    assert len(calls) == result["n_evaluations"] == 6
    assert np.isnan(result["y_evaluated"][0])
    assert np.all((result["X_evaluated"] >= -1) & (result["X_evaluated"] <= 1))
    assert np.isfinite(result["best_loglik"])
