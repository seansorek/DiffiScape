import pickle

import numpy as np
import pytest

from diffiscape import fit, predict


@pytest.mark.parametrize("model_type", ["loglinear", "mlp", "conv", "spline_gam"])
def test_physical_models_fit_predict_reload(model_type):
    mask = np.ones(20, dtype=bool)
    mask[7] = False
    basis = np.linspace(0, 1, mask.sum() * 2).reshape(-1, 2)
    counts = np.arange(mask.sum()) % 3
    configurations = {"loglinear": {}, "mlp": {"hidden_dim": 3, "n_hidden_layers": 1},
        "conv": {"hidden_dim": 3, "n_hidden_layers": 1, "conv_channels": 2, "n_conv_layers": 1},
        "spline_gam": {"n_knots": 2, "spline_degree": 2, "include_interactions": True}}
    result = fit(basis, counts, mask, 4, 5, model_type=model_type,
        model_config=configurations[model_type],
        source_spacing=2, n_epochs=4, warmup_epochs=0, verbose=False)
    assert np.isfinite(result["best_loss"])
    assert min(result["loss_history"]) < result["loss_history"][0]
    assert result["gamma"] > 0
    predicted = predict(pickle.loads(pickle.dumps(result)), basis)
    np.testing.assert_allclose(predicted["intensity"].ravel()[mask], result["intensity"], rtol=1e-10)
    assert np.isnan(predicted["intensity"].ravel()[~mask]).all()
    assert np.isfinite(predicted["voltage"].ravel()[mask]).all()


@pytest.mark.parametrize("method", ["adam", "lbfgs"])
def test_physical_bounds_and_likelihood_match_prediction(method):
    basis = np.linspace(-1, 1, 20).reshape(-1, 1)
    counts = np.arange(20) % 4
    result = fit(basis, counts, np.ones(20, dtype=bool), 4, 5, method=method,
        source_spacing=2, init_params=[3, 0], lower_bounds=[2.9, -0.1], upper_bounds=[3.1, 0.1],
        n_epochs=4, warmup_epochs=0, verbose=False)
    assert np.all(result["best_params"] >= [2.9, -0.1])
    assert np.all(result["best_params"] <= [3.1, 0.1])
    prediction = predict(result, basis)
    intensity = prediction["intensity"].ravel()
    np.testing.assert_allclose(result["best_loglik"], counts @ np.log(intensity) - intensity.sum())


@pytest.mark.parametrize("model_type, config", [
    ("spline_gam", {"n_knots": 1, "spline_degree": 2, "intensity_spline": True, "intensity_n_knots": 1, "intensity_degree": 2}),
    ("conv", {"conv_channels": 2, "n_conv_layers": 1, "hidden_dim": 2, "intensity_hidden": 2}),
])
def test_learned_intensity_prediction(model_type, config):
    basis = np.linspace(0, 1, 12).reshape(-1, 1)
    fitted = fit(basis, np.arange(12) % 3, np.ones(12, dtype=bool), 3, 4,
        model_type=model_type, model_config=config, source_spacing=2, n_epochs=2, verbose=False)
    np.testing.assert_allclose(predict(fitted, basis)["intensity"].ravel(), fitted["intensity"], rtol=1e-10)
