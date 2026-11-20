"""Installed public API: fitting, masked prediction, and saved fit reuse."""
import pickle

import numpy as np
import pytest

from diffiscape import fit, predict, sample


def test_fit_predict_and_reload():
    mask = np.array([True, False, True, True, True, True])
    basis = np.linspace(-1, 1, mask.sum()).reshape(-1, 1)
    result = fit(basis.tolist(), [0, 2, 0, 1, 3], mask.tolist(), 2, 3,
                 formulation="window_resistance",
                 radius=1, block_size=1, method="adam", n_epochs=4, verbose=False)
    assert np.isfinite(result["best_loglik"])
    assert result["loss_history"][-1] < result["loss_history"][0]
    prediction = predict(result, basis)
    assert prediction["intensity"].shape == (2, 3)
    assert np.isnan(prediction["intensity"].ravel()[~mask]).all()
    assert (prediction["intensity"].ravel()[mask] > 0).all()
    restored = pickle.loads(pickle.dumps(result))
    for name, values in prediction.items():
        np.testing.assert_allclose(predict(restored, basis)[name], values, equal_nan=True)


@pytest.mark.parametrize("counts", [[1], [0, -1], [0, np.nan]])
def test_invalid_counts_rejected(counts):
    with pytest.raises(ValueError, match="obs_counts"):
        fit([[0], [1]], counts, [True, True], 1, 2)


def test_sampling_rejects_unvalidated_grid_model():
    with pytest.raises(NotImplementedError, match="IRL is retired"):
        sample([[0], [1]], [0, 1], [True, True], 1, 2, model_type="irl")


@pytest.mark.parametrize("model_type, model_config", [
    ("mlp", {"hidden_dim": 2, "n_hidden_layers": 1}),
    ("conv", {"conv_channels": 2, "n_conv_layers": 1}),
    ("spline_gam", {"n_knots": 2}),
])
def test_legacy_neural_prediction_uses_saved_state(model_type, model_config):
    mask = np.array([True, False, True, True, True, True])
    basis = np.linspace(0, 1, 5).reshape(-1, 1)
    result = fit(basis, [0, 1, 2, 0, 1], mask, 2, 3,
        formulation="window_resistance", model_type=model_type, model_config=model_config,
        radius=1, block_size=1, optim_config={"n_epochs": 2}, verbose=False)
    predicted = predict(pickle.loads(pickle.dumps(result)), basis)
    np.testing.assert_allclose(predicted["surface"].ravel()[mask], result["resistance"])
    expected = predicted["intensity"].ravel()[mask]
    np.testing.assert_allclose(result["best_loglik"], np.array([0, 1, 2, 0, 1]) @ np.log(expected) - expected.sum())
