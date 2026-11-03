import numpy as np

from diffiscape.likelihoods import standardise_connectivity, residualise_connectivity, predict_residualisation


def test_standardization_sample_sd_and_fitted_scale():
    result = standardise_connectivity([0, 1, 3], [0, 2, 4])
    assert result["c_scale"] == 2
    np.testing.assert_allclose(np.mean(result["z_obs"]), 0, atol=1e-15)
    np.testing.assert_allclose(np.std(result["z_obs"], ddof=1), 1)
    np.testing.assert_allclose(result["z_int"], (np.log1p(np.array([0, 2, 4]) / 2) - result["mu"]) / result["sigma"])


def test_residualization_and_prediction_use_same_coefficients():
    x = np.array([0.1, 0.3, 0.8, 0.9])
    observed = 0.7 + 2 * x + np.array([0.01, -0.01, 0.02, -0.02])
    result = residualise_connectivity(observed, [0.2, 1.4], {"x": x}, {"x": [0.2, 0.6]})
    np.testing.assert_allclose(predict_residualisation(result["aux_model"], {"x": [0.2, 0.6]}) + result["z_int_resid"], [0.2, 1.4])
    assert 0 <= result["aux_r2"] <= 1
