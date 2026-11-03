import numpy as np

from diffiscape.diagnostics import information_criteria, posterior_summary


def test_information_criteria_preserves_small_sample_convention():
    result = information_criteria(-10, 3, 20)
    assert result["AIC"] == 26
    np.testing.assert_allclose(result["BIC"], 20 + np.log(20) * 3)
    assert information_criteria(-10, 3, 2)["AICc"] == 50


def test_posterior_summary_sample_sd_missing_values_and_interval():
    result = posterior_summary({"a": [1, 2, 3, np.nan], "loglik": [-2, -3, -4, -5]}, prob=0.8)
    assert result["parameter"].tolist() == ["a"]
    np.testing.assert_allclose(result["mean"], 2)
    np.testing.assert_allclose(result["sd"], 1)
    np.testing.assert_allclose(result["lower"], 1.2)
    np.testing.assert_allclose(result["upper"], 2.8)
