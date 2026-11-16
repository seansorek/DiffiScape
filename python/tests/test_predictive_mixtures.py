import numpy as np
from diffiscape.diagnostics import simulate_counts
from diffiscape.diagnostics import intensity_predictive_checks


def test_aggregate_negative_binomial_and_zero_inflation():
    mu = np.array([2., 3., 5.])
    draws = simulate_counts(mu, "negbin", size=2, n_sim=40000)
    np.testing.assert_allclose(draws.sum(axis=1).mean(), 10, rtol=.02)
    np.testing.assert_allclose(draws.sum(axis=1).var(), 10 + 100 / 2, rtol=.04)
    assert np.cov(draws.T)[0, 1] > 2
    zero = simulate_counts(mu, "zinb", size=2, pi=1, n_sim=10)
    assert not zero.any()


def test_shared_intensity_predictive_operation_is_reproducible():
    args = dict(z=[0, .5, 1], covariates={}, observed_counts=[0, 1, 2],
        map_parameters={"alpha": 0, "gamma": .1},
        draws={"alpha": np.zeros(5), "gamma": np.ones(5) * .1}, family="poisson")
    output = intensity_predictive_checks(**args)
    assert output == intensity_predictive_checks(**args)
    assert output["n_sim"] == 5
    assert output["observed"]["total_count"] == 3
