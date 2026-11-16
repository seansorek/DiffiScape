import numpy as np
import pytest
import jax
import jax.numpy as jnp
from numpyro.infer.util import log_density
from numpyro.distributions import Normal

from diffiscape import fit, sample
from diffiscape.physical import _posterior_model


@pytest.fixture(scope="module")
def problem():
    basis = np.linspace(0, 1, 6).reshape(-1, 1)
    mask = np.ones(6, dtype=bool)
    counts = np.array([0, 1, 0, 2, 0, 1])
    fitted = fit(basis, counts, mask, 2, 3, model_type="spline_gam",
        model_config={"n_knots": 1, "spline_degree": 2, "include_interactions": False},
        source_spacing=2, n_epochs=2, verbose=False)
    return basis, mask, counts, fitted


def test_posterior_prior_and_penalty_density(problem):
    from diffiscape.physical import _prepare, _evaluate
    basis, mask, counts, fitted = problem
    model, initial, unpack = _posterior_model(fitted, basis, counts, fix_smoothing=False)
    total, trace = log_density(model, (), {}, {"parameters": initial})
    cfg = fitted["configuration"]
    resistance_model, args, _, state = _prepare(basis, mask, cfg)
    p = unpack(initial)
    _, _, _, log_lambda, penalty = _evaluate(p, resistance_model, args, mask, cfg, state)
    prior = trace["parameters"]["fn"].log_prob(initial)
    expected = prior + counts @ log_lambda - jnp.exp(log_lambda).sum() - penalty
    np.testing.assert_allclose(total, expected, rtol=1e-12)
    # Unlike the old flat target, the approved Normal priors penalize tails.
    shifted, _ = log_density(model, (), {}, {"parameters": initial + 100})
    assert shifted < total - 1000
    assert np.isfinite(jax.grad(lambda v: log_density(model, (), {}, {"parameters": v})[0])(initial)).all()


def test_physical_nuts_smoke(problem):
    basis, mask, counts, fitted = problem
    draws = sample(basis, counts, mask, 2, 3, fitted=fitted,
                   n_samples=4, warmup=4, max_treedepth=2)
    assert draws["samples"]["parameters"].shape[0] == 4
    assert np.isfinite(draws["samples"]["parameters"]).all()
    assert (draws["samples"]["gamma"] > 0).all()
    assert draws["n_divergences"] >= 0


def test_physical_advi_smoke(problem):
    import diffiscape
    basis, mask, counts, fitted = problem
    draws = diffiscape.sample(basis, counts, mask, 2, 3, fitted=fitted, method="advi", n_samples=4, max_iter=3)
    assert callable(diffiscape.sample)
    assert draws["samples"]["parameters"].shape[0] == 4
    assert np.isfinite(draws["samples"]["parameters"]).all()
    assert np.isfinite(draws["best_elbo"])
