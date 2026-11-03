import numpy as np


def test_custom_initial_bounds_are_respected():
    from diffiscape.likelihoods import fit_intensity, family_spec
    initial = family_spec("zinb", 0)
    initial["start"][-1] = -5
    initial["lower"][-1] = -5 - 1e-6
    initial["upper"][-1] = -5 + 1e-6
    result = fit_intensity([0., 1.], [0., .5, 1.], [1., 1., 1.], [1., 1.], family="zinb", initial=initial)
    assert abs(result["par"][-1] + 5) < 2e-6
import pytest
import jax
import jax.numpy as jnp

from diffiscape.likelihoods import negative_loglik, fit_intensity


@pytest.mark.parametrize("family, expected", [
    ("poisson", 1.72486352037781), ("negbin", 1.95128065817946),
    ("gaussian", 4.03061191101804), ("zinb", 2.46429591057941),
    ("rsf", 1.50339583828903), ("rsp", 929.485163120228), ("clogit", 2.05142471982199),
])
def test_matches_original_r_density_and_finite_difference(family, expected):
    # Captured from the original R 0.1.0 package before replacing its families.
    theta = [0.7, 0.3] if family in ("rsf", "clogit") else [-0.2, 0.7, 0.3]
    if family in ("negbin", "zinb"):
        theta += [np.log(2)]
    if family == "zinb":
        theta += [-0.4]
    if family == "gaussian":
        theta += [np.log(0.8)]
    cfg = dict(stratum_ids_used=[1, 2], stratum_ids_avail=[1, 1, 2]) if family == "clogit" else {}
    def objective(p):
        return negative_loglik(p, [-0.3, 0.8], [-0.5, 0.1, 1.2], [0.2, 0.5, 0.4], [2, 1],
            {"x": [0.2, 0.6]}, {"x": [0.1, 0.5, 0.7]}, ["x"], [-0.4, 1.5], family, **cfg)
    np.testing.assert_allclose(objective(jnp.array(theta)), expected, rtol=1e-12, atol=1e-12)
    gradient = jax.grad(objective)(jnp.array(theta))
    for index in range(len(theta)):
        upper, lower = np.array(theta), np.array(theta)
        upper[index] += 1e-5
        lower[index] -= 1e-5
        np.testing.assert_allclose(gradient[index], (objective(upper) - objective(lower)) / 2e-5, rtol=1e-5, atol=1e-6)


def test_poisson_fit_recovers_total_and_hessian():
    fitted = fit_intensity([0, 0, 0], [0, 0], [2, 3], [1, 2, 3])
    np.testing.assert_allclose(np.exp(fitted["par"][0]), 6 / 5, rtol=1e-5)
    assert fitted["convergence"] == 0
    np.testing.assert_allclose(fitted["hessian"][0, 0], 6, rtol=1e-5)


def test_zinb_zero_count_probability():
    theta = jnp.array([0.0, 0.0, np.log(2), -0.4])
    expected = -np.log(jax.nn.sigmoid(-0.4) + jax.nn.sigmoid(0.4) * (2 / 5) ** 2)
    np.testing.assert_allclose(negative_loglik(theta, [0], [0], [3], [0], family="zinb"), expected)
    assert np.isfinite(jax.grad(lambda p: negative_loglik(p, [0], [0], [3], [0], family="zinb"))(theta)).all()
