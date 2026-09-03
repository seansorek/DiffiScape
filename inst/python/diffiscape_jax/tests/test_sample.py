"""Tests for Bayesian sampling in diffiscape_jax.sample."""
import numpy as np
import pytest

# Skip all tests if dependencies are not available
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
pytest.importorskip("numpyro")
pytest.importorskip("jaxscape")

import numpyro

from diffiscape_jax.sample import (
    _build_numpyro_model,
    _summarize_samples,
    run_nuts_sampling,
    run_advi_sampling,
)
from diffiscape_jax.resistance import ResistanceMLP


@pytest.fixture(scope="module")
def tiny_problem():
    n_rows, n_cols, n_basis = 6, 6, 2
    n_cells = n_rows * n_cols
    rng = np.random.default_rng(42)
    basis = rng.standard_normal((n_cells, n_basis))
    obs = rng.poisson(3, n_cells).astype(float)
    valid = np.ones(n_cells, dtype=bool)
    model = ResistanceMLP(features=8, n_hidden=1)
    init_params = model.init(jax.random.PRNGKey(0), jnp.array(basis))
    return {
        "model": model, "init_params": init_params, "basis": basis,
        "obs": obs, "valid": valid, "n_rows": n_rows, "n_cols": n_cols,
    }


@pytest.fixture(scope="module")
def nuts_result(tiny_problem):
    return run_nuts_sampling(
        tiny_problem["model"], tiny_problem["init_params"],
        tiny_problem["basis"], tiny_problem["obs"], tiny_problem["valid"],
        tiny_problem["n_rows"], tiny_problem["n_cols"], cell_area=1.0,
        parameterization="resistance", n_samples=10, warmup=5,
        radius=3, block_size=2, seed=42,
    )


@pytest.fixture(scope="module")
def advi_result(tiny_problem):
    return run_advi_sampling(
        tiny_problem["model"], tiny_problem["init_params"],
        tiny_problem["basis"], tiny_problem["obs"], tiny_problem["valid"],
        tiny_problem["n_rows"], tiny_problem["n_cols"], cell_area=1.0,
        parameterization="resistance", n_samples=10, max_iter=20,
        radius=3, block_size=2, seed=42,
    )


class TestBuildNumpyroModel:
    """Tests for _build_numpyro_model helper."""

    def test_returns_tuple_of_three(self, tiny_problem):
        """Test that _build_numpyro_model returns (model_fn, flat_init, unflatten_fn)."""
        result = _build_numpyro_model(
            tiny_problem["model"],
            jnp.array(tiny_problem["basis"]),
            jnp.array(tiny_problem["obs"]),
            jnp.array(tiny_problem["valid"]),
            tiny_problem["n_rows"],
            tiny_problem["n_cols"],
            cell_area=1.0,
            parameterization="resistance",
            init_params=tiny_problem["init_params"],
            radius=3, block_size=2,
        )
        assert len(result) == 3
        model_fn, flat_init, unflatten_fn = result
        assert callable(model_fn)
        assert callable(unflatten_fn)
        assert flat_init.ndim == 1

    def test_flat_params_roundtrip(self, tiny_problem):
        """Test that flattening and unflattening params preserves structure."""
        _, flat_init, unflatten_fn = _build_numpyro_model(
            tiny_problem["model"],
            jnp.array(tiny_problem["basis"]),
            jnp.array(tiny_problem["obs"]),
            jnp.array(tiny_problem["valid"]),
            tiny_problem["n_rows"],
            tiny_problem["n_cols"],
            cell_area=1.0,
            parameterization="resistance",
            init_params=tiny_problem["init_params"],
            radius=3, block_size=2,
        )
        rebuilt = unflatten_fn(flat_init)
        orig_leaves = jax.tree.leaves(tiny_problem["init_params"])
        rebuilt_leaves = jax.tree.leaves(rebuilt)
        assert len(orig_leaves) == len(rebuilt_leaves)
        for o, r in zip(orig_leaves, rebuilt_leaves):
            np.testing.assert_array_equal(np.array(o), np.array(r))

    def test_uses_moving_window_connectivity_not_corner_source(self, tiny_problem):
        """Regression test for GH #123.

        The NumPyro likelihood must use the same moving-window
        ``cumulative_current_core`` operator as the point-estimate optimizer
        (core.py's ``_connectivity_objective``), not a single-source
        ``ResistanceDistance`` map rooted at grid cell (0, 0). Evaluates the
        model's connectivity at a fixed parameter draw and checks it equals
        ``cumulative_current_core`` computed independently on the same
        resistance surface.
        """
        from diffiscape_jax.core import prepare_permeability, cumulative_current_core

        model_fn, flat_init, unflatten = _build_numpyro_model(
            tiny_problem["model"],
            jnp.array(tiny_problem["basis"]),
            jnp.array(tiny_problem["obs"]),
            jnp.array(tiny_problem["valid"]),
            tiny_problem["n_rows"],
            tiny_problem["n_cols"],
            cell_area=1.0,
            parameterization="resistance",
            init_params=tiny_problem["init_params"],
            radius=3, block_size=2,
        )

        params_tree = unflatten(flat_init)
        log_r = tiny_problem["model"].apply(params_tree, jnp.array(tiny_problem["basis"]))
        resistance = jnp.exp(log_r)
        valid_mask = jnp.array(tiny_problem["valid"])
        n_rows, n_cols = tiny_problem["n_rows"], tiny_problem["n_cols"]

        fill_val = jnp.mean(resistance)
        full_surface = jnp.ones(n_rows * n_cols) * fill_val
        full_surface = full_surface.at[valid_mask].set(resistance)
        surface_2d = full_surface.reshape((n_rows, n_cols))
        perm = prepare_permeability(surface_2d, "resistance")

        expected = cumulative_current_core(perm, n_rows, n_cols, radius=3, block_size=2)

        with numpyro.handlers.seed(rng_seed=0), \
             numpyro.handlers.substitute(data={
                 "params": flat_init,
                 "alpha": jnp.array(0.0),
                 "gamma": jnp.array(1.0),
             }), \
             numpyro.handlers.trace() as tr:
            model_fn()

        # No node is directly the connectivity surface (it feeds a
        # `numpyro.factor`, not a `sample`/`deterministic`), so recompute it
        # via the loglik factor's dependency on the same inputs is indirect;
        # instead assert the model is internally consistent with the
        # independently-computed moving-window surface by checking the
        # factor log-likelihood matches ppp_loglik on `expected`.
        from diffiscape_jax.core import ppp_loglik
        conn_valid = expected.ravel()[valid_mask]
        expected_loglik = ppp_loglik(
            conn_valid, jnp.array(tiny_problem["obs"]), 1.0,
            jnp.array(0.0), jnp.array(1.0),
        )
        actual_loglik = tr["loglik"]["fn"].log_factor
        np.testing.assert_allclose(
            np.array(actual_loglik), np.array(expected_loglik), rtol=1e-5,
        )


class TestSummarizeSamples:
    """Tests for _summarize_samples helper."""

    def test_scalar_samples(self):
        """Test summary of a 1-D sample array."""
        samples = {"x": np.random.randn(100)}
        summary = _summarize_samples(samples)
        assert "x" in summary
        for key in ("mean", "sd", "q025", "q50", "q975"):
            assert key in summary["x"]
        assert summary["x"]["q025"] <= summary["x"]["q50"] <= summary["x"]["q975"]

    def test_vector_samples(self):
        """Test summary of a 2-D sample array (n_samples x n_params)."""
        samples = {"params": np.random.randn(50, 5)}
        summary = _summarize_samples(samples)
        assert "params" in summary
        # Per-column summaries
        for key in ("mean", "sd", "q025", "q50", "q975"):
            assert key in summary["params"]
            assert len(summary["params"][key]) == 5

    def test_empty_dict(self):
        """Test that an empty samples dict returns empty summary."""
        assert _summarize_samples({}) == {}


class TestRunNutsSampling:
    """Tests for run_nuts_sampling."""

    def test_nuts_returns_expected_keys(self, nuts_result):
        """Test that NUTS sampling returns dict with all expected keys."""
        result = nuts_result

        expected_keys = {"samples", "summary", "n_divergences", "elapsed"}
        assert expected_keys == set(result.keys())

    def test_nuts_samples_shape(self, nuts_result):
        """Test that NUTS produces samples with correct shape."""
        n_samples = 10
        result = nuts_result

        assert "params" in result["samples"]
        assert "alpha" in result["samples"]
        assert "gamma" in result["samples"]
        assert result["samples"]["params"].shape[0] == n_samples
        assert result["samples"]["alpha"].shape[0] == n_samples
        assert result["samples"]["gamma"].shape[0] == n_samples
        assert isinstance(result["samples"]["params"], np.ndarray)

    def test_nuts_divergences_is_int(self, nuts_result):
        """Test that n_divergences is a non-negative integer."""
        result = nuts_result

        assert isinstance(result["n_divergences"], int)
        assert result["n_divergences"] >= 0

    def test_nuts_requests_diverging_extra_field(self, tiny_problem, monkeypatch):
        """Regression test for GH #124: mcmc.run() must be called with
        extra_fields=("diverging",), or numpyro never records divergences
        and n_divergences is hardcoded 0 regardless of what actually
        happened during sampling."""
        from numpyro.infer import MCMC

        captured = {}
        original_run = MCMC.run

        def spy_run(self, *args, **kwargs):
            captured["extra_fields"] = kwargs.get("extra_fields")
            return original_run(self, *args, **kwargs)

        monkeypatch.setattr(MCMC, "run", spy_run)

        run_nuts_sampling(
            tiny_problem["model"], tiny_problem["init_params"],
            tiny_problem["basis"], tiny_problem["obs"], tiny_problem["valid"],
            tiny_problem["n_rows"], tiny_problem["n_cols"], cell_area=1.0,
            parameterization="resistance", n_samples=5, warmup=2, seed=0,
        )

        assert captured["extra_fields"] == ("diverging",)

    def test_diverging_is_collected_by_numpyro_without_extra_fields(self, tiny_problem):
        """Documents the actual mechanism behind GH #124, per review on #132:
        NUTS/HMC's `default_fields` is `("z", "diverging")` (see numpyro's
        `infer.hmc.HMC.default_fields`), so `mcmc.get_extra_fields()` already
        contains "diverging" for every NUTS run whether or not `extra_fields=`
        is passed to `mcmc.run()` -- passing `extra_fields=("diverging",)`
        (#132's fix) is harmless but not what makes `n_divergences` real; it
        always was. This runs the same model-building path run_nuts_sampling
        uses, deliberately WITHOUT extra_fields=, so a future numpyro release
        that ever stopped collecting "diverging" by default -- the actual
        failure mode #124 described -- would fail this test instead of
        silently falling back to run_nuts_sampling's n_divergences=0."""
        from numpyro.infer import MCMC, NUTS

        numpyro_model, flat_init, _ = _build_numpyro_model(
            tiny_problem["model"], jnp.array(tiny_problem["basis"]),
            jnp.array(tiny_problem["obs"]), jnp.array(tiny_problem["valid"]),
            tiny_problem["n_rows"], tiny_problem["n_cols"], 1.0,
            "resistance", tiny_problem["init_params"],
        )
        kernel = NUTS(numpyro_model)
        mcmc = MCMC(kernel, num_warmup=2, num_samples=5, progress_bar=False)
        with jax.disable_jit():
            mcmc.run(jax.random.PRNGKey(0), init_params={
                "params": flat_init,
                "alpha": jnp.array(0.0),
                "gamma": jnp.array(1.0),
            })

        extra = mcmc.get_extra_fields()
        assert "diverging" in extra
        assert np.array(extra["diverging"]).shape == (5,)

    def test_nuts_elapsed_positive(self, nuts_result):
        """Test that elapsed time is positive."""
        result = nuts_result

        assert result["elapsed"] > 0

    def test_nuts_summary_has_quantiles(self, nuts_result):
        """Test that summary contains quantile information."""
        result = nuts_result

        assert "params" in result["summary"]
        assert "alpha" in result["summary"]
        assert "gamma" in result["summary"]
        param_summary = result["summary"]["params"]
        for key in ("mean", "sd", "q025", "q50", "q975"):
            assert key in param_summary
        for site in ("alpha", "gamma"):
            for key in ("mean", "sd", "q025", "q50", "q975"):
                assert key in result["summary"][site]


class TestRunAdviSampling:
    """Tests for run_advi_sampling."""

    def test_advi_returns_expected_keys(self, advi_result):
        """Test that ADVI sampling returns dict with all expected keys."""
        result = advi_result

        expected_keys = {"samples", "summary", "best_elbo", "converged", "elapsed"}
        assert expected_keys == set(result.keys())

    def test_advi_samples_shape(self, advi_result):
        """Test that ADVI produces samples with correct shape."""
        n_samples = 10
        result = advi_result

        assert "params" in result["samples"]
        assert "alpha" in result["samples"]
        assert "gamma" in result["samples"]
        assert result["samples"]["params"].shape[0] == n_samples
        assert result["samples"]["alpha"].shape[0] == n_samples
        assert result["samples"]["gamma"].shape[0] == n_samples
        assert isinstance(result["samples"]["params"], np.ndarray)

    def test_advi_best_elbo_is_float(self, advi_result):
        """Test that best_elbo is a float."""
        result = advi_result

        assert isinstance(result["best_elbo"], float)

    def test_advi_converged_is_bool(self, advi_result):
        """Test that converged is a boolean."""
        result = advi_result

        assert isinstance(result["converged"], bool)

    def test_advi_elapsed_positive(self, advi_result):
        """Test that elapsed time is positive."""
        result = advi_result

        assert result["elapsed"] > 0
