"""Tests for diffiscape_jax.core module."""
import numpy as np
import pytest

# Try to import JAX - skip tests if not available
jax_available = False
try:
    import jax
    import jax.numpy as jnp
    jax_available = True
except ImportError:
    pass

# Skip all tests if JAX is not available
pytestmark = pytest.mark.skipif(not jax_available, reason="jax not installed")

from diffiscape_jax.core import prepare_permeability, forward_solve


class TestPreparePermeability:
    """Tests for prepare_permeability function."""

    def test_prepare_permeability_resistance_mode(self):
        """Test conversion from resistance to permeability."""
        resistance = jnp.array([[10.0, 20.0], [50.0, 100.0]])
        perm = prepare_permeability(resistance, "resistance")
        expected = 1.0 / resistance
        np.testing.assert_allclose(perm, expected, rtol=1e-6)

    def test_prepare_permeability_permeability_mode(self):
        """Test pass-through in permeability mode."""
        permeability = jnp.array([[0.1, 0.05], [0.02, 0.01]])
        result = prepare_permeability(permeability, "permeability")
        np.testing.assert_allclose(result, permeability, rtol=1e-6)

    def test_prepare_permeability_clamps_resistance(self):
        """Test clamping of resistance values."""
        resistance = jnp.array([[0.001, 10000.0]])
        perm = prepare_permeability(resistance, "resistance", r_min=1.0, r_max=5000.0)
        assert float(perm[0, 0]) == pytest.approx(1.0 / 1.0, rel=1e-5)
        assert float(perm[0, 1]) == pytest.approx(1.0 / 5000.0, rel=1e-5)

    def test_prepare_permeability_clamps_permeability(self):
        """Test clamping of permeability values."""
        permeability = jnp.array([[0.0001, 2.0]])
        p_min = 1.0 / 5000.0
        p_max = 1.0
        result = prepare_permeability(
            permeability, "permeability", p_min=p_min, p_max=p_max
        )
        assert float(result[0, 0]) == pytest.approx(p_min, rel=1e-5)
        assert float(result[0, 1]) == pytest.approx(p_max, rel=1e-5)


class TestForwardSolve:
    """Tests for forward_solve function."""

    def test_forward_solve_returns_correct_shape(self):
        """Test that forward_solve returns correct output shape."""
        pytest.importorskip("jaxscape")

        n_rows, n_cols = 10, 10
        resistance = np.ones((n_rows, n_cols)) * 10.0
        result = forward_solve(resistance, n_rows, n_cols)
        assert result["connectivity"].shape == (n_rows, n_cols)
        assert result["elapsed"] > 0
        assert np.all(np.isfinite(result["connectivity"]))

    def test_forward_solve_permeability_mode(self):
        """Test forward_solve in permeability mode."""
        pytest.importorskip("jaxscape")

        n_rows, n_cols = 10, 10
        permeability = np.ones((n_rows, n_cols)) * 0.1
        result = forward_solve(
            permeability, n_rows, n_cols, parameterization="permeability"
        )
        assert result["connectivity"].shape == (n_rows, n_cols)
        assert np.all(np.isfinite(result["connectivity"]))

    def test_forward_solve_with_custom_sources(self):
        """Test forward_solve with custom source indices."""
        pytest.importorskip("jaxscape")

        n_rows, n_cols = 5, 5
        resistance = np.ones((n_rows, n_cols)) * 10.0
        sources = jnp.array([0, 5])  # Two source nodes
        result = forward_solve(resistance, n_rows, n_cols, sources=sources)
        assert result["connectivity"].shape == (n_rows, n_cols)
        assert np.all(np.isfinite(result["connectivity"]))

    def test_forward_solve_returns_dict_with_keys(self):
        """Test that forward_solve returns dict with expected keys."""
        pytest.importorskip("jaxscape")

        n_rows, n_cols = 5, 5
        resistance = np.ones((n_rows, n_cols)) * 10.0
        result = forward_solve(resistance, n_rows, n_cols)
        assert isinstance(result, dict)
        assert "connectivity" in result
        assert "elapsed" in result


class TestCumulativeCurrentCore:
    """Tests for cumulative_current_core -- the differentiable moving-window
    operator shared by _connectivity_objective (training) and
    window.cumulative_current (forward/evaluation), added for GH #105."""

    def test_matches_window_cumulative_current(self):
        """The differentiable core must reproduce window.cumulative_current
        exactly for the same inputs -- this IS the fix: training and
        evaluation share one operator instead of two different ones."""
        pytest.importorskip("jaxscape")
        from diffiscape_jax.core import (
            cumulative_current_core, prepare_permeability,
        )
        from diffiscape_jax.window import cumulative_current

        n_rows, n_cols = 12, 12
        rng = np.random.default_rng(0)
        resistance = rng.uniform(1, 100, (n_rows, n_cols))

        forward = cumulative_current(
            resistance, n_rows, n_cols, radius=4, block_size=3
        )

        permeability = prepare_permeability(
            jnp.array(resistance, dtype=jnp.float64), "resistance"
        )
        objective_side = cumulative_current_core(
            permeability, n_rows, n_cols, radius=4, block_size=3
        )

        np.testing.assert_allclose(
            np.array(objective_side), forward["current"], rtol=1e-8, atol=1e-8
        )

    def test_radius_and_block_size_are_honored(self):
        """Changing radius/block_size must change the connectivity output --
        the pre-fix objective silently ignored both (GH #105)."""
        pytest.importorskip("jaxscape")
        from diffiscape_jax.core import (
            cumulative_current_core, prepare_permeability,
        )

        n_rows, n_cols = 12, 12
        rng = np.random.default_rng(1)
        resistance = rng.uniform(1, 100, (n_rows, n_cols))
        permeability = prepare_permeability(
            jnp.array(resistance, dtype=jnp.float64), "resistance"
        )

        small_window = cumulative_current_core(
            permeability, n_rows, n_cols, radius=2, block_size=2
        )
        large_window = cumulative_current_core(
            permeability, n_rows, n_cols, radius=5, block_size=3
        )

        assert not np.allclose(
            np.array(small_window), np.array(large_window)
        )

    def test_differentiable_wrt_permeability(self):
        """cumulative_current_core must be usable inside jax.grad -- the
        core requirement for using it as a training objective (GH #105).
        The old window.cumulative_current() was not (it converted to numpy
        internally), which is why the objective couldn't call it before."""
        pytest.importorskip("jaxscape")
        from diffiscape_jax.core import cumulative_current_core

        n_rows, n_cols = 8, 8
        rng = np.random.default_rng(2)
        permeability = jnp.array(
            rng.uniform(0.01, 1.0, (n_rows, n_cols)), dtype=jnp.float64
        )

        def summed(perm):
            return jnp.sum(
                cumulative_current_core(perm, n_rows, n_cols, radius=2, block_size=2)
            )

        grad = jax.grad(summed)(permeability)
        assert grad.shape == permeability.shape
        assert np.all(np.isfinite(np.array(grad)))
        assert np.any(np.array(grad) != 0)


class TestConnectivityObjectiveMatchesForwardPath:
    """Regression tests for GH #105: the gradient objective's connectivity
    definition, sign convention, and radius/block_size handling must match
    window.cumulative_current() (the forward/evaluation path consumed by
    evaluate_full_model(), diffiscape() step 6, and ds_posterior())."""

    def _make_problem(self, n_rows=10, n_cols=10, n_basis=1, seed=0):
        rng = np.random.default_rng(seed)
        basis = rng.standard_normal((n_rows * n_cols, n_basis))
        obs = rng.poisson(3, n_rows * n_cols).astype(float)
        valid = np.ones(n_rows * n_cols, dtype=bool)
        return basis, obs, valid, n_rows, n_cols

    def test_objective_connectivity_uses_windowed_operator(self):
        """_connectivity_objective's internal connectivity surface (before
        the PPP likelihood is applied) must equal cumulative_current_core
        run on the same resistance surface -- i.e. it is no longer a
        single-source ResistanceDistance call from cell (0, 0)."""
        pytest.importorskip("jaxscape")
        from diffiscape_jax.core import (
            _apply_link, cumulative_current_core, prepare_permeability,
        )

        basis, obs, valid, n_rows, n_cols = self._make_problem()
        params = jnp.array([3.0, 0.5, 0.0, 1.0])  # [r_0, z_1, alpha, gamma]

        resistance_flat = _apply_link(params[:-2], jnp.array(basis), "exp")
        full_surface = jnp.ones(n_rows * n_cols) * jnp.mean(resistance_flat)
        full_surface = full_surface.at[jnp.array(valid)].set(resistance_flat)
        surface_2d = full_surface.reshape((n_rows, n_cols))
        permeability = prepare_permeability(surface_2d, "resistance")

        expected = cumulative_current_core(
            permeability, n_rows, n_cols, radius=3, block_size=2
        )

        # Reimplements the loglik-free half of _connectivity_objective to
        # isolate the connectivity computation for direct comparison.
        conn_valid_expected = expected.ravel()[jnp.array(valid)]

        from diffiscape_jax.core import ppp_loglik, _connectivity_objective

        loglik = _connectivity_objective(
            params, jnp.array(basis), jnp.array(valid), n_rows, n_cols,
            1.0, "exp", 3, 2, "resistance", jnp.array(obs),
        )
        expected_loglik = ppp_loglik(
            conn_valid_expected, jnp.array(obs), 1.0, params[-2], params[-1]
        )

        assert float(loglik) == pytest.approx(float(expected_loglik), rel=1e-8)

    def test_objective_honors_radius_and_block_size(self):
        """The pre-fix objective accepted radius/block_size but silently
        discarded them (GH #105). After the fix, the log-likelihood must
        change when radius/block_size change (holding params fixed)."""
        pytest.importorskip("jaxscape")
        from diffiscape_jax.core import _connectivity_objective

        basis, obs, valid, n_rows, n_cols = self._make_problem()
        params = jnp.array([3.0, 0.5, 0.0, 1.0])

        loglik_small = _connectivity_objective(
            params, jnp.array(basis), jnp.array(valid), n_rows, n_cols,
            1.0, "exp", 2, 2, "resistance", jnp.array(obs),
        )
        loglik_large = _connectivity_objective(
            params, jnp.array(basis), jnp.array(valid), n_rows, n_cols,
            1.0, "exp", 4, 3, "resistance", jnp.array(obs),
        )

        assert float(loglik_small) != pytest.approx(float(loglik_large), rel=1e-8)

    def test_objective_and_forward_path_agree_on_connectivity_direction(self):
        """A uniformly MORE permeable (lower resistance) surface must move
        the objective's connectivity covariate in the SAME direction as it
        moves window.cumulative_current()'s output -- i.e. the objective and
        the forward/evaluation path share one sign convention. Before the
        fix, the objective used plain ResistanceDistance (increases with
        isolation) while the forward path also accumulates
        ResistanceDistance across windows; after the fix they are the exact
        same computation, so this holds by construction for any resistance
        surface, not just a hand-picked one."""
        pytest.importorskip("jaxscape")
        from diffiscape_jax.core import cumulative_current_core, prepare_permeability
        from diffiscape_jax.window import cumulative_current

        n_rows, n_cols = 10, 10
        rng = np.random.default_rng(3)
        low_resistance = rng.uniform(1, 20, (n_rows, n_cols))
        high_resistance = low_resistance * 5.0  # uniformly more isolated

        forward_low = cumulative_current(
            low_resistance, n_rows, n_cols, radius=3, block_size=2
        )["current"]
        forward_high = cumulative_current(
            high_resistance, n_rows, n_cols, radius=3, block_size=2
        )["current"]

        perm_low = prepare_permeability(jnp.array(low_resistance), "resistance")
        perm_high = prepare_permeability(jnp.array(high_resistance), "resistance")
        objective_low = np.array(
            cumulative_current_core(perm_low, n_rows, n_cols, 3, 2)
        )
        objective_high = np.array(
            cumulative_current_core(perm_high, n_rows, n_cols, 3, 2)
        )

        # Direction of change (mean level) must agree between the two paths.
        forward_direction = np.sign(forward_high.mean() - forward_low.mean())
        objective_direction = np.sign(objective_high.mean() - objective_low.mean())
        assert forward_direction == objective_direction
        assert forward_direction != 0


