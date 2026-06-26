"""Tests for diffiscape_jax.core module."""
import numpy as np
import pytest

# Try to import JAX - skip tests if not available
jax_available = False
try:
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
