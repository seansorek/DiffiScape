"""Tests for diffiscape_jax.window module."""
import numpy as np
import pytest

# Try to import JAX and JAXScape - skip tests if not available
jax_available = False
jaxscape_available = False
try:
    import jax.numpy as jnp
    jax_available = True
except ImportError:
    pass

try:
    from jaxscape import GridGraph, ResistanceDistance
    jaxscape_available = True
except ImportError:
    pass

# Skip all tests if JAX or JAXScape is not available
pytestmark = pytest.mark.skipif(
    not (jax_available and jaxscape_available),
    reason="jax or jaxscape not installed"
)

from diffiscape_jax.window import cumulative_current


def test_cumulative_current_shape():
    """Test that cumulative_current returns correct output shape."""
    n_rows, n_cols = 20, 20
    resistance = np.random.uniform(1, 100, (n_rows, n_cols))
    result = cumulative_current(resistance, n_rows, n_cols,
                                radius=5, block_size=3)
    assert result["current"].shape == (n_rows, n_cols)
    assert result["elapsed"] > 0


def test_cumulative_current_output_modes():
    """Test that cumulative_current respects output parameter."""
    n_rows, n_cols = 15, 15
    resistance = np.ones((n_rows, n_cols)) * 10.0

    # Current mode should work
    curr = cumulative_current(resistance, n_rows, n_cols,
                              radius=5, block_size=3, output="current")
    assert curr["current"] is not None
    assert curr["voltage"] is None

    # Voltage and "both" modes should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        cumulative_current(resistance, n_rows, n_cols,
                          radius=5, block_size=3, output="voltage")

    with pytest.raises(NotImplementedError):
        cumulative_current(resistance, n_rows, n_cols,
                          radius=5, block_size=3, output="both")


def test_cumulative_current_all_keys_present():
    """Test that all expected keys are present in output."""
    n_rows, n_cols = 10, 10
    resistance = np.ones((n_rows, n_cols)) * 10.0
    result = cumulative_current(resistance, n_rows, n_cols)
    assert "current" in result
    assert "voltage" in result
    assert "elapsed" in result


def test_cumulative_current_nonzero_accumulation():
    """Test that cumulative_current produces non-zero accumulated values."""
    n_rows, n_cols = 15, 15
    # Use varying resistance to ensure non-trivial circuit solutions
    resistance = np.random.uniform(1, 100, (n_rows, n_cols))
    result = cumulative_current(resistance, n_rows, n_cols,
                                radius=5, block_size=3)
    # Central region should have accumulated non-zero values
    # (after multiple window passes)
    assert np.any(result["current"] != 0), \
        "Accumulated current should contain non-zero values"
    assert np.max(result["current"]) > 0, \
        "Maximum accumulated current should be positive"


def test_cumulative_current_input_validation():
    """Test that input validation catches mismatched shapes and invalid radius."""
    n_rows, n_cols = 10, 10
    resistance = np.ones((n_rows, n_cols)) * 10.0

    # Test shape mismatch
    with pytest.raises(ValueError, match="resistance_matrix.shape"):
        cumulative_current(resistance, n_rows=15, n_cols=n_cols,
                          radius=5)

    # Test invalid radius (< 1)
    with pytest.raises(ValueError, match="radius must be >= 1"):
        cumulative_current(resistance, n_rows, n_cols, radius=0)
