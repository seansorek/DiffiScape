"""Tests for neural optimization in diffiscape_jax.optimize."""
import numpy as np
import pytest

# Skip all tests if dependencies are not available
jax = pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("jaxscape")

from diffiscape_jax.optimize import run_neural_optimization


class TestRunNeuralOptimization:
    """Tests for run_neural_optimization."""

    @pytest.fixture()
    def small_problem(self):
        """Create a small optimization problem for testing."""
        n_rows, n_cols, n_basis = 8, 8, 2
        n_cells = n_rows * n_cols
        rng = np.random.default_rng(42)
        basis = rng.standard_normal((n_cells, n_basis))
        obs = rng.poisson(3, n_cells).astype(float)
        valid = np.ones(n_cells, dtype=bool)
        return {
            "basis_values": basis,
            "obs_counts": obs,
            "valid_mask": valid,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_cells": n_cells,
        }

    def test_mlp_returns_expected_keys(self, small_problem):
        """Test that MLP optimization returns dict with all expected keys."""
        result = run_neural_optimization(
            small_problem["basis_values"],
            small_problem["obs_counts"],
            small_problem["valid_mask"],
            small_problem["n_rows"],
            small_problem["n_cols"],
            cell_area=1.0,
            model_type="mlp",
            model_config={"hidden_dim": 16, "n_hidden_layers": 1},
            optim_config={"lr": 0.01, "n_epochs": 5, "patience": 10},
            parameterization="resistance",
            seed=42,
            verbose=False,
        )

        expected_keys = {
            "resistance", "best_loglik", "loss_history",
            "n_epochs_run", "elapsed", "model_type",
        }
        assert expected_keys == set(result.keys())

    def test_mlp_resistance_shape(self, small_problem):
        """Test that MLP produces resistance with correct shape."""
        result = run_neural_optimization(
            small_problem["basis_values"],
            small_problem["obs_counts"],
            small_problem["valid_mask"],
            small_problem["n_rows"],
            small_problem["n_cols"],
            cell_area=1.0,
            model_type="mlp",
            model_config={"hidden_dim": 16, "n_hidden_layers": 1},
            optim_config={"lr": 0.01, "n_epochs": 5, "patience": 10},
            seed=42,
            verbose=False,
        )

        assert result["resistance"].shape == (small_problem["n_cells"],)
        assert isinstance(result["resistance"], np.ndarray)

    def test_mlp_finite_values(self, small_problem):
        """Test that MLP optimization produces finite resistance values."""
        result = run_neural_optimization(
            small_problem["basis_values"],
            small_problem["obs_counts"],
            small_problem["valid_mask"],
            small_problem["n_rows"],
            small_problem["n_cols"],
            cell_area=1.0,
            model_type="mlp",
            model_config={"hidden_dim": 16, "n_hidden_layers": 1},
            optim_config={"lr": 0.01, "n_epochs": 5, "patience": 10},
            seed=42,
            verbose=False,
        )

        assert np.all(np.isfinite(result["resistance"]))
        assert np.all(result["resistance"] > 0)
        assert np.isfinite(result["best_loglik"])

    def test_mlp_runs_epochs(self, small_problem):
        """Test that MLP runs the expected number of epochs."""
        n_epochs = 10
        result = run_neural_optimization(
            small_problem["basis_values"],
            small_problem["obs_counts"],
            small_problem["valid_mask"],
            small_problem["n_rows"],
            small_problem["n_cols"],
            cell_area=1.0,
            model_type="mlp",
            model_config={"hidden_dim": 16, "n_hidden_layers": 1},
            optim_config={"lr": 0.01, "n_epochs": n_epochs, "patience": n_epochs + 1},
            seed=42,
            verbose=False,
        )

        assert result["n_epochs_run"] == n_epochs
        assert len(result["loss_history"]) == n_epochs
        assert result["elapsed"] > 0

    def test_mlp_early_stopping(self, small_problem):
        """Test that MLP early-stops when loss plateaus."""
        patience = 3
        result = run_neural_optimization(
            small_problem["basis_values"],
            small_problem["obs_counts"],
            small_problem["valid_mask"],
            small_problem["n_rows"],
            small_problem["n_cols"],
            cell_area=1.0,
            model_type="mlp",
            model_config={"hidden_dim": 16, "n_hidden_layers": 1},
            optim_config={"lr": 0.0, "n_epochs": 100, "patience": patience},
            seed=42,
            verbose=False,
        )

        # Zero LR -> no progress -> early stop
        assert result["n_epochs_run"] <= patience + 1

    def test_model_type_echoed(self, small_problem):
        """Test that model_type is echoed in the result."""
        result = run_neural_optimization(
            small_problem["basis_values"],
            small_problem["obs_counts"],
            small_problem["valid_mask"],
            small_problem["n_rows"],
            small_problem["n_cols"],
            model_type="mlp",
            model_config={"hidden_dim": 8, "n_hidden_layers": 1},
            optim_config={"lr": 0.01, "n_epochs": 3, "patience": 10},
            seed=42,
            verbose=False,
        )

        assert result["model_type"] == "mlp"

    def test_spline_gam_runs(self, small_problem):
        """Test that spline_gam model type runs without error."""
        result = run_neural_optimization(
            small_problem["basis_values"],
            small_problem["obs_counts"],
            small_problem["valid_mask"],
            small_problem["n_rows"],
            small_problem["n_cols"],
            cell_area=1.0,
            model_type="spline_gam",
            model_config={"n_knots": 5},
            optim_config={"lr": 0.01, "n_epochs": 3, "patience": 10},
            seed=42,
            verbose=False,
        )

        assert result["resistance"].shape == (small_problem["n_cells"],)
        assert result["model_type"] == "spline_gam"
        assert np.all(np.isfinite(result["resistance"]))

    def test_irl_runs(self, small_problem):
        """Test that IRL model type runs without error (no adjacency)."""
        result = run_neural_optimization(
            small_problem["basis_values"],
            small_problem["obs_counts"],
            small_problem["valid_mask"],
            small_problem["n_rows"],
            small_problem["n_cols"],
            cell_area=1.0,
            model_type="irl",
            model_config={"hidden_dim": 8, "n_hidden_layers": 1},
            optim_config={"lr": 0.01, "n_epochs": 3, "patience": 10},
            seed=42,
            verbose=False,
        )

        assert result["resistance"].shape == (small_problem["n_cells"],)
        assert result["model_type"] == "irl"
        assert np.all(np.isfinite(result["resistance"]))

    def test_invalid_model_type_raises(self, small_problem):
        """Test that an invalid model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type"):
            run_neural_optimization(
                small_problem["basis_values"],
                small_problem["obs_counts"],
                small_problem["valid_mask"],
                small_problem["n_rows"],
                small_problem["n_cols"],
                model_type="invalid",
                verbose=False,
            )

    def test_default_configs(self, small_problem):
        """Test that None configs use sensible defaults."""
        result = run_neural_optimization(
            small_problem["basis_values"],
            small_problem["obs_counts"],
            small_problem["valid_mask"],
            small_problem["n_rows"],
            small_problem["n_cols"],
            model_type="mlp",
            model_config=None,
            optim_config=None,
            seed=42,
            verbose=False,
        )

        # Should run with defaults (300 epochs, patience 30)
        assert result["n_epochs_run"] > 0
        assert "resistance" in result
