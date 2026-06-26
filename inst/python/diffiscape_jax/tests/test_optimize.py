"""Tests for diffiscape_jax.optimize module."""
import numpy as np
import pytest


# Skip all tests if jax, jaxopt, optax, or jaxscape are not available
jax = pytest.importorskip("jax")
pytest.importorskip("jaxopt")
pytest.importorskip("optax")
pytest.importorskip("jaxscape")

from diffiscape_jax.optimize import run_parametric_optimization


class TestRunParametricOptimization:
    """Tests for run_parametric_optimization."""

    @pytest.fixture()
    def small_problem(self):
        """Create a small optimization problem for testing."""
        n_rows, n_cols = 10, 10
        n_basis = 2
        rng = np.random.default_rng(42)
        basis = rng.standard_normal((n_rows * n_cols, n_basis))
        obs = rng.poisson(5, n_rows * n_cols).astype(float)
        valid = np.ones(n_rows * n_cols, dtype=bool)
        init_params = np.array([3.0, 0.0, 0.0])
        return {
            "basis_values": basis,
            "obs_counts": obs,
            "valid_mask": valid,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "init_params": init_params,
        }

    def test_lbfgs_returns_expected_keys(self, small_problem):
        """Test that L-BFGS returns dict with all expected keys."""
        result = run_parametric_optimization(
            small_problem["basis_values"],
            small_problem["obs_counts"],
            small_problem["valid_mask"],
            small_problem["n_rows"],
            small_problem["n_cols"],
            cell_area=1.0,
            init_params=small_problem["init_params"],
            link_fn="exp",
            radius=3,
            block_size=2,
            parameterization="resistance",
            method="lbfgs",
            n_epochs=50,
            verbose=False,
        )

        assert "best_params" in result
        assert "best_loglik" in result
        assert "loss_history" in result
        assert "n_epochs_run" in result
        assert "elapsed" in result
        assert "converged" in result

    def test_lbfgs_param_shape(self, small_problem):
        """Test that L-BFGS returns params with correct shape."""
        result = run_parametric_optimization(
            small_problem["basis_values"],
            small_problem["obs_counts"],
            small_problem["valid_mask"],
            small_problem["n_rows"],
            small_problem["n_cols"],
            cell_area=1.0,
            init_params=small_problem["init_params"],
            link_fn="exp",
            radius=3,
            block_size=2,
            parameterization="resistance",
            method="lbfgs",
            n_epochs=50,
            verbose=False,
        )

        assert len(result["best_params"]) == 3
        assert isinstance(result["best_params"], np.ndarray)

    def test_lbfgs_runs_epochs(self, small_problem):
        """Test that L-BFGS runs at least one epoch."""
        result = run_parametric_optimization(
            small_problem["basis_values"],
            small_problem["obs_counts"],
            small_problem["valid_mask"],
            small_problem["n_rows"],
            small_problem["n_cols"],
            cell_area=1.0,
            init_params=small_problem["init_params"],
            link_fn="exp",
            radius=3,
            block_size=2,
            parameterization="resistance",
            method="lbfgs",
            n_epochs=50,
            verbose=False,
        )

        assert result["n_epochs_run"] > 0
        assert isinstance(result["loss_history"], list)
        assert result["elapsed"] > 0

    def test_lbfgs_loglik_finite(self, small_problem):
        """Test that L-BFGS produces a finite log-likelihood."""
        result = run_parametric_optimization(
            small_problem["basis_values"],
            small_problem["obs_counts"],
            small_problem["valid_mask"],
            small_problem["n_rows"],
            small_problem["n_cols"],
            cell_area=1.0,
            init_params=small_problem["init_params"],
            link_fn="exp",
            radius=3,
            block_size=2,
            parameterization="resistance",
            method="lbfgs",
            n_epochs=50,
            verbose=False,
        )

        assert np.isfinite(result["best_loglik"])
        assert np.all(np.isfinite(result["best_params"]))

    def test_adam_returns_expected_keys(self, small_problem):
        """Test that Adam returns dict with all expected keys."""
        result = run_parametric_optimization(
            small_problem["basis_values"],
            small_problem["obs_counts"],
            small_problem["valid_mask"],
            small_problem["n_rows"],
            small_problem["n_cols"],
            cell_area=1.0,
            init_params=small_problem["init_params"],
            link_fn="exp",
            radius=3,
            block_size=2,
            parameterization="resistance",
            method="adam",
            lr=0.01,
            n_epochs=20,
            patience=10,
            verbose=False,
        )

        assert "best_params" in result
        assert "best_loglik" in result
        assert "loss_history" in result
        assert "n_epochs_run" in result
        assert "elapsed" in result
        assert "converged" in result

    def test_adam_records_loss_history(self, small_problem):
        """Test that Adam records per-epoch loss history."""
        n_epochs = 15
        result = run_parametric_optimization(
            small_problem["basis_values"],
            small_problem["obs_counts"],
            small_problem["valid_mask"],
            small_problem["n_rows"],
            small_problem["n_cols"],
            cell_area=1.0,
            init_params=small_problem["init_params"],
            link_fn="exp",
            radius=3,
            block_size=2,
            parameterization="resistance",
            method="adam",
            lr=0.01,
            n_epochs=n_epochs,
            patience=n_epochs + 1,  # No early stopping
            verbose=False,
        )

        assert len(result["loss_history"]) == n_epochs
        assert result["n_epochs_run"] == n_epochs

    def test_adam_early_stopping(self, small_problem):
        """Test that Adam early-stops when loss plateaus."""
        patience = 3
        result = run_parametric_optimization(
            small_problem["basis_values"],
            small_problem["obs_counts"],
            small_problem["valid_mask"],
            small_problem["n_rows"],
            small_problem["n_cols"],
            cell_area=1.0,
            init_params=small_problem["init_params"],
            link_fn="exp",
            radius=3,
            block_size=2,
            parameterization="resistance",
            method="adam",
            lr=0.0,  # Zero LR -> no progress -> early stop
            n_epochs=100,
            patience=patience,
            verbose=False,
        )

        # With zero LR, loss never improves after initial eval,
        # so should stop at exactly patience epochs
        assert result["n_epochs_run"] <= patience + 1
        assert result["converged"] is False

    def test_invalid_method_raises(self, small_problem):
        """Test that an invalid method name raises ValueError."""
        with pytest.raises(ValueError, match="method"):
            run_parametric_optimization(
                small_problem["basis_values"],
                small_problem["obs_counts"],
                small_problem["valid_mask"],
                small_problem["n_rows"],
                small_problem["n_cols"],
                cell_area=1.0,
                init_params=small_problem["init_params"],
                method="invalid_method",
                verbose=False,
            )
