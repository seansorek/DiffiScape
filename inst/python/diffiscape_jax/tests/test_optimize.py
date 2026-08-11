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

    @pytest.fixture(scope="class")
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

    @pytest.fixture(scope="class")
    def lbfgs_result(self, small_problem):
        return run_parametric_optimization(
            small_problem["basis_values"], small_problem["obs_counts"],
            small_problem["valid_mask"], small_problem["n_rows"],
            small_problem["n_cols"], cell_area=1.0,
            init_params=small_problem["init_params"], link_fn="exp",
            radius=3, block_size=2, parameterization="resistance",
            method="lbfgs", n_epochs=50, verbose=False,
        )

    @pytest.fixture(scope="class")
    def adam_result(self, small_problem):
        return run_parametric_optimization(
            small_problem["basis_values"], small_problem["obs_counts"],
            small_problem["valid_mask"], small_problem["n_rows"],
            small_problem["n_cols"], cell_area=1.0,
            init_params=small_problem["init_params"], link_fn="exp",
            radius=3, block_size=2, parameterization="resistance",
            method="adam", lr=0.01, n_epochs=20, patience=21,
            verbose=False,
        )

    def test_lbfgs_returns_expected_keys(self, lbfgs_result):
        """Test that L-BFGS returns dict with all expected keys."""
        result = lbfgs_result

        assert "best_params" in result
        assert "alpha" in result
        assert "gamma" in result
        assert "best_loglik" in result
        assert "loss_history" in result
        assert "n_epochs_run" in result
        assert "elapsed" in result
        assert "converged" in result
        assert isinstance(result["alpha"], float)
        assert isinstance(result["gamma"], float)

    def test_lbfgs_param_shape(self, lbfgs_result):
        """Test that L-BFGS returns params with correct shape."""
        result = lbfgs_result

        assert len(result["best_params"]) == 3
        assert isinstance(result["best_params"], np.ndarray)

    def test_lbfgs_runs_epochs(self, lbfgs_result):
        """Test that L-BFGS runs at least one epoch."""
        result = lbfgs_result

        assert result["n_epochs_run"] > 0
        assert isinstance(result["loss_history"], list)
        assert result["elapsed"] > 0

    def test_lbfgs_loglik_finite(self, lbfgs_result):
        """Test that L-BFGS produces a finite log-likelihood."""
        result = lbfgs_result

        assert np.isfinite(result["best_loglik"])
        assert np.all(np.isfinite(result["best_params"]))

    def test_adam_returns_expected_keys(self, adam_result):
        """Test that Adam returns dict with all expected keys."""
        result = adam_result

        assert "best_params" in result
        assert "alpha" in result
        assert "gamma" in result
        assert "best_loglik" in result
        assert "loss_history" in result
        assert "n_epochs_run" in result
        assert "elapsed" in result
        assert "converged" in result
        assert isinstance(result["alpha"], float)
        assert isinstance(result["gamma"], float)

    def test_adam_records_loss_history(self, adam_result):
        """Test that Adam records per-epoch loss history."""
        n_epochs = 20
        result = adam_result

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
        # converged means "the loss plateaued" (the early-stop condition
        # itself firing), which is exactly what happened here (issue #92).
        assert result["converged"] is True

    def test_adam_not_converged_when_epochs_exhausted_while_improving(self, small_problem):
        """converged must be False when the epoch budget runs out before
        the loss ever plateaus -- the inverse of test_adam_early_stopping,
        and the case the old `stall < patience` semantics got backwards."""
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
            lr=0.05,
            n_epochs=5,
            patience=1000,  # far larger than n_epochs -> never plateaus out
            verbose=False,
        )

        assert result["n_epochs_run"] == 5
        assert result["converged"] is False

    def test_adam_best_params_matches_best_loglik(self, small_problem):
        """best_params must be the exact params that were scored to produce
        best_loglik (issue #92) -- not the params one optimizer step later,
        which optax.apply_updates would have produced under the old
        snapshot-after-update ordering. Recomputing the objective at
        best_params independently, via the same function the optimizer
        itself minimizes, must reproduce best_loglik exactly."""
        from diffiscape_jax.core import _connectivity_objective

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
            lr=0.05,
            n_epochs=25,
            patience=25,  # large enough that it never early-stops
            verbose=False,
        )

        full_params = jax.numpy.concatenate([
            jax.numpy.asarray(result["best_params"]),
            jax.numpy.array([result["alpha"], result["gamma"]]),
        ])
        recomputed_loglik = float(_connectivity_objective(
            full_params,
            jax.numpy.array(small_problem["basis_values"]),
            jax.numpy.array(small_problem["valid_mask"]),
            small_problem["n_rows"],
            small_problem["n_cols"],
            1.0,
            "exp",
            3,
            2,
            "resistance",
            jax.numpy.array(small_problem["obs_counts"]),
        ))

        assert recomputed_loglik == pytest.approx(result["best_loglik"], rel=1e-6)

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

    def test_none_init_params_defaults_to_zeros(self, small_problem):
        """init_params=None should default to zeros of length n_basis + 1."""
        result = run_parametric_optimization(
            small_problem["basis_values"],
            small_problem["obs_counts"],
            small_problem["valid_mask"],
            small_problem["n_rows"],
            small_problem["n_cols"],
            cell_area=1.0,
            init_params=None,
            radius=3, block_size=2,
            method="lbfgs", n_epochs=5, verbose=False,
        )

        n_basis = small_problem["basis_values"].shape[1]
        assert len(result["best_params"]) == n_basis + 1


class TestBoundsEnforcement:
    """GH #118: bounds must actually constrain the fit, not just seed the
    starting point -- the JAX solvers used to be fully unconstrained."""

    @pytest.fixture(scope="class")
    def small_problem(self):
        n_rows, n_cols = 10, 10
        n_basis = 2
        rng = np.random.default_rng(7)
        basis = rng.standard_normal((n_rows * n_cols, n_basis))
        obs = rng.poisson(5, n_rows * n_cols).astype(float)
        valid = np.ones(n_rows * n_cols, dtype=bool)
        # Tight box placed away from init_params, so the unconstrained
        # optimum (which the pre-fix solvers would happily wander to)
        # lies outside it: enforcement is only observable if the box
        # actually forces the fit off its unconstrained trajectory.
        init_params = np.array([3.0, 0.0, 0.0])
        lower_bounds = np.array([2.9, -0.1, -0.1])
        upper_bounds = np.array([3.1, 0.1, 0.1])
        return {
            "basis_values": basis,
            "obs_counts": obs,
            "valid_mask": valid,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "init_params": init_params,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
        }

    @pytest.mark.parametrize("method", ["lbfgs", "adam"])
    def test_resistance_params_stay_within_bounds(self, small_problem, method):
        result = run_parametric_optimization(
            small_problem["basis_values"], small_problem["obs_counts"],
            small_problem["valid_mask"], small_problem["n_rows"],
            small_problem["n_cols"], cell_area=1.0,
            init_params=small_problem["init_params"],
            lower_bounds=small_problem["lower_bounds"],
            upper_bounds=small_problem["upper_bounds"],
            radius=3, block_size=2, parameterization="resistance",
            method=method, lr=0.05, n_epochs=50, patience=50,
            verbose=False,
        )

        best_params = np.asarray(result["best_params"])
        assert np.all(best_params >= small_problem["lower_bounds"] - 1e-6)
        assert np.all(best_params <= small_problem["upper_bounds"] + 1e-6)

    @pytest.mark.parametrize("method", ["lbfgs", "adam"])
    def test_alpha_gamma_remain_unconstrained(self, small_problem, method):
        """alpha/gamma are appended internally and must not be clipped by
        resistance-parameter bounds."""
        result = run_parametric_optimization(
            small_problem["basis_values"], small_problem["obs_counts"],
            small_problem["valid_mask"], small_problem["n_rows"],
            small_problem["n_cols"], cell_area=1.0,
            init_params=small_problem["init_params"],
            lower_bounds=small_problem["lower_bounds"],
            upper_bounds=small_problem["upper_bounds"],
            radius=3, block_size=2, parameterization="resistance",
            method=method, lr=0.05, n_epochs=50, patience=50,
            verbose=False,
        )

        # alpha/gamma start at 0.0/1.0, well outside the tight resistance
        # box; if they were accidentally included in the clip they would be
        # pinned there instead of moving with the fit.
        assert result["alpha"] != pytest.approx(0.0, abs=1e-9) or \
            result["gamma"] != pytest.approx(1.0, abs=1e-9)

    def test_no_bounds_remains_unconstrained(self, small_problem):
        """lower_bounds/upper_bounds=None (the default) must not change
        behavior for existing callers that don't pass bounds."""
        result = run_parametric_optimization(
            small_problem["basis_values"], small_problem["obs_counts"],
            small_problem["valid_mask"], small_problem["n_rows"],
            small_problem["n_cols"], cell_area=1.0,
            init_params=small_problem["init_params"],
            radius=3, block_size=2, parameterization="resistance",
            method="lbfgs", n_epochs=50, verbose=False,
        )
        assert np.isfinite(result["best_params"]).all()
