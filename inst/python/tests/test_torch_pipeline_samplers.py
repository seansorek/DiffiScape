"""Characterization/smoke tests for the heavyweight entry points in
05_torch_pipeline.py: run_torch_optimization (mlp / spline_gam model types)
and the three Bayesian samplers (Langevin, HMC/NUTS, ADVI).

These were written to close a coverage gap identified during the
torch_pipeline package-split refactor (architecture review issue #3):
run_torch_optimization was previously only exercised indirectly via the IRL
path in test_irl_resistance.py, and run_langevin_sampling / run_hmc_sampling
/ run_advi had zero test coverage at all. They are intentionally cheap
(tiny grids, few epochs/iterations/samples) so they run fast in CI while
still exercising the full code path: MAP optimisation -> checkpoint ->
sampler -> posterior summary.

Used both as a pre-refactor baseline and a post-refactor regression check
(see Step A / Step B in the architecture-review plan).
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyamg")

_SRC = Path(__file__).parent.parent / "diff_cs"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tp = _load("torch_pipeline_samplers_under_test", "05_torch_pipeline.py")


def _toy_problem(nr=10, nc=10, nf=2, seed=0, n_obs=15):
    """Small grid + sparse Poisson-ish observation counts."""
    rng = np.random.default_rng(seed)
    n_cells = nr * nc
    valid = np.ones(n_cells, dtype=bool)
    basis = rng.standard_normal((n_cells, nf))
    obs = np.zeros(n_cells)
    obs[rng.choice(n_cells, n_obs, replace=False)] = rng.integers(1, 4, n_obs)
    return basis, obs, valid


class TestRunTorchOptimizationModelTypes:
    """run_torch_optimization with model_type="mlp" and "spline_gam" — the
    existing suite only covers model_type="irl" (test_irl_resistance.py)."""

    def test_mlp_model_type(self):
        basis, obs, valid = _toy_problem(seed=1)
        out = tempfile.mkdtemp()
        res = tp.run_torch_optimization(
            basis, obs, 10, 10, valid, cell_area=900.0,
            model_type="mlp", solver="global_absorption", absorption=0.05,
            n_epochs=8, patience=20, hidden_dim=8, n_hidden_layers=1,
            lr=0.02, warmup_epochs=2, source_spacing=1,
            output_dir=out, seed=1, verbose=False,
        )
        for key in ("resistance", "connectivity", "log_lambda", "alpha",
                    "gamma", "loglik", "loss_history", "best_epoch",
                    "n_params", "total_time", "model_type"):
            assert key in res, f"missing key {key!r} in result dict"
        assert np.isfinite(res["loglik"])
        assert np.isfinite(res["resistance"]).all()
        assert np.isfinite(res["connectivity"]).all()
        assert res["model_type"] == "mlp"
        assert os.path.exists(os.path.join(out, "resistance_nn.pt"))

    def test_spline_gam_model_type(self):
        basis, obs, valid = _toy_problem(seed=2)
        out = tempfile.mkdtemp()
        res = tp.run_torch_optimization(
            basis, obs, 10, 10, valid, cell_area=900.0,
            model_type="spline_gam", solver="global_absorption",
            absorption=0.05, n_epochs=8, patience=20, n_knots=4,
            spline_degree=3, include_interactions=False,
            lr=0.02, warmup_epochs=2, source_spacing=1,
            output_dir=out, seed=2, verbose=False,
        )
        for key in ("resistance", "connectivity", "log_lambda", "alpha",
                    "gamma", "loglik", "loss_history", "best_epoch",
                    "n_params", "total_time", "model_type"):
            assert key in res, f"missing key {key!r} in result dict"
        assert np.isfinite(res["loglik"])
        assert np.isfinite(res["resistance"]).all()
        assert np.isfinite(res["connectivity"]).all()
        assert res["model_type"] == "spline_gam"
        assert os.path.exists(os.path.join(out, "resistance_nn.pt"))


@pytest.fixture(scope="module")
def spline_checkpoint_dir():
    """Run a tiny spline_gam MAP optimisation once and reuse the checkpoint
    dir for all three sampler smoke tests (they're independent of each
    other given a fixed checkpoint)."""
    basis, obs, valid = _toy_problem(nr=8, nc=8, nf=2, seed=3, n_obs=10)
    out = tempfile.mkdtemp()
    tp.run_torch_optimization(
        basis, obs, 8, 8, valid, cell_area=900.0,
        model_type="spline_gam", solver="global_absorption",
        absorption=0.05, n_epochs=6, patience=20, n_knots=4,
        spline_degree=3, include_interactions=False,
        lr=0.02, warmup_epochs=1, source_spacing=1,
        output_dir=out, seed=3, verbose=False,
    )
    return {"basis": basis, "obs": obs, "valid": valid, "out": out}


class TestSamplersSmoke:
    """Smoke tests: run each sampler for a handful of samples/iterations on
    a real (tiny) MAP checkpoint. Asserts output dict shape, finiteness, and
    that no exception is raised."""

    def _common_kwargs(self, fix):
        return dict(
            basis_values_np=fix["basis"], obs_counts_np=fix["obs"],
            n_rows=8, n_cols=8, valid_mask_np=fix["valid"], cell_area=900.0,
            model_path=os.path.join(fix["out"], "resistance_nn.pt"),
            source_spacing=1, source_from_resistance=True,
            solver="global_absorption", absorption=0.05,
            n_knots=4, spline_degree=3, include_interactions=False,
            lambda_min=0.1, cg_tol=1e-6, seed=11, verbose=False,
        )

    def test_langevin_sampling(self, spline_checkpoint_dir):
        kw = self._common_kwargs(spline_checkpoint_dir)
        res = tp.run_langevin_sampling(
            n_samples=5, burn_in=3, thin=1, precondition=True,
            precondition_warmup=2, fix_smoothing=False, fix_intensity=False,
            use_mala=True, **kw,
        )
        for key in ("samples_effective_loglinear", "samples_alpha",
                    "samples_gamma", "log_posterior_trace", "ess",
                    "summary", "acceptance_rate", "elapsed_time"):
            assert key in res, f"missing key {key!r} in Langevin result"
        eff_ll = np.asarray(res["samples_effective_loglinear"])
        assert eff_ll.shape[0] == 5
        assert np.isfinite(eff_ll).all()
        assert np.isfinite(res["samples_alpha"]).all()
        assert np.isfinite(res["samples_gamma"]).all()
        assert np.isfinite(np.asarray(res["log_posterior_trace"])).all()
        assert 0.0 <= res["acceptance_rate"] <= 1.0

    def test_hmc_sampling(self, spline_checkpoint_dir):
        kw = self._common_kwargs(spline_checkpoint_dir)
        kw.pop("lambda_min")
        res = tp.run_hmc_sampling(
            n_samples=4, warmup=3, max_treedepth=4, target_accept=0.8,
            adapt_mass_matrix=True, fix_smoothing=True, fix_intensity=False,
            lambda_min=0.1, **kw,
        )
        for key in ("samples_effective_loglinear", "samples_alpha",
                    "samples_gamma", "summary", "n_divergences",
                    "elapsed_time"):
            assert key in res, f"missing key {key!r} in HMC result"
        eff_ll = np.asarray(res["samples_effective_loglinear"])
        assert eff_ll.shape[0] == 4
        assert np.isfinite(eff_ll).all()
        assert np.isfinite(res["samples_alpha"]).all()
        assert np.isfinite(res["samples_gamma"]).all()

    def test_advi(self, spline_checkpoint_dir):
        kw = self._common_kwargs(spline_checkpoint_dir)
        kw.pop("lambda_min")
        res = tp.run_advi(
            n_samples=5, max_iter=8, lr=0.05, n_elbo_samples=1,
            full_rank=False, patience=20, fix_smoothing=True,
            fix_intensity=False, lambda_min=0.1, **kw,
        )
        for key in ("samples_effective_loglinear", "samples_alpha",
                    "samples_gamma", "summary", "best_elbo", "converged",
                    "elapsed_time"):
            assert key in res, f"missing key {key!r} in ADVI result"
        eff_ll = np.asarray(res["samples_effective_loglinear"])
        assert eff_ll.shape[0] == 5
        assert np.isfinite(eff_ll).all()
        assert np.isfinite(res["samples_alpha"]).all()
        assert np.isfinite(res["samples_gamma"]).all()
        assert np.isfinite(res["best_elbo"])
