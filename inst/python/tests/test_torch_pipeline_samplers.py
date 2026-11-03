"""Optimization, checkpoint, and posterior checks for the Torch pipeline."""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("submodule,function,options", [
    ("langevin", "run_langevin_sampling", dict(
        burn_in=500, thin=1, precondition=False, step_size=0.3, use_mala=True)),
    ("hmc", "run_hmc_sampling", dict(warmup=500, max_treedepth=4)),
    ("advi", "run_advi", dict(
        max_iter=500, n_elbo_samples=8, lr=0.03, patience=500)),
])
def test_sampler_recovers_known_gaussian(monkeypatch, submodule, function, options):
    """Exercise the real inference algorithm against N(2, 0.7**2)."""
    module = tp._tp_import("samplers." + submodule)

    class GaussianNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.intercept = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        def forward_with_log_R(self, basis):
            return self.intercept.exp().expand(1), self.intercept.expand(1)

        def smoothing_penalty(self):
            return 0.5 * ((self.intercept - 2.0) / 0.7) ** 2

        def get_effective_loglinear(self):
            return [self.intercept.item(), 0.0]

        def get_partial_effects(self, n_grid, posterior_cov=None):
            return [{"effect": [0.0] * n_grid}]

    net = GaussianNet()
    cs = SimpleNamespace(CG_RTOL=1e-6, CG_WARM_START=False,
                         enable_gpu=lambda enabled: None)
    state = dict(
        net=net, alpha=torch.tensor(0.0, dtype=torch.float64),
        raw_gamma=torch.tensor(0.0, dtype=torch.float64),
        basis_t=torch.zeros((1, 1), dtype=torch.float64),
        obs_t=torch.zeros(1, dtype=torch.float64), n_valid=1, n_features=1,
        sampled_params=[net.intercept], param_names=["intercept"], n_total_scalar=1,
        use_absorption=True, use_intensity_spline=False, use_cuda=False,
        dev=torch.device("cpu"), cs=cs, original_rtol=1e-6,
    )
    monkeypatch.setattr(module, "_setup_sampling_state", lambda *args: state)
    monkeypatch.setattr(module, "_AbsorptionCircuitSolveFn", SimpleNamespace(
        apply=lambda resistance, *args: torch.zeros_like(resistance)))
    monkeypatch.setattr(module, "_ppp_loglik", lambda log_lam, *args: log_lam.sum() * 0)
    result = getattr(module, function)(
        np.zeros((1, 1)), np.zeros(1), 1, 1, np.ones(1, dtype=bool), 1.0,
        n_samples=2500, reg_mean=0.0, penalty_scale=1.0,
        seed=42, verbose=False, **options,
    )
    draws = np.asarray(result["samples_effective_loglinear"])[:, 0]
    assert draws.mean() == pytest.approx(2.0, abs=0.08)
    assert draws.var() == pytest.approx(0.7 ** 2, abs=0.08)
    if "n_divergences" in result:
        assert result["n_divergences"] == 0
        with torch.no_grad():
            net.intercept.zero_()
        monkeypatch.setattr(net, "smoothing_penalty",
                            lambda: 0.5 * ((net.intercept - 2.0) / 0.001) ** 2)
        unstable = module.run_hmc_sampling(
            np.zeros((1, 1)), np.zeros(1), 1, 1, np.ones(1, dtype=bool), 1.0,
            n_samples=3, warmup=0, init_step_size=1.0, max_treedepth=1,
            adapt_mass_matrix=False, reg_mean=0.0, penalty_scale=1.0,
            seed=42, verbose=False,
        )
        assert unstable["n_divergences"] == 3


def test_retired_torch_irl_requests_fail_before_fitting():
    with pytest.raises(ValueError, match="model_type"):
        tp.run_torch_optimization(
            np.zeros((1, 1)), np.zeros(1), 1, 1, np.ones(1, dtype=bool), 1.0,
            model_type="irl", verbose=False,
        )
    with pytest.raises(ValueError, match="spline_gam checkpoint"):
        tp.run_hmc_sampling(
            np.zeros((1, 1)), np.zeros(1), 1, 1, np.ones(1, dtype=bool), 1.0,
            model_state={"model_type": "irl"}, verbose=False,
        )


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
    """Run each retained resistance model through the full training path."""

    def test_single_covariate_and_best_checkpoint_match_scored_loss(self):
        rng = np.random.default_rng(1)
        basis = rng.random((64, 1))
        obs = np.zeros(64)
        obs[[9, 18, 27, 36]] = 1
        res = tp.run_torch_optimization(
            basis, obs, 8, 8, np.ones(64, dtype=bool), cell_area=1.0,
            n_epochs=2, warmup_epochs=0, lr=10.0, cg_tol=1e-10,
            reg_mean=0.0, reg_var=0.0, reg_skip=0.0, weight_decay=0.0,
            output_dir=None, seed=1, verbose=False, device="cpu",
        )
        assert np.asarray(res["effective_loglinear"]).shape == (2,)
        assert -res["loglik"] == pytest.approx(min(res["loss_history"]), rel=1e-8)

    @pytest.mark.parametrize("model_type", ["mlp", "conv"])
    def test_neural_model_type(self, model_type):
        basis, obs, valid = _toy_problem(seed=1)
        out = tempfile.mkdtemp()
        res = tp.run_torch_optimization(
            basis, obs, 10, 10, valid, cell_area=900.0,
            model_type=model_type, solver="global_absorption", absorption=0.05,
            n_epochs=3, patience=20, hidden_dim=8, n_hidden_layers=1,
            conv_channels=2, n_conv_layers=1,
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
        assert res["model_type"] == model_type
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
