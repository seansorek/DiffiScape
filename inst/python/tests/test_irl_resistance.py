"""Tests for the IRL (value-shaped) resistance model in 05_torch_pipeline.py.

Covers the IRLResistanceNet forward pass, soft value-iteration behaviour, the
end-to-end finite-difference gradient check through the value iteration and
circuit solve, and a tiny end-to-end MAP optimisation run.
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip the whole module if torch (or the circuit solver's AMG dep) is absent.
torch = pytest.importorskip("torch")
pytest.importorskip("pyamg")

_SRC = Path(__file__).parent.parent / "diff_cs"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tp = _load("torch_pipeline", "05_torch_pipeline.py")


def _grid(n_rows=12, n_cols=12, n_feat=3, n_invalid=3, seed=0):
    """Return (basis_valid, valid_mask, basis_grid_tensor, dims)."""
    rng = np.random.default_rng(seed)
    n_cells = n_rows * n_cols
    valid = np.ones(n_cells, dtype=bool)
    if n_invalid:
        valid[rng.choice(n_cells, n_invalid, replace=False)] = False
    basis_valid = rng.standard_normal((int(valid.sum()), n_feat))
    grid = np.zeros((n_feat, n_cells))
    grid[:, valid] = basis_valid.T
    grid = grid.reshape(1, n_feat, n_rows, n_cols)
    return basis_valid, valid, torch.tensor(grid, dtype=torch.float64), (n_rows, n_cols, n_feat)


class TestIRLResistanceNet:
    def test_forward_shape_and_finiteness(self):
        basis_valid, valid, grid, (nr, nc, nf) = _grid()
        net = tp.IRLResistanceNet(nf, hidden=8, n_layers=2,
                                  vi_beta=1.0, gamma_d=0.9, n_value_iter=40).double()
        R, log_R = net.forward_with_log_R(grid, valid, torch.tensor(basis_valid))
        assert R.shape == (int(valid.sum()),)
        assert log_R.shape == (int(valid.sum()),)
        assert torch.isfinite(R).all() and torch.isfinite(log_R).all()
        assert (R > 0).all()

    def test_logR_mean_tracks_offset(self):
        """Centring V makes `offset` the mean log-resistance level."""
        basis_valid, valid, grid, (nr, nc, nf) = _grid(seed=1)
        net = tp.IRLResistanceNet(nf, hidden=8, n_layers=2, n_value_iter=40).double()
        with torch.no_grad():
            net.offset.copy_(torch.tensor(3.0, dtype=torch.float64))
            _, log_R = net.forward_with_log_R(grid, valid, torch.tensor(basis_valid))
        # Mean log R should sit at offset (within clamp tolerance) and vary.
        assert abs(float(log_R.mean()) - 3.0) < 0.2
        assert float(log_R.std()) > 1e-3

    def test_value_iteration_contracts(self):
        """Soft value iteration is a contraction for gamma_d < 1: the change
        between successive sweeps shrinks toward a fixed point."""
        basis_valid, valid, grid, (nr, nc, nf) = _grid(seed=2)
        net = tp.IRLResistanceNet(nf, hidden=8, n_layers=2,
                                  vi_beta=1.0, gamma_d=0.9).double()
        _, n_feat, H, W = grid.shape
        phi = grid.squeeze(0).reshape(n_feat, -1).T
        reward = (net.skip(phi) + net.mlp(phi)).squeeze(-1).reshape(H, W).detach()
        vg = torch.as_tensor(valid.astype(np.float64).reshape(H, W), dtype=torch.float64)
        # Re-implement the sweep to measure successive-iterate deltas.
        b, gd, NEG = net.vi_beta, net.gamma_d, -1e9
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        nbr_valid = [net._shift(vg, dr, dc) for dr, dc in dirs]
        V = torch.zeros_like(reward)
        deltas = []
        for _ in range(120):
            qs = [reward + gd * V]
            for (dr, dc), nv in zip(dirs, nbr_valid):
                q = reward + gd * net._shift(V, dr, dc)
                qs.append(torch.where(nv > 0.5, q, torch.full_like(q, NEG)))
            V_new = torch.logsumexp(b * torch.stack(qs, 0), 0) / b
            V_new = torch.where(vg > 0.5, V_new, torch.zeros_like(V_new))
            deltas.append(float((V_new - V).abs().max()))
            V = V_new
        # The soft Bellman operator is a gamma_d-contraction: successive-iterate
        # deltas decay geometrically toward a fixed point.
        assert deltas[-1] < deltas[0]
        assert deltas[-1] < 1e-3
        assert deltas[-1] < 0.05 * deltas[0]

    def test_warm_start_sets_reward_skip(self):
        """warm_start seeds the reward skip (sign-flipped) and offset from
        log-linear params [r_0, z_1..z_K]."""
        basis_valid, valid, grid, (nr, nc, nf) = _grid(seed=5)
        net = tp.IRLResistanceNet(nf, hidden=8, n_layers=2, n_value_iter=20).double()
        theta = np.array([2.5] + [0.3 * (k + 1) for k in range(nf)])
        net.warm_start(theta)
        with torch.no_grad():
            # offset takes r_0; skip weights take -z (reward = -cost).
            assert abs(float(net.offset) - 2.5) < 1e-9
            np.testing.assert_allclose(
                net.skip.weight.detach().numpy().ravel(), -theta[1:], atol=1e-9)
            # forward still produces finite, positive resistance afterwards.
            R, _ = net.forward_with_log_R(grid, valid, torch.tensor(basis_valid))
        assert torch.isfinite(R).all() and (R > 0).all()

    def test_invalid_cells_have_zero_value(self):
        basis_valid, valid, grid, (nr, nc, nf) = _grid(seed=3)
        net = tp.IRLResistanceNet(nf, hidden=8, n_layers=2, n_value_iter=30).double()
        _, n_feat, H, W = grid.shape
        phi = grid.squeeze(0).reshape(n_feat, -1).T
        reward = (net.skip(phi) + net.mlp(phi)).squeeze(-1).reshape(H, W).detach()
        vg = torch.as_tensor(valid.astype(np.float64).reshape(H, W), dtype=torch.float64)
        V = net._soft_value_iteration(reward, vg)
        inv = (vg < 0.5)
        assert torch.allclose(V[inv], torch.zeros_like(V[inv]))


class TestIRLGradient:
    def test_softrl_gradient_check(self):
        basis_valid, valid, _, (nr, nc, nf) = _grid(n_rows=10, n_cols=10, seed=4)
        res = tp.verify_softrl_gradient(
            basis_valid, valid, nr, nc,
            hidden_dim=8, n_hidden_layers=2,
            beta=1.0, gamma_d=0.9, n_value_iter=20,
            solver="global_absorption", absorption=0.05, seed=4,
        )
        assert res["pass"], f"max_rel_error={res['max_rel_error']:.2e}"


class TestIRLOptimization:
    def test_end_to_end_map_run(self):
        rng = np.random.default_rng(7)
        nr, nc, nf = 14, 14, 2
        n_cells = nr * nc
        valid = np.ones(n_cells, dtype=bool)
        basis = rng.standard_normal((n_cells, nf))
        obs = np.zeros(n_cells)
        obs[rng.choice(n_cells, 20, replace=False)] = rng.integers(1, 4, 20)

        out = tempfile.mkdtemp()
        res = tp.run_torch_optimization(
            basis, obs, nr, nc, valid, cell_area=900.0,
            model_type="irl", solver="global_absorption", absorption=0.05,
            n_epochs=10, patience=20, hidden_dim=8, n_hidden_layers=2,
            beta=1.0, gamma_d=0.9, n_value_iter=25,
            lr=0.02, warmup_epochs=2, source_spacing=1,
            output_dir=out, seed=7, verbose=False,
        )
        assert np.isfinite(res["loglik"])
        assert np.isfinite(res["resistance"]).all()
        assert np.isfinite(res["connectivity"]).all()
        assert res["model_type"] == "irl"

        ckpt = torch.load(os.path.join(out, "resistance_nn.pt"), weights_only=False)
        assert ckpt["model_type"] == "irl"
        for k in ("beta", "gamma_d", "n_value_iter", "value_scale_init"):
            assert k in ckpt
