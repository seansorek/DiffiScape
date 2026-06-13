import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Skip entire module at collection time if torch is absent — this prevents
# ModuleNotFoundError when _load() tries to exec 05_torch_pipeline.py.
torch = pytest.importorskip("torch")

_SRC = Path(__file__).parent.parent / "diff_cs"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tp = _load("torch_pipeline", "05_torch_pipeline.py")


def _grid(n_rows=8, n_cols=8, n_feat=3, seed=0):
    """Build (basis_grid, valid_mask, basis_valid) for grid-context nets."""
    rng = np.random.default_rng(seed)
    n_cells = n_rows * n_cols
    valid = np.ones(n_cells, dtype=bool)
    basis_valid = rng.standard_normal((int(valid.sum()), n_feat))
    grid = np.zeros((n_feat, n_cells))
    grid[:, valid] = basis_valid.T
    grid = grid.reshape(1, n_feat, n_rows, n_cols)
    return (torch.tensor(grid, dtype=torch.float64),
            valid,
            torch.tensor(basis_valid, dtype=torch.float64))


class TestResistanceNet:
    def test_forward_shape(self):
        net = tp.ResistanceNet(n_features=3)
        out = net(torch.randn(25, 3))
        assert out.shape == (25,)

    def test_output_positive(self):
        net = tp.ResistanceNet(n_features=3)
        out = net(torch.randn(25, 3))
        assert (out > 0).all()

    def test_gradient_flows(self):
        net = tp.ResistanceNet(n_features=2)
        out = net(torch.randn(9, 2))
        out.sum().backward()
        assert all(p.grad is not None for p in net.parameters())


class TestConvResistanceNet:
    def test_forward_shape(self):
        net = tp.ConvResistanceNet(n_features=3).double()
        grid, valid, basis_valid = _grid(n_feat=3)
        out = net(grid, valid, basis_valid)
        # Returns one resistance value per valid pixel.
        assert out.shape == (int(valid.sum()),)

    def test_output_positive(self):
        net = tp.ConvResistanceNet(n_features=3).double()
        grid, valid, basis_valid = _grid(n_feat=3)
        out = net(grid, valid, basis_valid)
        assert (out > 0).all()

    def test_gradient_flows(self):
        net = tp.ConvResistanceNet(n_features=2).double()
        grid, valid, basis_valid = _grid(n_feat=2)
        out = net(grid, valid, basis_valid)
        out.sum().backward()
        assert all(p.grad is not None for p in net.parameters()
                   if p.requires_grad)
