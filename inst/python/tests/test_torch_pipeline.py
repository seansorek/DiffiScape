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


class TestResistanceNet:
    def test_forward_shape(self):
        net = tp.ResistanceNet(n_covariates=3)
        out = net(torch.randn(25, 3))
        assert out.shape == (25,)

    def test_output_positive(self):
        net = tp.ResistanceNet(n_covariates=3)
        out = net(torch.randn(25, 3))
        assert (out > 0).all()

    def test_gradient_flows(self):
        net = tp.ResistanceNet(n_covariates=2)
        out = net(torch.randn(9, 2))
        out.sum().backward()
        assert all(p.grad is not None for p in net.parameters())


class TestConvResistanceNet:
    def test_forward_shape(self):
        net = tp.ConvResistanceNet(n_covariates=3)
        x = torch.randn(1, 3, 7, 7)
        out = net(x)
        assert out.shape == (1, 1, 7, 7)

    def test_output_positive(self):
        net = tp.ConvResistanceNet(n_covariates=3)
        out = net(torch.randn(1, 3, 7, 7))
        assert (out > 0).all()
