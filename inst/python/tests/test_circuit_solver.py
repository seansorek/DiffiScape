import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).parent.parent / "diff_cs"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cs = _load("circuit_solver", "03_circuit_solver.py")


class TestBuildEdgeList:
    def test_2x2_edge_count(self):
        src, dst, w = cs.build_edge_list(np.ones(4), 2, 2)
        assert len(src) == len(dst) == len(w) == 8

    def test_conductance_harmonic_mean(self):
        # w_ij = 2 / (R_i + R_j)
        R = np.array([1.0, 3.0, 1.0, 3.0])
        _, _, w = cs.build_edge_list(R, 2, 2)
        assert np.any(np.isclose(w, 2.0 / (1.0 + 3.0)))

    def test_no_self_loops(self):
        src, dst, _ = cs.build_edge_list(np.ones(9), 3, 3)
        assert not np.any(src == dst)

    def test_all_weights_positive(self):
        _, _, w = cs.build_edge_list(np.ones(6), 2, 3)
        assert np.all(w > 0)


class TestBuildLaplacian:
    def test_row_sums_zero(self):
        src, dst, w = cs.build_edge_list(np.ones(4), 2, 2)
        L = cs.build_laplacian(src, dst, w, 4)
        row_sums = np.asarray(L.sum(axis=1)).ravel()
        assert np.allclose(row_sums, 0.0, atol=1e-12)

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        R = 1.0 + rng.random(9)
        src, dst, w = cs.build_edge_list(R, 3, 3)
        L = cs.build_laplacian(src, dst, w, 9)
        diff = (L - L.T).data
        assert np.allclose(diff, 0.0, atol=1e-12)

    def test_diagonal_positive(self):
        src, dst, w = cs.build_edge_list(np.ones(9), 3, 3)
        L = cs.build_laplacian(src, dst, w, 9)
        assert np.all(L.diagonal() > 0)


class TestBoundaryMask:
    def test_3x3_all_boundary(self):
        assert cs.get_boundary_mask(3, 3).sum() == 9

    def test_5x5_interior_count(self):
        # (5-2) * (5-2) = 9 interior pixels
        assert (~cs.get_boundary_mask(5, 5)).sum() == 9

    def test_corners_are_boundary(self):
        mask = cs.get_boundary_mask(4, 4)
        assert mask[0] and mask[3] and mask[12] and mask[15]

    def test_first_row_all_boundary(self):
        mask = cs.get_boundary_mask(5, 5)
        assert np.all(mask[:5])


class TestSolveCircuit:
    def test_current_density_nonnegative(self):
        result = cs.solve_circuit(np.ones((5, 5)) * 2.0)
        assert np.all(result["current_density"] >= 0.0)

    def test_boundary_voltage_zero(self):
        result = cs.solve_circuit(np.ones((5, 5)) * 2.0)
        boundary = cs.get_boundary_mask(5, 5)
        assert np.allclose(result["v"][boundary], 0.0, atol=1e-8)

    def test_uniform_resistance_left_right_symmetry(self):
        c2d = cs.solve_circuit(np.ones((7, 7)) * 3.0)["current_density_2d"]
        assert np.allclose(c2d, c2d[:, ::-1], atol=1e-6)

    def test_uniform_resistance_top_bottom_symmetry(self):
        c2d = cs.solve_circuit(np.ones((7, 7)) * 3.0)["current_density_2d"]
        assert np.allclose(c2d, c2d[::-1, :], atol=1e-6)

    def test_output_shape(self):
        result = cs.solve_circuit(np.ones((6, 8)) * 2.0)
        assert result["current_density_2d"].shape == (6, 8)
        assert result["n_rows"] == 6
        assert result["n_cols"] == 8


class TestSolveCircuitAbsorption:
    def test_output_has_required_keys(self):
        result = cs.solve_circuit_absorption(np.ones((5, 5)) * 2.0)
        assert "absorption" in result
        assert "current_density" in result
        assert "current_density_2d" in result

    def test_current_density_nonnegative(self):
        result = cs.solve_circuit_absorption(np.ones((5, 5)) * 2.0)
        assert np.all(result["current_density"] >= 0.0)

    def test_absorption_value_stored(self):
        result = cs.solve_circuit_absorption(np.ones((5, 5)) * 2.0, absorption=0.05)
        assert result["absorption"] == pytest.approx(0.05)
