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


do = _load("diff_omniscape", "04_diff_omniscape.py")


class TestSolveLocalFocal:
    def test_returns_float_and_dict(self):
        R = np.ones((9, 9)) * 2.0
        C, state = do.solve_local_focal(R, focal_row=4, focal_col=4, radius=2)
        assert isinstance(C, float)
        assert isinstance(state, dict)

    def test_current_nonnegative(self):
        R = np.ones((9, 9)) * 2.0
        C, _ = do.solve_local_focal(R, focal_row=4, focal_col=4, radius=2)
        assert C >= 0.0

    def test_higher_resistance_lower_current(self):
        R_lo = np.ones((9, 9)) * 1.0
        R_hi = np.ones((9, 9)) * 100.0
        C_lo, _ = do.solve_local_focal(R_lo, focal_row=4, focal_col=4, radius=3)
        C_hi, _ = do.solve_local_focal(R_hi, focal_row=4, focal_col=4, radius=3)
        assert C_hi < C_lo

    def test_edge_focal_pixel(self):
        R = np.ones((9, 9)) * 2.0
        C, state = do.solve_local_focal(R, focal_row=1, focal_col=1, radius=2)
        assert C >= 0.0
        assert isinstance(state, dict)


class TestSolveDiffOmniscape:
    def test_output_shape(self):
        R = np.ones((9, 9)) * 2.0
        C_map, forward_states, elapsed = do.solve_diff_omniscape(
            R, radius=2, block_size=2
        )
        assert C_map.shape == (9, 9)

    def test_current_nonnegative(self):
        R = np.ones((9, 9)) * 2.0
        C_map, _, _ = do.solve_diff_omniscape(R, radius=2, block_size=2)
        assert np.all(C_map >= 0.0)

    def test_uniform_symmetry(self):
        R = np.ones((9, 9)) * 2.0
        C_map, _, _ = do.solve_diff_omniscape(R, radius=3, block_size=2)
        # Nonzero entries (focal pixels) should be symmetric for uniform R
        nonzero = C_map > 0
        rows, cols = np.where(nonzero)
        for r, c in zip(rows, cols):
            r_mirror = r
            c_mirror = 8 - c
            if nonzero[r_mirror, c_mirror]:
                assert np.isclose(C_map[r, c], C_map[r_mirror, c_mirror], rtol=1e-5)

    def test_forward_states_list(self):
        R = np.ones((9, 9)) * 2.0
        _, forward_states, _ = do.solve_diff_omniscape(R, radius=2, block_size=3)
        assert isinstance(forward_states, list)
        assert len(forward_states) > 0
        # Each entry is (focal_row, focal_col, state_dict, C_focal)
        fr, fc, state, C_f = forward_states[0]
        assert isinstance(fr, int) and isinstance(fc, int)
        assert isinstance(state, dict)
        assert C_f >= 0.0

    def test_elapsed_is_float(self):
        R = np.ones((9, 9)) * 2.0
        _, _, elapsed = do.solve_diff_omniscape(R, radius=2, block_size=3)
        assert isinstance(elapsed, float)
        assert elapsed >= 0.0
