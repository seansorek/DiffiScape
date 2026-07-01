"""Shared numerical constants and GPU-availability flags for the torch
pipeline package.

Single source of truth for the resistance clamp range and the circuit-solver
tolerance — every net class and ``run_*`` entry point defaults to these names
instead of repeating the literals.
"""
import torch

# ---------------------------------------------------------------------------
# Default numerical constants
# ---------------------------------------------------------------------------

# Differentiable resistance clamp: R is constrained to [R_MIN, R_MAX] via a
# double-softplus in log-space (see _softplus_clamp and the net forward passes).
DEFAULT_R_MIN = 1.0        # minimum resistance (Ω) — open/low-cost habitat
DEFAULT_R_MAX = 5000.0     # maximum resistance (Ω) — near-impassable
DEFAULT_CLAMP_BETA = 5.0   # softplus sharpness for the clamp

# Conjugate-gradient relative tolerance for the circuit solve.
DEFAULT_CG_TOL = 1e-6


# ---------------------------------------------------------------------------
# GPU support
# ---------------------------------------------------------------------------

_GPU_AVAILABLE = torch.cuda.is_available()

try:
    import cupy as cp
    _CUPY_AVAILABLE = cp.cuda.is_available()
except ImportError:
    _CUPY_AVAILABLE = False
