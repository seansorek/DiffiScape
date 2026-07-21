"""
05_torch_pipeline.py
Neural Network Resistance + Differentiable Circuit Solve + PPP Likelihood

Thin re-export aggregator. The implementation lives in the ``torch_pipeline/``
package (sibling directory of this file) — see that package's modules for the
actual code:

  - torch_pipeline/constants.py            numerical defaults, GPU flags
  - torch_pipeline/_module_loaders.py       lazy imports of 03_/04_ scripts
  - torch_pipeline/autograd_functions.py    differentiable circuit solves
  - torch_pipeline/resistance_nets.py       MLP / Conv / IRL / Spline-GAM nets
  - torch_pipeline/verify_gradients.py      finite-difference gradient checks
  - torch_pipeline/optimization.py          run_torch_optimization (MAP/Adam)
  - torch_pipeline/samplers/                Langevin (MALA/ULA), HMC/NUTS, ADVI

This file stays at its exact original path/name because R/torch_bridge.R
loads it via ``reticulate::import_from_path("05_torch_pipeline", path = ...)``
and resolves entry points by string name
(``.ds_env$torch_module[[fn_name]](...)`` in ``ds_torch_call()``), and the
Python test suite (``inst/python/tests/test_*.py``) loads it the same way via
``importlib.util.spec_from_file_location``. Every top-level name that used to
live directly in this file (before the package split) is re-exported here
unchanged, so neither R nor the existing tests need any changes.

Replaces the R-based GAM profile optimizer (05e / 06c) with an end-to-end
PyTorch pipeline:

  - ResistanceNet: MLP(covariates) → R(x) with linear skip connection
    (starts at the log-linear model, learns nonlinear corrections)
  - Global circuit solve via custom autograd Function (reuses 03_circuit_solver)
  - Parametric PPP intensity: log λ = α + γ · log(1 + C)
  - Adam optimizer with cosine LR scheduling + early stopping

Speed gains vs the GAM-profile pipeline (06c):
  1. Global solver (1 CG solve) instead of diff_omniscape (~23K local solves)
  2. No GAM inner loop (bam fits eliminated)
  3. Relaxed CG tolerance (1e-6 default instead of 1e-10)
  4. Adam handles noisy gradients → no restart loop

Typical runtime: ~5-15 min on 777×745 grid (vs hours for 06c)

Called from R via reticulate (03c_torch_interface.R / 06d_run_torch_pipeline.R).
"""

# ---------------------------------------------------------------------------
# Incidental module-level imports — re-exported for exact namespace parity
# with the pre-split monolith (some external code may reference
# 05_torch_pipeline.torch / .np / etc. directly, not just the functions).
# ---------------------------------------------------------------------------
import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import spsolve

# ---------------------------------------------------------------------------
# Import the torch_pipeline/ implementation package.
#
# This file is loaded both by R (reticulate::import_from_path, which uses
# importlib machinery without a real parent package) and by the test suite
# (importlib.util.spec_from_file_location, registering the loaded module
# under the name "torch_pipeline" in sys.modules — see inst/python/tests/
# test_torch_pipeline.py / test_irl_resistance.py). Neither gives this file a
# package context, AND the test loaders' choice of module name collides with
# the subpackage directory's own name, so:
#   - "from .torch_pipeline import ..." fails ("no known parent package"),
#   - "import torch_pipeline" / "from torch_pipeline import ..." would bind
#     to whatever the test harness already stuffed into
#     sys.modules["torch_pipeline"] (this very shim module, mid-import),
#     not the subpackage.
# So the subpackage is loaded explicitly by file path under a private
# sys.modules key that cannot collide with anything else, mirroring how
# _get_circuit_module() / _get_diff_omniscape_module() resolve their sibling
# scripts by path rather than by ambient import.
# ---------------------------------------------------------------------------
_PKG_DIR = Path(__file__).resolve().parent / "torch_pipeline"
_PKG_KEY = "_diff_cs_torch_pipeline_impl"

if _PKG_KEY not in sys.modules:
    _pkg_spec = importlib.util.spec_from_file_location(
        _PKG_KEY, str(_PKG_DIR / "__init__.py"),
        submodule_search_locations=[str(_PKG_DIR)],
    )
    _pkg_mod = importlib.util.module_from_spec(_pkg_spec)
    sys.modules[_PKG_KEY] = _pkg_mod
    _pkg_spec.loader.exec_module(_pkg_mod)

_tp = sys.modules[_PKG_KEY]


def _tp_import(submodule_name):
    """Import a torch_pipeline submodule (e.g. 'constants', 'samplers.hmc')
    under the private package key, so its own internal relative imports
    ("from .constants import ...", "from ..resistance_nets import ...")
    resolve correctly."""
    full_name = f"{_PKG_KEY}.{submodule_name}"
    if full_name not in sys.modules:
        importlib.import_module(full_name)
    return sys.modules[full_name]


# ---------------------------------------------------------------------------
# Constants / GPU flags
# ---------------------------------------------------------------------------
_constants = _tp_import("constants")
DEFAULT_R_MIN = _constants.DEFAULT_R_MIN
DEFAULT_R_MAX = _constants.DEFAULT_R_MAX
DEFAULT_CLAMP_BETA = _constants.DEFAULT_CLAMP_BETA
DEFAULT_CG_TOL = _constants.DEFAULT_CG_TOL
_GPU_AVAILABLE = _constants._GPU_AVAILABLE
_CUPY_AVAILABLE = _constants._CUPY_AVAILABLE

# ---------------------------------------------------------------------------
# Lazy module loaders (03_circuit_solver.py / 04_diff_omniscape.py)
# ---------------------------------------------------------------------------
_module_loaders = _tp_import("_module_loaders")
_SCRIPT_DIR = _module_loaders._SCRIPT_DIR
_get_circuit_module = _module_loaders._get_circuit_module
_get_diff_omniscape_module = _module_loaders._get_diff_omniscape_module

# ---------------------------------------------------------------------------
# Autograd functions: cupy/torch bridges, adjoint gradients, circuit solves
# ---------------------------------------------------------------------------
_autograd_functions = _tp_import("autograd_functions")
_torch_to_cupy = _autograd_functions._torch_to_cupy
_cupy_to_torch = _autograd_functions._cupy_to_torch
_softplus_clamp = _autograd_functions._softplus_clamp
_gradient_wrt_R = _autograd_functions._gradient_wrt_R
_CircuitSolveFn = _autograd_functions._CircuitSolveFn
_AbsorptionCircuitSolveFn = _autograd_functions._AbsorptionCircuitSolveFn
_compute_interp_weights = _autograd_functions._compute_interp_weights
_interpolate_block_grid = _autograd_functions._interpolate_block_grid
_interpolate_backward_fast = _autograd_functions._interpolate_backward_fast
_local_adjoint_dl_dR = _autograd_functions._local_adjoint_dl_dR
_DiffOmniscapeSolveFn = _autograd_functions._DiffOmniscapeSolveFn

# ---------------------------------------------------------------------------
# Resistance nets (+ shared PPP log-likelihood)
# ---------------------------------------------------------------------------
_resistance_nets = _tp_import("resistance_nets")
LogLinearResistanceNet = _resistance_nets.LogLinearResistanceNet
ResistanceNet = _resistance_nets.ResistanceNet
_ConvResBlock = _resistance_nets._ConvResBlock
ConvResistanceNet = _resistance_nets.ConvResistanceNet
IRLResistanceNet = _resistance_nets.IRLResistanceNet
_bspline_knots = _resistance_nets._bspline_knots
_bspline_basis_matrix = _resistance_nets._bspline_basis_matrix
_bspline_basis_matrix_torch = _resistance_nets._bspline_basis_matrix_torch
_diff_penalty_matrix = _resistance_nets._diff_penalty_matrix
SplineResistanceNet = _resistance_nets.SplineResistanceNet
_ppp_loglik = _resistance_nets._ppp_loglik

# ---------------------------------------------------------------------------
# Gradient verification (finite-difference smoke tests)
# ---------------------------------------------------------------------------
_verify_gradients = _tp_import("verify_gradients")
verify_circuit_gradient = _verify_gradients.verify_circuit_gradient
verify_conv_gradient = _verify_gradients.verify_conv_gradient
verify_softrl_gradient = _verify_gradients.verify_softrl_gradient
verify_spline_gradient = _verify_gradients.verify_spline_gradient

# ---------------------------------------------------------------------------
# MAP optimisation
# ---------------------------------------------------------------------------
_optimization = _tp_import("optimization")
run_torch_optimization = _optimization.run_torch_optimization

# ---------------------------------------------------------------------------
# Bayesian samplers
# ---------------------------------------------------------------------------
_samplers_common = _tp_import("samplers.common")
_compute_ess_chain = _samplers_common._compute_ess_chain
_setup_sampling_state = _samplers_common._setup_sampling_state

_samplers_langevin = _tp_import("samplers.langevin")
run_langevin_sampling = _samplers_langevin.run_langevin_sampling

_samplers_hmc = _tp_import("samplers.hmc")
run_hmc_sampling = _samplers_hmc.run_hmc_sampling

_samplers_advi = _tp_import("samplers.advi")
run_advi = _samplers_advi.run_advi

__all__ = [
    "DEFAULT_R_MIN", "DEFAULT_R_MAX", "DEFAULT_CLAMP_BETA", "DEFAULT_CG_TOL",
    "_GPU_AVAILABLE", "_CUPY_AVAILABLE",
    "_SCRIPT_DIR", "_get_circuit_module", "_get_diff_omniscape_module",
    "_torch_to_cupy", "_cupy_to_torch", "_softplus_clamp", "_gradient_wrt_R",
    "_CircuitSolveFn", "_AbsorptionCircuitSolveFn",
    "_compute_interp_weights", "_interpolate_block_grid",
    "_interpolate_backward_fast", "_local_adjoint_dl_dR",
    "_DiffOmniscapeSolveFn",
    "LogLinearResistanceNet", "ResistanceNet", "_ConvResBlock", "ConvResistanceNet", "IRLResistanceNet",
    "_bspline_knots", "_bspline_basis_matrix", "_bspline_basis_matrix_torch",
    "_diff_penalty_matrix", "SplineResistanceNet", "_ppp_loglik",
    "verify_circuit_gradient", "verify_conv_gradient",
    "verify_softrl_gradient", "verify_spline_gradient",
    "run_torch_optimization",
    "_compute_ess_chain", "_setup_sampling_state",
    "run_langevin_sampling", "run_hmc_sampling", "run_advi",
]
