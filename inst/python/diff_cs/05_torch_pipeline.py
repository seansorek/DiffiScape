"""
05_torch_pipeline.py
Neural Network Resistance + Differentiable Circuit Solve + PPP Likelihood

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

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util
import sys
import os
import time
from pathlib import Path
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Default numerical constants
#
# Single source of truth for the resistance clamp range and the circuit-solver
# tolerance. Previously these literals (1.0 / 5000.0 / 5.0 / 1e-6) were repeated
# as default arguments across every net class and run_* entry point, so a change
# meant hunting down five copies. Functions still accept overrides — they just
# default to these names now.
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


def _torch_to_cupy(t):
    """Zero-copy torch CUDA tensor → cupy array via DLPack."""
    return cp.from_dlpack(t)


def _cupy_to_torch(a, device):
    """Zero-copy cupy array → torch tensor via DLPack."""
    return torch.from_dlpack(a).to(device)


# ---------------------------------------------------------------------------
# Import existing circuit solver
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent


def _get_circuit_module():
    """Lazy-import 03_circuit_solver.py."""
    if not hasattr(_get_circuit_module, "_mod"):
        spec = importlib.util.spec_from_file_location(
            "circuit_solver_03",
            str(_SCRIPT_DIR / "03_circuit_solver.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _get_circuit_module._mod = mod
    return _get_circuit_module._mod


def _get_diff_omniscape_module():
    """Lazy-import 04_diff_omniscape.py."""
    if not hasattr(_get_diff_omniscape_module, "_mod"):
        spec = importlib.util.spec_from_file_location(
            "diff_omniscape_04",
            str(_SCRIPT_DIR / "04_diff_omniscape.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _get_diff_omniscape_module._mod = mod
    return _get_diff_omniscape_module._mod


# ===========================================================================
# Helpers
# ===========================================================================

def _softplus_clamp(R_raw, R_min=DEFAULT_R_MIN, R_max=DEFAULT_R_MAX, beta=DEFAULT_CLAMP_BETA):
    """Double-softplus differentiable clamp to [R_min, R_max].
    Matches 02_resistance_surface.R exactly."""
    R_lo = R_min + F.softplus(R_raw - R_min, beta=beta)
    R = R_max - F.softplus(R_max - R_lo, beta=beta)
    return R


def _gradient_wrt_R(v, adjoint_lambda, dl_dc,
                     edge_src, edge_dst, edge_w,
                     R_flat, n_nodes,
                     source_from_resistance, source_mask):
    """
    Compute dl/dR at each pixel from the adjoint solution.

    Supports both numpy (CPU) and cupy (GPU) arrays — dispatches automatically
    based on the array module of the input arguments.
    """
    cs = _get_circuit_module()
    xp = cs._xp()

    dl_dw = cs.compute_gradient_wrt_weights(
        v, adjoint_lambda, dl_dc,
        edge_src, edge_dst, edge_w, n_nodes,
    )

    denom2 = (R_flat[edge_src] + R_flat[edge_dst]) ** 2
    dw_dR = -2.0 / denom2

    dl_dR = xp.zeros(n_nodes, dtype=xp.float64)
    if cs._use_gpu:
        cp.add.at(dl_dR, edge_src, dl_dw * dw_dR)
        cp.add.at(dl_dR, edge_dst, dl_dw * dw_dR)
    else:
        np.add.at(dl_dR, edge_src, dl_dw * dw_dR)
        np.add.at(dl_dR, edge_dst, dl_dw * dw_dR)

    # Source correction for 1/R injection (∂b/∂R term)
    if source_from_resistance and source_mask is not None:
        sm = xp.asarray(source_mask, dtype=bool).ravel()
        dl_dR[sm] -= adjoint_lambda[sm] / (R_flat[sm] ** 2)

    return dl_dR


# ===========================================================================
# Custom autograd: differentiable global circuit solve
# ===========================================================================

class _CircuitSolveFn(torch.autograd.Function):
    """
    Forward:  R(valid pixels) → C(valid pixels) via AMG-preconditioned CG
    Backward: dl/dC → dl/dR via the adjoint method (reuses cached AMG)

    Supports both CPU and GPU. On GPU, uses cupy ↔ torch DLPack zero-copy
    transfers to keep data on-device throughout.
    """

    @staticmethod
    def forward(ctx, R_valid, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance):
        cs = _get_circuit_module()
        use_gpu = cs._use_gpu
        xp = cs._xp()
        n = n_rows * n_cols

        if use_gpu:
            # GPU path: torch CUDA → cupy (zero-copy)
            R_cp = _torch_to_cupy(R_valid.detach().contiguous())
            R_full = cp.full(n, 1e6, dtype=cp.float64)
            R_full[valid_mask_np] = R_cp.astype(cp.float64)
        else:
            R_np = R_valid.detach().cpu().numpy().astype(np.float64)
            R_full = np.full(n, 1e6, dtype=np.float64)
            R_full[valid_mask_np] = R_np

        result = cs.solve_circuit(
            R_full.reshape(n_rows, n_cols),
            source_spacing=int(source_spacing),
            source_from_resistance=bool(source_from_resistance),
        )

        C_valid = result["current_density"][valid_mask_np].copy()

        ctx.fwd = result
        ctx.valid_mask_np = valid_mask_np
        ctx.n = n
        ctx.sfr = bool(source_from_resistance)
        ctx.use_gpu = use_gpu

        if use_gpu:
            return _cupy_to_torch(C_valid, R_valid.device)
        else:
            return torch.tensor(C_valid, dtype=torch.float64)

    @staticmethod
    def backward(ctx, grad_C):
        fwd = ctx.fwd
        vm = ctx.valid_mask_np
        n = ctx.n
        cs = _get_circuit_module()
        use_gpu = ctx.use_gpu
        xp = cs._xp()

        if use_gpu:
            dl_dc = cp.zeros(n, dtype=cp.float64)
            dl_dc[vm] = _torch_to_cupy(grad_C.detach().contiguous())
        else:
            dl_dc = np.zeros(n, dtype=np.float64)
            dl_dc[vm] = grad_C.detach().cpu().numpy()

        # Adjoint RHS:  dl/dc → dl/dv_interior
        rhs = cs.compute_adjoint_rhs(
            fwd["v"], dl_dc,
            fwd["edge_src"], fwd["edge_dst"], fwd["edge_w"],
            n, fwd["interior_mask"],
        )

        # Adjoint solve (reuses cached preconditioner)
        lam = cs.adjoint_solve(rhs)

        # dl/dR per pixel
        dl_dR = _gradient_wrt_R(
            fwd["v"], lam, dl_dc,
            fwd["edge_src"], fwd["edge_dst"], fwd["edge_w"],
            fwd["R_flat"], n,
            ctx.sfr, fwd.get("source_mask"),
        )

        if use_gpu:
            grad_R = _cupy_to_torch(dl_dR[vm], grad_C.device)
        else:
            grad_R = torch.tensor(dl_dR[vm], dtype=torch.float64)

        # Free forward state to release memory
        del ctx.fwd

        # Gradients: R_valid gets grad; other args are non-tensor → None
        return grad_R, None, None, None, None, None


# ===========================================================================
# Custom autograd: absorption-based global circuit solve (no boundary grounding)
# ===========================================================================

class _AbsorptionCircuitSolveFn(torch.autograd.Function):
    """
    Forward:  R(valid pixels) → C(valid pixels) via absorption-based circuit solve
    Backward: dl/dC → dl/dR via the adjoint method

    Uses (L + α·I)v = b instead of boundary grounding, eliminating corner
    current accumulation while keeping a single global solve.
    """

    @staticmethod
    def forward(ctx, R_valid, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance, absorption_alpha):
        cs = _get_circuit_module()
        use_gpu = cs._use_gpu
        xp = cs._xp()
        n = n_rows * n_cols

        if use_gpu:
            R_cp = _torch_to_cupy(R_valid.detach().contiguous())
            R_full = cp.full(n, 1e6, dtype=cp.float64)
            R_full[valid_mask_np] = R_cp.astype(cp.float64)
        else:
            R_np = R_valid.detach().cpu().numpy().astype(np.float64)
            R_full = np.full(n, 1e6, dtype=np.float64)
            R_full[valid_mask_np] = R_np

        result = cs.solve_circuit_absorption(
            R_full.reshape(n_rows, n_cols),
            absorption=float(absorption_alpha),
            source_spacing=int(source_spacing),
            source_from_resistance=bool(source_from_resistance),
        )

        C_valid = result["current_density"][valid_mask_np].copy()

        ctx.fwd = result
        ctx.valid_mask_np = valid_mask_np
        ctx.n = n
        ctx.sfr = bool(source_from_resistance)
        ctx.use_gpu = use_gpu

        if use_gpu:
            return _cupy_to_torch(C_valid, R_valid.device)
        else:
            return torch.tensor(C_valid, dtype=torch.float64)

    @staticmethod
    def backward(ctx, grad_C):
        fwd = ctx.fwd
        vm = ctx.valid_mask_np
        n = ctx.n
        cs = _get_circuit_module()
        use_gpu = ctx.use_gpu
        xp = cs._xp()

        if use_gpu:
            dl_dc = cp.zeros(n, dtype=cp.float64)
            dl_dc[vm] = _torch_to_cupy(grad_C.detach().contiguous())
        else:
            dl_dc = np.zeros(n, dtype=np.float64)
            dl_dc[vm] = grad_C.detach().cpu().numpy()

        # Adjoint RHS: dl/dc → dl/dv
        # With absorption, all nodes are interior (interior_mask = True everywhere)
        rhs = cs.compute_adjoint_rhs(
            fwd["v"], dl_dc,
            fwd["edge_src"], fwd["edge_dst"], fwd["edge_w"],
            n, fwd["interior_mask"],
        )

        # Adjoint solve (reuses cached AMG/preconditioner for L + α·I)
        lam = cs.adjoint_solve(rhs)

        # dl/dR per pixel (same formula — absorption doesn't add R-dependent terms)
        dl_dR = _gradient_wrt_R(
            fwd["v"], lam, dl_dc,
            fwd["edge_src"], fwd["edge_dst"], fwd["edge_w"],
            fwd["R_flat"], n,
            ctx.sfr, fwd.get("source_mask"),
        )

        if use_gpu:
            grad_R = _cupy_to_torch(dl_dR[vm], grad_C.device)
        else:
            grad_R = torch.tensor(dl_dR[vm], dtype=torch.float64)

        del ctx.fwd

        # R_valid gets grad; other args are non-tensor → None
        return grad_R, None, None, None, None, None, None


# ===========================================================================
# Bilinear interpolation for block-grid connectivity
# ===========================================================================

def _compute_interp_weights(n_rows, n_cols, focal_rows, focal_cols):
    """
    Pre-compute bilinear interpolation indices and weights for every pixel.

    Returns
    -------
    r0_idx, r1_idx : ndarray (n_rows,) int
        Surrounding focal row indices for each row.
    c0_idx, c1_idx : ndarray (n_cols,) int
        Surrounding focal column indices for each column.
    wr : ndarray (n_rows,)
        Row interpolation weights (0-1).
    wc : ndarray (n_cols,)
        Column interpolation weights (0-1).
    """
    fr = np.array(focal_rows, dtype=np.float64)
    fc = np.array(focal_cols, dtype=np.float64)
    n_fr = len(fr)
    n_fc = len(fc)

    all_r = np.arange(n_rows, dtype=np.float64)
    all_c = np.arange(n_cols, dtype=np.float64)

    # Row weights: for each row, find the surrounding focal row segment
    ri = np.searchsorted(fr, all_r, side="right") - 1
    ri = np.clip(ri, 0, n_fr - 2)
    r0_idx = ri.copy()
    r1_idx = np.minimum(ri + 1, n_fr - 1)

    r0_vals = fr[r0_idx]
    r1_vals = fr[r1_idx]
    dr = r1_vals - r0_vals
    wr = np.where(dr > 0, np.clip((all_r - r0_vals) / dr, 0.0, 1.0), 0.0)

    # Column weights
    ci = np.searchsorted(fc, all_c, side="right") - 1
    ci = np.clip(ci, 0, n_fc - 2)
    c0_idx = ci.copy()
    c1_idx = np.minimum(ci + 1, n_fc - 1)

    c0_vals = fc[c0_idx]
    c1_vals = fc[c1_idx]
    dc = c1_vals - c0_vals
    wc = np.where(dc > 0, np.clip((all_c - c0_vals) / dc, 0.0, 1.0), 0.0)

    return r0_idx, r1_idx, c0_idx, c1_idx, wr, wc


def _interpolate_block_grid(C_map, block_size, focal_rows, focal_cols,
                            n_rows, n_cols):
    """
    Bilinear-interpolate a sparse block-grid C_map to full resolution.

    Uses explicit weight computation (not scipy) so forward and backward
    are guaranteed to be exact transposes.
    """
    fr = np.array(focal_rows)
    fc = np.array(focal_cols)
    C_focal = C_map[np.ix_(fr, fc)]  # (n_fr, n_fc)

    r0_idx, r1_idx, c0_idx, c1_idx, wr, wc = _compute_interp_weights(
        n_rows, n_cols, focal_rows, focal_cols
    )

    C_full = np.zeros((n_rows, n_cols), dtype=np.float64)
    for r in range(n_rows):
        w0r = 1.0 - wr[r]
        w1r = wr[r]
        ri0, ri1 = r0_idx[r], r1_idx[r]

        # Vectorized across columns
        w0c = 1.0 - wc
        w1c = wc
        C_full[r, :] = (
            w0r * (w0c * C_focal[ri0, c0_idx] + w1c * C_focal[ri0, c1_idx])
            + w1r * (w0c * C_focal[ri1, c0_idx] + w1c * C_focal[ri1, c1_idx])
        )

    return C_full


def _interpolate_backward_fast(grad_full, focal_rows, focal_cols,
                               n_rows, n_cols):
    """
    Exact transpose of _interpolate_block_grid: scatter full-grid gradients
    back to focal grid using the same bilinear weights.
    """
    n_fr = len(focal_rows)
    n_fc = len(focal_cols)

    r0_idx, r1_idx, c0_idx, c1_idx, wr, wc = _compute_interp_weights(
        n_rows, n_cols, focal_rows, focal_cols
    )

    grad_focal = np.zeros((n_fr, n_fc), dtype=np.float64)

    for r in range(n_rows):
        g_row = grad_full[r, :]
        w0r = 1.0 - wr[r]
        w1r = wr[r]
        ri0, ri1 = r0_idx[r], r1_idx[r]

        w0c = 1.0 - wc
        w1c = wc

        np.add.at(grad_focal[ri0], c0_idx, w0r * w0c * g_row)
        np.add.at(grad_focal[ri0], c1_idx, w0r * w1c * g_row)
        np.add.at(grad_focal[ri1], c0_idx, w1r * w0c * g_row)
        np.add.at(grad_focal[ri1], c1_idx, w1r * w1c * g_row)

    return grad_focal


# ===========================================================================
# Local adjoint: dl/dR (stops before NN chain — torch autograd handles that)
# ===========================================================================

def _local_adjoint_dl_dR(forward_state, dl_dC_focal, n_global):
    """
    Compute dl/dR contribution from one focal pixel's local adjoint.

    This is Steps 1-4 of _local_adjoint_gradient() in 04_diff_omniscape.py,
    stopping at dl/dR per global pixel. Step 5 (dl/dR → dl/dθ via basis/softplus
    chain) is NOT done here — PyTorch autograd handles the NN→R chain.

    Parameters
    ----------
    forward_state : dict
        From solve_local_focal().
    dl_dC_focal : float
        dL/dC_f — gradient of loss w.r.t. current at this focal pixel.
    n_global : int
        Total number of pixels in the global grid (n_rows * n_cols).

    Returns
    -------
    dl_dR_global : ndarray (n_global,)
        Gradient contribution at each global pixel. Mostly zeros.
    """
    v_sub = forward_state["v_sub"]
    L_sub = forward_state["L_sub"]
    focal_j_nonfocal = forward_state["focal_j_nonfocal"]
    focal_w = forward_state["focal_w"]
    focal_j_sub = forward_state["focal_j_sub"]
    f_local = int(forward_state["f_local"])
    nf_edge_src_nf = forward_state["nf_edge_src_nf"]
    nf_edge_dst_nf = forward_state["nf_edge_dst_nf"]
    nf_edge_src_sub = forward_state["nf_edge_src_sub"]
    nf_edge_dst_sub = forward_state["nf_edge_dst_sub"]
    nonfocal_to_sub = forward_state["nonfocal_to_sub"]
    sub_global_map = forward_state["sub_global_map"]
    R_sub_flat = forward_state["R_sub_flat"]
    n_nonfocal = int(forward_state["n_nonfocal"])
    n_sub = int(forward_state["n_sub"])
    source_from_resistance = bool(forward_state["source_from_resistance"])

    # Step 1: Adjoint RHS
    dl_dv_sub = np.zeros(n_nonfocal, dtype=np.float64)
    np.add.at(dl_dv_sub, focal_j_nonfocal, dl_dC_focal * focal_w)

    # Step 2: Adjoint solve
    lambda_sub = spsolve(L_sub, dl_dv_sub)

    # Step 3: dl/dw per edge
    dlam_nf = lambda_sub[nf_edge_src_nf] - lambda_sub[nf_edge_dst_nf]
    dv_nf = v_sub[nf_edge_src_nf] - v_sub[nf_edge_dst_nf]
    dL_dw_nf = -0.5 * dlam_nf * dv_nf

    dL_dw_focal = (dl_dC_focal - lambda_sub[focal_j_nonfocal]) * v_sub[focal_j_nonfocal]

    # Step 4: dl/dR for each sub-window pixel
    dL_dR_sub = np.zeros(n_sub, dtype=np.float64)

    if len(nf_edge_src_sub) > 0:
        denom_sq_nf = (R_sub_flat[nf_edge_src_sub] + R_sub_flat[nf_edge_dst_sub]) ** 2
        dw_dR_nf = -2.0 / denom_sq_nf
        scaled = dL_dw_nf * dw_dR_nf
        np.add.at(dL_dR_sub, nf_edge_src_sub, scaled)
        np.add.at(dL_dR_sub, nf_edge_dst_sub, scaled)

    if len(focal_j_sub) > 0:
        denom_sq_f = (R_sub_flat[f_local] + R_sub_flat[focal_j_sub]) ** 2
        dw_dR_f = -2.0 / denom_sq_f
        scaled_f = dL_dw_focal * dw_dR_f
        np.add.at(dL_dR_sub, f_local, np.sum(scaled_f))
        np.add.at(dL_dR_sub, focal_j_sub, scaled_f)

    # Source-from-resistance correction
    if source_from_resistance and n_nonfocal > 0:
        R_nonfocal = R_sub_flat[nonfocal_to_sub]
        correction = -lambda_sub / (R_nonfocal ** 2)
        np.add.at(dL_dR_sub, nonfocal_to_sub, correction)

    # Map sub-window dl/dR to global indices
    dl_dR_global = np.zeros(n_global, dtype=np.float64)
    np.add.at(dl_dR_global, sub_global_map, dL_dR_sub)

    return dl_dR_global


# ===========================================================================
# Custom autograd: differentiable diff_omniscape circuit solve
# ===========================================================================

class _DiffOmniscapeSolveFn(torch.autograd.Function):
    """
    Forward:  R(valid pixels) → C(valid pixels) via diff_omniscape
              (moving-window focal-pixel-as-ground, bilinear interpolated)
    Backward: dl/dC → dl/dR via per-focal local adjoint
    """

    @staticmethod
    def forward(ctx, R_valid, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
                radius, block_size, focal_fraction, rng_seed):
        # Diff_omniscape uses many small direct solves — stays on CPU
        R_np = R_valid.detach().cpu().numpy().astype(np.float64)
        n = n_rows * n_cols
        ctx._input_device = R_valid.device

        R_full = np.full(n, 1e6, dtype=np.float64)
        R_full[valid_mask_np] = R_np
        R_matrix = R_full.reshape(n_rows, n_cols)

        do = _get_diff_omniscape_module()

        # Compute all focal pixel positions
        offset = block_size // 2
        focal_rows = list(range(offset, n_rows, block_size))
        focal_cols = list(range(offset, n_cols, block_size))

        # Solve all focal pixels
        all_states = []
        C_map = np.zeros((n_rows, n_cols), dtype=np.float64)

        for fr in focal_rows:
            for fc in focal_cols:
                C_f, fwd = do.solve_local_focal(
                    R_matrix, fr, fc, radius, source_from_resistance
                )
                all_states.append((fr, fc, fwd, C_f))
                C_map[fr, fc] = C_f

        # Stochastic focal sampling: select subset for backward
        n_total = len(all_states)
        frac = min(1.0, max(0.1, float(focal_fraction)))
        if frac < 1.0 and rng_seed is not None:
            rng = np.random.RandomState(rng_seed)
            n_keep = max(1, int(n_total * frac))
            keep_idx = rng.choice(n_total, n_keep, replace=False)
            backward_states = [all_states[i] for i in keep_idx]
            backward_scale = n_total / n_keep
        else:
            backward_states = all_states
            backward_scale = 1.0

        # Bilinear interpolate to full grid
        C_full = _interpolate_block_grid(
            C_map, block_size, focal_rows, focal_cols, n_rows, n_cols
        )
        C_valid_np = C_full.ravel()[valid_mask_np].copy()

        # Save for backward
        ctx.backward_states = backward_states
        ctx.backward_scale = backward_scale
        ctx.focal_rows = focal_rows
        ctx.focal_cols = focal_cols
        ctx.valid_mask_np = valid_mask_np
        ctx.n_rows = n_rows
        ctx.n_cols = n_cols
        ctx.n = n
        ctx.sfr = bool(source_from_resistance)

        return torch.tensor(C_valid_np, dtype=torch.float64, device=ctx._input_device)

    @staticmethod
    def backward(ctx, grad_C):
        vm = ctx.valid_mask_np
        n = ctx.n
        n_rows = ctx.n_rows
        n_cols = ctx.n_cols

        # Expand dl/dC to full grid, then bilinear backward to focal grid
        dl_dc_full = np.zeros(n, dtype=np.float64)
        dl_dc_full[vm] = grad_C.detach().cpu().numpy()
        dl_dc_full_2d = dl_dc_full.reshape(n_rows, n_cols)

        grad_focal = _interpolate_backward_fast(
            dl_dc_full_2d, ctx.focal_rows, ctx.focal_cols, n_rows, n_cols
        )

        # Build focal (row,col) → focal grid index mapping
        fr_arr = np.array(ctx.focal_rows)
        fc_arr = np.array(ctx.focal_cols)
        fr_to_idx = {int(fr_arr[i]): i for i in range(len(fr_arr))}
        fc_to_idx = {int(fc_arr[i]): i for i in range(len(fc_arr))}

        # Accumulate dl/dR across selected focal pixels
        dl_dR = np.zeros(n, dtype=np.float64)
        scale = ctx.backward_scale

        for fr, fc, fwd, C_f in ctx.backward_states:
            fi = fr_to_idx[fr]
            fj = fc_to_idx[fc]
            dl_dC_focal = float(grad_focal[fi, fj]) * scale

            if abs(dl_dC_focal) < 1e-30:
                continue

            dl_dR += _local_adjoint_dl_dR(fwd, dl_dC_focal, n)

        grad_R = torch.tensor(dl_dR[vm], dtype=torch.float64, device=ctx._input_device)

        # Free cached states
        del ctx.backward_states

        # R_valid gets grad; other args are non-tensor → None
        return grad_R, None, None, None, None, None, None, None, None, None


# ===========================================================================
# Neural-network resistance model
# ===========================================================================

class ResistanceNet(nn.Module):
    """
    MLP with linear skip (residual) connection for resistance mapping.

    log R(x) = linear(φ(x)) + MLP(φ(x))

    The linear branch starts as the log-linear model (r_0 + z · φ).
    The MLP starts at zero, so the network begins at the log-linear baseline
    and learns nonlinear corrections (interactions, thresholds, etc.)
    """

    def __init__(self, n_features=4, hidden=32, n_layers=2,
                 R_min=DEFAULT_R_MIN, R_max=DEFAULT_R_MAX, beta=DEFAULT_CLAMP_BETA):
        super().__init__()
        self.R_min = R_min
        self.R_max = R_max
        self.beta = beta

        # Linear skip connection (matches the log-linear model exactly)
        self.skip = nn.Linear(n_features, 1)

        # Nonlinear MLP (learns corrections to the linear model)
        layers = []
        d = n_features
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Start near R ≈ exp(3) ≈ 20 with spatial variation from covariates."""
        # Non-zero skip weights break the flat-surface saddle point.
        # Without this, var(logR)=0 so d(var_penalty)/d(weights)=0 identically.
        nn.init.normal_(self.skip.weight, std=0.1)
        nn.init.constant_(self.skip.bias, 3.0)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.05)
                nn.init.zeros_(m.bias)
        # Last MLP layer small → MLP output ≈ 0 initially
        last = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(last.weight, std=0.01)
        nn.init.zeros_(last.bias)

    def warm_start(self, theta):
        """Initialize skip from existing log-linear params [r_0, z_1..z_4]."""
        theta = np.asarray(theta, dtype=np.float64).ravel()
        with torch.no_grad():
            self.skip.weight.copy_(
                torch.tensor([theta[1:]], dtype=torch.float64)
            )
            self.skip.bias.copy_(
                torch.tensor([theta[0]], dtype=torch.float64)
            )
            # Reset MLP to zero
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        R, _ = self.forward_with_log_R(x)
        return R

    def forward_with_log_R(self, x):
        """Return (R, log_R_raw) where log_R_raw is pre-clamp for regularization."""
        log_R = (self.skip(x) + self.mlp(x)).squeeze(-1)
        # Smooth double-softplus clamp in LOG-space, then exponentiate.
        # This guarantees R ∈ [R_min, R_max] with nonzero gradients
        # everywhere.  The old approach (clamp → exp → softplus in R-space)
        # created gradient dead zones when exp() overshot R_max.
        log_lo = math.log(self.R_min)   # 0
        log_hi = math.log(self.R_max)   # ~8.52
        clamped = log_lo + F.softplus(log_R - log_lo, beta=self.beta)
        clamped = log_hi - F.softplus(log_hi - clamped, beta=self.beta)
        return torch.exp(clamped), log_R


class _ConvResBlock(nn.Module):
    """Conv2d + GroupNorm + SiLU residual block with optional dilation."""

    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(channels, channels, kernel_size,
                              padding=pad, dilation=dilation)
        self.norm = nn.GroupNorm(1, channels)  # num_groups=1 → LayerNorm-like
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.norm(self.conv(x)))


class ConvResistanceNet(nn.Module):
    """
    Convolutional encoder + MLP head with linear skip for resistance mapping.

    log R(x) = skip(φ(x)) + MLP(conv_features(x))

    Improvements over the basic sequential version:
      - GroupNorm (num_groups=1, i.e. LayerNorm) after each conv for stable
        activations at batch_size=1.
      - Residual connections in the conv encoder (skip = identity after the
        projection layer) for better gradient flow.
      - Multi-scale dilated convolutions: dilation increases with depth
        (1, 2, 4, ...) so the receptive field grows exponentially without
        more parameters.
      - Dropout in the MLP head (disabled at eval/final forward).
      - Learned intensity MLP on connectivity (optional, replaces
        the fixed log λ = α + γ log(1+C) parametric form).

    Conv weights are initialised small so the network starts near the
    skip-only (log-linear) baseline.
    """

    def __init__(self, n_features=4, conv_channels=16, n_conv_layers=3,
                 conv_kernel_size=3, hidden=16, n_mlp_layers=1,
                 R_min=DEFAULT_R_MIN, R_max=DEFAULT_R_MAX, beta=DEFAULT_CLAMP_BETA,
                 dropout=0.0, use_dilated=True,
                 intensity_hidden=0):
        super().__init__()
        self.R_min = R_min
        self.R_max = R_max
        self.beta = beta
        self.conv_channels = conv_channels

        # --- Input projection: n_features → conv_channels ---
        pad0 = conv_kernel_size // 2
        self.input_proj = nn.Sequential(
            nn.Conv2d(n_features, conv_channels, conv_kernel_size, padding=pad0),
            nn.GroupNorm(1, conv_channels),
            nn.SiLU(),
        )

        # --- Residual conv blocks with increasing dilation ---
        blocks = []
        for i in range(n_conv_layers):
            dil = (2 ** i) if use_dilated else 1
            blocks.append(_ConvResBlock(conv_channels, conv_kernel_size,
                                        dilation=dil))
        self.conv_encoder = nn.Sequential(*blocks)

        # --- Pointwise MLP head on conv features (with dropout) ---
        mlp_layers = []
        d = conv_channels
        for _ in range(n_mlp_layers):
            mlp_layers += [nn.Linear(d, hidden), nn.SiLU()]
            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))
            d = hidden
        mlp_layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*mlp_layers)

        # --- Linear skip on raw covariates (warm-start compatible) ---
        self.skip = nn.Linear(n_features, 1)

        # --- Optional learned intensity MLP ---
        self.intensity_hidden = intensity_hidden
        if intensity_hidden > 0:
            self.intensity_mlp = nn.Sequential(
                nn.Linear(1, intensity_hidden),
                nn.SiLU(),
                nn.Linear(intensity_hidden, 1),
            )
        else:
            self.intensity_mlp = None

        self._init_weights()

    def _init_weights(self):
        """Small init so conv+MLP ≈ 0 initially → starts at skip baseline."""
        nn.init.normal_(self.skip.weight, std=0.1)
        nn.init.constant_(self.skip.bias, 3.0)

        for m in self.input_proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

        for blk in self.conv_encoder:
            if isinstance(blk, _ConvResBlock):
                nn.init.normal_(blk.conv.weight, std=0.02)
                nn.init.zeros_(blk.conv.bias)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.05)
                nn.init.zeros_(m.bias)
        # Last MLP layer very small → MLP output ≈ 0 initially
        last = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(last.weight, std=0.01)
        nn.init.zeros_(last.bias)

        # Intensity MLP: init so output ≈ log1p(C) (identity passthrough)
        if self.intensity_mlp is not None:
            layers = [m for m in self.intensity_mlp.modules()
                      if isinstance(m, nn.Linear)]
            # First layer: weight≈1, bias≈0 → pass through
            nn.init.constant_(layers[0].weight, 1.0)
            nn.init.zeros_(layers[0].bias)
            # Last layer: weight≈1, bias≈0 → near-identity
            nn.init.constant_(layers[-1].weight, 1.0)
            nn.init.zeros_(layers[-1].bias)

    def warm_start(self, theta):
        """Initialize skip from existing log-linear params [r_0, z_1..z_4]."""
        theta = np.asarray(theta, dtype=np.float64).ravel()
        with torch.no_grad():
            self.skip.weight.copy_(
                torch.tensor(theta[1:].reshape(1, -1), dtype=torch.float64)
            )
            self.skip.bias.copy_(
                torch.tensor([theta[0]], dtype=torch.float64)
            )
            # Reset conv + MLP to near-zero
            for m in self.input_proj.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.zeros_(m.bias)
            for blk in self.conv_encoder:
                if isinstance(blk, _ConvResBlock):
                    nn.init.normal_(blk.conv.weight, std=0.001)
                    nn.init.zeros_(blk.conv.bias)
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward_with_log_R(self, basis_grid, valid_mask_flat, basis_valid):
        """
        Parameters
        ----------
        basis_grid : Tensor (1, n_features, H, W)
            Full 2D covariate grid (NaN cells zero-filled).
        valid_mask_flat : ndarray (H*W,) bool
            Which cells are valid.
        basis_valid : Tensor (n_valid, n_features)
            Raw covariates at valid pixels (for skip connection).

        Returns
        -------
        R : Tensor (n_valid,)
        log_R_raw : Tensor (n_valid,)   (pre-clamp, for regularisation)
        """
        # Input projection + residual conv blocks: (1, C_in, H, W) → (1, C_conv, H, W)
        h = self.input_proj(basis_grid)
        feat_map = self.conv_encoder(h)

        # Extract valid pixels: (1, C_conv, H, W) → (n_valid, C_conv)
        feat_flat = feat_map.squeeze(0).reshape(self.conv_channels, -1).T  # (H*W, C_conv)
        feat_valid = feat_flat[valid_mask_flat]  # (n_valid, C_conv)

        # MLP head on conv features + skip on raw covariates
        mlp_out = self.mlp(feat_valid).squeeze(-1)        # (n_valid,)
        skip_out = self.skip(basis_valid).squeeze(-1)      # (n_valid,)
        log_R = skip_out + mlp_out

        # Smooth double-softplus clamp in log-space
        log_lo = math.log(self.R_min)
        log_hi = math.log(self.R_max)
        clamped = log_lo + F.softplus(log_R - log_lo, beta=self.beta)
        clamped = log_hi - F.softplus(log_hi - clamped, beta=self.beta)
        return torch.exp(clamped), log_R

    def forward(self, basis_grid, valid_mask_flat, basis_valid):
        R, _ = self.forward_with_log_R(basis_grid, valid_mask_flat, basis_valid)
        return R


class IRLResistanceNet(nn.Module):
    """
    Inverse-reinforcement-learning (value-shaped) resistance model.

    A learned reward field over covariates is turned into a resistance surface
    via entropy-regularised soft value iteration on the 4-connected grid MDP:

        r(x)        = skip(phi(x)) + MLP(phi(x))                  # reward (negative cost)
        V_{t+1}(s)  = (1/b) logsumexp_{a in {stay,N,S,E,W}} b [ r(s) + gd * V_t(s_a) ]
        log R(x)    = clamp( offset - scale * V(x) )              # high value -> low resistance

    The soft value iteration is a smooth (logsumexp) contraction for the
    discount ``gamma_d < 1``, so V — and therefore R — is differentiable w.r.t.
    the reward parameters; gradients from the downstream circuit solve flow back
    through the unrolled iteration into the reward network.  Resistance encodes
    the long-range desirability of the landscape (an agent's plan-to-go value)
    rather than purely local habitat.

    The forward signature matches :class:`ConvResistanceNet`: value iteration
    needs the full 2D grid for spatial adjacency.  ``R`` and ``log_R_raw`` are
    returned per valid pixel, so the circuit solver / intensity link / samplers
    consume this model identically to the MLP / conv / spline models.
    """

    def __init__(self, n_features=4, hidden=32, n_layers=2,
                 R_min=DEFAULT_R_MIN, R_max=DEFAULT_R_MAX, beta=DEFAULT_CLAMP_BETA,
                 vi_beta=1.0, gamma_d=0.9, n_value_iter=60,
                 value_scale_init=1.0):
        super().__init__()
        self.R_min = R_min
        self.R_max = R_max
        self.beta = beta                  # log-space clamp sharpness
        self.vi_beta = float(vi_beta)     # soft value-iteration temperature
        self.gamma_d = float(gamma_d)     # discount / leakage (must be < 1)
        self.n_value_iter = int(n_value_iter)

        # Reward field = linear skip (warm-start compatible) + nonlinear MLP.
        self.skip = nn.Linear(n_features, 1)
        layers = []
        d = n_features
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

        # log R = offset - softplus(raw_scale) * V   (scale > 0 keeps the sign:
        # higher value -> lower resistance).
        self.offset = nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
        _raw_scale = math.log(math.expm1(max(float(value_scale_init), 1e-3)))
        self.raw_scale = nn.Parameter(torch.tensor(_raw_scale, dtype=torch.float64))

        # No learned intensity transform (uses the parametric link).
        self.intensity_mlp = None

        self._init_weights()

    def _init_weights(self):
        """Small reward variation so V starts smooth; MLP correction ~ 0."""
        nn.init.normal_(self.skip.weight, std=0.1)
        nn.init.zeros_(self.skip.bias)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.05)
                nn.init.zeros_(m.bias)
        last = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(last.weight, std=0.01)
        nn.init.zeros_(last.bias)

    def warm_start(self, theta):
        """Initialise the reward skip from log-linear params [r_0, z_1..z_K].

        The covariate weights are sign-flipped because reward = -cost: a
        covariate direction that raises resistance lowers reward.  The intercept
        seeds ``offset`` (the resistance level)."""
        theta = np.asarray(theta, dtype=np.float64).ravel()
        with torch.no_grad():
            self.skip.weight.copy_(
                torch.tensor(-theta[1:].reshape(1, -1), dtype=torch.float64)
            )
            self.skip.bias.zero_()
            self.offset.copy_(torch.tensor(theta[0], dtype=torch.float64))
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _shift(t, dr, dc):
        """Neighbour lookup: out[r, c] = t[r + dr, c + dc]; out-of-grid -> 0."""
        H, W = t.shape
        out = torch.zeros_like(t)
        r0s, r1s = max(0, dr), min(H, H + dr)
        c0s, c1s = max(0, dc), min(W, W + dc)
        r0d, r1d = max(0, -dr), min(H, H - dr)
        c0d, c1d = max(0, -dc), min(W, W - dc)
        out[r0d:r1d, c0d:c1d] = t[r0s:r1s, c0s:c1s]
        return out

    def _soft_value_iteration(self, reward_grid, valid_grid):
        """Entropy-regularised soft value iteration on the 4-connected grid.

        ``reward_grid`` and ``valid_grid`` are (H, W); ``valid_grid`` is float
        0/1.  Moves into out-of-grid or invalid cells are masked out of the
        soft maximisation; the always-available 'stay' action guarantees at
        least one finite term.  Invalid cells are pinned to V = 0 so they never
        leak value into valid neighbours."""
        b, gd, NEG = self.vi_beta, self.gamma_d, -1e9
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        nbr_valid = [self._shift(valid_grid, dr, dc) for dr, dc in dirs]
        V = torch.zeros_like(reward_grid)
        for _ in range(self.n_value_iter):
            qs = [reward_grid + gd * V]  # stay action (always valid)
            for (dr, dc), nv in zip(dirs, nbr_valid):
                q = reward_grid + gd * self._shift(V, dr, dc)
                qs.append(torch.where(nv > 0.5, q, torch.full_like(q, NEG)))
            V_new = torch.logsumexp(b * torch.stack(qs, dim=0), dim=0) / b
            V = torch.where(valid_grid > 0.5, V_new, torch.zeros_like(V_new))
        return V

    def forward_with_log_R(self, basis_grid, valid_mask_flat, basis_valid):
        """See :meth:`ConvResistanceNet.forward_with_log_R` for arg shapes."""
        _, n_feat, H, W = basis_grid.shape
        phi_all = basis_grid.squeeze(0).reshape(n_feat, -1).T   # (H*W, n_features)
        reward_grid = (self.skip(phi_all) + self.mlp(phi_all)).squeeze(-1).reshape(H, W)

        valid_grid = torch.as_tensor(
            np.asarray(valid_mask_flat, dtype=np.float64).reshape(H, W),
            dtype=reward_grid.dtype, device=reward_grid.device,
        )
        V = self._soft_value_iteration(reward_grid, valid_grid)
        V_valid = V.reshape(-1)[valid_mask_flat]               # (n_valid,)

        # Centre V: the absolute value level is an arbitrary constant (soft VI
        # accumulates a near-uniform entropy/discount offset).  Only spatial
        # contrasts in V should drive resistance, and centring makes ``offset``
        # the interpretable mean log-resistance (anchored by the mean penalty).
        V_valid = V_valid - V_valid.mean()
        scale = F.softplus(self.raw_scale)
        log_R = self.offset - scale * V_valid

        # Smooth double-softplus clamp in log-space (matches the other nets).
        log_lo = math.log(self.R_min)
        log_hi = math.log(self.R_max)
        clamped = log_lo + F.softplus(log_R - log_lo, beta=self.beta)
        clamped = log_hi - F.softplus(log_hi - clamped, beta=self.beta)
        return torch.exp(clamped), log_R

    def forward(self, basis_grid, valid_mask_flat, basis_valid):
        R, _ = self.forward_with_log_R(basis_grid, valid_mask_flat, basis_valid)
        return R


# ===========================================================================
# B-spline utilities for SplineResistanceNet
# ===========================================================================

def _bspline_knots(n_internal, degree=3, xmin=0.0, xmax=1.0):
    """
    Create B-spline knot vector with boundary knots repeated degree+1 times.

    Returns
    -------
    knots : ndarray (n_internal + 2*degree + 2,)
    """
    internal = np.linspace(xmin, xmax, n_internal + 2)  # includes endpoints
    knots = np.concatenate([
        np.repeat(xmin, degree),
        internal,
        np.repeat(xmax, degree),
    ])
    return knots


def _bspline_basis_matrix(x_np, n_internal=10, degree=3, xmin=0.0, xmax=1.0):
    """
    Evaluate cubic B-spline basis at x values using de Boor recursion.

    Parameters
    ----------
    x_np : ndarray (n,)
        Evaluation points (should be in [xmin, xmax]).
    n_internal : int
        Number of internal knots (total basis functions = n_internal + degree + 1).
    degree : int
        B-spline degree (3 = cubic).

    Returns
    -------
    B : ndarray (n, n_basis)
        Basis matrix.
    knots : ndarray
        Full knot vector.
    """
    knots = _bspline_knots(n_internal, degree, xmin, xmax)
    n_basis = len(knots) - degree - 1
    n = len(x_np)

    # Clamp x to [xmin, xmax] to avoid extrapolation issues.
    # Pull boundary points epsilon inward so the last knot span captures x=xmax
    # (standard B-spline convention: last span is [t_{n-1}, t_n] closed on right
    # but de Boor recursion with repeated boundary knots loses it).
    x = np.clip(x_np, xmin, xmax)
    eps = (xmax - xmin) * 1e-10
    x = np.where(x >= xmax, xmax - eps, x)

    # de Boor recursion
    # Start with degree-0 (piecewise constant) basis
    B = np.zeros((n, n_basis + degree), dtype=np.float64)
    for j in range(n_basis + degree):
        if j < len(knots) - 1:
            if j == len(knots) - 2:
                # Last interval: closed on right
                B[:, j] = ((x >= knots[j]) & (x <= knots[j + 1])).astype(np.float64)
            else:
                B[:, j] = ((x >= knots[j]) & (x < knots[j + 1])).astype(np.float64)

    # Recurse up to target degree
    for d in range(1, degree + 1):
        B_new = np.zeros((n, n_basis + degree - d), dtype=np.float64)
        for j in range(B_new.shape[1]):
            left_denom = knots[j + d] - knots[j]
            right_denom = knots[j + d + 1] - knots[j + 1]
            if left_denom > 0:
                B_new[:, j] += (x - knots[j]) / left_denom * B[:, j]
            if right_denom > 0:
                B_new[:, j] += (knots[j + d + 1] - x) / right_denom * B[:, j + 1]
        B = B_new

    return B[:, :n_basis], knots


def _bspline_basis_matrix_torch(x, knots_t, n_basis, degree=3):
    """
    Differentiable B-spline basis evaluation using de Boor recursion in PyTorch.

    Unlike `_bspline_basis_matrix` (numpy, non-differentiable), this operates
    on a torch tensor *x* and returns a basis matrix through which autograd
    can back-propagate.  Used for the intensity spline where the input
    log(1 + C) changes every optimisation step.

    Parameters
    ----------
    x : Tensor (n,)
        Evaluation points.
    knots_t : Tensor (n_knots_total,)
        Full B-spline knot vector (same convention as `_bspline_knots`).
    n_basis : int
        Number of basis functions.
    degree : int
        B-spline degree (3 = cubic).

    Returns
    -------
    B : Tensor (n, n_basis)
        Basis matrix (differentiable w.r.t. *x*).
    """
    n = x.shape[0]
    xmin = knots_t[0].item()
    xmax = knots_t[-1].item()
    eps = (xmax - xmin) * 1e-10

    # Soft clamp into [xmin, xmax - eps] — differentiable
    x_c = x.clamp(min=xmin, max=xmax - eps)

    n_intervals = n_basis + degree
    # Degree-0 basis (piecewise constant): B[j](x) = 1_{x in [t_j, t_{j+1})}
    # Use detached knots for interval indicators (no grad needed for knots).
    knots_np = knots_t.detach()
    # Build (n, n_intervals) indicator matrix.  These 0/1 masks are not
    # differentiated — all x-dependence enters through the recursion below.
    cols = []
    for j in range(n_intervals):
        if j < len(knots_np) - 1:
            lo = knots_np[j]
            hi = knots_np[j + 1]
            if j == n_intervals - 1:
                # Last span: closed on right
                indicator = ((x_c.detach() >= lo) & (x_c.detach() <= hi)).double()
            else:
                indicator = ((x_c.detach() >= lo) & (x_c.detach() < hi)).double()
        else:
            indicator = torch.zeros(n, dtype=x.dtype, device=x.device)
        cols.append(indicator)
    B = torch.stack(cols, dim=1)  # (n, n_intervals)

    # De Boor recursion: degree 1 … degree
    for d in range(1, degree + 1):
        n_new = n_intervals - d
        new_cols = []
        for j in range(n_new):
            left_denom = (knots_np[j + d] - knots_np[j]).item()
            right_denom = (knots_np[j + d + 1] - knots_np[j + 1]).item()

            val = torch.zeros(n, dtype=x.dtype, device=x.device)
            if left_denom > 0:
                val = val + (x_c - knots_np[j].item()) / left_denom * B[:, j]
            if right_denom > 0:
                val = val + (knots_np[j + d + 1].item() - x_c) / right_denom * B[:, j + 1]
            new_cols.append(val)
        B = torch.stack(new_cols, dim=1)

    return B[:, :n_basis]


def _diff_penalty_matrix(n_basis, order=2):
    """
    Second-order difference penalty matrix S = D2^T @ D2.

    For P-spline smoothing: penalty = β^T S β penalises wiggliness.

    Returns
    -------
    S : ndarray (n_basis, n_basis)
    """
    D = np.eye(n_basis, dtype=np.float64)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return D.T @ D


class SplineResistanceNet(nn.Module):
    """
    Penalised additive model (P-spline GAM) for resistance mapping.

    log R(x) = r_0 + Σ_k f_k(φ_k(x)) + Σ_{j<k} f_{jk}(φ_j(x), φ_k(x))

    Each f_k is a cubic B-spline with second-order difference penalty.
    Each f_{jk} is a tensor product of two B-spline bases with marginal penalties.
    Per-term smoothing parameters λ_k are learnable (log-parameterised).

    This model sits between the rigid log-linear (5 params) and the flexible
    MLP (~1250 params): it has ~1000+ params but strong regularisation via
    smoothing penalties, so it is much less prone to overfitting than the NN
    while still capturing nonlinear covariate effects and interactions.
    """

    def __init__(self, n_features=5, n_knots=10, degree=3,
                 include_interactions=True,
                 R_min=DEFAULT_R_MIN, R_max=DEFAULT_R_MAX, beta=DEFAULT_CLAMP_BETA,
                 lambda_init_marginal=0.0, lambda_init_interaction=2.0,
                 lambda_min=0.0,
                 intensity_spline=False, intensity_n_knots=5,
                 intensity_degree=3, lambda_init_intensity=2.0,
                 intensity_log1p_max=10.0):
        super().__init__()
        self.n_features = n_features
        self.n_knots = n_knots
        self.degree = degree
        self.include_interactions = include_interactions
        self.R_min = R_min
        self.R_max = R_max
        self.beta = beta
        self.lambda_init_marginal = lambda_init_marginal
        self.lambda_init_interaction = lambda_init_interaction
        self.lambda_min = lambda_min

        # B-spline basis info (precomputed in setup_basis_matrices)
        self.n_basis = n_knots + degree + 1  # per covariate
        self._basis_matrices = None  # set by setup_basis_matrices()
        self._penalty_matrices = None

        # Skip connection (log-linear component, for warm start compatibility)
        self.skip = nn.Linear(n_features, 1)

        # Intercept (r_0)
        self.intercept = nn.Parameter(torch.tensor(3.0, dtype=torch.float64))

        # Marginal spline coefficients: one set per covariate
        self.spline_coefs = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n_basis, dtype=torch.float64))
            for _ in range(n_features)
        ])

        # Per-term log smoothing parameters (marginal)
        self.log_lambda_smooth = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            for _ in range(n_features)
        ])

        # Interaction terms (all pairwise)
        self.interaction_pairs = []
        if include_interactions:
            for j in range(n_features):
                for k in range(j + 1, n_features):
                    self.interaction_pairs.append((j, k))

        n_interact = len(self.interaction_pairs)
        n_tp_basis = self.n_basis ** 2

        self.interaction_coefs = nn.ParameterList([
            nn.Parameter(torch.zeros(n_tp_basis, dtype=torch.float64))
            for _ in range(n_interact)
        ])

        # Per-interaction log smoothing (2 per pair: row and column marginal)
        self.log_lambda_interact_row = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            for _ in range(n_interact)
        ])
        self.log_lambda_interact_col = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            for _ in range(n_interact)
        ])

        # --- Intensity spline: f_intensity(log(1+C)) ---
        self.intensity_spline = intensity_spline
        self.intensity_n_knots = intensity_n_knots
        self.intensity_degree = intensity_degree
        self.lambda_init_intensity = lambda_init_intensity
        self.intensity_log1p_max = intensity_log1p_max

        if intensity_spline:
            self.intensity_n_basis = intensity_n_knots + intensity_degree + 1
            # Knot vector on [0, intensity_log1p_max] — registered as buffer
            knots_np = _bspline_knots(intensity_n_knots, intensity_degree,
                                      xmin=0.0, xmax=intensity_log1p_max)
            self.register_buffer(
                "_intensity_knots",
                torch.tensor(knots_np, dtype=torch.float64))
            # Penalty matrix (second-order difference)
            S_np = _diff_penalty_matrix(self.intensity_n_basis, order=2)
            self.register_buffer(
                "_intensity_penalty",
                torch.tensor(S_np, dtype=torch.float64))
            # Learnable spline coefficients
            self.intensity_coefs = nn.Parameter(
                torch.zeros(self.intensity_n_basis, dtype=torch.float64))
            # Learnable log-smoothing parameter
            self.log_lambda_intensity = nn.Parameter(
                torch.tensor(lambda_init_intensity, dtype=torch.float64))
        else:
            self.intensity_n_basis = 0
            self.intensity_coefs = None
            self.log_lambda_intensity = None

        self._init_weights()

    def _init_weights(self):
        """Small init for spline coefficients; skip near log-linear baseline."""
        nn.init.normal_(self.skip.weight, std=0.1)
        nn.init.constant_(self.skip.bias, 0.0)  # intercept handled separately

        for coef in self.spline_coefs:
            nn.init.normal_(coef, std=0.01)
        for coef in self.interaction_coefs:
            nn.init.normal_(coef, std=0.001)

        # Smoothing params: configurable initial values
        for lam in self.log_lambda_smooth:
            nn.init.constant_(lam, self.lambda_init_marginal)
        for lam in self.log_lambda_interact_row:
            nn.init.constant_(lam, self.lambda_init_interaction)
        for lam in self.log_lambda_interact_col:
            nn.init.constant_(lam, self.lambda_init_interaction)

        # Intensity spline: small init so it starts near linear
        if self.intensity_spline:
            nn.init.normal_(self.intensity_coefs, std=0.01)
            nn.init.constant_(self.log_lambda_intensity, self.lambda_init_intensity)

    def setup_basis_matrices(self, basis_values_np, device=None):
        """
        Precompute B-spline basis matrices and penalty matrices.

        Must be called once after construction with the actual covariate values.

        Parameters
        ----------
        basis_values_np : ndarray (n_valid, n_features)
            Covariate values at valid pixels.
        device : torch.device or None
        """
        if device is None:
            device = self.intercept.device

        n_valid, n_feat = basis_values_np.shape
        assert n_feat == self.n_features

        basis_mats = []
        penalty_mats = []

        # Marginal penalty matrix (same for all covariates)
        S_np = _diff_penalty_matrix(self.n_basis, order=2)
        S_t = torch.tensor(S_np, dtype=torch.float64, device=device)

        for k in range(n_feat):
            B_np, _ = _bspline_basis_matrix(
                basis_values_np[:, k],
                n_internal=self.n_knots,
                degree=self.degree,
            )
            B_t = torch.tensor(B_np, dtype=torch.float64, device=device)
            basis_mats.append(B_t)
            penalty_mats.append(S_t)

        # Tensor product basis matrices for interactions
        tp_basis_mats = []
        tp_penalty_row = []
        tp_penalty_col = []
        I_nb = torch.eye(self.n_basis, dtype=torch.float64, device=device)

        for j, k in self.interaction_pairs:
            B_j = basis_mats[j]  # (n_valid, n_basis)
            B_k = basis_mats[k]  # (n_valid, n_basis)
            # Tensor product basis: (n_valid, n_basis^2)
            # B_tp[i, a*n_basis + b] = B_j[i, a] * B_k[i, b]
            n_b = self.n_basis
            B_tp = (B_j.unsqueeze(2) * B_k.unsqueeze(1)).reshape(n_valid, n_b * n_b)
            tp_basis_mats.append(B_tp)

            # Row penalty: S_j ⊗ I  (penalise wiggliness in j-direction)
            S_row = torch.kron(S_t, I_nb)
            # Col penalty: I ⊗ S_k  (penalise wiggliness in k-direction)
            S_col = torch.kron(I_nb, S_t)
            tp_penalty_row.append(S_row)
            tp_penalty_col.append(S_col)

        self._basis_matrices = basis_mats
        self._penalty_matrices = penalty_mats
        self._tp_basis_matrices = tp_basis_mats
        self._tp_penalty_row = tp_penalty_row
        self._tp_penalty_col = tp_penalty_col

    def warm_start(self, theta):
        """
        Initialize from log-linear params [r_0, z_1, ..., z_K].

        Sets the intercept to r_0 and projects each z_k onto the B-spline
        basis (as a linear function: coefs proportional to knot positions).
        """
        theta = np.asarray(theta, dtype=np.float64).ravel()
        with torch.no_grad():
            self.intercept.copy_(torch.tensor(theta[0], dtype=torch.float64))
            # Set skip to the same log-linear model
            self.skip.weight.copy_(
                torch.tensor(theta[1:].reshape(1, -1), dtype=torch.float64)
            )
            self.skip.bias.copy_(torch.tensor([0.0], dtype=torch.float64))

            # Project z_k into spline basis: f_k(φ) ≈ z_k * φ
            # The B-spline basis spans polynomials up to degree d,
            # so we solve min_β ||B β - z_k * x||^2 for each covariate.
            if self._basis_matrices is not None:
                for k in range(min(len(theta) - 1, self.n_features)):
                    z_k = theta[k + 1]
                    B = self._basis_matrices[k]  # (n_valid, n_basis)
                    # Target: z_k * φ_k (the covariate values)
                    # φ_k is embedded in the basis: we want spline ≈ z_k * φ_k
                    # but we only have B, not φ_k directly.
                    # Use skip for the linear part, zero the spline coefs.
                    self.spline_coefs[k].zero_()

            # Zero interaction coefs
            for coef in self.interaction_coefs:
                coef.zero_()

    def forward_with_log_R(self, x):
        """
        Parameters
        ----------
        x : Tensor (n_valid, n_features)
            Covariate values at valid pixels.

        Returns
        -------
        R : Tensor (n_valid,)
        log_R_raw : Tensor (n_valid,)  (pre-clamp, for regularisation)
        """
        if self._basis_matrices is None:
            raise RuntimeError(
                "Call setup_basis_matrices() before forward pass")

        # Skip (linear) + intercept
        log_R = self.skip(x).squeeze(-1) + self.intercept

        # Marginal smooth terms: Σ_k B_k @ β_k
        for k in range(self.n_features):
            B_k = self._basis_matrices[k]
            log_R = log_R + B_k @ self.spline_coefs[k]

        # Tensor product interaction terms: Σ_{j<k} B_tp @ β_{jk}
        for idx, (j, k) in enumerate(self.interaction_pairs):
            B_tp = self._tp_basis_matrices[idx]
            log_R = log_R + B_tp @ self.interaction_coefs[idx]

        # Double-softplus clamp in log-space
        log_lo = math.log(self.R_min)
        log_hi = math.log(self.R_max)
        clamped = log_lo + F.softplus(log_R - log_lo, beta=self.beta)
        clamped = log_hi - F.softplus(log_hi - clamped, beta=self.beta)
        return torch.exp(clamped), log_R

    def forward(self, x):
        R, _ = self.forward_with_log_R(x)
        return R

    def eval_intensity_spline(self, log1p_C):
        """
        Evaluate the intensity P-spline: f_intensity(log(1 + C)).

        The result is centered (pixel-mean subtracted) so that the intercept
        alpha remains identifiable as the overall log-intensity level.

        Parameters
        ----------
        log1p_C : Tensor (n_valid,)
            log(1 + C) at valid pixels.

        Returns
        -------
        f_val : Tensor (n_valid,)
            Centered spline contribution to log-intensity.
        """
        if not self.intensity_spline:
            raise RuntimeError("intensity_spline is not enabled")
        B = _bspline_basis_matrix_torch(
            log1p_C, self._intensity_knots,
            self.intensity_n_basis, self.intensity_degree)
        f_raw = B @ self.intensity_coefs       # (n_valid,)
        return f_raw - f_raw.mean()            # center for identifiability

    def smoothing_penalty(self):
        """
        Compute the total smoothing penalty.

        Returns Σ_k λ_k β_k^T S_k β_k + Σ_{jk} [λ_row β^T S_row β + λ_col β^T S_col β]

        The result is a differentiable scalar (torch autograd tracks gradients
        through β AND log_λ).

        lambda_min provides a floor: λ = lambda_min + exp(log_lambda), preventing
        the optimizer from turning off smoothing entirely.
        """
        pen = torch.tensor(0.0, dtype=torch.float64,
                           device=self.intercept.device)

        # Marginal penalties
        for k in range(self.n_features):
            lam = self.lambda_min + torch.exp(self.log_lambda_smooth[k])
            beta_k = self.spline_coefs[k]
            S_k = self._penalty_matrices[k]
            pen = pen + lam * (beta_k @ S_k @ beta_k)

        # Interaction penalties (row + column marginal)
        for idx in range(len(self.interaction_pairs)):
            beta_tp = self.interaction_coefs[idx]
            lam_r = self.lambda_min + torch.exp(self.log_lambda_interact_row[idx])
            lam_c = self.lambda_min + torch.exp(self.log_lambda_interact_col[idx])
            S_r = self._tp_penalty_row[idx]
            S_c = self._tp_penalty_col[idx]
            pen = pen + lam_r * (beta_tp @ S_r @ beta_tp)
            pen = pen + lam_c * (beta_tp @ S_c @ beta_tp)

        # Intensity spline penalty
        if self.intensity_spline:
            lam_int = self.lambda_min + torch.exp(self.log_lambda_intensity)
            beta_int = self.intensity_coefs
            S_int = self._intensity_penalty
            pen = pen + lam_int * (beta_int @ S_int @ beta_int)

        return pen

    def get_partial_effects(self, n_grid=100, posterior_cov=None):
        """
        Evaluate marginal partial effects f_k(φ) on a regular grid.

        Returns dict mapping covariate index to (grid, effect) arrays.
        If posterior_cov is provided (from compute_posterior_covariance),
        also returns SE and 95% credible intervals per covariate.
        """
        device = self.intercept.device
        results = {}

        for k in range(self.n_features):
            grid_np = np.linspace(0.0, 1.0, n_grid)
            B_np, _ = _bspline_basis_matrix(
                grid_np, n_internal=self.n_knots, degree=self.degree)
            B_t = torch.tensor(B_np, dtype=torch.float64, device=device)
            with torch.no_grad():
                effect = (B_t @ self.spline_coefs[k]).cpu().numpy()
                # Add the skip (linear) contribution for this covariate
                skip_w = self.skip.weight.squeeze()
                effect = effect + skip_w[k].item() * grid_np

            entry = {"grid": grid_np, "effect": effect}

            # Bayesian credible bands if posterior covariance is available
            if posterior_cov is not None and k in posterior_cov:
                V_k = posterior_cov[k]  # (n_basis, n_basis) numpy array
                # Var[f_k(φ)] = B_k(φ)^T V_k B_k(φ)
                # B_np is (n_grid, n_basis)
                BV = B_np @ V_k  # (n_grid, n_basis)
                var_f = np.sum(BV * B_np, axis=1)  # (n_grid,)
                var_f = np.maximum(var_f, 0.0)  # numerical safety
                se = np.sqrt(var_f)
                entry["se"] = se
                entry["lower_95"] = effect - 1.96 * se
                entry["upper_95"] = effect + 1.96 * se

            results[k] = entry

        return results

    def get_interaction_effects(self, n_grid=30):
        """
        Evaluate tensor product interaction effects f_{jk}(φ_j, φ_k) on a grid.

        Returns list of dicts with keys:
            j, k        : covariate indices (0-based)
            grid_j      : 1-D array of φ_j values
            grid_k      : 1-D array of φ_k values
            effect_matrix : (n_grid, n_grid) array of f_{jk} values

        Returns empty list if no interactions were fitted.
        """
        if not self.interaction_pairs:
            return []

        device = self.intercept.device
        grid = np.linspace(0.0, 1.0, n_grid)

        # Build marginal B-spline basis on the grid (same for every covariate)
        B_np, _ = _bspline_basis_matrix(
            grid, n_internal=self.n_knots, degree=self.degree)
        B_t = torch.tensor(B_np, dtype=torch.float64, device=device)
        n_b = self.n_basis

        results = []
        with torch.no_grad():
            for idx, (j, k) in enumerate(self.interaction_pairs):
                # Tensor product basis on the 2-D grid
                # B_j[a] and B_k[b] → B_tp[a, b] = kron(B_j[a,:], B_k[b,:])
                # but we want a (n_grid*n_grid, n_b^2) matrix
                B_j_grid = B_t  # (n_grid, n_b)  – basis for covariate j
                B_k_grid = B_t  # (n_grid, n_b)  – basis for covariate k

                # Expand to (n_grid, n_grid, n_b, n_b) → flatten last two
                # B_tp[i, l, a, b] = B_j[i, a] * B_k[l, b]
                n_g = n_grid
                B_tp = (B_j_grid.unsqueeze(1).unsqueeze(3)
                        * B_k_grid.unsqueeze(0).unsqueeze(2))
                B_tp = B_tp.reshape(n_g * n_g, n_b * n_b)

                effect_flat = (B_tp @ self.interaction_coefs[idx]).cpu().numpy()
                effect_matrix = effect_flat.reshape(n_g, n_g)

                results.append({
                    "j": int(j),
                    "k": int(k),
                    "grid_j": grid.tolist(),
                    "grid_k": grid.tolist(),
                    "effect_matrix": effect_matrix.tolist(),
                })

        return results

    def get_effective_loglinear(self):
        """
        Approximate effective log-linear coefficients for compatibility.

        Returns [r_0_eff, z_1_eff, ..., z_K_eff] where z_k ≈ f_k(1) - f_k(0).
        """
        with torch.no_grad():
            r0 = self.intercept.item() + self.skip.bias.item()
            ell = [r0]
            for k in range(self.n_features):
                # Evaluate spline at 0 and 1
                x0 = np.array([0.0])
                x1 = np.array([1.0])
                B0, _ = _bspline_basis_matrix(
                    x0, n_internal=self.n_knots, degree=self.degree)
                B1, _ = _bspline_basis_matrix(
                    x1, n_internal=self.n_knots, degree=self.degree)
                B0_t = torch.tensor(B0, dtype=torch.float64,
                                    device=self.intercept.device)
                B1_t = torch.tensor(B1, dtype=torch.float64,
                                    device=self.intercept.device)
                f0 = (B0_t @ self.spline_coefs[k]).item()
                f1 = (B1_t @ self.spline_coefs[k]).item()
                # Add skip linear component
                skip_w = self.skip.weight.squeeze()[k].item()
                z_k = (f1 + skip_w) - f0
                ell.append(z_k)
            return ell

    # ==================================================================
    # Bayesian UQ: Laplace approximation on spline coefficients
    # ==================================================================

    def _collect_spline_params(self):
        """Return (params_list, slices) for marginal spline coefs only."""
        params = []
        slices = {}
        offset = 0
        for k in range(self.n_features):
            n_b = self.spline_coefs[k].numel()
            params.append(self.spline_coefs[k])
            slices[k] = slice(offset, offset + n_b)
            offset += n_b
        return params, slices, offset

    def compute_nll_hessian_block(self, nll_fn, block_only=True):
        """
        Compute the Hessian of the negative log-likelihood w.r.t. marginal
        spline coefficients using pure finite differences (no autograd).

        This uses central differences on the NLL *value* to build the Hessian:
            H[i,j] ≈ (f(θ+εe_i+εe_j) - f(θ+εe_i-εe_j) - f(θ-εe_i+εe_j) + f(θ-εe_i-εe_j)) / (4ε²)

        For diagonal and near-diagonal elements, uses the more efficient:
            H[i,i] ≈ (f(θ+εe_i) - 2f(θ) + f(θ-εe_i)) / ε²
            H[i,j] ≈ (f(θ+ε(e_i+e_j)) - f(θ+εe_i) - f(θ+εe_j) + f(θ)) / ε²

        Parameters
        ----------
        nll_fn : callable
            A function that takes no arguments, recomputes the forward pass
            (including circuit solve) and returns the negative log-likelihood
            as a Python float. Must not require gradients.
        block_only : bool
            If True, compute per-covariate blocks only (fast: K × n_basis
            function evaluations). If False, compute the full Hessian over
            all marginal spline coefs.

        Returns
        -------
        dict with:
            'blocks' : list of (n_basis, n_basis) numpy arrays if block_only
            'full'   : (n_total, n_total) numpy array if not block_only
            'slices' : dict mapping covariate index to slice into flat vector
        """
        params, slices, n_total = self._collect_spline_params()
        eps = 1e-4

        # Baseline NLL
        f0 = nll_fn()

        if block_only:
            blocks = []
            for k in range(self.n_features):
                beta_k = self.spline_coefs[k]
                n_b = beta_k.numel()
                H_k = np.zeros((n_b, n_b), dtype=np.float64)

                # First compute f(θ+εe_i) for all i (reused in off-diagonal)
                f_plus = np.zeros(n_b, dtype=np.float64)
                f_minus = np.zeros(n_b, dtype=np.float64)
                for i in range(n_b):
                    beta_k.data[i] += eps
                    f_plus[i] = nll_fn()
                    beta_k.data[i] -= 2.0 * eps
                    f_minus[i] = nll_fn()
                    beta_k.data[i] += eps  # restore

                # Diagonal: H[i,i] = (f+ - 2*f0 + f-) / eps^2
                for i in range(n_b):
                    H_k[i, i] = (f_plus[i] - 2.0 * f0 + f_minus[i]) / (eps * eps)

                # Off-diagonal: H[i,j] = (f(θ+ε(ei+ej)) - f+_i - f+_j + f0) / eps^2
                for i in range(n_b):
                    for j in range(i + 1, n_b):
                        beta_k.data[i] += eps
                        beta_k.data[j] += eps
                        f_ij = nll_fn()
                        beta_k.data[i] -= eps
                        beta_k.data[j] -= eps  # restore

                        H_k[i, j] = (f_ij - f_plus[i] - f_plus[j] + f0) / (eps * eps)
                        H_k[j, i] = H_k[i, j]

                blocks.append(H_k)

            return {"blocks": blocks, "slices": slices}

        else:
            # Full Hessian over all marginal spline coefficients
            H = np.zeros((n_total, n_total), dtype=np.float64)

            # Compute all single-perturbation values
            f_plus_all = np.zeros(n_total, dtype=np.float64)
            f_minus_all = np.zeros(n_total, dtype=np.float64)

            for k in range(self.n_features):
                beta_k = self.spline_coefs[k]
                s = slices[k]
                n_b = beta_k.numel()
                for i in range(n_b):
                    beta_k.data[i] += eps
                    f_plus_all[s.start + i] = nll_fn()
                    beta_k.data[i] -= 2.0 * eps
                    f_minus_all[s.start + i] = nll_fn()
                    beta_k.data[i] += eps  # restore

            # Diagonal
            for idx in range(n_total):
                H[idx, idx] = (f_plus_all[idx] - 2.0 * f0 + f_minus_all[idx]) / (eps * eps)

            # Off-diagonal (only within same covariate for block structure,
            # and cross-covariate if full)
            for k1 in range(self.n_features):
                s1 = slices[k1]
                beta_k1 = self.spline_coefs[k1]
                for k2 in range(k1, self.n_features):
                    s2 = slices[k2]
                    beta_k2 = self.spline_coefs[k2]
                    for i in range(s1.stop - s1.start):
                        j_start = (i + 1) if k1 == k2 else 0
                        for j in range(j_start, s2.stop - s2.start):
                            gi = s1.start + i
                            gj = s2.start + j
                            beta_k1.data[i] += eps
                            beta_k2.data[j] += eps
                            f_ij = nll_fn()
                            beta_k1.data[i] -= eps
                            beta_k2.data[j] -= eps

                            H[gi, gj] = (f_ij - f_plus_all[gi] - f_plus_all[gj] + f0) / (eps * eps)
                            H[gj, gi] = H[gi, gj]

            return {"full": H, "slices": slices}

    def compute_posterior_covariance(self, hessian_result, penalty_scale=1.0):
        """
        Compute posterior covariance V = (H_nll + penalty_scale × Σ_k λ_k S_k)^{-1}

        Uses the Bayesian P-spline interpretation where the smoothing penalty
        acts as an improper Gaussian prior.

        Parameters
        ----------
        hessian_result : dict
            Output from compute_nll_hessian_block.
        penalty_scale : float
            The penalty_scale used during optimization.

        Returns
        -------
        dict mapping covariate index k to (n_basis, n_basis) posterior covariance.
        Also includes 'edf' (effective degrees of freedom per covariate) and
        'significance' (approximate p-values).
        """
        posterior_cov = {}
        edf_dict = {}
        significance = {}

        if "blocks" in hessian_result:
            # Block-diagonal approximation
            for k in range(self.n_features):
                H_k = hessian_result["blocks"][k]
                lam_k = (self.lambda_min
                         + torch.exp(self.log_lambda_smooth[k]).item())
                S_k = self._penalty_matrices[k].cpu().numpy()
                P_k = penalty_scale * lam_k * S_k

                # Posterior precision = H_nll + penalty
                precision_k = H_k + P_k

                # Regularize for numerical stability
                eigvals = np.linalg.eigvalsh(precision_k)
                min_eig = eigvals.min()
                if min_eig < 1e-8:
                    precision_k += (1e-8 - min_eig) * np.eye(precision_k.shape[0])

                V_k = np.linalg.inv(precision_k)
                posterior_cov[k] = V_k

                # EDF = trace(V_k @ H_k) — how many params each smooth uses
                edf_k = np.trace(V_k @ H_k)
                edf_dict[k] = float(edf_k)

                # Significance test: β_k^T P_r β_k / rank(S_k) ~ chi2
                # where P_r = pseudoinverse of V_k restricted to range(S_k)
                beta_k = self.spline_coefs[k].detach().cpu().numpy()
                # Rank-based test (Wood 2013): use edf as approximate rank
                edf_r = max(edf_k, 1.0)
                test_stat = float(beta_k @ np.linalg.pinv(V_k) @ beta_k)
                from scipy import stats as sp_stats
                p_val = float(1.0 - sp_stats.chi2.cdf(test_stat, df=edf_r))
                significance[k] = {
                    "chi_sq": test_stat,
                    "edf": edf_k,
                    "p_value": p_val,
                }

        elif "full" in hessian_result:
            H = hessian_result["full"]
            slices = hessian_result["slices"]
            n_total = H.shape[0]

            # Build full penalty matrix
            P = np.zeros((n_total, n_total), dtype=np.float64)
            for k in range(self.n_features):
                s = slices[k]
                lam_k = (self.lambda_min
                         + torch.exp(self.log_lambda_smooth[k]).item())
                S_k = self._penalty_matrices[k].cpu().numpy()
                P[s, s] += penalty_scale * lam_k * S_k

            precision = H + P
            eigvals = np.linalg.eigvalsh(precision)
            min_eig = eigvals.min()
            if min_eig < 1e-8:
                precision += (1e-8 - min_eig) * np.eye(n_total)

            V_full = np.linalg.inv(precision)

            for k in range(self.n_features):
                s = slices[k]
                V_k = V_full[s, s]
                H_k = H[s, s]
                posterior_cov[k] = V_k
                edf_k = np.trace(V_k @ H_k)
                edf_dict[k] = float(edf_k)

                beta_k = self.spline_coefs[k].detach().cpu().numpy()
                edf_r = max(edf_k, 1.0)
                test_stat = float(beta_k @ np.linalg.pinv(V_k) @ beta_k)
                from scipy import stats as sp_stats
                p_val = float(1.0 - sp_stats.chi2.cdf(test_stat, df=edf_r))
                significance[k] = {
                    "chi_sq": test_stat,
                    "edf": edf_k,
                    "p_value": p_val,
                }

        return {
            "covariance": posterior_cov,
            "edf": edf_dict,
            "significance": significance,
        }


# ===========================================================================
# PPP log-likelihood
# ===========================================================================

def _ppp_loglik(log_lam, obs_counts, cell_area):
    """
    Poisson point process log-likelihood on raster cells.

    log L = Σ_k n_k log λ_k  −  A Σ_k λ_k

    where n_k = # GPS fixes in cell k, A = cell area (m²).
    """
    term1 = (obs_counts * log_lam).sum()
    term2 = cell_area * torch.exp(log_lam.clamp(max=20.0)).sum()
    return term1 - term2


# ===========================================================================
# Gradient verification
# ===========================================================================

def verify_circuit_gradient(basis_values_np, valid_mask_np,
                            n_rows, n_cols, source_spacing=5,
                            source_from_resistance=True,
                            eps=1e-4, seed=42,
                            solver="diff_omniscape",
                            radius=15, block_size=10,
                            absorption=0.01):
    """
    Finite-difference check for the custom autograd circuit solve.
    Uses a random subset of pixels for speed.

    Returns dict with max_rel_error and pass/fail.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_valid = valid_mask_np.sum()
    R_val = np.full(n_valid, 20.0, dtype=np.float64)
    # Add spatial variation
    R_val += np.random.randn(n_valid) * 5.0
    R_val = np.clip(R_val, 1.0, 5000.0)

    R_t = torch.tensor(R_val, dtype=torch.float64, requires_grad=True)

    use_diff_omniscape = (solver == "diff_omniscape")
    use_absorption = (solver == "global_absorption")

    def _apply_circuit(R_tensor):
        if use_diff_omniscape:
            return _DiffOmniscapeSolveFn.apply(
                R_tensor, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
                radius, block_size, 1.0, seed,  # focal_fraction=1.0 for exact grad check
            )
        elif use_absorption:
            return _AbsorptionCircuitSolveFn.apply(
                R_tensor, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
                absorption,
            )
        else:
            return _CircuitSolveFn.apply(
                R_tensor, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
            )

    # Forward
    C = _apply_circuit(R_t)

    # Random loss direction
    dl_dC = torch.randn_like(C, dtype=torch.float64)
    loss = (C * dl_dC).sum()
    loss.backward()
    analytic_grad = R_t.grad.numpy().copy()

    # Finite differences on a random subset
    n_check = min(20, n_valid)
    check_idx = np.random.choice(n_valid, n_check, replace=False)
    fd_grad = np.zeros(n_check)

    for i, idx in enumerate(check_idx):
        R_plus = R_val.copy()
        R_plus[idx] += eps
        C_plus = _apply_circuit(
            torch.tensor(R_plus, dtype=torch.float64),
        )
        loss_plus = (C_plus * dl_dC).sum().item()

        R_minus = R_val.copy()
        R_minus[idx] -= eps
        C_minus = _apply_circuit(
            torch.tensor(R_minus, dtype=torch.float64),
        )
        loss_minus = (C_minus * dl_dC).sum().item()

        fd_grad[i] = (loss_plus - loss_minus) / (2 * eps)

    analytic_at_check = analytic_grad[check_idx]
    rel_errors = np.abs(fd_grad - analytic_at_check) / (np.maximum(np.abs(fd_grad), np.abs(analytic_at_check)) + 1e-7)
    max_rel = float(np.max(rel_errors))

    print(f"  Gradient check: max rel error = {max_rel:.2e} "
          f"(checked {n_check} pixels)")

    return {
        "max_rel_error": max_rel,
        "pass": max_rel < 0.05,
        "rel_errors": rel_errors.tolist(),
    }


def verify_conv_gradient(basis_values_np, valid_mask_np,
                         n_rows, n_cols, source_spacing=1,
                         source_from_resistance=True,
                         conv_channels=16, n_conv_layers=3,
                         conv_kernel_size=3, hidden_dim=16,
                         n_mlp_layers=1,
                         eps=1e-4, seed=42,
                         solver="global_absorption",
                         radius=15, block_size=10,
                         absorption=0.01,
                         dropout=0.0, use_dilated=True,
                         intensity_hidden=0):
    """
    End-to-end finite-difference gradient check for ConvResistanceNet.

    Tests the full chain: conv(basis_grid) → R → circuit → C → scalar loss.
    Perturbs individual conv/MLP/skip parameters and compares FD vs autograd.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_valid = int(valid_mask_np.sum())
    n_features = basis_values_np.shape[1]

    # Build 2D grid
    n_cells = n_rows * n_cols
    grid_np = np.zeros((n_features, n_cells), dtype=np.float64)
    grid_np[:, valid_mask_np] = basis_values_np.T
    grid_np = grid_np.reshape(1, n_features, n_rows, n_cols)
    basis_grid_t = torch.tensor(grid_np, dtype=torch.float64)
    basis_valid_t = torch.tensor(basis_values_np, dtype=torch.float64)

    net = ConvResistanceNet(
        n_features, conv_channels=conv_channels,
        n_conv_layers=n_conv_layers,
        conv_kernel_size=conv_kernel_size,
        hidden=hidden_dim, n_mlp_layers=n_mlp_layers,
        dropout=0.0,  # No dropout for gradient check (deterministic)
        use_dilated=use_dilated,
        intensity_hidden=intensity_hidden,
    ).double()

    use_absorption = (solver == "global_absorption")
    use_diff_omniscape = (solver == "diff_omniscape")

    def _forward_loss(net_):
        R = net_(basis_grid_t, valid_mask_np, basis_valid_t)
        if use_diff_omniscape:
            C = _DiffOmniscapeSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
                radius, block_size, 1.0, seed,
            )
        elif use_absorption:
            C = _AbsorptionCircuitSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
                absorption,
            )
        else:
            C = _CircuitSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
            )
        return (C * dl_dC).sum()

    # Random loss direction (fixed across perturbations)
    with torch.no_grad():
        R_tmp = net(basis_grid_t, valid_mask_np, basis_valid_t)
    dl_dC = torch.randn(n_valid, dtype=torch.float64)

    # Analytic gradient
    loss = _forward_loss(net)
    loss.backward()

    # Collect a subset of parameters to check — save analytic grads upfront
    # (net.zero_grad() in the FD loop would clear them otherwise)
    all_params = [(n, p, p.grad.detach().cpu().numpy().ravel().copy())
                  for n, p in net.named_parameters()
                  if p.requires_grad and p.grad is not None]
    n_check_per_param = 3
    fd_results = []

    for pname, param, analytic_g in all_params:
        n_elem = len(analytic_g)
        check_idx = np.random.choice(n_elem, min(n_check_per_param, n_elem), replace=False)

        for idx in check_idx:
            orig = param.data.view(-1)[idx].item()

            param.data.view(-1)[idx] = orig + eps
            net.zero_grad()
            lp = _forward_loss(net).item()

            param.data.view(-1)[idx] = orig - eps
            net.zero_grad()
            lm = _forward_loss(net).item()

            param.data.view(-1)[idx] = orig

            fd = (lp - lm) / (2 * eps)
            an = analytic_g[idx]
            rel_err = abs(fd - an) / (max(abs(fd), abs(an)) + 1e-7)
            fd_results.append((pname, idx, fd, an, rel_err))

    rel_errors = [r[4] for r in fd_results]
    max_rel = float(max(rel_errors))

    print(f"  Conv gradient check: max rel error = {max_rel:.2e} "
          f"(checked {len(fd_results)} param elements across {len(all_params)} params)")
    for pname, idx, fd, an, re in fd_results:
        status = "OK" if re < 0.05 else "FAIL"
        print(f"    {pname}[{idx}]: FD={fd:.6e}, AN={an:.6e}, rel={re:.2e} {status}")

    return {
        "max_rel_error": max_rel,
        "pass": max_rel < 0.05,
        "rel_errors": [float(r) for r in rel_errors],
    }


def verify_softrl_gradient(basis_values_np, valid_mask_np,
                           n_rows, n_cols, source_spacing=1,
                           source_from_resistance=True,
                           hidden_dim=16, n_hidden_layers=2,
                           beta=1.0, gamma_d=0.9, n_value_iter=30,
                           value_scale_init=1.0,
                           eps=1e-4, seed=42,
                           solver="global_absorption",
                           radius=15, block_size=10,
                           absorption=0.01):
    """
    End-to-end finite-difference gradient check for IRLResistanceNet.

    Tests the full chain: reward(basis) → soft value iteration → R → circuit
    → C → scalar loss.  Perturbs individual reward/offset/scale parameters and
    compares finite-difference vs autograd gradients through the unrolled value
    iteration.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_valid = int(valid_mask_np.sum())
    n_features = basis_values_np.shape[1]

    # Build 2D grid (same layout as the conv/IRL training path)
    n_cells = n_rows * n_cols
    grid_np = np.zeros((n_features, n_cells), dtype=np.float64)
    grid_np[:, valid_mask_np] = basis_values_np.T
    grid_np = grid_np.reshape(1, n_features, n_rows, n_cols)
    basis_grid_t = torch.tensor(grid_np, dtype=torch.float64)
    basis_valid_t = torch.tensor(basis_values_np, dtype=torch.float64)

    net = IRLResistanceNet(
        n_features, hidden=hidden_dim, n_layers=n_hidden_layers,
        vi_beta=float(beta), gamma_d=float(gamma_d),
        n_value_iter=int(n_value_iter),
        value_scale_init=float(value_scale_init),
    ).double()

    use_absorption = (solver == "global_absorption")
    use_diff_omniscape = (solver == "diff_omniscape")

    def _forward_loss(net_):
        R = net_(basis_grid_t, valid_mask_np, basis_valid_t)
        if use_diff_omniscape:
            C = _DiffOmniscapeSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
                radius, block_size, 1.0, seed,
            )
        elif use_absorption:
            C = _AbsorptionCircuitSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
                absorption,
            )
        else:
            C = _CircuitSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
            )
        return (C * dl_dC).sum()

    # Random loss direction (fixed across perturbations)
    dl_dC = torch.randn(n_valid, dtype=torch.float64)

    # Analytic gradient
    loss = _forward_loss(net)
    loss.backward()

    all_params = [(n, p, p.grad.detach().cpu().numpy().ravel().copy())
                  for n, p in net.named_parameters()
                  if p.requires_grad and p.grad is not None]
    n_check_per_param = 3
    fd_results = []

    for pname, param, analytic_g in all_params:
        n_elem = len(analytic_g)
        check_idx = np.random.choice(n_elem, min(n_check_per_param, n_elem),
                                     replace=False)
        for idx in check_idx:
            orig = param.data.view(-1)[idx].item()

            param.data.view(-1)[idx] = orig + eps
            net.zero_grad()
            lp = _forward_loss(net).item()

            param.data.view(-1)[idx] = orig - eps
            net.zero_grad()
            lm = _forward_loss(net).item()

            param.data.view(-1)[idx] = orig

            fd = (lp - lm) / (2 * eps)
            an = analytic_g[idx]
            rel_err = abs(fd - an) / (max(abs(fd), abs(an)) + 1e-7)
            fd_results.append((pname, idx, fd, an, rel_err))

    rel_errors = [r[4] for r in fd_results]
    max_rel = float(max(rel_errors))

    print(f"  IRL gradient check: max rel error = {max_rel:.2e} "
          f"(checked {len(fd_results)} param elements across {len(all_params)} params)")
    for pname, idx, fd, an, re in fd_results:
        status = "OK" if re < 0.05 else "FAIL"
        print(f"    {pname}[{idx}]: FD={fd:.6e}, AN={an:.6e}, rel={re:.2e} {status}")

    return {
        "max_rel_error": max_rel,
        "pass": max_rel < 0.05,
        "rel_errors": [float(r) for r in rel_errors],
    }


def verify_spline_gradient(basis_values_np, valid_mask_np,
                           n_rows, n_cols, source_spacing=1,
                           source_from_resistance=True,
                           n_knots=10, degree=3,
                           include_interactions=True,
                           eps=1e-4, seed=42,
                           solver="global_absorption",
                           radius=15, block_size=10,
                           absorption=0.01,
                           intensity_spline=False,
                           intensity_n_knots=5,
                           intensity_degree=3,
                           intensity_log1p_max=10.0):
    """
    End-to-end finite-difference gradient check for SplineResistanceNet.

    Tests the full chain: B-spline(basis) → R → circuit → C → scalar loss.
    Perturbs individual spline parameters and compares FD vs autograd.
    When intensity_spline=True, also tests the intensity spline path:
    C → log(1+C) → B-spline → f_intensity → loss.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_valid = int(valid_mask_np.sum())
    n_features = basis_values_np.shape[1]

    basis_t = torch.tensor(basis_values_np, dtype=torch.float64)

    net = SplineResistanceNet(
        n_features, n_knots=n_knots, degree=degree,
        include_interactions=include_interactions,
        intensity_spline=bool(intensity_spline),
        intensity_n_knots=int(intensity_n_knots),
        intensity_degree=int(intensity_degree),
        intensity_log1p_max=float(intensity_log1p_max),
    ).double()
    net.setup_basis_matrices(basis_values_np)

    use_absorption = (solver == "global_absorption")
    use_diff_omniscape = (solver == "diff_omniscape")

    # Random loss direction (fixed across perturbations)
    dl_dC = torch.randn(n_valid, dtype=torch.float64)

    def _forward_loss(net_):
        R, _ = net_.forward_with_log_R(basis_t)
        if use_diff_omniscape:
            C = _DiffOmniscapeSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
                radius, block_size, 1.0, seed,
            )
        elif use_absorption:
            C = _AbsorptionCircuitSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
                absorption,
            )
        else:
            C = _CircuitSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
            )
        if net_.intensity_spline:
            # Test the full intensity spline path
            log1p_C = torch.log1p(C.clamp(min=0))
            f_int = net_.eval_intensity_spline(log1p_C)
            return (f_int * dl_dC).sum()
        else:
            return (C * dl_dC).sum()

    # Analytic gradient
    loss = _forward_loss(net)
    loss.backward()

    # Collect subset of parameters to check
    all_params = [(n, p, p.grad.detach().cpu().numpy().ravel().copy())
                  for n, p in net.named_parameters()
                  if p.requires_grad and p.grad is not None]
    n_check_per_param = 3
    fd_results = []

    for pname, param, analytic_g in all_params:
        n_elem = len(analytic_g)
        check_idx = np.random.choice(n_elem, min(n_check_per_param, n_elem),
                                     replace=False)

        for idx in check_idx:
            orig = param.data.view(-1)[idx].item()

            param.data.view(-1)[idx] = orig + eps
            net.zero_grad()
            lp = _forward_loss(net).item()

            param.data.view(-1)[idx] = orig - eps
            net.zero_grad()
            lm = _forward_loss(net).item()

            param.data.view(-1)[idx] = orig

            fd = (lp - lm) / (2 * eps)
            an = analytic_g[idx]
            rel_err = abs(fd - an) / (max(abs(fd), abs(an)) + 1e-7)
            fd_results.append((pname, idx, fd, an, rel_err))

    rel_errors = [r[4] for r in fd_results]
    max_rel = float(max(rel_errors))

    print(f"  Spline gradient check: max rel error = {max_rel:.2e} "
          f"(checked {len(fd_results)} param elements across "
          f"{len(all_params)} params)"
          f"{' [intensity spline ON]' if intensity_spline else ''}")
    for pname, idx, fd, an, re in fd_results:
        status = "OK" if re < 0.05 else "FAIL"
        print(f"    {pname}[{idx}]: FD={fd:.6e}, AN={an:.6e}, "
              f"rel={re:.2e} {status}")

    return {
        "max_rel_error": max_rel,
        "pass": max_rel < 0.05,
        "rel_errors": [float(r) for r in rel_errors],
    }


# ===========================================================================
# Main training loop
# ===========================================================================

def run_torch_optimization(
    basis_values_np,             # (n_valid, n_feat) float64
    obs_counts_np,               # (n_valid,) float64  — GPS count per valid cell
    n_rows,                      # int
    n_cols,                      # int
    valid_mask_np,               # (n_cells,) bool
    cell_area,                   # float (m²)
    source_spacing=5,
    source_from_resistance=True,
    hidden_dim=32,
    n_hidden_layers=2,
    lr=0.01,
    weight_decay=1e-4,
    n_epochs=300,
    patience=30,
    grad_clip=10.0,
    cg_tol=DEFAULT_CG_TOL,
    warm_start_theta=None,       # [r_0, z_1..z_4] from log-linear optimization
    reg_mean=1.0,                # Penalty weight: anchor mean(log_R) near baseline
    reg_var=0.1,                 # Penalty weight: encourage variance in log_R
    target_logR_var=1.0,         # Target var(log_R); penalty activates below this
    reg_skip=0.1,                # Penalty weight: prevent skip weights collapsing to 0
    log_R_baseline=3.0,          # Target mean(log_R); 3.0 → R ≈ 20
    output_dir=None,
    seed=42,
    verbose=True,
    # Solver selection
    solver="diff_omniscape",     # "diff_omniscape", "global", or "global_absorption"
    radius=15,                   # diff_omniscape: focal neighbourhood half-width
    block_size=10,               # diff_omniscape: spacing between focal pixels
    focal_fraction=0.5,          # diff_omniscape: fraction of focals used per epoch
    absorption=0.01,             # global_absorption: leakage rate (higher = more local)
    # Device selection
    device="auto",               # "auto", "cuda", or "cpu"
    # Convolutional encoder
    use_conv=False,              # Enable ConvResistanceNet (spatial context)
    n_conv_layers=3,             # Number of Conv2d layers
    conv_channels=16,            # Conv feature channels
    conv_kernel_size=3,          # Conv kernel size
    # New architecture options
    dropout=0.0,                 # Dropout rate in MLP head (0 = off)
    use_dilated=True,            # Dilated convolutions in conv encoder
    intensity_hidden=0,          # Learned intensity MLP width (0 = parametric)
    warmup_epochs=10,            # LR linear warm-up epochs (0 = off)
    absorption_schedule=None,    # Absorption curriculum: [start, end] or None
    # Model type
    model_type="mlp",            # "mlp", "conv", or "spline_gam"
    n_knots=10,                  # SplineResistanceNet: internal knots per covariate
    spline_degree=3,             # SplineResistanceNet: B-spline degree
    include_interactions=True,   # SplineResistanceNet: pairwise tensor products
    penalty_scale=1.0,           # Global multiplier on smoothing penalty (>1 = smoother)
    lambda_init_marginal=0.0,    # Initial log-lambda for marginal splines
    lambda_init_interaction=2.0, # Initial log-lambda for interaction terms
    lambda_min=0.0,              # Floor on lambda: λ = lambda_min + exp(log_lambda)
    compute_uq=False,            # Compute Bayesian credible bands post-optimization
    uq_block_only=True,          # True = block-diagonal Hessian (fast)
    # Intensity spline (spline_gam only)
    intensity_spline=False,      # Replace parametric γ·log(1+C) with P-spline f(log(1+C))
    intensity_n_knots=5,         # Internal knots for intensity spline
    intensity_degree=3,          # B-spline degree for intensity spline
    lambda_init_intensity=2.0,   # Initial log-lambda for intensity smoothing
    intensity_log1p_max=10.0,    # Upper knot boundary for log(1+C)
    # IRL (value-shaped) resistance — model_type="irl"
    beta=1.0,                    # Soft value-iteration temperature
    gamma_d=0.9,                 # Discount / leakage of the MDP (must be < 1)
    n_value_iter=60,             # Soft value-iteration steps (unrolled)
    value_scale_init=1.0,        # Initial scale in log R = offset - scale * V
):
    """
    End-to-end neural-network PPP-circuit optimization.

    Parameters
    ----------
    basis_values_np : ndarray (n_valid, n_features)
        Covariate values at valid pixels (e.g. canopy, impervious, water, fence, elevation).
    obs_counts_np : ndarray (n_valid,)
        Number of GPS observations per valid cell (0 for most cells).
    n_rows, n_cols : int
        Raster grid dimensions.
    valid_mask_np : ndarray (n_cells,) bool
        Which cells are valid (not NA).
    cell_area : float
        Cell area in m² (e.g. 900 for 30m grid).
    source_spacing : int
        Source lattice spacing for circuit solver.
    source_from_resistance : bool
        Use 1/R source injection (matches Omniscape).
    hidden_dim : int
        MLP hidden layer width.
    n_hidden_layers : int
        Number of MLP hidden layers.
    lr : float
        Initial Adam learning rate.
    weight_decay : float
        L2 regularization strength.
    n_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience.
    grad_clip : float
        Gradient norm clipping.
    cg_tol : float
        CG solver tolerance (1e-6 is fast; 1e-10 is precise).
    warm_start_theta : array or None
        Log-linear parameters [r_0, z_1, ..., z_K] to warm-start from.
    output_dir : str or None
        Directory to save model weights.
    seed : int
        Random seed.
    verbose : bool

    Returns
    -------
    dict with numpy arrays and Python scalars (reticulate-safe):
        resistance, connectivity, log_lambda, alpha, gamma,
        loglik, loss_history, best_epoch, n_params, total_time
    """
    t0_all = time.time()

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n_valid = basis_values_np.shape[0]
    n_features = basis_values_np.shape[1]
    valid_mask_np = np.asarray(valid_mask_np, dtype=bool)

    use_diff_omniscape = (solver == "diff_omniscape")
    use_absorption = (solver == "global_absorption")

    # Absorption solver: force source_spacing=1 (sparse sources create grid
    # artifacts with no speed benefit for a single global solve).
    if use_absorption and source_spacing > 1:
        if verbose:
            print(f"  NOTE: source_spacing={source_spacing} overridden to 1 "
                  f"for absorption solver (avoids grid artifacts)")
        source_spacing = 1

    # ---- Device selection ----
    if device == "auto":
        use_cuda = _GPU_AVAILABLE and (use_diff_omniscape or _CUPY_AVAILABLE)
        dev = torch.device("cuda" if use_cuda else "cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda is not available")
        if not _CUPY_AVAILABLE:
            raise RuntimeError("CUDA requested but cupy is not installed")
        dev = torch.device("cuda")
        use_cuda = True
    else:
        dev = torch.device("cpu")
        use_cuda = False

    # Enable/disable GPU in the circuit solver
    cs = _get_circuit_module()
    if use_cuda and not use_diff_omniscape:
        cs.enable_gpu(True)
        if seed is not None:
            cp.random.seed(seed)
    else:
        cs.enable_gpu(False)

    # ---- Resolve model_type from use_conv flag (backward compat) ----
    if model_type == "mlp" and use_conv:
        model_type = "conv"
    use_conv = (model_type == "conv")
    use_spline = (model_type == "spline_gam")
    use_irl = (model_type == "irl")
    # Grid-context nets (conv encoder, IRL value iteration) need the 2D basis
    # grid and the (basis_grid, valid_mask, basis_valid) forward signature.
    use_grid = use_conv or use_irl

    if verbose:
        n_obs = int(obs_counts_np.sum())
        n_obs_cells = int((obs_counts_np > 0).sum())
        print(f"\n{'='*60}")
        if use_spline:
            print(f"TORCH PIPELINE: P-spline GAM Resistance + Circuit + PPP")
        elif use_irl:
            print(f"TORCH PIPELINE: IRL value-shaped Resistance + Circuit + PPP")
        else:
            print(f"TORCH PIPELINE: Neural Net Resistance + Circuit + PPP")
        print(f"{'='*60}")
        print(f"  Device: {dev}" + (f" ({torch.cuda.get_device_name()})" if use_cuda else ""))
        print(f"  Grid: {n_rows} x {n_cols} ({n_valid} valid cells)")
        print(f"  Observations: {n_obs} GPS fixes in {n_obs_cells} cells")
        print(f"  Cell area: {cell_area:.1f} m²")
        print(f"  Solver: {solver}")
        print(f"  Model type: {model_type}")
        if use_diff_omniscape:
            offset = block_size // 2
            n_focal = len(range(offset, n_rows, block_size)) * len(range(offset, n_cols, block_size))
            print(f"  Diff-omniscape: radius={radius}, block_size={block_size}, "
                  f"focal_fraction={focal_fraction:.0%}, n_focal={n_focal}")
            if use_cuda:
                print(f"  NOTE: diff_omniscape uses CPU for local solves; NN on GPU")
        if use_absorption:
            print(f"  Absorption: α={absorption:.4f} "
                  f"(effective radius ≈ {1.0/np.sqrt(absorption):.0f} px)")
        print(f"  NN architecture: {n_features} → "
              + " → ".join([str(hidden_dim)] * n_hidden_layers)
              + " → 1 (+ linear skip)")
        if use_spline:
            n_basis_per = n_knots + spline_degree + 1
            n_marginal = n_features * n_basis_per
            n_pairs = n_features * (n_features - 1) // 2 if include_interactions else 0
            n_interact = n_pairs * n_basis_per ** 2
            n_smooth_params = n_features + 2 * n_pairs
            print(f"  Spline: {n_knots} knots, degree {spline_degree}, "
                  f"{n_basis_per} basis/cov")
            print(f"  Marginal terms: {n_features} × {n_basis_per} = "
                  f"{n_marginal} coefs")
            if include_interactions:
                print(f"  Interactions: {n_pairs} pairs × "
                      f"{n_basis_per**2} = {n_interact} coefs")
            print(f"  Smoothing params: {n_smooth_params}")
        if use_conv:
            # Receptive field with dilated convolutions: sum of dilations * (k-1) + 1
            if use_dilated:
                rf = 1 + sum(2**i * (conv_kernel_size - 1) for i in range(n_conv_layers))
                # +1 for the input projection layer
                rf += (conv_kernel_size - 1)
            else:
                rf = 2 * n_conv_layers + 1 + (conv_kernel_size - 1)
            print(f"  Conv encoder: {n_conv_layers} × ResBlock({conv_channels}, "
                  f"k={conv_kernel_size}, dilated={use_dilated}) → "
                  f"receptive field {rf}×{rf} px")
            if dropout > 0:
                print(f"  MLP dropout: {dropout}")
            if intensity_hidden > 0:
                print(f"  Learned intensity MLP: 1 → {intensity_hidden} → 1")
        if use_irl:
            print(f"  Soft value iteration: β={beta}, γ_d={gamma_d}, "
                  f"{n_value_iter} iters → log R = offset − scale·V")
        if warmup_epochs > 0:
            print(f"  LR warm-up: {warmup_epochs} epochs")
        if absorption_schedule is not None:
            print(f"  Absorption curriculum: {absorption_schedule[0]:.4f} → "
                  f"{absorption_schedule[1]:.4f}")
        print(f"  CG tolerance: {cg_tol:.1e}")
        print(f"  LR: {lr}, weight_decay: {weight_decay}, patience: {patience}")
        print(f"  Regularization: reg_mean={reg_mean}, reg_var={reg_var}, "
              f"target_logR_var={target_logR_var}, reg_skip={reg_skip}, "
              f"log_R_baseline={log_R_baseline}")

    # ---- Relax CG tolerance for speed ----
    original_rtol = getattr(cs, "CG_RTOL", 1e-10)
    cs.CG_RTOL = float(cg_tol)
    cs.CG_WARM_START = True

    # ---- Tensors (on device) ----
    basis_t = torch.tensor(basis_values_np, dtype=torch.float64, device=dev)
    obs_t = torch.tensor(obs_counts_np, dtype=torch.float64, device=dev)

    # ---- 2D grid for grid-context nets (conv encoder / IRL value iteration) ----
    basis_grid_t = None
    if use_grid:
        n_cells = n_rows * n_cols
        grid_np = np.zeros((n_features, n_cells), dtype=np.float64)
        grid_np[:, valid_mask_np] = basis_values_np.T  # fill valid; rest stays 0
        grid_np = grid_np.reshape(1, n_features, n_rows, n_cols)
        basis_grid_t = torch.tensor(grid_np, dtype=torch.float64, device=dev)

    # ---- Model (on device) ----
    if use_spline:
        net = SplineResistanceNet(
            n_features, n_knots=n_knots, degree=spline_degree,
            include_interactions=include_interactions,
            lambda_init_marginal=float(lambda_init_marginal),
            lambda_init_interaction=float(lambda_init_interaction),
            lambda_min=float(lambda_min),
            intensity_spline=bool(intensity_spline),
            intensity_n_knots=int(intensity_n_knots),
            intensity_degree=int(intensity_degree),
            lambda_init_intensity=float(lambda_init_intensity),
            intensity_log1p_max=float(intensity_log1p_max),
        ).double().to(dev)
        net.setup_basis_matrices(basis_values_np, device=dev)
    elif use_conv:
        net = ConvResistanceNet(
            n_features, conv_channels=conv_channels,
            n_conv_layers=n_conv_layers,
            conv_kernel_size=conv_kernel_size,
            hidden=hidden_dim, n_mlp_layers=n_hidden_layers,
            dropout=dropout, use_dilated=use_dilated,
            intensity_hidden=intensity_hidden,
        ).double().to(dev)
    elif use_irl:
        net = IRLResistanceNet(
            n_features, hidden=hidden_dim, n_layers=n_hidden_layers,
            vi_beta=float(beta), gamma_d=float(gamma_d),
            n_value_iter=int(n_value_iter),
            value_scale_init=float(value_scale_init),
        ).double().to(dev)
    else:
        net = ResistanceNet(n_features, hidden_dim, n_hidden_layers).double().to(dev)
    alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, device=dev))
    # Reparameterize gamma through softplus so gamma is always positive.
    # Initialize raw so that softplus(raw) = 1.0 (matching previous init).
    _raw_gamma_init = float(np.log(np.exp(1.0) - 1.0))  # ≈ 0.5413
    raw_gamma = nn.Parameter(torch.tensor(_raw_gamma_init, dtype=torch.float64, device=dev))

    # When intensity spline is active, gamma is absorbed into the spline
    use_intensity_spline = use_spline and net.intensity_spline
    if use_intensity_spline:
        raw_gamma.requires_grad_(False)

    if warm_start_theta is not None:
        ws = np.asarray(warm_start_theta, dtype=np.float64).ravel()
        if verbose:
            print(f"  Warm start from log-linear: [{', '.join(f'{w:.4f}' for w in ws)}]")
        net.warm_start(ws)

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad) + 2
    if use_intensity_spline:
        n_params -= 1  # gamma is frozen
    if verbose:
        print(f"  Total parameters: {n_params} ({n_params - 2} NN + "
              f"{'1 intensity (alpha only, gamma absorbed into spline)' if use_intensity_spline else '2 intensity'}")

    # ---- Optimizer ----
    # Separate param groups with appropriate weight_decay.
    if use_spline:
        # Spline model: smoothing penalty handles regularisation, so no L2 on
        # spline coefficients. Log-lambda params are self-regularising.
        spline_coef_params = (list(net.spline_coefs.parameters())
                              + list(net.interaction_coefs.parameters()))
        log_lambda_params = (list(net.log_lambda_smooth.parameters())
                             + list(net.log_lambda_interact_row.parameters())
                             + list(net.log_lambda_interact_col.parameters()))
        skip_params = [net.skip.weight, net.skip.bias, net.intercept]
        # Intensity params: alpha always; gamma only if no intensity spline
        intensity_params = [alpha]
        if not use_intensity_spline:
            intensity_params.append(raw_gamma)
        # Intensity spline coefficients and smoothing
        if use_intensity_spline:
            spline_coef_params = spline_coef_params + [net.intensity_coefs]
            log_lambda_params = log_lambda_params + [net.log_lambda_intensity]
        all_params = [
            {"params": skip_params, "weight_decay": 0.0},
            {"params": spline_coef_params, "weight_decay": 0.0},
            {"params": log_lambda_params, "weight_decay": 0.0, "lr": lr * 0.5},
            {"params": intensity_params, "weight_decay": 0.0},
        ]
        all_params_flat = (skip_params + spline_coef_params
                           + log_lambda_params + intensity_params)
    else:
        # NN models: skip.weight gets NO weight_decay so L2 doesn't
        # push it back to zero (which would re-create the flat saddle point).
        skip_params = [net.skip.weight]
        other_nn_params = [p for n, p in net.named_parameters()
                           if not n.startswith("skip.weight")]
        all_params = [
            {"params": skip_params, "weight_decay": 0.0},
            {"params": other_nn_params, "weight_decay": weight_decay},
            {"params": [alpha, raw_gamma], "weight_decay": 0.0},
        ]
        all_params_flat = skip_params + other_nn_params + [alpha, raw_gamma]
    opt = torch.optim.Adam(all_params, lr=lr)
    # LR schedule: optional linear warm-up then cosine decay
    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.02, end_factor=1.0,
            total_iters=warmup_epochs
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(n_epochs - warmup_epochs, 1), eta_min=lr * 0.01
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs]
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs, eta_min=lr * 0.01
        )

    # ---- Training loop ----
    history = []
    best = dict(loss=float("inf"), epoch=0, state=None, alpha=0.0, gamma=1.0)
    wait = 0

    if verbose:
        print(f"\n  {'Epoch':>5}  {'Loss':>10}  {'LogLik':>10}  "
              f"{'Alpha':>8}  {'Gamma':>8}  {'LR':>10}  {'Time':>6}")
        print(f"  {'-'*65}")

    for ep in range(n_epochs):
        t0 = time.time()

        # Absorption curriculum: linearly interpolate from start to end
        if absorption_schedule is not None and use_absorption:
            frac = min(ep / max(n_epochs - 1, 1), 1.0)
            absorption_ep = (absorption_schedule[0]
                             + frac * (absorption_schedule[1] - absorption_schedule[0]))
        else:
            absorption_ep = absorption

        try:
            opt.zero_grad()

            # Forward: covariates → R → circuit → C → log λ → loglik
            if use_grid:
                net.train()
                R, log_R_raw = net.forward_with_log_R(
                    basis_grid_t, valid_mask_np, basis_t)
            elif use_spline:
                R, log_R_raw = net.forward_with_log_R(basis_t)
            else:
                R, log_R_raw = net.forward_with_log_R(basis_t)
            if use_diff_omniscape:
                rng_seed = (seed + ep) if seed is not None else ep
                C = _DiffOmniscapeSolveFn.apply(
                    R, valid_mask_np, n_rows, n_cols,
                    source_spacing, source_from_resistance,
                    radius, block_size, focal_fraction, rng_seed,
                )
            elif use_absorption:
                C = _AbsorptionCircuitSolveFn.apply(
                    R, valid_mask_np, n_rows, n_cols,
                    source_spacing, source_from_resistance,
                    absorption_ep,
                )
            else:
                C = _CircuitSolveFn.apply(
                    R, valid_mask_np, n_rows, n_cols,
                    source_spacing, source_from_resistance,
                )

            gamma = F.softplus(raw_gamma)
            log1p_C = torch.log1p(C.clamp(min=0))
            # Intensity: spline or learned MLP or parametric
            if use_intensity_spline:
                log_lam = alpha + net.eval_intensity_spline(log1p_C)
            elif use_conv and net.intensity_mlp is not None:
                log_lam = alpha + gamma * net.intensity_mlp(
                    log1p_C.unsqueeze(-1)).squeeze(-1)
            else:
                log_lam = alpha + gamma * log1p_C
            ll = _ppp_loglik(log_lam, obs_t, float(cell_area))

            # Regularization: prevent degenerate R → R_max everywhere.
            # Penalties are scaled by n_valid so they are commensurate with
            # the PPP integral term (≈ n_valid × cell_area × mean(λ)).
            mean_logR = log_R_raw.mean()
            pen_mean = n_valid * reg_mean * (mean_logR - log_R_baseline) ** 2

            if use_spline:
                # P-spline smoothing penalty replaces reg_var and reg_skip.
                # penalty_scale > 1 biases toward smoother fits.
                pen_smooth = penalty_scale * net.smoothing_penalty()
                loss = -ll + pen_mean + pen_smooth
            else:
                var_logR = log_R_raw.var()
                # One-sided hinge: penalise var(logR) < target, no reward above.
                log_var = torch.log(var_logR + 0.01)
                pen_var = n_valid * reg_var * F.softplus(
                    math.log(target_logR_var) - log_var
                )
                # Prevent skip weights collapsing to zero.
                skip_norm = net.skip.weight.norm()
                pen_skip = n_valid * reg_skip / (skip_norm + 0.01)
                loss = -ll + pen_mean + pen_var + pen_skip

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params_flat, grad_clip)

            opt.step()
            sched.step()

        except RuntimeError as e:
            if "CG did not converge" in str(e):
                if verbose:
                    print(f"  [{ep:3d}] CG failed — halving learning rate")
                for pg in opt.param_groups:
                    pg["lr"] *= 0.5
                history.append(float("inf"))
                continue
            raise

        lv = loss.item()
        elapsed = time.time() - t0
        history.append(lv)

        # Track best
        if lv < best["loss"]:
            best.update(
                loss=lv,
                epoch=ep,
                alpha=alpha.item(),
                gamma=F.softplus(raw_gamma).item(),
                state={k: v.clone() for k, v in net.state_dict().items()},
            )
            wait = 0
        else:
            wait += 1

        if verbose and (ep % 10 == 0 or ep < 5 or ep == n_epochs - 1):
            abs_str = (f", α_abs={absorption_ep:.4f}"
                       if absorption_schedule is not None and use_absorption else "")
            print(f"  {ep:5d}  {lv:+10.2f}  {ll.item():10.2f}  "
                  f"{alpha.item():8.4f}  {F.softplus(raw_gamma).item():8.4f}  "
                  f"{sched.get_last_lr()[0]:10.1e}  {elapsed:5.1f}s{abs_str}")
            with torch.no_grad():
                net.eval()
                if use_grid:
                    R_mon, logR_mon = net.forward_with_log_R(
                        basis_grid_t, valid_mask_np, basis_t)
                else:
                    R_mon, logR_mon = net.forward_with_log_R(basis_t)
                net.train()
                print(f"         R: min={R_mon.min():.1f}, "
                      f"med={R_mon.median():.1f}, "
                      f"max={R_mon.max():.1f}, "
                      f"std={R_mon.std():.1f}")
                pm = n_valid * reg_mean * (logR_mon.mean() - log_R_baseline) ** 2
                if use_spline:
                    ps = net.smoothing_penalty().item()
                    print(f"         logR: mean={logR_mon.mean():.2f}, "
                          f"std={logR_mon.std():.2f}, "
                          f"pen_m={pm:.0f}, pen_smooth={ps:.1f}")
                else:
                    lv_mon = torch.log(logR_mon.var() + 0.01)
                    pv = n_valid * reg_var * F.softplus(
                        math.log(target_logR_var) - lv_mon
                    ).item()
                    sn = net.skip.weight.norm().item()
                    ps = n_valid * reg_skip / (sn + 0.01)
                    print(f"         logR: mean={logR_mon.mean():.2f}, "
                          f"std={logR_mon.std():.2f}, "
                          f"pen_m={pm:.0f}, pen_v={pv:.0f}, "
                          f"skip_norm={sn:.4f}, pen_s={ps:.0f}")

        if wait >= patience:
            if verbose:
                print(f"\n  Early stopping at epoch {ep} (patience={patience})")
            break

    # ---- Restore best ----
    if best["state"] is None:
        if verbose:
            print("  WARNING: No improvement found. Using final state.")
        best.update(
            alpha=alpha.item(),
            gamma=F.softplus(raw_gamma).item(),
            state={k: v.clone() for k, v in net.state_dict().items()},
            epoch=len(history) - 1,
        )

    net.load_state_dict(best["state"])
    alpha_v = best["alpha"]
    gamma_v = best["gamma"]

    if verbose:
        print(f"\n  Best epoch: {best['epoch']}, loss: {best['loss']:.2f}")
        print(f"  α = {alpha_v:.4f}, γ = {gamma_v:.4f}")

    # ---- Final forward (no grad) ----
    net.eval()
    with torch.no_grad():
        if use_grid:
            R_fin = net(basis_grid_t, valid_mask_np, basis_t).cpu().numpy()
        else:
            R_fin = net(basis_t).cpu().numpy()

    # Final circuit solve at full precision (always CPU for pyamg AMG accuracy)
    if use_cuda and not use_diff_omniscape:
        cs.enable_gpu(False)
    n = n_rows * n_cols
    Rf = np.full(n, 1e6, dtype=np.float64)
    Rf[valid_mask_np] = R_fin

    if use_diff_omniscape:
        # Full-precision diff_omniscape with smaller block_size, no subsampling
        do = _get_diff_omniscape_module()
        final_block = min(block_size, 5)  # finer grid for final output
        C_map_fin, _, _ = do.solve_diff_omniscape(
            Rf.reshape(n_rows, n_cols),
            radius=int(radius),
            block_size=int(final_block),
            source_from_resistance=bool(source_from_resistance),
        )
        # Interpolate to full grid
        offset_fin = final_block // 2
        focal_rows_fin = list(range(offset_fin, n_rows, final_block))
        focal_cols_fin = list(range(offset_fin, n_cols, final_block))
        C_full_fin = _interpolate_block_grid(
            C_map_fin, final_block, focal_rows_fin, focal_cols_fin,
            n_rows, n_cols
        )
        C_fin = C_full_fin.ravel()[valid_mask_np].copy()
        if verbose:
            print(f"  Final diff_omniscape solve: block_size={final_block}, "
                  f"n_focal={len(focal_rows_fin)*len(focal_cols_fin)}")
    elif use_absorption:
        cs.CG_RTOL = 1e-10
        result_fin = cs.solve_circuit_absorption(
            Rf.reshape(n_rows, n_cols),
            absorption=float(absorption),
            source_spacing=int(source_spacing),
            source_from_resistance=bool(source_from_resistance),
        )
        C_fin = result_fin["current_density"][valid_mask_np].copy()
        if verbose:
            print(f"  Final absorption solve: α={absorption:.4f}")
    else:
        cs.CG_RTOL = 1e-10
        result_fin = cs.solve_circuit(
            Rf.reshape(n_rows, n_cols),
            source_spacing=int(source_spacing),
            source_from_resistance=bool(source_from_resistance),
        )
        C_fin = result_fin["current_density"][valid_mask_np].copy()

    # Compute final intensity using learned MLP or parametric form
    log1p_C_fin = np.log1p(np.maximum(C_fin, 0.0))
    if use_intensity_spline:
        with torch.no_grad():
            log1p_t = torch.tensor(log1p_C_fin, dtype=torch.float64, device=dev)
            f_int = net.eval_intensity_spline(log1p_t).cpu().numpy()
        log_lam_fin = alpha_v + f_int
    elif use_conv and net.intensity_mlp is not None:
        with torch.no_grad():
            log1p_t = torch.tensor(log1p_C_fin, dtype=torch.float64, device=dev)
            intensity_transform = net.intensity_mlp(
                log1p_t.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        log_lam_fin = alpha_v + gamma_v * intensity_transform
    else:
        log_lam_fin = alpha_v + gamma_v * log1p_C_fin
    ll_fin = float(
        np.sum(obs_counts_np * log_lam_fin)
        - cell_area * np.sum(np.exp(np.minimum(log_lam_fin, 20.0)))
    )

    # Restore original CG tolerance and disable GPU for clean state
    cs.CG_RTOL = original_rtol
    cs.CG_WARM_START = False
    cs.enable_gpu(False)

    total_time = time.time() - t0_all

    # ---- Save model weights (always on CPU for portability) ----
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        net_cpu = {k: v.cpu() for k, v in net.state_dict().items()}
        save_dict = {
            "net": net_cpu,
            "alpha": alpha_v,
            "gamma": gamma_v,
            "n_features": n_features,
            "model_type": model_type,
            "hidden_dim": hidden_dim,
            "n_hidden_layers": n_hidden_layers,
            "use_conv": use_conv,
            "conv_channels": conv_channels if use_conv else None,
            "n_conv_layers": n_conv_layers if use_conv else None,
            "conv_kernel_size": conv_kernel_size if use_conv else None,
            "dropout": dropout if use_conv else None,
            "use_dilated": use_dilated if use_conv else None,
            "intensity_hidden": intensity_hidden if use_conv else None,
        }
        if use_spline:
            save_dict.update({
                "n_knots": n_knots,
                "spline_degree": spline_degree,
                "include_interactions": include_interactions,
                "intensity_spline": bool(intensity_spline),
                "intensity_n_knots": int(intensity_n_knots),
                "intensity_degree": int(intensity_degree),
                "lambda_init_intensity": float(lambda_init_intensity),
                "intensity_log1p_max": float(intensity_log1p_max),
            })
        if use_irl:
            save_dict.update({
                "beta": float(beta),
                "gamma_d": float(gamma_d),
                "n_value_iter": int(n_value_iter),
                "value_scale_init": float(value_scale_init),
            })
        torch.save(save_dict, os.path.join(output_dir, "resistance_nn.pt"))

    if verbose:
        print(f"\n  Done in {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"  Final loglik: {ll_fin:.2f}")
        print(f"  Parameters: {n_params}")

    # ---- Effective log-linear coefficients ----
    if use_spline:
        effective_loglinear = np.array(
            net.get_effective_loglinear(), dtype=np.float64)
    else:
        with torch.no_grad():
            skip_w = net.skip.weight.squeeze().cpu().numpy().copy()
            skip_b = net.skip.bias.cpu().item()
        effective_loglinear = np.concatenate([[skip_b], skip_w])

    # ---- Partial effects and UQ (spline only) ----
    partial_effects = None
    interaction_effects = None
    uq_results = None
    if use_spline:
        posterior_cov = None

        if compute_uq:
            if verbose:
                print(f"\n  Computing Bayesian UQ (block_only={uq_block_only})...")
                t_uq = time.time()

            # Build NLL function for Hessian computation.
            # Each call re-runs forward + circuit solve and returns NLL as float.
            n = n_rows * n_cols
            def _nll_fn():
                """Negative log-likelihood (no penalty) for Hessian FD."""
                with torch.no_grad():
                    R_uq = net(basis_t).cpu().numpy()
                R_full_uq = np.full(n, 1e6, dtype=np.float64)
                R_full_uq[valid_mask_np] = R_uq

                if use_absorption:
                    res_uq = cs.solve_circuit_absorption(
                        R_full_uq.reshape(n_rows, n_cols),
                        absorption=float(absorption),
                        source_spacing=int(source_spacing),
                        source_from_resistance=bool(source_from_resistance),
                    )
                    C_uq = res_uq["current_density"][valid_mask_np]
                else:
                    res_uq = cs.solve_circuit(
                        R_full_uq.reshape(n_rows, n_cols),
                        source_spacing=int(source_spacing),
                        source_from_resistance=bool(source_from_resistance),
                    )
                    C_uq = res_uq["current_density"][valid_mask_np]

                log1p_C_uq = np.log1p(np.maximum(C_uq, 0.0))
                if use_intensity_spline:
                    with torch.no_grad():
                        log1p_t_uq = torch.tensor(log1p_C_uq, dtype=torch.float64, device=dev)
                        f_int_uq = net.eval_intensity_spline(log1p_t_uq).cpu().numpy()
                    log_lam_uq = alpha_v + f_int_uq
                else:
                    log_lam_uq = alpha_v + gamma_v * log1p_C_uq
                nll = -(
                    np.sum(obs_counts_np * log_lam_uq)
                    - cell_area * np.sum(np.exp(np.minimum(log_lam_uq, 20.0)))
                )
                return float(nll)

            hess_result = net.compute_nll_hessian_block(
                _nll_fn, block_only=uq_block_only)

            uq_output = net.compute_posterior_covariance(
                hess_result, penalty_scale=float(penalty_scale))
            posterior_cov = uq_output["covariance"]

            uq_results = {
                "edf": uq_output["edf"],
                "significance": {
                    int(k): {
                        "chi_sq": float(v["chi_sq"]),
                        "edf": float(v["edf"]),
                        "p_value": float(v["p_value"]),
                    }
                    for k, v in uq_output["significance"].items()
                },
            }

            if verbose:
                dt = time.time() - t_uq
                print(f"  UQ completed in {dt:.1f}s")
                for k in range(net.n_features):
                    sig = uq_output["significance"][k]
                    print(f"    covariate {k}: EDF={sig['edf']:.1f}, "
                          f"chi2={sig['chi_sq']:.1f}, "
                          f"p={sig['p_value']:.4f}")

        pe = net.get_partial_effects(n_grid=100, posterior_cov=posterior_cov)
        partial_effects = {}
        for k, v in pe.items():
            entry = {
                "grid": v["grid"].tolist(),
                "effect": v["effect"].tolist(),
            }
            if "se" in v:
                entry["se"] = v["se"].tolist()
                entry["lower_95"] = v["lower_95"].tolist()
                entry["upper_95"] = v["upper_95"].tolist()
            partial_effects[int(k)] = entry

        interaction_effects = net.get_interaction_effects(n_grid=30)

    # Intensity spline partial effect
    intensity_partial_effect = None
    if use_intensity_spline:
        with torch.no_grad():
            # Evaluate on a grid from 0 to the max observed log(1+C)
            grid_max = float(np.max(log1p_C_fin)) * 1.05 + 0.1
            grid_np = np.linspace(0.0, grid_max, 100)
            grid_t = torch.tensor(grid_np, dtype=torch.float64, device=dev)
            B_int = _bspline_basis_matrix_torch(
                grid_t, net._intensity_knots,
                net.intensity_n_basis, net.intensity_degree)
            f_raw = (B_int @ net.intensity_coefs).cpu().numpy()
            # Don't center the display — show raw f(x)
            intensity_partial_effect = {
                "grid": grid_np.tolist(),
                "effect": f_raw.tolist(),
            }

    result = {
        "resistance": R_fin,
        "connectivity": C_fin,
        "log_lambda": log_lam_fin,
        "alpha": float(alpha_v),
        "gamma": float(gamma_v),
        "loglik": float(ll_fin),
        "loss_history": [float(x) for x in history],
        "best_epoch": int(best["epoch"]),
        "n_params": int(n_params),
        "n_epochs_run": int(len(history)),
        "total_time": float(total_time),
        "effective_loglinear": effective_loglinear.tolist(),
        "solver": str(solver),
        "device": str(dev),
        "model_type": str(model_type),
        "intensity_spline": bool(use_intensity_spline),
    }
    if partial_effects is not None:
        result["partial_effects"] = partial_effects
    if interaction_effects is not None:
        result["interaction_effects"] = interaction_effects
    if uq_results is not None:
        result["uq_results"] = uq_results
    if intensity_partial_effect is not None:
        result["intensity_partial_effect"] = intensity_partial_effect

    return result


# ===========================================================================
# Full Bayesian inference via Langevin Monte Carlo
# ===========================================================================

def run_langevin_sampling(
    basis_values_np,             # (n_valid, n_feat) float64
    obs_counts_np,               # (n_valid,) float64
    n_rows,                      # int
    n_cols,                      # int
    valid_mask_np,               # (n_cells,) bool
    cell_area,                   # float (m²)
    # Pre-trained model checkpoint (from run_torch_optimization output_dir)
    model_path=None,             # Path to resistance_nn.pt
    # Alternatively, pass the state dict directly (for in-session use)
    model_state=None,            # dict with 'net', 'alpha', 'gamma', etc.
    # Circuit solver settings (must match optimization)
    source_spacing=1,
    source_from_resistance=True,
    solver="global_absorption",
    absorption=0.002,
    # Sampler settings
    n_samples=2000,              # Posterior samples to collect
    burn_in=500,                 # Burn-in iterations (discarded)
    thin=5,                      # Thinning: collect every thin-th sample
    step_size=1e-5,              # Langevin step size (auto if None)
    precondition=True,           # Use Adam second-moment preconditioner
    precondition_warmup=50,      # Steps to build preconditioner before sampling
    fix_smoothing=False,         # Fix smoothing params λ_k at MAP
    fix_intensity=False,         # Fix alpha, gamma at MAP
    use_mala=True,               # True = MALA (MH correction); False = ULA
    target_accept=0.574,         # Target acceptance rate for adaptive step size
    # Regularization (must match optimization)
    reg_mean=1.0,
    log_R_baseline=3.0,
    penalty_scale=5.0,
    # Model architecture (must match optimization)
    n_knots=10,
    spline_degree=3,
    include_interactions=True,
    lambda_min=0.1,
    # Misc
    grad_clip=100.0,             # Per-param gradient clipping (safety)
    cg_tol=DEFAULT_CG_TOL,
    seed=42,
    verbose=True,
    output_dir=None,
    device="auto",               # "auto", "cuda", or "cpu"
):
    """
    Full Bayesian posterior sampling via Metropolis-adjusted Langevin (MALA)
    or unadjusted Langevin (ULA).

    Starts from the MAP estimate found by run_torch_optimization and explores
    the posterior using Langevin dynamics with optional MH correction:

        Proposal: θ* = θ_t + (ε/2) M⁻¹ ∇log p(θ|y) + √ε M^{-1/2} N(0,I)
        Accept:   min(1, p(θ*)/p(θ) × q(θ|θ*)/q(θ*|θ))  [MALA]

    where ∇log p(θ|y) = ∇(log L + log prior) and M is a diagonal
    preconditioner estimated from Adam-like second moments during burn-in.

    The P-spline smoothing penalty acts as the prior:
        log p(θ) ∝ -penalty_scale × Σ_k λ_k β_k^T S_k β_k

    MALA (use_mala=True, default): eliminates ULA discretization bias via
    Metropolis-Hastings correction. Each step costs ~2× forward passes but
    produces an exact posterior sample. Step size is adapted during burn-in
    to target ~57.4% acceptance (MALA optimal in high dimensions).

    ULA (use_mala=False): no accept/reject; faster per step but accumulates
    discretization bias proportional to ε. Only recommended for diagnostics.

    The preconditioner is frozen after burn-in to ensure a stationary
    transition kernel during sampling.

    Parameters
    ----------
    basis_values_np : ndarray (n_valid, n_features)
    obs_counts_np : ndarray (n_valid,)
    n_rows, n_cols : int
    valid_mask_np : ndarray (n_cells,) bool
    cell_area : float
    model_path : str or None
        Path to resistance_nn.pt checkpoint.
    model_state : dict or None
        Alternative to model_path: pass state dict directly.
    n_samples : int
        Number of posterior samples to collect after burn-in.
    burn_in : int
        Burn-in steps (discarded).
    thin : int
        Collect a sample every `thin` steps.
    step_size : float or None
        Langevin step size ε. If None, auto-tuned from posterior curvature.
    precondition : bool
        Use diagonal preconditioner from running second moments.
    use_mala : bool
        If True (default), use Metropolis-adjusted Langevin algorithm.
        If False, use unadjusted Langevin (legacy behavior).
    target_accept : float
        Target acceptance rate for adaptive step size during burn-in.
        Default 0.574 (MALA-optimal for high dimensions).
    fix_smoothing : bool
        Fix smoothing parameters at MAP (simpler; sample only spline coefs).
    fix_intensity : bool
        Fix alpha, gamma at MAP (sample only resistance params).

    Returns
    -------
    dict with:
        samples_effective_loglinear : (n_samples, K+1) array
        samples_alpha : (n_samples,) array
        samples_gamma : (n_samples,) array
        samples_partial_effects : dict[int, (n_samples, n_grid) array]
        log_posterior_trace : list of floats (burn_in + n_samples*thin)
        acceptance_rate : float (meaningful for MALA; always 1.0 for ULA)
        ess : dict of effective sample sizes per parameter
        summary : dict of posterior means, sds, quantiles
        elapsed_time : float (seconds)
    """
    t0_all = time.time()

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # ---- Load model ----
    if model_state is not None:
        checkpoint = model_state
    elif model_path is not None:
        checkpoint = torch.load(model_path, map_location="cpu",
                                weights_only=False)
    else:
        raise ValueError("Either model_path or model_state must be provided")

    n_features = int(checkpoint.get("n_features",
                     basis_values_np.shape[1] if basis_values_np.ndim > 1
                     else 1))
    n_knots_ckpt = int(checkpoint.get("n_knots", n_knots))
    degree_ckpt = int(checkpoint.get("spline_degree", spline_degree))
    interactions_ckpt = bool(checkpoint.get("include_interactions",
                                            include_interactions))

    # Intensity spline settings from checkpoint
    intensity_spline_ckpt = bool(checkpoint.get("intensity_spline", False))
    intensity_n_knots_ckpt = int(checkpoint.get("intensity_n_knots", 5))
    intensity_degree_ckpt = int(checkpoint.get("intensity_degree", 3))
    lambda_init_intensity_ckpt = float(checkpoint.get("lambda_init_intensity", 2.0))
    intensity_log1p_max_ckpt = float(checkpoint.get("intensity_log1p_max", 10.0))

    use_absorption = (str(solver).lower() in ("global_absorption", "absorption"))

    # ---- Device selection (same logic as training loop) ----
    if device == "auto":
        use_cuda = _GPU_AVAILABLE and _CUPY_AVAILABLE
        dev = torch.device("cuda" if use_cuda else "cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda is not available")
        if not _CUPY_AVAILABLE:
            raise RuntimeError("CUDA requested but cupy is not installed")
        dev = torch.device("cuda")
        use_cuda = True
    else:
        dev = torch.device("cpu")
        use_cuda = False

    net = SplineResistanceNet(
        n_features=n_features,
        n_knots=n_knots_ckpt,
        degree=degree_ckpt,
        include_interactions=interactions_ckpt,
        lambda_init_marginal=0.0,   # Will be overwritten by state_dict
        lambda_init_interaction=0.0,
        lambda_min=float(lambda_min),
        intensity_spline=intensity_spline_ckpt,
        intensity_n_knots=intensity_n_knots_ckpt,
        intensity_degree=intensity_degree_ckpt,
        lambda_init_intensity=lambda_init_intensity_ckpt,
        intensity_log1p_max=intensity_log1p_max_ckpt,
    ).double().to(dev)
    net.load_state_dict(checkpoint["net"])
    net.setup_basis_matrices(np.asarray(basis_values_np, dtype=np.float64),
                             device=dev)
    net.train()

    use_intensity_spline = intensity_spline_ckpt and net.intensity_spline

    alpha_map = float(checkpoint["alpha"])
    gamma_map = float(checkpoint["gamma"])

    # Parameterize intensity: alpha, raw_gamma (gamma = softplus(raw_gamma))
    alpha = nn.Parameter(torch.tensor(alpha_map, dtype=torch.float64, device=dev))
    _raw_gamma_init = float(np.log(np.exp(gamma_map) - 1.0 + 1e-8))
    raw_gamma = nn.Parameter(torch.tensor(_raw_gamma_init, dtype=torch.float64,
                                          device=dev))
    if use_intensity_spline:
        raw_gamma.requires_grad_(False)

    # ---- Prepare data tensors ----
    basis_t = torch.tensor(np.asarray(basis_values_np, dtype=np.float64),
                           dtype=torch.float64, device=dev)
    obs_t = torch.tensor(np.asarray(obs_counts_np, dtype=np.float64),
                         dtype=torch.float64, device=dev)
    obs_counts_np_local = np.asarray(obs_counts_np, dtype=np.float64)
    n_valid = basis_t.shape[0]

    # ---- Initialize circuit solver ----
    cs = _get_circuit_module()
    original_rtol = getattr(cs, "CG_RTOL", 1e-6)
    cs.CG_RTOL = float(cg_tol)
    cs.CG_WARM_START = True

    # Enable GPU in circuit solver (cupy sparse solves)
    if use_cuda:
        cs.enable_gpu(True)
        if seed is not None:
            cp.random.seed(seed)
        if verbose:
            print(f"  Device: CUDA (GPU-accelerated circuit solves)")
    else:
        cs.enable_gpu(False)
        if verbose:
            print(f"  Device: CPU")

    # ---- Identify parameters to sample ----
    sampled_params = []
    param_names = []

    # Always sample spline coefficients and skip/intercept
    for k, p in enumerate(net.spline_coefs):
        sampled_params.append(p)
        param_names.append(f"spline_coef_{k}")
    for idx, p in enumerate(net.interaction_coefs):
        sampled_params.append(p)
        param_names.append(f"interaction_coef_{idx}")
    sampled_params.extend([net.skip.weight, net.skip.bias, net.intercept])
    param_names.extend(["skip_weight", "skip_bias", "intercept"])

    if not fix_smoothing:
        for k, p in enumerate(net.log_lambda_smooth):
            sampled_params.append(p)
            param_names.append(f"log_lambda_smooth_{k}")
        for idx, p in enumerate(net.log_lambda_interact_row):
            sampled_params.append(p)
            param_names.append(f"log_lambda_interact_row_{idx}")
        for idx, p in enumerate(net.log_lambda_interact_col):
            sampled_params.append(p)
            param_names.append(f"log_lambda_interact_col_{idx}")
        if use_intensity_spline and hasattr(net, 'log_lambda_intensity'):
            sampled_params.append(net.log_lambda_intensity)
            param_names.append("log_lambda_intensity")

    if not fix_intensity:
        sampled_params.append(alpha)
        param_names.append("alpha")
        if use_intensity_spline:
            sampled_params.append(net.intensity_coefs)
            param_names.append("intensity_coefs")
        else:
            sampled_params.append(raw_gamma)
            param_names.append("raw_gamma")

    n_total_scalar = sum(p.numel() for p in sampled_params)
    if verbose:
        sampler_name = "MALA" if use_mala else "ULA"
        print(f"\n  Langevin sampler ({sampler_name}): {n_total_scalar} scalar parameters")
        print(f"    Burn-in: {burn_in}, Samples: {n_samples}, Thin: {thin}")
        print(f"    Total steps: {burn_in + n_samples * thin}")
        print(f"    Fix smoothing: {fix_smoothing}, Fix intensity: {fix_intensity}")
        if use_mala:
            print(f"    Target acceptance rate: {target_accept:.3f}")
            print(f"    Adaptive step size during burn-in: ON")

    # ---- Auto step-size ----
    if step_size is None:
        # Heuristic: scale by 1/sqrt(n_obs) — Langevin optimal scaling
        n_obs = float(obs_counts_np_local.sum())
        step_size = 2.0 / (n_obs + n_valid)
        if verbose:
            print(f"    Auto step_size: {step_size:.2e}")

    # ---- Preconditioner: running second moment (Adam-like) ----
    if precondition:
        v_hat = [torch.ones_like(p) for p in sampled_params]
        beta2 = 0.999
    else:
        v_hat = None

    # ---- Forward + loss function (reuses training loop logic) ----
    def compute_loss_and_grad():
        """Compute -log p(θ|y) and its gradient w.r.t. sampled params."""
        for p in sampled_params:
            if p.grad is not None:
                p.grad.zero_()

        # Forward: covariates → R → circuit → C → log λ → loglik
        R, log_R_raw = net.forward_with_log_R(basis_t)

        if use_absorption:
            C = _AbsorptionCircuitSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
                absorption,
            )
        else:
            C = _CircuitSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
            )

        gamma_val = F.softplus(raw_gamma)
        log1p_C = torch.log1p(C.clamp(min=0))
        if use_intensity_spline:
            log_lam = alpha + net.eval_intensity_spline(log1p_C)
        else:
            log_lam = alpha + gamma_val * log1p_C
        ll = _ppp_loglik(log_lam, obs_t, float(cell_area))

        # Prior = smoothing penalty + mean regularization
        mean_logR = log_R_raw.mean()
        pen_mean = n_valid * reg_mean * (mean_logR - log_R_baseline) ** 2
        pen_smooth = penalty_scale * net.smoothing_penalty()
        neg_log_post = -ll + pen_mean + pen_smooth

        neg_log_post.backward()

        return float(neg_log_post.item()), float(ll.item())

    # ---- Storage for samples ----
    n_grid_pe = 50  # Grid points for partial effects
    total_steps = burn_in + n_samples * thin

    log_post_trace = []
    ll_trace = []

    # Pre-allocate sample arrays
    eff_ll_dim = n_features + 1  # [r_0, z_1, ..., z_K]
    samples_eff_ll = np.zeros((n_samples, eff_ll_dim), dtype=np.float64)
    samples_alpha = np.zeros(n_samples, dtype=np.float64)
    samples_gamma = np.zeros(n_samples, dtype=np.float64)
    # Partial effects: (n_samples, n_grid) per covariate
    samples_pe = {k: np.zeros((n_samples, n_grid_pe), dtype=np.float64)
                  for k in range(n_features)}

    # ---- Sampling loop ----
    sample_idx = 0
    n_accept = 0       # Accepted proposals (MALA)
    n_total_mh = 0     # Total MH proposals attempted

    # Adaptive step size state (dual averaging, Hoffman & Gelman 2014)
    mu_da = math.log(10.0 * step_size)   # Anchor point (log(10 × ε_0))
    log_eps = math.log(step_size)
    log_eps_bar = 0.0            # Smoothed log step size (starts at 0 → eps_bar=1)
    adapt_kappa = 0.75           # Relaxation exponent
    adapt_gamma = 0.05           # Shrinkage towards anchor
    adapt_t0 = 10                # Stabilization offset
    H_bar = 0.0                  # Running mean of acceptance stat

    # Frozen preconditioner: snapshot v_hat at end of burn-in
    v_hat_frozen = None

    if verbose:
        header = (f"  {'Step':>6}  {'NegLogPost':>12}  {'LogLik':>10}  "
                  f"{'StepSize':>10}")
        if use_mala:
            header += f"  {'Accept%':>8}"
        header += f"  {'Time':>6}"
        print(f"\n{header}")
        print(f"  {'-' * (65 if use_mala else 55)}")

    for step in range(total_steps):
        t0 = time.time()

        # Determine whether to update preconditioner (only during burn-in)
        update_precond = precondition and (step < burn_in) and v_hat is not None
        use_frozen = precondition and (step >= burn_in) and v_hat_frozen is not None

        # Compute gradient of negative log-posterior at current θ
        neg_lp, ll_val = compute_loss_and_grad()
        log_post_trace.append(-neg_lp)
        ll_trace.append(ll_val)

        # ---- Langevin proposal ----
        eps = step_size
        accepted = True   # Default for ULA (always accept)

        if use_mala:
            # -----------------------------------------------------------------
            # MALA: propose θ* via Langevin, evaluate log p(θ*), accept/reject
            # IMPORTANT: compute_loss_and_grad() calls .backward() so it must
            # be called OUTSIDE any torch.no_grad() block.
            # -----------------------------------------------------------------

            # Step 1: collect current grads + update/read preconditioner
            with torch.no_grad():
                theta_old = [p.data.clone() for p in sampled_params]
                grads_old = []
                precond_list = []

                for i, p in enumerate(sampled_params):
                    if p.grad is None:
                        grads_old.append(torch.zeros_like(p))
                        precond_list.append(1.0)
                        continue

                    g = p.grad.clone()
                    g_norm = g.norm()
                    if g_norm > grad_clip:
                        g = g * (grad_clip / g_norm)
                    grads_old.append(g)

                    # Update or read preconditioner
                    if update_precond:
                        v_hat[i] = beta2 * v_hat[i] + (1 - beta2) * g * g
                        v_corrected = v_hat[i] / (1 - beta2 ** (step + 1))
                        pc = 1.0 / (torch.sqrt(v_corrected) + 1e-8)
                    elif use_frozen:
                        pc = v_hat_frozen[i]
                    elif precondition and v_hat is not None:
                        v_hat[i] = beta2 * v_hat[i] + (1 - beta2) * g * g
                        v_corrected = v_hat[i] / (1 - beta2 ** (step + 1))
                        pc = 1.0 / (torch.sqrt(v_corrected) + 1e-8)
                    else:
                        pc = 1.0
                    precond_list.append(pc)

                # Step 2: Langevin proposal  θ* = θ - (ε/2) M⁻¹ g + √ε M^{-1/2} z
                for i, p in enumerate(sampled_params):
                    noise = torch.randn_like(p)
                    pc = precond_list[i]
                    g = grads_old[i]
                    if isinstance(pc, float):
                        p.data -= (eps / 2) * g
                        p.data += math.sqrt(eps) * noise
                    else:
                        p.data -= (eps / 2) * pc * g
                        p.data += math.sqrt(eps) * torch.sqrt(pc) * noise

            # Step 3: evaluate log p(θ*) and its gradient (needs autograd)
            neg_lp_star, _ = compute_loss_and_grad()

            # Step 4: MH ratio + accept/reject (no autograd needed)
            with torch.no_grad():
                # Collect proposal grads for reverse proposal density
                grads_star = []
                for p in sampled_params:
                    if p.grad is None:
                        grads_star.append(torch.zeros_like(p))
                    else:
                        g = p.grad.clone()
                        g_norm = g.norm()
                        if g_norm > grad_clip:
                            g = g * (grad_clip / g_norm)
                        grads_star.append(g)

                # Log-proposal densities for MH correction
                # q(θ*|θ) = N(θ* ; θ - (ε/2) M⁻¹ g(θ), ε M⁻¹)
                # log q ∝ -1/(2ε) Σ_i (1/M⁻¹_ii)(θ*_i - μ_i)²
                log_q_forward = 0.0  # log q(θ*|θ)
                log_q_reverse = 0.0  # log q(θ|θ*)

                for i in range(len(sampled_params)):
                    theta_star_i = sampled_params[i].data
                    theta_old_i = theta_old[i]
                    pc = precond_list[i]

                    if isinstance(pc, float):
                        mu_fwd = theta_old_i - (eps / 2) * grads_old[i]
                        mu_rev = theta_star_i - (eps / 2) * grads_star[i]
                        diff_fwd = theta_star_i - mu_fwd
                        diff_rev = theta_old_i - mu_rev
                        log_q_forward += -0.5 / eps * (diff_fwd * diff_fwd).sum().item()
                        log_q_reverse += -0.5 / eps * (diff_rev * diff_rev).sum().item()
                    else:
                        mu_fwd = theta_old_i - (eps / 2) * pc * grads_old[i]
                        mu_rev = theta_star_i - (eps / 2) * pc * grads_star[i]
                        inv_pc = 1.0 / (pc + 1e-30)
                        diff_fwd = theta_star_i - mu_fwd
                        diff_rev = theta_old_i - mu_rev
                        log_q_forward += (-0.5 / eps * (inv_pc * diff_fwd * diff_fwd).sum()).item()
                        log_q_reverse += (-0.5 / eps * (inv_pc * diff_rev * diff_rev).sum()).item()

                # log α = [log p(θ*) - log p(θ)] + [log q(θ|θ*) - log q(θ*|θ)]
                log_alpha = (-neg_lp_star + neg_lp) + (log_q_reverse - log_q_forward)
                log_alpha = min(0.0, log_alpha)

                u = math.log(max(np.random.uniform(), 1e-300))
                accepted = (u < log_alpha)
                n_total_mh += 1

                if accepted:
                    n_accept += 1
                else:
                    # Reject: restore θ_old
                    for i, p in enumerate(sampled_params):
                        p.data.copy_(theta_old[i])

            # Step 5: adaptive step size during burn-in (dual averaging)
            if step < burn_in:
                accept_prob = min(1.0, math.exp(log_alpha))
                m = step + 1
                H_bar = (1.0 - 1.0 / (m + adapt_t0)) * H_bar + \
                        (1.0 / (m + adapt_t0)) * (target_accept - accept_prob)
                log_eps = mu_da - math.sqrt(m) / adapt_gamma * H_bar
                eta = m ** (-adapt_kappa)
                log_eps_bar = eta * log_eps + (1.0 - eta) * log_eps_bar
                step_size = math.exp(log_eps)
                step_size = max(1e-10, min(step_size, 1.0))

        else:
            # -----------------------------------------------------------------
            # ULA: unconditional update (legacy, no MH correction)
            # -----------------------------------------------------------------
            with torch.no_grad():
                for i, p in enumerate(sampled_params):
                    if p.grad is None:
                        continue

                    g = p.grad.clone()
                    g_norm = g.norm()
                    if g_norm > grad_clip:
                        g = g * (grad_clip / g_norm)

                    if update_precond:
                        v_hat[i] = beta2 * v_hat[i] + (1 - beta2) * g * g
                        v_corrected = v_hat[i] / (1 - beta2 ** (step + 1))
                        pc = 1.0 / (torch.sqrt(v_corrected) + 1e-8)
                    elif use_frozen:
                        pc = v_hat_frozen[i]
                    elif precondition and v_hat is not None:
                        v_hat[i] = beta2 * v_hat[i] + (1 - beta2) * g * g
                        v_corrected = v_hat[i] / (1 - beta2 ** (step + 1))
                        pc = 1.0 / (torch.sqrt(v_corrected) + 1e-8)
                    else:
                        pc = 1.0

                    noise = torch.randn_like(p)
                    if isinstance(pc, float):
                        p.data -= (eps / 2) * g
                        p.data += math.sqrt(eps) * noise
                    else:
                        p.data -= (eps / 2) * pc * g
                        p.data += math.sqrt(eps) * torch.sqrt(pc) * noise

        # ---- Freeze preconditioner at end of burn-in ----
        if step == burn_in - 1 and precondition and v_hat is not None:
            # Convert raw second moments → actual preconditioner (1/sqrt)
            # so that post-burn-in sampling uses the same pc as burn-in
            v_hat_frozen = []
            for i in range(len(v_hat)):
                v_corrected = v_hat[i] / (1 - beta2 ** burn_in)
                v_hat_frozen.append(
                    (1.0 / (torch.sqrt(v_corrected) + 1e-8)).clone()
                )
            if use_mala:
                # Lock step size at smoothed value from dual averaging
                step_size = math.exp(log_eps_bar)
                step_size = max(1e-10, min(step_size, 1.0))
            if verbose:
                print(f"\n  >>> Burn-in complete at step {step + 1}")
                print(f"      Preconditioner frozen")
                if use_mala:
                    acc_rate = n_accept / max(1, n_total_mh)
                    print(f"      Burn-in acceptance rate: {acc_rate:.3f}")
                    print(f"      Adapted step size: {step_size:.2e}")
                print()

        # ---- Collect sample ----
        is_past_burnin = (step >= burn_in)
        is_thin_step = ((step - burn_in) % thin == 0) if is_past_burnin else False

        if is_past_burnin and is_thin_step and sample_idx < n_samples:
            with torch.no_grad():
                # Effective log-linear coefficients
                ell = net.get_effective_loglinear()
                samples_eff_ll[sample_idx] = ell

                # Intensity params
                samples_alpha[sample_idx] = alpha.item()
                samples_gamma[sample_idx] = F.softplus(raw_gamma).item()

                # Partial effects on grid
                pe = net.get_partial_effects(n_grid=n_grid_pe, posterior_cov=None)
                for k in range(n_features):
                    samples_pe[k][sample_idx] = pe[k]["effect"]

            sample_idx += 1

        # ---- Logging ----
        elapsed = time.time() - t0
        if verbose and (step % 100 == 0 or step < 5
                        or step == burn_in
                        or step == total_steps - 1):
            phase = "burn-in" if step < burn_in else f"sample {sample_idx}/{n_samples}"
            line = (f"  {step:6d}  {neg_lp:12.2f}  {ll_val:10.2f}  "
                    f"{step_size:10.2e}")
            if use_mala:
                acc_pct = 100.0 * n_accept / max(1, n_total_mh)
                line += f"  {acc_pct:7.1f}%"
            line += f"  {elapsed:5.2f}s  [{phase}]"
            if use_mala and not accepted:
                line += " [REJ]"
            print(line)

    # ---- Restore CG tolerance and disable GPU ----
    cs.CG_RTOL = original_rtol
    cs.CG_WARM_START = False
    cs.enable_gpu(False)

    total_time = time.time() - t0_all
    acceptance_rate = n_accept / max(1, n_total_mh) if use_mala else 1.0
    if verbose:
        print(f"\n  Sampling complete: {sample_idx} samples in {total_time:.1f}s "
              f"({total_time / 60:.1f} min)")
        if use_mala:
            print(f"  Final acceptance rate: {acceptance_rate:.3f}")
            print(f"  Final step size: {step_size:.2e}")

    # ---- Compute ESS (effective sample size) via autocorrelation ----
    def _compute_ess(chain):
        """Estimate ESS from 1-D chain using initial monotone sequence."""
        n = len(chain)
        if n < 10:
            return float(n)
        chain = chain - np.mean(chain)
        var0 = np.var(chain)
        if var0 < 1e-30:
            return float(n)
        # Autocorrelation via FFT
        fft_chain = np.fft.fft(chain, n=2 * n)
        acf = np.fft.ifft(fft_chain * np.conj(fft_chain)).real[:n] / (n * var0)
        # Initial positive sequence estimator
        tau = 1.0
        for lag in range(1, n // 2):
            rho = acf[lag]
            if rho < 0.05:
                break
            tau += 2 * rho
        return max(1.0, n / tau)

    ess = {}
    basis_names = ["canopy", "impervious", "water", "fence", "elevation"]
    for k in range(eff_ll_dim):
        nm = f"r_0" if k == 0 else (
            f"z_{k}_{basis_names[k-1]}" if k <= len(basis_names)
            else f"z_{k}")
        ess[nm] = float(_compute_ess(samples_eff_ll[:sample_idx, k]))
    ess["alpha"] = float(_compute_ess(samples_alpha[:sample_idx]))
    ess["gamma"] = float(_compute_ess(samples_gamma[:sample_idx]))

    # ---- Posterior summary ----
    n_collected = sample_idx
    summary = {}
    for k in range(eff_ll_dim):
        nm = f"r_0" if k == 0 else (
            f"z_{k}_{basis_names[k-1]}" if k <= len(basis_names)
            else f"z_{k}")
        chain = samples_eff_ll[:n_collected, k]
        summary[nm] = {
            "mean": float(np.mean(chain)),
            "sd": float(np.std(chain)),
            "q025": float(np.percentile(chain, 2.5)),
            "q50": float(np.median(chain)),
            "q975": float(np.percentile(chain, 97.5)),
            "ess": ess[nm],
        }
    for nm, chain in [("alpha", samples_alpha[:n_collected]),
                      ("gamma", samples_gamma[:n_collected])]:
        summary[nm] = {
            "mean": float(np.mean(chain)),
            "sd": float(np.std(chain)),
            "q025": float(np.percentile(chain, 2.5)),
            "q50": float(np.median(chain)),
            "q975": float(np.percentile(chain, 97.5)),
            "ess": ess[nm],
        }

    # ---- Partial effect credible bands (pointwise quantiles) ----
    pe_grid = np.linspace(0.0, 1.0, n_grid_pe)
    partial_effects_summary = {}
    for k in range(n_features):
        chain_pe = samples_pe[k][:n_collected]  # (n_collected, n_grid)
        partial_effects_summary[int(k)] = {
            "grid": pe_grid.tolist(),
            "effect": np.mean(chain_pe, axis=0).tolist(),
            "se": np.std(chain_pe, axis=0).tolist(),
            "lower_95": np.percentile(chain_pe, 2.5, axis=0).tolist(),
            "upper_95": np.percentile(chain_pe, 97.5, axis=0).tolist(),
            "lower_50": np.percentile(chain_pe, 25, axis=0).tolist(),
            "upper_50": np.percentile(chain_pe, 75, axis=0).tolist(),
        }

    # ---- Save checkpoint ----
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(output_dir, "mcmc_samples.npz"),
            effective_loglinear=samples_eff_ll[:n_collected],
            alpha=samples_alpha[:n_collected],
            gamma=samples_gamma[:n_collected],
            log_posterior=np.array(log_post_trace),
            log_likelihood=np.array(ll_trace),
        )

    if verbose:
        print("\n  Posterior summary (effective log-linear):")
        print(f"  {'Param':>20}  {'Mean':>8}  {'SD':>8}  "
              f"{'2.5%':>8}  {'97.5%':>8}  {'ESS':>6}")
        print(f"  {'-' * 65}")
        for nm, s in summary.items():
            print(f"  {nm:>20}  {s['mean']:8.4f}  {s['sd']:8.4f}  "
                  f"{s['q025']:8.4f}  {s['q975']:8.4f}  {s['ess']:6.0f}")

    return {
        "samples_effective_loglinear": samples_eff_ll[:n_collected].tolist(),
        "samples_alpha": samples_alpha[:n_collected].tolist(),
        "samples_gamma": samples_gamma[:n_collected].tolist(),
        "partial_effects": partial_effects_summary,
        "log_posterior_trace": [float(x) for x in log_post_trace],
        "log_likelihood_trace": [float(x) for x in ll_trace],
        "ess": {str(k): float(v) for k, v in ess.items()},
        "summary": {str(k): {str(kk): float(vv) for kk, vv in v.items()}
                    for k, v in summary.items()},
        "n_samples": int(n_collected),
        "burn_in": int(burn_in),
        "thin": int(thin),
        "step_size": float(step_size),
        "acceptance_rate": float(acceptance_rate),
        "use_mala": bool(use_mala),
        "elapsed_time": float(total_time),
    }


# ==============================================================================
# HMC / NUTS sampler
# ==============================================================================


def _compute_ess_chain(chain):
    """Estimate ESS from 1-D chain using initial positive sequence estimator."""
    n = len(chain)
    if n < 10:
        return float(n)
    chain = chain - np.mean(chain)
    var0 = np.var(chain)
    if var0 < 1e-30:
        return float(n)
    fft_chain = np.fft.fft(chain, n=2 * n)
    acf = np.fft.ifft(fft_chain * np.conj(fft_chain)).real[:n] / (n * var0)
    tau = 1.0
    for lag in range(1, n // 2):
        rho = acf[lag]
        if rho < 0.05:
            break
        tau += 2 * rho
    return max(1.0, n / tau)


def _setup_sampling_state(
    basis_values_np, obs_counts_np, n_rows, n_cols, valid_mask_np, cell_area,
    model_path, model_state,
    source_spacing, source_from_resistance, solver, absorption,
    reg_mean, log_R_baseline, penalty_scale,
    n_knots, spline_degree, include_interactions, lambda_min,
    fix_smoothing, fix_intensity,
    cg_tol, seed, device,
):
    """Shared setup for MALA and HMC samplers: load model, build tensors."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Load checkpoint
    if model_state is not None:
        checkpoint = model_state
    elif model_path is not None:
        checkpoint = torch.load(model_path, map_location="cpu",
                                weights_only=False)
    else:
        raise ValueError("Either model_path or model_state must be provided")

    n_features = int(checkpoint.get(
        "n_features",
        basis_values_np.shape[1] if basis_values_np.ndim > 1 else 1))
    n_knots_ckpt = int(checkpoint.get("n_knots", n_knots))
    degree_ckpt = int(checkpoint.get("spline_degree", spline_degree))
    interactions_ckpt = bool(checkpoint.get("include_interactions",
                                            include_interactions))
    use_absorption = str(solver).lower() in ("global_absorption", "absorption")

    # Device
    if device == "auto":
        use_cuda = _GPU_AVAILABLE and _CUPY_AVAILABLE
        dev = torch.device("cuda" if use_cuda else "cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if not _CUPY_AVAILABLE:
            raise RuntimeError("CUDA requested but cupy not installed")
        dev = torch.device("cuda")
        use_cuda = True
    else:
        dev = torch.device("cpu")
        use_cuda = False

    # Intensity spline settings from checkpoint
    intensity_spline_ckpt = bool(checkpoint.get("intensity_spline", False))
    intensity_n_knots_ckpt = int(checkpoint.get("intensity_n_knots", 5))
    intensity_degree_ckpt = int(checkpoint.get("intensity_degree", 3))
    lambda_init_intensity_ckpt = float(checkpoint.get("lambda_init_intensity", 2.0))
    intensity_log1p_max_ckpt = float(checkpoint.get("intensity_log1p_max", 10.0))

    # Build network
    net = SplineResistanceNet(
        n_features=n_features,
        n_knots=n_knots_ckpt,
        degree=degree_ckpt,
        include_interactions=interactions_ckpt,
        lambda_init_marginal=0.0,
        lambda_init_interaction=0.0,
        lambda_min=float(lambda_min),
        intensity_spline=intensity_spline_ckpt,
        intensity_n_knots=intensity_n_knots_ckpt,
        intensity_degree=intensity_degree_ckpt,
        lambda_init_intensity=lambda_init_intensity_ckpt,
        intensity_log1p_max=intensity_log1p_max_ckpt,
    ).double().to(dev)
    net.load_state_dict(checkpoint["net"])
    net.setup_basis_matrices(np.asarray(basis_values_np, dtype=np.float64),
                             device=dev)
    net.train()

    use_intensity_spline = intensity_spline_ckpt and net.intensity_spline

    # Intensity parameters
    alpha_map = float(checkpoint["alpha"])
    gamma_map = float(checkpoint["gamma"])
    alpha = nn.Parameter(torch.tensor(alpha_map, dtype=torch.float64, device=dev))
    _raw_gamma_init = float(np.log(np.exp(gamma_map) - 1.0 + 1e-8))
    raw_gamma = nn.Parameter(torch.tensor(_raw_gamma_init, dtype=torch.float64,
                                          device=dev))
    if use_intensity_spline:
        raw_gamma.requires_grad_(False)

    # Data tensors
    basis_t = torch.tensor(np.asarray(basis_values_np, dtype=np.float64),
                           dtype=torch.float64, device=dev)
    obs_t = torch.tensor(np.asarray(obs_counts_np, dtype=np.float64),
                         dtype=torch.float64, device=dev)
    n_valid = basis_t.shape[0]

    # Circuit solver
    cs = _get_circuit_module()
    original_rtol = getattr(cs, "CG_RTOL", 1e-6)
    cs.CG_RTOL = float(cg_tol)
    if use_cuda:
        cs.enable_gpu(True)
    else:
        cs.enable_gpu(False)

    # Identify parameters to sample
    sampled_params = []
    param_names = []
    for k, p in enumerate(net.spline_coefs):
        sampled_params.append(p)
        param_names.append(f"spline_coef_{k}")
    for idx, p in enumerate(net.interaction_coefs):
        sampled_params.append(p)
        param_names.append(f"interaction_coef_{idx}")
    sampled_params.extend([net.skip.weight, net.skip.bias, net.intercept])
    param_names.extend(["skip_weight", "skip_bias", "intercept"])
    if not fix_smoothing:
        for k, p in enumerate(net.log_lambda_smooth):
            sampled_params.append(p)
            param_names.append(f"log_lambda_smooth_{k}")
        for idx, p in enumerate(net.log_lambda_interact_row):
            sampled_params.append(p)
            param_names.append(f"log_lambda_interact_row_{idx}")
        for idx, p in enumerate(net.log_lambda_interact_col):
            sampled_params.append(p)
            param_names.append(f"log_lambda_interact_col_{idx}")
        if use_intensity_spline and hasattr(net, 'log_lambda_intensity'):
            sampled_params.append(net.log_lambda_intensity)
            param_names.append("log_lambda_intensity")
    if not fix_intensity:
        sampled_params.append(alpha)
        param_names.append("alpha")
        if use_intensity_spline:
            sampled_params.append(net.intensity_coefs)
            param_names.append("intensity_coefs")
        else:
            sampled_params.append(raw_gamma)
            param_names.append("raw_gamma")

    n_total_scalar = sum(p.numel() for p in sampled_params)

    return {
        "net": net,
        "alpha": alpha,
        "raw_gamma": raw_gamma,
        "basis_t": basis_t,
        "obs_t": obs_t,
        "n_valid": n_valid,
        "n_features": n_features,
        "sampled_params": sampled_params,
        "param_names": param_names,
        "n_total_scalar": n_total_scalar,
        "use_absorption": use_absorption,
        "use_intensity_spline": use_intensity_spline,
        "use_cuda": use_cuda,
        "dev": dev,
        "cs": cs,
        "original_rtol": original_rtol,
        "checkpoint": checkpoint,
    }


def run_hmc_sampling(
    basis_values_np,
    obs_counts_np,
    n_rows,
    n_cols,
    valid_mask_np,
    cell_area,
    model_path=None,
    model_state=None,
    source_spacing=1,
    source_from_resistance=True,
    solver="global_absorption",
    absorption=0.002,
    # HMC / NUTS settings
    n_samples=1000,
    warmup=1000,
    max_treedepth=10,
    target_accept=0.80,
    init_step_size=None,
    adapt_mass_matrix=True,
    fix_smoothing=True,
    fix_intensity=False,
    # Regularization
    reg_mean=1.0,
    log_R_baseline=3.0,
    penalty_scale=5.0,
    # Architecture
    n_knots=10,
    spline_degree=3,
    include_interactions=False,
    lambda_min=0.1,
    # Misc
    grad_clip=100.0,
    cg_tol=DEFAULT_CG_TOL,
    seed=42,
    verbose=True,
    output_dir=None,
    device="auto",
):
    """
    Full Bayesian posterior sampling via No-U-Turn Sampler (NUTS).

    Uses Hamiltonian Monte Carlo with NUTS tree doubling (Hoffman & Gelman
    2014) for automatic trajectory length selection.  Step size is adapted
    via dual averaging during warmup, and an optional diagonal mass matrix
    is estimated from warmup samples.

    The sampler starts from the MAP estimate saved by run_torch_optimization
    and differentiates through the full pipeline:
        basis -> SplineResistanceNet -> R -> circuit solve -> C -> PPP loglik

    Parameters
    ----------
    n_samples : int
        Posterior samples to collect after warmup.
    warmup : int
        Warmup iterations (adaptation; discarded).
    max_treedepth : int
        Maximum NUTS tree depth (2^d leapfrog steps worst case).
    target_accept : float
        Target MH acceptance probability (0.80 recommended for NUTS).
    init_step_size : float or None
        Initial leapfrog step size (None = auto from posterior scale).
    adapt_mass_matrix : bool
        Adapt diagonal mass matrix from warmup variance.

    Returns
    -------
    dict — same structure as run_langevin_sampling plus:
        treedepth_trace : list of int (tree depth per sample)
        n_divergences : int (should be 0)
        energy_trace : list of float (Hamiltonian energy per sample)
    """
    t0_all = time.time()

    # ---- Shared setup ----
    state = _setup_sampling_state(
        basis_values_np, obs_counts_np, n_rows, n_cols, valid_mask_np,
        cell_area, model_path, model_state,
        source_spacing, source_from_resistance, solver, absorption,
        reg_mean, log_R_baseline, penalty_scale,
        n_knots, spline_degree, include_interactions, lambda_min,
        fix_smoothing, fix_intensity,
        cg_tol, seed, device,
    )
    net = state["net"]
    alpha = state["alpha"]
    raw_gamma = state["raw_gamma"]
    basis_t = state["basis_t"]
    obs_t = state["obs_t"]
    n_valid = state["n_valid"]
    n_features = state["n_features"]
    sampled_params = state["sampled_params"]
    use_absorption = state["use_absorption"]
    use_intensity_spline = state.get("use_intensity_spline", False)
    use_cuda = state["use_cuda"]
    dev = state["dev"]
    cs = state["cs"]
    original_rtol = state["original_rtol"]
    n_total_scalar = state["n_total_scalar"]

    if verbose:
        print(f"\n  NUTS sampler: {n_total_scalar} scalar parameters")
        print(f"    Warmup: {warmup}, Samples: {n_samples}")
        print(f"    Max tree depth: {max_treedepth}")
        print(f"    Target acceptance: {target_accept:.3f}")
        print(f"    Fix smoothing: {fix_smoothing}, Fix intensity: {fix_intensity}")
        print(f"    Adapt mass matrix: {adapt_mass_matrix}")

    # ---- Flatten / unflatten utilities ----
    def _flatten(params):
        return torch.cat([p.data.reshape(-1) for p in params])

    def _flatten_grad(params):
        gs = []
        for p in params:
            if p.grad is not None:
                g = p.grad.reshape(-1).clone()
            else:
                g = torch.zeros(p.numel(), dtype=torch.float64, device=dev)
            gs.append(g)
        return torch.cat(gs)

    def _unflatten(flat, params):
        offset = 0
        for p in params:
            n = p.numel()
            p.data.copy_(flat[offset:offset + n].reshape(p.shape))
            offset += n

    # ---- Potential energy (negative log-posterior) ----
    def compute_U():
        """Return U(θ) and populate .grad on sampled_params."""
        for p in sampled_params:
            if p.grad is not None:
                p.grad.zero_()

        R, log_R_raw = net.forward_with_log_R(basis_t)
        if use_absorption:
            C = _AbsorptionCircuitSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance, absorption)
        else:
            C = _CircuitSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance)

        gamma_val = F.softplus(raw_gamma)
        log1p_C = torch.log1p(C.clamp(min=0))
        if use_intensity_spline:
            log_lam = alpha + net.eval_intensity_spline(log1p_C)
        else:
            log_lam = alpha + gamma_val * log1p_C
        ll = _ppp_loglik(log_lam, obs_t, float(cell_area))

        mean_logR = log_R_raw.mean()
        pen_mean = n_valid * reg_mean * (mean_logR - log_R_baseline) ** 2
        pen_smooth = penalty_scale * net.smoothing_penalty()
        U = -ll + pen_mean + pen_smooth

        U.backward()
        return float(U.item()), float(ll.item())

    # ---- Leapfrog integrator ----
    def leapfrog(q0, p0, grad_U0, eps, inv_mass):
        """Single leapfrog trajectory of L=1 step.
        Returns q1, p1, grad_U1, U1, ll1.

        NOTE: No gradient clipping here — clipping biases the Hamiltonian
        dynamics and produces incorrect posterior samples.  If gradients
        explode, the root cause is numerical instability (fix cg_tol or
        regularisation instead)."""
        # Half-step momentum
        p_half = p0 - 0.5 * eps * grad_U0

        # Full-step position
        q1 = q0 + eps * inv_mass * p_half
        _unflatten(q1, sampled_params)

        # Gradient at new position
        U1, ll1 = compute_U()
        grad_U1 = _flatten_grad(sampled_params)

        # Half-step momentum
        p1 = p_half - 0.5 * eps * grad_U1

        return q1, p1, grad_U1, U1, ll1

    # ---- Kinetic energy ----
    def kinetic_energy(p, inv_mass):
        """K(p) = 0.5 p^T M^{-1} p."""
        return 0.5 * (p * inv_mass * p).sum().item()

    # ---- NUTS tree building (iterative, Hoffman & Gelman 2014 Algorithm 6) ----
    DIVERGENCE_THRESHOLD = 1000.0

    def _build_tree(q, p, grad_U, log_u_slice, direction, depth, eps, inv_mass,
                    H0):
        """Recursively build NUTS tree.
        Returns (q_minus, p_minus, grad_minus, q_plus, p_plus, grad_plus,
                 q_prime, grad_prime, U_prime, ll_prime, n_valid_states,
                 is_valid, alpha_sum, n_alpha, n_leapfrog).

        grad_prime is the gradient at q_prime so the caller can reuse it
        without an extra compute_U() call.

        log_u_slice: log of the slice variable (log-space to avoid overflow)."""
        if depth == 0:
            # Base case: single leapfrog step
            q_new, p_new, grad_new, U_new, ll_new = leapfrog(
                q, p, grad_U, direction * eps, inv_mass)
            H_new = U_new + kinetic_energy(p_new, inv_mass)
            delta_H = H_new - H0
            # Check divergence (include NaN/inf guard)
            divergent = (delta_H > DIVERGENCE_THRESHOLD) or \
                math.isnan(delta_H) or math.isinf(delta_H)
            # Slice check (log-space: log_u <= -(H_new - H0) = H0 - H_new)
            valid = (log_u_slice <= -delta_H) and (not divergent)
            n_valid_states = 1 if valid else 0
            # Acceptance stat for dual averaging
            log_accept = min(0.0, -delta_H)
            accept_prob = math.exp(log_accept) if not math.isnan(log_accept) else 0.0
            return (q_new, p_new, grad_new,
                    q_new, p_new, grad_new,
                    q_new, grad_new, U_new, ll_new,
                    n_valid_states, not divergent,
                    accept_prob, 1, 1)

        # Recursion: build left half
        (q_minus, p_minus, grad_minus, q_plus, p_plus, grad_plus,
         q_prime, grad_prime, U_prime, ll_prime,
         n_prime, is_valid_prime,
         alpha_prime, n_alpha_prime,
         n_lf_prime) = _build_tree(
            q, p, grad_U, log_u_slice, direction, depth - 1, eps, inv_mass, H0)

        if not is_valid_prime:
            return (q_minus, p_minus, grad_minus, q_plus, p_plus, grad_plus,
                    q_prime, grad_prime, U_prime, ll_prime,
                    n_prime, False, alpha_prime, n_alpha_prime, n_lf_prime)

        # Build right half from the appropriate end
        if direction == -1:
            (q_minus, p_minus, grad_minus, _, _, _,
             q_dprime, grad_dprime, U_dprime, ll_dprime,
             n_dprime, is_valid_dprime,
             alpha_dprime, n_alpha_dprime,
             n_lf_dprime) = _build_tree(
                q_minus, p_minus, grad_minus, log_u_slice, direction,
                depth - 1, eps, inv_mass, H0)
        else:
            (_, _, _, q_plus, p_plus, grad_plus,
             q_dprime, grad_dprime, U_dprime, ll_dprime,
             n_dprime, is_valid_dprime,
             alpha_dprime, n_alpha_dprime,
             n_lf_dprime) = _build_tree(
                q_plus, p_plus, grad_plus, log_u_slice, direction,
                depth - 1, eps, inv_mass, H0)

        # Multinomial sampling: accept q_dprime with prob n'' / (n' + n'')
        n_total = n_prime + n_dprime
        if n_dprime > 0 and n_total > 0:
            accept_dprime = n_dprime / n_total
            if np.random.uniform() < accept_dprime:
                q_prime = q_dprime
                grad_prime = grad_dprime
                U_prime = U_dprime
                ll_prime = ll_dprime

        # U-turn check
        dq = q_plus - q_minus
        is_valid = is_valid_dprime and \
            (dq.dot(p_minus) >= 0) and (dq.dot(p_plus) >= 0)

        return (q_minus, p_minus, grad_minus, q_plus, p_plus, grad_plus,
                q_prime, grad_prime, U_prime, ll_prime,
                n_total, is_valid,
                alpha_prime + alpha_dprime,
                n_alpha_prime + n_alpha_dprime,
                n_lf_prime + n_lf_dprime)

    # ---- Mass matrix (diagonal) ----
    inv_mass = torch.ones(n_total_scalar, dtype=torch.float64, device=dev)
    # During warmup we collect samples for mass matrix adaptation
    warmup_samples_for_mass = []

    # ---- CG tolerance: full precision throughout ----
    # Relaxed CG during warmup introduces gradient noise that corrupts
    # dual-averaging step-size adaptation and Hamiltonian dynamics.
    cg_tol_sampling = cg_tol

    # Enable CG warm-start for faster solves during NUTS
    cs.CG_WARM_START = True

    # ---- Initial gradient (needed before step size search) ----
    cs.CG_RTOL = cg_tol_sampling
    U_curr, ll_curr = compute_U()
    grad_curr = _flatten_grad(sampled_params)
    q_curr = _flatten(sampled_params)

    # ---- Auto step size (bisection to ~65% acceptance on single leapfrog) ----
    if init_step_size is None:
        # Start at 0.01 and halve/double to find eps giving ~65% acceptance
        test_eps = 0.01
        p_test = torch.randn(n_total_scalar, dtype=torch.float64, device=dev)
        H0_test = U_curr + kinetic_energy(p_test, inv_mass)
        _, p1_test, _, U1_test, _ = leapfrog(
            q_curr.clone(), p_test, grad_curr.clone(), test_eps, inv_mass)
        H1_test = U1_test + kinetic_energy(p1_test, inv_mass)
        log_accept_test = -(H1_test - H0_test)
        # If accept prob > 0.5, double eps; otherwise halve — up to 10 iterations
        direction_test = 1 if log_accept_test > math.log(0.65) else -1
        for _ in range(10):
            test_eps_new = test_eps * (2.0 ** direction_test)
            test_eps_new = max(1e-10, min(test_eps_new, 1.0))
            p_test2 = torch.randn(n_total_scalar, dtype=torch.float64, device=dev)
            H0_t2 = U_curr + kinetic_energy(p_test2, inv_mass)
            _, p1_t2, _, U1_t2, _ = leapfrog(
                q_curr.clone(), p_test2, grad_curr.clone(), test_eps_new, inv_mass)
            H1_t2 = U1_t2 + kinetic_energy(p1_t2, inv_mass)
            log_accept_t2 = -(H1_t2 - H0_t2)
            if (direction_test == 1 and log_accept_t2 < math.log(0.65)) or \
               (direction_test == -1 and log_accept_t2 > math.log(0.65)):
                break
            test_eps = test_eps_new
        init_step_size = max(1e-10, min(test_eps, 1.0))
        # Re-evaluate at MAP (leapfrog may have changed params)
        _unflatten(q_curr, sampled_params)
        U_curr, ll_curr = compute_U()
        grad_curr = _flatten_grad(sampled_params)
        if verbose:
            print(f"    Auto init step size: {init_step_size:.2e}")

    step_size = init_step_size

    # ---- Dual averaging state ----
    mu_da = math.log(10.0 * step_size)
    log_eps = math.log(step_size)
    log_eps_bar = 0.0
    adapt_kappa = 0.75
    adapt_gamma_da = 0.05
    adapt_t0 = 10
    H_bar = 0.0

    # ---- Storage ----
    n_grid_pe = 50
    eff_ll_dim = n_features + 1
    samples_eff_ll = np.zeros((n_samples, eff_ll_dim), dtype=np.float64)
    samples_alpha_arr = np.zeros(n_samples, dtype=np.float64)
    samples_gamma_arr = np.zeros(n_samples, dtype=np.float64)
    samples_pe = {k: np.zeros((n_samples, n_grid_pe), dtype=np.float64)
                  for k in range(n_features)}

    log_post_trace = []
    ll_trace = []
    treedepth_trace = []
    energy_trace = []
    n_divergences = 0
    n_leapfrog_total = 0
    sample_idx = 0

    # ---- Mass matrix adaptation windows (Stan-style) ----
    # Window I:   [0, init_window)    — step size adaptation only
    # Window II:  [init_window, term_start) — step size + mass matrix
    # Window III: [term_start, warmup)      — final step size adaptation
    init_window = min(75, warmup // 4)
    term_window = min(50, warmup // 4)
    term_start = warmup - term_window

    total_steps = warmup + n_samples

    if verbose:
        print(f"\n  {'Step':>6}  {'NegLogPost':>12}  {'LogLik':>10}  "
              f"{'StepSize':>10}  {'Depth':>5}  {'nLF':>5}  "
              f"{'Accept':>7}  {'Time':>6}")
        print(f"  {'-' * 75}")

    for step in range(total_steps):
        t0_step = time.time()

        is_warmup = (step < warmup)

        cs.CG_RTOL = cg_tol_sampling

        # Sample momentum
        p_curr = torch.randn(n_total_scalar, dtype=torch.float64,
                              device=dev)
        if adapt_mass_matrix:
            # p ~ N(0, M): scale by sqrt(mass) = 1/sqrt(inv_mass)
            p_curr = p_curr / torch.sqrt(inv_mass)

        # Current Hamiltonian
        H_curr = U_curr + kinetic_energy(p_curr, inv_mass)

        # Slice variable (log-space to avoid overflow when H_curr << 0)
        # log(u) where u ~ Uniform(0, exp(-H_curr))
        # = log(Uniform(0,1)) + (-H_curr) = log(Uniform(0,1)) - H_curr
        log_u_slice = math.log(np.random.uniform()) - H_curr

        # ---- NUTS tree doubling ----
        q_minus = q_curr.clone()
        q_plus = q_curr.clone()
        p_minus = p_curr.clone()
        p_plus = p_curr.clone()
        grad_minus = grad_curr.clone()
        grad_plus = grad_curr.clone()

        q_prop = q_curr.clone()
        grad_prop = grad_curr.clone()
        U_prop = U_curr
        ll_prop = ll_curr
        depth = 0
        n_valid_states = 1
        keep_going = True
        alpha_sum = 0.0
        n_alpha = 0
        n_leapfrog_step = 0
        divergent = False

        while keep_going and depth < max_treedepth:
            # Choose direction
            direction = 1 if np.random.uniform() < 0.5 else -1

            if direction == -1:
                (q_minus, p_minus, grad_minus, _, _, _,
                 q_prime, grad_q_prime, U_prime, ll_prime,
                 n_prime, is_valid,
                 alpha_sub, n_alpha_sub,
                 n_lf_sub) = _build_tree(
                    q_minus, p_minus, grad_minus, log_u_slice,
                    direction, depth, step_size, inv_mass, H_curr)
            else:
                (_, _, _, q_plus, p_plus, grad_plus,
                 q_prime, grad_q_prime, U_prime, ll_prime,
                 n_prime, is_valid,
                 alpha_sub, n_alpha_sub,
                 n_lf_sub) = _build_tree(
                    q_plus, p_plus, grad_plus, log_u_slice,
                    direction, depth, step_size, inv_mass, H_curr)

            if is_valid and n_prime > 0:
                accept_prob = min(1.0, n_prime / n_valid_states)
                if np.random.uniform() < accept_prob:
                    q_prop = q_prime
                    grad_prop = grad_q_prime
                    U_prop = U_prime
                    ll_prop = ll_prime

            n_valid_states += n_prime
            alpha_sum += alpha_sub
            n_alpha += n_alpha_sub
            n_leapfrog_step += n_lf_sub

            # U-turn check on full tree
            dq = q_plus - q_minus
            keep_going = is_valid and \
                (dq.dot(p_minus) >= 0) and (dq.dot(p_plus) >= 0)

            if not is_valid:
                divergent = True
                n_divergences += 1

            depth += 1

        # ---- Accept proposal (NUTS always accepts within tree) ----
        q_curr = q_prop.clone()
        _unflatten(q_curr, sampled_params)

        # Reuse cached U / gradient from the tree (saves one circuit solve).
        U_curr = U_prop
        ll_curr = ll_prop
        grad_curr = grad_prop if grad_prop is not None else _flatten_grad(sampled_params)

        n_leapfrog_total += n_leapfrog_step
        mean_accept = alpha_sum / max(1, n_alpha)

        # Traces
        log_post_trace.append(-U_curr)
        ll_trace.append(ll_curr)
        treedepth_trace.append(depth)
        energy_trace.append(U_curr + kinetic_energy(p_curr, inv_mass))

        # ---- Adaptation during warmup ----
        if is_warmup:
            # Dual averaging for step size
            m = step + 1
            H_bar = (1.0 - 1.0 / (m + adapt_t0)) * H_bar + \
                    (1.0 / (m + adapt_t0)) * (target_accept - mean_accept)
            log_eps = mu_da - math.sqrt(m) / adapt_gamma_da * H_bar
            eta = m ** (-adapt_kappa)
            log_eps_bar = eta * log_eps + (1.0 - eta) * log_eps_bar
            step_size = math.exp(log_eps)
            step_size = max(1e-10, min(step_size, 1.0))

            # Mass matrix adaptation (window II only)
            if adapt_mass_matrix and init_window <= step < term_start:
                warmup_samples_for_mass.append(q_curr.detach().cpu().numpy())
                # Update mass matrix at end of each doubling window
                n_mass = len(warmup_samples_for_mass)
                if n_mass >= 20 and (n_mass & (n_mass - 1) == 0):
                    # Power-of-2 checkpoints: 32, 64, 128, ...
                    stacked = np.stack(warmup_samples_for_mass)
                    var_est = np.var(stacked, axis=0)
                    var_est = np.clip(var_est, 1e-8, 1e8)
                    inv_mass = torch.tensor(
                        1.0 / var_est, dtype=torch.float64, device=dev)
                    if verbose:
                        print(f"  >>> Mass matrix updated (n={n_mass}, "
                              f"var range: [{var_est.min():.2e}, "
                              f"{var_est.max():.2e}])")

            # Final mass matrix update at transition to terminal window
            if adapt_mass_matrix and step == term_start - 1:
                if len(warmup_samples_for_mass) >= 10:
                    stacked = np.stack(warmup_samples_for_mass)
                    var_est = np.var(stacked, axis=0)
                    var_est = np.clip(var_est, 1e-8, 1e8)
                    inv_mass = torch.tensor(
                        1.0 / var_est, dtype=torch.float64, device=dev)
                    if verbose:
                        print(f"  >>> Final mass matrix "
                              f"(n={len(warmup_samples_for_mass)})")
                # Reset step size adaptation for terminal window
                mu_da = math.log(10.0 * step_size)
                log_eps = math.log(step_size)
                log_eps_bar = math.log(step_size)
                H_bar = 0.0

            # Lock step size at end of warmup
            if step == warmup - 1:
                step_size = math.exp(log_eps_bar)
                step_size = max(1e-10, min(step_size, 1.0))
                if verbose:
                    print(f"\n  >>> Warmup complete")
                    print(f"      Adapted step size: {step_size:.2e}")
                    if adapt_mass_matrix and len(warmup_samples_for_mass) > 0:
                        print(f"      Mass matrix from "
                              f"{len(warmup_samples_for_mass)} samples")
                    print(f"      Divergences during warmup: {n_divergences}")
                    print()
                # Reset divergence counter for sampling phase
                n_divergences = 0

        # ---- Collect sample (post-warmup) ----
        if not is_warmup:
            with torch.no_grad():
                ell = net.get_effective_loglinear()
                samples_eff_ll[sample_idx] = ell
                samples_alpha_arr[sample_idx] = alpha.item()
                samples_gamma_arr[sample_idx] = F.softplus(raw_gamma).item()
                pe = net.get_partial_effects(n_grid=n_grid_pe,
                                             posterior_cov=None)
                for k in range(n_features):
                    samples_pe[k][sample_idx] = pe[k]["effect"]
            sample_idx += 1

        # ---- Logging ----
        elapsed_step = time.time() - t0_step
        if verbose and (step % 100 == 0 or step < 5
                        or step == warmup or step == total_steps - 1):
            phase = "warmup" if is_warmup else \
                f"sample {sample_idx}/{n_samples}"
            div_str = " [DIV]" if divergent else ""
            print(f"  {step:6d}  {U_curr:12.2f}  {ll_curr:10.2f}  "
                  f"{step_size:10.2e}  {depth:5d}  "
                  f"{n_leapfrog_step:5d}  {mean_accept:6.3f}  "
                  f"{elapsed_step:5.1f}s  [{phase}]{div_str}")

    # ---- Cleanup ----
    cs.CG_RTOL = original_rtol
    cs.CG_WARM_START = False
    cs.enable_gpu(False)

    total_time = time.time() - t0_all
    if verbose:
        print(f"\n  Sampling complete: {sample_idx} samples in "
              f"{total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"  Post-warmup divergences: {n_divergences}")
        print(f"  Total leapfrog steps: {n_leapfrog_total}")
        print(f"  Mean leapfrog/sample: "
              f"{n_leapfrog_total / max(1, total_steps):.1f}")
        print(f"  Final step size: {step_size:.2e}")

    # ---- ESS ----
    n_collected = sample_idx
    ess = {}
    basis_names = ["canopy", "impervious", "water", "fence", "elevation"]
    for k in range(eff_ll_dim):
        nm = "r_0" if k == 0 else (
            f"z_{k}_{basis_names[k-1]}" if k <= len(basis_names)
            else f"z_{k}")
        ess[nm] = float(_compute_ess_chain(samples_eff_ll[:n_collected, k]))
    ess["alpha"] = float(_compute_ess_chain(samples_alpha_arr[:n_collected]))
    ess["gamma"] = float(_compute_ess_chain(samples_gamma_arr[:n_collected]))

    # ---- Posterior summary ----
    summary = {}
    for k in range(eff_ll_dim):
        nm = "r_0" if k == 0 else (
            f"z_{k}_{basis_names[k-1]}" if k <= len(basis_names)
            else f"z_{k}")
        chain = samples_eff_ll[:n_collected, k]
        summary[nm] = {
            "mean": float(np.mean(chain)),
            "sd": float(np.std(chain)),
            "q025": float(np.percentile(chain, 2.5)),
            "q50": float(np.median(chain)),
            "q975": float(np.percentile(chain, 97.5)),
            "ess": ess[nm],
        }
    for nm, chain in [("alpha", samples_alpha_arr[:n_collected]),
                      ("gamma", samples_gamma_arr[:n_collected])]:
        summary[nm] = {
            "mean": float(np.mean(chain)),
            "sd": float(np.std(chain)),
            "q025": float(np.percentile(chain, 2.5)),
            "q50": float(np.median(chain)),
            "q975": float(np.percentile(chain, 97.5)),
            "ess": ess[nm],
        }

    # ---- Partial effects credible bands ----
    pe_grid = np.linspace(0.0, 1.0, n_grid_pe)
    partial_effects_summary = {}
    for k in range(n_features):
        chain_pe = samples_pe[k][:n_collected]
        partial_effects_summary[int(k)] = {
            "grid": pe_grid.tolist(),
            "effect": np.mean(chain_pe, axis=0).tolist(),
            "se": np.std(chain_pe, axis=0).tolist(),
            "lower_95": np.percentile(chain_pe, 2.5, axis=0).tolist(),
            "upper_95": np.percentile(chain_pe, 97.5, axis=0).tolist(),
            "lower_50": np.percentile(chain_pe, 25, axis=0).tolist(),
            "upper_50": np.percentile(chain_pe, 75, axis=0).tolist(),
        }

    # ---- Save checkpoint ----
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(output_dir, "hmc_samples.npz"),
            effective_loglinear=samples_eff_ll[:n_collected],
            alpha=samples_alpha_arr[:n_collected],
            gamma=samples_gamma_arr[:n_collected],
            log_posterior=np.array(log_post_trace),
            log_likelihood=np.array(ll_trace),
            treedepth=np.array(treedepth_trace),
            energy=np.array(energy_trace),
        )

    if verbose:
        print("\n  Posterior summary (effective log-linear):")
        print(f"  {'Param':>20}  {'Mean':>8}  {'SD':>8}  "
              f"{'2.5%':>8}  {'97.5%':>8}  {'ESS':>6}")
        print(f"  {'-' * 65}")
        for nm, s in summary.items():
            print(f"  {nm:>20}  {s['mean']:8.4f}  {s['sd']:8.4f}  "
                  f"{s['q025']:8.4f}  {s['q975']:8.4f}  {s['ess']:6.0f}")

    return {
        "samples_effective_loglinear": samples_eff_ll[:n_collected].tolist(),
        "samples_alpha": samples_alpha_arr[:n_collected].tolist(),
        "samples_gamma": samples_gamma_arr[:n_collected].tolist(),
        "partial_effects": partial_effects_summary,
        "log_posterior_trace": [float(x) for x in log_post_trace],
        "log_likelihood_trace": [float(x) for x in ll_trace],
        "ess": {str(k): float(v) for k, v in ess.items()},
        "summary": {str(k): {str(kk): float(vv) for kk, vv in v.items()}
                    for k, v in summary.items()},
        "n_samples": int(n_collected),
        "burn_in": int(warmup),  # Compat key
        "thin": 1,               # No thinning in NUTS
        "step_size": float(step_size),
        "acceptance_rate": float(
            sum(treedepth_trace[warmup:]) / max(1, len(treedepth_trace[warmup:]))
            if len(treedepth_trace) > warmup else 0.0),
        "use_mala": False,
        "elapsed_time": float(total_time),
        # HMC-specific
        "treedepth_trace": [int(d) for d in treedepth_trace],
        "n_divergences": int(n_divergences),
        "energy_trace": [float(e) for e in energy_trace],
    }


# ==============================================================================
# ADVI: Automatic Differentiation Variational Inference
# ==============================================================================

def run_advi(
    basis_values_np,
    obs_counts_np,
    n_rows,
    n_cols,
    valid_mask_np,
    cell_area,
    model_path=None,
    model_state=None,
    source_spacing=1,
    source_from_resistance=True,
    solver="global_absorption",
    absorption=0.002,
    # ADVI settings
    n_samples=2000,
    max_iter=2000,
    lr=0.01,
    n_elbo_samples=1,
    full_rank=False,
    patience=100,
    fix_smoothing=True,
    fix_intensity=False,
    # Regularization
    reg_mean=1.0,
    log_R_baseline=3.0,
    penalty_scale=5.0,
    # Architecture
    n_knots=3,
    spline_degree=3,
    include_interactions=True,
    lambda_min=0.1,
    # Misc
    cg_tol=DEFAULT_CG_TOL,
    seed=42,
    verbose=True,
    output_dir=None,
    device="auto",
):
    """
    Automatic Differentiation Variational Inference (ADVI).

    Fits a Gaussian variational approximation q(θ) = N(μ, Σ) to the posterior
    by maximizing the ELBO via reparameterization gradients.  Supports
    mean-field (diagonal Σ) and full-rank parameterizations.

    Much faster than MCMC since each iteration requires only O(n_elbo_samples)
    forward+adjoint circuit solves (typically 1), vs. O(2^treedepth) for NUTS.

    Parameters
    ----------
    n_samples : int
        Number of posterior samples to draw from fitted q(θ) after convergence.
    max_iter : int
        Maximum ADVI optimization iterations.
    lr : float
        Adam learning rate for variational parameters (μ, log_σ or L).
    n_elbo_samples : int
        MC samples per ELBO gradient estimate (1 is standard; more reduces
        variance but costs proportionally more circuit solves).
    full_rank : bool
        If False, mean-field (diagonal Σ, 2d params).
        If True, full-rank (lower-triangular L with Σ = LL^T, d + d(d+1)/2 params).
    patience : int
        Early stopping: stop if ELBO doesn't improve for this many iterations.
    fix_smoothing : bool
        Fix smoothing parameters at MAP values (exclude from variational params).
    fix_intensity : bool
        Fix intensity parameters (alpha, gamma) at MAP values.

    Returns
    -------
    dict — same structure as run_hmc_sampling/run_langevin_sampling for
    downstream compatibility, plus ADVI-specific fields:
        elbo_trace : list of float (ELBO per iteration)
        converged : bool
        variational_mean : list (fitted μ)
        variational_std : list (fitted marginal σ)
    """
    t0_all = time.time()

    # ---- Shared setup ----
    state = _setup_sampling_state(
        basis_values_np, obs_counts_np, n_rows, n_cols, valid_mask_np,
        cell_area, model_path, model_state,
        source_spacing, source_from_resistance, solver, absorption,
        reg_mean, log_R_baseline, penalty_scale,
        n_knots, spline_degree, include_interactions, lambda_min,
        fix_smoothing, fix_intensity,
        cg_tol, seed, device,
    )
    net = state["net"]
    alpha = state["alpha"]
    raw_gamma = state["raw_gamma"]
    basis_t = state["basis_t"]
    obs_t = state["obs_t"]
    n_valid = state["n_valid"]
    n_features = state["n_features"]
    sampled_params = state["sampled_params"]
    use_absorption = state["use_absorption"]
    use_intensity_spline = state.get("use_intensity_spline", False)
    dev = state["dev"]
    cs = state["cs"]
    original_rtol = state["original_rtol"]
    n_total_scalar = state["n_total_scalar"]

    if verbose:
        mode_str = "full-rank" if full_rank else "mean-field"
        print(f"\n  ADVI ({mode_str}): {n_total_scalar} scalar parameters")
        n_var_params = (n_total_scalar + n_total_scalar * (n_total_scalar + 1) // 2
                        if full_rank else 2 * n_total_scalar)
        print(f"    Variational parameters: {n_var_params}")
        print(f"    Max iterations: {max_iter}, LR: {lr}")
        print(f"    ELBO MC samples: {n_elbo_samples}")
        print(f"    Patience: {patience}")
        print(f"    Fix smoothing: {fix_smoothing}, Fix intensity: {fix_intensity}")

    # ---- Flatten / unflatten utilities ----
    def _flatten(params):
        return torch.cat([p.data.reshape(-1) for p in params])

    def _unflatten(flat, params):
        offset = 0
        for p in params:
            n = p.numel()
            p.data.copy_(flat[offset:offset + n].reshape(p.shape))
            offset += n

    # ---- Initialize variational parameters at MAP ----
    d = n_total_scalar
    mu = nn.Parameter(_flatten(sampled_params).clone())
    # Initialize log_sigma small so q starts concentrated around MAP
    log_sigma = nn.Parameter(torch.full((d,), -3.0, dtype=torch.float64,
                                        device=dev))

    if full_rank:
        # L is lower-triangular: Σ = L L^T
        # Initialize as diagonal (= mean-field start)
        L_diag = nn.Parameter(torch.full((d,), -3.0, dtype=torch.float64,
                                         device=dev))
        L_offdiag = nn.Parameter(torch.zeros(d * (d - 1) // 2,
                                             dtype=torch.float64, device=dev))
        # Slower LR for off-diagonal (correlation structure) to improve stability
        var_params = [mu, L_diag, L_offdiag]
        optimizer = torch.optim.Adam([
            {"params": [mu, L_diag], "lr": lr},
            {"params": [L_offdiag], "lr": lr * 0.1},
        ])
    else:
        var_params = [mu, log_sigma]
        optimizer = torch.optim.Adam(var_params, lr=lr)

    # LR schedule: linear warm-up then cosine decay
    n_warmup = min(50, max_iter // 10)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=max(n_warmup, 1))
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(max_iter - n_warmup, 1), eta_min=lr * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[n_warmup])

    # ---- Helper: build L matrix for full-rank ----
    tril_idx = torch.tril_indices(d, d, offset=-1) if full_rank else None

    def _build_L():
        L = torch.zeros(d, d, dtype=torch.float64, device=dev)
        L[range(d), range(d)] = torch.exp(L_diag)  # positive diagonal
        L[tril_idx[0], tril_idx[1]] = L_offdiag
        return L

    # ---- Collect gradients for one theta sample ----
    def _collect_grad_theta():
        """
        Return grad_theta = d(-log_p)/d(theta) as a flat vector,
        by reading the .grad fields of sampled_params after backward().
        """
        parts = []
        for p in sampled_params:
            if p.grad is not None:
                parts.append(p.grad.reshape(-1).clone())
            else:
                parts.append(torch.zeros(p.numel(), dtype=torch.float64,
                                         device=dev))
        return torch.cat(parts)

    # ---- Log-posterior (populates .grad on sampled_params) ----
    def compute_log_posterior():
        """Compute log p(y, θ) = log_lik - penalty."""
        R, log_R_raw = net.forward_with_log_R(basis_t)
        if use_absorption:
            C = _AbsorptionCircuitSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance, absorption)
        else:
            C = _CircuitSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance)

        gamma_val = F.softplus(raw_gamma)
        if use_intensity_spline:
            log_lam = alpha + net.eval_intensity_spline(
                torch.log1p(C.clamp(min=0)))
        else:
            log_lam = alpha + gamma_val * torch.log1p(C.clamp(min=0))
        ll = _ppp_loglik(log_lam, obs_t, float(cell_area))

        mean_logR = log_R_raw.mean()
        pen_mean = n_valid * reg_mean * (mean_logR - log_R_baseline) ** 2
        pen_smooth = penalty_scale * net.smoothing_penalty()

        return ll - pen_mean - pen_smooth

    # ---- Manual reparameterization gradient ----
    # _unflatten uses .data.copy_() which severs the autograd graph between
    # the variational params (mu, sigma/L) and the network params.  So we
    # cannot rely on neg_elbo.backward() flowing gradients to mu/sigma.
    # Instead, we:
    #   1. Sample eps, compute theta = mu + scale(eps)
    #   2. _unflatten(theta) into network params (non-differentiable copy)
    #   3. Forward + backward on -log_p to get d(-log_p)/d(theta) via .grad
    #   4. Apply chain rule manually:
    #        d(-ELBO)/d(mu)        = E[ d(-log_p)/d(theta) ]
    #        d(-ELBO)/d(log_sigma) = E[ d(-log_p)/d(theta) * sigma * eps ] - 1
    #        (full-rank analogous via d(theta)/d(L))
    #   5. Step optimizer on variational params only.

    # ---- Optimization loop ----
    elbo_trace = []
    elbo_logp_trace = []   # E_q[log p] component
    elbo_entropy_trace = []  # H[q] component
    best_elbo = -float("inf")
    best_iter = 0
    best_mu = mu.data.clone()
    best_log_sigma = log_sigma.data.clone() if not full_rank else None
    best_L_diag = L_diag.data.clone() if full_rank else None
    best_L_offdiag = L_offdiag.data.clone() if full_rank else None

    if verbose:
        print(f"\n  {'Iter':>6}  {'ELBO':>12}  {'E[logp]':>12}  {'H[q]':>10}  "
              f"{'|g_mu|':>8}  {'|g_sig|':>8}  {'mean(sig)':>9}")
        print(f"  {'-' * 75}")

    for it in range(max_iter):
        # Zero variational param grads
        optimizer.zero_grad()

        # Accumulators for manual chain-rule gradients
        grad_mu_accum = torch.zeros(d, dtype=torch.float64, device=dev)
        logp_accum = 0.0

        if full_rank:
            grad_L_diag_accum = torch.zeros(d, dtype=torch.float64, device=dev)
            grad_L_offdiag_accum = torch.zeros_like(L_offdiag.data)
        else:
            grad_log_sigma_accum = torch.zeros(d, dtype=torch.float64,
                                               device=dev)

        for _ in range(n_elbo_samples):
            eps = torch.randn(d, dtype=torch.float64, device=dev)

            # Reparameterize: theta = mu + scale(eps)
            with torch.no_grad():
                if full_rank:
                    L_mat = _build_L()
                    theta = mu.data + L_mat.detach() @ eps
                else:
                    sigma = torch.exp(log_sigma.data)
                    theta = mu.data + sigma * eps

            # Copy theta into network params (non-differentiable)
            _unflatten(theta, sampled_params)

            # Forward + backward on log_p w.r.t. sampled_params
            for p in sampled_params:
                if p.grad is not None:
                    p.grad.zero_()
            log_p = compute_log_posterior()
            neg_log_p = -log_p
            neg_log_p.backward()

            # d(-log_p)/d(theta) from network param grads
            g_theta = _collect_grad_theta()  # d(-log_p)/d(theta)

            logp_accum += float(log_p.item())

            # Chain rule for variational params:
            # d(-ELBO)/d(mu) = E[g_theta]  (since d(theta)/d(mu) = I)
            grad_mu_accum += g_theta

            if full_rank:
                # d(theta)/d(L_diag_k) = eps_k * exp(L_diag_k) (at diagonal)
                # d(theta_i)/d(L_{i,j}) = eps_j for i > j (lower triangular)
                exp_L_diag = torch.exp(L_diag.data)
                # Diagonal: g_theta[k] * eps[k] * exp(L_diag[k])
                grad_L_diag_accum += g_theta * eps * exp_L_diag
                # Off-diagonal: for each (i,j) in lower triangle,
                # d(-ELBO)/d(L_{i,j}) = g_theta[i] * eps[j]
                g_expanded = g_theta[tril_idx[0]]  # g_theta at row indices
                e_expanded = eps[tril_idx[1]]       # eps at col indices
                grad_L_offdiag_accum += g_expanded * e_expanded
            else:
                # d(theta)/d(log_sigma) = sigma * eps
                # d(-ELBO)/d(log_sigma) = E[g_theta * sigma * eps]
                grad_log_sigma_accum += g_theta * sigma * eps

        # Average over MC samples
        inv_S = 1.0 / n_elbo_samples
        grad_mu_accum *= inv_S
        logp_mean = logp_accum * inv_S

        # Assign gradients to variational params (likelihood term)
        mu.grad = grad_mu_accum.clone()

        if full_rank:
            grad_L_diag_accum *= inv_S
            grad_L_offdiag_accum *= inv_S
            # Entropy gradient: d(-H)/d(L_diag) = -1
            L_diag.grad = grad_L_diag_accum - 1.0
            L_offdiag.grad = grad_L_offdiag_accum.clone()
        else:
            grad_log_sigma_accum *= inv_S
            # Entropy gradient: d(-H)/d(log_sigma) = -1
            log_sigma.grad = grad_log_sigma_accum - 1.0

        # Compute ELBO for logging
        if full_rank:
            entropy_val = (0.5 * d * (1.0 + math.log(2.0 * math.pi))
                           + float(L_diag.data.sum().item()))
        else:
            entropy_val = (float(log_sigma.data.sum().item())
                           + 0.5 * d * (1.0 + math.log(2.0 * math.pi)))
        elbo_val = logp_mean + entropy_val

        # Gradient clipping on variational params
        torch.nn.utils.clip_grad_norm_(var_params, 100.0)

        optimizer.step()
        scheduler.step()

        elbo_trace.append(elbo_val)
        elbo_logp_trace.append(logp_mean)
        elbo_entropy_trace.append(entropy_val)

        # Track best
        if elbo_val > best_elbo:
            best_elbo = elbo_val
            best_iter = it
            best_mu = mu.data.clone()
            if full_rank:
                best_L_diag = L_diag.data.clone()
                best_L_offdiag = L_offdiag.data.clone()
            else:
                best_log_sigma = log_sigma.data.clone()

        # Early stopping
        if it - best_iter >= patience:
            if verbose:
                print(f"\n  Early stopping at iter {it} "
                      f"(best ELBO {best_elbo:.2f} at iter {best_iter})")
            break

        # Logging
        if verbose and (it % 50 == 0 or it < 5 or it == max_iter - 1):
            delta = elbo_val - elbo_trace[-2] if len(elbo_trace) > 1 else 0.0
            g_mu_norm = float(mu.grad.norm().item()) if mu.grad is not None else 0.0
            if full_rank:
                g_sig_norm = float(L_diag.grad.norm().item()) if L_diag.grad is not None else 0.0
                mean_sig = float(torch.exp(L_diag.data).mean().item())
            else:
                g_sig_norm = float(log_sigma.grad.norm().item()) if log_sigma.grad is not None else 0.0
                mean_sig = float(torch.exp(log_sigma.data).mean().item())
            print(f"  {it:6d}  {elbo_val:12.2f}  {logp_mean:12.2f}  "
                  f"{entropy_val:10.2f}  {g_mu_norm:8.3f}  {g_sig_norm:8.3f}  "
                  f"{mean_sig:9.5f}")

    converged = (best_iter < max_iter - 1)
    n_iter_run = min(it + 1, max_iter)

    if verbose:
        print(f"\n  ADVI converged: {converged} "
              f"(best ELBO {best_elbo:.2f} at iter {best_iter}/{n_iter_run})")

    # ---- Restore best variational parameters ----
    mu.data.copy_(best_mu)
    if full_rank:
        L_diag.data.copy_(best_L_diag)
        L_offdiag.data.copy_(best_L_offdiag)
    else:
        log_sigma.data.copy_(best_log_sigma)

    # ---- Draw posterior samples from fitted q(θ) ----
    if verbose:
        print(f"\n  Drawing {n_samples} posterior samples from q(θ)...")

    n_grid_pe = 50
    eff_ll_dim = n_features + 1
    samples_eff_ll = np.zeros((n_samples, eff_ll_dim), dtype=np.float64)
    samples_alpha_arr = np.zeros(n_samples, dtype=np.float64)
    samples_gamma_arr = np.zeros(n_samples, dtype=np.float64)
    samples_pe = {k: np.zeros((n_samples, n_grid_pe), dtype=np.float64)
                  for k in range(n_features)}
    log_post_trace_samples = []

    with torch.no_grad():
        if full_rank:
            L_mat = _build_L()

        for s in range(n_samples):
            eps = torch.randn(d, dtype=torch.float64, device=dev)

            if full_rank:
                theta = mu + L_mat @ eps
            else:
                theta = mu + torch.exp(log_sigma) * eps

            _unflatten(theta, sampled_params)

            ell = net.get_effective_loglinear()
            samples_eff_ll[s] = ell
            samples_alpha_arr[s] = alpha.item()
            samples_gamma_arr[s] = F.softplus(raw_gamma).item()

            pe = net.get_partial_effects(n_grid=n_grid_pe, posterior_cov=None)
            for k in range(n_features):
                samples_pe[k][s] = pe[k]["effect"]

    # Restore MAP params for any downstream use
    _unflatten(best_mu, sampled_params)

    # ---- ESS (for VI samples these are independent, so ESS ≈ n_samples) ----
    ess = {}
    basis_names = ["canopy", "impervious", "water", "fence", "elevation"]
    for k in range(eff_ll_dim):
        nm = "r_0" if k == 0 else (
            f"z_{k}_{basis_names[k-1]}" if k <= len(basis_names)
            else f"z_{k}")
        ess[nm] = float(n_samples)  # iid samples from q
    ess["alpha"] = float(n_samples)
    ess["gamma"] = float(n_samples)

    # ---- Posterior summary ----
    summary = {}
    for k in range(eff_ll_dim):
        nm = "r_0" if k == 0 else (
            f"z_{k}_{basis_names[k-1]}" if k <= len(basis_names)
            else f"z_{k}")
        chain = samples_eff_ll[:, k]
        summary[nm] = {
            "mean": float(np.mean(chain)),
            "sd": float(np.std(chain)),
            "q025": float(np.percentile(chain, 2.5)),
            "q50": float(np.median(chain)),
            "q975": float(np.percentile(chain, 97.5)),
            "ess": float(n_samples),
        }
    for nm, chain in [("alpha", samples_alpha_arr),
                      ("gamma", samples_gamma_arr)]:
        summary[nm] = {
            "mean": float(np.mean(chain)),
            "sd": float(np.std(chain)),
            "q025": float(np.percentile(chain, 2.5)),
            "q50": float(np.median(chain)),
            "q975": float(np.percentile(chain, 97.5)),
            "ess": float(n_samples),
        }

    # ---- Partial effects credible bands ----
    pe_grid = np.linspace(0.0, 1.0, n_grid_pe)
    partial_effects_summary = {}
    for k in range(n_features):
        chain_pe = samples_pe[k]
        partial_effects_summary[int(k)] = {
            "grid": pe_grid.tolist(),
            "effect": np.mean(chain_pe, axis=0).tolist(),
            "se": np.std(chain_pe, axis=0).tolist(),
            "lower_95": np.percentile(chain_pe, 2.5, axis=0).tolist(),
            "upper_95": np.percentile(chain_pe, 97.5, axis=0).tolist(),
            "lower_50": np.percentile(chain_pe, 25, axis=0).tolist(),
            "upper_50": np.percentile(chain_pe, 75, axis=0).tolist(),
        }

    # ---- Save checkpoint ----
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_dict = {
            "effective_loglinear": samples_eff_ll,
            "alpha": samples_alpha_arr,
            "gamma": samples_gamma_arr,
            "elbo_trace": np.array(elbo_trace),
            "variational_mean": best_mu.cpu().numpy(),
        }
        if full_rank:
            save_dict["variational_L_diag"] = best_L_diag.cpu().numpy()
            save_dict["variational_L_offdiag"] = best_L_offdiag.cpu().numpy()
        else:
            save_dict["variational_log_sigma"] = best_log_sigma.cpu().numpy()
        np.savez_compressed(
            os.path.join(output_dir, "advi_samples.npz"), **save_dict)

    if verbose:
        print("\n  Posterior summary (ADVI):")
        print(f"  {'Param':>20}  {'Mean':>8}  {'SD':>8}  "
              f"{'2.5%':>8}  {'97.5%':>8}")
        print(f"  {'-' * 55}")
        for nm, s in summary.items():
            print(f"  {nm:>20}  {s['mean']:8.4f}  {s['sd']:8.4f}  "
                  f"{s['q025']:8.4f}  {s['q975']:8.4f}")

    # ---- Cleanup ----
    cs.CG_RTOL = original_rtol

    total_time = time.time() - t0_all
    if verbose:
        print(f"\n  ADVI complete: {n_iter_run} iterations + "
              f"{n_samples} samples in {total_time:.1f}s "
              f"({total_time / 60:.1f} min)")

    return {
        "samples_effective_loglinear": samples_eff_ll.tolist(),
        "samples_alpha": samples_alpha_arr.tolist(),
        "samples_gamma": samples_gamma_arr.tolist(),
        "partial_effects": partial_effects_summary,
        "log_posterior_trace": [float(x) for x in elbo_trace],
        "log_likelihood_trace": [float(x) for x in elbo_logp_trace],
        "elbo_logp_trace": [float(x) for x in elbo_logp_trace],
        "elbo_entropy_trace": [float(x) for x in elbo_entropy_trace],
        "ess": {str(k): float(v) for k, v in ess.items()},
        "summary": {str(k): {str(kk): float(vv) for kk, vv in v.items()}
                    for k, v in summary.items()},
        "n_samples": int(n_samples),
        "burn_in": int(n_iter_run),  # Compat: report optimization iters
        "thin": 1,
        "step_size": float(lr),
        "acceptance_rate": 1.0,  # Not applicable for VI
        "use_mala": False,
        "elapsed_time": float(total_time),
        # ADVI-specific
        "elbo_trace": [float(x) for x in elbo_trace],
        "converged": converged,
        "variational_mean": best_mu.cpu().numpy().tolist(),
        "variational_std": (
            torch.exp(best_log_sigma).cpu().numpy().tolist()
            if not full_rank
            else torch.exp(best_L_diag).cpu().numpy().tolist()
        ),
        "full_rank": full_rank,
        "n_iter": int(n_iter_run),
        "best_elbo": float(best_elbo),
    }
