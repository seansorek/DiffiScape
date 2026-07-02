"""Custom torch.autograd.Function subclasses for differentiable circuit-theory
solving, plus the cupy/torch tensor bridges and adjoint gradient math they
depend on.

Contains:
  - _torch_to_cupy / _cupy_to_torch: zero-copy DLPack bridges for the GPU path.
  - _softplus_clamp / _gradient_wrt_R: shared adjoint-gradient helpers.
  - _CircuitSolveFn / _AbsorptionCircuitSolveFn: global circuit solves
    (AMG-preconditioned CG, with and without absorption).
  - Bilinear interpolation helpers (_compute_interp_weights,
    _interpolate_block_grid, _interpolate_backward_fast) and the per-focal
    local adjoint (_local_adjoint_dl_dR) used by the diff_omniscape solve.
  - _DiffOmniscapeSolveFn: differentiable diff_omniscape (moving-window
    focal-pixel-as-ground) circuit solve.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import spsolve

from .constants import DEFAULT_R_MIN, DEFAULT_R_MAX, DEFAULT_CLAMP_BETA
from ._module_loaders import _get_circuit_module, _get_diff_omniscape_module

try:
    import cupy as cp
except ImportError:
    cp = None


def _torch_to_cupy(t):
    """Zero-copy torch CUDA tensor → cupy array via DLPack."""
    return cp.from_dlpack(t)


def _cupy_to_torch(a, device):
    """Zero-copy cupy array → torch tensor via DLPack."""
    return torch.from_dlpack(a).to(device)


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
