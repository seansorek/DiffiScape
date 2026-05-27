"""
04_diff_omniscape.py
Differentiable Omniscape-Style Connectivity Solver

Implements the Omniscape moving-window algorithm with adjoint-based gradient support.
For each focal pixel on a block grid, a local circuit is solved with the focal pixel
as ground (v=0) and all surrounding pixels within radius r as current sources.

Key difference from 03_circuit_solver.py (global solve):
  03_circuit_solver: boundary pixels = ground, all interior pixels = sources
  04_diff_omniscape:  focal pixel = ground, pixels within radius = sources

Forward pass (per focal pixel f):
  1. Extract (2r+1)×(2r+1) sub-window around f (clipped at grid edges)
  2. Build 4-connected Laplacian for sub-window
  3. Ground focal pixel: L_sub = L_full[non-focal, non-focal]
  4. Set sources: b_sub = 1 or 1/R_k for non-focal pixels
  5. Solve: L_sub * v_sub = b_sub (direct sparse solve, small system)
  6. C_f = Σ_{j~f} w_fj * v_sub[j]   (current flowing into grounded focal)

Backward pass (adjoint, per focal pixel):
  1. dl/dv_sub[j] = dl/dC_f * w_fj  for j in focal_neighbors, 0 elsewhere
  2. Adjoint solve: L_sub * λ_sub = dl/dv_sub
  3. dl/dw for nonfocal-nonfocal edges: -0.5*(λ_i-λ_j)*(v_i-v_j)
     dl/dw_fj for focal-nonfocal edges: (dl/dC_f - λ_j)*v_j
  4. Chain dl/dw → dl/dR → dl/dη → dl/dθ  (same as 03_circuit_solver)
  5. Accumulate contributions across all focal pixels

Grid convention: row-major flattening, k = row * n_cols + col (matches 03_circuit_solver)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time

_EPS = 1e-8

# Python-side cache for forward states.
# Avoids roundtripping forward_states through R/reticulate, which corrupts
# numpy dtypes (intp → float64) and scipy sparse matrices.
_cached_forward_states = None
_cached_R_matrix = None


def _smooth_abs(x):
    return np.sqrt(x * x + _EPS)


def _smooth_sign(x):
    return x / np.sqrt(x * x + _EPS)


# =============================================================================
# Local sub-window helpers
# =============================================================================

def _build_subwindow_edges(n_rows_sub, n_cols_sub, R_sub_flat):
    """
    Build 4-connected edge list for a sub-window.

    Returns edge_src, edge_dst (sub-local indices), edge_w (conductances).
    Both directions stored for each undirected edge.
    """
    # Horizontal edges: (r, c) -- (r, c+1)
    rows_h = np.arange(n_rows_sub)
    cols_h = np.arange(n_cols_sub - 1)
    ri, ci = np.meshgrid(rows_h, cols_h, indexing="ij")
    src_h = (ri * n_cols_sub + ci).ravel()
    dst_h = src_h + 1

    # Vertical edges: (r, c) -- (r+1, c)
    rows_v = np.arange(n_rows_sub - 1)
    cols_v = np.arange(n_cols_sub)
    ri, ci = np.meshgrid(rows_v, cols_v, indexing="ij")
    src_v = (ri * n_cols_sub + ci).ravel()
    dst_v = src_v + n_cols_sub

    # Combine (both directions)
    edge_src = np.concatenate([src_h, dst_h, src_v, dst_v])
    edge_dst = np.concatenate([dst_h, src_h, dst_v, src_v])

    w_h = 2.0 / (R_sub_flat[src_h] + R_sub_flat[dst_h])
    w_v = 2.0 / (R_sub_flat[src_v] + R_sub_flat[dst_v])
    edge_w = np.concatenate([w_h, w_h, w_v, w_v])

    return edge_src, edge_dst, edge_w


def _build_subwindow_laplacian(edge_src, edge_dst, edge_w, n_sub):
    """Build sparse CSR Laplacian from sub-window edge list."""
    L_off = sparse.coo_matrix(
        (-edge_w, (edge_src, edge_dst)),
        shape=(n_sub, n_sub)
    )
    diag_vals = np.zeros(n_sub)
    np.add.at(diag_vals, edge_src, edge_w)
    L = (L_off + sparse.diags(diag_vals)).tocsr()
    return L


# =============================================================================
# Single focal solve
# =============================================================================

def solve_local_focal(R_matrix, focal_row, focal_col, radius,
                      source_from_resistance=True):
    """
    Solve the local circuit for a single focal pixel.

    The focal pixel is grounded (v=0). All other pixels in the
    (2r+1)×(2r+1) sub-window (clipped at grid edges) are current sources.

    Parameters
    ----------
    R_matrix : ndarray (n_rows_global, n_cols_global)
        Global resistance matrix. NA values should already be replaced by 1e6
        by the caller (or are handled here via np.isnan check).
    focal_row, focal_col : int
        Grid coordinates of focal pixel.
    radius : int
        Half-width of sub-window (sub-window is (2r+1)×(2r+1)).
    source_from_resistance : bool
        If True, source strength at pixel k is 1/R_k (conductance-weighted).
        If False, uniform injection b_k = 1.

    Returns
    -------
    C_focal : float
        Total current flowing into grounded focal pixel.
    forward_state : dict
        Everything needed for the adjoint gradient computation.
    """
    n_rows_global, n_cols_global = R_matrix.shape

    # Clip sub-window to grid bounds
    r0 = max(0, focal_row - radius)
    r1 = min(n_rows_global, focal_row + radius + 1)
    c0 = max(0, focal_col - radius)
    c1 = min(n_cols_global, focal_col + radius + 1)

    n_rows_sub = r1 - r0
    n_cols_sub = c1 - c0
    n_sub = n_rows_sub * n_cols_sub

    # Extract sub-window; replace NA with high resistance
    R_sub = R_matrix[r0:r1, c0:c1].copy().astype(np.float64)
    na_mask_sub = np.isnan(R_sub)
    R_sub[na_mask_sub] = 1e6
    R_sub_flat = R_sub.ravel()

    # Focal pixel's position in sub-local coordinates
    f_row_local = focal_row - r0
    f_col_local = focal_col - c0
    f_local = f_row_local * n_cols_sub + f_col_local  # sub-local index

    # Global flat index of focal pixel
    f_global = focal_row * n_cols_global + focal_col

    # Build full sub-window edge list (sub-local indices, both directions)
    edge_src_sub, edge_dst_sub, edge_w_sub = _build_subwindow_edges(
        n_rows_sub, n_cols_sub, R_sub_flat
    )

    # Build full sub-window Laplacian
    L_full = _build_subwindow_laplacian(
        edge_src_sub, edge_dst_sub, edge_w_sub, n_sub
    )

    # Non-focal index array: maps nonfocal_index -> sub-local index
    # (skips f_local, so nonfocal index 0..f_local-1 maps to sub-local 0..f_local-1
    #  and nonfocal index f_local..n_sub-2 maps to sub-local f_local+1..n_sub-1)
    nonfocal_to_sub = np.concatenate([
        np.arange(f_local, dtype=np.intp),
        np.arange(f_local + 1, n_sub, dtype=np.intp)
    ])
    n_nonfocal = len(nonfocal_to_sub)  # = n_sub - 1

    # L_sub = L_full restricted to non-focal nodes
    # This automatically includes focal-neighbor diagonal contributions (w_fj in L[j,j])
    # because grounding the focal pixel eliminates its row/column but preserves
    # the edge weight contributions to neighboring diagonal entries.
    L_sub = L_full[np.ix_(nonfocal_to_sub, nonfocal_to_sub)].tocsc()

    # Source vector for non-focal pixels
    if source_from_resistance:
        R_nonfocal = R_sub_flat[nonfocal_to_sub]
        b_sub = 1.0 / R_nonfocal
    else:
        b_sub = np.ones(n_nonfocal, dtype=np.float64)

    # Solve: L_sub * v_sub = b_sub
    v_sub = spsolve(L_sub, b_sub)

    # Focal-neighbor edges: edges from focal to its non-focal neighbors
    # Only keep src=f_local direction (each undirected edge counted once)
    focal_src_mask = edge_src_sub == f_local
    focal_j_sub = edge_dst_sub[focal_src_mask]     # sub-local indices of focal's neighbors
    focal_w = edge_w_sub[focal_src_mask]            # w_fj conductances

    # Convert focal neighbors to nonfocal indices
    # sub-local j -> nonfocal index: j if j < f_local, else j-1
    focal_j_nonfocal = np.where(focal_j_sub < f_local,
                                 focal_j_sub,
                                 focal_j_sub - 1).astype(np.intp)

    # Current into focal: C_f = sum_j w_fj * v_sub[j_nonfocal]
    C_focal = float(np.dot(focal_w, v_sub[focal_j_nonfocal]))

    # Non-focal–non-focal edges (neither endpoint is focal)
    nonfocal_mask = (edge_src_sub != f_local) & (edge_dst_sub != f_local)
    nf_edge_src_sub = edge_src_sub[nonfocal_mask]
    nf_edge_dst_sub = edge_dst_sub[nonfocal_mask]
    nf_edge_w = edge_w_sub[nonfocal_mask]

    # Convert to nonfocal indices
    nf_edge_src_nf = np.where(nf_edge_src_sub < f_local,
                               nf_edge_src_sub,
                               nf_edge_src_sub - 1).astype(np.intp)
    nf_edge_dst_nf = np.where(nf_edge_dst_sub < f_local,
                               nf_edge_dst_sub,
                               nf_edge_dst_sub - 1).astype(np.intp)

    # Sub-window global index map: sub_global_map[k_sub] = global flat index
    row_offsets = (np.arange(n_rows_sub) + r0) * n_cols_global
    col_offsets = np.arange(n_cols_sub) + c0
    sub_global_map = (row_offsets[:, None] + col_offsets[None, :]).ravel()

    forward_state = {
        "v_sub": v_sub,
        "L_sub": L_sub,                         # (n_nonfocal, n_nonfocal) CSC
        "f_local": f_local,                     # sub-local index of focal
        "f_global": f_global,                   # global flat index of focal
        "focal_j_sub": focal_j_sub,             # sub-local indices of focal's neighbors
        "focal_j_nonfocal": focal_j_nonfocal,   # nonfocal indices of focal's neighbors
        "focal_w": focal_w,                     # conductances w_fj
        "nf_edge_src_nf": nf_edge_src_nf,       # nonfocal indices (for adjoint)
        "nf_edge_dst_nf": nf_edge_dst_nf,
        "nf_edge_src_sub": nf_edge_src_sub,     # sub-local indices (for dL/dR chain)
        "nf_edge_dst_sub": nf_edge_dst_sub,
        "nf_edge_w": nf_edge_w,
        "nonfocal_to_sub": nonfocal_to_sub,     # nonfocal index -> sub-local index
        "sub_global_map": sub_global_map,       # sub-local index -> global flat index
        "R_sub_flat": R_sub_flat,
        "na_mask_sub": na_mask_sub.ravel(),
        "n_nonfocal": n_nonfocal,
        "n_sub": n_sub,
        "source_from_resistance": source_from_resistance,
    }

    return C_focal, forward_state


# =============================================================================
# Full Omniscape forward pass
# =============================================================================

def solve_diff_omniscape(R_matrix, radius=13, block_size=5,
                          source_from_resistance=True):
    """
    Full differential Omniscape forward pass over a block grid of focal pixels.

    Iterates over focal pixels spaced every block_size rows/cols (centred offset).
    Each focal pixel gets its own local circuit solve.

    Parameters
    ----------
    R_matrix : ndarray (n_rows, n_cols)
        Resistance values. NA should be replaced by 1e6 before calling, or
        this function handles it per sub-window.
    radius : int
        Local neighbourhood radius (sub-window is (2r+1)×(2r+1)).
    block_size : int
        Spacing between focal pixels in both row and column directions.
    source_from_resistance : bool
        If True, source strength at pixel k is 1/R_k.

    Returns
    -------
    C_map : ndarray (n_rows, n_cols)
        Connectivity (current into focal) at each focal pixel; 0 elsewhere.
    forward_states : list of (focal_row, focal_col, forward_state, C_focal)
        One entry per focal pixel, for use in gradient computation.
    elapsed_seconds : float
    """
    n_rows, n_cols = R_matrix.shape
    C_map = np.zeros((n_rows, n_cols), dtype=np.float64)
    forward_states = []

    offset = block_size // 2
    focal_rows = list(range(offset, n_rows, block_size))
    focal_cols = list(range(offset, n_cols, block_size))

    t0 = time.time()
    for fr in focal_rows:
        for fc in focal_cols:
            C_f, fwd = solve_local_focal(
                R_matrix, fr, fc, radius, source_from_resistance
            )
            C_map[fr, fc] = C_f
            forward_states.append((fr, fc, fwd, C_f))

    elapsed = time.time() - t0

    # Cache forward states on the Python side so they never roundtrip through R
    global _cached_forward_states, _cached_R_matrix
    _cached_forward_states = forward_states
    _cached_R_matrix = R_matrix.copy()

    return C_map, forward_states, elapsed


# =============================================================================
# Local adjoint gradient
# =============================================================================

def _local_adjoint_gradient(forward_state, dl_dC_focal,
                             basis_values_global, R_flat_global,
                             R_min, R_max, use_softplus, beta, theta,
                             nodata_mask_flat):
    """
    Compute dl/dtheta contribution from one focal pixel's local adjoint solve.

    Parameters
    ----------
    forward_state : dict
        From solve_local_focal().
    dl_dC_focal : float
        dL/dC_f — gradient of loss w.r.t. current at this focal pixel.
    basis_values_global : ndarray (n_nodes, n_basis)
        Basis function values at all global pixels.
    R_flat_global : ndarray (n_nodes,)
        Flattened global resistance (with NA → 1e6).
    R_min, R_max, use_softplus, beta, theta, nodata_mask_flat :
        Same as gradient_wrt_resistance() in 03_circuit_solver.py.

    Returns
    -------
    dl_dtheta : ndarray (p,) where p = n_basis + 1
    """
    v_sub = np.asarray(forward_state["v_sub"], dtype=np.float64)
    L_sub = forward_state["L_sub"]
    # Index arrays may have been converted to float64 by reticulate round-trip;
    # cast them back to integer so they work as numpy indices.
    focal_j_nonfocal = np.asarray(forward_state["focal_j_nonfocal"], dtype=np.intp)
    focal_w = np.asarray(forward_state["focal_w"], dtype=np.float64)
    focal_j_sub = np.asarray(forward_state["focal_j_sub"], dtype=np.intp)
    f_local = int(forward_state["f_local"])
    nf_edge_src_nf = np.asarray(forward_state["nf_edge_src_nf"], dtype=np.intp)
    nf_edge_dst_nf = np.asarray(forward_state["nf_edge_dst_nf"], dtype=np.intp)
    nf_edge_src_sub = np.asarray(forward_state["nf_edge_src_sub"], dtype=np.intp)
    nf_edge_dst_sub = np.asarray(forward_state["nf_edge_dst_sub"], dtype=np.intp)
    nf_edge_w = np.asarray(forward_state["nf_edge_w"], dtype=np.float64)
    nonfocal_to_sub = np.asarray(forward_state["nonfocal_to_sub"], dtype=np.intp)
    sub_global_map = np.asarray(forward_state["sub_global_map"], dtype=np.intp)
    R_sub_flat = np.asarray(forward_state["R_sub_flat"], dtype=np.float64)
    na_mask_sub = np.asarray(forward_state["na_mask_sub"], dtype=bool)
    n_nonfocal = int(forward_state["n_nonfocal"])
    n_sub = int(forward_state["n_sub"])
    source_from_resistance = bool(forward_state["source_from_resistance"])

    # -------------------------------------------------------------------------
    # Step 1: Adjoint RHS — dl/dv_sub
    # dl/dv_j = dl/dC_f * w_fj  for j in focal_neighbors (nonfocal index)
    # -------------------------------------------------------------------------
    dl_dv_sub = np.zeros(n_nonfocal, dtype=np.float64)
    np.add.at(dl_dv_sub, focal_j_nonfocal, dl_dC_focal * focal_w)

    # -------------------------------------------------------------------------
    # Step 2: Adjoint solve (L_sub is symmetric)
    # L_sub * lambda_sub = dl/dv_sub
    # -------------------------------------------------------------------------
    lambda_sub = spsolve(L_sub, dl_dv_sub)

    # -------------------------------------------------------------------------
    # Step 3: dl/dw per edge
    #
    # Nonfocal–nonfocal edges (standard adjoint formula, 0.5 for both-direction storage):
    #   dl/dw_ij = -0.5 * (lambda_i - lambda_j) * (v_i - v_j)
    #
    # Focal–nonfocal edges (one direction only, direct + adjoint contributions):
    #   dl/dw_fj = (dl/dC_f - lambda_j) * v_j
    #   Direct:   dl/dC_f * v_j  (C_f = w_fj * v_j)
    #   Adjoint:  -lambda_j * v_j  (w_fj appears in L_sub diagonal of j)
    # -------------------------------------------------------------------------
    dlam_nf = lambda_sub[nf_edge_src_nf] - lambda_sub[nf_edge_dst_nf]
    dv_nf = v_sub[nf_edge_src_nf] - v_sub[nf_edge_dst_nf]
    dL_dw_nf = -0.5 * dlam_nf * dv_nf

    dL_dw_focal = (dl_dC_focal - lambda_sub[focal_j_nonfocal]) * v_sub[focal_j_nonfocal]

    # -------------------------------------------------------------------------
    # Step 4: dl/dR for each sub-window pixel
    # dw_ij / dR_k = -2 / (R_i + R_j)^2  for k in {i, j}
    # -------------------------------------------------------------------------
    dL_dR_sub = np.zeros(n_sub, dtype=np.float64)

    # From nonfocal-nonfocal edges
    if len(nf_edge_src_sub) > 0:
        denom_sq_nf = (R_sub_flat[nf_edge_src_sub] + R_sub_flat[nf_edge_dst_sub]) ** 2
        dw_dR_nf = -2.0 / denom_sq_nf
        scaled = dL_dw_nf * dw_dR_nf
        np.add.at(dL_dR_sub, nf_edge_src_sub, scaled)
        np.add.at(dL_dR_sub, nf_edge_dst_sub, scaled)

    # From focal-nonfocal edges
    if len(focal_j_sub) > 0:
        denom_sq_f = (R_sub_flat[f_local] + R_sub_flat[focal_j_sub]) ** 2
        dw_dR_f = -2.0 / denom_sq_f
        scaled_f = dL_dw_focal * dw_dR_f
        np.add.at(dL_dR_sub, f_local, np.sum(scaled_f))
        np.add.at(dL_dR_sub, focal_j_sub, scaled_f)

    # Source-from-resistance correction: b_k = 1/R_k
    # Extra term in gradient: dl/dR_k += -lambda_k / R_k^2  for source pixels (non-focal)
    if source_from_resistance and n_nonfocal > 0:
        R_nonfocal = R_sub_flat[nonfocal_to_sub]
        correction = -lambda_sub / (R_nonfocal ** 2)
        np.add.at(dL_dR_sub, nonfocal_to_sub, correction)

    # -------------------------------------------------------------------------
    # Step 5: dl/dR → dl/dη → dl/dθ  via global basis and clamping chain
    #
    # Uses global R and basis values at sub-window pixel locations.
    # -------------------------------------------------------------------------
    global_idxs = sub_global_map  # sub-local -> global flat index

    # dR/deta chain (same logic as gradient_wrt_resistance in 03_circuit_solver)
    if use_softplus:
        if theta is not None:
            theta_arr = np.asarray(theta, dtype=np.float64).ravel()
            basis_sub = basis_values_global[global_idxs]   # (n_sub, n_basis)
            eta = theta_arr[0] + basis_sub @ theta_arr[1:]
            eta = np.clip(eta, -50.0, 50.0)
            exp_eta = np.exp(eta)
        else:
            exp_eta = R_flat_global[global_idxs].copy()

        x_lower = beta * (exp_eta - R_min)
        sp_lower = np.where(
            x_lower > 20, x_lower / beta,
            np.log1p(np.exp(np.clip(x_lower, -50, 20))) / beta
        )
        R_L = R_min + sp_lower
        sig_lower = np.where(x_lower > 20, 1.0,
                             1.0 / (1.0 + np.exp(-np.clip(x_lower, -50, 20))))
        x_upper = beta * (R_max - R_L)
        sig_upper = np.where(x_upper > 20, 1.0,
                             1.0 / (1.0 + np.exp(-np.clip(x_upper, -50, 20))))
        dR_deta = sig_upper * sig_lower * exp_eta
    else:
        R_sub_global = R_flat_global[global_idxs]
        clamp_mask = (R_sub_global > R_min) & (R_sub_global < R_max)
        dR_deta = np.where(clamp_mask, R_sub_global, 0.0)

    # Zero out NA pixels and nodata pixels (their resistance is fixed)
    dR_deta[na_mask_sub] = 0.0
    if nodata_mask_flat is not None:
        dR_deta[nodata_mask_flat[global_idxs]] = 0.0

    dl_deta = dL_dR_sub * dR_deta

    # Chain dl/deta → dl/dtheta via basis functions
    basis_sub = basis_values_global[global_idxs]   # (n_sub, n_basis)
    p = basis_sub.shape[1] + 1
    dl_dtheta = np.zeros(p, dtype=np.float64)
    dl_dtheta[0] = np.sum(dl_deta)                        # intercept r_0
    dl_dtheta[1:] = basis_sub.T @ dl_deta                 # covariates z_1..z_{p-1}

    return dl_dtheta


# =============================================================================
# Full gradient aggregation
# =============================================================================

def diff_omniscape_gradient(forward_states, dl_dC_flat, R_matrix,
                             basis_values,
                             R_min=1.0, R_max=5000.0,
                             use_softplus=True, beta=5.0,
                             theta=None, nodata_mask=None):
    """
    Aggregate adjoint gradients across all focal-pixel local solves.

    Parameters
    ----------
    forward_states : list of (focal_row, focal_col, forward_state, C_focal)
        From solve_diff_omniscape().
    dl_dC_flat : ndarray (n_nodes,)
        Gradient of loss w.r.t. connectivity at each global pixel (flat, row-major).
        Only focal pixel locations will be non-zero; others are ignored.
    R_matrix : ndarray (n_rows, n_cols)
        Global resistance matrix (NA → 1e6).
    basis_values : ndarray (n_nodes, n_basis)
        Basis function values at all global pixels (flat, row-major).
    R_min, R_max : float
        Resistance clamping bounds.
    use_softplus : bool
        If True, use differentiable softplus clamp; else hard clamp.
    beta : float
        Softplus sharpness.
    theta : ndarray (p,) or None
        Resistance parameters [r_0, z_1, ..., z_{p-1}]. Required for softplus.
    nodata_mask : ndarray (bool, n_nodes) or None
        True for pixels whose resistance is fixed (NA → 1e6).

    Returns
    -------
    dl_dtheta : ndarray (p,)
        Gradient of loss w.r.t. resistance parameters.
    """
    n_rows, n_cols = R_matrix.shape
    R_flat_global = R_matrix.ravel().astype(np.float64)

    nodata_flat = None
    if nodata_mask is not None:
        nodata_flat = np.asarray(nodata_mask, dtype=bool).ravel()

    p = basis_values.shape[1] + 1
    dl_dtheta = np.zeros(p, dtype=np.float64)

    for fr, fc, fwd, C_f in forward_states:
        f_global = int(fwd["f_global"])
        dl_dC_focal = float(dl_dC_flat[f_global])

        if dl_dC_focal == 0.0:
            continue

        dl_dtheta_local = _local_adjoint_gradient(
            fwd, dl_dC_focal,
            basis_values, R_flat_global,
            R_min, R_max, use_softplus, beta, theta,
            nodata_flat
        )
        dl_dtheta += dl_dtheta_local

    return dl_dtheta


def diff_omniscape_gradient_cached(dl_dC_flat, basis_values,
                                    R_min=1.0, R_max=5000.0,
                                    use_softplus=True, beta=5.0,
                                    theta=None, nodata_mask=None):
    """
    Like diff_omniscape_gradient but uses Python-cached forward states.

    This avoids the reticulate roundtrip that corrupts numpy dtypes and
    scipy sparse matrices in the forward_state dicts.

    Must be called after solve_diff_omniscape() which populates the cache.
    """
    global _cached_forward_states, _cached_R_matrix
    if _cached_forward_states is None or _cached_R_matrix is None:
        raise RuntimeError("No cached forward states. Call solve_diff_omniscape first.")

    return diff_omniscape_gradient(
        _cached_forward_states, dl_dC_flat, _cached_R_matrix,
        basis_values,
        R_min=R_min, R_max=R_max,
        use_softplus=use_softplus, beta=beta,
        theta=theta, nodata_mask=nodata_mask
    )
