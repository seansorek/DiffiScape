"""
03_circuit_solver.py
Differentiable Circuit-Theory Connectivity Solver

Replaces Omniscape with a graph-Laplacian circuit solver that supports
analytical backpropagation via the adjoint method. Takes a resistance raster,
solves the global circuit system L(w)v = b, and returns current density.

Forward pass:
  1. Raster → edge weights (harmonic mean of adjacent resistances)
  2. Build sparse Laplacian L(w)
  3. Apply Dirichlet boundary conditions (v=0 on boundary)
  4. Solve L_int v = b via AMG-preconditioned CG
  5. Compute current density c_k = Σ_{j~k} w_kj * smooth_abs(v_k - v_j)

Backward pass (adjoint method):
  1. Given dl/dc, compute dl/dv via chain rule through current density
  2. Solve L_int λ = dl/dv (reusing cached AMG hierarchy)
  3. Assemble dl/dw from v, λ, dl/dc
  4. Chain through w → R → θ to get dl/dθ

Grid convention: row-major flattening, k = row * n_cols + col
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
import pyamg
import hashlib
import time


# =============================================================================
# GPU backend (cupy) — optional, auto-detected
# =============================================================================

_GPU_AVAILABLE = False
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    import cupyx.scipy.sparse.linalg as cp_splinalg
    _GPU_AVAILABLE = cp.cuda.is_available()
except ImportError:
    pass

_AMGX_AVAILABLE = False
_AMGX_INITIALIZED = False
try:
    import pyamgx
    _AMGX_AVAILABLE = True
except ImportError:
    pass

# Module-level GPU toggle
_use_gpu = False
_gpu_solver = "jacobi"  # "jacobi" (CuPy) or "amgx" (NVIDIA AmgX)


def enable_gpu(enable=True, solver="jacobi"):
    """Enable or disable GPU acceleration for the circuit solver.

    Parameters
    ----------
    enable : bool
    solver : str
        "jacobi" — CuPy Jacobi-preconditioned CG (default, fast for uniform absorption).
        "amgx"  — NVIDIA AmgX GPU-native AMG (better for ill-conditioned systems).
    """
    global _use_gpu, _gpu_solver
    if enable and not _GPU_AVAILABLE:
        raise RuntimeError(
            "GPU not available. Install cupy matching your CUDA version:\n"
            "  pip install cupy-cuda12x   # CUDA 12.x\n"
            "  pip install cupy-cuda11x   # CUDA 11.x"
        )
    if enable and solver == "amgx" and not _AMGX_AVAILABLE:
        raise RuntimeError(
            "AMGX not available. Install pyamgx:\n"
            "  1. Build AMGX from source: https://github.com/NVIDIA/AMGX\n"
            "  2. pip install pyamgx (with AMGX_DIR set)"
        )
    _use_gpu = enable
    _gpu_solver = solver
    if enable:
        cuda_ver = cp.cuda.runtime.runtimeGetVersion()
        print(f"  Circuit solver: GPU enabled (CUDA {cuda_ver}, solver={solver})")
    return _use_gpu


def gpu_available():
    """Check if GPU acceleration is available."""
    return _GPU_AVAILABLE


def amgx_available():
    """Check if AMGX GPU solver is available."""
    return _AMGX_AVAILABLE


def _xp():
    """Return cupy if GPU enabled, else numpy."""
    return cp if _use_gpu else np


def _to_device(arr):
    """Move numpy array to GPU if enabled."""
    if _use_gpu:
        return cp.asarray(arr)
    return np.asarray(arr)


def _to_host(arr):
    """Move array to CPU (numpy). No-op if already numpy."""
    if _use_gpu:
        return cp.asnumpy(arr)
    return np.asarray(arr)


# =============================================================================
# Module-level cache for AMG hierarchy
# =============================================================================

_cache = {
    "amg": None,
    "R_hash": None,
    "L_interior": None,
    "interior_idx": None,
    "boundary_idx": None,
}

_amgx_cache = {
    "resources": None,
    "config": None,
    "solver": None,
    "matrix": None,
    "rhs": None,
    "sol": None,
    "R_hash": None,
    "n": None,
}


def _amgx_ensure_resources():
    """Create persistent AMGX resources and config (once per session)."""
    global _AMGX_INITIALIZED
    if _amgx_cache["resources"] is not None:
        return
    if not _AMGX_INITIALIZED:
        # CuPy must initialize the CUDA context before AMGX can see the device.
        if _GPU_AVAILABLE:
            cp.cuda.Device(0).use()
            cp.zeros(1)
        pyamgx.initialize()
        _AMGX_INITIALIZED = True
    cfg = pyamgx.Config()
    cfg.create_from_dict({
        "config_version": 2,
        "solver": {
            "solver": "PCG",
            "max_iters": 2000,
            "tolerance": 1e-10,
            "monitor_residual": 1,
            "obtain_timings": 0,
            "print_solve_stats": 0,
            "preconditioner": {
                "solver": "AMG",
                "algorithm": "AGGREGATION",
                "selector": "SIZE_2",
                "max_iters": 1,
                "cycle": "V",
                "smoother": {
                    "solver": "MULTICOLOR_GS",
                    "max_iters": 2,
                    "relaxation_factor": 1.0,
                },
                "coarse_solver": "DENSE_LU_SOLVER",
            },
        },
    })
    rsc = pyamgx.Resources()
    rsc.create_simple(cfg)
    solver = pyamgx.Solver()
    solver.create(rsc, cfg)
    mat = pyamgx.Matrix()
    mat.create(rsc)
    rhs = pyamgx.Vector()
    rhs.create(rsc)
    sol = pyamgx.Vector()
    sol.create(rsc)
    _amgx_cache.update({
        "resources": rsc, "config": cfg, "solver": solver,
        "matrix": mat, "rhs": rhs, "sol": sol,
    })


def _amgx_solve(L_csr_scipy, b_np, x0=None):
    """Solve Ax=b using AMGX GPU-native AMG. Caches hierarchy across calls."""
    _amgx_ensure_resources()
    n = L_csr_scipy.shape[0]
    R_hash = _hash_array(L_csr_scipy.data)

    mat = _amgx_cache["matrix"]
    solver = _amgx_cache["solver"]
    rhs = _amgx_cache["rhs"]
    sol = _amgx_cache["sol"]

    if _amgx_cache["R_hash"] != R_hash or _amgx_cache["n"] != n:
        L_csr_scipy.sort_indices()
        mat.upload_CSR(L_csr_scipy)
        solver.setup(mat)
        _amgx_cache["R_hash"] = R_hash
        _amgx_cache["n"] = n

    rhs.upload(b_np.astype(np.float64))
    if x0 is not None:
        sol.upload(x0.astype(np.float64))
    else:
        sol.upload(np.zeros(n, dtype=np.float64))

    solver.solve(rhs, sol)

    result = np.zeros(n, dtype=np.float64)
    sol.download(result)
    return result

# Smooth absolute value parameter
_EPS = 1e-8

# CG solver tolerance — can be relaxed (e.g. 1e-6) for faster optimization
CG_RTOL = 1e-10

# CG warm-start: reuse previous voltage solution as x0.
# Enabled during NUTS sampling (parameters change slowly between leapfrog steps).
# Off by default to avoid perturbing gradient checks.
CG_WARM_START = False


def _hash_array(arr):
    """Fast hash of a numpy array for cache invalidation."""
    arr_cpu = _to_host(arr) if _use_gpu else arr
    return hashlib.md5(arr_cpu.tobytes()).hexdigest()


def _smooth_abs(x):
    """Smooth approximation to |x|: sqrt(x^2 + eps)."""
    xp = _xp()
    return xp.sqrt(x * x + _EPS)


def _smooth_sign(x):
    """Derivative of smooth_abs: x / sqrt(x^2 + eps)."""
    xp = _xp()
    return x / xp.sqrt(x * x + _EPS)


# =============================================================================
# Edge construction
# =============================================================================

def build_edge_list(R_flat, n_rows, n_cols):
    """
    Build edge list for a 4-connected grid from flattened resistance values.

    Parameters
    ----------
    R_flat : ndarray, shape (n_nodes,)
        Flattened resistance values (row-major).
    n_rows, n_cols : int
        Grid dimensions.

    Returns
    -------
    edge_src : ndarray of int
        Source node indices for each edge.
    edge_dst : ndarray of int
        Destination node indices for each edge.
    edge_w : ndarray of float
        Edge conductance weights: w_ij = 2 / (R_i + R_j).
    """
    xp = _xp()
    n_nodes = n_rows * n_cols

    # Horizontal edges: (row, col) -- (row, col+1)
    rows_h = xp.arange(n_rows)
    cols_h = xp.arange(n_cols - 1)
    row_idx_h, col_idx_h = xp.meshgrid(rows_h, cols_h, indexing="ij")
    src_h = (row_idx_h * n_cols + col_idx_h).ravel()
    dst_h = src_h + 1

    # Vertical edges: (row, col) -- (row+1, col)
    rows_v = xp.arange(n_rows - 1)
    cols_v = xp.arange(n_cols)
    row_idx_v, col_idx_v = xp.meshgrid(rows_v, cols_v, indexing="ij")
    src_v = (row_idx_v * n_cols + col_idx_v).ravel()
    dst_v = src_v + n_cols

    # Combine (undirected: store both directions)
    edge_src = xp.concatenate([src_h, dst_h, src_v, dst_v])
    edge_dst = xp.concatenate([dst_h, src_h, dst_v, src_v])

    # Conductance weights: harmonic mean
    w_h = 2.0 / (R_flat[src_h] + R_flat[dst_h])
    w_v = 2.0 / (R_flat[src_v] + R_flat[dst_v])
    edge_w = xp.concatenate([w_h, w_h, w_v, w_v])  # same weight both directions

    return edge_src, edge_dst, edge_w


# =============================================================================
# Laplacian construction
# =============================================================================

def build_laplacian(edge_src, edge_dst, edge_w, n_nodes):
    """
    Build sparse CSR Laplacian from edge list.

    L[i,j] = -w_ij  for i != j
    L[i,i] = sum_{j~i} w_ij

    Parameters
    ----------
    edge_src, edge_dst : ndarray of int
        Edge endpoint indices.
    edge_w : ndarray of float
        Edge weights (conductances).
    n_nodes : int
        Total number of nodes.

    Returns
    -------
    L : sparse CSR matrix, shape (n_nodes, n_nodes)
        Graph Laplacian (scipy on CPU, cupyx on GPU).
    """
    if _use_gpu:
        # Off-diagonal entries: -w_ij
        L_off = cp_sparse.coo_matrix(
            (-edge_w, (edge_src, edge_dst)),
            shape=(n_nodes, n_nodes)
        )
        # Diagonal entries: sum of weights per node
        diag_vals = cp.zeros(n_nodes, dtype=cp.float64)
        cp.add.at(diag_vals, edge_src, edge_w)
        L_diag = cp_sparse.diags(diag_vals, format="csr")
        L = (L_off + L_diag).tocsr()
    else:
        # Off-diagonal entries: -w_ij
        L_off = sparse.coo_matrix(
            (-edge_w, (edge_src, edge_dst)),
            shape=(n_nodes, n_nodes)
        )
        # Diagonal entries: sum of weights per node
        diag_vals = np.zeros(n_nodes)
        np.add.at(diag_vals, edge_src, edge_w)
        L_diag = sparse.diags(diag_vals, format="csr")
        L = (L_off + L_diag).tocsr()

    return L


# =============================================================================
# Boundary identification
# =============================================================================

def get_boundary_mask(n_rows, n_cols):
    """
    Return boolean mask (length n_nodes) that is True for boundary pixels
    (first/last row, first/last column).
    """
    n_nodes = n_rows * n_cols
    xp = _xp()
    mask = xp.zeros(n_nodes, dtype=bool)

    # First and last row
    mask[:n_cols] = True
    mask[-n_cols:] = True

    # First and last column (vectorized)
    rows = xp.arange(n_rows)
    mask[rows * n_cols] = True
    mask[rows * n_cols + n_cols - 1] = True

    return mask


def get_source_lattice_mask(n_rows, n_cols, spacing, interior_mask):
    """
    Return boolean mask (length n_nodes) that is True for interior pixels
    lying on a regular lattice with the given spacing.

    The lattice is centred: the first source row/col is at offset = spacing // 2,
    then every `spacing` pixels.  Only interior (non-boundary) pixels are
    included, so any lattice point that falls on the perimeter is excluded.

    Parameters
    ----------
    n_rows, n_cols : int
        Grid dimensions.
    spacing : int
        Distance (in pixels) between adjacent lattice points in both the
        row and column directions.  spacing=1 places a source at every
        interior pixel (equivalent to the original uniform injection).
    interior_mask : ndarray, bool, shape (n_nodes,)
        True for interior (non-boundary) pixels.

    Returns
    -------
    mask : ndarray, bool, shape (n_nodes,)
        True at lattice source pixels.
    """
    n_nodes = n_rows * n_cols
    xp = _xp()
    mask = xp.zeros(n_nodes, dtype=bool)
    offset = spacing // 2

    r_idx = xp.arange(offset, n_rows, spacing)
    c_idx = xp.arange(offset, n_cols, spacing)
    rr, cc = xp.meshgrid(r_idx, c_idx, indexing='ij')
    mask[(rr * n_cols + cc).ravel()] = True

    # Intersect with interior — boundary pixels must never inject
    mask &= interior_mask
    return mask


# =============================================================================
# Absorption-based solve (no boundary grounding)
# =============================================================================

def solve_circuit_absorption(R_matrix, absorption=0.01, source_spacing=1,
                             source_from_resistance=False, rebuild_amg=None,
                             output="both"):
    """
    Forward pass with distributed absorption instead of boundary grounding.

    Solves (L + α·I) v = b where α > 0 makes the system positive definite
    without needing Dirichlet boundary conditions. Current leaks to a
    distributed ground at every pixel, eliminating corner/boundary artifacts
    while keeping a single global solve.

    The absorption parameter α controls the spatial scale of current flow:
    larger α → more local (current decays faster with distance).
    Approximately, the effective radius ~ 1/√(α · R_typical).

    Parameters
    ----------
    R_matrix : ndarray, shape (n_rows, n_cols)
        Resistance values. Must be > 0 everywhere.
    absorption : float
        Absorption/leakage rate. Added to the diagonal of the Laplacian.
        Typical values: 0.001–0.1. Higher = more localized current.
    source_spacing : int
        Source lattice spacing (same as solve_circuit).
    source_from_resistance : bool
        If True, source strength = 1/R (matches Omniscape convention).
    rebuild_amg : bool or None
        If None, auto-detect from cache.
    output : str
        Which quantities to include in the returned dict.
        ``"both"`` (default) — include ``v`` and ``current_density*``.
        ``"voltage"`` — include only ``v``; skip current density computation.
        ``"current"`` — include only ``current_density*``; omit ``v``.

    Returns
    -------
    result : dict (same keys as solve_circuit, plus 'absorption')
    """
    # Absorption solver injects at ALL pixels — sparse sources create grid
    # artifacts with no speed benefit (same matrix, same single CG solve).
    if source_spacing > 1:
        import warnings
        warnings.warn(
            f"source_spacing={source_spacing} ignored for absorption solver "
            f"(forced to 1 to avoid grid artifacts)"
        )
        source_spacing = 1

    n_rows, n_cols = R_matrix.shape
    n_nodes = n_rows * n_cols
    xp = _xp()

    R_flat = (xp.asarray(R_matrix).ravel().astype(xp.float64)
              if _use_gpu else R_matrix.ravel().astype(np.float64))

    # Build edges and Laplacian
    edge_src, edge_dst, edge_w = build_edge_list(R_flat, n_rows, n_cols)
    L = build_laplacian(edge_src, edge_dst, edge_w, n_nodes)

    # Add absorption: L' = L + α·I (positive definite, no boundary grounding)
    if _use_gpu:
        L_abs = L + absorption * cp_sparse.eye(n_nodes, dtype=cp.float64, format="csr")
    else:
        L_abs = L + absorption * sparse.eye(n_nodes, dtype=np.float64, format="csr")

    # All pixels are "interior" (no boundary grounding)
    all_mask = xp.ones(n_nodes, dtype=bool)

    # Source lattice — uses all_mask since there's no boundary to exclude
    if source_spacing > 1:
        source_mask = get_source_lattice_mask(n_rows, n_cols, source_spacing, all_mask)
        b = source_mask.astype(xp.float64)
    else:
        source_mask = all_mask.copy()
        b = xp.ones(n_nodes, dtype=xp.float64)

    if source_from_resistance:
        R_src = R_flat.copy()
        R_src[R_src < 1e-8] = 1e-8
        b = b / R_src

    # Solve
    if _use_gpu and _gpu_solver == "amgx":
        # AMGX path: GPU-native AMG-preconditioned solve
        # AMGX operates on scipy CSR (host), uploads to GPU internally
        L_csr_gpu = L_abs.tocsr()
        L_scipy = sparse.csr_matrix(
            (cp.asnumpy(L_csr_gpu.data),
             cp.asnumpy(L_csr_gpu.indices),
             cp.asnumpy(L_csr_gpu.indptr)),
            shape=L_abs.shape,
        )
        b_np = _to_host(b)
        t0 = time.time()
        v_np = _amgx_solve(L_scipy, b_np)
        solve_time = time.time() - t0
        v = _to_device(v_np)

        _cache["amg"] = None
        _cache["R_hash"] = _hash_array(R_flat)
        _cache["L_interior"] = L_scipy  # keep scipy CSR for adjoint reuse
        _cache["interior_idx"] = xp.arange(n_nodes)
        _cache["boundary_idx"] = xp.array([], dtype=int)
        _cache["v_interior_gpu"] = v

    elif _use_gpu:
        # CuPy Jacobi path
        t0 = time.time()
        L_csr = L_abs.tocsr()

        diag_L = L_csr.diagonal()
        diag_inv = 1.0 / cp.maximum(diag_L, 1e-14)
        M = cp_splinalg.LinearOperator(
            shape=L_csr.shape,
            matvec=lambda x: diag_inv * x,
            dtype=cp.float64,
        )

        x0 = _cache.get("v_interior_gpu")
        if x0 is not None and x0.shape == b.shape:
            pass
        else:
            x0 = None

        v, info = cp_splinalg.cg(L_csr, b, x0=x0, M=M, rtol=CG_RTOL, maxiter=2000)
        solve_time = time.time() - t0

        if info != 0:
            v, info = cp_splinalg.cg(L_csr, b, M=M, rtol=CG_RTOL, maxiter=3000)
            if info != 0:
                raise RuntimeError(f"GPU CG did not converge (info={info})")

        _cache["amg"] = None
        _cache["R_hash"] = _hash_array(R_flat)
        _cache["L_interior"] = L_csr
        _cache["interior_idx"] = xp.arange(n_nodes)
        _cache["boundary_idx"] = xp.array([], dtype=int)
        _cache["diag_inv"] = diag_inv
        _cache["v_interior_gpu"] = v

    else:
        R_hash = _hash_array(R_flat)
        if rebuild_amg is None:
            rebuild_amg = (R_hash != _cache["R_hash"])

        if rebuild_amg or _cache["amg"] is None:
            t0 = time.time()
            amg = pyamg.smoothed_aggregation_solver(L_abs.tocsr())
            _cache["amg"] = amg
            _cache["R_hash"] = R_hash
            _cache["L_interior"] = L_abs
            _cache["interior_idx"] = np.arange(n_nodes)
            _cache["boundary_idx"] = np.array([], dtype=int)
        else:
            amg = _cache["amg"]

        M = amg.aspreconditioner()
        # Warm-start CG from previous voltage solution (halves iterations
        # when resistance changes slowly, e.g. during NUTS leapfrog steps)
        x0 = None
        if CG_WARM_START:
            x0 = _cache.get("v_interior_cpu")
            if x0 is not None and x0.shape != b.shape:
                x0 = None
        t0 = time.time()
        v, info = cg(L_abs, b, x0=x0, M=M, rtol=CG_RTOL, maxiter=1000)
        solve_time = time.time() - t0

        if info != 0:
            # Retry without warm start in case it misled CG
            v, info = cg(L_abs, b, M=M, rtol=CG_RTOL, maxiter=2000)
            if info != 0:
                raise RuntimeError(f"CG did not converge (info={info})")
        if CG_WARM_START:
            _cache["v_interior_cpu"] = v.copy()

    result = {
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_w": edge_w,
        "R_flat": R_flat,
        "interior_mask": all_mask,
        "source_mask": source_mask,
        "source_from_resistance": source_from_resistance,
        "interior_idx": xp.arange(n_nodes),
        "boundary_idx": xp.array([], dtype=int),
        "absorption": absorption,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "solve_time": solve_time,
    }
    if output in ("voltage", "both"):
        result["v"] = v
    if output in ("current", "both"):
        current_density = _compute_current_density(v, edge_src, edge_dst, edge_w, n_nodes)
        result["current_density"]    = current_density
        result["current_density_2d"] = current_density.reshape(n_rows, n_cols)
    return result


# =============================================================================
# Forward solve
# =============================================================================

def solve_circuit(R_matrix, rebuild_amg=None, source_spacing=1,
                  source_from_resistance=False, output="both"):
    """
    Full forward pass: resistance matrix → voltage and current density.

    Parameters
    ----------
    R_matrix : ndarray, shape (n_rows, n_cols)
        Resistance values. Must be > 0 everywhere.
    rebuild_amg : bool or None
        If None, auto-detect from cache. If True, force rebuild.
    source_spacing : int
        Spacing (in pixels) between lattice source points.  Default 1
        injects current at every interior pixel (original behaviour).
        Values > 1 place sources on a centred sub-grid spaced every
        ``source_spacing`` pixels in both row and column directions.
    source_from_resistance : bool
        If True, source strength at pixel k is 1/R_k (conductance-weighted),
        matching Omniscape's ``source_from_resistance = true`` mode.
        If False (default), uniform injection b_k = 1.0 at every source pixel.
        When True, the source vector b depends on R, introducing an extra
        adjoint term ∂b/∂R in the gradient computation.
    output : str
        Which quantities to include in the returned dict.
        ``"both"`` (default) — include ``v`` and ``current_density*``.
        ``"voltage"`` — include only ``v``; skip current density computation.
        ``"current"`` — include only ``current_density*``; omit ``v``.
        Auxiliary fields (``edge_src``, ``edge_dst``, ``edge_w``, ``R_flat``,
        ``interior_mask``, etc.) are always returned as they are required by
        gradient/adjoint functions.

    Returns
    -------
    result : dict with keys:
        v : ndarray (n_nodes,) — voltage at each pixel [when output != "current"]
        current_density : ndarray (n_nodes,) — current density [when output != "voltage"]
        current_density_2d : ndarray (n_rows, n_cols) — 2D current density [when output != "voltage"]
        edge_src, edge_dst, edge_w : edge list arrays
        interior_mask : boolean mask
        source_mask : boolean mask (n_nodes,) — True at lattice source pixels
        source_from_resistance : bool — whether 1/R sources were used
        n_rows, n_cols : grid dimensions
        solve_time : float — seconds for the linear solve
    """
    n_rows, n_cols = R_matrix.shape
    n_nodes = n_rows * n_cols
    xp = _xp()
    R_flat = xp.asarray(R_matrix).ravel().astype(xp.float64) if _use_gpu else R_matrix.ravel().astype(np.float64)

    # Build edges and Laplacian
    edge_src, edge_dst, edge_w = build_edge_list(R_flat, n_rows, n_cols)
    L = build_laplacian(edge_src, edge_dst, edge_w, n_nodes)

    # Boundary conditions
    boundary_mask = get_boundary_mask(n_rows, n_cols)
    interior_mask = ~boundary_mask
    interior_idx = xp.where(interior_mask)[0]
    boundary_idx = xp.where(boundary_mask)[0]
    n_interior = len(interior_idx)

    # Slice Laplacian to interior nodes
    if _use_gpu:
        # cupy sparse indexing
        L_interior = L[interior_idx][:, interior_idx]
    else:
        L_interior = L[np.ix_(interior_idx, interior_idx)]

    # Source injection — lattice or uniform
    if source_spacing > 1:
        source_mask = get_source_lattice_mask(
            n_rows, n_cols, source_spacing, interior_mask
        )
        b = source_mask[interior_idx].astype(xp.float64)
    else:
        source_mask = interior_mask.copy()
        b = xp.ones(n_interior, dtype=xp.float64)

    # Conductance-weighted sources: b_k = 1/R_k (matches Omniscape)
    if source_from_resistance:
        R_interior = R_flat[interior_idx]
        b = b / R_interior

    # Solve
    if _use_gpu and _gpu_solver == "amgx":
        # AMGX path: convert to scipy CSR, solve on GPU via AmgX
        L_int_csr = L_interior.tocsr()
        L_scipy = sparse.csr_matrix(
            (cp.asnumpy(L_int_csr.data),
             cp.asnumpy(L_int_csr.indices),
             cp.asnumpy(L_int_csr.indptr)),
            shape=L_interior.shape,
        )
        b_np = _to_host(b)
        t0 = time.time()
        v_int_np = _amgx_solve(L_scipy, b_np)
        solve_time = time.time() - t0
        v_interior = _to_device(v_int_np)

        _cache["amg"] = None
        _cache["R_hash"] = _hash_array(R_flat)
        _cache["L_interior"] = L_scipy
        _cache["interior_idx"] = interior_idx
        _cache["boundary_idx"] = boundary_idx
        _cache["v_interior_gpu"] = v_interior

    elif _use_gpu:
        # GPU path: Jacobi-preconditioned CG via cupy (warm-started)
        t0 = time.time()
        L_int_csr = L_interior.tocsr()

        # Jacobi (diagonal) preconditioner: M = diag(L)^{-1}
        diag_L = L_int_csr.diagonal()
        diag_inv = 1.0 / cp.maximum(diag_L, 1e-14)
        M = cp_splinalg.LinearOperator(
            shape=L_int_csr.shape,
            matvec=lambda x: diag_inv * x,
            dtype=cp.float64,
        )

        # Warm-start: reuse previous solution as initial guess if same grid shape
        x0 = None
        prev_v = _cache.get("v_interior_gpu")
        if prev_v is not None and prev_v.shape == b.shape:
            x0 = prev_v

        v_interior, info = cp_splinalg.cg(L_int_csr, b, x0=x0, M=M, rtol=CG_RTOL, maxiter=2000)
        solve_time = time.time() - t0

        if info != 0:
            # fallback: restart from zero (in case warm start caused issues)
            v_interior, info = cp_splinalg.cg(L_int_csr, b, M=M, rtol=CG_RTOL, maxiter=3000)
            if info != 0:
                raise RuntimeError(f"GPU CG did not converge (info={info})")

        # Cache GPU objects for adjoint reuse
        _cache["amg"] = None  # no AMG on GPU
        _cache["R_hash"] = _hash_array(R_flat)
        _cache["L_interior"] = L_int_csr
        _cache["interior_idx"] = interior_idx
        _cache["boundary_idx"] = boundary_idx
        _cache["diag_inv"] = diag_inv  # Jacobi preconditioner for adjoint reuse
        _cache["v_interior_gpu"] = v_interior  # warm-start cache

    else:
        # CPU path: AMG-preconditioned CG via pyamg (original)
        R_hash = _hash_array(R_flat)
        if rebuild_amg is None:
            rebuild_amg = (R_hash != _cache["R_hash"])

        if rebuild_amg or _cache["amg"] is None:
            t0 = time.time()
            amg = pyamg.smoothed_aggregation_solver(L_interior.tocsr())
            _cache["amg"] = amg
            _cache["R_hash"] = R_hash
            _cache["L_interior"] = L_interior
            _cache["interior_idx"] = interior_idx
            _cache["boundary_idx"] = boundary_idx
        else:
            amg = _cache["amg"]

        M = amg.aspreconditioner()
        t0 = time.time()
        v_interior, info = cg(L_interior, b, M=M, rtol=CG_RTOL, maxiter=1000)
        solve_time = time.time() - t0

        if info != 0:
            raise RuntimeError(f"CG did not converge (info={info})")

    # Reconstruct full voltage vector (boundary = 0)
    v = xp.zeros(n_nodes, dtype=xp.float64)
    v[interior_idx] = v_interior

    result = {
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_w": edge_w,
        "R_flat": R_flat,
        "interior_mask": interior_mask,
        "source_mask": source_mask,
        "source_from_resistance": source_from_resistance,
        "interior_idx": interior_idx,
        "boundary_idx": boundary_idx,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "solve_time": solve_time,
    }
    if output in ("voltage", "both"):
        result["v"] = v
    if output in ("current", "both"):
        current_density = _compute_current_density(v, edge_src, edge_dst, edge_w, n_nodes)
        result["current_density"]    = current_density
        result["current_density_2d"] = current_density.reshape(n_rows, n_cols)
    return result


def _compute_current_density(v, edge_src, edge_dst, edge_w, n_nodes):
    """
    Current density at node k: c_k = Σ_{j~k} w_kj * smooth_abs(v_k - v_j)

    Uses smooth absolute value for differentiability.
    Each undirected edge is stored twice (i→j and j→i), so we only accumulate
    at the source node (which sees each neighbor exactly once across both
    directions). Division by 2 is NOT needed because we sum over directed
    edges, and each undirected edge contributes to c_i and c_j separately.
    Actually, since we store both directions, node k accumulates:
      from (k→j): w_kj * smooth_abs(v_k - v_j)
      from (j→k): this goes to node j, not k
    So each directed edge (src=k, dst=j) contributes to c_k. With both
    directions stored, node k sees each neighbor exactly once as source.
    But we also have the reverse: node k appears as dst in (j→k).
    
    To avoid double-counting: only accumulate at src for each directed edge.
    Since we stored both directions, each undirected {i,j} appears as both
    (i,j) and (j,i). Accumulating at src for (i,j) gives contribution to c_i,
    and (j,i) gives contribution to c_j. This is correct — each node k gets
    one contribution per neighbor.
    """
    dv = v[edge_src] - v[edge_dst]
    contributions = edge_w * _smooth_abs(dv)

    xp = _xp()
    current_density = xp.zeros(n_nodes, dtype=xp.float64)
    if _use_gpu:
        cp.add.at(current_density, edge_src, contributions)
    else:
        np.add.at(current_density, edge_src, contributions)

    return current_density


# =============================================================================
# Adjoint solve
# =============================================================================

def adjoint_solve(dl_dv_interior, use_cache=True):
    """
    Solve the adjoint system L_int λ = dl_dv_interior using cached AMG.

    Parameters
    ----------
    dl_dv_interior : ndarray, shape (n_interior,)
        RHS of the adjoint system (gradient of loss w.r.t. interior voltages).
    use_cache : bool
        If True, reuse cached AMG hierarchy and L_interior.

    Returns
    -------
    lambda_full : ndarray, shape (n_nodes,)
        Adjoint variable at all nodes (zero at boundary).
    """
    if use_cache:
        L_interior = _cache["L_interior"]
        interior_idx = _cache["interior_idx"]
    else:
        raise ValueError("No cached state. Run solve_circuit() first.")

    xp = _xp()

    if _use_gpu and _gpu_solver == "amgx":
        # AMGX path: reuses cached AMG hierarchy for adjoint solve
        dl_dv_np = _to_host(dl_dv_interior)
        lam_np = _amgx_solve(L_interior, dl_dv_np)
        lam_interior = _to_device(lam_np)
    elif _use_gpu:
        # GPU path: Jacobi-preconditioned CG
        diag_inv = _cache.get("diag_inv")
        if diag_inv is None:
            diag_L = L_interior.diagonal()
            diag_inv = 1.0 / cp.maximum(diag_L, 1e-14)

        M = cp_splinalg.LinearOperator(
            shape=L_interior.shape,
            matvec=lambda x: diag_inv * x,
            dtype=cp.float64,
        )
        lam_interior, info = cp_splinalg.cg(
            L_interior, _to_device(dl_dv_interior), M=M, rtol=CG_RTOL, maxiter=2000
        )
        if info != 0:
            raise RuntimeError(f"Adjoint CG did not converge (info={info})")
    else:
        # CPU path: AMG-preconditioned CG
        amg = _cache["amg"]
        if amg is None:
            raise ValueError("No cached AMG hierarchy. Run solve_circuit() first.")
        M = amg.aspreconditioner()
        lam_interior, info = cg(L_interior, dl_dv_interior, M=M, rtol=CG_RTOL, maxiter=1000)
        if info != 0:
            raise RuntimeError(f"Adjoint CG did not converge (info={info})")

    n_nodes = len(interior_idx) + len(_cache["boundary_idx"])
    lambda_full = xp.zeros(n_nodes, dtype=xp.float64)
    lambda_full[interior_idx] = lam_interior

    return lambda_full


# =============================================================================
# Adjoint RHS: dl/dc → dl/dv
# =============================================================================

def compute_adjoint_rhs(v, dl_dc, edge_src, edge_dst, edge_w, n_nodes, interior_mask):
    """
    Compute ∂L/∂v_m from ∂L/∂c_k through the current density formula.

    c_k = Σ_{j~k} w_kj * smooth_abs(v_k - v_j)

    ∂c_k/∂v_k = Σ_{j~k} w_kj * smooth_sign(v_k - v_j)     [for node k]
    ∂c_k/∂v_j = -w_kj * smooth_sign(v_k - v_j)              [for neighbor j of k]

    But with directed edges (src→dst stored both ways), for edge (src, dst):
      c_src depends on v_src and v_dst:
        ∂c_src/∂v_src += w * smooth_sign(v_src - v_dst)
        ∂c_src/∂v_dst += -w * smooth_sign(v_src - v_dst)

    So: ∂L/∂v_m = Σ_k (∂L/∂c_k) * (∂c_k/∂v_m)

    For each directed edge (s, d):
      ∂L/∂v_s += ∂L/∂c_s * w_sd * smooth_sign(v_s - v_d)
      ∂L/∂v_d += ∂L/∂c_s * (-w_sd) * smooth_sign(v_s - v_d)

    Parameters
    ----------
    v : ndarray (n_nodes,)
    dl_dc : ndarray (n_nodes,)  — ∂L/∂c_k for each node
    edge_src, edge_dst, edge_w : edge arrays
    n_nodes : int
    interior_mask : boolean mask

    Returns
    -------
    dl_dv_interior : ndarray (n_interior,) — RHS for adjoint system
    """
    dv = v[edge_src] - v[edge_dst]
    s_dv = _smooth_sign(dv)

    xp = _xp()
    dl_dv = xp.zeros(n_nodes, dtype=xp.float64)
    contrib = dl_dc[edge_src] * edge_w * s_dv
    if _use_gpu:
        cp.add.at(dl_dv, edge_src, contrib)
        cp.add.at(dl_dv, edge_dst, -contrib)
    else:
        np.add.at(dl_dv, edge_src, contrib)
        np.add.at(dl_dv, edge_dst, -contrib)

    # Return only interior nodes
    interior_idx = xp.where(interior_mask)[0]
    return dl_dv[interior_idx]


# =============================================================================
# Gradient assembly: dl/dw → dl/dR → dl/dtheta
# =============================================================================

def compute_gradient_wrt_weights(v, adjoint_lambda, dl_dc, 
                                  edge_src, edge_dst, edge_w, n_nodes):
    """
    Compute ∂L/∂w_ij for each edge.

    The loss L depends on w through two paths:
    1. Through current density: c_k = Σ_{j~k} w_kj * smooth_abs(v_k - v_j)
       Direct contribution: ∂c_k/∂w_kj = smooth_abs(v_k - v_j)
    2. Through voltage (via Laplacian): L v = b
       Adjoint contribution: -λ^T (∂L_mat/∂w_ij) v

    For edge (i,j), ∂L_mat/∂w_ij changes:
      L[i,j] by -1, L[j,i] by -1, L[i,i] by +1, L[j,j] by +1
    So: -λ^T (∂L_mat/∂w) v = -(λ_i - λ_j)(v_i - v_j)  per undirected edge

    But we store directed edges. For directed edge (s=i, d=j):
    Total ∂L/∂w for this directed edge = smooth_abs(v_s - v_d) * dl_dc[s]
                                         - (λ_s - λ_d) * (v_s - v_d)

    Since each undirected edge {i,j} is stored as both (i,j) and (j,i),
    the undirected gradient is the sum of both directed halves. But
    for the chain rule to resistance, we accumulate per-directed-edge,
    so we effectively attribute half the adjoint contribution to each
    direction. This works because the w values are shared.

    Returns
    -------
    dl_dw : ndarray, shape (n_edges_directed,) — per directed edge
    """
    dv = v[edge_src] - v[edge_dst]
    d_lam = adjoint_lambda[edge_src] - adjoint_lambda[edge_dst]

    # Direct path through current density
    direct = _smooth_abs(dv) * dl_dc[edge_src]

    # Adjoint path through voltage
    adjoint_term = -0.5 * d_lam * dv

    dl_dw = direct + adjoint_term
    return dl_dw


def gradient_wrt_resistance(v, adjoint_lambda, dl_dc,
                             edge_src, edge_dst, edge_w,
                             R_flat, basis_values, n_nodes,
                             R_min=1.0, R_max=5000.0,
                             use_softplus=True, beta=5.0,
                             theta=None, nodata_mask=None,
                             source_from_resistance=False,
                             source_mask=None):
    """
    Full chain rule: dl/dw → dl/dR → dl/deta → dl/dtheta.

    When source_from_resistance=True the source vector b depends on R:
        b_k = source_mask_k / R_k
    This introduces an extra adjoint contribution to dl/dR:
        (dl/dR_k)_source = -λ_k · source_mask_k / R_k²
    where λ_k is the adjoint variable (from L λ = dl/dv).

    Parameters
    ----------
    v : ndarray (n_nodes,)
    adjoint_lambda : ndarray (n_nodes,)
    dl_dc : ndarray (n_nodes,)
    edge_src, edge_dst, edge_w : edge arrays
    R_flat : ndarray (n_nodes,) — resistance values (already clamped/softplus'd)
    basis_values : ndarray (n_nodes, p-1) — covariate matrix [canopy, imp, water, fence]
        Each column corresponds to a basis function. The intercept (r_0) is implicit.
    n_nodes : int
    R_min, R_max : float — clamping bounds for resistance
    use_softplus : bool — if True, use differentiable softplus clamp;
                          if False, use hard clamp (original behavior)
    beta : float — softplus sharpness (higher = closer to hard clamp)
    theta : ndarray (p,) or None — resistance parameters [r_0, z_1, ..., z_{p-1}].
        Required when use_softplus=True to compute exact exp(eta) for the
        derivative chain rule. If None, falls back to using R_flat as proxy
        (only valid when all pixels are far from the clamp boundaries).
    nodata_mask : ndarray (bool, 2D or 1D) or None — True for pixels whose
        resistance is fixed (e.g. NA → 1e6) and NOT a function of theta.
        Their dR/deta is zeroed out so they don't leak into the gradient.
    source_from_resistance : bool
        If True, adds the ∂b/∂R adjoint correction for 1/R source injection.
    source_mask : ndarray (bool, n_nodes) or None
        Required when source_from_resistance=True. True at source pixels.

    Returns
    -------
    dl_dtheta : ndarray (p,) — gradient w.r.t. [r_0, z_1, z_2, z_3, z_4]
    """
    # Step 1: dl/dw per directed edge
    dl_dw = compute_gradient_wrt_weights(
        v, adjoint_lambda, dl_dc, edge_src, edge_dst, edge_w, n_nodes
    )

    # Step 2: dw/dR — chain through w_ij = 2 / (R_i + R_j)
    # dw/dR_i = -2 / (R_i + R_j)^2
    # For directed edge (s, d): dw/dR_s = -2 / (R_s + R_d)^2
    #                           dw/dR_d = -2 / (R_s + R_d)^2
    denom_sq = (R_flat[edge_src] + R_flat[edge_dst]) ** 2
    dw_dR = -2.0 / denom_sq  # symmetric: dw/dR_src == dw/dR_dst

    # Accumulate dl/dR_i = Σ_{edges involving i} dl/dw * dw/dR_i
    xp = _xp()
    dl_dR = xp.zeros(n_nodes, dtype=xp.float64)
    dl_dw_dR = dl_dw * dw_dR
    if _use_gpu:
        cp.add.at(dl_dR, edge_src, dl_dw_dR)
        cp.add.at(dl_dR, edge_dst, dl_dw_dR)
    else:
        np.add.at(dl_dR, edge_src, dl_dw_dR)
        np.add.at(dl_dR, edge_dst, dl_dw_dR)

    # Step 2b: ∂b/∂R correction for source_from_resistance mode.
    # When b_k = source_mask_k / R_k, the Lagrangian picks up an extra term:
    #   L = loss(c(v)) + λᵀ(Lv − b)   ⟹   ∂L/∂R_k includes −λ_k · ∂b_k/∂R_k
    #   ∂b_k/∂R_k = −source_mask_k / R_k²
    #   ⟹ extra term = −λ_k · (−source_mask_k / R_k²) = +λ_k · source_mask_k / R_k²
    # But we want dl/dR, and the Lagrangian is L = loss − λᵀ(Lv − b)
    # (sign convention: L_int v = b, adjoint: L_int λ = dl/dv).
    # The total derivative is:
    #   dL/dR_k = (dl/dR_k)_weights + λ_k · (db_k/dR_k)
    # where db_k/dR_k = −source_mask_k / R_k².
    # So: dL/dR_k += λ_k · (−source_mask_k / R_k²) = −λ_k · source_mask_k / R_k²
    if source_from_resistance and source_mask is not None:
        src = xp.asarray(source_mask, dtype=bool).ravel()
        dl_dR[src] += -adjoint_lambda[src] / (R_flat[src] ** 2)

    # Step 3: dR/deta — chain through R = f(exp(eta))
    # Bring to CPU for theta/basis chain (always numpy from R)
    dl_dR = _to_host(dl_dR)
    R_flat_cpu = _to_host(R_flat)
    if use_softplus:
        # Softplus double-clamp:
        #   raw = exp(eta)                            (pre-clamp)
        #   R_L = R_min + softplus(raw - R_min)/beta  (lower bound)
        #   R   = R_max - softplus(R_max - R_L)/beta  (upper bound)
        #
        # Chain rule:
        #   dR/deta = dR/dR_L * dR_L/draw * draw/deta
        #   draw/deta = raw = exp(eta)
        #   dR_L/draw = sigmoid(beta * (raw - R_min))
        #   dR/dR_L   = sigmoid(beta * (R_max - R_L))

        # Compute exact exp(eta) from theta and basis_values
        if theta is not None:
            theta = np.asarray(theta, dtype=np.float64).ravel()
            eta = theta[0] + basis_values @ theta[1:]
            # Clamp eta to avoid overflow in exp (eta > 20 is enough for
            # R_max=5000 since exp(20)≈5e8 >> 5000, so softplus saturates)
            eta = np.clip(eta, -50.0, 50.0)
            exp_eta = np.exp(eta)
        else:
            # Fallback: use R_flat (only valid far from clamp boundaries)
            exp_eta = R_flat_cpu.copy()

        # Recompute R_L = R_min + softplus(exp_eta - R_min) / beta
        x_lower = beta * (exp_eta - R_min)
        sp_lower = np.where(x_lower > 20, x_lower / beta,
                           np.log1p(np.exp(np.clip(x_lower, -50, 20))) / beta)
        R_L = R_min + sp_lower

        # Lower sigmoid: dR_L / d(exp_eta)
        sig_lower = np.where(x_lower > 20, 1.0,
                            1.0 / (1.0 + np.exp(-np.clip(x_lower, -50, 20))))

        # Upper sigmoid: dR / dR_L
        x_upper = beta * (R_max - R_L)
        sig_upper = np.where(x_upper > 20, 1.0,
                            1.0 / (1.0 + np.exp(-np.clip(x_upper, -50, 20))))

        # Full derivative: dR/deta = sig_upper * sig_lower * exp(eta)
        dR_deta = sig_upper * sig_lower * exp_eta
    else:
        # Original hard clamp: dR/deta = R if R_min < R < R_max, else 0
        clamp_mask = (R_flat_cpu > R_min) & (R_flat_cpu < R_max)
        dR_deta = np.where(clamp_mask, R_flat_cpu, 0.0)

    # Zero out derivatives for nodata pixels (fixed resistance, not a function of theta)
    if nodata_mask is not None:
        nodata_flat = np.asarray(nodata_mask, dtype=bool).ravel()
        dR_deta[nodata_flat] = 0.0

    dl_deta = dl_dR * dR_deta

    # Step 4: deta/dtheta — eta_i = theta[0] + theta[1]*phi_1_i + ... + theta[p-1]*phi_{p-1}_i
    # dl/dtheta[0] = Σ_i dl/deta_i * 1           (intercept)
    # dl/dtheta[k] = Σ_i dl/deta_i * phi_k_i     (covariate k)
    p = basis_values.shape[1] + 1  # +1 for intercept
    dl_dtheta = np.empty(p, dtype=np.float64)
    dl_dtheta[0] = np.sum(dl_deta)
    dl_dtheta[1:] = basis_values.T @ dl_deta

    return dl_dtheta


# =============================================================================
# Convenience: combined forward + gradient
# =============================================================================

def forward_and_gradient(R_matrix, basis_values, dl_dc_func,
                          R_min=1.0, R_max=5000.0,
                          use_softplus=True, beta=5.0,
                          source_spacing=1,
                          source_from_resistance=False):
    """
    Combined forward solve + gradient computation.

    Parameters
    ----------
    R_matrix : ndarray (n_rows, n_cols) — resistance raster
    basis_values : ndarray (n_nodes, n_basis) — covariate matrix
    dl_dc_func : callable
        Function that takes current_density (n_nodes,) and returns dl_dc (n_nodes,).
        This allows the R caller to compute the PPP likelihood gradient w.r.t.
        current density externally.
    R_min, R_max : clamping bounds
    use_softplus : bool — if True, use differentiable softplus clamp
    beta : float — softplus sharpness
    source_spacing : int
        Lattice spacing for source injection (default 1 = all interior).
    source_from_resistance : bool
        If True, use 1/R source injection (matches Omniscape).

    Returns
    -------
    result : dict with forward pass results plus:
        dl_dtheta : ndarray (p,) — gradient w.r.t. resistance params
        adjoint_lambda : ndarray (n_nodes,) — adjoint variable
    """
    # Forward solve
    fwd = solve_circuit(R_matrix, source_spacing=source_spacing,
                        source_from_resistance=source_from_resistance)

    # Get dl/dc from the external loss function
    dl_dc = dl_dc_func(fwd["current_density"])

    # Compute adjoint RHS: dl/dc → dl/dv
    dl_dv_int = compute_adjoint_rhs(
        fwd["v"], dl_dc,
        fwd["edge_src"], fwd["edge_dst"], fwd["edge_w"],
        fwd["n_rows"] * fwd["n_cols"],
        fwd["interior_mask"]
    )

    # Adjoint solve: L_int λ = dl/dv_int
    adjoint_lambda = adjoint_solve(dl_dv_int)

    # Gradient assembly
    n_nodes = fwd["n_rows"] * fwd["n_cols"]
    dl_dtheta = gradient_wrt_resistance(
        fwd["v"], adjoint_lambda, dl_dc,
        fwd["edge_src"], fwd["edge_dst"], fwd["edge_w"],
        fwd["R_flat"], basis_values, n_nodes,
        R_min=R_min, R_max=R_max,
        use_softplus=use_softplus, beta=beta,
        theta=None, nodata_mask=None,
        source_from_resistance=source_from_resistance,
        source_mask=fwd["source_mask"]
    )

    fwd["dl_dtheta"] = dl_dtheta
    fwd["adjoint_lambda"] = adjoint_lambda
    fwd["dl_dc"] = dl_dc

    return fwd
