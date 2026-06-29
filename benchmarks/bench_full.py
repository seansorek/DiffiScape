#!/usr/bin/env python3
"""Full benchmark: Torch vs JAX × Uniform vs Boundary absorption × CPU/GPU/AMGX."""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "True"

import sys, time, importlib.util, json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
import pyamg

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    import cupyx.scipy.sparse.linalg as cp_splinalg
    _CUPY = cp.cuda.is_available()
except ImportError:
    _CUPY = False

sys.path.insert(0, "inst/python")

spec = importlib.util.spec_from_file_location(
    "cs", "inst/python/diff_cs/03_circuit_solver.py"
)
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)

import diffiscape_jax
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from diffiscape_jax.core import (
    prepare_permeability, circuit_solve, circuit_solve_init,
    _mean_weight, _boundary_mask, _EPS,
)
from jaxscape import GridGraph
from jaxscape.solvers import AMJaxCGSolver, BCOOLinearOperator
from jaxscape.utils import graph_laplacian
import lineax as lx

# =====================================================================
# Configuration
# =====================================================================
N_RUNS = 3
ABSORPTION = 0.01
SIZES = [100, 500, 1000]

cpu = jax.devices("cpu")[0]
try:
    gpu = jax.devices("gpu")[0]
except Exception:
    gpu = None

_HAS_AMGX = cs.amgx_available()

print("=" * 80)
print("FULL BENCHMARK: Torch vs JAX × Uniform vs Boundary × CPU/GPU/AMGX")
print("=" * 80)
print(f"CuPy GPU: {cs.gpu_available()}")
print(f"AMGX:     {_HAS_AMGX}")
try:
    print(f"JAX GPU:  {jax.devices('gpu')}")
except Exception:
    print("JAX GPU:  not available")
print(f"Sizes:    {SIZES}")
print(f"Runs:     {N_RUNS}")

if _HAS_AMGX:
    cs.enable_gpu(True, solver="amgx")
    cs._amgx_ensure_resources()
    print("AMGX pre-initialized")
    cs.enable_gpu(False)

print(flush=True)


# =====================================================================
# Helpers
# =====================================================================
def make_grid(size):
    return np.random.RandomState(42).uniform(1.0, 100.0, (size, size)).astype(np.float64)


def time_fn(fn, n_runs=N_RUNS):
    fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


# =====================================================================
# Torch boundary absorption (custom, not in main module)
# =====================================================================
def torch_boundary_cpu(R, absorption=0.01):
    """Torch CPU: boundary absorption with pyamg AMG preconditioner."""
    n_rows, n_cols = R.shape
    n = n_rows * n_cols
    R_flat = R.ravel().astype(np.float64)

    cs.enable_gpu(False)
    edge_src, edge_dst, edge_w = cs.build_edge_list(R_flat, n_rows, n_cols)
    L = cs.build_laplacian(edge_src, edge_dst, edge_w, n)

    boundary = cs.get_boundary_mask(n_rows, n_cols)
    interior = ~boundary
    abs_diag = np.where(boundary, absorption, 0.0)
    L_grounded = L + sparse.diags(abs_diag, format="csr")

    b = interior.astype(np.float64)

    amg = pyamg.smoothed_aggregation_solver(L_grounded)
    M = amg.aspreconditioner()
    v, info = cg(L_grounded, b, M=M, rtol=1e-10, maxiter=1000)
    return v


def torch_boundary_gpu_jacobi(R, absorption=0.01):
    """Torch GPU: boundary absorption with Jacobi preconditioner."""
    n_rows, n_cols = R.shape
    n = n_rows * n_cols
    R_flat = cp.asarray(R.ravel(), dtype=cp.float64)

    old_gpu = cs._use_gpu
    cs._use_gpu = True
    edge_src, edge_dst, edge_w = cs.build_edge_list(R_flat, n_rows, n_cols)
    L = cs.build_laplacian(edge_src, edge_dst, edge_w, n)
    boundary_np = cs.get_boundary_mask(n_rows, n_cols)
    cs._use_gpu = old_gpu

    boundary = cp.asarray(boundary_np)
    interior = ~boundary
    abs_diag = cp.where(boundary, absorption, 0.0)
    L_grounded = L + cp_sparse.diags(abs_diag, format="csr")

    b = interior.astype(cp.float64)

    diag_L = L_grounded.diagonal()
    diag_inv = 1.0 / cp.maximum(diag_L, 1e-14)
    M = cp_splinalg.LinearOperator(
        shape=L_grounded.shape,
        matvec=lambda x: diag_inv * x,
        dtype=cp.float64,
    )
    v, info = cp_splinalg.cg(L_grounded, b, M=M, rtol=1e-10, maxiter=2000)
    cp.cuda.Device(0).synchronize()
    return cp.asnumpy(v)


def torch_boundary_gpu_amgx(R, absorption=0.01):
    """Torch GPU: boundary absorption with AMGX AMG solver."""
    n_rows, n_cols = R.shape
    n = n_rows * n_cols
    R_flat = cp.asarray(R.ravel(), dtype=cp.float64)

    old_gpu = cs._use_gpu
    cs._use_gpu = True
    edge_src, edge_dst, edge_w = cs.build_edge_list(R_flat, n_rows, n_cols)
    L = cs.build_laplacian(edge_src, edge_dst, edge_w, n)
    boundary_np = cs.get_boundary_mask(n_rows, n_cols)
    cs._use_gpu = old_gpu

    boundary = cp.asarray(boundary_np)
    interior = ~boundary
    abs_diag = cp.where(boundary, absorption, 0.0)
    L_grounded = L + cp_sparse.diags(abs_diag, format="csr")

    b = interior.astype(cp.float64)

    L_csr = L_grounded.tocsr()
    L_scipy = sparse.csr_matrix(
        (cp.asnumpy(L_csr.data),
         cp.asnumpy(L_csr.indices),
         cp.asnumpy(L_csr.indptr)),
        shape=L_grounded.shape,
    )
    b_np = cp.asnumpy(b)
    v_np = cs._amgx_solve(L_scipy, b_np)
    return v_np


# =====================================================================
# JAX uniform absorption
# =====================================================================
def jax_uniform_init(permeability, n_rows, n_cols, absorption=0.01):
    grid = GridGraph(grid=permeability, fun=_mean_weight)
    A = grid.get_adjacency_matrix()
    L = graph_laplacian(A)

    n = n_rows * n_cols
    idx = jnp.arange(n, dtype=L.indices.dtype)
    abs_indices = jnp.stack([idx, idx], axis=-1)
    abs_data = jnp.full(n, absorption, dtype=L.data.dtype)
    absorption_diag = BCOO((abs_data, abs_indices), shape=L.shape)
    L_grounded = L + absorption_diag

    solver = AMJaxCGSolver(rtol=1e-6, atol=1e-6)
    operator = BCOOLinearOperator(L_grounded)
    state = solver.init(operator, {})
    return {
        "operator": operator, "solver": solver, "state": state,
        "A": A, "n": n,
    }


def jax_uniform_solve(permeability, n_rows, n_cols, solver_state):
    operator = solver_state["operator"]
    solver = solver_state["solver"]
    state = solver_state["state"]
    A = solver_state["A"]
    n = solver_state["n"]

    b = jnp.ones(n)
    v = lx.linear_solve(operator, b, solver=solver, state=state).value

    rows = A.indices[:, 0]
    cols = A.indices[:, 1]
    edge_currents = A.data * jnp.sqrt((v[rows] - v[cols]) ** 2 + 1e-8)
    cd = jnp.zeros(n).at[rows].add(edge_currents) * 0.5
    return cd, v


# =====================================================================
# Run benchmarks
# =====================================================================
results = {}

for size in SIZES:
    R = make_grid(size)
    perm_np = prepare_permeability(jnp.array(R), "resistance")
    tag = f"{size}x{size}"
    results[tag] = {"uniform": {}, "boundary": {}}

    print(f"\n{'=' * 80}")
    print(f"  {tag} ({size*size:,} cells)")
    print(f"{'=' * 80}")

    # ==================================================================
    # UNIFORM ABSORPTION: (L + α·I)v = 1
    # ==================================================================
    print(f"\n  UNIFORM ABSORPTION: (L + α·I)v = 1")
    print(f"  {'-' * 50}")

    # Torch CPU
    cs.enable_gpu(False)
    m, s = time_fn(
        lambda: cs.solve_circuit_absorption(R, absorption=ABSORPTION, output="current")
    )
    results[tag]["uniform"]["Torch CPU (pyamg)"] = (m, s)
    print(f"    Torch CPU (pyamg):      {m:>8.1f} ± {s:>5.1f} ms")

    # Torch GPU Jacobi
    if _CUPY:
        try:
            cs.enable_gpu(True, solver="jacobi")
            m, s = time_fn(
                lambda: cs.solve_circuit_absorption(R, absorption=ABSORPTION, output="current")
            )
            results[tag]["uniform"]["Torch GPU (Jacobi)"] = (m, s)
            print(f"    Torch GPU (Jacobi):     {m:>8.1f} ± {s:>5.1f} ms")
        except Exception as e:
            print(f"    Torch GPU (Jacobi):     FAIL ({e})")

    # Torch GPU AMGX
    if _HAS_AMGX:
        try:
            cs.enable_gpu(True, solver="amgx")
            m, s = time_fn(
                lambda: cs.solve_circuit_absorption(R, absorption=ABSORPTION, output="current")
            )
            results[tag]["uniform"]["Torch GPU (AMGX)"] = (m, s)
            print(f"    Torch GPU (AMGX):       {m:>8.1f} ± {s:>5.1f} ms")
        except Exception as e:
            print(f"    Torch GPU (AMGX):       FAIL ({e})")

    # JAX CPU
    with jax.default_device(cpu):
        perm_c = jax.device_put(perm_np, cpu)
        ss = jax_uniform_init(perm_c, size, size, ABSORPTION)
        m, s = time_fn(
            lambda: jax_uniform_solve(perm_c, size, size, ss)[0].block_until_ready()
        )
    results[tag]["uniform"]["JAX CPU (AMJax)"] = (m, s)
    print(f"    JAX CPU (AMJax):        {m:>8.1f} ± {s:>5.1f} ms")

    # JAX GPU
    if gpu:
        try:
            with jax.default_device(gpu):
                perm_g = jax.device_put(perm_np, gpu)
                ss_g = jax_uniform_init(perm_g, size, size, ABSORPTION)
                m, s = time_fn(
                    lambda: jax_uniform_solve(perm_g, size, size, ss_g)[0].block_until_ready()
                )
            results[tag]["uniform"]["JAX GPU (AMJax)"] = (m, s)
            print(f"    JAX GPU (AMJax):        {m:>8.1f} ± {s:>5.1f} ms")
        except Exception as e:
            print(f"    JAX GPU (AMJax):        FAIL ({e})")

    # ==================================================================
    # BOUNDARY ABSORPTION: (L + α·I_∂)v = 1_int
    # ==================================================================
    print(f"\n  BOUNDARY ABSORPTION: (L + α·I_∂)v = 1_int")
    print(f"  {'-' * 50}")

    # Torch CPU (pyamg)
    m, s = time_fn(lambda: torch_boundary_cpu(R, ABSORPTION))
    results[tag]["boundary"]["Torch CPU (pyamg)"] = (m, s)
    print(f"    Torch CPU (pyamg):      {m:>8.1f} ± {s:>5.1f} ms")

    # Torch GPU (Jacobi)
    if _CUPY:
        try:
            m, s = time_fn(lambda: torch_boundary_gpu_jacobi(R, ABSORPTION))
            results[tag]["boundary"]["Torch GPU (Jacobi)"] = (m, s)
            print(f"    Torch GPU (Jacobi):     {m:>8.1f} ± {s:>5.1f} ms")
        except Exception as e:
            print(f"    Torch GPU (Jacobi):     FAIL ({e})")

    # Torch GPU (AMGX) -- THE KEY TEST
    if _HAS_AMGX:
        try:
            m, s = time_fn(lambda: torch_boundary_gpu_amgx(R, ABSORPTION))
            results[tag]["boundary"]["Torch GPU (AMGX)"] = (m, s)
            print(f"    Torch GPU (AMGX):       {m:>8.1f} ± {s:>5.1f} ms  ◀ KEY TEST")
        except Exception as e:
            print(f"    Torch GPU (AMGX):       FAIL ({e})")

    # JAX CPU (boundary)
    with jax.default_device(cpu):
        perm_c = jax.device_put(perm_np, cpu)
        ss = circuit_solve_init(perm_c, size, size, ABSORPTION)
        m, s = time_fn(
            lambda: circuit_solve(perm_c, size, size, solver_state=ss)[0].block_until_ready()
        )
    results[tag]["boundary"]["JAX CPU (AMJax)"] = (m, s)
    print(f"    JAX CPU (AMJax):        {m:>8.1f} ± {s:>5.1f} ms")

    # JAX GPU (boundary)
    if gpu:
        try:
            with jax.default_device(gpu):
                perm_g = jax.device_put(perm_np, gpu)
                ss_g = circuit_solve_init(perm_g, size, size, ABSORPTION)
                m, s = time_fn(
                    lambda: circuit_solve(perm_g, size, size, solver_state=ss_g)[0].block_until_ready()
                )
            results[tag]["boundary"]["JAX GPU (AMJax)"] = (m, s)
            print(f"    JAX GPU (AMJax):        {m:>8.1f} ± {s:>5.1f} ms")
        except Exception as e:
            print(f"    JAX GPU (AMJax):        FAIL ({e})")

    print(flush=True)


# =====================================================================
# Verification: cross-framework correlation
# =====================================================================
print(f"\n{'=' * 80}")
print("VERIFICATION: Cross-framework correlation (100×100)")
print("=" * 80)

R_v = make_grid(100)
perm_v = prepare_permeability(jnp.array(R_v), "resistance")

# Uniform: Torch vs JAX
cs.enable_gpu(False)
torch_uni = cs.solve_circuit_absorption(R_v, absorption=ABSORPTION, output="current")["current_density"]
with jax.default_device(cpu):
    perm_c = jax.device_put(perm_v, cpu)
    ss = jax_uniform_init(perm_c, 100, 100, ABSORPTION)
    jax_uni = np.array(jax_uniform_solve(perm_c, 100, 100, ss)[0])
r_uni = np.corrcoef(torch_uni.ravel(), jax_uni.ravel())[0, 1]
print(f"  Uniform:  Pearson r = {r_uni:.8f}")

# Boundary: Torch vs JAX
torch_bdy = torch_boundary_cpu(R_v, ABSORPTION)
with jax.default_device(cpu):
    perm_c = jax.device_put(perm_v, cpu)
    ss = circuit_solve_init(perm_c, 100, 100, ABSORPTION)
    jax_bdy_cd, jax_bdy_v = circuit_solve(perm_c, 100, 100, solver_state=ss)
    jax_bdy = np.array(jax_bdy_v)
r_bdy = np.corrcoef(torch_bdy.ravel(), jax_bdy.ravel())[0, 1]
print(f"  Boundary: Pearson r = {r_bdy:.8f}")

# AMGX gradient check
if _HAS_AMGX:
    print(f"\n  AMGX gradient check (uniform, 100×100):")
    n_nodes = 100 * 100
    basis = np.random.RandomState(99).randn(n_nodes, 3)

    cs.enable_gpu(False)
    cs._cache["amg"] = None
    cs._cache["R_hash"] = None
    res_cpu = cs.forward_and_gradient(R_v, basis, lambda cd: np.ones_like(cd), source_from_resistance=False)
    grad_cpu = res_cpu["dl_dtheta"]

    cs.enable_gpu(True, solver="amgx")
    cs._cache["amg"] = None
    cs._cache["R_hash"] = None
    res_amgx = cs.forward_and_gradient(R_v, basis, lambda cd: np.ones_like(cd), source_from_resistance=False)
    grad_amgx = res_amgx["dl_dtheta"]

    corr = np.corrcoef(grad_cpu, grad_amgx)[0, 1]
    rel_err = np.max(np.abs(grad_cpu - grad_amgx)) / (np.max(np.abs(grad_cpu)) + 1e-15)
    print(f"    CPU vs AMGX gradient r = {corr:.10f}")
    print(f"    Max relative error:      {rel_err:.2e}")

print(flush=True)

# =====================================================================
# Summary tables
# =====================================================================
print(f"\n{'=' * 80}")
print("SUMMARY TABLES")
print("=" * 80)

backends = [
    "Torch CPU (pyamg)",
    "Torch GPU (Jacobi)",
    "Torch GPU (AMGX)",
    "JAX CPU (AMJax)",
    "JAX GPU (AMJax)",
]

for strategy in ["uniform", "boundary"]:
    label = "UNIFORM" if strategy == "uniform" else "BOUNDARY"
    print(f"\n  {label} ABSORPTION:")
    header = f"  {'Grid':<12}" + "".join(f"{'':>3}{b:>22}" for b in backends)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for size in SIZES:
        tag = f"{size}x{size}"
        row = f"  {tag:<12}"
        for b in backends:
            if b in results[tag][strategy]:
                m, s = results[tag][strategy][b]
                row += f"{'':>3}{m:>17.1f} ms"
            else:
                row += f"{'':>3}{'N/A':>20}"
        print(row)

# Save raw results as JSON
out = {}
for tag in results:
    out[tag] = {}
    for strat in results[tag]:
        out[tag][strat] = {}
        for backend, (m, s) in results[tag][strat].items():
            out[tag][strat][backend] = {"mean_ms": round(m, 1), "std_ms": round(s, 1)}

with open("bench_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nResults saved to bench_results.json")
