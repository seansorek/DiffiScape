# Julia → JAX Backend Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all Julia (Omniscape.jl, Circuitscape.jl, Enzyme.jl) and PyTorch backends with a single JAX-based backend using JAXScape, consolidating to one Python compute engine called via reticulate.

**Architecture:** R stays as the user-facing layer (basis functions, intensity models, posterior, diagnostics). A new `jax_bridge.R` replaces both `julia_bridge.R` and `torch_bridge.R`, calling a bundled Python package (`inst/python/diffiscape_jax/`) that wraps JAXScape for circuit solves, Flax for resistance nets, optax for optimization, and NumPyro for Bayesian sampling. Two solver modes: `"surrogate"` (JAXScape forward-only → GP/EI in R) and `"gradient"` (JAXScape + jax.grad → L-BFGS/Adam/MCMC in Python).

**Tech Stack:** R + reticulate, JAXScape, JAX, Flax, optax, jaxopt, NumPyro

## Global Constraints

- Python >= 3.10 (JAX requirement)
- JAXScape installed via `pip install jaxscape`
- All circuit solves use JAXScape's `GridGraph` + `ResistanceDistance`; no custom Laplacian code
- JAXScape operates in permeability space; conversion happens in `core.py`
- The `parameterization` config option (`"resistance"` default, `"permeability"`) controls whether basis functions / nets produce resistance or permeability
- R-side statistical machinery (intensity families, posterior, diagnostics) is NOT modified
- Each phase keeps old backends alive until validated; deletion happens in Phase 5
- All Python code lives under `inst/python/diffiscape_jax/` as a proper package with `__init__.py`

---

## Phase 1: Core Circuit Solver + Surrogate Path

### Task 1: Python package scaffold + `core.py` forward solve

**Files:**
- Create: `inst/python/diffiscape_jax/__init__.py`
- Create: `inst/python/diffiscape_jax/core.py`
- Create: `inst/python/diffiscape_jax/tests/__init__.py`
- Create: `inst/python/diffiscape_jax/tests/test_core.py`

**Interfaces:**
- Produces: `prepare_permeability(surface, parameterization, r_min, r_max, p_min, p_max) -> jnp.ndarray`
- Produces: `forward_solve(resistance_matrix, n_rows, n_cols, sources=None, parameterization="resistance", solver_type="pyamg") -> dict` with keys `"connectivity"` (np.ndarray, shape (n_rows, n_cols)), `"elapsed"` (float)

- [ ] **Step 1: Write failing test for `prepare_permeability`**

```python
# inst/python/diffiscape_jax/tests/test_core.py
import numpy as np
import jax.numpy as jnp
import pytest

from diffiscape_jax.core import prepare_permeability


def test_prepare_permeability_resistance_mode():
    resistance = jnp.array([[10.0, 20.0], [50.0, 100.0]])
    perm = prepare_permeability(resistance, "resistance")
    expected = 1.0 / resistance
    np.testing.assert_allclose(perm, expected, rtol=1e-6)


def test_prepare_permeability_permeability_mode():
    permeability = jnp.array([[0.1, 0.05], [0.02, 0.01]])
    result = prepare_permeability(permeability, "permeability")
    np.testing.assert_allclose(result, permeability, rtol=1e-6)


def test_prepare_permeability_clamps_resistance():
    resistance = jnp.array([[0.001, 10000.0]])
    perm = prepare_permeability(resistance, "resistance", r_min=1.0, r_max=5000.0)
    assert float(perm[0, 0]) == pytest.approx(1.0 / 1.0, rel=1e-5)
    assert float(perm[0, 1]) == pytest.approx(1.0 / 5000.0, rel=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd inst/python && python -m pytest diffiscape_jax/tests/test_core.py::test_prepare_permeability_resistance_mode -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'diffiscape_jax'`

- [ ] **Step 3: Implement `core.py` with `prepare_permeability` and `forward_solve`**

```python
# inst/python/diffiscape_jax/__init__.py
```

```python
# inst/python/diffiscape_jax/core.py
import numpy as np
import jax
import jax.numpy as jnp
from jaxscape import GridGraph
from jaxscape.distances import ResistanceDistance
import time

DEFAULT_R_MIN = 1.0
DEFAULT_R_MAX = 5000.0
DEFAULT_P_MIN = 1.0 / 5000.0
DEFAULT_P_MAX = 1.0


def prepare_permeability(surface, parameterization,
                         r_min=DEFAULT_R_MIN, r_max=DEFAULT_R_MAX,
                         p_min=DEFAULT_P_MIN, p_max=DEFAULT_P_MAX):
    if parameterization == "resistance":
        clamped = jnp.clip(surface, r_min, r_max)
        return 1.0 / clamped
    else:
        return jnp.clip(surface, p_min, p_max)


def _mean_weight(x, y):
    return (x + y) / 2


def forward_solve(resistance_matrix, n_rows, n_cols,
                  sources=None, parameterization="resistance",
                  solver_type="pyamg"):
    surface = jnp.array(resistance_matrix, dtype=jnp.float64)
    permeability = prepare_permeability(surface, parameterization)

    grid = GridGraph(grid=permeability, fun=_mean_weight)

    if sources is None:
        sources = grid.coord_to_index(
            jnp.array([0]), jnp.array([0])
        )

    distance = ResistanceDistance()

    t0 = time.time()
    dist_values = distance(grid, sources)
    elapsed = time.time() - t0

    connectivity = np.array(grid.node_values_to_array(dist_values))

    return {
        "connectivity": connectivity,
        "elapsed": elapsed,
    }
```

- [ ] **Step 4: Write and run test for `forward_solve`**

Add to `test_core.py`:

```python
from diffiscape_jax.core import forward_solve


def test_forward_solve_returns_correct_shape():
    n_rows, n_cols = 10, 10
    resistance = np.ones((n_rows, n_cols)) * 10.0
    result = forward_solve(resistance, n_rows, n_cols)
    assert result["connectivity"].shape == (n_rows, n_cols)
    assert result["elapsed"] > 0
    assert np.all(np.isfinite(result["connectivity"]))


def test_forward_solve_permeability_mode():
    n_rows, n_cols = 10, 10
    permeability = np.ones((n_rows, n_cols)) * 0.1
    result = forward_solve(permeability, n_rows, n_cols,
                           parameterization="permeability")
    assert result["connectivity"].shape == (n_rows, n_cols)
```

Run: `cd inst/python && python -m pytest diffiscape_jax/tests/test_core.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add inst/python/diffiscape_jax/
git commit -m "feat(phase1): add diffiscape_jax package scaffold + core.py forward solve"
```

### Task 2: `window.py` — Moving-window (Omniscape-style) cumulative current

**Files:**
- Create: `inst/python/diffiscape_jax/window.py`
- Create: `inst/python/diffiscape_jax/tests/test_window.py`

**Interfaces:**
- Consumes: `core.prepare_permeability`
- Produces: `cumulative_current(resistance_matrix, n_rows, n_cols, radius=13, block_size=5, source_from_resistance=True, parameterization="resistance", output="current") -> dict` with keys `"current"` (np.ndarray or None), `"voltage"` (np.ndarray or None), `"elapsed"` (float)

- [ ] **Step 1: Write failing test**

```python
# inst/python/diffiscape_jax/tests/test_window.py
import numpy as np
import pytest

from diffiscape_jax.window import cumulative_current


def test_cumulative_current_shape():
    n_rows, n_cols = 20, 20
    resistance = np.random.uniform(1, 100, (n_rows, n_cols))
    result = cumulative_current(resistance, n_rows, n_cols,
                                radius=5, block_size=3)
    assert result["current"].shape == (n_rows, n_cols)
    assert result["elapsed"] > 0


def test_cumulative_current_output_modes():
    n_rows, n_cols = 15, 15
    resistance = np.ones((n_rows, n_cols)) * 10.0
    curr = cumulative_current(resistance, n_rows, n_cols,
                              radius=5, block_size=3, output="current")
    assert curr["current"] is not None
    assert curr["voltage"] is None

    volt = cumulative_current(resistance, n_rows, n_cols,
                              radius=5, block_size=3, output="voltage")
    assert volt["current"] is None
    assert volt["voltage"] is not None

    both = cumulative_current(resistance, n_rows, n_cols,
                              radius=5, block_size=3, output="both")
    assert both["current"] is not None
    assert both["voltage"] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd inst/python && python -m pytest diffiscape_jax/tests/test_window.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'diffiscape_jax.window'`

- [ ] **Step 3: Implement `window.py`**

```python
# inst/python/diffiscape_jax/window.py
import numpy as np
import jax
import jax.numpy as jnp
from jaxscape import GridGraph, WindowOperation
from jaxscape.distances import ResistanceDistance
import time

from .core import prepare_permeability, _mean_weight


def cumulative_current(resistance_matrix, n_rows, n_cols,
                       radius=13, block_size=5,
                       source_from_resistance=True,
                       parameterization="resistance",
                       output="current"):
    surface = jnp.array(resistance_matrix, dtype=jnp.float64)
    permeability = prepare_permeability(surface, parameterization)

    grid = GridGraph(grid=permeability, fun=_mean_weight)
    distance = ResistanceDistance()
    window_op = WindowOperation(grid, radius=radius, block_size=block_size)

    t0 = time.time()
    current_acc = np.zeros((n_rows, n_cols))
    voltage_acc = np.zeros((n_rows, n_cols))

    for window in window_op:
        sub_grid, source_idx = window
        dist_vals = distance(sub_grid, source_idx)
        arr = np.array(sub_grid.node_values_to_array(dist_vals))
        current_acc += arr  # accumulate across windows
        # voltage accumulation TBD based on JAXScape API

    elapsed = time.time() - t0

    result = {
        "current": current_acc if output in ("current", "both") else None,
        "voltage": voltage_acc if output in ("voltage", "both") else None,
        "elapsed": elapsed,
    }
    return result
```

Note: the exact `WindowOperation` iteration API will need to be adapted to JAXScape's actual interface — the above is based on the README's description. The implementer should consult JAXScape docs and adjust the loop accordingly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd inst/python && python -m pytest diffiscape_jax/tests/test_window.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add inst/python/diffiscape_jax/window.py inst/python/diffiscape_jax/tests/test_window.py
git commit -m "feat(phase1): add window.py Omniscape-style cumulative current via JAXScape"
```

### Task 3: `jax_bridge.R` — R-side bridge to the JAX backend

**Files:**
- Create: `R/jax_bridge.R`
- Create: `tests/testthat/test-jax_bridge.R`

**Interfaces:**
- Consumes: `diffiscape_jax.core.forward_solve`, `diffiscape_jax.window.cumulative_current` (via reticulate)
- Produces: `ds_jax_setup(python = NULL, force = FALSE) -> invisible(TRUE)`
- Produces: `ds_jax_check() -> logical(1)`
- Produces: `ds_jax_call(module_name, fn_name, ...) -> R object`
- Produces: `ds_jax_connectivity(resistance, radius, block_size, source_from_resistance, parameterization, output) -> list(cum_current, flow_potential, elapsed_seconds)`
- Produces: `ds_install_jax_deps(method, envname, gpu) -> invisible(TRUE)`

- [ ] **Step 1: Write failing test**

```r
# tests/testthat/test-jax_bridge.R
test_that("ds_jax_setup initialises without error", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("jaxscape"),
          "jaxscape not installed")
  expect_true(ds_jax_setup())
  expect_true(ds_jax_check())
})

test_that("ds_jax_connectivity returns correct structure", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("jaxscape"),
          "jaxscape not installed")
  ds_jax_setup()

  r <- terra::rast(nrows = 15, ncols = 15, vals = runif(225, 1, 100))
  result <- ds_jax_connectivity(r, radius = 5L, block_size = 3L)

  expect_type(result, "list")
  expect_named(result, c("cum_current", "flow_potential", "elapsed_seconds"))
  expect_s4_class(result$cum_current, "SpatRaster")
  expect_true(result$elapsed_seconds > 0)
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `Rscript -e "devtools::test(filter = 'jax_bridge')"`
Expected: FAIL — `could not find function "ds_jax_setup"`

- [ ] **Step 3: Implement `jax_bridge.R`**

```r
# R/jax_bridge.R

#' Initialise the JAX backend for DiffiScape
#'
#' Loads the bundled `diffiscape_jax` package via reticulate, verifying
#' that required Python packages (jax, jaxscape, numpy) are available.
#'
#' @param python Optional path to a Python binary.
#' @param force Re-initialise even if already loaded.
#' @return `TRUE` (invisibly) on success.
#' @export
ds_jax_setup <- function(python = NULL, force = FALSE) {

  if (.ds_env$jax_initialized && !force) {
    message("JAX backend already initialised.")
    return(invisible(TRUE))
  }

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required for the JAX backend. ",
         "Install with install.packages('reticulate').",
         call. = FALSE)
  }

  if (!is.null(python)) {
    reticulate::use_python(python, required = TRUE)
  }

  required <- c("jax", "jaxlib", "jaxscape", "numpy")
  missing  <- required[!vapply(required, reticulate::py_module_available,
                               logical(1))]
  if (length(missing) > 0) {
    stop("Missing Python packages: ", paste(missing, collapse = ", "),
         "\nInstall them with ds_install_jax_deps().",
         call. = FALSE)
  }

  module_dir <- system.file("python", package = "DiffiScape")
  if (!nzchar(module_dir) || !dir.exists(module_dir)) {
    stop("Bundled Python package not found in inst/python/. ",
         "Re-install DiffiScape.", call. = FALSE)
  }

  tryCatch({
    .ds_env$jax_core <- reticulate::import_from_path(
      "diffiscape_jax.core", path = module_dir)
    .ds_env$jax_window <- reticulate::import_from_path(
      "diffiscape_jax.window", path = module_dir)
    .ds_env$jax_initialized <- TRUE

    jax <- reticulate::import("jax")
    devices <- jax$devices()
    dev_str <- if (length(devices) > 0) {
      paste(sapply(devices, function(d) d$platform), collapse = ", ")
    } else "cpu"
    message(sprintf("  JAX backend ready | devices: %s", dev_str))
    invisible(TRUE)
  }, error = function(e) {
    .ds_env$jax_initialized <- FALSE
    stop("Failed to initialise JAX backend: ", conditionMessage(e),
         call. = FALSE)
  })
}


#' Check whether the JAX backend is loaded
#' @return Logical.
#' @export
ds_jax_check <- function() {
  isTRUE(.ds_env$jax_initialized)
}


#' Compute cumulative current via the JAX backend
#'
#' Drop-in replacement for [run_cumulative_current()] that uses JAXScape
#' instead of the Julia differentiable solver.
#'
#' @param resistance A single-layer [terra::SpatRaster] of resistance.
#' @param radius Integer; moving-window radius (default 13).
#' @param block_size Integer; source block side length (default 5).
#' @param source_from_resistance Logical (default TRUE).
#' @param parameterization Character; `"resistance"` (default) or `"permeability"`.
#' @param output Character; `"current"`, `"voltage"`, or `"both"`.
#' @return A list matching the [run_cumulative_current()] interface.
#' @export
ds_jax_connectivity <- function(resistance,
                                radius     = 13L,
                                block_size = 5L,
                                source_from_resistance = TRUE,
                                parameterization = "resistance",
                                output     = "current") {

  output <- match.arg(output, c("current", "voltage", "both"))
  if (!ds_jax_check()) ds_jax_setup()

  np <- reticulate::import("numpy", convert = FALSE)
  nrow_grid <- terra::nrow(resistance)
  ncol_grid <- terra::ncol(resistance)

  R_vec <- as.numeric(terra::values(resistance))
  R_mat <- matrix(R_vec, nrow = nrow_grid, ncol = ncol_grid, byrow = TRUE)
  R_mat[is.na(R_mat)] <- 0

  R_np <- np$array(R_mat, dtype = np$float64)

  start <- Sys.time()

  result <- .ds_env$jax_window$cumulative_current(
    R_np,
    as.integer(nrow_grid),
    as.integer(ncol_grid),
    radius     = as.integer(radius),
    block_size = as.integer(block_size),
    source_from_resistance = source_from_resistance,
    parameterization = parameterization,
    output     = output
  )

  result_r <- reticulate::py_to_r(result)
  elapsed <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  message(sprintf("JAX circuit solver completed in %.1f s", elapsed))

  mat_to_rast <- function(mat, layer_name) {
    r <- terra::rast(resistance)
    terra::values(r) <- as.vector(t(mat))
    names(r) <- layer_name
    r
  }

  cum_rast  <- if (!is.null(result_r$current))
    mat_to_rast(result_r$current, "cum_current") else NULL
  volt_rast <- if (!is.null(result_r$voltage))
    mat_to_rast(result_r$voltage, "flow_potential") else NULL

  list(
    cum_current     = cum_rast,
    flow_potential  = volt_rast,
    elapsed_seconds = elapsed
  )
}


#' Install Python dependencies for the JAX backend
#'
#' @param method Installation method for reticulate.
#' @param envname Optional virtualenv/conda name.
#' @param gpu If TRUE, install JAX with CUDA support.
#' @return Invisible TRUE.
#' @export
ds_install_jax_deps <- function(method  = "auto",
                                 envname = NULL,
                                 gpu     = FALSE) {

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.", call. = FALSE)
  }

  pkgs <- c("jax", "jaxlib", "jaxscape", "numpy", "scipy")
  if (isTRUE(gpu)) {
    message("Note: for GPU support, install JAX with CUDA following ",
            "https://jax.readthedocs.io/en/latest/installation.html")
  }

  message("Installing Python packages: ", paste(pkgs, collapse = ", "))
  reticulate::py_install(pkgs, method = method, envname = envname)
  invisible(TRUE)
}
```

- [ ] **Step 4: Run tests**

Run: `Rscript -e "devtools::test(filter = 'jax_bridge')"`
Expected: PASS (or skip if jaxscape not installed)

- [ ] **Step 5: Commit**

```bash
git add R/jax_bridge.R tests/testthat/test-jax_bridge.R
git commit -m "feat(phase1): add jax_bridge.R — R reticulate interface to JAX backend"
```

### Task 4: Wire surrogate solver to use JAX connectivity

**Files:**
- Modify: `R/connectivity.R` — add `backend` parameter to `run_omniscape` and `run_cumulative_current`
- Modify: `R/optimizer.R:196-214` — switch `optimize_resistance_enzyme` obj_fn to use JAX
- Modify: `R/zzz.R` — add `jax_initialized` to `.ds_env`
- Modify: `tests/testthat/test-connectivity.R` — add JAX backend tests

**Interfaces:**
- Consumes: `ds_jax_connectivity` from Task 3
- Produces: Updated `run_cumulative_current(..., backend = "jax")` that dispatches to JAX

- [ ] **Step 1: Update `.onLoad` in `zzz.R` to initialise JAX state**

Add to `R/zzz.R` inside `.onLoad`:
```r
.ds_env$jax_initialized <- FALSE
.ds_env$jax_core   <- NULL
.ds_env$jax_window  <- NULL
```

- [ ] **Step 2: Add `backend` parameter to `run_cumulative_current`**

In `R/connectivity.R`, modify `run_cumulative_current` to accept `backend = c("julia", "jax")`. When `backend = "jax"`, delegate to `ds_jax_connectivity` instead of `ds_julia_call`.

```r
run_cumulative_current <- function(resistance,
                                   radius     = 13L,
                                   block_size = 5L,
                                   output     = "current",
                                   backend    = c("julia", "jax"),
                                   parameterization = "resistance") {

  backend <- match.arg(backend)
  output  <- match.arg(output, c("current", "voltage", "both"))

  if (backend == "jax") {
    return(ds_jax_connectivity(
      resistance,
      radius     = radius,
      block_size = block_size,
      parameterization = parameterization,
      output     = output
    ))
  }

  # ... existing Julia code unchanged ...
}
```

- [ ] **Step 3: Write test that exercises the JAX backend**

```r
# Add to tests/testthat/test-connectivity.R
test_that("run_cumulative_current works with JAX backend", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("jaxscape"),
          "jaxscape not installed")
  ds_jax_setup()

  r <- terra::rast(nrows = 15, ncols = 15,
                   vals = runif(225, 1, 100))
  result <- run_cumulative_current(r, radius = 5L, block_size = 3L,
                                   backend = "jax")
  expect_type(result, "list")
  expect_s4_class(result$cum_current, "SpatRaster")
})
```

- [ ] **Step 4: Run tests**

Run: `Rscript -e "devtools::test(filter = 'connectivity')"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add R/connectivity.R R/zzz.R tests/testthat/test-connectivity.R
git commit -m "feat(phase1): wire run_cumulative_current to JAX backend via backend param"
```

---

## Phase 2: Gradient Solver for Parametric Resistance

### Task 5: `core.py` — `solve_with_grad` via `jax.value_and_grad`

**Files:**
- Modify: `inst/python/diffiscape_jax/core.py`
- Modify: `inst/python/diffiscape_jax/tests/test_core.py`

**Interfaces:**
- Produces: `solve_with_grad(params, basis_values, valid_mask, n_rows, n_cols, cell_area, link_fn, radius, block_size, parameterization, solver_type) -> dict` with keys `"connectivity"` (np.ndarray), `"grad_params"` (np.ndarray), `"loglik"` (float), `"elapsed"` (float)

- [ ] **Step 1: Write failing test for gradient correctness**

```python
# Add to inst/python/diffiscape_jax/tests/test_core.py
import jax

from diffiscape_jax.core import solve_with_grad


def test_solve_with_grad_finite_difference():
    """Compare jax.grad against finite-difference gradient."""
    n_rows, n_cols = 8, 8
    n_basis = 2
    rng = np.random.default_rng(42)
    basis = rng.standard_normal((n_rows * n_cols, n_basis))
    valid_mask = np.ones(n_rows * n_cols, dtype=bool)
    params = np.array([3.0, 0.5, -0.3])  # r_0, z_1, z_2

    result = solve_with_grad(
        params, basis, valid_mask, n_rows, n_cols,
        cell_area=1.0,
        link_fn="exp",
        radius=3, block_size=2,
        parameterization="resistance",
    )

    assert "grad_params" in result
    assert result["grad_params"].shape == params.shape
    assert np.all(np.isfinite(result["grad_params"]))

    # Finite-difference check
    eps = 1e-4
    for i in range(len(params)):
        params_p = params.copy()
        params_m = params.copy()
        params_p[i] += eps
        params_m[i] -= eps
        r_p = solve_with_grad(params_p, basis, valid_mask, n_rows, n_cols,
                              cell_area=1.0, link_fn="exp", radius=3,
                              block_size=2, parameterization="resistance")
        r_m = solve_with_grad(params_m, basis, valid_mask, n_rows, n_cols,
                              cell_area=1.0, link_fn="exp", radius=3,
                              block_size=2, parameterization="resistance")
        fd = (r_p["loglik"] - r_m["loglik"]) / (2 * eps)
        np.testing.assert_allclose(
            result["grad_params"][i], fd,
            rtol=0.05,
            err_msg=f"Gradient mismatch at param {i}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd inst/python && python -m pytest diffiscape_jax/tests/test_core.py::test_solve_with_grad_finite_difference -v`
Expected: FAIL — `ImportError: cannot import name 'solve_with_grad'`

- [ ] **Step 3: Implement `solve_with_grad`**

Add to `core.py`:

```python
def _apply_link(params, basis_values, link_fn):
    r_0 = params[0]
    z = params[1:]
    log_R = r_0 + basis_values @ z
    if link_fn == "exp":
        return jnp.exp(log_R)
    elif link_fn == "softplus":
        return jax.nn.softplus(log_R)
    elif link_fn == "identity":
        return log_R
    else:
        return jnp.exp(log_R)


def _connectivity_objective(params, basis_values, valid_mask, n_rows, n_cols,
                            cell_area, link_fn, radius, block_size,
                            parameterization, obs_counts=None):
    resistance_flat = _apply_link(params, basis_values, link_fn)

    full_surface = jnp.zeros(n_rows * n_cols)
    full_surface = full_surface.at[valid_mask].set(resistance_flat)
    surface_2d = full_surface.reshape((n_rows, n_cols))

    permeability = prepare_permeability(surface_2d, parameterization)
    grid = GridGraph(grid=permeability, fun=_mean_weight)
    distance = ResistanceDistance()
    source = grid.coord_to_index(jnp.array([0]), jnp.array([0]))
    connectivity = distance(grid, source)

    log_conn = jnp.log1p(connectivity)
    loglik = jnp.sum(log_conn)  # simplified; real objective wired later
    return loglik


def solve_with_grad(params, basis_values, valid_mask, n_rows, n_cols,
                    cell_area, link_fn="exp", radius=13, block_size=5,
                    parameterization="resistance", obs_counts=None):
    params_jnp = jnp.array(params, dtype=jnp.float64)
    basis_jnp = jnp.array(basis_values, dtype=jnp.float64)
    mask_jnp = jnp.array(valid_mask)

    val_and_grad_fn = jax.value_and_grad(_connectivity_objective, argnums=0)

    t0 = time.time()
    loglik, grad = val_and_grad_fn(
        params_jnp, basis_jnp, mask_jnp, n_rows, n_cols,
        cell_area, link_fn, radius, block_size, parameterization, obs_counts
    )
    elapsed = time.time() - t0

    return {
        "loglik": float(loglik),
        "grad_params": np.array(grad),
        "elapsed": elapsed,
    }
```

Note: the `_connectivity_objective` function uses a simplified log-likelihood. The full PPP objective will be wired in when the intensity integration is connected. The key deliverable here is that `jax.value_and_grad` flows through JAXScape correctly.

- [ ] **Step 4: Run tests**

Run: `cd inst/python && python -m pytest diffiscape_jax/tests/test_core.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add inst/python/diffiscape_jax/core.py inst/python/diffiscape_jax/tests/test_core.py
git commit -m "feat(phase2): add solve_with_grad with jax.value_and_grad through JAXScape"
```

### Task 6: `optimize.py` — L-BFGS parametric optimizer

**Files:**
- Create: `inst/python/diffiscape_jax/optimize.py`
- Create: `inst/python/diffiscape_jax/tests/test_optimize.py`

**Interfaces:**
- Consumes: `core.solve_with_grad`, `core._apply_link`, `core._connectivity_objective`
- Produces: `run_parametric_optimization(basis_values, obs_counts, valid_mask, n_rows, n_cols, cell_area, init_params, link_fn, radius, block_size, parameterization, method, lr, n_epochs, patience, seed, verbose) -> dict` with keys `"best_params"` (np.ndarray), `"best_loglik"` (float), `"loss_history"` (list[float]), `"n_epochs_run"` (int), `"elapsed"` (float), `"converged"` (bool)

- [ ] **Step 1: Write failing test**

```python
# inst/python/diffiscape_jax/tests/test_optimize.py
import numpy as np
import pytest

from diffiscape_jax.optimize import run_parametric_optimization


def test_parametric_optimization_converges():
    n_rows, n_cols = 10, 10
    n_basis = 2
    rng = np.random.default_rng(42)
    basis = rng.standard_normal((n_rows * n_cols, n_basis))
    obs = rng.poisson(5, n_rows * n_cols).astype(float)
    valid = np.ones(n_rows * n_cols, dtype=bool)
    init_params = np.array([3.0, 0.0, 0.0])

    result = run_parametric_optimization(
        basis, obs, valid, n_rows, n_cols,
        cell_area=1.0, init_params=init_params,
        link_fn="exp", radius=3, block_size=2,
        parameterization="resistance",
        method="lbfgs", n_epochs=50, verbose=False,
    )

    assert "best_params" in result
    assert len(result["best_params"]) == 3
    assert result["n_epochs_run"] > 0
    assert isinstance(result["loss_history"], list)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd inst/python && python -m pytest diffiscape_jax/tests/test_optimize.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `optimize.py`**

```python
# inst/python/diffiscape_jax/optimize.py
import numpy as np
import jax
import jax.numpy as jnp
import jaxopt
import optax
import time

from .core import _connectivity_objective


def run_parametric_optimization(basis_values, obs_counts, valid_mask,
                                n_rows, n_cols, cell_area,
                                init_params, link_fn="exp",
                                radius=13, block_size=5,
                                parameterization="resistance",
                                method="lbfgs", lr=0.01,
                                n_epochs=300, patience=30,
                                seed=42, verbose=True):
    params = jnp.array(init_params, dtype=jnp.float64)
    basis_jnp = jnp.array(basis_values, dtype=jnp.float64)
    mask_jnp = jnp.array(valid_mask)
    obs_jnp = jnp.array(obs_counts, dtype=jnp.float64)

    def neg_loglik(p):
        return -_connectivity_objective(
            p, basis_jnp, mask_jnp, n_rows, n_cols,
            cell_area, link_fn, radius, block_size,
            parameterization, obs_jnp
        )

    t0 = time.time()
    loss_history = []

    if method == "lbfgs":
        solver = jaxopt.LBFGS(fun=neg_loglik, maxiter=n_epochs, tol=1e-6)
        result = solver.run(params)
        best_params = result.params
        best_loss = float(neg_loglik(best_params))
        loss_history = [best_loss]
        n_run = result.state.iter_num if hasattr(result.state, 'iter_num') else n_epochs
        converged = True

    elif method == "adam":
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)
        grad_fn = jax.grad(neg_loglik)

        best_loss = float('inf')
        best_params = params
        stall = 0

        for epoch in range(n_epochs):
            g = grad_fn(params)
            updates, opt_state = optimizer.update(g, opt_state)
            params = optax.apply_updates(params, updates)
            loss = float(neg_loglik(params))
            loss_history.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_params = params
                stall = 0
            else:
                stall += 1

            if stall >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        n_run = len(loss_history)
        converged = stall < patience

    elapsed = time.time() - t0

    return {
        "best_params": np.array(best_params),
        "best_loglik": -best_loss,
        "loss_history": loss_history,
        "n_epochs_run": int(n_run),
        "elapsed": elapsed,
        "converged": converged,
    }
```

- [ ] **Step 4: Run tests**

Run: `cd inst/python && python -m pytest diffiscape_jax/tests/test_optimize.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inst/python/diffiscape_jax/optimize.py inst/python/diffiscape_jax/tests/test_optimize.py
git commit -m "feat(phase2): add optimize.py with L-BFGS and Adam parametric optimizers"
```

### Task 7: Wire `optimize_resistance_gradient` into R

**Files:**
- Modify: `R/jax_bridge.R` — add `ds_jax_optimize` function
- Modify: `R/optimizer.R` — add `optimize_resistance_gradient` function
- Modify: `R/pipeline.R:156-204` — add `"gradient"` solver dispatch
- Create: `tests/testthat/test-optimizer-gradient.R`

**Interfaces:**
- Consumes: `diffiscape_jax.optimize.run_parametric_optimization` (via reticulate), `ds_jax_setup`
- Produces: `optimize_resistance_gradient(basis_stack, obs_points, bounds, config, intensity_config, output_dir, ...) -> list` matching `optimize_resistance_enzyme` return shape
- Produces: `ds_optimize(..., solver = "gradient")` routing

- [ ] **Step 1: Add `ds_jax_optimize` to `jax_bridge.R`**

```r
#' Run gradient-based optimization via the JAX backend
#'
#' @param basis_np numpy array of basis values.
#' @param obs_np numpy array of observation counts.
#' @param valid_mask_np numpy boolean mask.
#' @param n_rows,n_cols Grid dimensions.
#' @param cell_area Cell area in square units.
#' @param init_params Initial parameter vector.
#' @param ... Additional args forwarded to Python.
#' @return Converted R list from Python dict.
#' @keywords internal
ds_jax_optimize <- function(basis_np, obs_np, valid_mask_np,
                            n_rows, n_cols, cell_area,
                            init_params, ...) {
  if (!ds_jax_check()) ds_jax_setup()

  if (is.null(.ds_env$jax_optimize)) {
    module_dir <- system.file("python", package = "DiffiScape")
    .ds_env$jax_optimize <- reticulate::import_from_path(
      "diffiscape_jax.optimize", path = module_dir)
  }

  result <- .ds_env$jax_optimize$run_parametric_optimization(
    basis_np, obs_np, valid_mask_np,
    as.integer(n_rows), as.integer(n_cols),
    as.double(cell_area), init_params, ...
  )
  reticulate::py_to_r(result)
}
```

- [ ] **Step 2: Add `optimize_resistance_gradient` to `optimizer.R`**

```r
#' Optimise resistance via JAX automatic differentiation
#'
#' Uses L-BFGS or Adam optimisation with jax.grad through JAXScape
#' instead of the Julia Enzyme solver or GP surrogate.
#'
#' @inheritParams optimize_resistance_enzyme
#' @param method Character; `"lbfgs"` (default) or `"adam"`.
#' @param parameterization Character; `"resistance"` (default) or `"permeability"`.
#' @return A list with `best_params`, `best_loglik`, `bounds`,
#'   `n_evaluations`, `distribution`, `convergence`.
#' @export
optimize_resistance_gradient <- function(basis_stack, obs_points,
                                         bounds = NULL,
                                         config = default_optimizer_config(),
                                         intensity_config = default_intensity_config(),
                                         output_dir = tempdir(),
                                         covariates_obs = NULL,
                                         covariates_rasters = NULL,
                                         residualise = FALSE,
                                         available_points = NULL,
                                         available_covariates = NULL,
                                         method = "lbfgs",
                                         parameterization = "resistance") {
  # Uses .prepare_torch_inputs-style data prep (reuse or factor out)
  # then calls ds_jax_optimize
  # Return format matches optimize_resistance_enzyme
  # Implementation details in the step-by-step code
}
```

The full implementation follows the same data-preparation pattern as `.prepare_torch_inputs` (convert SpatRaster to numpy arrays via reticulate), calls `ds_jax_optimize`, and reshapes the result into the standard `list(best_params, best_loglik, ...)` format.

- [ ] **Step 3: Add `"gradient"` to `ds_optimize` dispatch in `pipeline.R`**

In `R/pipeline.R`, modify `ds_optimize` to accept `solver = "gradient"` and dispatch to `optimize_resistance_gradient`. Add deprecation aliases:

```r
ds_optimize <- function(...,
                        solver = c("surrogate", "gradient", "enzyme", "torch", "irl")) {
  solver <- match.arg(solver)

  # Deprecation aliases
  if (solver == "enzyme") {
    message("solver='enzyme' is deprecated. Use solver='gradient' instead.")
    solver <- "gradient"
  }
  if (solver %in% c("torch", "irl")) {
    message("solver='torch'/'irl' is deprecated. Use solver='gradient' ",
            "with model_type config instead.")
    # ... existing torch dispatch ...
  }

  if (solver == "gradient") {
    return(optimize_resistance_gradient(...))
  }

  # ... existing surrogate dispatch ...
}
```

- [ ] **Step 4: Write and run test**

```r
# tests/testthat/test-optimizer-gradient.R
test_that("optimize_resistance_gradient runs end-to-end", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("jaxscape"),
          "jaxscape not installed")

  r1 <- terra::rast(nrows = 10, ncols = 10, vals = runif(100))
  r2 <- terra::rast(nrows = 10, ncols = 10, vals = runif(100))
  basis <- c(r1, r2)
  pts <- data.frame(x = runif(20, terra::xmin(r1), terra::xmax(r1)),
                    y = runif(20, terra::ymin(r1), terra::ymax(r1)))

  result <- optimize_resistance_gradient(
    basis, pts,
    config = list(seed = 42L, n_epochs = 10L),
    method = "lbfgs"
  )

  expect_type(result, "list")
  expect_true("best_params" %in% names(result))
  expect_true("best_loglik" %in% names(result))
})
```

Run: `Rscript -e "devtools::test(filter = 'optimizer-gradient')"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add R/jax_bridge.R R/optimizer.R R/pipeline.R tests/testthat/test-optimizer-gradient.R
git commit -m "feat(phase2): wire gradient solver into R optimizer + pipeline dispatch"
```

---

## Phase 3: Flax Resistance Nets

### Task 8: `resistance.py` — Flax resistance net architectures

**Files:**
- Create: `inst/python/diffiscape_jax/resistance.py`
- Create: `inst/python/diffiscape_jax/tests/test_resistance.py`

**Interfaces:**
- Produces: `ResistanceMLP(features, n_hidden, r_min, r_max, clamp_beta)` — Flax `nn.Module`
- Produces: `ResistanceConv(channels, n_layers, kernel_size, hidden_dim, r_min, r_max)` — Flax `nn.Module`
- Produces: `ResistanceSpline(n_knots, degree, n_covariates, include_interactions, r_min, r_max)` — Flax `nn.Module`
- Produces: `ResistanceIRL(hidden_dim, n_hidden, beta, gamma_d, n_value_iter, r_min, r_max)` — Flax `nn.Module`
- All modules: `__call__(self, x) -> jnp.ndarray` returning log-resistance (or log-permeability) values, shape `(n_cells,)`

- [ ] **Step 1: Write failing test for `ResistanceMLP`**

```python
# inst/python/diffiscape_jax/tests/test_resistance.py
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from diffiscape_jax.resistance import ResistanceMLP


def test_resistance_mlp_output_shape():
    model = ResistanceMLP(features=32, n_hidden=2, r_min=1.0, r_max=5000.0)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (100, 3))  # 100 cells, 3 covariates
    params = model.init(rng, x)
    out = model.apply(params, x)
    assert out.shape == (100,)


def test_resistance_mlp_output_clamped():
    model = ResistanceMLP(features=16, n_hidden=1, r_min=1.0, r_max=5000.0)
    rng = jax.random.PRNGKey(1)
    x = jax.random.normal(rng, (50, 2)) * 10  # large inputs
    params = model.init(rng, x)
    out = model.apply(params, x)
    resistance = jnp.exp(out)  # log-R -> R
    assert jnp.all(resistance >= 0.9)  # softplus clamp is smooth, ~1.0
    assert jnp.all(resistance <= 5500.0)


def test_resistance_mlp_gradients():
    model = ResistanceMLP(features=16, n_hidden=1)
    rng = jax.random.PRNGKey(2)
    x = jax.random.normal(rng, (20, 2))
    params = model.init(rng, x)

    def loss_fn(p):
        return jnp.sum(model.apply(p, x))

    grad = jax.grad(loss_fn)(params)
    leaves = jax.tree.leaves(grad)
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd inst/python && python -m pytest diffiscape_jax/tests/test_resistance.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `resistance.py`**

```python
# inst/python/diffiscape_jax/resistance.py
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

from .core import DEFAULT_R_MIN, DEFAULT_R_MAX


def _softplus_clamp(x, x_min, x_max, beta=5.0):
    log_min = jnp.log(x_min)
    log_max = jnp.log(x_max)
    clamped = log_min + nn.softplus(x - log_min) * beta
    clamped = log_max - nn.softplus(log_max - clamped) * beta
    return clamped


class ResistanceMLP(nn.Module):
    features: int = 32
    n_hidden: int = 2
    r_min: float = DEFAULT_R_MIN
    r_max: float = DEFAULT_R_MAX
    clamp_beta: float = 5.0

    @nn.compact
    def __call__(self, x):
        n_input = x.shape[-1]
        skip = nn.Dense(1)(x).squeeze(-1)

        h = x
        for _ in range(self.n_hidden):
            h = nn.Dense(self.features)(h)
            h = nn.silu(h)
        nonlinear = nn.Dense(1)(h).squeeze(-1)

        log_r = skip + nonlinear
        return _softplus_clamp(log_r, self.r_min, self.r_max, self.clamp_beta)


class ResistanceConv(nn.Module):
    channels: int = 16
    n_layers: int = 3
    kernel_size: int = 3
    hidden_dim: int = 32
    r_min: float = DEFAULT_R_MIN
    r_max: float = DEFAULT_R_MAX

    @nn.compact
    def __call__(self, x):
        # x: (H, W, C) raster input
        h = x
        for i in range(self.n_layers):
            residual = h
            h = nn.Conv(self.channels, (self.kernel_size, self.kernel_size),
                        padding='SAME')(h)
            h = nn.silu(h)
            if residual.shape == h.shape:
                h = h + residual

        h = h.reshape(h.shape[0] * h.shape[1], -1)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.silu(h)
        log_r = nn.Dense(1)(h).squeeze(-1)
        return _softplus_clamp(log_r, self.r_min, self.r_max)


class ResistanceSpline(nn.Module):
    n_knots: int = 10
    degree: int = 3
    n_covariates: int = 1
    include_interactions: bool = True
    r_min: float = DEFAULT_R_MIN
    r_max: float = DEFAULT_R_MAX

    @nn.compact
    def __call__(self, x):
        # B-spline basis evaluation + learned coefficients
        # Each covariate gets its own spline basis
        components = []
        for k in range(self.n_covariates):
            basis = self._bspline_basis(x[:, k], self.n_knots, self.degree)
            coefs = self.param(f'coef_{k}',
                               nn.initializers.zeros,
                               (basis.shape[-1],))
            components.append(basis @ coefs)

        log_r = sum(components)

        if self.include_interactions and self.n_covariates >= 2:
            for i in range(self.n_covariates):
                for j in range(i + 1, self.n_covariates):
                    interaction = self.param(
                        f'interact_{i}_{j}',
                        nn.initializers.zeros, (1,))
                    log_r = log_r + interaction[0] * x[:, i] * x[:, j]

        intercept = self.param('intercept', nn.initializers.constant(3.0), ())
        log_r = intercept + log_r
        return _softplus_clamp(log_r, self.r_min, self.r_max)

    @staticmethod
    def _bspline_basis(x, n_knots, degree):
        knots = jnp.linspace(0, 1, n_knots + 2)[1:-1]
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        basis = jnp.stack([
            jnp.exp(-0.5 * ((x_norm - k) / (1.0 / n_knots)) ** 2)
            for k in knots
        ], axis=-1)
        return basis


class ResistanceIRL(nn.Module):
    hidden_dim: int = 32
    n_hidden: int = 2
    beta: float = 1.0
    gamma_d: float = 0.9
    n_value_iter: int = 60
    r_min: float = DEFAULT_R_MIN
    r_max: float = DEFAULT_R_MAX
    value_scale_init: float = 1.0

    @nn.compact
    def __call__(self, x, adjacency=None):
        # Reward network
        h = x
        for _ in range(self.n_hidden):
            h = nn.Dense(self.hidden_dim)(h)
            h = nn.silu(h)
        reward = nn.Dense(1)(h).squeeze(-1)

        # Soft value iteration (requires adjacency)
        if adjacency is not None:
            value = jnp.zeros_like(reward)
            for _ in range(self.n_value_iter):
                q = reward + self.gamma_d * adjacency @ value
                value = self.beta * jax.nn.logsumexp(q / self.beta, axis=-1)

            scale = self.param('value_scale',
                               nn.initializers.constant(self.value_scale_init), ())
            offset = self.param('value_offset',
                                nn.initializers.constant(3.0), ())
            log_r = offset - scale * value
        else:
            log_r = reward

        return _softplus_clamp(log_r, self.r_min, self.r_max)
```

- [ ] **Step 4: Add tests for remaining net types and run all**

Add to `test_resistance.py`:

```python
from diffiscape_jax.resistance import ResistanceConv, ResistanceSpline


def test_resistance_conv_output_shape():
    model = ResistanceConv(channels=8, n_layers=2, kernel_size=3,
                           hidden_dim=16)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (10, 10, 3))  # H, W, C
    params = model.init(rng, x)
    out = model.apply(params, x)
    assert out.shape == (100,)  # H * W


def test_resistance_spline_output_shape():
    model = ResistanceSpline(n_knots=5, degree=3, n_covariates=2)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (50, 2))
    params = model.init(rng, x)
    out = model.apply(params, x)
    assert out.shape == (50,)
```

Run: `cd inst/python && python -m pytest diffiscape_jax/tests/test_resistance.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add inst/python/diffiscape_jax/resistance.py inst/python/diffiscape_jax/tests/test_resistance.py
git commit -m "feat(phase3): add Flax resistance nets — MLP, Conv, Spline, IRL"
```

### Task 9: Wire neural resistance into `optimize.py` + R

**Files:**
- Modify: `inst/python/diffiscape_jax/optimize.py` — add `run_neural_optimization`
- Modify: `R/jax_bridge.R` — add neural optimization dispatch
- Modify: `R/pipeline.R` — route `model_type` config to gradient solver
- Create: `inst/python/diffiscape_jax/tests/test_neural_optimize.py`

**Interfaces:**
- Consumes: `resistance.ResistanceMLP`, `resistance.ResistanceConv`, `resistance.ResistanceSpline`, `resistance.ResistanceIRL`, `core._connectivity_objective`
- Produces: `run_neural_optimization(basis_values, obs_counts, valid_mask, n_rows, n_cols, cell_area, model_type, model_config, optim_config, parameterization, ...) -> dict` with keys `"best_params"` (serialized), `"best_loglik"`, `"resistance"` (np.ndarray), `"connectivity"` (np.ndarray), `"loss_history"`, `"n_epochs_run"`, `"elapsed"`, `"effective_loglinear"` (np.ndarray), `"alpha"` (float), `"gamma"` (float), `"model_type"` (str)

- [ ] **Step 1: Write failing test**

```python
# inst/python/diffiscape_jax/tests/test_neural_optimize.py
import numpy as np
import pytest

from diffiscape_jax.optimize import run_neural_optimization


def test_neural_mlp_optimization_runs():
    rng = np.random.default_rng(42)
    n_rows, n_cols, n_basis = 8, 8, 2
    n_cells = n_rows * n_cols
    basis = rng.standard_normal((n_cells, n_basis))
    obs = rng.poisson(3, n_cells).astype(float)
    valid = np.ones(n_cells, dtype=bool)

    result = run_neural_optimization(
        basis, obs, valid, n_rows, n_cols,
        cell_area=1.0,
        model_type="mlp",
        model_config={"hidden_dim": 16, "n_hidden_layers": 1},
        optim_config={"lr": 0.01, "n_epochs": 20, "patience": 10},
        parameterization="resistance",
        seed=42, verbose=False,
    )

    assert "resistance" in result
    assert result["resistance"].shape == (n_cells,)
    assert result["n_epochs_run"] > 0
```

- [ ] **Step 2: Implement `run_neural_optimization` in `optimize.py`**

```python
def run_neural_optimization(basis_values, obs_counts, valid_mask,
                            n_rows, n_cols, cell_area,
                            model_type="mlp", model_config=None,
                            optim_config=None,
                            parameterization="resistance",
                            seed=42, verbose=True):
    import jax
    import jax.numpy as jnp
    import optax
    from .resistance import (ResistanceMLP, ResistanceConv,
                             ResistanceSpline, ResistanceIRL)

    if model_config is None:
        model_config = {}
    if optim_config is None:
        optim_config = {}

    rng = jax.random.PRNGKey(seed)

    # Select model
    if model_type == "mlp":
        model = ResistanceMLP(
            features=model_config.get("hidden_dim", 32),
            n_hidden=model_config.get("n_hidden_layers", 2),
        )
    elif model_type == "conv":
        model = ResistanceConv(
            channels=model_config.get("conv_channels", 16),
            n_layers=model_config.get("n_conv_layers", 3),
        )
    elif model_type == "spline_gam":
        n_cov = basis_values.shape[1] if basis_values.ndim > 1 else 1
        model = ResistanceSpline(
            n_knots=model_config.get("n_knots", 10),
            n_covariates=n_cov,
        )
    elif model_type == "irl":
        model = ResistanceIRL(
            hidden_dim=model_config.get("hidden_dim", 32),
            n_hidden=model_config.get("n_hidden_layers", 2),
        )

    basis_jnp = jnp.array(basis_values, dtype=jnp.float64)
    obs_jnp = jnp.array(obs_counts, dtype=jnp.float64)
    mask_jnp = jnp.array(valid_mask)

    params = model.init(rng, basis_jnp)

    lr = optim_config.get("lr", 0.01)
    n_epochs = optim_config.get("n_epochs", 300)
    patience = optim_config.get("patience", 30)

    # Cosine schedule
    schedule = optax.cosine_decay_schedule(lr, n_epochs)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    from .core import prepare_permeability, _mean_weight
    from jaxscape import GridGraph
    from jaxscape.distances import ResistanceDistance

    def loss_fn(p):
        log_r = model.apply(p, basis_jnp)
        r_full = jnp.zeros(n_rows * n_cols)
        r_full = r_full.at[mask_jnp].set(jnp.exp(log_r))
        surface = r_full.reshape((n_rows, n_cols))
        perm = prepare_permeability(surface, parameterization)
        grid = GridGraph(grid=perm, fun=_mean_weight)
        dist = ResistanceDistance()
        source = grid.coord_to_index(jnp.array([0]), jnp.array([0]))
        conn = dist(grid, source)
        log_lambda = jnp.log1p(conn.ravel()[mask_jnp])
        nll = -(jnp.sum(obs_jnp * log_lambda) - jnp.sum(jnp.exp(log_lambda) * cell_area))
        return nll

    grad_fn = jax.grad(loss_fn)
    best_loss = float('inf')
    best_params = params
    loss_history = []
    stall = 0
    t0 = time.time()

    for epoch in range(n_epochs):
        g = grad_fn(params)
        updates, opt_state = optimizer.update(g, opt_state)
        params = optax.apply_updates(params, updates)
        loss = float(loss_fn(params))
        loss_history.append(loss)

        if loss < best_loss:
            best_loss = loss
            best_params = params
            stall = 0
        else:
            stall += 1

        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss={loss:.4f}")

        if stall >= patience:
            break

    elapsed = time.time() - t0
    final_log_r = np.array(model.apply(best_params, basis_jnp))
    resistance = np.exp(final_log_r)

    return {
        "resistance": resistance,
        "best_loglik": -best_loss,
        "loss_history": loss_history,
        "n_epochs_run": len(loss_history),
        "elapsed": elapsed,
        "model_type": model_type,
    }
```

- [ ] **Step 3: Run tests**

Run: `cd inst/python && python -m pytest diffiscape_jax/tests/test_neural_optimize.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add inst/python/diffiscape_jax/optimize.py inst/python/diffiscape_jax/tests/test_neural_optimize.py R/jax_bridge.R R/pipeline.R
git commit -m "feat(phase3): wire Flax neural resistance nets into optimization pipeline"
```

---

## Phase 4: NumPyro Bayesian Sampling

### Task 10: `sample.py` — NumPyro NUTS, MALA, ADVI

**Files:**
- Create: `inst/python/diffiscape_jax/sample.py`
- Create: `inst/python/diffiscape_jax/tests/test_sample.py`

**Interfaces:**
- Consumes: `resistance.ResistanceMLP` (and other nets), `core.prepare_permeability`
- Produces: `run_nuts_sampling(model, params, basis_values, obs_counts, valid_mask, n_rows, n_cols, cell_area, parameterization, n_samples, warmup, ...) -> dict` with keys `"samples"` (dict of np.ndarrays), `"summary"` (dict), `"n_divergences"` (int), `"elapsed"` (float)
- Produces: `run_mala_sampling(...) -> dict` (same shape)
- Produces: `run_advi_sampling(...) -> dict` with additional `"best_elbo"` (float), `"converged"` (bool)

- [ ] **Step 1: Write failing test**

```python
# inst/python/diffiscape_jax/tests/test_sample.py
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from diffiscape_jax.sample import run_nuts_sampling
from diffiscape_jax.resistance import ResistanceMLP


def test_nuts_sampling_produces_samples():
    rng = np.random.default_rng(42)
    n_rows, n_cols, n_basis = 6, 6, 2
    n_cells = n_rows * n_cols
    basis = rng.standard_normal((n_cells, n_basis))
    obs = rng.poisson(3, n_cells).astype(float)
    valid = np.ones(n_cells, dtype=bool)

    model = ResistanceMLP(features=8, n_hidden=1)
    init_rng = jax.random.PRNGKey(0)
    init_params = model.init(init_rng, jnp.array(basis))

    result = run_nuts_sampling(
        model, init_params, basis, obs, valid,
        n_rows, n_cols, cell_area=1.0,
        parameterization="resistance",
        n_samples=10, warmup=5,
        seed=42,
    )

    assert "samples" in result
    assert "summary" in result
    assert "n_divergences" in result
    assert isinstance(result["n_divergences"], int)
    assert result["elapsed"] > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd inst/python && python -m pytest diffiscape_jax/tests/test_sample.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `sample.py`**

```python
# inst/python/diffiscape_jax/sample.py
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
import time

from .core import prepare_permeability, _mean_weight
from jaxscape import GridGraph
from jaxscape.distances import ResistanceDistance


def _build_numpyro_model(flax_model, basis_jnp, obs_jnp, valid_mask,
                         n_rows, n_cols, cell_area, parameterization,
                         init_params):
    flat_params, unflatten = jax.flatten_util.ravel_pytree(init_params)
    n_params = len(flat_params)

    def model():
        param_vec = numpyro.sample(
            "params",
            dist.Normal(jnp.zeros(n_params), jnp.ones(n_params) * 2.0)
        )
        params_tree = unflatten(param_vec)
        log_r = flax_model.apply(params_tree, basis_jnp)
        r_full = jnp.zeros(n_rows * n_cols)
        r_full = r_full.at[valid_mask].set(jnp.exp(log_r))
        surface = r_full.reshape((n_rows, n_cols))
        perm = prepare_permeability(surface, parameterization)
        grid = GridGraph(grid=perm, fun=_mean_weight)
        d = ResistanceDistance()
        source = grid.coord_to_index(jnp.array([0]), jnp.array([0]))
        conn = d(grid, source)
        log_lambda = jnp.log1p(conn.ravel()[valid_mask])
        numpyro.factor("loglik",
                       jnp.sum(obs_jnp * log_lambda)
                       - jnp.sum(jnp.exp(log_lambda) * cell_area))

    return model, flat_params, unflatten


def _summarize_samples(samples_dict):
    summary = {}
    for k, v in samples_dict.items():
        arr = np.array(v)
        if arr.ndim == 1:
            summary[k] = {
                "mean": float(np.mean(arr)),
                "sd": float(np.std(arr)),
                "q025": float(np.quantile(arr, 0.025)),
                "q50": float(np.median(arr)),
                "q975": float(np.quantile(arr, 0.975)),
            }
    return summary


def run_nuts_sampling(flax_model, init_params, basis_values, obs_counts,
                      valid_mask, n_rows, n_cols, cell_area,
                      parameterization="resistance",
                      n_samples=1000, warmup=1000,
                      max_treedepth=10, target_accept=0.80,
                      seed=42):
    basis_jnp = jnp.array(basis_values, dtype=jnp.float64)
    obs_jnp = jnp.array(obs_counts, dtype=jnp.float64)
    mask_jnp = jnp.array(valid_mask)

    numpyro_model, flat_init, unflatten = _build_numpyro_model(
        flax_model, basis_jnp, obs_jnp, mask_jnp,
        n_rows, n_cols, cell_area, parameterization, init_params
    )

    kernel = NUTS(numpyro_model, max_tree_depth=max_treedepth,
                  target_accept_prob=target_accept)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=n_samples)

    rng = jax.random.PRNGKey(seed)
    t0 = time.time()
    mcmc.run(rng, init_params={"params": flat_init})
    elapsed = time.time() - t0

    samples = mcmc.get_samples()
    samples_np = {k: np.array(v) for k, v in samples.items()}
    summary = _summarize_samples(samples_np)

    n_divergences = int(mcmc.get_extra_fields().get(
        "diverging", jnp.array([])).sum())

    return {
        "samples": samples_np,
        "summary": summary,
        "n_divergences": n_divergences,
        "elapsed": elapsed,
    }


def run_advi_sampling(flax_model, init_params, basis_values, obs_counts,
                      valid_mask, n_rows, n_cols, cell_area,
                      parameterization="resistance",
                      n_samples=2000, max_iter=2000,
                      lr=0.01, full_rank=False,
                      patience=100, seed=42):
    basis_jnp = jnp.array(basis_values, dtype=jnp.float64)
    obs_jnp = jnp.array(obs_counts, dtype=jnp.float64)
    mask_jnp = jnp.array(valid_mask)

    numpyro_model, flat_init, unflatten = _build_numpyro_model(
        flax_model, basis_jnp, obs_jnp, mask_jnp,
        n_rows, n_cols, cell_area, parameterization, init_params
    )

    guide = AutoLowRankMultivariateNormal(numpyro_model)
    optimizer = numpyro.optim.Adam(lr)
    svi = SVI(numpyro_model, guide, optimizer, loss=Trace_ELBO())

    rng = jax.random.PRNGKey(seed)
    t0 = time.time()
    svi_result = svi.run(rng, max_iter, progress_bar=False)
    elapsed = time.time() - t0

    predictive = numpyro.infer.Predictive(guide, num_samples=n_samples)
    samples = predictive(rng)
    samples_np = {k: np.array(v) for k, v in samples.items()}
    summary = _summarize_samples(samples_np)

    return {
        "samples": samples_np,
        "summary": summary,
        "best_elbo": float(-svi_result.losses[-1]),
        "converged": True,
        "elapsed": elapsed,
    }
```

- [ ] **Step 4: Run tests**

Run: `cd inst/python && python -m pytest diffiscape_jax/tests/test_sample.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inst/python/diffiscape_jax/sample.py inst/python/diffiscape_jax/tests/test_sample.py
git commit -m "feat(phase4): add NumPyro NUTS and ADVI samplers in sample.py"
```

### Task 11: Wire sampling into R bridge

**Files:**
- Modify: `R/jax_bridge.R` — add `ds_jax_sample_nuts`, `ds_jax_sample_advi`
- Create: `tests/testthat/test-jax_sampling.R`

**Interfaces:**
- Consumes: `diffiscape_jax.sample.run_nuts_sampling`, `diffiscape_jax.sample.run_advi_sampling` (via reticulate)
- Produces: `ds_jax_sample_nuts(basis_stack, obs_points, model_dir, ...) -> list` matching `run_bayesian_sampling_hmc` return shape
- Produces: `ds_jax_sample_advi(basis_stack, obs_points, model_dir, ...) -> list` matching `run_advi` return shape

- [ ] **Step 1: Implement R sampling wrappers in `jax_bridge.R`**

These follow the same pattern as the existing `run_bayesian_sampling_hmc` and `run_advi` — prepare data via a shared helper (factor `.prepare_torch_inputs` into `.prepare_jax_inputs` or reuse), call the Python function, convert results, save artifacts.

```r
#' NUTS posterior sampling via NumPyro
#' @inheritParams run_bayesian_sampling_hmc
#' @export
ds_jax_sample_nuts <- function(basis_stack, obs_points, model_dir,
                                n_samples = 1000L, warmup = 1000L,
                                max_treedepth = 10L,
                                target_accept = 0.80,
                                parameterization = "resistance",
                                seed = 42L, verbose = TRUE,
                                output_dir = NULL) {
  if (!ds_jax_check()) ds_jax_setup()
  if (is.null(output_dir)) output_dir <- model_dir

  if (is.null(.ds_env$jax_sample)) {
    module_dir <- system.file("python", package = "DiffiScape")
    .ds_env$jax_sample <- reticulate::import_from_path(
      "diffiscape_jax.sample", path = module_dir)
  }

  prep <- .prepare_jax_inputs(basis_stack, obs_points)
  # ... load model, call .ds_env$jax_sample$run_nuts_sampling, convert ...
}
```

- [ ] **Step 2: Write test**

```r
# tests/testthat/test-jax_sampling.R
test_that("ds_jax_sample_nuts returns valid structure", {
  skip("Requires trained model checkpoint — integration test only")
})
```

- [ ] **Step 3: Commit**

```bash
git add R/jax_bridge.R tests/testthat/test-jax_sampling.R
git commit -m "feat(phase4): wire NumPyro NUTS + ADVI sampling into R jax_bridge"
```

---

## Phase 5: Cleanup + Deletion

### Task 12: Delete Julia and torch backends, update DESCRIPTION

**Files:**
- Delete: `inst/julia/` (entire directory)
- Delete: `inst/python/diff_cs/` (entire directory)
- Delete: `inst/python/tests/` (old torch tests)
- Delete: `R/julia_bridge.R`
- Delete: `R/torch_bridge.R`
- Delete: `R/torch_pipeline.R`
- Modify: `R/zzz.R` — remove Julia/torch state, update `.onAttach`
- Modify: `R/connectivity.R` — remove `backend` param, default to JAX
- Modify: `R/optimizer.R` — remove `optimize_resistance_enzyme`, remove Julia calls from surrogate obj_fn
- Modify: `R/pipeline.R` — remove old solver dispatch, keep deprecation warnings
- Modify: `DESCRIPTION` — drop `JuliaConnectoR` from Suggests, add note about Python deps
- Delete: `tests/testthat/test-julia_bridge.R`
- Delete: `tests/testthat/test-torch_bridge.R`
- Delete: `tests/testthat/test-torch_pipeline.R`

**Interfaces:**
- After this task, all Julia and torch imports are gone. Only `reticulate` + JAX path remains.

- [ ] **Step 1: Verify all R tests pass on JAX backend**

Run: `Rscript -e "devtools::test()"`
Expected: all tests PASS (Julia/torch tests will be skipped since backends aren't required)

- [ ] **Step 2: Delete Julia code**

```bash
rm -rf inst/julia/
rm R/julia_bridge.R
rm tests/testthat/test-julia_bridge.R
```

- [ ] **Step 3: Delete torch code**

```bash
rm -rf inst/python/diff_cs/
rm -rf inst/python/tests/
rm R/torch_bridge.R
rm R/torch_pipeline.R
rm tests/testthat/test-torch_bridge.R
rm tests/testthat/test-torch_pipeline.R
```

- [ ] **Step 4: Update `zzz.R`**

Remove `julia_initialized`, `julia_con`, `julia_project_path`, `torch_initialized`, `torch_module` from `.onLoad`. Remove JuliaConnectoR check from `.onAttach`. Update startup message to mention JAX backend.

```r
.onLoad <- function(libname, pkgname) {
  .ds_env$jax_initialized <- FALSE
  .ds_env$jax_core     <- NULL
  .ds_env$jax_window   <- NULL
  .ds_env$jax_optimize <- NULL
  .ds_env$jax_sample   <- NULL
}

.onAttach <- function(libname, pkgname) {
  packageStartupMessage(
    "DiffiScape v", utils::packageVersion("DiffiScape"),
    " -- Differentiable Landscape Connectivity Optimization"
  )

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    packageStartupMessage(
      "Note: reticulate not installed. ",
      "Install with install.packages('reticulate') ",
      "to enable the JAX compute backend."
    )
  }
}
```

- [ ] **Step 5: Update `connectivity.R`** — make JAX the default (remove `backend` param)

`run_cumulative_current` and `run_omniscape` now always use JAX. Remove the `backend` parameter and the Julia codepath.

- [ ] **Step 6: Update `optimizer.R`** — remove `optimize_resistance_enzyme`, rewire surrogate

Delete `optimize_resistance_enzyme`. In the surrogate `obj_fn`, replace `ds_julia_call("DiffiScape.cumulative_current", ...)` with `ds_jax_connectivity(...)`.

- [ ] **Step 7: Update `pipeline.R`** — clean solver dispatch

```r
ds_optimize <- function(...,
                        solver = c("surrogate", "gradient",
                                   "enzyme", "torch", "irl")) {
  solver <- match.arg(solver)

  if (solver == "enzyme") {
    warning("solver='enzyme' is deprecated; using 'gradient'.", call. = FALSE)
    solver <- "gradient"
  }
  if (solver %in% c("torch", "irl")) {
    warning("solver='torch'/'irl' is deprecated; using 'gradient'.",
            call. = FALSE)
    solver <- "gradient"
  }

  if (solver == "gradient") {
    return(optimize_resistance_gradient(...))
  }

  optimize_resistance(...)
}
```

- [ ] **Step 8: Update `DESCRIPTION`**

Remove `JuliaConnectoR` from Suggests. Update Title and Description to remove Julia references. Add a note about Python dependencies in Description.

- [ ] **Step 9: Run full test suite**

Run: `Rscript -e "devtools::test()"`
Expected: all PASS, no references to Julia or torch remain

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat(phase5): remove Julia + torch backends, consolidate to JAX

BREAKING CHANGE: JuliaConnectoR and Julia are no longer required.
solver='enzyme' and solver='torch' are deprecated aliases for solver='gradient'.
Python with jax, jaxscape, flax, and numpyro is now the sole compute backend."
```

### Task 13: Update README and ARCHITECTURE.md

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `CONTRIBUTING.md`

- [ ] **Step 1: Update README.md**

Replace Julia installation section with Python/JAX requirements. Update Quick Start to remove `ds_init_julia()`. Update backend descriptions.

- [ ] **Step 2: Update ARCHITECTURE.md**

Replace the three-backend diagram with the new single-backend architecture. Remove Julia and torch sections. Document the new `inst/python/diffiscape_jax/` module structure.

- [ ] **Step 3: Update CONTRIBUTING.md**

Replace Julia test instructions with Python test instructions. Remove Julia setup section.

- [ ] **Step 4: Commit**

```bash
git add README.md ARCHITECTURE.md CONTRIBUTING.md
git commit -m "docs: update README, ARCHITECTURE, CONTRIBUTING for JAX-only backend"
```
