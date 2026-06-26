# ============================================================================
# JAX bridge via reticulate
#
# Loads the bundled diffiscape_jax Python package and exposes a thin
# call interface analogous to the PyTorch bridge in torch_bridge.R.
# ============================================================================

# Internal module state (Task 4 will migrate these into .ds_env via zzz.R)
.jax_env <- new.env(parent = emptyenv())
.jax_env$initialized <- FALSE
.jax_env$core_module <- NULL
.jax_env$window_module <- NULL
.jax_env$optimize_module <- NULL


#' Initialise the JAX backend for DiffiScape
#'
#' Loads the bundled `diffiscape_jax` package via \pkg{reticulate},
#' verifying that the required Python packages (`jax`, `jaxlib`,
#' `jaxscape`, `numpy`) are available. Call once per R session before
#' any of the JAX-based solvers.
#'
#' @param python Optional path to a Python binary; passed to
#'   [reticulate::use_python()] before module import.
#' @param force Re-initialise even if the module is already loaded.
#' @return `TRUE` (invisibly) on success.
#' @export
ds_jax_setup <- function(python = NULL, force = FALSE) {

  if (.jax_env$initialized && !force) {
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
    .jax_env$core_module <- reticulate::import_from_path(
      "diffiscape_jax.core", path = module_dir)
    .jax_env$window_module <- reticulate::import_from_path(
      "diffiscape_jax.window", path = module_dir)
    .jax_env$initialized <- TRUE

    jax <- reticulate::import("jax")
    devices <- jax$devices()
    dev_str <- if (length(devices) > 0) {
      paste(vapply(devices, function(d) d$platform, character(1)),
            collapse = ", ")
    } else {
      "cpu"
    }
    message(sprintf("  JAX backend ready | devices: %s", dev_str))
    invisible(TRUE)
  }, error = function(e) {
    .jax_env$initialized <- FALSE
    .jax_env$core_module <- NULL
    .jax_env$window_module <- NULL
    .jax_env$optimize_module <- NULL
    stop("Failed to initialise JAX backend: ", conditionMessage(e),
         call. = FALSE)
  })
}


#' Check whether the JAX backend is loaded
#'
#' @return Logical; `TRUE` if [ds_jax_setup()] has been called
#'   successfully.
#' @export
ds_jax_check <- function() {
  isTRUE(.jax_env$initialized)
}


#' Call a function from the bundled JAX modules
#'
#' Thin wrapper that ensures the backend is initialised, then resolves the
#' named function on the cached Python module handle and forwards
#' arguments to it via reticulate.
#'
#' @param module_name Character; `"core"` or `"window"`.
#' @param fn_name Character; bare function name in the module
#'   (e.g. `"forward_solve"`).
#' @param ... Arguments forwarded to the Python function.
#' @return The value returned by Python (use [reticulate::py_to_r()] if a
#'   manual conversion is needed).
#' @export
ds_jax_call <- function(module_name, fn_name, ...) {
  if (!.jax_env$initialized) {
    stop("JAX backend not initialised. Call ds_jax_setup() first.",
         call. = FALSE)
  }

  mod <- switch(module_name,
    core   = .jax_env$core_module,
    window = .jax_env$window_module,
    stop("Unknown module: ", module_name,
         ". Use 'core' or 'window'.", call. = FALSE)
  )

  fn <- tryCatch(mod[[fn_name]], error = function(e) NULL)
  if (is.null(fn)) {
    stop("Function not found in diffiscape_jax.", module_name, ": ", fn_name,
         call. = FALSE)
  }
  fn(...)
}


#' Compute cumulative current via the JAX backend
#'
#' Drop-in replacement for [run_cumulative_current()] that uses JAXScape
#' instead of the Julia differentiable solver.
#'
#' @param resistance A single-layer [terra::SpatRaster] of resistance.
#' @param radius Integer; moving-window radius (default 13).
#' @param block_size Integer; source block side length (default 5).
#' @param source_from_resistance Logical (default `TRUE`).
#' @param parameterization Character; `"resistance"` (default) or
#'   `"permeability"`.
#' @param output Character; `"current"`, `"voltage"`, or `"both"`.
#' @return A list matching the [run_cumulative_current()] interface:
#'   \describe{
#'     \item{cum_current}{A [terra::SpatRaster] of cumulative current, or
#'       `NULL` if `output = "voltage"`.}
#'     \item{flow_potential}{A [terra::SpatRaster] of flow potential, or
#'       `NULL` if `output = "current"`.}
#'     \item{elapsed_seconds}{Numeric; wall-clock seconds.}
#'   }
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

  # terra values -> R matrix (row-major -> column-major)
  R_vec <- as.numeric(terra::values(resistance))
  R_mat <- matrix(R_vec, nrow = nrow_grid, ncol = ncol_grid, byrow = TRUE)
  R_mat[is.na(R_mat)] <- 0  # nodata -> zero resistance -> zero conductance

  R_np <- np$array(R_mat, dtype = np$float64)

  start <- Sys.time()

  result <- .jax_env$window_module$cumulative_current(
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
  elapsed  <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  message(sprintf("JAX circuit solver completed in %.1f s", elapsed))

  # Helper: Python matrix -> terra raster
  mat_to_rast <- function(mat, layer_name) {
    r <- terra::rast(resistance)
    terra::values(r) <- as.vector(t(mat))
    names(r) <- layer_name
    r
  }

  cum_rast  <- if (!is.null(result_r$current)) {
    mat_to_rast(result_r$current, "cum_current")
  } else {
    NULL
  }
  volt_rast <- if (!is.null(result_r$voltage)) {
    mat_to_rast(result_r$voltage, "flow_potential")
  } else {
    NULL
  }

  list(
    cum_current     = cum_rast,
    flow_potential  = volt_rast,
    elapsed_seconds = elapsed
  )
}


#' Run gradient-based parametric optimisation via the JAX backend
#'
#' Calls `diffiscape_jax.optimize.run_parametric_optimization`, which
#' minimises the negative log-likelihood of a connectivity-based
#' point-process model using L-BFGS or Adam with JAX auto-diff gradients.
#'
#' The optimize module is imported lazily and cached in `.jax_env$optimize_module`.
#'
#' @param basis_np Numpy array of basis values (n_valid_cells x n_basis).
#' @param obs_np Numpy array of observation counts per valid cell.
#' @param valid_mask_np Numpy boolean mask of shape (n_rows * n_cols,).
#' @param n_rows,n_cols Integer grid dimensions.
#' @param cell_area Numeric cell area in square map units.
#' @param init_params Numeric vector of initial parameters (intercept +
#'   basis coefficients).  If `NULL`, defaults to zeros.
#' @param ... Additional arguments forwarded to the Python function
#'   (e.g. `method`, `lr`, `n_epochs`, `patience`, `parameterization`,
#'   `link_fn`, `radius`, `block_size`, `seed`, `verbose`).
#' @return A list (converted from the Python dict) with `best_params`,
#'   `best_loglik`, `loss_history`, `n_epochs_run`, `elapsed`, `converged`.
#' @keywords internal
ds_jax_optimize <- function(basis_np, obs_np, valid_mask_np,
                            n_rows, n_cols, cell_area,
                            init_params = NULL, ...) {

  if (!ds_jax_check()) ds_jax_setup()

  if (is.null(.jax_env$optimize_module)) {
    module_dir <- system.file("python", package = "DiffiScape")
    .jax_env$optimize_module <- reticulate::import_from_path(
      "diffiscape_jax.optimize", path = module_dir)
  }

  np <- reticulate::import("numpy", convert = FALSE)
  init_np <- if (!is.null(init_params)) {
    np$array(as.double(init_params), dtype = np$float64)
  } else {
    NULL
  }

  result <- .jax_env$optimize_module$run_parametric_optimization(
    basis_np, obs_np, valid_mask_np,
    as.integer(n_rows), as.integer(n_cols),
    cell_area  = as.double(cell_area),
    init_params = init_np,
    ...
  )
  reticulate::py_to_r(result)
}


#' Install Python dependencies for the JAX backend
#'
#' Convenience wrapper around [reticulate::py_install()] that installs the
#' packages required by the `diffiscape_jax` module.
#'
#' @param method Installation method passed to [reticulate::py_install()]
#'   (`"auto"`, `"virtualenv"`, or `"conda"`).
#' @param envname Optional virtualenv/conda environment name.
#' @param gpu If `TRUE`, prints a note about GPU JAX installation, which
#'   requires manual steps for CUDA support.
#' @return Invisible `TRUE`.
#' @export
ds_install_jax_deps <- function(method  = "auto",
                                envname = NULL,
                                gpu     = FALSE) {

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required. ",
         "Install with install.packages('reticulate').",
         call. = FALSE)
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
