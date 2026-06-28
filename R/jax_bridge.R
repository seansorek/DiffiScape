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
.jax_env$sample_module <- NULL


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
    .jax_env$sample_module <- NULL
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
#' Core connectivity solver that uses JAXScape's differentiable circuit
#' solver via the JAX backend.
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


#' Run neural-network resistance optimisation via the JAX backend
#'
#' Calls `diffiscape_jax.optimize.run_neural_optimization`, which
#' trains a Flax neural network (MLP, Conv, Spline-GAM, or IRL)
#' to produce a resistance surface that maximises the connectivity-based
#' point-process log-likelihood.
#'
#' The optimize module is imported lazily and cached in `.jax_env$optimize_module`.
#'
#' @param basis_np Numpy array of basis / covariate values.
#' @param obs_np Numpy array of observation counts per valid cell.
#' @param valid_mask_np Numpy boolean mask of shape (n_rows * n_cols,).
#' @param n_rows,n_cols Integer grid dimensions.
#' @param cell_area Numeric cell area in square map units.
#' @param model_type Character; `"mlp"` (default), `"conv"`,
#'   `"spline_gam"`, or `"irl"`.
#' @param model_config Named list of model-specific parameters forwarded
#'   to the Flax module constructor.
#' @param optim_config Named list with `lr`, `n_epochs`, `patience`.
#' @param ... Additional arguments forwarded to the Python function
#'   (e.g. `parameterization`, `seed`, `verbose`).
#' @return A list (converted from the Python dict) with `resistance`,
#'   `best_loglik`, `loss_history`, `n_epochs_run`, `elapsed`, `model_type`.
#' @keywords internal
ds_jax_neural_optimize <- function(basis_np, obs_np, valid_mask_np,
                                    n_rows, n_cols, cell_area,
                                    model_type = "mlp",
                                    model_config = list(),
                                    optim_config = list(), ...) {

  if (!ds_jax_check()) ds_jax_setup()

  if (is.null(.jax_env$optimize_module)) {
    module_dir <- system.file("python", package = "DiffiScape")
    .jax_env$optimize_module <- reticulate::import_from_path(
      "diffiscape_jax.optimize", path = module_dir)
  }

  # Convert R lists to Python dicts via reticulate
  model_cfg_py <- if (length(model_config) > 0) model_config else NULL
  optim_cfg_py <- if (length(optim_config) > 0) optim_config else NULL

  result <- .jax_env$optimize_module$run_neural_optimization(
    basis_np, obs_np, valid_mask_np,
    as.integer(n_rows), as.integer(n_cols),
    cell_area    = as.double(cell_area),
    model_type   = model_type,
    model_config = model_cfg_py,
    optim_config = optim_cfg_py,
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

  pkgs <- c("jax", "jaxlib", "jaxscape", "numpy", "scipy",
             "jaxopt", "optax", "flax", "numpyro")
  if (isTRUE(gpu)) {
    message("Note: for GPU support, install JAX with CUDA following ",
            "https://jax.readthedocs.io/en/latest/installation.html")
  }

  message("Installing Python packages: ", paste(pkgs, collapse = ", "))
  reticulate::py_install(pkgs, method = method, envname = envname)
  invisible(TRUE)
}


# ============================================================================
# Shared data-preparation helper
# ============================================================================

#' Prepare inputs for JAX-backed sampling and optimisation
#'
#' Converts a SpatRaster basis stack and observation coordinates into numpy
#' arrays suitable for the bundled `diffiscape_jax` Python modules.  Mirrors
#' the internal `.prepare_torch_inputs()` helper so any downstream R code can
#' treat both backends symmetrically.
#'
#' @param basis_stack A [terra::SpatRaster] with K covariate layers.
#' @param obs_points Data.frame with `x, y` columns (projected coords).
#' @return A named list with:
#'   \describe{
#'     \item{basis_np}{numpy array, shape (n_valid, K).}
#'     \item{obs_np}{numpy array of observation counts per valid cell.}
#'     \item{vmask_np}{numpy boolean mask, shape (n_rows * n_cols,).}
#'     \item{valid_mask}{Logical vector (R side).}
#'     \item{n_rows, n_cols}{Integer grid dimensions.}
#'     \item{cell_area}{Numeric cell area in map units squared.}
#'     \item{n_valid}{Number of valid (non-NA) cells.}
#'     \item{n_obs}{Total observation count.}
#'   }
#' @keywords internal
.prepare_jax_inputs <- function(basis_stack, obs_points) {

  np <- reticulate::import("numpy", convert = FALSE)

  n_rows    <- terra::nrow(basis_stack)
  n_cols    <- terra::ncol(basis_stack)
  cell_area <- prod(terra::res(basis_stack))

  basis_matrix <- as.matrix(basis_stack)
  valid_mask   <- stats::complete.cases(basis_matrix)
  basis_values <- basis_matrix[valid_mask, , drop = FALSE]

  cell_indices <- terra::cellFromXY(
    basis_stack,
    cbind(obs_points$x, obs_points$y)
  )
  valid_obs <- !is.na(cell_indices) & valid_mask[cell_indices]
  if (any(!valid_obs)) {
    message(sprintf("    Dropped %d obs outside valid cells",
                    sum(!valid_obs)))
  }
  cell_indices <- cell_indices[valid_obs]

  obs_table        <- table(cell_indices)
  obs_counts_full  <- rep(0L, terra::ncell(basis_stack))
  obs_counts_full[as.integer(names(obs_table))] <- as.integer(obs_table)
  obs_counts_valid <- obs_counts_full[valid_mask]

  list(
    basis_np   = np$array(basis_values, dtype = np$float64),
    obs_np     = np$array(as.double(obs_counts_valid), dtype = np$float64),
    vmask_np   = np$array(valid_mask, dtype = np$bool_),
    valid_mask = valid_mask,
    n_rows     = n_rows,
    n_cols     = n_cols,
    cell_area  = cell_area,
    n_valid    = sum(valid_mask),
    n_obs      = sum(obs_counts_valid)
  )
}


# ============================================================================
# Bayesian samplers via NumPyro
# ============================================================================

#' Build a Flax resistance model and initialise parameters
#'
#' Internal helper that constructs the appropriate Flax `nn.Module` from
#' `model_type` and `model_config`, then initialises parameters with a dummy
#' forward pass.
#'
#' @param basis_np numpy array of basis values (used for model init shape).
#' @param model_type Character; `"mlp"`, `"conv"`, `"spline_gam"`, or `"irl"`.
#' @param model_config Named list of model-specific knobs (forwarded to the
#'   Flax module constructor).
#' @param seed Integer; PRNG seed for parameter initialisation.
#' @return A named list with `model` (Flax Module) and `init_params` (pytree).
#' @keywords internal
.build_flax_model <- function(basis_np, model_type, model_config, seed) {

  jax <- reticulate::import("jax", convert = FALSE)
  resistance_mod <- reticulate::import_from_path(
    "diffiscape_jax.resistance",
    path = system.file("python", package = "DiffiScape")
  )

  cfg <- if (length(model_config) > 0) model_config else list()

  model <- switch(model_type,
    mlp = resistance_mod$ResistanceMLP(
      features = as.integer(cfg$hidden_dim %||% 32L),
      n_hidden = as.integer(cfg$n_hidden_layers %||% 2L)
    ),
    conv = resistance_mod$ResistanceConv(
      channels = as.integer(cfg$conv_channels %||% 16L),
      n_layers = as.integer(cfg$n_conv_layers %||% 3L)
    ),
    spline_gam = {
      # Infer n_covariates from basis shape
      np <- reticulate::import("numpy", convert = FALSE)
      shape <- reticulate::py_to_r(np$array(basis_np$shape))
      n_cov <- if (length(shape) > 1) shape[2] else 1L
      resistance_mod$ResistanceSpline(
        n_knots     = as.integer(cfg$n_knots %||% 10L),
        n_covariates = as.integer(n_cov)
      )
    },
    irl = resistance_mod$ResistanceIRL(
      hidden_dim = as.integer(cfg$hidden_dim %||% 32L),
      n_hidden   = as.integer(cfg$n_hidden_layers %||% 2L)
    ),
    stop("Unknown model_type: ", model_type,
         ". Use 'mlp', 'conv', 'spline_gam', or 'irl'.", call. = FALSE)
  )

  rng_key <- jax$random$PRNGKey(as.integer(seed))
  init_params <- model$`init`(rng_key, basis_np)

  list(model = model, init_params = init_params)
}


#' NUTS posterior sampling via NumPyro
#'
#' Draws posterior samples for the parameters of a Flax neural-network
#' resistance model using the No-U-Turn Sampler implemented in NumPyro.
#' A `Normal(0, 2)` prior is placed on each (flattened) model parameter, and
#' the likelihood chains through JAXScape circuit-theory resistance distances.
#'
#' @param basis_stack A [terra::SpatRaster] with K covariate layers.
#' @param obs_points Data.frame with `x, y` columns (projected coords).
#' @param n_samples Integer; number of posterior draws (default 1000).
#' @param warmup Integer; warm-up / adaptation steps (default 1000).
#' @param max_treedepth Integer; NUTS maximum tree depth (default 10).
#' @param target_accept Numeric; target acceptance probability (default 0.80).
#' @param parameterization Character; `"resistance"` (default) or
#'   `"permeability"`.
#' @param model_type Character; Flax model architecture -- `"mlp"` (default),
#'   `"conv"`, `"spline_gam"`, or `"irl"`.
#' @param model_config Named list of model-specific knobs forwarded to the
#'   Flax module constructor (e.g., `list(hidden_dim = 32L)`).
#' @param seed Integer; random seed (default 42).
#' @param verbose Logical; print progress messages (default `TRUE`).
#' @param output_dir Character; directory for saving MCMC artifacts.
#'   Defaults to `tempdir()`.
#' @return A list with:
#'   \describe{
#'     \item{samples_effective_loglinear}{Matrix of posterior samples (or
#'       `NULL` if unavailable from the sampler output).}
#'     \item{samples_alpha}{`NULL` (placeholder for compatibility).}
#'     \item{samples_gamma}{`NULL` (placeholder for compatibility).}
#'     \item{partial_effects}{`NULL` (placeholder for compatibility).}
#'     \item{log_posterior_trace}{`NULL` (placeholder for compatibility).}
#'     \item{ess}{`NULL` (placeholder for compatibility).}
#'     \item{summary}{Per-parameter posterior summaries.}
#'     \item{elapsed_time}{Numeric; wall-clock seconds.}
#'     \item{n_divergences}{Integer; number of divergent transitions.}
#'   }
#' @export
ds_jax_sample_nuts <- function(basis_stack,
                                obs_points,
                                n_samples       = 1000L,
                                warmup          = 1000L,
                                max_treedepth   = 10L,
                                target_accept   = 0.80,
                                parameterization = "resistance",
                                model_type      = "mlp",
                                model_config    = list(),
                                seed            = 42L,
                                verbose         = TRUE,
                                output_dir      = NULL) {

  if (!ds_jax_check()) ds_jax_setup()

  if (is.null(output_dir)) output_dir <- tempdir()
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  # Lazy-load the sample module

  if (is.null(.jax_env$sample_module)) {
    module_dir <- system.file("python", package = "DiffiScape")
    .jax_env$sample_module <- reticulate::import_from_path(
      "diffiscape_jax.sample", path = module_dir)
  }

  if (verbose) message("\n  Preparing data for NUTS sampling (JAX)...")
  prep <- .prepare_jax_inputs(basis_stack, obs_points)
  if (verbose) {
    message(sprintf("    Grid: %d x %d (%d valid cells)",
                    prep$n_rows, prep$n_cols, prep$n_valid))
    message(sprintf("    Observations: %d GPS fixes", prep$n_obs))
  }

  # Build the Flax model and initialise parameters
  if (verbose) message(sprintf("    Model type: %s", model_type))
  flax_built <- .build_flax_model(prep$basis_np, model_type, model_config, seed)

  if (verbose) {
    message(sprintf("\n  Starting NUTS (No-U-Turn Sampler) via NumPyro..."))
    message(sprintf("    Samples: %d, Warmup: %d -> %d total iterations",
                    n_samples, warmup, warmup + n_samples))
  }

  result <- .jax_env$sample_module$run_nuts_sampling(
    flax_model       = flax_built$model,
    init_params      = flax_built$init_params,
    basis_values     = prep$basis_np,
    obs_counts       = prep$obs_np,
    valid_mask       = prep$vmask_np,
    n_rows           = as.integer(prep$n_rows),
    n_cols           = as.integer(prep$n_cols),
    cell_area        = as.double(prep$cell_area),
    parameterization = parameterization,
    n_samples        = as.integer(n_samples),
    warmup           = as.integer(warmup),
    max_treedepth    = as.integer(max_treedepth),
    target_accept    = as.double(target_accept),
    seed             = as.integer(seed)
  )

  results_r <- reticulate::py_to_r(result)

  if (verbose) {
    message(sprintf("  NUTS completed in %.1f s (%d divergent transitions)",
                    results_r$elapsed, results_r$n_divergences))
  }

  if (!is.null(results_r$n_divergences) && results_r$n_divergences > 0) {
    warning(sprintf(
      "NUTS reported %d divergent transitions. Consider increasing target_accept.",
      results_r$n_divergences
    ))
  }

  # Reshape to match run_bayesian_sampling_hmc return format
  out <- list(
    samples_effective_loglinear = results_r$samples$params,
    samples_alpha               = NULL,
    samples_gamma               = NULL,
    partial_effects             = NULL,
    log_posterior_trace          = NULL,
    ess                         = NULL,
    summary                     = results_r$summary,
    elapsed_time                = results_r$elapsed,
    n_divergences               = results_r$n_divergences
  )

  # Persist artifacts
  .save_jax_mcmc_artifacts(out, output_dir)

  out
}


#' ADVI posterior sampling via NumPyro
#'
#' Fits a low-rank multivariate normal variational posterior using NumPyro's
#' Stochastic Variational Inference (SVI) with Trace ELBO, then draws
#' posterior samples from the fitted guide.
#'
#' @inheritParams ds_jax_sample_nuts
#' @param n_samples Integer; number of posterior draws (default 2000).
#' @param max_iter Integer; maximum SVI iterations (default 2000).
#' @param lr Numeric; Adam learning rate for the ELBO (default 0.01).
#' @return A list with:
#'   \describe{
#'     \item{samples_effective_loglinear}{Matrix of posterior samples (or
#'       `NULL` if unavailable).}
#'     \item{samples_alpha}{`NULL` (placeholder).}
#'     \item{samples_gamma}{`NULL` (placeholder).}
#'     \item{partial_effects}{`NULL` (placeholder).}
#'     \item{log_posterior_trace}{`NULL` (placeholder).}
#'     \item{ess}{`NULL` (placeholder).}
#'     \item{summary}{Per-parameter posterior summaries.}
#'     \item{elapsed_time}{Numeric; wall-clock seconds.}
#'     \item{n_divergences}{Integer 0 (ADVI has no divergences).}
#'     \item{best_elbo}{Numeric; best (final) ELBO estimate.}
#'     \item{converged}{Logical; whether the ELBO stabilised.}
#'   }
#' @export
ds_jax_sample_advi <- function(basis_stack,
                                obs_points,
                                n_samples       = 2000L,
                                max_iter        = 2000L,
                                lr              = 0.01,
                                parameterization = "resistance",
                                model_type      = "mlp",
                                model_config    = list(),
                                seed            = 42L,
                                verbose         = TRUE,
                                output_dir      = NULL) {

  if (!ds_jax_check()) ds_jax_setup()

  if (is.null(output_dir)) output_dir <- tempdir()
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  # Lazy-load the sample module
  if (is.null(.jax_env$sample_module)) {
    module_dir <- system.file("python", package = "DiffiScape")
    .jax_env$sample_module <- reticulate::import_from_path(
      "diffiscape_jax.sample", path = module_dir)
  }

  if (verbose) message("\n  Preparing data for ADVI (JAX)...")
  prep <- .prepare_jax_inputs(basis_stack, obs_points)
  if (verbose) {
    message(sprintf("    Grid: %d x %d (%d valid cells)",
                    prep$n_rows, prep$n_cols, prep$n_valid))
    message(sprintf("    Observations: %d GPS fixes", prep$n_obs))
  }

  # Build the Flax model and initialise parameters
  if (verbose) message(sprintf("    Model type: %s", model_type))
  flax_built <- .build_flax_model(prep$basis_np, model_type, model_config, seed)

  if (verbose) {
    message(sprintf("\n  Starting ADVI via NumPyro..."))
    message(sprintf("    Max iterations: %d, LR: %.4f", max_iter, lr))
  }

  result <- .jax_env$sample_module$run_advi_sampling(
    flax_model       = flax_built$model,
    init_params      = flax_built$init_params,
    basis_values     = prep$basis_np,
    obs_counts       = prep$obs_np,
    valid_mask       = prep$vmask_np,
    n_rows           = as.integer(prep$n_rows),
    n_cols           = as.integer(prep$n_cols),
    cell_area        = as.double(prep$cell_area),
    parameterization = parameterization,
    n_samples        = as.integer(n_samples),
    max_iter         = as.integer(max_iter),
    lr               = as.double(lr),
    seed             = as.integer(seed)
  )

  results_r <- reticulate::py_to_r(result)

  if (verbose) {
    message(sprintf("  ADVI completed in %.1f s | best ELBO = %.2f | converged = %s",
                    results_r$elapsed,
                    results_r$best_elbo,
                    if (isTRUE(results_r$converged)) "TRUE" else "FALSE"))
  }

  if (!is.null(results_r$converged) && !isTRUE(results_r$converged)) {
    warning("ADVI did not converge. Consider increasing max_iter or adjusting lr.")
  }

  # Reshape to match run_advi return format
  out <- list(
    samples_effective_loglinear = results_r$samples$params,
    samples_alpha               = NULL,
    samples_gamma               = NULL,
    partial_effects             = NULL,
    log_posterior_trace          = NULL,
    ess                         = NULL,
    summary                     = results_r$summary,
    elapsed_time                = results_r$elapsed,
    n_divergences               = 0L,
    best_elbo                   = results_r$best_elbo,
    converged                   = results_r$converged
  )

  # Persist artifacts
  .save_jax_mcmc_artifacts(out, output_dir)

  out
}


#' Save JAX MCMC artifacts to disk
#'
#' Persists `mcmc_results.rds` and `posterior_summary.csv` from the NumPyro
#' sampler output.  Handles both scalar and vector summary formats returned
#' by `_summarize_samples`.
#'
#' @param results_r List; the R-converted sampler output.
#' @param output_dir Character; directory for saved files.
#' @keywords internal
.save_jax_mcmc_artifacts <- function(results_r, output_dir) {
  saveRDS(results_r, file.path(output_dir, "mcmc_results.rds"))
  message("  Saved mcmc_results.rds")

  if (!is.null(results_r$summary)) {
    rows <- list()
    for (nm in names(results_r$summary)) {
      s <- results_r$summary[[nm]]
      # Vector parameters (e.g. "params") expand to one row per element
      if (is.list(s$mean) || length(s$mean) > 1) {
        n_elem <- length(s$mean)
        for (i in seq_len(n_elem)) {
          rows[[length(rows) + 1L]] <- data.frame(
            parameter = paste0(nm, "[", i, "]"),
            mean   = s$mean[[i]],
            sd     = s$sd[[i]],
            q025   = s$q025[[i]],
            median = s$q50[[i]],
            q975   = s$q975[[i]],
            ess    = NA_real_,
            stringsAsFactors = FALSE
          )
        }
      } else {
        rows[[length(rows) + 1L]] <- data.frame(
          parameter = nm,
          mean   = s$mean,
          sd     = s$sd,
          q025   = s$q025,
          median = s$q50,
          q975   = s$q975,
          ess    = NA_real_,
          stringsAsFactors = FALSE
        )
      }
    }
    if (length(rows) > 0) {
      sum_df <- do.call(rbind, rows)
      utils::write.csv(sum_df,
                       file.path(output_dir, "posterior_summary.csv"),
                       row.names = FALSE)
      message("  Saved posterior_summary.csv")
    }
  }
}
