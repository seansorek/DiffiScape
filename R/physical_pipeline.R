#' Fit a physical-current resistance model through the installed Python core
#'
#' @param basis_stack Basis rasters in fitted covariate order.
#' @param obs_points Observation coordinates with x and y columns.
#' @param model_type Resistance model: loglinear, mlp, conv, or spline_gam (B-splines).
#' @param formulation Connectivity: global_absorption, boundary, or focal_current.
#' @param method Optimization method: adam or lbfgs (bounded L-BFGS-B).
#' @param model_config Named model-specific settings passed to Python.
#' @param bounds Optional named loglinear parameter bounds.
#' @param output_dir Directory for rasters and saved fit.
#' @param ... Optimizer and physical settings passed to Python fit.
#' @return A fitted result with raster fields and serializable python_fit state.
#' @export
run_jax_pipeline <- function(basis_stack, obs_points, model_type = "loglinear",
                             formulation = "global_absorption", method = "adam",
                             model_config = list(), bounds = NULL, output_dir = tempdir(), ...) {
  if (!ds_jax_check()) ds_jax_setup()
  prep <- .prepare_backend_inputs(basis_stack, obs_points)
  extra <- list(...)
  integer_keys <- c("n_epochs", "patience", "source_spacing", "radius", "block_size", "warmup_epochs", "seed")
  for (name in intersect(names(extra), integer_keys)) {
    if (length(extra[[name]]) != 1L || !is.finite(extra[[name]]) || extra[[name]] != as.integer(extra[[name]])) stop(name, " must be an integer", call. = FALSE)
    extra[[name]] <- as.integer(extra[[name]])
  }
  for (name in intersect(names(model_config), c("hidden_dim", "n_hidden_layers", "conv_channels", "n_conv_layers", "conv_kernel_size", "intensity_hidden", "n_knots", "spline_degree", "intensity_n_knots", "intensity_degree"))) {
    if (length(model_config[[name]]) != 1L || !is.finite(model_config[[name]]) || model_config[[name]] != as.integer(model_config[[name]])) stop(name, " must be an integer", call. = FALSE)
    model_config[[name]] <- as.integer(model_config[[name]])
  }
  if (!is.null(bounds)) {
    expected <- c("r_0", paste0("z_", seq_len(terra::nlyr(basis_stack))))
    if (!setequal(names(bounds), expected) || anyDuplicated(names(bounds))) stop("bounds must name r_0 and each z_1, z_2, ... coefficient", call. = FALSE)
    bounds <- bounds[expected]
    extra$lower_bounds <- vapply(bounds, `[`, numeric(1), 1)
    extra$upper_bounds <- vapply(bounds, `[`, numeric(1), 2)
  }
  result <- do.call(.jax_env$package$fit, c(list(
    basis_values = prep$basis_np, obs_counts = prep$obs_np, valid_mask = prep$vmask_np,
    n_rows = as.integer(prep$n_rows), n_cols = as.integer(prep$n_cols), cell_area = prep$cell_area,
    model_type = model_type, formulation = formulation, method = method,
    model_config = if (length(model_config)) model_config else NULL), extra))
  fitted <- reticulate::py_to_r(result)
  fitted$loss_history <- as.numeric(unlist(fitted$loss_history))
  as_raster <- function(values, name) {
    raster <- basis_stack[[1]]
    full <- rep(NA_real_, terra::ncell(raster))
    full[prep$valid_mask] <- as.numeric(values)
    terra::values(raster) <- full
    names(raster) <- name
    raster
  }
  resistance <- as_raster(fitted$resistance, "resistance")
  connectivity <- as_raster(as.numeric(fitted$current)[prep$valid_mask], "connectivity")
  intensity <- as_raster(fitted$intensity, "intensity")
  effective <- as.numeric(fitted$effective_loglinear)
  names(effective) <- c("r_0", paste0("z_", seq_len(terra::nlyr(basis_stack))))
  output <- list(best_params = as.data.frame(as.list(effective)), best_loglik = fitted$best_loglik,
    alpha = fitted$alpha, gamma = fitted$gamma, resistance_raster = resistance,
    connectivity_raster = connectivity, intensity_raster = intensity, loss_history = fitted$loss_history,
    best_epoch = which.min(fitted$loss_history), n_params = fitted$n_params,
    n_epochs_run = fitted$n_epochs_run, total_time = fitted$elapsed, method = paste0("jax_", model_type, "_circuit"),
    distribution = "poisson_parametric", effective_loglinear = effective, model_type = model_type,
    partial_effects = fitted$partial_effects, interaction_effects = fitted$interaction_effects, uq_results = NULL,
    intensity_params = c(alpha = fitted$alpha, gamma = fitted$gamma),
    intensity_se = c(alpha = NA_real_, gamma = NA_real_),
    convergence = if (isTRUE(fitted$converged)) 0L else 1L,
    convergence_message = fitted$convergence_message, n_evaluations = fitted$n_epochs_run,
    solver = formulation, bounds = bounds, python_fit = fitted)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  terra::writeRaster(resistance, file.path(output_dir, "optimized_resistance.tif"), overwrite = TRUE)
  terra::writeRaster(connectivity, file.path(output_dir, "optimized_connectivity.tif"), overwrite = TRUE)
  terra::writeRaster(intensity, file.path(output_dir, "predicted_intensity.tif"), overwrite = TRUE)
  saveRDS(output, file.path(output_dir, "optimization_results.rds"))
  output
}

.verify_jax_gradient <- function(basis_stack, crop_size, model_type, model_config,
                                 solver, absorption, radius = 3L, block_size = 2L) {
  if (!ds_jax_check()) ds_jax_setup()
  cr <- .crop_basis_for_gradient(basis_stack, crop_size)
  formulation <- switch(solver, global_absorption = "global_absorption", global = "boundary",
    boundary = "boundary", diff_omniscape = "focal_current", focal_current = "focal_current",
    stop("Unknown formulation: ", solver, call. = FALSE))
  module <- reticulate::import("diffiscape.physical", convert = FALSE)
  np <- reticulate::import("numpy", convert = FALSE)
  reticulate::py_to_r(module$verify_gradient(np$array(cr$basis_vals, dtype = np$float64),
    np$array(cr$valid, dtype = np$bool_), as.integer(cr$n_rows), as.integer(cr$n_cols),
    model_type = model_type, model_config = if (length(model_config)) model_config else NULL,
    formulation = formulation, absorption = absorption, radius = as.integer(radius),
    block_size = as.integer(block_size)))
}
