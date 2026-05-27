# ============================================================================
# PyTorch neural-network resistance pipeline
#
# R wrappers around inst/python/diff_cs/05_torch_pipeline.py:
#   - run_torch_pipeline()           MAP optimisation via Adam
#   - run_bayesian_sampling()        Langevin / MALA posterior
#   - run_bayesian_sampling_hmc()    HMC / NUTS posterior
#   - run_advi()                     ADVI variational posterior
#   - verify_torch_gradient()        Adjoint vs autograd gradient check
#   - verify_conv_gradient()         Conv resistance net gradient check
#   - verify_spline_gradient()       Spline-GAM resistance gradient check
#
# All functions assume ds_torch_setup() has been called.
# ============================================================================


# --------------- Data-preparation helper ------------------------------------
#
# Builds the numpy inputs the Python backend expects from a SpatRaster of
# basis layers and a data.frame of observation coordinates. Shared by every
# entry point in this file.
#
#' @keywords internal
.prepare_torch_inputs <- function(basis_stack, obs_points) {

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


# --------------- Torch pipeline configuration --------------------------------

#' Default PyTorch pipeline configuration
#'
#' Returns a named list of model hyperparameters used by
#' [run_torch_pipeline()]. All parameters may be overridden by modifying
#' the returned list and passing it via the `config` argument.
#'
#' Workflow/output parameters (`verbose`, `output_dir`, `warm_start_theta`,
#' `absorption_schedule`, `compute_uq`, `uq_block_only`) are intentionally
#' excluded — pass those directly to [run_torch_pipeline()].
#'
#' @return A named list of hyperparameters.
#' @export
default_torch_config <- function() {
  list(
    # --- Neural-network architecture ---
    hidden_dim              = 32L,
    n_hidden_layers         = 2L,

    # --- Optimiser ---
    lr                      = 0.01,
    weight_decay            = 1e-4,
    n_epochs                = 300L,
    patience                = 30L,
    grad_clip               = 10.0,
    warmup_epochs           = 10L,

    # --- Solver ---
    solver                  = "diff_omniscape",
    radius                  = 15L,
    block_size              = 10L,
    focal_fraction          = 0.5,
    absorption              = 0.01,
    cg_tol                  = 1e-6,
    source_spacing          = 5L,
    source_from_resistance  = TRUE,

    # --- Regularisation ---
    reg_mean                = 1.0,
    reg_var                 = 0.1,
    target_logR_var         = 1.0,
    reg_skip                = 0.1,
    log_R_baseline          = 3.0,

    # --- Misc ---
    seed                    = 42L,
    device                  = "auto",

    # --- Conv architecture ---
    use_conv                = FALSE,
    n_conv_layers           = 3L,
    conv_channels           = 16L,
    conv_kernel_size        = 3L,
    dropout                 = 0.0,
    use_dilated             = TRUE,
    intensity_hidden        = 0L,

    # --- Spline-GAM model ---
    model_type              = "mlp",
    n_knots                 = 10L,
    spline_degree           = 3L,
    include_interactions    = TRUE,
    penalty_scale           = 1.0,
    lambda_init_marginal    = 0.0,
    lambda_init_interaction = 2.0,
    lambda_min              = 0.0,

    # --- Intensity spline ---
    intensity_spline        = FALSE,
    intensity_n_knots       = 5L,
    intensity_degree        = 3L,
    lambda_init_intensity   = 2.0,
    intensity_log1p_max     = 10.0
  )
}


# --------------- MAP optimisation -------------------------------------------

#' Optimise resistance with a PyTorch neural network
#'
#' Trains a neural-network resistance surface (MLP / Conv / Spline-GAM) by
#' backpropagating through a differentiable circuit solver, maximising a
#' Poisson point-process log-likelihood with smoothness regularisation.
#'
#' Wraps `run_torch_optimization()` in the bundled Python module
#' `diff_cs/05_torch_pipeline.py`. Requires [ds_torch_setup()] to have been
#' called.
#'
#' @param basis_stack A [terra::SpatRaster] with K covariate layers.
#' @param obs_points Data.frame with `x, y` columns (projected coords).
#' @param config A named list of hyperparameters; see [default_torch_config()].
#'   Individual parameters may still be overridden by passing them directly.
#' @param hidden_dim Integer; MLP hidden width.
#' @param n_hidden_layers Integer; number of MLP hidden layers.
#' @param lr Numeric; Adam learning rate.
#' @param weight_decay Numeric; L2 weight decay.
#' @param n_epochs Integer; max training epochs.
#' @param patience Integer; early-stopping patience.
#' @param grad_clip Numeric; gradient-norm clip.
#' @param cg_tol Numeric; CG solver tolerance.
#' @param source_spacing Integer; circuit source lattice spacing.
#' @param source_from_resistance Logical; weight sources by resistance.
#' @param reg_mean Numeric; weight on `(mean(log R) - log_R_baseline)^2`.
#' @param reg_var Numeric; weight on a variance hinge penalty in `log R`.
#' @param target_logR_var Numeric; activation threshold for the var hinge.
#' @param reg_skip Numeric; weight on a skip-connection magnitude penalty.
#' @param log_R_baseline Numeric; target mean of `log R`.
#' @param warm_start_theta Optional numeric vector of log-linear params
#'   `(r_0, z_1, ..., z_K)` to warm-start from.
#' @param output_dir Character; directory for saved rasters and
#'   checkpoints.
#' @param seed Integer; RNG seed.
#' @param verbose Logical.
#' @param solver Character; `"diff_omniscape"`, `"global"`, or
#'   `"global_absorption"`.
#' @param radius Integer; Omniscape focal radius.
#' @param block_size Integer; Omniscape focal block size.
#' @param focal_fraction Numeric; stochastic focal sampling fraction.
#' @param absorption Numeric; absorption rate for the global solver.
#' @param device Character; `"auto"`, `"cuda"`, or `"cpu"`.
#' @param use_conv Logical; enable ConvResistanceNet.
#' @param n_conv_layers Integer; Conv2d residual blocks.
#' @param conv_channels Integer; conv feature channels.
#' @param conv_kernel_size Integer; conv kernel size.
#' @param dropout Numeric; dropout rate in the MLP head.
#' @param use_dilated Logical; dilated convolutions.
#' @param intensity_hidden Integer; learned intensity MLP width
#'   (`0` = parametric).
#' @param warmup_epochs Integer; linear LR warm-up epochs.
#' @param absorption_schedule Optional numeric vector for an absorption
#'   curriculum.
#' @param model_type Character; `"mlp"` or `"spline_gam"`.
#' @param n_knots Integer; spline knots per covariate.
#' @param spline_degree Integer; B-spline degree.
#' @param include_interactions Logical; tensor-product interactions.
#' @param penalty_scale Numeric; smoothing-penalty multiplier.
#' @param lambda_init_marginal Numeric; init marginal smoothing param.
#' @param lambda_init_interaction Numeric; init interaction smoothing.
#' @param lambda_min Numeric; floor on smoothing parameters.
#' @param compute_uq Logical; compute partial-effect uncertainty.
#' @param uq_block_only Logical; restrict UQ to spline blocks.
#' @param intensity_spline Logical; spline transform on connectivity in
#'   intensity model.
#' @param intensity_n_knots Integer; spline knots for the intensity term.
#' @param intensity_degree Integer; B-spline degree for the intensity term.
#' @param lambda_init_intensity Numeric; init smoothing for the intensity
#'   spline.
#' @param intensity_log1p_max Numeric; clamp on `log1p(connectivity)`.
#' @return A list with `best_params`, `best_loglik`, `alpha`, `gamma`,
#'   `resistance_raster`, `connectivity_raster`, `intensity_raster`,
#'   `loss_history`, `best_epoch`, `n_params`, `n_epochs_run`,
#'   `total_time`, `method`, `distribution`, `effective_loglinear`,
#'   `model_type`, plus the diagnostics returned by the Python optimiser.
#' @export
run_torch_pipeline <- function(basis_stack,
                                obs_points,
                                config                  = default_torch_config(),
                                hidden_dim              = config$hidden_dim,
                                n_hidden_layers         = config$n_hidden_layers,
                                lr                      = config$lr,
                                weight_decay            = config$weight_decay,
                                n_epochs                = config$n_epochs,
                                patience                = config$patience,
                                grad_clip               = config$grad_clip,
                                cg_tol                  = config$cg_tol,
                                source_spacing          = config$source_spacing,
                                source_from_resistance  = config$source_from_resistance,
                                reg_mean                = config$reg_mean,
                                reg_var                 = config$reg_var,
                                target_logR_var         = config$target_logR_var,
                                reg_skip                = config$reg_skip,
                                log_R_baseline          = config$log_R_baseline,
                                warm_start_theta        = NULL,
                                output_dir              = "output/torch",
                                seed                    = config$seed,
                                verbose                 = TRUE,
                                solver                  = config$solver,
                                radius                  = config$radius,
                                block_size              = config$block_size,
                                focal_fraction          = config$focal_fraction,
                                absorption              = config$absorption,
                                device                  = config$device,
                                use_conv                = config$use_conv,
                                n_conv_layers           = config$n_conv_layers,
                                conv_channels           = config$conv_channels,
                                conv_kernel_size        = config$conv_kernel_size,
                                dropout                 = config$dropout,
                                use_dilated             = config$use_dilated,
                                intensity_hidden        = config$intensity_hidden,
                                warmup_epochs           = config$warmup_epochs,
                                absorption_schedule     = NULL,
                                model_type              = config$model_type,
                                n_knots                 = config$n_knots,
                                spline_degree           = config$spline_degree,
                                include_interactions    = config$include_interactions,
                                penalty_scale           = config$penalty_scale,
                                lambda_init_marginal    = config$lambda_init_marginal,
                                lambda_init_interaction = config$lambda_init_interaction,
                                lambda_min              = config$lambda_min,
                                compute_uq              = FALSE,
                                uq_block_only           = TRUE,
                                intensity_spline        = config$intensity_spline,
                                intensity_n_knots       = config$intensity_n_knots,
                                intensity_degree        = config$intensity_degree,
                                lambda_init_intensity   = config$lambda_init_intensity,
                                intensity_log1p_max     = config$intensity_log1p_max) {

  if (!.ds_env$torch_initialized) ds_torch_setup()

  np <- reticulate::import("numpy", convert = FALSE)

  message("\n  Preparing data for PyTorch pipeline...")
  prep <- .prepare_torch_inputs(basis_stack, obs_points)
  message(sprintf("    Grid: %d x %d (%d valid cells)",
                  prep$n_rows, prep$n_cols, prep$n_valid))
  message(sprintf("    Observations: %d GPS fixes", prep$n_obs))
  message(sprintf("    Cell area: %.1f m^2", prep$cell_area))

  warm_theta_np <- if (!is.null(warm_start_theta)) {
    np$array(as.double(warm_start_theta), dtype = np$float64)
  } else NULL

  abs_sched_np <- if (!is.null(absorption_schedule)) {
    np$array(as.double(absorption_schedule), dtype = np$float64)
  } else NULL

  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  message("\n  Starting PyTorch optimisation...")
  results <- ds_torch_call(
    "run_torch_optimization",
    basis_values_np   = prep$basis_np,
    obs_counts_np     = prep$obs_np,
    n_rows            = as.integer(prep$n_rows),
    n_cols            = as.integer(prep$n_cols),
    valid_mask_np     = prep$vmask_np,
    cell_area         = as.double(prep$cell_area),
    source_spacing    = as.integer(source_spacing),
    source_from_resistance = source_from_resistance,
    hidden_dim        = as.integer(hidden_dim),
    n_hidden_layers   = as.integer(n_hidden_layers),
    lr                = as.double(lr),
    weight_decay      = as.double(weight_decay),
    n_epochs          = as.integer(n_epochs),
    patience          = as.integer(patience),
    grad_clip         = as.double(grad_clip),
    cg_tol            = as.double(cg_tol),
    warm_start_theta  = warm_theta_np,
    reg_mean          = as.double(reg_mean),
    reg_var           = as.double(reg_var),
    target_logR_var   = as.double(target_logR_var),
    reg_skip          = as.double(reg_skip),
    log_R_baseline    = as.double(log_R_baseline),
    output_dir        = output_dir,
    seed              = as.integer(seed),
    verbose           = verbose,
    solver            = solver,
    radius            = as.integer(radius),
    block_size        = as.integer(block_size),
    focal_fraction    = as.double(focal_fraction),
    absorption        = as.double(absorption),
    device            = as.character(device),
    use_conv          = use_conv,
    n_conv_layers     = as.integer(n_conv_layers),
    conv_channels     = as.integer(conv_channels),
    conv_kernel_size  = as.integer(conv_kernel_size),
    dropout           = as.double(dropout),
    use_dilated       = use_dilated,
    intensity_hidden  = as.integer(intensity_hidden),
    warmup_epochs     = as.integer(warmup_epochs),
    absorption_schedule    = abs_sched_np,
    model_type             = as.character(model_type),
    n_knots                = as.integer(n_knots),
    spline_degree          = as.integer(spline_degree),
    include_interactions   = include_interactions,
    penalty_scale          = as.double(penalty_scale),
    lambda_init_marginal   = as.double(lambda_init_marginal),
    lambda_init_interaction = as.double(lambda_init_interaction),
    lambda_min             = as.double(lambda_min),
    compute_uq             = compute_uq,
    uq_block_only          = uq_block_only,
    intensity_spline       = intensity_spline,
    intensity_n_knots      = as.integer(intensity_n_knots),
    intensity_degree       = as.integer(intensity_degree),
    lambda_init_intensity  = as.double(lambda_init_intensity),
    intensity_log1p_max    = as.double(intensity_log1p_max)
  )

  results_r <- reticulate::py_to_r(results)

  message("\n  Converting results to SpatRasters...")
  template <- basis_stack[[1]]

  R_full <- rep(NA_real_, terra::ncell(template))
  R_full[prep$valid_mask] <- results_r$resistance
  resistance_rast <- terra::rast(template)
  terra::values(resistance_rast) <- R_full
  names(resistance_rast) <- "resistance"

  C_full <- rep(NA_real_, terra::ncell(template))
  C_full[prep$valid_mask] <- results_r$connectivity
  connectivity_rast <- terra::rast(template)
  terra::values(connectivity_rast) <- C_full
  names(connectivity_rast) <- "connectivity"

  loglam_full <- rep(NA_real_, terra::ncell(template))
  loglam_full[prep$valid_mask] <- results_r$log_lambda
  intensity_rast <- terra::rast(template)
  terra::values(intensity_rast) <- exp(loglam_full)
  names(intensity_rast) <- "intensity"

  terra::writeRaster(resistance_rast,
                     file.path(output_dir, "optimized_resistance.tif"),
                     overwrite = TRUE)
  terra::writeRaster(connectivity_rast,
                     file.path(output_dir, "optimized_connectivity.tif"),
                     overwrite = TRUE)
  terra::writeRaster(intensity_rast,
                     file.path(output_dir, "predicted_intensity.tif"),
                     overwrite = TRUE)
  message("  Saved resistance, connectivity, intensity rasters")

  effective_ll <- results_r$effective_loglinear
  n_basis <- length(effective_ll) - 1L
  bp_list <- list(r_0 = effective_ll[1])
  for (k in seq_len(n_basis)) {
    bp_list[[paste0("z_", k)]] <- effective_ll[k + 1L]
  }
  best_params <- as.data.frame(bp_list)

  n_obs <- prep$n_obs
  method_tag <- if (model_type == "spline_gam") "torch_spline_gam_circuit"
                else "torch_nn_circuit"

  output <- list(
    best_params         = best_params,
    best_loglik         = results_r$loglik,
    alpha               = results_r$alpha,
    gamma               = results_r$gamma,
    resistance_raster   = resistance_rast,
    connectivity_raster = connectivity_rast,
    intensity_raster    = intensity_rast,
    loss_history        = results_r$loss_history,
    best_epoch          = results_r$best_epoch,
    n_params            = results_r$n_params,
    n_epochs_run        = results_r$n_epochs_run,
    total_time          = results_r$total_time,
    method              = method_tag,
    distribution        = "poisson_parametric",
    effective_loglinear = effective_ll,
    model_type          = model_type,
    partial_effects     = results_r$partial_effects,
    interaction_effects = results_r$interaction_effects,
    uq_results          = results_r$uq_results,
    intensity_params    = c(alpha = results_r$alpha, gamma = results_r$gamma),
    intensity_se        = c(alpha = NA_real_,        gamma = NA_real_),
    convergence         = 0L,
    convergence_message = sprintf("early_stop_epoch_%d", results_r$best_epoch),
    n_evaluations       = results_r$n_epochs_run,
    solver              = results_r$solver %||% "circuit_global"
  )

  saveRDS(output, file.path(output_dir, "torch_optimization_results.rds"))

  summary_report <- list(
    resistance_params = best_params,
    intensity_params  = output$intensity_params,
    intensity_se      = output$intensity_se,
    loglik            = results_r$loglik,
    distribution      = "poisson_parametric",
    info_criteria     = list(
      aic = -2 * results_r$loglik + 2 * results_r$n_params,
      bic = -2 * results_r$loglik + log(n_obs) * results_r$n_params
    ),
    n_evaluations    = results_r$n_epochs_run,
    n_basis          = terra::nlyr(basis_stack),
    method           = method_tag,
    solver           = output$solver,
    optim_convergence = 0L,
    optim_message    = "torch_adam"
  )
  saveRDS(summary_report, file.path(output_dir, "final_summary.rds"))

  output
}


# --------------- Gradient verification helpers ------------------------------
#
# All three helpers crop a small window around the centre of `basis_stack`
# to keep finite-difference gradient checks fast, then dispatch into the
# Python `verify_*_gradient()` functions.

#' @keywords internal
.crop_basis_for_gradient <- function(basis_stack, crop_size) {
  ext <- terra::ext(basis_stack)
  cx  <- (ext$xmin + ext$xmax) / 2
  cy  <- (ext$ymin + ext$ymax) / 2
  cell_size <- terra::res(basis_stack)[1]
  half <- crop_size * cell_size / 2
  crop_ext <- terra::ext(cx - half, cx + half, cy - half, cy + half)
  basis_crop <- terra::crop(basis_stack, crop_ext)

  basis_mat <- as.matrix(basis_crop)
  valid     <- stats::complete.cases(basis_mat)
  basis_vals <- basis_mat[valid, , drop = FALSE]

  list(
    basis_crop = basis_crop,
    basis_vals = basis_vals,
    valid      = valid,
    n_rows     = terra::nrow(basis_crop),
    n_cols     = terra::ncol(basis_crop)
  )
}


#' Verify the differentiable circuit-solver gradient
#'
#' Finite-difference gradient check on the bundled differentiable circuit
#' solver. Useful as a smoke test before running an expensive optimisation.
#'
#' @param basis_stack [terra::SpatRaster] of basis layers.
#' @param crop_size Integer; side length (cells) of the centre crop.
#' @param solver Character; solver to verify.
#' @param radius Integer; Omniscape radius.
#' @param block_size Integer; Omniscape block size.
#' @param absorption Numeric.
#' @return A list with `pass` (logical) and `max_rel_error`.
#' @export
verify_torch_gradient <- function(basis_stack,
                                   crop_size  = 20L,
                                   solver     = "diff_omniscape",
                                   radius     = 15L,
                                   block_size = 10L,
                                   absorption = 0.01) {

  if (!.ds_env$torch_initialized) ds_torch_setup()
  np <- reticulate::import("numpy", convert = FALSE)
  cr <- .crop_basis_for_gradient(basis_stack, crop_size)

  effective_radius <- min(as.integer(radius),
                          cr$n_rows %/% 2 - 1,
                          cr$n_cols %/% 2 - 1)
  effective_radius <- max(effective_radius, 3L)
  effective_block  <- min(as.integer(block_size), cr$n_rows %/% 3)
  effective_block  <- max(effective_block, 2L)

  if (effective_radius != radius || effective_block != block_size) {
    message(sprintf(
      "  Gradient check: adjusted radius=%d->%d, block_size=%d->%d for %dx%d crop",
      radius, effective_radius, block_size, effective_block,
      cr$n_rows, cr$n_cols
    ))
  }

  result <- ds_torch_call(
    "verify_circuit_gradient",
    basis_values_np = np$array(cr$basis_vals, dtype = np$float64),
    valid_mask_np   = np$array(cr$valid,      dtype = np$bool_),
    n_rows          = as.integer(cr$n_rows),
    n_cols          = as.integer(cr$n_cols),
    source_spacing  = 5L,
    source_from_resistance = TRUE,
    solver     = solver,
    radius     = as.integer(effective_radius),
    block_size = as.integer(effective_block),
    absorption = as.double(absorption)
  )

  result_r <- reticulate::py_to_r(result)
  if (isTRUE(result_r$pass)) {
    message("  Torch gradient check PASSED (max rel error: ",
            sprintf("%.2e", result_r$max_rel_error), ")")
  } else {
    warning("  Torch gradient check FAILED (max rel error: ",
            sprintf("%.2e", result_r$max_rel_error), ")")
  }
  result_r
}


#' Verify the ConvResistanceNet end-to-end gradient
#'
#' @param basis_stack [terra::SpatRaster] of basis layers.
#' @param crop_size Integer; centre crop side length.
#' @param solver Character; circuit solver.
#' @param absorption Numeric.
#' @param conv_channels,n_conv_layers,conv_kernel_size,hidden_dim Conv-net
#'   architecture knobs (see [run_torch_pipeline()]).
#' @param use_dilated Logical.
#' @param intensity_hidden Integer.
#' @return A list with `pass` and `max_rel_error`.
#' @export
verify_conv_gradient <- function(basis_stack,
                                  crop_size       = 20L,
                                  solver          = "global_absorption",
                                  absorption      = 0.01,
                                  conv_channels   = 16L,
                                  n_conv_layers   = 3L,
                                  conv_kernel_size = 3L,
                                  hidden_dim      = 16L,
                                  use_dilated     = TRUE,
                                  intensity_hidden = 0L) {

  if (!.ds_env$torch_initialized) ds_torch_setup()
  np <- reticulate::import("numpy", convert = FALSE)
  cr <- .crop_basis_for_gradient(basis_stack, crop_size)

  result <- ds_torch_call(
    "verify_conv_gradient",
    basis_values_np  = np$array(cr$basis_vals, dtype = np$float64),
    valid_mask_np    = np$array(cr$valid,      dtype = np$bool_),
    n_rows           = as.integer(cr$n_rows),
    n_cols           = as.integer(cr$n_cols),
    source_spacing   = 1L,
    source_from_resistance = TRUE,
    conv_channels    = as.integer(conv_channels),
    n_conv_layers    = as.integer(n_conv_layers),
    conv_kernel_size = as.integer(conv_kernel_size),
    hidden_dim       = as.integer(hidden_dim),
    solver           = solver,
    absorption       = as.double(absorption),
    use_dilated      = use_dilated,
    intensity_hidden = as.integer(intensity_hidden)
  )

  result_r <- reticulate::py_to_r(result)
  if (isTRUE(result_r$pass)) {
    message("  Conv gradient check PASSED (max rel error: ",
            sprintf("%.2e", result_r$max_rel_error), ")")
  } else {
    warning("  Conv gradient check FAILED (max rel error: ",
            sprintf("%.2e", result_r$max_rel_error), ")")
  }
  result_r
}


#' Verify the SplineResistanceNet end-to-end gradient
#'
#' @param basis_stack [terra::SpatRaster] of basis layers.
#' @param crop_size Integer; centre crop side length.
#' @param solver Character.
#' @param absorption Numeric.
#' @param n_knots Integer; internal knots per covariate.
#' @param spline_degree Integer.
#' @param include_interactions Logical.
#' @param intensity_spline Logical.
#' @param intensity_n_knots Integer.
#' @param intensity_degree Integer.
#' @param intensity_log1p_max Numeric.
#' @return A list with `pass` and `max_rel_error`.
#' @export
verify_spline_gradient <- function(basis_stack,
                                    crop_size           = 20L,
                                    solver              = "global_absorption",
                                    absorption          = 0.01,
                                    n_knots             = 10L,
                                    spline_degree       = 3L,
                                    include_interactions = TRUE,
                                    intensity_spline    = FALSE,
                                    intensity_n_knots   = 5L,
                                    intensity_degree    = 3L,
                                    intensity_log1p_max = 10.0) {

  if (!.ds_env$torch_initialized) ds_torch_setup()
  np <- reticulate::import("numpy", convert = FALSE)
  cr <- .crop_basis_for_gradient(basis_stack, crop_size)

  result <- ds_torch_call(
    "verify_spline_gradient",
    basis_values_np      = np$array(cr$basis_vals, dtype = np$float64),
    valid_mask_np        = np$array(cr$valid,      dtype = np$bool_),
    n_rows               = as.integer(cr$n_rows),
    n_cols               = as.integer(cr$n_cols),
    source_spacing       = 1L,
    source_from_resistance = TRUE,
    n_knots              = as.integer(n_knots),
    degree               = as.integer(spline_degree),
    include_interactions = include_interactions,
    solver               = solver,
    absorption           = as.double(absorption),
    intensity_spline     = intensity_spline,
    intensity_n_knots    = as.integer(intensity_n_knots),
    intensity_degree     = as.integer(intensity_degree),
    intensity_log1p_max  = as.double(intensity_log1p_max)
  )

  result_r <- reticulate::py_to_r(result)
  if (isTRUE(result_r$pass)) {
    message("  Spline gradient check PASSED (max rel error: ",
            sprintf("%.2e", result_r$max_rel_error), ")")
  } else {
    warning("  Spline gradient check FAILED (max rel error: ",
            sprintf("%.2e", result_r$max_rel_error), ")")
  }
  result_r
}


# --------------- Bayesian samplers ------------------------------------------
#
# Shared post-processing: persist mcmc_results.rds + posterior_summary.csv.
#
#' @keywords internal
.save_mcmc_artifacts <- function(results_r, output_dir) {
  saveRDS(results_r, file.path(output_dir, "mcmc_results.rds"))
  message("  Saved mcmc_results.rds")

  if (!is.null(results_r$summary)) {
    sum_df <- do.call(rbind, lapply(names(results_r$summary), function(nm) {
      s <- results_r$summary[[nm]]
      data.frame(
        parameter = nm,
        mean      = s$mean, sd = s$sd,
        q025      = s$q025, median = s$q50, q975 = s$q975,
        ess       = s$ess,
        stringsAsFactors = FALSE
      )
    }))
    utils::write.csv(sum_df,
                     file.path(output_dir, "posterior_summary.csv"),
                     row.names = FALSE)
    message("  Saved posterior_summary.csv")
  }
}


#' Langevin / MALA posterior sampling from a trained torch model
#'
#' Starts from a MAP fit produced by [run_torch_pipeline()] and explores the
#' joint posterior using preconditioned Langevin dynamics (or MALA when
#' `use_mala = TRUE`).
#'
#' @param basis_stack [terra::SpatRaster] of basis layers (must match the
#'   stack used for the MAP run).
#' @param obs_points Data.frame with `x, y` columns.
#' @param model_dir Character; directory containing `resistance_nn.pt`
#'   (the MAP checkpoint produced by [run_torch_pipeline()]).
#' @param n_samples,burn_in,thin Sampling schedule.
#' @param step_size Numeric or `NULL` for auto.
#' @param precondition Logical; Adam-style diagonal preconditioner.
#' @param precondition_warmup Integer.
#' @param fix_smoothing,fix_intensity Logical; freeze blocks at MAP.
#' @param use_mala Logical; Metropolis-adjusted Langevin.
#' @param target_accept Numeric; MALA target acceptance.
#' @param solver,absorption,source_spacing,source_from_resistance Solver
#'   knobs (must match the MAP run).
#' @param reg_mean,log_R_baseline,penalty_scale Prior knobs.
#' @param n_knots,spline_degree,include_interactions Spline model knobs.
#' @param lambda_min,grad_clip,cg_tol Numerical settings.
#' @param seed Integer.
#' @param verbose Logical.
#' @param output_dir Character; defaults to `model_dir`.
#' @param device Character; `"auto"`, `"cuda"`, or `"cpu"`.
#' @return A list with `samples_effective_loglinear`, `samples_alpha`,
#'   `samples_gamma`, `partial_effects`, `log_posterior_trace`, `ess`,
#'   `summary`, `elapsed_time`.
#' @export
run_bayesian_sampling <- function(basis_stack,
                                   obs_points,
                                   model_dir,
                                   n_samples            = 2000L,
                                   burn_in              = 500L,
                                   thin                 = 5L,
                                   step_size            = NULL,
                                   precondition         = TRUE,
                                   precondition_warmup  = 50L,
                                   fix_smoothing        = FALSE,
                                   fix_intensity        = FALSE,
                                   use_mala             = TRUE,
                                   target_accept        = 0.574,
                                   solver               = "global_absorption",
                                   absorption           = 0.002,
                                   source_spacing       = 1L,
                                   source_from_resistance = TRUE,
                                   reg_mean             = 1.0,
                                   log_R_baseline       = 3.0,
                                   penalty_scale        = 5.0,
                                   n_knots              = 10L,
                                   spline_degree        = 3L,
                                   include_interactions = TRUE,
                                   lambda_min           = 0.1,
                                   grad_clip            = 100.0,
                                   cg_tol               = 1e-6,
                                   seed                 = 42L,
                                   verbose              = TRUE,
                                   output_dir           = NULL,
                                   device               = "auto") {

  if (!.ds_env$torch_initialized) ds_torch_setup()

  model_path <- file.path(model_dir, "resistance_nn.pt")
  if (!file.exists(model_path)) {
    stop("Model checkpoint not found: ", model_path,
         "\nRun run_torch_pipeline() first.", call. = FALSE)
  }
  if (is.null(output_dir)) output_dir <- model_dir

  message("\n  Preparing data for Langevin sampling...")
  prep <- .prepare_torch_inputs(basis_stack, obs_points)
  message(sprintf("    Grid: %d x %d (%d valid cells)",
                  prep$n_rows, prep$n_cols, prep$n_valid))
  message(sprintf("    Observations: %d GPS fixes", prep$n_obs))

  message("\n  Starting Langevin Monte Carlo sampling...")
  message(sprintf("    Samples: %d, Burn-in: %d, Thin: %d -> %d total steps",
                  n_samples, burn_in, thin, burn_in + n_samples * thin))

  results <- ds_torch_call(
    "run_langevin_sampling",
    basis_values_np    = prep$basis_np,
    obs_counts_np      = prep$obs_np,
    n_rows             = as.integer(prep$n_rows),
    n_cols             = as.integer(prep$n_cols),
    valid_mask_np      = prep$vmask_np,
    cell_area          = as.double(prep$cell_area),
    model_path         = model_path,
    source_spacing     = as.integer(source_spacing),
    source_from_resistance = source_from_resistance,
    solver             = as.character(solver),
    absorption         = as.double(absorption),
    n_samples          = as.integer(n_samples),
    burn_in            = as.integer(burn_in),
    thin               = as.integer(thin),
    step_size          = if (!is.null(step_size)) as.double(step_size) else NULL,
    precondition       = precondition,
    precondition_warmup = as.integer(precondition_warmup),
    fix_smoothing      = fix_smoothing,
    fix_intensity      = fix_intensity,
    use_mala           = use_mala,
    target_accept      = as.double(target_accept),
    reg_mean           = as.double(reg_mean),
    log_R_baseline     = as.double(log_R_baseline),
    penalty_scale      = as.double(penalty_scale),
    n_knots            = as.integer(n_knots),
    spline_degree      = as.integer(spline_degree),
    include_interactions = include_interactions,
    lambda_min         = as.double(lambda_min),
    grad_clip          = as.double(grad_clip),
    cg_tol             = as.double(cg_tol),
    seed               = as.integer(seed),
    verbose            = verbose,
    output_dir         = output_dir,
    device             = as.character(device)
  )

  results_r <- reticulate::py_to_r(results)
  .save_mcmc_artifacts(results_r, output_dir)
  results_r
}


#' HMC / NUTS posterior sampling from a trained torch model
#'
#' Runs the No-U-Turn Sampler over the joint posterior, starting from the
#' MAP checkpoint produced by [run_torch_pipeline()].
#'
#' @inheritParams run_bayesian_sampling
#' @param n_samples Integer; number of posterior draws.
#' @param warmup Integer; warm-up iterations (adapts step size / mass).
#' @param max_treedepth Integer; NUTS max tree depth.
#' @param init_step_size Numeric or `NULL` for auto.
#' @param adapt_mass_matrix Logical.
#' @return A list with posterior samples, `summary`, diagnostics
#'   (`n_divergences`, etc.), and `elapsed_time`.
#' @export
run_bayesian_sampling_hmc <- function(basis_stack,
                                       obs_points,
                                       model_dir,
                                       n_samples            = 1000L,
                                       warmup               = 1000L,
                                       max_treedepth        = 10L,
                                       target_accept        = 0.80,
                                       init_step_size       = NULL,
                                       adapt_mass_matrix    = TRUE,
                                       fix_smoothing        = TRUE,
                                       fix_intensity        = FALSE,
                                       solver               = "global_absorption",
                                       absorption           = 0.002,
                                       source_spacing       = 1L,
                                       source_from_resistance = TRUE,
                                       reg_mean             = 1.0,
                                       log_R_baseline       = 3.0,
                                       penalty_scale        = 5.0,
                                       n_knots              = 10L,
                                       spline_degree        = 3L,
                                       include_interactions = FALSE,
                                       lambda_min           = 0.1,
                                       grad_clip            = 100.0,
                                       cg_tol               = 1e-6,
                                       seed                 = 42L,
                                       verbose              = TRUE,
                                       output_dir           = NULL,
                                       device               = "auto") {

  if (!.ds_env$torch_initialized) ds_torch_setup()

  model_path <- file.path(model_dir, "resistance_nn.pt")
  if (!file.exists(model_path)) {
    stop("Model checkpoint not found: ", model_path,
         "\nRun run_torch_pipeline() first.", call. = FALSE)
  }
  if (is.null(output_dir)) output_dir <- model_dir

  message("\n  Preparing data for HMC/NUTS sampling...")
  prep <- .prepare_torch_inputs(basis_stack, obs_points)
  message(sprintf("    Grid: %d x %d (%d valid cells)",
                  prep$n_rows, prep$n_cols, prep$n_valid))
  message(sprintf("    Observations: %d GPS fixes", prep$n_obs))

  message("\n  Starting NUTS (No-U-Turn Sampler)...")
  message(sprintf("    Samples: %d, Warmup: %d -> %d total iterations",
                  n_samples, warmup, warmup + n_samples))

  results <- ds_torch_call(
    "run_hmc_sampling",
    basis_values_np    = prep$basis_np,
    obs_counts_np      = prep$obs_np,
    n_rows             = as.integer(prep$n_rows),
    n_cols             = as.integer(prep$n_cols),
    valid_mask_np      = prep$vmask_np,
    cell_area          = as.double(prep$cell_area),
    model_path         = model_path,
    source_spacing     = as.integer(source_spacing),
    source_from_resistance = source_from_resistance,
    solver             = as.character(solver),
    absorption         = as.double(absorption),
    n_samples          = as.integer(n_samples),
    warmup             = as.integer(warmup),
    max_treedepth      = as.integer(max_treedepth),
    target_accept      = as.double(target_accept),
    init_step_size     = if (!is.null(init_step_size)) as.double(init_step_size) else NULL,
    adapt_mass_matrix  = adapt_mass_matrix,
    fix_smoothing      = fix_smoothing,
    fix_intensity      = fix_intensity,
    reg_mean           = as.double(reg_mean),
    log_R_baseline     = as.double(log_R_baseline),
    penalty_scale      = as.double(penalty_scale),
    n_knots            = as.integer(n_knots),
    spline_degree      = as.integer(spline_degree),
    include_interactions = include_interactions,
    lambda_min         = as.double(lambda_min),
    grad_clip          = as.double(grad_clip),
    cg_tol             = as.double(cg_tol),
    seed               = as.integer(seed),
    verbose            = verbose,
    output_dir         = output_dir,
    device             = as.character(device)
  )

  results_r <- reticulate::py_to_r(results)
  .save_mcmc_artifacts(results_r, output_dir)

  if (!is.null(results_r$n_divergences) && results_r$n_divergences > 0) {
    warning(sprintf(
      "HMC reported %d divergent transitions. Consider increasing target_accept or tightening cg_tol.",
      results_r$n_divergences
    ))
  }
  results_r
}


#' Automatic Differentiation Variational Inference (ADVI)
#'
#' Fits a mean-field or full-rank Gaussian variational posterior by
#' optimising the ELBO with Adam. Faster than HMC/Langevin but biased to
#' the variational family.
#'
#' @inheritParams run_bayesian_sampling_hmc
#' @param n_samples Integer; number of posterior draws.
#' @param lambda_min Numeric; floor on smoothing parameters.
#' @param cg_tol Numeric; CG solver tolerance.
#' @param max_iter Integer; ADVI iterations.
#' @param lr Numeric; ELBO Adam learning rate.
#' @param n_elbo_samples Integer; MC samples per ELBO gradient.
#' @param full_rank Logical; full-rank Gaussian variational family.
#' @param patience Integer; ELBO plateau patience.
#' @return A list with posterior samples / summary plus
#'   `best_elbo`, `converged`.
#' @export
run_advi <- function(basis_stack,
                      obs_points,
                      model_dir,
                      n_samples            = 2000L,
                      max_iter             = 2000L,
                      lr                   = 0.01,
                      n_elbo_samples       = 1L,
                      full_rank            = FALSE,
                      patience             = 100L,
                      fix_smoothing        = TRUE,
                      fix_intensity        = FALSE,
                      solver               = "global_absorption",
                      absorption           = 0.002,
                      source_spacing       = 1L,
                      source_from_resistance = TRUE,
                      reg_mean             = 1.0,
                      log_R_baseline       = 3.0,
                      penalty_scale        = 5.0,
                      n_knots              = 3L,
                      spline_degree        = 3L,
                      include_interactions = TRUE,
                      lambda_min           = 0.1,
                      cg_tol               = 1e-6,
                      seed                 = 42L,
                      verbose              = TRUE,
                      output_dir           = NULL,
                      device               = "auto") {

  if (!.ds_env$torch_initialized) ds_torch_setup()

  model_path <- file.path(model_dir, "resistance_nn.pt")
  if (!file.exists(model_path)) {
    stop("Model checkpoint not found: ", model_path,
         "\nRun run_torch_pipeline() first.", call. = FALSE)
  }
  if (is.null(output_dir)) output_dir <- model_dir

  message("\n  Preparing data for ADVI...")
  prep <- .prepare_torch_inputs(basis_stack, obs_points)
  message(sprintf("    Grid: %d x %d (%d valid cells)",
                  prep$n_rows, prep$n_cols, prep$n_valid))
  message(sprintf("    Observations: %d GPS fixes", prep$n_obs))

  mode_str <- if (full_rank) "full-rank" else "mean-field"
  message(sprintf("\n  Starting ADVI (%s)...", mode_str))
  message(sprintf("    Max iterations: %d, LR: %.4f, ELBO MC samples: %d",
                  max_iter, lr, n_elbo_samples))

  results <- ds_torch_call(
    "run_advi",
    basis_values_np      = prep$basis_np,
    obs_counts_np        = prep$obs_np,
    n_rows               = as.integer(prep$n_rows),
    n_cols               = as.integer(prep$n_cols),
    valid_mask_np        = prep$vmask_np,
    cell_area            = as.double(prep$cell_area),
    model_path           = model_path,
    source_spacing       = as.integer(source_spacing),
    source_from_resistance = source_from_resistance,
    solver               = as.character(solver),
    absorption           = as.double(absorption),
    n_samples            = as.integer(n_samples),
    max_iter             = as.integer(max_iter),
    lr                   = as.double(lr),
    n_elbo_samples       = as.integer(n_elbo_samples),
    full_rank            = full_rank,
    patience             = as.integer(patience),
    fix_smoothing        = fix_smoothing,
    fix_intensity        = fix_intensity,
    reg_mean             = as.double(reg_mean),
    log_R_baseline       = as.double(log_R_baseline),
    penalty_scale        = as.double(penalty_scale),
    n_knots              = as.integer(n_knots),
    spline_degree        = as.integer(spline_degree),
    include_interactions = include_interactions,
    lambda_min           = as.double(lambda_min),
    cg_tol               = as.double(cg_tol),
    seed                 = as.integer(seed),
    verbose              = verbose,
    output_dir           = output_dir,
    device               = as.character(device)
  )

  results_r <- reticulate::py_to_r(results)
  .save_mcmc_artifacts(results_r, output_dir)

  if (!is.null(results_r$converged) && !isTRUE(results_r$converged)) {
    warning("ADVI did not converge. Consider increasing max_iter or adjusting lr.")
  }
  message(sprintf("  ADVI: best ELBO = %.2f, converged = %s",
                  results_r$best_elbo,
                  if (isTRUE(results_r$converged)) "TRUE" else "FALSE"))
  results_r
}
