# ============================================================================
# Deprecated physical-model compatibility entry points
#
# Supported old fitting calls translate to the installed physical JAX core.
# Checkpoint samplers fail with refitting instructions; gradient checks use JAX.

# --------------- Torch pipeline configuration --------------------------------

#' Deprecated physical-model configuration
#'
#' Compatibility settings for run_torch_pipeline(). Use run_jax_pipeline() for new calls. weight_decay and stochastic focal_fraction are retired. Device selection follows the JAX environment.
#'
#' @export
#' @return Named compatibility configuration.
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
    solver                  = "global_absorption",
    radius                  = 15L,
    block_size              = 10L,
    focal_fraction          = 0.5,
    absorption              = 0.01,
    cg_tol                  = 1e-6,
    source_spacing          = 1L,
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
    model_type              = "loglinear",
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

#' Deprecated physical-model fitting entry point
#'
#' Translates supported settings to run_jax_pipeline() using JAX. Old checkpoint UQ, absorption curricula, neural warm starts and Torch-specific devices require migration. Use ds_posterior() for NumPyro uncertainty.
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
#' @param model_type Character; `"loglinear"`, `"mlp"`, `"conv"`, or `"spline_gam"`.
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
#' @export
#' @return The physical JAX fitting result; no Torch checkpoint is written.
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

  model_type <- match.arg(model_type, c("loglinear", "mlp", "conv", "spline_gam"))
  if (isTRUE(use_conv)) model_type <- "conv"
  if (isTRUE(compute_uq)) stop("compute_uq is unavailable after migration; use ds_posterior() with NumPyro for uncertainty.", call. = FALSE)
  if (!is.null(absorption_schedule)) stop("absorption_schedule has no validated translation; fit with a fixed absorption value.", call. = FALSE)
  if (!is.null(warm_start_theta) && model_type != "loglinear") stop("Neural warm starts have no validated translation; use the saved Python model state.", call. = FALSE)
  if (!device %in% c("auto", "cpu")) stop("Select an accelerator through JAX installation/configuration; Torch device values cannot be translated.", call. = FALSE)
  formulation <- switch(solver, global_absorption = "global_absorption", absorption = "global_absorption",
    global = "boundary", diff_omniscape = "focal_current", stop("Unknown connectivity formulation: ", solver, call. = FALSE))
  if (formulation == "global_absorption" && source_spacing != 1L) {
    message("Legacy global absorption uses source_spacing=1; use run_jax_pipeline() to choose another spacing.")
    source_spacing <- 1L
  }
  model_config <- switch(model_type,
    loglinear = list(),
    mlp = list(hidden_dim = hidden_dim, n_hidden_layers = n_hidden_layers),
    conv = list(hidden_dim = hidden_dim, n_hidden_layers = n_hidden_layers, conv_channels = conv_channels,
      n_conv_layers = n_conv_layers, conv_kernel_size = conv_kernel_size, dropout = dropout,
      use_dilated = use_dilated, intensity_hidden = intensity_hidden),
    spline_gam = list(n_knots = n_knots, spline_degree = spline_degree, include_interactions = include_interactions,
      lambda_init_marginal = lambda_init_marginal, lambda_init_interaction = lambda_init_interaction,
      lambda_min = lambda_min, intensity_spline = intensity_spline, intensity_n_knots = intensity_n_knots,
      intensity_degree = intensity_degree, lambda_init_intensity = lambda_init_intensity, intensity_log1p_max = intensity_log1p_max))
  message("run_torch_pipeline() is deprecated; using JAX with the same physical model. Use run_jax_pipeline(). Unused weight_decay and stochastic focal_fraction options are retired.")
  output <- run_jax_pipeline(basis_stack, obs_points, model_type = model_type, formulation = formulation,
    model_config = model_config, output_dir = output_dir, lr = lr, n_epochs = n_epochs,
    patience = patience, grad_clip = grad_clip, warmup_epochs = warmup_epochs,
    source_spacing = source_spacing, source_from_resistance = source_from_resistance, absorption = absorption,
    radius = radius, block_size = block_size, reg_mean = reg_mean, reg_var = reg_var,
    reg_skip = reg_skip, target_logR_var = target_logR_var, log_R_baseline = log_R_baseline,
    penalty_scale = penalty_scale, init_params = warm_start_theta, seed = seed, verbose = verbose, cg_tol = cg_tol)
  saveRDS(output, file.path(output_dir, "torch_optimization_results.rds"))
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
#' Finite-difference gradient check on the installed JAX circuit
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
  .verify_jax_gradient(basis_stack, crop_size, "loglinear", list(), solver, absorption, radius, block_size)
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
  .verify_jax_gradient(basis_stack, crop_size, "conv", list(conv_channels = as.integer(conv_channels), n_conv_layers = as.integer(n_conv_layers), conv_kernel_size = as.integer(conv_kernel_size), hidden_dim = as.integer(hidden_dim), use_dilated = use_dilated, intensity_hidden = as.integer(intensity_hidden)), solver, absorption)
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
  .verify_jax_gradient(basis_stack, crop_size, "spline_gam", list(n_knots = as.integer(n_knots), spline_degree = as.integer(spline_degree), include_interactions = include_interactions, intensity_spline = intensity_spline, intensity_n_knots = as.integer(intensity_n_knots), intensity_degree = as.integer(intensity_degree), intensity_log1p_max = intensity_log1p_max), solver, absorption)
}


# --------------- Bayesian samplers ------------------------------------------
#
# Shared post-processing: persist mcmc_results.rds + posterior_summary.csv.
#
#' Retired checkpoint sampler
#'
#' Old .pt checkpoints cannot be loaded by JAX. Refit with ds_optimize() and use ds_posterior(method = "nuts") or method = "advi". Custom MALA is retired.
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
#' @export
#' @return An actionable migration error.
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
  stop("Checkpoint-based MALA is retired. Refit with ds_optimize() and use ds_posterior(..., method = 'nuts') or method = 'advi'. Old .pt files cannot be loaded by JAX.", call. = FALSE)
}


#' Retired checkpoint sampler
#'
#' Old .pt checkpoints cannot be loaded by JAX. Refit with ds_optimize() and use ds_posterior(method = "nuts") or method = "advi". Custom MALA is retired.
#'
#' @inheritParams run_bayesian_sampling
#' @param n_samples Integer; number of posterior draws.
#' @param warmup Integer; warm-up iterations (adapts step size / mass).
#' @param max_treedepth Integer; NUTS max tree depth.
#' @param init_step_size Numeric or `NULL` for auto.
#' @param adapt_mass_matrix Logical.
#' @export
#' @return An actionable migration error.
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
  stop("Torch checkpoints are retired. Refit with ds_optimize() and use ds_posterior(..., method = 'nuts'). Old .pt files cannot be loaded by JAX.", call. = FALSE)
}


#' Retired checkpoint sampler
#'
#' Old .pt checkpoints cannot be loaded by JAX. Refit with ds_optimize() and use ds_posterior(method = "nuts") or method = "advi". Custom MALA is retired.
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
#' @export
#' @return An actionable migration error.
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
  stop("Torch checkpoints are retired. Refit with ds_optimize() and use ds_posterior(..., method = 'advi'). Old .pt files cannot be loaded by JAX.", call. = FALSE)
}
