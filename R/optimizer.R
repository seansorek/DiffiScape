# ============================================================================
# Two-stage optimiser
#
# Outer loop: LHS + GP surrogate + Thompson Sampling over resistance params
# Inner loop: MLE for intensity params (via fit_intensity_nb / fit_intensity_gam)
#
# The JAX gradient pathway (optimize_resistance_gradient) provides direct
# gradient-based optimisation via the JAXScape differentiable solver.
# ============================================================================

# --------------- Surrogate configuration ------------------------------------

#' Default surrogate optimisation configuration
#'
#' Returns a named list of tuning parameters for the GP surrogate optimisation
#' loop.  Modify individual elements and pass the result via the `config`
#' argument of [optimize_resistance()] or [diffiscape()].
#'
#' @section Sampling budget:
#' \describe{
#'   \item{`n_init`}{Integer. Latin Hypercube Sampling design points used for
#'     the initial space-filling exploration. Default: `20`.}
#'   \item{`n_iter`}{Integer. Surrogate-guided (GP + acquisition) iterations
#'     after the initial LHS phase. Default: `50`.}
#'   \item{`seed`}{Integer. RNG seed for reproducibility. Default: `42`.}
#' }
#'
#' @section Acquisition function:
#' \describe{
#'   \item{`acquisition`}{Character. `"TS"` (Thompson Sampling, default) or
#'     `"EI"` (Expected Improvement).}
#'   \item{`ts_min_sd`}{Numeric. Floor on GP predictive SD for Thompson
#'     Sampling, preventing numerical collapse. Default: `1e-6`.}
#'   \item{`ei_xi_scale_factor`}{Numeric. EI exploration scaling — the initial
#'     xi is `(max(y) - min(y)) * ei_xi_scale_factor`. Default: `0.1`.}
#'   \item{`ei_xi_min`}{Numeric. Minimum xi for EI (exploitation floor).
#'     Default: `0.02`.}
#'   \item{`ei_decay_rate_divisor`}{Numeric. Controls the xi decay rate;
#'     `decay_rate = n_iter / ei_decay_rate_divisor`. Default: `5`.}
#' }
#'
#' @section Adaptive local search:
#' \describe{
#'   \item{`sigma_initial`}{Numeric. Initial neighbourhood radius (in
#'     standardised parameter space). Default: `0.15`.}
#'   \item{`sigma_min`}{Numeric. Minimum neighbourhood radius. Default: `0.02`.}
#'   \item{`stall_threshold`}{Integer. Iterations without improvement before
#'     sigma is shrunk. Default: `3`.}
#'   \item{`decay_factor`}{Numeric. Multiplicative sigma reduction on stall.
#'     Default: `0.7`.}
#'   \item{`local_frac_initial`}{Numeric. Initial fraction of candidates drawn
#'     from the local neighbourhood. Default: `0.5`.}
#'   \item{`local_frac_max`}{Numeric. Maximum local fraction. Default: `0.8`.}
#'   \item{`local_frac_increment`}{Numeric. Per-stall increment to the local
#'     fraction. Default: `0.05`.}
#'   \item{`restart_threshold`}{Integer. Stall count that triggers a hard
#'     restart from the best parameters found so far. Default: `5`.}
#'   \item{`n_candidates`}{Integer. Candidate points sampled per iteration.
#'     Default: `3000`.}
#' }
#'
#' @section Likelihood model:
#' \describe{
#'   \item{`distribution`}{Character. Likelihood for the inner MLE step:
#'     `"negbin"` (negative-binomial, default) or `"gam"` (GAM via mgcv).}
#'   \item{`family`}{An [intensity_family] object, or `NULL` (inferred from
#'     `distribution`). Overrides `distribution` when set.}
#'   \item{`resistance_link`}{A [resistance_link] object, or `NULL` (uses
#'     [link_exp()]). Controls how raw parameters map to resistance values.}
#' }
#'
#' @section Surrogate model:
#' \describe{
#'   \item{`surrogate_config`}{Named list of extra arguments forwarded to
#'     the SMT KRG implementation. Default: `list()`.}
#' }
#'
#' @section Connectivity solver:
#' \describe{
#'   \item{`omniscape`}{Named list passed to the Omniscape solver:
#'     `radius` (focal radius in cells, default `13`),
#'     `block_size` (focal block size, default `5`),
#'     `cleanup` (remove temp files after each run, default `TRUE`).}
#' }
#'
#' @return A named list of tuning parameters.
#' @seealso [optimize_resistance()], [diffiscape()]
#' @export
default_optimizer_config <- function() {
  list(
    n_init    = 20L,    # LHS design points
    n_iter    = 50L,    # Surrogate-guided iterations
    seed      = 42L,

    # Acquisition: "TS" (Thompson Sampling, default) or "EI" (Expected Improvement)
    acquisition = "TS",

    # Thompson Sampling
    ts_min_sd = 1e-6,   # floor on GP predictive SD

    # Expected Improvement: dynamic xi decay (exploration-exploitation balance)
    # xi_initial = (max(y) - min(y)) * ei_xi_scale_factor (scaled from LHS scores)
    # xi_eff(iter) = max(xi_initial * exp(-iter / decay_rate), ei_xi_min)
    # decay_rate = n_iter / ei_decay_rate_divisor
    ei_xi_scale_factor    = 0.1,
    ei_xi_min             = 0.02,
    ei_decay_rate_divisor = 5,

    # Adaptive local search
    sigma_initial       = 0.15,
    sigma_min           = 0.02,
    stall_threshold     = 3L,
    decay_factor        = 0.7,
    local_frac_initial  = 0.5,
    local_frac_max      = 0.8,
    local_frac_increment = 0.05,
    restart_threshold   = 5L,
    n_candidates        = 3000L,

    # Likelihood model for inner loop
    distribution = "negbin",   # "negbin" or "gam"
    family       = NULL,       # intensity_family object (overrides distribution)

    # Resistance link
    resistance_link = NULL,    # resistance_link object (NULL -> link_exp())

    # GP surrogate for the outer loop
    surrogate_config = list(),

    # Omniscape / connectivity
    omniscape = list(radius = 13L, block_size = 5L, cleanup = TRUE)
  )
}


# --------------- JAX gradient optimiser -------------------------------------

#' Optimise resistance via JAX automatic differentiation
#'
#' Uses L-BFGS (via SciPy L-BFGS-B) or Adam (via optax) optimisation with
#' `jax.grad` through the JAXScape differentiable circuit solver.
#' This replaces the former Julia/Enzyme.jl optimizer.
#'
#' When `model_type` is not `"parametric"`, dispatches to the Flax
#' neural-network optimizer ([ds_jax_neural_optimize()]) instead of
#' the parametric optimizer.  Recognised neural model types are
#' `"mlp"`, `"conv"`, `"spline_gam"`, and `"irl"`.
#'
#' @param basis_stack A [terra::SpatRaster] of basis functions.
#' @param obs_points Data.frame with `x, y` columns.
#' @param bounds Named list of `c(lower, upper)` per parameter (or
#'   `NULL` for defaults).
#' @param config Optimiser configuration (see [default_optimizer_config()]).
#' @param intensity_config Intensity config (see [default_intensity_config()]).
#' @param output_dir Directory for logs and results.
#' @param covariates_obs Named list of covariate vectors at obs.
#' @param covariates_rasters Named list of [terra::SpatRaster] covariates.
#' @param residualise Logical; residualise connectivity.
#' @param available_points Optional data.frame with `x, y` columns of
#'   available/background locations for selection function families.
#' @param available_covariates Named list of covariate vectors at
#'   `available_points` locations.
#' @param method Character; `"lbfgs"` (default) or `"adam"`.
#'   Ignored when `model_type` is not `"parametric"`.
#' @param parameterization Character; `"resistance"` (default) or
#'   `"permeability"`.
#' @param model_type Character; `"parametric"` (default, existing
#'   L-BFGS / Adam path), `"mlp"`, `"conv"`, `"spline_gam"`, or
#'   `"irl"` (Flax neural-network resistance models).
#' @param model_config Named list of model-specific parameters
#'   forwarded to the Flax module constructor (ignored for
#'   `model_type = "parametric"`).
#' @param optim_config Named list with `lr`, `n_epochs`, `patience`
#'   for the neural optimizer (ignored for `model_type = "parametric"`).
#' @return A list with `best_params`, `best_loglik`, `bounds`,
#'   `n_evaluations`, `distribution`, `convergence`.
#' @export
optimize_resistance_gradient <- function(basis_stack,
                                         obs_points,
                                         bounds               = NULL,
                                         config               = default_optimizer_config(),
                                         intensity_config     = default_intensity_config(),
                                         output_dir           = tempdir(),
                                         covariates_obs       = NULL,
                                         covariates_rasters   = NULL,
                                         residualise          = FALSE,
                                         available_points     = NULL,
                                         available_covariates = NULL,
                                         method               = "lbfgs",
                                         parameterization     = "resistance",
                                         model_type           = "parametric",
                                         model_config         = list(),
                                         optim_config         = list()) {

  model_type <- match.arg(model_type,
    c("parametric", "mlp", "conv", "spline_gam", "irl"))

  # The JAX gradient/neural objective is a covariate-free Poisson PPP
  # (log lambda = alpha + gamma * log1p(C)) -- it has no negative-binomial,
  # ZINB, GAM, RSF, or conditional-logit path, and no way for a covariate
  # to enter. optimize_resistance() and run_torch_pipeline() do honor these
  # arguments, so silently dropping them here would let switching solvers
  # change the model being fit without any indication (GH #106).
  if (!is.null(covariates_obs) || !is.null(covariates_rasters) ||
      !is.null(available_points) || !is.null(available_covariates) ||
      isTRUE(residualise)) {
    stop(
      "optimize_resistance_gradient() (solver = \"gradient\") fits a ",
      "covariate-free Poisson PPP and does not support covariates_obs, ",
      "covariates_rasters, available_points, available_covariates, or ",
      "residualise = TRUE. Use solver = \"ei\" or solver = \"torch\" instead.",
      call. = FALSE
    )
  }

  # --- Neural path: dispatch to Flax optimizer ----------------------------
  if (model_type != "parametric") {
    return(.optimize_neural(
      basis_stack      = basis_stack,
      obs_points       = obs_points,
      config           = config,
      output_dir       = output_dir,
      parameterization = parameterization,
      model_type       = model_type,
      model_config     = model_config,
      optim_config     = optim_config
    ))
  }

  method <- match.arg(method, c("lbfgs", "adam"))

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required for the JAX gradient solver. ",
         "Install with install.packages('reticulate').",
         call. = FALSE)
  }

  prep <- .prepare_backend_inputs(basis_stack, obs_points)
  n_basis <- terra::nlyr(basis_stack)
  n_rows <- prep$n_rows
  n_cols <- prep$n_cols
  cell_area <- prep$cell_area
  valid_mask <- prep$valid_mask
  basis_np <- prep$basis_np
  obs_np <- prep$obs_np
  vmask_np <- prep$vmask_np
  obs_counts_valid <- reticulate::py_to_r(obs_np)
  if (is.null(bounds)) bounds <- get_default_bounds(n_basis)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  # Initial parameter vector: midpoints of bounds
  lower_b <- vapply(bounds, `[`, numeric(1), 1)
  upper_b <- vapply(bounds, `[`, numeric(1), 2)
  init_params <- (lower_b + upper_b) / 2

  # Solver settings
  solver_radius <- config$omniscape$radius     %||% 13L
  solver_block  <- config$omniscape$block_size  %||% 5L
  seed          <- config$seed                  %||% 42L
  n_epochs      <- config$n_iter                %||% 300L
  patience      <- 30L

  # Map resistance_link to a link_fn string for Python
  res_link <- config$resistance_link %||% link_exp()
  link_fn  <- if (inherits(res_link, "resistance_link")) {
    res_link$name
  } else {
    "exp"
  }

  # The gradient objective always fits a Poisson PPP -- report that instead
  # of echoing config$distribution, which would otherwise misreport the
  # fitted model whenever a non-default distribution was requested.
  distribution <- "poisson"

  message("\n", strrep("=", 60))
  message(sprintf("JAX gradient optimiser (%s, %s parameterization)",
                  method, parameterization))
  message(sprintf("  Grid: %d x %d (%d valid cells)",
                  n_rows, n_cols, sum(valid_mask)))
  message(sprintf("  Observations: %d GPS fixes",
                  sum(obs_counts_valid)))
  message(strrep("=", 60))

  # --- Call Python optimiser via jax_bridge -----------------------------------
  result <- ds_jax_optimize(
    basis_np      = basis_np,
    obs_np        = obs_np,
    valid_mask_np = vmask_np,
    n_rows        = n_rows,
    n_cols        = n_cols,
    cell_area     = cell_area,
    init_params   = init_params,
    lower_bounds       = lower_b,
    upper_bounds       = upper_b,
    link_fn            = link_fn,
    radius             = as.integer(solver_radius),
    block_size         = as.integer(solver_block),
    parameterization   = parameterization,
    method             = method,
    lr                 = 0.01,
    n_epochs           = as.integer(n_epochs),
    patience           = as.integer(patience),
    seed               = as.integer(seed),
    verbose            = TRUE
  )

  # --- Reshape result to standard optimizer return format --------------------
  best_params <- params_vector_to_list(as.numeric(result$best_params), n_basis)

  convergence <- if (isTRUE(result$converged)) 0L else 1L

  message(sprintf(
    "\nOptimisation complete: %d iterations, loglik = %.2f (%.1f s)",
    result$n_epochs_run, result$best_loglik, result$elapsed
  ))

  list(
    best_params   = best_params,
    best_loglik   = result$best_loglik,
    X_evaluated   = NULL,
    y_evaluated   = NULL,
    surrogate     = NULL,
    bounds        = bounds,
    n_evaluations = result$n_epochs_run,
    distribution  = distribution,
    convergence   = convergence,
    python_fit    = result
  )
}


# --------------- Neural optimizer helper ------------------------------------

#' Internal dispatch to Flax neural-network resistance optimization
#'
#' Prepares inputs and calls [ds_jax_neural_optimize()], then reshapes
#' the result to match the return format of [optimize_resistance_gradient()].
#'
#' @keywords internal
.optimize_neural <- function(basis_stack,
                              obs_points,
                              config           = default_optimizer_config(),
                              output_dir       = tempdir(),
                              parameterization = "resistance",
                              model_type       = "mlp",
                              model_config     = list(),
                              optim_config     = list()) {

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required for neural optimization. ",
         "Install with install.packages('reticulate').",
         call. = FALSE)
  }

  prep <- .prepare_backend_inputs(basis_stack, obs_points)
  n_rows <- prep$n_rows
  n_cols <- prep$n_cols
  cell_area <- prep$cell_area
  valid_mask <- prep$valid_mask
  obs_counts_valid <- reticulate::py_to_r(prep$obs_np)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  basis_np <- prep$basis_np

  # Solver settings -- same radius/block_size honored by the parametric
  # (JAX gradient) path and by ds_jax_connectivity()'s forward/evaluation
  # solve, so the neural loss trains against the same connectivity
  # definition it will later be evaluated against (GH #105).
  solver_radius <- config$omniscape$radius     %||% 13L
  solver_block  <- config$omniscape$block_size %||% 5L
  seed <- config$seed %||% 42L

  # Fill optim_config defaults from config if not already set
  if (is.null(optim_config$n_epochs)) {
    optim_config$n_epochs <- as.integer(config$n_iter %||% 300L)
  }

  # Same covariate-free Poisson PPP objective as the parametric gradient
  # path (see GH #106) -- report the model actually fit, not the requested
  # config$distribution.
  distribution <- "poisson"

  message("\n", strrep("=", 60))
  message(sprintf("JAX neural optimizer (%s, %s parameterization)",
                  model_type, parameterization))
  message(sprintf("  Grid: %d x %d (%d valid cells)",
                  n_rows, n_cols, sum(valid_mask)))
  message(sprintf("  Observations: %d GPS fixes", sum(obs_counts_valid)))
  message(strrep("=", 60))

  result <- ds_jax_neural_optimize(
    basis_np      = basis_np,
    obs_np        = prep$obs_np,
    valid_mask_np = prep$vmask_np,
    n_rows        = n_rows,
    n_cols        = n_cols,
    cell_area     = cell_area,
    model_type    = model_type,
    model_config  = model_config,
    optim_config  = optim_config,
    parameterization = parameterization,
    radius            = as.integer(solver_radius),
    block_size        = as.integer(solver_block),
    seed             = as.integer(seed),
    verbose          = TRUE
  )

  message(sprintf(
    "\nNeural optimisation complete: %d epochs, loglik = %.2f (%.1f s)",
    result$n_epochs_run, result$best_loglik, result$elapsed
  ))

  list(
    best_params   = NULL,
    best_loglik   = result$best_loglik,
    resistance    = result$resistance,
    X_evaluated   = NULL,
    y_evaluated   = NULL,
    surrogate     = NULL,
    bounds        = NULL,
    n_evaluations = result$n_epochs_run,
    distribution  = distribution,
    convergence   = if (isTRUE(result$converged)) 0L else 1L,
    loss_history  = result$loss_history,
    model_type    = result$model_type,
    python_fit    = result
  )
}




# --------------- Full-model evaluation function -----------------------------

#' Evaluate the full model for one set of resistance parameters
#'
#' Creates a resistance surface, runs connectivity computation, and fits
#' the inner-loop intensity model.
#'
#' @param resistance_params Resistance parameter vector or list.
#' @param basis_stack A [terra::SpatRaster] of basis functions.
#' @param obs_points Data.frame with `x, y` columns.
#' @param distribution Count family (`"negbin"`, `"poisson"`, or `"zinb"`),
#'   or `"gam"`. `family` overrides the count family. `NULL` uses `"negbin"`.
#' @param omniscape_settings List with `radius`, `block_size`, `cleanup`.
#' @param intensity_config List from [default_intensity_config()].
#' @param covariates_obs Named list of covariate vectors at obs.
#' @param covariates_rasters Named list of [terra::SpatRaster] covariates.
#' @param residualise Logical.
#' @param verbose Logical.
#' @param link A [resistance_link] object (default [link_exp()]).
#' @param family An [intensity_family] object, or `NULL` to use the
#'   default for the chosen `distribution`.
#' @param available_points Optional data.frame with `x, y` columns of
#'   available/background locations for selection function families.  When
#'   supplied, bypasses raster quadrature and uses these locations with unit
#'   weights instead.  `NULL` (default) uses standard area-weighted raster
#'   integration.  Not supported when `distribution = "gam"`; supplying
#'   both raises an error.
#' @param available_covariates Named list of covariate vectors at
#'   `available_points` locations.  Required when `available_points` is
#'   supplied and the intensity model includes covariates.
#' @param .min_obs Internal minimum valid observation count for Laplace refits.
#' @return A list with `loglik`, `intensity_params`, `intensity_se`,
#'   `hessian`, `total_time`, `omniscape_time`, `convergence`, `distribution`.
#' @export
evaluate_full_model <- function(resistance_params,
                                basis_stack,
                                obs_points,
                                distribution        = "negbin",
                                omniscape_settings  = list(),
                                intensity_config    = default_intensity_config(),
                                covariates_obs      = NULL,
                                covariates_rasters  = NULL,
                                residualise         = FALSE,
                                verbose             = TRUE,
                                link                = link_exp(),
                                family              = NULL,
                                available_points    = NULL,
                                available_covariates = NULL,
                                .min_obs            = 0L) {

  # Guard against an explicit NULL distribution being passed through by a
  # caller (e.g. ds_fit_intensity() forwarding opt_result$distribution
  # without a %||% fallback). An explicit NULL argument still binds the
  # formal, so the default above does not protect against it -- this line
  # does.
  distribution <- distribution %||% "negbin"

  if (distribution == "gam" && !is.null(available_points)) {
    stop("available_points is not supported with distribution = 'gam'.",
         call. = FALSE)
  }

  if (distribution != "gam") {
    family <- resolve_family(family %||% intensity_config$family, distribution)
  }

  t0 <- Sys.time()

  # Step 1: resistance surface
  if (verbose) message("  Creating resistance surface...")
  resistance <- create_resistance_surface(resistance_params, basis_stack,
                                          link = link)

  # Step 2: connectivity
  if (verbose) message("  Running connectivity solver...")
  omni_def <- list(radius = 13L, block_size = 5L)
  omni_set <- utils::modifyList(omni_def, omniscape_settings)

  omni <- ds_jax_connectivity(
    resistance,
    radius     = omni_set$radius,
    block_size = omni_set$block_size
  )
  connectivity <- omni$cum_current

  # Step 3: extract connectivity at observations
  conn_obs <- extract_connectivity(connectivity, obs_points)

  # Drop observations outside valid mask
  valid <- !is.na(conn_obs)
  if (any(!valid)) {
    if (!is.null(family) && identical(family$name, "clogit") &&
        isTRUE(family$uses_strata)) {
      stop(sprintf(
        "%d of %d obs_points fall outside the connectivity raster's valid ",
        sum(!valid), length(valid)),
        "mask. family_clogit() precomputes its stratum idx_map from the ",
        "positions of stratum_ids_used/stratum_ids_avail at construction ",
        "time; silently dropping rows here would shift those positions and ",
        "corrupt the used/available pairing without any error. Pre-filter ",
        "obs_points together with stratum_ids_used (dropping the same rows ",
        "from both) before calling.",
        call. = FALSE
      )
    }
    if (verbose) message(sprintf("  Dropping %d/%d obs outside mask",
                                 sum(!valid), length(valid)))
    conn_obs   <- conn_obs[valid]
    obs_points <- obs_points[valid, , drop = FALSE]
    if (!is.null(covariates_obs)) {
      covariates_obs <- lapply(covariates_obs, function(v) v[valid])
    }
  }

  if (sum(valid) < .min_obs) return(list(loglik = -1e10))

  # Step 3b: extract connectivity at available locations (selection mode)
  avail_conn <- NULL
  if (!is.null(available_points)) {
    if (verbose) message("  Extracting connectivity at available locations...")
    avail_conn_raw <- extract_connectivity(connectivity, available_points)
    avail_valid    <- !is.na(avail_conn_raw)
    if (any(!avail_valid) && !is.null(family) &&
        identical(family$name, "clogit") && isTRUE(family$uses_strata)) {
      stop(sprintf(
        "%d of %d available_points fall outside the connectivity raster's ",
        sum(!avail_valid), length(avail_valid)),
        "valid mask. family_clogit() precomputes its stratum idx_map from ",
        "the positions of stratum_ids_used/stratum_ids_avail at ",
        "construction time; silently dropping rows here would shift those ",
        "positions and corrupt the used/available pairing without any ",
        "error. Pre-filter available_points together with ",
        "stratum_ids_avail (dropping the same rows from both) before ",
        "calling.",
        call. = FALSE
      )
    }
    avail_conn     <- avail_conn_raw[avail_valid]
    if (!is.null(available_covariates)) {
      available_covariates <- lapply(available_covariates,
                                     function(v) v[avail_valid])
    }
  }

  # Step 4: fit intensity
  fit_fn <- if (distribution == "gam") fit_intensity_gam else fit_intensity_nb

  if (verbose) message(sprintf("  Fitting intensity (%s)...", distribution))
  int_args <- list(
    connectivity_at_obs  = conn_obs,
    connectivity_raster  = connectivity,
    obs_coords           = obs_points,
    covariates_obs       = covariates_obs,
    covariates_rasters   = covariates_rasters,
    residualise          = residualise,
    config               = intensity_config
  )
  if (distribution != "gam" && !is.null(family))
    int_args$family <- family
  if (distribution != "gam" && !is.null(avail_conn)) {
    int_args$available_connectivity <- avail_conn
    int_args$available_covariates   <- available_covariates
  }
  int_fit <- do.call(fit_fn, int_args)

  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  list(
    loglik           = int_fit$loglik,
    intensity_params = int_fit$estimates,
    intensity_fit_obj = int_fit,
    intensity_se     = int_fit$se,
    hessian          = int_fit$hessian,
    convergence      = int_fit$convergence,
    distribution     = distribution,
    total_time       = elapsed,
    omniscape_time   = omni$elapsed_seconds
  )
}


# --------------- Outer-loop objective ---------------------------------------

# Helper function for the outer-loop surrogate optimisation. Evaluates the full model for a given set of resistance parameters and returns the negative log-likelihood.
#' @keywords internal
.outer_objective <- function(theta,
                              basis_stack,
                              obs_points,
                              omniscape_settings,
                              eval_counter,
                              log_file,
                              distribution,
                              intensity_config,
                              covariates_obs,
                              covariates_rasters,
                              residualise,
                              link                = link_exp(),
                              family              = NULL,
                              available_points    = NULL,
                              available_covariates = NULL) {

  n_basis <- terra::nlyr(basis_stack)
  eval_counter$n <- eval_counter$n + 1L

  params <- params_vector_to_list(theta, n_basis)

  message(sprintf("\n=== Evaluation %d ===", eval_counter$n))
  z_str <- paste(vapply(seq_len(n_basis), function(k)
    sprintf("z%d=%.3f", k, params[[paste0("z_", k)]]), character(1)),
    collapse = ", ")
  message(sprintf("  r_0=%.3f, %s", params$r_0, z_str))

  result <- tryCatch(
    evaluate_full_model(
      resistance_params    = params,
      basis_stack          = basis_stack,
      obs_points           = obs_points,
      distribution         = distribution,
      omniscape_settings   = omniscape_settings,
      intensity_config     = intensity_config,
      covariates_obs       = covariates_obs,
      covariates_rasters   = covariates_rasters,
      residualise          = residualise,
      verbose              = TRUE,
      link                 = link,
      family               = family,
      available_points     = available_points,
      available_covariates = available_covariates
    ),
    error = function(e) {
      message("  ERROR: ", conditionMessage(e))
      list(loglik = NA_real_, intensity_params = c(alpha = NA, gamma = NA),
           intensity_se = c(alpha = NA, gamma = NA),
           convergence = 1L, total_time = NA_real_,
           distribution = distribution)
    }
  )

  # Log
  if (!is.null(log_file)) {
    entry <- data.frame(eval = eval_counter$n, r_0 = theta[1])
    for (k in seq_len(n_basis)) entry[[paste0("z_", k)]] <- theta[k + 1]
    entry$alpha   <- result$intensity_params["alpha"] %||% NA_real_
    entry$gamma   <- result$intensity_params["gamma"] %||% NA_real_
    entry$loglik  <- result$loglik
    entry$time    <- result$total_time
    entry$converged <- result$convergence == 0

    exists_ <- file.exists(log_file)
    utils::write.table(entry, log_file, append = exists_,
                       row.names = FALSE, col.names = !exists_, sep = ",")
  }

  # A failed/non-finite evaluation is reported as NA rather than a fixed
  # sentinel value. NA is distinguishable from a genuinely bad-but-valid
  # score and lets callers apply a *relative* penalty (scaled to the
  # observed objective range) instead of an absolute value that can be
  # many orders of magnitude outside that range and distort the surrogate
  # fit (see #101).
  neg_ll <- -result$loglik
  if (!is.finite(neg_ll)) neg_ll <- NA_real_
  neg_ll
}


# --------------- LHS design -------------------------------------------------

#' @keywords internal
.create_lhs_design <- function(n, bounds) {
  result <- .py_surrogate("lhs_design", as.integer(n), do.call(rbind, bounds), .surrogate_seed())
  result <- as.data.frame(result)
  names(result) <- names(bounds)
  result
}


# --------------- GP surrogate ----------------------------------------------

#' Fit a Gaussian-process surrogate for the outer optimisation loop
#'
#' Uses the installed Python core and SMT ordinary kriging with a Matern 5/2 kernel. The fitted surrogate
#' supplies means and uncertainty for acquisition and posterior calculations.
#'
#' @param X Numeric matrix or data.frame of evaluated parameter vectors.
#' @param y Numeric vector of objective values, one per row of `X`.
#' @param type Character; only `"gp"` is supported.
#' @param config SMT KRG options such as `eval_noise`, `noise0`, `n_start`, and `hyper_opt`. Legacy DiceKriging config keys require translation.
#' @return A `"ds_surrogate"` object with `type`, `model`, and `predictor_names`.
#' @seealso [predict_surrogate()], [default_optimizer_config()]
#' @export
fit_surrogate <- function(X, y, type = "gp", config = list()) {
  type <- match.arg(type, "gp")
  if (!ds_jax_check()) ds_jax_setup()
  module <- reticulate::import("diffiscape.surrogate", convert = FALSE)
  if (length(config) && any(names(config) %in% c("formula", "design", "response", "covtype", "nugget.estim", "control"))) {
    stop("surrogate_config now uses SMT KRG options; refit using corr='matern52', poly='constant', eval_noise, noise0 and hyper_opt. See ?fit_surrogate.", call. = FALSE)
  }
  X <- as.matrix(X)
  fitted <- reticulate::py_to_r(module$fit_surrogate(X, as.double(y), if (length(config)) config else NULL))
  structure(list(type = type, model = fitted, predictor_names = colnames(X)), class = "ds_surrogate")
}


#' Predict from a GP surrogate model
#'
#' @param surrogate A `"ds_surrogate"` object from [fit_surrogate()], or its
#'   serialized Python `model` field. Legacy `km` objects must be refitted.
#' @param newdata Numeric matrix or data.frame of query points.
#' @return A list with `mean` and `sd`, one value per row of `newdata`.
#' @export
predict_surrogate <- function(surrogate, newdata) {
  if (is.vector(newdata)) newdata <- matrix(newdata, nrow = 1)
  if (inherits(surrogate, "ds_surrogate")) {
    if (!identical(surrogate$type, "gp")) stop("Only GP surrogates are supported; refit with type = 'gp'.", call. = FALSE)
    surrogate <- surrogate$model
  }
  if (!is.list(surrogate) || !identical(surrogate$engine, "smt")) {
    stop("surrogate must be a ds_surrogate from the installed Python core; legacy km objects must be refitted.", call. = FALSE)
  }
  if (!ds_jax_check()) ds_jax_setup()
  module <- reticulate::import("diffiscape.surrogate", convert = FALSE)
  result <- reticulate::py_to_r(module$predict_surrogate(surrogate, as.matrix(newdata)))
  lapply(result, as.numeric)
}


# --------------- Thompson Sampling -----------------------------------------
# Perform Thompson Sampling acquisition by drawing a random sample from the surrogate model.
#' @keywords internal
.thompson_sampling <- function(x, model, min_sd = 1e-6) {
  # TODO 1: try more advanced TS techniques
  pred <- predict_surrogate(model, x)
  mu   <- pred$mean
  sig  <- pmax(pred$sd, min_sd)
  stats::rnorm(length(mu), mean = mu, sd = sig)
}


# --------------- Expected Improvement --------------------------------------
# Vectorised EI acquisition with dynamic xi decay.
# Returns one EI value per candidate row in `x`; caller selects argmax.
# For minimisation (we minimise negative log-likelihood):
#   improvement = (y_best - mu - xi)
#   EI = improvement * Phi(z) + sigma * phi(z),  z = improvement / sigma
#' @keywords internal
.expected_improvement <- function(x, model, y_best, xi_initial = 0.1,
                                   iter = NULL, n_iter = NULL, xi_min = 0.02,
                                   decay_divisor = 5, min_sd = 1e-10) {
  prediction <- predict_surrogate(model, x)
  as.numeric(.py_surrogate("expected_improvement", prediction$mean, prediction$sd, y_best,
    xi_initial, iter, n_iter, xi_min, decay_divisor, min_sd))
}


# --------------- Candidate generation ----------------------------------------

#' @keywords internal
.generate_candidates <- function(n, bounds, best_point = NULL, sigma_vector = NULL, local_frac = 0.5) {
  result <- .py_surrogate("candidates", as.integer(n), do.call(rbind, bounds), best_point,
    sigma_vector, local_frac, .surrogate_seed())
  result <- as.data.frame(result)
  names(result) <- names(bounds)
  result
}


# --------------- Failure-penalty scratch value for surrogate fitting -------

# `y_eval` is a negative log-likelihood being minimised, so a failed
# evaluation must be replaced with a value strictly *worse* (larger) than
# every valid observation. A multiplicative bump (`worst * 1.1`) only does
# that when `worst` is positive: when valid scores are negative (as they can
# be for a continuous/point-process log-likelihood, where loglik > 0), it
# instead pulls the placeholder *below* the worst value -- and potentially
# below every valid value -- letting a failed row masquerade as the optimum.
# An additive, range-scaled margin is worse in the same direction regardless
# of sign.
#' @keywords internal
.failure_penalty <- function(valid) {
  .py_surrogate("failure_penalty", as.list(valid))
}


# --------------- Hessian-based dimension scaling ----------------------------

# We use different scales for each parameter in the local search to account for different sensitivities. 
# This function computes scaling factors based on the curvature of the surrogate model at the best point.
#' @keywords internal
.compute_dimension_scales <- function(surrogate, best_point, bounds, max_ratio = 5) {
  model <- if (inherits(surrogate, "ds_surrogate")) surrogate$model else surrogate
  stats::setNames(as.numeric(.py_surrogate("dimension_scales", model, as.double(best_point), max_ratio)), names(bounds))
}


# --------------- Main surrogate optimiser -----------------------------------

#' Optimise resistance parameters via GP surrogate + Thompson Sampling
#'
#' Two-stage loop:
#' 1. **Outer**: Latin Hypercube initial design, then GP surrogate with
#'    Thompson Sampling acquisition.
#' 2. **Inner**: MLE for intensity parameters given each connectivity
#'    surface.
#'
#' @param basis_stack A [terra::SpatRaster] of basis functions.
#' @param obs_points Data.frame with `x, y` columns of observed locations.
#' @param bounds Named list of `c(lower, upper)` per parameter (see
#'   [get_default_bounds()]).  `NULL` uses defaults.
#' @param config List from [default_optimizer_config()].
#' @param intensity_config List from [default_intensity_config()].
#' @param output_dir Directory for logs and results.
#' @param covariates_obs Named list of covariate vectors at obs.
#' @param covariates_rasters Named list of [terra::SpatRaster] covariates.
#' @param residualise Logical; residualise connectivity.
#' @param available_points Optional data.frame with `x, y` columns of
#'   available/background locations for selection function families.  When
#'   supplied, bypasses raster quadrature and uses these locations with unit
#'   weights instead.  `NULL` (default) uses standard area-weighted raster
#'   integration.
#' @param available_covariates Named list of covariate vectors at
#'   `available_points` locations.  Required when `available_points` is
#'   supplied and the intensity model includes covariates.
#' @return A list with `best_params`, `best_loglik`, `X_evaluated`,
#'   `y_evaluated`, `surrogate`, `bounds`, `n_evaluations`.  `surrogate` is
#'   a `"ds_surrogate"` object (see [fit_surrogate()]); use
#'   [predict_surrogate()] to query its mean and uncertainty.
#' @export
optimize_resistance <- function(basis_stack,
                                obs_points,
                                bounds               = NULL,
                                config               = default_optimizer_config(),
                                intensity_config     = default_intensity_config(),
                                output_dir           = tempdir(),
                                covariates_obs       = NULL,
                                covariates_rasters   = NULL,
                                residualise          = FALSE,
                                available_points     = NULL,
                                available_covariates = NULL) {
 # TODO refactor this function to work with more flexible resistance models (e.g. non-linear, non-parametric, ML-based).
  if (!is.null(config$surrogate_type) && config$surrogate_type != "gp") {
    stop("Only GP surrogates are supported; remove surrogate_type = 'rf'.", call. = FALSE)
  }
  n_basis <- terra::nlyr(basis_stack)
  if (is.null(bounds)) bounds <- get_default_bounds(n_basis)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  log_file <- file.path(output_dir, "optimization_log.csv")
  if (file.exists(log_file)) unlink(log_file)
  eval_counter <- new.env(parent = emptyenv())
  eval_counter$n <- 0L
  objective <- function(theta) {
    if (inherits(theta, "python.builtin.object")) theta <- reticulate::py_to_r(theta)
    .outer_objective(as.numeric(theta), basis_stack, obs_points, config$omniscape,
      eval_counter, log_file, config$distribution, intensity_config,
      covariates_obs, covariates_rasters, residualise,
      link = config$resistance_link %||% link_exp(), family = config$family,
      available_points = available_points, available_covariates = available_covariates)
  }
  keys <- c("n_init", "n_iter", "seed", "acquisition", "n_candidates", "ts_min_sd",
    "ei_xi_scale_factor", "ei_xi_min", "ei_decay_rate_divisor", "sigma_initial", "sigma_min",
    "stall_threshold", "decay_factor", "local_frac_initial", "local_frac_max", "local_frac_increment", "restart_threshold")
  bo_config <- config[intersect(keys, names(config))]
  if (!ds_jax_check()) ds_jax_setup()
  module <- reticulate::import("diffiscape.surrogate", convert = FALSE)
  result <- module$optimize_surrogate(objective, do.call(rbind, bounds),
    if (length(bo_config)) bo_config else NULL,
    if (length(config$surrogate_config)) config$surrogate_config else NULL)
  result <- reticulate::py_to_r(result)
  result$best_params <- params_vector_to_list(as.numeric(result$best_params), n_basis)
  result$best_idx <- result$best_idx + 1L
  result$X_evaluated <- as.data.frame(result$X_evaluated)
  names(result$X_evaluated) <- names(bounds)
  result$y_evaluated <- as.numeric(result$y_evaluated)
  result$surrogate <- structure(list(type = "gp", model = result$surrogate,
    predictor_names = names(bounds)), class = "ds_surrogate")
  result$bounds <- bounds
  result$distribution <- config$distribution
  result$config <- config
  saveRDS(result, file.path(output_dir, "optimization_results.rds"))
  result
}
