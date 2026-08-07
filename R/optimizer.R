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
#'   \item{`surrogate_type`}{Character. `"gp"` (Gaussian Process via
#'     \pkg{DiceKriging}, default) or `"rf"` (Random Forest via
#'     \pkg{ranger}). Selects the model used to approximate the outer-loop
#'     objective surface for the Thompson Sampling / Expected Improvement
#'     acquisition loop. See [fit_surrogate()].}
#'   \item{`surrogate_config`}{Named list of extra arguments forwarded to
#'     the surrogate constructor (e.g. `list(num.trees = 500)` for `"rf"`).
#'     Default: `list()`.}
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

    # Surrogate model for the outer loop: "gp" (default) or "rf"
    surrogate_type   = "gp",
    surrogate_config = list(),

    # Omniscape / connectivity
    omniscape = list(radius = 13L, block_size = 5L, cleanup = TRUE)
  )
}


# --------------- JAX gradient optimiser -------------------------------------

#' Optimise resistance via JAX automatic differentiation
#'
#' Uses L-BFGS (via jaxopt) or Adam (via optax) optimisation with
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

  # --- Data preparation (same pattern as .prepare_backend_inputs) -------------
  np <- reticulate::import("numpy", convert = FALSE)

  n_basis   <- terra::nlyr(basis_stack)
  n_rows    <- terra::nrow(basis_stack)
  n_cols    <- terra::ncol(basis_stack)
  cell_area <- prod(terra::res(basis_stack))

  if (is.null(bounds)) bounds <- get_default_bounds(n_basis)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  basis_matrix <- as.matrix(basis_stack)
  valid_mask   <- stats::complete.cases(basis_matrix)
  basis_values <- basis_matrix[valid_mask, , drop = FALSE]

  # Tabulate observation counts per valid cell
  cell_indices <- terra::cellFromXY(
    basis_stack,
    cbind(obs_points$x, obs_points$y)
  )
  valid_obs <- !is.na(cell_indices) & valid_mask[cell_indices]
  if (any(!valid_obs)) {
    message(sprintf("  Dropped %d obs outside valid cells", sum(!valid_obs)))
  }
  cell_indices <- cell_indices[valid_obs]

  obs_table        <- table(cell_indices)
  obs_counts_full  <- rep(0L, terra::ncell(basis_stack))
  obs_counts_full[as.integer(names(obs_table))] <- as.integer(obs_table)
  obs_counts_valid <- obs_counts_full[valid_mask]

  basis_np <- np$array(basis_values, dtype = np$float64)
  obs_np   <- np$array(as.double(obs_counts_valid), dtype = np$float64)
  vmask_np <- np$array(valid_mask, dtype = np$bool_)

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
    convergence   = convergence
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

  np <- reticulate::import("numpy", convert = FALSE)

  n_basis   <- terra::nlyr(basis_stack)
  n_rows    <- terra::nrow(basis_stack)
  n_cols    <- terra::ncol(basis_stack)
  cell_area <- prod(terra::res(basis_stack))

  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  basis_matrix <- as.matrix(basis_stack)
  valid_mask   <- stats::complete.cases(basis_matrix)
  basis_values <- basis_matrix[valid_mask, , drop = FALSE]

  # Tabulate observation counts per valid cell
  cell_indices <- terra::cellFromXY(
    basis_stack,
    cbind(obs_points$x, obs_points$y)
  )
  valid_obs <- !is.na(cell_indices) & valid_mask[cell_indices]
  if (any(!valid_obs)) {
    message(sprintf("  Dropped %d obs outside valid cells", sum(!valid_obs)))
  }
  cell_indices <- cell_indices[valid_obs]

  obs_table        <- table(cell_indices)
  obs_counts_full  <- rep(0L, terra::ncell(basis_stack))
  obs_counts_full[as.integer(names(obs_table))] <- as.integer(obs_table)
  obs_counts_valid <- obs_counts_full[valid_mask]

  basis_np <- np$array(basis_values, dtype = np$float64)
  obs_np   <- np$array(as.double(obs_counts_valid), dtype = np$float64)
  vmask_np <- np$array(valid_mask, dtype = np$bool_)

  seed <- config$seed %||% 42L

  # Solver settings -- same radius/block_size honored by the parametric
  # (JAX gradient) path and by ds_jax_connectivity()'s forward/evaluation
  # solve, so the neural loss trains against the same connectivity
  # definition it will later be evaluated against (GH #105).
  solver_radius <- config$omniscape$radius     %||% 13L
  solver_block  <- config$omniscape$block_size %||% 5L

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
    obs_np        = obs_np,
    valid_mask_np = vmask_np,
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
    convergence   = 0L,
    loss_history  = result$loss_history,
    model_type    = result$model_type
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
#' @param distribution `"negbin"` or `"gam"`.  `NULL` is treated as
#'   `"negbin"` (the formal default does not protect against an explicit
#'   `NULL` argument, since R argument matching still binds the parameter
#'   in that case).
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
                                available_covariates = NULL) {

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
  fit_fn <- switch(distribution,
    negbin = fit_intensity_nb,
    gam    = fit_intensity_gam,
    stop("Unknown distribution: ", distribution, call. = FALSE)
  )

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
.create_lhs_design <- function(n_points, bounds) {
  n_params <- length(bounds)
  design_u <- lhs::randomLHS(n_points, n_params)
  design   <- matrix(NA_real_, n_points, n_params)
  colnames(design) <- names(bounds)
  for (i in seq_along(bounds)) {
    lo <- bounds[[i]][1]; hi <- bounds[[i]][2]
    design[, i] <- lo + design_u[, i] * (hi - lo)
  }
  as.data.frame(design)
}


# --------------- Surrogate model abstraction --------------------------------
#
# The outer loop needs a regression surrogate for the (expensive) objective
# surface. Two backends are supported:
#   "gp" - Gaussian Process regression via DiceKriging::km() (default;
#          preserves all prior behaviour exactly).
#   "rf" - Random Forest regression via ranger::ranger(), using its
#          jackknife-after-bootstrap variance estimator to provide the
#          predictive SD required by the TS/EI acquisition functions.
#
# Every surrogate is wrapped in a small S3 object (class "ds_surrogate")
# carrying `$type`, `$model`, and `$predictor_names` so downstream code
# (.thompson_sampling, .expected_improvement, .compute_dimension_scales,
# loo_cv_surrogate, laplace_resistance) can call predict_surrogate() without
# needing to know which backend produced the fit.
# ------------------------------------------------------------------------

#' Fit a surrogate regression model for the outer optimisation loop
#'
#' Factory function that fits either a Gaussian Process (`"gp"`, the
#' default and historical behaviour) or a Random Forest (`"rf"`) regression
#' surrogate of the (expensive) outer-loop objective, and wraps it in a
#' common `"ds_surrogate"` object so both backends can be used
#' interchangeably by the Thompson Sampling / Expected Improvement
#' acquisition functions and by [loo_cv_surrogate()] / [laplace_resistance()].
#'
#' @param X Numeric matrix or data.frame of evaluated parameter vectors
#'   (one row per evaluation).
#' @param y Numeric vector of objective values (one per row of `X`).
#' @param type Character; `"gp"` (default, [DiceKriging::km()] with a
#'   Matern 5/2 kernel) or `"rf"` (Random Forest via [ranger::ranger()],
#'   using jackknife-after-bootstrap variance estimates as the predictive
#'   SD needed for acquisition). Defaults to `"gp"` so existing callers of
#'   [optimize_resistance()] are unaffected.
#' @param config Named list of extra arguments forwarded to the underlying
#'   constructor (`DiceKriging::km()` for `"gp"`, `ranger::ranger()` for
#'   `"rf"`).
#' @return An object of class `"ds_surrogate"`: a list with `type`, `model`
#'   (the fitted backend object), and `predictor_names`.
#' @seealso [predict_surrogate()], [default_optimizer_config()]
#' @export
fit_surrogate <- function(X, y, type = c("gp", "rf"), config = list()) {
  type <- match.arg(type)

  if (is.data.frame(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"

  # Exact-duplicate rows make the GP correlation matrix singular; remove
  # them for both backends for consistent behaviour.
  keep <- !duplicated(X)
  X <- X[keep, , drop = FALSE]
  y <- y[keep]

  pnames <- colnames(X)

  model <- switch(type,
    gp = .fit_surrogate_gp(X, y, config),
    rf = .fit_surrogate_rf(X, y, config)
  )

  structure(
    list(type = type, model = model, predictor_names = pnames),
    class = "ds_surrogate"
  )
}


#' Legacy raw-GP surrogate fit (backward-compatible internal helper)
#'
#' Equivalent to `fit_surrogate(X, y, type = "gp")$model`: fits a
#' [DiceKriging::km()] GP directly and returns the raw `km` object (not the
#' `ds_surrogate` wrapper). Kept for internal callers and historical tests
#' that operate on the raw GP object directly.
#'
#' @keywords internal
.fit_surrogate <- function(X, y, config = list()) {
  if (is.data.frame(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  keep <- !duplicated(X)
  X <- X[keep, , drop = FALSE]
  y <- y[keep]
  .fit_surrogate_gp(X, y, config)
}


#' @keywords internal
.fit_surrogate_gp <- function(X, y, config = list()) {
  args_default <- list(
    formula      = ~1,
    design       = X,
    response     = y,
    covtype      = "matern5_2",
    control      = list(trace = FALSE),
    nugget.estim = TRUE,
    nugget       = 1
  )
  args <- utils::modifyList(args_default, config)

  tryCatch(
    do.call(DiceKriging::km, args),
    error = function(e) {
      # Near-duplicate points can still make the estimated-nugget path fail.
      # Retry with a fixed regularising nugget (1 % of response variance).
      args_fallback <- utils::modifyList(args, list(
        nugget.estim = FALSE,
        nugget       = max(stats::var(y) * 0.01, 1e-2)
      ))
      do.call(DiceKriging::km, args_fallback)
    }
  )
}


#' @keywords internal
.fit_surrogate_rf <- function(X, y, config = list()) {
  if (!requireNamespace("ranger", quietly = TRUE)) {
    stop("Package 'ranger' is required for surrogate_type = 'rf'. ",
         "Install with install.packages('ranger').", call. = FALSE)
  }

  df <- as.data.frame(X)
  df$.y <- y

  args_default <- list(
    formula    = .y ~ .,
    data       = df,
    num.trees  = 500L,
    keep.inbag = TRUE
  )
  args <- utils::modifyList(args_default, config)
  args$data <- df  # ensure config can't drop the response column
  if (isFALSE(args$keep.inbag)) {
    warning("surrogate_config$keep.inbag = FALSE is incompatible with the ",
            "jackknife SD estimate used by predict_surrogate(); overriding ",
            "to keep.inbag = TRUE.", call. = FALSE)
    args$keep.inbag <- TRUE
  }

  do.call(ranger::ranger, args)
}


#' Predict from a surrogate model
#'
#' Uniform prediction interface across surrogate backends, returning a mean
#' and predictive standard deviation regardless of whether the surrogate is
#' a GP ([DiceKriging::km()]) or a Random Forest ([ranger::ranger()]).
#'
#' For backward compatibility, `surrogate` may also be a raw
#' [DiceKriging::km()] object (as returned by the legacy internal
#' `.fit_surrogate()` helper) in addition to a `"ds_surrogate"` object from
#' [fit_surrogate()].
#'
#' @param surrogate A `"ds_surrogate"` object from [fit_surrogate()], or a
#'   raw `km` object.
#' @param newdata Numeric matrix or data.frame of query points.
#' @return A list with `mean` and `sd`, one value per row of `newdata`.
#' @export
predict_surrogate <- function(surrogate, newdata) {
  if (is.vector(newdata)) newdata <- matrix(newdata, nrow = 1)
  if (is.data.frame(newdata)) newdata <- as.matrix(newdata)

  # Backward-compat: a raw km object (e.g. from the legacy .fit_surrogate())
  if (methods::is(surrogate, "km")) {
    colnames(newdata) <- colnames(surrogate@X)
    pred <- stats::predict(surrogate, newdata = newdata, type = "UK")
    return(list(mean = pred$mean, sd = pred$sd))
  }

  if (!inherits(surrogate, "ds_surrogate")) {
    stop("`surrogate` must be a `ds_surrogate` object from fit_surrogate() ",
         "(or a raw km object).", call. = FALSE)
  }

  colnames(newdata) <- surrogate$predictor_names

  if (surrogate$type == "gp") {
    pred <- stats::predict(surrogate$model, newdata = newdata, type = "UK")
    list(mean = pred$mean, sd = pred$sd)
  } else {
    pred <- stats::predict(surrogate$model, data = as.data.frame(newdata),
                           type = "se", se.method = "jack")
    list(mean = pred$predictions, sd = pred$se)
  }
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
.expected_improvement <- function(x, model, y_best,
                                   xi_initial    = 0.1,
                                   iter          = NULL,
                                   n_iter        = NULL,
                                   xi_min        = 0.02,
                                   decay_divisor = 5,
                                   min_sd        = 1e-10) {
  pred  <- predict_surrogate(model, x)
  mu    <- pred$mean
  sigma <- pred$sd

  # Dynamic xi: decays exponentially toward xi_min across the surrogate phase.
  if (!is.null(iter) && !is.null(n_iter) && n_iter > 0) {
    decay_rate   <- n_iter / decay_divisor
    xi_effective <- max(xi_initial * exp(-iter / decay_rate), xi_min)
  } else {
    xi_effective <- xi_initial
  }

  improvement <- y_best - mu - xi_effective
  ei <- ifelse(
    sigma < min_sd,
    0,
    improvement * stats::pnorm(improvement / sigma) +
      sigma * stats::dnorm(improvement / sigma)
  )
  ei
}


# --------------- Candidate generation ----------------------------------------

#' @keywords internal
.generate_candidates <- function(n, bounds, best_point = NULL,
                                  sigma_vector = NULL,
                                  local_frac = 0.5) {
  n_p <- length(bounds)
  pnames <- names(bounds)
  if (is.null(sigma_vector)) {
    sigma_vector <- rep(0.05, n_p)
    names(sigma_vector) <- pnames
  }

  n_local <- floor(n * local_frac)
  n_lhs   <- n - n_local

  cand_lhs <- .create_lhs_design(n_lhs, bounds)

  if (n_local > 0 && !is.null(best_point)) {
    center <- as.numeric(best_point)
    local_m <- matrix(NA_real_, n_local, n_p)
    for (i in seq_len(n_local)) {
      for (j in seq_len(n_p)) {
        rng <- bounds[[j]][2] - bounds[[j]][1]
        local_m[i, j] <- center[j] + stats::rnorm(1, 0, sigma_vector[j] * rng)
        local_m[i, j] <- max(bounds[[j]][1], min(bounds[[j]][2], local_m[i, j]))
      }
    }
    cand_local <- as.data.frame(local_m)
    colnames(cand_local) <- pnames
    rbind(cand_lhs, cand_local)
  } else {
    cand_lhs
  }
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
  if (length(valid) == 0) {
    return(1e10)
  }
  worst  <- max(valid)
  spread <- diff(range(valid))
  margin <- if (spread > 0) 0.1 * spread else max(0.1 * abs(worst), 1e-6)
  worst + margin
}


# --------------- Hessian-based dimension scaling ----------------------------

# We use different scales for each parameter in the local search to account for different sensitivities. 
# This function computes scaling factors based on the curvature of the surrogate model at the best point.
#' @keywords internal
.compute_dimension_scales <- function(surrogate, best_point, bounds,
                                       max_ratio = 5) {
  n_p    <- length(bounds)
  pnames <- names(bounds)
  x_best <- as.numeric(best_point)
  names(x_best) <- pnames

  surrogate_mean <- function(x) {
    xm <- matrix(x, nrow = 1)
    colnames(xm) <- pnames
    predict_surrogate(surrogate, xm)$mean
  }

  H <- tryCatch(numDeriv::hessian(surrogate_mean, x_best), error = function(e) NULL)
  if (is.null(H)) {
    sc <- rep(1, n_p); names(sc) <- pnames; return(sc)
  }

  curvs <- pmax(abs(diag(H)), 1e-10)
  raw   <- 1 / sqrt(curvs)
  sc    <- raw / mean(raw)

  # clamp ratio
  ratio <- max(sc) / min(sc)
  if (ratio > max_ratio) {
    lsc <- log(sc)
    lsc <- lsc * (log(max_ratio) / log(ratio))
    sc  <- exp(lsc); sc <- sc / mean(sc)
  }
  names(sc) <- pnames
  sc
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
#'   [predict_surrogate()] to query it regardless of the backend selected
#'   via `config$surrogate_type` (`"gp"`, the default, or `"rf"`).
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
  withr::local_seed(config$seed)

  n_basis  <- terra::nlyr(basis_stack)
  if (is.null(bounds)) bounds <- get_default_bounds(n_basis)

  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  log_file <- file.path(output_dir, "optimization_log.csv")
  if (file.exists(log_file)) unlink(log_file)

  eval_counter <- new.env(parent = emptyenv())
  eval_counter$n <- 0L

  X_eval <- data.frame()
  y_eval <- numeric()

  distribution     <- config$distribution
  res_link         <- config$resistance_link %||% link_exp()
  int_family       <- config$family  # may be NULL (uses default for distribution)
  surrogate_type   <- config$surrogate_type %||% "gp"
  surrogate_config <- config$surrogate_config %||% list()

  # ======= Phase 1: LHS ====================================================
  message("\n", strrep("=", 60))
  message(sprintf("PHASE 1: Latin Hypercube Sampling (%d points)",
                  config$n_init))
  message(strrep("=", 60))

  init_design <- .create_lhs_design(config$n_init, bounds)

  for (i in seq_len(config$n_init)) {
    theta <- as.numeric(init_design[i, ])
    y <- .outer_objective(
      theta, basis_stack, obs_points, config$omniscape,
      eval_counter, log_file, distribution, intensity_config,
      covariates_obs, covariates_rasters, residualise,
      link = res_link, family = int_family,
      available_points     = available_points,
      available_covariates = available_covariates
    )
    X_eval <- rbind(X_eval, init_design[i, ])
    y_eval <- c(y_eval, y)
  }

  # ======= Phase 2: Surrogate-guided =======================================
  acquisition <- match.arg(config$acquisition %||% "TS", c("TS", "EI"))

  message("\n", strrep("=", 60))
  message(sprintf("PHASE 2: %s acquisition (%d iterations)",
                  if (acquisition == "EI") "Expected Improvement" else "Thompson Sampling",
                  config$n_iter))
  message(strrep("=", 60))

  # EI hyperparameters (only used when acquisition = "EI")
  xi_initial <- NULL
  decay_rate <- NULL
  if (acquisition == "EI") {
    score_range <- max(y_eval, na.rm = TRUE) - min(y_eval, na.rm = TRUE)
    xi_initial  <- score_range * (config$ei_xi_scale_factor %||% 0.1)
    decay_rate  <- (config$n_iter %||% 50L) /
      (config$ei_decay_rate_divisor %||% 5)
    message(sprintf("  EI xi_initial: %.4f (score range: %.2f), decay_rate: %.2f",
                    xi_initial, score_range, decay_rate))
  }

  # Adaptive state
  sigma_vec <- rep(config$sigma_initial, length(bounds))
  names(sigma_vec) <- names(bounds)
  dim_scales    <- rep(1, length(bounds))
  names(dim_scales) <- names(bounds)
  local_frac    <- config$local_frac_initial
  best_so_far   <- min(y_eval, na.rm = TRUE)
  stall_count   <- 0L
  stalls_at_min <- 0L
  restart_count <- 0L

  for (iter in seq_len(config$n_iter)) {

    # Failed evaluations (NA/non-finite) are never fed to the surrogate as
    # a fixed absolute sentinel. Instead they are penalised *relative* to
    # the observed objective range on a scratch copy (`y_fit`) used only
    # for surrogate fitting and candidate selection; the true values
    # (including NA for failures) are kept in `y_eval` for reporting.
    y_fit <- y_eval
    bad   <- !is.finite(y_fit)
    if (any(bad)) {
      y_fit[bad] <- .failure_penalty(y_fit[!bad])
    }

    surrogate <- fit_surrogate(X_eval, y_fit, type = surrogate_type,
                               config = surrogate_config)

    best_idx   <- which.min(y_fit)
    best_point <- as.numeric(X_eval[best_idx, ])
    names(best_point) <- names(bounds)
    y_best <- y_fit[best_idx]

    # Hessian scaling
    if (iter > 1) {
      dim_scales <- .compute_dimension_scales(surrogate, best_point,
                                               bounds, max_ratio = 5)
    }
    eff_sigma <- sigma_vec * dim_scales

    message(sprintf("\n--- Iteration %d/%d (best: %.2f) ---",
                    iter, config$n_iter, -y_best))

    candidates <- .generate_candidates(
      config$n_candidates, bounds, best_point, eff_sigma, local_frac
    )

    if (acquisition == "EI") {
      ei_vals <- .expected_improvement(
        as.matrix(candidates), surrogate, y_best,
        xi_initial    = xi_initial,
        iter          = iter,
        n_iter        = config$n_iter,
        xi_min        = config$ei_xi_min %||% 0.02,
        decay_divisor = config$ei_decay_rate_divisor %||% 5
      )
      next_idx <- which.max(ei_vals)
      next_pt  <- candidates[next_idx, ]
      message(sprintf("    EI (best candidate): %.4f", ei_vals[next_idx]))
    } else {
      ts_vals  <- .thompson_sampling(as.matrix(candidates), surrogate,
                                      config$ts_min_sd)
      next_idx <- which.min(ts_vals)
      next_pt  <- candidates[next_idx, ]
    }

    theta <- as.numeric(next_pt)
    y <- .outer_objective(
      theta, basis_stack, obs_points, config$omniscape,
      eval_counter, log_file, distribution, intensity_config,
      covariates_obs, covariates_rasters, residualise,
      link = res_link, family = int_family,
      available_points     = available_points,
      available_covariates = available_covariates
    )

    X_eval <- rbind(X_eval, next_pt)
    y_eval <- c(y_eval, y)

    # --- adaptive state update ---
    if (is.finite(y) && y < best_so_far) {
      message(sprintf("    IMPROVEMENT: %.4f -> %.2f", best_so_far - y, -y))
      best_so_far   <- y
      stall_count   <- 0L
      stalls_at_min <- 0L
    } else {
      stall_count <- stall_count + 1L
      if (stall_count >= config$stall_threshold) {
        if (mean(sigma_vec) <= config$sigma_min * 1.01) {
          stalls_at_min <- stalls_at_min + 1L
          if (stalls_at_min >= config$restart_threshold) {
            message("    >>> RESTART sigma <<<")
            sigma_vec <- rep(config$sigma_initial, length(bounds))
            names(sigma_vec) <- names(bounds)
            local_frac    <- config$local_frac_initial
            stalls_at_min <- 0L
            restart_count <- restart_count + 1L
          }
        } else {
          sigma_vec <- pmax(sigma_vec * config$decay_factor,
                            config$sigma_min)
          local_frac <- min(local_frac + config$local_frac_increment,
                            config$local_frac_max)
        }
        stall_count <- 0L
      }
    }
  }

  # ======= Results =========================================================
  best_idx <- which.min(y_eval)
  bp <- X_eval[best_idx, ]
  best_params <- params_vector_to_list(as.numeric(bp), n_basis)

  message("\n", strrep("=", 60))
  message("OPTIMISATION COMPLETE")
  message(sprintf("  Best log-likelihood: %.2f", -y_eval[best_idx]))
  message(strrep("=", 60))

  y_final_fit <- y_eval
  bad_final   <- !is.finite(y_final_fit)
  if (any(bad_final)) {
    y_final_fit[bad_final] <- .failure_penalty(y_final_fit[!bad_final])
  }
  final_surrogate <- fit_surrogate(X_eval, y_final_fit, type = surrogate_type,
                                   config = surrogate_config)

  results <- list(
    best_params   = best_params,
    best_loglik   = -y_eval[best_idx],
    best_idx      = best_idx,
    X_evaluated   = X_eval,
    y_evaluated   = y_eval,
    n_evaluations = eval_counter$n,
    surrogate     = final_surrogate,
    bounds        = bounds,
    distribution  = distribution,
    acquisition   = acquisition,
    xi_initial    = xi_initial,
    decay_rate    = decay_rate,
    config        = config
  )

  saveRDS(results, file.path(output_dir, "optimization_results.rds"))
  results
}
