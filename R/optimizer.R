# ============================================================================
# Two-stage optimiser
#
# Outer loop: LHS + GP surrogate + Thompson Sampling over resistance params
# Inner loop: MLE for intensity params (via fit_intensity_nb / fit_intensity_gam)
#
# Enzyme.jl pathway: stubbed -- will replace the surrogate loop with
# direct gradient-based optimisation once differentiable connectivity is
# available.
# ============================================================================

# --------------- Surrogate configuration ------------------------------------

#' Default surrogate optimisation configuration
#'
#' @return A named list of tuning parameters.
#' @export
default_optimizer_config <- function() {
  list(
    n_init    = 20L,    # LHS design points
    n_iter    = 50L,    # Surrogate-guided iterations
    seed      = 42L,

    # Thompson Sampling
    ts_min_sd = 1e-6,   # floor on GP predictive SD

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

    # Omniscape / connectivity
    omniscape = list(radius = 13L, block_size = 5L, cleanup = TRUE)
  )
}


# --------------- Enzyme stub ------------------------------------------------

#' Optimise resistance via Enzyme.jl automatic differentiation
#'
#' **Not yet implemented.**
#' This function will use Enzyme.jl to compute gradients of
#' connectivity with respect to resistance parameters and run L-BFGS
#' optimisation.
#'
#' @param basis_stack A [terra::SpatRaster] of basis functions.
#' @param obs_points Data.frame with `x, y` columns.
#' @param bounds Named list of `c(lower, upper)` per parameter.
#' @param config Optimiser configuration (list).
#' @param ... Additional arguments (reserved for future use).
#' @return Currently errors with an informative message.
#' @export
optimize_resistance_enzyme <- function(basis_stack,
                                       obs_points,
                                       bounds = NULL,
                                       config = list(),
                                       ...) {

  stop(
    "Enzyme.jl gradient pathway is not yet implemented.\n",
    "Use optimize_resistance() with method = 'surrogate' instead.\n",
    "See ?optimize_resistance for details.",
    call. = FALSE
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
#' @param distribution `"negbin"` or `"gam"`.
#' @param omniscape_settings List with `radius`, `block_size`, `cleanup`.
#' @param intensity_config List from [default_intensity_config()].
#' @param covariates_obs Named list of covariate vectors at obs.
#' @param covariates_rasters Named list of [terra::SpatRaster] covariates.
#' @param residualise Logical.
#' @param verbose Logical.
#' @return A list with `loglik`, `intensity_params`, `intensity_se`,
#'   `hessian`, `total_time`, `convergence`, `distribution`.
#' @export
evaluate_full_model <- function(resistance_params,
                                basis_stack,
                                obs_points,
                                distribution      = "negbin",
                                omniscape_settings = list(),
                                intensity_config  = default_intensity_config(),
                                covariates_obs    = NULL,
                                covariates_rasters = NULL,
                                residualise       = FALSE,
                                verbose           = TRUE) {

  t0 <- Sys.time()

  # Step 1: resistance surface
  if (verbose) message("  Creating resistance surface...")
  resistance <- create_resistance_surface(resistance_params, basis_stack)

  # Step 2: connectivity
  if (verbose) message("  Running Omniscape...")
  omni_def <- list(radius = 13L, block_size = 5L, cleanup = TRUE)
  omni_set <- utils::modifyList(omni_def, omniscape_settings)

  omni <- run_omniscape(
    resistance,
    radius     = omni_set$radius,
    block_size = omni_set$block_size,
    cleanup    = omni_set$cleanup
  )
  connectivity <- omni$cum_current

  # Step 3: extract connectivity at observations
  conn_obs <- extract_connectivity(connectivity, obs_points)

  # Drop observations outside valid mask
  valid <- !is.na(conn_obs)
  if (any(!valid)) {
    if (verbose) message(sprintf("  Dropping %d/%d obs outside mask",
                                 sum(!valid), length(valid)))
    conn_obs   <- conn_obs[valid]
    obs_points <- obs_points[valid, , drop = FALSE]
    if (!is.null(covariates_obs)) {
      covariates_obs <- lapply(covariates_obs, function(v) v[valid])
    }
  }

  # Step 4: fit intensity
  fit_fn <- switch(distribution,
    negbin = fit_intensity_nb,
    gam    = fit_intensity_gam,
    stop("Unknown distribution: ", distribution, call. = FALSE)
  )

  if (verbose) message(sprintf("  Fitting intensity (%s)...", distribution))
  int_fit <- fit_fn(
    connectivity_at_obs  = conn_obs,
    connectivity_raster  = connectivity,
    obs_coords           = obs_points,
    covariates_obs       = covariates_obs,
    covariates_rasters   = covariates_rasters,
    residualise          = residualise,
    config               = intensity_config
  )

  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  list(
    loglik           = int_fit$loglik,
    intensity_params = int_fit$estimates,
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
                              residualise) {

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
      resistance_params  = params,
      basis_stack        = basis_stack,
      obs_points         = obs_points,
      distribution       = distribution,
      omniscape_settings = omniscape_settings,
      intensity_config   = intensity_config,
      covariates_obs     = covariates_obs,
      covariates_rasters = covariates_rasters,
      residualise        = residualise,
      verbose            = TRUE
    ),
    error = function(e) {
      message("  ERROR: ", conditionMessage(e))
      list(loglik = -1e10, intensity_params = c(alpha = NA, gamma = NA),
           intensity_se = c(alpha = NA, gamma = NA),
           convergence = 1L, total_time = NA_real_,
           distribution = distribution)
    }
  )

  # Log
  if (!is.null(log_file)) {
    entry <- data.frame(eval = eval_counter$n, r_0 = theta[1])
    for (k in seq_len(n_basis)) entry[[paste0("z_", k)]] <- theta[k + 1]
    entry$alpha   <- result$intensity_params["alpha"]
    entry$gamma   <- result$intensity_params["gamma"]
    entry$loglik  <- result$loglik
    entry$time    <- result$total_time
    entry$converged <- result$convergence == 0

    exists_ <- file.exists(log_file)
    utils::write.table(entry, log_file, append = exists_,
                       row.names = FALSE, col.names = !exists_, sep = ",")
  }

  neg_ll <- -result$loglik
  if (!is.finite(neg_ll) || neg_ll > 1e10) neg_ll <- 1e10
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


# --------------- GP surrogate -----------------------------------------------

#' @keywords internal
.fit_surrogate <- function(X, y) {
  if (is.data.frame(X)) X <- as.matrix(X)
  DiceKriging::km(
    formula    = ~1,
    design     = X,
    response   = y,
    covtype    = "matern5_2", # TODO make the GP fully configurable
    control    = list(trace = FALSE),
    nugget.estim = TRUE,
    nugget     = 1
  )
}


# --------------- Thompson Sampling -----------------------------------------
# Perform Thompson Sampling acquisition by drawing a random sample from the surrogate model.
#' @keywords internal
.thompson_sampling <- function(x, model, min_sd = 1e-6) {
  # TODO 1: try more advanced TS techniques
  # TODO 2: add an option for Expected Improvement or other acquisition functions (random forests work well).
  if (is.vector(x)) x <- matrix(x, nrow = 1)
  colnames(x) <- colnames(model@X)
  pred <- stats::predict(model, newdata = x, type = "UK")
  mu   <- pred$mean
  sig  <- pmax(pred$sd, min_sd)
  stats::rnorm(length(mu), mean = mu, sd = sig) 
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

  gp_mean <- function(x) {
    xm <- matrix(x, nrow = 1)
    colnames(xm) <- pnames
    stats::predict(surrogate, newdata = xm, type = "UK")$mean
  }

  H <- tryCatch(numDeriv::hessian(gp_mean, x_best), error = function(e) NULL)
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
#' @return A list with `best_params`, `best_loglik`, `X_evaluated`,
#'   `y_evaluated`, `surrogate`, `bounds`, `n_evaluations`.
#' @export
optimize_resistance <- function(basis_stack,
                                obs_points,
                                bounds           = NULL,
                                config           = default_optimizer_config(),
                                intensity_config = default_intensity_config(),
                                output_dir       = tempdir(),
                                covariates_obs   = NULL,
                                covariates_rasters = NULL,
                                residualise      = FALSE) {
 # TODO refactor this function to work with more flexible resistance models (e.g. non-linear, non-parametric, ML-based).
  set.seed(config$seed)

  n_basis <- terra::nlyr(basis_stack)
  if (is.null(bounds)) bounds <- get_default_bounds(n_basis)

  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  log_file <- file.path(output_dir, "optimization_log.csv")
  if (file.exists(log_file)) unlink(log_file)

  eval_counter <- new.env(parent = emptyenv())
  eval_counter$n <- 0L

  X_eval <- data.frame()
  y_eval <- numeric()

  distribution <- config$distribution

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
      covariates_obs, covariates_rasters, residualise
    )
    X_eval <- rbind(X_eval, init_design[i, ])
    y_eval <- c(y_eval, y)
  }

  # ======= Phase 2: Surrogate-guided =======================================
  message("\n", strrep("=", 60))
  message(sprintf("PHASE 2: Thompson Sampling (%d iterations)",
                  config$n_iter))
  message(strrep("=", 60))

  # Adaptive state
  sigma_vec <- rep(config$sigma_initial, length(bounds))
  names(sigma_vec) <- names(bounds)
  dim_scales    <- rep(1, length(bounds))
  names(dim_scales) <- names(bounds)
  local_frac    <- config$local_frac_initial
  best_so_far   <- min(y_eval)
  stall_count   <- 0L
  stalls_at_min <- 0L
  restart_count <- 0L

  for (iter in seq_len(config$n_iter)) {

    # sanitise
    bad <- !is.finite(y_eval)
    if (any(bad)) {
      penalty <- if (any(!bad)) max(y_eval[!bad]) * 1.1 else 1e10
      y_eval[bad] <- max(penalty, 1e10)
    }

    surrogate <- .fit_surrogate(X_eval, y_eval)

    best_idx   <- which.min(y_eval)
    best_point <- as.numeric(X_eval[best_idx, ])
    names(best_point) <- names(bounds)
    y_best <- y_eval[best_idx]

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

    ts_vals  <- .thompson_sampling(as.matrix(candidates), surrogate,
                                    config$ts_min_sd)
    next_idx <- which.min(ts_vals)
    next_pt  <- candidates[next_idx, ]

    theta <- as.numeric(next_pt)
    y <- .outer_objective(
      theta, basis_stack, obs_points, config$omniscape,
      eval_counter, log_file, distribution, intensity_config,
      covariates_obs, covariates_rasters, residualise
    )

    X_eval <- rbind(X_eval, next_pt)
    y_eval <- c(y_eval, y)

    # --- adaptive state update ---
    if (y < best_so_far) {
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

  final_surrogate <- .fit_surrogate(X_eval, y_eval)

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
    config        = config
  )

  saveRDS(results, file.path(output_dir, "optimization_results.rds"))
  results
}
