# ============================================================================
# Posterior inference
# Doing full Bayesian inference through graphical laplacians is a whole can of worms.
# Laplace approximation for resistance params, Monte Carlo composition
# for joint posterior, GP emulator validation (LOO-CV).
# ============================================================================

# --------------- Laplace approximation for resistance params ----------------

#' Laplace approximation at the optimised resistance parameters
#'
#' Computes a Gaussian approximation to the marginal posterior of the
#' resistance parameters by evaluating (or re-using) the GP surrogate
#' at the MAP estimate.
#'
#' @param opt_result Result from [optimize_resistance()] or
#'   [optimize_resistance_enzyme()].
#' @param basis_stack A [terra::SpatRaster] of basis functions (required
#'   when `refit = TRUE` or when no surrogate is available).
#' @param obs_points Data.frame with `x, y` (required when `refit = TRUE`).
#' @param refit Logical; use `numDeriv::hessian()` on the full model
#'   instead of the GP surrogate.  Automatically set to `TRUE` when
#'   `opt_result` has no `$surrogate` (i.e., from
#'   [optimize_resistance_enzyme()]).
#' @param step Finite-difference step size for Hessian computation.
#' @param omniscape_settings Named list of solver overrides: `radius`
#'   (default `13L`) and `block_size` (default `5L`). Used when `refit = TRUE`.
#' @param intensity_config Optional [default_intensity_config()] list.
#' @param covariates_obs Named list of covariate vectors.
#' @param covariates_rasters Named list of covariate rasters.
#' @param residualise Logical.
#' @param link A [resistance_link] object (default [link_exp()]).
#' @param family An [intensity_family] object, or `NULL`.
#' @return A list with `mode`, `covariance`, `precision`, `std_error`.
#' @export
laplace_resistance <- function(opt_result,
                               basis_stack          = NULL,
                               obs_points           = NULL,
                               refit                = FALSE,
                               step                 = 1e-3,
                               omniscape_settings   = list(),
                               intensity_config     = default_intensity_config(),
                               covariates_obs       = NULL,
                               covariates_rasters   = NULL,
                               residualise          = FALSE,
                               available_points     = NULL,
                               available_covariates = NULL,
                               link                 = link_exp(),
                               family               = NULL) {

  best_vec <- .params_to_vector(opt_result$best_params)
  p        <- length(best_vec)

  # Auto-refit when no surrogate is available (Enzyme/direct path)
  if (is.null(opt_result$surrogate)) refit <- TRUE

  if (!refit && !is.null(opt_result$surrogate)) {
    # Use the GP surrogate's predicted mean as the log-likelihood
    surrogate  <- opt_result$surrogate
    pnames     <- names(opt_result$bounds)

    surrogate_fn <- function(theta) {
      x <- matrix(theta, nrow = 1)
      colnames(x) <- pnames
      -stats::predict(surrogate, newdata = x, type = "UK")$mean
    }

    H <- numDeriv::hessian(surrogate_fn, best_vec, method.args = list(eps = step))

  } else if (refit && !is.null(basis_stack) && !is.null(obs_points)) {
    # Direct Hessian via the fast Julia solver
    distribution <- opt_result$distribution %||% "negbin"
    n_basis      <- terra::nlyr(basis_stack)
    nrow_grid    <- terra::nrow(basis_stack)
    ncol_grid    <- terra::ncol(basis_stack)
    basis_vals   <- terra::values(basis_stack)
    template     <- basis_stack[[1]]
    omni_def      <- list(radius = 13L, block_size = 5L)
    omni_cfg      <- utils::modifyList(omni_def, omniscape_settings)
    solver_radius <- omni_cfg$radius
    solver_block  <- omni_cfg$block_size

    refit_fn <- function(theta) {
      R_vec <- quick_resistance(theta, basis_vals, link = link)
      R_mat <- matrix(R_vec, nrow = nrow_grid, ncol = ncol_grid,
                      byrow = TRUE)
      R_mat[is.na(R_mat)] <- 0

      cum_mat  <- ds_julia_call("DiffiScapeMod.cumulative_current",
                                 R_mat, solver_radius, solver_block)
      cum_vec  <- as.vector(t(cum_mat))
      cum_rast <- terra::rast(template)
      terra::values(cum_rast) <- cum_vec

      conn_obs <- extract_connectivity(cum_rast, obs_points)
      valid    <- !is.na(conn_obs)
      if (sum(valid) < 3) return(1e10)

      fit_fn <- switch(distribution,
        negbin = fit_intensity_nb, gam = fit_intensity_gam)

      int_args <- list(
        connectivity_at_obs = conn_obs[valid],
        connectivity_raster = cum_rast,
        obs_coords          = obs_points[valid, , drop = FALSE],
        covariates_obs      = if (!is.null(covariates_obs))
          lapply(covariates_obs, function(v) v[valid]) else NULL,
        covariates_rasters  = covariates_rasters,
        residualise         = residualise,
        config              = intensity_config
      )
      if (distribution != "gam" && !is.null(family))
        int_args$family <- family
      if (!is.null(available_points)) {
        avail_conn_raw <- extract_connectivity(cum_rast, available_points)
        avail_valid    <- !is.na(avail_conn_raw)
        avail_conn     <- avail_conn_raw[avail_valid]
        avail_cov      <- if (!is.null(available_covariates))
          lapply(available_covariates, function(v) v[avail_valid]) else NULL
        int_args$available_connectivity <- avail_conn
        int_args$available_covariates   <- avail_cov
      }
      int_fit <- do.call(fit_fn, int_args)

      neg_ll <- -int_fit$loglik
      if (!is.finite(neg_ll)) 1e10 else neg_ll
    }

    H <- numDeriv::hessian(refit_fn, best_vec,
                            method.args = list(eps = step))
  } else {
    stop("Must supply either opt_result$surrogate, or basis_stack + obs_points with refit=TRUE",
         call. = FALSE)
  }

  # H is the Hessian of the neg-LL -> neg-H is Hessian of LL
  neg_H <- -H

  # try Cholesky; fall back to nearPD
 cov_mat <- tryCatch({
    solve(neg_H)
  }, error = function(e) {
    message("  Hessian not positive-definite, using nearPD correction")
    pd <- Matrix::nearPD(neg_H, ensureSymmetry = TRUE)
    solve(as.matrix(pd$mat))
  })

  # Make sure cov is symmetric
  cov_mat <- 0.5 * (cov_mat + t(cov_mat))

  diag_var <- pmax(diag(cov_mat), 0)

  list(
    mode       = best_vec,
    covariance = cov_mat,
    precision  = neg_H,
    std_error  = sqrt(diag_var)
  )
}


# --------------- Monte Carlo composition ------------------------------------

#' Joint posterior via Monte Carlo composition sampling
#'
#' For each draw of resistance parameters from the Laplace posterior,
#' evaluates the inner-loop MLE and draws intensity parameters from
#' the asymptotic normal.
#'
#' @param laplace Result from [laplace_resistance()].
#' @param opt_result Result from [optimize_resistance()].
#' @param basis_stack A [terra::SpatRaster] of basis functions.
#' @param obs_points Data.frame with `x, y`.
#' @param n_draws Number of posterior draws.
#' @param n_inner Number of inner-loop draws per outer draw.
#' @param bounds Named list of parameter bounds.
#' @param omniscape_settings List with connectivity settings.
#' @param intensity_config List from [default_intensity_config()].
#' @param covariates_obs Named list of covariate vectors.
#' @param covariates_rasters Named list of covariate rasters.
#' @param residualise Logical.
#' @param link A [resistance_link] object (default [link_exp()]).
#' @param family An [intensity_family] object, or `NULL`.
#' @param seed Random seed.
#' @return A data.frame with posterior samples (one row per draw).
#' @export
posterior_sample <- function(laplace,
                             opt_result,
                             basis_stack,
                             obs_points,
                             n_draws              = 200L,
                             n_inner              = 5L,
                             bounds               = NULL,
                             omniscape_settings   = list(),
                             intensity_config     = default_intensity_config(),
                             covariates_obs       = NULL,
                             covariates_rasters   = NULL,
                             residualise          = FALSE,
                             available_points     = NULL,
                             available_covariates = NULL,
                             link                 = link_exp(),
                             family               = NULL,
                             seed                 = 42L) {

  set.seed(seed)

  n_basis  <- terra::nlyr(basis_stack)
  if (is.null(bounds)) bounds <- opt_result$bounds

  pnames <- names(bounds)
  mu     <- laplace$mode
  sigma  <- laplace$covariance

  # Cholesky for sampling
  L <- tryCatch(chol(sigma), error = function(e) {
    pd <- Matrix::nearPD(sigma, ensureSymmetry = TRUE)
    chol(as.matrix(pd$mat))
  })

  samples_list <- vector("list", n_draws)

  for (d in seq_len(n_draws)) {
    message(sprintf("\n=== Posterior draw %d/%d ===", d, n_draws))

    z     <- stats::rnorm(length(mu))
    theta <- mu + as.numeric(z %*% L)

    # clip to bounds
    for (j in seq_along(bounds)) {
      theta[j] <- max(bounds[[j]][1], min(bounds[[j]][2], theta[j]))
    }

    params <- params_vector_to_list(theta, n_basis)

    result <- tryCatch(
      evaluate_full_model(
        resistance_params    = params,
        basis_stack          = basis_stack,
        obs_points           = obs_points,
        distribution         = opt_result$distribution,
        omniscape_settings   = omniscape_settings,
        intensity_config     = intensity_config,
        covariates_obs       = covariates_obs,
        covariates_rasters   = covariates_rasters,
        residualise          = residualise,
        verbose              = FALSE,
        link                 = link,
        family               = family,
        available_points     = available_points,
        available_covariates = available_covariates
      ),
      error = function(e) {
        message("  ERROR in posterior draw: ", conditionMessage(e))
        NULL
      }
    )

    if (is.null(result)) next

    # inner-loop draws from asymptotic Normal
    inner_mu <- result$intensity_params
    inner_se <- result$intensity_se

    if (any(is.na(inner_se))) {
      inner_draws <- matrix(inner_mu, nrow = 1, ncol = length(inner_mu))
      colnames(inner_draws) <- names(inner_mu)
    } else {
      inner_draws <- matrix(NA_real_, n_inner, length(inner_mu))
      colnames(inner_draws) <- names(inner_mu)
      for (k in seq_len(n_inner)) {
        inner_draws[k, ] <- stats::rnorm(length(inner_mu), inner_mu, inner_se)
      }
    }

    for (k in seq_len(nrow(inner_draws))) {
      row <- as.list(theta)
      names(row) <- pnames
      for (nm in colnames(inner_draws)) row[[nm]] <- inner_draws[k, nm]
      row$loglik <- result$loglik
      samples_list[[length(samples_list) + 1L]] <- as.data.frame(row)
    }
  }

  samples <- do.call(rbind, samples_list[vapply(samples_list, is.data.frame, logical(1))])
  if (is.null(samples) || nrow(samples) == 0) {
    warning("No valid posterior samples obtained")
    return(data.frame())
  }
  rownames(samples) <- NULL
  samples
}


# --------------- LOO-CV for GP emulator -------------------------------------

#' Leave-one-out cross-validation of the GP surrogate
#'
#' Uses the analytical LOO formula for Gaussian processes.
#'
#' @param opt_result Result from [optimize_resistance()].
#' @return A list with `observed`, `predicted`, `residuals`, `rmse`,
#'   `r_squared`, `coverage_95`.
#' @export
loo_cv_surrogate <- function(opt_result) {

  surrogate <- opt_result$surrogate
  if (is.null(surrogate)) stop("No surrogate in opt_result", call. = FALSE)

  X <- as.matrix(opt_result$X_evaluated)
  y <- opt_result$y_evaluated

  n <- length(y)

  loo_pred  <- numeric(n)
  loo_sd    <- numeric(n)

  for (i in seq_len(n)) {
    gp_loo <- tryCatch(
      .fit_surrogate(X[-i, , drop = FALSE], y[-i]),
      error = function(e) NULL
    )
    if (is.null(gp_loo)) {
      loo_pred[i] <- NA
      loo_sd[i]   <- NA
      next
    }
    xi <- matrix(X[i, ], nrow = 1)
    colnames(xi) <- colnames(X)
    p <- stats::predict(gp_loo, newdata = xi, type = "UK")
    loo_pred[i] <- p$mean
    loo_sd[i]   <- p$sd
  }

  valid <- !is.na(loo_pred)
  resid <- y[valid] - loo_pred[valid]
  rmse  <- sqrt(mean(resid^2))
  ss_res <- sum(resid^2)
  ss_tot <- sum((y[valid] - mean(y[valid]))^2)
  r2 <- 1 - ss_res / ss_tot

  # 95% coverage
  z95 <- stats::qnorm(0.975)
  in_ci <- abs(resid) <= z95 * loo_sd[valid]
  coverage <- mean(in_ci, na.rm = TRUE)

  list(
    observed    = y,
    predicted   = loo_pred,
    residuals   = loo_pred - y,
    rmse        = rmse,
    r_squared   = r2,
    coverage_95 = coverage
  )
}


# --------------- Posterior summary & visualisation ---------------------------

#' Summarise posterior samples
#'
#' @param samples Data.frame from [posterior_sample()].
#' @param prob Width of credible interval (default 0.95).
#' @return A data.frame with posterior summaries per parameter.
#' @export
posterior_summary <- function(samples, prob = 0.95) {
  if (nrow(samples) == 0) return(data.frame())
  alpha <- (1 - prob) / 2
  qs    <- c(alpha, 0.5, 1 - alpha)

  params <- setdiff(names(samples), "loglik")
  out <- data.frame(parameter = params, stringsAsFactors = FALSE)
  out$mean   <- vapply(params, function(p) mean(samples[[p]], na.rm = TRUE), numeric(1))
  out$sd     <- vapply(params, function(p) stats::sd(samples[[p]], na.rm = TRUE), numeric(1))

  quantiles <- t(vapply(params, function(p) {
    stats::quantile(samples[[p]], probs = qs, na.rm = TRUE)
  }, numeric(3)))
  colnames(quantiles) <- c("lower", "median", "upper")
  cbind(out, as.data.frame(quantiles), row.names = NULL)
}


#' Plot posterior density of a parameter
#'
#' @param samples Data.frame from [posterior_sample()].
#' @param param Character; name of the parameter to plot.
#' @param ... Extra arguments passed to [graphics::hist()].
#' @return Invisible `NULL`; a plot is produced as a side-effect.
#' @export
plot_posterior <- function(samples, param, ...) {
  if (!param %in% names(samples)) {
    stop("Parameter '", param, "' not found in samples", call. = FALSE)
  }
  vals <- samples[[param]]
  graphics::hist(vals, 40, freq = FALSE, col = "steelblue",
                 border = "white", main = paste("Posterior:", param),
                 xlab = param, ...)
  graphics::lines(stats::density(vals, na.rm = TRUE), lwd = 2, col = "red")
  invisible(NULL)
}
