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
#' resistance parameters by numerically differentiating the true PPP
#' log-likelihood at the MAP estimate (default).  When `basis_stack` and
#' `obs_points` are not supplied, the function falls back to differentiating
#' the GP surrogate's predicted-mean surface and emits a warning.
#'
#' @param opt_result Result from [optimize_resistance()] or
#'   [optimize_resistance_gradient()].
#' @param basis_stack A [terra::SpatRaster] of basis functions (required
#'   when `refit = TRUE` or when no surrogate is available).
#' @param obs_points Data.frame with `x, y` (required when `refit = TRUE`).
#' @param refit Logical; when `TRUE` (the default) the Hessian is computed
#'   on the true PPP log-likelihood via the JAX solver, which requires
#'   `basis_stack` and `obs_points`.  When both of those are `NULL` and a
#'   GP surrogate is available, a warning is emitted and the function falls
#'   back to the surrogate-mean Hessian automatically so that existing
#'   single-argument calls remain valid.  Set `refit = FALSE` explicitly to
#'   suppress the warning and force the surrogate path.  Always set to
#'   `TRUE` automatically when `opt_result` has no `$surrogate` (i.e., from
#'   [optimize_resistance_gradient()]).
#' @param step Finite-difference step size for Hessian computation.
#' @param omniscape_settings Named list of solver overrides: `radius`
#'   (default `13L`) and `block_size` (default `5L`). Used when `refit = TRUE`.
#' @param intensity_config Optional [default_intensity_config()] list.
#' @param covariates_obs Named list of covariate vectors.
#' @param covariates_rasters Named list of covariate rasters.
#' @param residualise Logical.
#' @param available_points Optional data.frame with `x, y` columns of
#'   available/background locations for selection function families.  When
#'   supplied, bypasses raster quadrature and uses these locations with unit
#'   weights instead.  `NULL` (default) uses standard area-weighted raster
#'   integration.
#' @param available_covariates Named list of covariate vectors at
#'   `available_points` locations.  Required when `available_points` is
#'   supplied and the intensity model includes covariates.
#' @param link A [resistance_link] object (default [link_exp()]).
#' @param family An [intensity_family] object, or `NULL`.
#' @return A list with `mode`, `covariance`, `precision`, `std_error`.
#' @export
laplace_resistance <- function(opt_result,
                               basis_stack          = NULL,
                               obs_points           = NULL,
                               refit                = TRUE,
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

  n_basis  <- length(opt_result$bounds) - 1L
  best_vec <- .params_to_vector(opt_result$best_params, n_basis)
  p        <- length(best_vec)

  # Auto-refit when no surrogate is available (gradient/direct path)
  if (is.null(opt_result$surrogate)) refit <- TRUE

  # Graceful fallback: refit requested but full-model inputs missing
  if (refit && !is.null(opt_result$surrogate) &&
      (is.null(basis_stack) || is.null(obs_points))) {
    warning(
      "refit = TRUE requires basis_stack and obs_points; ",
      "falling back to surrogate-mean Hessian. ",
      "Supply basis_stack and obs_points for statistically correct intervals.",
      call. = FALSE
    )
    refit <- FALSE
  }

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
    # Direct Hessian via the JAX solver
    distribution <- opt_result$distribution %||% "negbin"
    n_basis      <- terra::nlyr(basis_stack)
    basis_vals   <- terra::values(basis_stack)
    template     <- basis_stack[[1]]
    omni_def      <- list(radius = 13L, block_size = 5L)
    omni_cfg      <- utils::modifyList(omni_def, omniscape_settings)
    solver_radius <- omni_cfg$radius
    solver_block  <- omni_cfg$block_size

    refit_fn <- function(theta) {
      resistance <- create_resistance_surface(
        params_vector_to_list(theta, n_basis),
        basis_stack, link = link
      )

      jax_result <- ds_jax_connectivity(
        resistance,
        radius     = solver_radius,
        block_size = solver_block
      )
      cum_rast <- jax_result$cum_current

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
    param_names <- names(opt_result$bounds)
    flat_idx    <- which(diag(neg_H) <= 0)
    param_note  <- if (length(flat_idx) > 0)
      paste0("; near-zero curvature in: ",
             paste(param_names[flat_idx], collapse = ", "))
    else ""
    warning(
      "Hessian is not positive-definite: the MAP is not a proper local maximum",
      param_note, ". ",
      "This signals non-identifiability; basis layers may be highly correlated. ",
      "The Laplace approximation is unreliable and credible intervals may be invalid. ",
      "Run check_basis_correlations() and consider reducing K or adding informative priors ",
      "before trusting these results.",
      call. = FALSE
    )
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
#' @param available_points Optional data.frame with `x, y` columns of
#'   available/background locations for selection function families.  When
#'   supplied, bypasses raster quadrature and uses these locations with unit
#'   weights instead.  `NULL` (default) uses standard area-weighted raster
#'   integration.
#' @param available_covariates Named list of covariate vectors at
#'   `available_points` locations.  Required when `available_points` is
#'   supplied and the intensity model includes covariates.
#' @param link A [resistance_link] object (default [link_exp()]).
#' @param family An [intensity_family] object, or `NULL`.
#' @param seed Random seed.
#' @details
#' **Approximation note.** The joint posterior produced by this function is a
#' plug-in (asymptotic) approximation, *not* a full Bayesian posterior.  The
#' outer resistance draw comes from the Laplace approximation to the PPP
#' log-likelihood; the inner intensity draw is an independent normal centred on
#' the conditional MLE with frequentist asymptotic standard errors.  Because the
#' two stages are sampled independently, the correlation between the inner
#' standard errors and the outer \eqn{\theta} draw is ignored, which can cause
#' joint credible intervals -- especially for connectivity parameters such as
#' \eqn{\gamma} -- to be too narrow.  Interpret credible intervals as approximate
#' guides rather than exact probability statements.
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
    small_idx  <- which(diag(sigma) < .Machine$double.eps * max(diag(sigma)))
    param_note <- if (length(small_idx) > 0)
      paste0("; near-zero variance in: ",
             paste(pnames[small_idx], collapse = ", "))
    else ""
    warning(
      "Posterior covariance is not positive-definite",
      param_note, ". ",
      "This signals non-identifiability; basis layers may be highly correlated. ",
      "Samples are drawn from a nearPD approximation and credible intervals may be invalid. ",
      "Run check_basis_correlations() and consider reducing K or adding informative priors.",
      call. = FALSE
    )
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

    if (!is.null(result$convergence) && result$convergence != 0L) {
      warning(sprintf(
        "posterior draw %d: inner MLE did not converge (convergence = %d); intensity estimates may be unreliable.",
        d, result$convergence
      ), call. = FALSE)
    }

    # inner-loop draws from asymptotic Normal
    inner_mu <- result$intensity_params
    inner_se <- result$intensity_se

    if (any(is.na(inner_se))) {
      warning(sprintf(
        "posterior draw %d: some inner standard errors are NA; using point estimate only.",
        d
      ), call. = FALSE)
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
#'   `r_squared`, `coverage_95`, `n_failed`. The `observed`, `predicted`,
#'   and `residuals` vectors are all restricted to the subset of points
#'   for which the leave-one-out refit succeeded (i.e. they are the same
#'   length and in the same order as each other, and are the exact values
#'   used to compute `rmse`, `r_squared`, and `coverage_95`). `n_failed`
#'   reports how many of the `n` leave-one-out fits failed and were
#'   dropped from these vectors and summary statistics.
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
    observed    = y[valid],
    predicted   = loo_pred[valid],
    residuals   = resid,
    rmse        = rmse,
    r_squared   = r2,
    coverage_95 = coverage,
    n_failed    = sum(!valid)
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
