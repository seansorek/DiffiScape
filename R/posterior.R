# ============================================================================
# Posterior inference
# Doing full Bayesian inference through graphical laplacians is a whole can of worms.
# Laplace approximation for resistance params, Monte Carlo composition
# for joint posterior, GP emulator validation (LOO-CV).
# Profile likelihood CIs for resistance params via Wilks' theorem.
# ============================================================================

# --------------- Laplace approximation for resistance params ----------------

#' Laplace approximation at the optimised resistance parameters
#'
#' Computes a Gaussian approximation to the marginal posterior of the
#' resistance parameters by evaluating (or re-using) the GP surrogate
#' at the MAP estimate.
#'
#' @param opt_result Result from [optimize_resistance()].
#' @param basis_stack A [terra::SpatRaster] of basis functions (used to
#'   count dimensions; not re-run unless `refit = TRUE`).
#' @param refit Logical; re-evaluate the Hessian around the MAP using the
#'   full model instead of the surrogate.
#' @param step Finite-difference step size for Hessian computation if refitting.
#' @return A list with `mode`, `covariance`, `precision`, `std_error`.
#' @export
laplace_resistance <- function(opt_result,
                               basis_stack = NULL,
                               refit = FALSE,
                               step  = 1e-3) {

  best_vec <- .params_to_vector(opt_result$best_params)
  p        <- length(best_vec)

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

  } else if (refit && !is.null(basis_stack)) {
    stop("Full-model Hessian refit not implemented yet. ",
         "Use refit = FALSE to approximate from the surrogate.",
         call. = FALSE)
  } else {
    stop("Must supply either opt_result$surrogate or basis_stack with refit=TRUE",
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
#' @param seed Random seed.
#' @return A data.frame with posterior samples (one row per draw).
#' @export
posterior_sample <- function(laplace,
                             opt_result,
                             basis_stack,
                             obs_points,
                             n_draws           = 200L,
                             n_inner           = 5L,
                             bounds            = NULL,
                             omniscape_settings = list(),
                             intensity_config   = default_intensity_config(),
                             covariates_obs     = NULL,
                             covariates_rasters = NULL,
                             residualise        = FALSE,
                             seed               = 42L) {

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
        resistance_params  = params,
        basis_stack        = basis_stack,
        obs_points         = obs_points,
        distribution       = opt_result$distribution,
        omniscape_settings = omniscape_settings,
        intensity_config   = intensity_config,
        covariates_obs     = covariates_obs,
        covariates_rasters = covariates_rasters,
        residualise        = residualise,
        verbose            = FALSE
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


# --------------- Profile likelihood for resistance params --------------------

#' Compute profile log-likelihood for one resistance parameter
#'
#' For a grid of values of a single resistance parameter, optimises the
#' remaining parameters via the GP surrogate to obtain the profile
#' log-likelihood.  The surrogate predicts minus the log-likelihood
#' returned by the inner-loop MLE, so the profile is obtained by
#' minimising the surrogate prediction over the nuisance parameters at
#' each fixed value of the parameter of interest.
#'
#' @param opt_result Result from [optimize_resistance()].
#' @param param Character; name of the parameter to profile (e.g.
#'   `"r_0"` or `"z_1"`).
#' @param n_points Integer; number of grid points along the parameter
#'   axis (default 50).
#' @param range_mult Numeric; how many profile-likelihood standard
#'   errors (from the Laplace approximation) to extend the grid
#'   beyond the MLE in each direction (default 3).
#' @param grid Optional numeric vector of values at which to evaluate
#'   the profile.  When supplied, \code{n_points} and
#'   \code{range_mult} are ignored.
#' @return A list with components
#'   \describe{
#'     \item{param}{Name of the profiled parameter.}
#'     \item{values}{Numeric vector of parameter values evaluated.}
#'     \item{profile_loglik}{Profile log-likelihood at each value.}
#'     \item{max_loglik}{Maximum log-likelihood (at the MLE).}
#'     \item{mle}{MLE of the profiled parameter.}
#'   }
#' @export
profile_loglik <- function(opt_result,
                           param,
                           n_points   = 50L,
                           range_mult = 3,
                           grid       = NULL) {

  surrogate <- opt_result$surrogate
  if (is.null(surrogate)) {
    stop("No GP surrogate found in opt_result", call. = FALSE)
  }

  pnames <- names(opt_result$bounds)
  if (!param %in% pnames) {
    stop("Parameter '", param, "' not found in bounds (available: ",
         paste(pnames, collapse = ", "), ")", call. = FALSE)
  }

  n_basis  <- length(pnames) - 1L
  best_vec <- .params_to_vector(opt_result$best_params, n_basis)
  pidx     <- match(param, pnames)
  mle_val  <- best_vec[pidx]

  # Build the grid --------------------------------------------------------
  if (is.null(grid)) {
    # Determine a reasonable range from the Laplace approximation
    lap <- laplace_resistance(opt_result)
    se  <- lap$std_error[pidx]
    if (is.na(se) || se <= 0) se <- 0.1 * abs(mle_val) + 0.01

    lo <- max(opt_result$bounds[[param]][1], mle_val - range_mult * se)
    hi <- min(opt_result$bounds[[param]][2], mle_val + range_mult * se)
    grid <- seq(lo, hi, length.out = n_points)
  }

  # Profile: for each grid value, optimise nuisance params ----------------
  other_idx <- setdiff(seq_along(pnames), pidx)
  lower_all <- vapply(opt_result$bounds, `[`, numeric(1), 1)
  upper_all <- vapply(opt_result$bounds, `[`, numeric(1), 2)

  surrogate_fn <- function(theta_full) {
    x <- matrix(theta_full, nrow = 1)
    colnames(x) <- pnames
    stats::predict(surrogate, newdata = x, type = "UK")$mean
  }

  prof_ll <- numeric(length(grid))

  for (g in seq_along(grid)) {
    if (length(other_idx) == 0) {
      # Only one parameter — no nuisance to optimise
      theta_full <- best_vec
      theta_full[pidx] <- grid[g]
      prof_ll[g] <- -surrogate_fn(theta_full)
    } else {
      # Fix the profiled parameter, optimise the rest
      obj_fn <- function(theta_other) {
        theta_full <- numeric(length(pnames))
        theta_full[pidx]      <- grid[g]
        theta_full[other_idx] <- theta_other
        surrogate_fn(theta_full)
      }

      start <- best_vec[other_idx]
      lo_b  <- lower_all[other_idx]
      hi_b  <- upper_all[other_idx]

      opt <- tryCatch(
        stats::optim(
          par    = start,
          fn     = obj_fn,
          method = if (length(other_idx) > 1) "L-BFGS-B" else "Brent",
          lower  = lo_b,
          upper  = hi_b
        ),
        error = function(e) {
          list(value = surrogate_fn(best_vec))
        }
      )

      prof_ll[g] <- -opt$value
    }
  }

  max_ll <- -min(opt_result$y_evaluated)

  list(
    param          = param,
    values         = grid,
    profile_loglik = prof_ll,
    max_loglik     = max_ll,
    mle            = mle_val
  )
}


#' Profile likelihood confidence intervals via Wilks' theorem
#'
#' Computes profile-likelihood-based confidence intervals for each
#' resistance parameter.
#' By Wilks' theorem, \eqn{2[\ell(\hat\theta) - \ell_P(\theta_i)]}
#' is asymptotically \eqn{\chi^2(1)}, so the \eqn{(1 - \alpha)} CI
#' is the set of values where the deviance is below
#' \code{qchisq(1 - alpha, 1)}.
#'
#' @param opt_result Result from [optimize_resistance()].
#' @param level Confidence level (default 0.95).
#' @param n_points Grid resolution per parameter (default 50).
#' @param range_mult Multiplier on the Laplace SE for grid range
#'   (default 3).
#' @return A data.frame with one row per parameter and columns
#'   \code{parameter}, \code{mle}, \code{lower}, \code{upper},
#'   \code{level}.  An attribute \code{"profiles"} contains the
#'   full profile-likelihood objects (one per parameter) for further
#'   inspection.
#' @export
profile_ci <- function(opt_result,
                       level    = 0.95,
                       n_points = 50L,
                       range_mult = 3) {

  pnames  <- names(opt_result$bounds)
  cutoff  <- stats::qchisq(level, df = 1)

  profiles <- stats::setNames(
    lapply(pnames, function(p) {
      profile_loglik(opt_result, p,
                     n_points   = n_points,
                     range_mult = range_mult)
    }),
    pnames
  )

  rows <- lapply(pnames, function(p) {
    pr <- profiles[[p]]
    deviance <- 2 * (pr$max_loglik - pr$profile_loglik)
    in_ci    <- which(deviance <= cutoff)

    if (length(in_ci) == 0) {
      lo <- NA_real_; hi <- NA_real_
    } else {
      lo <- min(pr$values[in_ci])
      hi <- max(pr$values[in_ci])
    }

    data.frame(
      parameter = p,
      mle       = pr$mle,
      lower     = lo,
      upper     = hi,
      level     = level,
      stringsAsFactors = FALSE
    )
  })

  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  attr(out, "profiles") <- profiles
  out
}


#' Plot profile log-likelihood for a resistance parameter
#'
#' Draws the profile log-likelihood curve together with the
#' chi-squared cutoff line that defines the confidence interval.
#'
#' @param profile Result from [profile_loglik()] for a single
#'   parameter.
#' @param level Confidence level for the cutoff line (default 0.95).
#' @param ... Extra arguments passed to [graphics::plot()].
#' @return Invisible `NULL`; a plot is produced as a side-effect.
#' @export
plot_profile <- function(profile, level = 0.95, ...) {

  deviance <- 2 * (profile$max_loglik - profile$profile_loglik)
  cutoff   <- stats::qchisq(level, df = 1)

  graphics::plot(
    profile$values, deviance,
    type = "l", lwd = 2, col = "steelblue",
    xlab = profile$param,
    ylab = "Deviance  (2 * [max LL - profile LL])",
    main = paste("Profile likelihood:", profile$param),
    ...
  )
  graphics::abline(h = cutoff, lty = 2, col = "red", lwd = 1.5)
  graphics::abline(v = profile$mle, lty = 3, col = "grey40")

  # Shade the CI region
  in_ci <- which(deviance <= cutoff)
  if (length(in_ci) >= 2) {
    ci_lo <- min(profile$values[in_ci])
    ci_hi <- max(profile$values[in_ci])
    graphics::rect(
      ci_lo, graphics::par("usr")[3],
      ci_hi, cutoff,
      col = grDevices::adjustcolor("steelblue", alpha.f = 0.15),
      border = NA
    )
  }

  graphics::legend(
    "topright",
    legend = c(
      "Profile deviance",
      sprintf("%g%% cutoff (%.3f)", level * 100, cutoff),
      "MLE"
    ),
    col  = c("steelblue", "red", "grey40"),
    lty  = c(1, 2, 3),
    lwd  = c(2, 1.5, 1),
    bg   = "white"
  )

  invisible(NULL)
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
