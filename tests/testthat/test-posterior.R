# Tests for R/posterior.R

test_that("posterior_summary works on sample data.frame", {
  samples <- data.frame(
    r_0    = rnorm(100, 1, 0.2),
    z_1    = rnorm(100, 0.5, 0.3),
    loglik = rnorm(100, -200, 10)
  )

  summ <- posterior_summary(samples)
  expect_s3_class(summ, "data.frame")
  expect_true("parameter" %in% names(summ))
  expect_true("mean" %in% names(summ))
  expect_true("sd" %in% names(summ))
  expect_true("lower" %in% names(summ))
  expect_true("upper" %in% names(summ))
  expect_equal(nrow(summ), 2)  # r_0 and z_1 (not loglik)
})


test_that("posterior_summary returns empty for empty input", {
  summ <- posterior_summary(data.frame())
  expect_equal(nrow(summ), 0)
})


test_that("plot_posterior runs without error", {
  skip_on_cran()
  samples <- data.frame(r_0 = rnorm(100), z_1 = rnorm(100))

  expect_silent({
    pdf(tempfile(fileext = ".pdf"))
    plot_posterior(samples, "r_0")
    dev.off()
  })
})


test_that("plot_posterior errors on missing parameter", {
  samples <- data.frame(r_0 = rnorm(10))
  expect_error(plot_posterior(samples, "z_99"), "not found")
})


test_that("posterior_summary quantile width grows with uncertainty", {
  set.seed(5)
  # Narrow distribution -> narrow interval
  narrow  <- data.frame(r_0 = rnorm(1000, mean = 1, sd = 0.1))
  # Wide distribution -> wider interval
  wide    <- data.frame(r_0 = rnorm(1000, mean = 1, sd = 2.0))

  summ_narrow <- posterior_summary(narrow)
  summ_wide   <- posterior_summary(wide)

  width_narrow <- summ_narrow$upper - summ_narrow$lower
  width_wide   <- summ_wide$upper - summ_wide$lower

  expect_true(width_wide > width_narrow)
})

test_that("posterior_summary mean and sd match sample statistics", {
  set.seed(6)
  samples <- data.frame(r_0 = rnorm(500, mean = 3, sd = 0.5),
                        z_1 = rnorm(500, mean = -1, sd = 1.0))
  summ <- posterior_summary(samples)

  r_row  <- summ[summ$parameter == "r_0", ]
  z_row  <- summ[summ$parameter == "z_1", ]

  expect_equal(r_row$mean, mean(samples$r_0), tolerance = 1e-10)
  expect_equal(r_row$sd,   sd(samples$r_0),   tolerance = 1e-10)
  expect_equal(z_row$mean, mean(samples$z_1), tolerance = 1e-10)
})

test_that("posterior_summary respects custom prob argument", {
  set.seed(7)
  samples <- data.frame(r_0 = rnorm(1000))

  summ_95 <- posterior_summary(samples, prob = 0.95)
  summ_50 <- posterior_summary(samples, prob = 0.50)

  # 50% CI must be narrower than 95% CI
  width_95 <- summ_95$upper - summ_95$lower
  width_50 <- summ_50$upper - summ_50$lower

  expect_true(width_50 < width_95)
})

test_that("posterior_summary lower < median < upper for unimodal distribution", {
  set.seed(8)
  samples <- data.frame(r_0 = rnorm(200, 2, 0.3))
  summ    <- posterior_summary(samples)

  expect_true(summ$lower < summ$median)
  expect_true(summ$median < summ$upper)
})

test_that("posterior_summary excludes loglik column from parameters", {
  samples <- data.frame(r_0 = rnorm(100), z_1 = rnorm(100),
                        loglik = rnorm(100, -200, 5))
  summ <- posterior_summary(samples)
  expect_false("loglik" %in% summ$parameter)
  expect_equal(nrow(summ), 2)
})


# ---------------------------------------------------------------------------
# laplace_resistance (surrogate path)
# ---------------------------------------------------------------------------

.make_surrogate_opt_result <- function(seed = 1) {
  set.seed(seed)
  bounds <- list(r_0 = c(-2, 2), z_1 = c(-3, 3))
  X      <- .create_lhs_design(15, bounds)
  y      <- rowSums(as.matrix(X)^2) + rnorm(15, sd = 0.05)
  gp     <- .fit_surrogate(as.matrix(X), y)

  best_idx <- which.min(y)
  best_params <- list(r_0 = X$r_0[best_idx], z_1 = X$z_1[best_idx])

  list(
    best_params = best_params,
    surrogate   = gp,
    bounds      = bounds
  )
}

test_that("laplace_resistance returns correct structure via surrogate path", {
  skip_on_cran()
  skip_if_not_installed("numDeriv")
  opt_result <- .make_surrogate_opt_result()

  lap <- suppressWarnings(laplace_resistance(opt_result))

  expect_named(lap, c("mode", "covariance", "precision", "std_error"))
  expect_length(lap$mode, 2)
  expect_equal(nrow(lap$covariance), 2)
  expect_equal(ncol(lap$covariance), 2)
  expect_length(lap$std_error, 2)
})


test_that("laplace_resistance mode matches best_params", {
  skip_on_cran()
  skip_if_not_installed("numDeriv")
  opt_result <- .make_surrogate_opt_result(seed = 2)

  lap  <- suppressWarnings(laplace_resistance(opt_result))
  best <- unlist(opt_result$best_params)

  expect_equal(lap$mode, unname(best), tolerance = 1e-10)
})


test_that("laplace_resistance std_error is non-negative", {
  skip_on_cran()
  skip_if_not_installed("numDeriv")
  opt_result <- .make_surrogate_opt_result(seed = 3)

  lap <- suppressWarnings(laplace_resistance(opt_result))
  expect_true(all(lap$std_error >= 0))
})


test_that("laplace_resistance covariance is symmetric", {
  skip_on_cran()
  skip_if_not_installed("numDeriv")
  opt_result <- .make_surrogate_opt_result(seed = 4)

  lap <- suppressWarnings(laplace_resistance(opt_result))
  expect_equal(lap$covariance, t(lap$covariance), tolerance = 1e-10)
})


test_that("laplace_resistance warns when refit=TRUE but basis_stack/obs_points absent", {
  skip_on_cran()
  skip_if_not_installed("numDeriv")
  opt_result <- .make_surrogate_opt_result(seed = 5)

  expect_warning(
    laplace_resistance(opt_result),
    regexp = "falling back to surrogate-mean Hessian"
  )
})


# ---------------------------------------------------------------------------
# nearPD fallback warnings (issue #16)
# ---------------------------------------------------------------------------

test_that("laplace_resistance warns when Hessian is not positive-definite", {
  skip_on_cran()
  skip_if_not_installed("numDeriv")
  opt <- .make_surrogate_opt_result()
  # H = -[[1,1],[1,1]] (rank-1 NSD) => neg_H = [[1,1],[1,1]] (PSD, singular)
  # solve(neg_H) fails; nearPD can handle PSD input
  local_mocked_bindings(
    hessian = function(func, x, ...) matrix(c(-1, -1, -1, -1), 2, 2),
    .package = "numDeriv"
  )
  expect_warning(
    suppressMessages(laplace_resistance(opt, refit = FALSE)),
    regexp = "not positive-definite"
  )
})

test_that("laplace_resistance nearPD warning mentions check_basis_correlations", {
  skip_on_cran()
  skip_if_not_installed("numDeriv")
  opt <- .make_surrogate_opt_result()
  local_mocked_bindings(
    hessian = function(func, x, ...) matrix(c(-1, -1, -1, -1), 2, 2),
    .package = "numDeriv"
  )
  expect_warning(
    suppressMessages(laplace_resistance(opt, refit = FALSE)),
    regexp = "check_basis_correlations"
  )
})

test_that("posterior_sample warns when covariance is not positive-definite", {
  skip_on_cran()
  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  laplace     <- list(mode       = c(r_0 = 0, z_1 = 0),
                      # PSD but singular (rank-1): chol fails, nearPD can handle it
                      covariance = matrix(c(1, 1, 1, 1), 2, 2))
  opt_result  <- list(bounds       = list(r_0 = c(-2, 2), z_1 = c(-3, 3)),
                      distribution = "nb")
  obs_points  <- data.frame(x = 0, y = 0)
  local_mocked_bindings(
    evaluate_full_model = function(...) list(
      loglik           = -100,
      intensity_params = c(alpha = 1),
      intensity_se     = c(alpha = 0.1),
      convergence      = 0L
    ),
    params_vector_to_list = function(theta, n_basis) list(r_0 = theta[1]),
    .package = "DiffiScape"
  )
  expect_warning(
    posterior_sample(laplace, opt_result, basis_stack, obs_points,
                     n_draws = 1L, n_inner = 1L),
    "not positive-definite"
  )
})


# ---------------------------------------------------------------------------
# posterior_sample — plug-in approximation warnings (issue #17)
# ---------------------------------------------------------------------------

.make_minimal_posterior_args <- function() {
  list(
    laplace     = list(mode       = c(r_0 = 0),
                       covariance = matrix(0.01, 1, 1)),
    opt_result  = list(bounds       = list(r_0 = c(-2, 2)),
                       distribution = "nb"),
    basis_stack = terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1),
    obs_points  = data.frame(x = 0, y = 0)
  )
}

test_that("posterior_sample warns on non-zero inner convergence", {
  skip_on_cran()
  args <- .make_minimal_posterior_args()
  local_mocked_bindings(
    evaluate_full_model = function(...) list(
      loglik           = -100,
      intensity_params = c(alpha = 1),
      intensity_se     = c(alpha = 0.1),
      convergence      = 1L
    ),
    params_vector_to_list = function(theta, n_basis) list(r_0 = theta[1]),
    .package = "DiffiScape"
  )
  expect_warning(
    posterior_sample(args$laplace, args$opt_result,
                     args$basis_stack, args$obs_points,
                     n_draws = 1L, n_inner = 1L),
    "did not converge"
  )
})

test_that("posterior_sample warns when inner standard errors are NA", {
  skip_on_cran()
  args <- .make_minimal_posterior_args()
  local_mocked_bindings(
    evaluate_full_model = function(...) list(
      loglik           = -100,
      intensity_params = c(alpha = 1),
      intensity_se     = c(alpha = NA_real_),
      convergence      = 0L
    ),
    params_vector_to_list = function(theta, n_basis) list(r_0 = theta[1]),
    .package = "DiffiScape"
  )
  expect_warning(
    posterior_sample(args$laplace, args$opt_result,
                     args$basis_stack, args$obs_points,
                     n_draws = 1L, n_inner = 1L),
    "standard errors are NA"
  )
})


# ---------------------------------------------------------------------------
# loo_cv_surrogate (issue #40) - consistent vectors and surfaced failures
# ---------------------------------------------------------------------------

.make_loo_opt_result <- function(seed = 1, n = 15) {
  set.seed(seed)
  bounds <- list(r_0 = c(-2, 2), z_1 = c(-3, 3))
  X      <- .create_lhs_design(n, bounds)
  y      <- rowSums(as.matrix(X)^2) + rnorm(n, sd = 0.05)
  gp     <- .fit_surrogate(as.matrix(X), y)

  list(
    X_evaluated = X,
    y_evaluated = y,
    surrogate   = gp,
    bounds      = bounds
  )
}

test_that("loo_cv_surrogate returns observed/predicted/residuals of equal, consistent length", {
  skip_on_cran()
  opt_result <- .make_loo_opt_result()

  res <- loo_cv_surrogate(opt_result)

  expect_equal(length(res$observed), length(res$predicted))
  expect_equal(length(res$observed), length(res$residuals))
  expect_equal(res$residuals, res$observed - res$predicted)
})

test_that("loo_cv_surrogate rmse/r_squared are computed from the returned residuals", {
  skip_on_cran()
  opt_result <- .make_loo_opt_result(seed = 2)

  res <- loo_cv_surrogate(opt_result)

  expect_equal(res$rmse, sqrt(mean(res$residuals^2)), tolerance = 1e-10)

  ss_res <- sum(res$residuals^2)
  ss_tot <- sum((res$observed - mean(res$observed))^2)
  expect_equal(res$r_squared, 1 - ss_res / ss_tot, tolerance = 1e-10)
})

test_that("loo_cv_surrogate reports n_failed = 0 when all LOO fits succeed", {
  skip_on_cran()
  opt_result <- .make_loo_opt_result(seed = 3)

  res <- loo_cv_surrogate(opt_result)

  expect_identical(res$n_failed, 0L)
  expect_equal(length(res$observed), length(opt_result$y_evaluated))
})

test_that("loo_cv_surrogate surfaces n_failed and shrinks vectors when LOO fits fail", {
  skip_on_cran()
  opt_result <- .make_loo_opt_result(seed = 4)
  n <- length(opt_result$y_evaluated)

  # Force the 1st and 3rd leave-one-out refits to fail. loo_cv_surrogate()
  # dispatches through fit_surrogate() -> .fit_surrogate_gp() for the "gp"
  # backend, so that's the function to intercept (not the legacy
  # .fit_surrogate() wrapper, which is no longer on this code path).
  real_fit_surrogate_gp <- .fit_surrogate_gp
  call_count <- 0L
  local_mocked_bindings(
    .fit_surrogate_gp = function(X, y, config = list()) {
      call_count <<- call_count + 1L
      if (call_count %in% c(1L, 3L)) stop("forced failure")
      real_fit_surrogate_gp(X, y, config)
    },
    .package = "DiffiScape"
  )

  res <- loo_cv_surrogate(opt_result)

  expect_identical(res$n_failed, 2L)
  expect_equal(length(res$observed), n - 2L)
  expect_equal(length(res$predicted), n - 2L)
  expect_equal(length(res$residuals), n - 2L)
  expect_equal(res$residuals, res$observed - res$predicted)
})


test_that("loo_cv_surrogate threads opt_result$config$surrogate_config through to LOO refits", {
  skip_on_cran()
  opt_result <- .make_loo_opt_result(seed = 5)
  opt_result$surrogate <- structure(
    list(type = "gp", model = opt_result$surrogate, predictor_names = colnames(opt_result$X_evaluated)),
    class = "ds_surrogate"
  )
  opt_result$config <- list(surrogate_type = "gp",
                            surrogate_config = list(nugget.estim = FALSE, nugget = 0.5))

  seen_configs <- list()
  real_fit_surrogate <- fit_surrogate
  local_mocked_bindings(
    fit_surrogate = function(X, y, type = c("gp", "rf"), config = list()) {
      seen_configs[[length(seen_configs) + 1L]] <<- config
      real_fit_surrogate(X, y, type = type, config = config)
    },
    .package = "DiffiScape"
  )

  loo_cv_surrogate(opt_result)

  expect_true(length(seen_configs) > 0)
  for (cfg in seen_configs) {
    expect_identical(cfg, list(nugget.estim = FALSE, nugget = 0.5))
  }
})


# ---------------------------------------------------------------------------
# posterior_sample — list accumulation correctness (issue #33)
# ---------------------------------------------------------------------------

test_that("posterior_sample returns n_draws * n_inner rows with no missing blocks", {
  skip_on_cran()
  args <- .make_minimal_posterior_args()
  n_draws <- 4L
  n_inner <- 3L
  local_mocked_bindings(
    evaluate_full_model = function(...) list(
      loglik           = -100,
      intensity_params = c(alpha = 1),
      intensity_se     = c(alpha = 0.1),
      convergence      = 0L
    ),
    params_vector_to_list = function(theta, n_basis) list(r_0 = theta[1]),
    .package = "DiffiScape"
  )

  samples <- posterior_sample(args$laplace, args$opt_result,
                              args$basis_stack, args$obs_points,
                              n_draws = n_draws, n_inner = n_inner)

  expect_s3_class(samples, "data.frame")
  expect_equal(nrow(samples), n_draws * n_inner)
  # No NULL/NA blocks: every row must be fully populated (no NA rows
  # introduced by leftover/misaligned list slots)
  expect_false(anyNA(samples))
  expect_true(all(vapply(seq_len(nrow(samples)), function(i) {
    is.numeric(samples$r_0[i]) && is.numeric(samples$alpha[i]) &&
      is.numeric(samples$loglik[i])
  }, logical(1))))
})
