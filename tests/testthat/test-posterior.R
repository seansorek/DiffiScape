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
# laplace_resistance (surrogate path — no Julia required)
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

  lap <- laplace_resistance(opt_result)

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

  lap  <- laplace_resistance(opt_result)
  best <- unlist(opt_result$best_params)

  expect_equal(lap$mode, unname(best), tolerance = 1e-10)
})


test_that("laplace_resistance std_error is non-negative", {
  skip_on_cran()
  skip_if_not_installed("numDeriv")
  opt_result <- .make_surrogate_opt_result(seed = 3)

  lap <- laplace_resistance(opt_result)
  expect_true(all(lap$std_error >= 0))
})


test_that("laplace_resistance covariance is symmetric", {
  skip_on_cran()
  skip_if_not_installed("numDeriv")
  opt_result <- .make_surrogate_opt_result(seed = 4)

  lap <- laplace_resistance(opt_result)
  expect_equal(lap$covariance, t(lap$covariance), tolerance = 1e-10)
})
