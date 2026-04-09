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


# ---- Profile likelihood tests ----

# Helper: build a mock opt_result with a GP surrogate for profile tests
.mock_opt_result <- function(seed = 1L) {
  set.seed(seed)
  # 2-parameter problem: r_0, z_1
  bounds <- list(r_0 = c(0, 6), z_1 = c(-3, 3))
  n <- 30
  X <- data.frame(
    r_0 = runif(n, 0, 6),
    z_1 = runif(n, -3, 3)
  )
  # Quadratic objective with known min near (3, 0)
  y <- (X$r_0 - 3)^2 + (X$z_1 - 0)^2 + rnorm(n, 0, 0.1)

  surrogate <- .fit_surrogate(as.matrix(X), y)

  best_idx <- which.min(y)
  best_vec <- as.numeric(X[best_idx, ])
  best_params <- list(r_0 = best_vec[1], z_1 = best_vec[2])

  list(
    best_params   = best_params,
    best_loglik   = -y[best_idx],
    best_idx      = best_idx,
    X_evaluated   = X,
    y_evaluated   = y,
    n_evaluations = n,
    surrogate     = surrogate,
    bounds        = bounds,
    distribution  = "negbin",
    config        = default_optimizer_config()
  )
}


test_that("profile_loglik returns correct structure", {
  skip_on_cran()
  opt <- .mock_opt_result()

  prof <- profile_loglik(opt, "r_0", n_points = 20)

  expect_type(prof, "list")
  expect_equal(prof$param, "r_0")
  expect_length(prof$values, 20)
  expect_length(prof$profile_loglik, 20)
  expect_true(is.numeric(prof$max_loglik))
  expect_true(is.numeric(prof$mle))
})


test_that("profile_loglik works with a custom grid", {
  skip_on_cran()
  opt <- .mock_opt_result()

  grid <- seq(1, 5, length.out = 10)
  prof <- profile_loglik(opt, "z_1", grid = grid)

  expect_equal(prof$values, grid)
  expect_length(prof$profile_loglik, 10)
})


test_that("profile_loglik errors on unknown parameter", {
  skip_on_cran()
  opt <- .mock_opt_result()
  expect_error(profile_loglik(opt, "z_99"), "not found")
})


test_that("profile_loglik errors without surrogate", {
  opt <- .mock_opt_result()
  opt$surrogate <- NULL
  expect_error(profile_loglik(opt, "r_0"), "No GP surrogate")
})


test_that("profile_ci returns data.frame with correct columns", {
  skip_on_cran()
  opt <- .mock_opt_result()

  ci <- profile_ci(opt, level = 0.95, n_points = 20)

  expect_s3_class(ci, "data.frame")
  expect_true(all(c("parameter", "mle", "lower", "upper", "level")
                  %in% names(ci)))
  expect_equal(nrow(ci), 2)  # r_0, z_1
  expect_true(all(ci$level == 0.95))
})


test_that("profile_ci intervals contain the MLE", {
  skip_on_cran()
  opt <- .mock_opt_result()

  ci <- profile_ci(opt, level = 0.95, n_points = 30)

  for (i in seq_len(nrow(ci))) {
    if (!is.na(ci$lower[i]) && !is.na(ci$upper[i])) {
      expect_true(ci$lower[i] <= ci$mle[i],
                  info = paste("lower bound > MLE for", ci$parameter[i]))
      expect_true(ci$upper[i] >= ci$mle[i],
                  info = paste("upper bound < MLE for", ci$parameter[i]))
    }
  }
})


test_that("profile_ci has profiles attribute", {
  skip_on_cran()
  opt <- .mock_opt_result()

  ci <- profile_ci(opt, n_points = 15)
  profiles <- attr(ci, "profiles")

  expect_type(profiles, "list")
  expect_equal(length(profiles), 2)
  expect_true(all(c("r_0", "z_1") %in% names(profiles)))
})


test_that("plot_profile produces a plot without error", {
  skip_on_cran()
  opt  <- .mock_opt_result()
  prof <- profile_loglik(opt, "r_0", n_points = 15)

  expect_silent({
    pdf(tempfile(fileext = ".pdf"))
    plot_profile(prof)
    dev.off()
  })
})


test_that("ds_profile wrapper returns ci and profiles", {
  skip_on_cran()
  opt <- .mock_opt_result()

  result <- ds_profile(opt, n_points = 15, plot = FALSE)

  expect_type(result, "list")
  expect_true("ci" %in% names(result))
  expect_true("profiles" %in% names(result))
  expect_s3_class(result$ci, "data.frame")
  expect_equal(nrow(result$ci), 2)
})
