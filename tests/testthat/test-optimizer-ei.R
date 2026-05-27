# Tests for the Expected Improvement acquisition path in R/optimizer.R

test_that("default_optimizer_config defaults to Thompson Sampling", {
  cfg <- default_optimizer_config()
  expect_equal(cfg$acquisition, "TS")
  expect_true("ei_xi_scale_factor" %in% names(cfg))
  expect_true("ei_xi_min" %in% names(cfg))
  expect_true("ei_decay_rate_divisor" %in% names(cfg))
  expect_equal(cfg$ei_xi_scale_factor, 0.1)
  expect_equal(cfg$ei_xi_min, 0.02)
  expect_equal(cfg$ei_decay_rate_divisor, 5)
})


test_that(".expected_improvement returns one EI value per candidate", {
  skip_on_cran()
  set.seed(1)
  X <- matrix(runif(20), ncol = 2)
  colnames(X) <- c("r_0", "z_1")
  y <- rowSums(X^2)
  gp <- .fit_surrogate(X, y)

  cand <- matrix(runif(10), ncol = 2)
  colnames(cand) <- c("r_0", "z_1")
  ei <- .expected_improvement(cand, gp, y_best = min(y),
                              xi_initial = 0.1,
                              iter = 1, n_iter = 10)
  expect_length(ei, 5)
  expect_true(all(is.finite(ei)))
  expect_true(all(ei >= 0))
})


test_that(".expected_improvement xi decays toward xi_min across iterations", {
  skip_on_cran()
  set.seed(2)
  X <- matrix(runif(20), ncol = 2)
  colnames(X) <- c("r_0", "z_1")
  y <- rowSums(X^2)
  gp <- .fit_surrogate(X, y)

  cand <- matrix(c(0.5, 0.5), nrow = 1)
  colnames(cand) <- c("r_0", "z_1")

  # Early iteration -> larger xi -> more exploration -> EI shifted by
  # an extra -xi_initial relative to a later iteration.
  ei_early <- .expected_improvement(cand, gp, y_best = min(y),
                                    xi_initial = 1.0,
                                    iter = 1, n_iter = 50,
                                    xi_min = 0.0)
  ei_late  <- .expected_improvement(cand, gp, y_best = min(y),
                                    xi_initial = 1.0,
                                    iter = 1000, n_iter = 50,
                                    xi_min = 0.0)
  # ei_late uses near-zero xi -> improvement is closer to (y_best - mu).
  # ei_early uses xi = 1 -> improvement is reduced by ~1.
  # Therefore ei_early <= ei_late by construction.
  expect_lte(ei_early, ei_late + 1e-6)
})


test_that(".expected_improvement returns 0 when sigma collapses", {
  # Build a GP that interpolates the training set exactly, then query
  # a training point so the predictive sigma is ~0.
  skip_on_cran()
  set.seed(3)
  X <- matrix(runif(20), ncol = 2)
  colnames(X) <- c("r_0", "z_1")
  y <- rowSums(X^2)
  gp <- .fit_surrogate(X, y)

  ei <- .expected_improvement(X[1, , drop = FALSE], gp, y_best = min(y),
                              xi_initial = 0.0,
                              iter = 1, n_iter = 1,
                              min_sd = 1e-3)
  # With sigma below min_sd and zero xi, EI must clamp to 0.
  expect_equal(ei, 0)
})
