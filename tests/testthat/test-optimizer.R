# Tests for R/optimizer.R

test_that("default_optimizer_config has required fields", {
  cfg <- default_optimizer_config()
  expect_true("n_init" %in% names(cfg))
  expect_true("n_iter" %in% names(cfg))
  expect_true("ts_min_sd" %in% names(cfg))
  expect_true("distribution" %in% names(cfg))
  expect_true("sigma_initial" %in% names(cfg))
  expect_true("n_candidates" %in% names(cfg))
})


test_that("optimize_resistance_enzyme errors on NULL inputs", {
  expect_error(
    optimize_resistance_enzyme(NULL, NULL)
  )
})


test_that(".create_lhs_design generates correct dimensions", {
  bounds <- list(r_0 = c(-2, 2), z_1 = c(-3, 3))
  design <- .create_lhs_design(10, bounds)
  expect_equal(nrow(design), 10)
  expect_equal(ncol(design), 2)
  expect_equal(names(design), c("r_0", "z_1"))

  # Values should be within bounds
  expect_true(all(design$r_0 >= -2 & design$r_0 <= 2))
  expect_true(all(design$z_1 >= -3 & design$z_1 <= 3))
})


test_that(".fit_surrogate fits a GP model", {
  skip_on_cran()
  set.seed(1)
  X <- matrix(runif(20), ncol = 2)
  colnames(X) <- c("r_0", "z_1")
  y <- rowSums(X^2)

  gp <- .fit_surrogate(X, y)
  expect_s4_class(gp, "km")
})


test_that(".thompson_sampling returns one value per candidate", {
  skip_on_cran()
  set.seed(1)
  X <- matrix(runif(20), ncol = 2)
  colnames(X) <- c("r_0", "z_1")
  y <- rowSums(X^2)
  gp <- .fit_surrogate(X, y)

  cand <- matrix(runif(10), ncol = 2)
  colnames(cand) <- c("r_0", "z_1")
  ts <- .thompson_sampling(cand, gp)
  expect_length(ts, 5)
})


test_that(".generate_candidates returns correct dimensions", {
  bounds <- list(r_0 = c(-2, 2), z_1 = c(-3, 3))
  cand <- .generate_candidates(100, bounds, best_point = c(0, 0),
                                sigma_vector = c(0.1, 0.1),
                                local_frac = 0.5)
  expect_equal(ncol(cand), 2)
  expect_equal(nrow(cand), 100)
})
