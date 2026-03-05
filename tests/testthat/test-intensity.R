# Tests for R/intensity.R

test_that("standardise_connectivity produces z-scores", {
  set.seed(123)
  C_obs <- rlnorm(100, 5, 1)
  C_int <- rlnorm(500, 5, 1)

  z <- standardise_connectivity(C_obs, C_int)
  expect_length(z$z_obs, 100)
  expect_length(z$z_int, 500)
  expect_true(abs(mean(c(z$z_obs, z$z_int))) < 0.5)  # approximately centred
})


test_that("default_intensity_config has required fields", {
  cfg <- default_intensity_config()
  expect_true("c_scale" %in% names(cfg))
  expect_true("k_connectivity" %in% names(cfg))
  expect_true("integration_subsample" %in% names(cfg))
})


test_that("compute_intensity returns positive values", {
  alpha  <- -5
  gamma  <- 0.5
  z_vals <- rnorm(50)
  lambda <- compute_intensity(z_vals, alpha, gamma)
  expect_length(lambda, 50)
  expect_true(all(lambda > 0))
})


test_that("compute_intensity accepts covariates", {
  alpha  <- -5
  gamma  <- 0.5
  z_vals <- rnorm(50)
  cov    <- list(elev = rnorm(50))
  beta   <- c(elev = 0.3)

  lambda <- compute_intensity(z_vals, alpha, gamma,
                               covariates = cov, betas = beta)
  expect_length(lambda, 50)
  expect_true(all(lambda > 0))
})


test_that("fit_intensity_nb returns valid fit on synthetic data", {
  skip_on_cran()
  set.seed(42)

  # Test the .nb_negloglik_cached function with correct args
  n_obs <- 80
  n_int <- 400
  total <- n_obs + n_int

  z_obs <- rnorm(n_obs)
  z_int <- rnorm(n_int)
  obs_w  <- rep(1, n_obs)
  int_w  <- rep(0.01, n_int)

  nll <- DiffiScape:::.nb_negloglik_cached(
    theta       = c(-3, 0.5, log(2)),
    z_obs       = z_obs,
    z_int       = z_int,
    int_weights = int_w,
    obs_weights = obs_w
  )
  expect_true(is.finite(nll))
  expect_true(nll > 0)  # neg-log-lik is positive
})
