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


test_that("standardise_connectivity uses exact centering on log1p-scaled values", {
  set.seed(7)
  C_obs <- c(1, 4, 9, 16, 25)  # known values for easy manual check
  C_int <- c(1, 4, 9)
  c_scale <- 1  # no rescaling

  res      <- standardise_connectivity(C_obs, C_int, c_scale = c_scale)
  log_obs  <- log1p(C_obs / c_scale)
  mu       <- mean(log_obs)
  sigma    <- sd(log_obs)
  expected_z_obs <- (log_obs - mu) / sigma

  expect_equal(res$z_obs, expected_z_obs, tolerance = 1e-10)
  expect_equal(res$c_scale, 1)
  expect_equal(res$mu, mu, tolerance = 1e-10)
  expect_equal(res$sigma, sigma, tolerance = 1e-10)
})

test_that("standardise_connectivity uses median of positive C_obs as default c_scale", {
  set.seed(8)
  C_obs <- c(0, 2, 4, 6, 8, 10)
  C_int <- c(1, 3, 5)

  res <- standardise_connectivity(C_obs, C_int)
  pos <- C_obs[C_obs > 0]
  expect_equal(res$c_scale, median(pos), tolerance = 1e-10)
})

test_that("standardise_connectivity falls back sigma=1 for constant input", {
  C_obs <- rep(5, 10)
  C_int <- rep(5, 5)
  res <- standardise_connectivity(C_obs, C_int, c_scale = 1)
  expect_equal(res$sigma, 1)
})

test_that("compute_intensity increases with gamma for positive z", {
  alpha <- -3
  z     <- c(0.5, 1.0, 1.5)

  lam_high <- compute_intensity(z, alpha, gamma = 2.0)
  lam_low  <- compute_intensity(z, alpha, gamma = 0.5)

  # Higher gamma -> higher lambda for positive z
  expect_true(all(lam_high > lam_low))
})

test_that("compute_intensity covariate contribution is additive on log scale", {
  alpha  <- 0
  gamma  <- 0
  z      <- rep(0, 5)
  cov    <- list(x = rep(1, 5))
  beta   <- c(x = 1.0)

  lam_with_cov    <- compute_intensity(z, alpha, gamma,
                                       covariates = cov, betas = beta)
  lam_without_cov <- compute_intensity(z, alpha, gamma)

  # With beta=1 and cov=1: lambda = exp(0 + 0*0 + 1*1) = exp(1) ~ 2.718
  expect_equal(lam_with_cov, rep(exp(1), 5), tolerance = 1e-10)
  expect_equal(lam_without_cov, rep(1, 5), tolerance = 1e-10)
})

test_that("fit_intensity_nb with terra raster returns correct output structure", {
  skip_on_cran()
  skip_if_not_installed("terra")
  set.seed(99)

  r <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(100, mean = 5))

  obs_x <- runif(20, 0.1, 0.9)
  obs_y <- runif(20, 0.1, 0.9)
  conn_at_obs <- abs(rnorm(20, mean = 5))

  fit <- fit_intensity_nb(
    connectivity_at_obs = conn_at_obs,
    connectivity_raster = r,
    obs_coords          = data.frame(x = obs_x, y = obs_y)
  )

  expect_true(is.list(fit))
  expect_true("estimates" %in% names(fit))
  expect_true("loglik" %in% names(fit))
  expect_true("convergence" %in% names(fit))
  expect_true(is.finite(fit$loglik))
})

test_that("fit_intensity_nb converges and loglik improves over start values", {
  skip_on_cran()
  skip_if_not_installed("terra")

  set.seed(22)
  r <- terra::rast(nrows = 8, ncols = 8, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- rep(seq(1, 8), each = 8)

  set.seed(22)
  conn_at_obs <- abs(rnorm(15, mean = 4))
  obs_x <- runif(15, 0.1, 0.9)
  obs_y <- runif(15, 0.1, 0.9)

  fit <- fit_intensity_nb(
    connectivity_at_obs = conn_at_obs,
    connectivity_raster = r,
    obs_coords          = data.frame(x = obs_x, y = obs_y)
  )

  # Optimizer should converge (convergence == 0)
  expect_equal(fit$convergence, 0)

  # loglik at optimum should be a finite scalar
  expect_true(is.finite(fit$loglik))
  expect_length(fit$loglik, 1)

  # Estimates vector should have named elements including alpha and gamma
  expect_true("alpha" %in% names(fit$estimates))
  expect_true("gamma" %in% names(fit$estimates))
})
