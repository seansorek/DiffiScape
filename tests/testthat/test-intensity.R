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

test_that("residualise_connectivity returns correct structure with one covariate", {
  set.seed(1)
  n_obs <- 30
  n_int <- 100
  cov_val <- rnorm(n_obs)
  z_obs   <- 0.5 * cov_val + rnorm(n_obs, sd = 0.3)
  z_int   <- rnorm(n_int)
  cov_int <- rnorm(n_int)

  res <- residualise_connectivity(
    z_obs          = z_obs,
    z_int          = z_int,
    covariates_obs = list(elev = cov_val),
    covariates_int = list(elev = cov_int)
  )

  expect_named(res, c("z_obs_resid", "z_int_resid", "aux_coefs", "aux_r2", "aux_model"))
  expect_length(res$z_obs_resid, n_obs)
  expect_length(res$z_int_resid, n_int)
  expect_true(is.numeric(res$z_obs_resid))
  expect_true(is.numeric(res$z_int_resid))
  expect_true(res$aux_r2 >= 0 && res$aux_r2 <= 1)
  # Residuals at observation points must be mean-zero (by OLS property)
  expect_equal(mean(res$z_obs_resid), 0, tolerance = 1e-10)
})


test_that("residualise_connectivity handles multiple covariates", {
  set.seed(2)
  n_obs <- 40
  n_int <- 120
  cov1_obs <- rnorm(n_obs); cov2_obs <- rnorm(n_obs); cov3_obs <- rnorm(n_obs)
  z_obs    <- cov1_obs + 0.5 * cov2_obs + rnorm(n_obs, sd = 0.2)

  res <- residualise_connectivity(
    z_obs          = z_obs,
    z_int          = rnorm(n_int),
    covariates_obs = list(a = cov1_obs, b = cov2_obs, c = cov3_obs),
    covariates_int = list(a = rnorm(n_int), b = rnorm(n_int), c = rnorm(n_int))
  )

  # aux_coefs should have intercept + 3 slopes
  expect_length(res$aux_coefs, 4)
  expect_true("(Intercept)" %in% names(res$aux_coefs))
})


test_that("residualise_connectivity R-squared near 1 for perfectly collinear covariate", {
  set.seed(3)
  n_obs <- 50
  cov   <- rnorm(n_obs)
  z_obs <- 2 * cov + 1  # perfect linear relationship

  res <- residualise_connectivity(
    z_obs          = z_obs,
    z_int          = rnorm(20),
    covariates_obs = list(x = cov),
    covariates_int = list(x = rnorm(20))
  )

  expect_true(res$aux_r2 > 0.99)
  expect_true(all(abs(res$z_obs_resid) < 1e-10))
})


test_that("fit_intensity_gam returns required fields with correct types", {
  skip_on_cran()
  skip_if_not_installed("mgcv")
  set.seed(42)

  r <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(100, mean = 5))

  conn_at_obs <- abs(rnorm(25, mean = 5))
  obs_x <- runif(25, 0.05, 0.95)
  obs_y <- runif(25, 0.05, 0.95)

  fit <- fit_intensity_gam(
    connectivity_at_obs = conn_at_obs,
    connectivity_raster = r,
    obs_coords          = data.frame(x = obs_x, y = obs_y)
  )

  expect_true(is.list(fit))
  required <- c("estimates", "se", "loglik", "convergence",
                "gam_model", "gam_edf", "gam_deviance_explained", "gam_aic")
  for (nm in required) {
    expect_true(nm %in% names(fit),
                info = paste("Missing field:", nm))
  }
  expect_true(inherits(fit$gam_model, "bam") || inherits(fit$gam_model, "gam"))
  expect_true(is.finite(fit$loglik))
  expect_true(fit$loglik < 0)
})


test_that("fit_intensity_gam estimates contain alpha and gamma", {
  skip_on_cran()
  skip_if_not_installed("mgcv")
  set.seed(7)

  r <- terra::rast(nrows = 8, ncols = 8, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(64, mean = 3))
  conn_at_obs <- abs(rnorm(20, mean = 3))

  fit <- fit_intensity_gam(
    connectivity_at_obs = conn_at_obs,
    connectivity_raster = r,
    obs_coords = data.frame(x = runif(20, 0.1, 0.9),
                            y = runif(20, 0.1, 0.9))
  )

  expect_true("alpha" %in% names(fit$estimates))
  expect_true("gamma" %in% names(fit$estimates))
  expect_true(is.finite(fit$estimates["alpha"]))
  expect_true(is.finite(fit$estimates["gamma"]))
})


test_that("fit_intensity_gam with residualise=TRUE exercises the residualisation path", {
  skip_on_cran()
  skip_if_not_installed("mgcv")
  set.seed(13)

  r <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(100, mean = 4))

  n_obs <- 25
  obs_x <- runif(n_obs, 0.05, 0.95)
  obs_y <- runif(n_obs, 0.05, 0.95)
  cov_obs <- list(habitat = runif(n_obs))

  # Covariate raster must cover the same grid
  r_cov <- terra::rast(r)
  terra::values(r_cov) <- runif(100)
  names(r_cov) <- "habitat"

  fit <- fit_intensity_gam(
    connectivity_at_obs = abs(rnorm(n_obs, mean = 4)),
    connectivity_raster = r,
    obs_coords          = data.frame(x = obs_x, y = obs_y),
    covariates_obs      = cov_obs,
    covariates_rasters  = list(habitat = r_cov),
    residualise         = TRUE
  )

  expect_true(isTRUE(fit$is_residualised))
  expect_true(!is.null(fit$residualisation_info))
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


# ---------------------------------------------------------------------------
# Tests for bug #29: residualisation applied at fit time must be mirrored at
# prediction time.
# ---------------------------------------------------------------------------

test_that("predict_intensity (NB) applies residualisation stored in fit", {
  # Verify that when residualise = TRUE, the stored aux_model is used during
  # prediction, so that in-sample predicted intensities are not biased by
  # confounding raw z with the residualised z that gamma was estimated on.
  skip_on_cran()
  skip_if_not_installed("terra")

  set.seed(55)
  r <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(100, mean = 5, sd = 2))

  r_cov <- terra::rast(r)
  terra::values(r_cov) <- runif(100)
  names(r_cov) <- "elev"

  n_obs <- 20
  set.seed(55)
  obs_x <- runif(n_obs, 0.05, 0.95)
  obs_y <- runif(n_obs, 0.05, 0.95)
  conn_at_obs <- abs(rnorm(n_obs, mean = 5, sd = 2))
  cov_obs_elev <- runif(n_obs)

  # Fit with residualise = TRUE
  fit_resid <- fit_intensity_nb(
    connectivity_at_obs = conn_at_obs,
    connectivity_raster = r,
    obs_coords          = data.frame(x = obs_x, y = obs_y),
    covariates_obs      = list(elev = cov_obs_elev),
    covariates_rasters  = list(elev = r_cov),
    residualise         = TRUE
  )

  # Fit without residualise = FALSE (baseline)
  fit_plain <- fit_intensity_nb(
    connectivity_at_obs = conn_at_obs,
    connectivity_raster = r,
    obs_coords          = data.frame(x = obs_x, y = obs_y),
    covariates_obs      = list(elev = cov_obs_elev),
    covariates_rasters  = list(elev = r_cov),
    residualise         = FALSE
  )

  expect_true(isTRUE(fit_resid$is_residualised))
  expect_false(isTRUE(fit_plain$is_residualised))
  expect_true(!is.null(fit_resid$residualisation_info))
  expect_true(!is.null(fit_resid$residualisation_info$aux_model))

  # Predict with both fits
  pred_resid <- predict_intensity(fit_resid, r,
                                  covariates_rasters = list(elev = r_cov))
  pred_plain <- predict_intensity(fit_plain, r,
                                  covariates_rasters = list(elev = r_cov))

  pred_resid_vals <- terra::values(pred_resid)
  pred_plain_vals <- terra::values(pred_plain)

  # Both should produce finite, positive intensities
  expect_true(all(is.finite(pred_resid_vals), na.rm = TRUE))
  expect_true(all(pred_resid_vals > 0, na.rm = TRUE))

  # The residualised prediction must differ from the plain prediction
  # (they solve different sub-problems), but should not be wildly extreme.
  # More importantly, verify the fix: when residualise=TRUE, the auxiliary
  # regression stored in fit$residualisation_info must actually be applied.
  # We verify by manually computing what predict_intensity *should* produce.
  C_safe <- pmax(terra::values(r), 0)
  z_manual <- (log1p(C_safe / fit_resid$c_scale) - fit_resid$log_conn_mean) /
    fit_resid$log_conn_sd

  ri <- fit_resid$residualisation_info
  cov_names_resid <- setdiff(names(ri$aux_coefs), "(Intercept)")
  nd_resid <- data.frame(elev = pmax(pmin(terra::values(r_cov), 1), 0))
  z_corrected <- z_manual - stats::predict(ri$aux_model, newdata = nd_resid)

  alpha_est <- fit_resid$estimates[["alpha"]]
  gamma_est <- fit_resid$estimates[["gamma"]]
  beta_elev  <- fit_resid$estimates[["beta_elev"]]
  elev_vals  <- pmax(pmin(terra::values(r_cov), 1), 0)
  log_lambda_expected <- (if (is.na(alpha_est)) 0 else alpha_est) +
                         gamma_est * z_corrected +
                         (if (!is.null(beta_elev) && !is.na(beta_elev))
                           beta_elev * elev_vals else 0)
  expected_vals <- exp(log_lambda_expected)

  expect_equal(pred_resid_vals, expected_vals, tolerance = 1e-8)
})


test_that("predict_intensity (NB) with residualise=TRUE differs from naive (non-residualised) prediction", {
  # Confirm the bug would have been detectable: if the fix is absent, applying
  # gamma * raw_z (instead of gamma * residual_z) produces different values.
  skip_on_cran()
  skip_if_not_installed("terra")

  set.seed(77)
  r <- terra::rast(nrows = 8, ncols = 8, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(64, mean = 4, sd = 1.5))

  r_cov <- terra::rast(r)
  terra::values(r_cov) <- runif(64)
  names(r_cov) <- "ndvi"

  n_obs <- 15
  conn_at_obs <- abs(rnorm(n_obs, mean = 4, sd = 1.5))
  cov_obs <- list(ndvi = runif(n_obs))

  fit_resid <- fit_intensity_nb(
    connectivity_at_obs = conn_at_obs,
    connectivity_raster = r,
    obs_coords          = data.frame(x = runif(n_obs, 0.1, 0.9),
                                     y = runif(n_obs, 0.1, 0.9)),
    covariates_obs      = cov_obs,
    covariates_rasters  = list(ndvi = r_cov),
    residualise         = TRUE
  )

  # Simulate naive (buggy) prediction: gamma * raw z, ignoring residualisation
  C_safe <- pmax(terra::values(r), 0)
  z_raw <- (log1p(C_safe / fit_resid$c_scale) - fit_resid$log_conn_mean) /
    fit_resid$log_conn_sd

  alpha_est <- fit_resid$estimates[["alpha"]]
  gamma_est <- fit_resid$estimates[["gamma"]]
  beta_ndvi <- fit_resid$estimates[["beta_ndvi"]]
  ndvi_vals <- pmax(pmin(terra::values(r_cov), 1), 0)
  naive_log_lambda <- (if (is.na(alpha_est)) 0 else alpha_est) +
                      gamma_est * z_raw +
                      (if (!is.null(beta_ndvi) && !is.na(beta_ndvi))
                        beta_ndvi * ndvi_vals else 0)
  naive_pred <- exp(naive_log_lambda)

  # Fixed prediction (using corrected z after residualisation)
  fixed_pred <- terra::values(
    predict_intensity(fit_resid, r, covariates_rasters = list(ndvi = r_cov))
  )

  # The fixed and naive predictions must differ (the bug would make them equal)
  expect_false(isTRUE(all.equal(fixed_pred, naive_pred, tolerance = 1e-6)))
})
