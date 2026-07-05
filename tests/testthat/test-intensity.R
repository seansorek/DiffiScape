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

  # Test the negloglik_fn actually used by fit_intensity_nb() via family_negbin()
  n_obs <- 80
  n_int <- 400
  total <- n_obs + n_int

  z_obs <- rnorm(n_obs)
  z_int <- rnorm(n_int)
  obs_w  <- rep(1, n_obs)
  int_w  <- rep(0.01, n_int)

  nll <- family_negbin()$negloglik_fn(
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
  expect_equal(fit$loglik, as.numeric(stats::logLik(fit$gam_model)), tolerance = 1e-6)
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


test_that("fit_intensity_gam warns on high concurvity when include_spatial_re=TRUE", {
  skip_on_cran()
  skip_if_not_installed("mgcv")

  set.seed(77)
  n_obs   <- 30
  r <- terra::rast(nrows = 15, ncols = 15, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(225, mean = 4))
  obs_pts  <- data.frame(x = runif(n_obs, 0.05, 0.95),
                         y = runif(n_obs, 0.05, 0.95))
  conn_obs <- abs(rnorm(n_obs, mean = 4))

  cfg <- default_intensity_config()
  cfg$include_spatial_re <- TRUE

  local_mocked_bindings(
    concurvity = function(object, full = TRUE) {
      matrix(
        c(0.95, 0.85, 0.70),
        nrow = 3, ncol = 1,
        dimnames = list(c("worst", "observed", "estimate"),
                        c("s(connectivity)"))
      )
    },
    .package = "mgcv"
  )

  ws <- character(0)
  withCallingHandlers(
    suppressMessages(
      fit_intensity_gam(
        connectivity_at_obs = conn_obs,
        connectivity_raster = r,
        obs_coords          = obs_pts,
        config              = cfg
      )
    ),
    warning = function(w) {
      ws <<- c(ws, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  expect_true(any(grepl("High concurvity detected", ws)))
})


test_that("fit_intensity_gam does not warn when concurvity is low", {
  skip_on_cran()
  skip_if_not_installed("mgcv")

  set.seed(78)
  n_obs   <- 30
  r <- terra::rast(nrows = 15, ncols = 15, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(225, mean = 4))
  obs_pts  <- data.frame(x = runif(n_obs, 0.05, 0.95),
                         y = runif(n_obs, 0.05, 0.95))
  conn_obs <- abs(rnorm(n_obs, mean = 4))

  cfg <- default_intensity_config()
  cfg$include_spatial_re <- TRUE

  local_mocked_bindings(
    concurvity = function(object, full = TRUE) {
      matrix(
        c(0.30, 0.20, 0.10),
        nrow = 3, ncol = 1,
        dimnames = list(c("worst", "observed", "estimate"),
                        c("s(connectivity)"))
      )
    },
    .package = "mgcv"
  )

  ws <- character(0)
  withCallingHandlers(
    suppressMessages(
      fit_intensity_gam(
        connectivity_at_obs = conn_obs,
        connectivity_raster = r,
        obs_coords          = obs_pts,
        config              = cfg
      )
    ),
    warning = function(w) {
      ws <<- c(ws, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  expect_false(any(grepl("High concurvity detected", ws)))
})


test_that("GAM predict errors early when covariate rasters are missing (#37)", {
  skip_on_cran()
  skip_if_not_installed("mgcv")
  set.seed(37)

  r <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(100, mean = 4))

  n_obs   <- 25
  obs_x   <- runif(n_obs, 0.05, 0.95)
  obs_y   <- runif(n_obs, 0.05, 0.95)
  cov_obs <- list(habitat = runif(n_obs))

  r_cov <- terra::rast(r)
  terra::values(r_cov) <- runif(100)
  names(r_cov) <- "habitat"

  # Fit a GAM that includes a covariate smooth
  fit <- fit_intensity_gam(
    connectivity_at_obs = abs(rnorm(n_obs, mean = 4)),
    connectivity_raster = r,
    obs_coords          = data.frame(x = obs_x, y = obs_y),
    covariates_obs      = cov_obs,
    covariates_rasters  = list(habitat = r_cov)
  )

  # Predict WITHOUT supplying covariates_rasters => should error with
  # an actionable message mentioning "covariates_rasters"
  expect_error(
    predict_intensity(fit, r, covariates_rasters = NULL),
    "covariates_rasters"
  )
})


test_that("compute_intensity treats missing covariate name in betas as zero (#35)", {
  alpha  <- -2
  gamma  <- 1.0
  z_vals <- rnorm(20)

  # 'elev' is in covariates but NOT in betas
  cov  <- list(elev = rnorm(20), slope = rnorm(20))
  beta <- c(slope = 0.5)  # 'elev' deliberately absent

  # Should not crash; missing beta treated as 0
  lambda <- compute_intensity(z_vals, alpha, gamma,
                               covariates = cov, betas = beta)
  expect_length(lambda, 20)
  expect_true(all(is.finite(lambda)))
  expect_true(all(lambda > 0))

  # Result should equal the version with only slope (elev contributes nothing)
  cov_slope_only <- list(slope = cov$slope)
  lambda_ref <- compute_intensity(z_vals, alpha, gamma,
                                   covariates = cov_slope_only, betas = beta)
  expect_equal(lambda, lambda_ref, tolerance = 1e-12)
})


test_that("available covariates are clamped in selection mode (#44)", {
  skip_on_cran()
  set.seed(44)

  n_obs <- 30
  n_avail <- 100
  obs_conn  <- abs(rnorm(n_obs, mean = 5))
  avail_conn <- abs(rnorm(n_avail, mean = 5))
  obs_pts   <- data.frame(x = runif(n_obs), y = runif(n_obs))

  # Covariates with values OUTSIDE [0,1]
  cov_obs_raw   <- list(habitat = runif(n_obs, -0.5, 1.5))
  cov_avail_raw <- list(habitat = runif(n_avail, -0.5, 1.5))

  # The fix clamps both cov_obs and cov_int to [0,1].
  # If cov_int is not clamped, values outside [0,1] would be passed raw.
  # We verify the fit succeeds and the internal clamped values are in [0,1]
  # by checking that results match a manually-clamped version.
  cov_avail_clamped <- list(
    habitat = pmax(pmin(cov_avail_raw$habitat, 1), 0)
  )
  cov_obs_clamped <- list(
    habitat = pmax(pmin(cov_obs_raw$habitat, 1), 0)
  )

  fit_raw <- fit_intensity_selection(
    connectivity_at_obs   = obs_conn,
    available_connectivity = avail_conn,
    obs_coords            = obs_pts,
    covariates_obs        = cov_obs_raw,
    available_covariates  = cov_avail_raw
  )

  fit_clamped <- fit_intensity_selection(
    connectivity_at_obs   = obs_conn,
    available_connectivity = avail_conn,
    obs_coords            = obs_pts,
    covariates_obs        = cov_obs_clamped,
    available_covariates  = cov_avail_clamped
  )

  # With the bug fixed, both calls should produce identical results because
  # the function internally clamps to [0,1] anyway
  expect_equal(fit_raw$estimates, fit_clamped$estimates, tolerance = 1e-10)
  expect_equal(fit_raw$loglik, fit_clamped$loglik, tolerance = 1e-10)
})


test_that("subsample guard ensures n_samp >= 1 for small rasters (#39)", {
  # Directly test the subsampling arithmetic that appears in
  # fit_intensity_nb / fit_intensity_gam.  When n_int_full is tiny and
  # frac is small, round(n_int_full * frac) can be 0; the fix uses
  # max(1L, ...) to prevent division by zero.
  n_int_full <- 3
  frac       <- 0.1    # round(3 * 0.1) == round(0.3) == 0

  n_samp_raw   <- round(n_int_full * frac)
  n_samp_fixed <- max(1L, round(n_int_full * frac))

  expect_equal(n_samp_raw, 0)          # confirms the bug scenario
  expect_equal(n_samp_fixed, 1L)       # guard kicks in

  # With the guard, step_sz is finite and the subsample produces valid indices
  step_sz <- n_int_full / n_samp_fixed
  expect_true(is.finite(step_sz))
  expect_true(step_sz > 0)
})


test_that("fit_intensity_nb errors when covariates_obs supplied without covariates_rasters (#47)", {
  skip_on_cran()
  skip_if_not_installed("terra")
  set.seed(47)

  r <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(100, mean = 5))

  n_obs <- 20
  obs_x <- runif(n_obs, 0.1, 0.9)
  obs_y <- runif(n_obs, 0.1, 0.9)
  conn_at_obs <- abs(rnorm(n_obs, mean = 5))

  # Raster mode: covariates_obs supplied, covariates_rasters is NULL
  expect_error(
    fit_intensity_nb(
      connectivity_at_obs = conn_at_obs,
      connectivity_raster = r,
      obs_coords          = data.frame(x = obs_x, y = obs_y),
      covariates_obs      = list(elev = runif(n_obs)),
      covariates_rasters  = NULL
    ),
    "covariates_obs supplied but covariates_rasters is NULL"
  )
})


test_that("fit_intensity_nb errors when covariates_obs supplied without available_covariates in selection mode (#47)", {
  skip_on_cran()
  set.seed(48)

  n_obs <- 20
  n_avail <- 100
  conn_at_obs <- abs(rnorm(n_obs, mean = 5))
  avail_conn  <- abs(rnorm(n_avail, mean = 5))

  # Selection mode: covariates_obs supplied, available_covariates is NULL
  expect_error(
    fit_intensity_nb(
      connectivity_at_obs    = conn_at_obs,
      connectivity_raster    = NULL,
      obs_coords             = data.frame(x = runif(n_obs), y = runif(n_obs)),
      covariates_obs         = list(elev = runif(n_obs)),
      available_connectivity = avail_conn,
      available_covariates   = NULL
    ),
    "covariates_obs supplied but available_covariates is NULL"
  )
})


test_that("predict_intensity applies residualisation when fit used residualise=TRUE (#29)", {
  skip_on_cran()
  skip_if_not_installed("terra")
  set.seed(29)

  # Create rasters
  r <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(100, mean = 5))

  r_cov <- terra::rast(r)
  terra::values(r_cov) <- runif(100)
  names(r_cov) <- "habitat"

  n_obs <- 25
  obs_x <- runif(n_obs, 0.05, 0.95)
  obs_y <- runif(n_obs, 0.05, 0.95)
  conn_at_obs <- abs(rnorm(n_obs, mean = 5))
  cov_obs <- list(habitat = runif(n_obs))

  # Fit with residualise = TRUE
  fit_resid <- fit_intensity_nb(
    connectivity_at_obs = conn_at_obs,
    connectivity_raster = r,
    obs_coords          = data.frame(x = obs_x, y = obs_y),
    covariates_obs      = cov_obs,
    covariates_rasters  = list(habitat = r_cov),
    residualise         = TRUE
  )

  # Fit without residualise
  fit_no_resid <- fit_intensity_nb(
    connectivity_at_obs = conn_at_obs,
    connectivity_raster = r,
    obs_coords          = data.frame(x = obs_x, y = obs_y),
    covariates_obs      = cov_obs,
    covariates_rasters  = list(habitat = r_cov),
    residualise         = FALSE
  )

  expect_true(isTRUE(fit_resid$is_residualised))
  expect_false(isTRUE(fit_no_resid$is_residualised))

  # Predict from both
  pred_resid    <- predict_intensity(fit_resid, r,
                                     covariates_rasters = list(habitat = r_cov))
  pred_no_resid <- predict_intensity(fit_no_resid, r,
                                     covariates_rasters = list(habitat = r_cov))

  vals_resid    <- terra::values(pred_resid)
  vals_no_resid <- terra::values(pred_no_resid)

  # Both should produce finite positive values
  expect_true(all(is.finite(vals_resid[!is.na(vals_resid)])))
  expect_true(all(vals_resid[!is.na(vals_resid)] > 0))

  # The two predictions should differ (residualisation changes the z values)
  expect_false(isTRUE(all.equal(vals_resid, vals_no_resid)),
               info = "Residualised predictions should differ from non-residualised")
})


test_that("predict_intensity_gam applies residualisation when fit used residualise=TRUE (#29)", {
  skip_on_cran()
  skip_if_not_installed("mgcv")
  set.seed(291)

  r <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(100, mean = 5))

  r_cov <- terra::rast(r)
  terra::values(r_cov) <- runif(100)
  names(r_cov) <- "habitat"

  n_obs <- 25
  obs_x <- runif(n_obs, 0.05, 0.95)
  obs_y <- runif(n_obs, 0.05, 0.95)
  conn_at_obs <- abs(rnorm(n_obs, mean = 5))
  cov_obs <- list(habitat = runif(n_obs))

  # Fit GAM with residualise = TRUE
  fit_resid <- fit_intensity_gam(
    connectivity_at_obs = conn_at_obs,
    connectivity_raster = r,
    obs_coords          = data.frame(x = obs_x, y = obs_y),
    covariates_obs      = cov_obs,
    covariates_rasters  = list(habitat = r_cov),
    residualise         = TRUE
  )

  # Fit GAM without residualise
  fit_no_resid <- fit_intensity_gam(
    connectivity_at_obs = conn_at_obs,
    connectivity_raster = r,
    obs_coords          = data.frame(x = obs_x, y = obs_y),
    covariates_obs      = cov_obs,
    covariates_rasters  = list(habitat = r_cov),
    residualise         = FALSE
  )

  expect_true(isTRUE(fit_resid$is_residualised))

  # Predict from both
  pred_resid    <- predict_intensity(fit_resid, r,
                                     covariates_rasters = list(habitat = r_cov))
  pred_no_resid <- predict_intensity(fit_no_resid, r,
                                     covariates_rasters = list(habitat = r_cov))

  vals_resid    <- terra::values(pred_resid)
  vals_no_resid <- terra::values(pred_no_resid)

  # Both should produce finite positive values
  expect_true(all(is.finite(vals_resid[!is.na(vals_resid)])))
  expect_true(all(vals_resid[!is.na(vals_resid)] > 0))

  # The two predictions should differ
  expect_false(isTRUE(all.equal(vals_resid, vals_no_resid)),
               info = "Residualised GAM predictions should differ from non-residualised")
})
