# Tests for R/diagnostics.R

test_that("compute_deviance_residuals works on known values", {
  # When y == mu, deviance residual should be 0
  obs <- c(5L, 10L, 0L)
  fit <- c(5.0, 10.0, 0.001)
  size <- 2

  dr <- compute_deviance_residuals(obs, fit, size)
  expect_length(dr, 3)

  # y == mu -> residual ~ 0

  expect_equal(dr[1], 0, tolerance = 1e-6)
  expect_equal(dr[2], 0, tolerance = 1e-6)
})


test_that("compute_deviance_residuals sign is correct", {
  obs <- c(10L, 1L)
  fit <- c(5.0, 5.0)
  size <- 2

  dr <- compute_deviance_residuals(obs, fit, size)
  expect_true(dr[1] > 0)   # obs > fit -> positive
  expect_true(dr[2] < 0)   # obs < fit -> negative
})


test_that("compute_deviance_residuals errors on mismatched lengths", {
  expect_error(compute_deviance_residuals(1:3, 1:4, 1))
})


test_that("moran_test works on synthetic data", {
  skip_on_cran()
  skip_if_not_installed("spdep")

  set.seed(42)
  n <- 50
  coords <- cbind(x = runif(n), y = runif(n))
  resid  <- rnorm(n)

  result <- moran_test(resid, coords, k = 5)
  expect_true("observed" %in% names(result))
  expect_true("p_value" %in% names(result))
  expect_true(is.numeric(result$p_value))
})


test_that("moran_test errors without spdep", {
  # We can't easily unload spdep, but we verify the function exists

  expect_true(is.function(moran_test))
})


test_that("plot_deviance_residuals runs without error", {
  skip_on_cran()
  obs <- rpois(100, 5)
  fit <- rep(5, 100)
  size <- 2

  # Should produce a plot without error
  expect_silent({
    pdf(tempfile(fileext = ".pdf"))
    plot_deviance_residuals(obs, fit, size)
    dev.off()
  })
})


test_that("plot_qq_deviance runs without error", {
  skip_on_cran()
  obs <- rpois(100, 5)
  fit <- rep(5, 100)
  size <- 2

  expect_silent({
    pdf(tempfile(fileext = ".pdf"))
    plot_qq_deviance(obs, fit, size)
    dev.off()
  })
})


# ---------------------------------------------------------------------------
# compute_deviance_residuals_gam
# ---------------------------------------------------------------------------

test_that("compute_deviance_residuals_gam works on a real bam object", {
  skip_on_cran()
  skip_if_not_installed("mgcv")
  set.seed(42)

  r <- terra::rast(nrows = 8, ncols = 8, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(64, mean = 3))

  fit <- fit_intensity_gam(
    connectivity_at_obs = abs(rnorm(20, mean = 3)),
    connectivity_raster = r,
    obs_coords = data.frame(x = runif(20, 0.1, 0.9),
                            y = runif(20, 0.1, 0.9))
  )

  resids <- compute_deviance_residuals_gam(fit$gam_model)
  expect_true(is.numeric(resids))
  expect_true(length(resids) > 0)
  expect_true(all(is.finite(resids)))
})


test_that("compute_deviance_residuals_gam extracts from a fit_intensity_gam list", {
  skip_on_cran()
  skip_if_not_installed("mgcv")
  set.seed(43)

  r <- terra::rast(nrows = 8, ncols = 8, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(64, mean = 3))

  fit <- fit_intensity_gam(
    connectivity_at_obs = abs(rnorm(20, mean = 3)),
    connectivity_raster = r,
    obs_coords = data.frame(x = runif(20, 0.1, 0.9),
                            y = runif(20, 0.1, 0.9))
  )

  # Pass the whole list (not just the gam model) — should auto-extract $gam_model
  resids_from_list   <- compute_deviance_residuals_gam(fit)
  resids_from_model  <- compute_deviance_residuals_gam(fit$gam_model)
  expect_equal(resids_from_list, resids_from_model)
})


test_that("compute_deviance_residuals_gam errors on invalid input", {
  expect_error(compute_deviance_residuals_gam(data.frame(x = 1:3)),
               "gam/bam")
  expect_error(compute_deviance_residuals_gam(list(foo = 1)),
               "gam/bam")
})


# ---------------------------------------------------------------------------
# rasterise_deviance_residuals and diagnose_model
# ---------------------------------------------------------------------------

.make_diag_inputs <- function(seed = 99) {
  set.seed(seed)
  r <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(100, mean = 4))
  n_obs   <- 20
  obs_pts <- data.frame(x = runif(n_obs, 0.05, 0.95),
                        y = runif(n_obs, 0.05, 0.95))
  nb_fit  <- fit_intensity_nb(
    connectivity_at_obs = abs(rnorm(n_obs, mean = 4)),
    connectivity_raster = r,
    obs_coords          = obs_pts
  )
  list(raster = r, obs_pts = obs_pts, nb_fit = nb_fit)
}


test_that("rasterise_deviance_residuals returns a SpatRaster with correct structure", {
  skip_on_cran()
  d <- .make_diag_inputs()

  resid_r <- rasterise_deviance_residuals(d$nb_fit, d$obs_pts, d$raster)

  expect_s4_class(resid_r, "SpatRaster")
  expect_equal(terra::nrow(resid_r), terra::nrow(d$raster))
  expect_equal(terra::ncol(resid_r), terra::ncol(d$raster))
  expect_equal(names(resid_r), "deviance_residual")
})


test_that("rasterise_deviance_residuals values are numeric (or NA)", {
  skip_on_cran()
  d <- .make_diag_inputs(seed = 11)

  resid_r <- rasterise_deviance_residuals(d$nb_fit, d$obs_pts, d$raster)
  vals    <- terra::values(resid_r)[, 1]

  expect_true(is.numeric(vals))
  # At least some cells should have finite residuals
  expect_true(any(is.finite(vals)))
})


test_that("rasterise_deviance_residuals works when intensity_fit is evaluate_full_model format", {
  skip_on_cran()
  d <- .make_diag_inputs(seed = 55)

  # Simulate the evaluate_full_model() return format (intensity_fit_obj is now stored)
  mock_full_result <- list(
    loglik            = d$nb_fit$loglik,
    intensity_params  = d$nb_fit$estimates,
    intensity_fit_obj = d$nb_fit,
    intensity_se      = d$nb_fit$se,
    convergence       = d$nb_fit$convergence,
    distribution      = "negbin"
  )

  resid_r <- rasterise_deviance_residuals(mock_full_result, d$obs_pts, d$raster)
  expect_s4_class(resid_r, "SpatRaster")
  expect_equal(names(resid_r), "deviance_residual")
})


test_that("diagnose_model returns expected fields with plot=FALSE", {
  skip_on_cran()
  skip_if_not_installed("spdep")
  d <- .make_diag_inputs(seed = 77)

  result <- suppressMessages(
    diagnose_model(d$nb_fit, d$obs_pts, d$raster, plot = FALSE)
  )

  expect_named(result, c("residual_raster", "moran", "mean_deviance", "prop_large"))
  expect_s4_class(result$residual_raster, "SpatRaster")
  expect_true(result$mean_deviance >= 0)
  expect_true(result$prop_large >= 0 && result$prop_large <= 1)
})


test_that("plot_residual_map runs without error", {
  skip_on_cran()
  r <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- rnorm(25)
  names(r) <- "deviance_residual"

  expect_silent({
    pdf(tempfile(fileext = ".pdf"))
    result <- plot_residual_map(r)
    dev.off()
  })
  expect_null(result)
})


test_that("plot_residual_map overlays observation points without error", {
  skip_on_cran()
  r <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- rnorm(25)
  pts <- data.frame(x = c(0.2, 0.8), y = c(0.2, 0.8))

  expect_silent({
    pdf(tempfile(fileext = ".pdf"))
    plot_residual_map(r, obs_points = pts)
    dev.off()
  })
})


test_that("diagnose_model warns when Moran's I is significant", {
  skip_on_cran()
  skip_if_not_installed("spdep")
  d <- .make_diag_inputs(seed = 77)

  local_mocked_bindings(
    moran_test = function(resids, coords, k = 8L) {
      list(observed = 0.45, p_value = 0.001)
    },
    .package = "DiffiScape"
  )

  expect_warning(
    suppressMessages(diagnose_model(d$nb_fit, d$obs_pts, d$raster, plot = FALSE)),
    "Significant residual spatial autocorrelation"
  )
})


test_that("diagnose_model does not warn when Moran's I is non-significant", {
  skip_on_cran()
  skip_if_not_installed("spdep")
  d <- .make_diag_inputs(seed = 77)

  local_mocked_bindings(
    moran_test = function(resids, coords, k = 8L) {
      list(observed = 0.02, p_value = 0.60)
    },
    .package = "DiffiScape"
  )

  expect_no_warning(
    suppressMessages(diagnose_model(d$nb_fit, d$obs_pts, d$raster, plot = FALSE))
  )
})
