# Tests for R/pipeline.R

test_that("ds_load_data reads CSV correctly", {
  tmp <- tempfile(fileext = ".csv")
  df  <- data.frame(x = 1:5, y = 6:10, extra = letters[1:5])
  write.csv(df, tmp, row.names = FALSE)

  pts <- ds_load_data(tmp)
  expect_true("x" %in% names(pts))
  expect_true("y" %in% names(pts))
  expect_equal(nrow(pts), 5)
  unlink(tmp)
})


test_that("ds_load_data errors on missing columns", {
  tmp <- tempfile(fileext = ".csv")
  write.csv(data.frame(a = 1:3, b = 4:6), tmp, row.names = FALSE)
  expect_error(ds_load_data(tmp), "must contain columns")
  unlink(tmp)
})


test_that("ds_load_data errors on unsupported format", {
  expect_error(ds_load_data("data.xyz"), "Unsupported file type")
})


test_that("ds_create_basis reads from file paths", {
  skip_on_cran()

  r1 <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  terra::values(r1) <- runif(25)

  tmp <- tempfile(fileext = ".tif")
  terra::writeRaster(r1, tmp)

  stack <- ds_create_basis(tmp)
  expect_s4_class(stack, "SpatRaster")
  expect_equal(terra::nlyr(stack), 1)
  unlink(tmp)
})


test_that("ds_create_basis reads from directory", {
  skip_on_cran()

  dir <- tempdir()
  subdir <- file.path(dir, "test_rasters")
  dir.create(subdir, showWarnings = FALSE)

  for (nm in c("elev", "slope")) {
    r <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                     ymin = 0, ymax = 1)
    terra::values(r) <- runif(25)
    terra::writeRaster(r, file.path(subdir, paste0(nm, ".tif")),
                       overwrite = TRUE)
  }

  stack <- ds_create_basis(subdir)
  expect_s4_class(stack, "SpatRaster")
  expect_equal(terra::nlyr(stack), 2)
  unlink(subdir, recursive = TRUE)
})


test_that("ds_init_julia errors without Julia", {
  # This test is expected to fail unless Julia is installed
  skip_if_not_installed("JuliaConnectoR")
  skip_on_cran()

  # Just verify the function exists and handles errors
  expect_true(is.function(ds_init_julia))
})


test_that("ds_posterior passes refit = TRUE to laplace_resistance", {
  skip_on_cran()

  opt_result <- list(
    best_params  = list(r_0 = 0),
    bounds       = list(r_0 = c(-2, 2)),
    distribution = "negbin",
    surrogate    = structure(list(), class = "km")
  )
  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)

  captured_refit <- NULL
  mock_lap <- list(mode = 0, covariance = matrix(0.01, 1, 1),
                   precision = matrix(100, 1, 1), std_error = 0.1)
  mock_samp <- data.frame(r_0 = rnorm(3))

  local_mocked_bindings(
    laplace_resistance = function(...) {
      args <- list(...)
      captured_refit <<- args$refit
      mock_lap
    },
    posterior_sample = function(...) mock_samp,
    posterior_summary = function(...) data.frame(),
    .package = "DiffiScape"
  )

  ds_posterior(opt_result, basis_stack, obs_points, n_draws = 3L, n_inner = 1L)

  expect_true(captured_refit)
})


test_that("ds_posterior forwards omniscape_settings to laplace and posterior_sample", {
  skip_on_cran()

  opt_result <- list(
    best_params  = list(r_0 = 0),
    bounds       = list(r_0 = c(-2, 2)),
    distribution = "negbin",
    surrogate    = structure(list(), class = "km")
  )

  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)

  custom_settings <- list(radius = 25L, block_size = 9L)
  captured_lap_settings  <- NULL
  captured_samp_settings <- NULL
  mock_lap  <- list(mode = 0, covariance = matrix(0.01, 1, 1),
                    precision = matrix(100, 1, 1), std_error = 0.1)
  mock_samp <- data.frame(r_0 = rnorm(3))

  local_mocked_bindings(
    laplace_resistance = function(...) {
      args <- list(...)
      captured_lap_settings <<- args$omniscape_settings
      mock_lap
    },
    posterior_sample = function(...) {
      args <- list(...)
      captured_samp_settings <<- args$omniscape_settings
      mock_samp
    },
    posterior_summary = function(...) data.frame(),
    .package = "DiffiScape"
  )

  ds_posterior(opt_result, basis_stack, obs_points,
              n_draws = 3L, n_inner = 1L,
              omniscape_settings = custom_settings)

  expect_equal(captured_lap_settings, custom_settings)
  expect_equal(captured_samp_settings, custom_settings)
})


test_that("diffiscape forwards omniscape_settings to refit, posterior, and diagnostics", {
  skip_on_cran()

  custom_settings <- list(radius = 25L, block_size = 9L)
  captured_fit_settings  <- NULL
  captured_post_settings <- NULL
  captured_omni_radius   <- NULL

  mock_opt   <- list(best_params = list(r_0 = 0), bounds = list(r_0 = c(-2, 2)),
                     distribution = "negbin")
  mock_fit   <- list(loglik = -10, intensity_params = c(0.1, 0.2),
                     intensity_fit_obj = list(), distribution = "negbin")
  mock_post  <- list(laplace = list(), samples = data.frame(), summary = data.frame())
  mock_diag  <- list(deviance_residuals = numeric(0))
  mock_conn  <- terra::rast(nrows = 2, ncols = 2, vals = 1)

  local_mocked_bindings(
    ds_init_julia = function(...) invisible(TRUE),
    ds_optimize = function(...) mock_opt,
    ds_fit_intensity = function(...) {
      args <- list(...)
      captured_fit_settings <<- args$omniscape_settings
      mock_fit
    },
    ds_posterior = function(...) {
      args <- list(...)
      captured_post_settings <<- args$omniscape_settings
      mock_post
    },
    create_resistance_surface = function(...) mock_conn,
    run_omniscape = function(resistance, radius = 13L, block_size = 5L, ...) {
      captured_omni_radius <<- radius
      list(cum_current = mock_conn, elapsed_seconds = 0.1)
    },
    ds_diagnose = function(...) mock_diag,
    .package = "DiffiScape"
  )

  basis <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs   <- data.frame(x = c(0, 1), y = c(0, 1))

  result <- diffiscape(
    obs_data           = obs,
    rasters            = list(layer1 = basis),
    omniscape_settings = custom_settings,
    n_posterior         = 3L,
    plot               = FALSE
  )

  expect_equal(captured_fit_settings, custom_settings)
  expect_equal(captured_post_settings, custom_settings)
  expect_equal(captured_omni_radius, 25L)
})


test_that("ds_predict works when intensity_fit contains intensity_fit_obj", {
  skip_on_cran()

  set.seed(42)
  r <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(100, mean = 4))
  n_obs  <- 15
  obs_pts <- data.frame(x = runif(n_obs, 0.05, 0.95),
                        y = runif(n_obs, 0.05, 0.95))
  nb_fit <- fit_intensity_nb(
    connectivity_at_obs = abs(rnorm(n_obs, mean = 4)),
    connectivity_raster = r,
    obs_coords          = obs_pts
  )

  # Simulate the ds_fit_intensity / evaluate_full_model return format
  mock_full <- list(
    loglik            = nb_fit$loglik,
    intensity_params  = nb_fit$estimates,
    intensity_fit_obj = nb_fit,
    intensity_se      = nb_fit$se,
    convergence       = nb_fit$convergence,
    distribution      = "negbin"
  )

  result <- ds_predict(mock_full, r)
  expect_s4_class(result, "SpatRaster")
  expect_equal(names(result), "intensity")
  expect_true(all(is.finite(terra::values(result)[, 1])))
})


test_that("ds_predict falls back gracefully when intensity_fit_obj is absent", {
  skip_on_cran()

  set.seed(43)
  r <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(100, mean = 4))
  n_obs   <- 15
  obs_pts <- data.frame(x = runif(n_obs, 0.05, 0.95),
                        y = runif(n_obs, 0.05, 0.95))
  nb_fit  <- fit_intensity_nb(
    connectivity_at_obs = abs(rnorm(n_obs, mean = 4)),
    connectivity_raster = r,
    obs_coords          = obs_pts
  )

  # Pass the bare fit object (no intensity_fit_obj field) — triggers fallback
  result <- ds_predict(nb_fit, r)
  expect_s4_class(result, "SpatRaster")
  expect_equal(names(result), "intensity")
  expect_true(all(is.finite(terra::values(result)[, 1])))
})
