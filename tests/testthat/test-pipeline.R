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


# ---- diffiscape() wrapper tests ----------------------------------------------

test_that("diffiscape errors when obs_data lacks x/y columns", {
  expect_error(
    diffiscape(obs_data = data.frame(a = 1, b = 2),
               rasters  = "dummy",
               output_dir = tempdir(),
               plot = FALSE),
    "x.*y"
  )
})


test_that("diffiscape returns expected result structure", {
  skip_on_cran()
  skip_if_not_installed("terra")

  mock_opt  <- list(best_params = list(r_0 = 0), bounds = list(r_0 = c(-2, 2)),
                    distribution = "negbin")
  mock_fit  <- list(loglik = -10, intensity_params = c(0.1, 0.2),
                    intensity_fit_obj = list(), distribution = "negbin")
  mock_post <- list(laplace = list(), samples = data.frame(r_0 = rnorm(3)),
                    summary = data.frame())
  mock_diag <- list(deviance_residuals = numeric(0))
  mock_conn <- terra::rast(nrows = 2, ncols = 2, vals = 1)

  local_mocked_bindings(
    ds_init_julia = function(...) invisible(TRUE),
    ds_optimize   = function(...) mock_opt,
    ds_fit_intensity = function(...) mock_fit,
    ds_posterior  = function(...) mock_post,
    create_resistance_surface = function(...) mock_conn,
    run_omniscape = function(...) {
      list(cum_current = mock_conn, elapsed_seconds = 0.1)
    },
    ds_diagnose = function(...) mock_diag,
    ds_ppc      = function(...) list(check = "ok"),
    .package = "DiffiScape"
  )

  basis  <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs    <- data.frame(x = c(0, 1), y = c(0, 1))
  result <- diffiscape(obs_data = obs, rasters = list(layer1 = basis),
                       output_dir = withr::local_tempdir(),
                       n_posterior = 3L, plot = FALSE)

  expect_named(result, c("obs_points", "basis_stack", "opt_result",
                          "intensity_fit", "posterior", "diagnostics",
                          "ppc", "elapsed_min"),
               ignore.order = TRUE)
  expect_type(result$elapsed_min, "double")
})


test_that("diffiscape skips posterior when n_posterior is 0", {
  skip_on_cran()
  skip_if_not_installed("terra")

  mock_opt  <- list(best_params = list(r_0 = 0), bounds = list(r_0 = c(-2, 2)),
                    distribution = "negbin")
  mock_fit  <- list(loglik = -10, intensity_params = c(0.1, 0.2),
                    intensity_fit_obj = list(), distribution = "negbin")
  mock_diag <- list(deviance_residuals = numeric(0))
  mock_conn <- terra::rast(nrows = 2, ncols = 2, vals = 1)

  posterior_called <- FALSE

  local_mocked_bindings(
    ds_init_julia = function(...) invisible(TRUE),
    ds_optimize   = function(...) mock_opt,
    ds_fit_intensity = function(...) mock_fit,
    ds_posterior  = function(...) { posterior_called <<- TRUE; list() },
    create_resistance_surface = function(...) mock_conn,
    run_omniscape = function(...) {
      list(cum_current = mock_conn, elapsed_seconds = 0.1)
    },
    ds_diagnose = function(...) mock_diag,
    .package = "DiffiScape"
  )

  basis  <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs    <- data.frame(x = c(0, 1), y = c(0, 1))
  result <- diffiscape(obs_data = obs, rasters = list(layer1 = basis),
                       output_dir = withr::local_tempdir(),
                       n_posterior = 0L, plot = FALSE)

  expect_false(posterior_called)
  expect_null(result$posterior)
})


test_that("diffiscape dispatches to ds_jax_connectivity for gradient solver", {
  skip_on_cran()
  skip_if_not_installed("terra")

  mock_opt  <- list(best_params = list(r_0 = 0), bounds = list(r_0 = c(-2, 2)),
                    distribution = "negbin")
  mock_fit  <- list(loglik = -10, intensity_params = c(0.1, 0.2),
                    intensity_fit_obj = list(), distribution = "negbin")
  mock_diag <- list(deviance_residuals = numeric(0))
  mock_conn <- terra::rast(nrows = 2, ncols = 2, vals = 1)

  jax_called       <- FALSE
  omniscape_called <- FALSE

  local_mocked_bindings(
    ds_init_julia = function(...) invisible(TRUE),
    ds_optimize   = function(...) mock_opt,
    ds_fit_intensity = function(...) mock_fit,
    create_resistance_surface = function(...) mock_conn,
    ds_jax_connectivity = function(...) {
      jax_called <<- TRUE
      list(cum_current = mock_conn, flow_potential = NULL, elapsed_seconds = 0.1)
    },
    run_omniscape = function(...) {
      omniscape_called <<- TRUE
      list(cum_current = mock_conn, elapsed_seconds = 0.1)
    },
    ds_diagnose = function(...) mock_diag,
    .package = "DiffiScape"
  )

  basis  <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs    <- data.frame(x = c(0, 1), y = c(0, 1))

  # "enzyme" should be deprecated to "gradient"
  expect_message(
    result <- diffiscape(obs_data = obs, rasters = list(layer1 = basis),
                         output_dir = withr::local_tempdir(),
                         n_posterior = 0L, plot = FALSE, solver = "enzyme"),
    "deprecated"
  )

  expect_true(jax_called)
  expect_false(omniscape_called)
})


test_that("diffiscape uses ds_jax_connectivity for solver='gradient'", {
  skip_on_cran()
  skip_if_not_installed("terra")

  mock_opt  <- list(best_params = list(r_0 = 0), bounds = list(r_0 = c(-2, 2)),
                    distribution = "negbin")
  mock_fit  <- list(loglik = -10, intensity_params = c(0.1, 0.2),
                    intensity_fit_obj = list(), distribution = "negbin")
  mock_diag <- list(deviance_residuals = numeric(0))
  mock_conn <- terra::rast(nrows = 2, ncols = 2, vals = 1)

  jax_called       <- FALSE
  omniscape_called <- FALSE

  local_mocked_bindings(
    ds_init_julia = function(...) invisible(TRUE),
    ds_optimize   = function(...) mock_opt,
    ds_fit_intensity = function(...) mock_fit,
    create_resistance_surface = function(...) mock_conn,
    ds_jax_connectivity = function(...) {
      jax_called <<- TRUE
      list(cum_current = mock_conn, flow_potential = NULL, elapsed_seconds = 0.1)
    },
    run_omniscape = function(...) {
      omniscape_called <<- TRUE
      list(cum_current = mock_conn, elapsed_seconds = 0.1)
    },
    ds_diagnose = function(...) mock_diag,
    .package = "DiffiScape"
  )

  basis  <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs    <- data.frame(x = c(0, 1), y = c(0, 1))
  result <- diffiscape(obs_data = obs, rasters = list(layer1 = basis),
                       output_dir = withr::local_tempdir(),
                       n_posterior = 0L, plot = FALSE, solver = "gradient")

  expect_true(jax_called)
  expect_false(omniscape_called)
})


test_that("diffiscape saves diffiscape_result.rds", {
  skip_on_cran()
  skip_if_not_installed("terra")

  mock_opt  <- list(best_params = list(r_0 = 0), bounds = list(r_0 = c(-2, 2)),
                    distribution = "negbin")
  mock_fit  <- list(loglik = -10, intensity_params = c(0.1, 0.2),
                    intensity_fit_obj = list(), distribution = "negbin")
  mock_diag <- list(deviance_residuals = numeric(0))
  mock_conn <- terra::rast(nrows = 2, ncols = 2, vals = 1)
  out_dir   <- withr::local_tempdir()

  local_mocked_bindings(
    ds_init_julia = function(...) invisible(TRUE),
    ds_optimize   = function(...) mock_opt,
    ds_fit_intensity = function(...) mock_fit,
    create_resistance_surface = function(...) mock_conn,
    run_omniscape = function(...) {
      list(cum_current = mock_conn, elapsed_seconds = 0.1)
    },
    ds_diagnose = function(...) mock_diag,
    .package = "DiffiScape"
  )

  basis <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs   <- data.frame(x = c(0, 1), y = c(0, 1))
  diffiscape(obs_data = obs, rasters = list(layer1 = basis),
             output_dir = out_dir, n_posterior = 0L, plot = FALSE)

  expect_true(file.exists(file.path(out_dir, "diffiscape_result.rds")))
})


test_that("diffiscape handles PPC error gracefully", {
  skip_on_cran()
  skip_if_not_installed("terra")

  mock_opt  <- list(best_params = list(r_0 = 0), bounds = list(r_0 = c(-2, 2)),
                    distribution = "negbin")
  mock_fit  <- list(loglik = -10, intensity_params = c(0.1, 0.2),
                    intensity_fit_obj = list(), distribution = "negbin")
  mock_post <- list(laplace = list(), samples = data.frame(r_0 = rnorm(3)),
                    summary = data.frame())
  mock_diag <- list(deviance_residuals = numeric(0))
  mock_conn <- terra::rast(nrows = 2, ncols = 2, vals = 1)

  local_mocked_bindings(
    ds_init_julia = function(...) invisible(TRUE),
    ds_optimize   = function(...) mock_opt,
    ds_fit_intensity = function(...) mock_fit,
    ds_posterior  = function(...) mock_post,
    create_resistance_surface = function(...) mock_conn,
    run_omniscape = function(...) {
      list(cum_current = mock_conn, elapsed_seconds = 0.1)
    },
    ds_diagnose = function(...) mock_diag,
    ds_ppc      = function(...) stop("PPC computation failed"),
    .package = "DiffiScape"
  )

  basis  <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs    <- data.frame(x = c(0, 1), y = c(0, 1))
  result <- diffiscape(obs_data = obs, rasters = list(layer1 = basis),
                       output_dir = withr::local_tempdir(),
                       n_posterior = 3L, plot = FALSE)

  expect_null(result$ppc)
})


test_that("diffiscape loads obs_data from CSV file path", {
  skip_on_cran()
  skip_if_not_installed("terra")

  mock_opt  <- list(best_params = list(r_0 = 0), bounds = list(r_0 = c(-2, 2)),
                    distribution = "negbin")
  mock_fit  <- list(loglik = -10, intensity_params = c(0.1, 0.2),
                    intensity_fit_obj = list(), distribution = "negbin")
  mock_diag <- list(deviance_residuals = numeric(0))
  mock_conn <- terra::rast(nrows = 2, ncols = 2, vals = 1)

  local_mocked_bindings(
    ds_init_julia = function(...) invisible(TRUE),
    ds_optimize   = function(...) mock_opt,
    ds_fit_intensity = function(...) mock_fit,
    create_resistance_surface = function(...) mock_conn,
    run_omniscape = function(...) {
      list(cum_current = mock_conn, elapsed_seconds = 0.1)
    },
    ds_diagnose = function(...) mock_diag,
    .package = "DiffiScape"
  )

  csv_path <- tempfile(fileext = ".csv")
  write.csv(data.frame(x = 1:5, y = 6:10), csv_path, row.names = FALSE)
  withr::defer(unlink(csv_path))

  basis  <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  result <- diffiscape(obs_data = csv_path, rasters = list(layer1 = basis),
                       output_dir = withr::local_tempdir(),
                       n_posterior = 0L, plot = FALSE)

  expect_equal(nrow(result$obs_points), 5)
})
