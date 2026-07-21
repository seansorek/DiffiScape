# Tests for R/pipeline.R

test_that("torch is the default solver", {
  expect_identical(formals(ds_optimize)$solver[[1]], "torch")
  expect_identical(formals(diffiscape)$solver[[1]], "torch")
})

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
  captured_jax_radius    <- NULL

  mock_opt   <- list(best_params = list(r_0 = 0), bounds = list(r_0 = c(-2, 2)),
                     distribution = "negbin")
  mock_fit   <- list(loglik = -10, intensity_params = c(0.1, 0.2),
                     intensity_fit_obj = list(), distribution = "negbin")
  mock_post  <- list(laplace = list(), samples = data.frame(), summary = data.frame())
  mock_diag  <- list(deviance_residuals = numeric(0))
  mock_conn  <- terra::rast(nrows = 2, ncols = 2, vals = 1)

  local_mocked_bindings(
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
    ds_jax_connectivity = function(resistance, radius = 13L, block_size = 5L, ...) {
      captured_jax_radius <<- radius
      list(cum_current = mock_conn, flow_potential = NULL, elapsed_seconds = 0.1)
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
  expect_equal(captured_jax_radius, 25L)
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


test_that("diffiscape deprecates solver = 'enzyme' to gradient", {
  # The deprecation alias fires near the top of diffiscape(), before any
  # connectivity compute, so we catch the (expected) downstream error after
  # confirming the deprecation message was emitted. No Python/JAX required.
  expect_message(
    tryCatch(
      diffiscape(obs_data = data.frame(a = 1, b = 2),  # lacks x/y -> errors
                 rasters  = "dummy",
                 output_dir = withr::local_tempdir(),
                 solver = "enzyme",
                 plot = FALSE),
      error = function(e) invisible(NULL)
    ),
    "deprecated"
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
    ds_optimize   = function(...) mock_opt,
    ds_fit_intensity = function(...) mock_fit,
    ds_posterior  = function(...) mock_post,
    create_resistance_surface = function(...) mock_conn,
    ds_jax_connectivity = function(...) {
      list(cum_current = mock_conn, flow_potential = NULL, elapsed_seconds = 0.1)
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
    ds_optimize   = function(...) mock_opt,
    ds_fit_intensity = function(...) mock_fit,
    ds_posterior  = function(...) { posterior_called <<- TRUE; list() },
    create_resistance_surface = function(...) mock_conn,
    ds_jax_connectivity = function(...) {
      list(cum_current = mock_conn, flow_potential = NULL, elapsed_seconds = 0.1)
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

  jax_called <- FALSE

  local_mocked_bindings(
    ds_optimize   = function(...) mock_opt,
    ds_fit_intensity = function(...) mock_fit,
    create_resistance_surface = function(...) mock_conn,
    ds_jax_connectivity = function(...) {
      jax_called <<- TRUE
      list(cum_current = mock_conn, flow_potential = NULL, elapsed_seconds = 0.1)
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

  jax_called <- FALSE

  local_mocked_bindings(
    ds_optimize   = function(...) mock_opt,
    ds_fit_intensity = function(...) mock_fit,
    create_resistance_surface = function(...) mock_conn,
    ds_jax_connectivity = function(...) {
      jax_called <<- TRUE
      list(cum_current = mock_conn, flow_potential = NULL, elapsed_seconds = 0.1)
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
    ds_optimize   = function(...) mock_opt,
    ds_fit_intensity = function(...) mock_fit,
    create_resistance_surface = function(...) mock_conn,
    ds_jax_connectivity = function(...) {
      list(cum_current = mock_conn, flow_potential = NULL, elapsed_seconds = 0.1)
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
    ds_optimize   = function(...) mock_opt,
    ds_fit_intensity = function(...) mock_fit,
    ds_posterior  = function(...) mock_post,
    create_resistance_surface = function(...) mock_conn,
    ds_jax_connectivity = function(...) {
      list(cum_current = mock_conn, flow_potential = NULL, elapsed_seconds = 0.1)
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


# ---- ds_fit_intensity() solver routing (Problem B collapse) -----------------
# Characterization tests written ahead of collapsing the inline "gradient"
# branch of ds_fit_intensity() so it routes through evaluate_full_model()
# unconditionally, just like the "surrogate" branch already does.

test_that("ds_fit_intensity(solver = 'gradient') calls evaluate_full_model()", {
  skip_on_cran()

  opt_result <- list(best_params = list(r_0 = 0), distribution = "negbin")
  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)

  captured_args <- NULL
  mock_result <- list(loglik = -1, intensity_params = c(alpha = 0, gamma = 0),
                      intensity_fit_obj = list(), intensity_se = NULL,
                      hessian = NULL, convergence = 0L, distribution = "negbin",
                      total_time = 0.5, omniscape_time = 0.2)

  local_mocked_bindings(
    evaluate_full_model = function(...) {
      captured_args <<- list(...)
      mock_result
    },
    .package = "DiffiScape"
  )

  result <- ds_fit_intensity(opt_result, basis_stack, obs_points,
                             solver = "gradient")

  expect_false(is.null(captured_args))
  expect_equal(captured_args$resistance_params, opt_result$best_params)
  expect_identical(result, mock_result)
})


test_that("ds_fit_intensity(solver = 'surrogate') calls evaluate_full_model() identically to 'gradient'", {
  skip_on_cran()

  opt_result <- list(best_params = list(r_0 = 0), distribution = "negbin")
  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)

  captured_calls <- list()
  mock_result <- list(loglik = -1, intensity_params = c(alpha = 0, gamma = 0),
                      intensity_fit_obj = list(), intensity_se = NULL,
                      hessian = NULL, convergence = 0L, distribution = "negbin",
                      total_time = 0.5, omniscape_time = 0.2)

  local_mocked_bindings(
    evaluate_full_model = function(...) {
      captured_calls[[length(captured_calls) + 1L]] <<- list(...)
      mock_result
    },
    .package = "DiffiScape"
  )

  ds_fit_intensity(opt_result, basis_stack, obs_points, solver = "surrogate")
  ds_fit_intensity(opt_result, basis_stack, obs_points, solver = "gradient")

  expect_equal(length(captured_calls), 2L)
  expect_equal(captured_calls[[1]], captured_calls[[2]])
})


test_that("ds_fit_intensity(solver = 'gradient') returns evaluate_full_model()'s full return shape (total_time + omniscape_time)", {
  skip_on_cran()

  opt_result <- list(best_params = list(r_0 = 0), distribution = "negbin")
  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)

  mock_result <- list(loglik = -1, intensity_params = c(alpha = 0, gamma = 0),
                      intensity_fit_obj = list(), intensity_se = NULL,
                      hessian = NULL, convergence = 0L, distribution = "negbin",
                      total_time = 1.23, omniscape_time = 0.45)

  local_mocked_bindings(
    evaluate_full_model = function(...) mock_result,
    .package = "DiffiScape"
  )

  result <- ds_fit_intensity(opt_result, basis_stack, obs_points,
                             solver = "gradient")

  expect_equal(result$total_time, 1.23)
  expect_equal(result$omniscape_time, 0.45)
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
    ds_optimize   = function(...) mock_opt,
    ds_fit_intensity = function(...) mock_fit,
    create_resistance_surface = function(...) mock_conn,
    ds_jax_connectivity = function(...) {
      list(cum_current = mock_conn, flow_potential = NULL, elapsed_seconds = 0.1)
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


# ============================================================================
# Issue #4: S3 solver dispatch refactor (ds_optimize / ds_fit_intensity /
# diffiscape) -- characterization tests written BEFORE introducing
# solver_spec() + S3 dispatch generics, per TDD. Sections 1-2 lock in CURRENT
# if/else behavior; sections 3-4 encode TARGET behavior not yet implemented
# (ds_fit_intensity() torch/irl guard, diffiscape() torch/irl routing) so
# they fail red against the current code and pass once the refactor lands.
# ============================================================================

# ---- 1. ds_optimize(solver = "torch"/"irl") dispatch into run_torch_pipeline() ----

test_that("ds_optimize(solver = 'torch') dispatches to run_torch_pipeline() with config$torch args", {
  skip_on_cran()

  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)
  mock_result <- list(best_params = data.frame(r_0 = 0), best_loglik = -1)

  captured_args <- NULL
  local_mocked_bindings(
    run_torch_pipeline = function(...) {
      captured_args <<- list(...)
      mock_result
    },
    .package = "DiffiScape"
  )

  torch_cfg <- list(n_epochs = 7L, lr = 0.05)
  result <- ds_optimize(
    basis_stack, obs_points,
    config = list(torch = torch_cfg, seed = 99L),
    output_dir = "torch_out",
    solver = "torch"
  )

  expect_identical(result, mock_result)
  expect_false(is.null(captured_args))
  expect_identical(captured_args$basis_stack, basis_stack)
  expect_identical(captured_args$obs_points, obs_points)
  expect_equal(captured_args$n_epochs, 7L)
  expect_equal(captured_args$lr, 0.05)
  expect_equal(captured_args$output_dir, "torch_out")
  expect_equal(captured_args$seed, 99L)
  expect_null(captured_args$model_type)
})


test_that("ds_optimize(solver = 'irl') dispatches to run_torch_pipeline() with model_type forced to 'irl'", {
  skip_on_cran()

  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)
  mock_result <- list(best_params = data.frame(r_0 = 0), best_loglik = -1)

  captured_args <- NULL
  local_mocked_bindings(
    run_torch_pipeline = function(...) {
      captured_args <<- list(...)
      mock_result
    },
    .package = "DiffiScape"
  )

  result <- ds_optimize(
    basis_stack, obs_points,
    config = list(torch = list(n_epochs = 3L)),
    solver = "irl"
  )

  expect_identical(result, mock_result)
  expect_false(is.null(captured_args))
  expect_equal(captured_args$model_type, "irl")
  expect_equal(captured_args$n_epochs, 3L)
})


test_that("ds_optimize(solver = 'torch') errors when available_points is supplied", {
  skip_on_cran()

  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)
  avail_pts   <- data.frame(x = 0.5, y = 0.5)

  expect_error(
    ds_optimize(basis_stack, obs_points, solver = "torch",
                available_points = avail_pts),
    "available_points is not supported"
  )
})


test_that("ds_optimize(solver = 'torch') defaults output_dir from the function arg when config$torch$output_dir is absent", {
  skip_on_cran()

  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)
  mock_result <- list(best_params = data.frame(r_0 = 0), best_loglik = -1)

  captured_args <- NULL
  local_mocked_bindings(
    run_torch_pipeline = function(...) {
      captured_args <<- list(...)
      mock_result
    },
    .package = "DiffiScape"
  )

  ds_optimize(basis_stack, obs_points, solver = "torch",
              output_dir = "my_output_dir")

  expect_equal(captured_args$output_dir, "my_output_dir")
})


# ---- 2. Exact "enzyme" deprecation message text across all three entry points ----
# message() appends a trailing newline to the supplied string, so
# conditionMessage() includes it. Locking in the literal text byte-for-byte
# before solver_spec() centralizes it.

test_that("ds_optimize() 'enzyme' deprecation message text is exact", {
  msgs <- testthat::capture_messages(
    tryCatch(
      ds_optimize(NULL, NULL, solver = "enzyme"),
      error = function(e) invisible(NULL)
    )
  )

  expect_equal(length(msgs), 1L)
  expect_equal(
    msgs[[1]],
    "solver='enzyme' is deprecated. Use solver='gradient' instead.\n"
  )
})


test_that("ds_fit_intensity() 'enzyme' deprecation message text is exact", {
  opt_result <- list(best_params = list(r_0 = 0), distribution = "negbin")
  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)

  local_mocked_bindings(
    evaluate_full_model = function(...) list(loglik = -1),
    .package = "DiffiScape"
  )

  msgs <- testthat::capture_messages(
    ds_fit_intensity(opt_result, basis_stack, obs_points, solver = "enzyme")
  )

  expect_equal(length(msgs), 1L)
  expect_equal(
    msgs[[1]],
    "solver='enzyme' is deprecated. Use solver='gradient' instead.\n"
  )
})


test_that("diffiscape() 'enzyme' deprecation message text is exact", {
  msgs <- testthat::capture_messages(
    tryCatch(
      diffiscape(obs_data = data.frame(a = 1, b = 2),  # lacks x/y -> errors downstream
                 rasters  = "dummy",
                 output_dir = withr::local_tempdir(),
                 solver = "enzyme",
                 plot = FALSE),
      error = function(e) invisible(NULL)
    )
  )

  expect_true(any(grepl("deprecated", msgs, fixed = TRUE)))
  deprecation_msg <- msgs[grepl("deprecated", msgs, fixed = TRUE)][[1]]
  expect_equal(
    deprecation_msg,
    "solver='enzyme' is deprecated. Use solver='gradient' instead.\n"
  )
})


# ---- 3. ds_fit_intensity() rejecting torch/irl (TARGET behavior) ----
# Today these fail with R's generic match.arg error (not in the choices set).
# These tests assert the NEW, more actionable error message that section 3
# of the refactor introduces -- written as red tests against current code.

test_that("ds_fit_intensity(solver = 'torch') stops with an actionable message", {
  opt_result  <- list(best_params = list(r_0 = 0), distribution = "negbin")
  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)

  expect_error(
    ds_fit_intensity(opt_result, basis_stack, obs_points, solver = "torch"),
    "does not support solver"
  )
})


test_that("ds_fit_intensity(solver = 'irl') stops with an actionable message", {
  opt_result  <- list(best_params = list(r_0 = 0), distribution = "negbin")
  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)

  expect_error(
    ds_fit_intensity(opt_result, basis_stack, obs_points, solver = "irl"),
    "does not support solver"
  )
})


# ---- 4. diffiscape(solver = "torch"/"irl") end-to-end routing (TARGET behavior) ----
# diffiscape()'s match.arg currently only accepts c("surrogate", "gradient",
# "enzyme"), so these fail today with a match.arg error. After the refactor,
# diffiscape() should call ds_optimize(solver = "torch"/"irl", ...), skip the
# separate ds_fit_intensity() call entirely (the torch/irl backend already
# fits resistance + intensity jointly), and assemble the normal return shape
# from the torch/irl result fields.

test_that("diffiscape(solver = 'torch') does not call ds_fit_intensity and returns torch result fields", {
  skip_on_cran()
  skip_if_not_installed("terra")

  mock_torch_result <- list(
    best_params         = data.frame(r_0 = 0, z_1 = 0.1),
    best_loglik         = -12.3,
    alpha               = 1.1,
    gamma               = 2.2,
    resistance_raster   = terra::rast(nrows = 2, ncols = 2, vals = 1),
    connectivity_raster = terra::rast(nrows = 2, ncols = 2, vals = 1),
    intensity_raster    = terra::rast(nrows = 2, ncols = 2, vals = 1),
    loss_history        = c(1, 0.5),
    best_epoch          = 10L,
    n_params            = 5L,
    n_epochs_run        = 20L,
    total_time          = 3.4,
    method              = "torch_nn_circuit",
    distribution        = "poisson_parametric",
    effective_loglinear = c(0, 0.1),
    model_type          = "mlp",
    partial_effects     = NULL,
    interaction_effects = NULL,
    uq_results          = NULL,
    intensity_params    = c(alpha = 1.1, gamma = 2.2),
    intensity_se        = c(alpha = NA_real_, gamma = NA_real_),
    convergence         = 0L,
    convergence_message = "early_stop_epoch_10",
    n_evaluations       = 20L,
    solver              = "circuit_global"
  )

  fit_intensity_called <- FALSE

  local_mocked_bindings(
    ds_optimize = function(...) mock_torch_result,
    ds_fit_intensity = function(...) {
      fit_intensity_called <<- TRUE
      stop("should not be called for torch/irl")
    },
    ds_diagnose = function(...) list(deviance_residuals = numeric(0)),
    .package = "DiffiScape"
  )

  basis <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs   <- data.frame(x = c(0, 1), y = c(0, 1))

  result <- diffiscape(obs_data = obs, rasters = list(layer1 = basis),
                       output_dir = withr::local_tempdir(),
                       n_posterior = 0L, plot = FALSE, solver = "torch")

  expect_false(fit_intensity_called)
  expect_named(result, c("obs_points", "basis_stack", "opt_result",
                          "intensity_fit", "posterior", "diagnostics",
                          "ppc", "elapsed_min"),
               ignore.order = TRUE)
  expect_identical(result$opt_result, mock_torch_result)
})


test_that("diffiscape(solver = 'irl') does not call ds_fit_intensity and routes through ds_optimize", {
  skip_on_cran()
  skip_if_not_installed("terra")

  mock_torch_result <- list(
    best_params         = data.frame(r_0 = 0, z_1 = 0.1),
    best_loglik         = -5.6,
    alpha               = 0.9,
    gamma               = 1.8,
    resistance_raster   = terra::rast(nrows = 2, ncols = 2, vals = 1),
    connectivity_raster = terra::rast(nrows = 2, ncols = 2, vals = 1),
    intensity_raster    = terra::rast(nrows = 2, ncols = 2, vals = 1),
    loss_history        = c(1, 0.2),
    best_epoch          = 8L,
    n_params            = 4L,
    n_epochs_run        = 15L,
    total_time          = 2.1,
    method              = "torch_irl_value_circuit",
    distribution        = "poisson_parametric",
    effective_loglinear = c(0, 0.05),
    model_type          = "irl",
    partial_effects     = NULL,
    interaction_effects = NULL,
    uq_results          = NULL,
    intensity_params    = c(alpha = 0.9, gamma = 1.8),
    intensity_se        = c(alpha = NA_real_, gamma = NA_real_),
    convergence         = 0L,
    convergence_message = "early_stop_epoch_8",
    n_evaluations       = 15L,
    solver              = "circuit_global"
  )

  fit_intensity_called <- FALSE
  captured_solver <- NULL

  local_mocked_bindings(
    ds_optimize = function(..., solver) {
      captured_solver <<- solver
      mock_torch_result
    },
    ds_fit_intensity = function(...) {
      fit_intensity_called <<- TRUE
      stop("should not be called for torch/irl")
    },
    ds_diagnose = function(...) list(deviance_residuals = numeric(0)),
    .package = "DiffiScape"
  )

  basis <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs   <- data.frame(x = c(0, 1), y = c(0, 1))

  result <- diffiscape(obs_data = obs, rasters = list(layer1 = basis),
                       output_dir = withr::local_tempdir(),
                       n_posterior = 0L, plot = FALSE, solver = "irl")

  expect_false(fit_intensity_called)
  expect_equal(captured_solver, "irl")
  expect_identical(result$opt_result, mock_torch_result)
})
