test_that("installed Python and R fitting/prediction agree on a masked nonsquare grid", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("diffiscape"))
  ds_jax_setup(force = TRUE)
  np <- reticulate::import("numpy", convert = FALSE)
  py <- reticulate::import("diffiscape", convert = FALSE)
  raster <- terra::rast(nrows = 2, ncols = 3, xmin = 0, xmax = 3,
                         ymin = 0, ymax = 2, crs = "EPSG:3857")
  terra::values(raster) <- c(-1, NA, -0.5, 0, 0.5, 1)
  points <- as.data.frame(terra::xyFromCell(raster, c(3, 3, 5, 6, 6, 6)))
  cfg <- default_optimizer_config()
  cfg$n_iter <- 4L
  cfg$omniscape$radius <- 1L
  cfg$omniscape$block_size <- 1L
  bounds <- list(r_0 = c(-1, 1), z_1 = c(-1, 1))
  fitted <- optimize_resistance_gradient(raster, points, bounds = bounds,
    config = cfg, method = "adam", output_dir = withr::local_tempdir())
  basis <- np$array(matrix(c(-1, -0.5, 0, 0.5, 1), ncol = 1), dtype = np$float64)
  direct <- py$fit(basis, np$array(c(0, 2, 0, 1, 3)),
    np$array(c(TRUE, FALSE, TRUE, TRUE, TRUE, TRUE)), 2L, 3L,
    formulation = "window_resistance",
    init_params = np$array(c(0, 0)), lower_bounds = np$array(c(-1, -1)),
    upper_bounds = np$array(c(1, 1)), method = "adam", n_epochs = 4L,
    radius = 1L, block_size = 1L, verbose = FALSE)
  expected <- reticulate::py_to_r(py$predict(direct, basis))
  expect_equal(fitted$best_loglik, reticulate::py_to_r(direct[["best_loglik"]]), tolerance = 1e-9)
  prediction <- ds_predict(fitted, basis_stack = raster)
  expect_equal(as.numeric(terra::values(prediction)), as.vector(t(expected$intensity)), tolerance = 1e-9)
  saved <- tempfile(fileext = ".rds")
  on.exit(unlink(saved))
  saveRDS(fitted, saved)
  expect_equal(terra::values(ds_predict(readRDS(saved), basis_stack = raster)), terra::values(prediction))
})

test_that("the default physical model predicts, samples and diagnoses without Torch", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("diffiscape"))
  raster <- terra::rast(nrows = 2, ncols = 3, xmin = 0, xmax = 3,
    ymin = 0, ymax = 2, crs = "EPSG:3857", vals = c(0, NA, 0.2, 0.5, 0.8, 1))
  points <- as.data.frame(terra::xyFromCell(raster, c(1, 3, 3, 5, 6)))
  fitted <- ds_optimize(raster, points,
    config = list(fit = list(n_epochs = 2L, warmup_epochs = 0L, source_spacing = 2L)),
    output_dir = withr::local_tempdir())
  expect_identical(fitted$solver, "global_absorption")
  expect_match(fitted$method, "jax")
  expect_equal(terra::values(ds_predict(fitted, basis_stack = raster)), terra::values(fitted$intensity_raster))
  posterior <- ds_posterior(fitted, raster, points, n_draws = 3L, warmup = 3L, max_treedepth = 2L)
  expect_length(as.numeric(posterior$samples$alpha), 3)
  expect_true(all(as.numeric(posterior$samples$gamma) > 0))
  diagnostics <- ds_diagnose(fitted, points, fitted$connectivity_raster, plot = FALSE)
  expect_true(is.finite(diagnostics$mean_deviance))
  ppc <- ds_ppc(posterior, fitted, points, fitted$connectivity_raster,
    basis_stack = raster, n_sim = 3L, plot = FALSE)
  expect_s3_class(ppc, "ds_ppc")
  expect_equal(ppc$n_sim, 3L)
})

test_that("legacy neural fitting transfers valid cells and predicts from saved state", {
  skip_on_cran()
  skip_if_not(reticulate::py_module_available("diffiscape"))
  raster <- terra::rast(nrows = 2, ncols = 3, xmin = 0, xmax = 3, ymin = 0, ymax = 2,
    crs = "EPSG:3857", vals = c(0, NA, .2, .5, .8, 1))
  points <- as.data.frame(terra::xyFromCell(raster, c(1, 3, 3, 6)))
  fitted <- ds_optimize(raster, points, solver = "gradient", model_type = "conv",
    config = list(n_iter = 2L, seed = 7L, omniscape = list(radius = 1L, block_size = 1L),
      model_config = list(conv_channels = 2L, n_conv_layers = 1L)),
    output_dir = withr::local_tempdir())
  prediction <- ds_predict(fitted, basis_stack = raster)
  expect_equal(sum(is.na(terra::values(prediction))), 1L)
  intensity <- as.numeric(terra::values(prediction))
  intensity <- intensity[!is.na(intensity)]
  expect_equal(fitted$best_loglik, sum(c(1, 2, 0, 0, 1) * log(intensity)) - sum(intensity), tolerance = 1e-9)
})
