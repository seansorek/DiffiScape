# Tests for R/connectivity.R

test_that("extract_connectivity function exists", {
  expect_true(is.function(extract_connectivity))
})


# ---- extract_connectivity behavioral tests -----------------------------------

test_that("extract_connectivity returns correct values from data.frame input", {
  skip_if_not_installed("terra")

  r <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 5, ymin = 0, ymax = 5)
  terra::values(r) <- 1:25

  pts <- data.frame(x = c(0.5, 1.5, 2.5), y = c(4.5, 3.5, 2.5))
  result <- extract_connectivity(r, pts)
  expect_equal(result, c(1, 7, 13))
})

test_that("extract_connectivity accepts matrix input", {
  skip_if_not_installed("terra")

  r <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 5, ymin = 0, ymax = 5)
  terra::values(r) <- 1:25

  pts <- matrix(c(0.5, 1.5, 4.5, 3.5), ncol = 2)
  colnames(pts) <- c("x", "y")
  result <- extract_connectivity(r, pts)
  expect_equal(result, c(1, 7))
})

test_that("extract_connectivity accepts SpatVector input", {
  skip_if_not_installed("terra")

  r <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 5, ymin = 0, ymax = 5)
  terra::values(r) <- 1:25

  pts <- terra::vect(data.frame(x = c(0.5, 1.5), y = c(4.5, 3.5)),
                     geom = c("x", "y"), crs = terra::crs(r))
  result <- extract_connectivity(r, pts)
  expect_equal(result, c(1, 7))
})

test_that("extract_connectivity with buffer returns mean of surrounding cells", {
  skip_if_not_installed("terra")

  r <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 5, ymin = 0, ymax = 5)
  terra::values(r) <- 10
  pts <- data.frame(x = 2.5, y = 2.5)

  result <- extract_connectivity(r, pts, buffer = 0.5)
  expect_equal(result, 10)
})

test_that("run_cumulative_current function exists", {
  expect_true(is.function(run_cumulative_current))
})

test_that("run_cumulative_current accepts parameterization parameter", {
  args <- formals(run_cumulative_current)
  expect_true("parameterization" %in% names(args))
  expect_equal(args$parameterization, "resistance")
})

# -- JAX backend integration tests -------------------------------------------

test_that("run_cumulative_current works with JAX backend", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("jaxscape"),
          "jaxscape not installed")
  ds_jax_setup()

  r <- terra::rast(nrows = 15, ncols = 15,
                   vals = runif(225, 1, 100))
  result <- run_cumulative_current(r, radius = 5L, block_size = 3L)
  expect_type(result, "list")
  expect_s4_class(result$cum_current, "SpatRaster")
  expect_null(result$flow_potential)
  expect_true(is.numeric(result$elapsed_seconds))
})

test_that("run_cumulative_current JAX backend returns voltage", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("jaxscape"),
          "jaxscape not installed")
  ds_jax_setup()

  r <- terra::rast(nrows = 15, ncols = 15,
                   vals = runif(225, 1, 100))
  result <- run_cumulative_current(r, radius = 5L, block_size = 3L,
                                   output = "both")
  expect_type(result, "list")
  expect_s4_class(result$cum_current, "SpatRaster")
  expect_s4_class(result$flow_potential, "SpatRaster")
})
