# Tests for R/connectivity.R

test_that("extract_connectivity function exists", {
  expect_true(is.function(extract_connectivity))
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
