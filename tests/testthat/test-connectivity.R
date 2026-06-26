# Tests for R/connectivity.R

test_that("run_omniscape function exists", {
  expect_true(is.function(run_omniscape))
})

test_that("run_circuitscape function exists", {
  expect_true(is.function(run_circuitscape))
})

test_that("extract_connectivity function exists", {
  expect_true(is.function(extract_connectivity))
})

# Connectivity functions require Julia; integration tests would
# need a Julia installation with Omniscape.jl and Circuitscape.jl.
# These are skipped in unit testing.

test_that("run_omniscape errors without Julia", {
  skip_on_cran()
  r <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- runif(25)
  expect_error(run_omniscape(r))
})

test_that("run_cumulative_current function exists", {
  expect_true(is.function(run_cumulative_current))
})

test_that("run_cumulative_current accepts backend parameter", {
  # Verify the backend parameter is part of the function signature
  args <- formals(run_cumulative_current)
  expect_true("backend" %in% names(args))
  expect_equal(eval(args$backend), c("julia", "jax"))
})

test_that("run_cumulative_current accepts parameterization parameter", {
  args <- formals(run_cumulative_current)
  expect_true("parameterization" %in% names(args))
  expect_equal(args$parameterization, "resistance")
})

test_that("run_cumulative_current errors without Julia (default backend)", {
  skip_on_cran()
  r <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- runif(25)
  expect_error(run_cumulative_current(r),
               "Julia not initialised")
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
  result <- run_cumulative_current(r, radius = 5L, block_size = 3L,
                                   backend = "jax")
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
                                   output = "both", backend = "jax")
  expect_type(result, "list")
  expect_s4_class(result$cum_current, "SpatRaster")
  expect_s4_class(result$flow_potential, "SpatRaster")
})

test_that("run_cumulative_current rejects invalid backend", {
  r <- terra::rast(nrows = 5, ncols = 5, vals = runif(25))
  expect_error(run_cumulative_current(r, backend = "torch"))
})
