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
