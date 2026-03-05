# Tests for R/julia_bridge.R

test_that("ds_julia_check returns FALSE before setup", {
  # Should return FALSE if Julia hasn't been initialised
  expect_true(is.function(ds_julia_check))
})


test_that("ds_julia_call errors before setup", {
  expect_error(ds_julia_call("nonexistent"),
               "Julia not initialised|Julia has not been initialised")
})


test_that("ds_julia_setup function exists", {
  expect_true(is.function(ds_julia_setup))
})
