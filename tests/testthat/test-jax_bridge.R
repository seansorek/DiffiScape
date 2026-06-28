# Tests for R/jax_bridge.R

test_that("ds_jax_check function exists and returns logical", {
  expect_true(is.function(ds_jax_check))
  expect_true(is.logical(ds_jax_check()))
})


test_that("ds_jax_setup function exists", {
  expect_true(is.function(ds_jax_setup))
})


test_that("ds_install_jax_deps function exists", {
  expect_true(is.function(ds_install_jax_deps))
})


test_that("ds_jax_call errors before setup", {
  # Skip if reticulate happens to have already initialised the backend
  # earlier in this test session.
  skip_if(isTRUE(ds_jax_check()))
  expect_error(ds_jax_call("core", "forward_solve"),
               "JAX backend not initialised")
})


test_that("ds_jax_call rejects unknown module names", {
  skip_if(!isTRUE(ds_jax_check()))
  expect_error(ds_jax_call("nonexistent", "foo"),
               "Unknown module")
})


test_that("Python backend is vendored under inst/python/diffiscape_jax/", {
  expected <- c("__init__.py", "core.py", "window.py")
  for (f in expected) {
    p <- system.file("python", "diffiscape_jax", f, package = "DiffiScape")
    expect_true(nzchar(p),
                info = sprintf("missing inst/python/diffiscape_jax/%s", f))
    expect_true(file.exists(p),
                info = sprintf("missing inst/python/diffiscape_jax/%s", f))
  }
})


test_that("ds_jax_setup initialises without error", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("jaxscape"),
              "jaxscape not installed in active Python env")

  ok <- ds_jax_setup(force = TRUE)
  expect_true(ok)
  expect_true(ds_jax_check())
})


test_that("ds_jax_connectivity returns correct structure", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("jaxscape"),
              "jaxscape not installed in active Python env")

  ds_jax_setup()

  r <- terra::rast(nrows = 15, ncols = 15, vals = runif(225, 1, 100))
  result <- ds_jax_connectivity(r, radius = 5L, block_size = 3L)

  expect_type(result, "list")
  expect_named(result, c("cum_current", "flow_potential", "elapsed_seconds"))
  expect_s4_class(result$cum_current, "SpatRaster")
  expect_null(result$flow_potential)  # default output = "current"
  expect_true(result$elapsed_seconds > 0)

  # Spatial dimensions should match input
  expect_equal(terra::nrow(result$cum_current), 15L)
  expect_equal(terra::ncol(result$cum_current), 15L)
})


test_that("ds_jax_connectivity handles NA values in resistance", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("jaxscape"),
              "jaxscape not installed in active Python env")

  ds_jax_setup()

  vals <- runif(225, 1, 100)
  vals[c(1, 50, 100)] <- NA
  r <- terra::rast(nrows = 15, ncols = 15, vals = vals)
  result <- ds_jax_connectivity(r, radius = 5L, block_size = 3L)

  expect_type(result, "list")
  expect_s4_class(result$cum_current, "SpatRaster")
})


test_that("ds_jax_connectivity validates output argument before backend init", {
  # match.arg() on `output` runs before ds_jax_setup() inside
  # ds_jax_connectivity(), so an invalid value is rejected without needing a
  # working JAX backend. This keeps the validation under test in CI, where
  # jaxscape is not installed.
  r <- terra::rast(nrows = 5, ncols = 5, vals = runif(25, 1, 100))
  expect_error(ds_jax_connectivity(r, output = "invalid"))
})


test_that("JAX backend functions are exported and callable", {
  exported <- c("ds_jax_setup", "ds_jax_check", "ds_jax_call",
                "ds_jax_connectivity", "ds_jax_sample_nuts",
                "ds_jax_sample_advi", "ds_install_jax_deps")
  for (fn in exported) {
    expect_true(
      is.function(getExportedValue("DiffiScape", fn)),
      info = sprintf("%s should be an exported function", fn)
    )
  }
})
