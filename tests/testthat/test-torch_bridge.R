# Tests for R/torch_bridge.R

test_that("ds_torch_check function exists and returns logical", {
  expect_true(is.function(ds_torch_check))
  expect_true(is.logical(ds_torch_check()))
})


test_that("ds_torch_setup function exists", {
  expect_true(is.function(ds_torch_setup))
})


test_that("ds_install_torch_deps function exists", {
  expect_true(is.function(ds_install_torch_deps))
})


test_that("ds_torch_call errors before setup", {
  # Skip if reticulate happens to have already initialised the backend
  # earlier in this test session.
  skip_if(isTRUE(ds_torch_check()))
  expect_error(ds_torch_call("nonexistent"),
               "PyTorch backend not initialised")
})


test_that("Python backend is vendored under inst/python/diff_cs/", {
  expected <- c("03_circuit_solver.py",
                "04_diff_omniscape.py",
                "05_torch_pipeline.py",
                "requirements.txt")
  for (f in expected) {
    p <- system.file("python", "diff_cs", f, package = "DiffiScape")
    expect_true(nzchar(p),
                info = sprintf("missing inst/python/diff_cs/%s", f))
    expect_true(file.exists(p),
                info = sprintf("missing inst/python/diff_cs/%s", f))
  }
})


test_that("ds_torch_setup loads the bundled module when reticulate + torch are available", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("torch"),
              "torch not installed in active Python env")
  skip_if_not(reticulate::py_module_available("pyamg"),
              "pyamg not installed in active Python env")

  ok <- ds_torch_setup(force = TRUE)
  expect_true(ok)
  expect_true(ds_torch_check())
})
