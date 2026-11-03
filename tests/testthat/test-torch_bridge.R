test_that("legacy backend calls give migration messages without loading Torch", {
  expect_false(ds_torch_check())
  expect_error(ds_torch_call("old"), "run_jax_pipeline")
  expect_error(ds_install_torch_deps(gpu = TRUE), "CPU-only")
  local_mocked_bindings(ds_jax_setup = function(python, force) TRUE, .package = "DiffiScape")
  expect_true(ds_torch_setup())
})
