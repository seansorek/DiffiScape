# Tests for R/torch_pipeline.R
# Covers default_torch_config() (no external deps). The shared
# .prepare_backend_inputs() helper (formerly .prepare_torch_inputs(), now in
# R/utils.R) is exercised here via mocks on the Bayesian sampler entry points,
# and directly (with real terra/numpy) in test-utils.R.

# ---- default_torch_config --------------------------------------------------

test_that("default_torch_config returns a list", {
  cfg <- default_torch_config()
  expect_type(cfg, "list")
})

test_that("default_torch_config contains all documented top-level keys", {
  cfg      <- default_torch_config()
  required <- c(
    # Architecture
    "hidden_dim", "n_hidden_layers",
    # Optimiser
    "lr", "weight_decay", "n_epochs", "patience", "grad_clip", "warmup_epochs",
    # Solver
    "solver", "radius", "block_size", "focal_fraction", "absorption",
    "cg_tol", "source_spacing", "source_from_resistance",
    # Regularisation
    "reg_mean", "reg_var", "target_logR_var", "reg_skip", "log_R_baseline",
    # Misc
    "seed", "device",
    # Conv
    "use_conv", "n_conv_layers", "conv_channels", "conv_kernel_size",
    "dropout", "use_dilated", "intensity_hidden",
    # Spline-GAM
    "model_type", "n_knots", "spline_degree", "include_interactions",
    "penalty_scale", "lambda_init_marginal", "lambda_init_interaction",
    "lambda_min",
    # Intensity spline
    "intensity_spline", "intensity_n_knots", "intensity_degree",
    "lambda_init_intensity", "intensity_log1p_max"
  )
  missing_keys <- setdiff(required, names(cfg))
  expect_length(missing_keys, 0L)
})

test_that("default_torch_config defaults are within valid ranges", {
  cfg <- default_torch_config()
  expect_true(cfg$lr > 0)
  expect_true(cfg$patience < cfg$n_epochs)
  expect_true(cfg$block_size > 0)
  expect_true(cfg$focal_fraction > 0 && cfg$focal_fraction <= 1)
  expect_true(cfg$hidden_dim > 0)
  expect_true(cfg$n_hidden_layers >= 1)
  expect_true(cfg$radius > 0)
  expect_true(cfg$grad_clip > 0)
  expect_true(cfg$warmup_epochs >= 0)
  expect_true(cfg$n_knots > 0)
  expect_true(cfg$spline_degree >= 1)
})

test_that("default_torch_config defaults to global absorption", {
  cfg <- default_torch_config()
  expect_equal(cfg$solver, "global_absorption")
})

test_that("default_torch_config immutability: calls return independent lists", {
  cfg1 <- default_torch_config()
  cfg2 <- default_torch_config()
  cfg1$lr <- 999
  expect_equal(cfg2$lr, 0.01)
})

test_that("default_torch_config device default is auto", {
  cfg <- default_torch_config()
  expect_equal(cfg$device, "auto")
})

test_that("default_torch_config model_type default is loglinear", {
  cfg <- default_torch_config()
  expect_equal(cfg$model_type, "loglinear")
})

test_that("default_torch_config use_conv default is FALSE", {
  cfg <- default_torch_config()
  expect_false(cfg$use_conv)
})

test_that("default_torch_config intensity_spline default is FALSE", {
  cfg <- default_torch_config()
  expect_false(cfg$intensity_spline)
  expect_identical(cfg$intensity_hidden, 0L)
})








test_that("legacy pipeline translates physics and model settings without Torch", {
  passed <- NULL
  local_mocked_bindings(run_jax_pipeline = function(...) {
    passed <<- list(...)
    list(method = "jax_loglinear_circuit")
  }, .package = "DiffiScape")
  result <- run_torch_pipeline(NULL, NULL, solver = "global", n_epochs = 7L,
    output_dir = withr::local_tempdir())
  expect_equal(passed$formulation, "boundary")
  expect_equal(passed$model_type, "loglinear")
  expect_equal(passed$n_epochs, 7L)
  expect_match(result$method, "jax")
  expect_error(run_torch_pipeline(NULL, NULL, compute_uq = TRUE), "ds_posterior")
  expect_error(run_torch_pipeline(NULL, NULL, absorption_schedule = c(.1, .01)), "fixed absorption")
})

test_that("retired checkpoint samplers explain how to refit", {
  expect_error(run_bayesian_sampling(NULL, NULL, "old"), "ds_posterior")
  expect_error(run_bayesian_sampling_hmc(NULL, NULL, "old"), "refit|Refit")
  expect_error(run_advi(NULL, NULL, "old"), "refit|Refit")
})
