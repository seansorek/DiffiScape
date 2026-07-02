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
    "lambda_init_intensity", "intensity_log1p_max",
    # IRL (value-shaped) resistance
    "beta", "gamma_d", "n_value_iter", "value_scale_init"
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

test_that("default_torch_config default solver is diff_omniscape", {
  cfg <- default_torch_config()
  expect_equal(cfg$solver, "diff_omniscape")
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

test_that("default_torch_config model_type default is mlp", {
  cfg <- default_torch_config()
  expect_equal(cfg$model_type, "mlp")
})

test_that("default_torch_config use_conv default is FALSE", {
  cfg <- default_torch_config()
  expect_false(cfg$use_conv)
})

test_that("default_torch_config intensity_spline default is FALSE", {
  cfg <- default_torch_config()
  expect_false(cfg$intensity_spline)
})


# ---- IRL (value-shaped) resistance config ----------------------------------

test_that("default_torch_config IRL knobs have valid defaults", {
  cfg <- default_torch_config()
  expect_true(cfg$beta > 0)              # soft value-iteration temperature
  expect_true(cfg$gamma_d > 0 && cfg$gamma_d < 1)  # contraction requires < 1
  expect_true(cfg$n_value_iter >= 1)
  expect_true(cfg$value_scale_init > 0)
})

test_that("ds_optimize exposes the 'irl' solver option", {
  choices <- eval(formals(ds_optimize)$solver)
  expect_true("irl" %in% choices)
})

test_that("run_torch_pipeline accepts IRL arguments", {
  args <- names(formals(run_torch_pipeline))
  expect_true(all(c("beta", "gamma_d", "n_value_iter", "value_scale_init")
                  %in% args))
  expect_true("model_type" %in% args)
})

test_that("irl_resistance_model errors without a resistance_raster", {
  skip_if_not_installed("terra")
  r <- terra::rast(nrows = 4, ncols = 4, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- runif(16)
  basis <- create_basis_stack(list(a = r), rescale = FALSE)
  expect_error(
    irl_resistance_model(list(loglik = -1), basis),
    "resistance_raster"
  )
})


# ---- .prepare_backend_inputs (formerly .prepare_torch_inputs) ---------------
# .prepare_torch_inputs() was collapsed into the shared .prepare_backend_inputs()
# helper in R/utils.R; see test-utils.R for the structure/cell_area/grid-dims/
# parity tests previously duplicated here.

test_that("run_torch_pipeline uses the shared .prepare_backend_inputs helper", {
  expect_true(is.function(DiffiScape:::.prepare_backend_inputs))
  expect_true(".prepare_backend_inputs" %in%
              all.names(body(run_torch_pipeline)))
})


# ---- .save_mcmc_artifacts ----------------------------------------------------

test_that(".save_mcmc_artifacts saves RDS and summary CSV", {
  tmp <- withr::local_tempdir()
  results <- list(
    samples = matrix(rnorm(6), nrow = 3),
    summary = list(
      r_0 = list(mean = 1.0, sd = 0.1, q025 = 0.8, q50 = 1.0, q975 = 1.2,
                 ess = 500),
      z_1 = list(mean = 0.5, sd = 0.2, q025 = 0.1, q50 = 0.5, q975 = 0.9,
                 ess = 400)
    )
  )
  expect_message(
    DiffiScape:::.save_mcmc_artifacts(results, tmp),
    "mcmc_results\\.rds"
  )
  expect_true(file.exists(file.path(tmp, "mcmc_results.rds")))
  expect_true(file.exists(file.path(tmp, "posterior_summary.csv")))

  csv <- read.csv(file.path(tmp, "posterior_summary.csv"))
  expect_equal(nrow(csv), 2)
  expect_true(all(c("parameter", "mean", "sd", "q025", "median", "q975", "ess")
                  %in% names(csv)))
})


test_that(".save_mcmc_artifacts skips CSV when summary is NULL", {
  tmp <- withr::local_tempdir()
  results <- list(samples = matrix(rnorm(6), nrow = 3), summary = NULL)
  expect_message(
    DiffiScape:::.save_mcmc_artifacts(results, tmp),
    "mcmc_results\\.rds"
  )
  expect_true(file.exists(file.path(tmp, "mcmc_results.rds")))
  expect_false(file.exists(file.path(tmp, "posterior_summary.csv")))
})


# ---- Bayesian sampler error paths --------------------------------------------

test_that("run_bayesian_sampling errors when checkpoint is missing", {
  skip_on_cran()
  skip_if_not_installed("reticulate")

  tmp <- withr::local_tempdir()
  basis <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                       ymin = 0, ymax = 1)
  terra::values(basis) <- runif(25)
  obs <- data.frame(x = 0.5, y = 0.5)

  local_mocked_bindings(
    ds_torch_setup = function(...) invisible(TRUE),
    .package = "DiffiScape"
  )

  expect_error(
    run_bayesian_sampling(basis, obs, model_dir = tmp),
    "Model checkpoint not found"
  )
})


test_that("run_bayesian_sampling_hmc errors when checkpoint is missing", {
  skip_on_cran()
  skip_if_not_installed("reticulate")

  tmp <- withr::local_tempdir()
  basis <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                       ymin = 0, ymax = 1)
  terra::values(basis) <- runif(25)
  obs <- data.frame(x = 0.5, y = 0.5)

  local_mocked_bindings(
    ds_torch_setup = function(...) invisible(TRUE),
    .package = "DiffiScape"
  )

  expect_error(
    run_bayesian_sampling_hmc(basis, obs, model_dir = tmp),
    "Model checkpoint not found"
  )
})


test_that("run_advi errors when checkpoint is missing", {
  skip_on_cran()
  skip_if_not_installed("reticulate")

  tmp <- withr::local_tempdir()
  basis <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                       ymin = 0, ymax = 1)
  terra::values(basis) <- runif(25)
  obs <- data.frame(x = 0.5, y = 0.5)

  local_mocked_bindings(
    ds_torch_setup = function(...) invisible(TRUE),
    .package = "DiffiScape"
  )

  expect_error(
    run_advi(basis, obs, model_dir = tmp),
    "Model checkpoint not found"
  )
})


# ---- Bayesian sampler mocked integration tests ------------------------------

test_that("run_bayesian_sampling dispatches to run_langevin_sampling", {
  skip_on_cran()
  skip_if_not_installed("reticulate")

  tmp <- withr::local_tempdir()
  file.create(file.path(tmp, "resistance_nn.pt"))
  basis <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                       ymin = 0, ymax = 1)
  terra::values(basis) <- runif(25)
  obs <- data.frame(x = 0.5, y = 0.5)

  captured_fn <- NULL
  captured_output_dir <- NULL
  mock_result <- list(summary = NULL, elapsed_time = 0.1)

  local_mocked_bindings(
    ds_torch_setup = function(...) invisible(TRUE),
    .prepare_backend_inputs = function(...) {
      list(basis_np = matrix(0, 1, 1), obs_np = matrix(0, 1, 1),
           vmask_np = TRUE, n_rows = 5L, n_cols = 5L, n_valid = 25L,
           n_obs = 1L, cell_area = 0.04)
    },
    ds_torch_call = function(fn_name, ...) {
      captured_fn <<- fn_name
      mock_result
    },
    .save_mcmc_artifacts = function(results_r, output_dir) {
      captured_output_dir <<- output_dir
      invisible(NULL)
    },
    .package = "DiffiScape"
  )

  run_bayesian_sampling(basis, obs, model_dir = tmp, output_dir = NULL,
                        verbose = FALSE)
  expect_equal(captured_fn, "run_langevin_sampling")
  expect_equal(captured_output_dir, tmp)
})


test_that("run_bayesian_sampling_hmc dispatches to run_hmc_sampling", {
  skip_on_cran()
  skip_if_not_installed("reticulate")

  tmp <- withr::local_tempdir()
  file.create(file.path(tmp, "resistance_nn.pt"))
  basis <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                       ymin = 0, ymax = 1)
  terra::values(basis) <- runif(25)
  obs <- data.frame(x = 0.5, y = 0.5)

  captured_fn <- NULL
  mock_result <- list(summary = NULL, elapsed_time = 0.1, n_divergences = 0L)

  local_mocked_bindings(
    ds_torch_setup = function(...) invisible(TRUE),
    .prepare_backend_inputs = function(...) {
      list(basis_np = matrix(0, 1, 1), obs_np = matrix(0, 1, 1),
           vmask_np = TRUE, n_rows = 5L, n_cols = 5L, n_valid = 25L,
           n_obs = 1L, cell_area = 0.04)
    },
    ds_torch_call = function(fn_name, ...) {
      captured_fn <<- fn_name
      mock_result
    },
    .save_mcmc_artifacts = function(...) invisible(NULL),
    .package = "DiffiScape"
  )

  run_bayesian_sampling_hmc(basis, obs, model_dir = tmp, verbose = FALSE)
  expect_equal(captured_fn, "run_hmc_sampling")
})


test_that("run_advi dispatches to run_advi Python function", {
  skip_on_cran()
  skip_if_not_installed("reticulate")

  tmp <- withr::local_tempdir()
  file.create(file.path(tmp, "resistance_nn.pt"))
  basis <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                       ymin = 0, ymax = 1)
  terra::values(basis) <- runif(25)
  obs <- data.frame(x = 0.5, y = 0.5)

  captured_fn <- NULL
  mock_result <- list(summary = NULL, elapsed_time = 0.1,
                      converged = TRUE, best_elbo = -50.0)

  local_mocked_bindings(
    ds_torch_setup = function(...) invisible(TRUE),
    .prepare_backend_inputs = function(...) {
      list(basis_np = matrix(0, 1, 1), obs_np = matrix(0, 1, 1),
           vmask_np = TRUE, n_rows = 5L, n_cols = 5L, n_valid = 25L,
           n_obs = 1L, cell_area = 0.04)
    },
    ds_torch_call = function(fn_name, ...) {
      captured_fn <<- fn_name
      mock_result
    },
    .save_mcmc_artifacts = function(...) invisible(NULL),
    .package = "DiffiScape"
  )

  run_advi(basis, obs, model_dir = tmp, verbose = FALSE)
  expect_equal(captured_fn, "run_advi")
})


test_that("run_bayesian_sampling calls ds_torch_setup when not initialized", {
  skip_on_cran()
  skip_if_not_installed("reticulate")

  tmp <- withr::local_tempdir()
  file.create(file.path(tmp, "resistance_nn.pt"))
  basis <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                       ymin = 0, ymax = 1)
  terra::values(basis) <- runif(25)
  obs <- data.frame(x = 0.5, y = 0.5)

  setup_called <- FALSE
  mock_result <- list(summary = NULL, elapsed_time = 0.1)

  local_mocked_bindings(
    ds_torch_setup = function(...) { setup_called <<- TRUE; invisible(TRUE) },
    .prepare_backend_inputs = function(...) {
      list(basis_np = matrix(0, 1, 1), obs_np = matrix(0, 1, 1),
           vmask_np = TRUE, n_rows = 5L, n_cols = 5L, n_valid = 25L,
           n_obs = 1L, cell_area = 0.04)
    },
    ds_torch_call = function(...) mock_result,
    .save_mcmc_artifacts = function(...) invisible(NULL),
    .package = "DiffiScape"
  )

  run_bayesian_sampling(basis, obs, model_dir = tmp, verbose = FALSE)
  expect_true(setup_called)
})


test_that("run_bayesian_sampling skips ds_torch_setup when already initialized", {
  skip_on_cran()
  skip_if_not_installed("reticulate")

  tmp <- withr::local_tempdir()
  file.create(file.path(tmp, "resistance_nn.pt"))
  basis <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                       ymin = 0, ymax = 1)
  terra::values(basis) <- runif(25)
  obs <- data.frame(x = 0.5, y = 0.5)

  ds_env  <- get(".ds_env", envir = environment(run_bayesian_sampling))
  old_val <- ds_env$torch_initialized
  withr::defer(ds_env$torch_initialized <- old_val)
  ds_env$torch_initialized <- TRUE

  setup_called <- FALSE
  mock_result <- list(summary = NULL, elapsed_time = 0.1)

  local_mocked_bindings(
    ds_torch_setup = function(...) { setup_called <<- TRUE },
    .prepare_backend_inputs = function(...) {
      list(basis_np = matrix(0, 1, 1), obs_np = matrix(0, 1, 1),
           vmask_np = TRUE, n_rows = 5L, n_cols = 5L, n_valid = 25L,
           n_obs = 1L, cell_area = 0.04)
    },
    ds_torch_call = function(...) mock_result,
    .save_mcmc_artifacts = function(...) invisible(NULL),
    .package = "DiffiScape"
  )

  run_bayesian_sampling(basis, obs, model_dir = tmp, verbose = FALSE)
  expect_false(setup_called)
})


test_that("run_bayesian_sampling forwards custom output_dir", {
  skip_on_cran()
  skip_if_not_installed("reticulate")

  model_tmp <- withr::local_tempdir()
  out_tmp   <- withr::local_tempdir()
  file.create(file.path(model_tmp, "resistance_nn.pt"))
  basis <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                       ymin = 0, ymax = 1)
  terra::values(basis) <- runif(25)
  obs <- data.frame(x = 0.5, y = 0.5)

  captured_output_dir <- NULL
  mock_result <- list(summary = NULL, elapsed_time = 0.1)

  local_mocked_bindings(
    ds_torch_setup = function(...) invisible(TRUE),
    .prepare_backend_inputs = function(...) {
      list(basis_np = matrix(0, 1, 1), obs_np = matrix(0, 1, 1),
           vmask_np = TRUE, n_rows = 5L, n_cols = 5L, n_valid = 25L,
           n_obs = 1L, cell_area = 0.04)
    },
    ds_torch_call = function(...) mock_result,
    .save_mcmc_artifacts = function(results_r, output_dir) {
      captured_output_dir <<- output_dir
      invisible(NULL)
    },
    .package = "DiffiScape"
  )

  run_bayesian_sampling(basis, obs, model_dir = model_tmp,
                        output_dir = out_tmp, verbose = FALSE)
  expect_equal(captured_output_dir, out_tmp)
})


# ---- Bayesian sampler diagnostic warnings ------------------------------------

test_that("run_bayesian_sampling_hmc warns on divergent transitions", {
  skip_on_cran()
  skip_if_not_installed("reticulate")

  tmp <- withr::local_tempdir()
  file.create(file.path(tmp, "resistance_nn.pt"))
  basis <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                       ymin = 0, ymax = 1)
  terra::values(basis) <- runif(25)
  obs <- data.frame(x = 0.5, y = 0.5)

  local_mocked_bindings(
    ds_torch_setup = function(...) invisible(TRUE),
    .prepare_backend_inputs = function(...) {
      list(basis_np = matrix(0, 1, 1), obs_np = matrix(0, 1, 1),
           vmask_np = TRUE, n_rows = 5L, n_cols = 5L, n_valid = 25L,
           n_obs = 1L, cell_area = 0.04)
    },
    ds_torch_call = function(...) {
      list(summary = NULL, elapsed_time = 0.1, n_divergences = 5L)
    },
    .save_mcmc_artifacts = function(...) invisible(NULL),
    .package = "DiffiScape"
  )

  expect_warning(
    run_bayesian_sampling_hmc(basis, obs, model_dir = tmp, verbose = FALSE),
    "divergent transitions"
  )
})


test_that("run_bayesian_sampling_hmc does not warn with 0 divergences", {
  skip_on_cran()
  skip_if_not_installed("reticulate")

  tmp <- withr::local_tempdir()
  file.create(file.path(tmp, "resistance_nn.pt"))
  basis <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                       ymin = 0, ymax = 1)
  terra::values(basis) <- runif(25)
  obs <- data.frame(x = 0.5, y = 0.5)

  local_mocked_bindings(
    ds_torch_setup = function(...) invisible(TRUE),
    .prepare_backend_inputs = function(...) {
      list(basis_np = matrix(0, 1, 1), obs_np = matrix(0, 1, 1),
           vmask_np = TRUE, n_rows = 5L, n_cols = 5L, n_valid = 25L,
           n_obs = 1L, cell_area = 0.04)
    },
    ds_torch_call = function(...) {
      list(summary = NULL, elapsed_time = 0.1, n_divergences = 0L)
    },
    .save_mcmc_artifacts = function(...) invisible(NULL),
    .package = "DiffiScape"
  )

  expect_no_warning(
    run_bayesian_sampling_hmc(basis, obs, model_dir = tmp, verbose = FALSE)
  )
})


test_that("run_advi warns when not converged", {
  skip_on_cran()
  skip_if_not_installed("reticulate")

  tmp <- withr::local_tempdir()
  file.create(file.path(tmp, "resistance_nn.pt"))
  basis <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                       ymin = 0, ymax = 1)
  terra::values(basis) <- runif(25)
  obs <- data.frame(x = 0.5, y = 0.5)

  local_mocked_bindings(
    ds_torch_setup = function(...) invisible(TRUE),
    .prepare_backend_inputs = function(...) {
      list(basis_np = matrix(0, 1, 1), obs_np = matrix(0, 1, 1),
           vmask_np = TRUE, n_rows = 5L, n_cols = 5L, n_valid = 25L,
           n_obs = 1L, cell_area = 0.04)
    },
    ds_torch_call = function(...) {
      list(summary = NULL, elapsed_time = 0.1, converged = FALSE,
           best_elbo = -100.0)
    },
    .save_mcmc_artifacts = function(...) invisible(NULL),
    .package = "DiffiScape"
  )

  expect_warning(
    run_advi(basis, obs, model_dir = tmp, verbose = FALSE),
    "ADVI did not converge"
  )
})


test_that("run_advi does not warn when converged", {
  skip_on_cran()
  skip_if_not_installed("reticulate")

  tmp <- withr::local_tempdir()
  file.create(file.path(tmp, "resistance_nn.pt"))
  basis <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                       ymin = 0, ymax = 1)
  terra::values(basis) <- runif(25)
  obs <- data.frame(x = 0.5, y = 0.5)

  local_mocked_bindings(
    ds_torch_setup = function(...) invisible(TRUE),
    .prepare_backend_inputs = function(...) {
      list(basis_np = matrix(0, 1, 1), obs_np = matrix(0, 1, 1),
           vmask_np = TRUE, n_rows = 5L, n_cols = 5L, n_valid = 25L,
           n_obs = 1L, cell_area = 0.04)
    },
    ds_torch_call = function(...) {
      list(summary = NULL, elapsed_time = 0.1, converged = TRUE,
           best_elbo = -50.0)
    },
    .save_mcmc_artifacts = function(...) invisible(NULL),
    .package = "DiffiScape"
  )

  expect_no_warning(
    run_advi(basis, obs, model_dir = tmp, verbose = FALSE)
  )
})


# ---- verify_irl_gradient -----------------------------------------------------

test_that("verify_irl_gradient messages on pass", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if_not_installed("terra")
  skip_if_not(reticulate::py_module_available("numpy"),
              "numpy not available")

  r <- terra::rast(nrows = 25, ncols = 25, xmin = 0, xmax = 25,
                   ymin = 0, ymax = 25, nlyrs = 2)
  terra::values(r) <- runif(25 * 25 * 2)

  local_mocked_bindings(
    ds_torch_setup = function(...) invisible(TRUE),
    ds_torch_call = function(...) list(pass = TRUE, max_rel_error = 1e-5),
    .package = "DiffiScape"
  )

  expect_message(
    result <- verify_irl_gradient(r, crop_size = 10L),
    "PASSED"
  )
  expect_true(result$pass)
})


test_that("verify_irl_gradient warns on failure", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if_not_installed("terra")
  skip_if_not(reticulate::py_module_available("numpy"),
              "numpy not available")

  r <- terra::rast(nrows = 25, ncols = 25, xmin = 0, xmax = 25,
                   ymin = 0, ymax = 25, nlyrs = 2)
  terra::values(r) <- runif(25 * 25 * 2)

  local_mocked_bindings(
    ds_torch_setup = function(...) invisible(TRUE),
    ds_torch_call = function(...) list(pass = FALSE, max_rel_error = 0.5),
    .package = "DiffiScape"
  )

  expect_warning(
    result <- verify_irl_gradient(r, crop_size = 10L),
    "FAILED"
  )
  expect_false(result$pass)
})
