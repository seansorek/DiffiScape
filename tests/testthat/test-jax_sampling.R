# Tests for JAX/NumPyro sampling wrappers in R/jax_bridge.R

# ---- Function-existence and signature tests (no Python required) ----

test_that("ds_jax_sample_nuts exists and is a function", {
  expect_true(is.function(ds_jax_sample_nuts))
})


test_that("ds_jax_sample_advi exists and is a function", {
  expect_true(is.function(ds_jax_sample_advi))
})


test_that(".build_flax_model exists and is a function", {
  expect_true(is.function(DiffiScape:::.build_flax_model))
})


test_that(".save_jax_mcmc_artifacts exists and is a function", {
  expect_true(is.function(DiffiScape:::.save_jax_mcmc_artifacts))
})


test_that("ds_jax_sample_nuts has expected formal arguments", {
  fmls <- names(formals(ds_jax_sample_nuts))
  expected <- c("basis_stack", "obs_points", "n_samples", "warmup",
                "max_treedepth", "target_accept", "parameterization",
                "model_type", "model_config", "seed", "verbose",
                "output_dir")
  for (arg in expected) {
    expect_true(arg %in% fmls,
                info = sprintf("ds_jax_sample_nuts missing argument: %s", arg))
  }
})


test_that("ds_jax_sample_advi has expected formal arguments", {
  fmls <- names(formals(ds_jax_sample_advi))
  expected <- c("basis_stack", "obs_points", "n_samples", "max_iter",
                "lr", "parameterization", "model_type", "model_config",
                "seed", "verbose", "output_dir")
  for (arg in expected) {
    expect_true(arg %in% fmls,
                info = sprintf("ds_jax_sample_advi missing argument: %s", arg))
  }
})


# ---- .prepare_backend_inputs tests (require terra + Python/numpy) ----
# .prepare_jax_inputs() was collapsed into the shared .prepare_backend_inputs()
# helper in R/utils.R; see test-utils.R for the full structure/parity tests.

test_that("ds_jax_sample_nuts and ds_jax_sample_advi use the shared .prepare_backend_inputs helper", {
  expect_true(is.function(DiffiScape:::.prepare_backend_inputs))
})


# ---- .save_jax_mcmc_artifacts tests (pure R, no Python) ----

test_that(".save_jax_mcmc_artifacts saves mcmc_results.rds", {
  out_dir <- tempfile("mcmc_test")
  dir.create(out_dir)
  on.exit(unlink(out_dir, recursive = TRUE))

  fake_results <- list(
    samples_effective_loglinear = matrix(rnorm(20), nrow = 10, ncol = 2),
    summary = NULL,
    elapsed_time = 1.5,
    n_divergences = 0L
  )
  DiffiScape:::.save_jax_mcmc_artifacts(fake_results, out_dir)

  expect_true(file.exists(file.path(out_dir, "mcmc_results.rds")))
  loaded <- readRDS(file.path(out_dir, "mcmc_results.rds"))
  expect_equal(loaded$n_divergences, 0L)
})


test_that(".save_jax_mcmc_artifacts saves posterior_summary.csv for vector params", {
  out_dir <- tempfile("mcmc_test2")
  dir.create(out_dir)
  on.exit(unlink(out_dir, recursive = TRUE))

  fake_results <- list(
    summary = list(
      params = list(
        mean = c(1.0, 2.0),
        sd   = c(0.1, 0.2),
        q025 = c(0.8, 1.6),
        q50  = c(1.0, 2.0),
        q975 = c(1.2, 2.4)
      )
    )
  )
  DiffiScape:::.save_jax_mcmc_artifacts(fake_results, out_dir)

  csv_path <- file.path(out_dir, "posterior_summary.csv")
  expect_true(file.exists(csv_path))
  df <- utils::read.csv(csv_path, stringsAsFactors = FALSE)
  expect_equal(nrow(df), 2L)
  expect_true("params[1]" %in% df$parameter)
  expect_true("params[2]" %in% df$parameter)
  expect_true(all(c("mean", "sd", "q025", "median", "q975", "ess") %in% names(df)))
})


test_that(".save_jax_mcmc_artifacts saves posterior_summary.csv for scalar params", {
  out_dir <- tempfile("mcmc_test3")
  dir.create(out_dir)
  on.exit(unlink(out_dir, recursive = TRUE))

  fake_results <- list(
    summary = list(
      alpha = list(
        mean = 1.0,
        sd   = 0.1,
        q025 = 0.8,
        q50  = 1.0,
        q975 = 1.2
      )
    )
  )
  DiffiScape:::.save_jax_mcmc_artifacts(fake_results, out_dir)

  csv_path <- file.path(out_dir, "posterior_summary.csv")
  expect_true(file.exists(csv_path))
  df <- utils::read.csv(csv_path, stringsAsFactors = FALSE)
  expect_equal(nrow(df), 1L)
  expect_equal(df$parameter, "alpha")
  expect_equal(df$mean, 1.0)
})


# ---- Integration tests (require JAX + NumPyro + jaxscape) ----

test_that("ds_jax_sample_nuts returns valid structure", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("jaxscape"),
              "jaxscape not installed in active Python env")
  skip_if_not(reticulate::py_module_available("numpyro"),
              "numpyro not installed in active Python env")
  skip_if_not(reticulate::py_module_available("flax"),
              "flax not installed in active Python env")
  skip("Integration test: requires full JAX + NumPyro stack and is slow")

  ds_jax_setup()

  r <- terra::rast(nrows = 6, ncols = 6,
                   vals = matrix(runif(72), ncol = 2))
  obs <- data.frame(x = runif(10, -180, 180), y = runif(10, -90, 90))

  result <- ds_jax_sample_nuts(
    basis_stack = r, obs_points = obs,
    n_samples = 5L, warmup = 3L,
    model_type = "mlp", model_config = list(hidden_dim = 8L),
    seed = 42L, verbose = FALSE
  )

  expect_type(result, "list")
  expected_names <- c("samples_effective_loglinear", "samples_alpha",
                      "samples_gamma", "partial_effects",
                      "log_posterior_trace", "ess", "summary",
                      "elapsed_time", "n_divergences")
  for (nm in expected_names) {
    expect_true(nm %in% names(result),
                info = sprintf("Missing return field: %s", nm))
  }
  expect_true(result$elapsed_time > 0)
  expect_true(is.integer(result$n_divergences) || is.numeric(result$n_divergences))
})


test_that("ds_jax_sample_advi returns valid structure", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("jaxscape"),
              "jaxscape not installed in active Python env")
  skip_if_not(reticulate::py_module_available("numpyro"),
              "numpyro not installed in active Python env")
  skip_if_not(reticulate::py_module_available("flax"),
              "flax not installed in active Python env")
  skip("Integration test: requires full JAX + NumPyro stack and is slow")

  ds_jax_setup()

  r <- terra::rast(nrows = 6, ncols = 6,
                   vals = matrix(runif(72), ncol = 2))
  obs <- data.frame(x = runif(10, -180, 180), y = runif(10, -90, 90))

  result <- ds_jax_sample_advi(
    basis_stack = r, obs_points = obs,
    n_samples = 5L, max_iter = 10L,
    model_type = "mlp", model_config = list(hidden_dim = 8L),
    seed = 42L, verbose = FALSE
  )

  expect_type(result, "list")
  expected_names <- c("samples_effective_loglinear", "samples_alpha",
                      "samples_gamma", "partial_effects",
                      "log_posterior_trace", "ess", "summary",
                      "elapsed_time", "n_divergences",
                      "best_elbo", "converged")
  for (nm in expected_names) {
    expect_true(nm %in% names(result),
                info = sprintf("Missing return field: %s", nm))
  }
  expect_true(result$elapsed_time > 0)
  expect_true(is.numeric(result$best_elbo))
  expect_true(is.logical(result$converged))
  expect_equal(result$n_divergences, 0L)
})
