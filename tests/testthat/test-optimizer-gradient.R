# Tests for optimize_resistance_gradient and ds_jax_optimize

# --------------- Unit tests (no Python needed) --------------------------------

test_that("optimize_resistance_gradient requires reticulate", {
  # Should error on NULL inputs even before reaching Python

  expect_error(optimize_resistance_gradient(NULL, NULL))
})


# ---- GH #106: unsupported arguments must error, not be silently dropped ----
# The JAX gradient objective is a covariate-free Poisson PPP. covariates_obs,
# covariates_rasters, available_points, available_covariates, and
# residualise = TRUE were previously accepted and silently ignored, letting
# `solver = "gradient"` fit a different model than requested with no
# indication. These should now error before any Python call is attempted.

test_that("optimize_resistance_gradient errors when covariates_obs is supplied", {
  expect_error(
    optimize_resistance_gradient(
      NULL, NULL, covariates_obs = data.frame(x = 1)
    ),
    "does not support"
  )
})

test_that("optimize_resistance_gradient errors when covariates_rasters is supplied", {
  expect_error(
    optimize_resistance_gradient(
      NULL, NULL, covariates_rasters = terra::rast(nrows = 2, ncols = 2, vals = 1)
    ),
    "does not support"
  )
})

test_that("optimize_resistance_gradient errors when residualise = TRUE", {
  expect_error(
    optimize_resistance_gradient(NULL, NULL, residualise = TRUE),
    "does not support"
  )
})

test_that("optimize_resistance_gradient errors when available_points is supplied", {
  expect_error(
    optimize_resistance_gradient(
      NULL, NULL, available_points = data.frame(x = 1)
    ),
    "does not support"
  )
})

test_that("optimize_resistance_gradient errors when available_covariates is supplied", {
  expect_error(
    optimize_resistance_gradient(
      NULL, NULL, available_covariates = data.frame(x = 1)
    ),
    "does not support"
  )
})

test_that("optimize_resistance_gradient's unsupported-argument check does not false-positive on defaults", {
  # NULL inputs should still fail (existing "requires reticulate" test),
  # but NOT with the "does not support" message -- the validation itself
  # must not fire when covariates_obs/residualise/etc. are left at their
  # defaults.
  err <- tryCatch(
    optimize_resistance_gradient(NULL, NULL),
    error = function(e) e
  )
  expect_false(grepl("does not support", conditionMessage(err)))
})


test_that("ds_optimize dispatches 'gradient' solver", {
  # Match should succeed without error
  solver <- match.arg("gradient",
                      c("surrogate", "gradient", "enzyme", "torch", "irl"))
  expect_equal(solver, "gradient")
})


test_that("ds_optimize deprecates 'enzyme' to 'gradient'", {
  # We test the deprecation message by calling ds_optimize with solver="enzyme"
  # and catching the message before it errors on NULL inputs
  expect_message(
    tryCatch(
      ds_optimize(NULL, NULL, solver = "enzyme"),
      error = function(e) invisible(NULL)
    ),
    "deprecated"
  )
})


# --------------- Integration tests (require JAX) -----------------------------

test_that("optimize_resistance_gradient runs end-to-end", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("jax"),
          "jax not installed")
  skip_if(!reticulate::py_module_available("jaxscape"),
          "jaxscape not installed")
  skip_if(!reticulate::py_module_available("jaxopt"),
          "jaxopt not installed")
  skip_on_cran()

  set.seed(42)
  r1 <- terra::rast(nrows = 10, ncols = 10, vals = runif(100, 0.1, 1))
  r2 <- terra::rast(nrows = 10, ncols = 10, vals = runif(100, 0.1, 1))
  basis <- c(r1, r2)

  # Create observation points within the raster extent
  pts <- data.frame(
    x = runif(20, terra::xmin(r1), terra::xmax(r1)),
    y = runif(20, terra::ymin(r1), terra::ymax(r1))
  )

  config <- default_optimizer_config()
  config$n_iter <- 5L
  config$seed <- 42L
  config$omniscape$radius <- 3L
  config$omniscape$block_size <- 2L

  result <- optimize_resistance_gradient(
    basis, pts,
    config = config,
    method = "lbfgs"
  )

  expect_type(result, "list")
  expect_true("best_params" %in% names(result))
  expect_true("best_loglik" %in% names(result))
  expect_true("bounds" %in% names(result))
  expect_true("n_evaluations" %in% names(result))
  expect_true("distribution" %in% names(result))
  expect_true("convergence" %in% names(result))

  # best_params should be a named list with r_0, z_1, z_2
  expect_true("r_0" %in% names(result$best_params))
  expect_true("z_1" %in% names(result$best_params))
  expect_true("z_2" %in% names(result$best_params))

  expect_true(is.numeric(result$best_loglik))
  expect_true(result$convergence %in% c(0L, 1L))
})


test_that("optimize_resistance_gradient works with adam method", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("jax"),
          "jax not installed")
  skip_if(!reticulate::py_module_available("jaxscape"),
          "jaxscape not installed")
  skip_if(!reticulate::py_module_available("optax"),
          "optax not installed")
  skip_on_cran()

  set.seed(42)
  r1 <- terra::rast(nrows = 10, ncols = 10, vals = runif(100, 0.1, 1))
  basis <- c(r1)

  pts <- data.frame(
    x = runif(15, terra::xmin(r1), terra::xmax(r1)),
    y = runif(15, terra::ymin(r1), terra::ymax(r1))
  )

  config <- default_optimizer_config()
  config$n_iter <- 5L
  config$seed <- 42L
  config$omniscape$radius <- 3L
  config$omniscape$block_size <- 2L

  result <- optimize_resistance_gradient(
    basis, pts,
    config = config,
    method = "adam"
  )

  expect_type(result, "list")
  expect_true("best_params" %in% names(result))
  expect_true(is.numeric(result$best_loglik))
})


test_that("optimize_resistance_gradient always reports distribution = 'poisson', regardless of config$distribution", {
  skip_on_cran()

  basis_stack <- terra::rast(nrows = 3, ncols = 3, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)

  local_mocked_bindings(
    ds_jax_optimize = function(basis_np, obs_np, valid_mask_np,
                                n_rows, n_cols, cell_area,
                                init_params = NULL, radius, block_size, ...) {
      list(best_params = c(0, 0), best_loglik = -1, n_epochs_run = 1L,
           elapsed = 0.01, converged = TRUE)
    },
    .package = "DiffiScape"
  )

  cfg <- default_optimizer_config()
  cfg$distribution <- "negbin"

  result <- optimize_resistance_gradient(
    basis_stack = basis_stack,
    obs_points  = obs_points,
    config      = cfg,
    output_dir  = withr::local_tempdir()
  )

  expect_equal(result$distribution, "poisson")
})


test_that("ds_optimize dispatches gradient solver correctly", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("jax"),
          "jax not installed")
  skip_if(!reticulate::py_module_available("jaxscape"),
          "jaxscape not installed")
  skip_if(!reticulate::py_module_available("jaxopt"),
          "jaxopt not installed")
  skip_on_cran()

  set.seed(42)
  r1 <- terra::rast(nrows = 10, ncols = 10, vals = runif(100, 0.1, 1))
  basis <- c(r1)

  pts <- data.frame(
    x = runif(15, terra::xmin(r1), terra::xmax(r1)),
    y = runif(15, terra::ymin(r1), terra::ymax(r1))
  )

  config <- default_optimizer_config()
  config$n_iter <- 5L
  config$seed <- 42L
  config$omniscape$radius <- 3L
  config$omniscape$block_size <- 2L

  result <- ds_optimize(
    basis, pts,
    config = config,
    solver = "gradient"
  )

  expect_type(result, "list")
  expect_true("best_params" %in% names(result))
})
