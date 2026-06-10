# Tests for R/torch_pipeline.R
# Covers default_torch_config() (no external deps) and
# .prepare_torch_inputs() (skipped without reticulate).

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


# ---- .prepare_torch_inputs -------------------------------------------------

test_that(".prepare_torch_inputs drops obs outside valid cells with message", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("terra")

  # Create a small raster with an NA region
  r <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 5,
                   ymin = 0, ymax = 5)
  vals <- runif(25)
  vals[1] <- NA   # top-left cell is invalid
  terra::values(r) <- vals
  basis_stack <- c(r, r)

  # One point inside a valid cell, one that will map to NA cell
  obs_in_valid <- data.frame(x = 2.5, y = 2.5)
  obs_in_na    <- data.frame(x = 100, y = 100)  # outside extent
  obs_points   <- rbind(obs_in_valid, obs_in_na)

  expect_message(
    result <- DiffiScape:::.prepare_torch_inputs(basis_stack, obs_points),
    "Dropped"
  )
  # Only the valid obs contributes
  expect_equal(result$n_obs, 1L)
})

test_that(".prepare_torch_inputs computes cell_area correctly", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("terra")

  r <- terra::rast(nrows = 4, ncols = 4, xmin = 0, xmax = 2,
                   ymin = 0, ymax = 2)
  terra::values(r) <- runif(16)
  basis_stack <- c(r, r)

  obs_points <- data.frame(x = 1.0, y = 1.0)
  result     <- DiffiScape:::.prepare_torch_inputs(basis_stack, obs_points)

  # Cell resolution is 0.5 x 0.5, so area = 0.25
  expect_equal(result$cell_area, 0.25, tolerance = 1e-8)
})

test_that(".prepare_torch_inputs returns correct grid dimensions", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("terra")

  r <- terra::rast(nrows = 6, ncols = 8, xmin = 0, xmax = 8,
                   ymin = 0, ymax = 6)
  terra::values(r) <- runif(48)
  basis_stack <- c(r, r)

  obs_points <- data.frame(x = 4, y = 3)
  result     <- DiffiScape:::.prepare_torch_inputs(basis_stack, obs_points)

  expect_equal(result$n_rows, 6L)
  expect_equal(result$n_cols, 8L)
})
