# Tests for R/utils.R

test_that("%||% works", {
  expect_equal(DiffiScape:::`%||%`(NULL, 5), 5)
  expect_equal(DiffiScape:::`%||%`(3, 5), 3)
  expect_equal(DiffiScape:::`%||%`("a", "b"), "a")
})


test_that("as_coord_matrix handles data.frame", {
  df <- data.frame(x = 1:3, y = 4:6)
  m  <- as_coord_matrix(df)
  expect_true(is.matrix(m))
  expect_equal(ncol(m), 2)
  expect_equal(nrow(m), 3)
  expect_equal(colnames(m), c("x", "y"))
})


test_that("as_coord_matrix handles matrix", {
  m_in <- matrix(c(1, 2, 3, 4, 5, 6), ncol = 2)
  m    <- as_coord_matrix(m_in)
  expect_true(is.matrix(m))
  expect_equal(nrow(m), 3)
})


test_that("as_coord_matrix errors on bad input", {
  expect_error(as_coord_matrix(list(1, 2, 3)))
})


test_that("compute_information_criteria returns correct components", {
  ic <- compute_information_criteria(loglik = -100, k = 3, n = 50)
  expect_named(ic, c("AIC", "BIC", "AICc"))
  expect_equal(ic$AIC, 2 * 3 - 2 * (-100))
  expect_true(ic$AICc > ic$AIC)  # correction increases AIC
})


test_that("validate_raster_alignment works", {
  r1 <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  r2 <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  expect_true(validate_raster_alignment(r1, r2))

  r3 <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  expect_false(validate_raster_alignment(r1, r3))
})


test_that("extract_raster_values works on SpatRaster", {
  r   <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 10,
                     ymin = 0, ymax = 10)
  terra::values(r) <- 1:100
  pts <- data.frame(x = c(0.5, 5.5), y = c(0.5, 5.5))
  vals <- extract_raster_values(r, pts)
  expect_length(vals, 2)
  expect_true(all(!is.na(vals)))
})


test_that("extract_raster_values returns NA for points outside extent", {
  r <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 5,
                   ymin = 0, ymax = 5)
  terra::values(r) <- 1:25
  # Point far outside the raster extent
  pts <- data.frame(x = c(2.5, 99.0), y = c(2.5, 99.0))
  vals <- extract_raster_values(r, pts)
  expect_length(vals, 2)
  expect_true(!is.na(vals[1]))
  expect_true(is.na(vals[2]))
})


test_that("extract_raster_values returns NA for points on NA cells", {
  r <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 5,
                   ymin = 0, ymax = 5)
  v <- 1:25
  v[1] <- NA
  terra::values(r) <- v
  # First cell (bottom-left corner) is NA
  pts <- data.frame(x = c(0.5, 2.5), y = c(4.5, 2.5))
  vals <- extract_raster_values(r, pts)
  expect_true(is.na(vals[1]))
  expect_true(!is.na(vals[2]))
})


test_that("extract_raster_values handles SpatVector input", {
  r <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 5,
                   ymin = 0, ymax = 5)
  terra::values(r) <- seq_len(25)
  pts_df <- data.frame(x = c(0.5, 2.5), y = c(4.5, 2.5))
  sv <- terra::vect(pts_df, geom = c("x", "y"), crs = terra::crs(r))
  vals <- extract_raster_values(r, sv)
  expect_length(vals, 2)
  expect_true(all(!is.na(vals)))
})


test_that("compute_information_criteria values match known formulas", {
  ic <- compute_information_criteria(loglik = -100, k = 3, n = 50)
  expect_equal(ic$AIC,  -2 * (-100) + 2 * 3,            tolerance = 1e-10)
  expect_equal(ic$BIC,  -2 * (-100) + log(50) * 3,      tolerance = 1e-10)
  # AICc = AIC + 2k(k+1)/(n-k-1)
  expect_equal(ic$AICc, ic$AIC + (2 * 3 * 4) / (50 - 3 - 1), tolerance = 1e-10)
})


test_that("compute_information_criteria is safe when n is very small", {
  # When n - k - 1 <= 0, denominator is clamped to 1
  ic <- compute_information_criteria(loglik = -50, k = 5, n = 5)
  expect_true(is.finite(ic$AICc))
})


test_that("validate_raster_alignment rejects different resolutions", {
  r1 <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  r_coarse <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                          ymin = 0, ymax = 1)
  expect_false(validate_raster_alignment(r1, r_coarse))
})


test_that("validate_raster_alignment rejects different extents", {
  r1 <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  r_shifted <- terra::rast(nrows = 10, ncols = 10, xmin = 5, xmax = 6,
                            ymin = 0, ymax = 1)
  expect_false(validate_raster_alignment(r1, r_shifted))
})


test_that("validate_raster_alignment accepts a single raster", {
  r <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  expect_true(validate_raster_alignment(r))
})


test_that("as_coord_matrix errors on NULL input", {
  expect_error(as_coord_matrix(NULL), "NULL")
})


test_that("as_coord_matrix errors on single-column data.frame", {
  expect_error(as_coord_matrix(data.frame(x = 1:3)))
})


test_that("as_coord_matrix errors on single-column matrix", {
  expect_error(as_coord_matrix(matrix(1:3, ncol = 1)))
})


# ---- .prepare_backend_inputs() (Problem A collapse) --------------------------
# .prepare_jax_inputs() (R/jax_bridge.R) and .prepare_torch_inputs()
# (R/torch_pipeline.R) were confirmed line-for-line identical and have been
# collapsed into this single shared helper in R/utils.R. These tests pin its
# behaviour against fixed input fixtures and mirror the structure/behaviour
# tests that used to live in test-jax_sampling.R / test-torch_pipeline.R for
# the two old dot-functions.

test_that(".prepare_backend_inputs exists and is a function", {
  expect_true(is.function(DiffiScape:::.prepare_backend_inputs))
})


test_that(".prepare_backend_inputs returns expected structure", {
  skip_on_cran()
  skip_if_not_installed("terra")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("numpy"),
              "numpy not installed in active Python env")

  r <- terra::rast(nrows = 5, ncols = 5, vals = runif(25))
  obs <- data.frame(x = 0, y = 0)
  prep <- DiffiScape:::.prepare_backend_inputs(r, obs)

  expect_type(prep, "list")
  expected_names <- c("basis_np", "obs_np", "vmask_np", "valid_mask",
                      "n_rows", "n_cols", "cell_area", "n_valid", "n_obs")
  for (nm in expected_names) {
    expect_true(nm %in% names(prep),
                info = sprintf(".prepare_backend_inputs missing field: %s", nm))
  }
  expect_equal(prep$n_rows, 5L)
  expect_equal(prep$n_cols, 5L)
  expect_equal(prep$n_valid, 25L)
})


test_that(".prepare_backend_inputs drops obs outside valid cells with message", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("terra")
  skip_if_not(reticulate::py_module_available("numpy"),
              "numpy not installed in active Python env")

  r <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 5,
                   ymin = 0, ymax = 5)
  vals <- runif(25)
  vals[1] <- NA
  terra::values(r) <- vals
  basis_stack <- c(r, r)

  obs_in_valid <- data.frame(x = 2.5, y = 2.5)
  obs_in_na    <- data.frame(x = 100, y = 100)
  obs_points   <- rbind(obs_in_valid, obs_in_na)

  expect_message(
    result <- DiffiScape:::.prepare_backend_inputs(basis_stack, obs_points),
    "Dropped"
  )
  expect_equal(result$n_obs, 1L)
})


test_that(".prepare_backend_inputs computes cell_area correctly", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("terra")
  skip_if_not(reticulate::py_module_available("numpy"),
              "numpy not installed in active Python env")

  r <- terra::rast(nrows = 4, ncols = 4, xmin = 0, xmax = 2,
                   ymin = 0, ymax = 2)
  terra::values(r) <- runif(16)
  basis_stack <- c(r, r)

  obs_points <- data.frame(x = 1.0, y = 1.0)
  result     <- DiffiScape:::.prepare_backend_inputs(basis_stack, obs_points)

  expect_equal(result$cell_area, 0.25, tolerance = 1e-8)
})


test_that(".prepare_backend_inputs returns correct grid dimensions", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("terra")
  skip_if_not(reticulate::py_module_available("numpy"),
              "numpy not installed in active Python env")

  r <- terra::rast(nrows = 6, ncols = 8, xmin = 0, xmax = 8,
                   ymin = 0, ymax = 6)
  terra::values(r) <- runif(48)
  basis_stack <- c(r, r)

  obs_points <- data.frame(x = 4, y = 3)
  result     <- DiffiScape:::.prepare_backend_inputs(basis_stack, obs_points)

  expect_equal(result$n_rows, 6L)
  expect_equal(result$n_cols, 8L)
})


test_that(".prepare_backend_inputs parity: identical output across two equivalent fixture calls", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("terra")
  skip_if_not(reticulate::py_module_available("numpy"),
              "numpy not installed in active Python env")

  set.seed(99)
  r1 <- terra::rast(nrows = 6, ncols = 6, xmin = 0, xmax = 6,
                    ymin = 0, ymax = 6, vals = runif(36))
  r2 <- terra::rast(nrows = 6, ncols = 6, xmin = 0, xmax = 6,
                    ymin = 0, ymax = 6, vals = runif(36))
  basis_stack <- c(r1, r2)
  obs_points  <- data.frame(x = c(1.5, 3.5, 5.5), y = c(1.5, 3.5, 5.5))

  prep_a <- DiffiScape:::.prepare_backend_inputs(basis_stack, obs_points)
  prep_b <- DiffiScape:::.prepare_backend_inputs(basis_stack, obs_points)

  expect_equal(prep_a$valid_mask, prep_b$valid_mask)
  expect_equal(prep_a$n_rows, prep_b$n_rows)
  expect_equal(prep_a$n_cols, prep_b$n_cols)
  expect_equal(prep_a$cell_area, prep_b$cell_area)
  expect_equal(prep_a$n_valid, prep_b$n_valid)
  expect_equal(prep_a$n_obs, prep_b$n_obs)
  expect_equal(reticulate::py_to_r(prep_a$basis_np),
               reticulate::py_to_r(prep_b$basis_np))
  expect_equal(reticulate::py_to_r(prep_a$obs_np),
               reticulate::py_to_r(prep_b$obs_np))
})
