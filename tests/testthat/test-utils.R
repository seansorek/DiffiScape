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
