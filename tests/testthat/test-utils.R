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
