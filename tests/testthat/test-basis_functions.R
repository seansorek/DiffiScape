# Tests for R/basis_functions.R

test_that("create_basis_stack creates a multi-layer raster", {
  r1 <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  r2 <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  terra::values(r1) <- runif(100)
  terra::values(r2) <- runif(100)

  stack <- create_basis_stack(list(elev = r1, slope = r2), rescale = TRUE)
  expect_s4_class(stack, "SpatRaster")
  expect_equal(terra::nlyr(stack), 2)
  expect_equal(names(stack), c("elev", "slope"))
})


test_that("create_basis_stack rescales to [0, 1]", {
  r1 <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  terra::values(r1) <- seq(10, 1000, length.out = 100)

  stack <- create_basis_stack(list(test = r1), rescale = TRUE)
  vals  <- terra::values(stack)[, 1]
  expect_true(all(vals >= 0 & vals <= 1, na.rm = TRUE))
})


test_that("create_basis_stack errors with no layers", {
  expect_error(create_basis_stack(list()))
})


test_that("validate_basis_stack returns TRUE for valid stack", {
  r1 <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  terra::values(r1) <- runif(25)
  stack <- create_basis_stack(list(a = r1), rescale = FALSE)
  expect_true(validate_basis_stack(stack))
})


test_that("check_basis_correlations returns correlation matrix", {
  r1 <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  r2 <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  set.seed(123)
  terra::values(r1) <- runif(100)
  terra::values(r2) <- runif(100)
  stack <- create_basis_stack(list(a = r1, b = r2), rescale = FALSE)

  cm <- check_basis_correlations(stack)
  expect_true(is.matrix(cm))
  expect_equal(nrow(cm), 2)
  expect_equal(unname(diag(cm)), c(1, 1), tolerance = 1e-10)
})


test_that("basis_summary returns a data.frame", {
  r1 <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  terra::values(r1) <- 1:25
  stack <- create_basis_stack(list(x = r1), rescale = FALSE)
  s <- basis_summary(stack)
  expect_s3_class(s, "data.frame")
  expect_true("layer" %in% names(s))
})


test_that("create_basis_stack handles single-layer input", {
  r1 <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  terra::values(r1) <- seq(1, 100)
  stack <- create_basis_stack(list(only = r1), rescale = TRUE)
  expect_equal(terra::nlyr(stack), 1)
  expect_equal(names(stack), "only")
  vals <- terra::values(stack)[, 1]
  expect_true(min(vals, na.rm = TRUE) >= 0)
  expect_true(max(vals, na.rm = TRUE) <= 1)
})


test_that("check_basis_correlations returns scalar (1x1 matrix) for single layer", {
  r1 <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  terra::values(r1) <- runif(25)
  stack <- create_basis_stack(list(a = r1), rescale = FALSE)
  cm <- check_basis_correlations(stack)
  expect_true(is.matrix(cm))
  expect_equal(dim(cm), c(1L, 1L))
  expect_equal(cm[1, 1], 1, tolerance = 1e-10)
})


test_that("create_basis_stack handles NaN/Inf by preserving NA mask", {
  r1 <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  v <- 1:25
  v[3] <- NA
  terra::values(r1) <- v

  stack <- create_basis_stack(list(x = r1), rescale = TRUE)
  vals  <- terra::values(stack)[, 1]
  # NA in input should propagate to output
  expect_true(is.na(vals[3]))
  # Non-NA values should be in [0, 1]
  expect_true(all(vals[-3] >= 0 & vals[-3] <= 1, na.rm = TRUE))
})
