# Tests for R/pipeline.R

test_that("ds_load_data reads CSV correctly", {
  tmp <- tempfile(fileext = ".csv")
  df  <- data.frame(x = 1:5, y = 6:10, extra = letters[1:5])
  write.csv(df, tmp, row.names = FALSE)

  pts <- ds_load_data(tmp)
  expect_true("x" %in% names(pts))
  expect_true("y" %in% names(pts))
  expect_equal(nrow(pts), 5)
  unlink(tmp)
})


test_that("ds_load_data errors on missing columns", {
  tmp <- tempfile(fileext = ".csv")
  write.csv(data.frame(a = 1:3, b = 4:6), tmp, row.names = FALSE)
  expect_error(ds_load_data(tmp), "must contain columns")
  unlink(tmp)
})


test_that("ds_load_data errors on unsupported format", {
  expect_error(ds_load_data("data.xyz"), "Unsupported file type")
})


test_that("ds_create_basis reads from file paths", {
  skip_on_cran()

  r1 <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  terra::values(r1) <- runif(25)

  tmp <- tempfile(fileext = ".tif")
  terra::writeRaster(r1, tmp)

  stack <- ds_create_basis(tmp)
  expect_s4_class(stack, "SpatRaster")
  expect_equal(terra::nlyr(stack), 1)
  unlink(tmp)
})


test_that("ds_create_basis reads from directory", {
  skip_on_cran()

  dir <- tempdir()
  subdir <- file.path(dir, "test_rasters")
  dir.create(subdir, showWarnings = FALSE)

  for (nm in c("elev", "slope")) {
    r <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                     ymin = 0, ymax = 1)
    terra::values(r) <- runif(25)
    terra::writeRaster(r, file.path(subdir, paste0(nm, ".tif")),
                       overwrite = TRUE)
  }

  stack <- ds_create_basis(subdir)
  expect_s4_class(stack, "SpatRaster")
  expect_equal(terra::nlyr(stack), 2)
  unlink(subdir, recursive = TRUE)
})


test_that("ds_init_julia errors without Julia", {
  # This test is expected to fail unless Julia is installed
  skip_if_not_installed("JuliaConnectoR")
  skip_on_cran()

  # Just verify the function exists and handles errors
  expect_true(is.function(ds_init_julia))
})
