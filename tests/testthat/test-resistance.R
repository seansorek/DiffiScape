# Tests for R/resistance.R

test_that("create_resistance_surface produces valid raster", {
  r1 <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  r2 <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  terra::values(r1) <- runif(100)
  terra::values(r2) <- runif(100)
  basis <- create_basis_stack(list(a = r1, b = r2), rescale = TRUE)

  params <- list(r_0 = 0, z_1 = 1, z_2 = -0.5)
  resist <- create_resistance_surface(params, basis)

  expect_s4_class(resist, "SpatRaster")
  expect_equal(terra::nlyr(resist), 1)
  vals <- terra::values(resist)[, 1]
  expect_true(all(vals > 0, na.rm = TRUE))   # exp(.) > 0
})


test_that("create_resistance_surface clamps to bounds", {
  r1 <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  terra::values(r1) <- runif(25)
  basis <- create_basis_stack(list(a = r1), rescale = FALSE)

  params <- list(r_0 = 50, z_1 = 0)   # very high -> clamped to R_max
  resist <- create_resistance_surface(params, basis, R_min = 0.01, R_max = 100)
  vals <- terra::values(resist)[, 1]
  expect_true(all(vals <= 100, na.rm = TRUE))
  expect_true(all(vals >= 0.01, na.rm = TRUE))
})


test_that("quick_resistance matches create_resistance_surface", {
  r1 <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  terra::values(r1) <- seq(0, 1, length.out = 25)
  basis <- create_basis_stack(list(a = r1), rescale = FALSE)

  params <- list(r_0 = 1, z_1 = 0.5)
  resist_rast <- create_resistance_surface(params, basis)
  rast_vals   <- terra::values(resist_rast)[, 1]

  basis_mat <- matrix(terra::values(basis)[, 1], ncol = 1)
  theta     <- c(1, 0.5)
  quick_vals <- quick_resistance(theta, basis_mat)

  expect_equal(quick_vals, rast_vals, tolerance = 1e-10)
})


test_that("get_default_bounds has correct structure", {
  bounds <- get_default_bounds(3)
  expect_named(bounds, c("r_0", "z_1", "z_2", "z_3"))
  expect_length(bounds$r_0, 2)
  expect_true(bounds$r_0[1] < bounds$r_0[2])
})


test_that("params_vector_to_list round-trips", {
  params <- list(r_0 = 1.5, z_1 = 0.3, z_2 = -0.7)
  vec    <- DiffiScape:::.params_to_vector(params, 2)
  back   <- params_vector_to_list(vec, 2)
  expect_equal(back$r_0, 1.5)
  expect_equal(back$z_1, 0.3)
  expect_equal(back$z_2, -0.7)
})


test_that("resistance_sensitivity returns a data.frame", {
  r1 <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  terra::values(r1) <- runif(25)
  basis <- create_basis_stack(list(a = r1), rescale = FALSE)

  params <- list(r_0 = 1, z_1 = 0.5)
  sens <- resistance_sensitivity(params, basis)
  expect_s3_class(sens, "data.frame")
  expect_equal(nrow(sens), 2)  # r_0 and z_1
  expect_true("parameter" %in% names(sens))
  expect_true("mean_pct_change" %in% names(sens))
})
