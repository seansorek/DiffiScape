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


# ============================================================================
# S3 class: resistance_model
# ============================================================================

.make_basis <- function(n = 25, rescale = FALSE) {
  r1 <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  terra::values(r1) <- runif(n)
  create_basis_stack(list(a = r1), rescale = rescale)
}


test_that("resistance_model() constructs a parametric model correctly", {
  basis <- .make_basis()
  m <- resistance_model(c(r_0 = 1, z_1 = 0.5), basis)

  expect_s3_class(m, "resistance_model")
  # Unified class -- no subtype class
  expect_false(inherits(m, "resistance_model_parametric"))
  expect_equal(m$type, "parametric")
  expect_equal(m$R_min, 1)
  expect_equal(m$R_max, 5000)
  expect_equal(m$params, c(1, 0.5))   # normalised to numeric vector
  expect_true(is.function(m$fn))       # fn field present for consistent interface
})


test_that("resistance_model() stores extra metadata via ...", {
  basis <- .make_basis()
  m <- resistance_model(c(r_0 = 0, z_1 = 0), basis,
                        name = "test model", version = 2L)
  expect_equal(m$extra$name, "test model")
  expect_equal(m$extra$version, 2L)
})


test_that("resistance_model() constructs a custom model correctly", {
  basis <- .make_basis()
  fn <- function(bs, ...) terra::app(bs[[1]], exp)
  m <- resistance_model(fn, basis, type = "custom")

  expect_s3_class(m, "resistance_model")
  # Unified class -- no subtype class
  expect_false(inherits(m, "resistance_model_custom"))
  expect_equal(m$type, "custom")
  expect_true(is.function(m$fn))
  expect_null(m$params)   # no numeric params for custom type
})


test_that("resistance_model() errors informatively for bad custom params", {
  basis <- .make_basis()
  expect_error(resistance_model("not_a_function", basis, type = "custom"),
               "must be a function")
})


test_that("print.resistance_model() runs without error", {
  basis <- .make_basis()
  m <- resistance_model(c(r_0 = 1, z_1 = 0.5), basis)
  expect_output(print(m), "resistance_model")
  expect_output(print(m), "parametric")
  expect_output(print(m), "r_0")
})


test_that("summary.resistance_model() runs without error", {
  basis <- .make_basis()
  m <- resistance_model(c(r_0 = 1, z_1 = 0.5), basis)
  expect_output(summary(m), "Sensitivity")
})


test_that("predict() on parametric model matches create_resistance_surface()", {
  basis <- .make_basis()
  params <- c(r_0 = 1, z_1 = 0.5)
  m <- resistance_model(params, basis)

  R_obj  <- predict(m)
  R_func <- create_resistance_surface(params, basis)

  expect_s4_class(R_obj, "SpatRaster")
  expect_equal(terra::values(R_obj), terra::values(R_func), tolerance = 1e-10)
})


test_that("predict() on parametric model accepts return_log = TRUE", {
  basis <- .make_basis()
  m <- resistance_model(c(r_0 = 1, z_1 = 0), basis)
  logR <- predict(m, return_log = TRUE)

  expect_s4_class(logR, "SpatRaster")
  expect_equal(names(logR), "log_resistance")
})


test_that("predict() on parametric model accepts a replacement basis_stack", {
  r1 <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  terra::values(r1) <- runif(25)
  basis_a <- create_basis_stack(list(a = r1), rescale = FALSE)

  r2 <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  terra::values(r2) <- runif(25)
  basis_b <- create_basis_stack(list(a = r2), rescale = FALSE)

  m <- resistance_model(c(r_0 = 1, z_1 = 0.5), basis_a)
  R_a <- predict(m)
  R_b <- predict(m, basis_stack = basis_b)

  expect_false(isTRUE(all.equal(terra::values(R_a), terra::values(R_b))))
})


test_that("parametric and custom models are interchangeable via predict()", {
  basis <- .make_basis()
  params <- c(r_0 = 1, z_1 = 0.5)

  m_param  <- resistance_model(params, basis, R_min = 1, R_max = 5000)

  # Wrap the parametric formula as a custom function -- should give same result
  fn <- function(bs, ...) {
    create_resistance_surface(params, bs, R_min = 1, R_max = 5000)
  }
  m_custom <- resistance_model(fn, basis, type = "custom")

  R_param  <- predict(m_param)
  R_custom <- predict(m_custom)

  # Same values via the same predict() interface
  expect_equal(terra::values(R_param), terra::values(R_custom), tolerance = 1e-10)
})


test_that("predict() on custom model calls the user function", {
  basis <- .make_basis()
  called <- FALSE
  fn <- function(bs, ...) {
    called <<- TRUE
    terra::app(bs[[1]], exp)
  }
  m <- resistance_model(fn, basis, type = "custom")
  R <- predict(m)

  expect_true(called)
  expect_s4_class(R, "SpatRaster")
  expect_equal(terra::nlyr(R), 1L)
})


test_that("predict() errors if function returns wrong output", {
  basis <- .make_basis()
  fn_bad <- function(bs, ...) terra::values(bs[[1]])   # returns matrix, not SpatRaster
  m <- resistance_model(fn_bad, basis, type = "custom")
  expect_error(predict(m), "single-layer SpatRaster")
})


test_that("predict() on custom model accepts extra arguments via ...", {
  basis <- .make_basis()
  fn <- function(bs, scale = 1, ...) terra::app(bs[[1]], function(x) exp(x * scale))
  m <- resistance_model(fn, basis, type = "custom")

  R1 <- predict(m, scale = 1)
  R2 <- predict(m, scale = 2)

  expect_false(isTRUE(all.equal(terra::values(R1), terra::values(R2))))
})


test_that("resistance_sensitivity uses the provided link function (#48)", {
  r1 <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  terra::values(r1) <- runif(25)
  basis <- create_basis_stack(list(a = r1), rescale = FALSE)

  params <- list(r_0 = 1, z_1 = 0.5)

  # Default link (exp) vs softplus should give different sensitivity values
  sens_exp     <- resistance_sensitivity(params, basis, link = link_exp())
  sens_softplus <- resistance_sensitivity(params, basis, link = link_softplus())

  expect_s3_class(sens_exp, "data.frame")
  expect_s3_class(sens_softplus, "data.frame")
  expect_equal(nrow(sens_exp), 2)
  expect_equal(nrow(sens_softplus), 2)

  # The percentage changes should differ between link functions
  expect_false(
    isTRUE(all.equal(sens_exp$mean_pct_change,
                     sens_softplus$mean_pct_change)),
    info = "Sensitivity should differ between exp and softplus links"
  )
})


test_that("summary.resistance_model forwards link to resistance_sensitivity (#48)", {
  r1 <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  terra::values(r1) <- runif(25)
  basis <- create_basis_stack(list(a = r1), rescale = FALSE)

  # Create a model with softplus link
  m <- resistance_model(c(r_0 = 1, z_1 = 0.5), basis, link = link_softplus())
  expect_equal(m$link$name, "softplus")

  # summary() should run without error and use the correct link
  expect_output(summary(m), "Sensitivity")
})
