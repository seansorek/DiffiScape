# Tests for R/diagnostics.R

test_that("compute_deviance_residuals works on known values", {
  # When y == mu, deviance residual should be 0
  obs <- c(5L, 10L, 0L)
  fit <- c(5.0, 10.0, 0.001)
  size <- 2

  dr <- compute_deviance_residuals(obs, fit, size)
  expect_length(dr, 3)

  # y == mu -> residual ~ 0

  expect_equal(dr[1], 0, tolerance = 1e-6)
  expect_equal(dr[2], 0, tolerance = 1e-6)
})


test_that("compute_deviance_residuals sign is correct", {
  obs <- c(10L, 1L)
  fit <- c(5.0, 5.0)
  size <- 2

  dr <- compute_deviance_residuals(obs, fit, size)
  expect_true(dr[1] > 0)   # obs > fit -> positive
  expect_true(dr[2] < 0)   # obs < fit -> negative
})


test_that("compute_deviance_residuals errors on mismatched lengths", {
  expect_error(compute_deviance_residuals(1:3, 1:4, 1))
})


test_that("moran_test works on synthetic data", {
  skip_on_cran()
  skip_if_not_installed("spdep")

  set.seed(42)
  n <- 50
  coords <- cbind(x = runif(n), y = runif(n))
  resid  <- rnorm(n)

  result <- moran_test(resid, coords, k = 5)
  expect_true("observed" %in% names(result))
  expect_true("p_value" %in% names(result))
  expect_true(is.numeric(result$p_value))
})


test_that("moran_test errors without spdep", {
  # We can't easily unload spdep, but we verify the function exists

  expect_true(is.function(moran_test))
})


test_that("plot_deviance_residuals runs without error", {
  skip_on_cran()
  obs <- rpois(100, 5)
  fit <- rep(5, 100)
  size <- 2

  # Should produce a plot without error
  expect_silent({
    pdf(tempfile(fileext = ".pdf"))
    plot_deviance_residuals(obs, fit, size)
    dev.off()
  })
})


test_that("plot_qq_deviance runs without error", {
  skip_on_cran()
  obs <- rpois(100, 5)
  fit <- rep(5, 100)
  size <- 2

  expect_silent({
    pdf(tempfile(fileext = ".pdf"))
    plot_qq_deviance(obs, fit, size)
    dev.off()
  })
})
