# Tests for R/posterior.R

test_that("posterior_summary works on sample data.frame", {
  samples <- data.frame(
    r_0    = rnorm(100, 1, 0.2),
    z_1    = rnorm(100, 0.5, 0.3),
    loglik = rnorm(100, -200, 10)
  )

  summ <- posterior_summary(samples)
  expect_s3_class(summ, "data.frame")
  expect_true("parameter" %in% names(summ))
  expect_true("mean" %in% names(summ))
  expect_true("sd" %in% names(summ))
  expect_true("lower" %in% names(summ))
  expect_true("upper" %in% names(summ))
  expect_equal(nrow(summ), 2)  # r_0 and z_1 (not loglik)
})


test_that("posterior_summary returns empty for empty input", {
  summ <- posterior_summary(data.frame())
  expect_equal(nrow(summ), 0)
})


test_that("plot_posterior runs without error", {
  skip_on_cran()
  samples <- data.frame(r_0 = rnorm(100), z_1 = rnorm(100))

  expect_silent({
    pdf(tempfile(fileext = ".pdf"))
    plot_posterior(samples, "r_0")
    dev.off()
  })
})


test_that("plot_posterior errors on missing parameter", {
  samples <- data.frame(r_0 = rnorm(10))
  expect_error(plot_posterior(samples, "z_99"), "not found")
})
