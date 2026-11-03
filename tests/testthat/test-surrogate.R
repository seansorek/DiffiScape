test_that("the default surrogate is GP with no extra options", {
  cfg <- default_optimizer_config()
  expect_null(cfg$surrogate_type)
  expect_identical(cfg$surrogate_config, list())
})

test_that("GP wrappers and saved Python models give the same predictions", {
  withr::local_seed(1)
  X <- matrix(runif(20), ncol = 2, dimnames = list(NULL, c("r_0", "z_1")))
  surrogate <- fit_surrogate(X, rowSums(X^2))
  expect_s3_class(surrogate, "ds_surrogate")
  expect_identical(surrogate$type, "gp")
  expect_identical(surrogate$model$engine, "smt")
  expect_identical(surrogate$predictor_names, colnames(X))
  candidates <- matrix(runif(6), ncol = 2)
  expect_equal(predict_surrogate(surrogate, candidates),
               predict_surrogate(surrogate$model, candidates))
})

test_that("retired RF requests fail before fitting or writing output", {
  expect_error(fit_surrogate(matrix(1:4, 2), 1:2, type = "rf"), "gp")
  expect_error(optimize_resistance(NULL, NULL, config = list(surrogate_type = "rf")),
               "Only GP")
  old_rf <- structure(list(type = "rf"), class = "ds_surrogate")
  expect_error(predict_surrogate(old_rf, matrix(1:2, nrow = 1)), "Only GP")
  expect_error(loo_cv_surrogate(list(surrogate = old_rf)), "Only GP")
})

test_that("predict_surrogate rejects unsupported objects", {
  expect_error(predict_surrogate(list(a = 1), matrix(1:2, nrow = 1)), "ds_surrogate")
})
