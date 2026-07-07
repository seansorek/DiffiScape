# Tests for the configurable surrogate abstraction (fit_surrogate() /
# predict_surrogate()) introduced in R/optimizer.R (issue #43).

test_that("default_optimizer_config defaults to the GP surrogate", {
  cfg <- default_optimizer_config()
  expect_equal(cfg$surrogate_type, "gp")
  expect_type(cfg$surrogate_config, "list")
  expect_length(cfg$surrogate_config, 0)
})


test_that("fit_surrogate(type = 'gp') matches legacy .fit_surrogate() behaviour", {
  skip_on_cran()
  set.seed(1)
  X <- matrix(runif(20), ncol = 2)
  colnames(X) <- c("r_0", "z_1")
  y <- rowSums(X^2)

  legacy <- .fit_surrogate(X, y)
  wrapped <- fit_surrogate(X, y, type = "gp")

  expect_s3_class(wrapped, "ds_surrogate")
  expect_equal(wrapped$type, "gp")
  expect_s4_class(wrapped$model, "km")
  expect_equal(wrapped$predictor_names, colnames(X))

  cand <- matrix(runif(6), ncol = 2)
  colnames(cand) <- c("r_0", "z_1")

  pred_legacy  <- predict_surrogate(legacy, cand)
  pred_wrapped <- predict_surrogate(wrapped, cand)

  expect_equal(pred_wrapped$mean, pred_legacy$mean, tolerance = 1e-6)
  expect_equal(pred_wrapped$sd,   pred_legacy$sd,   tolerance = 1e-6)
})


test_that("fit_surrogate defaults to 'gp' when type is unspecified", {
  skip_on_cran()
  set.seed(2)
  X <- matrix(runif(20), ncol = 2)
  colnames(X) <- c("r_0", "z_1")
  y <- rowSums(X^2)

  s <- fit_surrogate(X, y)
  expect_equal(s$type, "gp")
})


test_that("fit_surrogate(type = 'rf') fits a ranger model and predicts sane shapes", {
  skip_on_cran()
  skip_if_not_installed("ranger")
  set.seed(3)
  X <- matrix(runif(40), ncol = 2)
  colnames(X) <- c("r_0", "z_1")
  y <- rowSums(X^2) + rnorm(20, sd = 0.01)

  rf <- fit_surrogate(X, y, type = "rf")
  expect_s3_class(rf, "ds_surrogate")
  expect_equal(rf$type, "rf")
  expect_s3_class(rf$model, "ranger")

  cand <- matrix(runif(10), ncol = 2)
  colnames(cand) <- c("r_0", "z_1")

  pred <- predict_surrogate(rf, cand)
  expect_length(pred$mean, 5)
  expect_length(pred$sd, 5)
  expect_true(all(is.finite(pred$mean)))
  expect_true(all(is.finite(pred$sd)))
  expect_true(all(pred$sd >= 0))
})


test_that("fit_surrogate(type = 'rf') errors clearly when ranger is unavailable", {
  skip_on_cran()
  local_mocked_bindings(
    requireNamespace = function(pkg, ...) if (identical(pkg, "ranger")) FALSE else TRUE,
    .package = "base"
  )
  X <- matrix(runif(20), ncol = 2)
  colnames(X) <- c("r_0", "z_1")
  y <- rowSums(X^2)

  expect_error(fit_surrogate(X, y, type = "rf"), "ranger")
})


test_that("predict_surrogate rejects objects that aren't a ds_surrogate or km", {
  expect_error(predict_surrogate(list(a = 1), matrix(1:2, nrow = 1)),
               "ds_surrogate")
})


test_that(".thompson_sampling and .expected_improvement work with an RF surrogate", {
  skip_on_cran()
  skip_if_not_installed("ranger")
  set.seed(4)
  X <- matrix(runif(40), ncol = 2)
  colnames(X) <- c("r_0", "z_1")
  y <- rowSums(X^2) + rnorm(20, sd = 0.01)
  rf <- fit_surrogate(X, y, type = "rf")

  cand <- matrix(runif(10), ncol = 2)
  colnames(cand) <- c("r_0", "z_1")

  ts <- .thompson_sampling(cand, rf)
  expect_length(ts, 5)
  expect_true(all(is.finite(ts)))

  ei <- .expected_improvement(cand, rf, y_best = min(y),
                              xi_initial = 0.1, iter = 1, n_iter = 10)
  expect_length(ei, 5)
  expect_true(all(is.finite(ei)))
  expect_true(all(ei >= 0))
})


# ---- optimize_resistance with surrogate_type = "rf" ------------------------

test_that("optimize_resistance runs end-to-end with surrogate_type = 'rf'", {
  skip_on_cran()
  skip_if_not_installed("terra")
  skip_if_not_installed("ranger")

  set.seed(46)
  basis <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                       ymin = 0, ymax = 1, nlyrs = 2)
  terra::values(basis) <- runif(50)
  obs <- data.frame(x = runif(10, 0.1, 0.9), y = runif(10, 0.1, 0.9))

  local_mocked_bindings(
    .outer_objective = function(theta, basis_stack, obs_points,
                                omniscape_settings, eval_counter,
                                log_file, ...) {
      eval_counter$n <- eval_counter$n + 1L
      sum(theta^2) + rnorm(1, sd = 0.01)
    },
    .package = "DiffiScape"
  )

  cfg <- default_optimizer_config()
  cfg$n_init         <- 5L
  cfg$n_iter         <- 2L
  cfg$seed           <- 46L
  cfg$surrogate_type <- "rf"

  result <- optimize_resistance(basis, obs, config = cfg,
                                output_dir = withr::local_tempdir())

  expect_type(result$best_params, "list")
  expect_true(all(c("r_0", "z_1", "z_2") %in% names(result$best_params)))
  expect_type(result$best_loglik, "double")
  expect_s3_class(result$surrogate, "ds_surrogate")
  expect_equal(result$surrogate$type, "rf")
  expect_s3_class(result$surrogate$model, "ranger")
  expect_equal(result$n_evaluations, 7L)
})
