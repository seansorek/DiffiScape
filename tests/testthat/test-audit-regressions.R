test_that("torch rejects unsupported modeling arguments before running Python", {
  cases <- list(
    list(covariates_obs = list(habitat = 1)),
    list(covariates_rasters = list(habitat = 1)),
    list(available_covariates = list(habitat = 1)),
    list(residualise = TRUE),
    list(bounds = list(r_0 = c(0, 1))),
    list(config = list(family = family_negbin())),
    list(intensity_config = list(family = family_zinb())),
    list(config = list(distribution = "gam")),
    list(config = list(resistance_link = link_identity()))
  )
  local_mocked_bindings(
    run_torch_pipeline = function(...) stop("unexpected Python call"),
    .package = "DiffiScape"
  )
  for (args in cases) {
    expect_error(
      do.call(ds_optimize, c(list(basis_stack = NULL, obs_points = NULL, solver = "torch"), args)),
      "does not support|supports only"
    )
  }
})

test_that("explicit torch model choices override nested config and Poisson is accepted", {
  passed <- NULL
  local_mocked_bindings(
    run_torch_pipeline = function(...) { passed <<- list(...); list() },
    .package = "DiffiScape"
  )
  ds_optimize(NULL, NULL, model_type = "conv",
              config = list(family = family_poisson(), distribution = "gam",
                            torch = list(model_type = "mlp")))
  expect_identical(passed$model_type, "conv")
  expect_error(ds_optimize(NULL, NULL, model_type = "irl"), "arg")
})

test_that("retired R and Torch IRL entry points are unavailable", {
  expect_error(ds_optimize(NULL, NULL, solver = "irl"), "arg")
  expect_error(run_torch_pipeline(NULL, NULL, model_type = "irl"), "arg")
  expect_false(any(c("beta", "gamma_d", "n_value_iter", "value_scale_init") %in%
                     names(default_torch_config())))
})

test_that("gradient-shaped Poisson results refit through the shared evaluator", {
  connectivity <- terra::rast(nrows = 2, ncols = 2, xmin = 0, xmax = 2,
                               ymin = 0, ymax = 2, crs = "EPSG:3857", vals = 1:4)
  obs <- data.frame(x = c(0.5, 1.5, 0.5), y = c(0.5, 0.5, 1.5))
  cfg <- default_intensity_config()
  cfg$integration_subsample <- 1
  local_mocked_bindings(
    ds_jax_connectivity = function(...) list(cum_current = connectivity, elapsed_seconds = 0),
    .package = "DiffiScape"
  )
  result <- ds_fit_intensity(
    list(best_params = list(r_0 = 1, z_1 = 0), distribution = "poisson"),
    connectivity, obs, solver = "gradient", intensity_config = cfg
  )
  expect_true(is.finite(result$loglik))
  expect_identical(result$distribution, "poisson")
  expect_named(result$intensity_params, c("alpha", "gamma"))
  expect_identical(evaluate_full_model(
    list(r_0 = 1, z_1 = 0), connectivity, obs[1:2, ], distribution = "poisson",
    .min_obs = 3L, verbose = FALSE
  )$loglik, -1e10)
})

test_that("raster and explicit quadrature share the same intensity fit", {
  connectivity <- terra::rast(nrows = 2, ncols = 2, xmin = 0, xmax = 2,
                               ymin = 0, ymax = 2, crs = "EPSG:3857", vals = 1:4)
  obs <- data.frame(x = c(0.5, 1.5, 0.5), y = c(0.5, 0.5, 1.5))
  observed <- extract_connectivity(connectivity, obs)
  cfg <- default_intensity_config()
  cfg$integration_subsample <- 1
  raster_fit <- fit_intensity_nb(observed, connectivity, obs,
                                 config = cfg, family = family_poisson())
  available_fit <- fit_intensity_selection(observed, 1:4, obs,
                                           config = cfg, family = family_poisson())
  expect_equal(available_fit, raster_fit)
})

test_that("Laplace refits delegate and preserve log-likelihood curvature", {
  basis <- terra::rast(nrows = 2, ncols = 2, vals = 1)
  forwarded <- NULL
  local_mocked_bindings(
    evaluate_full_model = function(resistance_params, ...) {
      forwarded <<- list(...)
      list(loglik = -sum(unlist(resistance_params)^2))
    },
    .package = "DiffiScape"
  )
  opt <- list(best_params = list(r_0 = 0, z_1 = 0),
              bounds = list(r_0 = c(-1, 1), z_1 = c(-1, 1)), distribution = "poisson")
  result <- laplace_resistance(opt, basis, data.frame(x = 1:3, y = 1:3))
  expect_identical(forwarded$.min_obs, 3L)
  expect_identical(forwarded$distribution, "poisson")
  expect_equal(unname(diag(result$covariance)), rep(0.5, 2), tolerance = 1e-4)
})
