# Tests for R/optimizer.R

test_that("default_optimizer_config has required fields", {
  cfg <- default_optimizer_config()
  expect_true("n_init" %in% names(cfg))
  expect_true("n_iter" %in% names(cfg))
  expect_true("ts_min_sd" %in% names(cfg))
  expect_true("distribution" %in% names(cfg))
  expect_true("sigma_initial" %in% names(cfg))
  expect_true("n_candidates" %in% names(cfg))
})


test_that(".create_lhs_design generates correct dimensions", {
  bounds <- list(r_0 = c(-2, 2), z_1 = c(-3, 3))
  design <- .create_lhs_design(10, bounds)
  expect_equal(nrow(design), 10)
  expect_equal(ncol(design), 2)
  expect_equal(names(design), c("r_0", "z_1"))

  # Values should be within bounds
  expect_true(all(design$r_0 >= -2 & design$r_0 <= 2))
  expect_true(all(design$z_1 >= -3 & design$z_1 <= 3))
})


test_that(".fit_surrogate fits a GP model", {
  skip_on_cran()
  set.seed(1)
  X <- matrix(runif(20), ncol = 2)
  colnames(X) <- c("r_0", "z_1")
  y <- rowSums(X^2)

  gp <- .fit_surrogate(X, y)
  expect_s4_class(gp, "km")
})


test_that(".thompson_sampling returns one value per candidate", {
  skip_on_cran()
  set.seed(1)
  X <- matrix(runif(20), ncol = 2)
  colnames(X) <- c("r_0", "z_1")
  y <- rowSums(X^2)
  gp <- .fit_surrogate(X, y)

  cand <- matrix(runif(10), ncol = 2)
  colnames(cand) <- c("r_0", "z_1")
  ts <- .thompson_sampling(cand, gp)
  expect_length(ts, 5)
})


test_that(".generate_candidates returns correct dimensions", {
  bounds <- list(r_0 = c(-2, 2), z_1 = c(-3, 3))
  cand <- .generate_candidates(100, bounds, best_point = c(0, 0),
                                sigma_vector = c(0.1, 0.1),
                                local_frac = 0.5)
  expect_equal(ncol(cand), 2)
  expect_equal(nrow(cand), 100)
})


test_that(".generate_candidates keeps candidates within bounds", {
  bounds <- list(r_0 = c(-1, 1), z_1 = c(0, 2))
  cand   <- .generate_candidates(200, bounds, best_point = c(0, 1),
                                  sigma_vector = c(0.5, 0.5),
                                  local_frac   = 0.7)
  expect_true(all(cand[, "r_0"] >= -1 & cand[, "r_0"] <= 1))
  expect_true(all(cand[, "z_1"] >=  0 & cand[, "z_1"] <= 2))
})


test_that(".create_lhs_design respects bounds for three parameters", {
  bounds <- list(r_0 = c(0, 1), z_1 = c(-5, 5), z_2 = c(2, 4))
  design <- .create_lhs_design(20, bounds)
  expect_equal(ncol(design), 3)
  expect_true(all(design$r_0 >= 0  & design$r_0 <= 1))
  expect_true(all(design$z_1 >= -5 & design$z_1 <= 5))
  expect_true(all(design$z_2 >= 2  & design$z_2 <= 4))
})


# ---- optimize_resistance -----------------------------------------------------

test_that("optimize_resistance returns expected result structure", {
  skip_on_cran()
  skip_if_not_installed("terra")

  set.seed(42)
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
  cfg$n_init <- 5L
  cfg$n_iter <- 2L
  cfg$seed   <- 42L

  out_dir <- withr::local_tempdir()
  result  <- optimize_resistance(basis, obs, config = cfg,
                                 output_dir = out_dir)

  expect_type(result$best_params, "list")
  expect_true(all(c("r_0", "z_1", "z_2") %in% names(result$best_params)))
  expect_type(result$best_loglik, "double")
  expect_s3_class(result$X_evaluated, "data.frame")
  expect_type(result$y_evaluated, "double")
  expect_equal(result$n_evaluations, 7L)
  expect_s3_class(result$surrogate, "ds_surrogate")
  expect_equal(result$surrogate$type, "gp")
  expect_s4_class(result$surrogate$model, "km")
  expect_type(result$bounds, "list")
})


test_that("optimize_resistance creates output directory and saves RDS", {
  skip_on_cran()
  skip_if_not_installed("terra")

  set.seed(43)
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
  cfg$n_init <- 5L
  cfg$n_iter <- 2L
  cfg$seed   <- 43L

  parent  <- withr::local_tempdir()
  out_dir <- file.path(parent, "opt_output")
  result  <- optimize_resistance(basis, obs, config = cfg,
                                 output_dir = out_dir)

  expect_true(dir.exists(out_dir))
  expect_true(file.exists(file.path(out_dir, "optimization_results.rds")))
})


test_that("optimize_resistance uses default bounds when NULL", {
  skip_on_cran()
  skip_if_not_installed("terra")

  set.seed(44)
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
  cfg$n_init <- 5L
  cfg$n_iter <- 2L
  cfg$seed   <- 44L

  result <- optimize_resistance(basis, obs, bounds = NULL, config = cfg,
                                output_dir = withr::local_tempdir())

  expect_equal(length(result$bounds), 3)
  expect_true(all(c("r_0", "z_1", "z_2") %in% names(result$bounds)))
})


test_that("optimize_resistance works with EI acquisition", {
  skip_on_cran()
  skip_if_not_installed("terra")

  set.seed(45)
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
  cfg$n_init      <- 5L
  cfg$n_iter      <- 2L
  cfg$seed        <- 45L
  cfg$acquisition <- "EI"

  result <- optimize_resistance(basis, obs, config = cfg,
                                output_dir = withr::local_tempdir())

  expect_equal(result$acquisition, "EI")
  expect_false(is.null(result$xi_initial))
})


# ---- failed-evaluation handling (#101) ----------------------------------
# A failed outer-loop evaluation used to be reported as a fixed 1e10
# sentinel, which then poisoned the surrogate's fit (see issue #101). These
# tests confirm failures are reported as NA instead, and that a relative
# (not absolute) penalty is applied only when fitting the surrogate.

test_that(".outer_objective returns NA_real_ (not a 1e10 sentinel) when evaluate_full_model errors", {
  skip_if_not_installed("terra")

  local_mocked_bindings(
    evaluate_full_model = function(...) stop("simulated solver failure"),
    .package = "DiffiScape"
  )

  basis <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                       ymin = 0, ymax = 1, nlyrs = 2)
  terra::values(basis) <- runif(50)

  eval_counter <- new.env(parent = emptyenv())
  eval_counter$n <- 0L

  y <- .outer_objective(
    theta               = c(0, 0, 0),
    basis_stack         = basis,
    obs_points          = NULL,
    omniscape_settings  = list(),
    eval_counter        = eval_counter,
    log_file            = NULL,
    distribution        = "negbin",
    intensity_config    = default_intensity_config(),
    covariates_obs      = NULL,
    covariates_rasters  = NULL,
    residualise         = FALSE
  )

  expect_true(is.na(y))
  expect_false(isTRUE(y == 1e10))
})


test_that("optimize_resistance reports NA (not 1e10) for failed evaluations and still fits a surrogate", {
  skip_on_cran()
  skip_if_not_installed("terra")

  set.seed(46)
  basis <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                       ymin = 0, ymax = 1, nlyrs = 2)
  terra::values(basis) <- runif(50)
  obs <- data.frame(x = runif(10, 0.1, 0.9), y = runif(10, 0.1, 0.9))

  # Every 3rd evaluation "fails" (mirrors .outer_objective's own contract
  # of returning NA_real_ on failure).
  local_mocked_bindings(
    .outer_objective = function(theta, basis_stack, obs_points,
                                omniscape_settings, eval_counter,
                                log_file, ...) {
      eval_counter$n <- eval_counter$n + 1L
      if (eval_counter$n %% 3 == 0) return(NA_real_)
      sum(theta^2) + rnorm(1, sd = 0.01)
    },
    .package = "DiffiScape"
  )

  cfg <- default_optimizer_config()
  cfg$n_init <- 6L
  cfg$n_iter <- 3L
  cfg$seed   <- 46L

  result <- optimize_resistance(basis, obs, config = cfg,
                                output_dir = withr::local_tempdir())

  # Failures are preserved as NA, not silently coerced into a 1e10
  # sentinel that would distort the surrogate.
  expect_true(any(is.na(result$y_evaluated)))
  expect_false(any(result$y_evaluated == 1e10, na.rm = TRUE))

  # The best result must come from a real (non-failed) evaluation.
  expect_true(is.finite(result$best_loglik))

  # The surrogate must still be fit successfully despite the failures,
  # and its scale should not be dominated by an out-of-range sentinel.
  expect_s3_class(result$surrogate, "ds_surrogate")
  gp_model <- result$surrogate$model
  expect_s4_class(gp_model, "km")
  fitted_range <- range(gp_model@y)
  expect_true(diff(fitted_range) < 1e6)
})


# ---- failure penalty must worsen the objective regardless of sign -------
# Codex review on #103 (discussion_r3702710257): a *multiplicative* penalty
# (`max(valid) * 1.1`) only pushes the surrogate-fitting placeholder to a
# worse (larger) value than the worst real evaluation when `max(valid)` is
# positive. When valid scores are negative -- which happens whenever the
# underlying log-likelihood is positive (continuous or point-process
# models) -- multiplying by 1.1 pulls the placeholder *toward* zero, i.e.
# better, so a failed evaluation could be selected as the apparent optimum
# by which.min(). `.failure_penalty()` uses an additive, range-scaled
# margin instead, which is worse in the same direction regardless of sign.

test_that(".failure_penalty is always worse (larger) than every valid value, including when all valid values are negative", {
  valid <- c(-50, -30, -10)
  penalty <- .failure_penalty(valid)
  expect_true(penalty > max(valid))
})

test_that(".failure_penalty scales with the observed range and falls back to an absolute margin for a degenerate (zero-range) set of valid values", {
  expect_equal(.failure_penalty(numeric(0)), 1e10)
  expect_equal(.failure_penalty(c(0, 100)), 110)
  expect_equal(.failure_penalty(c(5, 5, 5)), 5.5)
})

test_that("optimize_resistance never lets a failed evaluation masquerade as the best when the objective is negative-valued", {
  skip_on_cran()
  skip_if_not_installed("terra")

  set.seed(47)
  basis <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                       ymin = 0, ymax = 1, nlyrs = 2)
  terra::values(basis) <- runif(50)
  obs <- data.frame(x = runif(10, 0.1, 0.9), y = runif(10, 0.1, 0.9))

  # A negative-valued objective (as if log-likelihood > 0), with every 4th
  # evaluation failing -- the exact scenario the multiplicative penalty
  # mishandled.
  local_mocked_bindings(
    .outer_objective = function(theta, basis_stack, obs_points,
                                omniscape_settings, eval_counter,
                                log_file, ...) {
      eval_counter$n <- eval_counter$n + 1L
      if (eval_counter$n %% 4 == 0) return(NA_real_)
      -(10 + sum(theta^2)) + rnorm(1, sd = 0.01)
    },
    .package = "DiffiScape"
  )

  cfg <- default_optimizer_config()
  cfg$n_init <- 8L
  cfg$n_iter <- 4L
  cfg$seed   <- 47L

  result <- optimize_resistance(basis, obs, config = cfg,
                                output_dir = withr::local_tempdir())

  expect_true(any(is.na(result$y_evaluated)))
  # The reported best must come from a real evaluation, never a failure.
  expect_false(is.na(result$y_evaluated[result$best_idx]))
  expect_true(is.finite(result$best_loglik))
})


# ---- evaluate_full_model ------------------------------------------------
# Characterization tests written ahead of the Problem A/B refactor (issue #2):
# collapsing .prepare_jax_inputs()/.prepare_torch_inputs() into
# .prepare_backend_inputs(), and routing ds_fit_intensity()'s "gradient"
# branch through evaluate_full_model() instead of reimplementing it. These
# tests lock in evaluate_full_model()'s existing/target behaviour so the
# refactor can be verified mechanically.

test_that("evaluate_full_model resolves omniscape radius/block_size via modifyList with partial overrides", {
  skip_on_cran()

  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)
  mock_conn   <- terra::rast(nrows = 2, ncols = 2, vals = 1)

  captured_radius     <- NULL
  captured_block_size <- NULL

  local_mocked_bindings(
    create_resistance_surface = function(...) mock_conn,
    ds_jax_connectivity = function(resistance, radius, block_size, ...) {
      captured_radius     <<- radius
      captured_block_size <<- block_size
      list(cum_current = mock_conn, flow_potential = NULL, elapsed_seconds = 0.1)
    },
    extract_connectivity = function(...) 1.0,
    fit_intensity_nb = function(...) {
      list(loglik = -1, estimates = c(alpha = 0, gamma = 0), se = NULL,
           hessian = NULL, convergence = 0L)
    },
    .package = "DiffiScape"
  )

  # Partial override: only radius is supplied; block_size should fall back
  # to the default (5L) via utils::modifyList(omni_def, omniscape_settings).
  evaluate_full_model(
    resistance_params  = list(r_0 = 0),
    basis_stack         = basis_stack,
    obs_points          = obs_points,
    omniscape_settings  = list(radius = 25L),
    verbose             = FALSE
  )

  expect_equal(captured_radius, 25L)
  expect_equal(captured_block_size, 5L)
})


test_that("evaluate_full_model forwards available_points/available_covariates into fit_intensity_nb", {
  skip_on_cran()

  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)
  avail_pts   <- data.frame(x = c(0, 1), y = c(0, 1))
  mock_conn   <- terra::rast(nrows = 2, ncols = 2, vals = 1)

  captured_args <- NULL

  local_mocked_bindings(
    create_resistance_surface = function(...) mock_conn,
    ds_jax_connectivity = function(...) {
      list(cum_current = mock_conn, flow_potential = NULL, elapsed_seconds = 0.1)
    },
    extract_connectivity = function(connectivity, points, ...) {
      rep(1.0, nrow(as.data.frame(points)))
    },
    fit_intensity_nb = function(...) {
      captured_args <<- list(...)
      list(loglik = -1, estimates = c(alpha = 0, gamma = 0), se = NULL,
           hessian = NULL, convergence = 0L)
    },
    .package = "DiffiScape"
  )

  evaluate_full_model(
    resistance_params     = list(r_0 = 0),
    basis_stack            = basis_stack,
    obs_points              = obs_points,
    available_points        = avail_pts,
    available_covariates    = list(elev = c(0.1, 0.2)),
    verbose                 = FALSE
  )

  expect_false(is.null(captured_args$available_connectivity))
  expect_equal(captured_args$available_connectivity, c(1.0, 1.0))
  expect_equal(captured_args$available_covariates, list(elev = c(0.1, 0.2)))
})


test_that("evaluate_full_model errors clearly when distribution='gam' and available_points is supplied", {
  skip_on_cran()

  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)
  avail_pts   <- data.frame(x = c(0, 1), y = c(0, 1))
  mock_conn   <- terra::rast(nrows = 2, ncols = 2, vals = 1)

  local_mocked_bindings(
    create_resistance_surface = function(...) mock_conn,
    ds_jax_connectivity = function(...) {
      list(cum_current = mock_conn, flow_potential = NULL, elapsed_seconds = 0.1)
    },
    extract_connectivity = function(connectivity, points, ...) {
      rep(1.0, nrow(as.data.frame(points)))
    },
    .package = "DiffiScape"
  )

  expect_error(
    evaluate_full_model(
      resistance_params  = list(r_0 = 0),
      basis_stack          = basis_stack,
      obs_points            = obs_points,
      distribution          = "gam",
      available_points      = avail_pts,
      verbose               = FALSE
    ),
    "not supported with distribution"
  )
})


test_that("evaluate_full_model defaults distribution to 'negbin' when explicitly passed NULL", {
  skip_on_cran()

  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)
  mock_conn   <- terra::rast(nrows = 2, ncols = 2, vals = 1)

  captured_distribution <- "unset"

  local_mocked_bindings(
    create_resistance_surface = function(...) mock_conn,
    ds_jax_connectivity = function(...) {
      list(cum_current = mock_conn, flow_potential = NULL, elapsed_seconds = 0.1)
    },
    extract_connectivity = function(...) 1.0,
    fit_intensity_nb = function(...) {
      list(loglik = -1, estimates = c(alpha = 0, gamma = 0), se = NULL,
           hessian = NULL, convergence = 0L)
    },
    .package = "DiffiScape"
  )

  # Mimics ds_fit_intensity()'s surrogate branch calling with
  # distribution = opt_result$distribution where opt_result$distribution is
  # NULL -- this must not error and must resolve to "negbin" internally.
  result <- evaluate_full_model(
    resistance_params  = list(r_0 = 0),
    basis_stack          = basis_stack,
    obs_points            = obs_points,
    distribution          = NULL,
    verbose               = FALSE
  )

  expect_equal(result$distribution, "negbin")
})


test_that("evaluate_full_model errors instead of silently dropping out-of-mask obs_points for family_clogit() with strata (#90)", {
  skip_on_cran()

  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  # Two used locations (one per stratum); the mock extract_connectivity()
  # below returns NA for the first one, simulating a point that falls
  # outside the connectivity raster's valid mask.
  obs_points  <- data.frame(x = c(0, 1), y = c(0, 1))
  avail_pts   <- data.frame(x = c(0, 1, 2, 3), y = c(0, 1, 2, 3))
  mock_conn   <- terra::rast(nrows = 2, ncols = 2, vals = 1)

  fam <- family_clogit(
    stratum_ids_used  = c(1L, 2L),
    stratum_ids_avail = c(1L, 1L, 2L, 2L)
  )

  local_mocked_bindings(
    create_resistance_surface = function(...) mock_conn,
    ds_jax_connectivity = function(...) {
      list(cum_current = mock_conn, flow_potential = NULL, elapsed_seconds = 0.1)
    },
    extract_connectivity = function(connectivity, points, ...) {
      n <- nrow(as.data.frame(points))
      vals <- rep(1.0, n)
      vals[1] <- NA_real_  # first location falls outside the mask
      vals
    },
    .package = "DiffiScape"
  )

  expect_error(
    evaluate_full_model(
      resistance_params  = list(r_0 = 0),
      basis_stack          = basis_stack,
      obs_points            = obs_points,
      available_points      = avail_pts,
      family                = fam,
      verbose               = FALSE
    ),
    "stratum idx_map"
  )
})


test_that("evaluate_full_model errors instead of silently dropping out-of-mask available_points for family_clogit() with strata (#90)", {
  skip_on_cran()

  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = c(0, 1), y = c(0, 1))
  avail_pts   <- data.frame(x = c(0, 1, 2, 3), y = c(0, 1, 2, 3))
  mock_conn   <- terra::rast(nrows = 2, ncols = 2, vals = 1)

  fam <- family_clogit(
    stratum_ids_used  = c(1L, 2L),
    stratum_ids_avail = c(1L, 1L, 2L, 2L)
  )

  local_mocked_bindings(
    create_resistance_surface = function(...) mock_conn,
    ds_jax_connectivity = function(...) {
      list(cum_current = mock_conn, flow_potential = NULL, elapsed_seconds = 0.1)
    },
    extract_connectivity = function(connectivity, points, ...) {
      n <- nrow(as.data.frame(points))
      vals <- rep(1.0, n)
      if (n == 4) vals[1] <- NA_real_  # avail point outside the mask
      vals
    },
    .package = "DiffiScape"
  )

  expect_error(
    evaluate_full_model(
      resistance_params  = list(r_0 = 0),
      basis_stack          = basis_stack,
      obs_points            = obs_points,
      available_points      = avail_pts,
      family                = fam,
      verbose               = FALSE
    ),
    "stratum idx_map"
  )
})


test_that("evaluate_full_model still drops out-of-mask obs_points for non-strata families (unchanged behaviour)", {
  skip_on_cran()

  basis_stack <- terra::rast(nrows = 2, ncols = 2, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = c(0, 1), y = c(0, 1))
  mock_conn   <- terra::rast(nrows = 2, ncols = 2, vals = 1)

  local_mocked_bindings(
    create_resistance_surface = function(...) mock_conn,
    ds_jax_connectivity = function(...) {
      list(cum_current = mock_conn, flow_potential = NULL, elapsed_seconds = 0.1)
    },
    extract_connectivity = function(connectivity, points, ...) {
      n <- nrow(as.data.frame(points))
      vals <- rep(1.0, n)
      vals[1] <- NA_real_
      vals
    },
    fit_intensity_nb = function(...) {
      list(loglik = -1, estimates = c(alpha = 0, gamma = 0), se = NULL,
           hessian = NULL, convergence = 0L)
    },
    .package = "DiffiScape"
  )

  expect_no_error(
    evaluate_full_model(
      resistance_params  = list(r_0 = 0),
      basis_stack          = basis_stack,
      obs_points            = obs_points,
      verbose               = FALSE
    )
  )
})


# ---- GH #105: gradient objective radius/block_size threading -----------
# The Python gradient objective (_connectivity_objective) previously
# accepted radius/block_size but silently ignored them -- tuning the
# moving-window solve had no effect on the fit, even though the same
# radius/block_size changed every downstream connectivity surface. These
# tests lock in that optimize_resistance_gradient() and the neural
# dispatch (.optimize_neural()) both resolve radius/block_size from
# config$omniscape and forward them, unmodified, into the JAX bridge
# calls that in turn honor them in Python (core.cumulative_current_core).

test_that("optimize_resistance_gradient forwards config$omniscape radius/block_size to ds_jax_optimize", {
  skip_on_cran()

  basis_stack <- terra::rast(nrows = 3, ncols = 3, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)

  captured_radius     <- NULL
  captured_block_size <- NULL

  local_mocked_bindings(
    ds_jax_optimize = function(basis_np, obs_np, valid_mask_np,
                                n_rows, n_cols, cell_area,
                                init_params = NULL, radius, block_size, ...) {
      captured_radius     <<- radius
      captured_block_size <<- block_size
      list(best_params = c(0, 0), best_loglik = -1, n_epochs_run = 1L,
           elapsed = 0.01, converged = TRUE)
    },
    .package = "DiffiScape"
  )

  cfg <- default_optimizer_config()
  cfg$omniscape <- list(radius = 21L, block_size = 4L, cleanup = TRUE)

  optimize_resistance_gradient(
    basis_stack = basis_stack,
    obs_points  = obs_points,
    config      = cfg,
    output_dir  = withr::local_tempdir()
  )

  expect_equal(captured_radius, 21L)
  expect_equal(captured_block_size, 4L)
})


test_that("optimize_resistance_gradient uses default radius/block_size (13/5) when config$omniscape is unset", {
  skip_on_cran()

  basis_stack <- terra::rast(nrows = 3, ncols = 3, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)

  captured_radius     <- NULL
  captured_block_size <- NULL

  local_mocked_bindings(
    ds_jax_optimize = function(basis_np, obs_np, valid_mask_np,
                                n_rows, n_cols, cell_area,
                                init_params = NULL, radius, block_size, ...) {
      captured_radius     <<- radius
      captured_block_size <<- block_size
      list(best_params = c(0, 0), best_loglik = -1, n_epochs_run = 1L,
           elapsed = 0.01, converged = TRUE)
    },
    .package = "DiffiScape"
  )

  cfg <- default_optimizer_config()
  cfg$omniscape <- NULL

  optimize_resistance_gradient(
    basis_stack = basis_stack,
    obs_points  = obs_points,
    config      = cfg,
    output_dir  = withr::local_tempdir()
  )

  expect_equal(captured_radius, 13L)
  expect_equal(captured_block_size, 5L)
})


test_that(".optimize_neural forwards config$omniscape radius/block_size to ds_jax_neural_optimize", {
  skip_on_cran()

  basis_stack <- terra::rast(nrows = 3, ncols = 3, nlyrs = 1, vals = 1)
  obs_points  <- data.frame(x = 0, y = 0)

  captured_radius     <- NULL
  captured_block_size <- NULL

  local_mocked_bindings(
    ds_jax_neural_optimize = function(basis_np, obs_np, valid_mask_np,
                                       n_rows, n_cols, cell_area,
                                       model_type = "mlp",
                                       model_config = list(),
                                       optim_config = list(),
                                       radius, block_size, ...) {
      captured_radius     <<- radius
      captured_block_size <<- block_size
      list(resistance = rep(1, n_rows * n_cols), best_loglik = -1,
           loss_history = list(-1), n_epochs_run = 1L, elapsed = 0.01,
           model_type = model_type)
    },
    .package = "DiffiScape"
  )

  cfg <- default_optimizer_config()
  cfg$omniscape <- list(radius = 9L, block_size = 2L, cleanup = TRUE)

  optimize_resistance_gradient(
    basis_stack = basis_stack,
    obs_points  = obs_points,
    config      = cfg,
    model_type  = "mlp",
    output_dir  = withr::local_tempdir()
  )

  expect_equal(captured_radius, 9L)
  expect_equal(captured_block_size, 2L)
})
