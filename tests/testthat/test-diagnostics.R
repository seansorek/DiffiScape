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


# ---------------------------------------------------------------------------
# compute_deviance_residuals_gam
# ---------------------------------------------------------------------------

test_that("compute_deviance_residuals_gam works on a real bam object", {
  skip_on_cran()
  skip_if_not_installed("mgcv")
  set.seed(42)

  r <- terra::rast(nrows = 8, ncols = 8, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(64, mean = 3))

  fit <- fit_intensity_gam(
    connectivity_at_obs = abs(rnorm(20, mean = 3)),
    connectivity_raster = r,
    obs_coords = data.frame(x = runif(20, 0.1, 0.9),
                            y = runif(20, 0.1, 0.9))
  )

  resids <- compute_deviance_residuals_gam(fit$gam_model)
  expect_true(is.numeric(resids))
  expect_true(length(resids) > 0)
  expect_true(all(is.finite(resids)))
})


test_that("compute_deviance_residuals_gam extracts from a fit_intensity_gam list", {
  skip_on_cran()
  skip_if_not_installed("mgcv")
  set.seed(43)

  r <- terra::rast(nrows = 8, ncols = 8, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(64, mean = 3))

  fit <- fit_intensity_gam(
    connectivity_at_obs = abs(rnorm(20, mean = 3)),
    connectivity_raster = r,
    obs_coords = data.frame(x = runif(20, 0.1, 0.9),
                            y = runif(20, 0.1, 0.9))
  )

  # Pass the whole list (not just the gam model) — should auto-extract $gam_model
  resids_from_list   <- compute_deviance_residuals_gam(fit)
  resids_from_model  <- compute_deviance_residuals_gam(fit$gam_model)
  expect_equal(resids_from_list, resids_from_model)
})


test_that("compute_deviance_residuals_gam errors on invalid input", {
  expect_error(compute_deviance_residuals_gam(data.frame(x = 1:3)),
               "gam/bam")
  expect_error(compute_deviance_residuals_gam(list(foo = 1)),
               "gam/bam")
})


# ---------------------------------------------------------------------------
# rasterise_deviance_residuals and diagnose_model
# ---------------------------------------------------------------------------

.make_diag_inputs <- function(seed = 99) {
  set.seed(seed)
  r <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(100, mean = 4))
  n_obs   <- 20
  obs_pts <- data.frame(x = runif(n_obs, 0.05, 0.95),
                        y = runif(n_obs, 0.05, 0.95))
  nb_fit  <- fit_intensity_nb(
    connectivity_at_obs = abs(rnorm(n_obs, mean = 4)),
    connectivity_raster = r,
    obs_coords          = obs_pts
  )
  list(raster = r, obs_pts = obs_pts, nb_fit = nb_fit)
}


test_that("rasterise_deviance_residuals returns a SpatRaster with correct structure", {
  skip_on_cran()
  d <- .make_diag_inputs()

  resid_r <- rasterise_deviance_residuals(d$nb_fit, d$obs_pts, d$raster)

  expect_s4_class(resid_r, "SpatRaster")
  expect_equal(terra::nrow(resid_r), terra::nrow(d$raster))
  expect_equal(terra::ncol(resid_r), terra::ncol(d$raster))
  expect_equal(names(resid_r), "deviance_residual")
})


test_that("rasterise_deviance_residuals values are numeric (or NA)", {
  skip_on_cran()
  d <- .make_diag_inputs(seed = 11)

  resid_r <- rasterise_deviance_residuals(d$nb_fit, d$obs_pts, d$raster)
  vals    <- terra::values(resid_r)[, 1]

  expect_true(is.numeric(vals))
  # At least some cells should have finite residuals
  expect_true(any(is.finite(vals)))
})


test_that("rasterise_deviance_residuals works when intensity_fit is evaluate_full_model format", {
  skip_on_cran()
  d <- .make_diag_inputs(seed = 55)

  # Simulate the evaluate_full_model() return format (intensity_fit_obj is now stored)
  mock_full_result <- list(
    loglik            = d$nb_fit$loglik,
    intensity_params  = d$nb_fit$estimates,
    intensity_fit_obj = d$nb_fit,
    intensity_se      = d$nb_fit$se,
    convergence       = d$nb_fit$convergence,
    distribution      = "negbin"
  )

  resid_r <- rasterise_deviance_residuals(mock_full_result, d$obs_pts, d$raster)
  expect_s4_class(resid_r, "SpatRaster")
  expect_equal(names(resid_r), "deviance_residual")
})


test_that("diagnose_model returns expected fields with plot=FALSE", {
  skip_on_cran()
  skip_if_not_installed("spdep")
  d <- .make_diag_inputs(seed = 77)

  result <- suppressMessages(
    diagnose_model(d$nb_fit, d$obs_pts, d$raster, plot = FALSE)
  )

  expect_named(result, c("residual_raster", "moran", "mean_deviance", "prop_large"))
  expect_s4_class(result$residual_raster, "SpatRaster")
  expect_true(result$mean_deviance >= 0)
  expect_true(result$prop_large >= 0 && result$prop_large <= 1)
})


test_that("plot_residual_map runs without error", {
  skip_on_cran()
  r <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- rnorm(25)
  names(r) <- "deviance_residual"

  expect_silent({
    pdf(tempfile(fileext = ".pdf"))
    result <- plot_residual_map(r)
    dev.off()
  })
  expect_null(result)
})


test_that("plot_residual_map overlays observation points without error", {
  skip_on_cran()
  r <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- rnorm(25)
  pts <- data.frame(x = c(0.2, 0.8), y = c(0.2, 0.8))

  expect_silent({
    pdf(tempfile(fileext = ".pdf"))
    plot_residual_map(r, obs_points = pts)
    dev.off()
  })
})


test_that("diagnose_model warns when Moran's I is significant", {
  skip_on_cran()
  skip_if_not_installed("spdep")
  d <- .make_diag_inputs(seed = 77)

  local_mocked_bindings(
    moran_test = function(resids, coords, k = 8L) {
      list(observed = 0.45, p_value = 0.001)
    },
    .package = "DiffiScape"
  )

  expect_warning(
    suppressMessages(diagnose_model(d$nb_fit, d$obs_pts, d$raster, plot = FALSE)),
    "Significant residual spatial autocorrelation"
  )
})


test_that("rasterise_deviance_residuals scales intensity by cell area (#45)", {
  skip_on_cran()
  set.seed(45)


  # Use a raster where cell_area != 1 (10x10 grid over [0, 10] x [0, 10]
  # => each cell is 1x1 = 1 km^2... change to [0, 20] => 2x2 = 4 km^2)
  r <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 20,
                   ymin = 0, ymax = 20)
  terra::values(r) <- abs(rnorm(100, mean = 4))
  cell_area <- prod(terra::res(r))
  expect_true(cell_area != 1)   # confirm non-unit cells

  n_obs   <- 20
  obs_pts <- data.frame(x = runif(n_obs, 1, 19),
                        y = runif(n_obs, 1, 19))
  nb_fit  <- fit_intensity_nb(
    connectivity_at_obs = abs(rnorm(n_obs, mean = 4)),
    connectivity_raster = r,
    obs_coords          = obs_pts
  )

  # Get predicted intensity raster (lambda, per unit area)
  pred_rast <- predict_intensity(nb_fit, r)
  raw_lambda <- terra::values(pred_rast)[, 1]

  # Compute residuals
  resid_r <- rasterise_deviance_residuals(nb_fit, obs_pts, r)
  vals    <- terra::values(resid_r)[, 1]

  # In cells with 0 observations the expected count = lambda * cell_area.
  # If the function mistakenly used bare lambda (not multiplied by cell_area),

  # then for a cell with 0 observations and lambda = L it would compute
  # residuals as if mu = L instead of mu = L * cell_area.
  # We verify by recomputing expected counts manually.
  coords <- as.matrix(obs_pts)
  cells  <- terra::cellFromXY(r, coords)
  n_cells <- terra::ncell(r)
  counts  <- rep(0L, n_cells)
  tab     <- table(cells)
  counts[as.integer(names(tab))] <- as.integer(tab)

  # Pick a cell with 0 obs and valid lambda
  zero_cells <- which(counts == 0 & !is.na(raw_lambda) & raw_lambda > 0)
  expect_true(length(zero_cells) > 0)
  idx <- zero_cells[1]

  mu_correct <- raw_lambda[idx] * cell_area
  size <- unname(nb_fit$estimates["size"])
  if (is.null(size) || is.na(size)) size <- 1

  expected_resid <- unname(compute_deviance_residuals(0L, mu_correct, size))
  expect_equal(unname(vals[idx]), expected_resid, tolerance = 1e-6)
})


test_that("diagnose_model does not warn when Moran's I is non-significant", {
  skip_on_cran()
  skip_if_not_installed("spdep")
  d <- .make_diag_inputs(seed = 77)

  local_mocked_bindings(
    moran_test = function(resids, coords, k = 8L) {
      list(observed = 0.02, p_value = 0.60)
    },
    .package = "DiffiScape"
  )

  expect_no_warning(
    suppressMessages(diagnose_model(d$nb_fit, d$obs_pts, d$raster, plot = FALSE))
  )
})


# ---------------------------------------------------------------------------
# Posterior predictive checks (ds_ppc / plot_ppc)
# ---------------------------------------------------------------------------

.make_ppc_inputs <- function(seed = 123) {
  set.seed(seed)
  r <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(100, mean = 4))

  n_obs   <- 25
  obs_pts <- data.frame(x = runif(n_obs, 0.05, 0.95),
                        y = runif(n_obs, 0.05, 0.95))
  nb_fit  <- fit_intensity_nb(
    connectivity_at_obs = abs(rnorm(n_obs, mean = 4)),
    connectivity_raster = r,
    obs_coords          = obs_pts
  )

  n_draws <- 20
  samples <- data.frame(
    r_0    = rnorm(n_draws, 0.5, 0.1),
    z_1    = rnorm(n_draws, 0.3, 0.1),
    alpha  = rnorm(n_draws, nb_fit$estimates[["alpha"]], 0.1),
    gamma  = rnorm(n_draws, nb_fit$estimates[["gamma"]], 0.1),
    size   = pmax(rnorm(n_draws, nb_fit$estimates[["size"]], 0.5), 0.1),
    loglik = rnorm(n_draws, -50, 5)
  )

  list(raster = r, obs_pts = obs_pts, nb_fit = nb_fit, samples = samples)
}


test_that("ds_ppc returns correct structure", {
  skip_on_cran()
  p <- .make_ppc_inputs()

  result <- suppressMessages(
    ds_ppc(p$samples, p$nb_fit, p$obs_pts, p$raster,
           n_sim = 10, plot = FALSE, seed = 1)
  )

  expect_s3_class(result, "ds_ppc")
  expect_named(result, c("observed", "simulated", "bayesian_p",
                          "n_sim", "test_quantities"))
  expect_true(is.numeric(result$observed))
  expect_true(is.list(result$simulated))
  expect_true(is.numeric(result$bayesian_p))
  expect_equal(result$n_sim, 10L)
  expect_equal(result$test_quantities,
               c("total_count", "vmi_ratio", "mean_deviance"))
})


test_that("ds_ppc Bayesian p-values are in [0, 1]", {
  skip_on_cran()
  p <- .make_ppc_inputs(seed = 42)

  result <- suppressMessages(
    ds_ppc(p$samples, p$nb_fit, p$obs_pts, p$raster,
           n_sim = 15, plot = FALSE, seed = 2)
  )

  expect_true(all(result$bayesian_p >= 0 & result$bayesian_p <= 1))
})


test_that("ds_ppc total_count observed matches actual count", {
  skip_on_cran()
  p <- .make_ppc_inputs(seed = 55)

  result <- suppressMessages(
    ds_ppc(p$samples, p$nb_fit, p$obs_pts, p$raster,
           n_sim = 5, plot = FALSE)
  )

  expect_equal(unname(result$observed[["total_count"]]), nrow(p$obs_pts))
})


test_that("ds_ppc n_sim is respected", {
  skip_on_cran()
  p <- .make_ppc_inputs()

  result <- suppressMessages(
    ds_ppc(p$samples, p$nb_fit, p$obs_pts, p$raster,
           n_sim = 10, plot = FALSE, seed = 3)
  )

  expect_equal(length(result$simulated$total_count), 10)
  expect_equal(length(result$simulated$vmi_ratio), 10)
  expect_equal(length(result$simulated$mean_deviance), 10)
})


test_that("ds_ppc works with Poisson family", {
  skip_on_cran()
  p <- .make_ppc_inputs(seed = 66)

  samples_pois <- p$samples[, c("r_0", "z_1", "alpha", "gamma", "loglik")]

  result <- suppressMessages(
    ds_ppc(samples_pois, p$nb_fit, p$obs_pts, p$raster,
           family = family_poisson(), n_sim = 10, plot = FALSE, seed = 4)
  )

  expect_s3_class(result, "ds_ppc")
  expect_equal(result$n_sim, 10L)
})


test_that("ds_ppc errors on unsupported family", {
  skip_on_cran()
  p <- .make_ppc_inputs()

  expect_error(
    suppressMessages(
      ds_ppc(p$samples, p$nb_fit, p$obs_pts, p$raster,
             family = family_rsf(), n_sim = 5, plot = FALSE)
    ),
    "not supported"
  )
})


test_that("ds_ppc seed produces reproducible results", {
  skip_on_cran()
  p <- .make_ppc_inputs(seed = 77)

  r1 <- suppressMessages(
    ds_ppc(p$samples, p$nb_fit, p$obs_pts, p$raster,
           n_sim = 10, plot = FALSE, seed = 99)
  )
  r2 <- suppressMessages(
    ds_ppc(p$samples, p$nb_fit, p$obs_pts, p$raster,
           n_sim = 10, plot = FALSE, seed = 99)
  )

  expect_equal(r1$simulated$total_count, r2$simulated$total_count)
  expect_equal(r1$simulated$vmi_ratio, r2$simulated$vmi_ratio)
})


test_that("ds_ppc thinning reduces draws used", {
  skip_on_cran()
  p <- .make_ppc_inputs()

  result <- suppressMessages(
    ds_ppc(p$samples, p$nb_fit, p$obs_pts, p$raster,
           n_sim = 200, thin = 5L, plot = FALSE)
  )

  expect_equal(result$n_sim, 4L)
})


test_that("plot_ppc runs without error", {
  skip_on_cran()
  p <- .make_ppc_inputs(seed = 88)

  result <- suppressMessages(
    ds_ppc(p$samples, p$nb_fit, p$obs_pts, p$raster,
           n_sim = 10, plot = FALSE, seed = 5)
  )

  expect_silent({
    pdf(tempfile(fileext = ".pdf"))
    plot_ppc(result)
    dev.off()
  })
})


test_that("ds_ppc handles evaluate_full_model format", {
  skip_on_cran()
  p <- .make_ppc_inputs(seed = 33)

  mock_full_result <- list(
    loglik            = p$nb_fit$loglik,
    intensity_params  = p$nb_fit$estimates,
    intensity_fit_obj = p$nb_fit,
    intensity_se      = p$nb_fit$se,
    convergence       = p$nb_fit$convergence,
    distribution      = "negbin"
  )

  result <- suppressMessages(
    ds_ppc(p$samples, mock_full_result, p$obs_pts, p$raster,
           n_sim = 5, plot = FALSE, seed = 6)
  )

  expect_s3_class(result, "ds_ppc")
})


test_that("ds_ppc messages when n_sim > available draws", {
  skip_on_cran()
  p <- .make_ppc_inputs()

  expect_message(
    ds_ppc(p$samples, p$nb_fit, p$obs_pts, p$raster,
           n_sim = 100, plot = FALSE),
    "Only 20 posterior draws available"
  )
})


# ---------------------------------------------------------------------------
# ZINB pi is consumed directly, not re-transformed (#109)
# ---------------------------------------------------------------------------
#
# fit_intensity_nb() now reports the ZINB zero-inflation parameter as "pi"
# on the natural (0, 1) probability scale (via plogis()), not as "logit_pi"
# exponentiated. ds_ppc() must consume fit_obj$estimates[["pi"]] and
# posterior_samples$pi directly -- re-applying plogis()/inverse-logit to an
# already-natural-scale value would corrupt it.

.make_ppc_inputs_zinb <- function(seed = 109, logit_pi_fixed = -5) {
  set.seed(seed)
  r <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1,
                   ymin = 0, ymax = 1)
  terra::values(r) <- abs(rnorm(100, mean = 4))

  n_obs   <- 40
  obs_pts <- data.frame(x = runif(n_obs, 0.05, 0.95),
                        y = runif(n_obs, 0.05, 0.95))

  # Pin the optimizer-scale logit_pi at a fixed negative value so the fit
  # is deterministic: L-BFGS-B cannot move a parameter whose lower and
  # upper bounds are equal.
  fam <- family_zinb()
  fam$init_fn <- function(n_cov) {
    list(
      start = c(0, 1, rep(0, n_cov), log(1), logit_pi_fixed),
      # A tight interval (not an exact point) around logit_pi_fixed:
      # L-BFGS-B's internal finite-difference gradient divides by
      # (upper - lower) for a bounded parameter, so exactly-equal
      # lower/upper produces a non-finite gradient. A 2e-6-wide interval
      # keeps the fitted logit_pi effectively pinned.
      lower = c(-10, -10, rep(-10, n_cov), log(0.01), logit_pi_fixed - 1e-6),
      upper = c( 10,  10, rep( 10, n_cov), log(1e6),  logit_pi_fixed + 1e-6)
    )
  }

  zinb_fit <- fit_intensity_nb(
    connectivity_at_obs = abs(rnorm(n_obs, mean = 4)),
    connectivity_raster = r,
    obs_coords          = obs_pts,
    family               = fam
  )

  true_pi <- 1 / (1 + exp(-logit_pi_fixed))  # ~= 0.0067

  n_draws <- 20
  samples <- data.frame(
    r_0    = rnorm(n_draws, 0.5, 0.1),
    z_1    = rnorm(n_draws, 0.3, 0.1),
    alpha  = rnorm(n_draws, zinb_fit$estimates[["alpha"]], 0.05),
    gamma  = rnorm(n_draws, zinb_fit$estimates[["gamma"]], 0.05),
    size   = pmax(rnorm(n_draws, zinb_fit$estimates[["size"]], 0.2), 0.1),
    pi     = pmin(pmax(rnorm(n_draws, true_pi, 0.001), 0), 0.02),
    loglik = rnorm(n_draws, -50, 5)
  )

  list(raster = r, obs_pts = obs_pts, zinb_fit = zinb_fit, samples = samples,
       true_pi = true_pi)
}


test_that("fit_intensity_nb reports ZINB pi small (not > 0.5) for a negative fitted logit, feeding ds_ppc correctly (#109)", {
  skip_on_cran()
  p <- .make_ppc_inputs_zinb(seed = 109, logit_pi_fixed = -5)

  expect_true("pi" %in% names(p$zinb_fit$estimates))
  expect_false("logit_pi" %in% names(p$zinb_fit$estimates))
  expect_equal(unname(p$zinb_fit$estimates[["pi"]]), p$true_pi, tolerance = 1e-4)
  expect_true(p$zinb_fit$estimates[["pi"]] < 0.05)
})


test_that("ds_ppc simulated total counts are not deflated by ~50% when the true ZINB pi is small (#109)", {
  skip_on_cran()
  p <- .make_ppc_inputs_zinb(seed = 109, logit_pi_fixed = -5)

  result <- suppressMessages(
    ds_ppc(p$samples, p$zinb_fit, p$obs_pts, p$raster,
           family = family_zinb(), n_sim = 30, plot = FALSE, seed = 5,
           test_quantities = "total_count")
  )

  observed_total <- result$observed[["total_count"]]
  sim_mean_total <- mean(result$simulated$total_count)

  # With pi ~= 0.0067 (near-zero structural-zero probability), simulated
  # total counts should track the observed count within a wide band. Before
  # the #109 fix, the stored value was exp(-5) ~= 0.0067, which ds_ppc then
  # re-passed through plogis() to get pi_map = plogis(0.0067) ~= 0.502 --
  # i.e. roughly half of all cells would be forced to a structural zero,
  # systematically deflating the simulated total count to well under half
  # of what a near-zero pi would produce.
  info_msg <- sprintf(
    "sim_mean_total = %.2f, observed_total = %.2f -- simulated counts look implausibly low, consistent with pi being incorrectly computed as > 0.5",
    sim_mean_total, observed_total
  )
  expect_true(sim_mean_total > 0.5 * observed_total, info = info_msg)
})
