test_that("family_rsf() returns a valid intensity_family", {
  fam <- family_rsf()
  expect_s3_class(fam, "intensity_family")
  expect_equal(fam$name, "rsf")
  expect_equal(fam$n_extra_params, 0L)
  expect_false(is.null(fam$param_names_fn))
})

test_that("family_rsf() param_names_fn returns correct names", {
  fam <- family_rsf()
  expect_equal(fam$param_names_fn(character(0)), "gamma")
  expect_equal(fam$param_names_fn(c("forest", "slope")),
               c("gamma", "beta_forest", "beta_slope"))
})

test_that("family_rsf() init_fn returns correct structure", {
  fam <- family_rsf()
  inits <- fam$init_fn(2L)
  expect_length(inits$start, 3L)  # gamma + 2 betas
  expect_length(inits$lower, 3L)
  expect_length(inits$upper, 3L)
})

test_that("family_rsf() negloglik_fn returns finite scalar on synthetic data", {
  fam <- family_rsf()
  set.seed(1)
  z_obs <- rnorm(30, mean = 1)
  z_int <- rnorm(100, mean = 0)
  nll <- fam$negloglik_fn(
    theta = c(0.5),
    z_obs = z_obs, z_int = z_int,
    int_weights = rep(1, 100), obs_weights = rep(1, 30)
  )
  expect_true(is.finite(nll))
  expect_true(is.numeric(nll))
  expect_length(nll, 1L)
})

test_that("family_rsf() negloglik_fn correct for trivial case", {
  fam <- family_rsf()
  # When gamma = 0, all scores are 0; nll = -(0 - n*log(n_avail))
  z_obs <- rep(0, 5)
  z_int <- rep(0, 10)
  nll <- fam$negloglik_fn(
    theta = c(0),
    z_obs = z_obs, z_int = z_int,
    int_weights = rep(1, 10), obs_weights = rep(1, 5)
  )
  expected <- -(0 - 5 * log(10))
  expect_equal(nll, expected, tolerance = 1e-10)
})

test_that("family_rsf() deviance_residuals_fn returns NA", {
  fam <- family_rsf()
  res <- fam$deviance_residuals_fn(1:5, 1:5, list())
  expect_true(all(is.na(res)))
  expect_length(res, 5L)
})

test_that("family_rsp() returns a valid intensity_family", {
  fam <- family_rsp()
  expect_s3_class(fam, "intensity_family")
  expect_equal(fam$name, "rsp")
  expect_equal(fam$n_extra_params, 0L)
  expect_null(fam$param_names_fn)  # uses default alpha/gamma naming
})

test_that("family_rsp() negloglik_fn returns finite scalar", {
  fam <- family_rsp(background_weight = 500)
  set.seed(2)
  z_obs <- rnorm(20, mean = 0.5)
  z_int <- rnorm(80, mean = 0)
  nll <- fam$negloglik_fn(
    theta = c(0, 0.3),
    z_obs = z_obs, z_int = z_int,
    int_weights = rep(1, 80), obs_weights = rep(1, 20)
  )
  expect_true(is.finite(nll))
  expect_true(nll > 0)
})

test_that("family_rsp() background_weight affects negloglik", {
  set.seed(3)
  z_obs <- rnorm(10, mean = 0.5)
  z_int <- rnorm(50, mean = 0)
  fam1 <- family_rsp(background_weight = 1)
  fam2 <- family_rsp(background_weight = 1000)
  nll1 <- fam1$negloglik_fn(c(0, 0.5), z_obs, z_int, rep(1,50), rep(1,10))
  nll2 <- fam2$negloglik_fn(c(0, 0.5), z_obs, z_int, rep(1,50), rep(1,10))
  expect_false(isTRUE(all.equal(nll1, nll2)))
})

test_that("family_clogit() without strata degenerates to family_rsf()", {
  fam_clogit <- family_clogit()
  fam_rsf    <- family_rsf()
  set.seed(4)
  z_obs <- rnorm(10)
  z_int <- rnorm(40)
  nll_clogit <- fam_clogit$negloglik_fn(c(0.5), z_obs, z_int,
                                         rep(1,40), rep(1,10))
  nll_rsf    <- fam_rsf$negloglik_fn(c(0.5), z_obs, z_int,
                                      rep(1,40), rep(1,10))
  expect_equal(nll_clogit, nll_rsf, tolerance = 1e-10)
})

test_that("family_clogit() with strata computes per-stratum likelihood", {
  # 3 strata, each with 1 used + 4 available
  n_strata <- 3L
  n_avail  <- 4L
  set.seed(5)
  z_obs    <- rnorm(n_strata)
  z_int    <- rnorm(n_strata * n_avail)
  ids_used <- rep(1:n_strata, each = 1)
  ids_avail <- rep(1:n_strata, each = n_avail)

  fam <- family_clogit(stratum_ids_used  = ids_used,
                       stratum_ids_avail = ids_avail)

  nll <- fam$negloglik_fn(
    theta = c(0.5),
    z_obs = z_obs, z_int = z_int,
    int_weights = rep(1, n_strata * n_avail), obs_weights = rep(1, n_strata)
  )
  expect_true(is.finite(nll))
  expect_true(nll > 0)

  # Manual computation for stratum 1
  gamma <- 0.5
  f_used_1  <- gamma * z_obs[1]
  f_avail_1 <- gamma * z_int[1:n_avail]
  f_all_1   <- c(f_used_1, f_avail_1)
  nll_s1    <- log(sum(exp(f_all_1))) - f_used_1

  f_used_2  <- gamma * z_obs[2]
  f_avail_2 <- gamma * z_int[(n_avail+1):(2*n_avail)]
  f_all_2   <- c(f_used_2, f_avail_2)
  nll_s2    <- log(sum(exp(f_all_2))) - f_used_2

  f_used_3  <- gamma * z_obs[3]
  f_avail_3 <- gamma * z_int[(2*n_avail+1):(3*n_avail)]
  f_all_3   <- c(f_used_3, f_avail_3)
  nll_s3    <- log(sum(exp(f_all_3))) - f_used_3

  expected_nll <- nll_s1 + nll_s2 + nll_s3
  expect_equal(nll, expected_nll, tolerance = 1e-10)
})

test_that("family_clogit() validates mismatched strata", {
  expect_error(
    family_clogit(stratum_ids_used  = c(1L, 2L, 3L),
                  stratum_ids_avail = c(1L, 2L, 4L)),
    "same unique stratum IDs"
  )
})

test_that("family_clogit() requires both or neither stratum args", {
  expect_error(
    family_clogit(stratum_ids_used = c(1L, 2L)),
    "Both stratum_ids_used and stratum_ids_avail"
  )
})

test_that("family_clogit() validates one used location per stratum", {
  expect_error(
    family_clogit(stratum_ids_used  = c(1L, 1L, 2L),  # two used in stratum 1
                  stratum_ids_avail = c(1L, 1L, 2L, 2L)),
    "exactly one used location"
  )
})

test_that("fit_intensity_nb() works in selection mode with family_rsf()", {
  set.seed(42)
  n_obs  <- 50L
  n_avail <- 200L

  # Synthetic: used locations have higher connectivity
  conn_obs   <- abs(rnorm(n_obs, mean = 2, sd = 0.5))
  conn_avail <- abs(rnorm(n_avail, mean = 1, sd = 0.5))

  obs_coords <- data.frame(x = runif(n_obs), y = runif(n_obs))

  fit <- fit_intensity_nb(
    connectivity_at_obs    = conn_obs,
    connectivity_raster    = NULL,
    obs_coords             = obs_coords,
    family                 = family_rsf(),
    available_connectivity = conn_avail
  )

  expect_true(is.finite(fit$loglik))
  expect_equal(fit$convergence, 0L)
  expect_false("alpha" %in% names(fit$estimates))
  expect_true("gamma" %in% names(fit$estimates))
  # gamma should be positive (used locs have higher connectivity)
  expect_true(fit$estimates["gamma"] > 0)
})

test_that("fit_intensity_selection() convenience alias works", {
  set.seed(7)
  conn_obs   <- abs(rnorm(30, mean = 1.5))
  conn_avail <- abs(rnorm(120, mean = 1.0))
  obs_coords <- data.frame(x = runif(30), y = runif(30))

  fit <- fit_intensity_selection(
    connectivity_at_obs    = conn_obs,
    available_connectivity = conn_avail,
    obs_coords             = obs_coords
  )

  expect_type(fit, "list")
  expect_true(is.finite(fit$loglik))
  expect_true("gamma" %in% names(fit$estimates))
})

test_that("family_rsf() negloglik_fn handles covariates", {
  fam <- family_rsf()
  set.seed(8)
  z_obs <- rnorm(20)
  z_int <- rnorm(60)
  cov_obs <- list(elev = runif(20))
  cov_int <- list(elev = runif(60))
  nll <- fam$negloglik_fn(
    theta = c(0.3, 0.2),
    z_obs = z_obs, z_int = z_int,
    int_weights = rep(1, 60), obs_weights = rep(1, 20),
    cov_obs = cov_obs, cov_int = cov_int, cov_names = "elev"
  )
  expect_true(is.finite(nll))
})

test_that("intensity_family() constructor accepts param_names_fn", {
  pfn <- function(cov_names) c("gamma", paste0("beta_", cov_names))
  fam <- intensity_family(
    name                  = "test",
    negloglik_fn          = function(...) 0,
    deviance_residuals_fn = function(...) NA,
    init_fn               = function(n) list(start = 0, lower = -10, upper = 10),
    param_names_fn        = pfn
  )
  expect_equal(fam$param_names_fn(c("a", "b")), c("gamma", "beta_a", "beta_b"))
})
