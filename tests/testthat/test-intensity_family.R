# Tests for R/intensity_family.R

# ---- constructor -----------------------------------------------------------

test_that("intensity_family constructor creates correct S3 object", {
  fam <- intensity_family(
    name                  = "test",
    negloglik_fn          = function(...) 0,
    deviance_residuals_fn = function(observed, fitted, extra_params) rep(0, length(observed)),
    init_fn               = function(n_cov) list(start = 0, lower = -10, upper = 10),
    n_extra_params        = 1L,
    extra_param_names     = "log_theta"
  )
  expect_s3_class(fam, "intensity_family")
  expect_named(fam, c("name", "negloglik_fn", "deviance_residuals_fn",
                       "init_fn", "n_extra_params", "extra_param_names",
                       "param_names_fn"))
  expect_equal(fam$n_extra_params, 1L)
  expect_equal(fam$extra_param_names, "log_theta")
})

test_that("intensity_family rejects non-function arguments", {
  expect_error(
    intensity_family(
      name = "bad", negloglik_fn = 42,
      deviance_residuals_fn = function(...) 0,
      init_fn = function(n_cov) list()
    )
  )
  expect_error(
    intensity_family(
      name = "bad",
      negloglik_fn = function(...) 0,
      deviance_residuals_fn = "not_a_fn",
      init_fn = function(n_cov) list()
    )
  )
})


# ---- helper: synthetic PPP inputs ------------------------------------------

make_ppp_inputs <- function(seed = 1, n_obs = 30, n_int = 100) {
  set.seed(seed)
  list(
    z_obs       = rnorm(n_obs),
    z_int       = rnorm(n_int),
    obs_weights = rep(1, n_obs),
    int_weights = rep(0.01, n_int)
  )
}


# ---- family_negbin ---------------------------------------------------------

test_that("family_negbin returns intensity_family with correct metadata", {
  fam <- family_negbin()
  expect_s3_class(fam, "intensity_family")
  expect_equal(fam$name, "negbin")
  expect_equal(fam$n_extra_params, 1L)
  expect_equal(fam$extra_param_names, "log_nb_theta")
})

test_that("family_negbin negloglik returns finite positive scalar", {
  fam <- family_negbin()
  inp <- make_ppp_inputs()
  # theta = c(alpha, gamma, log_nb_theta)
  nll <- fam$negloglik_fn(
    theta       = c(-3, 0.5, log(2)),
    z_obs       = inp$z_obs,
    z_int       = inp$z_int,
    int_weights = inp$int_weights,
    obs_weights = inp$obs_weights
  )
  expect_true(is.finite(nll))
  expect_true(nll > 0)
})

test_that("family_negbin negloglik is finite for extreme nb_theta", {
  fam <- family_negbin()
  inp <- make_ppp_inputs()

  # Very small nb_theta -> large negative log_nb_theta
  nll_small <- fam$negloglik_fn(
    theta = c(-3, 0.5, log(0.01)),
    z_obs = inp$z_obs, z_int = inp$z_int,
    int_weights = inp$int_weights, obs_weights = inp$obs_weights
  )
  expect_true(is.finite(nll_small))

  # Very large nb_theta -> should approach Poisson
  nll_large <- fam$negloglik_fn(
    theta = c(-3, 0.5, log(1e5)),
    z_obs = inp$z_obs, z_int = inp$z_int,
    int_weights = inp$int_weights, obs_weights = inp$obs_weights
  )
  expect_true(is.finite(nll_large))
})

test_that("family_negbin negloglik matches correct NB-PPP formula for known inputs", {
  fam <- family_negbin()
  # alpha=0, gamma=0: lambda_obs = lambda_int = exp(0) = 1 everywhere
  n_obs_val <- 10
  n_int_val <- 20
  z_obs <- rep(0, n_obs_val)
  z_int <- rep(0, n_int_val)
  int_w <- rep(1, n_int_val)   # term2 = sum(int_w * 1) = 20
  obs_w <- rep(1, n_obs_val)   # n_obs = 10

  log_nb_theta <- log(2)
  nb_theta     <- 2
  term2        <- 20
  n_obs_sum    <- 10

  nb_adj <- lgamma(n_obs_sum + nb_theta) - lgamma(nb_theta) +
    nb_theta * log(nb_theta / (nb_theta + term2)) +
    n_obs_sum * log(term2 / (nb_theta + term2))

  # Correct formula: term1 - n_obs*log(term2) + nb_adj
  # term1 = sum(obs_w * log(1)) = 0
  expected_nll <- -(0 - n_obs_sum * log(term2) + nb_adj)

  nll <- fam$negloglik_fn(
    theta       = c(0, 0, log_nb_theta),
    z_obs       = z_obs, z_int = z_int,
    int_weights = int_w, obs_weights = obs_w
  )
  expect_equal(nll, expected_nll, tolerance = 1e-8)
})

test_that("family_negbin nests family_poisson as nb_theta -> Inf", {
  # As nb_theta -> Inf, NB(mu, size=nb_theta) -> Poisson(mu).
  # So family_negbin log-lik should approach family_poisson log-lik.
  fam_nb  <- family_negbin()
  fam_poi <- family_poisson()
  inp <- make_ppp_inputs(seed = 7)

  theta_base <- c(-3, 0.5)
  nll_poi <- fam_poi$negloglik_fn(
    theta = theta_base,
    z_obs = inp$z_obs, z_int = inp$z_int,
    int_weights = inp$int_weights, obs_weights = inp$obs_weights
  )
  # NB with very large nb_theta should give nearly the same nll
  nll_nb_large <- fam_nb$negloglik_fn(
    theta = c(theta_base, log(1e8)),
    z_obs = inp$z_obs, z_int = inp$z_int,
    int_weights = inp$int_weights, obs_weights = inp$obs_weights
  )
  expect_equal(nll_nb_large, nll_poi, tolerance = 1e-3)
})

test_that("family_negbin deviance_residuals have correct length and are finite", {
  fam <- family_negbin()
  y   <- c(0, 1, 5, 10, 20)
  mu  <- c(0.5, 1.2, 4.8, 11, 18)
  dr  <- fam$deviance_residuals_fn(y, mu, extra_params = list(size = 2))
  expect_length(dr, length(y))
  expect_true(all(is.finite(dr)))
})

test_that("family_negbin deviance_residuals sign matches obs - fitted", {
  fam <- family_negbin()
  y   <- c(1, 10)
  mu  <- c(5, 2)    # first: y < mu -> negative; second: y > mu -> positive
  dr  <- fam$deviance_residuals_fn(y, mu, extra_params = list(size = 1))
  expect_true(dr[1] < 0)
  expect_true(dr[2] > 0)
})

test_that("family_negbin deviance_residuals uses k=1 default when size missing", {
  fam <- family_negbin()
  y   <- c(2, 3)
  mu  <- c(2, 3)
  dr_with_null <- fam$deviance_residuals_fn(y, mu, extra_params = list())
  expect_true(all(is.finite(dr_with_null)))
})

test_that("family_negbin init_fn has correct parameter count", {
  fam <- family_negbin()
  for (n_cov in c(0, 1, 3)) {
    ini <- fam$init_fn(n_cov)
    expected_len <- n_cov + 3  # alpha, gamma, betas..., log_nb_theta
    expect_length(ini$start, expected_len)
    expect_length(ini$lower, expected_len)
    expect_length(ini$upper, expected_len)
    expect_true(all(ini$lower <= ini$start))
    expect_true(all(ini$start <= ini$upper))
  }
})


# ---- family_poisson --------------------------------------------------------

test_that("family_poisson returns intensity_family with correct metadata", {
  fam <- family_poisson()
  expect_s3_class(fam, "intensity_family")
  expect_equal(fam$name, "poisson")
  expect_equal(fam$n_extra_params, 0L)
  expect_length(fam$extra_param_names, 0)
})

test_that("family_poisson negloglik returns finite positive scalar", {
  fam <- family_poisson()
  inp <- make_ppp_inputs()
  nll <- fam$negloglik_fn(
    theta       = c(-3, 0.5),
    z_obs       = inp$z_obs,
    z_int       = inp$z_int,
    int_weights = inp$int_weights,
    obs_weights = inp$obs_weights
  )
  expect_true(is.finite(nll))
  expect_true(nll > 0)
})

test_that("family_poisson negloglik is finite for large counts", {
  fam <- family_poisson()
  # Large z values -> large lambda
  nll <- fam$negloglik_fn(
    theta       = c(0, 0.5),
    z_obs       = rep(10, 20),
    z_int       = rep(10, 50),
    int_weights = rep(0.01, 50),
    obs_weights = rep(1, 20)
  )
  expect_true(is.finite(nll))
})

test_that("family_poisson deviance_residuals have correct length and sign", {
  fam <- family_poisson()
  y   <- c(0, 1, 5, 10)
  mu  <- c(1, 1, 3, 12)
  dr  <- fam$deviance_residuals_fn(y, mu, extra_params = list())
  expect_length(dr, length(y))
  expect_true(all(is.finite(dr)))
  expect_true(dr[3] > 0)   # y=5 > mu=3
  expect_true(dr[4] < 0)   # y=10 < mu=12
})

test_that("family_poisson init_fn has correct parameter count", {
  fam <- family_poisson()
  for (n_cov in c(0, 2)) {
    ini <- fam$init_fn(n_cov)
    expected_len <- n_cov + 2  # alpha, gamma, betas...
    expect_length(ini$start, expected_len)
    expect_length(ini$lower, expected_len)
    expect_length(ini$upper, expected_len)
  }
})


# ---- family_gaussian -------------------------------------------------------

test_that("family_gaussian with known_sd = NULL has 1 extra param", {
  fam <- family_gaussian()
  expect_equal(fam$n_extra_params, 1L)
  expect_equal(fam$extra_param_names, "log_sd")
})

test_that("family_gaussian with known_sd has 0 extra params", {
  fam <- family_gaussian(known_sd = 1.5)
  expect_equal(fam$n_extra_params, 0L)
  expect_length(fam$extra_param_names, 0)
})

test_that("family_gaussian negloglik returns finite value", {
  fam <- family_gaussian()
  set.seed(10)
  z_obs <- rnorm(20)
  y_obs <- exp(abs(rnorm(20, mean = 2)))
  nll <- fam$negloglik_fn(
    theta       = c(0, 1, log(1)),
    z_obs       = z_obs,
    z_int       = rnorm(50),
    int_weights = rep(0.01, 50),
    obs_weights = rep(1, 20),
    y_obs       = y_obs
  )
  expect_true(is.finite(nll))
})

test_that("family_gaussian errors without y_obs", {
  fam <- family_gaussian()
  expect_error(
    fam$negloglik_fn(
      theta       = c(0, 1, log(1)),
      z_obs       = rnorm(10),
      z_int       = rnorm(20),
      int_weights = rep(0.01, 20),
      obs_weights = rep(1, 10)
    ),
    "y_obs"
  )
})

test_that("family_gaussian recovers known parameters", {
  set.seed(42)
  n <- 500
  z_obs <- rnorm(n)
  z_int <- rnorm(200)

  true_alpha <- 1.0
  true_gamma <- 0.5
  true_sd    <- 0.3

  log_mu <- true_alpha + true_gamma * z_obs
  y_obs  <- exp(rnorm(n, mean = log_mu, sd = true_sd))

  fam   <- family_gaussian()
  inits <- fam$init_fn(0)

  opt <- stats::optim(
    par    = inits$start,
    fn     = fam$negloglik_fn,
    method = "L-BFGS-B",
    lower  = inits$lower,
    upper  = inits$upper,
    z_obs  = z_obs, z_int = z_int,
    int_weights = rep(0.01, 200), obs_weights = rep(1, n),
    y_obs  = y_obs,
    cov_obs = NULL, cov_int = NULL, cov_names = character(0)
  )

  expect_equal(opt$par[1], true_alpha, tolerance = 0.15)
  expect_equal(opt$par[2], true_gamma, tolerance = 0.15)
  expect_equal(exp(opt$par[3]), true_sd, tolerance = 0.1)
})

test_that("family_gaussian init_fn includes log_sd when estimating SD", {
  fam <- family_gaussian()    # est_sd = TRUE
  ini <- fam$init_fn(n_cov = 1)
  expect_length(ini$start, 4)  # alpha, gamma, beta, log_sd
  expect_length(ini$lower, 4)
  expect_length(ini$upper, 4)
})

test_that("family_gaussian init_fn excludes log_sd with known_sd", {
  fam <- family_gaussian(known_sd = 2.0)  # est_sd = FALSE
  ini <- fam$init_fn(n_cov = 1)
  expect_length(ini$start, 3)  # alpha, gamma, beta only
})

test_that("family_gaussian deviance_residuals preserve sign", {
  fam <- family_gaussian()
  y   <- c(1, 5)
  mu  <- c(3, 2)
  dr  <- fam$deviance_residuals_fn(y, mu, extra_params = list())
  expect_true(dr[1] < 0)   # y < fitted -> negative
  expect_true(dr[2] > 0)   # y > fitted -> positive
})


# ---- family_zinb -----------------------------------------------------------

test_that("family_zinb returns intensity_family with correct metadata", {
  fam <- family_zinb()
  expect_s3_class(fam, "intensity_family")
  expect_equal(fam$name, "zinb")
  expect_equal(fam$n_extra_params, 2L)
  expect_equal(fam$extra_param_names, c("log_nb_theta", "logit_pi"))
})

test_that("family_zinb negloglik returns finite positive scalar", {
  fam <- family_zinb()
  inp <- make_ppp_inputs()
  # theta = c(alpha, gamma, log_nb_theta, logit_pi)
  nll <- fam$negloglik_fn(
    theta       = c(-3, 0.5, log(2), 0),
    z_obs       = inp$z_obs,
    z_int       = inp$z_int,
    int_weights = inp$int_weights,
    obs_weights = inp$obs_weights
  )
  expect_true(is.finite(nll))
  expect_true(nll > 0)
})

test_that("family_zinb mixing weight pi is always in [0, 1]", {
  # This tests the internal computation; verify via negloglik being finite
  # for extreme logit_pi values
  fam <- family_zinb()
  inp <- make_ppp_inputs()
  for (logit_pi in c(-10, 0, 10)) {
    nll <- fam$negloglik_fn(
      theta       = c(-3, 0.5, log(2), logit_pi),
      z_obs       = inp$z_obs, z_int = inp$z_int,
      int_weights = inp$int_weights, obs_weights = inp$obs_weights
    )
    expect_true(is.finite(nll),
                label = paste("ZINB negloglik finite at logit_pi =", logit_pi))
  }
})

test_that("family_zinb negloglik handles structural zeros", {
  fam <- family_zinb()
  # When pi ~= 1 (high zero-inflation), structural zeros shouldn't cause NaN
  nll <- fam$negloglik_fn(
    theta       = c(-3, 0.5, log(2), 5),  # logit_pi=5 -> pi ~= 0.993
    z_obs       = rep(0, 10),
    z_int       = rnorm(20),
    int_weights = rep(0.01, 20),
    obs_weights = rep(1, 10)
  )
  expect_true(is.finite(nll))
})

test_that("family_zinb deviance_residuals have correct length and finiteness", {
  fam <- family_zinb()
  y   <- c(0, 1, 5, 10)
  mu  <- c(0.5, 1, 4, 12)
  dr  <- fam$deviance_residuals_fn(y, mu, extra_params = list(size = 1))
  expect_length(dr, length(y))
  expect_true(all(is.finite(dr)))
})

test_that("family_zinb init_fn has correct parameter count", {
  fam <- family_zinb()
  for (n_cov in c(0, 2)) {
    ini <- fam$init_fn(n_cov)
    expected_len <- n_cov + 4  # alpha, gamma, betas..., log_nb_theta, logit_pi
    expect_length(ini$start, expected_len)
    expect_length(ini$lower, expected_len)
    expect_length(ini$upper, expected_len)
    expect_true(all(ini$lower <= ini$start))
    expect_true(all(ini$start <= ini$upper))
  }
})


# ---- resolve_family --------------------------------------------------------

test_that("resolve_family dispatches all 4 string names correctly", {
  expect_equal(DiffiScape:::resolve_family(distribution = "negbin")$name, "negbin")
  expect_equal(DiffiScape:::resolve_family(distribution = "poisson")$name, "poisson")
  expect_equal(DiffiScape:::resolve_family(distribution = "gaussian")$name, "gaussian")
  expect_equal(DiffiScape:::resolve_family(distribution = "zinb")$name, "zinb")
})

test_that("resolve_family returns family object as-is when provided", {
  fam    <- family_poisson()
  result <- DiffiScape:::resolve_family(family = fam)
  expect_identical(result, fam)
})

test_that("resolve_family throws informative error for unknown distribution", {
  expect_error(DiffiScape:::resolve_family(distribution = "gamma"),
               "Unknown distribution")
})

test_that("resolve_family throws error when family is not intensity_family", {
  expect_error(DiffiScape:::resolve_family(family = list(name = "fake")),
               "intensity_family")
})
