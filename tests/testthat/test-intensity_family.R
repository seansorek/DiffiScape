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

test_that("family_gaussian recovers known parameters (response-scale)", {
  set.seed(42)
  n <- 500
  z_obs <- rnorm(n)
  z_int <- rnorm(200)

  true_alpha <- 1.0
  true_gamma <- 0.5
  true_sd    <- 0.3

  # Response-scale Gaussian: y_obs ~ Normal(mu_obs, true_sd), where
  # mu_obs = exp(alpha + gamma * z) is the fitted MEAN response (#110).
  mu_obs <- exp(true_alpha + true_gamma * z_obs)
  y_obs  <- rnorm(n, mean = mu_obs, sd = true_sd)

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

test_that("family_gaussian negloglik uses response-scale residuals, not log-scale", {
  # For mu_obs == y_obs everywhere, the response-scale residual is exactly
  # zero (regardless of the response's magnitude), so the negloglik should
  # equal the pure normalising-constant term. A log-scale (lognormal)
  # residual would NOT be zero except when mu_obs == y_obs on the log
  # scale too -- this test also implicitly distinguishes the two by using
  # values where mu_obs == y_obs exactly on the response scale.
  fam <- family_gaussian(known_sd = 1)
  z_obs <- c(0, 0, 0)
  # alpha = 0, gamma = 0 -> mu_obs = exp(0) = 1 for all obs
  y_obs <- c(1, 1, 1)
  nll <- fam$negloglik_fn(
    theta       = c(0, 0),
    z_obs       = z_obs,
    z_int       = rnorm(10),
    int_weights = rep(0.01, 10),
    obs_weights = rep(1, 3),
    y_obs       = y_obs
  )
  expected <- 3 * (log(1) + 0.5 * log(2 * pi))  # resid = 0 for all obs
  expect_equal(nll, expected, tolerance = 1e-8)
})

test_that("family_gaussian accepts zero and negative responses without clamping", {
  # A lognormal model would clamp non-positive responses to 1e-300 (or
  # reject them outright); the response-scale Gaussian must treat them as
  # ordinary finite observations (#110).
  fam <- family_gaussian(known_sd = 1)
  z_obs <- c(0, 0, 0)
  y_obs <- c(-5, 0, 3)  # negative, zero, and positive responses

  nll <- fam$negloglik_fn(
    theta       = c(0, 0),
    z_obs       = z_obs,
    z_int       = rnorm(10),
    int_weights = rep(0.01, 10),
    obs_weights = rep(1, 3),
    y_obs       = y_obs
  )
  expect_true(is.finite(nll))

  # mu_obs = exp(0) = 1 for all three observations; residuals are
  # (mu_obs - y_obs) = (6, 1, -2), matching an ordinary Gaussian on the
  # response scale -- not the huge values a naive log(pmax(y, 1e-300))
  # transform would produce for the non-positive entries.
  mu_obs <- 1
  resid  <- mu_obs - y_obs
  expected <- 0.5 * sum(resid^2) + 3 * (log(1) + 0.5 * log(2 * pi))
  expect_equal(nll, expected, tolerance = 1e-8)
})

test_that("family_gaussian negloglik differs from a lognormal-style computation", {
  # Sanity check that the implemented likelihood is NOT the lognormal
  # residual formula from the pre-fix implementation (#110).
  fam <- family_gaussian(known_sd = 1)
  z_obs <- c(0.5, -0.3, 1.2)
  y_obs <- c(2, 4, 1.5)

  theta <- c(0.2, 0.7)
  nll_response <- fam$negloglik_fn(
    theta       = theta,
    z_obs       = z_obs,
    z_int       = rnorm(10),
    int_weights = rep(0.01, 10),
    obs_weights = rep(1, 3),
    y_obs       = y_obs
  )

  # Reconstruct what the old lognormal-residual formula would have given.
  mu_obs   <- exp(theta[1] + theta[2] * z_obs)
  resid_ln <- log(pmax(mu_obs, 1e-300)) - log(pmax(y_obs, 1e-300))
  nll_lognormal <- 0.5 * sum((resid_ln / 1)^2) + 3 * log(1) + 3 * 0.5 * log(2 * pi)

  expect_false(isTRUE(all.equal(nll_response, nll_lognormal)))
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

test_that("family_gaussian deviance_residuals are response-scale (observed - fitted)", {
  # A lognormal-style deviance residual would be log(observed) - log(fitted),
  # which is undefined/clamped for non-positive observed values. The
  # response-scale residual must handle zero/negative observations exactly
  # like an ordinary Gaussian (#110).
  fam <- family_gaussian()
  y   <- c(-3, 0, 4)
  mu  <- c(1, 1, 1)
  dr  <- fam$deviance_residuals_fn(y, mu, extra_params = list())
  expect_equal(dr, y - mu)
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

test_that("family_zinb negloglik_fn is sensitive to logit_pi at matched alpha (#91)", {
  # Regression test for #91: (1 - pi) previously scaled both the point-log
  # term and the integral term identically, so it cancelled exactly out of
  # the likelihood and logit_pi had zero effect at fixed alpha -- i.e. the
  # negloglik_fn was numerically *constant* in logit_pi. After the fix,
  # pi enters only via an additive log(1 - pi) term on the count marginal,
  # so two different logit_pi values (at matched alpha/gamma/nb_theta) must
  # give two different negloglik values.
  fam <- family_zinb()
  inp <- make_ppp_inputs()

  nll_pi_low <- fam$negloglik_fn(
    theta       = c(-3, 0.5, log(2), -2),   # logit_pi = -2 -> pi ~= 0.12
    z_obs       = inp$z_obs, z_int = inp$z_int,
    int_weights = inp$int_weights, obs_weights = inp$obs_weights
  )
  nll_pi_high <- fam$negloglik_fn(
    theta       = c(-3, 0.5, log(2), 2),    # logit_pi =  2 -> pi ~= 0.88
    z_obs       = inp$z_obs, z_int = inp$z_int,
    int_weights = inp$int_weights, obs_weights = inp$obs_weights
  )

  expect_false(isTRUE(all.equal(nll_pi_low, nll_pi_high)))

  # And the difference must match the closed-form additive offset predicted
  # by the fix: for n_obs > 0, negll(logit_pi) = const - log(1 - pi_val), so
  # negll_high - negll_low == log(1 - pi_low) - log(1 - pi_high) exactly
  # (independent of alpha/gamma/nb_theta/data).
  pi_low  <- 1 / (1 + exp(2))    # pi at logit_pi = -2
  pi_high <- 1 / (1 + exp(-2))   # pi at logit_pi =  2
  expect_equal(nll_pi_high - nll_pi_low,
               log(1 - pi_low) - log(1 - pi_high),
               tolerance = 1e-8)
})

test_that("family_zinb negloglik_fn no longer confounds alpha and logit_pi via a common (1 - pi) scale factor (#91)", {
  # Before the fix, negloglik_fn(alpha, ..., logit_pi) depended on alpha and
  # logit_pi only through the product exp(alpha) * (1 - pi_val): shifting
  # alpha by delta and simultaneously choosing a new logit_pi such that
  # (1 - pi_new) = (1 - pi_old) * exp(-delta) left the log-likelihood
  # exactly unchanged (a flat ridge). After the fix that exact
  # compensation no longer holds.
  fam <- family_zinb()
  inp <- make_ppp_inputs()

  alpha0    <- -3
  logit_pi0 <- 0
  pi0       <- 1 / (1 + exp(-logit_pi0))

  delta      <- 0.7
  alpha1     <- alpha0 + delta
  pi1        <- 1 - (1 - pi0) * exp(-delta)   # the old "compensating" pi
  logit_pi1  <- log(pi1 / (1 - pi1))

  nll0 <- fam$negloglik_fn(
    theta       = c(alpha0, 0.5, log(2), logit_pi0),
    z_obs       = inp$z_obs, z_int = inp$z_int,
    int_weights = inp$int_weights, obs_weights = inp$obs_weights
  )
  nll1 <- fam$negloglik_fn(
    theta       = c(alpha1, 0.5, log(2), logit_pi1),
    z_obs       = inp$z_obs, z_int = inp$z_int,
    int_weights = inp$int_weights, obs_weights = inp$obs_weights
  )

  expect_false(isTRUE(all.equal(nll0, nll1, tolerance = 1e-6)))
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
