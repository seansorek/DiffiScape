# ============================================================================
# Intensity family objects
#
# An `intensity_family` is an S3 object that abstracts the distributional
# assumption of the point-process intensity model.  By swapping family
# objects, the same optimisation / diagnostic / posterior pipeline can use
# different likelihood functions and residual formulas.
#
# Each family bundles:
#   negloglik_fn(theta, z_obs, z_int, int_weights, obs_weights,
#                cov_obs, cov_int, cov_names) -> scalar
#   deviance_residuals_fn(observed, fitted, extra_params) -> numeric vector
#   init_fn(n_cov) -> list(start, lower, upper)   starting values + bounds
#   n_extra_params  integer  (distribution-specific params beyond alpha/gamma/betas)
#   extra_param_names  character vector
# ============================================================================

# ---- constructor -----------------------------------------------------------

#' Create an intensity family object
#'
#' An `intensity_family` encapsulates the negative log-likelihood,
#' deviance residual formula, and starting-value logic for a specific
#' distributional family used in the point-process intensity model.
#'
#' @param name Character label (e.g. `"negbin"`, `"poisson"`).
#' @param negloglik_fn Function implementing the PPP negative
#'   log-likelihood.  Signature:
#'   `(theta, z_obs, z_int, int_weights, obs_weights, cov_obs, cov_int, cov_names)`
#'   returning a scalar.
#' @param deviance_residuals_fn Function computing per-cell deviance
#'   residuals.  Signature: `(observed, fitted, extra_params)` returning
#'   a numeric vector.
#' @param init_fn Function `(n_cov)` returning a list with elements
#'   `start`, `lower`, `upper` — numeric vectors of starting values and
#'   bounds for [stats::optim()].
#' @param n_extra_params Integer; number of distribution-specific
#'   parameters beyond `alpha, gamma, betas` (e.g. 1 for NB's theta).
#' @param extra_param_names Character vector naming the extra parameters.
#' @return An S3 object of class `"intensity_family"`.
#' @export
intensity_family <- function(name,
                             negloglik_fn,
                             deviance_residuals_fn,
                             init_fn,
                             n_extra_params    = 0L,
                             extra_param_names = character(0)) {
  stopifnot(is.character(name), length(name) == 1L)
  stopifnot(is.function(negloglik_fn))
  stopifnot(is.function(deviance_residuals_fn))
  stopifnot(is.function(init_fn))
  stopifnot(is.integer(as.integer(n_extra_params)))

  structure(
    list(
      name                  = name,
      negloglik_fn          = negloglik_fn,
      deviance_residuals_fn = deviance_residuals_fn,
      init_fn               = init_fn,
      n_extra_params        = as.integer(n_extra_params),
      extra_param_names     = extra_param_names
    ),
    class = "intensity_family"
  )
}

#' @export
print.intensity_family <- function(x, ...) {
  cat(sprintf("<intensity_family>  '%s'", x$name))
  if (x$n_extra_params > 0L) {
    cat(sprintf("  [extra: %s]",
                paste(x$extra_param_names, collapse = ", ")))
  }
  cat("\n")
  invisible(x)
}


# ---- built-in families -----------------------------------------------------

#' Negative Binomial intensity family (default)
#'
#' PPP likelihood with NB adjustment:
#' \deqn{-\ell = -(T_1 - T_2 + A_{\text{NB}})}
#' where \eqn{T_1 = \sum w_i \log \lambda_i}, \eqn{T_2 = \sum w_j \lambda_j},
#' and \eqn{A_{\text{NB}}} is the NB count adjustment involving the size
#' parameter \eqn{\theta}.
#'
#' @return An [intensity_family] object.
#' @export
family_negbin <- function() {
  intensity_family(
    name = "negbin",

    negloglik_fn = function(theta,
                            z_obs, z_int,
                            int_weights, obs_weights,
                            cov_obs   = NULL,
                            cov_int   = NULL,
                            cov_names = character(0)) {
      alpha <- theta[1]
      gamma <- theta[2]
      n_cov <- length(cov_names)
      betas <- if (n_cov > 0) {
        stats::setNames(theta[3:(2 + n_cov)], cov_names)
      } else {
        NULL
      }
      log_nb_theta <- theta[length(theta)]
      nb_theta     <- exp(log_nb_theta)

      lambda_obs <- compute_intensity(z_obs, alpha, gamma, cov_obs, betas)
      lambda_int <- compute_intensity(z_int, alpha, gamma, cov_int, betas)

      term1 <- sum(obs_weights * log(pmax(lambda_obs, 1e-300)))
      term2 <- sum(int_weights * lambda_int)

      n_obs  <- sum(obs_weights)
      nb_adj <- lgamma(n_obs + nb_theta) - lgamma(nb_theta) -
                lgamma(n_obs + 1) +
                nb_theta * log(nb_theta / (nb_theta + term2)) +
                n_obs    * log(term2 / (nb_theta + term2))

      negll <- -(term1 - term2 + nb_adj)
      if (!is.finite(negll)) negll <- 1e15
      negll
    },

    deviance_residuals_fn = function(observed, fitted, extra_params) {
      k  <- extra_params[["size"]]
      if (is.null(k) || is.na(k)) k <- 1
      y  <- observed
      mu <- pmax(fitted, 1e-10)
      t1 <- ifelse(y > 0, y * log(y / mu), 0)
      t2 <- (y + k) * log((y + k) / (mu + k))
      d_sq <- pmax(2 * (t1 - t2), 0)
      sign(y - mu) * sqrt(d_sq)
    },

    init_fn = function(n_cov) {
      # theta = c(alpha, gamma, betas..., log_nb_theta)
      list(
        start = c(0, 1, rep(0, n_cov), log(1)),
        lower = c(-10, -10, rep(-10, n_cov), log(0.01)),
        upper = c( 10,  10, rep( 10, n_cov), log(1e6))
      )
    },

    n_extra_params    = 1L,
    extra_param_names = "log_nb_theta"
  )
}


#' Poisson intensity family
#'
#' PPP likelihood without overdispersion adjustment:
#' \deqn{-\ell = -(T_1 - T_2)}
#' Simpler model suitable when there is no evidence of overdispersion.
#'
#' @return An [intensity_family] object.
#' @export
family_poisson <- function() {
  intensity_family(
    name = "poisson",

    negloglik_fn = function(theta,
                            z_obs, z_int,
                            int_weights, obs_weights,
                            cov_obs   = NULL,
                            cov_int   = NULL,
                            cov_names = character(0)) {
      alpha <- theta[1]
      gamma <- theta[2]
      n_cov <- length(cov_names)
      betas <- if (n_cov > 0) {
        stats::setNames(theta[3:(2 + n_cov)], cov_names)
      } else {
        NULL
      }

      lambda_obs <- compute_intensity(z_obs, alpha, gamma, cov_obs, betas)
      lambda_int <- compute_intensity(z_int, alpha, gamma, cov_int, betas)

      term1 <- sum(obs_weights * log(pmax(lambda_obs, 1e-300)))
      term2 <- sum(int_weights * lambda_int)

      negll <- -(term1 - term2)
      if (!is.finite(negll)) negll <- 1e15
      negll
    },

    deviance_residuals_fn = function(observed, fitted, extra_params) {
      y  <- observed
      mu <- pmax(fitted, 1e-10)
      t1 <- ifelse(y > 0, y * log(y / mu), 0)
      d_sq <- pmax(2 * (t1 - (y - mu)), 0)
      sign(y - mu) * sqrt(d_sq)
    },

    init_fn = function(n_cov) {
      list(
        start = c(0, 1, rep(0, n_cov)),
        lower = c(-10, -10, rep(-10, n_cov)),
        upper = c( 10,  10, rep( 10, n_cov))
      )
    },

    n_extra_params    = 0L,
    extra_param_names = character(0)
  )
}


#' Gaussian intensity family
#'
#' For continuous response data (e.g. density estimates).  Uses a
#' Gaussian likelihood with known or estimated standard deviation.
#'
#' @param known_sd Fixed standard deviation.  If `NULL` (default),
#'   `log_sd` is estimated as an extra parameter.
#' @return An [intensity_family] object.
#' @export
family_gaussian <- function(known_sd = NULL) {
  est_sd <- is.null(known_sd)

  intensity_family(
    name = "gaussian",

    negloglik_fn = function(theta,
                            z_obs, z_int,
                            int_weights, obs_weights,
                            cov_obs   = NULL,
                            cov_int   = NULL,
                            cov_names = character(0)) {
      alpha <- theta[1]
      gamma <- theta[2]
      n_cov <- length(cov_names)
      betas <- if (n_cov > 0) {
        stats::setNames(theta[3:(2 + n_cov)], cov_names)
      } else {
        NULL
      }

      if (est_sd) {
        log_sd <- theta[length(theta)]
        sd_val <- exp(log_sd)
      } else {
        sd_val <- known_sd
      }

      mu_obs <- compute_intensity(z_obs, alpha, gamma, cov_obs, betas)
      # Gaussian likelihood on log-intensity
      resid <- log(pmax(mu_obs, 1e-300)) - log(pmax(z_obs, 1e-300))
      negll <- 0.5 * sum(obs_weights * (resid / sd_val)^2) +
               sum(obs_weights) * log(sd_val) +
               0.5 * sum(obs_weights) * log(2 * pi)

      if (!is.finite(negll)) negll <- 1e15
      negll
    },

    deviance_residuals_fn = function(observed, fitted, extra_params) {
      sign(observed - fitted) * abs(observed - fitted)
    },

    init_fn = function(n_cov) {
      if (est_sd) {
        list(
          start = c(0, 1, rep(0, n_cov), log(1)),
          lower = c(-10, -10, rep(-10, n_cov), log(0.001)),
          upper = c( 10,  10, rep( 10, n_cov), log(100))
        )
      } else {
        list(
          start = c(0, 1, rep(0, n_cov)),
          lower = c(-10, -10, rep(-10, n_cov)),
          upper = c( 10,  10, rep( 10, n_cov))
        )
      }
    },

    n_extra_params    = if (est_sd) 1L else 0L,
    extra_param_names = if (est_sd) "log_sd" else character(0)
  )
}


#' Zero-inflated Negative Binomial intensity family
#'
#' Extends the NB model with an extra zero-inflation parameter
#' \eqn{\pi = \text{logit}^{-1}(\text{logit\_pi})}, where with
#' probability \eqn{\pi} the observation is a structural zero.
#'
#' @return An [intensity_family] object.
#' @export
family_zinb <- function() {
  intensity_family(
    name = "zinb",

    negloglik_fn = function(theta,
                            z_obs, z_int,
                            int_weights, obs_weights,
                            cov_obs   = NULL,
                            cov_int   = NULL,
                            cov_names = character(0)) {
      alpha <- theta[1]
      gamma <- theta[2]
      n_cov <- length(cov_names)
      betas <- if (n_cov > 0) {
        stats::setNames(theta[3:(2 + n_cov)], cov_names)
      } else {
        NULL
      }
      # Last two extra params: log_nb_theta, logit_pi
      logit_pi     <- theta[length(theta)]
      log_nb_theta <- theta[length(theta) - 1L]
      nb_theta     <- exp(log_nb_theta)
      pi_val       <- 1 / (1 + exp(-logit_pi))

      lambda_obs <- compute_intensity(z_obs, alpha, gamma, cov_obs, betas)
      lambda_int <- compute_intensity(z_int, alpha, gamma, cov_int, betas)

      # PPP terms (on non-inflated component)
      term1 <- sum(obs_weights * log(pmax((1 - pi_val) * lambda_obs, 1e-300)))
      term2 <- sum(int_weights * (1 - pi_val) * lambda_int)

      n_obs  <- sum(obs_weights)
      nb_adj <- lgamma(n_obs + nb_theta) - lgamma(nb_theta) -
                lgamma(n_obs + 1) +
                nb_theta * log(nb_theta / (nb_theta + term2)) +
                n_obs    * log(term2 / (nb_theta + term2))

      negll <- -(term1 - term2 + nb_adj)
      if (!is.finite(negll)) negll <- 1e15
      negll
    },

    deviance_residuals_fn = function(observed, fitted, extra_params) {
      # Use NB residuals as approximation for ZINB
      k  <- extra_params[["size"]]
      if (is.null(k) || is.na(k)) k <- 1
      y  <- observed
      mu <- pmax(fitted, 1e-10)
      t1 <- ifelse(y > 0, y * log(y / mu), 0)
      t2 <- (y + k) * log((y + k) / (mu + k))
      d_sq <- pmax(2 * (t1 - t2), 0)
      sign(y - mu) * sqrt(d_sq)
    },

    init_fn = function(n_cov) {
      # theta = c(alpha, gamma, betas..., log_nb_theta, logit_pi)
      list(
        start = c(0, 1, rep(0, n_cov), log(1), 0),
        lower = c(-10, -10, rep(-10, n_cov), log(0.01), -5),
        upper = c( 10,  10, rep( 10, n_cov), log(1e6),   5)
      )
    },

    n_extra_params    = 2L,
    extra_param_names = c("log_nb_theta", "logit_pi")
  )
}


# ---- helpers ---------------------------------------------------------------

#' Resolve a family from a distribution name
#'
#' Converts the legacy `distribution` string to an [intensity_family]
#' object.  Used for backward compatibility.
#'
#' @param family An [intensity_family] object, or `NULL`.
#' @param distribution Character string (`"negbin"`, `"poisson"`,
#'   `"gaussian"`, `"zinb"`).
#' @return An [intensity_family] object.
#' @keywords internal
resolve_family <- function(family = NULL, distribution = "negbin") {
  if (!is.null(family)) {
    if (!inherits(family, "intensity_family")) {
      stop("`family` must be an intensity_family object", call. = FALSE)
    }
    return(family)
  }
  switch(distribution,
    negbin   = family_negbin(),
    poisson  = family_poisson(),
    gaussian = family_gaussian(),
    zinb     = family_zinb(),
    stop("Unknown distribution: ", distribution,
         ". Use an intensity_family object instead.", call. = FALSE)
  )
}
