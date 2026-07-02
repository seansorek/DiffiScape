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
#' @param param_names_fn Optional function `(cov_names)` returning a
#'   character vector of parameter names for families with non-standard
#'   layouts (e.g., no `alpha`).  `NULL` uses the default
#'   `alpha`/`gamma`/`beta_*` naming.
#' @return An S3 object of class `"intensity_family"`.
#' @export
intensity_family <- function(name,
                             negloglik_fn,
                             deviance_residuals_fn,
                             init_fn,
                             n_extra_params    = 0L,
                             extra_param_names = character(0),
                             param_names_fn    = NULL) {
  stopifnot(is.character(name), length(name) == 1L)
  stopifnot(is.function(negloglik_fn))
  stopifnot(is.function(deviance_residuals_fn))
  stopifnot(is.function(init_fn))
  stopifnot(is.integer(as.integer(n_extra_params)))
  if (!is.null(param_names_fn)) stopifnot(is.function(param_names_fn))

  structure(
    list(
      name                  = name,
      negloglik_fn          = negloglik_fn,
      deviance_residuals_fn = deviance_residuals_fn,
      init_fn               = init_fn,
      n_extra_params        = as.integer(n_extra_params),
      extra_param_names     = extra_param_names,
      param_names_fn        = param_names_fn
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
                            cov_names = character(0),
                            ...) {
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
      nb_adj <- lgamma(n_obs + nb_theta) - lgamma(nb_theta) +
                nb_theta * log(nb_theta / (nb_theta + term2)) +
                n_obs    * log(term2 / (nb_theta + term2))

      # Correct NB-PPP log-likelihood: term1 (conditional point log-lik) plus
      # NB marginal log P(N=n_obs | mu=term2, size=nb_theta), derived from the
      # Cox-process (Gamma-mixed Poisson) representation.  The pmf must NOT
      # include an lgamma(n_obs + 1) factorial term: as in the Poisson PPP
      # likelihood, the unordered point pattern density already cancels it.
      negll <- -(term1 - n_obs * log(pmax(term2, 1e-300)) + nb_adj)
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
                            cov_names = character(0),
                            ...) {
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
                            cov_names = character(0),
                            y_obs     = NULL,
                            ...) {
      if (is.null(y_obs)) {
        stop("family_gaussian requires an observed response vector (y_obs). ",
             "Pass `response` to fit_intensity_nb() or supply y_obs directly.",
             call. = FALSE)
      }
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
      resid <- log(pmax(mu_obs, 1e-300)) - log(pmax(y_obs, 1e-300))
      negll <- 0.5 * sum(obs_weights * (resid / sd_val)^2) +
               sum(obs_weights) * log(sd_val) +
               0.5 * sum(obs_weights) * log(2 * pi)

      if (!is.finite(negll)) negll <- 1e15
      negll
    },

    deviance_residuals_fn = function(observed, fitted, extra_params) {
      log_obs <- log(pmax(observed, 1e-300))
      log_fit <- log(pmax(fitted,   1e-300))
      log_obs - log_fit
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
                            cov_names = character(0),
                            ...) {
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
      nb_adj <- lgamma(n_obs + nb_theta) - lgamma(nb_theta) +
                nb_theta * log(nb_theta / (nb_theta + term2)) +
                n_obs    * log(term2 / (nb_theta + term2))

      negll <- -(term1 - n_obs * log(pmax(term2, 1e-300)) + nb_adj)
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


# ---- selection function families -------------------------------------------

#' Resource Selection Function (RSF) intensity family
#'
#' Exponential RSF fitted via the Poisson/use-availability form.  Does not
#' include a baseline intercept (alpha) — only relative selection is estimated.
#' Requires explicitly-provided available/background locations; pass these via
#' the `available_points` argument of [optimize_resistance()], [ds_optimize()],
#' or [fit_intensity_selection()].
#'
#' The negative log-likelihood is:
#' \deqn{-\ell = -\left(\sum_i f(x_i) - n \cdot \log \sum_j w_j \exp f(x_j)\right)}
#' where \eqn{f(x) = \gamma z + \sum_k \beta_k c_k(x)} and
#' \eqn{n = \sum \text{obs\_weights}}.
#'
#' @return An [intensity_family] object.
#' @export
family_rsf <- function() {
  intensity_family(
    name = "rsf",

    negloglik_fn = function(theta,
                            z_obs, z_int,
                            int_weights, obs_weights,
                            cov_obs   = NULL,
                            cov_int   = NULL,
                            cov_names = character(0),
                            ...) {
      gamma <- theta[1]
      n_cov <- length(cov_names)
      betas <- if (n_cov > 0) {
        stats::setNames(theta[2:(1 + n_cov)], cov_names)
      } else {
        NULL
      }

      f_obs <- gamma * z_obs
      f_int <- gamma * z_int
      if (!is.null(betas)) {
        for (nm in names(betas)) {
          f_obs <- f_obs + betas[[nm]] * cov_obs[[nm]]
          f_int <- f_int + betas[[nm]] * cov_int[[nm]]
        }
      }

      n_obs <- sum(obs_weights)
      # log-sum-exp for numerical stability
      mx <- max(f_int)
      lse <- mx + log(sum(int_weights * exp(f_int - mx)))

      negll <- -(sum(obs_weights * f_obs) - n_obs * lse)
      if (!is.finite(negll)) negll <- 1e15
      negll
    },

    deviance_residuals_fn = function(observed, fitted, extra_params) {
      rep(NA_real_, length(observed))
    },

    init_fn = function(n_cov) {
      # theta = c(gamma, betas...)
      list(
        start = c(0, rep(0, n_cov)),
        lower = c(-10, rep(-10, n_cov)),
        upper = c( 10, rep( 10, n_cov))
      )
    },

    n_extra_params    = 0L,
    extra_param_names = character(0),
    param_names_fn    = function(cov_names) {
      if (length(cov_names) == 0L) return("gamma")
      c("gamma", paste0("beta_", cov_names))
    }
  )
}


#' Resource Selection Probability (RSP / logistic selection) intensity family
#'
#' Logistic regression with a large background weight, implementing the
#' Fithian & Hastie (2013) "infinite weight" trick.  Estimates a selection
#' probability surface.  Includes baseline alpha (log-odds intercept).
#'
#' The negative log-likelihood is:
#' \deqn{-\ell = -\left(\sum_i \log \sigma(f_i) +
#'   W \sum_j w_j \log(1 - \sigma(f_j))\right)}
#' where \eqn{\sigma} is the logistic function and \eqn{W} is the background
#' weight.
#'
#' @param background_weight Numeric scalar; multiplicative weight applied to
#'   background/available locations (default `1000`).  Larger values make the
#'   RSP asymptotically equivalent to the PPP.
#' @return An [intensity_family] object.
#' @export
family_rsp <- function(background_weight = 1000) {
  stopifnot(is.numeric(background_weight), length(background_weight) == 1L,
            background_weight > 0)

  intensity_family(
    name = "rsp",

    negloglik_fn = function(theta,
                            z_obs, z_int,
                            int_weights, obs_weights,
                            cov_obs   = NULL,
                            cov_int   = NULL,
                            cov_names = character(0),
                            ...) {
      alpha <- theta[1]
      gamma <- theta[2]
      n_cov <- length(cov_names)
      betas <- if (n_cov > 0) {
        stats::setNames(theta[3:(2 + n_cov)], cov_names)
      } else {
        NULL
      }

      f_obs <- alpha + gamma * z_obs
      f_int <- alpha + gamma * z_int
      if (!is.null(betas)) {
        for (nm in names(betas)) {
          f_obs <- f_obs + betas[[nm]] * cov_obs[[nm]]
          f_int <- f_int + betas[[nm]] * cov_int[[nm]]
        }
      }

      # Numerically stable log-sigmoid and log(1-sigmoid)
      log_p_obs   <- -log1p(exp(-f_obs))
      log_1mp_int <- -log1p(exp( f_int))

      negll <- -(sum(obs_weights * log_p_obs) +
                 background_weight * sum(int_weights * log_1mp_int))
      if (!is.finite(negll)) negll <- 1e15
      negll
    },

    deviance_residuals_fn = function(observed, fitted, extra_params) {
      rep(NA_real_, length(observed))
    },

    init_fn = function(n_cov) {
      # theta = c(alpha, gamma, betas...)
      list(
        start = c(0, 0, rep(0, n_cov)),
        lower = c(-10, -10, rep(-10, n_cov)),
        upper = c( 10,  10, rep( 10, n_cov))
      )
    },

    n_extra_params    = 0L,
    extra_param_names = character(0)
    # param_names_fn is NULL -> uses default alpha/gamma/betas naming
  )
}


#' Conditional logistic selection family (iSSA / SSA)
#'
#' Conditional logistic regression for paired used-available data, optionally
#' stratified (one used location per stratum, any number of available).
#' When strata are omitted, reduces to a global [family_rsf()].
#'
#' Requires explicitly-provided available locations via the `available_points`
#' argument of [optimize_resistance()], [ds_optimize()], or
#' [fit_intensity_selection()].
#'
#' For stratified data (e.g. step-selection analysis), pass stratum IDs at
#' family creation time and set
#' `intensity_config$integration_subsample = 1` to preserve stratum integrity.
#'
#' @param stratum_ids_used Integer or character vector of length \eqn{n_\text{used}},
#'   giving the stratum each used location belongs to.  Must be provided
#'   together with `stratum_ids_avail`.
#' @param stratum_ids_avail Integer or character vector of length
#'   \eqn{n_\text{avail}}, giving the stratum each available location belongs
#'   to.
#' @return An [intensity_family] object.
#' @export
family_clogit <- function(stratum_ids_used  = NULL,
                          stratum_ids_avail = NULL) {

  use_strata <- !is.null(stratum_ids_used) || !is.null(stratum_ids_avail)

  if (use_strata) {
    if (is.null(stratum_ids_used) || is.null(stratum_ids_avail)) {
      stop("Both stratum_ids_used and stratum_ids_avail must be provided together.",
           call. = FALSE)
    }

    strata_used  <- sort(unique(stratum_ids_used))
    strata_avail <- sort(unique(stratum_ids_avail))
    if (!identical(strata_used, strata_avail)) {
      stop("stratum_ids_used and stratum_ids_avail must share the same unique stratum IDs.",
           call. = FALSE)
    }

    tab_used <- table(stratum_ids_used)
    if (any(tab_used != 1L)) {
      stop("Each stratum must have exactly one used location (stratum_ids_used).",
           call. = FALSE)
    }

    # Precompute index mapping for performance: list of list(used, avail)
    idx_map <- lapply(strata_used, function(s) {
      list(
        used  = which(stratum_ids_used  == s),
        avail = which(stratum_ids_avail == s)
      )
    })
  } else {
    idx_map <- NULL
  }

  intensity_family(
    name = "clogit",

    negloglik_fn = function(theta,
                            z_obs, z_int,
                            int_weights, obs_weights,
                            cov_obs   = NULL,
                            cov_int   = NULL,
                            cov_names = character(0),
                            ...) {
      gamma <- theta[1]
      n_cov <- length(cov_names)
      betas <- if (n_cov > 0) {
        stats::setNames(theta[2:(1 + n_cov)], cov_names)
      } else {
        NULL
      }

      f_obs <- gamma * z_obs
      f_int <- gamma * z_int
      if (!is.null(betas)) {
        for (nm in names(betas)) {
          f_obs <- f_obs + betas[[nm]] * cov_obs[[nm]]
          f_int <- f_int + betas[[nm]] * cov_int[[nm]]
        }
      }

      if (is.null(idx_map)) {
        # Global (no strata): equivalent to family_rsf()
        n_obs <- sum(obs_weights)
        mx  <- max(f_int)
        lse <- mx + log(sum(int_weights * exp(f_int - mx)))
        negll <- -(sum(obs_weights * f_obs) - n_obs * lse)
      } else {
        # Per-stratum conditional logistic
        nll_total <- 0
        for (im in idx_map) {
          f_used  <- f_obs[im$used]
          f_avail <- f_int[im$avail]
          # Denominator includes the used location
          f_all   <- c(f_used, f_avail)
          mx_s    <- max(f_all)
          lse_s   <- mx_s + log(sum(exp(f_all - mx_s)))
          nll_total <- nll_total + (lse_s - f_used)
        }
        negll <- nll_total
      }

      if (!is.finite(negll)) negll <- 1e15
      negll
    },

    deviance_residuals_fn = function(observed, fitted, extra_params) {
      rep(NA_real_, length(observed))
    },

    init_fn = function(n_cov) {
      # theta = c(gamma, betas...)
      list(
        start = c(0, rep(0, n_cov)),
        lower = c(-10, rep(-10, n_cov)),
        upper = c( 10, rep( 10, n_cov))
      )
    },

    n_extra_params    = 0L,
    extra_param_names = character(0),
    param_names_fn    = function(cov_names) {
      if (length(cov_names) == 0L) return("gamma")
      c("gamma", paste0("beta_", cov_names))
    }
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
