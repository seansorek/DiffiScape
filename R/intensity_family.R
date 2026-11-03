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

# ---- extra-parameter scale transforms --------------------------------------
#
# "Extra" distributional parameters (e.g. log_nb_theta, logit_pi) are
# estimated on an unconstrained optimizer scale and must be mapped to a
# natural/reporting scale BY NAME, not uniformly via exp(). Exponentiation
# is only correct for log-scale parameters (log_nb_theta -> size); a
# logit-scale parameter (logit_pi -> pi) must go through plogis(), or the
# stored value is silently squashed into (0, Inf) instead of (0, 1) and any
# downstream consumer that re-applies plogis() sees a nonsensical value
# (always > 0.5, since exp() of anything is positive). See #109.

#' @keywords internal
#' @noRd
.extra_param_to_natural <- function(raw_name, raw_value) {
  if (identical(raw_name, "logit_pi")) {
    stats::plogis(raw_value)
  } else {
    exp(raw_value)
  }
}

# Delta-method scale factor mapping an optimizer-scale SE to a natural-scale
# SE: se_natural = se_raw * scale_factor. `natural_value` is the
# already-transformed (natural-scale) point estimate -- for logit_pi this is
# d(plogis)/dx = plogis(x) * (1 - plogis(x)) = pi * (1 - pi); for a
# log-scale parameter this is d(exp)/dx = exp(x), i.e. the natural value
# itself.
#' @keywords internal
#' @noRd
.extra_param_se_scale <- function(raw_name, natural_value) {
  if (identical(raw_name, "logit_pi")) {
    natural_value * (1 - natural_value)
  } else {
    natural_value
  }
}


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
  .python_intensity_family("negbin")
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
  .python_intensity_family("poisson")
}


#' Gaussian intensity family
#'
#' For continuous response data (e.g. density estimates).  Uses a
#' response-scale Gaussian likelihood, \eqn{y \sim \text{Normal}(\mu, \sigma)},
#' with known or estimated standard deviation \eqn{\sigma}.  `mu_obs` from
#' [compute_intensity()] is the fitted mean response (not a log-scale
#' median), and residuals are computed on the response scale
#' (`mu_obs - y_obs`), so finite negative or zero responses are valid
#' observations -- they are not clamped or otherwise transformed.
#'
#' @param known_sd Fixed standard deviation (on the response scale).  If
#'   `NULL` (default), `log_sd` is estimated as an extra parameter.
#' @return An [intensity_family] object.
#' @export
family_gaussian <- function(known_sd = NULL) {
  .python_intensity_family("gaussian", known_sd = known_sd)
}


#' Zero-inflated Negative Binomial intensity family
#'
#' Extends the aggregate NB-PPP model with an extra zero-inflation parameter
#' \eqn{\pi = \text{logit}^{-1}(\text{logit\_pi})}, where with
#' probability \eqn{\pi} the observed count \eqn{n_\text{obs}} is a
#' structural zero (independent of the intensity surface), and with
#' probability \eqn{1 - \pi} it is drawn from the ordinary NB-PPP model
#' used by [family_negbin()]:
#' \deqn{
#'   P(N = n) = \begin{cases}
#'     \pi + (1 - \pi)\,\text{NB}(0 \mid \Lambda, \theta) & n = 0 \\
#'     (1 - \pi)\,\text{NB}(n \mid \Lambda, \theta) & n > 0
#'   \end{cases}
#' }
#' where \eqn{\Lambda = \sum_j w_j \lambda_j} is the (non-inflated) mean
#' intensity integral and \eqn{\theta} is the NB size parameter. Critically,
#' \eqn{\Lambda} and the point-location density are *not* scaled by
#' \eqn{(1 - \pi)}: only the marginal probability of the aggregate count
#' \eqn{n_\text{obs}} is zero-inflated. This keeps \eqn{\pi} from being
#' absorbed into the intercept \eqn{\alpha} (see #91) -- with a single
#' realised \eqn{n_\text{obs} > 0} (the typical case), \eqn{\pi} enters the
#' log-likelihood only via an additive \eqn{\log(1 - \pi)} term, so it is
#' identified in principle but only weakly so from a single aggregate
#' count; expect it to be pinned near its lower bound unless the data
#' includes genuine zero-count replicates.
#'
#' @return An [intensity_family] object.
#' @export
family_zinb <- function() {
  .python_intensity_family("zinb")
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
  .python_intensity_family("rsf")
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
            is.finite(background_weight), background_weight > 0)
  .python_intensity_family("rsp", background_weight = background_weight)
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
#' family creation time. `intensity_config$integration_subsample` is not
#' consulted on the `available_connectivity` selection-mode path (see
#' `fit_intensity_nb()`), so it has no effect here -- stratum integrity is
#' instead protected by `evaluate_full_model()` and `laplace_resistance()`
#' refusing (via `stop()`) to silently drop `obs_points`/`available_points`
#' that fall outside the connectivity raster's valid mask. Ensure your
#' points and their matching `stratum_ids_used`/`stratum_ids_avail` are
#' pre-filtered to the valid mask before fitting.
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

  }

  .python_intensity_family("clogit", stratum_ids_used = stratum_ids_used,
                            stratum_ids_avail = stratum_ids_avail)
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
