# ============================================================================
# Resistance link functions
#
# A `resistance_link` is an S3 object that abstracts the mapping from the
# linear predictor eta = r_0 + sum_k z_k * phi_k(x) to the resistance
# surface R(x).  By swapping link objects the same basis-based
# parameterisation can produce very different resistance surfaces without
# touching the optimiser, Julia solver, or pipeline code.
#
# Each link bundles:
#   forward_fn(eta, R_min, R_max) -> R   (linear predictor -> resistance)
#   inverse_fn(R, R_min, R_max)   -> eta (resistance -> linear predictor)
#   deriv_fn(eta, R_min, R_max)   -> dR/deta (for Enzyme chain-rule)
#   needs_clamp  logical  (FALSE for naturally-bounded links like softplus)
#
# Optionally a link can override the linear predictor itself via:
#   eta_fn(theta, basis_values) -> eta
# This allows non-additive models (interactions, neural nets, etc.).
# When NULL (the default) the standard additive form is used:
#   eta = theta[1] + basis_values %*% theta[-1]
# ============================================================================

# ---- constructor -----------------------------------------------------------

#' Create a resistance link object
#'
#' A `resistance_link` encapsulates the mapping between the linear
#' predictor \eqn{\eta(x)} and the resistance surface \eqn{R(x)}, plus
#' the derivative needed for gradient-based optimisation.
#'
#' @param name Character label (e.g. `"exp"`, `"softplus"`).
#' @param forward_fn Function `(eta, R_min, R_max) -> R`.
#' @param inverse_fn Function `(R, R_min, R_max) -> eta`.
#' @param deriv_fn Function `(eta, R_min, R_max) -> dR/deta`.
#' @param needs_clamp Logical; does the link require external clamping
#'   to `[R_min, R_max]`?  Links that are naturally bounded (e.g.
#'   softplus) can set this to `FALSE`.
#' @param eta_fn Optional function `(theta, basis_values) -> eta`
#'   overriding the default additive linear predictor.  Use this for
#'   non-additive models (tensor products, neural nets, etc.).
#' @return An S3 object of class `"resistance_link"`.
#' @export
resistance_link <- function(name,
                            forward_fn,
                            inverse_fn,
                            deriv_fn,
                            needs_clamp = TRUE,
                            eta_fn      = NULL) {
  stopifnot(is.character(name), length(name) == 1L)
  stopifnot(is.function(forward_fn))
  stopifnot(is.function(inverse_fn))
  stopifnot(is.function(deriv_fn))
  if (!is.null(eta_fn)) stopifnot(is.function(eta_fn))

  structure(
    list(
      name        = name,
      forward_fn  = forward_fn,
      inverse_fn  = inverse_fn,
      deriv_fn    = deriv_fn,
      needs_clamp = needs_clamp,
      eta_fn      = eta_fn
    ),
    class = "resistance_link"
  )
}

#' @export
print.resistance_link <- function(x, ...) {
  cat(sprintf("<resistance_link>  '%s'", x$name))
  if (!x$needs_clamp) cat("  (self-bounded)")
  if (!is.null(x$eta_fn)) cat("  (custom eta)")
  cat("\n")
  invisible(x)
}


# ---- built-in links --------------------------------------------------------

#' Exponential link (default)
#'
#' \eqn{R(x) = \text{clamp}(\exp(\eta), R_\text{min}, R_\text{max})}.
#' This is the original DiffiScape parameterisation.
#'
#' @return A [resistance_link] object.
#' @export
link_exp <- function() {
  resistance_link(
    name = "exp",
    forward_fn = function(eta, R_min = 1, R_max = 5000) {
      pmin(pmax(exp(eta), R_min), R_max)
    },
    inverse_fn = function(R, R_min = 1, R_max = 5000) {
      log(pmin(pmax(R, R_min), R_max))
    },
    deriv_fn = function(eta, R_min = 1, R_max = 5000) {
      R <- exp(eta)
      ifelse(R >= R_min & R <= R_max, R, 0)
    },
    needs_clamp = TRUE
  )
}


#' Softplus link
#'
#' \eqn{R(x) = \log(1 + \exp(\eta)) + R_\text{min}}.
#' Naturally bounded below by `R_min`, smooth everywhere (no clamp
#' discontinuity), and strictly positive.
#'
#' @return A [resistance_link] object.
#' @export
link_softplus <- function() {
  resistance_link(
    name = "softplus",
    forward_fn = function(eta, R_min = 1, R_max = 5000) {
      # Numerically stable softplus
      sp <- ifelse(eta > 20, eta, log1p(exp(eta)))
      pmin(sp + R_min, R_max)
    },
    inverse_fn = function(R, R_min = 1, R_max = 5000) {
      R_shifted <- pmax(R - R_min, 1e-10)
      log(expm1(pmin(R_shifted, 700)))
    },
    deriv_fn = function(eta, R_min = 1, R_max = 5000) {
      sp <- ifelse(eta > 20, eta, log1p(exp(eta)))
      R  <- sp + R_min
      # sigmoid * (R < R_max)
      sig <- 1 / (1 + exp(-eta))
      ifelse(R <= R_max, sig, 0)
    },
    needs_clamp = FALSE
  )
}


#' Power link
#'
#' \eqn{R(x) = \text{clamp}(|\eta|^p, R_\text{min}, R_\text{max})}.
#' Useful when resistance should scale polynomially with the linear
#' predictor.
#'
#' @param p Exponent (default 2).
#' @return A [resistance_link] object.
#' @export
link_power <- function(p = 2) {
  stopifnot(is.numeric(p), length(p) == 1L, p > 0)
  resistance_link(
    name = sprintf("power(%g)", p),
    forward_fn = function(eta, R_min = 1, R_max = 5000) {
      pmin(pmax(abs(eta)^p, R_min), R_max)
    },
    inverse_fn = function(R, R_min = 1, R_max = 5000) {
      R_c <- pmin(pmax(R, R_min), R_max)
      R_c^(1 / p)
    },
    deriv_fn = function(eta, R_min = 1, R_max = 5000) {
      R <- abs(eta)^p
      in_range <- R >= R_min & R <= R_max
      ifelse(in_range, p * abs(eta)^(p - 1) * sign(eta), 0)
    },
    needs_clamp = TRUE
  )
}


#' Identity link
#'
#' \eqn{R(x) = \text{clamp}(\eta, R_\text{min}, R_\text{max})}.
#' Useful when basis functions are already on the resistance scale.
#'
#' @return A [resistance_link] object.
#' @export
link_identity <- function() {
  resistance_link(
    name = "identity",
    forward_fn = function(eta, R_min = 1, R_max = 5000) {
      pmin(pmax(eta, R_min), R_max)
    },
    inverse_fn = function(R, R_min = 1, R_max = 5000) {
      pmin(pmax(R, R_min), R_max)
    },
    deriv_fn = function(eta, R_min = 1, R_max = 5000) {
      ifelse(eta >= R_min & eta <= R_max, 1, 0)
    },
    needs_clamp = TRUE
  )
}


# ---- helpers ---------------------------------------------------------------

#' Compute the linear predictor from parameters and basis values
#'
#' Uses the link's `eta_fn` if present, otherwise the standard additive
#' form `eta = theta[1] + basis_values %*% theta[-1]`.
#'
#' @param link A [resistance_link] object.
#' @param theta Numeric parameter vector.
#' @param basis_values Matrix (cells x K) or [terra::SpatRaster].
#' @return Numeric vector (or SpatRaster) of linear predictor values.
#' @keywords internal
compute_eta <- function(link, theta, basis_values) {
  if (!is.null(link$eta_fn)) {
    return(link$eta_fn(theta, basis_values))
  }
  # Default additive linear predictor
  if (inherits(basis_values, "SpatRaster")) {
    n_basis <- terra::nlyr(basis_values)
    eta <- theta[1]
    z   <- theta[-1]
    for (k in seq_len(n_basis)) {
      eta <- eta + z[k] * basis_values[[k]]
    }
    eta
  } else {
    theta[1] + basis_values %*% theta[-1]
  }
}
