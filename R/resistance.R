# ============================================================================
# Parameterised resistance surfaces
#
# Model:
#   log R(x) = r_0 + sum_k z_k * phi_k(x)
#   R(x)     = clamp( exp(log R(x)), R_min, R_max )
# ============================================================================

#' Create a resistance surface from parameters and basis functions
#'
#' Computes the resistance raster using an exponential (log) link:
#' \deqn{\log R(x) = r_0 + \sum_{k=1}^{K} z_k \phi_k(x)}{
#'   log R(x) = r0 + z1*phi1(x) + ... + zK*phiK(x)}
#'
#' @param params Named numeric vector **or** list with elements
#'   `r_0, z_1, z_2, ..., z_K` where *K* = number of basis layers.
#'   The first element is the intercept; remaining elements are
#'   coefficients for each basis function.
#' @param basis_stack A [terra::SpatRaster] with *K* layers (from
#'   [create_basis_stack()]).
#' @param R_min,R_max Numeric.
#'   Hard floor / ceiling applied after exponentiation (default 1 / 5000).
#' @param return_log If `TRUE`, return the log-resistance surface instead.
#' @return A single-layer [terra::SpatRaster] named `"resistance"` or
#'   `"log_resistance"`.
#' @export
#' @examples
#' \dontrun{
#'   R <- create_resistance_surface(
#'     params = c(r_0 = 3, z_1 = -0.5, z_2 = 0.5),
#'     basis_stack = basis
#'   )
#' }
create_resistance_surface <- function(params,
                                       basis_stack,
                                       R_min = 1,
                                       R_max = 5000,
                                       return_log = FALSE) {

  validate_basis_stack(basis_stack)
  n_basis <- terra::nlyr(basis_stack)

  # Normalise params to numeric vector
  theta <- .params_to_vector(params, n_basis)

  r_0 <- theta[1]
  z   <- theta[-1]

  # Linear predictor (log-resistance)
  log_R <- r_0
  for (k in seq_len(n_basis)) {
    log_R <- log_R + z[k] * basis_stack[[k]]
  }

  if (return_log) {
    names(log_R) <- "log_resistance"
    return(log_R)
  }

  R <- terra::app(log_R, function(x) {
    pmin(pmax(exp(x), R_min), R_max)
  })
  names(R) <- "resistance"
  R
}


#' Fast resistance computation on raw values (matrix form)
#'
#' Avoids `SpatRaster` overhead. Useful inside optimisation loops.
#'
#' @param theta Numeric vector `c(r_0, z_1, ..., z_K)`.
#' @param basis_values A matrix with *K* columns (one per basis function),
#'   rows = cells.
#' @param R_min,R_max Clamping bounds.
#' @return Numeric vector of resistance values.
#' @export
quick_resistance <- function(theta, basis_values, R_min = 1, R_max = 5000) {

  n_basis <- ncol(basis_values)
  if (length(theta) != n_basis + 1L) {
    stop(sprintf("theta length (%d) must equal n_basis + 1 (%d).",
                 length(theta), n_basis + 1L), call. = FALSE)
  }

  log_R <- theta[1] + basis_values %*% theta[-1]
  as.vector(pmin(pmax(exp(log_R), R_min), R_max))
}


#' Generate default parameter bounds for the resistance model
#'
#' Returns a named list of `c(lower, upper)` pairs for each parameter.
#'
#' @param n_basis Integer; number of basis functions.
#' @param r0_range Bounds for the intercept (default `c(0, 6)`).
#' @param z_range Bounds for each coefficient (default `c(-3, 3)`).
#' @return Named list of length `n_basis + 1`.
#' @export
get_default_bounds <- function(n_basis,
                               r0_range = c(0, 6),
                               z_range  = c(-3, 3)) {
  bounds <- list(r_0 = r0_range)
  for (k in seq_len(n_basis)) {
    bounds[[paste0("z_", k)]] <- z_range
  }
  bounds
}


#' Convert parameter vector / list / data.frame row to numeric vector
#'
#' @param params Accepted formats: named numeric vector, list, or
#'   one-row data.frame with `r_0, z_1, ...` columns.
#' @param n_basis Expected number of basis functions.
#' @return Numeric vector of length `n_basis + 1`.
#' @keywords internal
.params_to_vector <- function(params, n_basis) {

  expected <- n_basis + 1L

  if (is.data.frame(params)) {
    params <- as.list(params[1, ])
  }

  if (is.list(params)) {
    theta <- c(params$r_0,
               vapply(seq_len(n_basis),
                      function(k) params[[paste0("z_", k)]],
                      numeric(1)))
  } else {
    theta <- as.numeric(params)
  }

  if (length(theta) != expected) {
    stop(sprintf("Expected %d parameters (r_0 + %d z's), got %d.",
                 expected, n_basis, length(theta)), call. = FALSE)
  }
  theta
}


#' Convert parameter vector to named list
#'
#' @param theta Numeric vector `c(r_0, z_1, ..., z_K)`.
#' @param n_basis Number of basis functions *K*.
#' @return Named list.
#' @export
params_vector_to_list <- function(theta, n_basis) {
  if (length(theta) != n_basis + 1L) {
    stop(sprintf("theta length (%d) != n_basis + 1 (%d).",
                 length(theta), n_basis + 1L), call. = FALSE)
  }
  out <- list(r_0 = theta[1])
  for (k in seq_len(n_basis)) {
    out[[paste0("z_", k)]] <- theta[k + 1L]
  }
  out
}


#' Numerical sensitivity analysis for resistance parameters
#'
#' Perturbs each parameter by `delta` and reports the percentage change
#' in resistance across the study area.
#'
#' @param params Parameters (vector, list, or data.frame).
#' @param basis_stack A [terra::SpatRaster].
#' @param delta Perturbation size (default 0.1).
#' @return A data.frame with columns `parameter`, `mean_pct_change`,
#'   `sd_pct_change`, `min_pct_change`, `max_pct_change`.
#' @export
resistance_sensitivity <- function(params, basis_stack, delta = 0.1) {

  validate_basis_stack(basis_stack)
  n_basis <- terra::nlyr(basis_stack)
  theta <- .params_to_vector(params, n_basis)

  base_R  <- create_resistance_surface(theta, basis_stack)
  base_v  <- terra::values(base_R)
  valid   <- !is.na(base_v)

  pnames <- c("r_0", paste0("z_", seq_len(n_basis)))

  rows <- lapply(seq_along(pnames), function(i) {
    tp        <- theta
    tp[i]     <- tp[i] + delta
    R_plus    <- create_resistance_surface(tp, basis_stack)
    plus_v    <- terra::values(R_plus)
    pct       <- (plus_v[valid] - base_v[valid]) / base_v[valid] * 100
    data.frame(
      parameter      = pnames[i],
      mean_pct_change = mean(pct),
      sd_pct_change   = stats::sd(pct),
      min_pct_change  = min(pct),
      max_pct_change  = max(pct),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}
