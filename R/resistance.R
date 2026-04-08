# ============================================================================
# Parameterised resistance surfaces
#
# Model:
#   log R(x) = r_0 + sum_k z_k * phi_k(x)
#   R(x)     = clamp( exp(log R(x)), R_min, R_max )
#
# S3 class hierarchy
#   resistance_model              (base)
#     resistance_model_parametric (log-linear, existing model)
#     resistance_model_custom     (user-supplied function; supports ML methods
#                                  such as IRL, CNNs, GAMs, etc.)
#
# The "custom" pathway gives full freedom to use any model that maps a
# basis SpatRaster to a single-layer resistance SpatRaster.  Because
# dConnectivity/d\theta is available from Enzyme.jl, gradient-based
# optimisation works for any differentiable custom model.  Expert-based
# maps make natural starting points before gradient descent.
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


# ============================================================================
# S3 class: resistance_model
# ============================================================================

#' Create a resistance model object
#'
#' Constructs an S3 object that encapsulates a resistance model together with
#' its basis stack and clamping bounds.  Two model types are supported:
#'
#' * `"parametric"` — the log-linear model
#'   \eqn{\log R(x) = r_0 + \sum_k z_k \phi_k(x)}{log R(x) = r0 + sum z_k * phi_k(x)}
#'   (the same model implemented by [create_resistance_surface()]).
#' * `"custom"` — any user-supplied function
#'   `f(basis_stack, ...)` that returns a single-layer [terra::SpatRaster].
#'   This pathway is designed for machine-learning approaches such as
#'   Inverse Reinforcement Learning (IRL) or convolutional neural networks
#'   (CNNs): because \eqn{d\mathrm{Connectivity}/d\theta} is available via
#'   Enzyme.jl, gradient-based optimisation works for any differentiable
#'   custom model.  Expert-based maps make natural starting points before
#'   gradient descent.
#'
#' @param params
#'   * **parametric**: named numeric vector or list with elements
#'     `r_0, z_1, ..., z_K` (see [create_resistance_surface()]).
#'   * **custom**: a function `f(basis_stack, ...)` returning a
#'     single-layer [terra::SpatRaster] of resistance values.
#' @param basis_stack A [terra::SpatRaster] with *K* layers.
#' @param type `"parametric"` (default) or `"custom"`.
#' @param R_min,R_max Hard clamping bounds applied after exponentiation in
#'   the parametric pathway (default 1 / 5000).  Ignored for custom models
#'   unless the function applies them internally.
#' @param ... Additional metadata stored on the object (e.g. `name`,
#'   `description`, or a reference to an external ML model object).
#' @return An S3 object with classes
#'   `c("resistance_model_<type>", "resistance_model")`.
#' @seealso [predict.resistance_model_parametric()],
#'   [predict.resistance_model_custom()],
#'   [create_resistance_surface()]
#' @export
#' @examples
#' \dontrun{
#'   ## Parametric model
#'   r1 <- terra::rast(nrows = 5, ncols = 5, xmin = 0, xmax = 1,
#'                     ymin = 0, ymax = 1)
#'   terra::values(r1) <- runif(25)
#'   basis <- create_basis_stack(list(a = r1), rescale = FALSE)
#'
#'   m <- resistance_model(c(r_0 = 1, z_1 = 0.5), basis)
#'   print(m)
#'   R <- predict(m)
#'
#'   ## Custom / ML model (e.g. IRL or CNN wrapper)
#'   ml_fn <- function(basis_stack, ...) {
#'     # any model that returns a SpatRaster of resistance values
#'     terra::app(basis_stack[[1]], function(x) exp(x))
#'   }
#'   m_ml <- resistance_model(ml_fn, basis, type = "custom",
#'                            name = "IRL model v1")
#'   R_ml <- predict(m_ml)
#' }
resistance_model <- function(params,
                              basis_stack,
                              type  = c("parametric", "custom"),
                              R_min = 1,
                              R_max = 5000,
                              ...) {

  type <- match.arg(type)
  validate_basis_stack(basis_stack)

  if (type == "parametric") {
    n_basis <- terra::nlyr(basis_stack)
    params  <- .params_to_vector(params, n_basis)   # normalise to numeric vector
  } else {
    if (!is.function(params)) {
      stop(
        "For type = 'custom', `params` must be a function ",
        "with signature f(basis_stack, ...) returning a SpatRaster.",
        call. = FALSE
      )
    }
  }

  structure(
    list(
      params      = params,
      basis_stack = basis_stack,
      type        = type,
      R_min       = R_min,
      R_max       = R_max,
      extra       = list(...)
    ),
    class = c(paste0("resistance_model_", type), "resistance_model")
  )
}


#' Print a resistance_model object
#'
#' @param x A [resistance_model] object.
#' @param ... Ignored.
#' @return `x` invisibly.
#' @export
print.resistance_model <- function(x, ...) {
  n_basis <- terra::nlyr(x$basis_stack)
  cat(sprintf("<resistance_model>  [%s]\n", x$type))
  cat(sprintf("  Basis layers : %d  (%s)\n",
              n_basis, paste(names(x$basis_stack), collapse = ", ")))
  cat(sprintf("  Clamping     : [%.4g, %.4g]\n", x$R_min, x$R_max))
  if (x$type == "parametric") {
    nms <- c("r_0", paste0("z_", seq_len(n_basis)))
    cat("  Parameters   :\n")
    for (i in seq_along(nms)) {
      cat(sprintf("    %-8s = %+.4f\n", nms[i], x$params[i]))
    }
  } else {
    cat("  Function     :", deparse(x$params)[1], "\n")
    if (length(x$extra) > 0) {
      cat("  Extra fields :", paste(names(x$extra), collapse = ", "), "\n")
    }
  }
  invisible(x)
}


#' Summarise a resistance_model object
#'
#' Prints the model and, for parametric models, a numerical sensitivity
#' analysis (see [resistance_sensitivity()]).
#'
#' @param object A [resistance_model] object.
#' @param ... Ignored.
#' @return `object` invisibly.
#' @export
summary.resistance_model <- function(object, ...) {
  print(object)
  if (object$type == "parametric") {
    cat("\nSensitivity (delta = 0.1):\n")
    sens <- resistance_sensitivity(object$params, object$basis_stack)
    print(sens, row.names = FALSE, digits = 4)
  }
  invisible(object)
}


#' Compute a resistance surface from a parametric resistance model
#'
#' Calls [create_resistance_surface()] with the parameters and basis stack
#' stored in `object`.
#'
#' @param object A `resistance_model_parametric` object (see
#'   [resistance_model()]).
#' @param basis_stack Optional replacement [terra::SpatRaster].  Supply
#'   this to predict onto a different spatial extent or resolution.
#'   Defaults to the basis stack stored inside `object`.
#' @param return_log Logical; if `TRUE`, return the log-resistance surface
#'   instead of the exponentiated, clamped surface.
#' @param ... Ignored.
#' @return A single-layer [terra::SpatRaster] named `"resistance"` or
#'   `"log_resistance"`.
#' @export
predict.resistance_model_parametric <- function(object,
                                                 basis_stack = NULL,
                                                 return_log  = FALSE,
                                                 ...) {
  bs <- if (is.null(basis_stack)) {
    object$basis_stack
  } else {
    validate_basis_stack(basis_stack)
    basis_stack
  }
  create_resistance_surface(
    params      = object$params,
    basis_stack = bs,
    R_min       = object$R_min,
    R_max       = object$R_max,
    return_log  = return_log
  )
}


#' Compute a resistance surface from a custom resistance model
#'
#' Calls the user-supplied function stored in `object$params`, enabling
#' arbitrary model forms including machine-learning methods such as
#' Inverse Reinforcement Learning (IRL) and convolutional neural networks
#' (CNNs).
#'
#' @param object A `resistance_model_custom` object (see [resistance_model()]).
#' @param basis_stack Optional replacement [terra::SpatRaster].
#'   Defaults to the basis stack stored inside `object`.
#' @param ... Additional arguments forwarded to the custom function.
#' @return A single-layer [terra::SpatRaster] of resistance values as
#'   returned by the custom function.
#' @export
predict.resistance_model_custom <- function(object,
                                             basis_stack = NULL,
                                             ...) {
  bs <- if (is.null(basis_stack)) {
    object$basis_stack
  } else {
    validate_basis_stack(basis_stack)
    basis_stack
  }
  result <- object$params(bs, ...)
  if (!inherits(result, "SpatRaster") || terra::nlyr(result) != 1L) {
    stop(
      "Custom resistance function must return a single-layer SpatRaster.",
      call. = FALSE
    )
  }
  result
}
