# ============================================================================
# Intensity models for point-process likelihood
#
# Supports:
#   - Negative Binomial PPP  (parametric: alpha + gamma * z(x) + beta * cov)
#   - GAM via mgcv::bam()    (smooth / linear with optional spatial RE)
#
# Model (parametric):
#   log lambda(x) = alpha + gamma * z(x) + sum_j beta_j * cov_j(x)
#   where z(x) = standardised log-connectivity
#
# PPP log-likelihood (quadrature approximation):
#   log L = sum_i log lambda(s_i) - integral lambda(x) dx
# ============================================================================

# ----------- configuration constants ----------------------------------------

#' Default intensity model configuration
#'
#' Returns a named list of tuning knobs for the point-process intensity model
#' used by [fit_intensity_nb()] and [fit_intensity_gam()].  Modify individual
#' elements and pass the result via the `intensity_config` argument of
#' [optimize_resistance()], [evaluate_full_model()], or [diffiscape()].
#'
#' @section Connectivity scaling:
#' \describe{
#'   \item{`min_connectivity`}{Numeric. Connectivity values below this
#'     threshold are clipped to zero before standardisation. Default: `0`.}
#'   \item{`c_scale`}{Numeric or `NULL`. Scale divisor applied as
#'     `log1p(C / c_scale)` before z-scoring. `NULL` uses the median of
#'     positive observed connectivity values. Default: `NULL`.}
#'   \item{`family`}{An [intensity_family] object, or `NULL` (inferred from
#'     the optimizer config's `distribution` field). Default: `NULL`.}
#' }
#'
#' @section GAM basis dimensions:
#' \describe{
#'   \item{`k_connectivity`}{Integer. Basis dimension (number of spline knots)
#'     for the connectivity smooth. Increase for highly non-linear responses.
#'     Default: `10`.}
#'   \item{`k_covariate`}{Integer. Basis dimension per covariate smooth.
#'     Default: `8`.}
#'   \item{`k_spatial`}{Integer. Basis dimension for the 2-D spatial random
#'     effect (only used when `include_spatial_re = TRUE`). Default: `30`.}
#' }
#'
#' @section GAM options:
#' \describe{
#'   \item{`covariate_type`}{Character. `"smooth"` (penalised spline term,
#'     default) or `"linear"` for fixed-slope covariate effects.}
#'   \item{`include_spatial_re`}{Logical. Add a 2-D spatial random effect
#'     `te(x, y)` to absorb unmeasured spatial structure. Default: `FALSE`.}
#'   \item{`spatial_bs`}{Character. Marginal basis for `te(x, y)`, passed
#'     directly to `mgcv`. Default: `"cr"` (cubic regression spline).}
#'   \item{`spatial_tensor`}{Logical. Use a tensor-product `te(x, y)` rather
#'     than an isotropic `s(x, y)` for the spatial term. Default: `TRUE`.}
#' }
#'
#' @section Quadrature:
#' \describe{
#'   \item{`integration_subsample`}{Numeric in `(0, 1]`. Fraction of raster
#'     cells used to approximate the PPP log-likelihood integral.  Smaller
#'     values speed up fitting at the cost of quadrature accuracy.
#'     Default: `0.25`.}
#'   \item{`seed`}{Integer. RNG seed for integration subsampling.
#'     Default: `42`.}
#' }
#'
#' @return A named list of intensity model settings.
#' @seealso [fit_intensity_nb()], [fit_intensity_gam()], [diffiscape()]
#' @export
default_intensity_config <- function() {
  list(
    # Connectivity scaling
    min_connectivity     = 0,
    c_scale              = NULL,   # NULL --> median(C_obs)

    # Intensity family (NULL -> infer from distribution for backward compat)
    family               = NULL,

    # GAM knots
    k_connectivity = 10L,
    k_covariate    = 8L,
    k_spatial      = 30L,

    # GAM options
    covariate_type      = "smooth",   # "smooth" or "linear"
    include_spatial_re   = FALSE,
    spatial_bs           = "cr",      # marginal basis for te(x,y)
    spatial_tensor       = TRUE,

    # Integration subsampling
    integration_subsample = 0.25,
    seed                  = 42L
  )
}


# ======================= CONNECTIVITY STANDARDISATION =======================

#' Standardise connectivity values
#'
#' Applies `log1p(C / c_scale)` then z-scores using mean/sd computed from
#' `C_obs`.  The same `c_scale`, `mu`, and `sigma` should be applied to
#' integration-grid values.
#'
#' @param C_obs Numeric vector of connectivity at observation locations.
#' @param C_int Numeric vector of connectivity at integration locations.
#' @param c_scale Scale divisor.
#'   If `NULL` (default) the median of positive `C_obs` is used.
#' @return A list with `z_obs`, `z_int`, `c_scale`, `mu`, `sigma`.
#' @export
standardise_connectivity <- function(C_obs, C_int, c_scale = NULL) {

  C_obs <- pmax(C_obs, 0)
  C_int <- pmax(C_int, 0)

  if (is.null(c_scale)) {
    pos <- C_obs[C_obs > 0]
    c_scale <- if (length(pos) > 0) stats::median(pos) else 1
  }

  log_obs <- log1p(C_obs / c_scale)
  mu      <- mean(log_obs)
  sigma   <- stats::sd(log_obs)
  if (sigma < 1e-10) sigma <- 1

  z_obs <- (log_obs - mu) / sigma
  z_int <- (log1p(C_int / c_scale) - mu) / sigma

  list(z_obs = z_obs, z_int = z_int,
       c_scale = c_scale, mu = mu, sigma = sigma)
}


#' Residualise connectivity against local covariates
#'
#' Regresses standardised log-connectivity on supplied covariates at
#' observation points, then applies the fitted regression to both
#' observation and integration grids.  The residual captures the
#' non-local (network-structure) component of connectivity.
#'
#' @param z_obs,z_int Standardised connectivity vectors.
#' @param covariates_obs,covariates_int Named lists of covariate vectors
#'   at observation / integration points (same names required).
#' @return A list with `z_obs_resid`, `z_int_resid`, `aux_coefs`,
#'   `aux_r2`, `aux_model`.
#' @export
residualise_connectivity <- function(z_obs, z_int,
                                     covariates_obs,
                                     covariates_int) {

  # Build data.frames
  df_obs <- data.frame(z = z_obs)
  df_int <- data.frame(dummy = seq_along(z_int))
  for (nm in names(covariates_obs)) {
    df_obs[[nm]] <- covariates_obs[[nm]]
    df_int[[nm]] <- covariates_int[[nm]]
  }

  fmla <- stats::as.formula(paste("z ~",
    paste(setdiff(names(df_obs), "z"), collapse = " + ")))
  fit  <- stats::lm(fmla, data = df_obs)
  smry <- summary(fit)

  z_obs_resid <- stats::residuals(fit)
  z_int_resid <- z_int - stats::predict(fit, newdata = df_int)

  list(
    z_obs_resid = as.numeric(z_obs_resid),
    z_int_resid = as.numeric(z_int_resid),
    aux_coefs   = stats::coef(fit),
    aux_r2      = smry$r.squared,
    aux_model   = fit
  )
}


# ======================= PARAMETRIC (NB) INTENSITY ==========================

#' Compute intensity at locations (parametric model)
#'
#' @param z Standardised connectivity vector.
#' @param alpha,gamma Intensity parameters.
#' @param covariates Optional named list of covariate vectors, each the
#'   same length as `z`.
#' @param betas Named numeric vector of covariate coefficients, matching
#'   names in `covariates`.
#' @return Numeric vector of intensity values.
#' @export
compute_intensity <- function(z, alpha, gamma,
                              covariates = NULL,
                              betas      = NULL) {
  log_lambda <- alpha + gamma * z

  if (!is.null(covariates) && !is.null(betas)) {
    for (nm in names(covariates)) {
      beta_nm <- betas[nm] %||% 0
      if (beta_nm != 0) {
        log_lambda <- log_lambda + beta_nm * covariates[[nm]]
      }
    }
  }

  exp(log_lambda)
}


#' Negative log-likelihood for the NB PPP intensity model (cached version)
#'
#' Operates on pre-extracted numeric vectors to avoid repeated raster I/O
#' inside the optimiser.
#'
#' @param theta Parameter vector.  Layout: `c(alpha, gamma, [betas...],
#'   log_nb_theta)`.  The last element is the log-dispersion parameter
#'   (`size` in [stats::dnbinom]).
#' @param z_obs,z_int Standardised connectivity at observations /
#'   integration points.
#' @param int_weights Quadrature weights for integration points.
#' @param obs_weights Observation weights (typically all 1).
#' @param cov_obs,cov_int Named lists of covariate vectors (or `NULL`).
#' @param cov_names Character vector of covariate names (determines
#'   which elements of `theta` map to betas).
#' @return Scalar negative log-likelihood.
#' @keywords internal
.nb_negloglik_cached <- function(theta,
                                  z_obs, z_int,
                                  int_weights, obs_weights,
                                  cov_obs    = NULL,
                                  cov_int    = NULL,
                                  cov_names  = character(0)) {

  alpha <- theta[1]
  gamma <- theta[2]

  n_cov  <- length(cov_names)
  betas  <- if (n_cov > 0) stats::setNames(theta[3:(2 + n_cov)], cov_names) else NULL

  log_nb_theta <- theta[length(theta)]
  nb_theta     <- exp(log_nb_theta)   # ensure > 0

  # Observation intensities
  lambda_obs <- compute_intensity(z_obs, alpha, gamma, cov_obs, betas)

  # Integration intensities
  lambda_int <- compute_intensity(z_int, alpha, gamma, cov_int, betas)

  # PPP terms
  term1 <- sum(obs_weights * log(pmax(lambda_obs, 1e-300)))
  term2 <- sum(int_weights * lambda_int)

  # NB adjustment
  n_obs   <- sum(obs_weights)
  nb_adj  <- lgamma(n_obs + nb_theta) - lgamma(nb_theta) -
             lgamma(n_obs + 1) +
             nb_theta * log(nb_theta / (nb_theta + term2)) +
             n_obs    * log(term2 / (nb_theta + term2))

  negll <- -(term1 - term2 + nb_adj)
  if (!is.finite(negll)) negll <- 1e15
  negll
}


#' Fit the negative-binomial PPP intensity model
#'
#' Inner-loop MLE for `alpha`, `gamma`, optional covariate betas, and NB
#' dispersion.  Uses [stats::optim()] with `"L-BFGS-B"`.
#'
#' @param connectivity_at_obs Numeric vector of raw connectivity at
#'   observation locations.
#' @param connectivity_raster A [terra::SpatRaster] for integration.
#' @param obs_coords Data.frame / matrix with `x, y`.
#' @param covariates_obs Named list of covariate vectors at observations
#'   (each 0–1 scaled; `NULL` to omit).
#' @param covariates_rasters Named list of [terra::SpatRaster] for
#'   integration-grid covariates (`NULL` to omit).
#' @param residualise Logical; residualise connectivity against
#'   covariates before fitting (default `FALSE`).
#' @param config List from [default_intensity_config()].
#' @param family An [intensity_family] object.  If `NULL` (default),
#'   uses [family_negbin()] for backward compatibility.
#' @param available_connectivity Numeric vector of raw connectivity at
#'   available/background locations.  When non-`NULL`, bypasses raster
#'   quadrature and uses these values with unit weights instead.
#' @param available_covariates Named list of covariate vectors at available
#'   locations.  Used together with `available_connectivity`.
#' @return A list with `estimates`, `se`, `loglik`, `convergence`,
#'   `c_scale`, `log_conn_mean`, `log_conn_sd`,
#'   `residualisation_info`.
#' @export
fit_intensity_nb <- function(connectivity_at_obs,
                             connectivity_raster,
                             obs_coords,
                             covariates_obs       = NULL,
                             covariates_rasters   = NULL,
                             residualise          = FALSE,
                             config               = default_intensity_config(),
                             family               = NULL,
                             available_connectivity = NULL,
                             available_covariates   = NULL) {

  family <- resolve_family(family %||% config$family, "negbin")

  if (!is.null(available_connectivity)) {
    # ---- selection mode: use explicit available locations -------------------
    C_obs_raw   <- pmax(connectivity_at_obs, config$min_connectivity)
    C_int_raw   <- pmax(available_connectivity, config$min_connectivity)
    int_weights <- rep(1, length(C_int_raw))
    obs_weights <- rep(1, length(C_obs_raw))

    std    <- standardise_connectivity(C_obs_raw, C_int_raw, config$c_scale)
    z_obs  <- std$z_obs
    z_int  <- std$z_int

    cov_int  <- available_covariates
    cov_obs  <- if (!is.null(covariates_obs) && length(covariates_obs) > 0) {
      lapply(covariates_obs, function(v) pmax(pmin(v, 1), 0))
    } else {
      NULL
    }
    cov_names <- if (!is.null(cov_int)) names(cov_int) else
                 if (!is.null(cov_obs)) names(cov_obs) else character(0)

    resid_info <- NULL
    if (residualise && !is.null(cov_obs) && length(cov_obs) > 0) {
      ri       <- residualise_connectivity(z_obs, z_int, cov_obs, cov_int)
      z_obs      <- ri$z_obs_resid
      z_int      <- ri$z_int_resid
      resid_info <- ri
    }

    n_cov <- length(cov_names)
    inits <- family$init_fn(n_cov)

    opt <- stats::optim(
      par    = inits$start,
      fn     = family$negloglik_fn,
      method = "L-BFGS-B",
      lower  = inits$lower,
      upper  = inits$upper,
      z_obs  = z_obs, z_int = z_int,
      int_weights = int_weights, obs_weights = obs_weights,
      cov_obs = cov_obs, cov_int = cov_int, cov_names = cov_names,
      hessian = TRUE
    )

    n_extra   <- family$n_extra_params
    n_base    <- length(opt$par) - n_extra
    if (!is.null(family$param_names_fn)) {
      est_names <- family$param_names_fn(cov_names)
      if (length(est_names) != length(opt$par)) {
        est_names <- if (n_base == 1L) "gamma" else c("alpha", "gamma")
        if (n_cov > 0 && n_base > 1L)
          est_names <- c(est_names, paste0("beta_", cov_names))
        if (n_extra > 0) est_names <- c(est_names, family$extra_param_names)
      }
    } else {
      if (n_base == 1L) {
        est_names <- "gamma"
      } else {
        est_names <- c("alpha", "gamma")
        if (n_cov > 0) est_names <- c(est_names, paste0("beta_", cov_names))
      }
      if (n_extra > 0) est_names <- c(est_names, family$extra_param_names)
    }

    estimates <- opt$par
    names(estimates) <- est_names

    se <- rep(NA_real_, length(estimates))
    tryCatch({
      H  <- opt$hessian
      V  <- solve(H)
      se <- sqrt(pmax(diag(V), 0))
    }, error = function(e) NULL)
    names(se) <- est_names

    return(list(
      estimates        = estimates,
      se               = se,
      loglik           = -opt$value,
      convergence      = opt$convergence,
      c_scale          = std$c_scale,
      log_conn_mean    = std$mu,
      log_conn_sd      = std$sigma,
      is_residualised  = !is.null(resid_info),
      residualisation_info = resid_info,
      hessian          = opt$hessian
    ))
  }

  # ---- extract integration grid -------------------------------------------
  C_all_raw   <- terra::values(connectivity_raster)
  valid_mask  <- !is.na(C_all_raw)
  C_int_raw   <- pmax(C_all_raw[valid_mask], config$min_connectivity)
  n_int_full  <- length(C_int_raw)
  cell_area   <- prod(terra::res(connectivity_raster))

  C_obs_raw <- pmax(connectivity_at_obs, config$min_connectivity)

  # ---- subsample integration points ---------------------------------------
  frac <- config$integration_subsample
  if (frac < 1 && frac > 0) {
    set.seed(config$seed)
    n_samp  <- round(n_int_full * frac)
    step_sz <- n_int_full / n_samp
    start_  <- stats::runif(1, 1, step_sz)
    idx     <- unique(round(seq(start_, n_int_full, by = step_sz)))
    idx     <- idx[idx <= n_int_full]
    C_int   <- C_int_raw[idx]
    wt_mult <- n_int_full / length(idx)
  } else {
    C_int   <- C_int_raw
    idx     <- seq_len(n_int_full)
    wt_mult <- 1
  }

  int_weights <- rep(cell_area * wt_mult, length(C_int))
  obs_weights <- rep(1, length(C_obs_raw))

  # ---- standardise connectivity ------------------------------------------
  std <- standardise_connectivity(C_obs_raw, C_int, config$c_scale)
  z_obs <- std$z_obs
  z_int <- std$z_int

  # ---- covariates at integration points -----------------------------------
  cov_int <- NULL
  cov_names <- character(0)
  if (!is.null(covariates_rasters) && length(covariates_rasters) > 0) {
    cov_int <- lapply(covariates_rasters, function(r) {
      v <- terra::values(r)[valid_mask]
      pmax(pmin(v[idx], 1), 0)
    })
    cov_names <- names(covariates_rasters)
  }

  cov_obs <- NULL
  if (!is.null(covariates_obs) && length(covariates_obs) > 0) {
    cov_obs <- lapply(covariates_obs, function(v) pmax(pmin(v, 1), 0))
    if (length(cov_names) == 0) cov_names <- names(covariates_obs)
  }

  # ---- optional residualisation -------------------------------------------
  resid_info <- NULL
  if (residualise && !is.null(cov_obs) && length(cov_obs) > 0) {
    ri <- residualise_connectivity(z_obs, z_int, cov_obs, cov_int)
    z_obs      <- ri$z_obs_resid
    z_int      <- ri$z_int_resid
    resid_info <- ri
  }

  # ---- optimise -----------------------------------------------------------
  n_cov   <- length(cov_names)
  inits   <- family$init_fn(n_cov)

  opt <- stats::optim(
    par    = inits$start,
    fn     = family$negloglik_fn,
    method = "L-BFGS-B",
    lower  = inits$lower,
    upper  = inits$upper,
    z_obs  = z_obs, z_int = z_int,
    int_weights = int_weights, obs_weights = obs_weights,
    cov_obs = cov_obs, cov_int = cov_int, cov_names = cov_names,
    hessian = TRUE
  )

  # ---- extract results ----------------------------------------------------
  n_extra   <- family$n_extra_params
  if (!is.null(family$param_names_fn)) {
    est_names <- family$param_names_fn(cov_names)
    if (n_extra > 0 && length(est_names) < length(opt$par)) {
      est_names <- c(est_names, family$extra_param_names)
    }
  } else {
    est_names <- c("alpha", "gamma")
    if (n_cov > 0) est_names <- c(est_names, paste0("beta_", cov_names))
    if (n_extra > 0) est_names <- c(est_names, family$extra_param_names)
  }

  estimates <- opt$par
  # Transform extra params to natural scale
  if (n_extra > 0) {
    extra_idx <- seq(length(estimates) - n_extra + 1L, length(estimates))
    for (ei in extra_idx) {
      estimates[ei] <- exp(estimates[ei])
    }
  }
  # Alias: first extra param is conventionally "size" / "theta"
  if (n_extra > 0 && family$name %in% c("negbin", "zinb")) {
    est_names[length(est_names) - n_extra + 1L] <- "size"
  }
  names(estimates) <- est_names

  se <- rep(NA_real_, length(estimates))
  tryCatch({
    H  <- opt$hessian
    V  <- solve(H)
    se <- sqrt(pmax(diag(V), 0))
    # delta-method for exp-transformed extra params
    if (n_extra > 0) {
      for (ei in extra_idx) {
        se[ei] <- se[ei] * estimates[ei]
      }
    }
  }, error = function(e) NULL)
  names(se) <- est_names

  list(
    estimates        = estimates,
    se               = se,
    loglik           = -opt$value,
    convergence      = opt$convergence,
    c_scale          = std$c_scale,
    log_conn_mean    = std$mu,
    log_conn_sd      = std$sigma,
    is_residualised  = !is.null(resid_info),
    residualisation_info = resid_info,
    hessian          = opt$hessian
  )
}


# ===================== GAM-BASED INTENSITY ==================================

#' Fit a GAM intensity model via Berman-Turner quadrature
#'
#' Uses [mgcv::bam()] with `family = nb()` to fit a log-Gaussian Cox
#' Process-style model through penalised regression splines.  Optionally
#' includes a 2-D spatial smooth `te(x, y)` to capture residual spatial
#' structure (Dovers et al., 2024, *The American Statistician*).
#'
#' @inheritParams fit_intensity_nb
#' @param config List from [default_intensity_config()].
#' @return A list matching the interface of [fit_intensity_nb()], plus
#'   `gam_model`, `gam_edf`, `gam_deviance_explained`, `gam_aic`.
#' @export
fit_intensity_gam <- function(connectivity_at_obs,
                              connectivity_raster,
                              obs_coords,
                              covariates_obs     = NULL,
                              covariates_rasters = NULL,
                              residualise        = FALSE,
                              config             = default_intensity_config()) {

  # ---- extract / subsample integration grid --------------------------------
  C_all_raw  <- terra::values(connectivity_raster)
  valid_mask <- !is.na(C_all_raw)
  C_int_raw  <- pmax(C_all_raw[valid_mask], config$min_connectivity)
  int_coords <- terra::crds(connectivity_raster, na.rm = TRUE)
  n_int_full <- length(C_int_raw)
  cell_area  <- prod(terra::res(connectivity_raster))

  frac <- config$integration_subsample
  if (frac < 1 && frac > 0) {
    set.seed(config$seed)
    n_samp  <- round(n_int_full * frac)
    step_sz <- n_int_full / n_samp
    start_  <- stats::runif(1, 1, step_sz)
    idx     <- unique(round(seq(start_, n_int_full, by = step_sz)))
    idx     <- idx[idx <= n_int_full]
    C_int   <- C_int_raw[idx]
    int_xy  <- int_coords[idx, , drop = FALSE]
    wt_mult <- n_int_full / length(idx)
  } else {
    C_int   <- C_int_raw
    int_xy  <- int_coords
    idx     <- seq_len(n_int_full)
    wt_mult <- 1
  }

  C_obs_raw <- pmax(connectivity_at_obs, config$min_connectivity)

  # ---- standardise connectivity ------------------------------------------
  std   <- standardise_connectivity(C_obs_raw, C_int, config$c_scale)
  z_obs <- std$z_obs
  z_int <- std$z_int

  # ---- covariates ---------------------------------------------------------
  cov_int <- cov_obs <- NULL
  cov_names <- character(0)
  if (!is.null(covariates_rasters) && length(covariates_rasters) > 0) {
    cov_int <- lapply(covariates_rasters, function(r) {
      v <- terra::values(r)[valid_mask]
      pmax(pmin(v[idx], 1), 0)
    })
    cov_names <- names(covariates_rasters)
  }
  if (!is.null(covariates_obs) && length(covariates_obs) > 0) {
    cov_obs <- lapply(covariates_obs, function(v) pmax(pmin(v, 1), 0))
    if (length(cov_names) == 0) cov_names <- names(covariates_obs)
  }

  # ---- optional residualisation -------------------------------------------
  resid_info <- NULL
  if (residualise && !is.null(cov_obs) && length(cov_obs) > 0) {
    ri <- residualise_connectivity(z_obs, z_int, cov_obs, cov_int)
    z_obs      <- ri$z_obs_resid
    z_int      <- ri$z_int_resid
    resid_info <- ri
  }

  # ---- build Berman-Turner data -------------------------------------------
  obs_mat <- as_coord_matrix(obs_coords)
  n_obs   <- nrow(obs_mat)
  n_int   <- length(C_int)

  # response: 1/weight at obs, 0 at integration
  bt_y   <- c(rep(1, n_obs), rep(0, n_int))
  bt_w   <- c(rep(1e-6, n_obs),
              rep(cell_area * wt_mult, n_int))
  bt_y_w <- bt_y / bt_w   # pseudo-response for Poisson trick

  bt_df <- data.frame(
    y_bt         = bt_y_w,
    wt           = bt_w,
    connectivity = c(z_obs, z_int),
    x_coord      = c(obs_mat[, 1], int_xy[, 1]),
    y_coord      = c(obs_mat[, 2], int_xy[, 2]),
    is_obs       = c(rep(1L, n_obs), rep(0L, n_int))
  )

  for (nm in cov_names) {
    bt_df[[nm]] <- c(
      if (!is.null(cov_obs)) cov_obs[[nm]] else rep(0, n_obs),
      if (!is.null(cov_int)) cov_int[[nm]] else rep(0, n_int)
    )
  }

  # ---- build formula ------------------------------------------------------
  terms <- character(0)
  k_con <- config$k_connectivity
  k_cov <- config$k_covariate

  if (config$covariate_type == "smooth") {
    terms <- c(terms, sprintf("s(connectivity, k = %d)", k_con))
    for (nm in cov_names) {
      terms <- c(terms, sprintf("s(%s, k = %d)", nm, k_cov))
    }
  } else {
    terms <- c(terms, "connectivity")
    terms <- c(terms, cov_names)
  }

  if (config$include_spatial_re) {
    k_sp <- config$k_spatial
    if (config$spatial_tensor) {
      terms <- c(terms,
        sprintf('te(x_coord, y_coord, k = c(%d, %d), bs = "%s")',
                k_sp, k_sp, config$spatial_bs))
    } else {
      terms <- c(terms,
        sprintf('s(x_coord, y_coord, k = %d)', k_sp))
    }
  }

  if (length(terms) == 0) terms <- "1"
  fmla <- stats::as.formula(paste("y_bt ~", paste(terms, collapse = " + ")))

  # ---- fit GAM ------------------------------------------------------------
  gam_fit <- mgcv::bam(
    fmla,
    family   = mgcv::nb(),
    data     = bt_df,
    weights  = bt_df$wt,
    method   = "fREML",
    discrete = TRUE
  )

  smry <- summary(gam_fit)

  # ---- extract effective alpha / gamma ------------------------------------
  # Approximate linear equivalents from the parametric / smooth terms
  alpha_eff <- stats::coef(gam_fit)["(Intercept)"]

  # Approximate gamma as slope of connectivity smooth at the mean (zero, since
  # z-scored) using finite difference
  eps <- 0.01
  nd0 <- bt_df[1, , drop = FALSE]; nd0$connectivity <- 0
  nd1 <- bt_df[1, , drop = FALSE]; nd1$connectivity <- eps
  p0  <- stats::predict(gam_fit, newdata = nd0, type = "link")
  p1  <- stats::predict(gam_fit, newdata = nd1, type = "link")
  gamma_eff <- as.numeric((p1 - p0) / eps)

  estimates <- c(alpha = unname(alpha_eff), gamma = unname(gamma_eff))
  se        <- c(alpha = NA_real_, gamma = NA_real_)

  edf <- sum(smry$edf) + length(stats::coef(gam_fit))

  list(
    estimates        = estimates,
    se               = se,
    loglik           = as.numeric(-0.5 * gam_fit$deviance),
    convergence      = if (gam_fit$converged) 0L else 1L,
    c_scale          = std$c_scale,
    log_conn_mean    = std$mu,
    log_conn_sd      = std$sigma,
    is_residualised  = !is.null(resid_info),
    residualisation_info = resid_info,
    gam_model        = gam_fit,
    gam_edf          = edf,
    gam_deviance_explained = smry$dev.expl,
    gam_aic          = stats::AIC(gam_fit),
    gam_summary      = smry,
    hessian          = NULL
  )
}


# ===================== INTENSITY PREDICTION =================================

#' Predict intensity surface from a fitted model
#'
#' @param fit Result from [fit_intensity_nb()] or [fit_intensity_gam()].
#' @param connectivity_raster A [terra::SpatRaster] of connectivity.
#' @param covariates_rasters Named list of [terra::SpatRaster] (or `NULL`).
#' @param config Optional intensity configuration list (currently unused;
#'   reserved for future extensions).
#' @return A [terra::SpatRaster] of predicted intensity.
#' @export
predict_intensity <- function(fit,
                              connectivity_raster,
                              covariates_rasters = NULL,
                              config = NULL) {

  if (!is.null(fit$gam_model)) {
    return(.predict_intensity_gam(fit, connectivity_raster,
                                  covariates_rasters))
  }

  # Parametric prediction
  alpha_raw <- fit$estimates[["alpha"]]
  alpha <- if (is.null(alpha_raw) || is.na(alpha_raw)) 0 else alpha_raw
  gamma <- fit$estimates[["gamma"]]

  C_vals <- terra::values(connectivity_raster)
  C_safe <- pmax(C_vals, 0)
  z_vals <- (log1p(C_safe / fit$c_scale) - fit$log_conn_mean) /
            fit$log_conn_sd

  # Apply stored residualisation if the model was fitted on residualised z.
  # Without this, gamma is estimated against the residual but multiplied by
  # the raw z, biasing the predicted intensity.
  if (isTRUE(fit$is_residualised) && !is.null(fit$residualisation_info)) {
    ri      <- fit$residualisation_info
    # Build a newdata frame matching the covariate names used in the aux lm.
    # covariate values come from covariates_rasters (same names).
    cov_names_resid <- setdiff(names(ri$aux_coefs), "(Intercept)")
    if (length(cov_names_resid) > 0 && !is.null(covariates_rasters)) {
      nd_resid <- data.frame(dummy = seq_along(z_vals))
      for (nm in cov_names_resid) {
        if (!is.null(covariates_rasters[[nm]])) {
          cv <- pmax(pmin(terra::values(covariates_rasters[[nm]]), 1), 0)
          nd_resid[[nm]] <- cv
        }
      }
      nd_resid$dummy <- NULL
      z_vals <- z_vals - stats::predict(ri$aux_model, newdata = nd_resid)
    }
  }

  log_lambda <- alpha + gamma * z_vals

  # Add covariate contributions
  if (!is.null(covariates_rasters)) {
    for (nm in names(covariates_rasters)) {
      beta_nm <- fit$estimates[[paste0("beta_", nm)]]
      if (!is.null(beta_nm) && !is.na(beta_nm)) {
        cv <- pmax(pmin(terra::values(covariates_rasters[[nm]]), 1), 0)
        log_lambda <- log_lambda + beta_nm * cv
      }
    }
  }

  out <- connectivity_raster[[1]]
  terra::values(out) <- exp(log_lambda)
  names(out) <- "intensity"
  out
}


#' Fit a selection function intensity model
#'
#' Convenience wrapper around [fit_intensity_nb()] for selection function
#' (RSF / RSP / iSSA) models.  Accepts explicit available/background locations
#' instead of a connectivity raster for quadrature.
#'
#' @param connectivity_at_obs Numeric vector of raw connectivity at
#'   used/presence locations.
#' @param available_connectivity Numeric vector of raw connectivity at
#'   available/background locations.
#' @param obs_coords Data.frame / matrix with `x, y` for used locations.
#' @param available_covariates Named list of covariate vectors at available
#'   locations (`NULL` to omit).
#' @param covariates_obs Named list of covariate vectors at used locations
#'   (`NULL` to omit).
#' @param config List from [default_intensity_config()].
#' @param family An [intensity_family] object (default [family_rsf()]).
#' @return A list matching the interface of [fit_intensity_nb()].
#' @export
fit_intensity_selection <- function(connectivity_at_obs,
                                    available_connectivity,
                                    obs_coords,
                                    available_covariates = NULL,
                                    covariates_obs       = NULL,
                                    config               = default_intensity_config(),
                                    family               = family_rsf()) {
  fit_intensity_nb(
    connectivity_at_obs    = connectivity_at_obs,
    connectivity_raster    = NULL,
    obs_coords             = obs_coords,
    covariates_obs         = covariates_obs,
    covariates_rasters     = NULL,
    config                 = config,
    family                 = family,
    available_connectivity = available_connectivity,
    available_covariates   = available_covariates
  )
}


#' @keywords internal
.predict_intensity_gam <- function(fit, connectivity_raster,
                                   covariates_rasters) {

  gam_mod <- fit$gam_model
  coords  <- terra::crds(connectivity_raster, na.rm = FALSE)
  C_all   <- terra::values(connectivity_raster)

  valid <- !is.na(C_all)
  C_v   <- pmax(C_all[valid], 0)
  z_v   <- (log1p(C_v / fit$c_scale) - fit$log_conn_mean) / fit$log_conn_sd

  # Apply stored residualisation if the model was fitted on residualised z.
  # Without this, the GAM smooth of connectivity is estimated against the
  # residual but evaluated on the raw z, biasing the predicted intensity.
  if (isTRUE(fit$is_residualised) && !is.null(fit$residualisation_info)) {
    ri <- fit$residualisation_info
    cov_names_resid <- setdiff(names(ri$aux_coefs), "(Intercept)")
    if (length(cov_names_resid) > 0 && !is.null(covariates_rasters)) {
      nd_resid <- data.frame(dummy = seq_along(z_v))
      for (nm in cov_names_resid) {
        if (!is.null(covariates_rasters[[nm]])) {
          cv <- pmax(pmin(terra::values(covariates_rasters[[nm]]), 1), 0)
          nd_resid[[nm]] <- cv[valid]
        }
      }
      nd_resid$dummy <- NULL
      z_v <- z_v - stats::predict(ri$aux_model, newdata = nd_resid)
    }
  }

  nd <- data.frame(
    connectivity = z_v,
    x_coord      = coords[valid, 1],
    y_coord      = coords[valid, 2]
  )

  if (!is.null(covariates_rasters)) {
    for (nm in names(covariates_rasters)) {
      cv <- terra::values(covariates_rasters[[nm]])
      nd[[nm]] <- pmax(pmin(cv[valid], 1), 0)
    }
  }

  pred_link <- stats::predict(gam_mod, newdata = nd, type = "response")

  out_v <- rep(NA_real_, length(C_all))
  out_v[valid] <- as.numeric(pred_link)
  out <- connectivity_raster[[1]]
  terra::values(out) <- out_v
  names(out) <- "intensity"
  out
}
