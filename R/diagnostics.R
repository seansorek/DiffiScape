# ============================================================================
# Diagnostics
#
# Deviance residuals, diagnostic plots, Moran's I spatial autocorrelation.
# Uses deviance residuals ONLY (not quadrat Pearson residuals).
# ============================================================================

# --------------- Deviance residuals -----------------------------------------

#' Compute deviance residuals for a negative binomial PPP model
#'
#' Each cell contributes a deviance residual defined as
#' \deqn{d_i = \mathrm{sign}(y_i - \hat{\mu}_i) \sqrt{2 \left[
#'   y_i \log\!\left(\frac{y_i}{\hat{\mu}_i}\right) - (y_i + k)
#'   \log\!\left(\frac{y_i + k}{\hat{\mu}_i + k}\right) \right]}}
#' where \eqn{k} is the negative binomial size parameter.
#'
#' @param observed Integer vector of observed counts per cell.
#' @param fitted Numeric vector of fitted means (\eqn{\hat{\mu}}).
#' @param size NB size parameter (theta).
#' @return Numeric vector of deviance residuals.
#' @export
compute_deviance_residuals <- function(observed, fitted, size) {

  n <- length(observed)
  if (length(fitted) != n) stop("observed and fitted must be same length",
                                 call. = FALSE)

  y    <- observed
  mu   <- pmax(fitted, 1e-10)
  k    <- size

  # Term 1: y * log(y / mu)  -- 0 when y == 0

  t1 <- ifelse(y > 0, y * log(y / mu), 0)
  # Term 2: (y + k) * log((y + k) / (mu + k))
  t2 <- (y + k) * log((y + k) / (mu + k))

  d_sq <- 2 * (t1 - t2)
  d_sq <- pmax(d_sq, 0)   # numerical safety

  sign(y - mu) * sqrt(d_sq)
}


#' Compute deviance residuals from a GAM intensity fit
#'
#' Extracts residuals from the fitted GAM model object.
#'
#' @param gam_fit A fitted GAM model from [fit_intensity_gam()].
#' @return Numeric vector of deviance residuals.
#' @export
compute_deviance_residuals_gam <- function(gam_fit) {
  if (inherits(gam_fit, "list") && !is.null(gam_fit$gam_model)) {
    gam_fit <- gam_fit$gam_model
  }
  if (!inherits(gam_fit, "gam") && !inherits(gam_fit, "bam")) {
    stop("Expected a gam/bam model object", call. = FALSE)
  }
  stats::residuals(gam_fit, type = "deviance")
}


# --------------- Rasterised deviance residuals ------------------------------

#' Rasterise deviance residuals onto a grid
#'
#' Counts observations per cell, computes predicted intensity per cell,
#' and returns a raster of deviance residuals.
#'
#' @param intensity_fit Result from [evaluate_full_model()] or
#'   [fit_intensity_nb()] / [fit_intensity_gam()].
#' @param obs_points Data.frame with `x, y`.
#' @param connectivity A [terra::SpatRaster] of connectivity values.
#' @param intensity_config Intensity config.
#' @param covariates_rasters Named list of covariate rasters.
#' @param family An [intensity_family] object, or `NULL` (uses NB default).
#' @return A [terra::SpatRaster] of deviance residuals.
#' @export
rasterise_deviance_residuals <- function(intensity_fit,
                                         obs_points,
                                         connectivity,
                                         intensity_config   = default_intensity_config(),
                                         covariates_rasters = NULL,
                                         family             = NULL) {

  # Template raster
  template <- connectivity

  # Count observations per cell
  coords <- as_coord_matrix(obs_points)
  cells  <- terra::cellFromXY(template, coords)

  n_cells <- terra::ncell(template)
  counts  <- rep(0L, n_cells)
  tab     <- table(cells)
  counts[as.integer(names(tab))] <- as.integer(tab)

  # Resolve the full fit object for predict_intensity.
  # intensity_fit may come from evaluate_full_model() (which stores the full
  # fit in $intensity_fit_obj and just estimates in $intensity_params) or
  # directly from fit_intensity_nb/gam (which has $estimates at the top level).
  fit_obj <- intensity_fit$intensity_fit_obj %||% intensity_fit

  # Predicted intensity per cell (lambda * cell_area = expected count)
  pred_rast <- predict_intensity(
    fit_obj,
    connectivity,
    covariates_rasters = covariates_rasters,
    config             = intensity_config
  )
  cell_area   <- prod(terra::res(connectivity))
  fitted_vals <- terra::values(pred_rast)[, 1] * cell_area

  # Resolve the estimates vector for size / extra-param extraction
  params_vec <- fit_obj$estimates %||% intensity_fit$intensity_params

  # Deviance residuals
  valid <- !is.na(fitted_vals) & fitted_vals > 0
  dev_resid <- rep(NA_real_, n_cells)

  if (!is.null(family)) {
    ep_names <- family$extra_param_names
    extra_p  <- params_vec[ep_names]
    dev_resid[valid] <- family$deviance_residuals_fn(
      counts[valid], fitted_vals[valid], extra_p
    )
  } else {
    size <- params_vec["size"]
    if (is.null(size) || is.na(size)) size <- 1
    dev_resid[valid] <- compute_deviance_residuals(
      counts[valid], fitted_vals[valid], size
    )
  }

  if (all(is.na(dev_resid))) {
    message("Deviance residuals not defined for this family (e.g. selection families); returning NA raster.")
  }

  result <- terra::rast(template)
  terra::values(result) <- dev_resid
  names(result) <- "deviance_residual"
  result
}


# --------------- Moran's I --------------------------------------------------

#' Moran's I test for spatial autocorrelation of residuals
#'
#' Tests whether the deviance residuals exhibit significant spatial
#' autocorrelation using a nearest-neighbour spatial weights matrix.
#'
#' @param residuals Numeric vector of residuals.
#' @param coords Matrix or data.frame of spatial coordinates (2 columns).
#' @param k Number of nearest neighbours for weights matrix.
#' @return A list with `observed`, `expected`, `variance`, `z_score`,
#'   `p_value`.
#' @export
moran_test <- function(residuals, coords, k = 8L) {

  if (!requireNamespace("spdep", quietly = TRUE)) {
    stop("Package 'spdep' is required for Moran's I test.\n",
         "Install with install.packages('spdep')", call. = FALSE)
  }

  valid <- !is.na(residuals)
  res   <- residuals[valid]
  crd   <- as.matrix(coords)[valid, , drop = FALSE]

  nb <- spdep::knearneigh(crd, k = k)
  nb <- spdep::knn2nb(nb)
  wt <- spdep::nb2listw(nb, style = "W")

  mt <- spdep::moran.test(res, wt, alternative = "two.sided")

  list(
    observed = mt$estimate["Moran I statistic"],
    expected = mt$estimate["Expectation"],
    variance = mt$estimate["Variance"],
    z_score  = mt$statistic,
    p_value  = mt$p.value
  )
}


# --------------- Diagnostic plots -------------------------------------------

#' Diagnostic plot: deviance residuals vs fitted values
#'
#' @param observed Integer vector of observed counts.
#' @param fitted Numeric vector of fitted means.
#' @param size NB size parameter (ignored when `family` is given).
#' @param family An [intensity_family] object, or `NULL`.
#' @param extra_params Named vector of extra distribution parameters
#'   (used with `family`).
#' @param ... Extra arguments passed to [graphics::plot()].
#' @return Invisible `NULL`.
#' @export
plot_deviance_residuals <- function(observed, fitted, size = NULL,
                                    family = NULL, extra_params = NULL,
                                    ...) {

  dev_r <- if (!is.null(family)) {
    family$deviance_residuals_fn(observed, fitted, extra_params)
  } else {
    compute_deviance_residuals(observed, fitted, size)
  }

  graphics::plot(fitted, dev_r,
                 pch = 16, cex = 0.5, col = grDevices::adjustcolor("black", 0.3),
                 xlab = "Fitted values", ylab = "Deviance residuals",
                 main = "Deviance Residuals vs Fitted", ...)
  graphics::abline(h = 0, lty = 2, col = "red")
  graphics::lines(stats::lowess(fitted, dev_r), col = "blue", lwd = 2)
  invisible(NULL)
}


#' Diagnostic plot: QQ-plot of deviance residuals
#'
#' @param observed Integer vector of observed counts.
#' @param fitted Numeric vector of fitted means.
#' @param size NB size parameter (ignored when `family` is given).
#' @param family An [intensity_family] object, or `NULL`.
#' @param extra_params Named vector of extra distribution parameters
#'   (used with `family`).
#' @param ... Extra arguments passed to [stats::qqnorm()].
#' @return Invisible `NULL`.
#' @export
plot_qq_deviance <- function(observed, fitted, size = NULL,
                             family = NULL, extra_params = NULL, ...) {

  dev_r <- if (!is.null(family)) {
    family$deviance_residuals_fn(observed, fitted, extra_params)
  } else {
    compute_deviance_residuals(observed, fitted, size)
  }

  stats::qqnorm(dev_r, main = "QQ-Plot: Deviance Residuals",
                pch = 16, cex = 0.5, ...)
  stats::qqline(dev_r, col = "red", lwd = 2)
  invisible(NULL)
}


#' Diagnostic plot: spatial map of deviance residuals
#'
#' @param resid_raster A [terra::SpatRaster] from
#'   [rasterise_deviance_residuals()].
#' @param obs_points Optional data.frame with `x, y` to overlay.
#' @param ... Extra arguments passed to [terra::plot()].
#' @return Invisible `NULL`.
#' @export
plot_residual_map <- function(resid_raster, obs_points = NULL, ...) {

  terra::plot(resid_raster, main = "Deviance Residuals (spatial)", ...)

  if (!is.null(obs_points)) {
    coords <- as_coord_matrix(obs_points)
    graphics::points(coords[, 1], coords[, 2], pch = 3, cex = 0.5)
  }
  invisible(NULL)
}


# --------------- Comprehensive diagnostics ----------------------------------

#' Run comprehensive model diagnostics
#'
#' Computes deviance residuals, runs Moran's I test, and optionally
#' produces diagnostic plots.
#'
#' @param intensity_fit Result from [evaluate_full_model()] or
#'   [fit_intensity_nb()].
#' @param obs_points Data.frame with `x, y`.
#' @param connectivity A [terra::SpatRaster].
#' @param intensity_config Intensity config.
#' @param covariates_rasters Named list of covariate rasters.
#' @param family An [intensity_family] object, or `NULL` (uses NB default).
#' @param plot Logical; produce diagnostic plots.
#' @return A list with `residual_raster`, `moran`, `mean_deviance`,
#'   `prop_large` (proportion of |resid| > 2).
#' @export
diagnose_model <- function(intensity_fit,
                           obs_points,
                           connectivity,
                           intensity_config   = default_intensity_config(),
                           covariates_rasters = NULL,
                           family             = NULL,
                           plot               = TRUE) {

  resid_rast <- rasterise_deviance_residuals(
    intensity_fit, obs_points, connectivity,
    intensity_config, covariates_rasters, family = family
  )

  resid_vals <- terra::values(resid_rast)[, 1]
  valid      <- !is.na(resid_vals)

  mean_dev   <- mean(abs(resid_vals[valid]))
  prop_large <- mean(abs(resid_vals[valid]) > 2)

  # Moran's I
  moran_result <- tryCatch({
    xy <- terra::xyFromCell(connectivity, which(valid))
    moran_test(resid_vals[valid], xy, k = 8L)
  }, error = function(e) {
    message("  Moran's I test failed: ", conditionMessage(e))
    NULL
  })

  if (plot) {
    # Get counts + fitted for cell-level diagnostics
    coords <- as_coord_matrix(obs_points)
    cells  <- terra::cellFromXY(connectivity, coords)
    n_cells <- terra::ncell(connectivity)
    counts <- rep(0L, n_cells)
    tab <- table(cells)
    counts[as.integer(names(tab))] <- as.integer(tab)

    fit_obj_plt  <- intensity_fit$intensity_fit_obj %||% intensity_fit
    params_vec_plt <- fit_obj_plt$estimates %||% intensity_fit$intensity_params

    pred_rast <- predict_intensity(
      fit_obj_plt, connectivity,
      covariates_rasters = covariates_rasters,
      config = intensity_config
    )
    fitted <- terra::values(pred_rast)[, 1]

    size <- params_vec_plt["size"]
    if (is.null(size) || is.na(size)) size <- 1

    ep_names    <- if (!is.null(family)) family$extra_param_names else NULL
    extra_p_plt <- params_vec_plt[ep_names]

    grDevices::dev.new()
    graphics::par(mfrow = c(2, 2))
    plot_deviance_residuals(counts[valid], fitted[valid], size,
                            family = family, extra_params = extra_p_plt)
    plot_qq_deviance(counts[valid], fitted[valid], size,
                     family = family, extra_params = extra_p_plt)
    plot_residual_map(resid_rast, obs_points)

    # Histogram
    graphics::hist(resid_vals[valid], 50, freq = FALSE,
                   col = "steelblue", border = "white",
                   main = "Deviance Residuals Distribution",
                   xlab = "Deviance Residual")
    graphics::abline(v = 0, lty = 2, col = "red", lwd = 2)
  }

  message(sprintf("  Mean |deviance residual|: %.3f", mean_dev))
  message(sprintf("  Proportion |resid| > 2: %.1f%%", 100 * prop_large))
  if (!is.null(moran_result)) {
    message(sprintf("  Moran's I: %.4f (p = %.4f)",
                    moran_result$observed, moran_result$p_value))
    if (isTRUE(moran_result$p_value < 0.05)) {
      warning(
        sprintf(
          "Significant residual spatial autocorrelation detected (Moran's I = %.4f, p = %.4f). ",
          moran_result$observed, moran_result$p_value
        ),
        "Intensity SEs are likely optimistic. ",
        "Consider setting include_spatial_re = TRUE in intensity_config to absorb residual spatial structure. ",
        "Note: include_spatial_re = TRUE introduces concurvity risk when connectivity is spatially smooth; ",
        "check mgcv::concurvity() afterward.",
        call. = FALSE
      )
    }
  }

  list(
    residual_raster = resid_rast,
    moran           = moran_result,
    mean_deviance   = mean_dev,
    prop_large      = prop_large
  )
}
