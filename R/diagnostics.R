
# fit_intensity_nb() renames the first extra parameter for negbin/zinb
# families to "size" in the estimates vector, but family$extra_param_names
# still reports the optimizer-scale name (e.g. "log_nb_theta"). Mirror that
# rename when looking up extra params on a fit, otherwise the family-aware
# deviance-residual path silently gets NA and falls back to k=1.
.fit_extra_param_names <- function(family) {
  ep <- family$extra_param_names
  if (length(ep) > 0 && isTRUE(family$name %in% c("negbin", "zinb"))) {
    ep[1L] <- "size"
  }
  ep
}

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
    ep_names <- .fit_extra_param_names(family)
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

    ep_names    <- if (!is.null(family)) .fit_extra_param_names(family) else NULL
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


# --------------- Posterior predictive checks ---------------------------------

# Simulate one replicate of cell counts from the posterior predictive.
.ppc_simulate_counts <- function(mu_cells, family_name, size = NULL,
                                 pi_val = NULL) {
  n <- length(mu_cells)
  switch(family_name,
    poisson = stats::rpois(n, lambda = mu_cells),
    negbin = {
      size_safe <- max(size, 1e-6)
      stats::rnbinom(n, mu = mu_cells, size = size_safe)
    },
    zinb = {
      size_safe <- max(size, 1e-6)
      is_zero <- stats::runif(n) < pi_val
      counts  <- stats::rnbinom(n, mu = mu_cells, size = size_safe)
      ifelse(is_zero, 0L, counts)
    },
    stop("Posterior predictive checks are not supported for family '",
         family_name, "'. PPC requires a count-valued distributional family ",
         "(negbin, poisson, zinb).", call. = FALSE)
  )
}


# Compute test quantities from a vector of cell counts.
.ppc_test_quantities <- function(counts, mu_cells, tq_names,
                                 family = NULL, size = NULL,
                                 coords = NULL, include_moran = FALSE) {
  out <- numeric(0)

  if ("total_count" %in% tq_names) {
    out[["total_count"]] <- sum(counts)
  }

  if ("vmi_ratio" %in% tq_names) {
    mn <- mean(counts)
    out[["vmi_ratio"]] <- stats::var(counts) / max(mn, 1e-10)
  }

  if ("mean_deviance" %in% tq_names) {
    valid <- mu_cells > 0 & !is.na(mu_cells)
    if (any(valid)) {
      dr <- if (!is.null(family) && family$n_extra_params > 0L) {
        ep_names <- .fit_extra_param_names(family)
        extra_p  <- stats::setNames(size, ep_names[1L])
        family$deviance_residuals_fn(counts[valid], mu_cells[valid], extra_p)
      } else if (!is.null(family)) {
        family$deviance_residuals_fn(counts[valid], mu_cells[valid], NULL)
      } else {
        s <- if (is.null(size) || is.na(size)) 1 else size
        compute_deviance_residuals(counts[valid], mu_cells[valid], s)
      }
      out[["mean_deviance"]] <- mean(abs(dr))
    } else {
      out[["mean_deviance"]] <- NA_real_
    }
  }

  if (include_moran && !is.null(coords)) {
    moran_res <- tryCatch({
      valid <- mu_cells > 0 & !is.na(mu_cells)
      s <- if (is.null(size) || is.na(size)) 1 else size
      dr <- compute_deviance_residuals(counts[valid], mu_cells[valid], s)
      moran_test(dr, coords[valid, , drop = FALSE], k = 8L)
    }, error = function(e) NULL)
    out[["moran_i"]] <- if (!is.null(moran_res)) moran_res$observed else NA_real_
  }

  out
}


#' Posterior predictive check
#'
#' Simulates replicate datasets from the posterior predictive distribution
#' and compares test quantities to the observed data.  Returns Bayesian
#' p-values for each test quantity.
#'
#' @param posterior_samples Data.frame from [posterior_sample()] (one row
#'   per draw, columns include intensity parameters such as `alpha`,
#'   `gamma`, `beta_*`, and optionally `size`).
#' @param intensity_fit Result from [fit_intensity_nb()],
#'   [evaluate_full_model()], or [ds_fit_intensity()].  Supplies the
#'   connectivity standardisation constants (`c_scale`, `log_conn_mean`,
#'   `log_conn_sd`).
#' @param obs_points Data.frame with `x, y`.
#' @param connectivity A [terra::SpatRaster] of connectivity values.
#' @param intensity_config Intensity config list.
#' @param covariates_rasters Named list of covariate rasters (or `NULL`).
#' @param family An [intensity_family] object, or `NULL` (defaults to NB).
#' @param n_sim Number of posterior draws to simulate from.
#' @param test_quantities Character vector of test quantity names.
#'   Supported: `"total_count"`, `"vmi_ratio"`, `"mean_deviance"`.
#' @param include_moran Logical; also compute Moran's I per replicate
#'   (slow; requires `spdep`).
#' @param thin Integer; use every `thin`-th row of `posterior_samples`.
#' @param seed Optional integer seed for reproducibility.
#' @param plot Logical; produce PPC diagnostic plots.
#' @param verbose Logical; emit progress messages.
#' @return A list with class `"ds_ppc"`:
#'   \describe{
#'     \item{`observed`}{Named numeric vector of observed test quantities.}
#'     \item{`simulated`}{Named list of numeric vectors (one per test
#'       quantity, length `n_sim`).}
#'     \item{`bayesian_p`}{Named numeric vector of two-sided Bayesian
#'       p-values.}
#'     \item{`n_sim`}{Integer; number of simulations performed.}
#'     \item{`test_quantities`}{Character vector of test quantity names.}
#'   }
#' @seealso [diagnose_model()], [posterior_sample()], [plot_ppc()]
#' @export
ds_ppc <- function(posterior_samples,
                   intensity_fit,
                   obs_points,
                   connectivity,
                   intensity_config   = default_intensity_config(),
                   covariates_rasters = NULL,
                   family             = NULL,
                   n_sim              = 200L,
                   test_quantities    = c("total_count", "vmi_ratio",
                                          "mean_deviance"),
                   include_moran      = FALSE,
                   thin               = 1L,
                   seed               = NULL,
                   plot               = TRUE,
                   verbose            = TRUE) {

  if (!is.data.frame(posterior_samples) || nrow(posterior_samples) == 0) {
    stop("posterior_samples must be a non-empty data.frame", call. = FALSE)
  }

  # Resolve fit object
  fit_obj <- intensity_fit$intensity_fit_obj %||% intensity_fit

  # Determine family name
  family_name <- if (!is.null(family)) {
    family$name
  } else if (!is.null(fit_obj$distribution)) {
    fit_obj$distribution
  } else {
    "negbin"
  }

  if (!is.null(fit_obj$gam_model)) {
    stop("PPC is currently supported only for parametric intensity models ",
         "(negbin, poisson, zinb), not GAM fits.", call. = FALSE)
  }

  if (!is.null(seed)) set.seed(seed)

  # --- Pre-extract raster data (once) ---
  C_vals    <- terra::values(connectivity)[, 1]
  valid     <- !is.na(C_vals)
  mc        <- fit_obj$min_connectivity %||% 0
  C_safe    <- pmax(C_vals[valid], mc)
  z_vals    <- (log1p(C_safe / fit_obj$c_scale) - fit_obj$log_conn_mean) /
               fit_obj$log_conn_sd
  cell_area <- prod(terra::res(connectivity))

  # Covariate values at valid cells
  cov_vals <- NULL
  cov_names_vec <- character(0)
  if (!is.null(covariates_rasters) && length(covariates_rasters) > 0) {
    cov_names_vec <- names(covariates_rasters)
    cov_vals <- lapply(covariates_rasters, function(r) {
      v <- terra::values(r)[, 1]
      pmax(pmin(v[valid], 1), 0)
    })
  }

  # Observed counts per valid cell
  coords    <- as_coord_matrix(obs_points)
  cells     <- terra::cellFromXY(connectivity, coords)
  valid_idx <- which(valid)
  n_valid   <- length(valid_idx)

  obs_counts <- rep(0L, n_valid)
  cell_map   <- match(cells, valid_idx)
  cell_map   <- cell_map[!is.na(cell_map)]
  tab        <- table(cell_map)
  obs_counts[as.integer(names(tab))] <- as.integer(tab)

  cell_coords <- if (include_moran) {
    terra::xyFromCell(connectivity, valid_idx)
  } else {
    NULL
  }

  # --- Select posterior draws ---
  draw_idx <- seq(1L, nrow(posterior_samples), by = thin)
  if (length(draw_idx) > n_sim) {
    draw_idx <- sort(sample(draw_idx, n_sim))
  } else if (length(draw_idx) < n_sim) {
    if (verbose) {
      message(sprintf("  Only %d posterior draws available (requested %d); using all.",
                      length(draw_idx), n_sim))
    }
  }
  actual_n_sim <- length(draw_idx)

  # --- Compute MAP expected counts for observed test quantities ---
  params_vec <- fit_obj$estimates %||% intensity_fit$intensity_params
  map_alpha  <- params_vec[["alpha"]]
  if (is.null(map_alpha) || is.na(map_alpha)) map_alpha <- 0
  map_gamma  <- params_vec[["gamma"]]
  map_size   <- params_vec[["size"]]

  map_log_lambda <- map_alpha + map_gamma * z_vals
  if (length(cov_names_vec) > 0) {
    for (nm in cov_names_vec) {
      b <- params_vec[[paste0("beta_", nm)]]
      if (!is.null(b) && !is.na(b)) {
        map_log_lambda <- map_log_lambda + b * cov_vals[[nm]]
      }
    }
  }
  map_mu <- exp(map_log_lambda) * cell_area

  tq_all <- test_quantities
  if (include_moran) tq_all <- union(tq_all, "moran_i")

  obs_tq <- .ppc_test_quantities(
    obs_counts, map_mu, tq_all,
    family = family, size = map_size,
    coords = cell_coords, include_moran = include_moran
  )

  # --- Simulation loop ---
  sim_tq <- lapply(tq_all, function(nm) numeric(actual_n_sim))
  names(sim_tq) <- tq_all

  for (i in seq_along(draw_idx)) {
    if (verbose && i %% 50 == 0) {
      message(sprintf("  PPC simulation %d/%d", i, actual_n_sim))
    }

    row <- posterior_samples[draw_idx[i], , drop = FALSE]

    alpha_i <- row[["alpha"]]
    if (is.null(alpha_i) || is.na(alpha_i)) alpha_i <- 0
    gamma_i <- row[["gamma"]]
    size_i  <- row[["size"]]

    log_lambda_i <- alpha_i + gamma_i * z_vals
    if (length(cov_names_vec) > 0) {
      for (nm in cov_names_vec) {
        b <- row[[paste0("beta_", nm)]]
        if (!is.null(b) && !is.na(b)) {
          log_lambda_i <- log_lambda_i + b * cov_vals[[nm]]
        }
      }
    }
    mu_cells_i <- exp(log_lambda_i) * cell_area

    pi_val_i <- if (family_name == "zinb" && "logit_pi" %in% names(row)) {
      1 / (1 + exp(-row[["logit_pi"]]))
    } else {
      NULL
    }

    sim_counts <- .ppc_simulate_counts(mu_cells_i, family_name,
                                       size = size_i, pi_val = pi_val_i)

    sim_vals <- .ppc_test_quantities(
      sim_counts, mu_cells_i, tq_all,
      family = family, size = size_i,
      coords = cell_coords, include_moran = include_moran
    )

    for (nm in tq_all) {
      sim_tq[[nm]][i] <- sim_vals[[nm]]
    }
  }

  # --- Bayesian p-values (two-sided) ---
  bayesian_p <- vapply(tq_all, function(nm) {
    obs_val <- obs_tq[[nm]]
    sim_vec <- sim_tq[[nm]]
    if (is.na(obs_val) || all(is.na(sim_vec))) return(NA_real_)
    p_upper <- mean(sim_vec >= obs_val, na.rm = TRUE)
    p_lower <- mean(sim_vec <= obs_val, na.rm = TRUE)
    min(2 * min(p_upper, p_lower), 1)
  }, numeric(1))

  if (verbose) {
    message("\n  Posterior predictive check summary:")
    for (nm in tq_all) {
      message(sprintf("    %-15s  observed = %8.2f  p = %.3f",
                      nm, obs_tq[[nm]], bayesian_p[[nm]]))
    }
  }

  result <- structure(
    list(
      observed        = obs_tq,
      simulated       = sim_tq,
      bayesian_p      = bayesian_p,
      n_sim           = actual_n_sim,
      test_quantities = tq_all
    ),
    class = "ds_ppc"
  )

  if (plot) {
    plot_ppc(result)
  }

  result
}


#' Plot posterior predictive check results
#'
#' Produces a multi-panel histogram comparing observed test quantities
#' (red vertical line) against the posterior predictive distribution
#' (steelblue histograms).
#'
#' @param ppc_result Object returned by [ds_ppc()].
#' @param ... Extra arguments passed to [graphics::hist()].
#' @return Invisible `NULL`.
#' @seealso [ds_ppc()]
#' @export
plot_ppc <- function(ppc_result, ...) {

  tq <- ppc_result$test_quantities
  n_panels <- length(tq)
  nc <- ceiling(sqrt(n_panels))
  nr <- ceiling(n_panels / nc)

  op <- graphics::par(mfrow = c(nr, nc))
  on.exit(graphics::par(op))

  for (nm in tq) {
    sim_vals <- ppc_result$simulated[[nm]]
    obs_val  <- ppc_result$observed[[nm]]
    p_val    <- ppc_result$bayesian_p[[nm]]

    graphics::hist(sim_vals, 40, freq = FALSE,
                   col = "steelblue", border = "white",
                   main = sprintf("PPC: %s  (p = %.3f)", nm, p_val),
                   xlab = nm, ...)
    graphics::abline(v = obs_val, col = "red", lwd = 2)
    graphics::legend("topright", legend = "observed",
                     col = "red", lwd = 2, bty = "n", cex = 0.8)
  }

  invisible(NULL)
}
