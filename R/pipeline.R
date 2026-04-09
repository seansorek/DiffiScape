# ============================================================================
# High-level pipeline API
#
# One-call wrapper `diffiscape()` + modular step functions:
#   ds_load_data  ->  ds_create_basis  ->  ds_init_julia  ->
#   ds_optimize   ->  ds_fit_intensity ->  ds_predict     ->
#   ds_posterior   ->  ds_diagnose
# ============================================================================

# TODO Refactor these functions to use S3 methods, especially for predict and summary functions.

# --------------- Modular step functions -------------------------------------

#' Load observation data
#'
#' Reads GPS point data from a CSV or shapefile and returns a clean
#' data.frame with `x, y` columns (projected coordinates).
#'
#' @param path File path to a CSV, shapefile (.shp), or GeoPackage.
#'   Must contain geographic coordinates.
#' @param crs Target CRS for reprojection ($proj4string or EPSG code).
#'   If `NULL`, the native CRS of the input is used.
#' @param x_col,y_col Column names for coordinates in a CSV.
#' @return A data.frame with at least `x` and `y` columns.
#' @export
ds_load_data <- function(path, crs = NULL, x_col = "x", y_col = "y") {

  ext <- tolower(tools::file_ext(path))

  if (ext == "csv") {
    pts <- utils::read.csv(path, stringsAsFactors = FALSE)
    if (!all(c(x_col, y_col) %in% names(pts))) {
      stop("CSV must contain columns '", x_col, "' and '", y_col, "'",
           call. = FALSE)
    }
    names(pts)[names(pts) == x_col] <- "x"
    names(pts)[names(pts) == y_col] <- "y"

  } else if (ext %in% c("shp", "gpkg", "geojson")) {
    v <- terra::vect(path)
    if (!is.null(crs)) v <- terra::project(v, crs)
    crds <- terra::crds(v)
    pts <- as.data.frame(v)
    pts$x <- crds[, 1]
    pts$y <- crds[, 2]

  } else {
    stop("Unsupported file type: ", ext, call. = FALSE)
  }

  if (!is.null(crs) && ext == "csv") {
    message("Note: CSV coordinates assumed already in target CRS")
  }

  pts
}


#' Create a basis function stack from raster files
#'
#' Wrapper around [create_basis_stack()] that accepts a directory or
#' file paths.
#'
#' @param rasters Either a character vector of raster file paths, a directory
#'   path, or a named list of [terra::SpatRaster] objects.
#' @param pattern Glob pattern to match files in a directory
#'   (default `"*.tif"`).
#' @param rescale Logical; rescale rasters to \[0, 1\].
#' @param mask_to Optional [terra::SpatRaster] to unify masks.
#' @return A [terra::SpatRaster] basis stack.
#' @export
ds_create_basis <- function(rasters,
                            pattern  = "*.tif",
                            rescale  = TRUE,
                            mask_to  = NULL) {

  if (is.character(rasters) && length(rasters) == 1 && dir.exists(rasters)) {
    files   <- list.files(rasters, pattern = utils::glob2rx(pattern),
                          full.names = TRUE)
    if (length(files) == 0) stop("No rasters found in ", rasters, call. = FALSE)
    rasters <- stats::setNames(
      lapply(files, terra::rast),
      tools::file_path_sans_ext(basename(files))
    )
  } else if (is.character(rasters)) {
    rasters <- stats::setNames(
      lapply(rasters, terra::rast),
      tools::file_path_sans_ext(basename(rasters))
    )
  }

  create_basis_stack(rasters, rescale = rescale, mask_to = mask_to)
}


#' Initialise the Julia backend
#'
#' Wrapper around [ds_julia_setup()] with user-friendly error messages.
#'
#' @param julia_home Path to the Julia binary. If `NULL`, uses PATH.
#' @param force Re-initialise even if already set up.
#' @return Invisible `TRUE` on success.
#' @export
ds_init_julia <- function(julia_home = NULL, force = FALSE) {
  tryCatch(
    ds_julia_setup(julia_home = julia_home, force = force),
    error = function(e) {
      stop(
        "Julia initialisation failed.\n",
        "Make sure Julia >= 1.9 is installed and JuliaConnectoR is available.\n",
        "Original error: ", conditionMessage(e),
        call. = FALSE
      )
    }
  )
  invisible(TRUE)
}


#' Optimise resistance parameters
#'
#' Wrapper around [optimize_resistance()] with sensible defaults.
#'
#' @param basis_stack Basis function stack.
#' @param obs_points Data.frame with `x, y`.
#' @param bounds Optional parameter bounds.
#' @param config Optimiser config (see [default_optimizer_config()]).
#' @param intensity_config Intensity config (see [default_intensity_config()]).
#' @param output_dir Directory for logs.
#' @param covariates_obs Named list of covariate vectors.
#' @param covariates_rasters Named list of covariate rasters.
#' @param residualise Logical.
#' @return Result from [optimize_resistance()].
#' @export
ds_optimize <- function(basis_stack,
                        obs_points,
                        bounds           = NULL,
                        config           = default_optimizer_config(),
                        intensity_config = default_intensity_config(),
                        output_dir       = tempdir(),
                        covariates_obs   = NULL,
                        covariates_rasters = NULL,
                        residualise      = FALSE) {

  optimize_resistance(
    basis_stack        = basis_stack,
    obs_points         = obs_points,
    bounds             = bounds,
    config             = config,
    intensity_config   = intensity_config,
    output_dir         = output_dir,
    covariates_obs     = covariates_obs,
    covariates_rasters = covariates_rasters,
    residualise        = residualise
  )
}


#' Fit the final intensity model at the optimised parameters
#'
#' Rebuilds connectivity at the MAP resistance parameters and fits the
#' intensity model.
#'
#' @param opt_result Result from [ds_optimize()].
#' @param basis_stack Basis function stack.
#' @param obs_points Data.frame with `x, y`.
#' @param omniscape_settings List of Omniscape settings.
#' @param intensity_config Intensity config.
#' @param covariates_obs Named list of covariate vectors.
#' @param covariates_rasters Named list of covariate rasters.
#' @param residualise Logical.
#' @return Result from [evaluate_full_model()].
#' @export
ds_fit_intensity <- function(opt_result,
                              basis_stack,
                              obs_points,
                              omniscape_settings = list(),
                              intensity_config   = default_intensity_config(),
                              covariates_obs     = NULL,
                              covariates_rasters = NULL,
                              residualise        = FALSE) {

  evaluate_full_model(
    resistance_params  = opt_result$best_params,
    basis_stack        = basis_stack,
    obs_points         = obs_points,
    distribution       = opt_result$distribution,
    omniscape_settings = omniscape_settings,
    intensity_config   = intensity_config,
    covariates_obs     = covariates_obs,
    covariates_rasters = covariates_rasters,
    residualise        = residualise
  )
}


#' Predict intensity surface
#'
#' Generates a raster of the predicted intensity surface.
#'
#' @param intensity_fit Result from [ds_fit_intensity()].
#' @param connectivity A [terra::SpatRaster] of connectivity values.
#' @param intensity_config Intensity config.
#' @param covariates_rasters Named list of covariate rasters.
#' @return A [terra::SpatRaster] of predicted intensity.
#' @export
ds_predict <- function(intensity_fit,
                       connectivity,
                       intensity_config   = default_intensity_config(),
                       covariates_rasters = NULL) {

  predict_intensity(
    intensity_fit$intensity_params,
    connectivity,
    covariates_rasters = covariates_rasters,
    config             = intensity_config
  )
}


#' Posterior inference
#'
#' Wrapper around Laplace approximation + Monte Carlo composition.
#'
#' @param opt_result Result from [ds_optimize()].
#' @param basis_stack Basis function stack.
#' @param obs_points Data.frame with `x, y`.
#' @param n_draws Number of posterior draws.
#' @param n_inner Inner-loop draws per outer draw.
#' @param omniscape_settings List of Omniscape settings.
#' @param intensity_config Intensity config.
#' @param covariates_obs Named list of covariate vectors.
#' @param covariates_rasters Named list of covariate rasters.
#' @param residualise Logical.
#' @return A list with `laplace`, `samples`, `summary`.
#' @export
ds_posterior <- function(opt_result,
                         basis_stack,
                         obs_points,
                         n_draws            = 200L,
                         n_inner            = 5L,
                         omniscape_settings = list(),
                         intensity_config   = default_intensity_config(),
                         covariates_obs     = NULL,
                         covariates_rasters = NULL,
                         residualise        = FALSE) {

  lap <- laplace_resistance(opt_result)

  samp <- posterior_sample(
    laplace            = lap,
    opt_result         = opt_result,
    basis_stack        = basis_stack,
    obs_points         = obs_points,
    n_draws            = n_draws,
    n_inner            = n_inner,
    bounds             = opt_result$bounds,
    omniscape_settings = omniscape_settings,
    intensity_config   = intensity_config,
    covariates_obs     = covariates_obs,
    covariates_rasters = covariates_rasters,
    residualise        = residualise
  )

  summ <- posterior_summary(samp)

  list(laplace = lap, samples = samp, summary = summ)
}


#' Run diagnostics on a fitted model
#'
#' Wrapper around [compute_deviance_residuals()] and [diagnose_model()].
#'
#' @param intensity_fit Result from [ds_fit_intensity()].
#' @param obs_points Data.frame with `x, y`.
#' @param connectivity A [terra::SpatRaster].
#' @param intensity_config Intensity config.
#' @param covariates_rasters Named list of covariate rasters.
#' @param plot Logical; produce diagnostic plots.
#' @return Result from [diagnose_model()].
#' @export
ds_diagnose <- function(intensity_fit,
                        obs_points,
                        connectivity,
                        intensity_config   = default_intensity_config(),
                        covariates_rasters = NULL,
                        plot               = TRUE) {

  diagnose_model(
    intensity_fit      = intensity_fit,
    obs_points         = obs_points,
    connectivity       = connectivity,
    intensity_config   = intensity_config,
    covariates_rasters = covariates_rasters,
    plot               = plot
  )
}


#' Profile likelihood inference for resistance parameters
#'
#' Wrapper around [profile_ci()] and [profile_loglik()] that returns
#' confidence intervals and the full profile objects for all resistance
#' parameters.
#'
#' @param opt_result Result from [ds_optimize()].
#' @param level Confidence level (default 0.95).
#' @param n_points Grid resolution per parameter (default 50).
#' @param range_mult Multiplier on the Laplace SE for grid range
#'   (default 3).
#' @param plot Logical; produce profile-likelihood plots (default
#'   `TRUE`).
#' @return A list with `ci` (data.frame of confidence intervals) and
#'   `profiles` (named list of profile-likelihood objects).
#' @export
ds_profile <- function(opt_result,
                       level      = 0.95,
                       n_points   = 50L,
                       range_mult = 3,
                       plot       = TRUE) {

  ci <- profile_ci(opt_result,
                    level      = level,
                    n_points   = n_points,
                    range_mult = range_mult)

  profiles <- attr(ci, "profiles")

  if (plot) {
    for (p in names(profiles)) {
      plot_profile(profiles[[p]], level = level)
    }
  }

  list(ci = ci, profiles = profiles)
}


# --------------- One-call wrapper -------------------------------------------

#' Run the full DiffiScape pipeline
#'
#' A single high-level function that chains every step of the pipeline:
#' data loading, basis creation, Julia start-up, optimisation, intensity
#' fitting, posterior inference, and diagnostics.
#'
#' @param obs_data Either a file path (CSV / shapefile) or a data.frame
#'   with `x, y` columns.
#' @param rasters Either a directory, file paths, or a named list of
#'   [terra::SpatRaster] objects.
#' @param julia_home Path to the Julia binary (or `NULL` for PATH).
#' @param optimizer_config Config list (see [default_optimizer_config()]).
#' @param intensity_config Config list (see [default_intensity_config()]).
#' @param bounds Parameter bounds (or `NULL` for defaults).
#' @param output_dir Directory for all outputs.
#' @param n_posterior Number of posterior draws (`0` to skip).
#' @param covariates_obs Named list of covariate vectors.
#' @param covariates_rasters Named list of covariate rasters.
#' @param residualise Logical.
#' @param plot Logical; produce diagnostic plots.
#' @param crs Target CRS for reprojection (if input is spatial file).
#' @param rescale_basis Logical; rescale basis rasters.
#' @param pattern File pattern for raster directory.
#' @return A list with `obs_points`, `basis_stack`, `opt_result`,
#'   `intensity_fit`, `posterior`, `diagnostics`.
#' @export
diffiscape <- function(obs_data,
                       rasters,
                       julia_home        = NULL,
                       optimizer_config  = default_optimizer_config(),
                       intensity_config  = default_intensity_config(),
                       bounds            = NULL,
                       output_dir        = "diffiscape_output",
                       n_posterior        = 200L,
                       covariates_obs    = NULL,
                       covariates_rasters = NULL,
                       residualise       = FALSE,
                       plot              = TRUE,
                       crs               = NULL,
                       rescale_basis     = TRUE,
                       pattern           = "*.tif") {

  t0 <- Sys.time()
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  # --- Step 1: Load data ---
  message("\n[1/7] Loading observation data...")
  if (is.character(obs_data)) {
    obs_points <- ds_load_data(obs_data, crs = crs)
  } else {
    obs_points <- obs_data
    if (!all(c("x", "y") %in% names(obs_points))) {
      stop("obs_data must have 'x' and 'y' columns", call. = FALSE)
    }
  }
  message(sprintf("  %d observations loaded", nrow(obs_points)))

  # --- Step 2: Basis stack ---
  message("\n[2/7] Creating basis function stack...")
  basis_stack <- ds_create_basis(rasters, pattern = pattern,
                                 rescale = rescale_basis)
  message(sprintf("  %d basis functions", terra::nlyr(basis_stack)))

  # --- Step 3: Julia ---
  message("\n[3/7] Initialising Julia backend...")
  ds_init_julia(julia_home = julia_home)

  # --- Step 4: Optimise ---
  message("\n[4/7] Optimising resistance parameters...")
  opt_result <- ds_optimize(
    basis_stack        = basis_stack,
    obs_points         = obs_points,
    bounds             = bounds,
    config             = optimizer_config,
    intensity_config   = intensity_config,
    output_dir         = output_dir,
    covariates_obs     = covariates_obs,
    covariates_rasters = covariates_rasters,
    residualise        = residualise
  )

  # --- Step 5: Final intensity fit ---
  message("\n[5/7] Fitting final intensity model...")
  intensity_fit <- ds_fit_intensity(
    opt_result         = opt_result,
    basis_stack        = basis_stack,
    obs_points         = obs_points,
    intensity_config   = intensity_config,
    covariates_obs     = covariates_obs,
    covariates_rasters = covariates_rasters,
    residualise        = residualise
  )

  # --- Step 6: Posterior ---
  posterior <- NULL
  if (n_posterior > 0) {
    message("\n[6/7] Posterior inference...")
    posterior <- ds_posterior(
      opt_result         = opt_result,
      basis_stack        = basis_stack,
      obs_points         = obs_points,
      n_draws            = n_posterior,
      intensity_config   = intensity_config,
      covariates_obs     = covariates_obs,
      covariates_rasters = covariates_rasters,
      residualise        = residualise
    )
  } else {
    message("\n[6/7] Posterior inference... SKIPPED")
  }

  # --- Step 7: Diagnostics ---
  message("\n[7/7] Diagnostics...")
  final_resistance  <- create_resistance_surface(opt_result$best_params, basis_stack)
  final_omni        <- run_omniscape(final_resistance)
  final_connectivity <- final_omni$cum_current

  diagnostics <- ds_diagnose(
    intensity_fit      = intensity_fit,
    obs_points         = obs_points,
    connectivity       = final_connectivity,
    intensity_config   = intensity_config,
    covariates_rasters = covariates_rasters,
    plot               = plot
  )

  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  message(sprintf("\nPipeline complete in %.1f minutes", elapsed))

  result <- list(
    obs_points     = obs_points,
    basis_stack    = basis_stack,
    opt_result     = opt_result,
    intensity_fit  = intensity_fit,
    posterior      = posterior,
    diagnostics    = diagnostics,
    elapsed_min    = elapsed
  )

  saveRDS(result, file.path(output_dir, "diffiscape_result.rds"))
  result
}
