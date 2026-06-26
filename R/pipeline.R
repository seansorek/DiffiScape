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
#' Wrapper that dispatches to [optimize_resistance()] (GP surrogate),
#' [optimize_resistance_gradient()] (L-BFGS / Adam via JAX auto-diff,
#' or Flax neural-network resistance when `model_type` is set),
#' [run_torch_pipeline()] (PyTorch neural-network resistance), depending
#' on `solver`.
#'
#' @param basis_stack Basis function stack.
#' @param obs_points Data.frame with `x, y`.
#' @param bounds Optional parameter bounds.
#' @param config Optimiser config (see [default_optimizer_config()]).
#'   For `solver = "torch"`, may contain a `torch` sublist whose entries
#'   are forwarded to [run_torch_pipeline()].
#'   For `solver = "gradient"` with a neural `model_type`, may contain
#'   `model_config` and `optim_config` sublists.
#' @param intensity_config Intensity config (see [default_intensity_config()]).
#' @param output_dir Directory for logs.
#' @param covariates_obs Named list of covariate vectors.
#' @param covariates_rasters Named list of covariate rasters.
#' @param residualise Logical.
#' @param available_points Optional data.frame with `x, y` columns of
#'   available/background locations for selection function families.  When
#'   supplied, bypasses raster quadrature and uses these locations with unit
#'   weights instead.  `NULL` (default) uses standard area-weighted raster
#'   integration.
#' @param available_covariates Named list of covariate vectors at
#'   `available_points` locations.  Required when `available_points` is
#'   supplied and the intensity model includes covariates.
#' @param solver Character; `"surrogate"` (GP + Thompson Sampling or
#'   Expected Improvement, default), `"gradient"` (L-BFGS / Adam via JAX
#'   auto-diff), `"enzyme"` (deprecated, alias for `"gradient"`),
#'   `"torch"` (PyTorch neural-network resistance), or `"irl"`
#'   (PyTorch value-shaped resistance: a reward network is turned into a
#'   resistance surface via soft value iteration, then run through the same
#'   differentiable circuit solver -- a convenience alias for `solver = "torch"`
#'   with `model_type = "irl"`).
#' @param model_type Character; `"parametric"` (default), `"mlp"`,
#'   `"conv"`, `"spline_gam"`, or `"irl"`.  When not `"parametric"`,
#'   the gradient solver dispatches to the Flax neural-network
#'   optimizer instead of the parametric L-BFGS/Adam path.
#' @return Result from [optimize_resistance()],
#'   [optimize_resistance_gradient()], or [run_torch_pipeline()].
#' @export
ds_optimize <- function(basis_stack,
                        obs_points,
                        bounds               = NULL,
                        config               = default_optimizer_config(),
                        intensity_config     = default_intensity_config(),
                        output_dir           = tempdir(),
                        covariates_obs       = NULL,
                        covariates_rasters   = NULL,
                        residualise          = FALSE,
                        available_points     = NULL,
                        available_covariates = NULL,
                        solver               = c("surrogate", "gradient",
                                                 "enzyme", "torch", "irl"),
                        model_type           = "parametric") {

  solver <- match.arg(solver)

  # Deprecation aliases
  if (solver == "enzyme") {
    message("solver='enzyme' is deprecated. Use solver='gradient' instead.")
    solver <- "gradient"
  }

  if (solver == "gradient") {
    return(optimize_resistance_gradient(
      basis_stack          = basis_stack,
      obs_points           = obs_points,
      bounds               = bounds,
      config               = config,
      intensity_config     = intensity_config,
      output_dir           = output_dir,
      covariates_obs       = covariates_obs,
      covariates_rasters   = covariates_rasters,
      residualise          = residualise,
      available_points     = available_points,
      available_covariates = available_covariates,
      model_type           = model_type,
      model_config         = config$model_config %||% list(),
      optim_config         = config$optim_config %||% list()
    ))
  }

  if (solver == "torch" || solver == "irl") {
    if (!is.null(available_points)) {
      stop("available_points is not supported with solver = '", solver, "'.",
           call. = FALSE)
    }
    torch_args <- config$torch %||% list()
    torch_args$basis_stack <- basis_stack
    torch_args$obs_points  <- obs_points
    if (is.null(torch_args$output_dir)) torch_args$output_dir <- output_dir
    if (is.null(torch_args$seed) && !is.null(config$seed)) {
      torch_args$seed <- config$seed
    }
    # "irl" is "torch" with the value-shaped resistance model.
    if (solver == "irl") torch_args$model_type <- "irl"
    return(do.call(run_torch_pipeline, torch_args))
  }

  opt_fn <- if (solver == "enzyme") optimize_resistance_enzyme
            else optimize_resistance

  opt_fn(
    basis_stack          = basis_stack,
    obs_points           = obs_points,
    bounds               = bounds,
    config               = config,
    intensity_config     = intensity_config,
    output_dir           = output_dir,
    covariates_obs       = covariates_obs,
    covariates_rasters   = covariates_rasters,
    residualise          = residualise,
    available_points     = available_points,
    available_covariates = available_covariates
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
#' @param available_points Optional data.frame with `x, y` columns of
#'   available/background locations for selection function families.  When
#'   supplied, bypasses raster quadrature and uses these locations with unit
#'   weights instead.  `NULL` (default) uses standard area-weighted raster
#'   integration.
#' @param available_covariates Named list of covariate vectors at
#'   `available_points` locations.  Required when `available_points` is
#'   supplied and the intensity model includes covariates.
#' @param solver Character; `"surrogate"` (Omniscape), `"gradient"` (JAX
#'   differentiable solver), or `"enzyme"` (deprecated, alias for
#'   `"gradient"`).
#' @param link A [resistance_link] object (default [link_exp()]).
#' @param family An [intensity_family] object, or `NULL`.
#' @return Result from [evaluate_full_model()].
#' @export
ds_fit_intensity <- function(opt_result,
                              basis_stack,
                              obs_points,
                              omniscape_settings   = list(),
                              intensity_config     = default_intensity_config(),
                              covariates_obs       = NULL,
                              covariates_rasters   = NULL,
                              residualise          = FALSE,
                              available_points     = NULL,
                              available_covariates = NULL,
                              solver               = c("surrogate", "gradient", "enzyme"),
                              link                 = link_exp(),
                              family               = NULL) {

  solver <- match.arg(solver)

  if (solver == "enzyme") {
    message("solver='enzyme' is deprecated. Use solver='gradient' instead.")
    solver <- "gradient"
  }

  if (solver == "gradient") {
    # Use the JAX differentiable solver
    resistance <- create_resistance_surface(opt_result$best_params, basis_stack,
                                            link = link)
    omni <- ds_jax_connectivity(
      resistance,
      radius     = omniscape_settings$radius     %||% 13L,
      block_size = omniscape_settings$block_size  %||% 5L
    )
    connectivity <- omni$cum_current
    conn_obs <- extract_connectivity(connectivity, obs_points)

    valid <- !is.na(conn_obs)
    distribution <- opt_result$distribution %||% "negbin"
    fit_fn <- switch(distribution,
      negbin = fit_intensity_nb, gam = fit_intensity_gam)

    int_args <- list(
      connectivity_at_obs = conn_obs[valid],
      connectivity_raster = connectivity,
      obs_coords          = obs_points[valid, , drop = FALSE],
      covariates_obs      = if (!is.null(covariates_obs))
        lapply(covariates_obs, function(v) v[valid]) else NULL,
      covariates_rasters  = covariates_rasters,
      residualise         = residualise,
      config              = intensity_config
    )
    if (distribution != "gam" && !is.null(family))
      int_args$family <- family

    if (!is.null(available_points)) {
      avail_conn_raw <- extract_connectivity(connectivity, available_points)
      avail_valid    <- !is.na(avail_conn_raw)
      avail_conn     <- avail_conn_raw[avail_valid]
      avail_cov      <- if (!is.null(available_covariates))
        lapply(available_covariates, function(v) v[avail_valid]) else NULL
      int_args$available_connectivity <- avail_conn
      int_args$available_covariates   <- avail_cov
    }

    int_fit <- do.call(fit_fn, int_args)

    return(list(
      loglik           = int_fit$loglik,
      intensity_params = int_fit$estimates,
      intensity_fit_obj = int_fit,
      intensity_se     = int_fit$se,
      hessian          = int_fit$hessian,
      convergence      = int_fit$convergence,
      distribution     = distribution,
      total_time       = omni$elapsed_seconds
    ))
  }

  evaluate_full_model(
    resistance_params    = opt_result$best_params,
    basis_stack          = basis_stack,
    obs_points           = obs_points,
    distribution         = opt_result$distribution,
    omniscape_settings   = omniscape_settings,
    intensity_config     = intensity_config,
    covariates_obs       = covariates_obs,
    covariates_rasters   = covariates_rasters,
    residualise          = residualise,
    link                 = link,
    family               = family,
    available_points     = available_points,
    available_covariates = available_covariates
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

  fit_obj <- intensity_fit$intensity_fit_obj %||% intensity_fit
  predict_intensity(
    fit_obj,
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
#' @param available_points Optional data.frame with `x, y` columns of
#'   available/background locations for selection function families.  When
#'   supplied, bypasses raster quadrature and uses these locations with unit
#'   weights instead.  `NULL` (default) uses standard area-weighted raster
#'   integration.
#' @param available_covariates Named list of covariate vectors at
#'   `available_points` locations.  Required when `available_points` is
#'   supplied and the intensity model includes covariates.
#' @param link A [resistance_link] object (default [link_exp()]).
#' @param family An [intensity_family] object, or `NULL`.
#' @return A list with `laplace`, `samples`, `summary`.
#' @export
ds_posterior <- function(opt_result,
                         basis_stack,
                         obs_points,
                         n_draws              = 200L,
                         n_inner              = 5L,
                         omniscape_settings   = list(),
                         intensity_config     = default_intensity_config(),
                         covariates_obs       = NULL,
                         covariates_rasters   = NULL,
                         residualise          = FALSE,
                         available_points     = NULL,
                         available_covariates = NULL,
                         link                 = link_exp(),
                         family               = NULL) {

  use_refit <- TRUE

  lap <- laplace_resistance(
    opt_result,
    basis_stack          = basis_stack,
    obs_points           = obs_points,
    refit                = use_refit,
    omniscape_settings   = omniscape_settings,
    intensity_config     = intensity_config,
    covariates_obs       = covariates_obs,
    covariates_rasters   = covariates_rasters,
    residualise          = residualise,
    available_points     = available_points,
    available_covariates = available_covariates,
    link                 = link,
    family               = family
  )

  samp <- posterior_sample(
    laplace              = lap,
    opt_result           = opt_result,
    basis_stack          = basis_stack,
    obs_points           = obs_points,
    n_draws              = n_draws,
    n_inner              = n_inner,
    bounds               = opt_result$bounds,
    omniscape_settings   = omniscape_settings,
    intensity_config     = intensity_config,
    covariates_obs       = covariates_obs,
    covariates_rasters   = covariates_rasters,
    residualise          = residualise,
    available_points     = available_points,
    available_covariates = available_covariates,
    link                 = link,
    family               = family
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
#' @param family An [intensity_family] object, or `NULL`.
#' @param plot Logical; produce diagnostic plots.
#' @return Result from [diagnose_model()].
#' @export
ds_diagnose <- function(intensity_fit,
                        obs_points,
                        connectivity,
                        intensity_config   = default_intensity_config(),
                        covariates_rasters = NULL,
                        family             = NULL,
                        plot               = TRUE) {

  diagnose_model(
    intensity_fit      = intensity_fit,
    obs_points         = obs_points,
    connectivity       = connectivity,
    intensity_config   = intensity_config,
    covariates_rasters = covariates_rasters,
    family             = family,
    plot               = plot
  )
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
#' @param available_points Optional data.frame with `x, y` columns of
#'   available/background locations for selection function families.  When
#'   supplied, bypasses raster quadrature and uses these locations with unit
#'   weights instead.  `NULL` (default) uses standard area-weighted raster
#'   integration.
#' @param available_covariates Named list of covariate vectors at
#'   `available_points` locations.  Required when `available_points` is
#'   supplied and the intensity model includes covariates.
#' @param plot Logical; produce diagnostic plots.
#' @param crs Target CRS for reprojection (if input is spatial file).
#' @param rescale_basis Logical; rescale basis rasters.
#' @param pattern File pattern for raster directory.
#' @param omniscape_settings Named list of Omniscape overrides forwarded to
#'   every connectivity step (final refit, posterior sampling, and
#'   diagnostics).  Recognised entries: `radius` (default `13L`),
#'   `block_size` (default `5L`), `cleanup` (default `TRUE`).
#' @param solver Character; `"surrogate"` (default, GP surrogate optimiser),
#'   `"gradient"` (L-BFGS / Adam via JAX auto-diff), or `"enzyme"` (deprecated,
#'   alias for `"gradient"`).
#' @return A list with `obs_points`, `basis_stack`, `opt_result`,
#'   `intensity_fit`, `posterior`, `diagnostics`.
#' @export
diffiscape <- function(obs_data,
                       rasters,
                       julia_home           = NULL,
                       optimizer_config     = default_optimizer_config(),
                       intensity_config     = default_intensity_config(),
                       bounds               = NULL,
                       output_dir           = "diffiscape_output",
                       n_posterior          = 200L,
                       covariates_obs       = NULL,
                       covariates_rasters   = NULL,
                       residualise          = FALSE,
                       available_points     = NULL,
                       available_covariates = NULL,
                       plot                 = TRUE,
                       crs                  = NULL,
                       rescale_basis        = TRUE,
                       pattern              = "*.tif",
                       omniscape_settings   = list(),
                       solver               = c("surrogate", "gradient", "enzyme")) {

  solver <- match.arg(solver)

  # Deprecation alias
  if (solver == "enzyme") {
    message("solver='enzyme' is deprecated. Use solver='gradient' instead.")
    solver <- "gradient"
  }

  # Extract link and family from configs
  res_link   <- optimizer_config$resistance_link %||% link_exp()
  int_family <- optimizer_config$family %||% intensity_config$family

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
    basis_stack          = basis_stack,
    obs_points           = obs_points,
    bounds               = bounds,
    config               = optimizer_config,
    intensity_config     = intensity_config,
    output_dir           = output_dir,
    covariates_obs       = covariates_obs,
    covariates_rasters   = covariates_rasters,
    residualise          = residualise,
    available_points     = available_points,
    available_covariates = available_covariates,
    solver               = solver
  )

  # --- Step 5: Final intensity fit ---
  message("\n[5/7] Fitting final intensity model...")
  intensity_fit <- ds_fit_intensity(
    opt_result           = opt_result,
    basis_stack          = basis_stack,
    obs_points           = obs_points,
    omniscape_settings   = omniscape_settings,
    intensity_config     = intensity_config,
    covariates_obs       = covariates_obs,
    covariates_rasters   = covariates_rasters,
    residualise          = residualise,
    available_points     = available_points,
    available_covariates = available_covariates,
    solver               = solver,
    link                 = res_link,
    family               = int_family
  )

  # --- Step 6: Posterior ---
  posterior <- NULL
  if (n_posterior > 0) {
    message("\n[6/7] Posterior inference...")
    posterior <- ds_posterior(
      opt_result           = opt_result,
      basis_stack          = basis_stack,
      obs_points           = obs_points,
      n_draws              = n_posterior,
      omniscape_settings   = omniscape_settings,
      intensity_config     = intensity_config,
      covariates_obs       = covariates_obs,
      covariates_rasters   = covariates_rasters,
      residualise          = residualise,
      available_points     = available_points,
      available_covariates = available_covariates,
      link                 = res_link,
      family               = int_family
    )
  } else {
    message("\n[6/7] Posterior inference... SKIPPED")
  }

  # --- Step 7: Diagnostics ---
  message("\n[7/7] Diagnostics...")
  final_resistance  <- create_resistance_surface(opt_result$best_params,
                                                   basis_stack, link = res_link)
  omni_def <- list(radius = 13L, block_size = 5L, cleanup = TRUE)
  omni_cfg <- utils::modifyList(omni_def, omniscape_settings)
  if (solver == "gradient") {
    final_omni <- ds_jax_connectivity(final_resistance,
                                       radius     = omni_cfg$radius,
                                       block_size = omni_cfg$block_size)
  } else {
    final_omni <- run_omniscape(final_resistance,
                                radius     = omni_cfg$radius,
                                block_size = omni_cfg$block_size,
                                cleanup    = omni_cfg$cleanup)
  }
  final_connectivity <- final_omni$cum_current

  diagnostics <- ds_diagnose(
    intensity_fit      = intensity_fit,
    obs_points         = obs_points,
    connectivity       = final_connectivity,
    intensity_config   = intensity_config,
    covariates_rasters = covariates_rasters,
    family             = int_family,
    plot               = plot
  )

  ppc <- NULL
  if (!is.null(posterior) && nrow(posterior$samples) > 0) {
    message("\n  Posterior predictive checks...")
    ppc <- tryCatch(
      ds_ppc(
        posterior_samples  = posterior$samples,
        intensity_fit      = intensity_fit,
        obs_points         = obs_points,
        connectivity       = final_connectivity,
        intensity_config   = intensity_config,
        covariates_rasters = covariates_rasters,
        family             = int_family,
        plot               = plot
      ),
      error = function(e) {
        message("  PPC skipped: ", conditionMessage(e))
        NULL
      }
    )
  }

  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  message(sprintf("\nPipeline complete in %.1f minutes", elapsed))

  result <- list(
    obs_points     = obs_points,
    basis_stack    = basis_stack,
    opt_result     = opt_result,
    intensity_fit  = intensity_fit,
    posterior      = posterior,
    diagnostics    = diagnostics,
    ppc            = ppc,
    elapsed_min    = elapsed
  )

  saveRDS(result, file.path(output_dir, "diffiscape_result.rds"))
  result
}
