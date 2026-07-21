# ============================================================================
# High-level pipeline API
#
# One-call wrapper `diffiscape()` + modular step functions:
#   ds_load_data  ->  ds_create_basis  ->  ds_optimize  ->
#   ds_fit_intensity  ->  ds_predict   ->  ds_posterior  ->
#   ds_diagnose
# ============================================================================

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


#' Optimise resistance parameters
#'
#' Wrapper that dispatches to [optimize_resistance()] (GP surrogate),
#' [optimize_resistance_gradient()] (L-BFGS / Adam via JAX auto-diff,
#' or Flax neural-network resistance when `model_type` is set), or
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
#' @param solver Character; `"torch"` (PyTorch log-linear resistance by
#'   default), `"surrogate"` (GP + Thompson Sampling or Expected Improvement),
#'   `"gradient"` (L-BFGS / Adam via JAX
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
#' @section Config conventions:
#' The three backend families are deliberately *not* unified onto a common
#' config shape -- each keeps its pre-existing convention, and dispatch
#' (via an internal `solver_spec()` S3 mechanism) is the one place that
#' knows how to extract the right sublist for its solver:
#' \describe{
#'   \item{`solver = "surrogate"`}{Plain top-level `config` entries (see
#'     [default_optimizer_config()]). No sublist.}
#'   \item{`solver = "gradient"` / `"enzyme"`}{Top-level `config` entries
#'     for the parametric L-BFGS/Adam path, plus optional
#'     `config$model_config` / `config$optim_config` sublists when
#'     `model_type` selects a neural resistance model.}
#'   \item{`solver = "torch"` / `"irl"`}{A `config$torch` sublist forwarded
#'     as named arguments to [run_torch_pipeline()].}
#' }
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
                        solver               = c("torch", "surrogate", "gradient",
                                                 "enzyme", "irl"),
                        model_type           = "parametric") {

  solver <- match.arg(solver)
  spec <- solver_spec(solver)

  .ds_optimize_dispatch(
    spec,
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
    model_type           = model_type
  )
}


#' @keywords internal
.ds_optimize_dispatch.ds_solver_gradient <- function(spec,
                                                       basis_stack,
                                                       obs_points,
                                                       bounds,
                                                       config,
                                                       intensity_config,
                                                       output_dir,
                                                       covariates_obs,
                                                       covariates_rasters,
                                                       residualise,
                                                       available_points,
                                                       available_covariates,
                                                       model_type,
                                                       ...) {
  optimize_resistance_gradient(
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
  )
}


#' @keywords internal
.ds_optimize_dispatch.ds_solver_torch <- function(spec,
                                                    basis_stack,
                                                    obs_points,
                                                    bounds,
                                                    config,
                                                    intensity_config,
                                                    output_dir,
                                                    covariates_obs,
                                                    covariates_rasters,
                                                    residualise,
                                                    available_points,
                                                    available_covariates,
                                                    model_type,
                                                    ...) {
  if (!is.null(available_points)) {
    stop("available_points is not supported with solver = '", spec$solver, "'.",
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
  if (spec$solver == "irl") torch_args$model_type <- "irl"
  do.call(run_torch_pipeline, torch_args)
}


#' @keywords internal
.ds_optimize_dispatch.ds_solver_irl <- .ds_optimize_dispatch.ds_solver_torch


#' @keywords internal
.ds_optimize_dispatch.ds_solver_surrogate <- function(spec,
                                                        basis_stack,
                                                        obs_points,
                                                        bounds,
                                                        config,
                                                        intensity_config,
                                                        output_dir,
                                                        covariates_obs,
                                                        covariates_rasters,
                                                        residualise,
                                                        available_points,
                                                        available_covariates,
                                                        model_type,
                                                        ...) {
  optimize_resistance(
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
#' @param solver Character; `"surrogate"`, `"gradient"`, or `"enzyme"`
#'   (deprecated, alias for `"gradient"`).  Retained for compatibility with
#'   [ds_optimize()]'s solver dispatch and to preserve the `"enzyme"`
#'   deprecation message, but no longer affects this function's internal
#'   connectivity-computation logic: both `"surrogate"` and `"gradient"`
#'   route through the same [evaluate_full_model()] call (JAX connectivity
#'   via [ds_jax_connectivity()]).  `"torch"` and `"irl"` are accepted as
#'   *values* (so a caller forwarding [ds_optimize()]'s solver gets a
#'   clear, actionable error instead of a generic `match.arg()` failure)
#'   but are **not supported** by this function: those backends fit
#'   resistance and intensity jointly inside a single [ds_optimize()] /
#'   [run_torch_pipeline()] call, so there is nothing left for
#'   `ds_fit_intensity()` to do for them -- calling this function with
#'   `solver = "torch"` or `"irl"` raises an error directing you to the
#'   already-complete result from [run_torch_pipeline()].
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
                              solver               = c("surrogate", "gradient",
                                                       "enzyme", "torch", "irl"),
                              link                 = link_exp(),
                              family               = NULL) {

  solver <- match.arg(solver)
  spec <- solver_spec(solver)

  .ds_fit_intensity_dispatch(
    spec,
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
    link                 = link,
    family               = family
  )
}


#' @keywords internal
.ds_fit_intensity_dispatch.ds_solver_surrogate <- function(spec,
                                                             opt_result,
                                                             basis_stack,
                                                             obs_points,
                                                             omniscape_settings,
                                                             intensity_config,
                                                             covariates_obs,
                                                             covariates_rasters,
                                                             residualise,
                                                             available_points,
                                                             available_covariates,
                                                             link,
                                                             family,
                                                             ...) {
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


#' @keywords internal
.ds_fit_intensity_dispatch.ds_solver_gradient <-
  .ds_fit_intensity_dispatch.ds_solver_surrogate


#' @keywords internal
.ds_fit_intensity_dispatch.ds_solver_torch <- function(spec, ...) {
  stop(
    "ds_fit_intensity() does not support solver = '", spec$solver, "' -- ",
    "these backends fit resistance and intensity jointly in a single ",
    "ds_optimize() call; the result already contains intensity_params, ",
    "intensity_se, etc. See ?run_torch_pipeline.",
    call. = FALSE
  )
}


#' @keywords internal
.ds_fit_intensity_dispatch.ds_solver_irl <-
  .ds_fit_intensity_dispatch.ds_solver_torch


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
#' data loading, basis creation, optimisation, intensity fitting,
#' posterior inference, and diagnostics.
#'
#' @param obs_data Either a file path (CSV / shapefile) or a data.frame
#'   with `x, y` columns.
#' @param rasters Either a directory, file paths, or a named list of
#'   [terra::SpatRaster] objects.
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
#' @param solver Character; `"torch"` (default, PyTorch log-linear resistance),
#'   `"surrogate"` (GP surrogate optimiser), `"gradient"` (L-BFGS / Adam via
#'   JAX auto-diff), `"enzyme"` (deprecated, alias for `"gradient"`),
#'   or `"irl"` (PyTorch value-shaped resistance). For `"torch"` / `"irl"`,
#'   [run_torch_pipeline()] fits resistance and intensity jointly in a
#'   single call, so the separate intensity-fitting step is skipped and
#'   `posterior`/`diagnostics`/`ppc` are `NULL` (those require either a
#'   parametric `bounds`-based Laplace approximation or a
#'   `fit_intensity_nb()`/`fit_intensity_gam()`-shaped fit object that the
#'   torch backend does not produce).
#' @return A list with `obs_points`, `basis_stack`, `opt_result`,
#'   `intensity_fit`, `posterior`, `diagnostics`.
#' @export
diffiscape <- function(obs_data,
                       rasters,
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
                       solver               = c("torch", "surrogate", "gradient",
                                                "enzyme", "irl")) {

  solver <- match.arg(solver)
  spec <- solver_spec(solver)
  solver <- spec$solver

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

  # --- Step 3: Optimise ---
  message("\n[3/6] Optimising resistance parameters...")
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

  if (solver %in% c("torch", "irl")) {
    # --- torch/irl: resistance + intensity are fit jointly inside
    # ds_optimize() -> run_torch_pipeline(), so there is no separate
    # intensity-fitting step. Build the normal intensity_fit shape directly
    # from fields already present on opt_result (see run_torch_pipeline()'s
    # return value in R/torch_pipeline.R).
    message("\n[4/6] Fitting final intensity model... SKIPPED ",
            "(solver = '", solver, "' fits resistance + intensity jointly)")
    intensity_fit <- list(
      loglik             = opt_result$best_loglik,
      intensity_params   = opt_result$intensity_params,
      # No fit_intensity_nb()/fit_intensity_gam()-shaped object exists for
      # the torch/irl path (predict_intensity() / diagnose_model() require
      # $estimates, $c_scale, $log_conn_mean, $log_conn_sd, none of which
      # the neural-network intensity head produces) -- NULL rather than a
      # fabricated stand-in.
      intensity_fit_obj  = NULL,
      intensity_se       = opt_result$intensity_se,
      # No Hessian is computed by the Adam/MAP torch optimiser.
      hessian            = NULL,
      convergence        = opt_result$convergence,
      distribution       = opt_result$distribution,
      total_time         = opt_result$total_time,
      # The torch pipeline fuses the connectivity solve into the training
      # loop rather than timing it as a separate post-hoc step.
      omniscape_time     = NULL
    )

    # --- Posterior / diagnostics / PPC: not available for torch/irl -------
    # ds_posterior() (Laplace approx + MC composition) requires
    # opt_result$bounds, a parametric r_0/z_1..z_K bounds list that the
    # torch/IRL neural-network parameterisation does not produce. Likewise
    # ds_diagnose()/ds_ppc() call predict_intensity() on a
    # fit_intensity_nb()/fit_intensity_gam()-shaped object that doesn't
    # exist here (see intensity_fit_obj note above). Rather than fabricate
    # these or silently change the result's field set, they are explicitly
    # NULL for this solver.
    message("\n[5/6] Posterior inference... SKIPPED (not available for solver = '",
            solver, "')")
    posterior <- NULL

    message("\n[6/6] Diagnostics... SKIPPED (not available for solver = '",
            solver, "')")
    diagnostics <- NULL
    ppc <- NULL

  } else {
    # --- Step 4: Final intensity fit ---
    message("\n[4/6] Fitting final intensity model...")
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

    # --- Step 5: Posterior ---
    posterior <- NULL
    if (n_posterior > 0) {
      message("\n[5/6] Posterior inference...")
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
      message("\n[5/6] Posterior inference... SKIPPED")
    }

    # --- Step 6: Diagnostics ---
    message("\n[6/6] Diagnostics...")
    final_resistance  <- create_resistance_surface(opt_result$best_params,
                                                     basis_stack, link = res_link)
    omni_def <- list(radius = 13L, block_size = 5L)
    omni_cfg <- utils::modifyList(omni_def, omniscape_settings)
    final_omni <- ds_jax_connectivity(final_resistance,
                                       radius     = omni_cfg$radius,
                                       block_size = omni_cfg$block_size)
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
