# ============================================================================
# Utility functions shared across the package
# ============================================================================

# Null-coalescing operator (internal)
`%||%` <- function(a, b) if (is.null(a)) b else a

#' Convert coordinates to a two-column matrix
#'
#' Accepts a data.frame (with `x` and `y` columns), matrix, or
#' [terra::SpatVector] and returns the coordinates as a two-column matrix.
#'
#' @param coords Coordinates as data.frame, matrix, or SpatVector.
#' @return A numeric matrix with columns `x` and `y`.
#' @export
as_coord_matrix <- function(coords) {

  if (is.null(coords)) {
    stop("Coordinates cannot be NULL", call. = FALSE)
  }

  if (inherits(coords, "SpatVector")) {
    return(terra::crds(coords))
  }


  if (is.data.frame(coords)) {
    if ("x" %in% names(coords) && "y" %in% names(coords)) {
      return(as.matrix(coords[, c("x", "y")]))
    }
    if (ncol(coords) >= 2) {
      return(as.matrix(coords[, 1:2]))
    }
    stop("Coordinate data.frame must have 'x','y' columns or >= 2 columns",
         call. = FALSE)
  }

  if (is.matrix(coords)) {
    if (ncol(coords) < 2) {
      stop("Coordinate matrix must have >= 2 columns", call. = FALSE)
    }
    return(coords[, 1:2, drop = FALSE])
  }

  stop("Coordinates must be a data.frame, matrix, or SpatVector",
       call. = FALSE)
}


#' Extract raster values at point locations
#'
#' Robust extraction that handles data.frames, matrices, and SpatVectors.
#'
#' @param raster A [terra::SpatRaster].
#' @param points Locations as data.frame (with x, y), matrix, or SpatVector.
#' @return Numeric vector of extracted values.
#' @export
extract_raster_values <- function(raster, points) {

  if (is.null(raster) || is.null(points)) return(NULL)

  if (is.data.frame(points) || is.matrix(points)) {
    mat <- as_coord_matrix(points)
    pts_df <- data.frame(x = mat[, 1], y = mat[, 2])
    pts_vect <- terra::vect(pts_df, geom = c("x", "y"),
                            crs = terra::crs(raster))
  } else if (inherits(points, "SpatVector")) {
    pts_vect <- points
  } else {
    stop("points must be a data.frame, matrix, or SpatVector", call. = FALSE)
  }

  terra::extract(raster, pts_vect)[, 2]
}


#' Compute AIC and BIC
#' 
#' @param loglik Log-likelihood value.
#' @param k Number of estimated parameters.
#' @param n Number of observations.
#' @return A named list with `AIC`, `BIC`, and `AICc`.
#' @export
compute_information_criteria <- function(loglik, k, n) {
  aic <- -2 * loglik + 2 * k
  bic <- -2 * loglik + log(n) * k
  aicc <- aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
  list(AIC = aic, BIC = bic, AICc = aicc)
}


#' Validate that rasters share the same geometry
#'
#' Checks CRS, extent, and resolution. Returns TRUE silently on success;
#' stops with an informative message on failure.
#'
#' @param ... One or more [terra::SpatRaster] objects.
#' @return `TRUE` if aligned, `FALSE` otherwise.
#' @keywords internal
validate_raster_alignment <- function(...) {
  rasters <- list(...)
  rasters <- rasters[!vapply(rasters, is.null, logical(1))]
  if (length(rasters) < 2) return(TRUE)

  ref <- rasters[[1]]
  for (i in seq_along(rasters)[-1]) {
    if (!terra::compareGeom(ref, rasters[[i]], stopOnError = FALSE)) {
      return(FALSE)
    }
  }
  TRUE
}


#' Prepare inputs for JAX/PyTorch-backed sampling and optimisation
#'
#' Converts a SpatRaster basis stack and observation coordinates into numpy
#' arrays suitable for the bundled Python modules used by both the JAX
#' bridge ([ds_jax_sample_nuts()], [ds_jax_sample_advi()]) and the PyTorch
#' pipeline ([run_torch_pipeline()], [run_bayesian_sampling()],
#' [run_bayesian_sampling_hmc()], [run_advi()]).  Shared by every entry
#' point that talks to either backend, so both treat data preparation
#' identically.
#'
#' @param basis_stack A [terra::SpatRaster] with K covariate layers.
#' @param obs_points Data.frame with `x, y` columns (projected coords).
#' @return A named list with:
#'   \describe{
#'     \item{basis_np}{numpy array, shape (n_valid, K).}
#'     \item{obs_np}{numpy array of observation counts per valid cell.}
#'     \item{vmask_np}{numpy boolean mask, shape (n_rows * n_cols,).}
#'     \item{valid_mask}{Logical vector (R side).}
#'     \item{n_rows, n_cols}{Integer grid dimensions.}
#'     \item{cell_area}{Numeric cell area in map units squared.}
#'     \item{n_valid}{Number of valid (non-NA) cells.}
#'     \item{n_obs}{Total observation count.}
#'   }
#' @keywords internal
.prepare_backend_inputs <- function(basis_stack, obs_points) {

  np <- reticulate::import("numpy", convert = FALSE)

  n_rows    <- terra::nrow(basis_stack)
  n_cols    <- terra::ncol(basis_stack)
  cell_area <- prod(terra::res(basis_stack))

  basis_matrix <- as.matrix(basis_stack)
  valid_mask   <- stats::complete.cases(basis_matrix)
  basis_values <- basis_matrix[valid_mask, , drop = FALSE]

  cell_indices <- terra::cellFromXY(
    basis_stack,
    cbind(obs_points$x, obs_points$y)
  )
  valid_obs <- !is.na(cell_indices) & valid_mask[cell_indices]
  if (any(!valid_obs)) {
    message(sprintf("    Dropped %d obs outside valid cells",
                    sum(!valid_obs)))
  }
  cell_indices <- cell_indices[valid_obs]

  obs_table        <- table(cell_indices)
  obs_counts_full  <- rep(0L, terra::ncell(basis_stack))
  obs_counts_full[as.integer(names(obs_table))] <- as.integer(obs_table)
  obs_counts_valid <- obs_counts_full[valid_mask]

  list(
    basis_np   = np$array(basis_values, dtype = np$float64),
    obs_np     = np$array(as.double(obs_counts_valid), dtype = np$float64),
    vmask_np   = np$array(valid_mask, dtype = np$bool_),
    valid_mask = valid_mask,
    n_rows     = n_rows,
    n_cols     = n_cols,
    cell_area  = cell_area,
    n_valid    = sum(valid_mask),
    n_obs      = sum(obs_counts_valid)
  )
}
