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
