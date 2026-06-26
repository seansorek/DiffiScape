# ============================================================================
# Connectivity computation via JAX differentiable solver
# ============================================================================


#' Extract connectivity values at observation locations
#'
#' @param connectivity A [terra::SpatRaster] of connectivity / current flow.
#' @param points Locations as data.frame (with x, y), matrix, or SpatVector.
#' @param buffer Optional extraction buffer radius (uses mean within buffer).
#' @return Numeric vector of connectivity values.
#' @export
extract_connectivity <- function(connectivity, points, buffer = NULL) {

  mat <- as_coord_matrix(points)
  pts_df <- data.frame(x = mat[, 1], y = mat[, 2])
  pts_vect <- terra::vect(pts_df, geom = c("x", "y"),
                          crs = terra::crs(connectivity))

  if (is.null(buffer)) {
    terra::extract(connectivity, pts_vect)[, 2]
  } else {
    terra::extract(connectivity, pts_vect, buffer = buffer, fun = mean)[, 2]
  }
}


# ===================== In-Memory Differentiable Solver ======================

#' Compute cumulative current (and optionally voltage) using the JAX
#' differentiable solver
#'
#' In-memory solver that requires no file I/O.  Delegates to
#' [ds_jax_connectivity()] which uses JAXScape's conjugate-gradient
#' circuit solver.
#'
#' @param resistance A single-layer [terra::SpatRaster] of resistance.
#' @param radius Integer; moving-window radius in pixels (default 13).
#' @param block_size Integer; source-block side length (default 5).
#' @param output Character; which quantity to return from the circuit solve.
#'   One of `"current"` (default), `"voltage"`, or `"both"`.
#'   - `"current"` returns only cumulative current density.
#'   - `"voltage"` returns only cumulative voltage (flow potential).
#'   - `"both"` returns both.
#' @param parameterization Character; `"resistance"` (default) or
#'   `"permeability"`.
#' @return A list with:
#'   \describe{
#'     \item{cum_current}{[terra::SpatRaster] of cumulative current, or `NULL`
#'       when `output = "voltage"`.}
#'     \item{flow_potential}{[terra::SpatRaster] of cumulative voltage
#'       (flow potential), or `NULL` when `output = "current"`.}
#'     \item{elapsed_seconds}{Wall-clock time.}
#'   }
#' @export
run_cumulative_current <- function(resistance,
                                   radius     = 13L,
                                   block_size = 5L,
                                   output     = "current",
                                   parameterization = "resistance") {

  output <- match.arg(output, c("current", "voltage", "both"))

  ds_jax_connectivity(
    resistance,
    radius           = radius,
    block_size       = block_size,
    parameterization = parameterization,
    output           = output
  )
}
