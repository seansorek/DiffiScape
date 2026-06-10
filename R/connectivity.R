# ============================================================================
# Connectivity computation via Julia (Omniscape / Circuitscape)
# ============================================================================

#' Run Omniscape to compute cumulative current flow
#'
#' Writes the resistance raster to a temporary file, calls the Julia
#' `DiffiScape.run_omniscape()` function via JuliaConnectoR, reads back
#' the cumulative-current-flow raster, and (optionally) cleans up.
#'
#' @param resistance A single-layer [terra::SpatRaster] of resistance
#'   values (from [create_resistance_surface()]).
#' @param radius Integer; Omniscape search radius in pixels (default 13).
#' @param block_size Integer; moving-window block size (default 5).
#' @param source_from_resistance Logical; derive source strength from
#'   resistance (default `TRUE`).
#' @param output_dir Optional directory for Omniscape output.
#'   If `NULL`, a temp directory is used and cleaned up afterwards.
#' @param cleanup Logical; remove temp files after reading results
#'   (default `TRUE`; only applies when `output_dir` is `NULL`).
#' @return A list with:
#'   \describe{
#'     \item{cum_current}{[terra::SpatRaster] of cumulative current flow.}
#'     \item{flow_potential}{Flow-potential raster (if available).}
#'     \item{elapsed_seconds}{Wall-clock time.}
#'   }
#' @export
run_omniscape <- function(resistance,
                          radius     = 13L,
                          block_size = 5L,
                          source_from_resistance = TRUE,
                          output_dir = NULL,
                          cleanup    = TRUE) {

  if (!ds_julia_check()) {
    stop("Julia not initialised. Call ds_julia_setup() first.",
         call. = FALSE)
  }

  using_temp <- is.null(output_dir)
  if (using_temp) {
    output_dir <- tempfile("omniscape_")
    dir.create(output_dir, recursive = TRUE)
  }

  # Write resistance raster to file
  res_path <- file.path(output_dir, "resistance.tif")
  terra::writeRaster(resistance, res_path, overwrite = TRUE,
                     datatype = "FLT4S")

  res_path_jl <- gsub("\\\\", "/", normalizePath(res_path, winslash = "/"))
  out_dir_jl  <- gsub("\\\\", "/", normalizePath(output_dir, winslash = "/"))

  start <- Sys.time()

  ds_julia_call(
    "DiffiScapeMod.run_omniscape",
    res_path_jl,
    out_dir_jl,
    as.integer(radius),
    as.integer(block_size),
    source_from_resistance
  )

  elapsed <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  message(sprintf("Omniscape completed in %.1f s", elapsed))

  # Read results
  run_dir <- file.path(output_dir, "omniscape_run")
  cum_path  <- file.path(run_dir, "cum_currmap.tif")
  flow_path <- file.path(run_dir, "flow_potential.tif")

  if (!file.exists(cum_path)) {
    stop("Omniscape output not found: ", cum_path, call. = FALSE)
  }

  cum_current <- terra::rast(cum_path) * 1   # force into memory
  flow_potential <- NULL
  if (file.exists(flow_path)) {
    flow_potential <- terra::rast(flow_path) * 1
  }

  if (cleanup && using_temp) {
    unlink(output_dir, recursive = TRUE)
  }

  list(
    cum_current    = cum_current,
    flow_potential = flow_potential,
    elapsed_seconds = elapsed
  )
}


#' Run Circuitscape for pairwise / one-to-all connectivity
#'
#' Calls the Julia `DiffiScape.run_circuitscape()` function.
#'
#' @param resistance A single-layer [terra::SpatRaster] of resistance.
#' @param focal_points A data.frame with `x, y` columns or a
#'   [terra::SpatVector] of focal locations.
#' @param mode Character; `"one-to-all"` (default) or `"pairwise"`.
#' @param output_dir Optional output directory (temp if `NULL`).
#' @param cleanup Logical; clean temp files (default `TRUE`).
#' @return A list with:
#'   \describe{
#'     \item{current_map}{[terra::SpatRaster] of current flow.}
#'     \item{resistance_distances}{Matrix of pairwise resistances
#'       (only for `mode = "pairwise"`).}
#'     \item{elapsed_seconds}{Wall-clock time.}
#'   }
#' @export
run_circuitscape <- function(resistance,
                             focal_points,
                             mode       = c("one-to-all", "pairwise"),
                             output_dir = NULL,
                             cleanup    = TRUE) {

  mode <- match.arg(mode)

  if (!ds_julia_check()) {
    stop("Julia not initialised. Call ds_julia_setup() first.",
         call. = FALSE)
  }

  using_temp <- is.null(output_dir)
  if (using_temp) {
    output_dir <- tempfile("circuitscape_")
    dir.create(output_dir, recursive = TRUE)
  }

  # Write resistance & focal nodes
  res_path <- file.path(output_dir, "resistance.tif")
  terra::writeRaster(resistance, res_path, overwrite = TRUE,
                     datatype = "FLT4S")

  coords <- as_coord_matrix(focal_points)
  focal_path <- file.path(output_dir, "focal_nodes.csv")
  utils::write.csv(data.frame(x = coords[, 1], y = coords[, 2]),
                    focal_path, row.names = FALSE)

  res_path_jl   <- gsub("\\\\", "/", normalizePath(res_path))
  focal_path_jl <- gsub("\\\\", "/", normalizePath(focal_path))
  out_dir_jl    <- gsub("\\\\", "/", normalizePath(output_dir))

  start <- Sys.time()

  ds_julia_call(
    "DiffiScapeMod.run_circuitscape",
    res_path_jl,
    focal_path_jl,
    out_dir_jl,
    mode
  )

  elapsed <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  message(sprintf("Circuitscape (%s) completed in %.1f s", mode, elapsed))

  # Read results
  cur_path <- file.path(output_dir, "curmap.tif")
  current_map <- NULL
  if (file.exists(cur_path)) {
    current_map <- terra::rast(cur_path) * 1
  }

  res_dist <- NULL
  rdist_path <- file.path(output_dir, "resistances.csv")
  if (file.exists(rdist_path)) {
    res_dist <- as.matrix(utils::read.csv(rdist_path, row.names = 1))
  }

  if (cleanup && using_temp) {
    unlink(output_dir, recursive = TRUE)
  }

  list(
    current_map          = current_map,
    resistance_distances = res_dist,
    elapsed_seconds      = elapsed
  )
}


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

#' Compute cumulative current (and optionally voltage) using the differentiable
#' Julia solver
#'
#' Replaces [run_omniscape()] with a pure-Julia, in-memory solver that
#' requires no file I/O.  The solver uses conjugate gradient on a
#' matrix-free grid Laplacian with threaded window parallelism.
#'
#' @param resistance A single-layer [terra::SpatRaster] of resistance.
#' @param radius Integer; moving-window radius in pixels (default 13).
#' @param block_size Integer; source-block side length (default 5).
#' @param output Character; which quantity to return from the circuit solve.
#'   One of `"current"` (default), `"voltage"`, or `"both"`.
#'   - `"current"` returns only cumulative current density.
#'   - `"voltage"` returns only cumulative voltage (flow potential).
#'   - `"both"` returns both.
#' @return A list matching the [run_omniscape()] interface:
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
                                   output     = "current") {

  output <- match.arg(output, c("current", "voltage", "both"))

  if (!ds_julia_check()) {
    stop("Julia not initialised. Call ds_julia_setup() first.",
         call. = FALSE)
  }

  nrow_grid <- terra::nrow(resistance)
  ncol_grid <- terra::ncol(resistance)

  # terra values → R matrix (row-major → column-major)
  R_vec <- as.numeric(terra::values(resistance))
  R_mat <- matrix(R_vec, nrow = nrow_grid, ncol = ncol_grid, byrow = TRUE)
  R_mat[is.na(R_mat)] <- 0  # nodata → zero resistance → zero conductance

  start <- Sys.time()

  julia_result <- ds_julia_call("DiffiScapeMod.cumulative_current",
                                R_mat,
                                as.integer(radius),
                                as.integer(block_size),
                                output)

  elapsed <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  message(sprintf("Differentiable solver completed in %.1f s", elapsed))

  # Helper: Julia matrix → terra raster (column-major → row-major)
  mat_to_rast <- function(mat, layer_name) {
    r <- terra::rast(resistance)
    terra::values(r) <- as.vector(t(mat))
    names(r) <- layer_name
    r
  }

  if (output == "both") {
    # JuliaConnectoR converts Julia named tuple (; current, voltage) → R list
    cum_rast  <- mat_to_rast(julia_result$current, "cum_current")
    volt_rast <- mat_to_rast(julia_result$voltage,  "flow_potential")
  } else if (output == "voltage") {
    cum_rast  <- NULL
    volt_rast <- mat_to_rast(julia_result, "flow_potential")
  } else {
    cum_rast  <- mat_to_rast(julia_result, "cum_current")
    volt_rast <- NULL
  }

  list(
    cum_current     = cum_rast,
    flow_potential  = volt_rast,
    elapsed_seconds = elapsed
  )
}
