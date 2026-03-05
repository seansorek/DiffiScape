# ============================================================================
# Basis functions -- general-purpose environmental raster preparation
# ============================================================================

#' Create a basis-function stack from environmental rasters
#'
#' Accepts an arbitrary named list of [terra::SpatRaster] layers, validates
#' alignment, optionally rescales each to \[0, 1\], and returns a multi-layer
#' `SpatRaster` ready for [create_resistance_surface()].
#'
#' @param rasters A **named** list of single-layer `SpatRaster` objects.
#'   Names become layer names in the output stack (e.g.
#'   `list(canopy = rast_canopy, impervious = rast_imp)`).
#' @param rescale Logical; if `TRUE` (default), each layer is linearly
#'   rescaled so that its valid (non-NA) values span \[0, 1\].
#' @param mask_to Character name of the layer whose NA pattern should be
#'   applied to **all** layers (unified study-area mask).
#'   `NULL` (default) skips masking.
#' @param cache_path Optional file path.
#'   If the file exists it is loaded and returned immediately;
#'   otherwise the new stack is written there for future re-use.
#' @return A [terra::SpatRaster] with one layer per input raster.
#' @details
#' Any input raster whose geometry (CRS, extent, resolution) does not match
#' the **first** raster is automatically resampled to it (bilinear for
#' continuous layers, nearest-neighbour for layers with only 0/1 values).
#' @export
#' @examples
#' \dontrun{
#'   canopy <- terra::rast("canopy.tif")
#'   imperv <- terra::rast("impervious.tif")
#'   basis  <- create_basis_stack(list(canopy = canopy, impervious = imperv))
#' }
create_basis_stack <- function(rasters,
                               rescale  = TRUE,
                               mask_to  = NULL,
                               cache_path = NULL) {

 # ---------- cache shortcut ------------------------------------------------
 if (!is.null(cache_path) && file.exists(cache_path)) {
    message("Loading cached basis stack from: ", cache_path)
    return(terra::rast(cache_path))
  }

  # ---------- validate inputs -----------------------------------------------
  if (!is.list(rasters) || length(rasters) == 0) {
    stop("`rasters` must be a non-empty named list of SpatRaster objects.",
         call. = FALSE)
  }
  if (is.null(names(rasters)) || any(names(rasters) == "")) {
    stop("Every element of `rasters` must be named.", call. = FALSE)
  }

  for (nm in names(rasters)) {
    if (!inherits(rasters[[nm]], "SpatRaster")) {
      stop(sprintf("Element '%s' is not a SpatRaster.", nm), call. = FALSE)
    }
    if (terra::nlyr(rasters[[nm]]) != 1) {
      stop(sprintf("Element '%s' must have exactly 1 layer (has %d).",
                   nm, terra::nlyr(rasters[[nm]])), call. = FALSE)
    }
  }

  # ---------- align to first raster -----------------------------------------
  ref <- rasters[[1]]
  for (nm in names(rasters)[-1]) {
    if (!terra::compareGeom(ref, rasters[[nm]], stopOnError = FALSE)) {
      vals <- terra::values(rasters[[nm]])
      is_binary <- all(vals[!is.na(vals)] %in% c(0, 1))
      method <- if (is_binary) "near" else "bilinear"
      message(sprintf("  Resampling '%s' to reference grid (method: %s)",
                      nm, method))
      rasters[[nm]] <- terra::resample(rasters[[nm]], ref, method = method)
    }
  }

  # ---------- rescale to [0, 1] ---------------------------------------------
  if (rescale) {
    for (nm in names(rasters)) {
      v <- terra::values(rasters[[nm]])
      vmin <- min(v, na.rm = TRUE)
      vmax <- max(v, na.rm = TRUE)
      rng  <- vmax - vmin
      if (rng > 0) {
        rasters[[nm]] <- (rasters[[nm]] - vmin) / rng
      } else {
        # constant layer (e.g. all-zero fence) -- leave as-is
        message(sprintf("  '%s' has zero range; skipping rescale.", nm))
      }
    }
  }

  # ---------- stack layers --------------------------------------------------
  stk <- terra::rast(rasters)
  names(stk) <- names(rasters)

  # ---------- unified mask --------------------------------------------------
  if (!is.null(mask_to)) {
    if (!(mask_to %in% names(stk))) {
      stop(sprintf("mask_to = '%s' not found in layer names.", mask_to),
           call. = FALSE)
    }
    stk <- terra::mask(stk, stk[[mask_to]])
  }

  # ---------- cache ---------------------------------------------------------
  if (!is.null(cache_path)) {
    cache_dir <- dirname(cache_path)
    if (!dir.exists(cache_dir)) dir.create(cache_dir, recursive = TRUE)
    terra::writeRaster(stk, cache_path, overwrite = TRUE)
    message("Cached basis stack to: ", cache_path)
  }

  stk
}


#' Validate a basis-function stack
#'
#' Checks that a `SpatRaster` looks like a valid basis stack (>= 1 layer,
#' named, no completely empty layers).
#'
#' @param basis_stack A [terra::SpatRaster].
#' @return `TRUE` (invisibly) on success; stops on failure.
#' @export
validate_basis_stack <- function(basis_stack) {

  if (!inherits(basis_stack, "SpatRaster")) {
    stop("`basis_stack` must be a SpatRaster.", call. = FALSE)
  }
  n <- terra::nlyr(basis_stack)
  if (n < 1) {
    stop("`basis_stack` must have >= 1 layer.", call. = FALSE)
  }
  nms <- names(basis_stack)
  if (is.null(nms) || any(nms == "")) {
    stop("All layers in `basis_stack` must be named.", call. = FALSE)
  }
  for (i in seq_len(n)) {
    v <- terra::values(basis_stack[[i]])
    if (all(is.na(v))) {
      stop(sprintf("Layer '%s' is entirely NA.", nms[i]), call. = FALSE)
    }
  }
  invisible(TRUE)
}


#' Check pairwise correlations among basis functions
#'
#' Computes Pearson correlation between all pairs of layers and warns when
#' any exceed `warn_threshold`.
#'
#' @param basis_stack A [terra::SpatRaster].
#' @param warn_threshold Absolute correlation above which a warning is
#'   emitted (default 0.7).
#' @return The correlation matrix (invisibly).
#' @export
check_basis_correlations <- function(basis_stack, warn_threshold = 0.7) {

  validate_basis_stack(basis_stack)
  n <- terra::nlyr(basis_stack)
  if (n < 2) return(invisible(NULL))

  vals <- terra::values(basis_stack)
  complete <- stats::complete.cases(vals)
  if (sum(complete) < 10) {
    warning("Fewer than 10 complete cells; skipping correlation check.")
    return(invisible(NULL))
  }

  vals <- vals[complete, , drop = FALSE]

  # Drop zero-variance layers
  vars <- apply(vals, 2, stats::var, na.rm = TRUE)
  keep <- vars > 0
  if (sum(keep) < 2) return(invisible(NULL))
  vals <- vals[, keep, drop = FALSE]

  cormat <- stats::cor(vals, use = "complete.obs")

  # Warn about high correlations
  upper <- which(abs(cormat) > warn_threshold & upper.tri(cormat),
                 arr.ind = TRUE)
  if (nrow(upper) > 0) {
    msg <- vapply(seq_len(nrow(upper)), function(k) {
      sprintf("  %s -- %s: r = %.3f",
              colnames(cormat)[upper[k, 1]],
              colnames(cormat)[upper[k, 2]],
              cormat[upper[k, 1], upper[k, 2]])
    }, character(1))
    warning("High correlations (|r| > ", warn_threshold, "):\n",
            paste(msg, collapse = "\n"), call. = FALSE)
  }

  invisible(cormat)
}


#' Get summary statistics for each basis layer
#'
#' @param basis_stack A [terra::SpatRaster].
#' @return A data.frame with one row per layer.
#' @export
basis_summary <- function(basis_stack) {
  validate_basis_stack(basis_stack)
  nms <- names(basis_stack)
  do.call(rbind, lapply(nms, function(nm) {
    v <- terra::values(basis_stack[[nm]])
    ok <- !is.na(v)
    data.frame(
      layer   = nm,
      mean    = mean(v[ok]),
      sd      = stats::sd(v[ok]),
      min     = min(v[ok]),
      max     = max(v[ok]),
      n_valid = sum(ok),
      n_na    = sum(!ok),
      stringsAsFactors = FALSE
    )
  }))
}
