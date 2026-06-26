# ============================================================================
# Package-level hooks and global state
# ============================================================================

#' @keywords internal
.ds_env <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {

  .ds_env$torch_initialized <- FALSE
  .ds_env$torch_module      <- NULL

  .ds_env$jax_initialized <- FALSE
  .ds_env$jax_core        <- NULL
  .ds_env$jax_window      <- NULL
}

.onAttach <- function(libname, pkgname) {

  packageStartupMessage(
    "DiffiScape v", utils::packageVersion("DiffiScape"),
    " -- Differentiable Landscape Connectivity Optimization"
  )

  reticulate_ok <- requireNamespace("reticulate", quietly = TRUE)
  if (!reticulate_ok) {
    packageStartupMessage(
      "Note: reticulate not installed. ",
      "Install with install.packages('reticulate') ",
      "to enable the JAX and PyTorch compute backends."
    )
  }
}
