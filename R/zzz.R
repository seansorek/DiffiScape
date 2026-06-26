# ============================================================================
# Package-level hooks and global state
# ============================================================================

#' @keywords internal
.ds_env <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {

  .ds_env$julia_initialized <- FALSE
  .ds_env$julia_con <- NULL
  .ds_env$julia_project_path <- system.file("julia", "DiffiScape",
                                             package = "DiffiScape")

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

  julia_ok <- requireNamespace("JuliaConnectoR", quietly = TRUE)
  if (!julia_ok) {
    packageStartupMessage(
      "Note: JuliaConnectoR not installed. ",
      "Install with install.packages('JuliaConnectoR') ",
      "and ensure Julia is on your PATH."
    )
  }

  reticulate_ok <- requireNamespace("reticulate", quietly = TRUE)
  if (!reticulate_ok) {
    packageStartupMessage(
      "Note: reticulate not installed. ",
      "Install with install.packages('reticulate') ",
      "to enable the PyTorch backend (run_torch_pipeline, run_bayesian_sampling, ...)."
    )
  }
}
