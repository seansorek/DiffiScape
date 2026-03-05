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
}
