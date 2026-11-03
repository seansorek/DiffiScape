# ============================================================================
# Package-level hooks and global state
# ============================================================================

.onLoad <- function(libname, pkgname) {

  if (requireNamespace("reticulate", quietly = TRUE)) {
    reticulate::py_require("diffiscape>=0.1.0,<0.2", python_version = ">=3.12")
    .jax_env$package <- reticulate::import("diffiscape", delay_load = TRUE,
                                           convert = FALSE)
  }
}

.onAttach <- function(libname, pkgname) {

  packageStartupMessage(
    "DiffiScape v", utils::packageVersion("DiffiScape"),
    " -- Differentiable Landscape Connectivity Optimization"
  )

}
