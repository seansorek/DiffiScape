# ============================================================================
# Julia bridge via JuliaConnectoR
# ============================================================================

#' Set up the Julia connection for DiffiScape
#'
#' Initialises a Julia session via \pkg{JuliaConnectoR}, activates the
#' bundled Julia project (which declares Omniscape, Circuitscape, and
#' Enzyme as dependencies), and loads the `DiffiScape` Julia module.
#'
#' @param julia_home Optional path to the Julia binary directory.
#'   If `NULL`, Julia must be discoverable on `PATH`.
#' @param force Re-initialise even if a session already exists.
#' @return `TRUE` (invisibly) on success.
#' @export
ds_julia_setup <- function(julia_home = NULL, force = FALSE) {

  if (.ds_env$julia_initialized && !force) {
    message("Julia session already initialised.")
    return(invisible(TRUE))
  }

  if (!requireNamespace("JuliaConnectoR", quietly = TRUE)) {
    stop("Package 'JuliaConnectoR' is required. ",
         "Install with install.packages('JuliaConnectoR')",
         call. = FALSE)
  }

  # Optionally set JULIA_BINDIR so JuliaConnectoR finds the right binary

  if (!is.null(julia_home)) {
    Sys.setenv(JULIA_BINDIR = julia_home)
  }

  # Locate bundled Julia project
  julia_proj <- .ds_env$julia_project_path
  if (!nzchar(julia_proj) || !dir.exists(julia_proj)) {
    stop("Bundled Julia project not found at: ", julia_proj,
         "\nRe-install DiffiScape or check inst/julia/DiffiScape/",
         call. = FALSE)
  }

  message("Activating Julia project: ", julia_proj)

  tryCatch({
    # Activate project & instantiate deps
    JuliaConnectoR::juliaEval(
      sprintf('import Pkg; Pkg.activate("%s"); Pkg.instantiate()',
              gsub("\\\\", "/", julia_proj))
    )

    # Load the DiffiScape Julia module
    src_path <- file.path(julia_proj, "src", "DiffiScape.jl")
    JuliaConnectoR::juliaEval(
      sprintf('include("%s")', gsub("\\\\", "/", src_path))
    )

    .ds_env$julia_initialized <- TRUE
    message("Julia + DiffiScape module ready.")
    invisible(TRUE)
  }, error = function(e) {
    .ds_env$julia_initialized <- FALSE
    stop("Failed to initialise Julia: ", conditionMessage(e),
         call. = FALSE)
  })
}


#' Check whether the Julia connection is active
#'
#' @return Logical; `TRUE` if `ds_julia_setup()` has been called
#'   successfully.
#' @export
ds_julia_check <- function() {
  .ds_env$julia_initialized
}


#' Call a Julia function from the DiffiScape module
#'
#' Thin wrapper around `JuliaConnectoR::juliaCall()` that ensures the
#' session is initialised first.
#'
#' @param fn_name Character. Fully-qualified Julia function name
#'   (e.g. `"DiffiScape.run_omniscape"`).
#' @param ... Arguments forwarded to the Julia function.
#' @return The value returned by Julia, converted to R by JuliaConnectoR.
#' @export
ds_julia_call <- function(fn_name, ...) {
  if (!.ds_env$julia_initialized) {
    stop("Julia not initialised. Call ds_julia_setup() first.",
         call. = FALSE)
  }
  JuliaConnectoR::juliaCall(fn_name, ...)
}
