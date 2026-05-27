# ============================================================================
# PyTorch bridge via reticulate
#
# Loads the bundled diff_cs/05_torch_pipeline.py module and exposes a thin
# call interface analogous to the JuliaConnectoR bridge in julia_bridge.R.
# ============================================================================

#' Initialise the PyTorch backend for DiffiScape
#'
#' Loads the bundled `diff_cs/05_torch_pipeline.py` module via \pkg{reticulate},
#' verifying that the required Python packages (`torch`, `numpy`, `scipy`,
#' `pyamg`) are available. Call once per R session before any of the
#' torch-based optimisers or samplers.
#'
#' @param python Optional path to a Python binary; passed to
#'   [reticulate::use_python()] before module import.
#' @param force Re-initialise even if the module is already loaded.
#' @return `TRUE` (invisibly) on success.
#' @export
ds_torch_setup <- function(python = NULL, force = FALSE) {

  if (.ds_env$torch_initialized && !force) {
    message("PyTorch backend already initialised.")
    return(invisible(TRUE))
  }

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required for the torch backend. ",
         "Install with install.packages('reticulate').",
         call. = FALSE)
  }

  if (!is.null(python)) {
    reticulate::use_python(python, required = TRUE)
  }

  required <- c("torch", "numpy", "scipy", "pyamg")
  missing  <- required[!vapply(required, reticulate::py_module_available,
                               logical(1))]
  if (length(missing) > 0) {
    stop("Missing Python packages: ", paste(missing, collapse = ", "),
         "\nInstall them with ds_install_torch_deps() or ",
         "reticulate::py_install(c(\"", paste(missing, collapse = "\", \""), "\")).",
         call. = FALSE)
  }

  module_dir <- system.file("python", "diff_cs", package = "DiffiScape")
  if (!nzchar(module_dir) || !dir.exists(module_dir)) {
    stop("Bundled Python module not found in inst/python/diff_cs/. ",
         "Re-install DiffiScape.",
         call. = FALSE)
  }

  tryCatch({
    py_mod <- reticulate::import_from_path("05_torch_pipeline",
                                            path = module_dir)
    .ds_env$torch_module      <- py_mod
    .ds_env$torch_initialized <- TRUE

    torch <- reticulate::import("torch")
    cuda_status <- if (isTRUE(torch$cuda$is_available())) {
      gpu_name <- tryCatch(torch$cuda$get_device_name(0L),
                           error = function(e) "unknown")
      sprintf("CUDA available (%s)", gpu_name)
    } else {
      "CPU only"
    }
    cupy_avail <- reticulate::py_module_available("cupy")
    message(sprintf("  PyTorch %s | %s | cupy: %s",
                    torch$`__version__`, cuda_status,
                    if (cupy_avail) "yes"
                    else "no (install cupy-cuda12x for GPU circuit solver)"))
    message("  Module: ", file.path(module_dir, "05_torch_pipeline.py"))

    invisible(TRUE)
  }, error = function(e) {
    .ds_env$torch_initialized <- FALSE
    .ds_env$torch_module      <- NULL
    stop("Failed to initialise PyTorch backend: ", conditionMessage(e),
         call. = FALSE)
  })
}


#' Check whether the PyTorch backend is loaded
#'
#' @return Logical; `TRUE` if [ds_torch_setup()] has been called
#'   successfully.
#' @export
ds_torch_check <- function() {
  .ds_env$torch_initialized
}


#' Call a function from the bundled PyTorch pipeline module
#'
#' Thin wrapper that ensures the backend is initialised, then resolves the
#' named function on the cached Python module handle and forwards
#' arguments to it via reticulate.
#'
#' @param fn_name Character; bare function name in `05_torch_pipeline`
#'   (e.g. `"run_torch_optimization"`).
#' @param ... Arguments forwarded to the Python function.
#' @return The value returned by Python (use [reticulate::py_to_r()] if a
#'   manual conversion is needed).
#' @export
ds_torch_call <- function(fn_name, ...) {
  if (!.ds_env$torch_initialized) {
    stop("PyTorch backend not initialised. Call ds_torch_setup() first.",
         call. = FALSE)
  }
  fn <- tryCatch(.ds_env$torch_module[[fn_name]],
                 error = function(e) NULL)
  if (is.null(fn)) {
    stop("Function not found in 05_torch_pipeline: ", fn_name,
         call. = FALSE)
  }
  fn(...)
}


#' Install Python dependencies for the torch backend
#'
#' Convenience wrapper around [reticulate::py_install()] that installs the
#' packages required by `05_torch_pipeline.py`. Optionally installs
#' `cupy-cuda12x` for GPU-accelerated circuit solves.
#'
#' @param method Installation method passed to [reticulate::py_install()]
#'   (`"auto"`, `"virtualenv"`, or `"conda"`).
#' @param envname Optional virtualenv/conda environment name.
#' @param gpu If `TRUE`, also install `cupy-cuda12x` for the GPU circuit
#'   solver. Requires a CUDA 12 capable GPU and matching CUDA toolkit.
#' @return Invisible `TRUE`.
#' @export
ds_install_torch_deps <- function(method  = "auto",
                                   envname = NULL,
                                   gpu     = FALSE) {

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required. ",
         "Install with install.packages('reticulate').",
         call. = FALSE)
  }

  pkgs <- c("torch", "numpy", "scipy", "pyamg")
  if (isTRUE(gpu)) pkgs <- c(pkgs, "cupy-cuda12x")

  message("Installing Python packages: ", paste(pkgs, collapse = ", "))
  reticulate::py_install(pkgs, method = method, envname = envname)
  invisible(TRUE)
}
