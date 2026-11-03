# Deprecated names retained for callers migrating to the installed Python core.

#' Deprecated backend setup
#' @param python Optional Python executable.
#' @param force Reinitialize the installed Python core.
#' @return Invisible TRUE after JAX setup.
#' @export
ds_torch_setup <- function(python = NULL, force = FALSE) {
  message("ds_torch_setup() is deprecated; use ds_jax_setup().")
  ds_jax_setup(python, force)
}

#' Check the deprecated backend
#' @return FALSE; the Torch implementation is retired.
#' @export
ds_torch_check <- function() FALSE

#' Retired direct backend calls
#' @param fn_name Former Python function name.
#' @param ... Former arguments.
#' @return Raises an actionable migration error.
#' @export
ds_torch_call <- function(fn_name, ...) {
  stop("Direct Torch calls are retired. Use run_jax_pipeline(), ds_predict(), or ds_posterior(); refit old .pt checkpoints.", call. = FALSE)
}

#' Install the replacement numerical core
#' @param method Installation method passed to reticulate.
#' @param envname Optional environment name.
#' @param gpu Select a separately configured JAX accelerator environment.
#' @return Invisible TRUE after installation.
#' @export
ds_install_torch_deps <- function(method = "auto", envname = NULL, gpu = FALSE) {
  message("ds_install_torch_deps() is deprecated; use ds_install_jax_deps().")
  if (isTRUE(gpu)) stop("Install JAX for your accelerator separately; the validated installation is CPU-only.", call. = FALSE)
  ds_install_jax_deps(method = method, envname = envname)
}
