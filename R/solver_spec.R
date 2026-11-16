# ============================================================================
# Solver dispatch (S3)
#
# `ds_optimize()`, `ds_fit_intensity()`, and `diffiscape()` all accept a
# `solver` string and used to branch on it with if/else chains.  This file
# centralises that branching into a small internal S3 hierarchy:
#
#   solver_spec(solver)  ->  object of class c("ds_solver_<canonical>", "ds_solver")
#
# Adam and L-BFGS-B fit physical-current models. Surrogate and gradient
# dispatch to the legacy window workflows. "torch" is a translation for
# physical-model settings, and "enzyme" aliases "gradient".
#
# Two internal generics replace the if/else chains:
#   .ds_optimize_dispatch(spec, ...)
#   .ds_fit_intensity_dispatch(spec, ...)
#
# Both are purely internal: not exported, not in NAMESPACE.  Public function
# signatures (ds_optimize(), ds_fit_intensity(), diffiscape()) are unchanged.
# ============================================================================

#' Construct an internal solver-dispatch spec
#'
#' Resolves the solver for [ds_optimize()], [ds_fit_intensity()], and
#' [diffiscape()] through S3 dispatch. The deprecated `enzyme` value resolves
#' to `gradient` and emits a message.
#'
#' @section Config conventions:
#' \describe{
#'   \item{`"adam"` and `"lbfgs"`}{The `config$fit` sublist supplies
#'     physical and model settings for [run_jax_pipeline()].}
#'   \item{`solver = "surrogate"`}{Plain top-level arguments / `config`
#'     entries consumed directly by [optimize_resistance()] (see
#'     [default_optimizer_config()]). No sublist.}
#'   \item{`solver = "gradient"` (or deprecated `"enzyme"`)}{Top-level
#'     `config` entries for the parametric L-BFGS/Adam path, plus optional
#'     `config$model_config` and `config$optim_config` sublists when
#'     `model_type` selects a neural resistance model (forwarded to
#'     [optimize_resistance_gradient()]).}
#'   \item{`"torch"` (deprecated)}{[run_torch_pipeline()] translates the
#'     legacy `config$torch` settings to the physical JAX pipeline.}
#' }
#'
#' @param solver One of `"adam"`, `"lbfgs"`, `"surrogate"`, `"gradient"`,
#'   or the deprecated values `"enzyme"` and `"torch"`.
#' @param choices Character vector of solver values accepted by the caller.
#' @return An object of class `c("ds_solver_<canonical>", "ds_solver")`
#'   with element `solver` (the canonical, post-alias-collapse value).
#' @keywords internal
solver_spec <- function(solver,
                         choices = c("adam", "lbfgs", "surrogate", "gradient", "enzyme",
                                     "torch")) {
  solver <- match.arg(solver, choices)

  # Deprecation alias: "enzyme" collapses into "gradient" here, once, for
  # all three public entry points. Message text is unchanged from the
  # pre-refactor inline checks (locked in by characterization tests).
  if (solver == "enzyme") {
    message("solver='enzyme' is deprecated. Use solver='gradient' instead.")
    solver <- "gradient"
  }

  structure(
    list(solver = solver),
    class = c(paste0("ds_solver_", solver), "ds_solver")
  )
}


#' Internal dispatch generic for [ds_optimize()]
#'
#' @param spec A `solver_spec()` object.
#' @param ... Named arguments forwarded from [ds_optimize()]'s formal
#'   argument list (each explicitly named at the call site, so a missing
#'   argument fails loudly rather than silently mispositioning).
#' @return Same return shape as [ds_optimize()].
#' @keywords internal
.ds_optimize_dispatch <- function(spec, ...) {
  UseMethod(".ds_optimize_dispatch")
}


#' Internal dispatch generic for [ds_fit_intensity()]
#'
#' @param spec A `solver_spec()` object.
#' @param ... Named arguments forwarded from [ds_fit_intensity()].
#' @return Same return shape as [ds_fit_intensity()].
#' @keywords internal
.ds_fit_intensity_dispatch <- function(spec, ...) {
  UseMethod(".ds_fit_intensity_dispatch")
}
