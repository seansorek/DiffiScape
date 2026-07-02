# ============================================================================
# Solver dispatch (S3)
#
# `ds_optimize()`, `ds_fit_intensity()`, and `diffiscape()` all accept a
# `solver` string and used to branch on it with if/else chains.  This file
# centralises that branching into a small internal S3 hierarchy:
#
#   solver_spec(solver)  ->  object of class c("ds_solver_<canonical>", "ds_solver")
#
# classes: ds_solver_surrogate, ds_solver_gradient, ds_solver_torch,
# ds_solver_irl.  "enzyme" is a deprecated alias for "gradient" and is
# collapsed into ds_solver_gradient at construction time -- it never gets
# its own class, matching the pre-refactor behavior exactly (including the
# deprecation message text, which is intentionally unchanged).
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
#' Validates a `solver` string, fires the (unchanged) `"enzyme"` deprecation
#' message when applicable, and returns a small S3 object used to dispatch
#' [ds_optimize()] / [ds_fit_intensity()] / [diffiscape()] onto their
#' per-solver implementations.  This replaces the if/else chains that used
#' to live inline in those three functions.
#'
#' @section Config conventions:
#' The three backend families keep their pre-existing, *not unified*,
#' config conventions -- each `.ds_optimize_dispatch()` / S3 method is the
#' one place that knows how to pull its own sublist out of `config`:
#' \describe{
#'   \item{`solver = "surrogate"`}{Plain top-level arguments / `config`
#'     entries consumed directly by [optimize_resistance()] (see
#'     [default_optimizer_config()]). No sublist.}
#'   \item{`solver = "gradient"` (or deprecated `"enzyme"`)}{Top-level
#'     `config` entries for the parametric L-BFGS/Adam path, plus optional
#'     `config$model_config` and `config$optim_config` sublists when
#'     `model_type` selects a neural resistance model (forwarded to
#'     [optimize_resistance_gradient()]).}
#'   \item{`solver = "torch"` / `"irl"`}{A `config$torch` sublist whose
#'     entries are forwarded as named arguments to [run_torch_pipeline()].
#'     `"irl"` is `"torch"` with `model_type` forced to `"irl"`.}
#' }
#'
#' @param solver Character scalar; one of `"surrogate"`, `"gradient"`,
#'   `"enzyme"` (deprecated alias for `"gradient"`), `"torch"`, `"irl"`.
#' @param choices Character vector of valid `solver` values for the calling
#'   function (passed through to `match.arg()`'s `choices`, so callers with
#'   a narrower accepted set -- e.g. [ds_fit_intensity()], which does not
#'   support `"torch"`/`"irl"` as a *fitting* solver -- get the right error
#'   message). Defaults to the full 5-value set.
#' @return An object of class `c("ds_solver_<canonical>", "ds_solver")`
#'   with element `solver` (the canonical, post-alias-collapse value).
#' @keywords internal
solver_spec <- function(solver,
                         choices = c("surrogate", "gradient", "enzyme",
                                     "torch", "irl")) {
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
