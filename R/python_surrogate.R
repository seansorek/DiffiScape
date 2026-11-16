.py_surrogate <- function(name, ...) {
  if (!ds_jax_check()) ds_jax_setup()
  module <- reticulate::import("diffiscape.surrogate", convert = FALSE)
  reticulate::py_to_r(do.call(module[[name]], list(...)))
}

.surrogate_seed <- function() as.integer(stats::runif(1, 0, .Machine$integer.max))
