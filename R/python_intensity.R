# Built-in numerical families live in Python; user callbacks remain R-only.
.python_intensity_family <- function(name, ...) {
  config <- c(list(family = name), list(...))
  module <- function() {
    if (!ds_jax_check()) ds_jax_setup()
    reticulate::import("diffiscape.likelihoods", convert = FALSE)
  }
  nll <- function(theta, z_obs, z_int, int_weights, obs_weights,
                  cov_obs = NULL, cov_int = NULL, cov_names = character(),
                  y_obs = NULL, ...) {
    value <- do.call(module()$negative_loglik, c(list(
      theta = as.list(theta), z_obs = as.list(z_obs), z_int = as.list(z_int),
      int_weights = as.list(int_weights), obs_weights = as.list(obs_weights),
      cov_obs = cov_obs, cov_int = cov_int, cov_names = as.list(cov_names),
      y_obs = if (!is.null(y_obs)) as.list(y_obs) else NULL), config))
    reticulate::py_to_r(value$item())
  }
  attr(nll, "python_config") <- config
  extra <- switch(name, negbin = "log_nb_theta", zinb = c("log_nb_theta", "logit_pi"),
                    gaussian = if (is.null(config$known_sd)) "log_sd" else character(), character())
  family <- intensity_family(name, nll,
    deviance_residuals_fn = function(observed, fitted, extra_params) {
      size <- extra_params[["size"]] %||% 1
      if (is.na(size)) size <- 1
      values <- module()$deviance_residuals(as.list(observed), as.list(fitted), family = name, size = size)
      as.numeric(reticulate::py_to_r(values$`__array__`()))
    },
    init_fn = function(n_cov) {
      spec <- reticulate::py_to_r(do.call(module()$family_spec, c(list(n_cov = as.integer(n_cov)), config)))
      lapply(spec[c("start", "lower", "upper")], as.numeric)
    }, n_extra_params = length(extra), extra_param_names = extra,
    param_names_fn = if (name %in% c("rsf", "clogit")) function(cov_names) {
      if (length(cov_names)) c("gamma", paste0("beta_", cov_names)) else "gamma"
    } else NULL)
  if (name == "clogit") family$uses_strata <- !is.null(config$stratum_ids_used)
  family
}

.fit_family <- function(family, z_obs, z_int, int_weights, obs_weights,
                         cov_obs, cov_int, cov_names, response) {
  config <- attr(family$negloglik_fn, "python_config")
  if (!is.null(config)) {
    if (!ds_jax_check()) ds_jax_setup()
    module <- reticulate::import("diffiscape.likelihoods", convert = FALSE)
    result <- do.call(module$fit_intensity, c(list(
      z_obs = as.list(z_obs), z_int = as.list(z_int), int_weights = as.list(int_weights),
      obs_weights = as.list(obs_weights), cov_obs = cov_obs, cov_int = cov_int,
      cov_names = as.list(cov_names), initial = lapply(family$init_fn(length(cov_names)), as.list),
      y_obs = if (!is.null(response)) as.list(response) else NULL), config))
    result <- reticulate::py_to_r(result)
    result$par <- as.numeric(result$par)
    return(result)
  }
  inits <- family$init_fn(length(cov_names))
  stats::optim(par = inits$start, fn = family$negloglik_fn, method = "L-BFGS-B",
    lower = inits$lower, upper = inits$upper, z_obs = z_obs, z_int = z_int,
    int_weights = int_weights, obs_weights = obs_weights, cov_obs = cov_obs,
    cov_int = cov_int, cov_names = cov_names, y_obs = response, hessian = TRUE)
}
