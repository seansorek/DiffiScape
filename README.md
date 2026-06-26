# DiffiScape 0.1.0

**Landscape Connectivity Optimization — a modular R framework for resistance and intensity model experimentation**

DiffiScape is an R package designed as a flexible glue layer for fitting landscape resistance surfaces against animal movement data. It bridges circuit-theory connectivity computation via JAX (primary) or PyTorch with a suite of swappable likelihood models, making it easy to experiment with different resistance and intensity model combinations. Both point process likelihoods (for occurrence/count data) and selection function likelihoods (RSF, RSP, and conditional logistic for iSSA/SSA) are supported, with the architecture designed to accommodate additional movement data likelihoods in the future.

Users supply environmental rasters and GPS locations (or paired used-available locations for selection models); DiffiScape estimates resistance parameters that best explain the observed spatial distribution of animal occurrences or movement choices. The package ships three optimization modes:

- **Surrogate BO** — Latin Hypercube Sampling → GP emulator → Thompson Sampling or Expected Improvement (selectable) via JAX connectivity
- **Gradient descent** — parametric or neural network resistance via automatic differentiation in JAX
- **PyTorch neural networks** — MLP / convolutional / spline-GAM / IRL value-shaped resistance networks trained with a differentiable circuit solver, plus MAP optimization and full Bayesian posterior sampling (Langevin/MALA, NUTS, ADVI)

---

## Installation

### R Package

```r
# Install from GitHub (requires remotes)
remotes::install_github("seansorek/DiffiScape")
```

### Python Requirements

DiffiScape now requires Python for all solver modes (surrogate, gradient, torch/irl). Before using the package you need:

- **Python ≥ 3.10** — [Download Python](https://www.python.org/downloads/) or use a conda distribution
- **JAX, JAXScape, Flax, NumPyro** — installed automatically on first use, or install manually via `pip install jax jaxscape flax numpyro`
- **PyTorch** (optional) — required only for `solver = "torch"` or `solver = "irl"` modes

```r
# R will use reticulate to manage your Python environment
# Install minimal deps on first use, or manually:
ds_install_jax_deps()      # JAX+JAXScape (recommended)
ds_install_torch_deps()    # Optional: PyTorch for neural network backend
```

---

## Quick Start

```r
library(DiffiScape)

# Run the full pipeline in one call (uses JAX surrogate solver by default)
# DiffiScape does not handle raster preprocessing. Make sure everything is aligned!
result <- diffiscape(
  obs_data  = "gps_locations.csv",   # CSV with x, y columns (or shapefile)
  rasters   = "env_rasters/",        # Directory of .tif environmental layers
  output_dir = "my_analysis"
)

# Inspect results
result$opt_result$best_loglik     # Best log-likelihood
result$opt_result$best_params     # Optimal resistance parameters
result$posterior$summary          # Posterior summary table

# Use different solvers:
result_gradient <- diffiscape(obs_data, rasters, output_dir, solver = "gradient")
result_torch <- diffiscape(obs_data, rasters, output_dir, solver = "torch")
```

---

## Design Philosophy

DiffiScape is built around easy **experimentation** with different resistance and intensity model combinations. Rather than committing to a single modelling approach, the package provides a clear interface so you can:

- **Swap resistance models** — supply any parameterised or data-driven function that maps environmental covariates to a resistance raster. Built-in helpers make it easy to get started with linear or log-linear formulations, and the architecture is open to ML-based or expert-driven alternatives.
- **Swap intensity models** — choose between parametric (negative binomial, Poisson) and nonparametric (GAM) count models for the inner-loop likelihood, or plug in your own.
- **Swap solvers** — use the JAX surrogate optimizer (robust, derivative-free, slow), JAX gradient descent (exact autodiff gradients, fast), or PyTorch neural networks (flexible, full Bayesian posteriors).

The surrogate optimizer (Latin Hypercube Sampling → GP + Thompson Sampling or Expected Improvement, powered by JAX) coordinates the outer search over resistance parameters while calling whichever intensity model you choose at each evaluation. The gradient descent pathway via JAX allows arbitrarily complex, differentiable resistance models with automatic differentiation. The PyTorch back-end adds neural-network resistance surfaces with full Bayesian posterior sampling.

---

## Modular Pipeline

The `diffiscape()` wrapper chains these modular step functions, which can also be called independently:

| Step | Function | Description |
|------|----------|-------------|
| 1 | `ds_load_data()` | Read GPS points from CSV, shapefile, or GeoPackage |
| 2 | `ds_create_basis()` | Build a multi-layer basis function stack from rasters |
| 3 | `ds_optimize()` | Outer solver loop (surrogate, gradient, or torch) for resistance parameters |
| 4 | `ds_fit_intensity()` | Fit the final intensity model at optimized parameters |
| 5 | `ds_predict()` | Generate a predicted intensity raster |
| 6 | `ds_posterior()` | Laplace approximation + Monte Carlo composition for uncertainty |
| 7 | `ds_diagnose()` | Deviance residuals, Moran's I, and diagnostic plots |

### Example: Step-by-step

```r
library(DiffiScape)

# Load data
pts   <- ds_load_data("gps_locations.csv")
basis <- ds_create_basis("env_rasters/")

# Optimize resistance parameters (swap intensity model via config)
cfg <- default_optimizer_config()
cfg$distribution <- "negbin"   # or "gam" / "poisson"
opt <- ds_optimize(basis, pts, config = cfg, solver = "surrogate")  # or "gradient" / "torch"

# Fit final model at optimized parameters
fit <- ds_fit_intensity(opt, basis, pts)

# Predict intensity surface
connectivity  <- run_omniscape(create_resistance_surface(opt$best_params, basis))$cum_current
intensity_map <- ds_predict(fit, connectivity)

# Posterior inference
post <- ds_posterior(opt, basis, pts, n_draws = 200L)
post$summary

# Diagnostics
diag <- ds_diagnose(fit, pts, connectivity)
```

---

## Key Functions

### Resistance

| Function | Description |
|----------|-------------|
| `create_resistance_surface()` | Build a resistance raster from parameters and basis stack |
| `quick_resistance()` | Fast matrix-form resistance (for optimization loops) |
| `get_default_bounds()` | Default parameter bounds (`r_0 ∈ [0,6]`, `z_k ∈ [-3,3]`) |
| `resistance_sensitivity()` | Numerical sensitivity of resistance to each parameter |

### Connectivity

| Function | Description |
|----------|-------------|
| `run_jax_connectivity()` | Compute cumulative current flow via JAX (primary) |
| `run_torch_connectivity()` | Compute cumulative current flow via PyTorch (optional, for torch backend) |
| `extract_connectivity()` | Extract connectivity values at point locations |
| `standardise_connectivity()` | Log-scale z-scoring of connectivity values |
| `residualise_connectivity()` | Regress local covariates out of connectivity, isolating network-structure variation |

### Intensity

| Function | Description |
|----------|-------------|
| `fit_intensity_nb()` | Negative binomial / selection-function MLE (general workhorse) |
| `fit_intensity_selection()` | Convenience wrapper for RSF / RSP / iSSA models with explicit available locations |
| `fit_intensity_gam()` | Poisson / GAM intensity via `mgcv::bam()` |
| `predict_intensity()` | Generate predicted intensity raster |
| `compute_information_criteria()` | AIC / BIC for model comparison |
| `intensity_family()` | Constructor for a custom distributional family |

### Basis Stack

| Function | Description |
|----------|-------------|
| `create_basis_stack()` | Build and align a multi-layer `SpatRaster` from a named list of rasters |
| `validate_basis_stack()` | Check that a basis stack is well-formed (named, non-empty, no all-NA layers) |
| `check_basis_correlations()` | Compute pairwise Pearson correlations and warn when `\|r\| > threshold` |
| `basis_summary()` | Data-frame summary of each layer (mean, SD, min, max, valid/NA cell counts) |

### Diagnostics

| Function | Description |
|----------|-------------|
| `compute_deviance_residuals()` | Cell-level NB deviance residuals |
| `rasterise_deviance_residuals()` | Map deviance residuals onto a grid |
| `moran_test()` | Moran's I spatial autocorrelation test |
| `plot_deviance_residuals()` | Residuals vs fitted plot |
| `plot_qq_deviance()` | QQ-plot of deviance residuals |
| `plot_residual_map()` | Spatial map of residuals |

### Posterior

| Function | Description |
|----------|-------------|
| `laplace_resistance()` | Gaussian approximation to the resistance posterior |
| `posterior_sample()` | Monte Carlo draws from the joint posterior |
| `posterior_summary()` | Summary table of posterior samples |
| `plot_posterior()` | Visualise marginal posteriors |
| `loo_cv_surrogate()` | LOO-CV validation of the GP emulator |

---

## Configuration

### Optimizer Config

```r
cfg <- default_optimizer_config()
# Key fields:
#   n_init       = 20L         # LHS design points
#   n_iter       = 50L         # Surrogate iterations
#   acquisition  = "TS"        # "TS" (Thompson Sampling) or "EI" (Expected Improvement)
#   distribution = "negbin"    # or "gam" / "poisson"
#   family       = NULL        # intensity_family object (overrides distribution)
#   resistance_link = NULL     # resistance_link object (NULL → link_exp())
#   omniscape    = list(radius = 13L, block_size = 5L)
#
# EI-specific knobs (used when acquisition = "EI"):
#   ei_xi_scale_factor  = 0.1   # xi = score_range * scale_factor at iteration 1
#   ei_xi_min           = 0.02  # floor xi decays toward
#   ei_decay_rate_divisor = 5   # controls how quickly xi decays

# Use Expected Improvement acquisition instead of Thompson Sampling:
cfg$acquisition <- "EI"
opt <- ds_optimize(basis, pts, config = cfg)
```

### Intensity Config

```r
int_cfg <- default_intensity_config()
# Key fields:
#   k_connectivity = 10L   # GAM knots for connectivity
#   covariate_type = "smooth"  # or "linear"
#   include_spatial_re = FALSE
#   integration_subsample = 0.25
```

---

## Intensity Families

The intensity model can use any of seven built-in distributional families — covering both point process count models and selection function models — or a fully custom one. Pass a family object via `config$family` to override the legacy `distribution` string.

**Point process / count families**

| Family | Constructor | Description |
|--------|-------------|-------------|
| Negative Binomial | `family_negbin()` | Default; handles overdispersion with a size parameter θ |
| Poisson | `family_poisson()` | Simpler model; no overdispersion adjustment |
| Gaussian | `family_gaussian(known_sd)` | For continuous density estimates; `known_sd = NULL` estimates σ |
| Zero-inflated NB | `family_zinb()` | Extends NB with a structural-zero probability π |

**Selection function families** (require explicit available/background locations)

| Family | Constructor | Description |
|--------|-------------|-------------|
| RSF | `family_rsf()` | Exponential resource selection function via Poisson use-availability likelihood; no intercept α |
| RSP | `family_rsp(background_weight)` | Logistic RSP via the Fithian & Hastie (2013) infinite-weight trick; estimates a selection probability surface |
| Conditional logistic | `family_clogit(stratum_ids_used, stratum_ids_avail)` | Paired used-available (iSSA/SSA); reduces to RSF when strata are omitted |

```r
cfg <- default_optimizer_config()

# Point process families:
cfg$family <- family_poisson()
cfg$family <- family_zinb()

# Selection function families — also pass available locations:
cfg$family           <- family_rsf()
cfg$available_points <- available_locs   # data.frame with x, y

opt <- ds_optimize(basis, pts, config = cfg)

# Conditional logistic (iSSA) with pre-defined strata:
cfg$family <- family_clogit(
  stratum_ids_used  = used_df$stratum_id,
  stratum_ids_avail = avail_df$stratum_id
)

# Convenience wrapper for selection models (bypasses raster quadrature):
fit <- fit_intensity_selection(
  connectivity_at_obs  = connectivity[used_idx],
  available_connectivity = connectivity[avail_idx],
  obs_coords           = used_df[, c("x", "y")],
  family               = family_rsf()
)
```

To define a completely custom family, supply a `negloglik_fn`, `deviance_residuals_fn`, and `init_fn` to `intensity_family()`:

```r
my_family <- intensity_family(
  name                  = "my_family",
  negloglik_fn          = function(theta, z_obs, z_int, int_weights, obs_weights,
                                   cov_obs, cov_int, cov_names) { ... },
  deviance_residuals_fn = function(observed, fitted, extra_params) { ... },
  init_fn               = function(n_cov) list(start = ..., lower = ..., upper = ...)
)
cfg$family <- my_family
```

---

## Resistance Link Functions

The link function maps the linear predictor η(x) = r₀ + Σₖ zₖ φₖ(x) to the resistance surface R(x). Swapping links changes the shape of the resistance surface without touching the optimizer or Julia solver.

| Link | Formula | Notes |
|------|---------|-------|
| `link_exp()` | R = clamp(exp(η), Rmin, Rmax) | Default; original DiffiScape parameterisation |
| `link_softplus()` | R = log(1 + exp(η)) + Rmin | Smooth, naturally bounded below, no clamp discontinuity |
| `link_power(p)` | R = clamp(\|η\|ᵖ, Rmin, Rmax) | Polynomial scaling; default p = 2 |
| `link_identity()` | R = clamp(η, Rmin, Rmax) | Use when basis functions are already on the resistance scale |

```r
# Pass a link directly to create_resistance_surface():
R <- create_resistance_surface(theta, basis, link = link_softplus())

# Or configure it for the full optimizer loop:
cfg <- default_optimizer_config()
cfg$resistance_link <- link_softplus()
opt <- ds_optimize(basis, pts, config = cfg)

# Power link with p = 3:
cfg$resistance_link <- link_power(p = 3)
```

For non-additive models (e.g. tensor products, neural nets), supply a custom `eta_fn` to `resistance_link()`:

```r
my_link <- resistance_link(
  name       = "my_link",
  forward_fn = function(eta, R_min, R_max) { ... },
  inverse_fn = function(R, R_min, R_max) { ... },
  deriv_fn   = function(eta, R_min, R_max) { ... },
  eta_fn     = function(theta, basis_values) { ... }   # optional
)
```

---

## Solver Modes

DiffiScape provides three distinct solver modes, each with different trade-offs.

### Surrogate Optimizer (JAX) — Default

The primary solver mode using Bayesian Optimization with Gaussian Process emulation. Fast, robust, and derivative-free for moderate-dimensional problems.

```r
# Default behavior
opt <- ds_optimize(basis, pts, config = cfg)

# Explicit specification
opt <- ds_optimize(basis, pts, config = cfg, solver = "surrogate")

# Custom configuration
cfg <- default_optimizer_config()
cfg$acquisition <- "EI"  # Expected Improvement (default: "TS" Thompson Sampling)
opt <- ds_optimize(basis, pts, config = cfg, solver = "surrogate")
```

### Gradient Descent (JAX)

For parametric or shallow neural-network resistance models with automatic differentiation via JAX.

```r
opt <- ds_optimize(basis, pts, solver = "gradient",
                   config = list(jax = list(
                     n_epochs = 100L,
                     learning_rate = 0.01,
                     optimizer = "adam"  # or "lbfgs", "sgd"
                   )))
```

### PyTorch Neural Networks

DiffiScape ships a fully self-contained neural-network resistance back-end. The Python code is vendored inside the package (`inst/python/diff_cs/`) — no external files or research-repository clone is needed.

#### Setup

```r
# Install Python dependencies into the active reticulate environment:
ds_install_torch_deps()           # CPU — installs torch, numpy, scipy, pyamg
ds_install_torch_deps(gpu = TRUE) # GPU — additionally installs cupy-cuda12x

# Initialise the Python session (done once per R session):
ds_torch_setup()
ds_torch_check()  # TRUE when ready
```

#### MAP Optimization (Adam)

```r
result <- run_torch_pipeline(
  basis_stack = basis,
  obs_points  = pts,
  model_type  = "mlp",        # "mlp", "conv", "spline_gam", or "irl"
  n_epochs    = 500L,
  output_dir  = "torch_results/"
)

result$resistance_raster   # SpatRaster — optimized resistance surface
result$connectivity_raster # SpatRaster — resulting current flow
result$intensity_raster    # SpatRaster — predicted intensity surface
result$best_loglik         # scalar — best log-likelihood achieved
```

#### Bayesian Posterior Sampling

```r
# Langevin / MALA sampler:
mcmc <- run_bayesian_sampling(
  basis_stack = basis, obs_points = pts,
  n_samples = 1000L, burnin = 200L
)

# NUTS (No-U-Turn Sampler):
nuts <- run_bayesian_sampling_hmc(
  basis_stack = basis, obs_points = pts,
  n_samples = 500L, n_chains = 2L
)

# ADVI (Automatic Differentiation Variational Inference):
vi <- run_advi(basis_stack = basis, obs_points = pts, n_iter = 2000L)
```

#### Gradient Checks

```r
verify_torch_gradient(basis, pts)    # MLP gradient check
verify_conv_gradient(basis, pts)     # convolutional resistance gradient
verify_spline_gradient(basis, pts)   # spline-GAM gradient
```

#### Using the PyTorch Back-end via `ds_optimize()`

```r
opt <- ds_optimize(basis, pts, solver = "torch",
                   config = list(torch = list(
                     model_type = "mlp", n_epochs = 300L
                   )))
```

#### Inverse Reinforcement Learning (value-shaped) resistance

The `"irl"` resistance model treats the landscape as a Markov decision process
(states = cells, actions = moves to neighbours) and learns the **reward**
(negative cost) an agent appears to be following. A reward network over the
basis covariates is turned into a resistance surface via entropy-regularised
**soft value iteration**, so resistance reflects the *long-range desirability*
of the landscape (an agent's plan-to-go value), not just local habitat:

```
reward rψ(x) → soft value iteration (β, γ_d) → V(x)
            → R(x) = clamp(offset − scale·V(x))
            → differentiable circuit solver → C(x) → log λ = α + γ·log(1+C) → PPP likelihood
```

Crucially this **keeps circuit theory as the forward model**: the learned
resistance flows through the same differentiable current solve as the other
PyTorch models, and gradients backpropagate through both the circuit solve and
the unrolled value iteration into the reward network. It needs only occurrence
points (no movement trajectories).

```r
# Convenience alias for solver = "torch" with model_type = "irl":
opt <- ds_optimize(basis, pts, solver = "irl",
                   config = list(torch = list(
                     beta = 1.0,        # soft value-iteration temperature
                     gamma_d = 0.9,     # MDP discount / leakage (< 1)
                     n_value_iter = 60L,
                     n_epochs = 300L
                   )))

# Package the fit as a custom resistance_model for predict()/ds_diagnose():
model <- irl_resistance_model(opt, basis)
R     <- predict(model)

# Gradient smoke-test through value iteration + circuit solve:
verify_irl_gradient(basis)
```

> **Note:** v1 supports IRL for MAP (Adam) optimization. Bayesian posterior
> sampling (`run_bayesian_sampling()` / `run_advi()`) currently targets the
> spline-GAM model only; IRL posterior sampling is planned future work.

---

## Vignettes

Five worked examples ship with the package and are built automatically when it is installed. Access them with:

```r
browseVignettes("DiffiScape")
```

| Vignette | Topic |
|----------|-------|
| `surrogate-jax` | Surrogate BO with JAX cumulative current |
| `gradient-jax` | Gradient-based optimization via JAX autodiff |
| `gam-profile` | GAM intensity profile workflow |
| `torch-mlp` | PyTorch MLP resistance network |
| `spline-gam` | Spline-GAM resistance network |

---

## Dependencies

**R Imports:** `terra`, `mgcv`, `numDeriv`, `DiceKriging`, `lhs`, `Matrix`

**R Suggests:** `reticulate` (required for all backends), `ggplot2`, `spdep`, `testthat`

**Python (required):** Python ≥ 3.10 with `jax`, `jaxscape`, `flax`, `numpyro` — installed via `ds_install_jax_deps()`

**Python (optional, for PyTorch back-end):** `torch`, `numpy`, `scipy`, `pyamg` — installed via `ds_install_torch_deps()`

---

## License

GPL (≥ 3) — see [LICENSE.md](LICENSE.md)

## Author

Sean Sorek (<sssorek1@gmail.com>)
