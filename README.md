# DiffiScape

**Landscape Connectivity Optimization — a modular R framework for resistance and intensity model experimentation**

DiffiScape is an R package designed as a flexible glue layer for fitting landscape resistance surfaces against animal movement data. It bridges circuit-theory connectivity computation ([Omniscape.jl](https://github.com/circuitscape/Omniscape.jl) / [Circuitscape.jl](https://github.com/circuitscape/Circuitscape.jl)) with a suite of swappable likelihood models, making it easy to experiment with different resistance and intensity model combinations. Point process likelihoods are currently implemented, with the architecture designed to accommodate other movement data likelihoods in the future.

Users supply environmental rasters and GPS locations; DiffiScape estimates resistance parameters that best explain the observed spatial distribution of animal occurrences. Negative binomial and Poisson likelihoods are currently supported. The package ships two optimization back-ends and a full neural-network resistance pipeline:

- **Surrogate BO** — Latin Hypercube Sampling → GP emulator → Thompson Sampling or Expected Improvement (selectable)
- **Enzyme.jl L-BFGS** — gradient-based optimization via automatic differentiation in Julia
- **PyTorch pipeline** — MLP / convolutional / spline-GAM resistance networks trained with a differentiable circuit solver, plus MAP optimization and full Bayesian posterior sampling (Langevin/MALA, NUTS, ADVI)

---

## Installation

### R Package

```r
# Install from GitHub (requires remotes)
remotes::install_github("seansorek/DiffiScape")
```

### Julia Requirements

DiffiScape calls Julia internally via [`JuliaConnectoR`](https://cran.r-project.org/package=JuliaConnectoR). Before using the package you need:

- **Julia ≥ 1.9** — [Download Julia](https://julialang.org/downloads/)
- **Circuitscape.jl ≥ 5.0** and **Omniscape.jl ≥ 0.6** — installed automatically on first use via the bundled Julia project

```r
install.packages("JuliaConnectoR")
```

---

## Quick Start

```r
library(DiffiScape)

# Run the full pipeline in one call
result <- diffiscape(
  obs_data  = "gps_locations.csv",   # CSV with x, y columns (or shapefile)
  rasters   = "env_rasters/",        # Directory of .tif environmental layers
  output_dir = "my_analysis"
)

# Inspect results
result$opt_result$best_loglik     # Best log-likelihood
result$opt_result$best_params     # Optimal resistance parameters
result$posterior$summary          # Posterior summary table
```

---

## Design Philosophy

DiffiScape is built around easy **experimentation** with different resistance and intensity model combinations. Rather than committing to a single modelling approach, the package provides a clear interface so you can:

- **Swap resistance models** — supply any parameterised or data-driven function that maps environmental covariates to a resistance raster. Built-in helpers make it easy to get started with linear or log-linear formulations, and the architecture is open to ML-based or expert-driven alternatives.
- **Swap intensity models** — choose between parametric (negative binomial, Poisson) and nonparametric (GAM) count models for the inner-loop likelihood, or plug in your own.
- **Swap connectivity engines** — use Omniscape cumulative current flow, Circuitscape pairwise resistances, or any other raster-valued connectivity surface.

The surrogate optimizer (Latin Hypercube Sampling → GP + Thompson Sampling or Expected Improvement) coordinates the outer search over resistance parameters while calling whichever intensity model you choose at each evaluation. A gradient-based pathway via Enzyme.jl allows arbitrarily complex, differentiable resistance models. The PyTorch back-end adds neural-network resistance surfaces with full Bayesian posterior sampling.

---

## Modular Pipeline

The `diffiscape()` wrapper chains these modular step functions, which can also be called independently:

| Step | Function | Description |
|------|----------|-------------|
| 1 | `ds_load_data()` | Read GPS points from CSV, shapefile, or GeoPackage |
| 2 | `ds_create_basis()` | Build a multi-layer basis function stack from rasters |
| 3 | `ds_init_julia()` | Start the Julia session and load the DiffiScape Julia module |
| 4 | `ds_optimize()` | Outer surrogate loop for resistance parameters |
| 5 | `ds_fit_intensity()` | Fit the final intensity model at optimized parameters |
| 6 | `ds_predict()` | Generate a predicted intensity raster |
| 7 | `ds_posterior()` | Laplace approximation + Monte Carlo composition for uncertainty |
| 8 | `ds_diagnose()` | Deviance residuals, Moran's I, and diagnostic plots |

### Example: Step-by-step

```r
library(DiffiScape)

# Load data
pts   <- ds_load_data("gps_locations.csv")
basis <- ds_create_basis("env_rasters/")

# Start Julia
ds_init_julia()

# Optimize resistance parameters (swap intensity model via config)
cfg <- default_optimizer_config()
cfg$distribution <- "negbin"   # or "gam" / "poisson"
opt <- ds_optimize(basis, pts, config = cfg)

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
| `run_omniscape()` | Compute cumulative current flow via Omniscape.jl |
| `run_circuitscape()` | Pairwise or one-to-all current flow via Circuitscape.jl |
| `extract_connectivity()` | Extract connectivity values at point locations |
| `standardise_connectivity()` | Log-scale z-scoring of connectivity values |

### Intensity

| Function | Description |
|----------|-------------|
| `fit_intensity_nb()` | Negative binomial PPP MLE |
| `fit_intensity_gam()` | Poisson / GAM intensity via `mgcv::bam()` |
| `predict_intensity()` | Generate predicted intensity raster |
| `compute_information_criteria()` | AIC / BIC for model comparison |

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

## PyTorch Pipeline

DiffiScape ships a fully self-contained neural-network resistance back-end. The Python code is vendored inside the package (`inst/python/diff_cs/`) — no external files or research-repository clone is needed.

### Setup

```r
# Install Python dependencies into the active reticulate environment:
ds_install_torch_deps()           # CPU — installs torch, numpy, scipy, pyamg
ds_install_torch_deps(gpu = TRUE) # GPU — additionally installs cupy-cuda12x

# Initialise the Python session (done once per R session):
ds_torch_setup()
ds_torch_check()  # TRUE when ready
```

### MAP Optimization (Adam)

```r
result <- run_torch_pipeline(
  basis_stack = basis,
  obs_points  = pts,
  model_type  = "mlp",        # "mlp", "conv", or "spline_gam"
  n_epochs    = 500L,
  output_dir  = "torch_results/"
)

result$resistance_raster   # SpatRaster — optimized resistance surface
result$connectivity_raster # SpatRaster — resulting current flow
result$intensity_raster    # SpatRaster — predicted intensity surface
result$best_loglik         # scalar — best log-likelihood achieved
```

### Bayesian Posterior Sampling

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

### Gradient Checks

```r
verify_torch_gradient(basis, pts)    # MLP gradient check
verify_conv_gradient(basis, pts)     # convolutional resistance gradient
verify_spline_gradient(basis, pts)   # spline-GAM gradient
```

### Using the Torch Back-end via `ds_optimize()`

```r
opt <- ds_optimize(basis, pts, solver = "torch",
                   config = list(torch = list(
                     model_type = "mlp", n_epochs = 300L
                   )))
```

---

## Dependencies

**R Imports:** `terra`, `mgcv`, `numDeriv`, `DiceKriging`, `lhs`, `Matrix`

**R Suggests:** `JuliaConnectoR` (required for Julia/connectivity back-end), `reticulate` (required for PyTorch back-end), `ggplot2`, `spdep`, `testthat`

**Julia:** Circuitscape.jl ≥ 5.0, Omniscape.jl ≥ 0.6 (Julia ≥ 1.9)

**Python (optional, for PyTorch back-end):** Python ≥ 3.9, `torch`, `numpy`, `scipy`, `pyamg` — installed via `ds_install_torch_deps()`

---

## License

GPL (≥ 3) — see [LICENSE.md](LICENSE.md)

## Author

Sean Sorek (<sssorek1@gmail.com>)
