# DiffiScape

**Differentiable Landscape Connectivity Optimization via CircuitScape.jl and Enzyme.jl**

DiffiScape is an R package for gradient-based and surrogate-based optimization of landscape connectivity surfaces. It combines circuit-theory connectivity computation ([Omniscape.jl](https://github.com/circuitscape/Omniscape.jl) / [Circuitscape.jl](https://github.com/circuitscape/Circuitscape.jl)) with automatic differentiation ([Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)) to fit **Poisson Point Process (PPP) models** of animal movement.

Users supply environmental rasters and GPS locations; DiffiScape estimates resistance parameters that best explain the observed spatial intensity of animal occurrences. Both negative binomial and GAM-based intensity models are supported via [`mgcv`](https://cran.r-project.org/package=mgcv).

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

## The Statistical Model

### Resistance Surface

Resistance is parameterised as a log-linear combination of spatial basis functions:

$$\log R(x) = r_0 + \sum_{k=1}^{K} z_k \, \phi_k(x)$$

$$R(x) = \text{clamp}\bigl(\exp(\log R(x)),\; R_{\min},\; R_{\max}\bigr)$$

where $\phi_k(x)$ are environmental rasters (e.g. land cover, elevation, NDVI) and $z_k$ are the coefficients to be estimated.

### Intensity Model

Given a connectivity surface $C(x)$ (cumulative current flow from Omniscape), the spatial intensity of animal observations is modelled as:

$$\log \lambda(x) = \alpha + \gamma \cdot z(x) + \sum_j \beta_j \cdot \text{cov}_j(x)$$

where $z(x)$ is a standardised log-transformation of $C(x)$, and the integral of $\lambda$ is approximated by quadrature for the PPP log-likelihood.

### Optimizer

Resistance parameters are estimated through a two-stage Bayesian optimization loop:

1. **Phase 1 — Latin Hypercube Sampling**: evaluates `n_init` (default 20) design points spread across the parameter space.
2. **Phase 2 — GP Surrogate + Thompson Sampling**: fits a Gaussian process (Matérn 5/2 kernel via `DiceKriging`) to the evaluated points and uses Thompson Sampling acquisition with adaptive local search for `n_iter` (default 50) further evaluations.

At each outer evaluation, the intensity parameters ($\alpha$, $\gamma$, $\beta_j$, size) are fitted by MLE (inner loop).

> **Note:** The Enzyme.jl automatic differentiation pathway (`optimize_resistance_enzyme()`) is planned but not yet implemented. The current default is the surrogate-based optimizer.

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

# Optimize resistance parameters
opt <- ds_optimize(basis, pts)

# Fit final model
fit <- ds_fit_intensity(opt, basis, pts)

# Predict intensity surface
connectivity <- run_omniscape(create_resistance_surface(opt$best_params, basis))$cum_current
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
| `fit_intensity_gam()` | GAM-based intensity via `mgcv::bam()` |
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
#   n_init    = 20L      # LHS design points
#   n_iter    = 50L      # Surrogate iterations
#   distribution = "negbin"  # or "gam"
#   omniscape = list(radius = 13L, block_size = 5L)
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

## Dependencies

**R Imports:** `terra`, `mgcv`, `numDeriv`, `DiceKriging`, `lhs`, `Matrix`

**R Suggests:** `JuliaConnectoR` (required for connectivity computation), `ggplot2`, `spdep`, `testthat`

**Julia:** Circuitscape.jl ≥ 5.0, Omniscape.jl ≥ 0.6 (Julia ≥ 1.9)

---

## License

GPL (≥ 3) — see [LICENSE.md](LICENSE.md)

## Author

Sean Sorek (<sssorek1@gmail.com>)
