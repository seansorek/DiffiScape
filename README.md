# DiffiScape

DiffiScape fits landscape resistance from environmental rasters and animal
locations using differentiable circuit connectivity. One Python scientific core
serves both independently installable Python and R packages.

## CPU installation

Python 3.12 is the tested interpreter. From this checkout:

```sh
python -m venv .venv
# Activate the environment, then:
python -m pip install -c python/constraints-cpu.txt ./python
```

Install the R package using `remotes::install_local(".")`. Before Python starts,
set `RETICULATE_PYTHON` to that environment's Python executable. Loading the R
package is lazy: it does not initialize Python, run a solver, or install a runtime.
The R package requires Python `diffiscape>=0.1.0,<0.2`. Until that Python release
is published, install from this checkout or the built wheel. Publish Python
before distributing an R release that depends on it.

```r
library(DiffiScape)
library(terra)
basis <- rast(nrows = 3, ncols = 4, xmin = 0, xmax = 4,
              ymin = 0, ymax = 3, crs = "EPSG:3857", vals = seq(0, 1, length.out = 12))
pts <- as.data.frame(xyFromCell(basis, c(1, 3, 3, 5, 8, 12)))
fit <- ds_optimize(basis, pts,
  config = list(fit = list(n_epochs = 10L, source_spacing = 2L)),
  output_dir = tempdir())
predicted <- ds_predict(fit, basis_stack = basis)
posterior <- ds_posterior(fit, basis, pts, n_draws = 100L, warmup = 100L)
checks <- ds_ppc(posterior, fit, pts, fit$connectivity_raster,
                 basis_stack = basis, n_sim = 100L, plot = FALSE)
```

`ds_optimize()` defaults to Optax Adam with log-linear resistance, harmonic
conductance, global absorption, and joint Poisson PPP intensity. Choose
`solver="lbfgs"` for bounded SciPy L-BFGS-B. Pass physical settings through
`config$fit`; choose `model_type="mlp"`, `"conv"`, or `"spline_gam"` independently.
`run_jax_pipeline()` exposes the same complete fitting operation directly.

Global absorption, fixed-zero boundary, and focal current are separate physical
formulations. The legacy `solver="gradient"` and surrogate workflow use
moving-window **effective resistance**, historically labelled cumulative current.
They do not reproduce physical current. The legacy window spline is an RBF;
the physical spline is a penalized B-spline with optional tensor interactions.
See [the numerical contract and validation](docs/JAX_MIGRATION_STATUS.md).

Built-in intensity families, residuals, summaries, predictive counts, kriging,
and optimization share Python implementations. The derivative-free surrogate
workflow supports the existing R-only `mgcv` GAM and custom R callback extensions.
Arbitrary R callbacks are not passed into JAX autodiff. Physical joint fitting
currently accepts the Poisson family without separate intensity covariates;
use the surrogate workflow for other families and availability strata.

## Migration behavior

`solver="torch"`, `config$torch`, and `run_torch_pipeline()` are deprecated
translations to the same physical JAX model. No Torch implementation is loaded.
Direct Torch calls and old `.pt` checkpoints require refitting. Use
`ds_posterior(method="nuts")` or `method="advi"`; custom MALA is retired.
Spline posteriors now use the approved proper priors described in the
[Python guide](python/README.md), retaining the existing penalties.
Curriculum absorption, old checkpoint UQ, neural warm starts, and Torch-specific
device choices raise migration errors. Stochastic focal subsampling and the
previous unused weight-decay option are retired with messages. IRL and RF
surrogates were removed by the concurrent simplification work. Raw DiceKriging
fits require refitting as SMT surrogates.

CPU is validated on Windows. GPU installation and numerical validation are
separate; these results do not establish accelerator support.

See [Python usage](python/README.md), [architecture](ARCHITECTURE.md),
[contributing](CONTRIBUTING.md), and the synthetic-data vignettes.
