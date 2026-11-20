# DiffiScape

DiffiScape estimates landscape resistance and the intensity of animal occurrences
from environmental rasters and observed locations. It uses differentiable circuit
models to relate landscape resistance to connectivity. The Python package uses
JAX for optimization, built-in intensity likelihoods, and posterior inference.
The R package calls the same Python code and handles spatial inputs, raster
outputs, and plotting.

## CPU installation

The CPU workflow was tested on Windows with Python 3.12. Run these commands from
the repository root.

```sh
python -m venv .venv
# Activate the environment, then:
python -m pip install -c python/constraints-cpu.txt ./python
```

In R, run `remotes::install_local(".")`. Set `RETICULATE_PYTHON` to the Python
executable in the virtual environment before reticulate starts Python. Loading
DiffiScape registers its Python requirements. Reticulate starts Python when you
first call a function that needs it.

The R package requires Python `diffiscape>=0.3.0,<0.4`. For an unreleased Python
version, install it from this checkout or from a wheel built from it. Publish
the Python release before distributing an R release that depends on it.

## Fit a model in R

This example fits a model to one small raster and six observations, including two
in the same cell. It predicts on the same grid, draws samples from the posterior,
and runs posterior predictive checks.

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

## Choose a model and solver

By default, `ds_optimize()` uses Optax Adam to fit log-linear resistance and
Poisson point-process intensity jointly. The circuit uses harmonic conductance
and global absorption. Set `solver="lbfgs"` to use SciPy's bounded L-BFGS-B.
Set the circuit options in `config$fit`, where you can choose global absorption,
a fixed-zero boundary, or focal current. Either optimizer can fit
`model_type="mlp"`, `"conv"`, or `"spline_gam"`. The physical spline is a penalized
B-spline with optional tensor interactions. You can also call the physical
fitting workflow directly with `run_jax_pipeline()`.

The legacy `solver="gradient"` and the surrogate workflow compute moving-window
effective resistance, called cumulative current in older versions. The legacy
window spline uses radial basis functions (RBFs). See the circuit equations
and CPU test results in the
[numerical contract and validation](docs/JAX_MIGRATION_STATUS.md).

The physical joint fit currently uses Poisson intensity with connectivity as
its predictor. Use the derivative-free surrogate workflow for other families,
additional intensity covariates, or availability strata. That workflow also
supports R-only `mgcv` GAMs and custom R callbacks, which run outside JAX automatic
differentiation. Both packages use the same Python code for built-in intensity
families, residuals, summaries, predictive counts, kriging, and optimization.

## Upgrade older fits

The deprecated `solver="torch"`, `config$torch`, and `run_torch_pipeline()` now
route to the physical JAX model. Refit models that used direct Torch calls or
old `.pt` checkpoints. Custom MALA has been retired. Use
`ds_posterior(method="nuts")` or `method="advi"` for posterior inference.
Spline posteriors keep their existing penalties and use the proper priors
described in the [Python guide](python/README.md).

Curriculum absorption, uncertainty quantification from old checkpoints, neural
warm starts, and Torch-specific device options raise migration errors. The
migration message flags the retirement of stochastic focal subsampling and the
previously unused weight-decay option. IRL and random-forest surrogates have also
been retired. Refit DiceKriging models as SMT surrogates.

## More documentation

The synthetic-data vignettes contain worked R examples. See the
[Python usage guide](python/README.md) for standalone Python examples, or read
about the [package architecture](ARCHITECTURE.md) and
[how to contribute](CONTRIBUTING.md).
