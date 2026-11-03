# DiffiScape Python

An array-based scientific core for landscape connectivity, resistance fitting,
intensity families, surrogate optimization, and NumPyro inference. R is not required.

Install the tested CPU stack with Python 3.12:

```sh
python -m pip install -c python/constraints-cpu.txt './python[dev]'
```

From a released distribution, install the matching `diffiscape` wheel instead.
GPU installation and validation are separate; native Windows CUDA is not supported
by the tested CPU installation.

```python
import numpy as np
from diffiscape import fit, predict, sample, posterior_predictive

basis = np.linspace(0, 1, 12).reshape(12, 1)
mask = np.ones(12, dtype=bool)
counts = np.array([0, 1, 0, 2, 0, 0, 1, 0, 0, 2, 0, 1])
result = fit(basis, counts, mask, 3, 4, source_spacing=2,
             n_epochs=10, verbose=False)
prediction = predict(result, basis)
assert prediction['intensity'].shape == (3, 4)
posterior = sample(basis, counts, mask, 3, 4, fitted=result,
                   n_samples=100, warmup=100)
replicas = posterior_predictive(result, basis, posterior)
```

The default is log-linear resistance, harmonic conductance, global absorption,
and Poisson point-process intensity `exp(alpha + softplus(raw_gamma)*log1p(C))`.
`method='adam'` uses Optax; `method='lbfgs'` uses SciPy L-BFGS-B. Log-linear
box bounds follow `[intercept, covariate coefficients...]`.
`model_type` also accepts `mlp`, `conv`, and `spline_gam` (penalized B-splines).
`model_config` supplies architecture/spline settings. `formulation` selects
`global_absorption`, fixed-zero `boundary`, or `focal_current`. The last measures
current into a single focal ground; conservation reduces it to a source sum.
These definitions are distinct from the explicitly selected legacy
`window_resistance`, whose outputs are accumulated effective resistance maps.
Its historical `spline_gam` is an RBF model, not a B-spline model.

Inputs are finite float64 `(n_valid, n_covariates)` arrays in C/row-major order,
with a Boolean full-grid mask. Conv grid construction happens inside Python.
Prediction uses the fit's geometry, mask, covariate order, and stored state;
masked outputs are NaN. Cell area is positive, in projected units squared.
Physical barriers use resistance 1e6; legacy windows use 5000.

Physical posterior inference uses proper Normal priors: free model parameters,
alpha, and sampled log-smoothing parameters have mean 0 and SD 2; raw gamma
has mean 0.54132485 and SD 1, with positive gamma through softplus. Existing
regularization penalties remain. Smoothing is fixed by default; pass
`fix_smoothing=False` to sample it. These priors replace an improper legacy
spline target and are an intentional statistical change.

Built-in `fit_intensity` covers Poisson, aggregate negative-binomial and
zero-inflated NB PPPs, Gaussian response, RSF/RSP, and conditional logit.
NB/ZINB predictive counts share one Gamma rate (and one structural zero state)
across cells per replicate, matching the aggregate likelihood.
`diffiscape.surrogate` uses SMT ordinary kriging and SciPy Latin hypercubes;
`loo_cv(X, y, config=None)` refits that same model and reports failed refits,
residuals, RMSE, R-squared, and interval coverage.

Results contain NumPy arrays and metadata. Pickle can preserve local fits and
surrogates; only load trusted artifacts. Torch checkpoints require refitting.
IRL and random-forest surrogates are retired. There is no Torch or JAXopt runtime.

```sh
python -m build python
python -m pip install --no-deps --force-reinstall python/dist/*.whl
# Run from outside the source directory, using an absolute tests path:
python -m pytest /path/to/DiffiScape/python/tests
```
