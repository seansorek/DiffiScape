# Architecture

The repository builds two packages. `python/src/diffiscape` owns shared science;
the root R package owns raster/point conversion, spatial metadata, S3 behavior,
plots, and explicit R-only extensions. Python never imports R.

| Responsibility | Implementation |
|---|---|
| Public arrays, fitting, saved-state prediction, sampling | `api.py` |
| Harmonic physical circuits and legacy arithmetic windows | `core.py`, `window.py` |
| Public AMJax init/reinit, sparse solves and derivatives | JAXScape, AMJax, Lineax |
| Physical log-linear/MLP/Conv/B-spline models | `physical.py`, `physical_models.py` |
| Legacy window MLP/Conv/RBF models | `resistance.py`, `optimize.py` |
| B-spline basis and derivatives | SciPy BSpline through `splines.py` |
| Adam, bounded optimization | Optax, SciPy L-BFGS-B |
| Built-in intensity likelihoods, MLE, Hessians, transforms | `likelihoods.py` |
| NUTS and variational inference | NumPyro through `physical.py`, `sampling.py` |
| Ordinary kriging, surrogate BO and leave-one-out checks | SMT and SciPy through `surrogate.py` |
| Residuals, summaries, criteria, predictive statistics | `diagnostics.py` |
| Shared spatial input conversion | `R/utils.R::.prepare_backend_inputs` |
| Installed package bridge and physical fitting | `R/jax_bridge.R`, `R/physical_pipeline.R` |

The public choices are optimization method, resistance model, connectivity
formulation, and likelihood. They are not interchangeable backend names.
The R default `adam` and `lbfgs` methods fit physical-current PPP models jointly.
The legacy `gradient` method fits a window-resistance PPP. `surrogate` retains
the derivative-free outer workflow and profiles built-in intensity families.
`mgcv` GAM, custom resistance links and their raster sensitivity helper, custom intensity callbacks, custom R
objectives, Laplace refits of those objectives, and spatial Moran statistics
remain explicitly R-only. The surrogate outer callback crosses languages only
as a derivative-free objective; JAX never differentiates through R.

Array ordering, masks, barriers, sources, grounding, current aggregation and
statistical definitions are specified in `docs/JAX_MIGRATION_STATUS.md` and
checked by independent tiny dense oracles. Effective resistance maps are kept
distinct from voltage and physical current. No production dense fallback,
custom adjoint, circuit solver kernel, or MCMC algorithm is maintained.

R imports the installed versioned Python package lazily. It does not construct
Flax models or optimize one iteration at a time. Fits preserve serializable
state and configuration for prediction. Only trusted local artifacts may be
unpickled. Deprecated Torch names are thin translations or actionable errors;
IRL, RF, Torch implementations, CuPy/AMGX paths and JAXopt are retired.

Scientific tests live in `python/tests`; R tests focus on conversion, delegation,
extensions and public behavior. CI builds a wheel, tests it outside the source
directory, installs the sdist, and checks R with the installed compatible core.
One CPU constraint file records tested dependencies; compatibility upgrades and
GPU validation require separate checks. Python must publish before dependent R.
