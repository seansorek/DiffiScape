# Contributing to DiffiScape

Thanks for your interest in DiffiScape. This is a multi-language package: an R
front end with **Python** compute backends (**JAX** for surrogate and gradient solvers,
**PyTorch** for neural networks). Read [ARCHITECTURE.md](ARCHITECTURE.md) first — it explains
which code runs for which `solver`.

## Prerequisites

| Backend | Needed for | Install |
|---------|-----------|---------|
| **R ≥ 4.1** | everything | the package itself + `Imports` in `DESCRIPTION` |
| **Python ≥ 3.10** | all solvers (JAX surrogate/gradient and PyTorch) | `pip install -r inst/python/diffiscape_jax/requirements.txt` and optionally `pip install -r inst/python/diff_cs/requirements.txt` |

Python is **required** (`Suggests` in `DESCRIPTION`): the R package installs without
it, but you need Python to use any solver. If working on JAX, you need JAX deps; if
working on PyTorch, you need torch deps.

## Local setup

```r
# R side
install.packages("devtools")
devtools::install_deps(dependencies = TRUE)
devtools::load_all()
```

```bash
# JAX backend (required for surrogate and gradient solvers)
pip install -r inst/python/diffiscape_jax/requirements.txt
# This installs: jax, jaxscape, flax, numpyro, numpy, scipy
```

```bash
# PyTorch backend (optional, for torch and irl solvers)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r inst/python/diff_cs/requirements.txt
# GPU (optional): also `pip install cupy-cuda12x` — auto-detected at runtime.
```

From R, initialize the JAX backend once per session before using surrogate/gradient:
`ds_install_jax_deps()` (or let `diffiscape()` do it automatically on first use).

## Running the tests

| Suite | Command | Location |
|-------|---------|----------|
| R | `devtools::test()` or `R CMD check` | `tests/testthat/` |
| Python JAX | `cd inst/python && pytest diffiscape_jax/tests/` | `inst/python/diffiscape_jax/tests/` |
| Python PyTorch | `cd inst/python && pytest tests/` | `inst/python/tests/` |

CI runs all three on every PR (`.github/workflows/{r,python,test-coverage}.yml`)
and uploads coverage to Codecov under the `r` / `jax` / `torch` flags
(configured in [codecov.yml](codecov.yml)). Please add or update tests for the
backend you touch — the JAX optimize and sample modules, and the PyTorch Bayesian samplers,
are areas where new tests are especially welcome.

### Gradient changes

If you modify a differentiable solver or a resistance net, run the matching
finite-difference check so you don't ship a wrong gradient:

- JAX: `verify_*` helpers in `inst/python/diffiscape_jax/tests/` (test modules).
- PyTorch: `verify_circuit_gradient` / `verify_conv_gradient` /
  `verify_spline_gradient` / `verify_irl_gradient` in `05_torch_pipeline.py`
  (exposed in R as `verify_*_gradient`).

## Conventions

- **R**: roxygen2 doc comments (`@export` public functions); regenerate docs with
  `devtools::document()` and commit the `man/*.Rd` + `NAMESPACE` changes. Internal
  helpers are prefixed `.` (e.g. `.outer_objective`). Follow the existing British
  spelling in user-facing strings (`standardise`, `residualise`).
- **JAX**: use `jax.vmap` and `jax.grad` for clean functional code. Avoid mutable state.
  Keep numeric defaults in module-level constants near the top of files.
- **PyTorch**: keep new numeric defaults in the `DEFAULT_*` constants block at the
  top of `05_torch_pipeline.py` rather than as scattered literals.

## Pull requests

- One logical change per PR; keep refactors separate from behavior changes.
- For anything touching the science (solvers, likelihoods, gradients), note in the
  PR how you verified numerical behavior is unchanged (e.g. a fixed-seed example or
  a gradient check).
- Make sure `R CMD check`, `pytest`, and the Julia tests pass for the backend(s)
  you changed.

## Known tech debt

A prioritized backlog of structural cleanups (JAX-PyTorch code consolidation, improved
test coverage for JAX samplers, S3 dispatch for resistance models) is tracked separately.
If you hit one of these while working, reference it rather than expanding your PR's scope.
