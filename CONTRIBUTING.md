# Contributing to DiffiScape

Thanks for your interest in DiffiScape. This is a multi-language package: an R
front end with optional **Julia** (Enzyme) and **Python** (PyTorch) compute
backends. Read [ARCHITECTURE.md](ARCHITECTURE.md) first — it explains which code
runs for which `solver`.

## Prerequisites

| Backend | Needed for | Install |
|---------|-----------|---------|
| **R ≥ 4.1** | everything | the package itself + `Imports` in `DESCRIPTION` |
| **Julia ≥ 1.9** | `solver = "surrogate"` and `"enzyme"` | [julialang.org/downloads](https://julialang.org/downloads/); Circuitscape/Omniscape/Enzyme install on first use |
| **Python ≥ 3.10** | `solver = "torch"` / `"irl"` | `pip install -r inst/python/diff_cs/requirements.txt` |

The Julia and Python stacks are **optional** (`Suggests`): the R package installs
and its R tests run without them. You only need a backend to work on or test that
backend.

## Local setup

```r
# R side
install.packages("devtools")
devtools::install_deps(dependencies = TRUE)
devtools::load_all()
```

```bash
# Python backend (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r inst/python/diff_cs/requirements.txt
# GPU (optional): also `pip install cupy-cuda12x` — auto-detected at runtime.
```

```julia
# Julia backend
import Pkg; Pkg.add("Enzyme")   # Circuitscape/Omniscape resolve via the bundled project
```

From R, initialize a backend once per session before using it:
`ds_julia_setup()` / `ds_torch_setup()` (or `ds_install_torch_deps()`).

## Running the tests

| Suite | Command | Location |
|-------|---------|----------|
| R | `devtools::test()` or `R CMD check` | `tests/testthat/` |
| Python | `cd inst/python && pytest tests/` | `inst/python/tests/` |
| Julia | `julia inst/julia/DiffiScape/test/runtests.jl` | `inst/julia/DiffiScape/test/` |

CI runs all three on every PR (`.github/workflows/{r,python,julia,test-coverage}.yml`)
and uploads coverage to Codecov under the `r` / `python` / `julia` flags
(configured in [codecov.yml](codecov.yml)). Please add or update tests for the
backend you touch — the Python torch pipeline and Bayesian samplers are the most
under-covered area, so new tests there are especially welcome.

### Gradient changes

If you modify a differentiable solver or a resistance net, run the matching
finite-difference check so you don't ship a wrong gradient:

- Python: `verify_circuit_gradient` / `verify_conv_gradient` /
  `verify_softrl_gradient` / `verify_spline_gradient` in `05_torch_pipeline.py`
  (exposed in R as `verify_*_gradient`).
- Julia: `verify_*` helpers in `inst/julia/DiffiScape/test/`.

## Conventions

- **R**: roxygen2 doc comments (`@export` public functions); regenerate docs with
  `devtools::document()` and commit the `man/*.Rd` + `NAMESPACE` changes. Internal
  helpers are prefixed `.` (e.g. `.outer_objective`). Follow the existing British
  spelling in user-facing strings (`standardise`, `residualise`).
- **Python**: keep new numeric defaults in the `DEFAULT_*` constants block at the
  top of `05_torch_pipeline.py` rather than as scattered literals.
- **Julia**: keep the differentiable solver Enzyme-compatible (no allocations that
  break reverse-mode AD).

## Pull requests

- One logical change per PR; keep refactors separate from behavior changes.
- For anything touching the science (solvers, likelihoods, gradients), note in the
  PR how you verified numerical behavior is unchanged (e.g. a fixed-seed example or
  a gradient check).
- Make sure `R CMD check`, `pytest`, and the Julia tests pass for the backend(s)
  you changed.

## Known tech debt

A prioritized backlog of structural cleanups (module split, S3 dispatch, the
hardcoded-`exp`-link gap, sampler test coverage) is tracked separately. If you hit
one of these while working, reference it rather than expanding your PR's scope.
