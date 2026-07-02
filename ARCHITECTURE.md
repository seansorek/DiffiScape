# DiffiScape Architecture

DiffiScape is an **R package that orchestrates Python compute backends** (JAX and PyTorch)
to fit landscape *resistance* surfaces against animal-movement data. R is the user-facing
glue layer; the numerically intensive circuit-theory solves and their gradients happen
in Python via JAX or PyTorch and are called back into R.

This document maps the moving parts so a new contributor knows **which code runs
for which solver**, and where the circuit-theory math is implemented.

```
                ┌──────────────────────────────────────────────┐
   user data →  │  R layer  (R/)                                │
  rasters+GPS   │  basis_functions → ds_optimize → intensity →  │
                │  posterior / diagnostics / predict            │
                └───────┬───────────────┬───────────────┐───────┘
            solver=      │ "surrogate"    │ "gradient"    │ "torch" / "irl"
                         ▼                ▼                ▼
                  JAX surrogate    JAX autodiff       PyTorch diff.
                  (DiceKriging     circuit solver +   circuit solver
                   + JAX current)  gradient descent   (reticulate)
```

## The R layer (`R/`)

| File | Responsibility |
|------|----------------|
| `pipeline.R` | One-call `diffiscape()` + modular steps (`ds_load_data`, `ds_create_basis`, `ds_optimize`, `ds_fit_intensity`, `ds_predict`, `ds_posterior`, `ds_diagnose`). `ds_optimize` is the **backend dispatcher**. |
| `basis_functions.R` | Build/validate the basis-function stack from environmental rasters. |
| `optimizer.R` | `optimize_resistance_surrogate` (JAX surrogate/GP), `optimize_resistance_gradient` (JAX autodiff), and the outer objective machinery. |
| `connectivity.R` | R wrappers over the Python (JAX/PyTorch) connectivity entry points. |
| `intensity.R`, `intensity_family.R` | Likelihood/intensity models: NB, Poisson, GAM, plus selection families (RSF, RSP, clogit). `resolve_family()` picks one. |
| `resistance.R`, `resistance_link.R` | Resistance model objects and the swappable link functions (`link_exp`, `link_power`, `link_softplus`, `link_identity`). |
| `posterior.R`, `diagnostics.R` | Laplace/posterior summaries and model diagnostics. |
| `jax_bridge.R`, `torch_bridge.R` | Thin call interfaces to the JAX and PyTorch backends. |
| `jax_pipeline.R`, `torch_pipeline.R` | R-side wrappers that marshal arguments into the Python pipelines. |
| `utils.R`, `zzz.R` | Helpers, package `.onLoad`, internal env (`.ds_env`). |

## Backend selection — `ds_optimize(..., solver=)`

`ds_optimize` ([R/pipeline.R](R/pipeline.R)) routes to one of three solvers:

| `solver` | Entry point | Backend | Gradient strategy |
|----------|-------------|---------|-------------------|
| `"surrogate"` (default) | `optimize_resistance_surrogate` | **JAX** — surrogate GP emulation + cumulative current | None; Bayesian optimization over a DiceKriging GP surrogate (LHS init + Thompson sampling / expected improvement) |
| `"gradient"` | `optimize_resistance_gradient` | **JAX** — pure-JAX autodiff circuit solver | JAX `vmap` + `grad` → Adam / L-BFGS |
| `"torch"` | `run_torch_pipeline` | **PyTorch** — differentiable circuit solver | torch autograd (custom `Function`s) → Adam / Bayesian samplers |
| `"irl"` | `run_torch_pipeline` (`model_type="irl"`) | **PyTorch** | Same as `torch`, with the value-iteration (IRL) resistance net |

Rule of thumb: **surrogate** = robust, derivative-free, good for moderate-D problems;
**gradient** = exact JAX autodiff gradients, flexible parametric models;
**torch/irl** = flexible neural / spline / value-shaped resistance and full Bayesian posteriors.

## The circuit-solver implementations

The circuit-theory connectivity math (resistance → conductance via harmonic mean →
Laplacian solve → current density) is implemented in two places:

1. **`inst/python/diffiscape_jax/`** (modules: `core.py`, `resistance.py`, `optimize.py`, `sample.py`)
   — pure-JAX with `jax.vmap`, `jax.grad` for autodiff. Used by both `surrogate` and `gradient`
   solvers. Includes cumulative current (global and windowed Omniscape-style).

2. **`inst/python/diff_cs/`**
   — PyTorch implementations. Used by `torch` / `irl` solvers. Includes custom autograd
   `Function`s wrapping the circuit solve for backpropagation.

### JAX backend (`inst/python/diffiscape_jax/`)

The JAX backend is organized into clean, reusable modules:

| Module | Responsibility |
|--------|---|
| `core.py` | Circuit solver (global and windowed), cumulative current computation, resistance ↔ conductance |
| `resistance.py` | Resistance nets (parametric linear basis, shallow MLPs for `gradient` mode) and link functions |
| `optimize.py` | Surrogate optimizer (LHS + GP emulation + Thompson/EI), gradient descent (Adam/L-BFGS) entry points |
| `sample.py` | Bayesian sampling (MALA, NUTS via JAXScape/Flax, ADVI via Numpyro) |
| `window.py` | Omniscape-style windowed cumulative current computation |

### PyTorch backend (`inst/python/diff_cs/`)

`05_torch_pipeline.py` is the module loaded by reticulate
(`R/torch_bridge.R` imports it by exact path; `R` resolves entry points off it
by string name via `ds_torch_call()`). It is a **thin re-export shim** — the
implementation lives in the sibling `torch_pipeline/` package, organized into:

| Module | Responsibility |
|--------|---|
| `torch_pipeline/constants.py` | Shared numerical defaults (`DEFAULT_R_MIN/MAX/CLAMP_BETA/CG_TOL`) and GPU/cupy availability flags. |
| `torch_pipeline/_module_loaders.py` | Lazy file-path imports of `03_circuit_solver.py` / `04_diff_omniscape.py` (siblings of `05_torch_pipeline.py`, one directory up from this package). |
| `torch_pipeline/autograd_functions.py` | Cupy/torch DLPack bridges, adjoint-gradient math, and the custom `torch.autograd.Function` subclasses wrapping the circuit solves (`_CircuitSolveFn`, `_AbsorptionCircuitSolveFn`, `_DiffOmniscapeSolveFn`) plus the diff_omniscape bilinear interpolation/adjoint helpers. |
| `torch_pipeline/resistance_nets.py` | Resistance nets — `ResistanceNet` (MLP), `ConvResistanceNet`, `IRLResistanceNet`, `SplineResistanceNet` (P-spline GAM, with its B-spline basis helpers) — plus the shared `_ppp_loglik` Poisson log-likelihood. |
| `torch_pipeline/verify_gradients.py` | Finite-difference gradient checks (`verify_circuit_gradient`, `verify_conv_gradient`, `verify_softrl_gradient`, `verify_spline_gradient`). |
| `torch_pipeline/optimization.py` | `run_torch_optimization` — MAP fitting via Adam. |
| `torch_pipeline/samplers/common.py` | Shared Bayesian-sampler setup: `_setup_sampling_state` (checkpoint loading, network reconstruction, parameter collection) and `_compute_ess_chain` (effective sample size), used by all three samplers below. |
| `torch_pipeline/samplers/langevin.py`, `hmc.py`, `advi.py` | `run_langevin_sampling` (MALA/ULA), `run_hmc_sampling` (NUTS), `run_advi`. |

`05_torch_pipeline.py` itself stays at its original path/name (R and the
Python test suite both load it there) and re-exports every top-level name the
pre-split monolith had, so neither R nor existing tests needed to change.

## Data flow (end to end)

1. **Load** GPS + rasters → `ds_load_data`, `ds_create_basis`.
2. **Optimize** resistance params → `ds_optimize(solver=...)` → backend computes
   connectivity (and gradients for JAX/torch).
3. **Fit intensity** at the MAP params → `ds_fit_intensity` (`resolve_family`).
4. **Predict / diagnose / posterior** → `ds_predict`, `ds_diagnose`, `ds_posterior`.

## Where things are tested

- R: `tests/testthat/` (run with `devtools::test()`), coverage flag `r`.
- Python JAX: `inst/python/diffiscape_jax/tests/` (run with `pytest`), coverage flag `jax`.
- Python PyTorch: `inst/python/tests/` (run with `pytest`), coverage flag `torch`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and how to run each suite.
