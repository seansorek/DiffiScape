# DiffiScape Architecture

DiffiScape is an **R package that orchestrates two heavy, optional compute
backends** (Julia and Python) to fit landscape *resistance* surfaces against
animal-movement data. R is the user-facing glue layer; the numerically intensive
circuit-theory solves and their gradients happen in Julia or Python and are
called back into R.

This document maps the moving parts so a new contributor knows **which code runs
for which solver**, and why the same circuit math appears in three places.

```
                ┌──────────────────────────────────────────────┐
   user data →  │  R layer  (R/)                                │
  rasters+GPS   │  basis_functions → ds_optimize → intensity →  │
                │  posterior / diagnostics / predict            │
                └───────┬───────────────┬───────────────┬───────┘
            solver=      │ "surrogate"    │ "enzyme"       │ "torch" / "irl"
                         ▼                ▼                ▼
                 Omniscape.jl /    pure-Julia diff.   PyTorch diff.
                 Circuitscape.jl   solver + Enzyme    circuit solver
                 (JuliaConnectoR)  (JuliaConnectoR)   (reticulate)
```

## The R layer (`R/`)

| File | Responsibility |
|------|----------------|
| `pipeline.R` | One-call `diffiscape()` + modular steps (`ds_load_data`, `ds_create_basis`, `ds_optimize`, `ds_fit_intensity`, `ds_predict`, `ds_posterior`, `ds_diagnose`). `ds_optimize` is the **backend dispatcher**. |
| `basis_functions.R` | Build/validate the basis-function stack from environmental rasters. |
| `optimizer.R` | `optimize_resistance` (surrogate/GP) and `optimize_resistance_enzyme` (Julia gradient) + the outer objective and surrogate machinery. |
| `connectivity.R` | R wrappers over the Julia connectivity entry points (`run_omniscape`, `run_circuitscape`, `run_cumulative_current`). |
| `intensity.R`, `intensity_family.R` | Likelihood/intensity models: NB, Poisson, GAM, plus selection families (RSF, RSP, clogit). `resolve_family()` picks one. |
| `resistance.R`, `resistance_link.R` | Resistance model objects and the swappable link functions (`link_exp`, `link_power`, `link_softplus`, `link_identity`). |
| `posterior.R`, `diagnostics.R` | Laplace/posterior summaries and model diagnostics. |
| `julia_bridge.R`, `torch_bridge.R` | Thin call interfaces to the two backends (`ds_julia_*`, `ds_torch_*`). |
| `torch_pipeline.R` | R-side wrapper that marshals arguments into the Python torch pipeline. |
| `utils.R`, `zzz.R` | Helpers, package `.onLoad`, internal env (`.ds_env`). |

## Backend selection — `ds_optimize(..., solver=)`

`ds_optimize` ([R/pipeline.R](R/pipeline.R)) routes to one of four solvers:

| `solver` | Entry point | Backend | Gradient strategy |
|----------|-------------|---------|-------------------|
| `"surrogate"` (default) | `optimize_resistance` | **Julia** — Omniscape.jl/Circuitscape.jl as a black box | None; Bayesian optimization over a DiceKriging GP surrogate (LHS init + Thompson sampling / expected improvement) |
| `"enzyme"` | `optimize_resistance_enzyme` | **Julia** — pure-Julia differentiable solver | Exact adjoint gradients via Enzyme.jl → L-BFGS |
| `"torch"` | `run_torch_pipeline` | **Python** — PyTorch differentiable circuit solver | torch autograd (custom `Function`s) → Adam / Bayesian samplers |
| `"irl"` | `run_torch_pipeline` (`model_type="irl"`) | **Python** | Same as `torch`, with the value-iteration (IRL) resistance net |

Rule of thumb: **surrogate** = robust, derivative-free, slow; **enzyme** = fast
exact gradients for parametric resistance; **torch/irl** = flexible neural /
spline / value-shaped resistance and full Bayesian posteriors.

## The three circuit-solver implementations

The same circuit-theory connectivity math (resistance → conductance via harmonic
mean → Laplacian solve → current density) is implemented **three times** because
each backend needs different differentiation support. This is deliberate, but it
means a change to the connectivity model may need to land in more than one place.

1. **Omniscape.jl / Circuitscape.jl** (external Julia packages) — used by the
   `surrogate` solver via `R/connectivity.R`. No gradients (hence the surrogate).
2. **`inst/julia/DiffiScape/src/differentiable_solver.jl`** (+ `enzyme_gradients.jl`)
   — pure-Julia, Enzyme-compatible. Used by the `enzyme` solver. Global solve +
   moving-window cumulative current.
   - ⚠️ Known gap: the Enzyme gradient path **hardcodes the `exp` link**
     (`enzyme_gradients.jl`); a non-`exp` `resistance_link` chosen in R is
     silently ignored on this backend. See the tech-debt backlog.
3. **`inst/python/diff_cs/03_circuit_solver.py`** (global) and
   **`04_diff_omniscape.py`** (Omniscape-style windowed) — used by the
   `torch`/`irl` solver, wrapped in torch autograd `Function`s inside
   `05_torch_pipeline.py`.

### Python torch backend internals (`inst/python/diff_cs/`)

`05_torch_pipeline.py` is the large entry-point module loaded by reticulate
(`R/torch_bridge.R` imports it; it lazily loads `03`/`04` by file path). It
contains:

- Autograd `Function`s wrapping the circuit solves (`_CircuitSolveFn`,
  `_AbsorptionCircuitSolveFn`, `_DiffOmniscapeSolveFn`).
- Resistance nets: `ResistanceNet` (MLP), `ConvResistanceNet`, `IRLResistanceNet`,
  `SplineResistanceNet` (GAM-style).
- Entry points: `run_torch_optimization` (MAP), `run_langevin_sampling` (MALA),
  `run_hmc_sampling` (NUTS), `run_advi`, and `verify_*_gradient` finite-difference
  checks.
- Shared numerical defaults live in the `DEFAULT_*` constants block near the top
  (resistance clamp range, CG tolerance).

> Note: the `03_`/`04_`/`05_` filename prefixes are historical pipeline-stage
> numbers. Because Python modules can't start with a digit, `05` imports `03`/`04`
> via `importlib.util.spec_from_file_location`. Cleaning this up is on the
> backlog (Phase 2).

## Data flow (end to end)

1. **Load** GPS + rasters → `ds_load_data`, `ds_create_basis`.
2. **Optimize** resistance params → `ds_optimize(solver=...)` → backend computes
   connectivity (and gradients for enzyme/torch).
3. **Fit intensity** at the MAP params → `ds_fit_intensity` (`resolve_family`).
4. **Predict / diagnose / posterior** → `ds_predict`, `ds_diagnose`, `ds_posterior`.

## Where things are tested

- R: `tests/testthat/` (run with `devtools::test()`), coverage flag `r`.
- Python: `inst/python/tests/` (run with `pytest`), coverage flag `python`.
- Julia: `inst/julia/DiffiScape/test/` (run with `Pkg.test()`), coverage flag `julia`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and how to run each suite.
