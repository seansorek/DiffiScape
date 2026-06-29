# DiffiScape: Julia → JAX Backend Migration

**Date**: 2026-06-25
**Status**: Approved
**Goal**: Replace all Julia backends (Omniscape.jl, Circuitscape.jl, Enzyme.jl) and the PyTorch backend (`inst/python/diff_cs/`) with a single JAX-based backend built on JAXScape. R remains the user-facing layer. No changes to the statistical modeling (intensity families, posterior, diagnostics).

---

## Capability Mapping

| Current component | JAXScape replacement |
|---|---|
| `Omniscape.jl` / `Circuitscape.jl` (surrogate solver) | `GridGraph` + `ResistanceDistance` forward-only |
| `differentiable_solver.jl` + `enzyme_gradients.jl` | `GridGraph` + `ResistanceDistance` + `jax.value_and_grad` |
| `03_circuit_solver.py` (torch global solve) | `GridGraph` + `ResistanceDistance` |
| `04_diff_omniscape.py` (torch windowed solve) | `WindowOperation` |
| `05_torch_pipeline.py` autograd `Function`s | Eliminated — `jax.grad` natively differentiates through JAXScape |
| PyAMG-preconditioned CG | `PyAMGSolver` or `AMJaxCGSolver` (JAXScape built-in) |
| CuPy GPU path | JAX GPU backend natively |

JAXScape operates in **permeability** space (high = easy to cross). Current DiffiScape operates in **resistance** space. The bridge handles conversion.

---

## Architecture

```
              ┌──────────────────────────────────────────────┐
 user data →  │  R layer  (R/) — UNCHANGED                    │
 rasters+GPS  │  basis_functions → ds_optimize → intensity →  │
              │  posterior / diagnostics / predict             │
              └───────┬───────────────┬────────────────────────┘
          solver=      │ "surrogate"    │ "gradient" (replaces "enzyme"+"torch"+"irl")
                       ▼                ▼
               ┌─────────────────────────────────────┐
               │  jax_bridge.R  (replaces julia_bridge│
               │  AND torch_bridge.R)                 │
               │  ds_jax_setup()                      │
               │  ds_jax_connectivity()               │
               │  ds_jax_optimize()                   │
               │  ds_jax_sample()                     │
               └───────┬─────────────────────────────┘
                       │  reticulate
                       ▼
               ┌─────────────────────────────────────┐
               │  inst/python/diffiscape_jax/         │
               │                                      │
               │  core.py        — GridGraph setup,   │
               │                   R↔permeability,    │
               │                   forward solve      │
               │  resistance.py  — Flax resistance    │
               │                   nets (MLP, Conv,   │
               │                   Spline, IRL)       │
               │  optimize.py    — L-BFGS / Adam      │
               │  sample.py      — NumPyro NUTS /     │
               │                   MALA / ADVI        │
               │  window.py      — WindowOperation    │
               │                   wrapper            │
               └─────────────────────────────────────┘
                       │
                       ▼
               JAXScape  +  JAX  +  Flax  +  NumPyro
```

### R-side changes

| File | Change |
|---|---|
| `julia_bridge.R` | **Deleted** |
| `torch_bridge.R` | **Deleted** |
| `torch_pipeline.R` | **Deleted** |
| `jax_bridge.R` | **New** — thin reticulate interface |
| `connectivity.R` | Rewire `run_omniscape` / `run_circuitscape` / `run_cumulative_current` to call `jax_bridge` |
| `optimizer.R` | `optimize_resistance_enzyme` → `optimize_resistance_gradient` calling JAX |
| `pipeline.R` | `solver="enzyme"` and `solver="torch"` collapse to `solver="gradient"` with deprecation aliases |
| `zzz.R` | Remove Julia init from `.onLoad`, add JAX lazy init |
| `DESCRIPTION` | Drop `JuliaConnectoR`, keep `reticulate`, document Python deps |

### Deleted entirely

- `inst/julia/` (all Julia code)
- `inst/python/diff_cs/` (all torch code)
- `inst/python/tests/` (replaced by new test suite)

### Preserved unchanged

- `intensity.R`, `intensity_family.R`, `resistance.R`, `resistance_link.R`, `basis_functions.R`, `posterior.R`, `diagnostics.R`
- GP/EI surrogate machinery in `optimizer.R`
- All R tests that don't directly call Julia or torch

---

## Permeability / Resistance Duality

Users can choose parameterization via config:

```r
config$parameterization <- "resistance"    # default, backward-compatible
config$parameterization <- "permeability"  # native to JAXScape
```

| Aspect | `"resistance"` | `"permeability"` |
|---|---|---|
| Basis function interpretation | Covariates → resistance via link fn | Covariates → permeability via link fn |
| Conversion in `core.py` | `permeability = 1/R` before JAXScape | Direct pass, no conversion |
| Resistance nets output | log-resistance, clamp `[R_MIN, R_MAX]` | log-permeability, clamp `[P_MIN, P_MAX]` |
| Gradient flow | Chains through `1/R` inversion | Direct |

Single conversion point in `core.py`:

```python
def prepare_permeability(surface, parameterization):
    if parameterization == "resistance":
        return 1.0 / jnp.clip(surface, R_MIN, R_MAX)
    else:
        return jnp.clip(surface, P_MIN, P_MAX)
```

---

## Python Backend Modules (`inst/python/diffiscape_jax/`)

### `core.py`

- `forward_solve(resistance_or_permeability, config)` → connectivity array
- `solve_with_grad(surface, params, basis, link_fn, config)` → (connectivity, grad_params) via `jax.value_and_grad`
- `prepare_permeability(surface, parameterization)` — single R↔permeability conversion point
- Edge weight function: `fun=lambda x, y: (x + y) / 2` (harmonic mean)
- Solver: `PyAMGSolver` default, `AMJaxCGSolver` for GPU

### `window.py`

- Wraps JAXScape `WindowOperation` for Omniscape-style moving-window cumulative current
- Configurable `radius`, `block_size`
- `jax.vmap` over windows for GPU parallelism

### `resistance.py`

Port of torch resistance nets to Flax:

| Torch class | Flax class |
|---|---|
| `ResistanceNet` | `ResistanceMLP` |
| `ConvResistanceNet` | `ResistanceConv` |
| `SplineResistanceNet` | `ResistanceSpline` |
| `IRLResistanceNet` | `ResistanceIRL` |

All output log-resistance or log-permeability (depending on parameterization) → softplus clamp.

### `optimize.py`

- Parametric: `jaxopt.LBFGS` (replaces Enzyme L-BFGS)
- Neural: `optax.adam` + cosine schedule + early stopping (replaces torch Adam)
- Objective: negative log-likelihood of PPP intensity
- Parametric path: intensity stays in R. Neural path: simple JAX log-intensity `log λ = α + γ · log(1 + C)`

### `sample.py`

| Torch sampler | NumPyro replacement |
|---|---|
| `run_langevin_sampling` | Custom MALA kernel or `numpyro.infer.SA` |
| `run_hmc_sampling` | `numpyro.infer.NUTS` |
| `run_advi` | `AutoLowRankMultivariateNormal` + `SVI` |

---

## Solver API (R-side)

After migration, `ds_optimize` supports two solvers:

| `solver` | Backend | Gradient strategy |
|---|---|---|
| `"surrogate"` (default) | JAXScape forward-only → GP/EI | None; Bayesian optimization |
| `"gradient"` | JAXScape + `jax.grad` | Exact autodiff → L-BFGS or Adam |

The `"gradient"` solver accepts `model_type` config:

- `"parametric"` — basis functions + link → resistance/permeability (replaces `"enzyme"`)
- `"mlp"`, `"conv"`, `"spline"`, `"irl"` — Flax net (replaces `"torch"` / `"irl"`)

Deprecation aliases: `solver="enzyme"` → `solver="gradient", model_type="parametric"`. `solver="torch"` → `solver="gradient", model_type="mlp"`.

---

## Migration Phases

### Phase 1: `core.py` + `window.py` — surrogate solver path

- Implement `forward_solve` and windowed cumulative current
- Write `jax_bridge.R` with `ds_jax_setup()`, `ds_jax_connectivity()`
- Rewire `connectivity.R`
- Validation: compare connectivity rasters against Julia output (correlation > 0.99)

### Phase 2: `optimize.py` — gradient solver for parametric resistance

- Implement `solve_with_grad` with `jax.value_and_grad`
- L-BFGS via `jaxopt`
- Wire into `optimizer.R` as `optimize_resistance_gradient`
- Validation: compare gradients against finite differences and Enzyme output

### Phase 3: `resistance.py` — Flax resistance nets

- Port four net architectures
- Adam/cosine via `optax`
- Validation: `jax.test_util.check_grads`, compare loss curves against torch

### Phase 4: `sample.py` — NumPyro Bayesian sampling

- NUTS, MALA, ADVI via NumPyro
- Validation: compare posterior summaries against torch sampler output

### Phase 5: Cleanup

- Delete `inst/julia/`, `inst/python/diff_cs/`
- Delete `julia_bridge.R`, `torch_bridge.R`, `torch_pipeline.R`
- Remove `JuliaConnectoR` from DESCRIPTION
- Add deprecation warnings for old solver names
- Update README, vignettes

### Test strategy

Each phase keeps old backends alive until replacement is validated. Old code deleted only after all R-level tests pass on JAX backends.

---

## Dependencies

**Added**: `jax`, `jaxscape`, `flax`, `optax`, `numpyro`, `jaxopt`

**Removed**: `torch`, `pyamg`, `cupy` (JAX handles GPU natively)

**Kept**: `numpy`, `scipy`, `reticulate`
