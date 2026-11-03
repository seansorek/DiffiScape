# JAX migration evidence

This is the implementation record for [the migration plan](JAX_MIGRATION_PLAN.md).
Evidence below is from November 3, 2026 on native Windows, Python 3.12.10, CPU.
An unchecked gate is not a supported replacement or a completed phase.

Scope update: the concurrent simplification PR is [#136](https://github.com/seansorek/DiffiScape/pull/136),
head `222b38a45d53332af6cb019bb71d14b11e0c0079`. Its working-tree changes were
preserved, including IRL/RF retirement, likelihood/Laplace fixes, unsupported-input
guards and automatic-merge removal. IRL has not been restored in JAX. PR #136
remains open; this integration does not merge or publish it. The working branch
is `codex/jax-migration`, based directly on that PR head. All 203 present working
files were SHA-256 checked before and after changing the branch base and remained
byte-for-byte identical; `main` and unrelated edits were preserved.
Torch sampler corrections are superseded by NumPyro; the best-parameter snapshot
and single-covariate behavior are covered in the replacement tests.

## Phase 0: scope and feature matrix

The initial working tree had edits to `.gitignore` and `.Rbuildignore`, plus the
untracked migration plan. Preserve all three. No applicable `AGENTS.md` was found
in the checkout or its ancestors. R 4.6.2 and the project Python environment are
available; neither is on the ordinary executable search path.

The **baseline** public R API is grouped below; utility and presentation functions remain R
adapters. Scientific rows must migrate before their existing implementation can
be retired.

| Entry points (including documented examples) | Resistance / intensity | Connectivity | Method and current owner | Replacement status |
| --- | --- | --- | --- | --- |
| `diffiscape`, `ds_optimize`, `run_torch_pipeline`, `default_torch_config` | Default log-linear; MLP, convolution, B-spline GAM, IRL; PPP `alpha + gamma * log1p(C)` (optional intensity network/spline) | Global absorption (default), fixed-zero boundary (`global`), focal moving windows (`diff_omniscape`) | Adam, Torch and SciPy/PyAMG; custom circuit adjoints | Required; keep reference |
| `optimize_resistance_gradient`, `solver="gradient"`, deprecated `"enzyme"` | Parametric exp/softplus/identity; Flax MLP, convolution, RBF called `spline_gam`, IRL; covariate-free PPP | Sum of moving-window effective resistance distances | JAXopt L-BFGS-B or Optax Adam, JAX | Existing behavior; not equivalent to default Torch |
| `optimize_resistance`, `evaluate_full_model`, `default_optimizer_config`, `solver="surrogate"` | Parametric resistance links; built-in intensity families, local covariates, residualization; R callbacks | Existing JAX window resistance-distance surface | R LHS, DiceKriging GP / ranger RF, marginal Thompson sampling / expected improvement; nested R intensity optimization | Shared built-ins required; callbacks R-only |
| `run_cumulative_current`, `ds_jax_connectivity`, `ds_jax_call` | Resistance or permeability arrays | JAX moving-window resistance distances, despite `current` names | JAXScape dense pseudoinverse by default | Keep meaning visible; not physical current |
| `ds_jax_setup`, `ds_jax_check`, `ds_install_jax_deps` | Backend/environment management | None | reticulate bundled file imports | Replace with lazy installed-package import after solver gate |
| `ds_torch_setup`, `ds_torch_check`, `ds_torch_call`, `ds_install_torch_deps` | Backend/environment management | None | reticulate bundled Torch files | Retire after replacement checks |
| `create_resistance_surface`, `quick_resistance`, `resistance_model`, `irl_resistance_model`, `predict/print/summary.resistance_model`, `resistance_link`, `link_exp/power/softplus/identity`, `params_vector_to_list`, `get_default_bounds` | Parametric links / custom R functions / IRL | None | R, with separate Python neural models | Shared built-ins required; custom R functions R-only |
| `intensity_family`, `family_negbin/poisson/gaussian/zinb/rsf/rsp/clogit`, `fit_intensity_nb`, `fit_intensity_selection`, `compute_intensity`, `predict_intensity`, `standardise_connectivity`, `residualise_connectivity`, `default_intensity_config` | Weighted PPP; NB count-mixture adjustment; response-scale Gaussian; cell-count ZINB; RSF/RSP normalized selection; stratum-normalized clogit | Supplied connectivity | R bounded likelihood optimization | Shared built-ins required; preserve weighting and transforms |
| `fit_intensity_gam`, `ds_fit_intensity` | `mgcv` spatial GAM or selected built-in family | Supplied/recomputed connectivity | R `mgcv` / nested likelihood fit | `mgcv` stays R-only; Torch/IRL joint fits explicitly rejected by `ds_fit_intensity` |
| `fit_surrogate`, `predict_surrogate`, `loo_cv_surrogate` | GP / RF objective surrogate | None | DiceKriging / ranger | Shared surrogate required if retained |
| `laplace_resistance`, `posterior_sample`, `posterior_summary`, `ds_posterior`, `plot_posterior` | Resistance/intensity Laplace approximation and draws | Recomputed circuit surfaces or surrogate curvature | R numerical Hessian, eigenvalue repair, optional intensity refitting | Shared numerical parts required; plotting R |
| `run_bayesian_sampling`, `run_bayesian_sampling_hmc`, `run_advi` | Torch fitted network, priors and PPP | Global absorption / fixed boundary | Custom MALA/ULA, NUTS/HMC and variational algorithms in Torch | Replace with checked NumPyro density; keep reference |
| `ds_jax_sample_nuts`, `ds_jax_sample_advi` | Flax network priors and PPP | JAX moving-window distances | NumPyro NUTS / AutoNormal | Existing behavior; density differs from Torch |
| `compute_deviance_residuals`, `compute_deviance_residuals_gam`, `compute_information_criteria`, `diagnose_model`, `ds_diagnose`, `ds_ppc`, `resistance_sensitivity` | Family residuals, information criteria, posterior prediction, sensitivity | Fit-dependent | R numerical calculations; some joint-fit paths skip diagnostics/posteriors | Shared calculations required; report unsupported outputs explicitly |
| `verify_torch/conv/spline/irl_gradient` | Torch model-specific finite differences | Selected Torch circuit operator | Python gradient checks called from R | Keep until replacements pass |
| `ds_load_data`, `as_coord_matrix`, `extract_raster_values`, `extract_connectivity`, `ds_create_basis`, `create_basis_stack`, `validate_basis_stack`, `basis_summary`, `check_basis_correlations`, `ds_predict`, `rasterise_deviance_residuals`, `moran_test`, `plot_deviance_residuals`, `plot_qq_deviance`, `plot_residual_map`, `plot_ppc` | Spatial preparation, extraction, S3 and presentation | Adapter only, except delegated predictions | R terra, plotting, spdep | Retain spatial adapters |

Examples in README and `gradient-jax`, `surrogate-omniscape`, `gam-profile`,
`torch-mlp`, `spline-gam`, and `irl` vignettes exercise these distinct paths.
`gam-profile` currently asks the restricted gradient path to fit a GAM; this is
not evidence that the combination works. Architecture documentation also has
stale defaults/module descriptions (there is no `R/jax_pipeline.R`).

## Numerical and language contract

Framework, formulation, resistance model, likelihood, and optimization method
are separate choices. IRL is a resistance model. Surrogate Bayesian optimization
is an optimization method. Existing solver strings conflate these dimensions;
do not reinterpret them until the same formulation is validated.

- Array axes are `(row, column, covariate)`; cells flatten in NumPy C order,
  `index = row * n_columns + column`, matching terra's cell numbering minus one.
  Valid-cell matrices retain this order, with shape `(n_valid, n_covariates)`;
  masks are full-grid Boolean arrays, never silently recycled. R matrix storage
  order must not be used as the cell order. Test a nonsquare grid.
- Scientific arrays and solver calculations use float64 explicitly. Public
  boundaries accept NumPy/array-like values and validate shape/finiteness.
  JAX arrays stay in JAX inside differentiated functions. CRS, extent and raster
  resolution metadata stay in the adapters.
- Missing covariates define invalid cells. The legacy JAX barrier is **finite**:
  `R=5000` or `P=1/5000`, not node deletion or exact zero conductance. Its
  arithmetic edge mean can still conduct strongly across a barrier beside a
  permeable cell. Preserve this legacy formulation explicitly; do not claim it
  implements the Torch harmonic barrier convention.
- Observation counts are nonnegative finite counts for valid cells; multiple
  points in a cell add. Points outside the grid/valid mask are excluded by the
  adapter and must be counted/reported. Cell area is positive, in the projected
  coordinate system's squared units. PPP observation terms use counts; the
  integral uses area times intensity over valid cells. Do not multiply the
  observation log term by cell area or add count factorials to the PPP density.
- Available-location arrays have their own weights and, for conditional logit,
  explicit used/available stratum IDs. They cannot be replaced by raster area
  quadrature. Preserve aggregate NB/ZINB mixtures and response-scale Gaussian
  semantics separately. R intensity fitting standardizes `log1p(C/c_scale)`
  using observed locations (sample SD); joint JAX/Torch PPP uses `log1p(C)`.
  Store the fitted transform and coefficients for prediction.
- Torch's 4-neighbor undirected conductance is `2/(R_i+R_j)`, equivalently the
  harmonic mean of permeabilities. Legacy JAX uses `(P_i+P_j)/2`. Neither has
  diagonal edges. Laplacian is `diag(A @ 1) - A`.
- Global absorption solves `(L + a I) v = b`, `a > 0`. Sources are a regular
  lattice with configured spacing, optionally weighted by `P`. Unit sources at
  **every** cell give constant `v=1/a`; raw physical edge current then vanishes.
  Such a case alone cannot verify landscape sensitivity.
- Torch fixed-boundary `global` solves the interior submatrix with perimeter
  `v=0` and sources only in the interior. Legacy JAX `circuit_solve` instead
  solves `(L + a I_boundary) v = b_interior` (soft/Robin boundary absorption).
  These are different boundary conditions, not numerical tolerances.
- Physical edge current is `A_ij * abs(v_i-v_j)`. Legacy global implementations
  smooth absolute value as `sqrt(dv**2 + 1e-8)`; this creates nonzero current at
  zero voltage drop. Torch node current sums all outgoing directed edges;
  legacy JAX multiplies this sum by one half. Retain and label these factors
  in reference tests rather than hiding the discrepancy in normalization.
- Legacy JAX windows edge-pad by `radius`, add divisibility padding, place one
  unit source at each square window center, and sum **effective resistance**
  maps across whole overlapping windows before cropping. Effective resistance
  is `(e_s-e_t)^T L^+ (e_s-e_t)` for each target; it is neither voltage nor
  physical current. `source_from_resistance=False` and voltage output are
  rejected. Torch focal Omniscape uses local source/sink solves and weighting;
  the two window operators must remain distinct.
- Tiny-grid oracles use independent NumPy dense Laplacians/solves/pseudoinverses
  only in tests. Test values, finite-difference gradients, changed-matrix cached
  state, barriers, and nonuniform sources. Dense reference code must never
  become a production fallback.

## Validation record

Initial environment: JAX/JAXlib 0.12.0.1, JAXScape 0.0.9, AMJax 0.0.3,
Lineax 0.1.1, Flax 0.12.12, Optax 0.2.8, NumPyro 0.22.0, JAXopt 0.8.5,
NumPy 2.5.4, SciPy 1.18.1, Torch 2.15.0+cpu, pytest 9.2.0.

JAXScape 0.0.15 was downloaded into `.local/jaxscape-0.0.15` without replacing
the initial environment. Its wheel imports successfully with the installed
stack. Import success alone does not pass Phase 1.

Upstream sources checked:

- [JAXScape 0.0.15 release](https://github.com/vboussange/jaxscape/releases/tag/v0.0.15):
  Python >=3.12, JAX >=0.11, public `init`/`reinit` solver state lifecycle.
- [Solver-state release PR](https://github.com/vboussange/jaxscape/pull/50):
  stale-matrix fixes, gradient failures on JAX 0.9–0.10, sparse window limitations.
- [JAX changelog](https://docs.jax.dev/en/latest/changelog.html) and
  [platform installation](https://docs.jax.dev/en/latest/installation.html).
  Native Windows CPU evidence is not CUDA validation.

## Implemented replacements and gates

| Retained feature | Final owner / behavior | Evidence |
| --- | --- | --- |
| Physical loglinear, MLP, Conv, penalized B-spline resistance | Flax; harmonic global absorption, fixed-zero boundary, focal current | Four frozen Torch model fixtures agree to 1e-11; dense forward/gradient oracles |
| Legacy window parametric, MLP, Conv, RBF | Existing JAXScape effective resistance; explicit `window_resistance` | Original scientific tests moved, saved-state API checks, R/Python mask ordering |
| Adam / bounded L-BFGS-B | Optax / SciPy consuming JAX value-and-grad | Objective improvement, bounds and truthful convergence tests |
| Poisson, NB, ZINB, Gaussian, RSF/RSP/clogit | Python built-in densities, weighting, strata, MLE, Hessian | Original R densities to 1e-12, finite differences, zero-count mixture check |
| Connectivity transform, residualization, intensity prediction | Python; fitted sample SD and least-squares coefficients stored | Cross-language existing fixtures and save/reload predictions |
| GP surrogate BO and leave-one-out validation | SMT ordinary Matern 5/2 kriging; SciPy Latin hypercube; TS/EI semantics retained | Fixed R kriging means within 1e-14, SD within 7e-9 at matched fitted kernel settings; failure/bounds/acquisition and LOO tests |
| Physical NUTS / ADVI | NumPyro; shared model, mask, penalty and proper priors | Exact tiny-case log density; real NUTS/ADVI draws and diagnostics |
| Summaries, criteria, residuals, count posterior prediction | Python; R handles raster/plot output | Numerical fixtures, aggregate-mixture variance test, physical R PPC |
| mgcv, arbitrary R callbacks, profile-objective Laplace refits, Moran statistics | Explicit R-only extensions | Existing R regressions; CPU vignette smoke checks |
| IRL / RF | Retired by concurrent simplification | Rejection tests; no implementations restored |

Phase 0: baseline R suite passed with three skips and one existing collinearity
warning; 19 targeted original JAX core tests passed. The initial full JAX baseline
was interrupted and is not claimed as evidence. The matrix and numerical
contract above record discrepancies before replacement.

Phase 1: JAX/JAXlib 0.12.0.1, JAXScape 0.0.15, AMJax 0.0.3 and Lineax 0.1.1
pass independent dense tests for global, boundary, and window formulations,
finite differences, changed-matrix state reuse, and multilevel forward/backward
solves. Public AMJax `init`/`reinit` replaces the old monkey patch and private
imports. Physical current sums outgoing edges with the legacy smoothed absolute
value; it is not halved. Physical invalid cells remain finite R=1e6 barriers.

The focal-current replacement uses conservation: with exactly one ground in a
local window, incoming current at that ground equals the sum of all other source
strengths. SciPy/NumPy dense circuit oracles verify this identity. Window sums
and the original focal lattice/bilinear interpolation preserve this quantity
without maintaining a second circuit solve or an adjoint.

Phases 2–3: source lives only under `python/src/diffiscape`. A wheel and sdist
were built and installed into `.local/migration-venv`; neither Torch nor JAXopt
is installed there. An earlier full installed-wheel suite passed 125 tests before
the final physical additions. Tests run outside the repository to prevent hidden
source imports. R imports the installed version lazily, constructs no Flax
objects, and shares one valid-cell array preparation helper. Independent R/Python
fits and predictions agree on a masked 2x3 grid at 1e-9; R save/reload and the
physical fit/predict/posterior/diagnostics/PPC workflow pass.

Phase 4: all retained shared features in the table have Python-only operations.
SciPy BSpline supplies values and derivatives; a small custom-JVP callback makes
those maintained derivatives available to JAX JIT/AD. It is not an in-house basis
recurrence. Intensity spline JIT/Hessian and end-to-end gradient checks pass.
The approved proper priors intentionally correct an improper legacy posterior:
free model parameters, alpha and sampled log-smoothing parameters use Normal(0,2);
raw gamma uses Normal(0.54132485,1) followed by softplus. Existing penalties
remain. Smoothing is fixed by default, optional to sample. Tiny-case density tests
include the priors, penalties and PPP and verify finite tail behavior.

On a 3x4 physical loglinear case, 150 warmup + 200 retained NUTS draws with target
acceptance .9 produced **0 divergences**, minimum ESS **85.76**, maximum split
R-hat **1.0113**, and nonconstant finite draws. This is a small numerical gate,
not a convergence claim for users' landscapes. End-to-end objective gradient
relative errors: loglinear 1.38e-7, Conv 6.52e-10, intensity B-spline 1.42e-10.

NB/ZINB predictive counts now use a shared Gamma rate per replicate (and a shared
structural-zero event for ZINB), matching the fitted aggregate PPP mixture.
The old independent per-cell draws were inconsistent with that density; this is
an explicit correction. Marginal count residual definitions remain unchanged.

Phase 5: Torch implementation, custom adjoints/MCMC, CuPy/AMGX paths, requirements,
and CI jobs are removed after replacement checks. JAXopt is removed. Deprecated
R fitting names translate physical settings; direct Torch calls, old .pt files,
checkpoint UQ and absorption curricula give actionable migration errors. Stochastic
focal subsampling and the old unused weight-decay option are retired with messages.
Old global-absorption calls retain their enforced source_spacing=1. New physical
calls may explicitly choose another spacing. Raw DiceKriging objects require refitting.
The guides and vignettes now use the installed package; obsolete executable
comparison notebooks were replaced by a synchronized matched CPU benchmark.

Phase 6 final wheel/sdist, complete R suite, R CMD check and vignette gate results
are recorded below. CPU dependency versions are fixed in
`python/constraints-cpu.txt`. New upstream versions and GPU execution are separate
compatibility checks and have not been validated here. Python must be published
before an R release requiring its API; no release has been published by this task.

## Final validation — all implementation gates passed

- Runtime R/Python source totals **17,947 → 10,223 lines**, a net reduction of
  **7,724 lines relative to PR #136**. This excludes tests, external documentation,
  benchmarks and dependencies. Existing custom R-link sensitivity and profile
  callback extensions remain R-only; shared GP diagnostics now run in Python.
- **Installed wheel:** `python -m pytest <absolute path>/python/tests -vv`
  from outside the repository: **148 passed**, 283.13 seconds. Seven warnings:
  six existing pytest class-fixture deprecations (pytest is pinned below 10),
  and one ESS divide-by-zero warning in the intentionally four-draw sampler
  smoke fixture. The longer diagnostic gate above has finite ESS and no divergences.
- **Source distribution:** built sdist installed into a separate package directory;
  its eight public API tests passed from outside the repository.
  Wheel and sdist contain one runtime source copy and the sdist includes JSON
  scientific fixtures. Neither environment requires R, Torch, or JAXopt.
- **R source/installed-core suite:** passed. The final independently built R
  source package was checked with `R CMD check --no-manual`: **Status: OK**,
  **0 errors, 0 warnings, 0 notes**. Its testthat result was **946 passes,
  0 failures, 0 warnings, 2 skips**. One skip tests a pre-initialization error
  after Python was already initialized; the other checks source vignette text,
  absent from the installed test layout. Vignette checks themselves passed.
- **Surrogate LOO:** complete refitting and numerical summaries now live in Python,
  including failed-fold accounting and exclusion of failed objective evaluations.
  Python tests cover injected refit failures and exact returned statistics;
  R tests cover delegation and configuration transfer.
- **Legacy R inference:** formerly unconditional NUTS/ADVI skips were replaced
  by real tiny-grid integration tests. Both wrappers also passed consecutively
  in a manual smoke run, including artifact serialization. This caught and fixed
  the Python `sample` module/export collision: the internal module is now named
  `sampling.py`, leaving the public `sample()` callable intact across calls.
- **R neural adapter:** a real masked Conv fit, its seed, saved-state prediction,
  and recomputed PPP likelihood pass, in addition to the physical integration gate.
- **Documentation:** all five synthetic CPU vignette workflows were executed,
  including the R-only mgcv extension. All HTML vignettes built and were rebuilt
  successfully by R CMD check. Generated NAMESPACE and help pages match R source.
- **Benchmark:** `benchmarks/bench_full.py --sizes 5` passed synchronized physical
  global/boundary forward and gradient smoke checks at equal float64/tolerance,
  source lattice, weighting, and state reuse. No cross-framework speed claim is made.
- **Retirement:** searches of runtime source, CI and benchmark commands find no
  Torch, CuPy, AMGX, JAXopt or repository-relative Python import requirements.
  Remaining names are deprecated API names, rejection tests, frozen reference
  fixtures, and migration/history documentation.

The user's `.Renviron` places TMPDIR inside this checkout. R CMD build copies
the source tree before applying excludes, which caused a recursive temporary
directory copy; it was stopped and its task-created scratch directory removed.
The successful R build used a clean staged source tree under `.local/r-source`
with only DESCRIPTION, NAMESPACE, R, man, tests and vignettes, and an external
temporary directory (`R_ENVIRON_USER=NULL`). The existing machine configuration
was preserved. Generated test outputs are now excluded by `.Rbuildignore`.
Pandoc was installed as a local build tool for HTML vignettes, outside the Python
runtime environment. **PDF manual generation was not checked** because LaTeX is
not installed. GPU execution and newer upstream compatibility remain untested;
they are separate gates from this completed CPU migration.
