# DiffiScape migration: one JAX core, Python and R packages

Prepared November 3, 2026. This is an implementation handoff, not a record of a completed migration. No environments, numerical results, or upstream compatibility combinations were validated while writing this plan.

## Objective and decisions

Reduce the maintainer's numerical implementation burden while making DiffiScape usable directly from Python and from R.

- Keep **one repository and two independently installable packages**.
- Keep the R package at the repository root. Add a Python package under `python/`.
- Make Python the single implementation of shared scientific calculations; R depends on Python, and Python never requires R.
- Use **JAX, JAXScape, Lineax, AMJax/PyAMG, existing Flax Linen, Optax, and NumPyro** where appropriate. NumPy and SciPy remain normal numerical dependencies.
- Delegate sparse solving, preconditioning, differentiation through solves, and posterior sampling to maintained libraries. Avoid new custom solver kernels, adjoint implementations, or MCMC algorithms.
- Remove PyTorch and JAXopt from the final runtime. Keep Torch temporarily as a migration reference until the required behavior has a validated replacement.
- Retain `mgcv` fitting as an explicitly R-only extension initially. Reimplementing a GAM library is outside this migration.
- Preserve public R functions where practical. Deprecate backend-specific names with actionable messages; do not silently change the scientific model when translating an old call.
- Prioritize CPU support. GPU support uses the same Python core, with a separately tested installation option; do not require GPU tooling for ordinary installation.

The user values less code to maintain more than a small dependency count. Do not replace an upstream numerical library with an in-house implementation just to reduce dependencies.

## Current code and important discrepancies

Start by reading `ARCHITECTURE.md`, `DESCRIPTION`, `NAMESPACE`, `R/solver_spec.R`, `R/pipeline.R`, `R/jax_bridge.R`, `R/optimizer.R`, and both Python backends.

- `inst/python/diffiscape_jax/` contains JAX calculations but has no standalone `pyproject.toml` or complete public Python API.
- `R/jax_bridge.R` loads bundled files with `system.file()` / `import_from_path()` and constructs Flax models in R.
- `R/intensity_family.R`, `R/intensity.R`, `R/optimizer.R`, `R/posterior.R`, and `R/diagnostics.R` own scientific functionality not yet exposed by an independent Python package.
- The default Torch workflow is log-linear resistance plus **global absorption** and `log(lambda) = alpha + gamma * log1p(connectivity)`.
- Existing JAX gradient and neural fitting use a **moving-window** operator and a covariate-free Poisson point-process objective. They are not replacements for every existing family or the default Torch workflow.
- JAX `ResistanceSpline` uses Gaussian radial basis functions and ignores `degree`. Torch's spline implementation uses B-splines and penalties. Matching names do not imply matching models.
- The JAX window code accumulates `ResistanceDistance` outputs under current-related names. Audit whether each output represents effective resistance, voltage, or physical current before claiming equivalence with Torch or Omniscape.
- Some existing workflows intentionally reject options or omit diagnostics/posteriors. Record actual supported behavior rather than treating README claims as proof.
- Existing benchmark results are from June 2026 and use unequal tolerances and setup reuse. They are not an acceptance test for this migration.

At plan-writing time, `.gitignore` and `.Rbuildignore` had uncommitted changes from other work. Recheck the working tree and preserve all unrelated changes. Do not reset files or assume runtime installation work is complete.

## Target structure

Keep the initial extraction close to existing modules; split further only when a concrete responsibility requires it.

```text
DiffiScape/
  DESCRIPTION
  NAMESPACE
  R/                         # R input/output adapters, public functions, S3, plots
  man/
  tests/testthat/            # adapter and cross-language integration checks
  vignettes/
  python/
    pyproject.toml
    src/diffiscape/
      __init__.py            # small public API
      core.py                # existing connectivity logic, upstream solver integration
      window.py
      resistance.py
      likelihoods.py
      optimize.py
      sample.py
      diagnostics.py         # shared numerical diagnostics as they are migrated
    tests/                   # scientific and Python package tests
  docs/
  .github/workflows/
```

The module list is a responsibility guide, not a requirement to create empty files. Reuse existing names and functions where possible.

## Phase 0: baseline and scientific contract

1. Read applicable repository instructions; inspect git status and available runtimes. Use the current suitable checkout unless repository rules require otherwise.
2. Build a compact feature matrix: entry point, model, likelihood, connectivity definition, optimizer/inference method, current backend, and migration status. Include the exported R API and documented examples.
3. Separate the meanings of **backend**, **connectivity formulation**, **resistance model**, **likelihood**, and **optimization method**. IRL is a model; surrogate BO is an optimization method. Neither should select a different framework implicitly.
4. Record current tests and failures. Run small deterministic reference examples wherever the existing environment permits; document unavailable checks rather than inventing baselines.
5. Establish the intended connectivity quantity and physical conventions for global absorption, boundary absorption, and moving windows: conductance mapping, source placement/weighting, grounding, masking, current aggregation, and normalization.
6. Define one array contract shared by both languages: valid-cell ordering, full-grid versus valid-cell arrays, `(rows, columns, covariates)` conventions, float64, masks, missing values, observation counts, cell area, and available-location strata. Keep CRS/extent metadata in language-specific adapters.
7. Use tiny grids with independently checkable dense linear algebra as numerical oracles. Compare forward values and finite-difference gradients against the production solver. A dense reference used only in tests is not a new production solver.

**Gate:** a written feature matrix and numerical contract exist; discrepancies are visible. Do not delete a backend or substitute a scientific formulation to make tests pass.

## Phase 1: validate the upstream solver combination

Recheck current primary documentation and release notes. The versions below are research leads, not a known-good lockfile:

- JAX 0.12.0.1 was released October 23, 2026.
- JAXScape 0.0.15 was released October 30 and requires Python >=3.12 and JAX >=0.11. It revises reusable solver state and adds an optional cuDSS backend.
- Its release PR reports gradient problems on JAX 0.9-0.10 for vmapped sparse solves, and limitations for windowed analyses with sparse solvers.
- Current JAXScape imports AMJax `MultilevelSolver`; the project's compatibility monkey patch may now be unnecessary.

Test the smallest upstream-supported path that covers the numerical contract. Reuse existing JAX global-absorption code where correct; update it to public upstream APIs. Confirm state reuse respects changing matrix values and verify both forward and backward solves.

Exercise global, boundary, and moving-window formulations separately. If a required formulation fails, investigate supported solver choices or a compatible release and document the exact limitation. Do not silently replace it with another formulation or begin writing a custom multigrid/adjoint implementation.

Keep float64 explicit. Use public imports and public `init` / `reinit` APIs where supported; remove private upstream imports and compatibility patches only after proving the replacement works.

**Gate:** a tested CPU dependency combination passes small-grid forward and gradient checks for the required formulations. If upstream cannot support a required formulation without substantial custom numerical code, report that concrete design blocker before expanding implementation scope.

## Phase 2: extract an installable Python package

1. Move the existing JAX source into `python/src/diffiscape/`; move its scientific tests into `python/tests/`. Keep one source copy. Do not vendor a second copy under `inst/python`.
2. Add `pyproject.toml` with package metadata, the existing license, supported Python version, numerical dependencies, and development dependencies. Keep tests out of runtime dependencies.
3. Declare supported dependency ranges in package metadata; record exact tested versions in a reproducible development/CI environment. The R package should depend on a compatible DiffiScape Python release, not duplicate every transitive pin.
4. Expose a small Python API for connectivity, fitting, prediction, and posterior inference. Move Flax construction and parameter initialization into Python. Keep JAX/Flax implementation objects out of the required R-facing arguments.
5. Keep existing result dictionaries initially where adequate. Ensure results contain enough configuration, fitted parameters, scaling, and model state to predict correctly without rerunning optimization. Test any supported save/reload workflow.
6. Accept NumPy/array-like input at the public boundary; keep differentiable computations in JAX internally. Avoid conversions to NumPy within the differentiated objective. Do not add mandatory rasterio/geopandas dependencies to an array-based numerical core.
7. Exclude `python/` from R source builds as appropriate. Update CI and references to source paths.

**Gate:** a built wheel and source distribution install into a clean environment; an example fits and predicts using Python alone, without R or repository-relative imports. Test the installed package from outside its source directory.

## Phase 3: make R call the installed Python package

1. Replace file-based Python loading with `reticulate::import("diffiscape", delay_load = TRUE)` and declare the supported Python package requirement using `py_require()` before Python initialization.
2. Keep Python initialization lazy and respect explicitly selected user environments. Do not download runtimes or run solver calculations merely to load the R package.
3. Centralize raster/point-to-array preparation and result-to-raster conversion. Remove repeated data preparation from individual fitting wrappers.
4. Call complete Python operations from R. Do not orchestrate optimizer iterations, build Flax objects, or implement the same likelihood in both languages.
5. Preserve R-facing result fields and S3 behavior where the meaning is unchanged. Report unavailable results explicitly instead of fabricating fit objects, uncertainties, or convergence status.
6. Compare the same tiny problem through Python and R. Include a nonsquare grid and masked cells to catch transposition, flattening, and indexing errors.

**Gate / first runnable milestone:** the same supported model fits and predicts through independent Python and R installations with matching numerical results. Torch may still exist temporarily as a migration reference.

## Phase 4: consolidate shared science and replace JAXopt

Work one supported model/family at a time, preserving reference behavior and deleting the displaced implementation after its checks pass.

- Port shared built-in likelihoods from R into Python, including their weighting, normalization, additional parameters, and selection strata. Cover Poisson, negative binomial, Gaussian, zero inflation, RSF/RSP, and conditional logistic according to the feature matrix. Fitting, prediction, sampling, and diagnostics must use consistent definitions.
- Preserve the default log-linear/global-absorption workflow in JAX. Do not let the existing moving-window PPP objective silently become its replacement.
- Decide the spline behavior explicitly. Preserve documented B-spline/penalty semantics where required, using maintained basis-construction tools; do not relabel the existing RBF model as an equivalent spline GAM.
- Preserve the existing MLP, convolutional, and IRL models as required. Verify grid inputs, adjacency, masked barriers, and that parameters have a measurable effect on predictions.
- Use Optax for Adam. Replace `jaxopt.LBFGSB` with a maintained bounded optimizer. A small SciPy L-BFGS-B adapter consuming `jax.value_and_grad` is a candidate because SciPy is already part of the numerical environment. Optax's unconstrained L-BFGS is not a drop-in substitute for L-BFGS-B. Verify bounds and convergence reporting.
- Move surrogate optimization into Python if retained as a shared feature. Use maintained GP/RF and Latin-hypercube tools; do not write new regression engines. Preserve acquisition semantics and compare surrogate predictions on fixed training data before assessing optimization results. Additional packages are justified when they remove substantial maintained code.
- Use NumPyro for required NUTS/variational inference. Preserve the intended priors, transforms, likelihood, connectivity operator, and masks. Test the intended posterior density on a tiny case; do not claim equivalence just because two samplers return draws.
- Move shared residuals, posterior summaries, information criteria, and posterior-predictive calculations to Python. Keep plotting and spatial presentation native to R where useful.
- Retain `mgcv` and R callback-based custom extensions as explicitly R-only features where necessary. Arbitrary R callbacks must not become hidden dependencies of Python-only fitting or be silently passed into a JAX differentiated objective. Document their supported execution paths.

**Gate:** every retained shared feature works without R; the R interface delegates to the same implementation. Any remaining R-only extensions are named and documented.

## Phase 5: retire Torch and simplify the public interface

1. Remove the Torch implementation, custom adjoints and samplers, CuPy/AMGX installation paths, Torch requirements, and backend-specific CI jobs once replacements pass.
2. Remove JAXopt imports and requirements. Remove obsolete JAX compatibility patches only for a dependency range actually covered by CI.
3. Make the default use JAX while retaining the intended default scientific model. Translate legacy `solver="torch"`, `config$torch`, and Torch-specific functions only when semantics can be preserved; otherwise provide a clear deprecation or migration error. Avoid an indefinite compatibility layer that imports Torch.
4. Make optimizer/model/connectivity options independent, with no redundant framework-selection abstraction. Simplify dispatch only as needed for the final supported choices.
5. Update `DESCRIPTION`, generated `NAMESPACE` / man pages, README, architecture/contributing docs, vignettes, examples, and benchmarks. Ensure generated documentation reflects actual behavior.
6. Remove unused R dependencies only after shared code has moved. Some R-only extensions may still require `mgcv`, `spdep`, or other existing packages.

**Gate:** both packages install and their documented JAX workflows run in an environment with neither Torch nor JAXopt. Searches for their names leave only intentional migration/history references, not runtime imports or install commands.

## Phase 6: CI, release, and completion

- Run scientific tests against the installed Python wheel, plus an sdist installation smoke check. Do not let source-tree imports hide missing package files.
- Run the R suite and `R CMD check` with the installed compatible Python package. Scientific checks live primarily in Python; R tests cover conversion, public behavior, and integration.
- Run a small numerical regression set with matched formulations, float64, tolerances, masks, and source/grounding settings. Include current-density/voltage distinctions and finite-difference gradients.
- Check fitting improves the intended objective, honors bounds, and produces valid predictions. Check sampler diagnostics on a suitably small problem without requiring identical random draws across implementations.
- Verify the documented CPU path first. Treat GPU testing as separate evidence; do not claim GPU support from import success or a CPU test run. Native Windows JAX CPU support does not imply native Windows CUDA support.
- Keep one tested dependency lock for routine CI; use a separate compatibility check when assessing newer upstream versions. Avoid making every normal check depend on the latest untested stack.
- Publish Python before an R release that requires its new API. Start with coordinated release tags and separate package build jobs; no monorepo orchestration framework is needed.

Completion means: one maintained Python scientific implementation; independent Python usage; the R package using that installed implementation; preserved or explicitly documented public behavior; successful packaging/integration checks; no Torch or JAXopt runtime requirement; and reproducible setup instructions.

## Suggested commands after extraction

Adapt executable paths to the actual environment. Run from the repository root unless stated otherwise.

```text
python -m pip install -e "./python[dev]"
python -m pytest python/tests
python -m build python
```

Then install the produced wheel into a separate clean environment and run smoke checks from a directory outside the repository. Configure `RETICULATE_PYTHON` to that interpreter before starting R, install the R package, and run its integration checks / `R CMD check`. Have the chosen build tooling expose the required test/build commands; do not assume `build` or a `dev` extra already exists today.

## Sources to recheck in the implementation session

- JAXScape 0.0.15 release: https://github.com/vboussange/jaxscape/releases/tag/v0.0.15
- Solver-state changes, numerical findings, and limitations: https://github.com/vboussange/jaxscape/pull/50
- AMJax: https://github.com/vboussange/AMJax
- JAX changelog: https://docs.jax.dev/en/latest/changelog.html
- JAX platform support: https://docs.jax.dev/en/latest/installation.html
- JAX sparse maintenance status: https://docs.jax.dev/en/latest/jax.experimental.sparse.html
- JAXopt maintenance status: https://github.com/google/jaxopt#status
- Flax Linen/NNX guidance: https://flax.readthedocs.io/en/latest/
- Reticulate package integration: https://pkgs.rstudio.com/reticulate/articles/package.html
- Python source layout: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/

## Prompt for a new session

> Implement the migration described in `docs/JAX_MIGRATION_PLAN.md`. My priority is reducing maintenance: use upstream numerical solvers and NumPyro rather than developing custom circuit solvers, adjoints, or samplers. Keep a monorepo with an independently installable Python scientific core under `python/` and the existing R package at the root as its adapter. Recheck repository instructions, working-tree changes, installed runtimes, and current upstream compatibility first. Work through the phases and their gates in order. Preserve scientific semantics, particularly global absorption versus moving windows, effective resistance versus current, likelihoods, masking, and spline behavior. Keep `mgcv` as an R-only extension initially. Preserve unrelated changes and do not remove Torch before replacement behavior is verified. Begin with the baseline, upstream solver validation, and a runnable Python/R integration milestone, then continue through consolidation and removal. Report concrete test results, remaining incompatibilities, and the next unfinished phase; do not claim full migration from successful imports or skipped tests.
