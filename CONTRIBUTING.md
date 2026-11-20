# Contributing

Use Python 3.12 and the CPU constraints for routine changes:

```sh
python -m pip install -c python/constraints-cpu.txt './python[dev]'
python -m build python
python -m pip install --no-deps --force-reinstall python/dist/*.whl
```

Run `python -m pytest /absolute/path/to/DiffiScape/python/tests` outside the
checkout to validate the installed package. Install the sdist and rerun the
public API smoke checks. Do not set PYTHONPATH for packaging gates.

In R, set RETICULATE_PYTHON to the same interpreter before starting R. Install R
dependencies, run `devtools::document()`, `devtools::test()`, then `R CMD build .`
and `R CMD check` on the source tarball. Integration tests need NOT_CRAN=true
for a local full run. Ordinary package loading must stay lazy.

Put shared numerical changes in Python, and adapter checks in R. Add a tiny
independent dense reference or finite-difference check for changed physical
semantics. Match formulations, float64, masks, sources, grounding, tolerances,
and solver-state reuse before comparing performance. Preserve existing failures
and unrelated working-tree changes. `docs/JAX_MIGRATION_STATUS.md` records the
scientific contract and tested evidence.

Use upstream sparse solving and autodiff, Optax/SciPy optimization, NumPyro
inference, SciPy splines, and SMT kriging. Do not add custom adjoints or samplers.
Keep mgcv and arbitrary R callbacks explicitly R-only. Generated NAMESPACE and
man pages must follow roxygen source. Update both package guides for API changes.

`benchmarks/bench_full.py` measures the current physical CPU formulations with
equal settings and synchronized JAX execution. June 2026 results are historical
and are not migration acceptance evidence. Accelerator tests are separate.
Publish Python before an R release requiring its API; coordinate release tags
without adding monorepo orchestration.
