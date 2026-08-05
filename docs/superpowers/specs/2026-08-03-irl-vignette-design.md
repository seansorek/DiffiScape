# IRL Vignette Design

## Goal

Add a compiled vignette that teaches DiffiScape's PyTorch inverse reinforcement
learning pathway through the same synthetic urban example used by
`spline-gam.Rmd` and `torch-mlp.Rmd`.

## Scope

Create `vignettes/irl.Rmd`, extend the existing vignette contract test, and
render the vignette to HTML. Keep the established IRL model, optimizer, R
wrapper, and Python tests unchanged unless compilation or testing exposes a
defect that blocks the vignette.

## Vignette Structure

The vignette will:

1. Build the reference 100 by 100 canopy-and-impervious landscape and sample
   1,000 occurrence points from its circuit-derived intensity.
2. Explain the IRL data flow: covariates produce reward, soft value iteration
   produces long-range value, value produces resistance, and the circuit solve
   produces connectivity for the point-process likelihood.
3. Run `verify_irl_gradient()` in the illustrative code.
4. Configure `run_torch_pipeline()` with `model_type = "irl"`, a small reward
   network, bounded value-iteration work, at most 10 optimization epochs, and
   CPU execution.
5. Check finite positive outputs, plot the loss and maps, wrap the fit with
   `irl_resistance_model()`, and save a compact result object.

Like the two reference vignettes, the document will set `eval = FALSE`. This
keeps package builds independent of a configured Python and PyTorch runtime;
rendering still validates the R Markdown structure and presentation.

## Tests

Extend `tests/testthat/test-torch-vignettes.R` before creating the vignette.
The first test run must fail because `vignettes/irl.Rmd` is absent. The updated
test will apply the shared fast-example contract to all three vignettes and
will require IRL-specific calls and configuration in the new document.

After implementation, run the focused vignette test, render `irl.Rmd` to HTML,
and run the relevant package test suite. Diagnose failures that arise in this
workflow and change production code only when a reproduced defect requires it.

## Output

Track the source vignette and test changes. Treat the rendered HTML as a build
artifact and place it in a temporary output directory unless the repository's
existing build process requires another location.
