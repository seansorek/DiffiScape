"""Array-based operations; spatial metadata belongs to the calling language."""
import numpy as np
import jax.numpy as jnp

from .core import _apply_link, invalid_cell_fill_value, prepare_permeability, cumulative_current_core
from .optimize import run_parametric_optimization


def connectivity(surface, formulation="global_absorption", absorption=0.01,
                 source_spacing=1, source_from_resistance=False,
                 parameterization="resistance", radius=3, block_size=2):
    """Compute physical current and voltage with explicit grounding conventions."""
    from .core import physical_solve
    surface = np.asarray(surface, dtype=np.float64)
    if surface.ndim != 2 or not surface.size or not np.isfinite(surface).all() or (surface <= 0).any():
        raise ValueError("surface must be a finite positive two-dimensional array")
    if not np.isfinite(absorption) or absorption <= 0:
        raise ValueError("absorption must be positive and finite")
    if not isinstance(source_spacing, (int, np.integer)) or source_spacing < 1:
        raise ValueError("source_spacing must be a positive integer")
    if parameterization not in ("resistance", "permeability"):
        raise ValueError("parameterization must be resistance or permeability")
    permeability = jnp.asarray(1 / surface if parameterization == "resistance" else surface)
    if formulation == "focal_current":
        from .core import focal_current
        if any(not isinstance(n, (int, np.integer)) or n < 1 for n in (radius, block_size)):
            raise ValueError("radius and block_size must be positive integers")
        return {"current": np.asarray(focal_current(permeability, radius, block_size, source_from_resistance)), "voltage": None}
    current, voltage = physical_solve(permeability, formulation, absorption,
                                      source_spacing, source_from_resistance)
    return {"current": np.asarray(current).reshape(surface.shape),
            "voltage": np.asarray(voltage).reshape(surface.shape)}


def _validate_arrays(basis_values, valid_mask, n_rows, n_cols):
    if any(not isinstance(n, (int, np.integer)) or n < 1 for n in (n_rows, n_cols)):
        raise ValueError("n_rows and n_cols must be positive integers")
    basis = np.asarray(basis_values, dtype=np.float64)
    mask = np.asarray(valid_mask)
    if mask.dtype.kind != "b" or mask.shape != (n_rows * n_cols,) or not mask.any():
        raise ValueError("valid_mask must be a nonempty full-grid Boolean vector")
    if basis.ndim != 2 or basis.shape[0] != mask.sum() or basis.shape[1] < 1 or not np.isfinite(basis).all():
        raise ValueError("basis_values must be a finite (n_valid, n_covariates) matrix")
    return basis, mask


def fit(basis_values, obs_counts, valid_mask, n_rows, n_cols, cell_area=1.0,
        formulation="global_absorption", model_type="loglinear", **options):
    """Fit resistance and PPP intensity using array-like input.

    Optimizer options match ``run_parametric_optimization``. This does not
    reinterpret the R/Torch global-absorption formulation as window resistance.
    """
    basis, mask = _validate_arrays(basis_values, valid_mask, n_rows, n_cols)
    counts = np.asarray(obs_counts, dtype=np.float64)
    if counts.shape != (mask.sum(),) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("obs_counts must contain nonnegative finite counts for valid cells")
    if not np.isfinite(cell_area) or cell_area <= 0:
        raise ValueError("cell_area must be positive and finite")
    if formulation != "window_resistance":
        from .physical import fit_physical
        return fit_physical(basis, counts, mask, n_rows, n_cols, cell_area,
                            formulation=formulation, model_type=model_type, **options)
    configuration = dict(n_rows=n_rows, n_cols=n_cols, valid_mask=mask.copy(),
                         n_covariates=basis.shape[1], cell_area=cell_area,
                         link_fn=options.get("link_fn", "exp"),
                         radius=options.get("radius", 13), block_size=options.get("block_size", 5),
                         parameterization=options.get("parameterization", "resistance"),
                         model_type=model_type,
                         formulation="window_resistance")
    if configuration["link_fn"] not in ("exp", "softplus", "identity"):
        raise ValueError("link_fn must be exp, softplus, or identity")
    if configuration["parameterization"] not in ("resistance", "permeability"):
        raise ValueError("parameterization must be resistance or permeability")
    for name in ("radius", "block_size"):
        value = configuration[name]
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if model_type == "loglinear":
        result = run_parametric_optimization(basis, counts, mask, n_rows, n_cols,
                                             cell_area=cell_area, **options)
    else:
        from .optimize import run_neural_optimization
        result = run_neural_optimization(basis, counts, mask, n_rows, n_cols,
                                          cell_area=cell_area, **options, model_type=model_type)
    result["configuration"] = configuration
    return result


def predict(result, basis_values):
    """Predict full-grid surface, effective resistance, and PPP intensity.

    Masked cells are NaN in returned arrays and finite barriers during solving.
    Configuration comes from the fit; optimization is never rerun.
    """
    cfg = result["configuration"]
    basis, mask = _validate_arrays(basis_values, cfg["valid_mask"], cfg["n_rows"], cfg["n_cols"])
    if basis.shape[1] != cfg["n_covariates"]:
        raise ValueError("prediction covariates must match the fitted covariate count and order")
    if cfg["formulation"] != "window_resistance":
        from .physical import predict_physical
        return predict_physical(result, basis)
    surface = jnp.full(mask.size, invalid_cell_fill_value(cfg["parameterization"]))
    if cfg.get("model_type", "loglinear") == "loglinear":
        values = _apply_link(jnp.asarray(result["best_params"]), jnp.asarray(basis), cfg["link_fn"])
    else:
        from .resistance import create_model, legacy_model_input
        model = create_model(cfg["model_type"], result["model_config"], basis.shape[1])
        inputs = legacy_model_input(jnp.asarray(basis), mask, cfg["n_rows"], cfg["n_cols"], cfg["model_type"])
        values = jnp.exp(model.apply(result["model_state"], inputs))
        if cfg["model_type"] == "conv":
            values = values[mask]
    surface = surface.at[mask].set(values)
    shape = (cfg["n_rows"], cfg["n_cols"])
    permeability = prepare_permeability(surface.reshape(shape), cfg["parameterization"])
    connectivity = cumulative_current_core(permeability, *shape, cfg["radius"], cfg["block_size"])
    intensity = jnp.exp(result["alpha"] + result["gamma"] * jnp.log1p(connectivity))
    return {name: np.where(mask.reshape(shape), np.asarray(value).reshape(shape), np.nan)
            for name, value in (("surface", surface), ("connectivity", connectivity), ("intensity", intensity))}


def sample(basis_values, obs_counts, valid_mask, n_rows, n_cols, cell_area=1.0,
           method="nuts", model_type="mlp", model_config=None, seed=42, fitted=None, **options):
    """NumPyro posterior inference without requiring a Flax object from callers."""
    from .resistance import create_model
    from .sampling import run_nuts_sampling, run_advi_sampling
    import jax

    basis, mask = _validate_arrays(basis_values, valid_mask, n_rows, n_cols)
    counts = np.asarray(obs_counts, dtype=np.float64)
    if counts.shape != (mask.sum(),) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("obs_counts must contain nonnegative finite counts for valid cells")
    if not np.isfinite(cell_area) or cell_area <= 0:
        raise ValueError("cell_area must be positive and finite")
    if fitted is not None:
        from .physical import sample_physical
        cfg = fitted["configuration"]
        if (cfg["n_rows"], cfg["n_cols"]) != (n_rows, n_cols) or not np.array_equal(cfg["valid_mask"], mask) or cfg["cell_area"] != cell_area:
            raise ValueError("posterior geometry, mask and cell area must match the fit")
        return sample_physical(fitted, basis, counts, method=method, seed=seed, **options)
    if model_type not in ("mlp", "spline_gam"):
        raise NotImplementedError("legacy window posteriors support mlp and RBF spline_gam; use fitted= for physical conv models. IRL is retired.")
    if method not in ("nuts", "advi"):
        raise ValueError("method must be nuts or advi")
    model = create_model(model_type, model_config, basis.shape[1])
    params = model.init(jax.random.PRNGKey(seed), jnp.asarray(basis))
    sampler = run_nuts_sampling if method == "nuts" else run_advi_sampling
    return sampler(model, params, basis, counts, mask, n_rows, n_cols,
                   cell_area=cell_area, seed=seed, **options)
