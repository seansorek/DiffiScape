"""Bayesian posterior sampling for DiffiScape connectivity surfaces.

Provides NUTS (via NumPyro MCMC) and ADVI (via NumPyro SVI) samplers that
draw posterior samples for Flax neural-network resistance-surface parameters.
Both samplers build a NumPyro model whose likelihood chains:

    Flax model -> exp(log_r) -> permeability -> GridGraph ->
    ResistanceDistance -> PPP log-likelihood

NumPyro is an optional dependency; all public functions raise ``ImportError``
with a helpful message when it is absent.
"""
import time

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
    from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
except ImportError:
    numpyro = None

from .core import prepare_permeability, _mean_weight, ppp_loglik

try:
    from jaxscape import GridGraph, ResistanceDistance
except ImportError:
    GridGraph = None
    ResistanceDistance = None


def _check_deps():
    """Raise ImportError if required dependencies are missing."""
    if jax is None:
        raise ImportError("JAX is required for sampling. Install with: pip install jax")
    if numpyro is None:
        raise ImportError(
            "NumPyro is required for Bayesian sampling. "
            "Install with: pip install numpyro"
        )
    if GridGraph is None or ResistanceDistance is None:
        raise ImportError("JAXScape is required. Install with: pip install jaxscape")


def _build_numpyro_model(flax_model, basis_jnp, obs_jnp, valid_mask,
                         n_rows, n_cols, cell_area, parameterization,
                         init_params):
    """Build a NumPyro model function from a Flax resistance model.

    Parameters
    ----------
    flax_model : flax.linen.Module
        A Flax resistance model (e.g., ResistanceMLP).
    basis_jnp : jnp.ndarray
        Basis covariate matrix, shape (n_cells, n_basis).
    obs_jnp : jnp.ndarray
        Observed counts per cell, shape (n_cells,).
    valid_mask : jnp.ndarray
        Boolean mask of shape (n_rows * n_cols,) indicating valid cells.
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.
    cell_area : float
        Area of each grid cell.
    parameterization : str
        Either "resistance" or "permeability".
    init_params : pytree
        Initial Flax parameter pytree (used to determine flattened shape).

    Returns
    -------
    tuple
        ``(model_fn, flat_init_params, unflatten_fn)`` where:

        - ``model_fn`` is a zero-argument callable suitable for NumPyro.
        - ``flat_init_params`` is a 1-D array of initial parameter values.
        - ``unflatten_fn`` reconstructs the Flax pytree from a flat vector.
    """
    flat_params, unflatten = jax.flatten_util.ravel_pytree(init_params)
    n_params = len(flat_params)

    def model():
        param_vec = numpyro.sample(
            "params",
            dist.Normal(jnp.zeros(n_params), jnp.ones(n_params) * 2.0),
        )
        alpha = numpyro.sample("alpha", dist.Normal(0.0, 2.0))
        gamma = numpyro.sample("gamma", dist.Normal(1.0, 1.0))

        params_tree = unflatten(param_vec)
        log_r = flax_model.apply(params_tree, basis_jnp)
        resistance = jnp.exp(log_r)

        # Embed valid-cell resistance into the full grid
        fill_val = jnp.mean(resistance)
        full_surface = jnp.ones(n_rows * n_cols) * fill_val
        full_surface = full_surface.at[valid_mask].set(resistance)
        surface_2d = full_surface.reshape((n_rows, n_cols))

        # Permeability conversion and circuit solve
        perm = prepare_permeability(surface_2d, parameterization)
        grid = GridGraph(grid=perm, fun=_mean_weight)
        dist_solver = ResistanceDistance()
        source = grid.coord_to_index(jnp.array([0]), jnp.array([0]))
        connectivity = dist_solver(grid, source)

        # PPP log-likelihood on valid cells with learnable alpha/gamma
        conn_valid = connectivity.ravel()[valid_mask]
        loglik = ppp_loglik(conn_valid, obs_jnp, cell_area, alpha, gamma)
        numpyro.factor("loglik", loglik)

    return model, flat_params, unflatten


def _summarize_samples(samples_dict):
    """Compute per-parameter posterior summaries.

    Parameters
    ----------
    samples_dict : dict[str, np.ndarray]
        Dictionary mapping parameter names to sample arrays.
        Each array has shape ``(n_samples,)`` for scalars or
        ``(n_samples, n_params)`` for vectors.

    Returns
    -------
    dict[str, dict]
        Dictionary mapping parameter names to summary dicts with keys
        ``mean``, ``sd``, ``q025``, ``q50``, ``q975``.  For vector
        parameters the values are arrays of length ``n_params``.
    """
    summary = {}
    for k, v in samples_dict.items():
        arr = np.array(v)
        if arr.ndim == 1:
            summary[k] = {
                "mean": float(np.mean(arr)),
                "sd": float(np.std(arr)),
                "q025": float(np.quantile(arr, 0.025)),
                "q50": float(np.median(arr)),
                "q975": float(np.quantile(arr, 0.975)),
            }
        elif arr.ndim == 2:
            summary[k] = {
                "mean": np.mean(arr, axis=0).tolist(),
                "sd": np.std(arr, axis=0).tolist(),
                "q025": np.quantile(arr, 0.025, axis=0).tolist(),
                "q50": np.median(arr, axis=0).tolist(),
                "q975": np.quantile(arr, 0.975, axis=0).tolist(),
            }
    return summary


def run_nuts_sampling(
    flax_model,
    init_params,
    basis_values,
    obs_counts,
    valid_mask,
    n_rows,
    n_cols,
    cell_area=1.0,
    parameterization="resistance",
    n_samples=1000,
    warmup=1000,
    max_treedepth=10,
    target_accept=0.80,
    seed=42,
):
    """Run NUTS (No-U-Turn Sampler) via NumPyro MCMC.

    Draws posterior samples for the parameters of a Flax resistance model
    using the NUTS algorithm.  The model places a ``Normal(0, 2)`` prior on
    each (flattened) parameter and evaluates a point-process log-likelihood
    that chains through JAXScape circuit-theory resistance distances.

    Parameters
    ----------
    flax_model : flax.linen.Module
        A Flax resistance model (e.g., ``ResistanceMLP``).
    init_params : pytree
        Initial Flax parameter pytree.
    basis_values : array-like
        Basis covariate matrix, shape ``(n_cells, n_basis)``.
    obs_counts : array-like
        Observed counts per cell, shape ``(n_cells,)``.
    valid_mask : array-like
        Boolean mask of shape ``(n_rows * n_cols,)`` indicating valid cells.
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.
    cell_area : float, optional
        Area of each grid cell (default: 1.0).
    parameterization : str, optional
        Either ``"resistance"`` or ``"permeability"`` (default: ``"resistance"``).
    n_samples : int, optional
        Number of posterior samples to draw (default: 1000).
    warmup : int, optional
        Number of warmup (adaptation) steps (default: 1000).
    max_treedepth : int, optional
        Maximum tree depth for NUTS (default: 10).
    target_accept : float, optional
        Target acceptance probability (default: 0.80).
    seed : int, optional
        Random seed (default: 42).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"samples"`` : dict of np.ndarrays -- posterior samples per parameter.
        - ``"summary"`` : dict -- per-parameter summaries (mean, sd, quantiles).
        - ``"n_divergences"`` : int -- number of divergent transitions.
        - ``"elapsed"`` : float -- wall-clock seconds.

    Raises
    ------
    ImportError
        If JAX, NumPyro, or JAXScape are not installed.
    """
    _check_deps()

    basis_jnp = jnp.array(basis_values, dtype=jnp.float64)
    obs_jnp = jnp.array(obs_counts, dtype=jnp.float64)
    mask_jnp = jnp.array(valid_mask)

    numpyro_model, flat_init, unflatten = _build_numpyro_model(
        flax_model, basis_jnp, obs_jnp, mask_jnp,
        n_rows, n_cols, cell_area, parameterization, init_params,
    )

    kernel = NUTS(
        numpyro_model,
        max_tree_depth=max_treedepth,
        target_accept_prob=target_accept,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=n_samples,
        progress_bar=False,
    )

    rng = jax.random.PRNGKey(seed)
    t0 = time.time()
    mcmc.run(rng, init_params={
        "params": flat_init,
        "alpha": jnp.array(0.0),
        "gamma": jnp.array(1.0),
    })
    elapsed = time.time() - t0

    samples = mcmc.get_samples()
    samples_np = {k: np.array(v) for k, v in samples.items()}
    summary = _summarize_samples(samples_np)

    extra = mcmc.get_extra_fields()
    if "diverging" in extra:
        n_divergences = int(np.array(extra["diverging"]).sum())
    else:
        n_divergences = 0

    return {
        "samples": samples_np,
        "summary": summary,
        "n_divergences": n_divergences,
        "elapsed": elapsed,
    }


def run_advi_sampling(
    flax_model,
    init_params,
    basis_values,
    obs_counts,
    valid_mask,
    n_rows,
    n_cols,
    cell_area=1.0,
    parameterization="resistance",
    n_samples=2000,
    max_iter=2000,
    lr=0.01,
    seed=42,
):
    """Run Automatic Differentiation Variational Inference (ADVI) via NumPyro.

    Fits a low-rank multivariate normal variational approximation to the
    posterior using NumPyro's ``SVI`` with ``Trace_ELBO``, then draws samples
    from the fitted guide.

    Parameters
    ----------
    flax_model : flax.linen.Module
        A Flax resistance model (e.g., ``ResistanceMLP``).
    init_params : pytree
        Initial Flax parameter pytree.
    basis_values : array-like
        Basis covariate matrix, shape ``(n_cells, n_basis)``.
    obs_counts : array-like
        Observed counts per cell, shape ``(n_cells,)``.
    valid_mask : array-like
        Boolean mask of shape ``(n_rows * n_cols,)`` indicating valid cells.
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.
    cell_area : float, optional
        Area of each grid cell (default: 1.0).
    parameterization : str, optional
        Either ``"resistance"`` or ``"permeability"`` (default: ``"resistance"``).
    n_samples : int, optional
        Number of posterior samples to draw from the fitted guide (default: 2000).
    max_iter : int, optional
        Maximum number of SVI iterations (default: 2000).
    lr : float, optional
        Adam learning rate (default: 0.01).
    seed : int, optional
        Random seed (default: 42).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"samples"`` : dict of np.ndarrays -- posterior samples per parameter.
        - ``"summary"`` : dict -- per-parameter summaries (mean, sd, quantiles).
        - ``"best_elbo"`` : float -- best (final) ELBO estimate.
        - ``"converged"`` : bool -- whether the ELBO stabilised.
        - ``"elapsed"`` : float -- wall-clock seconds.

    Raises
    ------
    ImportError
        If JAX, NumPyro, or JAXScape are not installed.
    """
    _check_deps()

    basis_jnp = jnp.array(basis_values, dtype=jnp.float64)
    obs_jnp = jnp.array(obs_counts, dtype=jnp.float64)
    mask_jnp = jnp.array(valid_mask)

    numpyro_model, flat_init, unflatten = _build_numpyro_model(
        flax_model, basis_jnp, obs_jnp, mask_jnp,
        n_rows, n_cols, cell_area, parameterization, init_params,
    )

    guide = AutoLowRankMultivariateNormal(numpyro_model)
    optimizer = numpyro.optim.Adam(lr)
    svi = SVI(numpyro_model, guide, optimizer, loss=Trace_ELBO())

    rng = jax.random.PRNGKey(seed)
    t0 = time.time()
    svi_result = svi.run(rng, max_iter, progress_bar=False)
    elapsed = time.time() - t0

    # Draw samples from the fitted variational guide
    predictive = Predictive(guide, params=svi_result.params, num_samples=n_samples)
    samples = predictive(rng)
    samples_np = {k: np.array(v) for k, v in samples.items()}
    summary = _summarize_samples(samples_np)

    # Assess convergence: check if ELBO stabilised in the final 10% of iters
    losses = np.array(svi_result.losses)
    tail_len = max(1, len(losses) // 10)
    tail = losses[-tail_len:]
    if len(tail) > 1 and np.std(tail) > 0:
        rel_change = np.std(tail) / (np.abs(np.mean(tail)) + 1e-8)
        converged = bool(rel_change < 0.1)
    else:
        converged = True

    return {
        "samples": samples_np,
        "summary": summary,
        "best_elbo": float(-losses[-1]),
        "converged": converged,
        "elapsed": elapsed,
    }
