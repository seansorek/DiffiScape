"""Shared sampler utilities used by run_langevin_sampling, run_hmc_sampling,
and run_advi.

  - _compute_ess_chain: effective-sample-size estimator (autocorrelation
    initial-positive-sequence method).
  - _setup_sampling_state: load a MAP checkpoint, rebuild the
    SplineResistanceNet, and identify the set of sampled parameters. Shared
    by all three samplers (Langevin sets one extra flag,
    ``cs.CG_WARM_START``, immediately after calling this — see
    run_langevin_sampling for why that isn't folded in here).
"""
import numpy as np
import torch
import torch.nn as nn

from ..constants import _GPU_AVAILABLE, _CUPY_AVAILABLE
from .._module_loaders import _get_circuit_module
from ..resistance_nets import SplineResistanceNet


def _compute_ess_chain(chain):
    """Estimate ESS from 1-D chain using initial positive sequence estimator."""
    n = len(chain)
    if n < 10:
        return float(n)
    chain = chain - np.mean(chain)
    var0 = np.var(chain)
    if var0 < 1e-30:
        return float(n)
    fft_chain = np.fft.fft(chain, n=2 * n)
    acf = np.fft.ifft(fft_chain * np.conj(fft_chain)).real[:n] / (n * var0)
    tau = 1.0
    for lag in range(1, n // 2):
        rho = acf[lag]
        if rho < 0.05:
            break
        tau += 2 * rho
    return max(1.0, n / tau)


def _setup_sampling_state(
    basis_values_np, obs_counts_np, n_rows, n_cols, valid_mask_np, cell_area,
    model_path, model_state,
    source_spacing, source_from_resistance, solver, absorption,
    reg_mean, log_R_baseline, penalty_scale,
    n_knots, spline_degree, include_interactions, lambda_min,
    fix_smoothing, fix_intensity,
    cg_tol, seed, device,
):
    """Shared setup for MALA and HMC samplers: load model, build tensors."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Load checkpoint
    if model_state is not None:
        checkpoint = model_state
    elif model_path is not None:
        checkpoint = torch.load(model_path, map_location="cpu",
                                weights_only=False)
    else:
        raise ValueError("Either model_path or model_state must be provided")

    _ckpt_model_type = checkpoint.get("model_type", "spline")
    if _ckpt_model_type == "irl":
        raise NotImplementedError(
            "Bayesian sampling (Langevin/HMC/ADVI) is not yet supported for "
            "model_type='irl'. Use run_torch_optimization() for MAP inference."
        )

    n_features = int(checkpoint.get(
        "n_features",
        basis_values_np.shape[1] if basis_values_np.ndim > 1 else 1))
    n_knots_ckpt = int(checkpoint.get("n_knots", n_knots))
    degree_ckpt = int(checkpoint.get("spline_degree", spline_degree))
    interactions_ckpt = bool(checkpoint.get("include_interactions",
                                            include_interactions))
    use_absorption = str(solver).lower() in ("global_absorption", "absorption")

    # Device
    if device == "auto":
        use_cuda = _GPU_AVAILABLE and _CUPY_AVAILABLE
        dev = torch.device("cuda" if use_cuda else "cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if not _CUPY_AVAILABLE:
            raise RuntimeError("CUDA requested but cupy not installed")
        dev = torch.device("cuda")
        use_cuda = True
    else:
        dev = torch.device("cpu")
        use_cuda = False

    # Intensity spline settings from checkpoint
    intensity_spline_ckpt = bool(checkpoint.get("intensity_spline", False))
    intensity_n_knots_ckpt = int(checkpoint.get("intensity_n_knots", 5))
    intensity_degree_ckpt = int(checkpoint.get("intensity_degree", 3))
    lambda_init_intensity_ckpt = float(checkpoint.get("lambda_init_intensity", 2.0))
    intensity_log1p_max_ckpt = float(checkpoint.get("intensity_log1p_max", 10.0))

    # Build network
    net = SplineResistanceNet(
        n_features=n_features,
        n_knots=n_knots_ckpt,
        degree=degree_ckpt,
        include_interactions=interactions_ckpt,
        lambda_init_marginal=0.0,
        lambda_init_interaction=0.0,
        lambda_min=float(lambda_min),
        intensity_spline=intensity_spline_ckpt,
        intensity_n_knots=intensity_n_knots_ckpt,
        intensity_degree=intensity_degree_ckpt,
        lambda_init_intensity=lambda_init_intensity_ckpt,
        intensity_log1p_max=intensity_log1p_max_ckpt,
    ).double().to(dev)
    net.load_state_dict(checkpoint["net"])
    net.setup_basis_matrices(np.asarray(basis_values_np, dtype=np.float64),
                             device=dev)
    net.train()

    use_intensity_spline = intensity_spline_ckpt and net.intensity_spline

    # Intensity parameters
    alpha_map = float(checkpoint["alpha"])
    gamma_map = float(checkpoint["gamma"])
    alpha = nn.Parameter(torch.tensor(alpha_map, dtype=torch.float64, device=dev))
    _raw_gamma_init = float(np.log(np.exp(gamma_map) - 1.0 + 1e-8))
    raw_gamma = nn.Parameter(torch.tensor(_raw_gamma_init, dtype=torch.float64,
                                          device=dev))
    if use_intensity_spline:
        raw_gamma.requires_grad_(False)

    # Data tensors
    basis_t = torch.tensor(np.asarray(basis_values_np, dtype=np.float64),
                           dtype=torch.float64, device=dev)
    obs_t = torch.tensor(np.asarray(obs_counts_np, dtype=np.float64),
                         dtype=torch.float64, device=dev)
    n_valid = basis_t.shape[0]

    # Circuit solver
    cs = _get_circuit_module()
    original_rtol = getattr(cs, "CG_RTOL", 1e-6)
    cs.CG_RTOL = float(cg_tol)
    if use_cuda:
        cs.enable_gpu(True)
    else:
        cs.enable_gpu(False)

    # Identify parameters to sample
    sampled_params = []
    param_names = []
    for k, p in enumerate(net.spline_coefs):
        sampled_params.append(p)
        param_names.append(f"spline_coef_{k}")
    for idx, p in enumerate(net.interaction_coefs):
        sampled_params.append(p)
        param_names.append(f"interaction_coef_{idx}")
    sampled_params.extend([net.skip.weight, net.skip.bias, net.intercept])
    param_names.extend(["skip_weight", "skip_bias", "intercept"])
    if not fix_smoothing:
        for k, p in enumerate(net.log_lambda_smooth):
            sampled_params.append(p)
            param_names.append(f"log_lambda_smooth_{k}")
        for idx, p in enumerate(net.log_lambda_interact_row):
            sampled_params.append(p)
            param_names.append(f"log_lambda_interact_row_{idx}")
        for idx, p in enumerate(net.log_lambda_interact_col):
            sampled_params.append(p)
            param_names.append(f"log_lambda_interact_col_{idx}")
        if use_intensity_spline and hasattr(net, 'log_lambda_intensity'):
            sampled_params.append(net.log_lambda_intensity)
            param_names.append("log_lambda_intensity")
    if not fix_intensity:
        sampled_params.append(alpha)
        param_names.append("alpha")
        if use_intensity_spline:
            sampled_params.append(net.intensity_coefs)
            param_names.append("intensity_coefs")
        else:
            sampled_params.append(raw_gamma)
            param_names.append("raw_gamma")

    n_total_scalar = sum(p.numel() for p in sampled_params)

    return {
        "net": net,
        "alpha": alpha,
        "raw_gamma": raw_gamma,
        "basis_t": basis_t,
        "obs_t": obs_t,
        "n_valid": n_valid,
        "n_features": n_features,
        "sampled_params": sampled_params,
        "param_names": param_names,
        "n_total_scalar": n_total_scalar,
        "use_absorption": use_absorption,
        "use_intensity_spline": use_intensity_spline,
        "use_cuda": use_cuda,
        "dev": dev,
        "cs": cs,
        "original_rtol": original_rtol,
        "checkpoint": checkpoint,
    }


