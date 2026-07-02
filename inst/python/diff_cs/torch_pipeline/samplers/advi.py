"""Automatic Differentiation Variational Inference (ADVI).

See run_advi() for the algorithm description. Shares checkpoint loading /
network setup with run_langevin_sampling and run_hmc_sampling via
_setup_sampling_state() (samplers/common.py).
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import DEFAULT_CG_TOL
from ..autograd_functions import _CircuitSolveFn, _AbsorptionCircuitSolveFn
from ..resistance_nets import _ppp_loglik
from .common import _setup_sampling_state


# ==============================================================================
# ADVI: Automatic Differentiation Variational Inference
# ==============================================================================

def run_advi(
    basis_values_np,
    obs_counts_np,
    n_rows,
    n_cols,
    valid_mask_np,
    cell_area,
    model_path=None,
    model_state=None,
    source_spacing=1,
    source_from_resistance=True,
    solver="global_absorption",
    absorption=0.002,
    # ADVI settings
    n_samples=2000,
    max_iter=2000,
    lr=0.01,
    n_elbo_samples=1,
    full_rank=False,
    patience=100,
    fix_smoothing=True,
    fix_intensity=False,
    # Regularization
    reg_mean=1.0,
    log_R_baseline=3.0,
    penalty_scale=5.0,
    # Architecture
    n_knots=3,
    spline_degree=3,
    include_interactions=True,
    lambda_min=0.1,
    # Misc
    cg_tol=DEFAULT_CG_TOL,
    seed=42,
    verbose=True,
    output_dir=None,
    device="auto",
):
    """
    Automatic Differentiation Variational Inference (ADVI).

    Fits a Gaussian variational approximation q(θ) = N(μ, Σ) to the posterior
    by maximizing the ELBO via reparameterization gradients.  Supports
    mean-field (diagonal Σ) and full-rank parameterizations.

    Much faster than MCMC since each iteration requires only O(n_elbo_samples)
    forward+adjoint circuit solves (typically 1), vs. O(2^treedepth) for NUTS.

    Parameters
    ----------
    n_samples : int
        Number of posterior samples to draw from fitted q(θ) after convergence.
    max_iter : int
        Maximum ADVI optimization iterations.
    lr : float
        Adam learning rate for variational parameters (μ, log_σ or L).
    n_elbo_samples : int
        MC samples per ELBO gradient estimate (1 is standard; more reduces
        variance but costs proportionally more circuit solves).
    full_rank : bool
        If False, mean-field (diagonal Σ, 2d params).
        If True, full-rank (lower-triangular L with Σ = LL^T, d + d(d+1)/2 params).
    patience : int
        Early stopping: stop if ELBO doesn't improve for this many iterations.
    fix_smoothing : bool
        Fix smoothing parameters at MAP values (exclude from variational params).
    fix_intensity : bool
        Fix intensity parameters (alpha, gamma) at MAP values.

    Returns
    -------
    dict — same structure as run_hmc_sampling/run_langevin_sampling for
    downstream compatibility, plus ADVI-specific fields:
        elbo_trace : list of float (ELBO per iteration)
        converged : bool
        variational_mean : list (fitted μ)
        variational_std : list (fitted marginal σ)
    """
    t0_all = time.time()

    # ---- Shared setup ----
    state = _setup_sampling_state(
        basis_values_np, obs_counts_np, n_rows, n_cols, valid_mask_np,
        cell_area, model_path, model_state,
        source_spacing, source_from_resistance, solver, absorption,
        reg_mean, log_R_baseline, penalty_scale,
        n_knots, spline_degree, include_interactions, lambda_min,
        fix_smoothing, fix_intensity,
        cg_tol, seed, device,
    )
    net = state["net"]
    alpha = state["alpha"]
    raw_gamma = state["raw_gamma"]
    basis_t = state["basis_t"]
    obs_t = state["obs_t"]
    n_valid = state["n_valid"]
    n_features = state["n_features"]
    sampled_params = state["sampled_params"]
    use_absorption = state["use_absorption"]
    use_intensity_spline = state.get("use_intensity_spline", False)
    dev = state["dev"]
    cs = state["cs"]
    original_rtol = state["original_rtol"]
    n_total_scalar = state["n_total_scalar"]

    if verbose:
        mode_str = "full-rank" if full_rank else "mean-field"
        print(f"\n  ADVI ({mode_str}): {n_total_scalar} scalar parameters")
        n_var_params = (n_total_scalar + n_total_scalar * (n_total_scalar + 1) // 2
                        if full_rank else 2 * n_total_scalar)
        print(f"    Variational parameters: {n_var_params}")
        print(f"    Max iterations: {max_iter}, LR: {lr}")
        print(f"    ELBO MC samples: {n_elbo_samples}")
        print(f"    Patience: {patience}")
        print(f"    Fix smoothing: {fix_smoothing}, Fix intensity: {fix_intensity}")

    # ---- Flatten / unflatten utilities ----
    def _flatten(params):
        return torch.cat([p.data.reshape(-1) for p in params])

    def _unflatten(flat, params):
        offset = 0
        for p in params:
            n = p.numel()
            p.data.copy_(flat[offset:offset + n].reshape(p.shape))
            offset += n

    # ---- Initialize variational parameters at MAP ----
    d = n_total_scalar
    mu = nn.Parameter(_flatten(sampled_params).clone())
    # Initialize log_sigma small so q starts concentrated around MAP
    log_sigma = nn.Parameter(torch.full((d,), -3.0, dtype=torch.float64,
                                        device=dev))

    if full_rank:
        # L is lower-triangular: Σ = L L^T
        # Initialize as diagonal (= mean-field start)
        L_diag = nn.Parameter(torch.full((d,), -3.0, dtype=torch.float64,
                                         device=dev))
        L_offdiag = nn.Parameter(torch.zeros(d * (d - 1) // 2,
                                             dtype=torch.float64, device=dev))
        # Slower LR for off-diagonal (correlation structure) to improve stability
        var_params = [mu, L_diag, L_offdiag]
        optimizer = torch.optim.Adam([
            {"params": [mu, L_diag], "lr": lr},
            {"params": [L_offdiag], "lr": lr * 0.1},
        ])
    else:
        var_params = [mu, log_sigma]
        optimizer = torch.optim.Adam(var_params, lr=lr)

    # LR schedule: linear warm-up then cosine decay
    n_warmup = min(50, max_iter // 10)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=max(n_warmup, 1))
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(max_iter - n_warmup, 1), eta_min=lr * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[n_warmup])

    # ---- Helper: build L matrix for full-rank ----
    tril_idx = torch.tril_indices(d, d, offset=-1) if full_rank else None

    def _build_L():
        L = torch.zeros(d, d, dtype=torch.float64, device=dev)
        L[range(d), range(d)] = torch.exp(L_diag)  # positive diagonal
        L[tril_idx[0], tril_idx[1]] = L_offdiag
        return L

    # ---- Collect gradients for one theta sample ----
    def _collect_grad_theta():
        """
        Return grad_theta = d(-log_p)/d(theta) as a flat vector,
        by reading the .grad fields of sampled_params after backward().
        """
        parts = []
        for p in sampled_params:
            if p.grad is not None:
                parts.append(p.grad.reshape(-1).clone())
            else:
                parts.append(torch.zeros(p.numel(), dtype=torch.float64,
                                         device=dev))
        return torch.cat(parts)

    # ---- Log-posterior (populates .grad on sampled_params) ----
    def compute_log_posterior():
        """Compute log p(y, θ) = log_lik - penalty."""
        R, log_R_raw = net.forward_with_log_R(basis_t)
        if use_absorption:
            C = _AbsorptionCircuitSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance, absorption)
        else:
            C = _CircuitSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance)

        gamma_val = F.softplus(raw_gamma)
        if use_intensity_spline:
            log_lam = alpha + net.eval_intensity_spline(
                torch.log1p(C.clamp(min=0)))
        else:
            log_lam = alpha + gamma_val * torch.log1p(C.clamp(min=0))
        ll = _ppp_loglik(log_lam, obs_t, float(cell_area))

        mean_logR = log_R_raw.mean()
        pen_mean = n_valid * reg_mean * (mean_logR - log_R_baseline) ** 2
        pen_smooth = penalty_scale * net.smoothing_penalty()

        return ll - pen_mean - pen_smooth

    # ---- Manual reparameterization gradient ----
    # _unflatten uses .data.copy_() which severs the autograd graph between
    # the variational params (mu, sigma/L) and the network params.  So we
    # cannot rely on neg_elbo.backward() flowing gradients to mu/sigma.
    # Instead, we:
    #   1. Sample eps, compute theta = mu + scale(eps)
    #   2. _unflatten(theta) into network params (non-differentiable copy)
    #   3. Forward + backward on -log_p to get d(-log_p)/d(theta) via .grad
    #   4. Apply chain rule manually:
    #        d(-ELBO)/d(mu)        = E[ d(-log_p)/d(theta) ]
    #        d(-ELBO)/d(log_sigma) = E[ d(-log_p)/d(theta) * sigma * eps ] - 1
    #        (full-rank analogous via d(theta)/d(L))
    #   5. Step optimizer on variational params only.

    # ---- Optimization loop ----
    elbo_trace = []
    elbo_logp_trace = []   # E_q[log p] component
    elbo_entropy_trace = []  # H[q] component
    best_elbo = -float("inf")
    best_iter = 0
    best_mu = mu.data.clone()
    best_log_sigma = log_sigma.data.clone() if not full_rank else None
    best_L_diag = L_diag.data.clone() if full_rank else None
    best_L_offdiag = L_offdiag.data.clone() if full_rank else None

    if verbose:
        print(f"\n  {'Iter':>6}  {'ELBO':>12}  {'E[logp]':>12}  {'H[q]':>10}  "
              f"{'|g_mu|':>8}  {'|g_sig|':>8}  {'mean(sig)':>9}")
        print(f"  {'-' * 75}")

    for it in range(max_iter):
        # Zero variational param grads
        optimizer.zero_grad()

        # Accumulators for manual chain-rule gradients
        grad_mu_accum = torch.zeros(d, dtype=torch.float64, device=dev)
        logp_accum = 0.0

        if full_rank:
            grad_L_diag_accum = torch.zeros(d, dtype=torch.float64, device=dev)
            grad_L_offdiag_accum = torch.zeros_like(L_offdiag.data)
        else:
            grad_log_sigma_accum = torch.zeros(d, dtype=torch.float64,
                                               device=dev)

        for _ in range(n_elbo_samples):
            eps = torch.randn(d, dtype=torch.float64, device=dev)

            # Reparameterize: theta = mu + scale(eps)
            with torch.no_grad():
                if full_rank:
                    L_mat = _build_L()
                    theta = mu.data + L_mat.detach() @ eps
                else:
                    sigma = torch.exp(log_sigma.data)
                    theta = mu.data + sigma * eps

            # Copy theta into network params (non-differentiable)
            _unflatten(theta, sampled_params)

            # Forward + backward on log_p w.r.t. sampled_params
            for p in sampled_params:
                if p.grad is not None:
                    p.grad.zero_()
            log_p = compute_log_posterior()
            neg_log_p = -log_p
            neg_log_p.backward()

            # d(-log_p)/d(theta) from network param grads
            g_theta = _collect_grad_theta()  # d(-log_p)/d(theta)

            logp_accum += float(log_p.item())

            # Chain rule for variational params:
            # d(-ELBO)/d(mu) = E[g_theta]  (since d(theta)/d(mu) = I)
            grad_mu_accum += g_theta

            if full_rank:
                # d(theta)/d(L_diag_k) = eps_k * exp(L_diag_k) (at diagonal)
                # d(theta_i)/d(L_{i,j}) = eps_j for i > j (lower triangular)
                exp_L_diag = torch.exp(L_diag.data)
                # Diagonal: g_theta[k] * eps[k] * exp(L_diag[k])
                grad_L_diag_accum += g_theta * eps * exp_L_diag
                # Off-diagonal: for each (i,j) in lower triangle,
                # d(-ELBO)/d(L_{i,j}) = g_theta[i] * eps[j]
                g_expanded = g_theta[tril_idx[0]]  # g_theta at row indices
                e_expanded = eps[tril_idx[1]]       # eps at col indices
                grad_L_offdiag_accum += g_expanded * e_expanded
            else:
                # d(theta)/d(log_sigma) = sigma * eps
                # d(-ELBO)/d(log_sigma) = E[g_theta * sigma * eps]
                grad_log_sigma_accum += g_theta * sigma * eps

        # Average over MC samples
        inv_S = 1.0 / n_elbo_samples
        grad_mu_accum *= inv_S
        logp_mean = logp_accum * inv_S

        # Assign gradients to variational params (likelihood term)
        mu.grad = grad_mu_accum.clone()

        if full_rank:
            grad_L_diag_accum *= inv_S
            grad_L_offdiag_accum *= inv_S
            # Entropy gradient: d(-H)/d(L_diag) = -1
            L_diag.grad = grad_L_diag_accum - 1.0
            L_offdiag.grad = grad_L_offdiag_accum.clone()
        else:
            grad_log_sigma_accum *= inv_S
            # Entropy gradient: d(-H)/d(log_sigma) = -1
            log_sigma.grad = grad_log_sigma_accum - 1.0

        # Compute ELBO for logging
        if full_rank:
            entropy_val = (0.5 * d * (1.0 + math.log(2.0 * math.pi))
                           + float(L_diag.data.sum().item()))
        else:
            entropy_val = (float(log_sigma.data.sum().item())
                           + 0.5 * d * (1.0 + math.log(2.0 * math.pi)))
        elbo_val = logp_mean + entropy_val

        # Gradient clipping on variational params
        torch.nn.utils.clip_grad_norm_(var_params, 100.0)

        optimizer.step()
        scheduler.step()

        elbo_trace.append(elbo_val)
        elbo_logp_trace.append(logp_mean)
        elbo_entropy_trace.append(entropy_val)

        # Track best
        if elbo_val > best_elbo:
            best_elbo = elbo_val
            best_iter = it
            best_mu = mu.data.clone()
            if full_rank:
                best_L_diag = L_diag.data.clone()
                best_L_offdiag = L_offdiag.data.clone()
            else:
                best_log_sigma = log_sigma.data.clone()

        # Early stopping
        if it - best_iter >= patience:
            if verbose:
                print(f"\n  Early stopping at iter {it} "
                      f"(best ELBO {best_elbo:.2f} at iter {best_iter})")
            break

        # Logging
        if verbose and (it % 50 == 0 or it < 5 or it == max_iter - 1):
            delta = elbo_val - elbo_trace[-2] if len(elbo_trace) > 1 else 0.0
            g_mu_norm = float(mu.grad.norm().item()) if mu.grad is not None else 0.0
            if full_rank:
                g_sig_norm = float(L_diag.grad.norm().item()) if L_diag.grad is not None else 0.0
                mean_sig = float(torch.exp(L_diag.data).mean().item())
            else:
                g_sig_norm = float(log_sigma.grad.norm().item()) if log_sigma.grad is not None else 0.0
                mean_sig = float(torch.exp(log_sigma.data).mean().item())
            print(f"  {it:6d}  {elbo_val:12.2f}  {logp_mean:12.2f}  "
                  f"{entropy_val:10.2f}  {g_mu_norm:8.3f}  {g_sig_norm:8.3f}  "
                  f"{mean_sig:9.5f}")

    converged = (best_iter < max_iter - 1)
    n_iter_run = min(it + 1, max_iter)

    if verbose:
        print(f"\n  ADVI converged: {converged} "
              f"(best ELBO {best_elbo:.2f} at iter {best_iter}/{n_iter_run})")

    # ---- Restore best variational parameters ----
    mu.data.copy_(best_mu)
    if full_rank:
        L_diag.data.copy_(best_L_diag)
        L_offdiag.data.copy_(best_L_offdiag)
    else:
        log_sigma.data.copy_(best_log_sigma)

    # ---- Draw posterior samples from fitted q(θ) ----
    if verbose:
        print(f"\n  Drawing {n_samples} posterior samples from q(θ)...")

    n_grid_pe = 50
    eff_ll_dim = n_features + 1
    samples_eff_ll = np.zeros((n_samples, eff_ll_dim), dtype=np.float64)
    samples_alpha_arr = np.zeros(n_samples, dtype=np.float64)
    samples_gamma_arr = np.zeros(n_samples, dtype=np.float64)
    samples_pe = {k: np.zeros((n_samples, n_grid_pe), dtype=np.float64)
                  for k in range(n_features)}
    log_post_trace_samples = []

    with torch.no_grad():
        if full_rank:
            L_mat = _build_L()

        for s in range(n_samples):
            eps = torch.randn(d, dtype=torch.float64, device=dev)

            if full_rank:
                theta = mu + L_mat @ eps
            else:
                theta = mu + torch.exp(log_sigma) * eps

            _unflatten(theta, sampled_params)

            ell = net.get_effective_loglinear()
            samples_eff_ll[s] = ell
            samples_alpha_arr[s] = alpha.item()
            samples_gamma_arr[s] = F.softplus(raw_gamma).item()

            pe = net.get_partial_effects(n_grid=n_grid_pe, posterior_cov=None)
            for k in range(n_features):
                samples_pe[k][s] = pe[k]["effect"]

    # Restore MAP params for any downstream use
    _unflatten(best_mu, sampled_params)

    # ---- ESS (for VI samples these are independent, so ESS ≈ n_samples) ----
    ess = {}
    basis_names = ["canopy", "impervious", "water", "fence", "elevation"]
    for k in range(eff_ll_dim):
        nm = "r_0" if k == 0 else (
            f"z_{k}_{basis_names[k-1]}" if k <= len(basis_names)
            else f"z_{k}")
        ess[nm] = float(n_samples)  # iid samples from q
    ess["alpha"] = float(n_samples)
    ess["gamma"] = float(n_samples)

    # ---- Posterior summary ----
    summary = {}
    for k in range(eff_ll_dim):
        nm = "r_0" if k == 0 else (
            f"z_{k}_{basis_names[k-1]}" if k <= len(basis_names)
            else f"z_{k}")
        chain = samples_eff_ll[:, k]
        summary[nm] = {
            "mean": float(np.mean(chain)),
            "sd": float(np.std(chain)),
            "q025": float(np.percentile(chain, 2.5)),
            "q50": float(np.median(chain)),
            "q975": float(np.percentile(chain, 97.5)),
            "ess": float(n_samples),
        }
    for nm, chain in [("alpha", samples_alpha_arr),
                      ("gamma", samples_gamma_arr)]:
        summary[nm] = {
            "mean": float(np.mean(chain)),
            "sd": float(np.std(chain)),
            "q025": float(np.percentile(chain, 2.5)),
            "q50": float(np.median(chain)),
            "q975": float(np.percentile(chain, 97.5)),
            "ess": float(n_samples),
        }

    # ---- Partial effects credible bands ----
    pe_grid = np.linspace(0.0, 1.0, n_grid_pe)
    partial_effects_summary = {}
    for k in range(n_features):
        chain_pe = samples_pe[k]
        partial_effects_summary[int(k)] = {
            "grid": pe_grid.tolist(),
            "effect": np.mean(chain_pe, axis=0).tolist(),
            "se": np.std(chain_pe, axis=0).tolist(),
            "lower_95": np.percentile(chain_pe, 2.5, axis=0).tolist(),
            "upper_95": np.percentile(chain_pe, 97.5, axis=0).tolist(),
            "lower_50": np.percentile(chain_pe, 25, axis=0).tolist(),
            "upper_50": np.percentile(chain_pe, 75, axis=0).tolist(),
        }

    # ---- Save checkpoint ----
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_dict = {
            "effective_loglinear": samples_eff_ll,
            "alpha": samples_alpha_arr,
            "gamma": samples_gamma_arr,
            "elbo_trace": np.array(elbo_trace),
            "variational_mean": best_mu.cpu().numpy(),
        }
        if full_rank:
            save_dict["variational_L_diag"] = best_L_diag.cpu().numpy()
            save_dict["variational_L_offdiag"] = best_L_offdiag.cpu().numpy()
        else:
            save_dict["variational_log_sigma"] = best_log_sigma.cpu().numpy()
        np.savez_compressed(
            os.path.join(output_dir, "advi_samples.npz"), **save_dict)

    if verbose:
        print("\n  Posterior summary (ADVI):")
        print(f"  {'Param':>20}  {'Mean':>8}  {'SD':>8}  "
              f"{'2.5%':>8}  {'97.5%':>8}")
        print(f"  {'-' * 55}")
        for nm, s in summary.items():
            print(f"  {nm:>20}  {s['mean']:8.4f}  {s['sd']:8.4f}  "
                  f"{s['q025']:8.4f}  {s['q975']:8.4f}")

    # ---- Cleanup ----
    cs.CG_RTOL = original_rtol

    total_time = time.time() - t0_all
    if verbose:
        print(f"\n  ADVI complete: {n_iter_run} iterations + "
              f"{n_samples} samples in {total_time:.1f}s "
              f"({total_time / 60:.1f} min)")

    return {
        "samples_effective_loglinear": samples_eff_ll.tolist(),
        "samples_alpha": samples_alpha_arr.tolist(),
        "samples_gamma": samples_gamma_arr.tolist(),
        "partial_effects": partial_effects_summary,
        "log_posterior_trace": [float(x) for x in elbo_trace],
        "log_likelihood_trace": [float(x) for x in elbo_logp_trace],
        "elbo_logp_trace": [float(x) for x in elbo_logp_trace],
        "elbo_entropy_trace": [float(x) for x in elbo_entropy_trace],
        "ess": {str(k): float(v) for k, v in ess.items()},
        "summary": {str(k): {str(kk): float(vv) for kk, vv in v.items()}
                    for k, v in summary.items()},
        "n_samples": int(n_samples),
        "burn_in": int(n_iter_run),  # Compat: report optimization iters
        "thin": 1,
        "step_size": float(lr),
        "acceptance_rate": 1.0,  # Not applicable for VI
        "use_mala": False,
        "elapsed_time": float(total_time),
        # ADVI-specific
        "elbo_trace": [float(x) for x in elbo_trace],
        "converged": converged,
        "variational_mean": best_mu.cpu().numpy().tolist(),
        "variational_std": (
            torch.exp(best_log_sigma).cpu().numpy().tolist()
            if not full_rank
            else torch.exp(best_L_diag).cpu().numpy().tolist()
        ),
        "full_rank": full_rank,
        "n_iter": int(n_iter_run),
        "best_elbo": float(best_elbo),
    }
