"""Full Bayesian posterior sampling via Langevin Monte Carlo (MALA / ULA).

See run_langevin_sampling() for the algorithm description. Shares checkpoint
loading / network setup with run_hmc_sampling and run_advi via
_setup_sampling_state() (samplers/common.py); see that function's docstring
and run_langevin_sampling's body for the one behavioural difference it adds
on top of the shared setup (cs.CG_WARM_START = True).
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
from .common import _compute_ess_chain, _setup_sampling_state

try:
    import cupy as cp
except ImportError:
    cp = None


# ===========================================================================
# Full Bayesian inference via Langevin Monte Carlo
# ===========================================================================

def run_langevin_sampling(
    basis_values_np,             # (n_valid, n_feat) float64
    obs_counts_np,               # (n_valid,) float64
    n_rows,                      # int
    n_cols,                      # int
    valid_mask_np,               # (n_cells,) bool
    cell_area,                   # float (m²)
    # Pre-trained model checkpoint (from run_torch_optimization output_dir)
    model_path=None,             # Path to resistance_nn.pt
    # Alternatively, pass the state dict directly (for in-session use)
    model_state=None,            # dict with 'net', 'alpha', 'gamma', etc.
    # Circuit solver settings (must match optimization)
    source_spacing=1,
    source_from_resistance=True,
    solver="global_absorption",
    absorption=0.002,
    # Sampler settings
    n_samples=2000,              # Posterior samples to collect
    burn_in=500,                 # Burn-in iterations (discarded)
    thin=5,                      # Thinning: collect every thin-th sample
    step_size=1e-5,              # Langevin step size (auto if None)
    precondition=True,           # Use Adam second-moment preconditioner
    precondition_warmup=50,      # Steps to build preconditioner before sampling
    fix_smoothing=False,         # Fix smoothing params λ_k at MAP
    fix_intensity=False,         # Fix alpha, gamma at MAP
    use_mala=True,               # True = MALA (MH correction); False = ULA
    target_accept=0.574,         # Target acceptance rate for adaptive step size
    # Regularization (must match optimization)
    reg_mean=1.0,
    log_R_baseline=3.0,
    penalty_scale=5.0,
    # Model architecture (must match optimization)
    n_knots=10,
    spline_degree=3,
    include_interactions=True,
    lambda_min=0.1,
    # Misc
    grad_clip=100.0,             # Per-param gradient clipping (safety)
    cg_tol=DEFAULT_CG_TOL,
    seed=42,
    verbose=True,
    output_dir=None,
    device="auto",               # "auto", "cuda", or "cpu"
):
    """
    Full Bayesian posterior sampling via Metropolis-adjusted Langevin (MALA)
    or unadjusted Langevin (ULA).

    Starts from the MAP estimate found by run_torch_optimization and explores
    the posterior using Langevin dynamics with optional MH correction:

        Proposal: θ* = θ_t + (ε/2) M⁻¹ ∇log p(θ|y) + √ε M^{-1/2} N(0,I)
        Accept:   min(1, p(θ*)/p(θ) × q(θ|θ*)/q(θ*|θ))  [MALA]

    where ∇log p(θ|y) = ∇(log L + log prior) and M is a diagonal
    preconditioner estimated from Adam-like second moments during burn-in.

    The P-spline smoothing penalty acts as the prior:
        log p(θ) ∝ -penalty_scale × Σ_k λ_k β_k^T S_k β_k

    MALA (use_mala=True, default): eliminates ULA discretization bias via
    Metropolis-Hastings correction. Each step costs ~2× forward passes but
    produces an exact posterior sample. Step size is adapted during burn-in
    to target ~57.4% acceptance (MALA optimal in high dimensions).

    ULA (use_mala=False): no accept/reject; faster per step but accumulates
    discretization bias proportional to ε. Only recommended for diagnostics.

    The preconditioner is frozen after burn-in to ensure a stationary
    transition kernel during sampling.

    Parameters
    ----------
    basis_values_np : ndarray (n_valid, n_features)
    obs_counts_np : ndarray (n_valid,)
    n_rows, n_cols : int
    valid_mask_np : ndarray (n_cells,) bool
    cell_area : float
    model_path : str or None
        Path to resistance_nn.pt checkpoint.
    model_state : dict or None
        Alternative to model_path: pass state dict directly.
    n_samples : int
        Number of posterior samples to collect after burn-in.
    burn_in : int
        Burn-in steps (discarded).
    thin : int
        Collect a sample every `thin` steps.
    step_size : float or None
        Langevin step size ε. If None, auto-tuned from posterior curvature.
    precondition : bool
        Use diagonal preconditioner from running second moments.
    use_mala : bool
        If True (default), use Metropolis-adjusted Langevin algorithm.
        If False, use unadjusted Langevin (legacy behavior).
    target_accept : float
        Target acceptance rate for adaptive step size during burn-in.
        Default 0.574 (MALA-optimal for high dimensions).
    fix_smoothing : bool
        Fix smoothing parameters at MAP (simpler; sample only spline coefs).
    fix_intensity : bool
        Fix alpha, gamma at MAP (sample only resistance params).

    Returns
    -------
    dict with:
        samples_effective_loglinear : (n_samples, K+1) array
        samples_alpha : (n_samples,) array
        samples_gamma : (n_samples,) array
        samples_partial_effects : dict[int, (n_samples, n_grid) array]
        log_posterior_trace : list of floats (burn_in + n_samples*thin)
        acceptance_rate : float (meaningful for MALA; always 1.0 for ULA)
        ess : dict of effective sample sizes per parameter
        summary : dict of posterior means, sds, quantiles
        elapsed_time : float (seconds)
    """
    t0_all = time.time()

    # ---- Shared setup (load checkpoint, build net, identify sampled params) ----
    # See _setup_sampling_state() docstring; this is also used by run_hmc_sampling
    # and run_advi. The one behavioral difference Langevin needs on top of the
    # shared setup is CG warm-starting between consecutive proposal evaluations
    # (HMC/ADVI don't request this), so it is set explicitly right after the
    # shared call rather than folded silently into the helper.
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
    param_names = state["param_names"]
    n_total_scalar = state["n_total_scalar"]
    use_absorption = state["use_absorption"]
    use_intensity_spline = state["use_intensity_spline"]
    use_cuda = state["use_cuda"]
    dev = state["dev"]
    cs = state["cs"]
    original_rtol = state["original_rtol"]

    obs_counts_np_local = np.asarray(obs_counts_np, dtype=np.float64)

    # Langevin-specific: warm-start the CG solve from the previous proposal's
    # solution between consecutive forward passes (not used by HMC/ADVI).
    cs.CG_WARM_START = True

    if use_cuda:
        if seed is not None:
            cp.random.seed(seed)
        if verbose:
            print(f"  Device: CUDA (GPU-accelerated circuit solves)")
    else:
        if verbose:
            print(f"  Device: CPU")

    if verbose:
        sampler_name = "MALA" if use_mala else "ULA"
        print(f"\n  Langevin sampler ({sampler_name}): {n_total_scalar} scalar parameters")
        print(f"    Burn-in: {burn_in}, Samples: {n_samples}, Thin: {thin}")
        print(f"    Total steps: {burn_in + n_samples * thin}")
        print(f"    Fix smoothing: {fix_smoothing}, Fix intensity: {fix_intensity}")
        if use_mala:
            print(f"    Target acceptance rate: {target_accept:.3f}")
            print(f"    Adaptive step size during burn-in: ON")

    # ---- Auto step-size ----
    if step_size is None:
        # Heuristic: scale by 1/sqrt(n_obs) — Langevin optimal scaling
        n_obs = float(obs_counts_np_local.sum())
        step_size = 2.0 / (n_obs + n_valid)
        if verbose:
            print(f"    Auto step_size: {step_size:.2e}")

    # ---- Preconditioner: running second moment (Adam-like) ----
    if precondition:
        v_hat = [torch.ones_like(p) for p in sampled_params]
        beta2 = 0.999
    else:
        v_hat = None

    # ---- Forward + loss function (reuses training loop logic) ----
    def compute_loss_and_grad():
        """Compute -log p(θ|y) and its gradient w.r.t. sampled params."""
        for p in sampled_params:
            if p.grad is not None:
                p.grad.zero_()

        # Forward: covariates → R → circuit → C → log λ → loglik
        R, log_R_raw = net.forward_with_log_R(basis_t)

        if use_absorption:
            C = _AbsorptionCircuitSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
                absorption,
            )
        else:
            C = _CircuitSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
            )

        gamma_val = F.softplus(raw_gamma)
        log1p_C = torch.log1p(C.clamp(min=0))
        if use_intensity_spline:
            log_lam = alpha + net.eval_intensity_spline(log1p_C)
        else:
            log_lam = alpha + gamma_val * log1p_C
        ll = _ppp_loglik(log_lam, obs_t, float(cell_area))

        # Prior = smoothing penalty + mean regularization
        mean_logR = log_R_raw.mean()
        pen_mean = n_valid * reg_mean * (mean_logR - log_R_baseline) ** 2
        pen_smooth = penalty_scale * net.smoothing_penalty()
        neg_log_post = -ll + pen_mean + pen_smooth

        neg_log_post.backward()

        return float(neg_log_post.item()), float(ll.item())

    # ---- Storage for samples ----
    n_grid_pe = 50  # Grid points for partial effects
    total_steps = burn_in + n_samples * thin

    log_post_trace = []
    ll_trace = []

    # Pre-allocate sample arrays
    eff_ll_dim = n_features + 1  # [r_0, z_1, ..., z_K]
    samples_eff_ll = np.zeros((n_samples, eff_ll_dim), dtype=np.float64)
    samples_alpha = np.zeros(n_samples, dtype=np.float64)
    samples_gamma = np.zeros(n_samples, dtype=np.float64)
    # Partial effects: (n_samples, n_grid) per covariate
    samples_pe = {k: np.zeros((n_samples, n_grid_pe), dtype=np.float64)
                  for k in range(n_features)}

    # ---- Sampling loop ----
    sample_idx = 0
    n_accept = 0       # Accepted proposals (MALA)
    n_total_mh = 0     # Total MH proposals attempted

    # Adaptive step size state (dual averaging, Hoffman & Gelman 2014)
    mu_da = math.log(10.0 * step_size)   # Anchor point (log(10 × ε_0))
    log_eps = math.log(step_size)
    log_eps_bar = 0.0            # Smoothed log step size (starts at 0 → eps_bar=1)
    adapt_kappa = 0.75           # Relaxation exponent
    adapt_gamma = 0.05           # Shrinkage towards anchor
    adapt_t0 = 10                # Stabilization offset
    H_bar = 0.0                  # Running mean of acceptance stat

    # Frozen preconditioner: snapshot v_hat at end of burn-in
    v_hat_frozen = None

    if verbose:
        header = (f"  {'Step':>6}  {'NegLogPost':>12}  {'LogLik':>10}  "
                  f"{'StepSize':>10}")
        if use_mala:
            header += f"  {'Accept%':>8}"
        header += f"  {'Time':>6}"
        print(f"\n{header}")
        print(f"  {'-' * (65 if use_mala else 55)}")

    for step in range(total_steps):
        t0 = time.time()

        # Determine whether to update preconditioner (only during burn-in)
        update_precond = precondition and (step < burn_in) and v_hat is not None
        use_frozen = precondition and (step >= burn_in) and v_hat_frozen is not None

        # Compute gradient of negative log-posterior at current θ
        neg_lp, ll_val = compute_loss_and_grad()
        log_post_trace.append(-neg_lp)
        ll_trace.append(ll_val)

        # ---- Langevin proposal ----
        eps = step_size
        accepted = True   # Default for ULA (always accept)

        if use_mala:
            # -----------------------------------------------------------------
            # MALA: propose θ* via Langevin, evaluate log p(θ*), accept/reject
            # IMPORTANT: compute_loss_and_grad() calls .backward() so it must
            # be called OUTSIDE any torch.no_grad() block.
            # -----------------------------------------------------------------

            # Step 1: collect current grads + update/read preconditioner
            with torch.no_grad():
                theta_old = [p.data.clone() for p in sampled_params]
                grads_old = []
                precond_list = []

                for i, p in enumerate(sampled_params):
                    if p.grad is None:
                        grads_old.append(torch.zeros_like(p))
                        precond_list.append(1.0)
                        continue

                    g = p.grad.clone()
                    g_norm = g.norm()
                    if g_norm > grad_clip:
                        g = g * (grad_clip / g_norm)
                    grads_old.append(g)

                    # Update or read preconditioner
                    if update_precond:
                        v_hat[i] = beta2 * v_hat[i] + (1 - beta2) * g * g
                        v_corrected = v_hat[i] / (1 - beta2 ** (step + 1))
                        pc = 1.0 / (torch.sqrt(v_corrected) + 1e-8)
                    elif use_frozen:
                        pc = v_hat_frozen[i]
                    elif precondition and v_hat is not None:
                        v_hat[i] = beta2 * v_hat[i] + (1 - beta2) * g * g
                        v_corrected = v_hat[i] / (1 - beta2 ** (step + 1))
                        pc = 1.0 / (torch.sqrt(v_corrected) + 1e-8)
                    else:
                        pc = 1.0
                    precond_list.append(pc)

                # Step 2: Langevin proposal  θ* = θ - (ε/2) M⁻¹ g + √ε M^{-1/2} z
                for i, p in enumerate(sampled_params):
                    noise = torch.randn_like(p)
                    pc = precond_list[i]
                    g = grads_old[i]
                    if isinstance(pc, float):
                        p.data -= (eps / 2) * g
                        p.data += math.sqrt(eps) * noise
                    else:
                        p.data -= (eps / 2) * pc * g
                        p.data += math.sqrt(eps) * torch.sqrt(pc) * noise

            # Step 3: evaluate log p(θ*) and its gradient (needs autograd)
            neg_lp_star, _ = compute_loss_and_grad()

            # Step 4: MH ratio + accept/reject (no autograd needed)
            with torch.no_grad():
                # Collect proposal grads for reverse proposal density
                grads_star = []
                for p in sampled_params:
                    if p.grad is None:
                        grads_star.append(torch.zeros_like(p))
                    else:
                        g = p.grad.clone()
                        g_norm = g.norm()
                        if g_norm > grad_clip:
                            g = g * (grad_clip / g_norm)
                        grads_star.append(g)

                # Log-proposal densities for MH correction
                # q(θ*|θ) = N(θ* ; θ - (ε/2) M⁻¹ g(θ), ε M⁻¹)
                # log q ∝ -1/(2ε) Σ_i (1/M⁻¹_ii)(θ*_i - μ_i)²
                log_q_forward = 0.0  # log q(θ*|θ)
                log_q_reverse = 0.0  # log q(θ|θ*)

                for i in range(len(sampled_params)):
                    theta_star_i = sampled_params[i].data
                    theta_old_i = theta_old[i]
                    pc = precond_list[i]

                    if isinstance(pc, float):
                        mu_fwd = theta_old_i - (eps / 2) * grads_old[i]
                        mu_rev = theta_star_i - (eps / 2) * grads_star[i]
                        diff_fwd = theta_star_i - mu_fwd
                        diff_rev = theta_old_i - mu_rev
                        log_q_forward += -0.5 / eps * (diff_fwd * diff_fwd).sum().item()
                        log_q_reverse += -0.5 / eps * (diff_rev * diff_rev).sum().item()
                    else:
                        mu_fwd = theta_old_i - (eps / 2) * pc * grads_old[i]
                        mu_rev = theta_star_i - (eps / 2) * pc * grads_star[i]
                        inv_pc = 1.0 / (pc + 1e-30)
                        diff_fwd = theta_star_i - mu_fwd
                        diff_rev = theta_old_i - mu_rev
                        log_q_forward += (-0.5 / eps * (inv_pc * diff_fwd * diff_fwd).sum()).item()
                        log_q_reverse += (-0.5 / eps * (inv_pc * diff_rev * diff_rev).sum()).item()

                # log α = [log p(θ*) - log p(θ)] + [log q(θ|θ*) - log q(θ*|θ)]
                log_alpha = (-neg_lp_star + neg_lp) + (log_q_reverse - log_q_forward)
                log_alpha = min(0.0, log_alpha)

                u = math.log(max(np.random.uniform(), 1e-300))
                accepted = (u < log_alpha)
                n_total_mh += 1

                if accepted:
                    n_accept += 1
                else:
                    # Reject: restore θ_old
                    for i, p in enumerate(sampled_params):
                        p.data.copy_(theta_old[i])

            # Step 5: adaptive step size during burn-in (dual averaging)
            if step < burn_in:
                accept_prob = min(1.0, math.exp(log_alpha))
                m = step + 1
                H_bar = (1.0 - 1.0 / (m + adapt_t0)) * H_bar + \
                        (1.0 / (m + adapt_t0)) * (target_accept - accept_prob)
                log_eps = mu_da - math.sqrt(m) / adapt_gamma * H_bar
                eta = m ** (-adapt_kappa)
                log_eps_bar = eta * log_eps + (1.0 - eta) * log_eps_bar
                step_size = math.exp(log_eps)
                step_size = max(1e-10, min(step_size, 1.0))

        else:
            # -----------------------------------------------------------------
            # ULA: unconditional update (legacy, no MH correction)
            # -----------------------------------------------------------------
            with torch.no_grad():
                for i, p in enumerate(sampled_params):
                    if p.grad is None:
                        continue

                    g = p.grad.clone()
                    g_norm = g.norm()
                    if g_norm > grad_clip:
                        g = g * (grad_clip / g_norm)

                    if update_precond:
                        v_hat[i] = beta2 * v_hat[i] + (1 - beta2) * g * g
                        v_corrected = v_hat[i] / (1 - beta2 ** (step + 1))
                        pc = 1.0 / (torch.sqrt(v_corrected) + 1e-8)
                    elif use_frozen:
                        pc = v_hat_frozen[i]
                    elif precondition and v_hat is not None:
                        v_hat[i] = beta2 * v_hat[i] + (1 - beta2) * g * g
                        v_corrected = v_hat[i] / (1 - beta2 ** (step + 1))
                        pc = 1.0 / (torch.sqrt(v_corrected) + 1e-8)
                    else:
                        pc = 1.0

                    noise = torch.randn_like(p)
                    if isinstance(pc, float):
                        p.data -= (eps / 2) * g
                        p.data += math.sqrt(eps) * noise
                    else:
                        p.data -= (eps / 2) * pc * g
                        p.data += math.sqrt(eps) * torch.sqrt(pc) * noise

        # ---- Freeze preconditioner at end of burn-in ----
        if step == burn_in - 1 and precondition and v_hat is not None:
            # Convert raw second moments → actual preconditioner (1/sqrt)
            # so that post-burn-in sampling uses the same pc as burn-in
            v_hat_frozen = []
            for i in range(len(v_hat)):
                v_corrected = v_hat[i] / (1 - beta2 ** burn_in)
                v_hat_frozen.append(
                    (1.0 / (torch.sqrt(v_corrected) + 1e-8)).clone()
                )
            if use_mala:
                # Lock step size at smoothed value from dual averaging
                step_size = math.exp(log_eps_bar)
                step_size = max(1e-10, min(step_size, 1.0))
            if verbose:
                print(f"\n  >>> Burn-in complete at step {step + 1}")
                print(f"      Preconditioner frozen")
                if use_mala:
                    acc_rate = n_accept / max(1, n_total_mh)
                    print(f"      Burn-in acceptance rate: {acc_rate:.3f}")
                    print(f"      Adapted step size: {step_size:.2e}")
                print()

        # ---- Collect sample ----
        is_past_burnin = (step >= burn_in)
        is_thin_step = ((step - burn_in) % thin == 0) if is_past_burnin else False

        if is_past_burnin and is_thin_step and sample_idx < n_samples:
            with torch.no_grad():
                # Effective log-linear coefficients
                ell = net.get_effective_loglinear()
                samples_eff_ll[sample_idx] = ell

                # Intensity params
                samples_alpha[sample_idx] = alpha.item()
                samples_gamma[sample_idx] = F.softplus(raw_gamma).item()

                # Partial effects on grid
                pe = net.get_partial_effects(n_grid=n_grid_pe, posterior_cov=None)
                for k in range(n_features):
                    samples_pe[k][sample_idx] = pe[k]["effect"]

            sample_idx += 1

        # ---- Logging ----
        elapsed = time.time() - t0
        if verbose and (step % 100 == 0 or step < 5
                        or step == burn_in
                        or step == total_steps - 1):
            phase = "burn-in" if step < burn_in else f"sample {sample_idx}/{n_samples}"
            line = (f"  {step:6d}  {neg_lp:12.2f}  {ll_val:10.2f}  "
                    f"{step_size:10.2e}")
            if use_mala:
                acc_pct = 100.0 * n_accept / max(1, n_total_mh)
                line += f"  {acc_pct:7.1f}%"
            line += f"  {elapsed:5.2f}s  [{phase}]"
            if use_mala and not accepted:
                line += " [REJ]"
            print(line)

    # ---- Restore CG tolerance and disable GPU ----
    cs.CG_RTOL = original_rtol
    cs.CG_WARM_START = False
    cs.enable_gpu(False)

    total_time = time.time() - t0_all
    acceptance_rate = n_accept / max(1, n_total_mh) if use_mala else 1.0
    if verbose:
        print(f"\n  Sampling complete: {sample_idx} samples in {total_time:.1f}s "
              f"({total_time / 60:.1f} min)")
        if use_mala:
            print(f"  Final acceptance rate: {acceptance_rate:.3f}")
            print(f"  Final step size: {step_size:.2e}")

    # ---- Compute ESS (effective sample size) via autocorrelation ----
    # Shared with run_hmc_sampling / run_advi — see _compute_ess_chain().
    ess = {}
    basis_names = ["canopy", "impervious", "water", "fence", "elevation"]
    for k in range(eff_ll_dim):
        nm = f"r_0" if k == 0 else (
            f"z_{k}_{basis_names[k-1]}" if k <= len(basis_names)
            else f"z_{k}")
        ess[nm] = float(_compute_ess_chain(samples_eff_ll[:sample_idx, k]))
    ess["alpha"] = float(_compute_ess_chain(samples_alpha[:sample_idx]))
    ess["gamma"] = float(_compute_ess_chain(samples_gamma[:sample_idx]))

    # ---- Posterior summary ----
    n_collected = sample_idx
    summary = {}
    for k in range(eff_ll_dim):
        nm = f"r_0" if k == 0 else (
            f"z_{k}_{basis_names[k-1]}" if k <= len(basis_names)
            else f"z_{k}")
        chain = samples_eff_ll[:n_collected, k]
        summary[nm] = {
            "mean": float(np.mean(chain)),
            "sd": float(np.std(chain)),
            "q025": float(np.percentile(chain, 2.5)),
            "q50": float(np.median(chain)),
            "q975": float(np.percentile(chain, 97.5)),
            "ess": ess[nm],
        }
    for nm, chain in [("alpha", samples_alpha[:n_collected]),
                      ("gamma", samples_gamma[:n_collected])]:
        summary[nm] = {
            "mean": float(np.mean(chain)),
            "sd": float(np.std(chain)),
            "q025": float(np.percentile(chain, 2.5)),
            "q50": float(np.median(chain)),
            "q975": float(np.percentile(chain, 97.5)),
            "ess": ess[nm],
        }

    # ---- Partial effect credible bands (pointwise quantiles) ----
    pe_grid = np.linspace(0.0, 1.0, n_grid_pe)
    partial_effects_summary = {}
    for k in range(n_features):
        chain_pe = samples_pe[k][:n_collected]  # (n_collected, n_grid)
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
        np.savez_compressed(
            os.path.join(output_dir, "mcmc_samples.npz"),
            effective_loglinear=samples_eff_ll[:n_collected],
            alpha=samples_alpha[:n_collected],
            gamma=samples_gamma[:n_collected],
            log_posterior=np.array(log_post_trace),
            log_likelihood=np.array(ll_trace),
        )

    if verbose:
        print("\n  Posterior summary (effective log-linear):")
        print(f"  {'Param':>20}  {'Mean':>8}  {'SD':>8}  "
              f"{'2.5%':>8}  {'97.5%':>8}  {'ESS':>6}")
        print(f"  {'-' * 65}")
        for nm, s in summary.items():
            print(f"  {nm:>20}  {s['mean']:8.4f}  {s['sd']:8.4f}  "
                  f"{s['q025']:8.4f}  {s['q975']:8.4f}  {s['ess']:6.0f}")

    return {
        "samples_effective_loglinear": samples_eff_ll[:n_collected].tolist(),
        "samples_alpha": samples_alpha[:n_collected].tolist(),
        "samples_gamma": samples_gamma[:n_collected].tolist(),
        "partial_effects": partial_effects_summary,
        "log_posterior_trace": [float(x) for x in log_post_trace],
        "log_likelihood_trace": [float(x) for x in ll_trace],
        "ess": {str(k): float(v) for k, v in ess.items()},
        "summary": {str(k): {str(kk): float(vv) for kk, vv in v.items()}
                    for k, v in summary.items()},
        "n_samples": int(n_collected),
        "burn_in": int(burn_in),
        "thin": int(thin),
        "step_size": float(step_size),
        "acceptance_rate": float(acceptance_rate),
        "use_mala": bool(use_mala),
        "elapsed_time": float(total_time),
    }
