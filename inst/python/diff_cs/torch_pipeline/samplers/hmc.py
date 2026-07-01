"""Full Bayesian posterior sampling via Hamiltonian Monte Carlo with the
No-U-Turn Sampler (NUTS).

See run_hmc_sampling() for the algorithm description. Shares checkpoint
loading / network setup with run_langevin_sampling and run_advi via
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
from .common import _compute_ess_chain, _setup_sampling_state


def run_hmc_sampling(
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
    # HMC / NUTS settings
    n_samples=1000,
    warmup=1000,
    max_treedepth=10,
    target_accept=0.80,
    init_step_size=None,
    adapt_mass_matrix=True,
    fix_smoothing=True,
    fix_intensity=False,
    # Regularization
    reg_mean=1.0,
    log_R_baseline=3.0,
    penalty_scale=5.0,
    # Architecture
    n_knots=10,
    spline_degree=3,
    include_interactions=False,
    lambda_min=0.1,
    # Misc
    grad_clip=100.0,
    cg_tol=DEFAULT_CG_TOL,
    seed=42,
    verbose=True,
    output_dir=None,
    device="auto",
):
    """
    Full Bayesian posterior sampling via No-U-Turn Sampler (NUTS).

    Uses Hamiltonian Monte Carlo with NUTS tree doubling (Hoffman & Gelman
    2014) for automatic trajectory length selection.  Step size is adapted
    via dual averaging during warmup, and an optional diagonal mass matrix
    is estimated from warmup samples.

    The sampler starts from the MAP estimate saved by run_torch_optimization
    and differentiates through the full pipeline:
        basis -> SplineResistanceNet -> R -> circuit solve -> C -> PPP loglik

    Parameters
    ----------
    n_samples : int
        Posterior samples to collect after warmup.
    warmup : int
        Warmup iterations (adaptation; discarded).
    max_treedepth : int
        Maximum NUTS tree depth (2^d leapfrog steps worst case).
    target_accept : float
        Target MH acceptance probability (0.80 recommended for NUTS).
    init_step_size : float or None
        Initial leapfrog step size (None = auto from posterior scale).
    adapt_mass_matrix : bool
        Adapt diagonal mass matrix from warmup variance.

    Returns
    -------
    dict — same structure as run_langevin_sampling plus:
        treedepth_trace : list of int (tree depth per sample)
        n_divergences : int (should be 0)
        energy_trace : list of float (Hamiltonian energy per sample)
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
    use_cuda = state["use_cuda"]
    dev = state["dev"]
    cs = state["cs"]
    original_rtol = state["original_rtol"]
    n_total_scalar = state["n_total_scalar"]

    if verbose:
        print(f"\n  NUTS sampler: {n_total_scalar} scalar parameters")
        print(f"    Warmup: {warmup}, Samples: {n_samples}")
        print(f"    Max tree depth: {max_treedepth}")
        print(f"    Target acceptance: {target_accept:.3f}")
        print(f"    Fix smoothing: {fix_smoothing}, Fix intensity: {fix_intensity}")
        print(f"    Adapt mass matrix: {adapt_mass_matrix}")

    # ---- Flatten / unflatten utilities ----
    def _flatten(params):
        return torch.cat([p.data.reshape(-1) for p in params])

    def _flatten_grad(params):
        gs = []
        for p in params:
            if p.grad is not None:
                g = p.grad.reshape(-1).clone()
            else:
                g = torch.zeros(p.numel(), dtype=torch.float64, device=dev)
            gs.append(g)
        return torch.cat(gs)

    def _unflatten(flat, params):
        offset = 0
        for p in params:
            n = p.numel()
            p.data.copy_(flat[offset:offset + n].reshape(p.shape))
            offset += n

    # ---- Potential energy (negative log-posterior) ----
    def compute_U():
        """Return U(θ) and populate .grad on sampled_params."""
        for p in sampled_params:
            if p.grad is not None:
                p.grad.zero_()

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
        log1p_C = torch.log1p(C.clamp(min=0))
        if use_intensity_spline:
            log_lam = alpha + net.eval_intensity_spline(log1p_C)
        else:
            log_lam = alpha + gamma_val * log1p_C
        ll = _ppp_loglik(log_lam, obs_t, float(cell_area))

        mean_logR = log_R_raw.mean()
        pen_mean = n_valid * reg_mean * (mean_logR - log_R_baseline) ** 2
        pen_smooth = penalty_scale * net.smoothing_penalty()
        U = -ll + pen_mean + pen_smooth

        U.backward()
        return float(U.item()), float(ll.item())

    # ---- Leapfrog integrator ----
    def leapfrog(q0, p0, grad_U0, eps, inv_mass):
        """Single leapfrog trajectory of L=1 step.
        Returns q1, p1, grad_U1, U1, ll1.

        NOTE: No gradient clipping here — clipping biases the Hamiltonian
        dynamics and produces incorrect posterior samples.  If gradients
        explode, the root cause is numerical instability (fix cg_tol or
        regularisation instead)."""
        # Half-step momentum
        p_half = p0 - 0.5 * eps * grad_U0

        # Full-step position
        q1 = q0 + eps * inv_mass * p_half
        _unflatten(q1, sampled_params)

        # Gradient at new position
        U1, ll1 = compute_U()
        grad_U1 = _flatten_grad(sampled_params)

        # Half-step momentum
        p1 = p_half - 0.5 * eps * grad_U1

        return q1, p1, grad_U1, U1, ll1

    # ---- Kinetic energy ----
    def kinetic_energy(p, inv_mass):
        """K(p) = 0.5 p^T M^{-1} p."""
        return 0.5 * (p * inv_mass * p).sum().item()

    # ---- NUTS tree building (iterative, Hoffman & Gelman 2014 Algorithm 6) ----
    DIVERGENCE_THRESHOLD = 1000.0

    def _build_tree(q, p, grad_U, log_u_slice, direction, depth, eps, inv_mass,
                    H0):
        """Recursively build NUTS tree.
        Returns (q_minus, p_minus, grad_minus, q_plus, p_plus, grad_plus,
                 q_prime, grad_prime, U_prime, ll_prime, n_valid_states,
                 is_valid, alpha_sum, n_alpha, n_leapfrog).

        grad_prime is the gradient at q_prime so the caller can reuse it
        without an extra compute_U() call.

        log_u_slice: log of the slice variable (log-space to avoid overflow)."""
        if depth == 0:
            # Base case: single leapfrog step
            q_new, p_new, grad_new, U_new, ll_new = leapfrog(
                q, p, grad_U, direction * eps, inv_mass)
            H_new = U_new + kinetic_energy(p_new, inv_mass)
            delta_H = H_new - H0
            # Check divergence (include NaN/inf guard)
            divergent = (delta_H > DIVERGENCE_THRESHOLD) or \
                math.isnan(delta_H) or math.isinf(delta_H)
            # Slice check (log-space: log_u <= -(H_new - H0) = H0 - H_new)
            valid = (log_u_slice <= -delta_H) and (not divergent)
            n_valid_states = 1 if valid else 0
            # Acceptance stat for dual averaging
            log_accept = min(0.0, -delta_H)
            accept_prob = math.exp(log_accept) if not math.isnan(log_accept) else 0.0
            return (q_new, p_new, grad_new,
                    q_new, p_new, grad_new,
                    q_new, grad_new, U_new, ll_new,
                    n_valid_states, not divergent,
                    accept_prob, 1, 1)

        # Recursion: build left half
        (q_minus, p_minus, grad_minus, q_plus, p_plus, grad_plus,
         q_prime, grad_prime, U_prime, ll_prime,
         n_prime, is_valid_prime,
         alpha_prime, n_alpha_prime,
         n_lf_prime) = _build_tree(
            q, p, grad_U, log_u_slice, direction, depth - 1, eps, inv_mass, H0)

        if not is_valid_prime:
            return (q_minus, p_minus, grad_minus, q_plus, p_plus, grad_plus,
                    q_prime, grad_prime, U_prime, ll_prime,
                    n_prime, False, alpha_prime, n_alpha_prime, n_lf_prime)

        # Build right half from the appropriate end
        if direction == -1:
            (q_minus, p_minus, grad_minus, _, _, _,
             q_dprime, grad_dprime, U_dprime, ll_dprime,
             n_dprime, is_valid_dprime,
             alpha_dprime, n_alpha_dprime,
             n_lf_dprime) = _build_tree(
                q_minus, p_minus, grad_minus, log_u_slice, direction,
                depth - 1, eps, inv_mass, H0)
        else:
            (_, _, _, q_plus, p_plus, grad_plus,
             q_dprime, grad_dprime, U_dprime, ll_dprime,
             n_dprime, is_valid_dprime,
             alpha_dprime, n_alpha_dprime,
             n_lf_dprime) = _build_tree(
                q_plus, p_plus, grad_plus, log_u_slice, direction,
                depth - 1, eps, inv_mass, H0)

        # Multinomial sampling: accept q_dprime with prob n'' / (n' + n'')
        n_total = n_prime + n_dprime
        if n_dprime > 0 and n_total > 0:
            accept_dprime = n_dprime / n_total
            if np.random.uniform() < accept_dprime:
                q_prime = q_dprime
                grad_prime = grad_dprime
                U_prime = U_dprime
                ll_prime = ll_dprime

        # U-turn check
        dq = q_plus - q_minus
        is_valid = is_valid_dprime and \
            (dq.dot(p_minus) >= 0) and (dq.dot(p_plus) >= 0)

        return (q_minus, p_minus, grad_minus, q_plus, p_plus, grad_plus,
                q_prime, grad_prime, U_prime, ll_prime,
                n_total, is_valid,
                alpha_prime + alpha_dprime,
                n_alpha_prime + n_alpha_dprime,
                n_lf_prime + n_lf_dprime)

    # ---- Mass matrix (diagonal) ----
    inv_mass = torch.ones(n_total_scalar, dtype=torch.float64, device=dev)
    # During warmup we collect samples for mass matrix adaptation
    warmup_samples_for_mass = []

    # ---- CG tolerance: full precision throughout ----
    # Relaxed CG during warmup introduces gradient noise that corrupts
    # dual-averaging step-size adaptation and Hamiltonian dynamics.
    cg_tol_sampling = cg_tol

    # Enable CG warm-start for faster solves during NUTS
    cs.CG_WARM_START = True

    # ---- Initial gradient (needed before step size search) ----
    cs.CG_RTOL = cg_tol_sampling
    U_curr, ll_curr = compute_U()
    grad_curr = _flatten_grad(sampled_params)
    q_curr = _flatten(sampled_params)

    # ---- Auto step size (bisection to ~65% acceptance on single leapfrog) ----
    if init_step_size is None:
        # Start at 0.01 and halve/double to find eps giving ~65% acceptance
        test_eps = 0.01
        p_test = torch.randn(n_total_scalar, dtype=torch.float64, device=dev)
        H0_test = U_curr + kinetic_energy(p_test, inv_mass)
        _, p1_test, _, U1_test, _ = leapfrog(
            q_curr.clone(), p_test, grad_curr.clone(), test_eps, inv_mass)
        H1_test = U1_test + kinetic_energy(p1_test, inv_mass)
        log_accept_test = -(H1_test - H0_test)
        # If accept prob > 0.5, double eps; otherwise halve — up to 10 iterations
        direction_test = 1 if log_accept_test > math.log(0.65) else -1
        for _ in range(10):
            test_eps_new = test_eps * (2.0 ** direction_test)
            test_eps_new = max(1e-10, min(test_eps_new, 1.0))
            p_test2 = torch.randn(n_total_scalar, dtype=torch.float64, device=dev)
            H0_t2 = U_curr + kinetic_energy(p_test2, inv_mass)
            _, p1_t2, _, U1_t2, _ = leapfrog(
                q_curr.clone(), p_test2, grad_curr.clone(), test_eps_new, inv_mass)
            H1_t2 = U1_t2 + kinetic_energy(p1_t2, inv_mass)
            log_accept_t2 = -(H1_t2 - H0_t2)
            if (direction_test == 1 and log_accept_t2 < math.log(0.65)) or \
               (direction_test == -1 and log_accept_t2 > math.log(0.65)):
                break
            test_eps = test_eps_new
        init_step_size = max(1e-10, min(test_eps, 1.0))
        # Re-evaluate at MAP (leapfrog may have changed params)
        _unflatten(q_curr, sampled_params)
        U_curr, ll_curr = compute_U()
        grad_curr = _flatten_grad(sampled_params)
        if verbose:
            print(f"    Auto init step size: {init_step_size:.2e}")

    step_size = init_step_size

    # ---- Dual averaging state ----
    mu_da = math.log(10.0 * step_size)
    log_eps = math.log(step_size)
    log_eps_bar = 0.0
    adapt_kappa = 0.75
    adapt_gamma_da = 0.05
    adapt_t0 = 10
    H_bar = 0.0

    # ---- Storage ----
    n_grid_pe = 50
    eff_ll_dim = n_features + 1
    samples_eff_ll = np.zeros((n_samples, eff_ll_dim), dtype=np.float64)
    samples_alpha_arr = np.zeros(n_samples, dtype=np.float64)
    samples_gamma_arr = np.zeros(n_samples, dtype=np.float64)
    samples_pe = {k: np.zeros((n_samples, n_grid_pe), dtype=np.float64)
                  for k in range(n_features)}

    log_post_trace = []
    ll_trace = []
    treedepth_trace = []
    energy_trace = []
    n_divergences = 0
    n_leapfrog_total = 0
    sample_idx = 0

    # ---- Mass matrix adaptation windows (Stan-style) ----
    # Window I:   [0, init_window)    — step size adaptation only
    # Window II:  [init_window, term_start) — step size + mass matrix
    # Window III: [term_start, warmup)      — final step size adaptation
    init_window = min(75, warmup // 4)
    term_window = min(50, warmup // 4)
    term_start = warmup - term_window

    total_steps = warmup + n_samples

    if verbose:
        print(f"\n  {'Step':>6}  {'NegLogPost':>12}  {'LogLik':>10}  "
              f"{'StepSize':>10}  {'Depth':>5}  {'nLF':>5}  "
              f"{'Accept':>7}  {'Time':>6}")
        print(f"  {'-' * 75}")

    for step in range(total_steps):
        t0_step = time.time()

        is_warmup = (step < warmup)

        cs.CG_RTOL = cg_tol_sampling

        # Sample momentum
        p_curr = torch.randn(n_total_scalar, dtype=torch.float64,
                              device=dev)
        if adapt_mass_matrix:
            # p ~ N(0, M): scale by sqrt(mass) = 1/sqrt(inv_mass)
            p_curr = p_curr / torch.sqrt(inv_mass)

        # Current Hamiltonian
        H_curr = U_curr + kinetic_energy(p_curr, inv_mass)

        # Slice variable (log-space to avoid overflow when H_curr << 0)
        # log(u) where u ~ Uniform(0, exp(-H_curr))
        # = log(Uniform(0,1)) + (-H_curr) = log(Uniform(0,1)) - H_curr
        log_u_slice = math.log(np.random.uniform()) - H_curr

        # ---- NUTS tree doubling ----
        q_minus = q_curr.clone()
        q_plus = q_curr.clone()
        p_minus = p_curr.clone()
        p_plus = p_curr.clone()
        grad_minus = grad_curr.clone()
        grad_plus = grad_curr.clone()

        q_prop = q_curr.clone()
        grad_prop = grad_curr.clone()
        U_prop = U_curr
        ll_prop = ll_curr
        depth = 0
        n_valid_states = 1
        keep_going = True
        alpha_sum = 0.0
        n_alpha = 0
        n_leapfrog_step = 0
        divergent = False

        while keep_going and depth < max_treedepth:
            # Choose direction
            direction = 1 if np.random.uniform() < 0.5 else -1

            if direction == -1:
                (q_minus, p_minus, grad_minus, _, _, _,
                 q_prime, grad_q_prime, U_prime, ll_prime,
                 n_prime, is_valid,
                 alpha_sub, n_alpha_sub,
                 n_lf_sub) = _build_tree(
                    q_minus, p_minus, grad_minus, log_u_slice,
                    direction, depth, step_size, inv_mass, H_curr)
            else:
                (_, _, _, q_plus, p_plus, grad_plus,
                 q_prime, grad_q_prime, U_prime, ll_prime,
                 n_prime, is_valid,
                 alpha_sub, n_alpha_sub,
                 n_lf_sub) = _build_tree(
                    q_plus, p_plus, grad_plus, log_u_slice,
                    direction, depth, step_size, inv_mass, H_curr)

            if is_valid and n_prime > 0:
                accept_prob = min(1.0, n_prime / n_valid_states)
                if np.random.uniform() < accept_prob:
                    q_prop = q_prime
                    grad_prop = grad_q_prime
                    U_prop = U_prime
                    ll_prop = ll_prime

            n_valid_states += n_prime
            alpha_sum += alpha_sub
            n_alpha += n_alpha_sub
            n_leapfrog_step += n_lf_sub

            # U-turn check on full tree
            dq = q_plus - q_minus
            keep_going = is_valid and \
                (dq.dot(p_minus) >= 0) and (dq.dot(p_plus) >= 0)

            if not is_valid:
                divergent = True
                n_divergences += 1

            depth += 1

        # ---- Accept proposal (NUTS always accepts within tree) ----
        q_curr = q_prop.clone()
        _unflatten(q_curr, sampled_params)

        # Reuse cached U / gradient from the tree (saves one circuit solve).
        U_curr = U_prop
        ll_curr = ll_prop
        grad_curr = grad_prop if grad_prop is not None else _flatten_grad(sampled_params)

        n_leapfrog_total += n_leapfrog_step
        mean_accept = alpha_sum / max(1, n_alpha)

        # Traces
        log_post_trace.append(-U_curr)
        ll_trace.append(ll_curr)
        treedepth_trace.append(depth)
        energy_trace.append(U_curr + kinetic_energy(p_curr, inv_mass))

        # ---- Adaptation during warmup ----
        if is_warmup:
            # Dual averaging for step size
            m = step + 1
            H_bar = (1.0 - 1.0 / (m + adapt_t0)) * H_bar + \
                    (1.0 / (m + adapt_t0)) * (target_accept - mean_accept)
            log_eps = mu_da - math.sqrt(m) / adapt_gamma_da * H_bar
            eta = m ** (-adapt_kappa)
            log_eps_bar = eta * log_eps + (1.0 - eta) * log_eps_bar
            step_size = math.exp(log_eps)
            step_size = max(1e-10, min(step_size, 1.0))

            # Mass matrix adaptation (window II only)
            if adapt_mass_matrix and init_window <= step < term_start:
                warmup_samples_for_mass.append(q_curr.detach().cpu().numpy())
                # Update mass matrix at end of each doubling window
                n_mass = len(warmup_samples_for_mass)
                if n_mass >= 20 and (n_mass & (n_mass - 1) == 0):
                    # Power-of-2 checkpoints: 32, 64, 128, ...
                    stacked = np.stack(warmup_samples_for_mass)
                    var_est = np.var(stacked, axis=0)
                    var_est = np.clip(var_est, 1e-8, 1e8)
                    inv_mass = torch.tensor(
                        1.0 / var_est, dtype=torch.float64, device=dev)
                    if verbose:
                        print(f"  >>> Mass matrix updated (n={n_mass}, "
                              f"var range: [{var_est.min():.2e}, "
                              f"{var_est.max():.2e}])")

            # Final mass matrix update at transition to terminal window
            if adapt_mass_matrix and step == term_start - 1:
                if len(warmup_samples_for_mass) >= 10:
                    stacked = np.stack(warmup_samples_for_mass)
                    var_est = np.var(stacked, axis=0)
                    var_est = np.clip(var_est, 1e-8, 1e8)
                    inv_mass = torch.tensor(
                        1.0 / var_est, dtype=torch.float64, device=dev)
                    if verbose:
                        print(f"  >>> Final mass matrix "
                              f"(n={len(warmup_samples_for_mass)})")
                # Reset step size adaptation for terminal window
                mu_da = math.log(10.0 * step_size)
                log_eps = math.log(step_size)
                log_eps_bar = math.log(step_size)
                H_bar = 0.0

            # Lock step size at end of warmup
            if step == warmup - 1:
                step_size = math.exp(log_eps_bar)
                step_size = max(1e-10, min(step_size, 1.0))
                if verbose:
                    print(f"\n  >>> Warmup complete")
                    print(f"      Adapted step size: {step_size:.2e}")
                    if adapt_mass_matrix and len(warmup_samples_for_mass) > 0:
                        print(f"      Mass matrix from "
                              f"{len(warmup_samples_for_mass)} samples")
                    print(f"      Divergences during warmup: {n_divergences}")
                    print()
                # Reset divergence counter for sampling phase
                n_divergences = 0

        # ---- Collect sample (post-warmup) ----
        if not is_warmup:
            with torch.no_grad():
                ell = net.get_effective_loglinear()
                samples_eff_ll[sample_idx] = ell
                samples_alpha_arr[sample_idx] = alpha.item()
                samples_gamma_arr[sample_idx] = F.softplus(raw_gamma).item()
                pe = net.get_partial_effects(n_grid=n_grid_pe,
                                             posterior_cov=None)
                for k in range(n_features):
                    samples_pe[k][sample_idx] = pe[k]["effect"]
            sample_idx += 1

        # ---- Logging ----
        elapsed_step = time.time() - t0_step
        if verbose and (step % 100 == 0 or step < 5
                        or step == warmup or step == total_steps - 1):
            phase = "warmup" if is_warmup else \
                f"sample {sample_idx}/{n_samples}"
            div_str = " [DIV]" if divergent else ""
            print(f"  {step:6d}  {U_curr:12.2f}  {ll_curr:10.2f}  "
                  f"{step_size:10.2e}  {depth:5d}  "
                  f"{n_leapfrog_step:5d}  {mean_accept:6.3f}  "
                  f"{elapsed_step:5.1f}s  [{phase}]{div_str}")

    # ---- Cleanup ----
    cs.CG_RTOL = original_rtol
    cs.CG_WARM_START = False
    cs.enable_gpu(False)

    total_time = time.time() - t0_all
    if verbose:
        print(f"\n  Sampling complete: {sample_idx} samples in "
              f"{total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"  Post-warmup divergences: {n_divergences}")
        print(f"  Total leapfrog steps: {n_leapfrog_total}")
        print(f"  Mean leapfrog/sample: "
              f"{n_leapfrog_total / max(1, total_steps):.1f}")
        print(f"  Final step size: {step_size:.2e}")

    # ---- ESS ----
    n_collected = sample_idx
    ess = {}
    basis_names = ["canopy", "impervious", "water", "fence", "elevation"]
    for k in range(eff_ll_dim):
        nm = "r_0" if k == 0 else (
            f"z_{k}_{basis_names[k-1]}" if k <= len(basis_names)
            else f"z_{k}")
        ess[nm] = float(_compute_ess_chain(samples_eff_ll[:n_collected, k]))
    ess["alpha"] = float(_compute_ess_chain(samples_alpha_arr[:n_collected]))
    ess["gamma"] = float(_compute_ess_chain(samples_gamma_arr[:n_collected]))

    # ---- Posterior summary ----
    summary = {}
    for k in range(eff_ll_dim):
        nm = "r_0" if k == 0 else (
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
    for nm, chain in [("alpha", samples_alpha_arr[:n_collected]),
                      ("gamma", samples_gamma_arr[:n_collected])]:
        summary[nm] = {
            "mean": float(np.mean(chain)),
            "sd": float(np.std(chain)),
            "q025": float(np.percentile(chain, 2.5)),
            "q50": float(np.median(chain)),
            "q975": float(np.percentile(chain, 97.5)),
            "ess": ess[nm],
        }

    # ---- Partial effects credible bands ----
    pe_grid = np.linspace(0.0, 1.0, n_grid_pe)
    partial_effects_summary = {}
    for k in range(n_features):
        chain_pe = samples_pe[k][:n_collected]
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
            os.path.join(output_dir, "hmc_samples.npz"),
            effective_loglinear=samples_eff_ll[:n_collected],
            alpha=samples_alpha_arr[:n_collected],
            gamma=samples_gamma_arr[:n_collected],
            log_posterior=np.array(log_post_trace),
            log_likelihood=np.array(ll_trace),
            treedepth=np.array(treedepth_trace),
            energy=np.array(energy_trace),
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
        "samples_alpha": samples_alpha_arr[:n_collected].tolist(),
        "samples_gamma": samples_gamma_arr[:n_collected].tolist(),
        "partial_effects": partial_effects_summary,
        "log_posterior_trace": [float(x) for x in log_post_trace],
        "log_likelihood_trace": [float(x) for x in ll_trace],
        "ess": {str(k): float(v) for k, v in ess.items()},
        "summary": {str(k): {str(kk): float(vv) for kk, vv in v.items()}
                    for k, v in summary.items()},
        "n_samples": int(n_collected),
        "burn_in": int(warmup),  # Compat key
        "thin": 1,               # No thinning in NUTS
        "step_size": float(step_size),
        "acceptance_rate": float(
            sum(treedepth_trace[warmup:]) / max(1, len(treedepth_trace[warmup:]))
            if len(treedepth_trace) > warmup else 0.0),
        "use_mala": False,
        "elapsed_time": float(total_time),
        # HMC-specific
        "treedepth_trace": [int(d) for d in treedepth_trace],
        "n_divergences": int(n_divergences),
        "energy_trace": [float(e) for e in energy_trace],
    }
