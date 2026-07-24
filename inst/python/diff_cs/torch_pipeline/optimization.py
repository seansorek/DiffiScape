"""MAP optimisation entry point: run_torch_optimization.

End-to-end neural-network/spline/IRL resistance + differentiable circuit
solve + Poisson point-process likelihood, trained with Adam.
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import DEFAULT_CG_TOL, _GPU_AVAILABLE, _CUPY_AVAILABLE
from ._module_loaders import _get_circuit_module, _get_diff_omniscape_module
from .autograd_functions import (
    _CircuitSolveFn,
    _AbsorptionCircuitSolveFn,
    _DiffOmniscapeSolveFn,
)
from .resistance_nets import (
    LogLinearResistanceNet, ResistanceNet, ConvResistanceNet, IRLResistanceNet,
    SplineResistanceNet,
    _ppp_loglik,
)

try:
    import cupy as cp
except ImportError:
    cp = None


# ===========================================================================
# Main training loop
# ===========================================================================

def run_torch_optimization(
    basis_values_np,             # (n_valid, n_feat) float64
    obs_counts_np,               # (n_valid,) float64  — GPS count per valid cell
    n_rows,                      # int
    n_cols,                      # int
    valid_mask_np,               # (n_cells,) bool
    cell_area,                   # float (m²)
    source_spacing=5,
    source_from_resistance=True,
    hidden_dim=32,
    n_hidden_layers=2,
    lr=0.01,
    weight_decay=1e-4,
    n_epochs=300,
    patience=30,
    grad_clip=10.0,
    cg_tol=DEFAULT_CG_TOL,
    warm_start_theta=None,       # [r_0, z_1..z_4] from log-linear optimization
    reg_mean=1.0,                # Penalty weight: anchor mean(log_R) near baseline
    reg_var=0.1,                 # Penalty weight: encourage variance in log_R
    target_logR_var=1.0,         # Target var(log_R); penalty activates below this
    reg_skip=0.1,                # Penalty weight: prevent skip weights collapsing to 0
    log_R_baseline=3.0,          # Target mean(log_R); 3.0 → R ≈ 20
    output_dir=None,
    seed=42,
    verbose=True,
    # Solver selection
    solver="global_absorption",  # "diff_omniscape", "global", or "global_absorption"
    radius=15,                   # diff_omniscape: focal neighbourhood half-width
    block_size=10,               # diff_omniscape: spacing between focal pixels
    focal_fraction=0.5,          # diff_omniscape: fraction of focals used per epoch
    absorption=0.01,             # global_absorption: leakage rate (higher = more local)
    # Device selection
    device="auto",               # "auto", "cuda", or "cpu"
    # Convolutional encoder
    use_conv=False,              # Enable ConvResistanceNet (spatial context)
    n_conv_layers=3,             # Number of Conv2d layers
    conv_channels=16,            # Conv feature channels
    conv_kernel_size=3,          # Conv kernel size
    # New architecture options
    dropout=0.0,                 # Dropout rate in MLP head (0 = off)
    use_dilated=True,            # Dilated convolutions in conv encoder
    intensity_hidden=0,          # Learned intensity MLP width (0 = parametric)
    warmup_epochs=10,            # LR linear warm-up epochs (0 = off)
    absorption_schedule=None,    # Absorption curriculum: [start, end] or None
    # Model type
    model_type="loglinear",      # "loglinear", "mlp", "conv", or "spline_gam"
    n_knots=10,                  # SplineResistanceNet: internal knots per covariate
    spline_degree=3,             # SplineResistanceNet: B-spline degree
    include_interactions=True,   # SplineResistanceNet: pairwise tensor products
    penalty_scale=1.0,           # Global multiplier on smoothing penalty (>1 = smoother)
    lambda_init_marginal=0.0,    # Initial log-lambda for marginal splines
    lambda_init_interaction=2.0, # Initial log-lambda for interaction terms
    lambda_min=0.0,              # Floor on lambda: λ = lambda_min + exp(log_lambda)
    compute_uq=False,            # Compute Bayesian credible bands post-optimization
    uq_block_only=True,          # True = block-diagonal Hessian (fast)
    # Intensity spline (spline_gam only)
    intensity_spline=False,      # Replace parametric γ·log(1+C) with P-spline f(log(1+C))
    intensity_n_knots=5,         # Internal knots for intensity spline
    intensity_degree=3,          # B-spline degree for intensity spline
    lambda_init_intensity=2.0,   # Initial log-lambda for intensity smoothing
    intensity_log1p_max=10.0,    # Upper knot boundary for log(1+C)
    # IRL (value-shaped) resistance — model_type="irl"
    beta=1.0,                    # Soft value-iteration temperature
    gamma_d=0.9,                 # Discount / leakage of the MDP (must be < 1)
    n_value_iter=60,             # Soft value-iteration steps (unrolled)
    value_scale_init=1.0,        # Initial scale in log R = offset - scale * V
):
    """
    End-to-end neural-network PPP-circuit optimization.

    Parameters
    ----------
    basis_values_np : ndarray (n_valid, n_features)
        Covariate values at valid pixels (e.g. canopy, impervious, water, fence, elevation).
    obs_counts_np : ndarray (n_valid,)
        Number of GPS observations per valid cell (0 for most cells).
    n_rows, n_cols : int
        Raster grid dimensions.
    valid_mask_np : ndarray (n_cells,) bool
        Which cells are valid (not NA).
    cell_area : float
        Cell area in m² (e.g. 900 for 30m grid).
    source_spacing : int
        Source lattice spacing for circuit solver.
    source_from_resistance : bool
        Use 1/R source injection (matches Omniscape).
    hidden_dim : int
        MLP hidden layer width.
    n_hidden_layers : int
        Number of MLP hidden layers.
    lr : float
        Initial Adam learning rate.
    weight_decay : float
        L2 regularization strength.
    n_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience.
    grad_clip : float
        Gradient norm clipping.
    cg_tol : float
        CG solver tolerance (1e-6 is fast; 1e-10 is precise).
    warm_start_theta : array or None
        Log-linear parameters [r_0, z_1, ..., z_K] to warm-start from.
    output_dir : str or None
        Directory to save model weights.
    seed : int
        Random seed.
    verbose : bool

    Returns
    -------
    dict with numpy arrays and Python scalars (reticulate-safe):
        resistance, connectivity, log_lambda, alpha, gamma,
        loglik, loss_history, best_epoch, n_params, total_time
    """
    t0_all = time.time()

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n_valid = basis_values_np.shape[0]
    n_features = basis_values_np.shape[1]
    valid_mask_np = np.asarray(valid_mask_np, dtype=bool)

    use_diff_omniscape = (solver == "diff_omniscape")
    use_absorption = (solver == "global_absorption")

    # Absorption solver: force source_spacing=1 (sparse sources create grid
    # artifacts with no speed benefit for a single global solve).
    if use_absorption and source_spacing > 1:
        if verbose:
            print(f"  NOTE: source_spacing={source_spacing} overridden to 1 "
                  f"for absorption solver (avoids grid artifacts)")
        source_spacing = 1

    # ---- Device selection ----
    if device == "auto":
        use_cuda = _GPU_AVAILABLE and (use_diff_omniscape or _CUPY_AVAILABLE)
        dev = torch.device("cuda" if use_cuda else "cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda is not available")
        if not _CUPY_AVAILABLE:
            raise RuntimeError("CUDA requested but cupy is not installed")
        dev = torch.device("cuda")
        use_cuda = True
    else:
        dev = torch.device("cpu")
        use_cuda = False

    # Enable/disable GPU in the circuit solver
    cs = _get_circuit_module()
    if use_cuda and not use_diff_omniscape:
        cs.enable_gpu(True)
        if seed is not None:
            cp.random.seed(seed)
    else:
        cs.enable_gpu(False)

    # ---- Resolve model_type from use_conv flag (backward compat) ----
    if model_type == "mlp" and use_conv:
        model_type = "conv"
    use_conv = (model_type == "conv")
    use_spline = (model_type == "spline_gam")
    use_loglinear = (model_type == "loglinear")
    use_irl = (model_type == "irl")
    # Grid-context nets (conv encoder, IRL value iteration) need the 2D basis
    # grid and the (basis_grid, valid_mask, basis_valid) forward signature.
    use_grid = use_conv or use_irl

    if verbose:
        n_obs = int(obs_counts_np.sum())
        n_obs_cells = int((obs_counts_np > 0).sum())
        print(f"\n{'='*60}")
        if use_spline:
            print(f"TORCH PIPELINE: P-spline GAM Resistance + Circuit + PPP")
        elif use_irl:
            print(f"TORCH PIPELINE: IRL value-shaped Resistance + Circuit + PPP")
        else:
            print(f"TORCH PIPELINE: Neural Net Resistance + Circuit + PPP")
        print(f"{'='*60}")
        print(f"  Device: {dev}" + (f" ({torch.cuda.get_device_name()})" if use_cuda else ""))
        print(f"  Grid: {n_rows} x {n_cols} ({n_valid} valid cells)")
        print(f"  Observations: {n_obs} GPS fixes in {n_obs_cells} cells")
        print(f"  Cell area: {cell_area:.1f} m²")
        print(f"  Solver: {solver}")
        print(f"  Model type: {model_type}")
        if use_diff_omniscape:
            offset = block_size // 2
            n_focal = len(range(offset, n_rows, block_size)) * len(range(offset, n_cols, block_size))
            print(f"  Diff-omniscape: radius={radius}, block_size={block_size}, "
                  f"focal_fraction={focal_fraction:.0%}, n_focal={n_focal}")
            if use_cuda:
                print(f"  NOTE: diff_omniscape uses CPU for local solves; NN on GPU")
        if use_absorption:
            print(f"  Absorption: α={absorption:.4f} "
                  f"(effective radius ≈ {1.0/np.sqrt(absorption):.0f} px)")
        print(f"  NN architecture: {n_features} → "
              + " → ".join([str(hidden_dim)] * n_hidden_layers)
              + " → 1 (+ linear skip)")
        if use_spline:
            n_basis_per = n_knots + spline_degree + 1
            n_marginal = n_features * n_basis_per
            n_pairs = n_features * (n_features - 1) // 2 if include_interactions else 0
            n_interact = n_pairs * n_basis_per ** 2
            n_smooth_params = n_features + 2 * n_pairs
            print(f"  Spline: {n_knots} knots, degree {spline_degree}, "
                  f"{n_basis_per} basis/cov")
            print(f"  Marginal terms: {n_features} × {n_basis_per} = "
                  f"{n_marginal} coefs")
            if include_interactions:
                print(f"  Interactions: {n_pairs} pairs × "
                      f"{n_basis_per**2} = {n_interact} coefs")
            print(f"  Smoothing params: {n_smooth_params}")
        if use_conv:
            # Receptive field with dilated convolutions: sum of dilations * (k-1) + 1
            if use_dilated:
                rf = 1 + sum(2**i * (conv_kernel_size - 1) for i in range(n_conv_layers))
                # +1 for the input projection layer
                rf += (conv_kernel_size - 1)
            else:
                rf = 2 * n_conv_layers + 1 + (conv_kernel_size - 1)
            print(f"  Conv encoder: {n_conv_layers} × ResBlock({conv_channels}, "
                  f"k={conv_kernel_size}, dilated={use_dilated}) → "
                  f"receptive field {rf}×{rf} px")
            if dropout > 0:
                print(f"  MLP dropout: {dropout}")
            if intensity_hidden > 0:
                print(f"  Learned intensity MLP: 1 → {intensity_hidden} → 1")
        if use_irl:
            print(f"  Soft value iteration: β={beta}, γ_d={gamma_d}, "
                  f"{n_value_iter} iters → log R = offset − scale·V")
        if warmup_epochs > 0:
            print(f"  LR warm-up: {warmup_epochs} epochs")
        if absorption_schedule is not None:
            print(f"  Absorption curriculum: {absorption_schedule[0]:.4f} → "
                  f"{absorption_schedule[1]:.4f}")
        print(f"  CG tolerance: {cg_tol:.1e}")
        print(f"  LR: {lr}, weight_decay: {weight_decay}, patience: {patience}")
        print(f"  Regularization: reg_mean={reg_mean}, reg_var={reg_var}, "
              f"target_logR_var={target_logR_var}, reg_skip={reg_skip}, "
              f"log_R_baseline={log_R_baseline}")

    # ---- Relax CG tolerance for speed ----
    original_rtol = getattr(cs, "CG_RTOL", 1e-10)
    cs.CG_RTOL = float(cg_tol)
    cs.CG_WARM_START = True

    # ---- Tensors (on device) ----
    basis_t = torch.tensor(basis_values_np, dtype=torch.float64, device=dev)
    obs_t = torch.tensor(obs_counts_np, dtype=torch.float64, device=dev)

    # ---- 2D grid for grid-context nets (conv encoder / IRL value iteration) ----
    basis_grid_t = None
    if use_grid:
        n_cells = n_rows * n_cols
        grid_np = np.zeros((n_features, n_cells), dtype=np.float64)
        grid_np[:, valid_mask_np] = basis_values_np.T  # fill valid; rest stays 0
        grid_np = grid_np.reshape(1, n_features, n_rows, n_cols)
        basis_grid_t = torch.tensor(grid_np, dtype=torch.float64, device=dev)

    # ---- Model (on device) ----
    if use_spline:
        net = SplineResistanceNet(
            n_features, n_knots=n_knots, degree=spline_degree,
            include_interactions=include_interactions,
            lambda_init_marginal=float(lambda_init_marginal),
            lambda_init_interaction=float(lambda_init_interaction),
            lambda_min=float(lambda_min),
            intensity_spline=bool(intensity_spline),
            intensity_n_knots=int(intensity_n_knots),
            intensity_degree=int(intensity_degree),
            lambda_init_intensity=float(lambda_init_intensity),
            intensity_log1p_max=float(intensity_log1p_max),
        ).double().to(dev)
        net.setup_basis_matrices(basis_values_np, device=dev)
    elif use_conv:
        net = ConvResistanceNet(
            n_features, conv_channels=conv_channels,
            n_conv_layers=n_conv_layers,
            conv_kernel_size=conv_kernel_size,
            hidden=hidden_dim, n_mlp_layers=n_hidden_layers,
            dropout=dropout, use_dilated=use_dilated,
            intensity_hidden=intensity_hidden,
        ).double().to(dev)
    elif use_irl:
        net = IRLResistanceNet(
            n_features, hidden=hidden_dim, n_layers=n_hidden_layers,
            vi_beta=float(beta), gamma_d=float(gamma_d),
            n_value_iter=int(n_value_iter),
            value_scale_init=float(value_scale_init),
        ).double().to(dev)
    elif use_loglinear:
        net = LogLinearResistanceNet(n_features).double().to(dev)
    else:
        net = ResistanceNet(n_features, hidden_dim, n_hidden_layers).double().to(dev)
    alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, device=dev))
    # Reparameterize gamma through softplus so gamma is always positive.
    # Initialize raw so that softplus(raw) = 1.0 (matching previous init).
    _raw_gamma_init = float(np.log(np.exp(1.0) - 1.0))  # ≈ 0.5413
    raw_gamma = nn.Parameter(torch.tensor(_raw_gamma_init, dtype=torch.float64, device=dev))

    # When intensity spline is active, gamma is absorbed into the spline
    use_intensity_spline = use_spline and net.intensity_spline
    if use_intensity_spline:
        raw_gamma.requires_grad_(False)

    if warm_start_theta is not None:
        ws = np.asarray(warm_start_theta, dtype=np.float64).ravel()
        if verbose:
            print(f"  Warm start from log-linear: [{', '.join(f'{w:.4f}' for w in ws)}]")
        net.warm_start(ws)

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad) + 2
    if use_intensity_spline:
        n_params -= 1  # gamma is frozen
    if verbose:
        print(f"  Total parameters: {n_params} ({n_params - 2} NN + "
              f"{'1 intensity (alpha only, gamma absorbed into spline)' if use_intensity_spline else '2 intensity'}")

    # ---- Optimizer ----
    # Separate param groups with appropriate weight_decay.
    if use_spline:
        # Spline model: smoothing penalty handles regularisation, so no L2 on
        # spline coefficients. Log-lambda params are self-regularising.
        spline_coef_params = (list(net.spline_coefs.parameters())
                              + list(net.interaction_coefs.parameters()))
        log_lambda_params = (list(net.log_lambda_smooth.parameters())
                             + list(net.log_lambda_interact_row.parameters())
                             + list(net.log_lambda_interact_col.parameters()))
        skip_params = [net.skip.weight, net.skip.bias, net.intercept]
        # Intensity params: alpha always; gamma only if no intensity spline
        intensity_params = [alpha]
        if not use_intensity_spline:
            intensity_params.append(raw_gamma)
        # Intensity spline coefficients and smoothing
        if use_intensity_spline:
            spline_coef_params = spline_coef_params + [net.intensity_coefs]
            log_lambda_params = log_lambda_params + [net.log_lambda_intensity]
        all_params = [
            {"params": skip_params, "weight_decay": 0.0},
            {"params": spline_coef_params, "weight_decay": 0.0},
            {"params": log_lambda_params, "weight_decay": 0.0, "lr": lr * 0.5},
            {"params": intensity_params, "weight_decay": 0.0},
        ]
        all_params_flat = (skip_params + spline_coef_params
                           + log_lambda_params + intensity_params)
    else:
        # NN models: skip.weight gets NO weight_decay so L2 doesn't
        # push it back to zero (which would re-create the flat saddle point).
        skip_params = [net.skip.weight]
        other_nn_params = [p for n, p in net.named_parameters()
                           if not n.startswith("skip.weight")]
        all_params = [
            {"params": skip_params, "weight_decay": 0.0},
            {"params": other_nn_params, "weight_decay": weight_decay},
            {"params": [alpha, raw_gamma], "weight_decay": 0.0},
        ]
        all_params_flat = skip_params + other_nn_params + [alpha, raw_gamma]
    opt = torch.optim.Adam(all_params, lr=lr)
    # LR schedule: optional linear warm-up then cosine decay
    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.02, end_factor=1.0,
            total_iters=warmup_epochs
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(n_epochs - warmup_epochs, 1), eta_min=lr * 0.01
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs]
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs, eta_min=lr * 0.01
        )

    # ---- Training loop ----
    history = []
    best = dict(loss=float("inf"), epoch=0, state=None, alpha=0.0, gamma=1.0)
    wait = 0

    if verbose:
        print(f"\n  {'Epoch':>5}  {'Loss':>10}  {'LogLik':>10}  "
              f"{'Alpha':>8}  {'Gamma':>8}  {'LR':>10}  {'Time':>6}")
        print(f"  {'-'*65}")

    for ep in range(n_epochs):
        t0 = time.time()

        # Absorption curriculum: linearly interpolate from start to end
        if absorption_schedule is not None and use_absorption:
            frac = min(ep / max(n_epochs - 1, 1), 1.0)
            absorption_ep = (absorption_schedule[0]
                             + frac * (absorption_schedule[1] - absorption_schedule[0]))
        else:
            absorption_ep = absorption

        try:
            opt.zero_grad()

            # Forward: covariates → R → circuit → C → log λ → loglik
            if use_grid:
                net.train()
                R, log_R_raw = net.forward_with_log_R(
                    basis_grid_t, valid_mask_np, basis_t)
            elif use_spline:
                R, log_R_raw = net.forward_with_log_R(basis_t)
            else:
                R, log_R_raw = net.forward_with_log_R(basis_t)
            if use_diff_omniscape:
                rng_seed = (seed + ep) if seed is not None else ep
                C = _DiffOmniscapeSolveFn.apply(
                    R, valid_mask_np, n_rows, n_cols,
                    source_spacing, source_from_resistance,
                    radius, block_size, focal_fraction, rng_seed,
                )
            elif use_absorption:
                C = _AbsorptionCircuitSolveFn.apply(
                    R, valid_mask_np, n_rows, n_cols,
                    source_spacing, source_from_resistance,
                    absorption_ep,
                )
            else:
                C = _CircuitSolveFn.apply(
                    R, valid_mask_np, n_rows, n_cols,
                    source_spacing, source_from_resistance,
                )

            gamma = F.softplus(raw_gamma)
            log1p_C = torch.log1p(C.clamp(min=0))
            # Intensity: spline or learned MLP or parametric
            if use_intensity_spline:
                log_lam = alpha + net.eval_intensity_spline(log1p_C)
            elif use_conv and net.intensity_mlp is not None:
                log_lam = alpha + gamma * net.intensity_mlp(
                    log1p_C.unsqueeze(-1)).squeeze(-1)
            else:
                log_lam = alpha + gamma * log1p_C
            ll = _ppp_loglik(log_lam, obs_t, float(cell_area))

            # Regularization: prevent degenerate R → R_max everywhere.
            # Penalties are scaled by n_valid so they are commensurate with
            # the PPP integral term (≈ n_valid × cell_area × mean(λ)).
            mean_logR = log_R_raw.mean()
            pen_mean = n_valid * reg_mean * (mean_logR - log_R_baseline) ** 2

            if use_spline:
                # P-spline smoothing penalty replaces reg_var and reg_skip.
                # penalty_scale > 1 biases toward smoother fits.
                pen_smooth = penalty_scale * net.smoothing_penalty()
                loss = -ll + pen_mean + pen_smooth
            else:
                var_logR = log_R_raw.var()
                # One-sided hinge: penalise var(logR) < target, no reward above.
                log_var = torch.log(var_logR + 0.01)
                pen_var = n_valid * reg_var * F.softplus(
                    math.log(target_logR_var) - log_var
                )
                # Prevent skip weights collapsing to zero.
                skip_norm = net.skip.weight.norm()
                pen_skip = n_valid * reg_skip / (skip_norm + 0.01)
                loss = -ll + pen_mean + pen_var + pen_skip

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params_flat, grad_clip)

            opt.step()
            sched.step()

        except RuntimeError as e:
            if "CG did not converge" in str(e):
                if verbose:
                    print(f"  [{ep:3d}] CG failed — halving learning rate")
                for pg in opt.param_groups:
                    pg["lr"] *= 0.5
                history.append(float("inf"))
                continue
            raise

        lv = loss.item()
        elapsed = time.time() - t0
        history.append(lv)

        # Track best
        if lv < best["loss"]:
            best.update(
                loss=lv,
                epoch=ep,
                alpha=alpha.item(),
                gamma=F.softplus(raw_gamma).item(),
                state={k: v.clone() for k, v in net.state_dict().items()},
            )
            wait = 0
        else:
            wait += 1

        if verbose and (ep % 10 == 0 or ep < 5 or ep == n_epochs - 1):
            abs_str = (f", α_abs={absorption_ep:.4f}"
                       if absorption_schedule is not None and use_absorption else "")
            print(f"  {ep:5d}  {lv:+10.2f}  {ll.item():10.2f}  "
                  f"{alpha.item():8.4f}  {F.softplus(raw_gamma).item():8.4f}  "
                  f"{sched.get_last_lr()[0]:10.1e}  {elapsed:5.1f}s{abs_str}")
            with torch.no_grad():
                net.eval()
                if use_grid:
                    R_mon, logR_mon = net.forward_with_log_R(
                        basis_grid_t, valid_mask_np, basis_t)
                else:
                    R_mon, logR_mon = net.forward_with_log_R(basis_t)
                net.train()
                print(f"         R: min={R_mon.min():.1f}, "
                      f"med={R_mon.median():.1f}, "
                      f"max={R_mon.max():.1f}, "
                      f"std={R_mon.std():.1f}")
                pm = n_valid * reg_mean * (logR_mon.mean() - log_R_baseline) ** 2
                if use_spline:
                    ps = net.smoothing_penalty().item()
                    print(f"         logR: mean={logR_mon.mean():.2f}, "
                          f"std={logR_mon.std():.2f}, "
                          f"pen_m={pm:.0f}, pen_smooth={ps:.1f}")
                else:
                    lv_mon = torch.log(logR_mon.var() + 0.01)
                    pv = n_valid * reg_var * F.softplus(
                        math.log(target_logR_var) - lv_mon
                    ).item()
                    sn = net.skip.weight.norm().item()
                    ps = n_valid * reg_skip / (sn + 0.01)
                    print(f"         logR: mean={logR_mon.mean():.2f}, "
                          f"std={logR_mon.std():.2f}, "
                          f"pen_m={pm:.0f}, pen_v={pv:.0f}, "
                          f"skip_norm={sn:.4f}, pen_s={ps:.0f}")

        if wait >= patience:
            if verbose:
                print(f"\n  Early stopping at epoch {ep} (patience={patience})")
            break

    # ---- Restore best ----
    if best["state"] is None:
        if verbose:
            print("  WARNING: No improvement found. Using final state.")
        best.update(
            alpha=alpha.item(),
            gamma=F.softplus(raw_gamma).item(),
            state={k: v.clone() for k, v in net.state_dict().items()},
            epoch=len(history) - 1,
        )

    net.load_state_dict(best["state"])
    alpha_v = best["alpha"]
    gamma_v = best["gamma"]

    if verbose:
        print(f"\n  Best epoch: {best['epoch']}, loss: {best['loss']:.2f}")
        print(f"  α = {alpha_v:.4f}, γ = {gamma_v:.4f}")

    # ---- Final forward (no grad) ----
    net.eval()
    with torch.no_grad():
        if use_grid:
            R_fin = net(basis_grid_t, valid_mask_np, basis_t).cpu().numpy()
        else:
            R_fin = net(basis_t).cpu().numpy()

    # Final circuit solve at full precision (always CPU for pyamg AMG accuracy)
    if use_cuda and not use_diff_omniscape:
        cs.enable_gpu(False)
    n = n_rows * n_cols
    Rf = np.full(n, 1e6, dtype=np.float64)
    Rf[valid_mask_np] = R_fin

    if use_diff_omniscape:
        # Full-precision diff_omniscape with smaller block_size, no subsampling
        do = _get_diff_omniscape_module()
        final_block = min(block_size, 5)  # finer grid for final output
        C_map_fin, _, _ = do.solve_diff_omniscape(
            Rf.reshape(n_rows, n_cols),
            radius=int(radius),
            block_size=int(final_block),
            source_from_resistance=bool(source_from_resistance),
        )
        # Interpolate to full grid
        offset_fin = final_block // 2
        focal_rows_fin = list(range(offset_fin, n_rows, final_block))
        focal_cols_fin = list(range(offset_fin, n_cols, final_block))
        C_full_fin = _interpolate_block_grid(
            C_map_fin, final_block, focal_rows_fin, focal_cols_fin,
            n_rows, n_cols
        )
        C_fin = C_full_fin.ravel()[valid_mask_np].copy()
        if verbose:
            print(f"  Final diff_omniscape solve: block_size={final_block}, "
                  f"n_focal={len(focal_rows_fin)*len(focal_cols_fin)}")
    elif use_absorption:
        cs.CG_RTOL = 1e-10
        result_fin = cs.solve_circuit_absorption(
            Rf.reshape(n_rows, n_cols),
            absorption=float(absorption),
            source_spacing=int(source_spacing),
            source_from_resistance=bool(source_from_resistance),
        )
        C_fin = result_fin["current_density"][valid_mask_np].copy()
        if verbose:
            print(f"  Final absorption solve: α={absorption:.4f}")
    else:
        cs.CG_RTOL = 1e-10
        result_fin = cs.solve_circuit(
            Rf.reshape(n_rows, n_cols),
            source_spacing=int(source_spacing),
            source_from_resistance=bool(source_from_resistance),
        )
        C_fin = result_fin["current_density"][valid_mask_np].copy()

    # Compute final intensity using learned MLP or parametric form
    log1p_C_fin = np.log1p(np.maximum(C_fin, 0.0))
    if use_intensity_spline:
        with torch.no_grad():
            log1p_t = torch.tensor(log1p_C_fin, dtype=torch.float64, device=dev)
            f_int = net.eval_intensity_spline(log1p_t).cpu().numpy()
        log_lam_fin = alpha_v + f_int
    elif use_conv and net.intensity_mlp is not None:
        with torch.no_grad():
            log1p_t = torch.tensor(log1p_C_fin, dtype=torch.float64, device=dev)
            intensity_transform = net.intensity_mlp(
                log1p_t.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        log_lam_fin = alpha_v + gamma_v * intensity_transform
    else:
        log_lam_fin = alpha_v + gamma_v * log1p_C_fin
    ll_fin = float(
        np.sum(obs_counts_np * log_lam_fin)
        - cell_area * np.sum(np.exp(np.minimum(log_lam_fin, 20.0)))
    )

    # Restore original CG tolerance and disable GPU for clean state
    cs.CG_RTOL = original_rtol
    cs.CG_WARM_START = False
    cs.enable_gpu(False)

    total_time = time.time() - t0_all

    # ---- Save model weights (always on CPU for portability) ----
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        net_cpu = {k: v.cpu() for k, v in net.state_dict().items()}
        save_dict = {
            "net": net_cpu,
            "alpha": alpha_v,
            "gamma": gamma_v,
            "n_features": n_features,
            "model_type": model_type,
            "hidden_dim": hidden_dim,
            "n_hidden_layers": n_hidden_layers,
            "use_conv": use_conv,
            "conv_channels": conv_channels if use_conv else None,
            "n_conv_layers": n_conv_layers if use_conv else None,
            "conv_kernel_size": conv_kernel_size if use_conv else None,
            "dropout": dropout if use_conv else None,
            "use_dilated": use_dilated if use_conv else None,
            "intensity_hidden": intensity_hidden if use_conv else None,
        }
        if use_spline:
            save_dict.update({
                "n_knots": n_knots,
                "spline_degree": spline_degree,
                "include_interactions": include_interactions,
                "intensity_spline": bool(intensity_spline),
                "intensity_n_knots": int(intensity_n_knots),
                "intensity_degree": int(intensity_degree),
                "lambda_init_intensity": float(lambda_init_intensity),
                "intensity_log1p_max": float(intensity_log1p_max),
            })
        if use_irl:
            save_dict.update({
                "beta": float(beta),
                "gamma_d": float(gamma_d),
                "n_value_iter": int(n_value_iter),
                "value_scale_init": float(value_scale_init),
            })
        torch.save(save_dict, os.path.join(output_dir, "resistance_nn.pt"))

    if verbose:
        print(f"\n  Done in {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"  Final loglik: {ll_fin:.2f}")
        print(f"  Parameters: {n_params}")

    # ---- Effective log-linear coefficients ----
    if use_spline:
        effective_loglinear = np.array(
            net.get_effective_loglinear(), dtype=np.float64)
    else:
        with torch.no_grad():
            skip_w = net.skip.weight.squeeze().cpu().numpy().copy()
            skip_b = net.skip.bias.cpu().item()
        effective_loglinear = np.concatenate([[skip_b], skip_w])

    # ---- Partial effects and UQ (spline only) ----
    partial_effects = None
    interaction_effects = None
    uq_results = None
    if use_spline:
        posterior_cov = None

        if compute_uq:
            if verbose:
                print(f"\n  Computing Bayesian UQ (block_only={uq_block_only})...")
                t_uq = time.time()

            # Build NLL function for Hessian computation.
            # Each call re-runs forward + circuit solve and returns NLL as float.
            n = n_rows * n_cols
            def _nll_fn():
                """Negative log-likelihood (no penalty) for Hessian FD."""
                with torch.no_grad():
                    R_uq = net(basis_t).cpu().numpy()
                R_full_uq = np.full(n, 1e6, dtype=np.float64)
                R_full_uq[valid_mask_np] = R_uq

                if use_absorption:
                    res_uq = cs.solve_circuit_absorption(
                        R_full_uq.reshape(n_rows, n_cols),
                        absorption=float(absorption),
                        source_spacing=int(source_spacing),
                        source_from_resistance=bool(source_from_resistance),
                    )
                    C_uq = res_uq["current_density"][valid_mask_np]
                else:
                    res_uq = cs.solve_circuit(
                        R_full_uq.reshape(n_rows, n_cols),
                        source_spacing=int(source_spacing),
                        source_from_resistance=bool(source_from_resistance),
                    )
                    C_uq = res_uq["current_density"][valid_mask_np]

                log1p_C_uq = np.log1p(np.maximum(C_uq, 0.0))
                if use_intensity_spline:
                    with torch.no_grad():
                        log1p_t_uq = torch.tensor(log1p_C_uq, dtype=torch.float64, device=dev)
                        f_int_uq = net.eval_intensity_spline(log1p_t_uq).cpu().numpy()
                    log_lam_uq = alpha_v + f_int_uq
                else:
                    log_lam_uq = alpha_v + gamma_v * log1p_C_uq
                nll = -(
                    np.sum(obs_counts_np * log_lam_uq)
                    - cell_area * np.sum(np.exp(np.minimum(log_lam_uq, 20.0)))
                )
                return float(nll)

            hess_result = net.compute_nll_hessian_block(
                _nll_fn, block_only=uq_block_only)

            uq_output = net.compute_posterior_covariance(
                hess_result, penalty_scale=float(penalty_scale))
            posterior_cov = uq_output["covariance"]

            uq_results = {
                "edf": uq_output["edf"],
                "significance": {
                    int(k): {
                        "chi_sq": float(v["chi_sq"]),
                        "edf": float(v["edf"]),
                        "p_value": float(v["p_value"]),
                    }
                    for k, v in uq_output["significance"].items()
                },
            }

            if verbose:
                dt = time.time() - t_uq
                print(f"  UQ completed in {dt:.1f}s")
                for k in range(net.n_features):
                    sig = uq_output["significance"][k]
                    print(f"    covariate {k}: EDF={sig['edf']:.1f}, "
                          f"chi2={sig['chi_sq']:.1f}, "
                          f"p={sig['p_value']:.4f}")

        pe = net.get_partial_effects(n_grid=100, posterior_cov=posterior_cov)
        partial_effects = {}
        for k, v in pe.items():
            entry = {
                "grid": v["grid"].tolist(),
                "effect": v["effect"].tolist(),
            }
            if "se" in v:
                entry["se"] = v["se"].tolist()
                entry["lower_95"] = v["lower_95"].tolist()
                entry["upper_95"] = v["upper_95"].tolist()
            partial_effects[int(k)] = entry

        interaction_effects = net.get_interaction_effects(n_grid=30)

    # Intensity spline partial effect
    intensity_partial_effect = None
    if use_intensity_spline:
        with torch.no_grad():
            # Evaluate on a grid from 0 to the max observed log(1+C)
            grid_max = float(np.max(log1p_C_fin)) * 1.05 + 0.1
            grid_np = np.linspace(0.0, grid_max, 100)
            grid_t = torch.tensor(grid_np, dtype=torch.float64, device=dev)
            B_int = _bspline_basis_matrix_torch(
                grid_t, net._intensity_knots,
                net.intensity_n_basis, net.intensity_degree)
            f_raw = (B_int @ net.intensity_coefs).cpu().numpy()
            # Don't center the display — show raw f(x)
            intensity_partial_effect = {
                "grid": grid_np.tolist(),
                "effect": f_raw.tolist(),
            }

    result = {
        "resistance": R_fin,
        "connectivity": C_fin,
        "log_lambda": log_lam_fin,
        "alpha": float(alpha_v),
        "gamma": float(gamma_v),
        "loglik": float(ll_fin),
        "loss_history": [float(x) for x in history],
        "best_epoch": int(best["epoch"]),
        "n_params": int(n_params),
        "n_epochs_run": int(len(history)),
        "total_time": float(total_time),
        "effective_loglinear": effective_loglinear.tolist(),
        "solver": str(solver),
        "device": str(dev),
        "model_type": str(model_type),
        "intensity_spline": bool(use_intensity_spline),
    }
    if partial_effects is not None:
        result["partial_effects"] = partial_effects
    if interaction_effects is not None:
        result["interaction_effects"] = interaction_effects
    if uq_results is not None:
        result["uq_results"] = uq_results
    if intensity_partial_effect is not None:
        result["intensity_partial_effect"] = intensity_partial_effect

    return result

