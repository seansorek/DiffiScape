"""Finite-difference gradient verification for the circuit-solve autograd
Functions and resistance-net forward chains.

These are smoke tests for the differentiable pipeline, exposed as importable
functions (not just pytest) so R callers can run a quick sanity check before
launching an expensive optimisation (see verify_torch_gradient() /
verify_conv_gradient() / verify_spline_gradient() / verify_irl_gradient() in
R/torch_pipeline.R).
"""
import numpy as np
import torch

from .autograd_functions import (
    _CircuitSolveFn,
    _AbsorptionCircuitSolveFn,
    _DiffOmniscapeSolveFn,
)
from .resistance_nets import ConvResistanceNet, IRLResistanceNet, SplineResistanceNet


# ===========================================================================
# Gradient verification
# ===========================================================================

def verify_circuit_gradient(basis_values_np, valid_mask_np,
                            n_rows, n_cols, source_spacing=5,
                            source_from_resistance=True,
                            eps=1e-4, seed=42,
                            solver="diff_omniscape",
                            radius=15, block_size=10,
                            absorption=0.01):
    """
    Finite-difference check for the custom autograd circuit solve.
    Uses a random subset of pixels for speed.

    Returns dict with max_rel_error and pass/fail.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_valid = valid_mask_np.sum()
    R_val = np.full(n_valid, 20.0, dtype=np.float64)
    # Add spatial variation
    R_val += np.random.randn(n_valid) * 5.0
    R_val = np.clip(R_val, 1.0, 5000.0)

    R_t = torch.tensor(R_val, dtype=torch.float64, requires_grad=True)

    use_diff_omniscape = (solver == "diff_omniscape")
    use_absorption = (solver == "global_absorption")

    def _apply_circuit(R_tensor):
        if use_diff_omniscape:
            return _DiffOmniscapeSolveFn.apply(
                R_tensor, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
                radius, block_size, 1.0, seed,  # focal_fraction=1.0 for exact grad check
            )
        elif use_absorption:
            return _AbsorptionCircuitSolveFn.apply(
                R_tensor, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
                absorption,
            )
        else:
            return _CircuitSolveFn.apply(
                R_tensor, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
            )

    # Forward
    C = _apply_circuit(R_t)

    # Random loss direction
    dl_dC = torch.randn_like(C, dtype=torch.float64)
    loss = (C * dl_dC).sum()
    loss.backward()
    analytic_grad = R_t.grad.numpy().copy()

    # Finite differences on a random subset
    n_check = min(20, n_valid)
    check_idx = np.random.choice(n_valid, n_check, replace=False)
    fd_grad = np.zeros(n_check)

    for i, idx in enumerate(check_idx):
        R_plus = R_val.copy()
        R_plus[idx] += eps
        C_plus = _apply_circuit(
            torch.tensor(R_plus, dtype=torch.float64),
        )
        loss_plus = (C_plus * dl_dC).sum().item()

        R_minus = R_val.copy()
        R_minus[idx] -= eps
        C_minus = _apply_circuit(
            torch.tensor(R_minus, dtype=torch.float64),
        )
        loss_minus = (C_minus * dl_dC).sum().item()

        fd_grad[i] = (loss_plus - loss_minus) / (2 * eps)

    analytic_at_check = analytic_grad[check_idx]
    rel_errors = np.abs(fd_grad - analytic_at_check) / (np.maximum(np.abs(fd_grad), np.abs(analytic_at_check)) + 1e-7)
    max_rel = float(np.max(rel_errors))

    print(f"  Gradient check: max rel error = {max_rel:.2e} "
          f"(checked {n_check} pixels)")

    return {
        "max_rel_error": max_rel,
        "pass": max_rel < 0.05,
        "rel_errors": rel_errors.tolist(),
    }


def verify_conv_gradient(basis_values_np, valid_mask_np,
                         n_rows, n_cols, source_spacing=1,
                         source_from_resistance=True,
                         conv_channels=16, n_conv_layers=3,
                         conv_kernel_size=3, hidden_dim=16,
                         n_mlp_layers=1,
                         eps=1e-4, seed=42,
                         solver="global_absorption",
                         radius=15, block_size=10,
                         absorption=0.01,
                         dropout=0.0, use_dilated=True,
                         intensity_hidden=0):
    """
    End-to-end finite-difference gradient check for ConvResistanceNet.

    Tests the full chain: conv(basis_grid) → R → circuit → C → scalar loss.
    Perturbs individual conv/MLP/skip parameters and compares FD vs autograd.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_valid = int(valid_mask_np.sum())
    n_features = basis_values_np.shape[1]

    # Build 2D grid
    n_cells = n_rows * n_cols
    grid_np = np.zeros((n_features, n_cells), dtype=np.float64)
    grid_np[:, valid_mask_np] = basis_values_np.T
    grid_np = grid_np.reshape(1, n_features, n_rows, n_cols)
    basis_grid_t = torch.tensor(grid_np, dtype=torch.float64)
    basis_valid_t = torch.tensor(basis_values_np, dtype=torch.float64)

    net = ConvResistanceNet(
        n_features, conv_channels=conv_channels,
        n_conv_layers=n_conv_layers,
        conv_kernel_size=conv_kernel_size,
        hidden=hidden_dim, n_mlp_layers=n_mlp_layers,
        dropout=0.0,  # No dropout for gradient check (deterministic)
        use_dilated=use_dilated,
        intensity_hidden=intensity_hidden,
    ).double()

    use_absorption = (solver == "global_absorption")
    use_diff_omniscape = (solver == "diff_omniscape")

    def _forward_loss(net_):
        R = net_(basis_grid_t, valid_mask_np, basis_valid_t)
        if use_diff_omniscape:
            C = _DiffOmniscapeSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
                radius, block_size, 1.0, seed,
            )
        elif use_absorption:
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
        return (C * dl_dC).sum()

    # Random loss direction (fixed across perturbations)
    with torch.no_grad():
        R_tmp = net(basis_grid_t, valid_mask_np, basis_valid_t)
    dl_dC = torch.randn(n_valid, dtype=torch.float64)

    # Analytic gradient
    loss = _forward_loss(net)
    loss.backward()

    # Collect a subset of parameters to check — save analytic grads upfront
    # (net.zero_grad() in the FD loop would clear them otherwise)
    all_params = [(n, p, p.grad.detach().cpu().numpy().ravel().copy())
                  for n, p in net.named_parameters()
                  if p.requires_grad and p.grad is not None]
    n_check_per_param = 3
    fd_results = []

    for pname, param, analytic_g in all_params:
        n_elem = len(analytic_g)
        check_idx = np.random.choice(n_elem, min(n_check_per_param, n_elem), replace=False)

        for idx in check_idx:
            orig = param.data.view(-1)[idx].item()

            param.data.view(-1)[idx] = orig + eps
            net.zero_grad()
            lp = _forward_loss(net).item()

            param.data.view(-1)[idx] = orig - eps
            net.zero_grad()
            lm = _forward_loss(net).item()

            param.data.view(-1)[idx] = orig

            fd = (lp - lm) / (2 * eps)
            an = analytic_g[idx]
            rel_err = abs(fd - an) / (max(abs(fd), abs(an)) + 1e-7)
            fd_results.append((pname, idx, fd, an, rel_err))

    rel_errors = [r[4] for r in fd_results]
    max_rel = float(max(rel_errors))

    print(f"  Conv gradient check: max rel error = {max_rel:.2e} "
          f"(checked {len(fd_results)} param elements across {len(all_params)} params)")
    for pname, idx, fd, an, re in fd_results:
        status = "OK" if re < 0.05 else "FAIL"
        print(f"    {pname}[{idx}]: FD={fd:.6e}, AN={an:.6e}, rel={re:.2e} {status}")

    return {
        "max_rel_error": max_rel,
        "pass": max_rel < 0.05,
        "rel_errors": [float(r) for r in rel_errors],
    }


def verify_softrl_gradient(basis_values_np, valid_mask_np,
                           n_rows, n_cols, source_spacing=1,
                           source_from_resistance=True,
                           hidden_dim=16, n_hidden_layers=2,
                           beta=1.0, gamma_d=0.9, n_value_iter=30,
                           value_scale_init=1.0,
                           eps=1e-4, seed=42,
                           solver="global_absorption",
                           radius=15, block_size=10,
                           absorption=0.01):
    """
    End-to-end finite-difference gradient check for IRLResistanceNet.

    Tests the full chain: reward(basis) → soft value iteration → R → circuit
    → C → scalar loss.  Perturbs individual reward/offset/scale parameters and
    compares finite-difference vs autograd gradients through the unrolled value
    iteration.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_valid = int(valid_mask_np.sum())
    n_features = basis_values_np.shape[1]

    # Build 2D grid (same layout as the conv/IRL training path)
    n_cells = n_rows * n_cols
    grid_np = np.zeros((n_features, n_cells), dtype=np.float64)
    grid_np[:, valid_mask_np] = basis_values_np.T
    grid_np = grid_np.reshape(1, n_features, n_rows, n_cols)
    basis_grid_t = torch.tensor(grid_np, dtype=torch.float64)
    basis_valid_t = torch.tensor(basis_values_np, dtype=torch.float64)

    net = IRLResistanceNet(
        n_features, hidden=hidden_dim, n_layers=n_hidden_layers,
        vi_beta=float(beta), gamma_d=float(gamma_d),
        n_value_iter=int(n_value_iter),
        value_scale_init=float(value_scale_init),
    ).double()

    use_absorption = (solver == "global_absorption")
    use_diff_omniscape = (solver == "diff_omniscape")

    def _forward_loss(net_):
        R = net_(basis_grid_t, valid_mask_np, basis_valid_t)
        if use_diff_omniscape:
            C = _DiffOmniscapeSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
                radius, block_size, 1.0, seed,
            )
        elif use_absorption:
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
        return (C * dl_dC).sum()

    # Random loss direction (fixed across perturbations)
    dl_dC = torch.randn(n_valid, dtype=torch.float64)

    # Analytic gradient
    loss = _forward_loss(net)
    loss.backward()

    all_params = [(n, p, p.grad.detach().cpu().numpy().ravel().copy())
                  for n, p in net.named_parameters()
                  if p.requires_grad and p.grad is not None]
    n_check_per_param = 3
    fd_results = []

    for pname, param, analytic_g in all_params:
        n_elem = len(analytic_g)
        check_idx = np.random.choice(n_elem, min(n_check_per_param, n_elem),
                                     replace=False)
        for idx in check_idx:
            orig = param.data.view(-1)[idx].item()

            param.data.view(-1)[idx] = orig + eps
            net.zero_grad()
            lp = _forward_loss(net).item()

            param.data.view(-1)[idx] = orig - eps
            net.zero_grad()
            lm = _forward_loss(net).item()

            param.data.view(-1)[idx] = orig

            fd = (lp - lm) / (2 * eps)
            an = analytic_g[idx]
            rel_err = abs(fd - an) / (max(abs(fd), abs(an)) + 1e-7)
            fd_results.append((pname, idx, fd, an, rel_err))

    rel_errors = [r[4] for r in fd_results]
    max_rel = float(max(rel_errors))

    print(f"  IRL gradient check: max rel error = {max_rel:.2e} "
          f"(checked {len(fd_results)} param elements across {len(all_params)} params)")
    for pname, idx, fd, an, re in fd_results:
        status = "OK" if re < 0.05 else "FAIL"
        print(f"    {pname}[{idx}]: FD={fd:.6e}, AN={an:.6e}, rel={re:.2e} {status}")

    return {
        "max_rel_error": max_rel,
        "pass": max_rel < 0.05,
        "rel_errors": [float(r) for r in rel_errors],
    }


def verify_spline_gradient(basis_values_np, valid_mask_np,
                           n_rows, n_cols, source_spacing=1,
                           source_from_resistance=True,
                           n_knots=10, degree=3,
                           include_interactions=True,
                           eps=1e-4, seed=42,
                           solver="global_absorption",
                           radius=15, block_size=10,
                           absorption=0.01,
                           intensity_spline=False,
                           intensity_n_knots=5,
                           intensity_degree=3,
                           intensity_log1p_max=10.0):
    """
    End-to-end finite-difference gradient check for SplineResistanceNet.

    Tests the full chain: B-spline(basis) → R → circuit → C → scalar loss.
    Perturbs individual spline parameters and compares FD vs autograd.
    When intensity_spline=True, also tests the intensity spline path:
    C → log(1+C) → B-spline → f_intensity → loss.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_valid = int(valid_mask_np.sum())
    n_features = basis_values_np.shape[1]

    basis_t = torch.tensor(basis_values_np, dtype=torch.float64)

    net = SplineResistanceNet(
        n_features, n_knots=n_knots, degree=degree,
        include_interactions=include_interactions,
        intensity_spline=bool(intensity_spline),
        intensity_n_knots=int(intensity_n_knots),
        intensity_degree=int(intensity_degree),
        intensity_log1p_max=float(intensity_log1p_max),
    ).double()
    net.setup_basis_matrices(basis_values_np)

    use_absorption = (solver == "global_absorption")
    use_diff_omniscape = (solver == "diff_omniscape")

    # Random loss direction (fixed across perturbations)
    dl_dC = torch.randn(n_valid, dtype=torch.float64)

    def _forward_loss(net_):
        R, _ = net_.forward_with_log_R(basis_t)
        if use_diff_omniscape:
            C = _DiffOmniscapeSolveFn.apply(
                R, valid_mask_np, n_rows, n_cols,
                source_spacing, source_from_resistance,
                radius, block_size, 1.0, seed,
            )
        elif use_absorption:
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
        if net_.intensity_spline:
            # Test the full intensity spline path
            log1p_C = torch.log1p(C.clamp(min=0))
            f_int = net_.eval_intensity_spline(log1p_C)
            return (f_int * dl_dC).sum()
        else:
            return (C * dl_dC).sum()

    # Analytic gradient
    loss = _forward_loss(net)
    loss.backward()

    # Collect subset of parameters to check
    all_params = [(n, p, p.grad.detach().cpu().numpy().ravel().copy())
                  for n, p in net.named_parameters()
                  if p.requires_grad and p.grad is not None]
    n_check_per_param = 3
    fd_results = []

    for pname, param, analytic_g in all_params:
        n_elem = len(analytic_g)
        check_idx = np.random.choice(n_elem, min(n_check_per_param, n_elem),
                                     replace=False)

        for idx in check_idx:
            orig = param.data.view(-1)[idx].item()

            param.data.view(-1)[idx] = orig + eps
            net.zero_grad()
            lp = _forward_loss(net).item()

            param.data.view(-1)[idx] = orig - eps
            net.zero_grad()
            lm = _forward_loss(net).item()

            param.data.view(-1)[idx] = orig

            fd = (lp - lm) / (2 * eps)
            an = analytic_g[idx]
            rel_err = abs(fd - an) / (max(abs(fd), abs(an)) + 1e-7)
            fd_results.append((pname, idx, fd, an, rel_err))

    rel_errors = [r[4] for r in fd_results]
    max_rel = float(max(rel_errors))

    print(f"  Spline gradient check: max rel error = {max_rel:.2e} "
          f"(checked {len(fd_results)} param elements across "
          f"{len(all_params)} params)"
          f"{' [intensity spline ON]' if intensity_spline else ''}")
    for pname, idx, fd, an, re in fd_results:
        status = "OK" if re < 0.05 else "FAIL"
        print(f"    {pname}[{idx}]: FD={fd:.6e}, AN={an:.6e}, "
              f"rel={re:.2e} {status}")

    return {
        "max_rel_error": max_rel,
        "pass": max_rel < 0.05,
        "rel_errors": [float(r) for r in rel_errors],
    }

