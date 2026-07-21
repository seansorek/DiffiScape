"""Resistance-mapping neural network / GAM models.

Contains the five resistance net classes:
  - LogLinearResistanceNet: log-linear covariate model.
  - ResistanceNet: MLP with linear skip connection.
  - ConvResistanceNet (+ _ConvResBlock): convolutional encoder + MLP head.
  - IRLResistanceNet: inverse-reinforcement-learning value-shaped resistance.
  - SplineResistanceNet: penalised additive (P-spline GAM) model, with its
    B-spline helpers (_bspline_knots, _bspline_basis_matrix,
    _bspline_basis_matrix_torch, _diff_penalty_matrix).

Also hosts _ppp_loglik, the shared Poisson point-process log-likelihood used
by run_torch_optimization and all three Bayesian samplers.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import DEFAULT_R_MIN, DEFAULT_R_MAX, DEFAULT_CLAMP_BETA


# ===========================================================================
# Neural-network resistance model
# ===========================================================================

class LogLinearResistanceNet(nn.Module):
    """Log R(x) = r_0 + z · φ(x)."""

    def __init__(self, n_features=4, R_min=DEFAULT_R_MIN,
                 R_max=DEFAULT_R_MAX, beta=DEFAULT_CLAMP_BETA):
        super().__init__()
        self.R_min = R_min
        self.R_max = R_max
        self.beta = beta
        self.skip = nn.Linear(n_features, 1)
        nn.init.normal_(self.skip.weight, std=0.1)
        nn.init.constant_(self.skip.bias, 3.0)

    def warm_start(self, theta):
        theta = np.asarray(theta, dtype=np.float64).ravel()
        with torch.no_grad():
            self.skip.weight.copy_(torch.tensor([theta[1:]], dtype=torch.float64))
            self.skip.bias.copy_(torch.tensor([theta[0]], dtype=torch.float64))

    def forward(self, x):
        R, _ = self.forward_with_log_R(x)
        return R

    def forward_with_log_R(self, x):
        log_R = self.skip(x).squeeze(-1)
        log_lo, log_hi = math.log(self.R_min), math.log(self.R_max)
        clamped = log_lo + F.softplus(log_R - log_lo, beta=self.beta)
        clamped = log_hi - F.softplus(log_hi - clamped, beta=self.beta)
        return torch.exp(clamped), log_R


class ResistanceNet(nn.Module):
    """
    MLP with linear skip (residual) connection for resistance mapping.

    log R(x) = linear(φ(x)) + MLP(φ(x))

    The linear branch starts as the log-linear model (r_0 + z · φ).
    The MLP starts at zero, so the network begins at the log-linear baseline
    and learns nonlinear corrections (interactions, thresholds, etc.)
    """

    def __init__(self, n_features=4, hidden=32, n_layers=2,
                 R_min=DEFAULT_R_MIN, R_max=DEFAULT_R_MAX, beta=DEFAULT_CLAMP_BETA):
        super().__init__()
        self.R_min = R_min
        self.R_max = R_max
        self.beta = beta

        # Linear skip connection (matches the log-linear model exactly)
        self.skip = nn.Linear(n_features, 1)

        # Nonlinear MLP (learns corrections to the linear model)
        layers = []
        d = n_features
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Start near R ≈ exp(3) ≈ 20 with spatial variation from covariates."""
        # Non-zero skip weights break the flat-surface saddle point.
        # Without this, var(logR)=0 so d(var_penalty)/d(weights)=0 identically.
        nn.init.normal_(self.skip.weight, std=0.1)
        nn.init.constant_(self.skip.bias, 3.0)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.05)
                nn.init.zeros_(m.bias)
        # Last MLP layer small → MLP output ≈ 0 initially
        last = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(last.weight, std=0.01)
        nn.init.zeros_(last.bias)

    def warm_start(self, theta):
        """Initialize skip from existing log-linear params [r_0, z_1..z_4]."""
        theta = np.asarray(theta, dtype=np.float64).ravel()
        with torch.no_grad():
            self.skip.weight.copy_(
                torch.tensor([theta[1:]], dtype=torch.float64)
            )
            self.skip.bias.copy_(
                torch.tensor([theta[0]], dtype=torch.float64)
            )
            # Reset MLP to zero
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        R, _ = self.forward_with_log_R(x)
        return R

    def forward_with_log_R(self, x):
        """Return (R, log_R_raw) where log_R_raw is pre-clamp for regularization."""
        log_R = (self.skip(x) + self.mlp(x)).squeeze(-1)
        # Smooth double-softplus clamp in LOG-space, then exponentiate.
        # This guarantees R ∈ [R_min, R_max] with nonzero gradients
        # everywhere.  The old approach (clamp → exp → softplus in R-space)
        # created gradient dead zones when exp() overshot R_max.
        log_lo = math.log(self.R_min)   # 0
        log_hi = math.log(self.R_max)   # ~8.52
        clamped = log_lo + F.softplus(log_R - log_lo, beta=self.beta)
        clamped = log_hi - F.softplus(log_hi - clamped, beta=self.beta)
        return torch.exp(clamped), log_R


class _ConvResBlock(nn.Module):
    """Conv2d + GroupNorm + SiLU residual block with optional dilation."""

    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(channels, channels, kernel_size,
                              padding=pad, dilation=dilation)
        self.norm = nn.GroupNorm(1, channels)  # num_groups=1 → LayerNorm-like
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.norm(self.conv(x)))


class ConvResistanceNet(nn.Module):
    """
    Convolutional encoder + MLP head with linear skip for resistance mapping.

    log R(x) = skip(φ(x)) + MLP(conv_features(x))

    Improvements over the basic sequential version:
      - GroupNorm (num_groups=1, i.e. LayerNorm) after each conv for stable
        activations at batch_size=1.
      - Residual connections in the conv encoder (skip = identity after the
        projection layer) for better gradient flow.
      - Multi-scale dilated convolutions: dilation increases with depth
        (1, 2, 4, ...) so the receptive field grows exponentially without
        more parameters.
      - Dropout in the MLP head (disabled at eval/final forward).
      - Learned intensity MLP on connectivity (optional, replaces
        the fixed log λ = α + γ log(1+C) parametric form).

    Conv weights are initialised small so the network starts near the
    skip-only (log-linear) baseline.
    """

    def __init__(self, n_features=4, conv_channels=16, n_conv_layers=3,
                 conv_kernel_size=3, hidden=16, n_mlp_layers=1,
                 R_min=DEFAULT_R_MIN, R_max=DEFAULT_R_MAX, beta=DEFAULT_CLAMP_BETA,
                 dropout=0.0, use_dilated=True,
                 intensity_hidden=0):
        super().__init__()
        self.R_min = R_min
        self.R_max = R_max
        self.beta = beta
        self.conv_channels = conv_channels

        # --- Input projection: n_features → conv_channels ---
        pad0 = conv_kernel_size // 2
        self.input_proj = nn.Sequential(
            nn.Conv2d(n_features, conv_channels, conv_kernel_size, padding=pad0),
            nn.GroupNorm(1, conv_channels),
            nn.SiLU(),
        )

        # --- Residual conv blocks with increasing dilation ---
        blocks = []
        for i in range(n_conv_layers):
            dil = (2 ** i) if use_dilated else 1
            blocks.append(_ConvResBlock(conv_channels, conv_kernel_size,
                                        dilation=dil))
        self.conv_encoder = nn.Sequential(*blocks)

        # --- Pointwise MLP head on conv features (with dropout) ---
        mlp_layers = []
        d = conv_channels
        for _ in range(n_mlp_layers):
            mlp_layers += [nn.Linear(d, hidden), nn.SiLU()]
            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))
            d = hidden
        mlp_layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*mlp_layers)

        # --- Linear skip on raw covariates (warm-start compatible) ---
        self.skip = nn.Linear(n_features, 1)

        # --- Optional learned intensity MLP ---
        self.intensity_hidden = intensity_hidden
        if intensity_hidden > 0:
            self.intensity_mlp = nn.Sequential(
                nn.Linear(1, intensity_hidden),
                nn.SiLU(),
                nn.Linear(intensity_hidden, 1),
            )
        else:
            self.intensity_mlp = None

        self._init_weights()

    def _init_weights(self):
        """Small init so conv+MLP ≈ 0 initially → starts at skip baseline."""
        nn.init.normal_(self.skip.weight, std=0.1)
        nn.init.constant_(self.skip.bias, 3.0)

        for m in self.input_proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

        for blk in self.conv_encoder:
            if isinstance(blk, _ConvResBlock):
                nn.init.normal_(blk.conv.weight, std=0.02)
                nn.init.zeros_(blk.conv.bias)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.05)
                nn.init.zeros_(m.bias)
        # Last MLP layer very small → MLP output ≈ 0 initially
        last = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(last.weight, std=0.01)
        nn.init.zeros_(last.bias)

        # Intensity MLP: init so output ≈ log1p(C) (identity passthrough)
        if self.intensity_mlp is not None:
            layers = [m for m in self.intensity_mlp.modules()
                      if isinstance(m, nn.Linear)]
            # First layer: weight≈1, bias≈0 → pass through
            nn.init.constant_(layers[0].weight, 1.0)
            nn.init.zeros_(layers[0].bias)
            # Last layer: weight≈1, bias≈0 → near-identity
            nn.init.constant_(layers[-1].weight, 1.0)
            nn.init.zeros_(layers[-1].bias)

    def warm_start(self, theta):
        """Initialize skip from existing log-linear params [r_0, z_1..z_4]."""
        theta = np.asarray(theta, dtype=np.float64).ravel()
        with torch.no_grad():
            self.skip.weight.copy_(
                torch.tensor(theta[1:].reshape(1, -1), dtype=torch.float64)
            )
            self.skip.bias.copy_(
                torch.tensor([theta[0]], dtype=torch.float64)
            )
            # Reset conv + MLP to near-zero
            for m in self.input_proj.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.zeros_(m.bias)
            for blk in self.conv_encoder:
                if isinstance(blk, _ConvResBlock):
                    nn.init.normal_(blk.conv.weight, std=0.001)
                    nn.init.zeros_(blk.conv.bias)
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward_with_log_R(self, basis_grid, valid_mask_flat, basis_valid):
        """
        Parameters
        ----------
        basis_grid : Tensor (1, n_features, H, W)
            Full 2D covariate grid (NaN cells zero-filled).
        valid_mask_flat : ndarray (H*W,) bool
            Which cells are valid.
        basis_valid : Tensor (n_valid, n_features)
            Raw covariates at valid pixels (for skip connection).

        Returns
        -------
        R : Tensor (n_valid,)
        log_R_raw : Tensor (n_valid,)   (pre-clamp, for regularisation)
        """
        # Input projection + residual conv blocks: (1, C_in, H, W) → (1, C_conv, H, W)
        h = self.input_proj(basis_grid)
        feat_map = self.conv_encoder(h)

        # Extract valid pixels: (1, C_conv, H, W) → (n_valid, C_conv)
        feat_flat = feat_map.squeeze(0).reshape(self.conv_channels, -1).T  # (H*W, C_conv)
        feat_valid = feat_flat[valid_mask_flat]  # (n_valid, C_conv)

        # MLP head on conv features + skip on raw covariates
        mlp_out = self.mlp(feat_valid).squeeze(-1)        # (n_valid,)
        skip_out = self.skip(basis_valid).squeeze(-1)      # (n_valid,)
        log_R = skip_out + mlp_out

        # Smooth double-softplus clamp in log-space
        log_lo = math.log(self.R_min)
        log_hi = math.log(self.R_max)
        clamped = log_lo + F.softplus(log_R - log_lo, beta=self.beta)
        clamped = log_hi - F.softplus(log_hi - clamped, beta=self.beta)
        return torch.exp(clamped), log_R

    def forward(self, basis_grid, valid_mask_flat, basis_valid):
        R, _ = self.forward_with_log_R(basis_grid, valid_mask_flat, basis_valid)
        return R


class IRLResistanceNet(nn.Module):
    """
    Inverse-reinforcement-learning (value-shaped) resistance model.

    A learned reward field over covariates is turned into a resistance surface
    via entropy-regularised soft value iteration on the 4-connected grid MDP:

        r(x)        = skip(phi(x)) + MLP(phi(x))                  # reward (negative cost)
        V_{t+1}(s)  = (1/b) logsumexp_{a in {stay,N,S,E,W}} b [ r(s) + gd * V_t(s_a) ]
        log R(x)    = clamp( offset - scale * V(x) )              # high value -> low resistance

    The soft value iteration is a smooth (logsumexp) contraction for the
    discount ``gamma_d < 1``, so V — and therefore R — is differentiable w.r.t.
    the reward parameters; gradients from the downstream circuit solve flow back
    through the unrolled iteration into the reward network.  Resistance encodes
    the long-range desirability of the landscape (an agent's plan-to-go value)
    rather than purely local habitat.

    The forward signature matches :class:`ConvResistanceNet`: value iteration
    needs the full 2D grid for spatial adjacency.  ``R`` and ``log_R_raw`` are
    returned per valid pixel, so the circuit solver / intensity link / samplers
    consume this model identically to the MLP / conv / spline models.
    """

    def __init__(self, n_features=4, hidden=32, n_layers=2,
                 R_min=DEFAULT_R_MIN, R_max=DEFAULT_R_MAX, beta=DEFAULT_CLAMP_BETA,
                 vi_beta=1.0, gamma_d=0.9, n_value_iter=60,
                 value_scale_init=1.0):
        super().__init__()
        self.R_min = R_min
        self.R_max = R_max
        self.beta = beta                  # log-space clamp sharpness
        self.vi_beta = float(vi_beta)     # soft value-iteration temperature
        self.gamma_d = float(gamma_d)     # discount / leakage (must be < 1)
        self.n_value_iter = int(n_value_iter)

        # Reward field = linear skip (warm-start compatible) + nonlinear MLP.
        self.skip = nn.Linear(n_features, 1)
        layers = []
        d = n_features
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

        # log R = offset - softplus(raw_scale) * V   (scale > 0 keeps the sign:
        # higher value -> lower resistance).
        self.offset = nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
        _raw_scale = math.log(math.expm1(max(float(value_scale_init), 1e-3)))
        self.raw_scale = nn.Parameter(torch.tensor(_raw_scale, dtype=torch.float64))

        # No learned intensity transform (uses the parametric link).
        self.intensity_mlp = None

        self._init_weights()

    def _init_weights(self):
        """Small reward variation so V starts smooth; MLP correction ~ 0."""
        nn.init.normal_(self.skip.weight, std=0.1)
        nn.init.zeros_(self.skip.bias)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.05)
                nn.init.zeros_(m.bias)
        last = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(last.weight, std=0.01)
        nn.init.zeros_(last.bias)

    def warm_start(self, theta):
        """Initialise the reward skip from log-linear params [r_0, z_1..z_K].

        The covariate weights are sign-flipped because reward = -cost: a
        covariate direction that raises resistance lowers reward.  The intercept
        seeds ``offset`` (the resistance level)."""
        theta = np.asarray(theta, dtype=np.float64).ravel()
        with torch.no_grad():
            self.skip.weight.copy_(
                torch.tensor(-theta[1:].reshape(1, -1), dtype=torch.float64)
            )
            self.skip.bias.zero_()
            self.offset.copy_(torch.tensor(theta[0], dtype=torch.float64))
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _shift(t, dr, dc):
        """Neighbour lookup: out[r, c] = t[r + dr, c + dc]; out-of-grid -> 0."""
        H, W = t.shape
        out = torch.zeros_like(t)
        r0s, r1s = max(0, dr), min(H, H + dr)
        c0s, c1s = max(0, dc), min(W, W + dc)
        r0d, r1d = max(0, -dr), min(H, H - dr)
        c0d, c1d = max(0, -dc), min(W, W - dc)
        out[r0d:r1d, c0d:c1d] = t[r0s:r1s, c0s:c1s]
        return out

    def _soft_value_iteration(self, reward_grid, valid_grid):
        """Entropy-regularised soft value iteration on the 4-connected grid.

        ``reward_grid`` and ``valid_grid`` are (H, W); ``valid_grid`` is float
        0/1.  Moves into out-of-grid or invalid cells are masked out of the
        soft maximisation; the always-available 'stay' action guarantees at
        least one finite term.  Invalid cells are pinned to V = 0 so they never
        leak value into valid neighbours.

        Peak memory scales with n_value_iter × grid_size because the loop is
        fully unrolled into the autograd graph; reduce n_value_iter if OOM."""
        b, gd, NEG = self.vi_beta, self.gamma_d, -1e9
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        nbr_valid = [self._shift(valid_grid, dr, dc) for dr, dc in dirs]
        V = torch.zeros_like(reward_grid)
        for _ in range(self.n_value_iter):
            qs = [reward_grid + gd * V]  # stay action (always valid)
            for (dr, dc), nv in zip(dirs, nbr_valid):
                q = reward_grid + gd * self._shift(V, dr, dc)
                qs.append(torch.where(nv > 0.5, q, torch.full_like(q, NEG)))
            V_new = torch.logsumexp(b * torch.stack(qs, dim=0), dim=0) / b
            V = torch.where(valid_grid > 0.5, V_new, torch.zeros_like(V_new))
        return V

    def forward_with_log_R(self, basis_grid, valid_mask_flat, basis_valid):
        """See :meth:`ConvResistanceNet.forward_with_log_R` for arg shapes."""
        _, n_feat, H, W = basis_grid.shape
        phi_all = basis_grid.squeeze(0).reshape(n_feat, -1).T   # (H*W, n_features)
        reward_grid = (self.skip(phi_all) + self.mlp(phi_all)).squeeze(-1).reshape(H, W)

        valid_grid = torch.as_tensor(
            np.asarray(valid_mask_flat, dtype=np.float64).reshape(H, W),
            dtype=reward_grid.dtype, device=reward_grid.device,
        )
        V = self._soft_value_iteration(reward_grid, valid_grid)
        V_valid = V.reshape(-1)[valid_mask_flat]               # (n_valid,)

        # Centre V: the absolute value level is an arbitrary constant (soft VI
        # accumulates a near-uniform entropy/discount offset).  Only spatial
        # contrasts in V should drive resistance, and centring makes ``offset``
        # the interpretable mean log-resistance (anchored by the mean penalty).
        V_valid = V_valid - V_valid.mean()
        scale = F.softplus(self.raw_scale)
        log_R = self.offset - scale * V_valid

        # Smooth double-softplus clamp in log-space (matches the other nets).
        log_lo = math.log(self.R_min)
        log_hi = math.log(self.R_max)
        clamped = log_lo + F.softplus(log_R - log_lo, beta=self.beta)
        clamped = log_hi - F.softplus(log_hi - clamped, beta=self.beta)
        return torch.exp(clamped), log_R

    def forward(self, basis_grid, valid_mask_flat, basis_valid):
        R, _ = self.forward_with_log_R(basis_grid, valid_mask_flat, basis_valid)
        return R


# ===========================================================================
# B-spline utilities for SplineResistanceNet
# ===========================================================================

def _bspline_knots(n_internal, degree=3, xmin=0.0, xmax=1.0):
    """
    Create B-spline knot vector with boundary knots repeated degree+1 times.

    Returns
    -------
    knots : ndarray (n_internal + 2*degree + 2,)
    """
    internal = np.linspace(xmin, xmax, n_internal + 2)  # includes endpoints
    knots = np.concatenate([
        np.repeat(xmin, degree),
        internal,
        np.repeat(xmax, degree),
    ])
    return knots


def _bspline_basis_matrix(x_np, n_internal=10, degree=3, xmin=0.0, xmax=1.0):
    """
    Evaluate cubic B-spline basis at x values using de Boor recursion.

    Parameters
    ----------
    x_np : ndarray (n,)
        Evaluation points (should be in [xmin, xmax]).
    n_internal : int
        Number of internal knots (total basis functions = n_internal + degree + 1).
    degree : int
        B-spline degree (3 = cubic).

    Returns
    -------
    B : ndarray (n, n_basis)
        Basis matrix.
    knots : ndarray
        Full knot vector.
    """
    knots = _bspline_knots(n_internal, degree, xmin, xmax)
    n_basis = len(knots) - degree - 1
    n = len(x_np)

    # Clamp x to [xmin, xmax] to avoid extrapolation issues.
    # Pull boundary points epsilon inward so the last knot span captures x=xmax
    # (standard B-spline convention: last span is [t_{n-1}, t_n] closed on right
    # but de Boor recursion with repeated boundary knots loses it).
    x = np.clip(x_np, xmin, xmax)
    eps = (xmax - xmin) * 1e-10
    x = np.where(x >= xmax, xmax - eps, x)

    # de Boor recursion
    # Start with degree-0 (piecewise constant) basis
    B = np.zeros((n, n_basis + degree), dtype=np.float64)
    for j in range(n_basis + degree):
        if j < len(knots) - 1:
            if j == len(knots) - 2:
                # Last interval: closed on right
                B[:, j] = ((x >= knots[j]) & (x <= knots[j + 1])).astype(np.float64)
            else:
                B[:, j] = ((x >= knots[j]) & (x < knots[j + 1])).astype(np.float64)

    # Recurse up to target degree
    for d in range(1, degree + 1):
        B_new = np.zeros((n, n_basis + degree - d), dtype=np.float64)
        for j in range(B_new.shape[1]):
            left_denom = knots[j + d] - knots[j]
            right_denom = knots[j + d + 1] - knots[j + 1]
            if left_denom > 0:
                B_new[:, j] += (x - knots[j]) / left_denom * B[:, j]
            if right_denom > 0:
                B_new[:, j] += (knots[j + d + 1] - x) / right_denom * B[:, j + 1]
        B = B_new

    return B[:, :n_basis], knots


def _bspline_basis_matrix_torch(x, knots_t, n_basis, degree=3):
    """
    Differentiable B-spline basis evaluation using de Boor recursion in PyTorch.

    Unlike `_bspline_basis_matrix` (numpy, non-differentiable), this operates
    on a torch tensor *x* and returns a basis matrix through which autograd
    can back-propagate.  Used for the intensity spline where the input
    log(1 + C) changes every optimisation step.

    Parameters
    ----------
    x : Tensor (n,)
        Evaluation points.
    knots_t : Tensor (n_knots_total,)
        Full B-spline knot vector (same convention as `_bspline_knots`).
    n_basis : int
        Number of basis functions.
    degree : int
        B-spline degree (3 = cubic).

    Returns
    -------
    B : Tensor (n, n_basis)
        Basis matrix (differentiable w.r.t. *x*).
    """
    n = x.shape[0]
    xmin = knots_t[0].item()
    xmax = knots_t[-1].item()
    eps = (xmax - xmin) * 1e-10

    # Soft clamp into [xmin, xmax - eps] — differentiable
    x_c = x.clamp(min=xmin, max=xmax - eps)

    n_intervals = n_basis + degree
    # Degree-0 basis (piecewise constant): B[j](x) = 1_{x in [t_j, t_{j+1})}
    # Use detached knots for interval indicators (no grad needed for knots).
    knots_np = knots_t.detach()
    # Build (n, n_intervals) indicator matrix.  These 0/1 masks are not
    # differentiated — all x-dependence enters through the recursion below.
    cols = []
    for j in range(n_intervals):
        if j < len(knots_np) - 1:
            lo = knots_np[j]
            hi = knots_np[j + 1]
            if j == n_intervals - 1:
                # Last span: closed on right
                indicator = ((x_c.detach() >= lo) & (x_c.detach() <= hi)).double()
            else:
                indicator = ((x_c.detach() >= lo) & (x_c.detach() < hi)).double()
        else:
            indicator = torch.zeros(n, dtype=x.dtype, device=x.device)
        cols.append(indicator)
    B = torch.stack(cols, dim=1)  # (n, n_intervals)

    # De Boor recursion: degree 1 … degree
    for d in range(1, degree + 1):
        n_new = n_intervals - d
        new_cols = []
        for j in range(n_new):
            left_denom = (knots_np[j + d] - knots_np[j]).item()
            right_denom = (knots_np[j + d + 1] - knots_np[j + 1]).item()

            val = torch.zeros(n, dtype=x.dtype, device=x.device)
            if left_denom > 0:
                val = val + (x_c - knots_np[j].item()) / left_denom * B[:, j]
            if right_denom > 0:
                val = val + (knots_np[j + d + 1].item() - x_c) / right_denom * B[:, j + 1]
            new_cols.append(val)
        B = torch.stack(new_cols, dim=1)

    return B[:, :n_basis]


def _diff_penalty_matrix(n_basis, order=2):
    """
    Second-order difference penalty matrix S = D2^T @ D2.

    For P-spline smoothing: penalty = β^T S β penalises wiggliness.

    Returns
    -------
    S : ndarray (n_basis, n_basis)
    """
    D = np.eye(n_basis, dtype=np.float64)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return D.T @ D


class SplineResistanceNet(nn.Module):
    """
    Penalised additive model (P-spline GAM) for resistance mapping.

    log R(x) = r_0 + Σ_k f_k(φ_k(x)) + Σ_{j<k} f_{jk}(φ_j(x), φ_k(x))

    Each f_k is a cubic B-spline with second-order difference penalty.
    Each f_{jk} is a tensor product of two B-spline bases with marginal penalties.
    Per-term smoothing parameters λ_k are learnable (log-parameterised).

    This model sits between the rigid log-linear (5 params) and the flexible
    MLP (~1250 params): it has ~1000+ params but strong regularisation via
    smoothing penalties, so it is much less prone to overfitting than the NN
    while still capturing nonlinear covariate effects and interactions.
    """

    def __init__(self, n_features=5, n_knots=10, degree=3,
                 include_interactions=True,
                 R_min=DEFAULT_R_MIN, R_max=DEFAULT_R_MAX, beta=DEFAULT_CLAMP_BETA,
                 lambda_init_marginal=0.0, lambda_init_interaction=2.0,
                 lambda_min=0.0,
                 intensity_spline=False, intensity_n_knots=5,
                 intensity_degree=3, lambda_init_intensity=2.0,
                 intensity_log1p_max=10.0):
        super().__init__()
        self.n_features = n_features
        self.n_knots = n_knots
        self.degree = degree
        self.include_interactions = include_interactions
        self.R_min = R_min
        self.R_max = R_max
        self.beta = beta
        self.lambda_init_marginal = lambda_init_marginal
        self.lambda_init_interaction = lambda_init_interaction
        self.lambda_min = lambda_min

        # B-spline basis info (precomputed in setup_basis_matrices)
        self.n_basis = n_knots + degree + 1  # per covariate
        self._basis_matrices = None  # set by setup_basis_matrices()
        self._penalty_matrices = None

        # Skip connection (log-linear component, for warm start compatibility)
        self.skip = nn.Linear(n_features, 1)

        # Intercept (r_0)
        self.intercept = nn.Parameter(torch.tensor(3.0, dtype=torch.float64))

        # Marginal spline coefficients: one set per covariate
        self.spline_coefs = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n_basis, dtype=torch.float64))
            for _ in range(n_features)
        ])

        # Per-term log smoothing parameters (marginal)
        self.log_lambda_smooth = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            for _ in range(n_features)
        ])

        # Interaction terms (all pairwise)
        self.interaction_pairs = []
        if include_interactions:
            for j in range(n_features):
                for k in range(j + 1, n_features):
                    self.interaction_pairs.append((j, k))

        n_interact = len(self.interaction_pairs)
        n_tp_basis = self.n_basis ** 2

        self.interaction_coefs = nn.ParameterList([
            nn.Parameter(torch.zeros(n_tp_basis, dtype=torch.float64))
            for _ in range(n_interact)
        ])

        # Per-interaction log smoothing (2 per pair: row and column marginal)
        self.log_lambda_interact_row = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            for _ in range(n_interact)
        ])
        self.log_lambda_interact_col = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            for _ in range(n_interact)
        ])

        # --- Intensity spline: f_intensity(log(1+C)) ---
        self.intensity_spline = intensity_spline
        self.intensity_n_knots = intensity_n_knots
        self.intensity_degree = intensity_degree
        self.lambda_init_intensity = lambda_init_intensity
        self.intensity_log1p_max = intensity_log1p_max

        if intensity_spline:
            self.intensity_n_basis = intensity_n_knots + intensity_degree + 1
            # Knot vector on [0, intensity_log1p_max] — registered as buffer
            knots_np = _bspline_knots(intensity_n_knots, intensity_degree,
                                      xmin=0.0, xmax=intensity_log1p_max)
            self.register_buffer(
                "_intensity_knots",
                torch.tensor(knots_np, dtype=torch.float64))
            # Penalty matrix (second-order difference)
            S_np = _diff_penalty_matrix(self.intensity_n_basis, order=2)
            self.register_buffer(
                "_intensity_penalty",
                torch.tensor(S_np, dtype=torch.float64))
            # Learnable spline coefficients
            self.intensity_coefs = nn.Parameter(
                torch.zeros(self.intensity_n_basis, dtype=torch.float64))
            # Learnable log-smoothing parameter
            self.log_lambda_intensity = nn.Parameter(
                torch.tensor(lambda_init_intensity, dtype=torch.float64))
        else:
            self.intensity_n_basis = 0
            self.intensity_coefs = None
            self.log_lambda_intensity = None

        self._init_weights()

    def _init_weights(self):
        """Small init for spline coefficients; skip near log-linear baseline."""
        nn.init.normal_(self.skip.weight, std=0.1)
        nn.init.constant_(self.skip.bias, 0.0)  # intercept handled separately

        for coef in self.spline_coefs:
            nn.init.normal_(coef, std=0.01)
        for coef in self.interaction_coefs:
            nn.init.normal_(coef, std=0.001)

        # Smoothing params: configurable initial values
        for lam in self.log_lambda_smooth:
            nn.init.constant_(lam, self.lambda_init_marginal)
        for lam in self.log_lambda_interact_row:
            nn.init.constant_(lam, self.lambda_init_interaction)
        for lam in self.log_lambda_interact_col:
            nn.init.constant_(lam, self.lambda_init_interaction)

        # Intensity spline: small init so it starts near linear
        if self.intensity_spline:
            nn.init.normal_(self.intensity_coefs, std=0.01)
            nn.init.constant_(self.log_lambda_intensity, self.lambda_init_intensity)

    def setup_basis_matrices(self, basis_values_np, device=None):
        """
        Precompute B-spline basis matrices and penalty matrices.

        Must be called once after construction with the actual covariate values.

        Parameters
        ----------
        basis_values_np : ndarray (n_valid, n_features)
            Covariate values at valid pixels.
        device : torch.device or None
        """
        if device is None:
            device = self.intercept.device

        n_valid, n_feat = basis_values_np.shape
        assert n_feat == self.n_features

        basis_mats = []
        penalty_mats = []

        # Marginal penalty matrix (same for all covariates)
        S_np = _diff_penalty_matrix(self.n_basis, order=2)
        S_t = torch.tensor(S_np, dtype=torch.float64, device=device)

        for k in range(n_feat):
            B_np, _ = _bspline_basis_matrix(
                basis_values_np[:, k],
                n_internal=self.n_knots,
                degree=self.degree,
            )
            B_t = torch.tensor(B_np, dtype=torch.float64, device=device)
            basis_mats.append(B_t)
            penalty_mats.append(S_t)

        # Tensor product basis matrices for interactions
        tp_basis_mats = []
        tp_penalty_row = []
        tp_penalty_col = []
        I_nb = torch.eye(self.n_basis, dtype=torch.float64, device=device)

        for j, k in self.interaction_pairs:
            B_j = basis_mats[j]  # (n_valid, n_basis)
            B_k = basis_mats[k]  # (n_valid, n_basis)
            # Tensor product basis: (n_valid, n_basis^2)
            # B_tp[i, a*n_basis + b] = B_j[i, a] * B_k[i, b]
            n_b = self.n_basis
            B_tp = (B_j.unsqueeze(2) * B_k.unsqueeze(1)).reshape(n_valid, n_b * n_b)
            tp_basis_mats.append(B_tp)

            # Row penalty: S_j ⊗ I  (penalise wiggliness in j-direction)
            S_row = torch.kron(S_t, I_nb)
            # Col penalty: I ⊗ S_k  (penalise wiggliness in k-direction)
            S_col = torch.kron(I_nb, S_t)
            tp_penalty_row.append(S_row)
            tp_penalty_col.append(S_col)

        self._basis_matrices = basis_mats
        self._penalty_matrices = penalty_mats
        self._tp_basis_matrices = tp_basis_mats
        self._tp_penalty_row = tp_penalty_row
        self._tp_penalty_col = tp_penalty_col

    def warm_start(self, theta):
        """
        Initialize from log-linear params [r_0, z_1, ..., z_K].

        Sets the intercept to r_0 and projects each z_k onto the B-spline
        basis (as a linear function: coefs proportional to knot positions).
        """
        theta = np.asarray(theta, dtype=np.float64).ravel()
        with torch.no_grad():
            self.intercept.copy_(torch.tensor(theta[0], dtype=torch.float64))
            # Set skip to the same log-linear model
            self.skip.weight.copy_(
                torch.tensor(theta[1:].reshape(1, -1), dtype=torch.float64)
            )
            self.skip.bias.copy_(torch.tensor([0.0], dtype=torch.float64))

            # Project z_k into spline basis: f_k(φ) ≈ z_k * φ
            # The B-spline basis spans polynomials up to degree d,
            # so we solve min_β ||B β - z_k * x||^2 for each covariate.
            if self._basis_matrices is not None:
                for k in range(min(len(theta) - 1, self.n_features)):
                    z_k = theta[k + 1]
                    B = self._basis_matrices[k]  # (n_valid, n_basis)
                    # Target: z_k * φ_k (the covariate values)
                    # φ_k is embedded in the basis: we want spline ≈ z_k * φ_k
                    # but we only have B, not φ_k directly.
                    # Use skip for the linear part, zero the spline coefs.
                    self.spline_coefs[k].zero_()

            # Zero interaction coefs
            for coef in self.interaction_coefs:
                coef.zero_()

    def forward_with_log_R(self, x):
        """
        Parameters
        ----------
        x : Tensor (n_valid, n_features)
            Covariate values at valid pixels.

        Returns
        -------
        R : Tensor (n_valid,)
        log_R_raw : Tensor (n_valid,)  (pre-clamp, for regularisation)
        """
        if self._basis_matrices is None:
            raise RuntimeError(
                "Call setup_basis_matrices() before forward pass")

        # Skip (linear) + intercept
        log_R = self.skip(x).squeeze(-1) + self.intercept

        # Marginal smooth terms: Σ_k B_k @ β_k
        for k in range(self.n_features):
            B_k = self._basis_matrices[k]
            log_R = log_R + B_k @ self.spline_coefs[k]

        # Tensor product interaction terms: Σ_{j<k} B_tp @ β_{jk}
        for idx, (j, k) in enumerate(self.interaction_pairs):
            B_tp = self._tp_basis_matrices[idx]
            log_R = log_R + B_tp @ self.interaction_coefs[idx]

        # Double-softplus clamp in log-space
        log_lo = math.log(self.R_min)
        log_hi = math.log(self.R_max)
        clamped = log_lo + F.softplus(log_R - log_lo, beta=self.beta)
        clamped = log_hi - F.softplus(log_hi - clamped, beta=self.beta)
        return torch.exp(clamped), log_R

    def forward(self, x):
        R, _ = self.forward_with_log_R(x)
        return R

    def eval_intensity_spline(self, log1p_C):
        """
        Evaluate the intensity P-spline: f_intensity(log(1 + C)).

        The result is centered (pixel-mean subtracted) so that the intercept
        alpha remains identifiable as the overall log-intensity level.

        Parameters
        ----------
        log1p_C : Tensor (n_valid,)
            log(1 + C) at valid pixels.

        Returns
        -------
        f_val : Tensor (n_valid,)
            Centered spline contribution to log-intensity.
        """
        if not self.intensity_spline:
            raise RuntimeError("intensity_spline is not enabled")
        B = _bspline_basis_matrix_torch(
            log1p_C, self._intensity_knots,
            self.intensity_n_basis, self.intensity_degree)
        f_raw = B @ self.intensity_coefs       # (n_valid,)
        return f_raw - f_raw.mean()            # center for identifiability

    def smoothing_penalty(self):
        """
        Compute the total smoothing penalty.

        Returns Σ_k λ_k β_k^T S_k β_k + Σ_{jk} [λ_row β^T S_row β + λ_col β^T S_col β]

        The result is a differentiable scalar (torch autograd tracks gradients
        through β AND log_λ).

        lambda_min provides a floor: λ = lambda_min + exp(log_lambda), preventing
        the optimizer from turning off smoothing entirely.
        """
        pen = torch.tensor(0.0, dtype=torch.float64,
                           device=self.intercept.device)

        # Marginal penalties
        for k in range(self.n_features):
            lam = self.lambda_min + torch.exp(self.log_lambda_smooth[k])
            beta_k = self.spline_coefs[k]
            S_k = self._penalty_matrices[k]
            pen = pen + lam * (beta_k @ S_k @ beta_k)

        # Interaction penalties (row + column marginal)
        for idx in range(len(self.interaction_pairs)):
            beta_tp = self.interaction_coefs[idx]
            lam_r = self.lambda_min + torch.exp(self.log_lambda_interact_row[idx])
            lam_c = self.lambda_min + torch.exp(self.log_lambda_interact_col[idx])
            S_r = self._tp_penalty_row[idx]
            S_c = self._tp_penalty_col[idx]
            pen = pen + lam_r * (beta_tp @ S_r @ beta_tp)
            pen = pen + lam_c * (beta_tp @ S_c @ beta_tp)

        # Intensity spline penalty
        if self.intensity_spline:
            lam_int = self.lambda_min + torch.exp(self.log_lambda_intensity)
            beta_int = self.intensity_coefs
            S_int = self._intensity_penalty
            pen = pen + lam_int * (beta_int @ S_int @ beta_int)

        return pen

    def get_partial_effects(self, n_grid=100, posterior_cov=None):
        """
        Evaluate marginal partial effects f_k(φ) on a regular grid.

        Returns dict mapping covariate index to (grid, effect) arrays.
        If posterior_cov is provided (from compute_posterior_covariance),
        also returns SE and 95% credible intervals per covariate.
        """
        device = self.intercept.device
        results = {}

        for k in range(self.n_features):
            grid_np = np.linspace(0.0, 1.0, n_grid)
            B_np, _ = _bspline_basis_matrix(
                grid_np, n_internal=self.n_knots, degree=self.degree)
            B_t = torch.tensor(B_np, dtype=torch.float64, device=device)
            with torch.no_grad():
                effect = (B_t @ self.spline_coefs[k]).cpu().numpy()
                # Add the skip (linear) contribution for this covariate
                skip_w = self.skip.weight.squeeze()
                effect = effect + skip_w[k].item() * grid_np

            entry = {"grid": grid_np, "effect": effect}

            # Bayesian credible bands if posterior covariance is available
            if posterior_cov is not None and k in posterior_cov:
                V_k = posterior_cov[k]  # (n_basis, n_basis) numpy array
                # Var[f_k(φ)] = B_k(φ)^T V_k B_k(φ)
                # B_np is (n_grid, n_basis)
                BV = B_np @ V_k  # (n_grid, n_basis)
                var_f = np.sum(BV * B_np, axis=1)  # (n_grid,)
                var_f = np.maximum(var_f, 0.0)  # numerical safety
                se = np.sqrt(var_f)
                entry["se"] = se
                entry["lower_95"] = effect - 1.96 * se
                entry["upper_95"] = effect + 1.96 * se

            results[k] = entry

        return results

    def get_interaction_effects(self, n_grid=30):
        """
        Evaluate tensor product interaction effects f_{jk}(φ_j, φ_k) on a grid.

        Returns list of dicts with keys:
            j, k        : covariate indices (0-based)
            grid_j      : 1-D array of φ_j values
            grid_k      : 1-D array of φ_k values
            effect_matrix : (n_grid, n_grid) array of f_{jk} values

        Returns empty list if no interactions were fitted.
        """
        if not self.interaction_pairs:
            return []

        device = self.intercept.device
        grid = np.linspace(0.0, 1.0, n_grid)

        # Build marginal B-spline basis on the grid (same for every covariate)
        B_np, _ = _bspline_basis_matrix(
            grid, n_internal=self.n_knots, degree=self.degree)
        B_t = torch.tensor(B_np, dtype=torch.float64, device=device)
        n_b = self.n_basis

        results = []
        with torch.no_grad():
            for idx, (j, k) in enumerate(self.interaction_pairs):
                # Tensor product basis on the 2-D grid
                # B_j[a] and B_k[b] → B_tp[a, b] = kron(B_j[a,:], B_k[b,:])
                # but we want a (n_grid*n_grid, n_b^2) matrix
                B_j_grid = B_t  # (n_grid, n_b)  – basis for covariate j
                B_k_grid = B_t  # (n_grid, n_b)  – basis for covariate k

                # Expand to (n_grid, n_grid, n_b, n_b) → flatten last two
                # B_tp[i, l, a, b] = B_j[i, a] * B_k[l, b]
                n_g = n_grid
                B_tp = (B_j_grid.unsqueeze(1).unsqueeze(3)
                        * B_k_grid.unsqueeze(0).unsqueeze(2))
                B_tp = B_tp.reshape(n_g * n_g, n_b * n_b)

                effect_flat = (B_tp @ self.interaction_coefs[idx]).cpu().numpy()
                effect_matrix = effect_flat.reshape(n_g, n_g)

                results.append({
                    "j": int(j),
                    "k": int(k),
                    "grid_j": grid.tolist(),
                    "grid_k": grid.tolist(),
                    "effect_matrix": effect_matrix.tolist(),
                })

        return results

    def get_effective_loglinear(self):
        """
        Approximate effective log-linear coefficients for compatibility.

        Returns [r_0_eff, z_1_eff, ..., z_K_eff] where z_k ≈ f_k(1) - f_k(0).
        """
        with torch.no_grad():
            r0 = self.intercept.item() + self.skip.bias.item()
            ell = [r0]
            for k in range(self.n_features):
                # Evaluate spline at 0 and 1
                x0 = np.array([0.0])
                x1 = np.array([1.0])
                B0, _ = _bspline_basis_matrix(
                    x0, n_internal=self.n_knots, degree=self.degree)
                B1, _ = _bspline_basis_matrix(
                    x1, n_internal=self.n_knots, degree=self.degree)
                B0_t = torch.tensor(B0, dtype=torch.float64,
                                    device=self.intercept.device)
                B1_t = torch.tensor(B1, dtype=torch.float64,
                                    device=self.intercept.device)
                f0 = (B0_t @ self.spline_coefs[k]).item()
                f1 = (B1_t @ self.spline_coefs[k]).item()
                # Add skip linear component
                skip_w = self.skip.weight.squeeze()[k].item()
                z_k = (f1 + skip_w) - f0
                ell.append(z_k)
            return ell

    # ==================================================================
    # Bayesian UQ: Laplace approximation on spline coefficients
    # ==================================================================

    def _collect_spline_params(self):
        """Return (params_list, slices) for marginal spline coefs only."""
        params = []
        slices = {}
        offset = 0
        for k in range(self.n_features):
            n_b = self.spline_coefs[k].numel()
            params.append(self.spline_coefs[k])
            slices[k] = slice(offset, offset + n_b)
            offset += n_b
        return params, slices, offset

    def compute_nll_hessian_block(self, nll_fn, block_only=True):
        """
        Compute the Hessian of the negative log-likelihood w.r.t. marginal
        spline coefficients using pure finite differences (no autograd).

        This uses central differences on the NLL *value* to build the Hessian:
            H[i,j] ≈ (f(θ+εe_i+εe_j) - f(θ+εe_i-εe_j) - f(θ-εe_i+εe_j) + f(θ-εe_i-εe_j)) / (4ε²)

        For diagonal and near-diagonal elements, uses the more efficient:
            H[i,i] ≈ (f(θ+εe_i) - 2f(θ) + f(θ-εe_i)) / ε²
            H[i,j] ≈ (f(θ+ε(e_i+e_j)) - f(θ+εe_i) - f(θ+εe_j) + f(θ)) / ε²

        Parameters
        ----------
        nll_fn : callable
            A function that takes no arguments, recomputes the forward pass
            (including circuit solve) and returns the negative log-likelihood
            as a Python float. Must not require gradients.
        block_only : bool
            If True, compute per-covariate blocks only (fast: K × n_basis
            function evaluations). If False, compute the full Hessian over
            all marginal spline coefs.

        Returns
        -------
        dict with:
            'blocks' : list of (n_basis, n_basis) numpy arrays if block_only
            'full'   : (n_total, n_total) numpy array if not block_only
            'slices' : dict mapping covariate index to slice into flat vector
        """
        params, slices, n_total = self._collect_spline_params()
        eps = 1e-4

        # Baseline NLL
        f0 = nll_fn()

        if block_only:
            blocks = []
            for k in range(self.n_features):
                beta_k = self.spline_coefs[k]
                n_b = beta_k.numel()
                H_k = np.zeros((n_b, n_b), dtype=np.float64)

                # First compute f(θ+εe_i) for all i (reused in off-diagonal)
                f_plus = np.zeros(n_b, dtype=np.float64)
                f_minus = np.zeros(n_b, dtype=np.float64)
                for i in range(n_b):
                    beta_k.data[i] += eps
                    f_plus[i] = nll_fn()
                    beta_k.data[i] -= 2.0 * eps
                    f_minus[i] = nll_fn()
                    beta_k.data[i] += eps  # restore

                # Diagonal: H[i,i] = (f+ - 2*f0 + f-) / eps^2
                for i in range(n_b):
                    H_k[i, i] = (f_plus[i] - 2.0 * f0 + f_minus[i]) / (eps * eps)

                # Off-diagonal: H[i,j] = (f(θ+ε(ei+ej)) - f+_i - f+_j + f0) / eps^2
                for i in range(n_b):
                    for j in range(i + 1, n_b):
                        beta_k.data[i] += eps
                        beta_k.data[j] += eps
                        f_ij = nll_fn()
                        beta_k.data[i] -= eps
                        beta_k.data[j] -= eps  # restore

                        H_k[i, j] = (f_ij - f_plus[i] - f_plus[j] + f0) / (eps * eps)
                        H_k[j, i] = H_k[i, j]

                blocks.append(H_k)

            return {"blocks": blocks, "slices": slices}

        else:
            # Full Hessian over all marginal spline coefficients
            H = np.zeros((n_total, n_total), dtype=np.float64)

            # Compute all single-perturbation values
            f_plus_all = np.zeros(n_total, dtype=np.float64)
            f_minus_all = np.zeros(n_total, dtype=np.float64)

            for k in range(self.n_features):
                beta_k = self.spline_coefs[k]
                s = slices[k]
                n_b = beta_k.numel()
                for i in range(n_b):
                    beta_k.data[i] += eps
                    f_plus_all[s.start + i] = nll_fn()
                    beta_k.data[i] -= 2.0 * eps
                    f_minus_all[s.start + i] = nll_fn()
                    beta_k.data[i] += eps  # restore

            # Diagonal
            for idx in range(n_total):
                H[idx, idx] = (f_plus_all[idx] - 2.0 * f0 + f_minus_all[idx]) / (eps * eps)

            # Off-diagonal (only within same covariate for block structure,
            # and cross-covariate if full)
            for k1 in range(self.n_features):
                s1 = slices[k1]
                beta_k1 = self.spline_coefs[k1]
                for k2 in range(k1, self.n_features):
                    s2 = slices[k2]
                    beta_k2 = self.spline_coefs[k2]
                    for i in range(s1.stop - s1.start):
                        j_start = (i + 1) if k1 == k2 else 0
                        for j in range(j_start, s2.stop - s2.start):
                            gi = s1.start + i
                            gj = s2.start + j
                            beta_k1.data[i] += eps
                            beta_k2.data[j] += eps
                            f_ij = nll_fn()
                            beta_k1.data[i] -= eps
                            beta_k2.data[j] -= eps

                            H[gi, gj] = (f_ij - f_plus_all[gi] - f_plus_all[gj] + f0) / (eps * eps)
                            H[gj, gi] = H[gi, gj]

            return {"full": H, "slices": slices}

    def compute_posterior_covariance(self, hessian_result, penalty_scale=1.0):
        """
        Compute posterior covariance V = (H_nll + penalty_scale × Σ_k λ_k S_k)^{-1}

        Uses the Bayesian P-spline interpretation where the smoothing penalty
        acts as an improper Gaussian prior.

        Parameters
        ----------
        hessian_result : dict
            Output from compute_nll_hessian_block.
        penalty_scale : float
            The penalty_scale used during optimization.

        Returns
        -------
        dict mapping covariate index k to (n_basis, n_basis) posterior covariance.
        Also includes 'edf' (effective degrees of freedom per covariate) and
        'significance' (approximate p-values).
        """
        posterior_cov = {}
        edf_dict = {}
        significance = {}

        if "blocks" in hessian_result:
            # Block-diagonal approximation
            for k in range(self.n_features):
                H_k = hessian_result["blocks"][k]
                lam_k = (self.lambda_min
                         + torch.exp(self.log_lambda_smooth[k]).item())
                S_k = self._penalty_matrices[k].cpu().numpy()
                P_k = penalty_scale * lam_k * S_k

                # Posterior precision = H_nll + penalty
                precision_k = H_k + P_k

                # Regularize for numerical stability
                eigvals = np.linalg.eigvalsh(precision_k)
                min_eig = eigvals.min()
                if min_eig < 1e-8:
                    precision_k += (1e-8 - min_eig) * np.eye(precision_k.shape[0])

                V_k = np.linalg.inv(precision_k)
                posterior_cov[k] = V_k

                # EDF = trace(V_k @ H_k) — how many params each smooth uses
                edf_k = np.trace(V_k @ H_k)
                edf_dict[k] = float(edf_k)

                # Significance test: β_k^T P_r β_k / rank(S_k) ~ chi2
                # where P_r = pseudoinverse of V_k restricted to range(S_k)
                beta_k = self.spline_coefs[k].detach().cpu().numpy()
                # Rank-based test (Wood 2013): use edf as approximate rank
                edf_r = max(edf_k, 1.0)
                test_stat = float(beta_k @ np.linalg.pinv(V_k) @ beta_k)
                from scipy import stats as sp_stats
                p_val = float(1.0 - sp_stats.chi2.cdf(test_stat, df=edf_r))
                significance[k] = {
                    "chi_sq": test_stat,
                    "edf": edf_k,
                    "p_value": p_val,
                }

        elif "full" in hessian_result:
            H = hessian_result["full"]
            slices = hessian_result["slices"]
            n_total = H.shape[0]

            # Build full penalty matrix
            P = np.zeros((n_total, n_total), dtype=np.float64)
            for k in range(self.n_features):
                s = slices[k]
                lam_k = (self.lambda_min
                         + torch.exp(self.log_lambda_smooth[k]).item())
                S_k = self._penalty_matrices[k].cpu().numpy()
                P[s, s] += penalty_scale * lam_k * S_k

            precision = H + P
            eigvals = np.linalg.eigvalsh(precision)
            min_eig = eigvals.min()
            if min_eig < 1e-8:
                precision += (1e-8 - min_eig) * np.eye(n_total)

            V_full = np.linalg.inv(precision)

            for k in range(self.n_features):
                s = slices[k]
                V_k = V_full[s, s]
                H_k = H[s, s]
                posterior_cov[k] = V_k
                edf_k = np.trace(V_k @ H_k)
                edf_dict[k] = float(edf_k)

                beta_k = self.spline_coefs[k].detach().cpu().numpy()
                edf_r = max(edf_k, 1.0)
                test_stat = float(beta_k @ np.linalg.pinv(V_k) @ beta_k)
                from scipy import stats as sp_stats
                p_val = float(1.0 - sp_stats.chi2.cdf(test_stat, df=edf_r))
                significance[k] = {
                    "chi_sq": test_stat,
                    "edf": edf_k,
                    "p_value": p_val,
                }

        return {
            "covariance": posterior_cov,
            "edf": edf_dict,
            "significance": significance,
        }


# ===========================================================================
# PPP log-likelihood
# ===========================================================================

def _ppp_loglik(log_lam, obs_counts, cell_area):
    """
    Poisson point process log-likelihood on raster cells.

    log L = Σ_k n_k log λ_k  −  A Σ_k λ_k

    where n_k = # GPS fixes in cell k, A = cell area (m²).
    """
    term1 = (obs_counts * log_lam).sum()
    term2 = cell_area * torch.exp(log_lam.clamp(max=20.0)).sum()
    return term1 - term2

