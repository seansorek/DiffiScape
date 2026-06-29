"""Flax neural-network resistance-surface models.

Four ``flax.linen.Module`` subclasses that map spatial covariates to
log-resistance values, porting the PyTorch nets in ``diff_cs/05_torch_pipeline.py``
to JAX/Flax:

* :class:`ResistanceMLP` — MLP with linear skip connection
* :class:`ResistanceConv` — Conv2d residual blocks + MLP head
* :class:`ResistanceSpline` — B-spline GAM-style additive model
* :class:`ResistanceIRL` — Reward MLP + soft value iteration

All modules output clamped log-resistance in ``[log(R_MIN), log(R_MAX)]``
via a double-softplus smooth clamp.
"""

try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
except ImportError:
    raise ImportError(
        "JAX and Flax are required for resistance models. "
        "Install with: pip install jax flax"
    )

from .core import DEFAULT_R_MIN, DEFAULT_R_MAX


# ---------------------------------------------------------------------------
# Differentiable clamp helper
# ---------------------------------------------------------------------------

def _scaled_softplus(x, beta=5.0):
    """Softplus with sharpness parameter: (1/beta) * log(1 + exp(beta * x)).

    Matches PyTorch's ``F.softplus(x, beta=beta)``.
    """
    return jax.nn.softplus(beta * x) / beta


def _softplus_clamp(x, r_min=DEFAULT_R_MIN, r_max=DEFAULT_R_MAX, beta=5.0):
    """Double-softplus differentiable clamp to [log(r_min), log(r_max)].

    Operates in log-space: *x* is a raw log-resistance value and the
    returned value is smoothly constrained to ``[log(r_min), log(r_max)]``.
    Matches the log-space clamp used in the PyTorch ``ResistanceNet``.

    Parameters
    ----------
    x : jnp.ndarray
        Raw log-resistance values.
    r_min, r_max : float
        Resistance bounds (in natural scale).
    beta : float
        Softplus sharpness (higher = sharper clamp).

    Returns
    -------
    jnp.ndarray
        Clamped log-resistance values.
    """
    log_min = jnp.log(jnp.float32(r_min))
    log_max = jnp.log(jnp.float32(r_max))
    # Lower clamp: push x above log_min
    clamped = log_min + _scaled_softplus(x - log_min, beta)
    # Upper clamp: push x below log_max
    clamped = log_max - _scaled_softplus(log_max - clamped, beta)
    return clamped


# ---------------------------------------------------------------------------
# 1. ResistanceMLP
# ---------------------------------------------------------------------------

class ResistanceMLP(nn.Module):
    """MLP with linear skip connection for resistance mapping.

    ``log R(x) = linear(x) + MLP(x)``

    The linear branch mirrors the log-linear model.  The MLP branch starts
    near zero (default Flax init) so the network begins at a near-linear
    baseline and learns nonlinear corrections.

    Parameters
    ----------
    features : int
        Hidden-layer width.
    n_hidden : int
        Number of hidden layers in the MLP branch.
    r_min, r_max : float
        Resistance bounds for the output clamp.
    clamp_beta : float
        Softplus sharpness for the clamp.
    """
    features: int = 32
    n_hidden: int = 2
    r_min: float = DEFAULT_R_MIN
    r_max: float = DEFAULT_R_MAX
    clamp_beta: float = 5.0

    @nn.compact
    def __call__(self, x):
        """Forward pass.

        Parameters
        ----------
        x : jnp.ndarray, shape ``(n_cells, n_covariates)``
            Basis covariates.

        Returns
        -------
        jnp.ndarray, shape ``(n_cells,)``
            Clamped log-resistance values.
        """
        # Linear skip connection
        skip = nn.Dense(1)(x).squeeze(-1)

        # Nonlinear MLP branch
        h = x
        for _ in range(self.n_hidden):
            h = nn.Dense(self.features)(h)
            h = nn.silu(h)
        nonlinear = nn.Dense(1)(h).squeeze(-1)

        log_r = skip + nonlinear
        return _softplus_clamp(log_r, self.r_min, self.r_max, self.clamp_beta)


# ---------------------------------------------------------------------------
# 2. ResistanceConv
# ---------------------------------------------------------------------------

class ResistanceConv(nn.Module):
    """Convolutional residual blocks + MLP head for resistance mapping.

    Input is a 2-D raster ``(H, W, C)`` (channels-last, JAX convention).
    The conv encoder uses residual connections where shapes match, then
    flattens to an MLP head that produces per-pixel log-resistance.

    Parameters
    ----------
    channels : int
        Number of conv feature channels.
    n_layers : int
        Number of residual conv blocks.
    kernel_size : int
        Spatial kernel size for conv layers.
    hidden_dim : int
        Width of the MLP head hidden layer.
    r_min, r_max : float
        Resistance bounds for the output clamp.
    clamp_beta : float
        Softplus sharpness for the clamp.
    """
    channels: int = 16
    n_layers: int = 3
    kernel_size: int = 3
    hidden_dim: int = 32
    r_min: float = DEFAULT_R_MIN
    r_max: float = DEFAULT_R_MAX
    clamp_beta: float = 5.0

    @nn.compact
    def __call__(self, x):
        """Forward pass.

        Parameters
        ----------
        x : jnp.ndarray, shape ``(H, W, C)``
            Raster input (channels-last).

        Returns
        -------
        jnp.ndarray, shape ``(H * W,)``
            Clamped log-resistance values.
        """
        H, W = x.shape[0], x.shape[1]
        h = x

        # Residual conv blocks
        for _ in range(self.n_layers):
            residual = h
            h = nn.Conv(self.channels,
                        kernel_size=(self.kernel_size, self.kernel_size),
                        padding='SAME')(h)
            h = nn.silu(h)
            if residual.shape == h.shape:
                h = h + residual

        # Flatten spatial dims → MLP head
        h = h.reshape(H * W, -1)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.silu(h)
        log_r = nn.Dense(1)(h).squeeze(-1)

        return _softplus_clamp(log_r, self.r_min, self.r_max, self.clamp_beta)


# ---------------------------------------------------------------------------
# 3. ResistanceSpline
# ---------------------------------------------------------------------------

class ResistanceSpline(nn.Module):
    """B-spline GAM-style additive model for resistance mapping.

    Per-covariate Gaussian-RBF basis with learned coefficients, plus
    optional pairwise product interactions.  The basis approximates a
    smooth spline: each basis function is a Gaussian centred at a knot.

    Parameters
    ----------
    n_knots : int
        Number of interior knots per covariate.
    degree : int
        Unused (kept for API compatibility); the RBF width is ``1/n_knots``.
    n_covariates : int
        Number of input covariates.
    include_interactions : bool
        Whether to include pairwise product interaction terms.
    r_min, r_max : float
        Resistance bounds for the output clamp.
    clamp_beta : float
        Softplus sharpness for the clamp.
    """
    n_knots: int = 10
    degree: int = 3
    n_covariates: int = 1
    include_interactions: bool = True
    x_min: float = 0.0
    x_max: float = 1.0
    r_min: float = DEFAULT_R_MIN
    r_max: float = DEFAULT_R_MAX
    clamp_beta: float = 5.0

    @nn.compact
    def __call__(self, x):
        """Forward pass.

        Parameters
        ----------
        x : jnp.ndarray, shape ``(n_cells, n_covariates)``
            Covariate values (should be normalized to ``[x_min, x_max]``).

        Returns
        -------
        jnp.ndarray, shape ``(n_cells,)``
            Clamped log-resistance values.
        """
        components = []
        for k in range(self.n_covariates):
            basis = self._rbf_basis(x[:, k], self.n_knots, self.x_min, self.x_max)
            coefs = self.param(
                f'coef_{k}',
                nn.initializers.zeros,
                (basis.shape[-1],),
            )
            components.append(basis @ coefs)

        log_r = sum(components)

        # Pairwise product interactions
        if self.include_interactions and self.n_covariates >= 2:
            for i in range(self.n_covariates):
                for j in range(i + 1, self.n_covariates):
                    interaction = self.param(
                        f'interact_{i}_{j}',
                        nn.initializers.zeros,
                        (1,),
                    )
                    log_r = log_r + interaction[0] * x[:, i] * x[:, j]

        intercept = self.param(
            'intercept',
            nn.initializers.constant(3.0),
            (),
        )
        log_r = intercept + log_r
        return _softplus_clamp(log_r, self.r_min, self.r_max, self.clamp_beta)

    @staticmethod
    def _rbf_basis(x, n_knots, x_min=0.0, x_max=1.0):
        """Gaussian RBF basis centred at evenly-spaced interior knots.

        Parameters
        ----------
        x : jnp.ndarray, shape ``(n,)``
            Input values for one covariate.
        n_knots : int
            Number of interior knots.
        x_min : float
            Lower bound for normalization (compile-time constant, JIT-safe).
        x_max : float
            Upper bound for normalization (compile-time constant, JIT-safe).

        Returns
        -------
        jnp.ndarray, shape ``(n, n_knots)``
            Basis matrix.
        """
        knots = jnp.linspace(0.0, 1.0, n_knots + 2)[1:-1]
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        sigma = 1.0 / n_knots
        basis = jnp.stack(
            [jnp.exp(-0.5 * ((x_norm - k) / sigma) ** 2) for k in knots],
            axis=-1,
        )
        return basis


# ---------------------------------------------------------------------------
# 4. ResistanceIRL
# ---------------------------------------------------------------------------

class ResistanceIRL(nn.Module):
    """Inverse-reinforcement-learning (value-shaped) resistance model.

    A learned reward field is turned into a resistance surface via
    entropy-regularised soft value iteration:

    * ``r(x) = MLP(x)``  — reward (negative cost)
    * ``V_{t+1}(s) = (1/beta) logsumexp_{a} beta [r(s) + gamma * V_t(s')]``
    * ``log R(x) = offset - scale * V(x)``

    When no adjacency matrix is provided, falls back to using the raw
    reward as log-resistance.

    Parameters
    ----------
    hidden_dim : int
        Reward MLP hidden-layer width.
    n_hidden : int
        Number of hidden layers in the reward MLP.
    beta : float
        Soft value-iteration temperature.
    gamma_d : float
        Discount factor (must be < 1 for convergence).
    n_value_iter : int
        Number of value-iteration sweeps.
    r_min, r_max : float
        Resistance bounds for the output clamp.
    clamp_beta : float
        Softplus sharpness for the output clamp.
    value_scale_init : float
        Initial value for the ``scale`` parameter.
    """
    hidden_dim: int = 32
    n_hidden: int = 2
    beta: float = 1.0
    gamma_d: float = 0.9
    n_value_iter: int = 60
    r_min: float = DEFAULT_R_MIN
    r_max: float = DEFAULT_R_MAX
    clamp_beta: float = 5.0
    value_scale_init: float = 1.0

    @nn.compact
    def __call__(self, x, adjacency=None):
        """Forward pass.

        Parameters
        ----------
        x : jnp.ndarray, shape ``(n_cells, n_features)``
            Covariate values.
        adjacency : jnp.ndarray or None, shape ``(n_cells, n_cells)``
            Adjacency/transition matrix.  If ``None``, raw reward is used
            as log-resistance (no value iteration).

        Returns
        -------
        jnp.ndarray, shape ``(n_cells,)``
            Clamped log-resistance values.
        """
        # Reward network
        h = x
        for _ in range(self.n_hidden):
            h = nn.Dense(self.hidden_dim)(h)
            h = nn.silu(h)
        reward = nn.Dense(1)(h).squeeze(-1)

        if adjacency is not None:
            # Soft value iteration with adjacency matrix via fori_loop
            # (avoids XLA graph explosion from unrolled Python loop)
            beta = self.beta
            gamma_d = self.gamma_d

            def vi_step(_, value):
                neighbor_values = adjacency @ value
                q_stay = reward + gamma_d * value
                q_move = reward + gamma_d * neighbor_values
                q_stack = jnp.stack([q_stay, q_move], axis=-1)
                return jax.nn.logsumexp(beta * q_stack, axis=-1) / beta

            value = jax.lax.fori_loop(
                0, self.n_value_iter, vi_step, jnp.zeros_like(reward)
            )

            scale = self.param(
                'value_scale',
                nn.initializers.constant(self.value_scale_init),
                (),
            )
            offset = self.param(
                'value_offset',
                nn.initializers.constant(3.0),
                (),
            )
            log_r = offset - scale * value
        else:
            # No adjacency: raw reward as log-resistance
            log_r = reward

        return _softplus_clamp(log_r, self.r_min, self.r_max, self.clamp_beta)
