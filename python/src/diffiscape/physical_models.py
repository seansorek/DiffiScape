"""Resistance models for physical-current fitting, including penalized B-splines."""
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from .resistance import _softplus_clamp, ResistanceMLP


class LogLinearResistance(nn.Module):
    @nn.compact
    def __call__(self, x, return_raw=False):
        raw = nn.Dense(1, name="skip", param_dtype=jnp.float64,
                       kernel_init=nn.initializers.normal(0.1),
                       bias_init=nn.initializers.constant(3.0))(x).squeeze(-1)
        clamped = _softplus_clamp(raw)
        return (clamped, raw) if return_raw else clamped


class IntensityMLP(nn.Module):
    hidden: int

    @nn.compact
    def __call__(self, x):
        h = nn.silu(nn.Dense(self.hidden, param_dtype=jnp.float64,
                            kernel_init=nn.initializers.ones)(x))
        return nn.Dense(1, param_dtype=jnp.float64, kernel_init=nn.initializers.ones)(h).squeeze(-1)


class BSplineResistance(nn.Module):
    n_covariates: int
    n_basis: int
    n_pairs: int
    lambda_init_marginal: float = 0.0
    lambda_init_interaction: float = 2.0

    @nn.compact
    def __call__(self, x, marginal, tensor, return_raw=False):
        raw = nn.Dense(1, name="skip", param_dtype=jnp.float64,
                       kernel_init=nn.initializers.normal(0.1))(x).squeeze(-1)
        raw = raw + self.param("intercept", nn.initializers.constant(3.0), (), jnp.float64)
        coefs = self.param("spline_coefs", nn.initializers.normal(0.01), (self.n_covariates, self.n_basis), jnp.float64)
        raw = raw + jnp.einsum("nkb,kb->n", marginal, coefs)
        interactions = self.param("interaction_coefs", nn.initializers.normal(0.001), (self.n_pairs, self.n_basis ** 2), jnp.float64)
        raw = raw + jnp.einsum("npb,pb->n", tensor, interactions)
        self.param("log_lambda_smooth", nn.initializers.constant(self.lambda_init_marginal), (self.n_covariates,), jnp.float64)
        self.param("log_lambda_interact", nn.initializers.constant(self.lambda_init_interaction), (self.n_pairs, 2), jnp.float64)
        clamped = _softplus_clamp(raw)
        return (clamped, raw) if return_raw else clamped


def smoothing_penalty(parameters, lambda_min=0.0):
    p = parameters["params"]
    marginal = jnp.sum(jnp.diff(p["spline_coefs"], n=2, axis=-1) ** 2, axis=-1)
    value = jnp.sum((lambda_min + jnp.exp(p["log_lambda_smooth"])) * marginal)
    n_basis = p["spline_coefs"].shape[-1]
    tensor = p["interaction_coefs"].reshape(-1, n_basis, n_basis)
    row = jnp.sum(jnp.diff(tensor, n=2, axis=1) ** 2, axis=(1, 2))
    col = jnp.sum(jnp.diff(tensor, n=2, axis=2) ** 2, axis=(1, 2))
    weights = lambda_min + jnp.exp(p["log_lambda_interact"])
    return value + jnp.sum(weights[:, 0] * row + weights[:, 1] * col)


class SpatialConvResistance(nn.Module):
    channels: int = 16
    n_layers: int = 3
    kernel_size: int = 3
    hidden: int = 16
    n_mlp_layers: int = 1
    use_dilated: bool = True
    dropout: float = 0.0

    @nn.compact
    def __call__(self, grid, mask, basis, return_raw=False, deterministic=True):
        h = nn.Conv(self.channels, (self.kernel_size, self.kernel_size), name="input_proj",
                    param_dtype=jnp.float64, kernel_init=nn.initializers.normal(0.02))(grid[None])
        h = nn.silu(nn.GroupNorm(num_groups=1, epsilon=1e-5, param_dtype=jnp.float64, name="input_norm")(h))
        for layer in range(self.n_layers):
            dilation = 2 ** layer if self.use_dilated else 1
            residual = nn.Conv(self.channels, (self.kernel_size, self.kernel_size),
                kernel_dilation=(dilation, dilation), param_dtype=jnp.float64,
                kernel_init=nn.initializers.normal(0.02), name=f"conv_{layer}")(h)
            h = h + nn.silu(nn.GroupNorm(num_groups=1, epsilon=1e-5, param_dtype=jnp.float64,
                                         name=f"norm_{layer}")(residual))
        h = h.reshape(-1, self.channels)[mask]
        for layer in range(self.n_mlp_layers):
            h = nn.silu(nn.Dense(self.hidden, param_dtype=jnp.float64,
                                kernel_init=nn.initializers.normal(0.05), name=f"head_{layer}")(h))
            if self.dropout:
                h = nn.Dropout(self.dropout)(h, deterministic=deterministic)
        correction = nn.Dense(1, param_dtype=jnp.float64, kernel_init=nn.initializers.normal(0.01), name="head_out")(h).squeeze(-1)
        skip = nn.Dense(1, name="skip", param_dtype=jnp.float64,
                        kernel_init=nn.initializers.normal(0.1), bias_init=nn.initializers.constant(3.0))(basis).squeeze(-1)
        raw = skip + correction
        clamped = _softplus_clamp(raw)
        return (clamped, raw) if return_raw else clamped


def create_physical_model(model_type, n_covariates, cfg):
    if model_type == "loglinear":
        return LogLinearResistance()
    if model_type == "mlp":
        return ResistanceMLP(features=cfg.get("hidden_dim", 32), n_hidden=cfg.get("n_hidden_layers", 2))
    if model_type == "spline_gam":
        n_basis = cfg.get("n_knots", 10) + cfg.get("spline_degree", 3) + 1
        n_pairs = n_covariates * (n_covariates - 1) // 2 if cfg.get("include_interactions", True) else 0
        return BSplineResistance(n_covariates, n_basis, n_pairs, cfg.get("lambda_init_marginal", 0.0), cfg.get("lambda_init_interaction", 2.0))
    if model_type == "conv":
        return SpatialConvResistance(channels=cfg.get("conv_channels", 16), n_layers=cfg.get("n_conv_layers", 3),
            kernel_size=cfg.get("conv_kernel_size", 3), hidden=cfg.get("hidden_dim", 16),
            n_mlp_layers=cfg.get("n_hidden_layers", 1), use_dilated=cfg.get("use_dilated", True), dropout=cfg.get("dropout", 0.0))
    raise ValueError("model_type must be loglinear, mlp, conv, or spline_gam")


def partial_effects(fitted, posterior=None, n_grid=100, interaction_grid=30):
    """Raw log-resistance effects, with draw-based bands when a posterior exists."""
    from .splines import design_matrix
    cfg = fitted["configuration"]
    if cfg["model_type"] != "spline_gam":
        return dict(partial_effects=None, interaction_effects=None)
    settings = cfg["model_config"]
    grid = np.linspace(0, 1, n_grid)
    B = design_matrix(grid[:, None], settings.get("n_knots", 10), settings.get("spline_degree", 3), False)[0][:, 0]
    p = fitted["model_state"]["resistance"]["params"]
    marginal = {}
    for index in range(cfg["n_covariates"]):
        effect = B @ np.asarray(p["spline_coefs"])[index] + np.asarray(p["skip"]["kernel"])[index, 0] * grid
        entry = dict(grid=grid, effect=effect)
        if posterior is not None:
            draws = posterior["parameter_draws"]["resistance"]["params"]
            values = np.asarray(draws["spline_coefs"])[:, index] @ B.T + np.asarray(draws["skip"]["kernel"])[:, index, 0, None] * grid
            entry.update(se=np.std(values, axis=0, ddof=1), lower_95=np.quantile(values, 0.025, axis=0), upper_95=np.quantile(values, 0.975, axis=0))
        marginal[str(index)] = entry
    interactions = []
    if settings.get("include_interactions", True):
        interaction_x = np.linspace(0, 1, interaction_grid)
        tensor_B = design_matrix(interaction_x[:, None], settings.get("n_knots", 10), settings.get("spline_degree", 3), False)[0][:, 0]
        pairs = [(j, k) for j in range(cfg["n_covariates"]) for k in range(j + 1, cfg["n_covariates"])]
        for index, (j, k) in enumerate(pairs):
            coefficients = np.asarray(p["interaction_coefs"])[index].reshape(B.shape[1], B.shape[1])
            interactions.append(dict(j=j, k=k, grid_j=interaction_x, grid_k=interaction_x, effect_matrix=tensor_B @ coefficients @ tensor_B.T))
    return dict(partial_effects=marginal, interaction_effects=interactions)
