"""Tests for diffiscape_jax.resistance module."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")

from diffiscape_jax.resistance import (
    ResistanceMLP,
    ResistanceConv,
    ResistanceSpline,
    ResistanceIRL,
    _softplus_clamp,
)


# ---------------------------------------------------------------------------
# _softplus_clamp
# ---------------------------------------------------------------------------

class TestSoftplusClamp:
    """Tests for the differentiable clamp helper."""

    def test_clamp_within_bounds(self):
        """Values already within bounds should pass through approximately."""
        x = jnp.array([1.0, 3.0, 5.0])  # log-R values
        clamped = _softplus_clamp(x, r_min=1.0, r_max=5000.0)
        # The clamp is smooth, so interior values should be very close
        np.testing.assert_allclose(clamped, x, atol=0.1)

    def test_clamp_lower_bound(self):
        """Very negative values should be pushed up toward log(r_min)."""
        x = jnp.array([-100.0, -50.0])
        clamped = _softplus_clamp(x, r_min=1.0, r_max=5000.0)
        log_min = np.log(1.0)
        assert jnp.all(clamped >= log_min - 0.5)

    def test_clamp_upper_bound(self):
        """Very large values should be pushed down toward log(r_max)."""
        x = jnp.array([100.0, 200.0])
        clamped = _softplus_clamp(x, r_min=1.0, r_max=5000.0)
        log_max = np.log(5000.0)
        assert jnp.all(clamped <= log_max + 0.5)

    def test_clamp_gradients_finite(self):
        """Gradients through the clamp should always be finite."""
        def f(x):
            return jnp.sum(_softplus_clamp(x, r_min=1.0, r_max=5000.0))

        x = jnp.array([-100.0, 0.0, 5.0, 100.0])
        grad = jax.grad(f)(x)
        assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# ResistanceMLP
# ---------------------------------------------------------------------------

class TestResistanceMLP:
    """Tests for the MLP resistance model."""

    def test_output_shape(self):
        """Output should be (n_cells,) for input (n_cells, n_covariates)."""
        model = ResistanceMLP(features=32, n_hidden=2, r_min=1.0, r_max=5000.0)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (100, 3))
        params = model.init(rng, x)
        out = model.apply(params, x)
        assert out.shape == (100,)

    def test_output_clamped(self):
        """Exponentiated output should be within [~r_min, ~r_max]."""
        model = ResistanceMLP(features=16, n_hidden=1, r_min=1.0, r_max=5000.0)
        rng = jax.random.PRNGKey(1)
        x = jax.random.normal(rng, (50, 2)) * 10  # large inputs
        params = model.init(rng, x)
        out = model.apply(params, x)
        resistance = jnp.exp(out)
        # Softplus clamp is smooth so allow small margin
        assert jnp.all(resistance >= 0.9)
        assert jnp.all(resistance <= 5500.0)

    def test_gradients_finite(self):
        """All parameter gradients should be finite."""
        model = ResistanceMLP(features=16, n_hidden=1)
        rng = jax.random.PRNGKey(2)
        x = jax.random.normal(rng, (20, 2))
        params = model.init(rng, x)

        def loss_fn(p):
            return jnp.sum(model.apply(p, x))

        grad = jax.grad(loss_fn)(params)
        leaves = jax.tree.leaves(grad)
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)

    def test_single_covariate(self):
        """Should work with a single covariate."""
        model = ResistanceMLP(features=8, n_hidden=1)
        rng = jax.random.PRNGKey(3)
        x = jax.random.normal(rng, (30, 1))
        params = model.init(rng, x)
        out = model.apply(params, x)
        assert out.shape == (30,)

    def test_different_hidden_depths(self):
        """Should work with varying n_hidden."""
        rng = jax.random.PRNGKey(4)
        x = jax.random.normal(rng, (10, 3))
        for n_hidden in [0, 1, 3]:
            model = ResistanceMLP(features=8, n_hidden=n_hidden)
            params = model.init(rng, x)
            out = model.apply(params, x)
            assert out.shape == (10,)


# ---------------------------------------------------------------------------
# ResistanceConv
# ---------------------------------------------------------------------------

class TestResistanceConv:
    """Tests for the convolutional resistance model."""

    def test_output_shape(self):
        """Output should be (H*W,) for input (H, W, C)."""
        model = ResistanceConv(channels=8, n_layers=2, kernel_size=3,
                               hidden_dim=16)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (10, 10, 3))
        params = model.init(rng, x)
        out = model.apply(params, x)
        assert out.shape == (100,)  # 10 * 10

    def test_non_square_raster(self):
        """Should handle non-square rasters."""
        model = ResistanceConv(channels=4, n_layers=1, kernel_size=3,
                               hidden_dim=8)
        rng = jax.random.PRNGKey(1)
        x = jax.random.normal(rng, (8, 12, 2))
        params = model.init(rng, x)
        out = model.apply(params, x)
        assert out.shape == (96,)  # 8 * 12

    def test_output_clamped(self):
        """Exponentiated output should respect resistance bounds."""
        model = ResistanceConv(channels=4, n_layers=1, hidden_dim=8,
                               r_min=1.0, r_max=5000.0)
        rng = jax.random.PRNGKey(2)
        x = jax.random.normal(rng, (5, 5, 2)) * 10
        params = model.init(rng, x)
        out = model.apply(params, x)
        resistance = jnp.exp(out)
        assert jnp.all(resistance >= 0.9)
        assert jnp.all(resistance <= 5500.0)

    def test_gradients_finite(self):
        """All parameter gradients should be finite."""
        model = ResistanceConv(channels=4, n_layers=1, hidden_dim=8)
        rng = jax.random.PRNGKey(3)
        x = jax.random.normal(rng, (5, 5, 2))
        params = model.init(rng, x)

        def loss_fn(p):
            return jnp.sum(model.apply(p, x))

        grad = jax.grad(loss_fn)(params)
        leaves = jax.tree.leaves(grad)
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)


# ---------------------------------------------------------------------------
# ResistanceSpline
# ---------------------------------------------------------------------------

class TestResistanceSpline:
    """Tests for the spline resistance model."""

    def test_output_shape(self):
        """Output should be (n_cells,) for input (n_cells, n_covariates)."""
        model = ResistanceSpline(n_knots=5, degree=3, n_covariates=2)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (50, 2))
        params = model.init(rng, x)
        out = model.apply(params, x)
        assert out.shape == (50,)

    def test_single_covariate(self):
        """Should work with a single covariate."""
        model = ResistanceSpline(n_knots=5, n_covariates=1)
        rng = jax.random.PRNGKey(1)
        x = jax.random.normal(rng, (30, 1))
        params = model.init(rng, x)
        out = model.apply(params, x)
        assert out.shape == (30,)

    def test_output_clamped(self):
        """Exponentiated output should respect resistance bounds."""
        model = ResistanceSpline(n_knots=5, n_covariates=2,
                                 r_min=1.0, r_max=5000.0)
        rng = jax.random.PRNGKey(2)
        x = jax.random.normal(rng, (40, 2))
        params = model.init(rng, x)
        out = model.apply(params, x)
        resistance = jnp.exp(out)
        assert jnp.all(resistance >= 0.9)
        assert jnp.all(resistance <= 5500.0)

    def test_gradients_finite(self):
        """All parameter gradients should be finite."""
        model = ResistanceSpline(n_knots=5, n_covariates=2)
        rng = jax.random.PRNGKey(3)
        x = jax.random.normal(rng, (20, 2))
        params = model.init(rng, x)

        def loss_fn(p):
            return jnp.sum(model.apply(p, x))

        grad = jax.grad(loss_fn)(params)
        leaves = jax.tree.leaves(grad)
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)

    def test_no_interactions(self):
        """Should work with interactions disabled."""
        model = ResistanceSpline(n_knots=5, n_covariates=3,
                                 include_interactions=False)
        rng = jax.random.PRNGKey(4)
        x = jax.random.normal(rng, (25, 3))
        params = model.init(rng, x)
        out = model.apply(params, x)
        assert out.shape == (25,)

    def test_interactions_present(self):
        """With interactions, model should have interaction params."""
        model = ResistanceSpline(n_knots=5, n_covariates=3,
                                 include_interactions=True)
        rng = jax.random.PRNGKey(5)
        x = jax.random.normal(rng, (20, 3))
        params = model.init(rng, x)
        # Should have interact_0_1, interact_0_2, interact_1_2 params
        param_names = set(params['params'].keys())
        assert 'interact_0_1' in param_names
        assert 'interact_0_2' in param_names
        assert 'interact_1_2' in param_names


# ---------------------------------------------------------------------------
# ResistanceIRL
# ---------------------------------------------------------------------------

class TestResistanceIRL:
    """Tests for the IRL resistance model."""

    def test_output_shape_no_adjacency(self):
        """Without adjacency, should output (n_cells,) from raw reward."""
        model = ResistanceIRL(hidden_dim=16, n_hidden=1)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (30, 3))
        params = model.init(rng, x)
        out = model.apply(params, x)
        assert out.shape == (30,)

    def test_output_shape_with_adjacency(self):
        """With adjacency matrix, should output (n_cells,) from value iteration."""
        model = ResistanceIRL(hidden_dim=8, n_hidden=1,
                              n_value_iter=5, gamma_d=0.8)
        rng = jax.random.PRNGKey(1)
        n = 20
        x = jax.random.normal(rng, (n, 2))
        # Simple adjacency: identity + nearest neighbor ring
        adj = jnp.eye(n) * 0.5
        adj = adj.at[jnp.arange(n - 1), jnp.arange(1, n)].set(0.25)
        adj = adj.at[jnp.arange(1, n), jnp.arange(n - 1)].set(0.25)
        params = model.init(rng, x, adjacency=adj)
        out = model.apply(params, x, adjacency=adj)
        assert out.shape == (n,)

    def test_output_clamped(self):
        """Exponentiated output should respect resistance bounds."""
        model = ResistanceIRL(hidden_dim=8, n_hidden=1,
                              r_min=1.0, r_max=5000.0)
        rng = jax.random.PRNGKey(2)
        x = jax.random.normal(rng, (25, 2)) * 5
        params = model.init(rng, x)
        out = model.apply(params, x)
        resistance = jnp.exp(out)
        assert jnp.all(resistance >= 0.9)
        assert jnp.all(resistance <= 5500.0)

    def test_gradients_finite_no_adjacency(self):
        """Parameter gradients without adjacency should be finite."""
        model = ResistanceIRL(hidden_dim=8, n_hidden=1)
        rng = jax.random.PRNGKey(3)
        x = jax.random.normal(rng, (15, 2))
        params = model.init(rng, x)

        def loss_fn(p):
            return jnp.sum(model.apply(p, x))

        grad = jax.grad(loss_fn)(params)
        leaves = jax.tree.leaves(grad)
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)

    def test_gradients_finite_with_adjacency(self):
        """Parameter gradients with adjacency should be finite."""
        model = ResistanceIRL(hidden_dim=8, n_hidden=1,
                              n_value_iter=3, gamma_d=0.8)
        rng = jax.random.PRNGKey(4)
        n = 10
        x = jax.random.normal(rng, (n, 2))
        adj = jnp.eye(n) * 0.5
        adj = adj.at[jnp.arange(n - 1), jnp.arange(1, n)].set(0.25)
        adj = adj.at[jnp.arange(1, n), jnp.arange(n - 1)].set(0.25)
        params = model.init(rng, x, adjacency=adj)

        def loss_fn(p):
            return jnp.sum(model.apply(p, x, adjacency=adj))

        grad = jax.grad(loss_fn)(params)
        leaves = jax.tree.leaves(grad)
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
