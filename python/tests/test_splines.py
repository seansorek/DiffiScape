import numpy as np
import jax
import jax.numpy as jnp

from diffiscape.splines import knots, basis, differentiable_basis, design_matrix


def test_scipy_basis_and_jit_derivatives():
    t = knots(3, 3)
    x = jnp.array([0.13, 0.42, 0.89])
    weights = jnp.arange(7, dtype=jnp.float64) ** 2
    objective = lambda v: jnp.sum(differentiable_basis(v, t) @ weights)
    np.testing.assert_allclose(differentiable_basis(x, t), basis(x, t))
    np.testing.assert_allclose(jax.jit(jax.grad(objective))(x), basis(x, t, derivative=1) @ weights, rtol=1e-12)
    np.testing.assert_allclose(np.diag(jax.hessian(objective)(x)), basis(x, t, derivative=2) @ weights, rtol=1e-12)


def test_basis_degree_tensor_product_and_clamping():
    x = np.array([[-0.1, 0.2], [0.5, 0.8], [1.1, 1]])
    marginal, tensor = design_matrix(x, 2, 2)
    assert marginal.shape == (3, 2, 5)
    assert tensor.shape == (3, 1, 25)
    np.testing.assert_allclose(marginal.sum(axis=-1), 1, atol=1e-12)
    np.testing.assert_allclose(tensor[:, 0], (marginal[:, 0, :, None] * marginal[:, 1, None, :]).reshape(3, -1))
    gradient = jax.grad(lambda v: jnp.sum(differentiable_basis(v, knots(2, 2)) ** 2))(jnp.array([-0.1, 1.1]))
    np.testing.assert_array_equal(gradient, 0)
