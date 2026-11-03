"""Independent dense oracles for the migration's CPU solver gate.

No production solver, adjoint, or preconditioner is implemented here.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp
import lineax as lx
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import AMJaxCGSolver, BCOOLinearOperator
from jaxscape.utils import graph_laplacian

jax.config.update("jax_enable_x64", True)


def dense_adjacency(permeability, harmonic=False):
    rows, cols = permeability.shape
    p = np.asarray(permeability).ravel()
    adjacency = np.zeros((p.size, p.size))
    for row in range(rows):
        for col in range(cols):
            i = row * cols + col
            for rr, cc in ((row + 1, col), (row, col + 1)):
                if rr < rows and cc < cols:
                    k = rr * cols + cc
                    w = 2 * p[i] * p[k] / (p[i] + p[k]) if harmonic else (p[i] + p[k]) / 2
                    adjacency[i, k] = adjacency[k, i] = w
    return adjacency


def dense_distance(permeability, source):
    adjacency = dense_adjacency(permeability)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    inverse = np.linalg.pinv(laplacian, hermitian=True)
    return inverse[source, source] + np.diag(inverse) - 2 * inverse[source]


def mean_weight(x, y):
    return (x + y) / 2


def harmonic_weight(x, y):
    return 2 * x * y / (x + y)


def finite_difference(fun, values, index, step=1e-5):
    upper, lower = np.array(values), np.array(values)
    upper[index] += step
    lower[index] -= step
    return (float(fun(upper)) - float(fun(lower))) / (2 * step)


@pytest.mark.parametrize("shape", [(4, 5), (12, 13)])
@pytest.mark.parametrize("boundary_only", [False, True])
def test_public_amjax_state_forward_and_backward(shape, boundary_only):
    """Test changing values, nonuniform sources, and a multilevel hierarchy."""
    p = np.linspace(0.1, 0.9, np.prod(shape)).reshape(shape)
    p[1, 2] = 1 / 5000  # finite masked-cell barrier, not a removed vertex
    n = p.size
    boundary = np.zeros(shape, dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    absorption = 0.2 * (boundary.ravel() if boundary_only else np.ones(n))
    source = np.linspace(0.2, 1.0, n)
    if boundary_only:
        source[boundary.ravel()] = 0

    def operator(values):
        adjacency = GridGraph(jnp.asarray(values), fun=harmonic_weight).get_adjacency_matrix()
        laplacian = graph_laplacian(adjacency)
        from jax.experimental.sparse import BCOO
        diagonal = jnp.stack([jnp.arange(n), jnp.arange(n)], axis=1)
        return BCOOLinearOperator(laplacian + BCOO((jnp.asarray(absorption), diagonal), shape=(n, n)))

    solver = AMJaxCGSolver(rtol=1e-10, atol=1e-10, max_steps=2000)
    state = solver.init(operator(p), {})

    def solve(values):
        op = operator(values)
        refreshed = solver.reinit(op, {}, state)
        return lx.linear_solve(op, jnp.asarray(source), solver=solver, state=refreshed).value

    changed = p * 0.73
    adjacency = dense_adjacency(changed, harmonic=True)
    expected = np.linalg.solve(np.diag(adjacency.sum(axis=1) + absorption) - adjacency, source)
    np.testing.assert_allclose(solve(changed), expected, rtol=2e-8, atol=2e-8)
    objective = lambda values: jnp.sum(solve(values) ** 2)
    derivative = jax.grad(objective)(jnp.asarray(changed))
    assert np.isfinite(derivative).all()
    for index in [(1, 1), (2, 3)]:
        np.testing.assert_allclose(derivative[index], finite_difference(objective, changed, index), rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("solver", [None, lx.CG(rtol=1e-10, atol=1e-10, max_steps=2000)])
def test_effective_resistance_forward_and_backward(solver):
    p = np.linspace(0.2, 0.9, 12).reshape(3, 4)
    distance = ResistanceDistance(solver=solver)

    def distances(values):
        return distance(GridGraph(jnp.asarray(values), fun=mean_weight), jnp.array([5]))[0]

    np.testing.assert_allclose(distances(p), dense_distance(p, 5), rtol=2e-8, atol=2e-8)
    objective = lambda values: jnp.sum(distances(values))
    gradient = jax.grad(objective)(jnp.asarray(p))
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(gradient[1, 2], finite_difference(objective, p, (1, 2)), rtol=2e-4, atol=2e-5)


def test_legacy_window_is_resistance_distance():
    """A nonsquare independent window oracle identifies the returned quantity."""
    from diffiscape.core import cumulative_current_core

    p = np.array([[0.2, 0.3, 0.5], [0.7, 0.4, 0.8]])
    padded = np.pad(p, 1, mode="edge")
    accumulated = np.zeros_like(padded)
    for row in range(p.shape[0]):
        for col in range(p.shape[1]):
            window = padded[row:row + 3, col:col + 3]
            accumulated[row:row + 3, col:col + 3] += dense_distance(window, 4).reshape(3, 3)
    actual = cumulative_current_core(jnp.asarray(p), *p.shape, radius=1, block_size=1)
    np.testing.assert_allclose(actual, accumulated[1:-1, 1:-1], rtol=2e-8, atol=2e-8)
    objective = lambda values: jnp.sum(cumulative_current_core(jnp.asarray(values), *p.shape, radius=1, block_size=1))
    gradient = jax.grad(objective)(jnp.asarray(p))
    np.testing.assert_allclose(gradient[0, 1], finite_difference(objective, p, (0, 1)), rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("formulation", ["global_absorption", "boundary"])
@pytest.mark.parametrize("weighted", [False, True])
def test_physical_current_and_voltage_match_dense_reference(formulation, weighted):
    from diffiscape.core import physical_solve, physical_solve_init
    p = np.linspace(0.1, 0.9, 30).reshape(5, 6)
    p[2, 3] = 1 / 5000
    state = physical_solve_init(jnp.asarray(p), formulation, 0.2)
    changed = p * 0.8
    adjacency = dense_adjacency(changed, harmonic=True)
    matrix = np.diag(adjacency.sum(axis=1)) - adjacency
    boundary = np.zeros(p.shape, dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    source = np.zeros(p.shape)
    source[1::2, 1::2] = 1
    if weighted:
        source *= changed
    if formulation == "boundary":
        source[boundary] = 0
        matrix[boundary.ravel(), :] = 0
        matrix[:, boundary.ravel()] = 0
        matrix[boundary.ravel(), boundary.ravel()] = 1
    else:
        matrix += 0.2 * np.eye(p.size)
    expected_voltage = np.linalg.solve(matrix, source.ravel())
    expected_current = (adjacency * np.sqrt((expected_voltage[:, None] - expected_voltage[None, :]) ** 2 + 1e-8)).sum(axis=1)

    def solve(values):
        return physical_solve(jnp.asarray(values), formulation, 0.2, 2, weighted, state)

    current, voltage = solve(changed)
    np.testing.assert_allclose(voltage, expected_voltage, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(current, expected_current, rtol=2e-8, atol=2e-8)
    objective = lambda values: jnp.sum(solve(values)[0] ** 2)
    derivative = jax.jit(jax.grad(objective))(jnp.asarray(changed))
    np.testing.assert_allclose(derivative[1, 2], finite_difference(objective, changed, (1, 2)), rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("weighted", [False, True])
def test_focal_current_conservation_against_dense_solve(weighted):
    from diffiscape.core import focal_current
    p = np.linspace(0.1, 0.9, 30).reshape(5, 6)
    expected = np.zeros_like(p)
    for row in range(5):
        for col in range(6):
            r0, r1 = max(row - 1, 0), min(row + 2, 5)
            c0, c1 = max(col - 1, 0), min(col + 2, 6)
            window = p[r0:r1, c0:c1]
            adjacency = dense_adjacency(window, harmonic=True)
            laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
            center = (row - r0) * window.shape[1] + col - c0
            keep = np.arange(window.size) != center
            source = window.ravel() if weighted else np.ones(window.size)
            voltage = np.zeros(window.size)
            voltage[keep] = np.linalg.solve(laplacian[np.ix_(keep, keep)], source[keep])
            expected[row, col] = adjacency[center] @ voltage
    np.testing.assert_allclose(focal_current(jnp.asarray(p), 1, 1, weighted), expected, rtol=1e-12, atol=1e-12)
    objective = lambda values: jnp.sum(focal_current(values, 1, 2, weighted) ** 2)
    gradient = jax.jit(jax.grad(objective))(jnp.asarray(p))
    np.testing.assert_allclose(gradient[2, 3], finite_difference(objective, p, (2, 3)), rtol=2e-4, atol=2e-5)


def test_public_physical_connectivity_preserves_high_resistance_barriers():
    from diffiscape import connectivity
    resistance = np.arange(1., 13.).reshape(3, 4)
    resistance[1, 2] = 1e6
    p = 1 / resistance
    adjacency = dense_adjacency(p, harmonic=True)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    voltage = np.linalg.solve(laplacian + .2 * np.eye(12), p.ravel())
    actual = connectivity(resistance, absorption=.2, source_from_resistance=True)
    np.testing.assert_allclose(actual["voltage"].ravel(), voltage, atol=1e-8)
    focal = connectivity(resistance, formulation="focal_current", radius=1, block_size=1,
                         source_from_resistance=True)
    assert focal["current"].shape == resistance.shape
    assert focal["voltage"] is None
