"""Matched float64 physical CPU checks; run against an installed wheel."""
import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from diffiscape.core import physical_solve, physical_solve_init


def benchmark(size, formulation, repeats=3):
    permeability = jnp.asarray(np.random.default_rng(42).uniform(.05, 1, (size, size)))
    state = physical_solve_init(permeability, formulation, .01, rtol=1e-8)
    solve = jax.jit(lambda p: physical_solve(p, formulation, .01, 2, True, state))
    # Compile once; time synchronized execution with the same geometry and state.
    jax.block_until_ready(solve(permeability))
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        current, voltage = jax.block_until_ready(solve(permeability))
        timings.append(time.perf_counter() - start)
    assert np.isfinite(current).all() and np.isfinite(voltage).all()
    gradient = jax.jit(jax.grad(lambda p: solve(p)[0].sum()))(permeability)
    assert np.isfinite(gradient).all()
    return dict(size=size, formulation=formulation, seconds=min(timings),
                tolerance=1e-8, source_spacing=2, weighted_sources=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 50, 100])
    args = parser.parse_args()
    with jax.default_device(jax.devices("cpu")[0]):
        for size in args.sizes:
            for formulation in ("global_absorption", "boundary"):
                print(json.dumps(benchmark(size, formulation)))
