"""Frozen outputs captured before retiring Torch; Torch is not a test dependency."""
import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import pytest

from diffiscape.physical import _prepare
from diffiscape.physical_models import smoothing_penalty


CASES = json.loads((Path(__file__).parent / "fixtures" / "torch_model_reference.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["configuration"]["model_type"])
def test_resistance_model_matches_original_torch(case):
    cfg = case["configuration"]
    model, args, initial, _ = _prepare(np.array(case["basis"]), np.array(case["mask"]), cfg)
    _, unpack = ravel_pytree(initial)
    parameters = unpack(jnp.array(case["parameters"]))
    log_r, raw = model.apply(parameters, *args, return_raw=True)
    np.testing.assert_allclose(raw, case["raw"], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(jnp.exp(log_r), case["resistance"], rtol=1e-11, atol=1e-11)
    if "penalty" in case:
        np.testing.assert_allclose(smoothing_penalty(parameters, cfg["model_config"]["lambda_min"]), case["penalty"], rtol=1e-12)
