from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from glm53flash.moe import TopKRouter
from tests.helpers import tiny_config


def test_router_selects_only_best_group_and_normalizes_original_scores():
    config = tiny_config(
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
    )
    router = TopKRouter(config)
    router.weight = mx.zeros((4, 32), dtype=mx.float32)
    router.e_score_correction_bias = mx.array([10.0, 9.0, 1.0, 0.0])
    _, weights, indices = router(mx.zeros((1, 32), dtype=mx.float32))
    mx.eval(weights, indices)
    assert set(np.array(indices).reshape(-1).tolist()) == {0, 1}
    np.testing.assert_allclose(np.array(weights), [[1.25, 1.25]], atol=1e-6)
