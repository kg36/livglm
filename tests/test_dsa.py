import mlx.core as mx
import numpy as np
import pytest

from glm53flash.contract import ContractError
from glm53flash.dsa import DenseEquivalentDSAAttention, dense_short_context_is_exact
from tests.helpers import tiny_config


def _rms(x, weight, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def test_dsa_real_mla_projections_match_dense_reference():
    config = tiny_config(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        v_head_dim=4,
        index_topk=8,
    )
    module = DenseEquivalentDSAAttention(config, 3, context_limit=4)
    rng = np.random.default_rng(20)

    def matrix(rows, cols):
        return rng.normal(0, 0.15, (rows, cols)).astype(np.float32)

    wa, wb = matrix(4, 8), matrix(8, 4)
    wka, wkb = matrix(4, 8), matrix(16, 4)
    wo = matrix(8, 8)
    qnorm = rng.uniform(0.8, 1.2, 4).astype(np.float32)
    kvnorm = rng.uniform(0.8, 1.2, 4).astype(np.float32)
    module.q_a_proj.weight, module.q_b_proj.weight = mx.array(wa), mx.array(wb)
    module.q_a_layernorm.weight = mx.array(qnorm)
    module.kv_a_proj_with_mqa.weight, module.kv_b_proj.weight = mx.array(wka), mx.array(wkb)
    module.kv_a_layernorm.weight = mx.array(kvnorm)
    module.o_proj.weight = mx.array(wo)

    cache = module.empty_cache()
    keys, values = [], []
    for token in rng.normal(0, 0.2, (3, 8)).astype(np.float32):
        x = token.reshape(1, 1, 8)
        q = _rms(x @ wa.T, qnorm, config.rms_norm_eps) @ wb.T
        q = q.reshape(1, 1, 2, 4).transpose(0, 2, 1, 3)
        kv = _rms(x @ wka.T, kvnorm, config.rms_norm_eps) @ wkb.T
        kv = kv.reshape(1, 1, 2, 8).transpose(0, 2, 1, 3)
        key, value = np.split(kv, [4], axis=-1)
        keys.append(key)
        values.append(value)
        all_keys = np.concatenate(keys, axis=2)
        all_values = np.concatenate(values, axis=2)
        scores = (q @ all_keys.swapaxes(-1, -2)) / 2.0
        scores -= scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        attended = (probs @ all_values).transpose(0, 2, 1, 3).reshape(1, 1, 8)
        expected = attended @ wo.T

        actual = module(mx.array(x), cache)
        mx.eval(actual, cache.keys, cache.values)
        np.testing.assert_allclose(np.array(actual), expected, rtol=2e-3, atol=2e-4)


def test_dsa_dense_domain_guard():
    assert dense_short_context_is_exact(history_length=128, index_topk=2048)
    assert not dense_short_context_is_exact(history_length=2049, index_topk=2048)
    assert not dense_short_context_is_exact(history_length=1, index_topk=2048, has_padding=True)

    config = tiny_config(index_topk=2)
    module = DenseEquivalentDSAAttention(config, 3, context_limit=2)
    # The guard occurs before any deferred parameter is touched.
    module.empty_cache().keys = mx.zeros((1, 4, 2, 8))
    cache = module.empty_cache()
    cache.keys = mx.zeros((1, 4, 2, 8))
    cache.values = mx.zeros((1, 4, 2, 8))
    with pytest.raises(ContractError, match="domain exceeded"):
        module(mx.zeros((1, 1, 32)), cache)
