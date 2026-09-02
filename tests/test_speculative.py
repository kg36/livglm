from types import SimpleNamespace

import mlx.core as mx

from glm53flash.cache import DSACache, KDACache, ModelCache
from glm53flash.mtp import GLMNextTokenPredictor


def test_width_two_rejection_commits_only_authoritative_cache_row():
    kda = KDACache.empty(
        batch_size=1,
        qkv_dim=4,
        conv_kernel=2,
        num_heads=1,
        head_dim=2,
    )
    dsa = DSACache()
    cache = ModelCache([kda, dsa], position=7)
    snapshot = cache.snapshot()

    first_kda = (
        mx.ones_like(kda.q_conv),
        mx.ones_like(kda.k_conv) * 2,
        mx.ones_like(kda.v_conv) * 3,
        mx.ones_like(kda.recurrent) * 4,
    )
    first_keys = mx.ones((1, 1, 8, 2), dtype=mx.bfloat16)
    first_values = mx.ones((1, 1, 8, 2), dtype=mx.bfloat16) * 5
    kda._speculative_first = first_kda
    dsa._speculative_first = (first_keys, first_values)
    cache.position = 9

    cache.commit_first_from_wide(snapshot)

    assert cache.position == 8
    assert kda.recurrent is first_kda[-1]
    assert dsa.keys is first_keys
    assert dsa.values is first_values
    assert not hasattr(kda, "_speculative_first")
    assert not hasattr(dsa, "_speculative_first")


def test_mtp_attention_only_catchup_appends_exact_projected_kv():
    mtp = GLMNextTokenPredictor.__new__(GLMNextTokenPredictor)
    object.__setattr__(mtp, "enorm", lambda value: value)
    object.__setattr__(mtp, "hnorm", lambda value: value)
    object.__setattr__(mtp, "eh_proj", lambda value: value[..., :2])
    attention = SimpleNamespace(
        kv_a_proj_with_mqa=lambda value: value,
        kv_a_layernorm=lambda value: value,
        kv_b_proj=lambda value: mx.concatenate((value, value), axis=-1),
        num_heads=1,
        qk_nope_head_dim=2,
        v_head_dim=2,
    )
    object.__setattr__(
        mtp,
        "decoder",
        SimpleNamespace(
            self_attn=attention,
            input_layernorm=lambda value: value,
        ),
    )
    cache = DSACache()
    embedding = mx.array([[[1.0, 2.0]]], dtype=mx.bfloat16)
    hidden = mx.array([[[3.0, 4.0]]], dtype=mx.bfloat16)

    anchor = mtp.advance_attention_cache(embedding, hidden, cache)
    mx.eval(anchor, cache.values)

    assert cache.length == 1
    assert cache.keys.shape == (1, 1, 1, 2)
    assert cache.values.shape == (1, 1, 1, 2)
    assert cache.keys.tolist() == [[[[1.0, 2.0]]]]
    assert cache.values.tolist() == [[[[1.0, 2.0]]]]
