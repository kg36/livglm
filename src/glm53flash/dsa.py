"""Short-context exact-dense DSA/MLA path for the v1 runtime."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .cache import DSACache
from .contract import ContractError
from .layers import DeferredLayerNorm, DeferredLinear, RMSNorm
from .model_config import GLMTextConfig


class DormantDSAIndexer(nn.Module):
    """Own all official indexer parameters while v1 uses its proven dense domain."""

    def __init__(self, config: GLMTextConfig):
        super().__init__()
        self.wq_b = DeferredLinear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            quantize_resident=False,
        )
        self.wk = DeferredLinear(
            config.hidden_size,
            config.index_head_dim,
            quantize_resident=False,
        )
        self.k_norm = DeferredLayerNorm(config.index_head_dim)
        self.weights_proj = DeferredLinear(
            config.hidden_size,
            config.index_n_heads,
            quantize_resident=False,
        )
        self.index_kpool_compress_ape = None
        self.index_kpool_compress_gate = None


def dense_short_context_is_exact(
    *,
    history_length: int,
    index_topk: int,
    token_at_a_time: bool = True,
    batch_size: int = 1,
    has_padding: bool = False,
) -> bool:
    """Whether the official DSA candidate budget necessarily covers all visible keys."""
    return (
        token_at_a_time
        and batch_size == 1
        and not has_padding
        and 0 < history_length <= index_topk
    )


class DenseEquivalentDSAAttention(nn.Module):
    """The real MLA projections with DSA scoring skipped only in the exact dense domain."""

    def __init__(self, config: GLMTextConfig, layer_idx: int, *, context_limit: int):
        super().__init__()
        if context_limit > config.index_topk:
            raise ContractError(
                f"v1 DSA context limit {context_limit} exceeds index_topk {config.index_topk}"
            )
        self.layer_idx = layer_idx
        self.context_limit = context_limit
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_head_dim = config.qk_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.scaling = self.qk_head_dim**-0.5

        self.q_a_proj = DeferredLinear(config.hidden_size, config.q_lora_rank)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = DeferredLinear(
            config.q_lora_rank,
            self.num_heads * self.qk_head_dim,
        )
        self.kv_a_proj_with_mqa = DeferredLinear(config.hidden_size, config.kv_lora_rank)
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = DeferredLinear(
            config.kv_lora_rank,
            self.num_heads * (config.qk_nope_head_dim + config.v_head_dim),
        )
        self.o_proj = DeferredLinear(
            self.num_heads * config.v_head_dim,
            config.hidden_size,
        )
        self.indexer = DormantDSAIndexer(config)

    def empty_cache(self) -> DSACache:
        return DSACache()

    def __call__(self, hidden_states: mx.array, cache: DSACache) -> mx.array:
        if hidden_states.ndim != 3 or hidden_states.shape[:2] != (1, 1):
            raise ContractError("v1 DSA requires batch=1 and token-at-a-time execution")
        next_length = cache.length + 1
        if not dense_short_context_is_exact(
            history_length=next_length,
            index_topk=self.context_limit,
        ):
            raise ContractError(
                f"v1 DSA dense-equivalence domain exceeded at token {next_length}; "
                f"hard limit is {self.context_limit}"
            )

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query = self.q_b_proj(q_resid)
        query = query.reshape(1, 1, self.num_heads, self.qk_head_dim)
        query = mx.swapaxes(query, 1, 2)

        compressed = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states))
        expanded = self.kv_b_proj(compressed)
        expanded = expanded.reshape(
            1,
            1,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        expanded = mx.swapaxes(expanded, 1, 2)
        key, value = mx.split(expanded, (self.qk_nope_head_dim,), axis=-1)
        cache.keys = key if cache.keys is None else mx.concatenate((cache.keys, key), axis=2)
        cache.values = value if cache.values is None else mx.concatenate((cache.values, value), axis=2)

        scores = mx.matmul(
            query.astype(mx.float32),
            mx.swapaxes(cache.keys.astype(mx.float32), -1, -2),
        ) * self.scaling
        probabilities = mx.softmax(scores, axis=-1).astype(query.dtype)
        attended = mx.matmul(probabilities, cache.values)
        attended = mx.swapaxes(attended, 1, 2).reshape(
            1,
            1,
            self.num_heads * self.v_head_dim,
        )
        return self.o_proj(attended)
