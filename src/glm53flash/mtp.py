"""Checkpoint-native GLM next-token predictor used for exact speculation."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .cache import DSACache
from .dsa import DenseEquivalentDSAAttention
from .layers import DeferredLinear, RMSNorm
from .model_config import GLMTextConfig
from .moe import ExpertSSDMoE


class MTPSharedHead(nn.Module):
    def __init__(self, config: GLMTextConfig):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class GLMMTPDecoderLayer(nn.Module):
    """Official Layer 45: ordinary residual DSA + sparse MoE."""

    def __init__(self, config: GLMTextConfig, expert_factory):
        super().__init__()
        self.self_attn = DenseEquivalentDSAAttention(
            config,
            45,
            context_limit=128,
        )
        self.mlp = ExpertSSDMoE(config, 45, expert_factory)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

    def empty_cache(self) -> DSACache:
        return self.self_attn.empty_cache()

    def __call__(self, hidden: mx.array, cache: DSACache) -> mx.array:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), cache)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class GLMNextTokenPredictor(nn.Module):
    """One exact official MTP draft step sharing target embedding and head."""

    def __init__(self, config: GLMTextConfig, expert_factory):
        super().__init__()
        self.enorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = DeferredLinear(2 * config.hidden_size, config.hidden_size)
        self.decoder = GLMMTPDecoderLayer(config, expert_factory)
        self.shared_head = MTPSharedHead(config)

    def empty_cache(self) -> DSACache:
        return self.decoder.empty_cache()

    def __call__(
        self,
        token_embedding: mx.array,
        target_hidden: mx.array,
        cache: DSACache,
    ) -> mx.array:
        hidden = self.eh_proj(
            mx.concatenate(
                (self.enorm(token_embedding), self.hnorm(target_hidden)),
                axis=-1,
            )
        )
        return self.shared_head.norm(self.decoder(hidden, cache))

    def advance_attention_cache(
        self,
        token_embedding: mx.array,
        target_hidden: mx.array,
        cache: DSACache,
    ) -> mx.array:
        """Append one exact MTP K/V entry without a discarded full block.

        After an accepted width-two verification, the MTP cache is one token
        behind the target. The next draft depends on that token's K/V entry,
        but not on its attention output, MoE, residual, normalization, or
        logits. This is the same exact catch-up reduction used by LivSeek.
        """

        hidden = self.eh_proj(
            mx.concatenate(
                (self.enorm(token_embedding), self.hnorm(target_hidden)),
                axis=-1,
            )
        )
        attention = self.decoder.self_attn
        hidden = self.decoder.input_layernorm(hidden)
        compressed = attention.kv_a_layernorm(
            attention.kv_a_proj_with_mqa(hidden)
        )
        expanded = attention.kv_b_proj(compressed).reshape(
            1,
            hidden.shape[1],
            attention.num_heads,
            attention.qk_nope_head_dim + attention.v_head_dim,
        )
        expanded = mx.swapaxes(expanded, 1, 2)
        key, value = mx.split(
            expanded,
            (attention.qk_nope_head_dim,),
            axis=-1,
        )
        cache.keys = key if cache.keys is None else mx.concatenate(
            (cache.keys, key), axis=2
        )
        cache.values = value if cache.values is None else mx.concatenate(
            (cache.values, value), axis=2
        )
        return cache.keys

    def quantize_resident_linears_mxfp4(self) -> dict[str, int]:
        modules: list[DeferredLinear] = []
        seen: set[int] = set()
        for _, module in tree_flatten(
            self.leaf_modules(),
            is_leaf=lambda value: isinstance(value, DeferredLinear),
        ):
            if (
                isinstance(module, DeferredLinear)
                and module.quantize_resident
                and id(module) not in seen
            ):
                seen.add(id(module))
                modules.append(module)
        source = destination = 0
        for module in modules:
            old, new = module.quantize_to_mxfp(4)
            source += old
            destination += new
        mx.clear_cache()
        return {
            "linear_count": len(modules),
            "source_bytes": source,
            "destination_bytes": destination,
        }

    @property
    def experts(self):
        return self.decoder.mlp.experts
