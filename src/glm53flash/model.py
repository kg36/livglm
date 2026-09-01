"""Text-only GLM-5.3-Flash MLX graph with mandatory ExpertSSD MoE layers."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .cache import DSACache, KDACache, LayerCache, ModelCache
from .contract import ContractError
from .dsa import DenseEquivalentDSAAttention
from .kda import KDALinearAttention
from .layers import DeferredEmbedding, DeferredLinear, DenseMLP, RMSNorm
from .mhc import HyperConnection, HyperHead, hyper_residual
from .model_config import GLMTextConfig
from .moe import ExpertSSDBackend, ExpertSSDMoE
from .trace import DecodeTrace, active_trace


ExpertFactory = Callable[[int], ExpertSSDBackend]


class GLMDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GLMTextConfig,
        layer_idx: int,
        *,
        context_limit: int,
        expert_factory: ExpertFactory,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_type = config.layer_types[layer_idx]
        self.self_attn = (
            KDALinearAttention(config, layer_idx)
            if self.block_type == "linear_attention"
            else DenseEquivalentDSAAttention(
                config,
                layer_idx,
                context_limit=context_limit,
            )
        )
        self.mlp = (
            DenseMLP(
                config.hidden_size,
                config.intermediate_size,
                config.swiglu_limit,
            )
            if config.mlp_layer_types[layer_idx] == "dense"
            else ExpertSSDMoE(config, layer_idx, expert_factory)
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def empty_cache(self, batch_size: int, dtype=mx.bfloat16) -> LayerCache:
        if isinstance(self.self_attn, KDALinearAttention):
            return self.self_attn.empty_cache(batch_size, dtype)
        return self.self_attn.empty_cache()

    def __call__(self, hidden_states: mx.array, cache: LayerCache) -> mx.array:
        trace = active_trace()
        if trace is None:
            return self._call_impl(hidden_states, cache, trace=None)
        with trace.span(
            "transformer_layer",
            category="model_structure",
            args={
                "decode_index": trace.current_decode_index,
                "layer": self.layer_idx,
                "attention_type": self.block_type,
                "ffn_type": "moe" if isinstance(self.mlp, ExpertSSDMoE) else "dense",
            },
        ):
            return self._call_impl(hidden_states, cache, trace=trace)

    def _call_impl(
        self,
        hidden_states: mx.array,
        cache: LayerCache,
        *,
        trace: DecodeTrace | None,
    ) -> mx.array:
        stage_args = {
            "decode_index": trace.current_decode_index if trace is not None else None,
            "layer": self.layer_idx,
            "semantics": (
                "MLX graph construction; exact GPU execution is in the paired "
                "Metal System Trace"
            ),
        }
        attention_context = (
            trace.span(
                "attention_graph_construct",
                category="mlx_submit",
                args={**stage_args, "attention_type": self.block_type},
            )
            if trace is not None
            else nullcontext({})
        )
        with attention_context:
            residual = hidden_states
            post, comb, collapsed = self.attn_hc(hidden_states)
            collapsed = self.input_layernorm(collapsed)
            if isinstance(self.self_attn, KDALinearAttention):
                if not isinstance(cache, KDACache):
                    raise ContractError(f"layer {self.layer_idx} expected a KDA cache")
                attended = self.self_attn(collapsed, cache)
            else:
                if not isinstance(cache, DSACache):
                    raise ContractError(f"layer {self.layer_idx} expected a DSA cache")
                attended = self.self_attn(collapsed, cache)
            hidden_states = hyper_residual(residual, attended, post, comb)

        ffn_context = (
            trace.span(
                "ffn_graph_construct",
                category="mlx_submit",
                args={
                    **stage_args,
                    "ffn_type": "moe" if isinstance(self.mlp, ExpertSSDMoE) else "dense",
                },
            )
            if trace is not None
            else nullcontext({})
        )
        with ffn_context:
            residual = hidden_states
            post, comb, collapsed = self.ffn_hc(hidden_states)
            fed_forward = self.mlp(self.post_attention_layernorm(collapsed))
            return hyper_residual(residual, fed_forward, post, comb)


class GLMTextModel(nn.Module):
    def __init__(
        self,
        config: GLMTextConfig,
        *,
        context_limit: int,
        expert_factory: ExpertFactory,
    ):
        super().__init__()
        self.config = config
        self.context_limit = context_limit
        self.embed_tokens = DeferredEmbedding(config.vocab_size, config.hidden_size)
        self.layers = [
            GLMDecoderLayer(
                config,
                index,
                context_limit=context_limit,
                expert_factory=expert_factory,
            )
            for index in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hc_head = HyperHead()

    def empty_cache(self, *, batch_size: int = 1, dtype=mx.bfloat16) -> ModelCache:
        return ModelCache(
            [layer.empty_cache(batch_size, dtype) for layer in self.layers]
        )

    def __call__(self, input_ids: mx.array, cache: ModelCache) -> mx.array:
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ContractError("v1 model execution accepts exactly one token per call")
        if input_ids.shape[0] != 1:
            raise ContractError("v1 model execution supports batch size one")
        if cache.position >= self.context_limit:
            raise ContractError(
                f"v1 context limit exceeded: {cache.position + 1} > {self.context_limit}"
            )
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = mx.broadcast_to(
            hidden_states[:, :, None, :],
            (*hidden_states.shape[:2], self.config.hc_mult, self.config.hidden_size),
        )
        for layer, layer_cache in zip(self.layers, cache.layers, strict=True):
            hidden_states = layer(hidden_states, layer_cache)
            # Each following MoE router materializes its indices before native
            # refill. The final logits barrier materializes the last layer and
            # every recurrent/cache update, so an additional finite-value GPU
            # round trip at every layer is redundant.
        cache.advance()
        return self.norm(self.hc_head(hidden_states))


class GLMForCausalLM(nn.Module):
    def __init__(
        self,
        config: GLMTextConfig,
        *,
        context_limit: int,
        expert_factory: ExpertFactory,
    ):
        super().__init__()
        self.config = config
        self.language_model = GLMTextModel(
            config,
            context_limit=context_limit,
            expert_factory=expert_factory,
        )
        self.lm_head = DeferredLinear(config.hidden_size, config.vocab_size)

    def empty_cache(self, *, batch_size: int = 1) -> ModelCache:
        return self.language_model.empty_cache(batch_size=batch_size)

    def __call__(self, input_ids: mx.array, cache: ModelCache) -> mx.array:
        return self.lm_head(self.language_model(input_ids, cache))

    def expert_layers(self) -> tuple[ExpertSSDMoE, ...]:
        values = [layer.mlp for layer in self.language_model.layers]
        return tuple(value for value in values if isinstance(value, ExpertSSDMoE))

    def quantize_resident_linears_mxfp8(self) -> dict[str, int | str]:
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

        source_bytes = 0
        destination_bytes = 0
        for index, module in enumerate(modules):
            source, destination = module.quantize_to_mxfp8()
            source_bytes += source
            destination_bytes += destination
            if index % 32 == 31:
                mx.clear_cache()
        mx.clear_cache()
        return {
            "format": "mxfp8",
            "linear_count": len(modules),
            "source_bytes": source_bytes,
            "destination_bytes": destination_bytes,
        }
