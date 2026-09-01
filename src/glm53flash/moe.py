"""GLM routing shell backed exclusively by ExpertSSD routed experts."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from .contract import ContractError
from .expert_ssd import ExpertSSD
from .layers import DenseMLP, require_weight
from .model_config import GLMTextConfig
from .native_expert_ssd import NativeExpertSSD


class TopKRouter(nn.Module):
    def __init__(self, config: GLMTextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = None
        self.e_score_correction_bias = None

    @staticmethod
    def _last_k_indices(values: mx.array, k: int) -> mx.array:
        return mx.argsort(values, axis=-1)[..., -k:]

    def __call__(self, hidden_states: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        rows = hidden_states.reshape(-1, self.hidden_dim).astype(mx.float32)
        logits = rows @ require_weight(self.weight, "router.weight").astype(mx.float32).T
        scores = mx.sigmoid(logits)
        choice = scores + require_weight(
            self.e_score_correction_bias,
            "router.e_score_correction_bias",
        ).astype(mx.float32)

        per_group = self.num_experts // self.num_group
        grouped = choice.reshape(-1, self.num_group, per_group)
        group_scores = mx.sum(mx.sort(grouped, axis=-1)[..., -2:], axis=-1)
        group_indices = self._last_k_indices(group_scores, self.topk_group)
        group_mask = mx.zeros(group_scores.shape, dtype=mx.bool_)
        group_mask = mx.put_along_axis(
            group_mask,
            group_indices,
            mx.ones(group_indices.shape, dtype=mx.bool_),
            axis=-1,
        )
        score_mask = mx.broadcast_to(group_mask[..., None], grouped.shape).reshape(choice.shape)
        choice = mx.where(score_mask, choice, mx.array(-float("inf"), mx.float32))
        indices = self._last_k_indices(choice, self.top_k).astype(mx.uint32)
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if self.norm_topk_prob:
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        return logits, weights, indices


ExpertSSDBackend = ExpertSSD | NativeExpertSSD
ExpertSSDFactory = Callable[[int], ExpertSSDBackend]


class ExpertSSDMoE(nn.Module):
    """Shared BF16 MLP plus mandatory on-demand MXFP4 ExpertSSD routes."""

    def __init__(
        self,
        config: GLMTextConfig,
        layer_idx: int,
        expert_factory: ExpertSSDFactory,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.gate = TopKRouter(config)
        self.experts = expert_factory(layer_idx)
        if not isinstance(self.experts, (ExpertSSD, NativeExpertSSD)):
            raise ContractError(
                f"sparse layer {layer_idx} must be backed by ExpertSSD, got "
                f"{type(self.experts).__name__}"
            )
        self.shared_experts = DenseMLP(
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
            config.swiglu_limit,
        )
        self.last_route: tuple[int, ...] = ()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        original_shape = hidden_states.shape
        rows = hidden_states.reshape(-1, self.hidden_size)
        _, weights, indices = self.gate(rows)
        if isinstance(self.experts, NativeExpertSSD):
            # Native reads run on I/O workers while the independent resident
            # shared expert executes on Metal.
            route_plan = self.experts.prepare(indices)
            shared = self.shared_experts(rows)
            mx.async_eval(shared)
            route_outputs = self.experts.finish(rows, route_plan)
        else:
            route_outputs = self.experts(rows, indices)
            shared = self.shared_experts(rows)
        routed = mx.sum(route_outputs * weights[..., None], axis=-2)
        self.last_route = self.experts.last_expert_ids
        return (routed + shared).reshape(original_shape)
