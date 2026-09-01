"""Bounded ExpertSSD routed-MoE implementation for native GLM experts."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn

from .contract import ContractError
from .expert_reader import LoadedExpert, NativeExpertReader


@dataclass(frozen=True)
class ExpertCacheStats:
    layer: int
    capacity: int
    resident: int
    hits: int
    misses: int
    evictions: int
    route_sync_seconds: float = 0.0
    slot_plan_seconds: float = 0.0
    policy: str = "lru"

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class ExpertLRU:
    """One layer's bounded cache of immutable native MXFP4 experts."""

    def __init__(
        self,
        reader: NativeExpertReader,
        *,
        layer: int,
        capacity: int,
    ):
        if not reader.plan.first_layer <= layer <= reader.plan.last_layer:
            raise ContractError(f"ExpertSSD layer is out of range: {layer}")
        if not 1 <= capacity <= reader.plan.experts_per_layer:
            raise ContractError(
                f"ExpertSSD capacity must be within 1..{reader.plan.experts_per_layer}: "
                f"{capacity}"
            )
        self.reader = reader
        self.layer = layer
        self.capacity = capacity
        self._experts: OrderedDict[int, LoadedExpert] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def ensure(self, experts: Iterable[int]) -> dict[int, LoadedExpert]:
        requested = tuple(dict.fromkeys(int(expert) for expert in experts))
        if len(requested) > self.capacity:
            raise ContractError(
                f"layer {self.layer} routed {len(requested)} unique experts, "
                f"exceeding ExpertSSD capacity {self.capacity}; v1 requires "
                "token-at-a-time execution with capacity >= top-k"
            )
        invalid = [
            expert
            for expert in requested
            if not 0 <= expert < self.reader.plan.experts_per_layer
        ]
        if invalid:
            raise ContractError(f"routed expert ids are out of range: {invalid}")
        for expert in requested:
            loaded = self._experts.pop(expert, None)
            if loaded is not None:
                self._experts[expert] = loaded
                self._hits += 1
                continue
            self._misses += 1
            if len(self._experts) == self.capacity:
                self._experts.popitem(last=False)
                self._evictions += 1
            self._experts[expert] = self.reader.load(self.layer, expert)
        return {expert: self._experts[expert] for expert in requested}

    @property
    def resident_experts(self) -> tuple[int, ...]:
        return tuple(self._experts)

    def stats(self) -> ExpertCacheStats:
        return ExpertCacheStats(
            layer=self.layer,
            capacity=self.capacity,
            resident=len(self._experts),
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
        )

    def clear(self) -> None:
        self._experts.clear()


def limited_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit > 0:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


def _flatten(values) -> list[int]:
    result: list[int] = []

    def visit(value) -> None:
        if isinstance(value, list):
            for child in value:
                visit(child)
        else:
            result.append(int(value))

    visit(values)
    return result


def _remap(values, mapping: dict[int, int]):
    if isinstance(values, list):
        return [_remap(value, mapping) for value in values]
    return mapping[int(values)]


class ExpertSSD(nn.Module):
    """Routed SwitchGLU whose only expert storage is a bounded SSD cache."""

    def __init__(
        self,
        reader: NativeExpertReader,
        *,
        layer: int,
        capacity: int,
        swiglu_limit: float,
    ):
        super().__init__()
        self.layer = layer
        self.capacity = capacity
        self.swiglu_limit = swiglu_limit
        self.cache = ExpertLRU(reader, layer=layer, capacity=capacity)
        self.last_expert_ids: tuple[int, ...] = ()

    @staticmethod
    def _compact(
        experts: list[LoadedExpert],
        projection: str,
    ) -> tuple[mx.array, mx.array]:
        weights, scales = zip(
            *(expert.projection(projection) for expert in experts),
            strict=True,
        )
        return mx.stack(weights), mx.stack(scales)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        if indices.ndim < 1 or indices.shape[-1] < 1:
            raise ContractError("routed expert indices must have a top-k dimension")
        nested = indices.tolist()
        routed = _flatten(nested)
        unique = list(dict.fromkeys(routed))
        self.last_expert_ids = tuple(routed)
        loaded = self.cache.ensure(unique)
        compact = [loaded[expert] for expert in unique]
        compact_ids = {expert: index for index, expert in enumerate(unique)}
        remapped = mx.array(_remap(nested, compact_ids), dtype=mx.uint32)
        expanded = mx.expand_dims(x, (-2, -3))

        def qmm(value: mx.array, projection: str) -> mx.array:
            weight, scales = self._compact(compact, projection)
            return mx.gather_qmm(
                value,
                weight,
                scales,
                rhs_indices=remapped,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
            )

        gate = qmm(expanded, "gate_proj")
        up = qmm(expanded, "up_proj")
        activated = limited_swiglu(gate, up, self.swiglu_limit)
        return qmm(activated, "down_proj").squeeze(-2)

    def stats(self) -> ExpertCacheStats:
        return self.cache.stats()

    def clear(self) -> None:
        self.cache.clear()
