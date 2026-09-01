"""Exact native-safetensors source plan for routed MXFP4 experts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contract import (
    ContractError,
    EXPERT_COMPONENTS,
    EXPERT_PARTS,
    EXPERTS_PER_LAYER,
    FIRST_MOE_LAYER,
    LAST_MAIN_LAYER,
    ModelContract,
    TensorSource,
)
from .model_config import GLMTextConfig


@dataclass(frozen=True)
class ExpertTensorSource:
    source_name: str
    destination_name: str
    shard_name: str
    absolute_offset: int
    byte_length: int
    source_shape: tuple[int, ...]
    mlx_dtype: str
    mlx_shape: tuple[int, ...]


@dataclass(frozen=True)
class ExpertReadRange:
    shard_name: str
    absolute_offset: int
    byte_length: int
    tensors: tuple[ExpertTensorSource, ...]

    def relative_offset(self, tensor: ExpertTensorSource) -> int:
        return tensor.absolute_offset - self.absolute_offset


@dataclass(frozen=True)
class NativeExpertSource:
    layer: int
    expert: int
    tensors: tuple[ExpertTensorSource, ...]
    read_ranges: tuple[ExpertReadRange, ...]

    @property
    def payload_bytes(self) -> int:
        return sum(tensor.byte_length for tensor in self.tensors)

    @property
    def read_bytes(self) -> int:
        return sum(item.byte_length for item in self.read_ranges)


@dataclass(frozen=True)
class NativeExpertSourcePlan:
    model_dir: str
    first_layer: int
    last_layer: int
    experts_per_layer: int
    packs: tuple[NativeExpertSource, ...]

    @property
    def layer_count(self) -> int:
        return self.last_layer - self.first_layer + 1

    @property
    def expert_count(self) -> int:
        return len(self.packs)

    @property
    def tensor_count(self) -> int:
        return sum(len(pack.tensors) for pack in self.packs)

    @property
    def payload_bytes(self) -> int:
        return sum(pack.payload_bytes for pack in self.packs)

    @property
    def read_bytes(self) -> int:
        return sum(pack.read_bytes for pack in self.packs)

    @property
    def source_shard_count(self) -> int:
        return len({item.shard_name for pack in self.packs for item in pack.read_ranges})

    @property
    def uniform_pack_bytes(self) -> int | None:
        values = {pack.payload_bytes for pack in self.packs}
        return next(iter(values)) if len(values) == 1 else None

    @property
    def uniform_read_count(self) -> int | None:
        values = {len(pack.read_ranges) for pack in self.packs}
        return next(iter(values)) if len(values) == 1 else None

    def expert(self, layer: int, expert: int) -> NativeExpertSource:
        if not self.first_layer <= layer <= self.last_layer:
            raise ContractError(f"expert layer is out of range: {layer}")
        if not 0 <= expert < self.experts_per_layer:
            raise ContractError(f"expert id is out of range: {expert}")
        offset = (layer - self.first_layer) * self.experts_per_layer + expert
        pack = self.packs[offset]
        if (pack.layer, pack.expert) != (layer, expert):
            raise ContractError("native expert source plan lost canonical ordering")
        return pack

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_kind": "livseek_glm_native_safetensors",
            "model_dir": self.model_dir,
            "first_layer": self.first_layer,
            "last_layer": self.last_layer,
            "layer_count": self.layer_count,
            "experts_per_layer": self.experts_per_layer,
            "expert_count": self.expert_count,
            "tensor_count": self.tensor_count,
            "payload_bytes": self.payload_bytes,
            "read_bytes": self.read_bytes,
            "source_shard_count": self.source_shard_count,
            "uniform_pack_bytes": self.uniform_pack_bytes,
            "uniform_read_count": self.uniform_read_count,
            "layers": [
                {
                    "layer": layer,
                    "expert_count": self.experts_per_layer,
                    "payload_bytes": sum(
                        self.expert(layer, expert).payload_bytes
                        for expert in range(self.experts_per_layer)
                    ),
                    "source_shards": sorted(
                        {
                            item.shard_name
                            for expert in range(self.experts_per_layer)
                            for item in self.expert(layer, expert).read_ranges
                        }
                    ),
                }
                for layer in range(self.first_layer, self.last_layer + 1)
            ],
        }


def _expected_shape(
    component: str,
    part: str,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[int, int]:
    rows, columns = (
        (hidden_size, intermediate_size)
        if component == "down_proj"
        else (intermediate_size, hidden_size)
    )
    divisor = 2 if part == "weight_packed" else 32
    if columns % divisor:
        raise ContractError(f"expert width is not divisible by {divisor}")
    return rows, columns // divisor


def _destination_tensor(source: TensorSource, component: str, part: str) -> ExpertTensorSource:
    if source.dtype != "U8":
        raise ContractError(f"native MXFP4 tensor must use U8 storage: {source.name}")
    if part == "weight_packed":
        if source.shape[-1] % 4:
            raise ContractError(f"packed expert width is not U32-viewable: {source.name}")
        mlx_dtype = "U32"
        mlx_shape = (*source.shape[:-1], source.shape[-1] // 4)
        destination_name = f"{component}.weight"
    else:
        mlx_dtype = "U8"
        mlx_shape = source.shape
        destination_name = f"{component}.scales"
    return ExpertTensorSource(
        source_name=source.name,
        destination_name=destination_name,
        shard_name=source.shard_name,
        absolute_offset=source.absolute_offset,
        byte_length=source.byte_length,
        source_shape=source.shape,
        mlx_dtype=mlx_dtype,
        mlx_shape=mlx_shape,
    )


def _coalesce(tensors: tuple[ExpertTensorSource, ...]) -> tuple[ExpertReadRange, ...]:
    ordered = sorted(tensors, key=lambda item: (item.shard_name, item.absolute_offset))
    ranges: list[ExpertReadRange] = []
    current: list[ExpertTensorSource] = []
    for tensor in ordered:
        if current:
            previous = current[-1]
            adjacent = (
                previous.shard_name == tensor.shard_name
                and previous.absolute_offset + previous.byte_length == tensor.absolute_offset
            )
            if not adjacent:
                first = current[0]
                end = previous.absolute_offset + previous.byte_length
                ranges.append(
                    ExpertReadRange(
                        shard_name=first.shard_name,
                        absolute_offset=first.absolute_offset,
                        byte_length=end - first.absolute_offset,
                        tensors=tuple(current),
                    )
                )
                current = []
        current.append(tensor)
    if current:
        first = current[0]
        last = current[-1]
        ranges.append(
            ExpertReadRange(
                shard_name=first.shard_name,
                absolute_offset=first.absolute_offset,
                byte_length=last.absolute_offset + last.byte_length - first.absolute_offset,
                tensors=tuple(current),
            )
        )
    return tuple(ranges)


def build_native_expert_source_plan(
    contract: ModelContract,
    config: GLMTextConfig,
    *,
    first_layer: int = FIRST_MOE_LAYER,
    last_layer: int = LAST_MAIN_LAYER,
) -> NativeExpertSourcePlan:
    packs: list[NativeExpertSource] = []
    for layer in range(first_layer, last_layer + 1):
        for expert in range(config.n_routed_experts):
            tensors: list[ExpertTensorSource] = []
            for component in EXPERT_COMPONENTS:
                for part in EXPERT_PARTS:
                    name = (
                        f"model.language_model.layers.{layer}.mlp.experts.{expert}."
                        f"{component}.{part}"
                    )
                    source = contract.tensor(name)
                    expected_shape = _expected_shape(
                        component,
                        part,
                        hidden_size=config.hidden_size,
                        intermediate_size=config.moe_intermediate_size,
                    )
                    if source.shape != expected_shape:
                        raise ContractError(
                            f"native MXFP4 shape mismatch for {name}: "
                            f"{source.shape} != {expected_shape}"
                        )
                    tensors.append(_destination_tensor(source, component, part))
            tensor_tuple = tuple(tensors)
            ranges = _coalesce(tensor_tuple)
            packs.append(
                NativeExpertSource(
                    layer=layer,
                    expert=expert,
                    tensors=tensor_tuple,
                    read_ranges=ranges,
                )
            )
    plan = NativeExpertSourcePlan(
        model_dir=str(contract.model_dir),
        first_layer=first_layer,
        last_layer=last_layer,
        experts_per_layer=config.n_routed_experts,
        packs=tuple(packs),
    )
    expected_experts = (last_layer - first_layer + 1) * config.n_routed_experts
    if plan.expert_count != expected_experts or plan.tensor_count != expected_experts * 6:
        raise ContractError("native expert source inventory is incomplete")
    if (
        first_layer == FIRST_MOE_LAYER
        and last_layer == LAST_MAIN_LAYER
        and config.n_routed_experts == EXPERTS_PER_LAYER
    ):
        if plan.uniform_pack_bytes != 13_369_344:
            raise ContractError(
                f"native expert record size changed: {plan.uniform_pack_bytes}"
            )
        if plan.uniform_read_count != 1 or plan.read_bytes != plan.payload_bytes:
            raise ContractError("native GLM experts are no longer exact one-range records")
    return plan
