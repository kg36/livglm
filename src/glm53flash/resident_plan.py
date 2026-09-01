"""Resident source plan for native official-remainder tensors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from .contract import ContractError, ModelContract, TensorSource


COPY = "copy"
BLOCK_FP8_TO_BF16 = "block_fp8_to_bf16"
MXFP8_GROUP_SIZE = 32
HC_RE = re.compile(
    r"^(language_model\.layers\.\d+)\.hc_(attn|ffn)_(fn|base|scale)$"
)


def destination_name(source_name: str) -> str:
    if source_name.startswith("model.language_model."):
        name = "language_model." + source_name.removeprefix("model.language_model.")
    elif source_name == "lm_head.weight":
        name = source_name
    else:
        raise ContractError(f"unsupported resident source name: {source_name}")
    match = HC_RE.fullmatch(name)
    if match is not None:
        site = "attn_hc" if match.group(2) == "attn" else "ffn_hc"
        return f"{match.group(1)}.{site}.{match.group(3)}"
    return name


@dataclass(frozen=True)
class ResidentTensorPlan:
    source_names: tuple[str, ...]
    destination_name: str
    sources: tuple[TensorSource, ...]
    destination_dtype: str
    destination_shape: tuple[int, ...]
    destination_bytes: int
    transform: str

    @property
    def source_bytes(self) -> int:
        return sum(source.byte_length for source in self.sources)

    @property
    def shard_name(self) -> str:
        names = {source.shard_name for source in self.sources}
        if len(names) != 1:
            raise ContractError(
                f"resident transform crosses shards: {self.destination_name}"
            )
        return next(iter(names))


@dataclass(frozen=True)
class ResidentSourcePlan:
    model_dir: str
    tensors: tuple[ResidentTensorPlan, ...]
    source_tensor_count: int

    @property
    def runtime_tensor_count(self) -> int:
        return len(self.tensors)

    @property
    def source_bytes(self) -> int:
        return sum(tensor.source_bytes for tensor in self.tensors)

    @property
    def destination_bytes(self) -> int:
        return sum(tensor.destination_bytes for tensor in self.tensors)

    @staticmethod
    def _mxfp8_linear(tensor: ResidentTensorPlan) -> bool:
        name = tensor.destination_name
        return (
            len(tensor.destination_shape) == 2
            and name.endswith(".weight")
            and ".indexer." not in name
            and "embed_tokens" not in name
            and not name.endswith(".mlp.gate.weight")
        )

    @property
    def mxfp8_linear_count(self) -> int:
        return sum(self._mxfp8_linear(tensor) for tensor in self.tensors)

    @property
    def mxfp8_source_linear_bytes(self) -> int:
        return sum(
            tensor.destination_bytes
            for tensor in self.tensors
            if self._mxfp8_linear(tensor)
        )

    @property
    def mxfp8_destination_linear_bytes(self) -> int:
        total = 0
        for tensor in self.tensors:
            if not self._mxfp8_linear(tensor):
                continue
            elements = tensor.destination_shape[0] * tensor.destination_shape[1]
            if elements % MXFP8_GROUP_SIZE:
                raise ContractError(
                    f"resident MXFP8 group geometry changed: {tensor.destination_name}"
                )
            total += elements + elements // MXFP8_GROUP_SIZE
        return total

    @property
    def mxfp4_destination_linear_bytes(self) -> int:
        total = 0
        for tensor in self.tensors:
            if not self._mxfp8_linear(tensor):
                continue
            elements = tensor.destination_shape[0] * tensor.destination_shape[1]
            if elements % MXFP8_GROUP_SIZE:
                raise ContractError(
                    f"resident MXFP4 group geometry changed: {tensor.destination_name}"
                )
            total += elements // 2 + elements // MXFP8_GROUP_SIZE
        return total

    @property
    def mxfp8_runtime_bytes(self) -> int:
        return (
            self.destination_bytes
            - self.mxfp8_source_linear_bytes
            + self.mxfp8_destination_linear_bytes
        )

    @property
    def mxfp4_runtime_bytes(self) -> int:
        return (
            self.destination_bytes
            - self.mxfp8_source_linear_bytes
            + self.mxfp4_destination_linear_bytes
        )

    @property
    def shard_count(self) -> int:
        return len({tensor.shard_name for tensor in self.tensors})

    def by_shard(self) -> tuple[tuple[str, tuple[ResidentTensorPlan, ...]], ...]:
        grouped: dict[str, list[ResidentTensorPlan]] = {}
        for tensor in self.tensors:
            grouped.setdefault(tensor.shard_name, []).append(tensor)
        return tuple(
            (shard, tuple(sorted(values, key=lambda item: item.destination_name)))
            for shard, values in sorted(grouped.items())
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_kind": "livseek_glm_native_safetensors",
            "model_dir": self.model_dir,
            "source_tensor_count": self.source_tensor_count,
            "runtime_tensor_count": self.runtime_tensor_count,
            "source_bytes": self.source_bytes,
            "destination_bytes": self.destination_bytes,
            "mxfp8_runtime_bytes": self.mxfp8_runtime_bytes,
            "mxfp4_runtime_bytes": self.mxfp4_runtime_bytes,
            "mxfp8_linear_count": self.mxfp8_linear_count,
            "shard_count": self.shard_count,
            "transforms": {
                COPY: sum(tensor.transform == COPY for tensor in self.tensors),
                BLOCK_FP8_TO_BF16: sum(
                    tensor.transform == BLOCK_FP8_TO_BF16 for tensor in self.tensors
                ),
            },
        }


def build_resident_source_plan(contract: ModelContract) -> ResidentSourcePlan:
    resident_names = set(contract.names_for("resident"))
    consumed: set[str] = set()
    plans: list[ResidentTensorPlan] = []
    for name in sorted(resident_names):
        if name in consumed:
            continue
        source = contract.tensor(name)
        if source.dtype == "F8_E4M3":
            if not name.endswith(".weight") or len(source.shape) != 2:
                raise ContractError(f"unexpected resident F8 tensor: {name}")
            scale_name = name + "_scale_inv"
            if scale_name not in resident_names:
                raise ContractError(f"resident FP8 weight has no scale: {name}")
            scale = contract.tensor(scale_name)
            expected_scale_shape = (
                source.shape[0] // 128,
                source.shape[1] // 128,
            )
            if (
                source.shape[0] % 128
                or source.shape[1] % 128
                or scale.dtype != "F32"
                or scale.shape != expected_scale_shape
            ):
                raise ContractError(
                    f"resident block-FP8 geometry mismatch: {name}, "
                    f"weight={source.shape}, scale={scale.shape}"
                )
            if source.shard_name != scale.shard_name:
                raise ContractError(f"resident FP8 pair crosses shards: {name}")
            plans.append(
                ResidentTensorPlan(
                    source_names=(name, scale_name),
                    destination_name=destination_name(name),
                    sources=(source, scale),
                    destination_dtype="BF16",
                    destination_shape=source.shape,
                    destination_bytes=source.byte_length * 2,
                    transform=BLOCK_FP8_TO_BF16,
                )
            )
            consumed.update((name, scale_name))
            continue
        if name.endswith(".weight_scale_inv"):
            raise ContractError(f"orphan resident FP8 scale: {name}")
        if source.dtype not in {"BF16", "F32"}:
            raise ContractError(f"unsupported resident tensor dtype: {name} {source.dtype}")
        plans.append(
            ResidentTensorPlan(
                source_names=(name,),
                destination_name=destination_name(name),
                sources=(source,),
                destination_dtype=source.dtype,
                destination_shape=source.shape,
                destination_bytes=source.byte_length,
                transform=COPY,
            )
        )
        consumed.add(name)
    if consumed != resident_names:
        raise ContractError(
            "resident source plan did not consume its inventory: "
            f"missing={sorted(resident_names - consumed)[:5]}"
        )
    destinations = [tensor.destination_name for tensor in plans]
    if len(set(destinations)) != len(destinations):
        raise ContractError("resident destination mapping contains a collision")
    plan = ResidentSourcePlan(
        model_dir=str(contract.model_dir),
        tensors=tuple(sorted(plans, key=lambda item: (item.shard_name, item.destination_name))),
        source_tensor_count=len(resident_names),
    )
    if len(resident_names) == 1_425:
        if plan.runtime_tensor_count != 1_246:
            raise ContractError(
                f"resident runtime tensor count changed: {plan.runtime_tensor_count}"
            )
        if plan.destination_bytes != 17_842_600_184:
            raise ContractError(
                f"resident destination byte budget changed: {plan.destination_bytes}"
            )
        expected_mxfp8 = (497, 16_230_907_904, 8_369_061_888, 9_980_754_168)
        actual_mxfp8 = (
            plan.mxfp8_linear_count,
            plan.mxfp8_source_linear_bytes,
            plan.mxfp8_destination_linear_bytes,
            plan.mxfp8_runtime_bytes,
        )
        if actual_mxfp8 != expected_mxfp8:
            raise ContractError(
                f"resident MXFP8 budget changed: {actual_mxfp8} != {expected_mxfp8}"
            )
        expected_mxfp4 = (4_311_334_912, 5_923_027_192)
        actual_mxfp4 = (
            plan.mxfp4_destination_linear_bytes,
            plan.mxfp4_runtime_bytes,
        )
        if actual_mxfp4 != expected_mxfp4:
            raise ContractError(
                f"resident MXFP4 budget changed: {actual_mxfp4} != {expected_mxfp4}"
            )
    return plan
