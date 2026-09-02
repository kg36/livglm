"""Resident source plan for the official GLM Layer 45 MTP block."""

from __future__ import annotations

from .contract import ContractError, ModelContract
from .resident_plan import (
    BLOCK_FP8_TO_BF16,
    COPY,
    ResidentSourcePlan,
    ResidentTensorPlan,
)


PREFIX = "model.language_model.layers.45."


def _destination(name: str) -> str:
    value = name.removeprefix(PREFIX)
    if value.startswith(("eh_proj.", "enorm.", "hnorm.", "shared_head.")):
        return value
    return "decoder." + value


def build_mtp_resident_source_plan(contract: ModelContract) -> ResidentSourcePlan:
    names = {
        name
        for name in contract.names_for("mtp")
        if ".mlp.experts." not in name
    }
    consumed: set[str] = set()
    plans: list[ResidentTensorPlan] = []
    for name in sorted(names):
        if name in consumed:
            continue
        source = contract.tensor(name)
        if source.dtype == "F8_E4M3":
            scale_name = name + "_scale_inv"
            if scale_name not in names:
                raise ContractError(f"MTP FP8 weight has no scale: {name}")
            scale = contract.tensor(scale_name)
            expected_scale = (source.shape[0] // 128, source.shape[1] // 128)
            if (
                len(source.shape) != 2
                or source.shape[0] % 128
                or source.shape[1] % 128
                or scale.dtype != "F32"
                or scale.shape != expected_scale
                or source.shard_name != scale.shard_name
            ):
                raise ContractError(f"MTP block-FP8 geometry mismatch: {name}")
            plans.append(
                ResidentTensorPlan(
                    source_names=(name, scale_name),
                    destination_name=_destination(name),
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
            raise ContractError(f"orphan MTP FP8 scale: {name}")
        if source.dtype not in {"BF16", "F32"}:
            raise ContractError(f"unsupported MTP resident dtype: {name} {source.dtype}")
        plans.append(
            ResidentTensorPlan(
                source_names=(name,),
                destination_name=_destination(name),
                sources=(source,),
                destination_dtype=source.dtype,
                destination_shape=source.shape,
                destination_bytes=source.byte_length,
                transform=COPY,
            )
        )
        consumed.add(name)
    if consumed != names:
        raise ContractError("MTP resident source inventory was not fully consumed")
    result = ResidentSourcePlan(
        model_dir=str(contract.model_dir),
        tensors=tuple(sorted(plans, key=lambda item: (item.shard_name, item.destination_name))),
        source_tensor_count=len(names),
    )
    if (result.source_tensor_count, result.runtime_tensor_count) != (32, 25):
        raise ContractError("official MTP resident tensor inventory changed")
    return result
