"""Stream native resident tensors into the deferred MLX model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import mmap
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .contract import ContractError, TensorSource
from .fp8 import dequantize_block_fp8
from .resident_plan import BLOCK_FP8_TO_BF16, COPY, ResidentSourcePlan, ResidentTensorPlan


@dataclass(frozen=True)
class ResidentReaderStats:
    source_tensors: int
    runtime_tensors: int
    source_bytes: int
    destination_bytes: int
    mapped_shards: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def bind_parameter(root: Any, dotted_name: str, value: mx.array) -> None:
    parts = dotted_name.split(".")
    target = root
    for part in parts[:-1]:
        if part.isdigit():
            target = target[int(part)]
        elif isinstance(target, dict):
            target = target[part]
        else:
            if not hasattr(target, part):
                raise ContractError(f"runtime parameter parent is absent: {dotted_name}")
            target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise ContractError(f"runtime parameter is absent: {dotted_name}")
    current = getattr(target, leaf)
    if current is not None:
        raise ContractError(f"runtime parameter was already populated: {dotted_name}")
    setattr(target, leaf, value)


class NativeResidentReader:
    def __init__(self, plan: ResidentSourcePlan):
        self.plan = plan
        self.model_dir = Path(plan.model_dir)
        self._source_tensors = 0
        self._runtime_tensors = 0
        self._source_bytes = 0
        self._destination_bytes = 0
        self._mapped_shards = 0

    @staticmethod
    def _slice(mapping: mmap.mmap, source: TensorSource) -> memoryview:
        start = source.absolute_offset
        end = start + source.byte_length
        return memoryview(mapping)[start:end]

    def _load(self, mapping: mmap.mmap, tensor: ResidentTensorPlan) -> mx.array:
        if tensor.transform == COPY:
            source = tensor.sources[0]
            raw = self._slice(mapping, source)
            if source.dtype == "BF16":
                view = np.frombuffer(raw, dtype=np.dtype("<u2")).reshape(source.shape)
                array = mx.array(view, dtype=mx.uint16).view(mx.bfloat16)
            elif source.dtype == "F32":
                view = np.frombuffer(raw, dtype=np.dtype("<f4")).reshape(source.shape)
                array = mx.array(view, dtype=mx.float32)
            else:
                raise ContractError(f"unsupported resident copy dtype: {source.dtype}")
        elif tensor.transform == BLOCK_FP8_TO_BF16:
            weight, scale = tensor.sources
            decoded = dequantize_block_fp8(
                self._slice(mapping, weight),
                weight.shape,
                self._slice(mapping, scale),
                scale.shape,
            )
            array = mx.array(decoded, dtype=mx.float32).astype(mx.bfloat16)
        else:
            raise ContractError(f"unknown resident transform: {tensor.transform}")
        if tuple(array.shape) != tensor.destination_shape:
            raise ContractError(
                f"resident destination shape mismatch: {tensor.destination_name} "
                f"{tuple(array.shape)} != {tensor.destination_shape}"
            )
        if array.nbytes != tensor.destination_bytes:
            raise ContractError(
                f"resident destination bytes changed: {tensor.destination_name} "
                f"{array.nbytes} != {tensor.destination_bytes}"
            )
        mx.eval(array)
        return array

    def load_into(self, model: Any) -> ResidentReaderStats:
        for shard_name, tensors in self.plan.by_shard():
            fd = os.open(self.model_dir / shard_name, os.O_RDONLY)
            try:
                mapping = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                self._mapped_shards += 1
                try:
                    for tensor in tensors:
                        array = self._load(mapping, tensor)
                        bind_parameter(model, tensor.destination_name, array)
                        self._source_tensors += len(tensor.sources)
                        self._runtime_tensors += 1
                        self._source_bytes += tensor.source_bytes
                        self._destination_bytes += tensor.destination_bytes
                finally:
                    mapping.close()
            finally:
                os.close(fd)
        stats = self.stats()
        if stats.source_tensors != self.plan.source_tensor_count:
            raise ContractError("resident reader did not consume the full source plan")
        if stats.runtime_tensors != self.plan.runtime_tensor_count:
            raise ContractError("resident reader did not populate the full runtime plan")
        return stats

    def stats(self) -> ResidentReaderStats:
        return ResidentReaderStats(
            source_tensors=self._source_tensors,
            runtime_tensors=self._runtime_tensors,
            source_bytes=self._source_bytes,
            destination_bytes=self._destination_bytes,
            mapped_shards=self._mapped_shards,
        )
