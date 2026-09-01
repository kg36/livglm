"""Synchronous exact-range reader for native GLM MXFP4 experts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import threading

import mlx.core as mx
import numpy as np

from .contract import ContractError
from .expert_source import NativeExpertSourcePlan


PROJECTIONS = ("down_proj", "gate_proj", "up_proj")


@dataclass(frozen=True)
class ReaderStats:
    expert_loads: int
    logical_reads: int
    system_reads: int
    read_bytes: int
    open_shards: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedExpert:
    layer: int
    expert: int
    tensors: dict[str, mx.array]
    logical_reads: int
    system_reads: int
    read_bytes: int

    def projection(self, name: str) -> tuple[mx.array, mx.array]:
        if name not in PROJECTIONS:
            raise ContractError(f"unknown routed projection: {name}")
        return self.tensors[f"{name}.weight"], self.tensors[f"{name}.scales"]

    @property
    def payload_bytes(self) -> int:
        return sum(value.nbytes for value in self.tensors.values())


class NativeExpertReader:
    def __init__(self, plan: NativeExpertSourcePlan):
        self.plan = plan
        self.model_dir = Path(plan.model_dir)
        self._fds: dict[str, int] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._expert_loads = 0
        self._logical_reads = 0
        self._system_reads = 0
        self._read_bytes = 0

    def __enter__(self) -> "NativeExpertReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _fd(self, shard_name: str) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("native expert reader is closed")
            fd = self._fds.get(shard_name)
            if fd is None:
                fd = os.open(self.model_dir / shard_name, os.O_RDONLY)
                self._fds[shard_name] = fd
            return fd

    def _pread_exact(self, shard_name: str, offset: int, length: int) -> bytes:
        fd = self._fd(shard_name)
        chunks: list[bytes] = []
        remaining = length
        cursor = offset
        self._logical_reads += 1
        while remaining:
            block = os.pread(fd, remaining, cursor)
            self._system_reads += 1
            if not block:
                raise ContractError(
                    f"short native expert read from {shard_name}: "
                    f"offset={offset}, expected={length}, received={length - remaining}"
                )
            chunks.append(block)
            cursor += len(block)
            remaining -= len(block)
            self._read_bytes += len(block)
        return chunks[0] if len(chunks) == 1 else b"".join(chunks)

    @staticmethod
    def _array(raw: memoryview, dtype: str, shape: tuple[int, ...]) -> mx.array:
        elements = 1
        for dimension in shape:
            elements *= dimension
        if dtype == "U32":
            expected = elements * 4
            if len(raw) != expected:
                raise ContractError(f"U32 expert view byte mismatch: {len(raw)} != {expected}")
            source = np.frombuffer(raw, dtype=np.dtype("<u4")).reshape(shape)
            return mx.array(source, dtype=mx.uint32)
        if dtype == "U8":
            expected = elements
            if len(raw) != expected:
                raise ContractError(f"U8 expert view byte mismatch: {len(raw)} != {expected}")
            source = np.frombuffer(raw, dtype=np.dtype("u1")).reshape(shape)
            return mx.array(source, dtype=mx.uint8)
        raise ContractError(f"unsupported native expert MLX dtype: {dtype}")

    def load(self, layer: int, expert: int) -> LoadedExpert:
        source = self.plan.expert(layer, expert)
        before_logical = self._logical_reads
        before_system = self._system_reads
        before_bytes = self._read_bytes
        tensors: dict[str, mx.array] = {}
        for source_range in source.read_ranges:
            raw = self._pread_exact(
                source_range.shard_name,
                source_range.absolute_offset,
                source_range.byte_length,
            )
            view = memoryview(raw)
            for tensor in source_range.tensors:
                relative = source_range.relative_offset(tensor)
                tensor_raw = view[relative : relative + tensor.byte_length]
                tensors[tensor.destination_name] = self._array(
                    tensor_raw,
                    tensor.mlx_dtype,
                    tensor.mlx_shape,
                )
        expected = {
            f"{projection}.{part}"
            for projection in PROJECTIONS
            for part in ("weight", "scales")
        }
        if set(tensors) != expected:
            raise ContractError(
                "native expert load produced the wrong tensor set: "
                f"missing={sorted(expected - set(tensors))}, "
                f"extra={sorted(set(tensors) - expected)}"
            )
        mx.eval(*tensors.values())
        self._expert_loads += 1
        return LoadedExpert(
            layer=layer,
            expert=expert,
            tensors=tensors,
            logical_reads=self._logical_reads - before_logical,
            system_reads=self._system_reads - before_system,
            read_bytes=self._read_bytes - before_bytes,
        )

    def stats(self) -> ReaderStats:
        return ReaderStats(
            expert_loads=self._expert_loads,
            logical_reads=self._logical_reads,
            system_reads=self._system_reads,
            read_bytes=self._read_bytes,
            open_shards=len(self._fds),
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            fds = tuple(self._fds.values())
            self._fds.clear()
        for fd in fds:
            os.close(fd)
