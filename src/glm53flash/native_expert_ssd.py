"""Native fixed-slot ExpertSSD data plane for GLM routed experts."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn

from .contract import ContractError
from .expert_reader import ReaderStats
from .expert_source import NativeExpertSourcePlan
from .expert_ssd import ExpertCacheStats, limited_swiglu
from .trace import DecodeTrace, active_trace


_DTYPES = {
    "U8": mx.uint8,
    "U32": mx.uint32,
}

_REQUIRED_NATIVE_SYMBOLS = (
    "_open_expert_safetensors_direct",
    "_expert_safetensors_direct_read_range_count",
    "_expert_ssd_direct_load_into",
    "_expert_ssd_route_plan",
    "_expert_ssd_markov_state_new",
    "_expert_ssd_route_cache_state_new",
    "_expert_ssd_route_cache_plan",
    "_expert_ssd_mxfp4_pair_qmv",
    "_expert_ssd_mxfp4_masked_qmv",
    "_expert_ssd_wire_arrays",
    "_expert_ssd_unwire_arrays",
)

_REQUIRED_SCALEX_SYMBOLS = (
    "_open_scalex_mode_a_direct",
    "_scalex_mode_b_load_experts_into_many",
    "_expert_ssd_scalex_mxfp4_qmv",
    "_expert_ssd_scalex_mxfp4_qmv_split_routes",
    "_expert_ssd_scalex_mxfp4_width2_pair_qmv",
    "_expert_ssd_scalex_mxfp4_width2_down_reduce",
)


def native_expert_ssd_available() -> bool:
    return all(hasattr(mx, name) for name in _REQUIRED_NATIVE_SYMBOLS)


def require_native_expert_ssd() -> None:
    missing = [name for name in _REQUIRED_NATIVE_SYMBOLS if not hasattr(mx, name)]
    if missing:
        raise ContractError(
            "the mlx-io-glm overlay is required for native ExpertSSD; "
            f"missing symbols: {', '.join(missing)}. Run ./build.sh."
        )


@dataclass(frozen=True)
class NativeRoutePlan:
    remapped: mx.array
    proposed_mapping: OrderedDict[int, int]
    miss_experts: tuple[int, ...]
    futures: tuple[Future[float], ...]
    hits: int
    misses: int
    evictions: int


def _run_traced_load(
    trace: DecodeTrace,
    flow_id: int,
    source: "NativeExpertLayerSource",
    expert: int,
    slot: int,
    destinations: tuple[mx.array, ...],
    args: dict[str, int | str],
) -> float:
    trace.flow("t", flow_id, args=args)
    try:
        with trace.span("SSD_read", category="ssd_worker", args=args, force=True):
            return source.load_into(expert, slot, destinations)
    finally:
        trace.flow("f", flow_id, args=args)


class NativeExpertPool:
    """Shared descriptors, I/O workers, and counters for all MoE layers."""

    def __init__(
        self,
        plan: NativeExpertSourcePlan,
        *,
        workers: int = 8,
        no_file_cache: bool = False,
        read_ahead: bool = True,
    ):
        require_native_expert_ssd()
        if workers < 1:
            raise ContractError("native ExpertSSD workers must be positive")
        self.plan = plan
        self.workers = workers
        self.no_file_cache = no_file_cache
        self.read_ahead = read_ahead
        if plan.scalex_mode_b:
            missing = [name for name in _REQUIRED_SCALEX_SYMBOLS if not hasattr(mx, name)]
            if missing:
                raise ContractError(
                    "the mlx-io-glm overlay lacks ScaleX Mode B symbols: "
                    + ", ".join(missing)
                )
        self.backend = (
            "mlx-io-glm/direct-to-slot/ScaleX-Mode-B"
            if plan.scalex_mode_b
            else "mlx-io-glm/direct-to-slot"
        )
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="glm-expert-ssd",
        )
        self._sources: dict[int, NativeExpertLayerSource] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._expert_loads = 0
        self._logical_reads = 0
        self._system_reads = 0
        self._read_bytes = 0
        self._read_seconds = 0.0
        self._io_wait_seconds = 0.0
        self._wired_bytes = 0

    def layer(self, layer: int) -> "NativeExpertLayerSource":
        if self._closed:
            raise RuntimeError("native ExpertSSD pool is closed")
        source = self._sources.get(layer)
        if source is None:
            source = NativeExpertLayerSource(self, layer=layer)
            self._sources[layer] = source
        return source

    def submit(
        self,
        source: "NativeExpertLayerSource",
        expert: int,
        slot: int,
        destinations: tuple[mx.array, ...],
    ) -> Future[float]:
        if self._closed:
            raise RuntimeError("native ExpertSSD pool is closed")
        trace = active_trace()
        if trace is not None:
            flow_id = trace.new_flow()
            args: dict[str, int | str] = {
                "decode_index": int(trace.current_decode_index or 0),
                "layer": source.layer,
                "expert": int(expert),
                "destination_slot": int(slot),
                "read_ranges": source.read_range_counts[int(expert)],
                "payload_bytes": source.payload_bytes[int(expert)],
                "shard": source.shard_name,
                "backend": self.backend,
            }
            trace.flow("s", flow_id, args=args)
            return self._executor.submit(
                _run_traced_load,
                trace,
                flow_id,
                source,
                expert,
                slot,
                destinations,
                args,
            )
        return self._executor.submit(source.load_into, expert, slot, destinations)

    def record_loads(
        self,
        source: "NativeExpertLayerSource",
        experts: Iterable[int],
        worker_seconds: Iterable[float],
        io_wait_seconds: float,
    ) -> None:
        expert_ids = tuple(int(value) for value in experts)
        seconds = tuple(float(value) for value in worker_seconds)
        if len(expert_ids) != len(seconds):
            raise RuntimeError("native ExpertSSD accounting lost a worker result")
        with self._lock:
            self._expert_loads += len(expert_ids)
            self._logical_reads += sum(source.read_range_counts[item] for item in expert_ids)
            # The native path normally issues one preadv per coalesced range.
            # A rare short-read recovery is intentionally not observable here.
            self._system_reads += sum(source.read_range_counts[item] for item in expert_ids)
            self._read_bytes += sum(source.payload_bytes[item] for item in expert_ids)
            self._read_seconds += sum(seconds)
            self._io_wait_seconds += float(io_wait_seconds)

    def add_wired_bytes(self, value: int) -> None:
        with self._lock:
            self._wired_bytes += int(value)

    def stats(self) -> ReaderStats:
        with self._lock:
            return ReaderStats(
                expert_loads=self._expert_loads,
                logical_reads=self._logical_reads,
                system_reads=self._system_reads,
                read_bytes=self._read_bytes,
                open_shards=len(self._sources),
                backend=self.backend,
                direct_to_slot=True,
                read_seconds=self._read_seconds,
                io_wait_seconds=self._io_wait_seconds,
                wired_bytes=self._wired_bytes,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)


class NativeExpertLayerSource:
    """One native shard handle and canonical tensor layout for a GLM layer."""

    def __init__(self, pool: NativeExpertPool, *, layer: int):
        self.pool = pool
        self.plan = pool.plan
        self.layer = layer
        self.scalex_mode_b = self.plan.scalex_mode_b
        sources = tuple(
            self.plan.expert(layer, expert)
            for expert in range(self.plan.experts_per_layer)
        )
        shard_names = {
            tensor.shard_name
            for source in sources
            for tensor in source.tensors
        }
        if len(shard_names) != 1:
            raise ContractError(
                f"native ExpertSSD layer {layer} spans shards: {sorted(shard_names)}"
            )
        if any(len(source.read_ranges) != 1 for source in sources):
            raise ContractError(
                f"native ExpertSSD layer {layer} is not one contiguous read per expert"
            )
        self.shard_name = next(iter(shard_names))
        shard_path = Path(self.plan.model_dir) / self.shard_name
        canonical = tuple(
            (tensor.destination_name, tensor.mlx_dtype, tensor.mlx_shape)
            for tensor in sources[0].tensors
        )
        for source in sources[1:]:
            layout = tuple(
                (tensor.destination_name, tensor.mlx_dtype, tensor.mlx_shape)
                for tensor in source.tensors
            )
            if layout != canonical:
                raise ContractError(
                    f"native ExpertSSD tensor layout changes within layer {layer}"
                )
        self.scalex_handle = None
        if self.scalex_mode_b:
            records = tuple(source.scalex_record for source in sources)
            if any(record is None for record in records):
                raise ContractError(f"ScaleX layer {layer} has a native expert")
            first_record = records[0]
            assert first_record is not None
            canonical_by_source = {
                tensor.source_name: tensor for tensor in sources[0].tensors
            }
            scale_tensors = tuple(
                canonical_by_source[name] for name in first_record.scale_names
            )
            weight_tensors = tuple(
                canonical_by_source[name] for name in first_record.weight_names
            )
            if tuple(tensor.byte_length for tensor in scale_tensors) != first_record.scale_nbytes:
                raise ContractError(f"ScaleX decoded-scale geometry changed in layer {layer}")
            if tuple(tensor.byte_length for tensor in weight_tensors) != first_record.weight_nbytes:
                raise ContractError(f"ScaleX weight geometry changed in layer {layer}")
            record_stride = max(
                record.mode_b_row_bytes for record in records if record is not None
            )
            self.tensor_layouts = (
                ("scalex_record", mx.uint8, (record_stride,)),
                *(
                    (
                        tensor.destination_name,
                        _DTYPES[tensor.mlx_dtype],
                        tuple(tensor.mlx_shape),
                    )
                    for tensor in weight_tensors
                ),
            )
            self.handle = None
            self.scalex_handle = mx._open_scalex_mode_a_direct(
                shard_path,
                [
                    [record.absolute_offset, record.encoded_bytes]
                    for record in records
                    if record is not None
                ],
                list(first_record.scale_nbytes),
                no_cache=pool.no_file_cache,
                read_ahead=pool.read_ahead,
            )
            self.read_range_counts = (1,) * self.plan.experts_per_layer
        else:
            self.tensor_layouts = tuple(
                (name, _DTYPES[dtype], tuple(shape))
                for name, dtype, shape in canonical
            )
            raw_specs = [
                [
                    (
                        tensor.destination_name,
                        tensor.mlx_dtype,
                        list(tensor.mlx_shape),
                        tensor.absolute_offset,
                    )
                    for tensor in source.tensors
                ]
                for source in sources
            ]
            self.handle = mx._open_expert_safetensors_direct(
                shard_path,
                raw_specs,
                no_cache=pool.no_file_cache,
                read_ahead=pool.read_ahead,
            )
            self.read_range_counts = tuple(
                int(mx._expert_safetensors_direct_read_range_count(self.handle, expert))
                for expert in range(self.plan.experts_per_layer)
            )
            expected = tuple(len(source.read_ranges) for source in sources)
            if self.read_range_counts != expected:
                raise ContractError(
                    f"native ExpertSSD range plan differs in layer {layer}: "
                    f"{self.read_range_counts} != {expected}"
                )
        self.payload_bytes = tuple(source.read_bytes for source in sources)

    def allocate_slots(self, capacity: int) -> tuple[mx.array, ...]:
        if not 1 <= capacity <= self.plan.experts_per_layer:
            raise ContractError(
                f"native ExpertSSD capacity must be within "
                f"1..{self.plan.experts_per_layer}: {capacity}"
            )
        arrays = tuple(
            mx.zeros((capacity, *shape), dtype=dtype)
            for _, dtype, shape in self.tensor_layouts
        )
        mx.eval(*arrays)
        return arrays

    def load_into(
        self,
        expert: int,
        slot: int,
        destinations: tuple[mx.array, ...],
    ) -> float:
        started = time.perf_counter()
        if self.scalex_mode_b:
            mx._scalex_mode_b_load_experts_into_many(
                self.scalex_handle,
                [int(expert)],
                [int(slot)],
                destinations[0],
                list(destinations[1:]),
            )
        else:
            mx._expert_ssd_direct_load_into(
                self.handle,
                int(expert),
                int(slot),
                list(destinations),
            )
        return time.perf_counter() - started


class NativeExpertSSD(nn.Module):
    """Fixed native slot bank with parallel direct SSD refill and LRU policy."""

    def __init__(
        self,
        pool: NativeExpertPool,
        *,
        layer: int,
        capacity: int,
        swiglu_limit: float,
        wire_slots: bool = True,
        defer_slots: bool = False,
    ):
        super().__init__()
        if capacity < 1:
            raise ContractError("native ExpertSSD capacity must be positive")
        self.pool = pool
        self.source = pool.layer(layer)
        self.layer = layer
        self.capacity = capacity
        self.swiglu_limit = swiglu_limit
        self._wire_slots = wire_slots
        self.slots: tuple[mx.array, ...] = ()
        self.wired_bytes = 0
        self._active = False
        self._expert_to_slot: OrderedDict[int, int] = OrderedDict()
        self._reset_policy()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._route_sync_seconds = 0.0
        self._slot_plan_seconds = 0.0
        self._closed = False
        self.last_expert_ids: tuple[int, ...] = ()
        if not defer_slots:
            self.activate()

    def activate(self) -> None:
        if self._closed:
            raise RuntimeError("native ExpertSSD layer is closed")
        if self._active:
            return
        self.slots = self.source.allocate_slots(self.capacity)
        self.wired_bytes = (
            int(mx._expert_ssd_wire_arrays(list(self.slots)))
            if self._wire_slots
            else 0
        )
        self.pool.add_wired_bytes(self.wired_bytes)
        self._active = True

    def _reset_policy(self) -> None:
        self._markov_state = mx._expert_ssd_markov_state_new(
            self.pool.plan.experts_per_layer,
            3,
        )
        self._route_cache_state = mx._expert_ssd_route_cache_state_new(
            self.pool.plan.experts_per_layer,
            self.capacity,
            0.90,
            0.25,
            4096.0,
            0.25,
            300.0,
            self._markov_state,
        )

    def prepare(self, indices: mx.array) -> NativeRoutePlan:
        if self._closed:
            raise RuntimeError("native ExpertSSD layer is closed")
        if not self._active:
            raise RuntimeError("native ExpertSSD slots were not activated")
        if indices.ndim < 1 or indices.shape[-1] < 1:
            raise ContractError("routed expert indices must have a top-k dimension")
        trace = active_trace()
        route_args = {
            "decode_index": trace.current_decode_index if trace is not None else None,
            "layer": self.layer,
            "semantics": (
                "host waits for routed indices and all upstream Metal work; "
                "actual kernels are in the paired Metal System Trace"
            ),
        }
        route_started = time.perf_counter()
        if trace is None:
            raw = mx._expert_ssd_route_cache_plan(self._route_cache_state, indices)
        else:
            with trace.span(
                "materialize_route",
                category="gpu_sync",
                args=route_args,
            ) as event_args:
                raw = mx._expert_ssd_route_cache_plan(self._route_cache_state, indices)
                event_args["routed_positions"] = len(raw["routed"])
                event_args["unique_experts"] = len(raw["unique"])
        route_seconds = time.perf_counter() - route_started
        slot_started = time.perf_counter()
        with (
            trace.span(
                "slot_plan",
                category="cache",
                args={
                    "decode_index": trace.current_decode_index,
                    "layer": self.layer,
                    "capacity": self.capacity,
                },
            )
            if trace is not None
            else nullcontext({})
        ) as slot_args:
            unique = tuple(int(value) for value in raw["unique"])
            self.last_expert_ids = tuple(int(value) for value in raw["routed"])
            if len(unique) > self.capacity:
                raise ContractError(
                    f"layer {self.layer} routed {len(unique)} unique experts, "
                    f"exceeding native ExpertSSD capacity {self.capacity}"
                )
            invalid = [
                expert
                for expert in unique
                if not 0 <= expert < self.pool.plan.experts_per_layer
            ]
            if invalid:
                raise ContractError(f"routed expert ids are out of range: {invalid}")

            proposed = self._expert_to_slot.copy()
            for expert in raw["evicted"]:
                proposed.pop(int(expert), None)
            gate_rows = tuple(int(value) for value in raw["gate_up_rows"])
            for expert, slot in zip(unique, gate_rows, strict=True):
                proposed.pop(expert, None)
                proposed[expert] = slot
            miss_experts = tuple(int(value) for value in raw["missing"])
            miss_slots = tuple(int(value) for value in raw["missing_gate_up_rows"])
            compact_slots = mx.array(gate_rows, dtype=mx.uint32)
            remapped = mx.take(compact_slots, raw["compact"])
            slot_args.update(
                {
                    "requested_experts": unique,
                    "missing_experts": miss_experts,
                    "hit_count": int(raw["hits"]),
                    "missing_count": int(raw["misses"]),
                    "eviction_count": len(raw["evicted"]),
                }
            )
        slot_seconds = time.perf_counter() - slot_started

        if trace is None:
            futures = tuple(
                self.pool.submit(self.source, expert, slot, self.slots)
                for expert, slot in zip(miss_experts, miss_slots, strict=True)
            )
        else:
            with trace.span(
                "issue_reads",
                category="ssd_issue",
                args={
                    "decode_index": trace.current_decode_index,
                    "layer": self.layer,
                    "experts": miss_experts,
                    "read_count": len(miss_experts),
                    "payload_bytes": sum(
                        self.source.payload_bytes[expert] for expert in miss_experts
                    ),
                },
            ):
                futures = tuple(
                    self.pool.submit(self.source, expert, slot, self.slots)
                    for expert, slot in zip(miss_experts, miss_slots, strict=True)
                )
        self._route_sync_seconds += route_seconds
        self._slot_plan_seconds += slot_seconds
        return NativeRoutePlan(
            remapped=remapped,
            proposed_mapping=proposed,
            miss_experts=miss_experts,
            futures=futures,
            hits=int(raw["hits"]),
            misses=int(raw["misses"]),
            evictions=len(raw["evicted"]),
        )

    def _finish_refill(self, plan: NativeRoutePlan) -> None:
        wait_started = time.perf_counter()
        trace = active_trace()
        if trace is None:
            worker_seconds = tuple(future.result() for future in plan.futures)
        else:
            with trace.span(
                "join_reads",
                category="ssd_join",
                args={
                    "decode_index": trace.current_decode_index,
                    "layer": self.layer,
                    "experts": plan.miss_experts,
                    "read_count": len(plan.futures),
                    "payload_bytes": sum(
                        self.source.payload_bytes[expert]
                        for expert in plan.miss_experts
                    ),
                },
            ):
                worker_seconds = tuple(future.result() for future in plan.futures)
        wait_seconds = time.perf_counter() - wait_started
        self.pool.record_loads(
            self.source,
            plan.miss_experts,
            worker_seconds,
            wait_seconds,
        )
        self._expert_to_slot = plan.proposed_mapping
        self._hits += plan.hits
        self._misses += plan.misses
        self._evictions += plan.evictions

    def finish(self, x: mx.array, plan: NativeRoutePlan) -> mx.array:
        self._finish_refill(plan)
        by_name = {
            name: self.slots[index]
            for index, (name, _, _) in enumerate(self.source.tensor_layouts)
        }

        fixed_qmv_geometry = (
            x.shape[-1] % 512 == 0
            and by_name["up_proj.weight"].shape[-2] % 8 == 0
        )
        production_native_qmv = (
            fixed_qmv_geometry
            and x.dtype == mx.bfloat16
            and x.size == x.shape[-1]
        )
        scalex_qmv = (
            self.source.scalex_mode_b
            and fixed_qmv_geometry
            and x.dtype in (mx.bfloat16, mx.float32)
        )
        trace = active_trace()
        with (
            trace.span(
                "routed_expert_graph_construct",
                category="mlx_submit",
                args={
                    "decode_index": trace.current_decode_index,
                    "layer": self.layer,
                    "routes": int(plan.remapped.size),
                    "misses": plan.misses,
                    "primitive": (
                        "scalex_mxfp4_qmv"
                        if scalex_qmv and production_native_qmv
                        else "scalex_mxfp4_qmv_serial_prefill"
                        if scalex_qmv
                        else "native_mxfp4_qmv"
                        if production_native_qmv
                        else "gather_qmm"
                    ),
                    "semantics": (
                        "Python/native graph construction, not GPU kernel time"
                    ),
                },
            )
            if trace is not None
            else nullcontext({})
        ):
            if scalex_qmv:
                route_width = plan.remapped.shape[-1]
                # GLM's residual stream is FP32. Keep the conversion lazy in
                # the MLX graph; the fixed ScaleX Metal kernel consumes BF16,
                # matching the checkpoint's effective expert precision.
                flat_x = x.astype(mx.bfloat16).reshape(-1, x.shape[-1])
                flat_routes = plan.remapped.reshape(-1, route_width).astype(mx.uint32)
                if flat_x.shape[0] != flat_routes.shape[0]:
                    raise ContractError(
                        "ScaleX token and routed-index counts differ during prefill"
                    )
                token_outputs = []
                for token in range(flat_x.shape[0]):
                    token_x = flat_x[token : token + 1]
                    routes = flat_routes[token]
                    gate = mx._expert_ssd_scalex_mxfp4_qmv(
                        token_x,
                        by_name["gate_proj.weight"],
                        by_name["scalex_record"],
                        routes,
                        0,
                    )
                    up = mx._expert_ssd_scalex_mxfp4_qmv(
                        token_x,
                        by_name["up_proj.weight"],
                        by_name["scalex_record"],
                        routes,
                        2,
                    )
                    activated = limited_swiglu(gate, up, self.swiglu_limit)
                    output = mx._expert_ssd_scalex_mxfp4_qmv(
                        activated,
                        by_name["down_proj.weight"],
                        by_name["scalex_record"],
                        routes,
                        1,
                    )
                    token_outputs.append(output.reshape(route_width, x.shape[-1]))
                return mx.stack(token_outputs, axis=0).reshape(
                    *plan.remapped.shape,
                    x.shape[-1],
                )

            if production_native_qmv:
                routes = plan.remapped.reshape(-1).astype(mx.uint32)
                up, gate = mx._expert_ssd_mxfp4_pair_qmv(
                    x.reshape(1, x.shape[-1]),
                    by_name["up_proj.weight"],
                    by_name["up_proj.scales"],
                    by_name["gate_proj.weight"],
                    by_name["gate_proj.scales"],
                    routes,
                )
                activated = limited_swiglu(gate, up, self.swiglu_limit)
                output = mx._expert_ssd_mxfp4_masked_qmv(
                    activated,
                    by_name["down_proj.weight"],
                    by_name["down_proj.scales"],
                    routes,
                )
                return output.reshape(
                    *plan.remapped.shape,
                    1,
                    x.shape[-1],
                ).squeeze(-2)

            if self.source.scalex_mode_b:
                raise ContractError(
                    "ScaleX Mode B received unsupported QMV geometry: "
                    f"dtype={x.dtype}, shape={tuple(x.shape)}, "
                    f"up_weight={tuple(by_name['up_proj.weight'].shape)}"
                )

            expanded = mx.expand_dims(x, (-2, -3))

            def qmm(value: mx.array, projection: str) -> mx.array:
                return mx.gather_qmm(
                    value,
                    by_name[f"{projection}.weight"],
                    by_name[f"{projection}.scales"],
                    rhs_indices=plan.remapped,
                    transpose=True,
                    group_size=32,
                    bits=4,
                    mode="mxfp4",
                )

            gate = qmm(expanded, "gate_proj")
            up = qmm(expanded, "up_proj")
            activated = limited_swiglu(gate, up, self.swiglu_limit)
            return qmm(activated, "down_proj").squeeze(-2)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        return self.finish(x, self.prepare(indices))

    def finish_width2_merged(
        self,
        x: mx.array,
        plan: NativeRoutePlan,
        scores: mx.array,
        shared: mx.array,
    ) -> mx.array:
        """Top-8 ScaleX width-two down projection and mixture reduction."""

        if (
            not self.source.scalex_mode_b
            or x.shape != (2, 4096)
            or plan.remapped.shape != (2, 8)
        ):
            raise ContractError("GLM fused verifier requires width-two/top-eight ScaleX")
        self._finish_refill(plan)
        by_name = {
            name: self.slots[index]
            for index, (name, _, _) in enumerate(self.source.tensor_layouts)
        }
        flat_x = x.astype(mx.bfloat16)
        flat_routes = plan.remapped.reshape(2, 8).astype(mx.uint32)
        up, gate = mx._expert_ssd_scalex_mxfp4_width2_pair_qmv(
            mx.contiguous(flat_x),
            by_name["up_proj.weight"],
            by_name["gate_proj.weight"],
            by_name["scalex_record"],
            mx.contiguous(flat_routes.reshape(-1)),
        )
        activated = limited_swiglu(gate, up, self.swiglu_limit)
        return mx._expert_ssd_scalex_mxfp4_width2_down_reduce(
            mx.contiguous(activated),
            by_name["down_proj.weight"],
            by_name["scalex_record"],
            flat_routes.reshape(-1),
            flat_routes.reshape(-1),
            mx.contiguous(scores.reshape(-1).astype(mx.float32)),
            mx.contiguous(shared.reshape(2, 1, 4096).astype(mx.bfloat16)),
        ).squeeze(1)

    def stats(self) -> ExpertCacheStats:
        return ExpertCacheStats(
            layer=self.layer,
            capacity=self.capacity,
            resident=len(self._expert_to_slot),
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            route_sync_seconds=self._route_sync_seconds,
            slot_plan_seconds=self._slot_plan_seconds,
            policy="native-markov-lhd",
        )

    @property
    def resident_experts(self) -> tuple[int, ...]:
        return tuple(self._expert_to_slot)

    def clear(self) -> None:
        self._expert_to_slot.clear()
        self._reset_policy()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.wired_bytes:
            mx._expert_ssd_unwire_arrays(list(self.slots))
        self._active = False
