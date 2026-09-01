"""Low-overhead, opt-in Perfetto/Chrome tracing for GLM decode.

The JSON trace records application-level CPU work and synchronization.  It
does not claim that MLX graph-construction spans are GPU execution time; real
kernel occupancy belongs in the paired Instruments Metal System Trace.
"""

from __future__ import annotations

from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterator


_ACTIVE_TRACE: DecodeTrace | None = None
_ACTIVE_TRACE_LOCK = threading.Lock()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return repr(value)


def active_trace() -> "DecodeTrace | None":
    """Return the trace while a selected decode step is recording."""

    trace = _ACTIVE_TRACE
    return trace if trace is not None and trace.recording else None


class DecodeTrace:
    """Thread-safe Perfetto JSON recorder for selected decode forwards."""

    def __init__(
        self,
        path: str | Path,
        *,
        decode_start: int = 0,
        decode_steps: int = 20,
        metadata: dict[str, Any] | None = None,
    ):
        if decode_start < 0:
            raise ValueError("trace decode_start must be non-negative")
        if decode_steps <= 0:
            raise ValueError("trace decode_steps must be positive")
        self.path = Path(path).expanduser().resolve()
        self.decode_start = int(decode_start)
        self.decode_stop = self.decode_start + int(decode_steps)
        self.metadata = dict(metadata or {})
        self.origin_monotonic_ns = time.perf_counter_ns()
        self.origin_wall_ns = time.time_ns()
        self.pid = os.getpid()
        self.recording = False
        self.current_decode_index: int | None = None
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._flow_ids = itertools.count(1)
        self._named_threads: set[int] = set()
        self._closed = False
        self._register_thread("Python main")
        self._metadata_event("process_name", {"name": "GLM-5.3-Flash"})
        self.instant(
            "trace_clock_anchor",
            category="metadata",
            args={
                "monotonic_origin_ns": self.origin_monotonic_ns,
                "wall_origin_unix_ns": self.origin_wall_ns,
                "timestamp_unit": "microseconds from monotonic_origin_ns",
            },
            force=True,
        )

    def __enter__(self) -> "DecodeTrace":
        global _ACTIVE_TRACE
        with _ACTIVE_TRACE_LOCK:
            if _ACTIVE_TRACE is not None and _ACTIVE_TRACE is not self:
                raise RuntimeError("another decode trace is already installed")
            _ACTIVE_TRACE = self
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def selected(self, decode_index: int) -> bool:
        return self.decode_start <= decode_index < self.decode_stop

    def _timestamp_us(self, timestamp_ns: int | None = None) -> float:
        value = time.perf_counter_ns() if timestamp_ns is None else timestamp_ns
        return (value - self.origin_monotonic_ns) / 1000.0

    @staticmethod
    def _thread_id() -> int:
        return threading.get_native_id()

    def _append(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)

    def _metadata_event(self, name: str, args: dict[str, Any], *, tid: int = 0) -> None:
        self._append(
            {
                "name": name,
                "ph": "M",
                "pid": self.pid,
                "tid": tid,
                "args": _json_value(args),
            }
        )

    def _register_thread(self, fallback_name: str | None = None) -> int:
        tid = self._thread_id()
        with self._lock:
            if tid in self._named_threads:
                return tid
            self._named_threads.add(tid)
            name = threading.current_thread().name or fallback_name or f"thread-{tid}"
            self._events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": self.pid,
                    "tid": tid,
                    "args": {"name": name},
                }
            )
        return tid

    @contextmanager
    def span(
        self,
        name: str,
        *,
        category: str,
        args: dict[str, Any] | None = None,
        force: bool = False,
    ) -> Iterator[dict[str, Any]]:
        if not (force or self.recording):
            yield {}
            return
        tid = self._register_thread()
        started = time.perf_counter_ns()
        cpu_started = time.thread_time_ns()
        event_args = dict(args or {})
        try:
            yield event_args
        finally:
            stopped = time.perf_counter_ns()
            cpu_stopped = time.thread_time_ns()
            wall_us = (stopped - started) / 1000.0
            cpu_us = (cpu_stopped - cpu_started) / 1000.0
            event_args["host_thread_cpu_us"] = cpu_us
            event_args["host_wall_minus_cpu_us"] = max(0.0, wall_us - cpu_us)
            self._append(
                {
                    "name": name,
                    "cat": category,
                    "ph": "X",
                    "pid": self.pid,
                    "tid": tid,
                    "ts": self._timestamp_us(started),
                    "dur": wall_us,
                    "args": _json_value(event_args),
                }
            )

    def instant(
        self,
        name: str,
        *,
        category: str,
        args: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        if not (force or self.recording):
            return
        self._append(
            {
                "name": name,
                "cat": category,
                "ph": "i",
                "s": "t",
                "pid": self.pid,
                "tid": self._register_thread(),
                "ts": self._timestamp_us(),
                "args": _json_value(args or {}),
            }
        )

    def counter(
        self,
        name: str,
        values: dict[str, int | float],
        *,
        category: str = "counters",
    ) -> None:
        if not self.recording:
            return
        self._append(
            {
                "name": name,
                "cat": category,
                "ph": "C",
                "pid": self.pid,
                "tid": self._thread_id(),
                "ts": self._timestamp_us(),
                "args": _json_value(values),
            }
        )

    def new_flow(self) -> int:
        return next(self._flow_ids)

    def flow(
        self,
        phase: str,
        flow_id: int,
        *,
        name: str = "expert_refill",
        category: str = "ssd_flow",
        args: dict[str, Any] | None = None,
    ) -> None:
        if phase not in {"s", "t", "f"}:
            raise ValueError(f"invalid flow phase: {phase}")
        self._append(
            {
                "name": name,
                "cat": category,
                "ph": phase,
                "id": flow_id,
                "bp": "e",
                "pid": self.pid,
                "tid": self._register_thread(),
                "ts": self._timestamp_us(),
                "args": _json_value(args or {}),
            }
        )

    @contextmanager
    def decode_step(self, decode_index: int, *, token_id: int) -> Iterator[bool]:
        selected = self.selected(decode_index)
        if not selected:
            yield False
            return
        if self._closed:
            raise RuntimeError("decode trace is already closed")
        self.recording = True
        self.current_decode_index = decode_index
        try:
            with self.span(
                "decode_step",
                category="python",
                args={"decode_index": decode_index, "input_token_id": token_id},
            ):
                yield True
        finally:
            self.current_decode_index = None
            self.recording = False

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            events = list(self._events)
        payload = {
            "traceEvents": events,
            "displayTimeUnit": "ms",
            "otherData": {
                "format": "livglm-perfetto-v1",
                "pid": self.pid,
                "decode_start": self.decode_start,
                "decode_stop": self.decode_stop,
                "monotonic_origin_ns": self.origin_monotonic_ns,
                "wall_origin_unix_ns": self.origin_wall_ns,
                **_json_value(self.metadata),
            },
        }
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        rendered = json.dumps(payload, separators=(",", ":")) + "\n"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def close(self) -> None:
        global _ACTIVE_TRACE
        if self._closed:
            return
        self._closed = True
        self.recording = False
        self.current_decode_index = None
        with _ACTIVE_TRACE_LOCK:
            if _ACTIVE_TRACE is self:
                _ACTIVE_TRACE = None
        self.write()
