"""Summarize a LivGLM Perfetto trace without requiring Perfetto itself."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence


def _interval_stats(events: list[dict[str, Any]]) -> tuple[float, int]:
    points: list[tuple[float, int]] = []
    for event in events:
        start = float(event["ts"])
        points.append((start, 1))
        points.append((start + float(event["dur"]), -1))
    active = 0
    peak = 0
    previous: float | None = None
    union_us = 0.0
    for timestamp, change in sorted(points, key=lambda point: (point[0], point[1])):
        if previous is not None and active > 0:
            union_us += timestamp - previous
        active += change
        peak = max(peak, active)
        previous = timestamp
    return union_us, peak


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize_trace(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("traceEvents must be a list")
    complete = [event for event in events if event.get("ph") == "X"]
    counters = [event for event in events if event.get("ph") == "C"]
    roots = sorted(
        (event for event in complete if event.get("name") == "decode_step"),
        key=lambda event: float(event["ts"]),
    )
    steps: list[dict[str, Any]] = []
    all_reads: list[dict[str, Any]] = []
    all_layers: list[dict[str, Any]] = []
    all_metal: list[dict[str, Any]] = []
    for root in roots:
        start = float(root["ts"])
        stop = start + float(root["dur"])
        inside = [
            event
            for event in complete
            if event is not root and start <= float(event["ts"]) < stop
        ]
        inside_counters = [
            event
            for event in counters
            if start <= float(event["ts"]) < stop
        ]
        reads = [event for event in inside if event.get("name") == "SSD_read"]
        metal = [event for event in inside if event.get("cat") == "metal_gpu"]
        layers = [
            event for event in inside if event.get("name") == "transformer_layer"
        ]
        all_reads.extend(reads)
        all_layers.extend(layers)
        all_metal.extend(metal)
        read_union_us, peak_concurrency = _interval_stats(reads)
        metal_union_us, metal_peak = _interval_stats(metal)
        read_payload = sum(
            int(event.get("args", {}).get("payload_bytes", 0)) for event in reads
        )

        def duration_ms(name: str) -> float:
            return sum(
                float(event["dur"])
                for event in inside
                if event.get("name") == name
            ) / 1000.0

        metrics: dict[str, Any] = {}
        for counter in inside_counters:
            metrics.update(counter.get("args", {}))
        steps.append(
            {
                "decode_index": root.get("args", {}).get("decode_index"),
                "wall_ms": float(root["dur"]) / 1000.0,
                "model_forward_ms": duration_ms("model_forward"),
                "final_gpu_wait_ms": duration_ms("final_logits_eval_wait"),
                "route_gpu_wait_ms": duration_ms("materialize_route"),
                "slot_plan_ms": duration_ms("slot_plan"),
                "ssd_issue_ms": duration_ms("issue_reads"),
                "ssd_join_ms": duration_ms("join_reads"),
                "ssd_worker_sum_ms": duration_ms("SSD_read"),
                "ssd_worker_union_ms": read_union_us / 1000.0,
                "ssd_peak_concurrency": peak_concurrency,
                "ssd_read_count": len(reads),
                "ssd_payload_bytes": read_payload,
                "ssd_active_gb_per_second": (
                    read_payload / (read_union_us / 1e6) / 1e9
                    if read_union_us
                    else None
                ),
                "metal_gpu_active_ms": metal_union_us / 1000.0,
                "metal_gpu_peak_concurrency": metal_peak,
                "metal_gpu_interval_count": len(metal),
                "transformer_layers": len(layers),
                "moe_layers": sum(
                    event.get("name") == "moe_layer" for event in inside
                ),
                "mlx_submit_span_sum_ms": sum(
                    float(event["dur"])
                    for event in inside
                    if event.get("cat") == "mlx_submit"
                ) / 1000.0,
                "python_main_thread_cpu_ms": float(
                    root.get("args", {}).get("host_thread_cpu_us", 0.0)
                ) / 1000.0,
                "ssd_worker_thread_cpu_ms": sum(
                    float(event.get("args", {}).get("host_thread_cpu_us", 0.0))
                    for event in reads
                ) / 1000.0,
                "metrics": metrics,
            }
        )

    read_ms = [float(event["dur"]) / 1000.0 for event in all_reads]
    layer_ms = [float(event["dur"]) / 1000.0 for event in all_layers]
    total_wall_ms = sum(step["wall_ms"] for step in steps)
    total_payload = sum(step["ssd_payload_bytes"] for step in steps)
    total_active_ms = sum(step["ssd_worker_union_ms"] for step in steps)
    covered_metal_steps = sum(
        bool(step["metal_gpu_interval_count"]) for step in steps
    )
    metal_union_us, metal_peak = _interval_stats(all_metal)
    return {
        "source": str(source),
        "format": payload.get("otherData", {}).get("format"),
        "event_count": len(events),
        "decode_steps": len(steps),
        "aggregate": {
            "wall_ms": total_wall_ms,
            "tps": len(steps) / (total_wall_ms / 1000.0) if total_wall_ms else None,
            "route_gpu_wait_ms_per_token": (
                sum(step["route_gpu_wait_ms"] for step in steps) / len(steps)
                if steps
                else None
            ),
            "final_gpu_wait_ms_per_token": (
                sum(step["final_gpu_wait_ms"] for step in steps) / len(steps)
                if steps
                else None
            ),
            "ssd_join_ms_per_token": (
                sum(step["ssd_join_ms"] for step in steps) / len(steps)
                if steps
                else None
            ),
            "ssd_payload_bytes": total_payload,
            "ssd_active_gb_per_second": (
                total_payload / (total_active_ms / 1000.0) / 1e9
                if total_active_ms
                else None
            ),
            "ssd_read_ms_p50": statistics.median(read_ms) if read_ms else None,
            "ssd_read_ms_p95": _percentile(read_ms, 0.95),
            "ssd_read_ms_max": max(read_ms) if read_ms else None,
            "metal_gpu_active_ms": metal_union_us / 1000.0,
            "metal_gpu_covered_steps": covered_metal_steps,
            "metal_gpu_active_ms_per_covered_token": (
                metal_union_us / 1000.0 / covered_metal_steps
                if covered_metal_steps
                else None
            ),
            "metal_gpu_peak_concurrency": metal_peak,
            "transformer_layer_ms_p50": (
                statistics.median(layer_ms) if layer_ms else None
            ),
            "transformer_layer_ms_p95": _percentile(layer_ms, 0.95),
        },
        "steps": steps,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="livglm-trace-summary",
        description="Summarize a LivGLM Perfetto/Chrome decode trace.",
    )
    parser.add_argument("trace", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(summarize_trace(args.trace), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
