from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import threading
from typing import Any

from .hf_range import copy_range_to_file, fetch_bytes, fetch_json, shard_url
from .planner import load_or_build_plan
from .scalex_container import is_scalex_layer
from .sources import EXPERTS, OFFICIAL, SOURCES


RANGE_CHUNK = 256 * 1024 * 1024
SYSTEM_RESERVE = 10 * 1024**3
PRINT_LOCK = threading.Lock()


def _header_bytes(shard: dict[str, Any]) -> bytes:
    header: dict[str, Any] = {
        "__metadata__": {
            "format": "pt",
            "livseek_composite": "glm53flash-v1",
            "group": shard["group"],
        }
    }
    for tensor in shard["tensors"]:
        start = tensor["output_start"]
        header[tensor["name"]] = {
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "data_offsets": [start, start + tensor["length"]],
        }
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    return struct.pack("<Q", len(raw)) + raw


def _operations(shard: dict[str, Any], data_base: int) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    for tensor in shard["tensors"]:
        candidate = {
            "source": tensor["source"],
            "source_shard": tensor["source_shard"],
            "source_start": tensor["source_start"],
            "length": tensor["length"],
            "output_start": data_base + tensor["output_start"],
        }
        if operations:
            previous = operations[-1]
            adjacent = (
                previous["source"] == candidate["source"]
                and previous["source_shard"] == candidate["source_shard"]
                and previous["source_start"] + previous["length"] == candidate["source_start"]
                and previous["output_start"] + previous["length"] == candidate["output_start"]
            )
            if adjacent:
                previous["length"] += candidate["length"]
                continue
        operations.append(candidate)
    return operations


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def _prepare_partial(partial: Path, header: bytes, total_size: int) -> None:
    if partial.exists():
        with partial.open("rb") as handle:
            existing = handle.read(len(header))
        if existing != header or partial.stat().st_size != total_size:
            raise ValueError(f"partial shard does not match cached plan: {partial}")
        return
    with partial.open("xb") as handle:
        handle.write(header)
        handle.truncate(total_size)
        handle.flush()
        os.fsync(handle.fileno())


def _download_shard(destination: Path, state_dir: Path, shard: dict[str, Any]) -> str:
    final = destination / shard["filename"]
    if final.exists():
        return f"SKIP {final.name} already complete"
    partial = destination / (shard["filename"] + ".partial")
    progress_path = state_dir / (shard["filename"] + ".progress.json")
    header = _header_bytes(shard)
    data_base = len(header)
    total_size = data_base + shard["payload_bytes"]
    _prepare_partial(partial, header, total_size)
    operations = _operations(shard, data_base)
    progress = {"operation": 0, "done": 0}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))

    with partial.open("r+b", buffering=0) as output:
        for op_index in range(progress["operation"], len(operations)):
            operation = operations[op_index]
            done = progress["done"] if op_index == progress["operation"] else 0
            while done < operation["length"]:
                amount = min(RANGE_CHUNK, operation["length"] - done)
                source = SOURCES[operation["source"]]
                url = shard_url(source, operation["source_shard"])
                copy_range_to_file(
                    url,
                    operation["source_start"] + done,
                    operation["source_start"] + done + amount - 1,
                    output,
                    operation["output_start"] + done,
                )
                output.flush()
                os.fsync(output.fileno())
                done += amount
                _atomic_json(progress_path, {"operation": op_index, "done": done})
            progress = {"operation": op_index + 1, "done": 0}
            _atomic_json(progress_path, progress)

    partial.replace(final)
    progress_path.unlink(missing_ok=True)
    return f"DONE {final.name} {total_size / 1e9:.3f} GB"


def _download_support_files(destination: Path) -> None:
    api = fetch_json(
        f"https://huggingface.co/api/models/{OFFICIAL.repo}/revision/{OFFICIAL.revision}"
    )
    for sibling in api["siblings"]:
        name = sibling["rfilename"]
        if name.endswith(".safetensors") or name == "model.safetensors.index.json" or name == ".gitattributes":
            continue
        target = destination / name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(fetch_bytes(f"{OFFICIAL.base_url}/{name}?download=true"))

    (destination / "MXFP4_SOURCE_README.md").write_bytes(
        fetch_bytes(f"{EXPERTS.base_url}/README.md?download=true")
    )
    (destination / "mxfp4-quantization-config.json").write_bytes(
        fetch_bytes(f"{EXPERTS.base_url}/quantization_config.json?download=true")
    )


def assemble(destination: Path, *, workers: int = 4) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    converted = [
        path.name
        for path in destination.glob("model-*.safetensors")
        if is_scalex_layer(path)
    ]
    if converted:
        raise RuntimeError(
            "download cannot run over a ScaleX checkpoint; restore it first with "
            f"./converter --native (found {len(converted)} converted shard(s))"
        )
    state_dir = destination / ".download"
    state_dir.mkdir(exist_ok=True)
    plan_path = state_dir / "assembly-plan.json"
    plan = load_or_build_plan(plan_path)
    _download_support_files(destination)

    allocated = sum(
        getattr(path.stat(), "st_blocks", 0) * 512
        for path in destination.glob("model-*.safetensors*")
    )
    remaining = max(0, plan["payload_bytes"] - allocated)
    available = shutil.disk_usage(destination).free
    if available < remaining + SYSTEM_RESERVE:
        raise OSError(
            f"insufficient disk: need {(remaining + SYSTEM_RESERVE)/1024**3:.1f} GiB, "
            f"have {available/1024**3:.1f} GiB"
        )

    print(
        f"Plan: {plan['tensor_count']:,} tensors, {plan['shard_count']} shards, "
        f"{plan['payload_bytes']/1e9:.3f} GB payload; workers={workers}"
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download_shard, destination, state_dir, shard) for shard in plan["shards"]]
        for future in as_completed(futures):
            message = future.result()
            with PRINT_LOCK:
                print(message, flush=True)

    index_document = {
        "metadata": {
            "total_size": plan["payload_bytes"],
            "format": plan["format"],
        },
        "weight_map": plan["weight_map"],
    }
    _atomic_json(destination / "model.safetensors.index.json", index_document)
    composite = {
        "format": plan["format"],
        "status": "assembled_unvalidated",
        "payload_bytes": plan["payload_bytes"],
        "tensor_count": plan["tensor_count"],
        "shard_count": plan["shard_count"],
        "sources": plan["sources"],
        "contract": {
            "official_remainder": "official Z.ai FP8 tensors, excluding all routed expert projections",
            "routed_experts": "layers 3-45, 288 experts, gate/up/down, group-32 MXFP4 E2M1 with U8 E8M0 scales",
            "runtime_compatibility": "custom LivSeek loader required",
        },
    }
    _atomic_json(destination / "livseek-composite.json", composite)
    return plan
