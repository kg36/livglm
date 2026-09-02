from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .hf_range import fetch_range, local_safetensors_header, shard_url
from .sources import SOURCES
from .scalex_container import (
    is_scalex_layer,
    read_scalex_layout,
    read_scalex_tensor,
    verify_scalex_layer,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(32 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate(destination: Path, *, full_hash: bool = True) -> dict[str, Any]:
    plan_path = destination / ".download" / "assembly-plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"missing assembly plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    model_index = json.loads((destination / "model.safetensors.index.json").read_text(encoding="utf-8"))
    if model_index["weight_map"] != plan["weight_map"]:
        raise ValueError("final weight map differs from the pinned assembly plan")

    hashes: list[tuple[str, str]] = []
    reconstructed_hashes: list[tuple[str, str]] = []
    audited_tensors = 0
    scalex_shard_count = 0
    for number, shard in enumerate(plan["shards"], 1):
        path = destination / shard["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        scalex = is_scalex_layer(path)
        if scalex:
            scalex_shard_count += 1
            layout = read_scalex_layout(path)
            scalex_report = verify_scalex_layer(path)
            data_base, header = layout.virtual_data_base, layout.virtual_header
        else:
            layout = None
            scalex_report = None
            data_base, header = local_safetensors_header(path)
        expected_names = {tensor["name"] for tensor in shard["tensors"]}
        actual_names = set(header) - {"__metadata__"}
        if actual_names != expected_names:
            raise ValueError(f"tensor inventory mismatch in {path.name}")
        for tensor in shard["tensors"]:
            meta = header[tensor["name"]]
            expected_meta = {
                "dtype": tensor["dtype"],
                "shape": tensor["shape"],
                "data_offsets": [tensor["output_start"], tensor["output_start"] + tensor["length"]],
            }
            if meta != expected_meta:
                raise ValueError(f"metadata mismatch for {tensor['name']}")
        expected_size = data_base + shard["payload_bytes"]
        actual_logical_size = layout.original_bytes if layout is not None else path.stat().st_size
        if actual_logical_size != expected_size:
            raise ValueError(f"file size mismatch for {path.name}")
        audited_tensors += len(shard["tensors"])
        if full_hash:
            digest = _sha256(path)
            hashes.append((digest, path.name))
            if scalex_report is not None:
                reconstructed_hashes.append(
                    (scalex_report.reconstructed_sha256, path.name)
                )
        print(f"AUDIT {number:03d}/{plan['shard_count']:03d} {path.name}", flush=True)

    samples = []
    all_tensors = [tensor for shard in plan["shards"] for tensor in shard["tensors"]]
    for tensor in (all_tensors[0], all_tensors[len(all_tensors)//2], all_tensors[-1]):
        local_shard = destination / plan["weight_map"][tensor["name"]]
        sample_length = min(1024 * 1024, tensor["length"])
        if is_scalex_layer(local_shard):
            local = read_scalex_tensor(
                read_scalex_layout(local_shard), tensor["name"]
            )[:sample_length]
        else:
            local_base, local_header = local_safetensors_header(local_shard)
            local_start = local_base + local_header[tensor["name"]]["data_offsets"][0]
            with local_shard.open("rb") as handle:
                handle.seek(local_start)
                local = handle.read(sample_length)
        source = SOURCES[tensor["source"]]
        remote = fetch_range(
            shard_url(source, tensor["source_shard"]),
            tensor["source_start"],
            tensor["source_start"] + sample_length - 1,
        )
        if local != remote:
            raise ValueError(f"remote byte sample mismatch for {tensor['name']}")
        samples.append(tensor["name"])

    if full_hash:
        (destination / "SHA256SUMS").write_text(
            "".join(f"{digest}  {name}\n" for digest, name in hashes), encoding="utf-8"
        )
        reconstructed_path = destination / "SCALEX_RECONSTRUCTED_SHA256SUMS"
        if reconstructed_hashes:
            reconstructed_path.write_text(
                "".join(
                    f"{digest}  {name}\n" for digest, name in reconstructed_hashes
                ),
                encoding="utf-8",
            )
        elif reconstructed_path.exists():
            reconstructed_path.unlink()
    report = {
        "status": "PASS",
        "payload_bytes": plan["payload_bytes"],
        "tensor_count": audited_tensors,
        "shard_count": plan["shard_count"],
        "full_sha256": full_hash,
        "scalex_shards": scalex_shard_count,
        "remote_byte_samples": samples,
    }
    (destination / "VALIDATION.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    composite_path = destination / "livseek-composite.json"
    composite = json.loads(composite_path.read_text(encoding="utf-8"))
    composite["status"] = "validated"
    composite_path.write_text(json.dumps(composite, indent=2) + "\n", encoding="utf-8")
    return report
