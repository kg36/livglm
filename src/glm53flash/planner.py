from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .hf_range import index, remote_safetensors_header
from .sources import EXPERTS, OFFICIAL, expert_layer, mxfp4_routed_tensor, official_routed_tensor


DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "F8_E8M0": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8,
}


def _payload_size(meta: dict[str, Any]) -> int:
    count = 1
    for dim in meta["shape"]:
        count *= dim
    try:
        return count * DTYPE_BYTES[meta["dtype"]]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {meta['dtype']}") from exc


def _selected(source_label: str, name: str) -> bool:
    if source_label == OFFICIAL.label:
        return not official_routed_tensor(name)
    if source_label == EXPERTS.label:
        return mxfp4_routed_tensor(name)
    return False


def build_plan() -> dict[str, Any]:
    source_indexes = {OFFICIAL.label: index(OFFICIAL), EXPERTS.label: index(EXPERTS)}
    sources = {OFFICIAL.label: OFFICIAL, EXPERTS.label: EXPERTS}
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    inventories: dict[str, dict[str, int]] = {}

    for label, source in sources.items():
        weight_map = source_indexes[label]["weight_map"]
        selected_names = [name for name in weight_map if _selected(label, name)]
        names_by_shard: dict[str, list[str]] = defaultdict(list)
        for name in selected_names:
            names_by_shard[weight_map[name]].append(name)

        payload = 0
        for shard in sorted(names_by_shard):
            data_base, header = remote_safetensors_header(source, shard)
            for name in names_by_shard[shard]:
                if name not in header:
                    raise ValueError(f"index/header mismatch for {label}:{name}")
                meta = header[name]
                start, end = meta["data_offsets"]
                length = end - start
                if length != _payload_size(meta):
                    raise ValueError(f"shape/dtype byte mismatch for {name}")
                payload += length
                group = (
                    f"expert-layer-{expert_layer(name):03d}"
                    if label == EXPERTS.label
                    else f"official-{shard.removesuffix('.safetensors')}"
                )
                by_group[group].append(
                    {
                        "name": name,
                        "source": label,
                        "source_shard": shard,
                        "source_start": data_base + start,
                        "length": length,
                        "dtype": meta["dtype"],
                        "shape": meta["shape"],
                    }
                )
        inventories[label] = {"tensor_count": len(selected_names), "payload_bytes": payload}

    expected_expert_tensors = 43 * 288 * 3 * 2
    if inventories[EXPERTS.label]["tensor_count"] != expected_expert_tensors:
        raise ValueError(
            f"MXFP4 expert inventory is incomplete: {inventories[EXPERTS.label]['tensor_count']} "
            f"!= {expected_expert_tensors}"
        )

    group_names = sorted(by_group, key=lambda x: (not x.startswith("official-"), x))
    shard_count = len(group_names)
    shards: list[dict[str, Any]] = []
    weight_map: dict[str, str] = {}

    for number, group in enumerate(group_names, 1):
        filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        tensors = sorted(
            by_group[group], key=lambda t: (t["source_shard"], t["source_start"], t["name"])
        )
        cursor = 0
        for tensor in tensors:
            tensor["output_start"] = cursor
            cursor += tensor["length"]
            weight_map[tensor["name"]] = filename
        shards.append(
            {"group": group, "filename": filename, "payload_bytes": cursor, "tensors": tensors}
        )

    total = sum(item["payload_bytes"] for item in shards)
    return {
        "format": "livseek-glm53flash-composite-v1",
        "sources": {label: asdict(source) for label, source in sources.items()},
        "inventories": inventories,
        "tensor_count": len(weight_map),
        "payload_bytes": total,
        "shard_count": shard_count,
        "weight_map": weight_map,
        "shards": shards,
    }


def write_plan(plan: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plan, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def load_or_build_plan(path: Path) -> dict[str, Any]:
    if path.exists():
        plan = json.loads(path.read_text(encoding="utf-8"))
        if plan.get("format") != "livseek-glm53flash-composite-v1":
            raise ValueError(f"unsupported cached plan: {path}")
        return plan
    plan = build_plan()
    write_plan(plan, path)
    return plan
