from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .hf_range import index, remote_safetensors_header, tensor_blob
from .sources import BF16, EXPERTS, OFFICIAL, mxfp4_routed_tensor, official_routed_tensor


SAMPLE_LAYERS = (3, 10, 44, 45)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _get(source, idx, name):
    shard = idx["weight_map"][name]
    data_base, header = remote_safetensors_header(source, shard)
    return header[name], tensor_blob(source, shard, data_base, header[name])


def run_preflight(report_path: Path) -> dict[str, Any]:
    official_index = index(OFFICIAL)
    expert_index = index(EXPERTS)
    bf16_index = index(BF16)
    official_names = official_index["weight_map"]
    expert_names = expert_index["weight_map"]

    counts = {
        "official_routed_tensors_replaced": sum(official_routed_tensor(n) for n in official_names),
        "official_remainder_tensors": sum(not official_routed_tensor(n) for n in official_names),
        "mxfp4_expert_tensors": sum(mxfp4_routed_tensor(n) for n in expert_names),
    }
    expected = 43 * 288 * 3 * 2
    if counts["official_routed_tensors_replaced"] != expected:
        raise ValueError(f"official routed inventory mismatch: {counts}")
    if counts["mxfp4_expert_tensors"] != expected:
        raise ValueError(f"MXFP4 routed inventory mismatch: {counts}")

    try:
        import mlx.core as mx
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numerical preflight requires NumPy and MLX") from exc

    results = []
    for layer in SAMPLE_LAYERS:
        for projection in PROJECTIONS:
            stem = f"model.language_model.layers.{layer}.mlp.experts.0.{projection}"
            packed_meta, packed_blob = _get(EXPERTS, expert_index, stem + ".weight_packed")
            scale_meta, scale_blob = _get(EXPERTS, expert_index, stem + ".weight_scale")
            bf16_meta, bf16_blob = _get(BF16, bf16_index, stem + ".weight")

            if packed_meta["dtype"] != "U8" or scale_meta["dtype"] != "U8":
                raise ValueError(f"unexpected MXFP4 storage dtype for {stem}")
            if bf16_meta["dtype"] != "BF16":
                raise ValueError(f"unexpected BF16 reference dtype for {stem}")
            rows, cols = bf16_meta["shape"]
            if packed_meta["shape"] != [rows, cols // 2]:
                raise ValueError(f"packed shape mismatch for {stem}")
            if scale_meta["shape"] != [rows, cols // 32]:
                raise ValueError(f"scale shape mismatch for {stem}")

            packed_np = np.frombuffer(packed_blob, dtype=np.uint8).reshape(packed_meta["shape"])
            scales_np = np.frombuffer(scale_blob, dtype=np.uint8).reshape(scale_meta["shape"])
            bf16_u16 = np.frombuffer(bf16_blob, dtype="<u2").reshape(bf16_meta["shape"])
            weight = mx.array(packed_np).view(mx.uint32)
            scales = mx.array(scales_np)
            reference_weight = mx.array(bf16_u16).view(mx.bfloat16)
            mx.random.seed(5300 + layer)
            x = mx.random.normal((2, cols)).astype(mx.bfloat16)
            quantized = mx.quantized_matmul(
                x, weight, scales, transpose=True, group_size=32, bits=4, mode="mxfp4"
            )
            reference = x @ reference_weight.T
            mx.eval(quantized, reference)
            q = np.asarray(quantized.astype(mx.float32))
            r = np.asarray(reference.astype(mx.float32))
            error = q - r
            nrmse = float(np.sqrt(np.mean(error * error)) / max(np.sqrt(np.mean(r * r)), 1e-12))
            cosine = float(np.sum(q * r) / max(np.linalg.norm(q) * np.linalg.norm(r), 1e-12))
            if not math.isfinite(nrmse) or not math.isfinite(cosine) or cosine < 0.95 or nrmse > 0.35:
                raise ValueError(
                    f"MXFP4 numerical gate failed for {stem}: cosine={cosine:.6f}, nrmse={nrmse:.6f}"
                )
            results.append(
                {
                    "layer": layer,
                    "expert": 0,
                    "projection": projection,
                    "shape": [rows, cols],
                    "cosine": cosine,
                    "nrmse": nrmse,
                }
            )
            print(f"PASS layer={layer:02d} {projection:9s} cosine={cosine:.6f} nrmse={nrmse:.6f}")

    report = {
        "status": "PASS",
        "sources": {
            source.label: {"repo": source.repo, "revision": source.revision}
            for source in (OFFICIAL, EXPERTS, BF16)
        },
        "inventory": counts,
        "samples": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
