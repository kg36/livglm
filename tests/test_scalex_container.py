import hashlib
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from glm53flash.contract import ModelContract
from glm53flash.scalex_container import (
    ScaleXContainerError,
    compress_scalex_layer,
    compress_scalex_layer_in_place,
    is_scalex_layer,
    read_scalex_layout,
    read_scalex_tensor,
    restore_scalex_layer,
    restore_scalex_layer_in_place,
    verify_scalex_layer,
)


def _native_layer(path: Path, *, layer: int = 3, experts: int = 2) -> None:
    rng = np.random.default_rng(5303)
    tensors = {}
    for expert in range(experts):
        for component in ("down_proj", "gate_proj", "up_proj"):
            prefix = (
                f"model.language_model.layers.{layer}.mlp.experts.{expert}."
                f"{component}"
            )
            tensors[prefix + ".weight_packed"] = rng.integers(
                0, 256, size=(32, 16), dtype=np.uint8
            )
            scales = np.resize(np.array([120, 121], np.uint8), (32, 1)).copy()
            tensors[prefix + ".weight_scale"] = scales
    save_file(tensors, path, metadata={"format": "synthetic-glm-mxfp4"})


def test_scalex_container_is_byte_reversible(tmp_path):
    native = tmp_path / "native.safetensors"
    compressed = tmp_path / "compressed.safetensors"
    restored = tmp_path / "restored.safetensors"
    _native_layer(native)
    original = native.read_bytes()

    report = compress_scalex_layer(native, compressed, layer=3, experts=2)
    assert report.operation == "compress"
    assert report.byte_identical
    assert report.bytes_saved == len(original) - compressed.stat().st_size
    assert is_scalex_layer(compressed)

    layout = read_scalex_layout(compressed)
    assert layout.layer == 3
    assert layout.experts == 2
    assert layout.records[0].codec == "one_bit"
    assert layout.records[0].scale_names[0].endswith("gate_proj.weight_scale")
    assert layout.records[0].weight_names[1].endswith("down_proj.weight_packed")
    first_weight = layout.records[0].weight_names[0]
    assert read_scalex_tensor(layout, first_weight)
    contract = ModelContract(
        tmp_path,
        {},
        {
            "weight_map": {
                name: compressed.name
                for name in layout.virtual_header
                if name != "__metadata__"
            }
        },
        {},
        {},
    )
    assert contract.tensor(first_weight).byte_length == len(
        read_scalex_tensor(layout, first_weight)
    )
    contract.audit_headers()
    assert verify_scalex_layer(compressed).reconstructed_sha256 == hashlib.sha256(original).hexdigest()

    restored_report = restore_scalex_layer(compressed, restored)
    assert restored_report.byte_identical
    assert restored.read_bytes() == original


def test_scalex_container_detects_payload_corruption(tmp_path):
    native = tmp_path / "native.safetensors"
    compressed = tmp_path / "compressed.safetensors"
    _native_layer(native)
    compress_scalex_layer(native, compressed, layer=3, experts=2)
    layout = read_scalex_layout(compressed)
    with compressed.open("r+b") as handle:
        handle.seek(layout.records[0].absolute_offset + layout.records[0].encoded_bytes)
        value = handle.read(1)
        handle.seek(-1, 1)
        handle.write(bytes([value[0] ^ 1]))
    with pytest.raises(ScaleXContainerError, match="hash differs"):
        verify_scalex_layer(compressed)


def test_scalex_in_place_conversion_is_idempotent_and_reversible(tmp_path):
    native = tmp_path / "model.safetensors"
    _native_layer(native)
    original = native.read_bytes()
    compressed = compress_scalex_layer_in_place(native, layer=3, experts=2)
    assert compressed.operation == "compress"
    assert is_scalex_layer(native)
    assert compress_scalex_layer_in_place(native, layer=3, experts=2).operation == "verify"
    restored = restore_scalex_layer_in_place(native)
    assert restored.operation == "restore"
    assert native.read_bytes() == original
