from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from glm53flash.contract import TensorSource
from glm53flash.resident_plan import (
    BLOCK_FP8_TO_BF16,
    COPY,
    ResidentSourcePlan,
    ResidentTensorPlan,
)
from glm53flash.resident_reader import NativeResidentReader


def test_resident_reader_copies_and_closes_mmap(tmp_path):
    values = np.array([[1.0, -2.0], [3.5, 0.25]], dtype=np.float32)
    encoded = np.array(mx.array(values).astype(mx.bfloat16).view(mx.uint16)).astype("<u2").tobytes()
    (tmp_path / "resident.bin").write_bytes(encoded)
    source = TensorSource("source", "resident.bin", 0, len(encoded), "BF16", (2, 2))
    tensor = ResidentTensorPlan(
        source_names=("source",),
        destination_name="weight",
        sources=(source,),
        destination_dtype="BF16",
        destination_shape=(2, 2),
        destination_bytes=8,
        transform=COPY,
    )
    plan = ResidentSourcePlan(str(tmp_path), (tensor,), 1)
    target = SimpleNamespace(weight=None)
    stats = NativeResidentReader(plan).load_into(target)
    assert stats.runtime_tensors == 1
    np.testing.assert_allclose(np.array(target.weight.astype(mx.float32)), values, atol=0.02)


def test_resident_reader_decodes_block_fp8(tmp_path):
    weight_raw = bytes([0x38]) * (128 * 128)
    scale_raw = np.array([[2.0]], dtype="<f4").tobytes()
    (tmp_path / "resident.bin").write_bytes(weight_raw + scale_raw)
    weight = TensorSource("weight", "resident.bin", 0, len(weight_raw), "F8_E4M3", (128, 128))
    scale = TensorSource("scale", "resident.bin", len(weight_raw), 4, "F32", (1, 1))
    tensor = ResidentTensorPlan(
        source_names=("weight", "scale"),
        destination_name="weight",
        sources=(weight, scale),
        destination_dtype="BF16",
        destination_shape=(128, 128),
        destination_bytes=128 * 128 * 2,
        transform=BLOCK_FP8_TO_BF16,
    )
    target = SimpleNamespace(weight=None)
    NativeResidentReader(ResidentSourcePlan(str(tmp_path), (tensor,), 2)).load_into(target)
    np.testing.assert_array_equal(
        np.array(target.weight.astype(mx.float32)),
        np.full((128, 128), 2.0, np.float32),
    )
