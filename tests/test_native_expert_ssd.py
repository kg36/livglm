from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from glm53flash.expert_reader import NativeExpertReader
from glm53flash.expert_source import (
    ExpertReadRange,
    ExpertTensorSource,
    NativeExpertSource,
    NativeExpertSourcePlan,
)
from glm53flash.expert_ssd import ExpertSSD
from glm53flash.native_expert_ssd import (
    NativeExpertPool,
    NativeExpertSSD,
    native_expert_ssd_available,
)
from glm53flash.runtime import DEFAULT_MODEL_DIR, TargetRuntime


pytestmark = pytest.mark.skipif(
    not native_expert_ssd_available(),
    reason="mlx-io-glm overlay is not active",
)


def _native_plan(tmp_path: Path, *, experts: int = 3):
    rng = np.random.default_rng(53)
    packs = []
    payload = bytearray()
    matrices = []
    for expert in range(experts):
        expert_tensors = []
        expert_matrices = {}
        expert_start = len(payload)
        for projection in ("down_proj", "gate_proj", "up_proj"):
            matrix = rng.normal(0, 0.2, (32, 32)).astype(np.float32)
            expert_matrices[projection] = matrix
            packed, scales = mx.quantize(
                mx.array(matrix),
                group_size=32,
                bits=4,
                mode="mxfp4",
            )
            mx.eval(packed, scales)
            for part, array, dtype in (
                ("weight", packed, "U32"),
                ("scales", scales, "U8"),
            ):
                values = np.array(array)
                raw = values.astype("<u4", copy=False).tobytes() if dtype == "U32" else values.tobytes()
                offset = len(payload)
                payload.extend(raw)
                expert_tensors.append(
                    ExpertTensorSource(
                        source_name=f"{projection}.{part}",
                        destination_name=f"{projection}.{part}",
                        shard_name="experts.bin",
                        absolute_offset=offset,
                        byte_length=len(raw),
                        source_shape=tuple(values.shape),
                        mlx_dtype=dtype,
                        mlx_shape=tuple(values.shape),
                    )
                )
        source_range = ExpertReadRange(
            "experts.bin",
            expert_start,
            len(payload) - expert_start,
            tuple(expert_tensors),
        )
        packs.append(
            NativeExpertSource(
                layer=0,
                expert=expert,
                tensors=tuple(expert_tensors),
                read_ranges=(source_range,),
            )
        )
        matrices.append(expert_matrices)
    (tmp_path / "experts.bin").write_bytes(payload)
    return NativeExpertSourcePlan(
        model_dir=str(tmp_path),
        first_layer=0,
        last_layer=0,
        experts_per_layer=experts,
        packs=tuple(packs),
    ), matrices


def test_native_direct_slots_match_python_reader_and_reuse_rows(tmp_path):
    plan, _ = _native_plan(tmp_path)
    rng = np.random.default_rng(8)
    x = mx.array(rng.normal(0, 0.2, (1, 32)).astype(np.float32))
    indices = mx.array([[2, 0]], dtype=mx.uint32)

    with NativeExpertReader(plan) as reader:
        reference = ExpertSSD(
            reader,
            layer=0,
            capacity=2,
            swiglu_limit=10.0,
        )
        expected = reference(x, indices)
        mx.eval(expected)

    pool = NativeExpertPool(plan, workers=2)
    native = NativeExpertSSD(
        pool,
        layer=0,
        capacity=2,
        swiglu_limit=10.0,
        wire_slots=False,
        defer_slots=True,
    )
    try:
        with pytest.raises(RuntimeError, match="not activated"):
            native(x, indices)
        native.activate()
        actual = native(x, indices)
        mx.eval(actual)
        np.testing.assert_array_equal(np.array(actual), np.array(expected))
        assert native.resident_experts == (2, 0)
        assert native.stats().misses == 2
        assert pool.stats().expert_loads == 2
        assert pool.stats().logical_reads == 2
        assert pool.stats().direct_to_slot is True

        warm = native(x, indices)
        mx.eval(warm)
        np.testing.assert_array_equal(np.array(warm), np.array(expected))
        assert native.stats().hits == 2
        assert pool.stats().expert_loads == 2

        replaced = native(x, mx.array([[1, 0]], dtype=mx.uint32))
        mx.eval(replaced)
        assert native.resident_experts == (1, 0)
        assert native.stats().hits == 3
        assert native.stats().misses == 3
        assert native.stats().evictions == 1
        assert pool.stats().expert_loads == 3
    finally:
        native.close()
        pool.close()


@pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / "VALIDATION.json").is_file(),
    reason="local composite checkpoint is absent",
)
def test_real_layer3_expert0_native_slot_is_byte_exact():
    preflight = TargetRuntime.preflight(
        DEFAULT_MODEL_DIR,
        memory_gib=24,
        physical_bytes=256 * 2**30,
    )
    with NativeExpertReader(preflight.expert_plan) as reader:
        expected = reader.load(3, 0)

    pool = NativeExpertPool(preflight.expert_plan, workers=1)
    source = pool.layer(3)
    slots = source.allocate_slots(1)
    try:
        elapsed = source.load_into(0, 0, slots)
        assert elapsed >= 0.0
        by_name = {
            name: slots[index][0]
            for index, (name, _, _) in enumerate(source.tensor_layouts)
        }
        for name, expected_array in expected.tensors.items():
            np.testing.assert_array_equal(
                np.array(by_name[name]),
                np.array(expected_array),
            )
    finally:
        pool.close()


@pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / "VALIDATION.json").is_file(),
    reason="local composite checkpoint is absent",
)
def test_real_layer3_native_qmv_is_bit_exact_to_stock_gather_qmm():
    preflight = TargetRuntime.preflight(
        DEFAULT_MODEL_DIR,
        memory_gib=24,
        physical_bytes=256 * 2**30,
    )
    rng = np.random.default_rng(5301)
    x = mx.array(
        rng.normal(0, 0.2, (1, 4096)).astype(np.float32)
    ).astype(mx.bfloat16)
    indices = mx.array([list(range(8))], dtype=mx.uint32)

    with NativeExpertReader(preflight.expert_plan) as reader:
        reference = ExpertSSD(
            reader,
            layer=3,
            capacity=8,
            swiglu_limit=10.0,
        )
        expected = reference(x, indices)
        mx.eval(expected)

    pool = NativeExpertPool(preflight.expert_plan, workers=8)
    native = NativeExpertSSD(
        pool,
        layer=3,
        capacity=8,
        swiglu_limit=10.0,
        wire_slots=False,
    )
    try:
        actual = native(x, indices)
        mx.eval(actual)
        np.testing.assert_array_equal(
            np.array(actual.astype(mx.float32)),
            np.array(expected.astype(mx.float32)),
        )
    finally:
        native.close()
        pool.close()
