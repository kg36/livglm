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
from glm53flash.scale_codec import decode_mode_a, encode_mode_a, encode_mode_b_row
from glm53flash.scalex_container import ScaleXRecord


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


def _scalex_plan(tmp_path: Path, *, experts: int = 2):
    rng = np.random.default_rng(5304)
    payload = bytearray()
    packs = []
    expected = []
    for expert in range(experts):
        tensors = []
        arrays = {}
        for projection in ("down_proj", "gate_proj", "up_proj"):
            matrix = rng.normal(0, 0.2, (32, 32)).astype(np.float32)
            packed, scales = mx.quantize(
                mx.array(matrix), group_size=32, bits=4, mode="mxfp4"
            )
            mx.eval(packed, scales)
            arrays[projection + ".weight"] = np.array(packed).astype("<u4", copy=False)
            arrays[projection + ".scales"] = np.array(scales).astype(np.uint8, copy=False)
        scale_names = tuple(f"{name}.scales" for name in ("gate_proj", "down_proj", "up_proj"))
        weight_names = tuple(f"{name}.weight" for name in ("gate_proj", "down_proj", "up_proj"))
        raw_scales = b"".join(arrays[name].tobytes() for name in scale_names)
        encoded = encode_mode_a(raw_scales)
        start = len(payload)
        payload.extend(encoded)
        for name in weight_names:
            payload.extend(arrays[name].tobytes())
        for projection in ("down_proj", "gate_proj", "up_proj"):
            for part, dtype in (("weight", "U32"), ("scales", "U8")):
                name = f"{projection}.{part}"
                value = arrays[name]
                tensors.append(
                    ExpertTensorSource(
                        source_name=name,
                        destination_name=name,
                        shard_name="scalex.bin",
                        absolute_offset=0,
                        byte_length=value.nbytes,
                        source_shape=tuple(value.shape),
                        mlx_dtype=dtype,
                        mlx_shape=tuple(value.shape),
                    )
                )
        record = ScaleXRecord(
            expert=expert,
            absolute_offset=start,
            encoded_bytes=len(encoded),
            decoded_bytes=len(raw_scales),
            scale_names=scale_names,
            scale_nbytes=tuple(arrays[name].nbytes for name in scale_names),
            weight_names=weight_names,
            weight_nbytes=tuple(arrays[name].nbytes for name in weight_names),
            codec="one_bit",
        )
        packs.append(
            NativeExpertSource(
                layer=0,
                expert=expert,
                tensors=tuple(tensors),
                read_ranges=(
                    ExpertReadRange(
                        "scalex.bin",
                        start,
                        record.physical_bytes,
                        tuple(tensor for tensor in tensors if tensor.destination_name.endswith(".weight")),
                    ),
                ),
                scalex_record=record,
            )
        )
        expected.append((encoded, arrays))
    (tmp_path / "scalex.bin").write_bytes(payload)
    return NativeExpertSourcePlan(
        model_dir=str(tmp_path),
        first_layer=0,
        last_layer=0,
        experts_per_layer=experts,
        packs=tuple(packs),
    ), expected


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


def test_native_scalex_mode_b_loads_compressed_record_and_weights(tmp_path):
    plan, expected = _scalex_plan(tmp_path)
    pool = NativeExpertPool(plan, workers=1)
    source = pool.layer(0)
    slots = source.allocate_slots(2)
    try:
        source.load_into(1, 0, slots)
        encoded, arrays = expected[1]
        record_row = np.array(slots[0][0])
        assert decode_mode_a(record_row[: len(encoded)].tobytes()) == b"".join(
            arrays[name].tobytes()
            for name in ("gate_proj.scales", "down_proj.scales", "up_proj.scales")
        )
        by_name = {
            name: np.array(slots[index][0])
            for index, (name, _, _) in enumerate(source.tensor_layouts)
            if name != "scalex_record"
        }
        for name in ("gate_proj.weight", "down_proj.weight", "up_proj.weight"):
            np.testing.assert_array_equal(by_name[name], arrays[name])
        assert plan.scalex_mode_b
        assert pool.stats().backend.endswith("ScaleX-Mode-B")
    finally:
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
        if preflight.expert_plan.scalex_mode_b:
            record = preflight.expert_plan.expert(3, 0).scalex_record
            assert record is not None
            raw_scales = decode_mode_a(
                np.array(by_name["scalex_record"], copy=False)[
                    : record.encoded_bytes
                ].tobytes()
            )
            cursor = 0
            for projection, length in zip(
                ("gate_proj", "down_proj", "up_proj"),
                record.scale_nbytes,
                strict=True,
            ):
                np.testing.assert_array_equal(
                    np.frombuffer(raw_scales[cursor : cursor + length], np.uint8),
                    np.array(expected.tensors[projection + ".scales"]).reshape(-1),
                )
                cursor += length
            for projection in ("gate_proj", "down_proj", "up_proj"):
                np.testing.assert_array_equal(
                    np.array(by_name[projection + ".weight"]),
                    np.array(expected.tensors[projection + ".weight"]),
                )
        else:
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
@pytest.mark.parametrize(
    ("token_count", "input_dtype"),
    ((1, mx.bfloat16), (2, mx.bfloat16), (2, mx.float32)),
)
def test_real_layer3_native_qmv_is_bit_exact_to_stock_gather_qmm(
    token_count,
    input_dtype,
):
    preflight = TargetRuntime.preflight(
        DEFAULT_MODEL_DIR,
        memory_gib=24,
        physical_bytes=256 * 2**30,
    )
    rng = np.random.default_rng(5301)
    x = mx.array(
        rng.normal(0, 0.2, (token_count, 4096)).astype(np.float32)
    ).astype(input_dtype)
    indices = mx.array([list(range(8))] * token_count, dtype=mx.uint32)

    with NativeExpertReader(preflight.expert_plan) as reader:
        reference = ExpertSSD(
            reader,
            layer=3,
            capacity=8,
            swiglu_limit=10.0,
        )
        reference_x = (
            x.astype(mx.bfloat16)
            if preflight.expert_plan.scalex_mode_b
            else x
        )
        expected = reference(reference_x, indices)
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


@pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / "VALIDATION.json").is_file(),
    reason="local composite checkpoint is absent",
)
def test_real_layer3_scalex_qmv_is_bit_exact_for_all_projections():
    preflight = TargetRuntime.preflight(
        DEFAULT_MODEL_DIR,
        memory_gib=24,
        physical_bytes=256 * 2**30,
    )
    with NativeExpertReader(preflight.expert_plan) as reader:
        expert = reader.load(3, 0)
    scale_raw = b"".join(
        np.array(expert.tensors[name]).tobytes()
        for name in (
            "gate_proj.scales",
            "down_proj.scales",
            "up_proj.scales",
        )
    )
    encoded = encode_mode_a(scale_raw)
    record = mx.array(np.frombuffer(encode_mode_b_row(encoded), np.uint8))[None]
    routes = mx.array([0], dtype=mx.uint32)
    for projection, name, width in (
        (0, "gate_proj", 4096),
        (1, "down_proj", 2048),
        (2, "up_proj", 4096),
    ):
        rng = np.random.default_rng(5310 + projection)
        x = mx.array(rng.normal(0, 0.2, (1, width)).astype(np.float32)).astype(
            mx.bfloat16
        )
        weight = expert.tensors[name + ".weight"][None]
        scales = expert.tensors[name + ".scales"][None]
        native = mx._expert_ssd_mxfp4_masked_qmv(x, weight, scales, routes)
        scalex = mx._expert_ssd_scalex_mxfp4_qmv(
            x, weight, record, routes, projection
        )
        mx.eval(native, scalex)
        np.testing.assert_array_equal(
            np.array(native.view(mx.uint16)),
            np.array(scalex.view(mx.uint16)),
        )


@pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / "VALIDATION.json").is_file(),
    reason="local composite checkpoint is absent",
)
def test_real_layer3_width2_pair_and_fused_reduce_are_valid():
    preflight = TargetRuntime.preflight(
        DEFAULT_MODEL_DIR,
        memory_gib=24,
        physical_bytes=256 * 2**30,
    )
    pool = NativeExpertPool(preflight.expert_plan, workers=8)
    native = NativeExpertSSD(
        pool,
        layer=3,
        capacity=16,
        swiglu_limit=10.0,
        wire_slots=False,
    )
    try:
        rng = np.random.default_rng(5320)
        x = mx.array(rng.normal(0, 0.2, (2, 4096)).astype(np.float32))
        indices = mx.array([list(range(8)), list(range(8, 16))], dtype=mx.uint32)
        plan = native.prepare(indices)
        shared = mx.zeros((2, 4096), dtype=mx.bfloat16)
        scores = mx.full((2, 8), 1 / 8, dtype=mx.float32)
        actual = native.finish_width2_merged(x, plan, scores, shared)
        mx.eval(actual)
        assert actual.shape == (2, 4096)
        assert bool(mx.all(mx.isfinite(actual)).item())
    finally:
        native.close()
        pool.close()
