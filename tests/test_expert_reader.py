from pathlib import Path

import numpy as np

import glm53flash.expert_reader as reader_module
from glm53flash.expert_reader import NativeExpertReader
from glm53flash.expert_source import (
    ExpertReadRange,
    ExpertTensorSource,
    NativeExpertSource,
    NativeExpertSourcePlan,
)


def test_native_reader_retries_short_preads_and_keeps_one_logical_range(tmp_path, monkeypatch):
    tensors = []
    payload = bytearray()
    offset = 0
    for projection in ("down_proj", "gate_proj", "up_proj"):
        for part, dtype, shape, byte_count in (
            ("weight", "U32", (32, 4), 512),
            ("scales", "U8", (32, 1), 32),
        ):
            raw = bytes([(offset // 32) & 0xFF]) * byte_count
            payload.extend(raw)
            tensors.append(
                ExpertTensorSource(
                    source_name=f"{projection}.{part}",
                    destination_name=f"{projection}.{part}",
                    shard_name="expert.bin",
                    absolute_offset=offset,
                    byte_length=byte_count,
                    source_shape=shape,
                    mlx_dtype=dtype,
                    mlx_shape=shape,
                )
            )
            offset += byte_count
    (tmp_path / "expert.bin").write_bytes(payload)
    source_range = ExpertReadRange("expert.bin", 0, len(payload), tuple(tensors))
    pack = NativeExpertSource(0, 0, tuple(tensors), (source_range,))
    plan = NativeExpertSourcePlan(str(tmp_path), 0, 0, 1, (pack,))

    original_pread = reader_module.os.pread

    def short_pread(fd, length, absolute_offset):
        return original_pread(fd, min(length, 97), absolute_offset)

    monkeypatch.setattr(reader_module.os, "pread", short_pread)
    with NativeExpertReader(plan) as reader:
        expert = reader.load(0, 0)
        stats = reader.stats()
        assert stats.logical_reads == 1
        assert stats.system_reads > 1
        assert stats.read_bytes == len(payload)
        assert expert.payload_bytes == len(payload)
        assert expert.projection("gate_proj")[0].shape == (32, 4)
