import numpy as np
import pytest

from glm53flash.scale_codec import (
    ScaleCodecError,
    decode_mode_a,
    encode_mode_a,
    encode_mode_b_row,
    inspect_mode_a,
    mode_b_indexed_nbytes,
)


@pytest.mark.parametrize(
    ("values", "codec"),
    [
        (np.resize(np.array([120, 121], np.uint8), 4096), "one_bit"),
        (np.resize(np.array([118, 119, 120, 121], np.uint8), 4096), "two_bit"),
        (np.arange(256, dtype=np.uint8).repeat(16), "raw"),
    ],
)
def test_scalex_round_trip_and_mode_selection(values, codec):
    raw = values.tobytes()
    encoded = encode_mode_a(raw)
    info = inspect_mode_a(encoded)
    assert info.codec == codec
    assert info.raw_bytes == len(raw)
    assert decode_mode_a(encoded) == raw
    row = encode_mode_b_row(encoded)
    assert len(row) == mode_b_indexed_nbytes(len(encoded), len(raw))


def test_scalex_rejects_truncated_record():
    with pytest.raises(ScaleCodecError, match="truncated"):
        decode_mode_a(b"LSA1")
