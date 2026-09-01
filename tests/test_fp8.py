import numpy as np
import pytest

from glm53flash.contract import ContractError
from glm53flash.fp8 import decode_e4m3fn, dequantize_block_fp8


def test_e4m3fn_known_values():
    raw = bytes([0x00, 0x01, 0x38, 0x3C, 0x7E, 0xB8])
    actual = decode_e4m3fn(raw, (6,))
    expected = np.array([0.0, 2**-9, 1.0, 1.5, 448.0, -1.0], np.float32)
    np.testing.assert_array_equal(actual, expected)


def test_block_fp8_applies_scale():
    weights = bytes([0x38]) * (128 * 128)
    scales = np.array([[2.5]], dtype="<f4").tobytes()
    actual = dequantize_block_fp8(weights, (128, 128), scales, (1, 1))
    np.testing.assert_array_equal(actual, np.full((128, 128), 2.5, np.float32))


def test_block_fp8_scale_grid_maps_to_matrix_quadrants():
    weights = bytes([0x38]) * (256 * 256)
    scales = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="<f4")
    actual = dequantize_block_fp8(weights, (256, 256), scales.tobytes(), (2, 2))
    assert np.all(actual[:128, :128] == 1.0)
    assert np.all(actual[:128, 128:] == 2.0)
    assert np.all(actual[128:, :128] == 3.0)
    assert np.all(actual[128:, 128:] == 4.0)


def test_block_fp8_rejects_non_finite_code():
    weights = bytes([0x7F]) * (128 * 128)
    scales = np.array([[1.0]], dtype="<f4").tobytes()
    with pytest.raises(ContractError, match="non-finite"):
        dequantize_block_fp8(weights, (128, 128), scales, (1, 1))
