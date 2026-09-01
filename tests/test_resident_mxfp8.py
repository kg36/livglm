import mlx.core as mx
import numpy as np

from glm53flash.layers import DeferredLinear


def test_deferred_linear_mxfp8_conversion_reduces_storage_and_stays_close():
    rng = np.random.default_rng(5308)
    layer = DeferredLinear(32, 64)
    layer.weight = mx.array(rng.normal(0, 0.05, (64, 32))).astype(mx.bfloat16)
    values = mx.array(rng.normal(0, 0.2, (2, 32))).astype(mx.bfloat16)
    expected = layer(values)
    mx.eval(expected)

    source_bytes, destination_bytes = layer.quantize_to_mxfp8()
    actual = layer(values)
    mx.eval(actual)

    assert source_bytes == 64 * 32 * 2
    assert destination_bytes == 64 * 32 + 64
    assert destination_bytes < source_bytes
    assert layer.quantization_mode == "mxfp8"
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=0.03,
        atol=0.015,
    )


def test_deferred_linear_mxfp4_conversion_reduces_storage_and_stays_close():
    rng = np.random.default_rng(5304)
    layer = DeferredLinear(32, 64)
    layer.weight = mx.array(rng.normal(0, 0.05, (64, 32))).astype(mx.bfloat16)
    values = mx.array(rng.normal(0, 0.2, (2, 32))).astype(mx.bfloat16)
    expected = layer(values)
    mx.eval(expected)

    source_bytes, destination_bytes = layer.quantize_to_mxfp4()
    actual = layer(values)
    mx.eval(actual)

    assert source_bytes == 64 * 32 * 2
    assert destination_bytes == 64 * 16 + 64
    assert destination_bytes < source_bytes
    assert layer.quantization_mode == "mxfp4"
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=0.20,
        atol=0.04,
    )
