import numpy as np
import mlx.core as mx

from glm53flash.mhc import HyperConnection, hyper_residual
from tests.helpers import tiny_config


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def test_hyper_connection_matches_reference():
    config = tiny_config(hidden_size=2, hc_mult=2, hc_sinkhorn_iters=3)
    layer = HyperConnection(config)
    streams = np.array([[[[0.5, -1.0], [1.5, 0.25]]]], np.float32)
    fn = np.arange(8 * 4, dtype=np.float32).reshape(8, 4) / 80.0 - 0.2
    base = np.linspace(-0.2, 0.2, 8, dtype=np.float32)
    scale = np.array([0.5, 0.75, 1.25], np.float32)
    layer.fn, layer.base, layer.scale = mx.array(fn), mx.array(base), mx.array(scale)

    post, comb, collapsed = layer(mx.array(streams))

    flat = streams.reshape(1, 1, 4).astype(np.float32)
    flat /= np.sqrt(np.mean(flat * flat, axis=-1, keepdims=True) + config.rms_norm_eps)
    mixed = flat @ fn.T
    pre_w, post_w, comb_w = np.split(mixed, [2, 4], axis=-1)
    pre_b, post_b, comb_b = np.split(base, [2, 4])
    pre = _sigmoid(pre_w * scale[0] + pre_b) + config.hc_eps
    expected_post = 2 * _sigmoid(post_w * scale[1] + post_b)
    logits = comb_w.reshape(1, 1, 2, 2) * scale[2] + comb_b.reshape(2, 2)
    logits -= logits.max(axis=-1, keepdims=True)
    expected_comb = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    expected_comb += config.hc_eps
    expected_comb /= expected_comb.sum(axis=-2, keepdims=True) + config.hc_eps
    for _ in range(2):
        expected_comb /= expected_comb.sum(axis=-1, keepdims=True) + config.hc_eps
        expected_comb /= expected_comb.sum(axis=-2, keepdims=True) + config.hc_eps
    expected_collapsed = (pre[..., None] * streams).sum(axis=2)

    np.testing.assert_allclose(np.array(post), expected_post, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.array(comb), expected_comb, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.array(collapsed), expected_collapsed, rtol=1e-5, atol=1e-6)

    sublayer = mx.array([[[0.2, -0.3]]], dtype=mx.float32)
    result = hyper_residual(mx.array(streams), sublayer, post, comb)
    expected = expected_post[..., None] * np.array(sublayer)[..., None, :]
    expected += np.swapaxes(expected_comb, -1, -2) @ streams
    # MLX's float32 Metal matmul uses the platform's fast accumulation path.
    np.testing.assert_allclose(np.array(result), expected, rtol=1e-3, atol=1e-4)
