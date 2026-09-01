import mlx.core as mx
import numpy as np

from glm53flash.kda import KDALinearAttention
from tests.helpers import tiny_config


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


def test_recurrent_kda_matches_numpy_at_every_token():
    config = tiny_config(
        hidden_size=8,
        linear_num_heads=2,
        linear_head_dim=4,
        linear_conv_kernel=3,
    )
    module = KDALinearAttention(config, 0)
    rng = np.random.default_rng(12)

    def matrix(rows, cols, scale=0.15):
        return rng.normal(0, scale, (rows, cols)).astype(np.float32)

    weights = {
        "q": matrix(8, 8),
        "k": matrix(8, 8),
        "v": matrix(8, 8),
        "qconv": matrix(8, 3),
        "kconv": matrix(8, 3),
        "vconv": matrix(8, 3),
        "fa": matrix(4, 8),
        "fb": matrix(8, 4),
        "b": matrix(2, 8),
        "ga": matrix(4, 8),
        "gb": matrix(8, 4),
        "onorm": rng.uniform(0.8, 1.2, 4).astype(np.float32),
        "o": matrix(8, 8),
        "dt": rng.normal(0, 0.1, 8).astype(np.float32),
        "A": rng.normal(-1.0, 0.1, 2).astype(np.float32),
    }
    module.q_proj.weight = mx.array(weights["q"])
    module.k_proj.weight = mx.array(weights["k"])
    module.v_proj.weight = mx.array(weights["v"])
    module.q_conv1d.weight = mx.array(weights["qconv"][:, None])
    module.k_conv1d.weight = mx.array(weights["kconv"][:, None])
    module.v_conv1d.weight = mx.array(weights["vconv"][:, None])
    module.f_a_proj.weight = mx.array(weights["fa"])
    module.f_b_proj.weight = mx.array(weights["fb"])
    module.b_proj.weight = mx.array(weights["b"])
    module.g_a_proj.weight = mx.array(weights["ga"])
    module.g_b_proj.weight = mx.array(weights["gb"])
    module.o_norm.weight = mx.array(weights["onorm"])
    module.o_proj.weight = mx.array(weights["o"])
    module.dt_bias = mx.array(weights["dt"])
    module.A_log = mx.array(weights["A"])

    cache = module.empty_cache(1, dtype=mx.float32)
    conv_states = {name: np.zeros((1, 8, 2), np.float32) for name in ("q", "k", "v")}
    recurrent = np.zeros((1, 2, 4, 4), np.float32)
    inputs = rng.normal(0, 0.2, (3, 8)).astype(np.float32)

    for token in inputs:
        x = token.reshape(1, 1, 8)
        projected = {}
        for name in ("q", "k", "v"):
            current = (x @ weights[name].T).transpose(0, 2, 1)
            window = np.concatenate((conv_states[name], current), axis=-1)
            projected[name] = _silu((window * weights[name + "conv"][None]).sum(-1)[:, None])
            conv_states[name] = window[..., -2:]

        q = projected["q"].reshape(1, 1, 2, 4).astype(np.float32)
        k = projected["k"].reshape(1, 1, 2, 4).astype(np.float32)
        v = projected["v"].reshape(1, 1, 2, 4).astype(np.float32)
        q /= np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
        q /= 2.0
        k /= np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        raw_g = ((x @ weights["fa"].T) @ weights["fb"].T + weights["dt"]).reshape(1, 1, 2, 4)
        forget = -5.0 * _sigmoid(np.exp(weights["A"])[None, None, :, None] * raw_g)
        beta = _sigmoid(x @ weights["b"].T)
        recurrent *= np.exp(forget[:, 0])[..., None]
        recalled = (recurrent * k[:, 0, :, :, None]).sum(-2)
        delta = (v[:, 0] - recalled) * beta[:, 0, :, None]
        recurrent += k[:, 0, :, :, None] * delta[..., None, :]
        core = (recurrent * q[:, 0, :, :, None]).sum(-2)[:, None]
        gate = ((x @ weights["ga"].T) @ weights["gb"].T).reshape(1, 1, 2, 4)
        norm = core / np.sqrt((core * core).mean(-1, keepdims=True) + config.rms_norm_eps)
        norm = norm * weights["onorm"] * _sigmoid(gate)
        expected = norm.reshape(1, 1, 8) @ weights["o"].T

        actual = module(mx.array(x), cache)
        mx.eval(actual, cache.recurrent, cache.q_conv, cache.k_conv, cache.v_conv)
        np.testing.assert_allclose(np.array(actual), expected, rtol=2e-3, atol=2e-4)
        np.testing.assert_allclose(np.array(cache.recurrent), recurrent, rtol=2e-3, atol=2e-4)
        np.testing.assert_allclose(np.array(cache.q_conv), conv_states["q"], rtol=1e-5, atol=1e-6)
