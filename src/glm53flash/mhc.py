"""Manifold-Constrained Hyper-Connections for GLM-5.3-Flash."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .layers import UnweightedRMSNorm, require_weight
from .model_config import GLMTextConfig


class HyperConnection(nn.Module):
    def __init__(self, config: GLMTextConfig):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = UnweightedRMSNorm(config.rms_norm_eps)
        self.fn = None
        self.base = None
        self.scale = None

    def __call__(self, streams: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        hc = self.hc_mult
        dtype = streams.dtype
        flat = self.input_norm(mx.flatten(streams, start_axis=2))
        fn = require_weight(self.fn, "hyper_connection.fn").astype(mx.float32)
        two_row = getattr(mx, "_expert_ssd_two_row_gemv", None)
        if two_row is not None and flat.size == 2 * flat.shape[-1]:
            mixed = two_row(mx.contiguous(flat), mx.contiguous(fn))
        else:
            mixed = flat @ fn.T
        pre_w, post_w, comb_w = mx.split(mixed, (hc, 2 * hc), axis=-1)
        base = require_weight(self.base, "hyper_connection.base").astype(mx.float32)
        pre_b, post_b, comb_b = mx.split(base, (hc, 2 * hc), axis=-1)
        scale = require_weight(self.scale, "hyper_connection.scale").astype(mx.float32)

        pre = mx.sigmoid(pre_w * scale[0] + pre_b) + self.hc_eps
        post = 2.0 * mx.sigmoid(post_w * scale[1] + post_b)
        comb_logits = comb_w.reshape(*comb_w.shape[:-1], hc, hc) * scale[2]
        comb_logits = comb_logits + comb_b.reshape(hc, hc)
        comb = mx.softmax(comb_logits, axis=-1) + self.hc_eps
        comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + self.hc_eps)
            comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + self.hc_eps)
        collapsed = mx.sum(pre[..., None] * streams, axis=2).astype(dtype)
        return post, comb, collapsed


def hyper_residual(
    residual: mx.array,
    sublayer_output: mx.array,
    post: mx.array,
    comb: mx.array,
) -> mx.array:
    dtype = residual.dtype
    placed = post.astype(dtype)[..., None] * sublayer_output[..., None, :]
    mixed = mx.matmul(mx.swapaxes(comb.astype(dtype), -1, -2), residual)
    return placed + mixed


class HyperHead(nn.Module):
    def __call__(self, streams: mx.array) -> mx.array:
        return mx.mean(streams, axis=2)
