"""Token-at-a-time Kimi Delta Attention used by GLM-5.3-Flash v1."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .cache import KDACache
from .contract import ContractError
from .layers import DeferredLinear, GatedRMSNorm, require_weight
from .model_config import GLMTextConfig


class DeferredDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = None

    def token(self, current: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        """Apply a causal depthwise convolution to one `[B, 1, C]` token."""
        if current.ndim != 3 or current.shape[1] != 1:
            raise ContractError("v1 KDA convolution accepts exactly one token")
        current_t = mx.swapaxes(current, 1, 2)
        window = mx.concatenate((state, current_t), axis=-1)
        weight = require_weight(self.weight, "depthwise_conv.weight").squeeze(1)
        output = mx.sum(window * weight[None, :, :], axis=-1)[:, None, :]
        return output, window[..., -(self.kernel_size - 1) :]


def l2_normalize(x: mx.array, eps: float = 1e-6) -> mx.array:
    x = x.astype(mx.float32)
    return x / mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)


class ForgetGate(nn.Module):
    def __init__(self, config: GLMTextConfig):
        super().__init__()
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.qkv_dim = config.qkv_dim
        self.f_a_proj = DeferredLinear(config.hidden_size, self.head_dim)
        self.f_b_proj = DeferredLinear(self.head_dim, self.qkv_dim)
        self.dt_bias = None
        self.A_log = None
        self.safe_gate_lower_bound = config.linear_gate_lower_bound

    def __call__(self, hidden_states: mx.array) -> mx.array:
        gate = self.f_b_proj(self.f_a_proj(hidden_states)).astype(mx.float32)
        gate = gate + require_weight(self.dt_bias, "forget_gate.dt_bias").astype(mx.float32)
        gate = gate.reshape(*hidden_states.shape[:2], self.num_heads, self.head_dim)
        decay = mx.exp(require_weight(self.A_log, "forget_gate.A_log").astype(mx.float32))
        decay = decay.reshape(1, 1, self.num_heads, 1)
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * mx.sigmoid(decay * gate)
        softplus = mx.where(gate > 20.0, gate, mx.log1p(mx.exp(gate)))
        return -decay * softplus


class KDALinearAttention(nn.Module):
    """Checkpoint-native, recurrent-only KDA; prefill also advances one token at a time."""

    def __init__(self, config: GLMTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        self.qkv_dim = config.qkv_dim
        self.conv_kernel_size = config.linear_conv_kernel

        self.q_proj = DeferredLinear(self.hidden_size, self.qkv_dim)
        self.k_proj = DeferredLinear(self.hidden_size, self.qkv_dim)
        self.v_proj = DeferredLinear(self.hidden_size, self.qkv_dim)
        self.q_conv1d = DeferredDepthwiseConv1d(self.qkv_dim, self.conv_kernel_size)
        self.k_conv1d = DeferredDepthwiseConv1d(self.qkv_dim, self.conv_kernel_size)
        self.v_conv1d = DeferredDepthwiseConv1d(self.qkv_dim, self.conv_kernel_size)

        # These names deliberately match the checkpoint's flattened forget-gate layout.
        self.f_a_proj = DeferredLinear(self.hidden_size, self.head_dim)
        self.f_b_proj = DeferredLinear(self.head_dim, self.qkv_dim)
        self.dt_bias = None
        self.A_log = None
        self.b_proj = DeferredLinear(self.hidden_size, self.num_heads)
        self.g_a_proj = DeferredLinear(self.hidden_size, self.head_dim)
        self.g_b_proj = DeferredLinear(self.head_dim, self.qkv_dim)
        self.o_norm = GatedRMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_proj = DeferredLinear(self.qkv_dim, self.hidden_size)
        self.safe_gate_lower_bound = config.linear_gate_lower_bound

    def empty_cache(self, batch_size: int, dtype=mx.bfloat16) -> KDACache:
        return KDACache.empty(
            batch_size=batch_size,
            qkv_dim=self.qkv_dim,
            conv_kernel=self.conv_kernel_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=dtype,
        )

    def _forget(self, x: mx.array) -> mx.array:
        gate = self.f_b_proj(self.f_a_proj(x)).astype(mx.float32)
        gate = gate + require_weight(self.dt_bias, "kda.dt_bias").astype(mx.float32)
        gate = gate.reshape(*x.shape[:2], self.num_heads, self.head_dim)
        decay = mx.exp(require_weight(self.A_log, "kda.A_log").astype(mx.float32))
        decay = decay.reshape(1, 1, self.num_heads, 1)
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * mx.sigmoid(decay * gate)
        softplus = mx.where(gate > 20.0, gate, mx.log1p(mx.exp(gate)))
        return -decay * softplus

    def __call__(self, hidden_states: mx.array, cache: KDACache) -> mx.array:
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ContractError("v1 KDA is token-at-a-time only")
        batch_size = hidden_states.shape[0]
        hidden_shape = (batch_size, 1, self.num_heads, self.head_dim)

        query, cache.q_conv = self.q_conv1d.token(self.q_proj(hidden_states), cache.q_conv)
        key, cache.k_conv = self.k_conv1d.token(self.k_proj(hidden_states), cache.k_conv)
        value, cache.v_conv = self.v_conv1d.token(self.v_proj(hidden_states), cache.v_conv)
        query = mx.sigmoid(query) * query
        key = mx.sigmoid(key) * key
        value = mx.sigmoid(value) * value

        query = l2_normalize(query.reshape(hidden_shape)) / math.sqrt(self.head_dim)
        key = l2_normalize(key.reshape(hidden_shape))
        value = value.reshape(hidden_shape).astype(mx.float32)
        forget = self._forget(hidden_states).astype(mx.float32)
        beta = mx.sigmoid(self.b_proj(hidden_states).astype(mx.float32))

        q_i = query[:, 0]
        k_i = key[:, 0]
        v_i = value[:, 0]
        state = cache.recurrent * mx.exp(forget[:, 0])[..., None]
        recalled = mx.sum(state * k_i[..., None], axis=-2)
        delta = (v_i - recalled) * beta[:, 0, :, None]
        state = state + k_i[..., None] * delta[..., None, :]
        core = mx.sum(state * q_i[..., None], axis=-2)[:, None]
        cache.recurrent = state

        gate = self.g_b_proj(self.g_a_proj(hidden_states)).reshape(hidden_shape)
        output = self.o_norm(core.astype(hidden_states.dtype), gate)
        return self.o_proj(output.reshape(batch_size, 1, self.qkv_dim))
