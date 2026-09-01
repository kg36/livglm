"""Small deferred-weight MLX layers used by the GLM text graph."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .contract import ContractError
from .expert_ssd import limited_swiglu


def require_weight(value: mx.array | None, name: str) -> mx.array:
    if value is None:
        raise ContractError(f"resident parameter was not loaded: {name}")
    return value


class DeferredLinear(nn.Module):
    """A linear layer whose storage is attached by the streamed loader."""

    def __init__(self, input_dims: int, output_dims: int, *, bias: bool = False):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.weight = None
        self.bias = None if bias else False

    def __call__(self, x: mx.array) -> mx.array:
        weight = require_weight(self.weight, "linear.weight")
        result = x @ weight.T
        if self.bias is not False:
            result = result + require_weight(self.bias, "linear.bias")
        return result


class DeferredEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weight = None

    def __call__(self, token_ids: mx.array) -> mx.array:
        return require_weight(self.weight, "embedding.weight")[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.weight = None

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x32 = x.astype(mx.float32)
        normalized = x32 * mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + self.eps)
        return normalized.astype(dtype) * require_weight(self.weight, "rms_norm.weight")


class UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x32 = x.astype(mx.float32)
        return x32 * mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + self.eps)


class GatedRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.weight = None

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        dtype = x.dtype
        x32 = x.astype(mx.float32)
        normalized = x32 * mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + self.eps)
        normalized = normalized * require_weight(self.weight, "gated_rms_norm.weight").astype(mx.float32)
        return (normalized * mx.sigmoid(gate.astype(mx.float32))).astype(dtype)


class DenseMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float):
        super().__init__()
        self.gate_proj = DeferredLinear(hidden_size, intermediate_size)
        self.up_proj = DeferredLinear(hidden_size, intermediate_size)
        self.down_proj = DeferredLinear(intermediate_size, hidden_size)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            limited_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit)
        )


class DeferredLayerNorm(nn.Module):
    """LayerNorm used only by the dormant short-context DSA indexer."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.weight = None
        self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        x32 = x.astype(mx.float32)
        mean = mx.mean(x32, axis=-1, keepdims=True)
        variance = mx.mean(mx.square(x32 - mean), axis=-1, keepdims=True)
        normalized = (x32 - mean) * mx.rsqrt(variance + self.eps)
        return (
            normalized * require_weight(self.weight, "layer_norm.weight").astype(mx.float32)
            + require_weight(self.bias, "layer_norm.bias").astype(mx.float32)
        ).astype(x.dtype)
