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

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        *,
        bias: bool = False,
        quantize_resident: bool = True,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.weight = None
        self.scales = None
        self.quantization_mode = None
        self.quantize_resident = quantize_resident
        self.bias = None if bias else False

    def __call__(self, x: mx.array) -> mx.array:
        weight = require_weight(self.weight, "linear.weight")
        if self.quantization_mode in {"mxfp4", "mxfp8"}:
            bits = 4 if self.quantization_mode == "mxfp4" else 8
            two_row = getattr(mx, "_expert_ssd_mxfp4_two_row_qmv", None)
            if (
                bits == 4
                and two_row is not None
                and x.ndim >= 2
                and x.size == 2 * self.input_dims
                and self.input_dims % 512 == 0
                and self.output_dims % 8 == 0
            ):
                result = two_row(
                    mx.contiguous(x.reshape(2, self.input_dims)),
                    mx.contiguous(weight),
                    mx.contiguous(require_weight(self.scales, "linear.scales")),
                ).reshape(*x.shape[:-1], self.output_dims)
            else:
                result = mx.quantized_matmul(
                    x,
                    weight,
                    require_weight(self.scales, "linear.scales"),
                    transpose=True,
                    group_size=32,
                    bits=bits,
                    mode=self.quantization_mode,
                )
        else:
            result = x @ weight.T
        if self.bias is not False:
            result = result + require_weight(self.bias, "linear.bias")
        return result

    def quantize_to_mxfp(self, bits: int) -> tuple[int, int]:
        mode = f"mxfp{bits}"
        if bits not in {4, 8}:
            raise ContractError(f"unsupported resident quantization: {mode}")
        if not self.quantize_resident:
            raise ContractError(
                f"this deferred linear is excluded from resident {mode.upper()}"
            )
        if self.quantization_mode is not None:
            raise ContractError("deferred linear was already quantized")
        weight = require_weight(self.weight, "linear.weight")
        if weight.ndim != 2 or weight.shape[-1] % 32:
            raise ContractError(
                f"resident {mode.upper()} requires a group-32 matrix: "
                f"{tuple(weight.shape)}"
            )
        source_bytes = weight.nbytes
        packed, scales = mx.quantize(
            weight,
            group_size=32,
            bits=bits,
            mode=mode,
        )
        mx.eval(packed, scales)
        self.weight = packed
        self.scales = scales
        self.quantization_mode = mode
        return source_bytes, packed.nbytes + scales.nbytes

    def quantize_to_mxfp8(self) -> tuple[int, int]:
        return self.quantize_to_mxfp(8)

    def quantize_to_mxfp4(self) -> tuple[int, int]:
        return self.quantize_to_mxfp(4)


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
        return mx.fast.rms_norm(
            x,
            require_weight(self.weight, "rms_norm.weight"),
            self.eps,
        )


class UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x.astype(mx.float32), None, self.eps)


class GatedRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.weight = None

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        dtype = x.dtype
        normalized = mx.fast.rms_norm(
            x.astype(mx.float32),
            require_weight(self.weight, "gated_rms_norm.weight").astype(mx.float32),
            self.eps,
        )
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
