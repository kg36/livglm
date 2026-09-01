"""Small, dependency-free E4M3FN decoder for official block-FP8 weights."""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from .contract import ContractError


@lru_cache(maxsize=1)
def e4m3fn_lut() -> np.ndarray:
    """Return the IEEE-like finite E4M3FN value for every possible byte."""

    values = np.empty(256, dtype=np.float32)
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exponent = (byte >> 3) & 0x0F
        mantissa = byte & 0x07
        if exponent == 0:
            value = math.ldexp(mantissa / 8.0, -6)
        elif exponent == 0x0F and mantissa == 0x07:
            value = math.nan
        else:
            value = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
        values[byte] = sign * value
    values.setflags(write=False)
    return values


def decode_e4m3fn(raw: bytes | bytearray | memoryview, shape: tuple[int, ...]) -> np.ndarray:
    elements = 1
    for dimension in shape:
        elements *= int(dimension)
    source = np.frombuffer(raw, dtype=np.uint8)
    if source.size != elements:
        raise ContractError(f"E4M3 byte count mismatch: {source.size} != {elements}")
    return e4m3fn_lut()[source].reshape(shape)


def dequantize_block_fp8(
    weight_raw: bytes | bytearray | memoryview,
    weight_shape: tuple[int, int],
    scale_raw: bytes | bytearray | memoryview,
    scale_shape: tuple[int, int],
    *,
    block_size: int = 128,
) -> np.ndarray:
    """Decode one 2-D E4M3FN matrix and multiply its F32 block scales."""

    rows, columns = weight_shape
    if rows % block_size or columns % block_size:
        raise ContractError(
            f"block-FP8 shape must be divisible by {block_size}: {weight_shape}"
        )
    expected_scale_shape = (rows // block_size, columns // block_size)
    if scale_shape != expected_scale_shape:
        raise ContractError(
            f"block-FP8 scale shape mismatch: {scale_shape} != {expected_scale_shape}"
        )
    scales = np.frombuffer(scale_raw, dtype=np.dtype("<f4"))
    if scales.size != scale_shape[0] * scale_shape[1]:
        raise ContractError("block-FP8 scale byte count changed")
    scales = scales.reshape(scale_shape)
    decoded = decode_e4m3fn(weight_raw, weight_shape)
    blocked = decoded.reshape(
        scale_shape[0],
        block_size,
        scale_shape[1],
        block_size,
    )
    result = blocked * scales[:, None, :, None]
    result = result.reshape(weight_shape)
    if not np.isfinite(result).all():
        raise ContractError("block-FP8 dequantization produced a non-finite weight")
    return result
