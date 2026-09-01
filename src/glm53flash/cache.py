"""Explicit token-at-a-time cache records for the v1 runtime."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class KDACache:
    q_conv: mx.array
    k_conv: mx.array
    v_conv: mx.array
    recurrent: mx.array

    @classmethod
    def empty(
        cls,
        *,
        batch_size: int,
        qkv_dim: int,
        conv_kernel: int,
        num_heads: int,
        head_dim: int,
        dtype=mx.bfloat16,
    ) -> "KDACache":
        conv_shape = (batch_size, qkv_dim, conv_kernel - 1)
        return cls(
            q_conv=mx.zeros(conv_shape, dtype=dtype),
            k_conv=mx.zeros(conv_shape, dtype=dtype),
            v_conv=mx.zeros(conv_shape, dtype=dtype),
            recurrent=mx.zeros(
                (batch_size, num_heads, head_dim, head_dim),
                dtype=mx.float32,
            ),
        )


@dataclass
class DSACache:
    keys: mx.array | None = None
    values: mx.array | None = None

    @property
    def length(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]


LayerCache = KDACache | DSACache


@dataclass
class ModelCache:
    layers: list[LayerCache]
    position: int = 0

    def advance(self) -> None:
        self.position += 1
