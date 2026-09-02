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

    def advance(self, tokens: int = 1) -> None:
        self.position += int(tokens)

    def snapshot(self) -> tuple[int, tuple[tuple[object, ...], ...]]:
        """Capture array identities for an exact speculative rollback."""

        layers: list[tuple[object, ...]] = []
        for cache in self.layers:
            if isinstance(cache, KDACache):
                layers.append(
                    ("kda", cache.q_conv, cache.k_conv, cache.v_conv, cache.recurrent)
                )
            else:
                layers.append(("dsa", cache.keys, cache.values))
        return self.position, tuple(layers)

    def restore(self, snapshot: tuple[int, tuple[tuple[object, ...], ...]]) -> None:
        position, layers = snapshot
        if len(layers) != len(self.layers):
            raise ValueError("cache snapshot layer inventory changed")
        for cache, values in zip(self.layers, layers, strict=True):
            if isinstance(cache, KDACache) and values[0] == "kda":
                _, cache.q_conv, cache.k_conv, cache.v_conv, cache.recurrent = values
            elif isinstance(cache, DSACache) and values[0] == "dsa":
                _, cache.keys, cache.values = values
            else:
                raise ValueError("cache snapshot type changed")
        self.position = int(position)

    def commit_first_from_wide(
        self,
        snapshot: tuple[int, tuple[tuple[object, ...], ...]],
    ) -> None:
        """Commit only verifier row zero without replaying the target."""

        position, _ = snapshot
        for cache in self.layers:
            values = getattr(cache, "_speculative_first", None)
            if values is None:
                raise ValueError("wide target cache did not stage its first row")
            if isinstance(cache, KDACache):
                cache.q_conv, cache.k_conv, cache.v_conv, cache.recurrent = values
            else:
                cache.keys, cache.values = values
            delattr(cache, "_speculative_first")
        self.position = int(position) + 1
