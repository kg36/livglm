"""Lossless ScaleX codec for MXFP4 E8M0 scale streams."""

from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np


MODE_A_MAGIC = b"LSA1"
MODE_A_HEADER = struct.Struct("<4sB4B3xIII")

A_RAW = 0
A_ONE_BIT = 1
A_TWO_BIT = 2
MODE_B_TILE_SCALES = 512


class ScaleCodecError(ValueError):
    """Raised when a ScaleX record is malformed."""


@dataclass(frozen=True)
class ModeAInfo:
    raw_bytes: int
    encoded_bytes: int
    codec: str
    palette: tuple[int, int, int, int]
    primary_bytes: int
    exceptions: int


def encode_mode_a(source: bytes) -> bytes:
    """Encode one complete expert's E8M0 scales without changing any value."""

    values = np.frombuffer(source, dtype=np.uint8)
    counts = np.bincount(values, minlength=256)
    ordered = np.array(
        sorted(range(256), key=lambda value: (-int(counts[value]), value))[:4],
        dtype=np.uint8,
    )
    palette = tuple(int(value) for value in ordered)
    candidates: list[tuple[int, int, bytes, list[tuple[int, int]]]] = []
    for codec, bits, width in (
        (A_ONE_BIT, 1, 2),
        (A_TWO_BIT, 2, 4),
    ):
        codes = np.zeros(values.size, dtype=np.uint8)
        represented = np.zeros(values.size, dtype=bool)
        for code, value in enumerate(ordered[:width]):
            mask = values == value
            codes[mask] = code
            represented |= mask
        if bits == 1:
            primary = np.packbits(codes, bitorder="little").tobytes()
        else:
            padding = (-codes.size) % 4
            packed_codes = np.pad(codes, (0, padding)) if padding else codes
            groups = packed_codes.reshape(-1, 4).astype(np.uint16)
            primary = (
                groups[:, 0]
                | (groups[:, 1] << 2)
                | (groups[:, 2] << 4)
                | (groups[:, 3] << 6)
            ).astype(np.uint8).tobytes()
        exception_positions = np.flatnonzero(~represented).astype("<u4")
        exception_values = values[~represented]
        exceptions = list(
            zip(exception_positions.tolist(), exception_values.tolist(), strict=True)
        )
        candidates.append(
            (len(primary) + 5 * len(exceptions), codec, primary, exceptions)
        )

    raw_candidate = (len(source), A_RAW, source, [])
    _, codec, primary, exceptions = min(
        (*candidates, raw_candidate), key=lambda candidate: candidate[0]
    )
    header = MODE_A_HEADER.pack(
        MODE_A_MAGIC,
        codec,
        *palette,
        len(source),
        len(primary),
        len(exceptions),
    )
    positions = b"".join(struct.pack("<I", position) for position, _ in exceptions)
    exception_values = bytes(value for _, value in exceptions)
    return header + primary + positions + exception_values


def inspect_mode_a(encoded: bytes | bytearray | memoryview) -> ModeAInfo:
    if len(encoded) < MODE_A_HEADER.size:
        raise ScaleCodecError("truncated Mode-A header")
    magic, codec, p0, p1, p2, p3, raw_size, primary_size, exceptions = (
        MODE_A_HEADER.unpack_from(encoded)
    )
    if magic != MODE_A_MAGIC or codec not in (A_RAW, A_ONE_BIT, A_TWO_BIT):
        raise ScaleCodecError("invalid Mode-A header")
    expected = MODE_A_HEADER.size + primary_size + 5 * exceptions
    if expected != len(encoded):
        raise ScaleCodecError("invalid Mode-A record length")
    codec_name = {A_RAW: "raw", A_ONE_BIT: "one_bit", A_TWO_BIT: "two_bit"}[codec]
    return ModeAInfo(
        raw_bytes=raw_size,
        encoded_bytes=len(encoded),
        codec=codec_name,
        palette=(p0, p1, p2, p3),
        primary_bytes=primary_size,
        exceptions=exceptions,
    )


def mode_b_tile_prefix(encoded: bytes | bytearray | memoryview) -> bytes:
    """Build the transient uint16 tile directory consumed by the Metal QMV."""

    info = inspect_mode_a(encoded)
    if info.exceptions > np.iinfo(np.uint16).max:
        raise ScaleCodecError("Mode-B exception prefix exceeds uint16")
    positions_offset = MODE_A_HEADER.size + info.primary_bytes
    positions = np.frombuffer(
        encoded,
        dtype="<u4",
        count=info.exceptions,
        offset=positions_offset,
    )
    if info.exceptions and (
        positions[-1] >= info.raw_bytes
        or np.any(positions[1:] <= positions[:-1])
    ):
        raise ScaleCodecError("invalid Mode-A exception position")
    tile_count = (
        info.raw_bytes + MODE_B_TILE_SCALES - 1
    ) // MODE_B_TILE_SCALES
    boundaries = np.minimum(
        np.arange(tile_count + 1, dtype=np.uint64) * MODE_B_TILE_SCALES,
        info.raw_bytes,
    )
    return np.searchsorted(positions, boundaries, side="left").astype("<u2").tobytes()


def mode_b_indexed_nbytes(encoded_bytes: int, raw_bytes: int) -> int:
    if encoded_bytes < MODE_A_HEADER.size or raw_bytes < 1:
        raise ScaleCodecError("invalid Mode-B row geometry")
    tile_count = (raw_bytes + MODE_B_TILE_SCALES - 1) // MODE_B_TILE_SCALES
    return encoded_bytes + (encoded_bytes & 1) + 2 * (tile_count + 1)


def encode_mode_b_row(encoded: bytes) -> bytes:
    """Append the transient tile directory used by a resident Mode-B row."""

    padding = b"\0" * (len(encoded) & 1)
    return encoded + padding + mode_b_tile_prefix(encoded)


def decode_mode_a(encoded: bytes | bytearray | memoryview) -> bytes:
    info = inspect_mode_a(encoded)
    codec = encoded[4]
    primary_start = MODE_A_HEADER.size
    primary = memoryview(encoded)[primary_start : primary_start + info.primary_bytes]
    if codec == A_RAW:
        if info.primary_bytes != info.raw_bytes or info.exceptions:
            raise ScaleCodecError("invalid raw Mode-A record")
        return bytes(primary)

    bits = 1 if codec == A_ONE_BIT else 2
    palette = info.palette[: 2 if bits == 1 else 4]
    minimum_primary = (info.raw_bytes * bits + 7) // 8
    if info.primary_bytes != minimum_primary:
        raise ScaleCodecError("invalid Mode-A primary length")
    packed = np.frombuffer(primary, dtype=np.uint8)
    if bits == 1:
        codes = np.unpackbits(packed, bitorder="little")[: info.raw_bytes]
    else:
        codes = np.empty(packed.size * 4, dtype=np.uint8)
        codes[0::4] = packed & 3
        codes[1::4] = (packed >> 2) & 3
        codes[2::4] = (packed >> 4) & 3
        codes[3::4] = packed >> 6
        codes = codes[: info.raw_bytes]
    output = np.array(palette, dtype=np.uint8)[codes]
    positions_start = primary_start + info.primary_bytes
    values_start = positions_start + 4 * info.exceptions
    positions = np.frombuffer(
        encoded,
        dtype="<u4",
        count=info.exceptions,
        offset=positions_start,
    )
    if info.exceptions:
        if positions[-1] >= info.raw_bytes or np.any(positions[1:] <= positions[:-1]):
            raise ScaleCodecError("invalid Mode-A exception position")
        exception_values = np.frombuffer(
            encoded,
            dtype=np.uint8,
            count=info.exceptions,
            offset=values_start,
        )
        output[positions] = exception_values
    return output.tobytes()
