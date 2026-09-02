"""Transactional, byte-reversible ScaleX containers for GLM expert shards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
from typing import Any, BinaryIO, Iterator

from .scale_codec import (
    decode_mode_a,
    encode_mode_a,
    inspect_mode_a,
    mode_b_indexed_nbytes,
    mode_b_tile_prefix,
)


SCALEX_FORMAT = "livglm_scalex_layer_v1"
SCALEX_MAGIC = b"LVGLMSX1"
SCALEX_VERSION = 1
SCALEX_PREFIX = struct.Struct("<8sIIQ")
COPY_CHUNK_BYTES = 16 * 2**20
MINIMUM_FREE_MARGIN_BYTES = 1 * 2**30

_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(down_proj|gate_proj|up_proj)\.(weight_packed|weight_scale)$"
)
_SCALE_COMPONENTS = ("gate_proj", "down_proj", "up_proj")
_WEIGHT_COMPONENTS = _SCALE_COMPONENTS


class ScaleXContainerError(RuntimeError):
    """A ScaleX shard is malformed or failed a transactional operation."""


@dataclass(frozen=True)
class ScaleXRecord:
    expert: int
    absolute_offset: int
    encoded_bytes: int
    decoded_bytes: int
    scale_names: tuple[str, str, str]
    scale_nbytes: tuple[int, int, int]
    weight_names: tuple[str, str, str]
    weight_nbytes: tuple[int, int, int]
    codec: str

    @property
    def physical_bytes(self) -> int:
        return self.encoded_bytes + sum(self.weight_nbytes)

    @property
    def mode_b_row_bytes(self) -> int:
        return mode_b_indexed_nbytes(self.encoded_bytes, self.decoded_bytes)


@dataclass(frozen=True)
class ScaleXLayout:
    path: Path
    metadata: dict[str, Any]
    data_base: int
    original_prefix: bytes
    virtual_header: dict[str, Any]
    records: tuple[ScaleXRecord, ...]

    @property
    def layer(self) -> int:
        return int(self.metadata["layer"])

    @property
    def experts(self) -> int:
        return len(self.records)

    @property
    def original_bytes(self) -> int:
        return int(self.metadata["original_bytes"])

    @property
    def original_sha256(self) -> str:
        return str(self.metadata["original_sha256"])

    @property
    def virtual_data_base(self) -> int:
        return len(self.original_prefix)

    @property
    def maximum_mode_b_row_bytes(self) -> int:
        return max(record.mode_b_row_bytes for record in self.records)

    def record(self, expert: int) -> ScaleXRecord:
        if not 0 <= expert < len(self.records):
            raise ScaleXContainerError(f"ScaleX expert is out of range: {expert}")
        record = self.records[expert]
        if record.expert != expert:
            raise ScaleXContainerError("ScaleX record ordering changed")
        return record


@dataclass(frozen=True)
class ScaleXReport:
    operation: str
    layer: int
    path: str
    experts: int
    original_bytes: int
    stored_bytes: int
    bytes_saved: int
    original_sha256: str
    reconstructed_sha256: str
    byte_identical: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(COPY_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _read_safetensors_prefix(path: Path) -> tuple[bytes, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ScaleXContainerError(f"truncated safetensors file: {path}")
        header_bytes = struct.unpack("<Q", raw_length)[0]
        if not 2 <= header_bytes <= 512 * 2**20:
            raise ScaleXContainerError(f"implausible safetensors header: {path}")
        raw_header = handle.read(header_bytes)
    if len(raw_header) != header_bytes:
        raise ScaleXContainerError(f"truncated safetensors header: {path}")
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise ScaleXContainerError(f"invalid safetensors header: {path}") from exc
    if not isinstance(header, dict):
        raise ScaleXContainerError(f"safetensors header is not an object: {path}")
    return raw_length + raw_header, header


def _tensor_range(header: dict[str, Any], name: str, data_base: int) -> tuple[int, int]:
    entry = header.get(name)
    if not isinstance(entry, dict):
        raise ScaleXContainerError(f"missing tensor in expert shard: {name}")
    try:
        start, end = (int(value) for value in entry["data_offsets"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ScaleXContainerError(f"invalid tensor offsets: {name}") from exc
    if start < 0 or end <= start:
        raise ScaleXContainerError(f"invalid tensor range: {name}")
    return data_base + start, end - start


def _expert_name(layer: int, expert: int, component: str, part: str) -> str:
    return (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}."
        f"{component}.{part}"
    )


def _validate_source_inventory(
    path: Path,
    header: dict[str, Any],
    *,
    layer: int,
    experts: int,
    data_base: int,
) -> None:
    expected = {
        _expert_name(layer, expert, component, part)
        for expert in range(experts)
        for component in _SCALE_COMPONENTS
        for part in ("weight_packed", "weight_scale")
    }
    actual = set(header) - {"__metadata__"}
    if actual != expected:
        raise ScaleXContainerError(
            f"expert shard inventory changed: missing={len(expected - actual)}, "
            f"extra={len(actual - expected)}"
        )
    ordered: list[tuple[int, int, str]] = []
    for name in actual:
        entry = header[name]
        if not isinstance(entry, dict) or entry.get("dtype") != "U8":
            raise ScaleXContainerError(f"ScaleX source tensor must be U8: {name}")
        absolute, length = _tensor_range(header, name, data_base)
        ordered.append((absolute, length, name))
    cursor = data_base
    for absolute, length, name in sorted(ordered):
        if absolute != cursor:
            raise ScaleXContainerError(f"expert shard has a gap before {name}")
        cursor += length
    if cursor != path.stat().st_size:
        raise ScaleXContainerError("expert shard payload does not cover the complete file")


def is_scalex_layer(path: str | Path) -> bool:
    source = Path(path)
    try:
        with source.open("rb") as handle:
            return handle.read(len(SCALEX_MAGIC)) == SCALEX_MAGIC
    except FileNotFoundError:
        return False


def _serialize_metadata(metadata: dict[str, Any]) -> bytes:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_scalex_layout(path: str | Path) -> ScaleXLayout:
    source = Path(path).expanduser().resolve()
    with source.open("rb") as handle:
        prefix = handle.read(SCALEX_PREFIX.size)
        if len(prefix) != SCALEX_PREFIX.size:
            raise ScaleXContainerError(f"truncated ScaleX prefix: {source}")
        magic, version, metadata_bytes, stored_bytes = SCALEX_PREFIX.unpack(prefix)
        if magic != SCALEX_MAGIC or version != SCALEX_VERSION:
            raise ScaleXContainerError(f"unsupported ScaleX container: {source}")
        if metadata_bytes < 2 or metadata_bytes > 64 * 2**20:
            raise ScaleXContainerError("invalid ScaleX metadata length")
        raw_metadata = handle.read(metadata_bytes)
        if len(raw_metadata) != metadata_bytes:
            raise ScaleXContainerError("truncated ScaleX metadata")
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError as exc:
            raise ScaleXContainerError("invalid ScaleX metadata") from exc
        if (
            not isinstance(metadata, dict)
            or metadata.get("format") != SCALEX_FORMAT
            or int(metadata.get("version", -1)) != SCALEX_VERSION
        ):
            raise ScaleXContainerError("unexpected ScaleX metadata format")
        data_base = SCALEX_PREFIX.size + metadata_bytes
        if data_base + int(stored_bytes) != source.stat().st_size:
            raise ScaleXContainerError("ScaleX stored length differs from file size")
        original_prefix_bytes = int(metadata.get("original_prefix_bytes", -1))
        if original_prefix_bytes < 10 or original_prefix_bytes > stored_bytes:
            raise ScaleXContainerError("invalid embedded safetensors prefix length")
        original_prefix = handle.read(original_prefix_bytes)
    if len(original_prefix) != original_prefix_bytes:
        raise ScaleXContainerError("truncated embedded safetensors prefix")
    if hashlib.sha256(original_prefix).hexdigest() != metadata.get("original_prefix_sha256"):
        raise ScaleXContainerError("embedded safetensors prefix hash changed")
    header_bytes = struct.unpack("<Q", original_prefix[:8])[0]
    if header_bytes + 8 != len(original_prefix):
        raise ScaleXContainerError("embedded safetensors prefix geometry changed")
    try:
        virtual_header = json.loads(original_prefix[8:])
    except json.JSONDecodeError as exc:
        raise ScaleXContainerError("invalid embedded safetensors header") from exc
    if not isinstance(virtual_header, dict):
        raise ScaleXContainerError("embedded safetensors header is not an object")

    raw_records = metadata.get("records")
    experts = int(metadata.get("experts", -1))
    if not isinstance(raw_records, list) or experts < 1 or len(raw_records) != experts:
        raise ScaleXContainerError("ScaleX record inventory changed")
    records: list[ScaleXRecord] = []
    cursor = original_prefix_bytes
    for expert, raw in enumerate(raw_records):
        if not isinstance(raw, dict) or int(raw.get("expert", -1)) != expert:
            raise ScaleXContainerError("ScaleX expert ordering changed")
        stored_offset = int(raw.get("stored_offset", -1))
        encoded_bytes = int(raw.get("encoded_bytes", -1))
        decoded_bytes = int(raw.get("decoded_bytes", -1))
        scale_names = raw.get("scale_names")
        scale_nbytes = raw.get("scale_nbytes")
        weight_names = raw.get("weight_names")
        weight_nbytes = raw.get("weight_nbytes")
        if (
            stored_offset != cursor
            or encoded_bytes < 24
            or decoded_bytes < 1
            or not isinstance(scale_names, list)
            or not isinstance(scale_nbytes, list)
            or not isinstance(weight_names, list)
            or not isinstance(weight_nbytes, list)
            or not all(len(value) == 3 for value in (scale_names, scale_nbytes, weight_names, weight_nbytes))
        ):
            raise ScaleXContainerError("invalid ScaleX expert record")
        record = ScaleXRecord(
            expert=expert,
            absolute_offset=data_base + stored_offset,
            encoded_bytes=encoded_bytes,
            decoded_bytes=decoded_bytes,
            scale_names=tuple(str(value) for value in scale_names),
            scale_nbytes=tuple(int(value) for value in scale_nbytes),
            weight_names=tuple(str(value) for value in weight_names),
            weight_nbytes=tuple(int(value) for value in weight_nbytes),
            codec=str(raw.get("codec")),
        )
        if sum(record.scale_nbytes) != decoded_bytes or any(
            value <= 0 for value in (*record.scale_nbytes, *record.weight_nbytes)
        ):
            raise ScaleXContainerError("invalid ScaleX tensor byte geometry")
        expected_scales = tuple(
            _expert_name(int(metadata["layer"]), expert, component, "weight_scale")
            for component in _SCALE_COMPONENTS
        )
        expected_weights = tuple(
            _expert_name(int(metadata["layer"]), expert, component, "weight_packed")
            for component in _WEIGHT_COMPONENTS
        )
        if record.scale_names != expected_scales or record.weight_names != expected_weights:
            raise ScaleXContainerError("ScaleX tensor ordering changed")
        if record.codec not in {"raw", "one_bit", "two_bit"}:
            raise ScaleXContainerError("invalid ScaleX codec name")
        for name, expected_bytes in zip(
            (*record.scale_names, *record.weight_names),
            (*record.scale_nbytes, *record.weight_nbytes),
            strict=True,
        ):
            entry = virtual_header.get(name)
            if not isinstance(entry, dict) or entry.get("dtype") != "U8":
                raise ScaleXContainerError(f"invalid embedded ScaleX tensor: {name}")
            _, actual_bytes = _tensor_range(
                virtual_header,
                name,
                len(original_prefix),
            )
            if actual_bytes != expected_bytes:
                raise ScaleXContainerError(f"ScaleX tensor length changed: {name}")
        cursor += record.physical_bytes
        records.append(record)
    if cursor != int(stored_bytes):
        raise ScaleXContainerError("ScaleX records do not cover stored data")
    original_bytes = int(metadata.get("original_bytes", -1))
    if original_bytes <= len(original_prefix):
        raise ScaleXContainerError("invalid original ScaleX file length")
    layout = ScaleXLayout(
        path=source,
        metadata=metadata,
        data_base=data_base,
        original_prefix=original_prefix,
        virtual_header=virtual_header,
        records=tuple(records),
    )
    _original_tensor_order(layout)
    return layout


def _read_exact(handle: BinaryIO, offset: int, length: int) -> bytes:
    handle.seek(offset)
    value = handle.read(length)
    if len(value) != length:
        raise ScaleXContainerError(f"short read at offset={offset}, length={length}")
    return value


def _record_tensors(
    layout: ScaleXLayout,
    handle: BinaryIO,
    expert: int,
) -> dict[str, bytes]:
    record = layout.record(expert)
    encoded = _read_exact(handle, record.absolute_offset, record.encoded_bytes)
    info = inspect_mode_a(encoded)
    if info.raw_bytes != record.decoded_bytes or info.codec != record.codec:
        raise ScaleXContainerError(f"ScaleX record metadata changed for expert {expert}")
    raw_scales = decode_mode_a(encoded)
    tensors: dict[str, bytes] = {}
    cursor = 0
    for name, length in zip(record.scale_names, record.scale_nbytes, strict=True):
        tensors[name] = raw_scales[cursor : cursor + length]
        cursor += length
    cursor = record.absolute_offset + record.encoded_bytes
    for name, length in zip(record.weight_names, record.weight_nbytes, strict=True):
        tensors[name] = _read_exact(handle, cursor, length)
        cursor += length
    return tensors


def read_scalex_tensor(layout: ScaleXLayout, name: str) -> bytes:
    """Read one virtual safetensors tensor from a compressed ScaleX shard."""

    match = _EXPERT_RE.fullmatch(name)
    if match is None or int(match.group(1)) != layout.layer:
        raise ScaleXContainerError(f"tensor is not part of ScaleX layer {layout.layer}: {name}")
    expert = int(match.group(2))
    with layout.path.open("rb") as handle:
        tensors = _record_tensors(layout, handle, expert)
    try:
        return tensors[name]
    except KeyError as exc:
        raise ScaleXContainerError(f"ScaleX record does not contain tensor: {name}") from exc


def _original_tensor_order(layout: ScaleXLayout) -> tuple[str, ...]:
    names = set(layout.virtual_header) - {"__metadata__"}
    ordered: list[tuple[int, int, str]] = []
    cursor = 0
    for name in names:
        entry = layout.virtual_header.get(name)
        if not isinstance(entry, dict):
            raise ScaleXContainerError(f"invalid embedded tensor metadata: {name}")
        try:
            start, end = (int(value) for value in entry["data_offsets"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ScaleXContainerError(f"invalid embedded tensor range: {name}") from exc
        ordered.append((start, end, name))
    result: list[str] = []
    for start, end, name in sorted(ordered):
        if start != cursor or end <= start:
            raise ScaleXContainerError(f"embedded payload has a gap before {name}")
        cursor = end
        result.append(name)
    if len(layout.original_prefix) + cursor != layout.original_bytes:
        raise ScaleXContainerError("embedded payload length differs from original file")
    return tuple(result)


def reconstructed_blocks(layout: ScaleXLayout) -> Iterator[bytes]:
    yield layout.original_prefix
    active_expert = -1
    tensors: dict[str, bytes] = {}
    with layout.path.open("rb") as handle:
        for name in _original_tensor_order(layout):
            match = _EXPERT_RE.fullmatch(name)
            if match is None or int(match.group(1)) != layout.layer:
                raise ScaleXContainerError(f"unexpected embedded tensor: {name}")
            expert = int(match.group(2))
            if expert != active_expert:
                tensors = _record_tensors(layout, handle, expert)
                active_expert = expert
            try:
                yield tensors[name]
            except KeyError as exc:
                raise ScaleXContainerError(f"record does not reconstruct {name}") from exc


def verify_scalex_layer(path: str | Path) -> ScaleXReport:
    layout = read_scalex_layout(path)
    digest = hashlib.sha256()
    reconstructed_bytes = 0
    for block in reconstructed_blocks(layout):
        digest.update(block)
        reconstructed_bytes += len(block)
    reconstructed_sha = digest.hexdigest()
    byte_identical = (
        reconstructed_bytes == layout.original_bytes
        and reconstructed_sha == layout.original_sha256
    )
    if not byte_identical:
        raise ScaleXContainerError("ScaleX reconstruction hash differs from source")
    stored_bytes = layout.path.stat().st_size
    return ScaleXReport(
        operation="verify",
        layer=layout.layer,
        path=str(layout.path),
        experts=layout.experts,
        original_bytes=layout.original_bytes,
        stored_bytes=stored_bytes,
        bytes_saved=layout.original_bytes - stored_bytes,
        original_sha256=layout.original_sha256,
        reconstructed_sha256=reconstructed_sha,
        byte_identical=True,
    )


def _copy_range(source: BinaryIO, destination: BinaryIO, offset: int, length: int) -> None:
    source.seek(offset)
    remaining = length
    while remaining:
        block = source.read(min(COPY_CHUNK_BYTES, remaining))
        if not block:
            raise ScaleXContainerError("short tensor read during ScaleX conversion")
        destination.write(block)
        remaining -= len(block)


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def compress_scalex_layer(
    source: str | Path,
    destination: str | Path,
    *,
    layer: int,
    experts: int,
) -> ScaleXReport:
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if is_scalex_layer(source_path):
        raise ScaleXContainerError("source is already a ScaleX layer")
    if destination_path.exists() or destination_path.is_symlink():
        raise ScaleXContainerError(f"destination already exists: {destination_path}")
    original_prefix, header = _read_safetensors_prefix(source_path)
    data_base = len(original_prefix)
    _validate_source_inventory(
        source_path,
        header,
        layer=layer,
        experts=experts,
        data_base=data_base,
    )
    original_sha = _sha256_file(source_path)
    encoded_records: list[bytes] = []
    raw_records: list[dict[str, Any]] = []
    stored_cursor = len(original_prefix)
    with source_path.open("rb") as handle:
        for expert in range(experts):
            scale_names = tuple(
                _expert_name(layer, expert, component, "weight_scale")
                for component in _SCALE_COMPONENTS
            )
            weight_names = tuple(
                _expert_name(layer, expert, component, "weight_packed")
                for component in _WEIGHT_COMPONENTS
            )
            scale_ranges = tuple(
                _tensor_range(header, name, data_base) for name in scale_names
            )
            weight_ranges = tuple(
                _tensor_range(header, name, data_base) for name in weight_names
            )
            raw_scales = b"".join(
                _read_exact(handle, offset, length) for offset, length in scale_ranges
            )
            encoded = encode_mode_a(raw_scales)
            if decode_mode_a(encoded) != raw_scales:
                raise ScaleXContainerError(f"ScaleX round trip failed for expert {expert}")
            # Reject a Mode-A record that cannot carry the uint16 Mode-B tile
            # directory before any checkpoint bytes are replaced.
            mode_b_tile_prefix(encoded)
            info = inspect_mode_a(encoded)
            encoded_records.append(encoded)
            record = {
                "expert": expert,
                "stored_offset": stored_cursor,
                "encoded_bytes": len(encoded),
                "decoded_bytes": len(raw_scales),
                "codec": info.codec,
                "scale_names": list(scale_names),
                "scale_nbytes": [length for _, length in scale_ranges],
                "weight_names": list(weight_names),
                "weight_nbytes": [length for _, length in weight_ranges],
            }
            raw_records.append(record)
            stored_cursor += len(encoded) + sum(record["weight_nbytes"])

    metadata: dict[str, Any] = {
        "format": SCALEX_FORMAT,
        "version": SCALEX_VERSION,
        "layer": int(layer),
        "experts": int(experts),
        "scale_order": list(_SCALE_COMPONENTS),
        "weight_order": list(_WEIGHT_COMPONENTS),
        "original_bytes": source_path.stat().st_size,
        "original_sha256": original_sha,
        "original_prefix_bytes": len(original_prefix),
        "original_prefix_sha256": hashlib.sha256(original_prefix).hexdigest(),
        "records": raw_records,
    }
    raw_metadata = _serialize_metadata(metadata)
    stored_bytes = stored_cursor
    output_bytes = SCALEX_PREFIX.size + len(raw_metadata) + stored_bytes
    free_bytes = shutil.disk_usage(destination_path.parent).free
    if free_bytes < output_bytes + MINIMUM_FREE_MARGIN_BYTES:
        raise ScaleXContainerError(
            f"ScaleX conversion needs {(output_bytes + MINIMUM_FREE_MARGIN_BYTES) / 2**30:.2f} "
            f"GiB free but only {free_bytes / 2**30:.2f} GiB is available"
        )
    temporary = destination_path.parent / f".{destination_path.name}.partial"
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    source_mode = source_path.stat().st_mode
    try:
        with source_path.open("rb") as input_handle, temporary.open("xb") as output_handle:
            output_handle.write(
                SCALEX_PREFIX.pack(
                    SCALEX_MAGIC,
                    SCALEX_VERSION,
                    len(raw_metadata),
                    stored_bytes,
                )
            )
            output_handle.write(raw_metadata)
            output_handle.write(original_prefix)
            for expert, encoded in enumerate(encoded_records):
                output_handle.write(encoded)
                for name in raw_records[expert]["weight_names"]:
                    offset, length = _tensor_range(header, name, data_base)
                    _copy_range(input_handle, output_handle, offset, length)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.chmod(temporary, source_mode)
        report = verify_scalex_layer(temporary)
        os.replace(temporary, destination_path)
        _fsync_parent(destination_path)
        return ScaleXReport(
            operation="compress",
            layer=report.layer,
            path=str(destination_path),
            experts=report.experts,
            original_bytes=report.original_bytes,
            stored_bytes=destination_path.stat().st_size,
            bytes_saved=report.bytes_saved,
            original_sha256=report.original_sha256,
            reconstructed_sha256=report.reconstructed_sha256,
            byte_identical=report.byte_identical,
        )
    finally:
        if temporary.exists():
            temporary.unlink()


def restore_scalex_layer(source: str | Path, destination: str | Path) -> ScaleXReport:
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if destination_path.exists() or destination_path.is_symlink():
        raise ScaleXContainerError(f"destination already exists: {destination_path}")
    layout = read_scalex_layout(source_path)
    free_bytes = shutil.disk_usage(destination_path.parent).free
    if free_bytes < layout.original_bytes + MINIMUM_FREE_MARGIN_BYTES:
        raise ScaleXContainerError(
            f"ScaleX restoration needs {(layout.original_bytes + MINIMUM_FREE_MARGIN_BYTES) / 2**30:.2f} "
            f"GiB free but only {free_bytes / 2**30:.2f} GiB is available"
        )
    temporary = destination_path.parent / f".{destination_path.name}.partial"
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    source_mode = source_path.stat().st_mode
    try:
        digest = hashlib.sha256()
        written = 0
        with temporary.open("xb") as output:
            for block in reconstructed_blocks(layout):
                output.write(block)
                digest.update(block)
                written += len(block)
            output.flush()
            os.fsync(output.fileno())
        if written != layout.original_bytes or digest.hexdigest() != layout.original_sha256:
            raise ScaleXContainerError("restored ScaleX bytes differ from original")
        os.chmod(temporary, source_mode)
        os.replace(temporary, destination_path)
        _fsync_parent(destination_path)
        return ScaleXReport(
            operation="restore",
            layer=layout.layer,
            path=str(destination_path),
            experts=layout.experts,
            original_bytes=layout.original_bytes,
            stored_bytes=destination_path.stat().st_size,
            bytes_saved=0,
            original_sha256=layout.original_sha256,
            reconstructed_sha256=digest.hexdigest(),
            byte_identical=True,
        )
    finally:
        if temporary.exists():
            temporary.unlink()


def compress_scalex_layer_in_place(
    path: str | Path,
    *,
    layer: int,
    experts: int,
) -> ScaleXReport:
    target = Path(path).expanduser().resolve()
    if is_scalex_layer(target):
        return verify_scalex_layer(target)
    temporary = target.parent / f".{target.name}.scalex.partial"
    if temporary.exists() or temporary.is_symlink():
        try:
            recovered = verify_scalex_layer(temporary)
        except (OSError, ScaleXContainerError):
            temporary.unlink()
        else:
            if (
                recovered.original_bytes == target.stat().st_size
                and recovered.original_sha256 == _sha256_file(target)
            ):
                os.replace(temporary, target)
                _fsync_parent(target)
                return ScaleXReport(**(recovered.as_dict() | {"path": str(target)}))
            temporary.unlink()
    report = compress_scalex_layer(target, temporary, layer=layer, experts=experts)
    os.replace(temporary, target)
    _fsync_parent(target)
    return ScaleXReport(**(report.as_dict() | {"path": str(target)}))


def restore_scalex_layer_in_place(path: str | Path) -> ScaleXReport:
    target = Path(path).expanduser().resolve()
    if not is_scalex_layer(target):
        raise ScaleXContainerError(f"layer is not ScaleX: {target}")
    temporary = target.parent / f".{target.name}.native.partial"
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    report = restore_scalex_layer(target, temporary)
    os.replace(temporary, target)
    _fsync_parent(target)
    return ScaleXReport(**(report.as_dict() | {"path": str(target)}))
