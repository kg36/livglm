from __future__ import annotations

import json
import re
import struct
import time
from typing import Any, BinaryIO
import urllib.error
import urllib.request

from .sources import Source


USER_AGENT = "LivSeek-GLM53Flash-composite/0.1"
CONTENT_RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def _request(url: str, *, start: int | None = None, end: int | None = None):
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if start is not None:
        if end is None or end < start:
            raise ValueError("invalid byte range")
        headers["Range"] = f"bytes={start}-{end}"
    return urllib.request.Request(url, headers=headers)


def fetch_bytes(url: str, *, retries: int = 6) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(_request(url), timeout=120) as response:
                return response.read()
        except (OSError, urllib.error.URLError) as exc:
            if attempt + 1 == retries:
                raise RuntimeError(f"failed to fetch {url}: {exc}") from exc
            time.sleep(min(2**attempt, 16))
    raise AssertionError("unreachable")


def fetch_json(url: str) -> dict[str, Any]:
    return json.loads(fetch_bytes(url))


def fetch_range(url: str, start: int, end: int, *, retries: int = 6) -> bytes:
    expected = end - start + 1
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(_request(url, start=start, end=end), timeout=180) as response:
                _check_range_response(response, start, end)
                data = response.read()
            if len(data) != expected:
                raise IOError(f"short range: wanted {expected}, received {len(data)}")
            return data
        except (OSError, urllib.error.URLError) as exc:
            if attempt + 1 == retries:
                raise RuntimeError(f"failed range {start}-{end} from {url}: {exc}") from exc
            time.sleep(min(2**attempt, 16))
    raise AssertionError("unreachable")


def copy_range_to_file(
    url: str,
    source_start: int,
    source_end: int,
    output: BinaryIO,
    output_start: int,
    *,
    retries: int = 6,
) -> int:
    expected = source_end - source_start + 1
    for attempt in range(retries):
        written = 0
        try:
            with urllib.request.urlopen(
                _request(url, start=source_start, end=source_end), timeout=300
            ) as response:
                _check_range_response(response, source_start, source_end)
                output.seek(output_start)
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    written += len(chunk)
            if written != expected:
                raise IOError(f"short range: wanted {expected}, received {written}")
            return written
        except (OSError, urllib.error.URLError) as exc:
            if attempt + 1 == retries:
                raise RuntimeError(
                    f"failed range {source_start}-{source_end} from {url}: {exc}"
                ) from exc
            time.sleep(min(2**attempt, 16))
    raise AssertionError("unreachable")


def _check_range_response(response, start: int, end: int) -> None:
    if response.status != 206:
        raise IOError(f"server ignored byte range (HTTP {response.status})")
    value = response.headers.get("Content-Range", "")
    match = CONTENT_RANGE_RE.fullmatch(value)
    if not match or (int(match.group(1)), int(match.group(2))) != (start, end):
        raise IOError(f"unexpected Content-Range: {value!r}")


def index(source: Source) -> dict[str, Any]:
    return fetch_json(f"{source.base_url}/model.safetensors.index.json?download=true")


def shard_url(source: Source, shard: str) -> str:
    return f"{source.base_url}/{shard}?download=true"


def remote_safetensors_header(source: Source, shard: str) -> tuple[int, dict[str, Any]]:
    url = shard_url(source, shard)
    raw_length = fetch_range(url, 0, 7)
    header_length = struct.unpack("<Q", raw_length)[0]
    if not 2 <= header_length <= 512 * 1024 * 1024:
        raise ValueError(f"implausible safetensors header in {shard}: {header_length}")
    raw_header = fetch_range(url, 8, 7 + header_length)
    return 8 + header_length, json.loads(raw_header)


def local_safetensors_header(path) -> tuple[int, dict[str, Any]]:
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors file: {path}")
        length = struct.unpack("<Q", raw)[0]
        header = handle.read(length)
        if len(header) != length:
            raise ValueError(f"truncated safetensors header: {path}")
    return 8 + length, json.loads(header)


def tensor_blob(source: Source, shard: str, data_base: int, meta: dict[str, Any]) -> bytes:
    start, end = meta["data_offsets"]
    return fetch_range(shard_url(source, shard), data_base + start, data_base + end - 1)
