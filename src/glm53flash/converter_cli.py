"""Public reversible converter for GLM routed-expert ScaleX shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .contract import EXPERTS_PER_LAYER, FIRST_MOE_LAYER, LAST_MAIN_LAYER
from .scalex_container import (
    ScaleXContainerError,
    compress_scalex_layer,
    compress_scalex_layer_in_place,
    is_scalex_layer,
    restore_scalex_layer,
    restore_scalex_layer_in_place,
    verify_scalex_layer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="./converter",
        description=(
            "Convert GLM-5.3-Flash routed-expert shards to lossless ScaleX "
            "Mode-B storage, or restore their exact native safetensors bytes."
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="operate on one main routed layer instead of layers 3-44",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="restore exact native MXFP4 safetensors bytes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="non-mutating output path; requires --layer",
    )
    parser.add_argument("folder", type=Path, help="validated GLM composite folder")
    return parser


def _layer_shard(model_dir: Path, layer: int) -> Path:
    index_path = model_dir / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    except (OSError, KeyError, json.JSONDecodeError, TypeError) as exc:
        raise ScaleXContainerError(f"cannot read model index: {index_path}") from exc
    prefix = f"model.language_model.layers.{layer}.mlp.experts."
    shards = {
        shard
        for name, shard in weight_map.items()
        if isinstance(name, str) and name.startswith(prefix)
    }
    if len(shards) != 1:
        raise ScaleXContainerError(
            f"layer {layer} does not resolve to one expert shard: {sorted(shards)}"
        )
    shard = next(iter(shards))
    if not isinstance(shard, str) or Path(shard).name != shard:
        raise ScaleXContainerError(f"unsafe expert shard path for layer {layer}")
    path = model_dir / shard
    if not path.is_file():
        raise ScaleXContainerError(f"expert shard is missing: {path}")
    return path


def _print_report(report) -> None:
    print(
        f"Layer {report.layer}: {report.operation} PASS; "
        f"{report.experts} experts, {report.stored_bytes / 2**30:.3f} GiB stored, "
        f"{report.bytes_saved / 2**30:.3f} GiB saved, byte-identical={report.byte_identical}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    model_dir = args.folder.expanduser().resolve()
    if not model_dir.is_dir():
        parser.error(f"model folder does not exist: {model_dir}")
    if args.output is not None and args.layer is None:
        parser.error("--output requires --layer")
    layers = (
        (args.layer,)
        if args.layer is not None
        else tuple(range(FIRST_MOE_LAYER, LAST_MAIN_LAYER + 1))
    )
    if any(layer < FIRST_MOE_LAYER or layer > LAST_MAIN_LAYER for layer in layers):
        parser.error(
            f"--layer must be within {FIRST_MOE_LAYER}..{LAST_MAIN_LAYER}"
        )
    try:
        reports = []
        for layer in layers:
            source = _layer_shard(model_dir, layer)
            if args.output is not None:
                destination = args.output.expanduser().resolve()
                report = (
                    restore_scalex_layer(source, destination)
                    if args.native
                    else compress_scalex_layer(
                        source,
                        destination,
                        layer=layer,
                        experts=EXPERTS_PER_LAYER,
                    )
                )
            elif args.native:
                if not is_scalex_layer(source):
                    print(f"Layer {layer}: already native; skipped", flush=True)
                    continue
                report = restore_scalex_layer_in_place(source)
            elif is_scalex_layer(source):
                report = verify_scalex_layer(source)
            else:
                report = compress_scalex_layer_in_place(
                    source,
                    layer=layer,
                    experts=EXPERTS_PER_LAYER,
                )
            reports.append(report)
            _print_report(report)
        if reports:
            saved = sum(report.bytes_saved for report in reports)
            print(
                f"ScaleX operation complete: {len(reports)} layer(s), "
                f"{saved / 2**30:.3f} GiB net saved.",
                flush=True,
            )
        return 0
    except (OSError, ScaleXContainerError) as exc:
        print(f"converter: {exc}", file=sys.stderr)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
