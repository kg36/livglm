"""DeepSeek-compatible command UX for the first GLM ExpertSSD runtime."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

from .contract import ContractError
from .runtime import DEFAULT_MODEL_DIR, TargetRuntime
from .trace import DecodeTrace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="./run.sh",
        description="Run GLM-5.3-Flash with native MXFP4 ExpertSSD experts.",
    )
    parser.add_argument("prompt", nargs="*", help="one-shot user prompt")
    parser.add_argument("--chat", action="store_true", help="open persistent terminal chat")
    parser.add_argument(
        "--memory",
        type=float,
        default=None,
        metavar="GB",
        help="hard total runtime allocation ceiling (default: 24 GiB)",
    )
    parser.add_argument("--max-tokens", type=int, default=None, metavar="N", help="maximum output tokens")
    parser.add_argument("--thinking", action="store_true", help="enable thinking mode")
    parser.add_argument(
        "--resident-bf16",
        action="store_true",
        help="retain the BF16 resident correctness oracle instead of startup MXFP8",
    )
    parser.add_argument(
        "--resident-mxfp4",
        action="store_true",
        help="use startup MXFP4 resident weights to leave more memory for ExpertSSD",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR, metavar="PATH", help=argparse.SUPPRESS)
    parser.add_argument("--preflight", action="store_true", help="validate metadata and print the runtime plan only")
    parser.add_argument(
        "--trace",
        type=Path,
        metavar="PATH",
        help="write a Perfetto/Chrome JSON trace for selected decode forwards",
    )
    parser.add_argument("--trace-decode-start", type=int, default=0, metavar="N")
    parser.add_argument("--trace-decode-steps", type=int, default=20, metavar="N")
    parser.add_argument(
        "--external-profile-ready",
        type=Path,
        metavar="PATH",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--external-profile-go",
        type=Path,
        metavar="PATH",
        help=argparse.SUPPRESS,
    )
    return parser


def _print_stats(stats) -> None:
    reader = stats.reader
    hits = sum(item.hits for item in stats.expert_caches)
    misses = sum(item.misses for item in stats.expert_caches)
    evictions = sum(item.evictions for item in stats.expert_caches)
    lookups = hits + misses
    hit_rate = 100.0 * hits / lookups if lookups else 0.0
    bandwidth = (
        reader.read_bytes / 1e9 / reader.io_wait_seconds
        if reader.io_wait_seconds
        else 0.0
    )
    decode_hits = sum(item.hits for item in stats.decode_expert_caches)
    decode_misses = sum(item.misses for item in stats.decode_expert_caches)
    decode_lookups = decode_hits + decode_misses
    decode_hit_rate = (
        100.0 * decode_hits / decode_lookups if decode_lookups else 0.0
    )
    route_sync = sum(item.route_sync_seconds for item in stats.expert_caches)
    slot_plan = sum(item.slot_plan_seconds for item in stats.expert_caches)
    decode_route_sync = sum(
        item.route_sync_seconds for item in stats.decode_expert_caches
    )
    decode_bandwidth = (
        stats.decode_reader.read_bytes / 1e9 / stats.decode_reader.io_wait_seconds
        if stats.decode_reader.io_wait_seconds
        else 0.0
    )
    prefill_rate = (
        stats.prompt_tokens / stats.prefill_seconds
        if stats.prefill_seconds
        else 0.0
    )
    decode_rate = (
        f"{stats.decode_tokens_per_second:.3f} tok/s"
        if stats.decode_forwards
        else "n/a"
    )
    print(
        f"\n[prompt={stats.prompt_tokens}, generated={stats.generated_tokens}, "
        f"prefill={prefill_rate:.3f} tok/s, decode={decode_rate}, "
        f"end-to-end={stats.tokens_per_second:.3f} output tok/s; "
        f"ExpertSSD={reader.backend}, loads={reader.expert_loads}, "
        f"reads={reader.logical_reads}, "
        f"I/O={reader.read_bytes / 2**30:.3f} GiB, hits={hits}, "
        f"misses={misses}, hit-rate={hit_rate:.1f}%, evictions={evictions}, "
        f"SSD-wait={reader.io_wait_seconds:.3f}s, "
        f"visible-bandwidth={bandwidth:.2f} GB/s, "
        f"decode-I/O={stats.decode_reader.read_bytes / 2**30:.3f} GiB, "
        f"decode-hit-rate={decode_hit_rate:.1f}%, "
        f"decode-SSD-wait={stats.decode_reader.io_wait_seconds:.3f}s, "
        f"decode-bandwidth={decode_bandwidth:.2f} GB/s, "
        f"route-sync={route_sync:.3f}s (decode {decode_route_sync:.3f}s), "
        f"slot-plan={slot_plan:.3f}s, "
        f"MLX={stats.active_memory_bytes / 2**30:.2f}/"
        f"{stats.peak_memory_bytes / 2**30:.2f} GiB active/peak]",
        file=sys.stderr,
    )


def _streamer(tokenizer):
    previous = ""

    def emit(_token: int, generated: list[int]) -> None:
        nonlocal previous
        current = tokenizer.decode(generated, skip_special_tokens=True)
        suffix = current[len(previous) :] if current.startswith(previous) else current
        if suffix:
            print(suffix, end="", flush=True)
        previous = current

    return emit


def _one_turn(
    runtime: TargetRuntime,
    messages: list[dict[str, str]],
    *,
    thinking: bool,
    maximum: int | None,
    trace: DecodeTrace | None = None,
    external_profile_ready: Path | None = None,
    external_profile_go: Path | None = None,
) -> str:
    prompt_ids = runtime.encode_messages(messages, thinking=thinking)
    generated, stats = runtime.generate(
        prompt_ids,
        max_new_tokens=maximum,
        on_token=_streamer(runtime.tokenizer),
        trace=trace,
        external_profile_ready=external_profile_ready,
        external_profile_go=external_profile_go,
    )
    print(flush=True)
    _print_stats(stats)
    return runtime.tokenizer.decode(generated, skip_special_tokens=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resident_bf16 and args.resident_mxfp4:
        parser.error("--resident-bf16 cannot be combined with --resident-mxfp4")
    if args.preflight:
        result = TargetRuntime.preflight(
            args.model,
            memory_gib=args.memory,
            resident_mxfp8=not args.resident_bf16,
            resident_mxfp4=args.resident_mxfp4,
        )
        print(json.dumps(result.as_dict(), indent=2))
        return 0
    if args.chat and args.prompt:
        parser.error("a prompt cannot be combined with --chat")
    if args.chat and args.trace is not None:
        parser.error("--trace currently supports one-shot generation only")
    if not args.chat and not args.prompt:
        parser.error("provide a prompt or use --chat")
    if (args.external_profile_ready is None) != (args.external_profile_go is None):
        parser.error("--external-profile-ready and --external-profile-go must be paired")
    if args.trace is None and (
        args.trace_decode_start != 0 or args.trace_decode_steps != 20
    ):
        parser.error("--trace-decode-start/steps require --trace")

    resident_label = (
        "BF16 oracle"
        if args.resident_bf16
        else "startup MXFP4"
        if args.resident_mxfp4
        else "startup MXFP8"
    )
    print(
        "Loading GLM-5.3-Flash resident weights; routed experts stay on SSD "
        f"({resident_label} resident, "
        "native MXFP4 ExpertSSD, no ScaleX)…",
        file=sys.stderr,
        flush=True,
    )
    runtime = TargetRuntime.load(
        args.model,
        memory_gib=args.memory,
        resident_mxfp8=not args.resident_bf16,
        resident_mxfp4=args.resident_mxfp4,
    )
    print(
        f"Ready: {runtime.profile.resident_gib:.2f} GiB "
        f"{runtime.profile.resident_format.upper()} resident, "
        f"{runtime.profile.expert_capacity} ExpertSSD slots/layer, "
        f"{runtime.profile.expert_cache_gib:.2f} GiB expert cache, "
        f"{runtime.profile.planned_gib:.2f}/{runtime.profile.effective_gib:.2f} "
        f"GiB planned, {runtime.active_memory_bytes / 2**30:.2f} GiB active, "
        f"{runtime.profile.context_limit}-token v1 context.",
        file=sys.stderr,
        flush=True,
    )
    try:
        if not args.chat:
            messages = [{"role": "user", "content": " ".join(args.prompt)}]
            decode_trace = (
                DecodeTrace(
                    args.trace,
                    decode_start=args.trace_decode_start,
                    decode_steps=args.trace_decode_steps,
                    metadata={
                        "model_dir": str(runtime.model_dir),
                        "resident_format": runtime.profile.resident_format,
                        "memory_gib": runtime.profile.effective_gib,
                        "expert_capacity": runtime.profile.expert_capacity,
                        "paired_gpu_trace": "Instruments Metal System Trace",
                    },
                )
                if args.trace is not None
                else None
            )
            with decode_trace if decode_trace is not None else nullcontext():
                _one_turn(
                    runtime,
                    messages,
                    thinking=args.thinking,
                    maximum=args.max_tokens,
                    trace=decode_trace,
                    external_profile_ready=args.external_profile_ready,
                    external_profile_go=args.external_profile_go,
                )
            if decode_trace is not None:
                print(
                    f"Perfetto trace: {decode_trace.path}",
                    file=sys.stderr,
                )
            return 0

        history: list[dict[str, str]] = []
        print("GLM-5.3-Flash ExpertSSD chat. Ctrl-D or /exit to quit.")
        while True:
            try:
                prompt = input("you> ").strip()
            except EOFError:
                print()
                break
            if prompt in {"/exit", "/quit"}:
                break
            if not prompt:
                continue
            history.append({"role": "user", "content": prompt})
            print("glm> ", end="", flush=True)
            reply = _one_turn(
                runtime,
                history,
                thinking=True,
                maximum=args.max_tokens,
            )
            history.append({"role": "assistant", "content": reply})
        return 0
    finally:
        runtime.close()


def entrypoint() -> None:
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    entrypoint()
