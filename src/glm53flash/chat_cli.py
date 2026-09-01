"""DeepSeek-compatible command UX for the first GLM ExpertSSD runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .contract import ContractError
from .runtime import DEFAULT_MODEL_DIR, TargetRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="./run.sh",
        description="Run GLM-5.3-Flash with native MXFP4 ExpertSSD experts.",
    )
    parser.add_argument("prompt", nargs="*", help="one-shot user prompt")
    parser.add_argument("--chat", action="store_true", help="open persistent terminal chat")
    parser.add_argument("--memory", type=float, default=None, metavar="GB", help="allocation ceiling (default: 24)")
    parser.add_argument("--max-tokens", type=int, default=None, metavar="N", help="maximum output tokens")
    parser.add_argument("--thinking", action="store_true", help="enable thinking mode")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR, metavar="PATH", help=argparse.SUPPRESS)
    parser.add_argument("--preflight", action="store_true", help="validate metadata and print the runtime plan only")
    return parser


def _print_stats(stats) -> None:
    reader = stats.reader
    hits = sum(item.hits for item in stats.expert_caches)
    misses = sum(item.misses for item in stats.expert_caches)
    evictions = sum(item.evictions for item in stats.expert_caches)
    print(
        f"\n[prompt={stats.prompt_tokens}, generated={stats.generated_tokens}, "
        f"{stats.tokens_per_second:.3f} tok/s; "
        f"ExpertSSD loads={reader.expert_loads}, reads={reader.logical_reads}, "
        f"I/O={reader.read_bytes / 2**30:.3f} GiB, hits={hits}, "
        f"misses={misses}, evictions={evictions}]",
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


def _one_turn(runtime: TargetRuntime, messages: list[dict[str, str]], *, thinking: bool, maximum: int | None) -> str:
    prompt_ids = runtime.encode_messages(messages, thinking=thinking)
    generated, stats = runtime.generate(
        prompt_ids,
        max_new_tokens=maximum,
        on_token=_streamer(runtime.tokenizer),
    )
    print(flush=True)
    _print_stats(stats)
    return runtime.tokenizer.decode(generated, skip_special_tokens=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.preflight:
        result = TargetRuntime.preflight(args.model, memory_gib=args.memory)
        print(json.dumps(result.as_dict(), indent=2))
        return 0
    if args.chat and args.prompt:
        parser.error("a prompt cannot be combined with --chat")
    if not args.chat and not args.prompt:
        parser.error("provide a prompt or use --chat")

    print(
        "Loading GLM-5.3-Flash resident weights; routed experts stay on SSD "
        "(native MXFP4 ExpertSSD, no ScaleX)…",
        file=sys.stderr,
        flush=True,
    )
    runtime = TargetRuntime.load(args.model, memory_gib=args.memory)
    print(
        f"Ready: {runtime.profile.resident_gib:.2f} GiB resident, "
        f"{runtime.profile.expert_capacity} ExpertSSD slots/layer, "
        f"{runtime.profile.context_limit}-token v1 context.",
        file=sys.stderr,
        flush=True,
    )
    try:
        if not args.chat:
            messages = [{"role": "user", "content": " ".join(args.prompt)}]
            _one_turn(
                runtime,
                messages,
                thinking=args.thinking,
                maximum=args.max_tokens,
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
