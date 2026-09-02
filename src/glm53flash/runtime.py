"""First-token GLM runtime assembly and token-at-a-time generation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import time
from typing import Callable

import mlx.core as mx

from .contract import ContractError, EXPERTS_PER_LAYER, ModelContract
from .expert_reader import ReaderStats
from .expert_source import NativeExpertSourcePlan, build_native_expert_source_plan
from .expert_ssd import ExpertCacheStats
from .model import GLMForCausalLM
from .model_config import GLMTextConfig
from .mtp import GLMNextTokenPredictor
from .mtp_resident import build_mtp_resident_source_plan
from .native_expert_ssd import NativeExpertPool, NativeExpertSSD
from .resident_plan import ResidentSourcePlan, build_resident_source_plan
from .resident_reader import NativeResidentReader, ResidentReaderStats
from .trace import DecodeTrace


DEFAULT_MODEL_DIR = Path("/Users/kumargaurav/Documents/livglm/GLM53Flash")
DEFAULT_MEMORY_GIB = 24.0
SYSTEM_RESERVE_GIB = 4.0
# The live 128-token runtime needs about 0.22 GiB beyond resident tensors and
# ExpertSSD rows. Keep more than twice that measured requirement while using
# the user's --memory ceiling instead of leaving 3 GiB permanently idle.
RUNTIME_RESERVE_GIB = 0.5
CONTEXT_LIMIT = 128
EXPERT_SLOT_BYTES = 13_369_344
MOE_LAYER_COUNT = 42
MINIMUM_EXPERT_CAPACITY = 8
MTP_EXPERT_CAPACITY = 48
MTP_AUXILIARY_GIB = 0.75


def _metal_profile_counts() -> dict[str, int]:
    profile = getattr(mx.metal, "_profile_counters", None)
    if profile is None:
        return {}
    return {
        str(name): int(value)
        for name, value in profile().items()
        if name != "enabled"
    }


def _wait_for_external_profiler(ready: Path, go: Path) -> None:
    """Pause after prefill so Instruments can attach to decode only."""

    ready = ready.expanduser().resolve()
    go = go.expanduser().resolve()
    if ready.exists():
        raise FileExistsError(f"external-profile ready path already exists: {ready}")
    if go.exists():
        raise FileExistsError(f"external-profile go path already exists: {go}")
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.write_text(
        f'{{"pid":{os.getpid()},"monotonic_ns":{time.perf_counter_ns()}}}\n',
        encoding="utf-8",
    )
    deadline = time.monotonic() + 120.0
    while not go.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"external profiler did not create go file within 120 seconds: {go}"
            )
        time.sleep(0.01)


@dataclass(frozen=True)
class RuntimeProfile:
    requested_gib: float
    effective_gib: float
    physical_gib: float
    resident_format: str
    resident_gib: float
    resident_load_gib: float
    resident_linear_count: int
    runtime_reserve_gib: float
    auxiliary_gib: float
    expert_source_format: str
    expert_slot_bytes_by_layer: tuple[int, ...]
    expert_capacity: int
    expert_cache_gib: float
    planned_gib: float
    budget_headroom_gib: float
    context_limit: int = CONTEXT_LIMIT

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def physical_memory_bytes() -> int:
    pages = int(os.sysconf("SC_PHYS_PAGES"))
    page_bytes = int(os.sysconf("SC_PAGE_SIZE"))
    total = pages * page_bytes
    if total <= 0:
        raise ContractError("cannot determine physical memory")
    return total


def resolve_runtime_profile(
    requested_gib: float | None,
    *,
    resident_bytes: int = 17_842_600_184,
    resident_load_bytes: int | None = None,
    resident_format: str = "bf16",
    resident_linear_count: int = 0,
    expert_source_format: str = "native_mxfp4",
    expert_slot_bytes_by_layer: tuple[int, ...] | None = None,
    physical_bytes: int | None = None,
    auxiliary_gib: float = 0.0,
) -> RuntimeProfile:
    physical_gib = (physical_bytes or physical_memory_bytes()) / 2**30
    requested = DEFAULT_MEMORY_GIB if requested_gib is None else float(requested_gib)
    if not math.isfinite(requested) or requested <= 0:
        raise ContractError("--memory must be a positive finite number")
    effective = min(requested, physical_gib - SYSTEM_RESERVE_GIB)
    resident_gib = resident_bytes / 2**30
    resident_load_gib = (resident_load_bytes or resident_bytes) / 2**30
    slot_bytes = (
        (EXPERT_SLOT_BYTES,) * MOE_LAYER_COUNT
        if expert_slot_bytes_by_layer is None
        else tuple(int(value) for value in expert_slot_bytes_by_layer)
    )
    if len(slot_bytes) != MOE_LAYER_COUNT or any(value <= 0 for value in slot_bytes):
        raise ContractError("expert slot geometry must cover 42 positive layer rows")
    bank_bytes = sum(slot_bytes)
    auxiliary = float(auxiliary_gib)
    if not math.isfinite(auxiliary) or auxiliary < 0:
        raise ContractError("auxiliary runtime reservation must be non-negative")
    available = (
        effective - resident_gib - RUNTIME_RESERVE_GIB - auxiliary
    ) * 2**30
    affordable_capacity = math.floor(available / bank_bytes)
    if affordable_capacity < MINIMUM_EXPERT_CAPACITY:
        minimum = (
            resident_gib
            + RUNTIME_RESERVE_GIB
            + auxiliary
            + MINIMUM_EXPERT_CAPACITY * bank_bytes / 2**30
        )
        raise ContractError(
            f"memory budget leaves capacity {affordable_capacity}, below routed top-k 8; "
            f"use at least {minimum:.1f} GiB"
        )
    capacity = min(EXPERTS_PER_LAYER, affordable_capacity)
    cache_gib = capacity * bank_bytes / 2**30
    planned_gib = resident_gib + RUNTIME_RESERVE_GIB + auxiliary + cache_gib
    headroom_gib = effective - planned_gib
    if headroom_gib < -1e-9:
        raise ContractError(
            f"resolved runtime plan exceeds --memory: {planned_gib:.3f} > "
            f"{effective:.3f} GiB"
        )
    return RuntimeProfile(
        requested_gib=requested,
        effective_gib=effective,
        physical_gib=physical_gib,
        resident_format=resident_format,
        resident_gib=resident_gib,
        resident_load_gib=resident_load_gib,
        resident_linear_count=resident_linear_count,
        runtime_reserve_gib=RUNTIME_RESERVE_GIB,
        auxiliary_gib=auxiliary,
        expert_source_format=expert_source_format,
        expert_slot_bytes_by_layer=slot_bytes,
        expert_capacity=capacity,
        expert_cache_gib=cache_gib,
        planned_gib=planned_gib,
        budget_headroom_gib=headroom_gib,
    )


@dataclass(frozen=True)
class PreflightResult:
    model_dir: str
    resident_plan: ResidentSourcePlan
    expert_plan: NativeExpertSourcePlan
    profile: RuntimeProfile

    def as_dict(self) -> dict:
        return {
            "model_dir": self.model_dir,
            "resident": self.resident_plan.as_dict(),
            "experts": self.expert_plan.as_dict(),
            "profile": self.profile.as_dict(),
            "expert_backend": (
                "mlx-io-glm/direct-to-slot/ScaleX-Mode-B"
                if self.expert_plan.scalex_mode_b
                else "mlx-io-glm/direct-to-slot/native-MXFP4"
            ),
            "scalex": self.expert_plan.scalex_mode_b,
        }


@dataclass(frozen=True)
class GenerationStats:
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    prefill_seconds: float
    decode_seconds: float
    decode_forwards: int
    steady_decode_seconds: float
    steady_decode_tokens: int
    speculative_cycles: int
    drafts_accepted: int
    drafts_rejected: int
    stopped_on_eos: bool
    reader: ReaderStats
    prefill_reader: ReaderStats
    decode_reader: ReaderStats
    expert_caches: tuple[ExpertCacheStats, ...]
    prefill_expert_caches: tuple[ExpertCacheStats, ...]
    decode_expert_caches: tuple[ExpertCacheStats, ...]
    active_memory_bytes: int
    peak_memory_bytes: int

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.elapsed_seconds if self.elapsed_seconds else 0.0

    @property
    def decode_tokens_per_second(self) -> float:
        decoded = max(0, self.generated_tokens - 1)
        return decoded / self.decode_seconds if self.decode_seconds else 0.0

    @property
    def steady_decode_tokens_per_second(self) -> float:
        return (
            self.steady_decode_tokens / self.steady_decode_seconds
            if self.steady_decode_seconds
            else 0.0
        )


def _reader_delta(after: ReaderStats, before: ReaderStats) -> ReaderStats:
    return ReaderStats(
        expert_loads=after.expert_loads - before.expert_loads,
        logical_reads=after.logical_reads - before.logical_reads,
        system_reads=after.system_reads - before.system_reads,
        read_bytes=after.read_bytes - before.read_bytes,
        open_shards=after.open_shards,
        backend=after.backend,
        direct_to_slot=after.direct_to_slot,
        read_seconds=after.read_seconds - before.read_seconds,
        io_wait_seconds=after.io_wait_seconds - before.io_wait_seconds,
        wired_bytes=after.wired_bytes,
    )


def _cache_delta(
    after: tuple[ExpertCacheStats, ...],
    before: tuple[ExpertCacheStats, ...],
) -> tuple[ExpertCacheStats, ...]:
    if len(after) != len(before):
        raise RuntimeError("ExpertSSD cache inventory changed during generation")
    return tuple(
        ExpertCacheStats(
            layer=end.layer,
            capacity=end.capacity,
            resident=end.resident,
            hits=end.hits - start.hits,
            misses=end.misses - start.misses,
            evictions=end.evictions - start.evictions,
            route_sync_seconds=end.route_sync_seconds - start.route_sync_seconds,
            slot_plan_seconds=end.slot_plan_seconds - start.slot_plan_seconds,
            policy=end.policy,
        )
        for end, start in zip(after, before, strict=True)
    )


class TargetRuntime:
    """Loaded resident graph plus one shared native expert reader."""

    def __init__(
        self,
        *,
        model_dir: Path,
        config: GLMTextConfig,
        profile: RuntimeProfile,
        model: GLMForCausalLM,
        expert_reader: NativeExpertPool,
        resident_stats: ResidentReaderStats,
        resident_quantization: dict[str, int | str],
        tokenizer,
        active_memory_bytes: int,
        peak_memory_bytes: int,
        mtp: GLMNextTokenPredictor | None = None,
        mtp_reader: NativeExpertPool | None = None,
    ):
        self.model_dir = model_dir
        self.config = config
        self.profile = profile
        self.model = model
        self.expert_reader = expert_reader
        self.resident_stats = resident_stats
        self.resident_quantization = resident_quantization
        self.tokenizer = tokenizer
        self.active_memory_bytes = active_memory_bytes
        self.peak_memory_bytes = peak_memory_bytes
        self.mtp = mtp
        self.mtp_reader = mtp_reader

    @classmethod
    def preflight(
        cls,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        memory_gib: float | None = None,
        physical_bytes: int | None = None,
        resident_mxfp8: bool = True,
        resident_mxfp4: bool = False,
        enable_mtp: bool = False,
    ) -> PreflightResult:
        contract = ModelContract.from_model_dir(model_dir)
        contract.validate_supported_profile()
        config = GLMTextConfig.from_model_dict(contract.config)
        resident = build_resident_source_plan(contract)
        experts = build_native_expert_source_plan(contract, config)
        resident_format = (
            "mxfp4" if resident_mxfp4 else "mxfp8" if resident_mxfp8 else "bf16"
        )
        resident_runtime_bytes = {
            "mxfp4": resident.mxfp4_runtime_bytes,
            "mxfp8": resident.mxfp8_runtime_bytes,
            "bf16": resident.destination_bytes,
        }[resident_format]
        profile = resolve_runtime_profile(
            memory_gib,
            resident_bytes=resident_runtime_bytes,
            resident_load_bytes=resident.destination_bytes,
            resident_format=resident_format,
            resident_linear_count=(
                resident.mxfp8_linear_count if resident_format != "bf16" else 0
            ),
            expert_source_format=experts.source_format,
            expert_slot_bytes_by_layer=experts.slot_bytes_by_layer,
            physical_bytes=physical_bytes,
            auxiliary_gib=MTP_AUXILIARY_GIB if enable_mtp else 0.0,
        )
        return PreflightResult(str(contract.model_dir), resident, experts, profile)

    @classmethod
    def load(
        cls,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        memory_gib: float | None = None,
        resident_mxfp8: bool = True,
        resident_mxfp4: bool = False,
        enable_mtp: bool = False,
    ) -> "TargetRuntime":
        preflight = cls.preflight(
            model_dir,
            memory_gib=memory_gib,
            resident_mxfp8=resident_mxfp8,
            resident_mxfp4=resident_mxfp4,
            enable_mtp=enable_mtp,
        )
        contract = ModelContract.from_model_dir(preflight.model_dir)
        config = GLMTextConfig.from_model_dict(contract.config)
        expert_reader = NativeExpertPool(preflight.expert_plan, workers=8)
        mtp_reader = None
        mtp = None
        memory_limit_bytes = int(preflight.profile.effective_gib * 2**30)

        def make_experts(layer: int) -> NativeExpertSSD:
            return NativeExpertSSD(
                expert_reader,
                layer=layer,
                capacity=preflight.profile.expert_capacity,
                swiglu_limit=config.swiglu_limit,
                defer_slots=True,
            )

        try:
            mx.set_memory_limit(memory_limit_bytes)
            mx.set_cache_limit(512 * 2**20)
            mx.reset_peak_memory()
            # Transformers is used only for tokenization; model-backend and
            # missing-PyTorch notices are irrelevant and otherwise look fatal.
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                preflight.model_dir,
                trust_remote_code=False,
            )
            model = GLMForCausalLM(
                config,
                context_limit=preflight.profile.context_limit,
                expert_factory=make_experts,
            )
            resident_stats = NativeResidentReader(preflight.resident_plan).load_into(model)
            if preflight.profile.resident_format in {"mxfp4", "mxfp8"}:
                bits = 4 if preflight.profile.resident_format == "mxfp4" else 8
                quantization = model.quantize_resident_linears_mxfp(bits)
                expected = {
                    "format": preflight.profile.resident_format,
                    "linear_count": preflight.resident_plan.mxfp8_linear_count,
                    "source_bytes": preflight.resident_plan.mxfp8_source_linear_bytes,
                    "destination_bytes": (
                        preflight.resident_plan.mxfp4_destination_linear_bytes
                        if bits == 4
                        else preflight.resident_plan.mxfp8_destination_linear_bytes
                    ),
                }
                if quantization != expected:
                    raise ContractError(
                        f"resident {preflight.profile.resident_format.upper()} "
                        f"conversion inventory changed: "
                        f"{quantization} != {expected}"
                    )
            else:
                quantization = {
                    "format": "bf16",
                    "linear_count": 0,
                    "source_bytes": 0,
                    "destination_bytes": 0,
                }
            if enable_mtp:
                mtp_expert_plan = build_native_expert_source_plan(
                    contract,
                    config,
                    first_layer=45,
                    last_layer=45,
                )
                mtp_reader = NativeExpertPool(mtp_expert_plan, workers=8)

                def make_mtp_experts(layer: int) -> NativeExpertSSD:
                    return NativeExpertSSD(
                        mtp_reader,
                        layer=layer,
                        capacity=MTP_EXPERT_CAPACITY,
                        swiglu_limit=config.swiglu_limit,
                        defer_slots=True,
                    )

                mtp = GLMNextTokenPredictor(config, make_mtp_experts)
                NativeResidentReader(
                    build_mtp_resident_source_plan(contract)
                ).load_into(mtp)
                mtp.quantize_resident_linears_mxfp4()
                mtp.experts.activate()
            for expert_layer in model.expert_layers():
                expert_layer.experts.activate()
            if mtp is None:
                mx.eval(model.parameters())
            else:
                mx.eval(model.parameters(), mtp.parameters())
            try:
                mx.clear_cache()
            except AttributeError:
                pass
            active_memory_bytes = int(mx.get_active_memory())
            peak_memory_bytes = int(mx.get_peak_memory())
            if active_memory_bytes > memory_limit_bytes:
                raise ContractError(
                    f"active MLX memory exceeded --memory: "
                    f"{active_memory_bytes / 2**30:.3f} > "
                    f"{preflight.profile.effective_gib:.3f} GiB"
                )
        except BaseException:
            if "model" in locals():
                for expert_layer in model.expert_layers():
                    expert_layer.experts.close()
            expert_reader.close()
            if mtp_reader is not None:
                mtp_reader.close()
            raise
        return cls(
            model_dir=Path(preflight.model_dir),
            config=config,
            profile=preflight.profile,
            model=model,
            expert_reader=expert_reader,
            resident_stats=resident_stats,
            resident_quantization=quantization,
            tokenizer=tokenizer,
            active_memory_bytes=active_memory_bytes,
            peak_memory_bytes=peak_memory_bytes,
            mtp=mtp,
            mtp_reader=mtp_reader,
        )

    def close(self) -> None:
        for layer in self.model.expert_layers():
            layer.experts.close()
        self.expert_reader.close()
        if self.mtp is not None:
            self.mtp.experts.close()
        if self.mtp_reader is not None:
            self.mtp_reader.close()

    def encode_messages(self, messages: list[dict[str, str]], *, thinking: bool) -> list[int]:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            clear_thinking=not thinking,
            reasoning_effort="max",
        )
        input_ids = [int(value) for value in encoded["input_ids"]]
        if not thinking:
            # GLM's official generation prompt always opens `<think>`. Match
            # the DeepSeek CLI's non-thinking UX by closing that empty section
            # before generation; `clear_thinking` only controls old messages.
            end_thinking = self.tokenizer.encode("</think>", add_special_tokens=False)
            if len(end_thinking) != 1:
                raise ContractError(
                    "official tokenizer no longer has a single </think> token"
                )
            input_ids.append(int(end_thinking[0]))
        return input_ids

    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int | None = None,
        on_token: Callable[[int, list[int]], None] | None = None,
        trace: DecodeTrace | None = None,
        external_profile_ready: Path | None = None,
        external_profile_go: Path | None = None,
    ) -> tuple[list[int], GenerationStats]:
        if not prompt_ids:
            raise ContractError("prompt tokenization produced no tokens")
        remaining = self.profile.context_limit - len(prompt_ids)
        if remaining <= 0:
            raise ContractError(
                f"prompt has {len(prompt_ids)} tokens; v1 limit is {self.profile.context_limit}"
            )
        limit = remaining if max_new_tokens is None else min(int(max_new_tokens), remaining)
        if limit < 1:
            raise ContractError("--max-tokens must be positive")
        if (external_profile_ready is None) != (external_profile_go is None):
            raise ValueError(
                "external_profile_ready and external_profile_go must be paired"
            )

        cache = self.model.empty_cache()
        mtp_cache = self.mtp.empty_cache() if self.mtp is not None else None
        logits = None
        hidden = None
        previous_hidden = None
        reader_start = self.expert_reader.stats()
        caches_start = tuple(layer.experts.stats() for layer in self.model.expert_layers())
        started = time.perf_counter()
        for token in prompt_ids:
            token_array = mx.array([[token]], dtype=mx.int32)
            if self.mtp is None:
                logits = self.model(token_array, cache)
                mx.eval(logits)
            else:
                logits, hidden = self.model(
                    token_array,
                    cache,
                    return_hidden=True,
                )
                mx.eval(logits, hidden)
                if previous_hidden is not None:
                    embedding = self.model.language_model.embed_tokens(token_array)
                    warm = self.mtp(embedding, previous_hidden, mtp_cache)
                    mx.eval(warm)
                previous_hidden = hidden
            if not bool(mx.all(mx.isfinite(logits)).item()):
                raise ContractError("non-finite output logits during prompt execution")
        assert logits is not None
        prefill_seconds = time.perf_counter() - started
        reader_after_prefill = self.expert_reader.stats()
        caches_after_prefill = tuple(
            layer.experts.stats() for layer in self.model.expert_layers()
        )
        if external_profile_ready is not None and external_profile_go is not None:
            _wait_for_external_profiler(
                external_profile_ready,
                external_profile_go,
            )

        generated: list[int] = []
        stopped = False
        decode_forwards = 0
        speculative_cycles = 0
        drafts_accepted = 0
        drafts_rejected = 0
        decode_started = time.perf_counter()
        steady_started: float | None = None
        steady_token_start = 0
        if self.mtp is not None:
            assert hidden is not None and mtp_cache is not None
            while len(generated) < limit:
                token = int(mx.argmax(logits[0, -1], axis=-1).item())
                generated.append(token)
                if on_token is not None:
                    on_token(token, generated)
                if token in self.config.eos_token_ids:
                    stopped = True
                    break
                if len(generated) >= limit:
                    break

                cycle = speculative_cycles
                trace_context = (
                    trace.decode_step(cycle, token_id=token)
                    if trace is not None
                    else nullcontext(False)
                )
                with trace_context as tracing:
                    draft_context = (
                        trace.span(
                            "mtp_draft",
                            category="mtp",
                            args={"decode_index": cycle, "depth": 1},
                        )
                        if tracing and trace is not None
                        else nullcontext({})
                    )
                    with draft_context:
                        token_array = mx.array([[token]], dtype=mx.int32)
                        embedding = self.model.language_model.embed_tokens(
                            token_array
                        )
                        draft_hidden = self.mtp(embedding, hidden, mtp_cache)
                        draft_logits = self.model.lm_head(draft_hidden)
                        mx.eval(draft_logits)
                        draft = int(
                            mx.argmax(draft_logits[0, -1], axis=-1).item()
                        )

                    snapshot = cache.snapshot()
                    forward_context = (
                        trace.span(
                            "model_forward",
                            category="python",
                            args={
                                "decode_index": cycle,
                                "width": 2,
                                "semantics": "MTP draft plus exact target verification",
                            },
                        )
                        if tracing and trace is not None
                        else nullcontext({})
                    )
                    with forward_context:
                        verify_logits, verify_hidden = self.model(
                            mx.array([[token, draft]], dtype=mx.int32),
                            cache,
                            return_hidden=True,
                        )
                    eval_context = (
                        trace.span(
                            "final_logits_eval_wait",
                            category="gpu_sync",
                            args={"decode_index": cycle, "width": 2},
                        )
                        if tracing and trace is not None
                        else nullcontext({})
                    )
                    with eval_context:
                        mx.eval(verify_logits, verify_hidden)
                    decode_forwards += 1
                    speculative_cycles += 1
                    if not bool(mx.all(mx.isfinite(verify_logits)).item()):
                        raise ContractError(
                            "non-finite output logits during speculative verification"
                        )
                    verified = int(
                        mx.argmax(verify_logits[0, 0], axis=-1).item()
                    )
                    if verified == draft:
                        drafts_accepted += 1
                        generated.append(draft)
                        if on_token is not None:
                            on_token(draft, generated)
                        logits = verify_logits[:, 1:2]
                        hidden = verify_hidden[:, 1:2]
                        if draft in self.config.eos_token_ids:
                            stopped = True
                            break
                        catchup_context = (
                            trace.span(
                                "mtp_kv_catchup",
                                category="mtp",
                                args={"decode_index": cycle, "positions": 1},
                            )
                            if tracing and trace is not None
                            else nullcontext({})
                        )
                        with catchup_context:
                            draft_array = mx.array([[draft]], dtype=mx.int32)
                            draft_embedding = (
                                self.model.language_model.embed_tokens(draft_array)
                            )
                            anchor = self.mtp.advance_attention_cache(
                                draft_embedding,
                                verify_hidden[:, 0:1],
                                mtp_cache,
                            )
                            mx.eval(anchor)
                    else:
                        drafts_rejected += 1
                        cache.commit_first_from_wide(snapshot)
                        logits = verify_logits[:, 0:1]
                        hidden = verify_hidden[:, 0:1]
                if steady_started is None and len(generated) >= 20:
                    steady_started = time.perf_counter()
                    steady_token_start = len(generated)
        else:
            for step in range(limit):
                token = int(mx.argmax(logits[0, -1], axis=-1).item())
                generated.append(token)
                if on_token is not None:
                    on_token(token, generated)
                if token in self.config.eos_token_ids:
                    stopped = True
                    break
                if step + 1 < limit:
                    trace_context = (
                        trace.decode_step(step, token_id=token)
                        if trace is not None
                        else nullcontext(False)
                    )
                    with trace_context as tracing:
                        forward_context = (
                            trace.span(
                                "model_forward",
                                category="python",
                                args={"decode_index": step, "width": 1},
                            )
                            if tracing and trace is not None
                            else nullcontext({})
                        )
                        with forward_context:
                            logits = self.model(
                                mx.array([[token]], dtype=mx.int32),
                                cache,
                            )
                        eval_context = (
                            trace.span(
                                "final_logits_eval_wait",
                                category="gpu_sync",
                                args={"decode_index": step, "width": 1},
                            )
                            if tracing and trace is not None
                            else nullcontext({})
                        )
                        with eval_context:
                            mx.eval(logits)
                        decode_forwards += 1
                        if not bool(mx.all(mx.isfinite(logits)).item()):
                            raise ContractError(
                                "non-finite output logits during generation"
                            )
        decode_seconds = time.perf_counter() - decode_started
        steady_decode_seconds = (
            time.perf_counter() - steady_started
            if steady_started is not None
            else 0.0
        )
        steady_decode_tokens = (
            len(generated) - steady_token_start
            if steady_started is not None
            else 0
        )
        elapsed = time.perf_counter() - started
        reader_end = self.expert_reader.stats()
        caches_end = tuple(layer.experts.stats() for layer in self.model.expert_layers())
        return generated, GenerationStats(
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(generated),
            elapsed_seconds=elapsed,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            decode_forwards=decode_forwards,
            steady_decode_seconds=steady_decode_seconds,
            steady_decode_tokens=steady_decode_tokens,
            speculative_cycles=speculative_cycles,
            drafts_accepted=drafts_accepted,
            drafts_rejected=drafts_rejected,
            stopped_on_eos=stopped,
            reader=_reader_delta(reader_end, reader_start),
            prefill_reader=_reader_delta(reader_after_prefill, reader_start),
            decode_reader=_reader_delta(reader_end, reader_after_prefill),
            expert_caches=_cache_delta(caches_end, caches_start),
            prefill_expert_caches=_cache_delta(caches_after_prefill, caches_start),
            decode_expert_caches=_cache_delta(caches_end, caches_after_prefill),
            active_memory_bytes=int(mx.get_active_memory()),
            peak_memory_bytes=int(mx.get_peak_memory()),
        )
