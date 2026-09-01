"""First-token GLM runtime assembly and token-at-a-time generation."""

from __future__ import annotations

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
from .native_expert_ssd import NativeExpertPool, NativeExpertSSD
from .resident_plan import ResidentSourcePlan, build_resident_source_plan
from .resident_reader import NativeResidentReader, ResidentReaderStats


DEFAULT_MODEL_DIR = Path("/Users/kumargaurav/Documents/livglm/GLM53Flash")
DEFAULT_MEMORY_GIB = 24.0
SYSTEM_RESERVE_GIB = 4.0
RUNTIME_RESERVE_GIB = 3.0
CONTEXT_LIMIT = 128
EXPERT_SLOT_BYTES = 13_369_344
MOE_LAYER_COUNT = 42
MINIMUM_EXPERT_CAPACITY = 8


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
    physical_bytes: int | None = None,
) -> RuntimeProfile:
    physical_gib = (physical_bytes or physical_memory_bytes()) / 2**30
    requested = DEFAULT_MEMORY_GIB if requested_gib is None else float(requested_gib)
    if not math.isfinite(requested) or requested <= 0:
        raise ContractError("--memory must be a positive finite number")
    effective = min(requested, physical_gib - SYSTEM_RESERVE_GIB)
    resident_gib = resident_bytes / 2**30
    resident_load_gib = (resident_load_bytes or resident_bytes) / 2**30
    available = (effective - resident_gib - RUNTIME_RESERVE_GIB) * 2**30
    affordable_capacity = math.floor(available / (MOE_LAYER_COUNT * EXPERT_SLOT_BYTES))
    if affordable_capacity < MINIMUM_EXPERT_CAPACITY:
        minimum = (
            resident_gib
            + RUNTIME_RESERVE_GIB
            + MINIMUM_EXPERT_CAPACITY * MOE_LAYER_COUNT * EXPERT_SLOT_BYTES / 2**30
        )
        raise ContractError(
            f"memory budget leaves capacity {affordable_capacity}, below routed top-k 8; "
            f"use at least {minimum:.1f} GiB"
        )
    capacity = min(EXPERTS_PER_LAYER, affordable_capacity)
    cache_gib = capacity * MOE_LAYER_COUNT * EXPERT_SLOT_BYTES / 2**30
    planned_gib = resident_gib + RUNTIME_RESERVE_GIB + cache_gib
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
            "expert_backend": "mlx-io-glm/direct-to-slot/native-MXFP4",
            "scalex": False,
        }


@dataclass(frozen=True)
class GenerationStats:
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    prefill_seconds: float
    decode_seconds: float
    decode_forwards: int
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
        return self.decode_forwards / self.decode_seconds if self.decode_seconds else 0.0


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

    @classmethod
    def preflight(
        cls,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        memory_gib: float | None = None,
        physical_bytes: int | None = None,
        resident_mxfp8: bool = True,
    ) -> PreflightResult:
        contract = ModelContract.from_model_dir(model_dir)
        contract.validate_supported_profile()
        config = GLMTextConfig.from_model_dict(contract.config)
        resident = build_resident_source_plan(contract)
        experts = build_native_expert_source_plan(contract, config)
        profile = resolve_runtime_profile(
            memory_gib,
            resident_bytes=(
                resident.mxfp8_runtime_bytes
                if resident_mxfp8
                else resident.destination_bytes
            ),
            resident_load_bytes=resident.destination_bytes,
            resident_format="mxfp8" if resident_mxfp8 else "bf16",
            resident_linear_count=(
                resident.mxfp8_linear_count if resident_mxfp8 else 0
            ),
            physical_bytes=physical_bytes,
        )
        return PreflightResult(str(contract.model_dir), resident, experts, profile)

    @classmethod
    def load(
        cls,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        memory_gib: float | None = None,
        resident_mxfp8: bool = True,
    ) -> "TargetRuntime":
        preflight = cls.preflight(
            model_dir,
            memory_gib=memory_gib,
            resident_mxfp8=resident_mxfp8,
        )
        contract = ModelContract.from_model_dir(preflight.model_dir)
        config = GLMTextConfig.from_model_dict(contract.config)
        expert_reader = NativeExpertPool(preflight.expert_plan, workers=8)
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
            if preflight.profile.resident_format == "mxfp8":
                quantization = model.quantize_resident_linears_mxfp8()
                expected = {
                    "format": "mxfp8",
                    "linear_count": preflight.resident_plan.mxfp8_linear_count,
                    "source_bytes": preflight.resident_plan.mxfp8_source_linear_bytes,
                    "destination_bytes": (
                        preflight.resident_plan.mxfp8_destination_linear_bytes
                    ),
                }
                if quantization != expected:
                    raise ContractError(
                        f"resident MXFP8 conversion inventory changed: "
                        f"{quantization} != {expected}"
                    )
            else:
                quantization = {
                    "format": "bf16",
                    "linear_count": 0,
                    "source_bytes": 0,
                    "destination_bytes": 0,
                }
            for expert_layer in model.expert_layers():
                expert_layer.experts.activate()
            mx.eval(model.parameters())
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
        )

    def close(self) -> None:
        for layer in self.model.expert_layers():
            layer.experts.close()
        self.expert_reader.close()

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

        cache = self.model.empty_cache()
        logits = None
        reader_start = self.expert_reader.stats()
        caches_start = tuple(layer.experts.stats() for layer in self.model.expert_layers())
        started = time.perf_counter()
        for token in prompt_ids:
            logits = self.model(mx.array([[token]], dtype=mx.int32), cache)
            mx.eval(logits)
            if not bool(mx.all(mx.isfinite(logits)).item()):
                raise ContractError("non-finite output logits during prompt execution")
        assert logits is not None
        prefill_seconds = time.perf_counter() - started
        reader_after_prefill = self.expert_reader.stats()
        caches_after_prefill = tuple(
            layer.experts.stats() for layer in self.model.expert_layers()
        )

        generated: list[int] = []
        stopped = False
        decode_forwards = 0
        decode_started = time.perf_counter()
        for step in range(limit):
            token = int(mx.argmax(logits[0, -1], axis=-1).item())
            generated.append(token)
            if on_token is not None:
                on_token(token, generated)
            if token in self.config.eos_token_ids:
                stopped = True
                break
            if step + 1 < limit:
                logits = self.model(mx.array([[token]], dtype=mx.int32), cache)
                mx.eval(logits)
                decode_forwards += 1
                if not bool(mx.all(mx.isfinite(logits)).item()):
                    raise ContractError("non-finite output logits during generation")
        decode_seconds = time.perf_counter() - decode_started
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
