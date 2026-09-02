"""Native composite checkpoint contract for the GLM-5.3-Flash runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Iterable

from .hf_range import local_safetensors_header
from .sources import EXPERTS, OFFICIAL

if TYPE_CHECKING:
    from .scalex_container import ScaleXLayout


COMPOSITE_FORMAT = "livseek-glm53flash-composite-v1"
MAIN_LAYER_COUNT = 45
FIRST_MOE_LAYER = 3
LAST_MAIN_LAYER = 44
MTP_LAYER = 45
EXPERTS_PER_LAYER = 288
TOP_K = 8
EXPERT_COMPONENTS = ("down_proj", "gate_proj", "up_proj")
EXPERT_PARTS = ("weight_packed", "weight_scale")

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}

ROUTED_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(down_proj|gate_proj|up_proj)\.(weight_packed|weight_scale)$"
)
LANGUAGE_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


class ContractError(ValueError):
    """The selected files do not satisfy the supported native contract."""


def tensor_payload_bytes(dtype: str, shape: Iterable[int]) -> int:
    try:
        element_bytes = DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ContractError(f"unsupported safetensors dtype: {dtype}") from exc
    elements = 1
    for raw_dimension in shape:
        dimension = int(raw_dimension)
        if dimension < 0:
            raise ContractError(f"negative tensor dimension: {dimension}")
        elements *= dimension
    return elements * element_bytes


@dataclass(frozen=True)
class TensorSource:
    name: str
    shard_name: str
    absolute_offset: int
    byte_length: int
    dtype: str
    shape: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContractInventory:
    resident_tensors: int
    routed_tensors: int
    mtp_tensors: int
    vision_tensors: int
    unknown_tensors: int
    resident_payload_bytes: int
    routed_payload_bytes: int
    mtp_payload_bytes: int
    vision_payload_bytes: int

    @property
    def target_tensors(self) -> int:
        return self.resident_tensors + self.routed_tensors

    @property
    def ignored_tensors(self) -> int:
        return self.mtp_tensors + self.vision_tensors

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "target_tensors": self.target_tensors,
            "ignored_tensors": self.ignored_tensors,
        }


class ModelContract:
    """Metadata-only view of the validated native composite."""

    def __init__(
        self,
        model_dir: Path,
        config: dict[str, Any],
        index: dict[str, Any],
        validation: dict[str, Any],
        composite: dict[str, Any],
    ):
        self.model_dir = model_dir
        self.config = config
        self.index = index
        self.validation = validation
        self.composite = composite
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ContractError("model index has no weight_map")
        if any(not isinstance(name, str) or not isinstance(shard, str) for name, shard in weight_map.items()):
            raise ContractError("model index weight_map must contain string names and shards")
        self.weight_map: dict[str, str] = dict(weight_map)
        self._headers: dict[str, tuple[int, dict[str, Any]]] = {}
        self._sources: dict[str, TensorSource] = {}
        self._scalex_layouts: dict[str, ScaleXLayout | None] = {}

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "ModelContract":
        root = Path(model_dir).expanduser().resolve()
        required = {
            "config": root / "config.json",
            "index": root / "model.safetensors.index.json",
            "validation": root / "VALIDATION.json",
            "composite": root / "livseek-composite.json",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise ContractError(f"native checkpoint metadata is incomplete: {missing}")

        def read(path: Path) -> dict[str, Any]:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ContractError(f"cannot read JSON metadata: {path}") from exc
            if not isinstance(value, dict):
                raise ContractError(f"JSON metadata must be an object: {path}")
            return value

        contract = cls(
            root,
            read(required["config"]),
            read(required["index"]),
            read(required["validation"]),
            read(required["composite"]),
        )
        contract.validate_native_format()
        return contract

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(self.weight_map)

    def validate_native_format(self) -> None:
        if self.validation.get("status") != "PASS":
            raise ContractError("VALIDATION.json does not report PASS")
        if self.composite.get("format") != COMPOSITE_FORMAT:
            raise ContractError(
                "unsupported expert format; v1 accepts only the native composite "
                f"{COMPOSITE_FORMAT!r}"
            )
        if self.composite.get("status") != "validated":
            raise ContractError("native composite is not marked validated")
        sources = self.composite.get("sources")
        expected = {
            OFFICIAL.label: (OFFICIAL.repo, OFFICIAL.revision),
            EXPERTS.label: (EXPERTS.repo, EXPERTS.revision),
        }
        if not isinstance(sources, dict):
            raise ContractError("native composite has no source identities")
        for label, (repo, revision) in expected.items():
            source = sources.get(label)
            if not isinstance(source, dict) or (
                source.get("repo"), source.get("revision")
            ) != (repo, revision):
                raise ContractError(f"native composite source changed: {label}")

    def validate_supported_profile(self) -> ContractInventory:
        text = self.config.get("text_config")
        if not isinstance(text, dict):
            raise ContractError("config.json has no text_config")
        expected = {
            "model_type": "glm5_next_text",
            "hidden_size": 4096,
            "num_hidden_layers": MAIN_LAYER_COUNT,
            "first_k_dense_replace": FIRST_MOE_LAYER,
            "n_routed_experts": EXPERTS_PER_LAYER,
            "num_experts_per_tok": TOP_K,
            "n_shared_experts": 1,
            "moe_intermediate_size": 2048,
            "hc_mult": 4,
            "qk_rope_head_dim": 0,
        }
        for key, value in expected.items():
            if text.get(key) != value:
                raise ContractError(
                    f"unsupported GLM text profile: {key}={text.get(key)!r}, expected {value!r}"
                )
        if self.config.get("architectures") != ["Glm5NextForConditionalGeneration"]:
            raise ContractError("unsupported GLM architecture declaration")
        layer_types = text.get("layer_types")
        mlp_types = text.get("mlp_layer_types")
        if not isinstance(layer_types, list) or len(layer_types) != MAIN_LAYER_COUNT:
            raise ContractError("layer_types must describe exactly 45 main layers")
        if not isinstance(mlp_types, list) or len(mlp_types) != MAIN_LAYER_COUNT:
            raise ContractError("mlp_layer_types must describe exactly 45 main layers")
        if mlp_types[:FIRST_MOE_LAYER] != ["dense"] * FIRST_MOE_LAYER or any(
            item != "sparse" for item in mlp_types[FIRST_MOE_LAYER:]
        ):
            raise ContractError("unsupported dense/MoE layer schedule")

        inventory = self.inventory(resolve_sources=True)
        expected_counts = {
            "resident_tensors": 1_425,
            "routed_tensors": 72_576,
            "mtp_tensors": 1_760,
            "vision_tensors": 347,
            "unknown_tensors": 0,
        }
        for key, value in expected_counts.items():
            if getattr(inventory, key) != value:
                raise ContractError(
                    f"native tensor inventory changed: {key}={getattr(inventory, key)} != {value}"
                )
        if len(self.weight_map) != 76_108:
            raise ContractError(
                f"native tensor count changed: {len(self.weight_map)} != 76108"
            )
        return inventory

    @staticmethod
    def partition(name: str) -> str:
        if name.startswith("model.visual."):
            return "vision"
        expert = ROUTED_EXPERT_RE.fullmatch(name)
        if expert is not None:
            layer = int(expert.group(1))
            if FIRST_MOE_LAYER <= layer <= LAST_MAIN_LAYER:
                return "routed"
            if layer == MTP_LAYER:
                return "mtp"
            return "unknown"
        layer_match = LANGUAGE_LAYER_RE.match(name)
        if layer_match is not None and int(layer_match.group(1)) == MTP_LAYER:
            return "mtp"
        if name.startswith("model.language_model.") or name == "lm_head.weight":
            return "resident"
        return "unknown"

    def names_for(self, partition: str) -> tuple[str, ...]:
        return tuple(name for name in self.weight_map if self.partition(name) == partition)

    def inventory(self, *, resolve_sources: bool = False) -> ContractInventory:
        counts = {key: 0 for key in ("resident", "routed", "mtp", "vision", "unknown")}
        payload = {key: 0 for key in counts}
        for name in self.weight_map:
            partition = self.partition(name)
            counts[partition] += 1
            if resolve_sources:
                payload[partition] += self.tensor(name).byte_length
        return ContractInventory(
            resident_tensors=counts["resident"],
            routed_tensors=counts["routed"],
            mtp_tensors=counts["mtp"],
            vision_tensors=counts["vision"],
            unknown_tensors=counts["unknown"],
            resident_payload_bytes=payload["resident"],
            routed_payload_bytes=payload["routed"],
            mtp_payload_bytes=payload["mtp"],
            vision_payload_bytes=payload["vision"],
        )

    def _header(self, shard_name: str) -> tuple[int, dict[str, Any]]:
        cached = self._headers.get(shard_name)
        if cached is not None:
            return cached
        path = self.model_dir / shard_name
        if not path.is_file():
            raise ContractError(f"indexed shard is missing: {path}")
        try:
            layout = self.scalex_layout(shard_name)
            if layout is None:
                data_base, header = local_safetensors_header(path)
            else:
                data_base, header = layout.virtual_data_base, layout.virtual_header
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot read safetensors header: {path}") from exc
        cached = (data_base, header)
        self._headers[shard_name] = cached
        return cached

    def scalex_layout(self, shard_name: str) -> ScaleXLayout | None:
        if shard_name in self._scalex_layouts:
            return self._scalex_layouts[shard_name]
        from .scalex_container import is_scalex_layer, read_scalex_layout

        path = self.model_dir / shard_name
        layout = read_scalex_layout(path) if is_scalex_layer(path) else None
        self._scalex_layouts[shard_name] = layout
        return layout

    def tensor(self, name: str) -> TensorSource:
        cached = self._sources.get(name)
        if cached is not None:
            return cached
        try:
            shard_name = self.weight_map[name]
        except KeyError as exc:
            raise ContractError(f"tensor is absent from model index: {name}") from exc
        data_base, header = self._header(shard_name)
        entry = header.get(name)
        if not isinstance(entry, dict):
            raise ContractError(f"indexed tensor is absent from {shard_name}: {name}")
        try:
            dtype = str(entry["dtype"])
            shape = tuple(int(value) for value in entry["shape"])
            start, end = (int(value) for value in entry["data_offsets"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid safetensors metadata for {name}") from exc
        byte_length = end - start
        expected = tensor_payload_bytes(dtype, shape)
        if start < 0 or byte_length != expected:
            raise ContractError(
                f"invalid tensor byte range for {name}: {byte_length} != {expected}"
            )
        absolute_offset = data_base + start
        layout = self.scalex_layout(shard_name)
        shard_size = (
            layout.original_bytes
            if layout is not None
            else (self.model_dir / shard_name).stat().st_size
        )
        if absolute_offset + byte_length > shard_size:
            raise ContractError(f"tensor exceeds shard size: {name}")
        source = TensorSource(
            name=name,
            shard_name=shard_name,
            absolute_offset=absolute_offset,
            byte_length=byte_length,
            dtype=dtype,
            shape=shape,
        )
        self._sources[name] = source
        return source

    def audit_headers(self) -> None:
        names_by_shard: dict[str, set[str]] = {}
        for name, shard in self.weight_map.items():
            names_by_shard.setdefault(shard, set()).add(name)
        for shard, indexed_names in names_by_shard.items():
            _, header = self._header(shard)
            actual_names = set(header) - {"__metadata__"}
            if actual_names != indexed_names:
                raise ContractError(
                    f"index/header inventory mismatch in {shard}: "
                    f"missing={len(indexed_names - actual_names)}, "
                    f"extra={len(actual_names - indexed_names)}"
                )
            for name in indexed_names:
                self.tensor(name)


def routed_expert_parts(name: str) -> tuple[int, int, str, str]:
    match = ROUTED_EXPERT_RE.fullmatch(name)
    if match is None:
        raise ContractError(f"not a native routed-expert tensor: {name}")
    return (
        int(match.group(1)),
        int(match.group(2)),
        match.group(3),
        match.group(4),
    )
