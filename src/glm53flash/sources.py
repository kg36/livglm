from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Source:
    label: str
    repo: str
    revision: str

    @property
    def base_url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}"


OFFICIAL = Source(
    "official_remainder",
    "zai-org/GLM-5.3-Flash",
    "03eb5366286afd40d2221b1d9c63a6dd1ba4832e",
)
EXPERTS = Source(
    "mxfp4_experts",
    "INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound",
    "8712b4a299e2cbb81c019d2c20084fb99cbc2d00",
)
BF16 = Source(
    "bf16_validation",
    "zai-org/GLM-5.3-Flash-BF16",
    "61f77a1e1a67c410650ce5017411337da0dcd11a",
)

SOURCES = {source.label: source for source in (OFFICIAL, EXPERTS, BF16)}

EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(down_proj|gate_proj|up_proj)\."
    r"(weight|weight_scale_inv|weight_packed|weight_scale)$"
)


def official_routed_tensor(name: str) -> bool:
    match = EXPERT_RE.match(name)
    return bool(match and 3 <= int(match.group(1)) <= 45)


def mxfp4_routed_tensor(name: str) -> bool:
    match = EXPERT_RE.match(name)
    return bool(
        match
        and 3 <= int(match.group(1)) <= 45
        and match.group(4) in {"weight_packed", "weight_scale"}
    )


def expert_layer(name: str) -> int:
    match = EXPERT_RE.match(name)
    if not match:
        raise ValueError(f"not a routed expert tensor: {name}")
    return int(match.group(1))
