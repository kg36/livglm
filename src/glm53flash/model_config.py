"""Normalized text-only GLM5Next configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contract import ContractError


@dataclass(frozen=True)
class GLMTextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    first_k_dense_replace: int
    layer_types: tuple[str, ...]
    mlp_layer_types: tuple[str, ...]
    hidden_act: str
    swiglu_limit: float
    rms_norm_eps: float
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    num_attention_heads: int
    num_key_value_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    linear_num_heads: int
    linear_head_dim: int
    linear_conv_kernel: int
    linear_gate_lower_bound: float | None
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    index_kpool_always_select_tail: bool
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    n_group: int
    topk_group: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    scoring_func: str
    topk_method: str
    eos_token_ids: tuple[int, ...]
    pad_token_id: int
    max_position_embeddings: int

    @classmethod
    def from_model_dict(cls, config: dict[str, Any]) -> "GLMTextConfig":
        text = config.get("text_config")
        if not isinstance(text, dict):
            raise ContractError("config has no text_config object")
        linear = text.get("linear_attn_config")
        if not isinstance(linear, dict):
            raise ContractError("text_config has no linear_attn_config object")

        def integer(mapping: dict[str, Any], key: str) -> int:
            value = mapping.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ContractError(f"configuration field must be an integer: {key}")
            return value

        def number(mapping: dict[str, Any], key: str) -> float:
            value = mapping.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractError(f"configuration field must be numeric: {key}")
            return float(value)

        eos = text.get("eos_token_id")
        if isinstance(eos, int):
            eos_ids = (eos,)
        elif isinstance(eos, list) and eos and all(isinstance(value, int) for value in eos):
            eos_ids = tuple(eos)
        else:
            raise ContractError("eos_token_id must be an integer or non-empty integer list")
        layer_types = text.get("layer_types")
        mlp_types = text.get("mlp_layer_types")
        if not isinstance(layer_types, list) or not all(isinstance(value, str) for value in layer_types):
            raise ContractError("layer_types must be a string list")
        if not isinstance(mlp_types, list) or not all(isinstance(value, str) for value in mlp_types):
            raise ContractError("mlp_layer_types must be a string list")

        result = cls(
            vocab_size=integer(text, "vocab_size"),
            hidden_size=integer(text, "hidden_size"),
            intermediate_size=integer(text, "intermediate_size"),
            moe_intermediate_size=integer(text, "moe_intermediate_size"),
            num_hidden_layers=integer(text, "num_hidden_layers"),
            first_k_dense_replace=integer(text, "first_k_dense_replace"),
            layer_types=tuple(layer_types),
            mlp_layer_types=tuple(mlp_types),
            hidden_act=str(text.get("hidden_act")),
            swiglu_limit=number(text, "swiglu_limit"),
            rms_norm_eps=number(text, "rms_norm_eps"),
            hc_mult=integer(text, "hc_mult"),
            hc_sinkhorn_iters=integer(text, "hc_sinkhorn_iters"),
            hc_eps=number(text, "hc_eps"),
            num_attention_heads=integer(text, "num_attention_heads"),
            num_key_value_heads=integer(text, "num_key_value_heads"),
            q_lora_rank=integer(text, "q_lora_rank"),
            kv_lora_rank=integer(text, "kv_lora_rank"),
            qk_nope_head_dim=integer(text, "qk_nope_head_dim"),
            qk_rope_head_dim=integer(text, "qk_rope_head_dim"),
            v_head_dim=integer(text, "v_head_dim"),
            linear_num_heads=integer(linear, "num_heads"),
            linear_head_dim=integer(linear, "head_dim"),
            linear_conv_kernel=integer(linear, "short_conv_kernel_size"),
            linear_gate_lower_bound=(
                None
                if linear.get("gate_lower_bound") is None
                else number(linear, "gate_lower_bound")
            ),
            index_n_heads=integer(text, "index_n_heads"),
            index_head_dim=integer(text, "index_head_dim"),
            index_topk=integer(text, "index_topk"),
            index_kpool=integer(text, "index_kpool"),
            index_kpool_always_select_tail=bool(text.get("index_kpool_always_select_tail")),
            n_routed_experts=integer(text, "n_routed_experts"),
            num_experts_per_tok=integer(text, "num_experts_per_tok"),
            n_shared_experts=integer(text, "n_shared_experts"),
            n_group=integer(text, "n_group"),
            topk_group=integer(text, "topk_group"),
            norm_topk_prob=bool(text.get("norm_topk_prob")),
            routed_scaling_factor=number(text, "routed_scaling_factor"),
            scoring_func=str(text.get("scoring_func")),
            topk_method=str(text.get("topk_method")),
            eos_token_ids=eos_ids,
            pad_token_id=integer(text, "pad_token_id"),
            max_position_embeddings=integer(text, "max_position_embeddings"),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.hidden_act != "silu":
            raise ContractError(f"v1 supports only SiLU, got {self.hidden_act!r}")
        if self.scoring_func != "sigmoid" or self.topk_method != "noaux_tc":
            raise ContractError("v1 supports only sigmoid/noaux_tc routing")
        if self.qk_rope_head_dim != 0:
            raise ContractError("v1 short-context DSA requires qk_rope_head_dim=0")
        if self.num_attention_heads != self.num_key_value_heads:
            raise ContractError("v1 DSA requires equal attention and KV head counts")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ContractError("layer_types length differs from num_hidden_layers")
        if len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ContractError("mlp_layer_types length differs from num_hidden_layers")
        if any(value not in {"linear_attention", "deepseek_sparse_attention"} for value in self.layer_types):
            raise ContractError("unsupported attention layer type")
        if any(value not in {"dense", "sparse"} for value in self.mlp_layer_types):
            raise ContractError("unsupported MLP layer type")
        if not 1 <= self.num_experts_per_tok <= self.n_routed_experts:
            raise ContractError("invalid routed top-k geometry")
        if self.n_routed_experts % self.n_group:
            raise ContractError("expert count must be divisible by n_group")
        if not 1 <= self.topk_group <= self.n_group:
            raise ContractError("invalid topk_group")
        if self.hc_mult < 1 or self.hc_sinkhorn_iters < 1:
            raise ContractError("invalid mHC configuration")
        if self.linear_conv_kernel < 1:
            raise ContractError("invalid KDA convolution kernel")
        if self.index_topk < 1 or self.index_kpool < 1:
            raise ContractError("invalid DSA index configuration")
        if self.index_topk % self.index_kpool:
            raise ContractError("v1 DSA requires index_topk divisible by index_kpool")
        if not self.index_kpool_always_select_tail:
            raise ContractError("v1 dense DSA equivalence requires the visible tail")
        if self.n_routed_experts // self.n_group < 2:
            raise ContractError("router groups must contain at least two experts")

    @property
    def qkv_dim(self) -> int:
        return self.linear_num_heads * self.linear_head_dim

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim
