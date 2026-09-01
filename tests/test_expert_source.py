from types import SimpleNamespace

from glm53flash.contract import TensorSource
from glm53flash.expert_source import build_native_expert_source_plan
from tests.helpers import tiny_config


class FakeContract:
    model_dir = "/tmp/native"

    def __init__(self):
        self.sources = {}
        offset = 64
        for expert in range(2):
            for component in ("down_proj", "gate_proj", "up_proj"):
                rows, columns = (32, 32)
                for part, width in (("weight_packed", columns // 2), ("weight_scale", columns // 32)):
                    name = f"model.language_model.layers.0.mlp.experts.{expert}.{component}.{part}"
                    byte_length = rows * width
                    self.sources[name] = TensorSource(
                        name=name,
                        shard_name="model.safetensors",
                        absolute_offset=offset,
                        byte_length=byte_length,
                        dtype="U8",
                        shape=(rows, width),
                    )
                    offset += byte_length

    def tensor(self, name):
        return self.sources[name]


def test_native_source_plan_coalesces_exact_expert_record():
    config = tiny_config(
        hidden_size=32,
        moe_intermediate_size=32,
        n_routed_experts=2,
        num_experts_per_tok=2,
    )
    plan = build_native_expert_source_plan(FakeContract(), config, first_layer=0, last_layer=0)
    assert plan.expert_count == 2
    assert plan.tensor_count == 12
    assert plan.uniform_read_count == 1
    assert plan.read_bytes == plan.payload_bytes
    assert plan.expert(0, 1).read_ranges[0].tensors[0].mlx_dtype == "U32"
