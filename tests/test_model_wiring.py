from pathlib import Path
from types import SimpleNamespace

import pytest

from glm53flash.contract import ContractError, ModelContract
from glm53flash.expert_reader import NativeExpertReader
from glm53flash.expert_ssd import ExpertSSD
from glm53flash.model import GLMForCausalLM
from glm53flash.model_config import GLMTextConfig
from glm53flash.runtime import DEFAULT_MODEL_DIR, TargetRuntime
from tests.helpers import tiny_config


class EmptyReader:
    plan = SimpleNamespace(first_layer=3, last_layer=3, experts_per_layer=4)

    def load(self, layer, expert):
        raise AssertionError("model construction must not read experts")


def test_sparse_model_rejects_non_expertssd_backend():
    with pytest.raises(ContractError, match="must be backed by ExpertSSD"):
        GLMForCausalLM(
            tiny_config(),
            context_limit=32,
            expert_factory=lambda _layer: object(),
        )


def test_sparse_model_has_no_resident_expert_fallback():
    reader = EmptyReader()
    model = GLMForCausalLM(
        tiny_config(),
        context_limit=32,
        expert_factory=lambda layer: ExpertSSD(
            reader,
            layer=layer,
            capacity=2,
            swiglu_limit=10.0,
        ),
    )
    assert len(model.expert_layers()) == 1
    assert isinstance(model.expert_layers()[0].experts, ExpertSSD)
    assert not hasattr(model.expert_layers()[0], "resident_experts")


@pytest.mark.skipif(not (DEFAULT_MODEL_DIR / "VALIDATION.json").is_file(), reason="local composite absent")
def test_real_checkpoint_all_resident_destinations_exist_without_loading_weights():
    preflight = TargetRuntime.preflight(
        DEFAULT_MODEL_DIR,
        memory_gib=24,
        physical_bytes=256 * 2**30,
    )
    contract = ModelContract.from_model_dir(preflight.model_dir)
    config = GLMTextConfig.from_model_dict(contract.config)
    reader = NativeExpertReader(preflight.expert_plan)
    try:
        model = GLMForCausalLM(
            config,
            context_limit=128,
            expert_factory=lambda layer: ExpertSSD(
                reader,
                layer=layer,
                capacity=8,
                swiglu_limit=config.swiglu_limit,
            ),
        )
        for tensor in preflight.resident_plan.tensors:
            target = model
            parts = tensor.destination_name.split(".")
            for part in parts[:-1]:
                target = target[int(part)] if part.isdigit() else getattr(target, part)
            assert hasattr(target, parts[-1]), tensor.destination_name
            assert getattr(target, parts[-1]) is None, tensor.destination_name
    finally:
        reader.close()
