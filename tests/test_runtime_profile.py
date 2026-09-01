import pytest

from glm53flash.contract import ContractError
from glm53flash.runtime import resolve_runtime_profile


def test_24_gib_profile_is_exact_topk_capacity():
    profile = resolve_runtime_profile(24, physical_bytes=256 * 2**30)
    assert profile.expert_capacity == 8
    assert profile.expert_cache_gib == pytest.approx(4.18359375)


def test_profile_clamps_to_physical_memory():
    profile = resolve_runtime_profile(200, physical_bytes=48 * 2**30)
    assert profile.effective_gib == 44
    assert profile.expert_capacity == 8


def test_v1_does_not_spend_large_machine_memory_before_tracing():
    profile = resolve_runtime_profile(200, physical_bytes=256 * 2**30)
    assert profile.expert_capacity == 8
    assert profile.expert_cache_gib == pytest.approx(4.18359375)


def test_profile_rejects_less_than_one_topk_bank():
    with pytest.raises(ContractError, match="below routed top-k 8"):
        resolve_runtime_profile(20, physical_bytes=256 * 2**30)
