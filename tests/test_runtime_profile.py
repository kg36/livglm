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
    assert profile.expert_capacity == 46
    assert profile.planned_gib <= profile.effective_gib


def test_large_budget_caps_at_the_complete_expert_bank():
    profile = resolve_runtime_profile(200, physical_bytes=256 * 2**30)
    assert profile.expert_capacity == 288
    assert profile.expert_cache_gib == pytest.approx(150.609375)
    assert profile.planned_gib <= profile.effective_gib


def test_30_gib_is_a_hard_total_budget_with_19_slots_per_layer():
    profile = resolve_runtime_profile(30, physical_bytes=256 * 2**30)
    assert profile.effective_gib == 30
    assert profile.expert_capacity == 19
    assert profile.expert_cache_gib == pytest.approx(9.93603515625)
    assert profile.planned_gib == pytest.approx(29.553251497)
    assert profile.budget_headroom_gib == pytest.approx(0.446748503)


def test_profile_rejects_less_than_one_topk_bank():
    with pytest.raises(ContractError, match="below routed top-k 8"):
        resolve_runtime_profile(20, physical_bytes=256 * 2**30)
