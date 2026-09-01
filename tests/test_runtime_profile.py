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


def test_30_gib_bf16_oracle_is_a_hard_total_budget_with_19_slots_per_layer():
    profile = resolve_runtime_profile(30, physical_bytes=256 * 2**30)
    assert profile.effective_gib == 30
    assert profile.expert_capacity == 19
    assert profile.expert_cache_gib == pytest.approx(9.93603515625)
    assert profile.planned_gib == pytest.approx(29.553251497)
    assert profile.budget_headroom_gib == pytest.approx(0.446748503)


def test_30_gib_mxfp8_runtime_has_33_slots_under_the_same_hard_budget():
    profile = resolve_runtime_profile(
        30,
        resident_bytes=9_980_754_168,
        resident_load_bytes=17_842_600_184,
        resident_format="mxfp8",
        resident_linear_count=497,
        physical_bytes=256 * 2**30,
    )
    assert profile.effective_gib == 30
    assert profile.resident_format == "mxfp8"
    assert profile.resident_gib == pytest.approx(9.295301668)
    assert profile.resident_load_gib == pytest.approx(16.617216341)
    assert profile.resident_linear_count == 497
    assert profile.expert_capacity == 33
    assert profile.expert_cache_gib == pytest.approx(17.257324219)
    assert profile.planned_gib == pytest.approx(29.552625887)
    assert profile.budget_headroom_gib == pytest.approx(0.447374113)


def test_profile_rejects_less_than_one_topk_bank():
    with pytest.raises(ContractError, match="below routed top-k 8"):
        resolve_runtime_profile(20, physical_bytes=256 * 2**30)
