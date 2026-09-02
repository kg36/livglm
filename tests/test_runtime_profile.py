import pytest

from glm53flash.contract import ContractError
from glm53flash.runtime import resolve_runtime_profile


def test_24_gib_profile_uses_all_safe_cache_headroom():
    profile = resolve_runtime_profile(24, physical_bytes=256 * 2**30)
    assert profile.expert_capacity == 13
    assert profile.expert_cache_gib == pytest.approx(6.79833984375)


def test_profile_clamps_to_physical_memory():
    profile = resolve_runtime_profile(200, physical_bytes=48 * 2**30)
    assert profile.effective_gib == 44
    assert profile.expert_capacity == 51
    assert profile.planned_gib <= profile.effective_gib


def test_large_budget_caps_at_the_complete_expert_bank():
    profile = resolve_runtime_profile(200, physical_bytes=256 * 2**30)
    assert profile.expert_capacity == 288
    assert profile.expert_cache_gib == pytest.approx(150.609375)
    assert profile.planned_gib <= profile.effective_gib


def test_30_gib_bf16_oracle_is_a_hard_total_budget_with_24_slots_per_layer():
    profile = resolve_runtime_profile(30, physical_bytes=256 * 2**30)
    assert profile.effective_gib == 30
    assert profile.expert_capacity == 24
    assert profile.expert_cache_gib == pytest.approx(12.55078125)
    assert profile.planned_gib == pytest.approx(29.667997591)
    assert profile.budget_headroom_gib == pytest.approx(0.332002409)


def test_30_gib_mxfp8_runtime_has_38_slots_under_the_same_hard_budget():
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
    assert profile.expert_capacity == 38
    assert profile.expert_cache_gib == pytest.approx(19.8720703125)
    assert profile.planned_gib == pytest.approx(29.667371981)
    assert profile.budget_headroom_gib == pytest.approx(0.332628019)


def test_30_gib_mxfp4_runtime_has_45_slots_under_the_same_hard_budget():
    profile = resolve_runtime_profile(
        30,
        resident_bytes=5_923_027_192,
        resident_load_bytes=17_842_600_184,
        resident_format="mxfp4",
        resident_linear_count=497,
        physical_bytes=256 * 2**30,
    )
    assert profile.resident_format == "mxfp4"
    assert profile.expert_capacity == 45
    assert profile.planned_gib <= 30


def test_30_gib_scalex_profile_uses_exact_per_layer_rows_for_48_slots():
    profile = resolve_runtime_profile(
        30,
        resident_bytes=5_923_027_192,
        resident_load_bytes=17_842_600_184,
        resident_format="mxfp4",
        resident_linear_count=497,
        expert_source_format="livglm_scalex_mode_b",
        expert_slot_bytes_by_layer=(12_715_000,) * 42,
        physical_bytes=256 * 2**30,
    )
    assert profile.expert_source_format == "livglm_scalex_mode_b"
    assert profile.expert_capacity == 48
    assert profile.planned_gib <= 30


def test_mtp_auxiliary_reservation_keeps_46_target_slots_under_30_gib():
    profile = resolve_runtime_profile(
        30,
        resident_bytes=5_923_027_192,
        resident_load_bytes=17_842_600_184,
        resident_format="mxfp4",
        resident_linear_count=497,
        expert_source_format="livglm_scalex_mode_b",
        expert_slot_bytes_by_layer=(12_715_000,) * 42,
        physical_bytes=256 * 2**30,
        auxiliary_gib=0.75,
    )
    assert profile.auxiliary_gib == 0.75
    assert profile.expert_capacity == 46
    assert profile.planned_gib <= 30


def test_profile_rejects_less_than_one_topk_bank():
    with pytest.raises(ContractError, match="below routed top-k 8"):
        resolve_runtime_profile(20, physical_bytes=256 * 2**30)
