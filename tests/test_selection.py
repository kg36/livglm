from glm53flash.sources import mxfp4_routed_tensor, official_routed_tensor


def test_routed_selection():
    base = "model.language_model.layers.3.mlp.experts.0.gate_proj."
    assert official_routed_tensor(base + "weight")
    assert official_routed_tensor(base + "weight_scale_inv")
    assert mxfp4_routed_tensor(base + "weight_packed")
    assert mxfp4_routed_tensor(base + "weight_scale")
    assert not official_routed_tensor("model.language_model.layers.2.mlp.gate_proj.weight")
