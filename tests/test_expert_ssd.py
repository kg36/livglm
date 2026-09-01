from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from glm53flash.expert_reader import LoadedExpert
from glm53flash.expert_ssd import ExpertSSD, limited_swiglu


class FakeReader:
    def __init__(self, experts):
        self.plan = SimpleNamespace(first_layer=0, last_layer=0, experts_per_layer=len(experts))
        self.experts = experts
        self.loads = []

    def load(self, layer, expert):
        self.loads.append((layer, expert))
        return self.experts[expert]


def _quantized_expert(expert_id, matrices):
    tensors = {}
    for name, matrix in matrices.items():
        packed, scales = mx.quantize(mx.array(matrix), group_size=32, bits=4, mode="mxfp4")
        tensors[f"{name}.weight"] = packed
        tensors[f"{name}.scales"] = scales
    mx.eval(*tensors.values())
    return LoadedExpert(0, expert_id, tensors, 1, 1, sum(x.nbytes for x in tensors.values()))


def test_expert_ssd_uses_selected_compact_experts_and_caches():
    rng = np.random.default_rng(8)
    matrices = []
    loaded = []
    for expert in range(3):
        values = {
            "gate_proj": rng.normal(0, 0.2, (32, 32)).astype(np.float32),
            "up_proj": rng.normal(0, 0.2, (32, 32)).astype(np.float32),
            "down_proj": rng.normal(0, 0.2, (32, 32)).astype(np.float32),
        }
        matrices.append(values)
        loaded.append(_quantized_expert(expert, values))
    reader = FakeReader(loaded)
    module = ExpertSSD(reader, layer=0, capacity=2, swiglu_limit=10.0)
    x = mx.array(rng.normal(0, 0.2, (1, 32)).astype(np.float32))
    indices = mx.array([[2, 0]], dtype=mx.uint32)

    actual = module(x, indices)
    expected = []
    for expert in (2, 0):
        item = loaded[expert]
        gate_w, gate_s = item.projection("gate_proj")
        up_w, up_s = item.projection("up_proj")
        down_w, down_s = item.projection("down_proj")
        gate = mx.quantized_matmul(x, gate_w, gate_s, transpose=True, group_size=32, bits=4, mode="mxfp4")
        up = mx.quantized_matmul(x, up_w, up_s, transpose=True, group_size=32, bits=4, mode="mxfp4")
        activated = limited_swiglu(gate, up, 10.0)
        expected.append(mx.quantized_matmul(activated, down_w, down_s, transpose=True, group_size=32, bits=4, mode="mxfp4"))
    expected = mx.stack(expected, axis=1)
    mx.eval(actual, expected)

    assert actual.shape == (1, 2, 32)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5)
    assert reader.loads == [(0, 2), (0, 0)]
    module(x, indices)
    mx.eval(actual)
    assert reader.loads == [(0, 2), (0, 0)]
    assert module.stats().hits == 2
