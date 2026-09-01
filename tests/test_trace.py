import json
import threading

from glm53flash.trace import DecodeTrace, active_trace
from glm53flash.trace_summary import summarize_trace


def test_decode_trace_records_selected_step_worker_flow_and_summary(tmp_path):
    path = tmp_path / "decode.json"
    with DecodeTrace(path, decode_start=1, decode_steps=1) as trace:
        with trace.decode_step(0, token_id=10) as selected:
            assert not selected
        with trace.decode_step(1, token_id=11) as selected:
            assert selected
            assert active_trace() is trace
            with trace.span("model_forward", category="python"):
                with trace.span(
                    "transformer_layer",
                    category="model_structure",
                    args={"layer": 3},
                ):
                    pass
            with trace.span("Metal GPU compute", category="metal_gpu"):
                pass
            flow_id = trace.new_flow()
            trace.flow("s", flow_id)

            def worker():
                trace.flow("t", flow_id)
                with trace.span(
                    "SSD_read",
                    category="ssd_worker",
                    args={"payload_bytes": 4096},
                    force=True,
                ):
                    pass
                trace.flow("f", flow_id)

            thread = threading.Thread(target=worker, name="glm-expert-ssd-test")
            thread.start()
            thread.join()
            trace.counter(
                "decode_step_metrics",
                {"logical_read_bytes": 4096, "metal_dispatches": 7},
            )
    assert active_trace() is None

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["otherData"]["format"] == "livglm-perfetto-v1"
    assert {event["name"] for event in payload["traceEvents"]} >= {
        "decode_step",
        "model_forward",
        "transformer_layer",
        "SSD_read",
        "decode_step_metrics",
    }
    assert any(
        event.get("args", {}).get("name") == "glm-expert-ssd-test"
        for event in payload["traceEvents"]
        if event.get("name") == "thread_name"
    )

    summary = summarize_trace(path)
    assert summary["decode_steps"] == 1
    assert summary["steps"][0]["ssd_read_count"] == 1
    assert summary["steps"][0]["ssd_payload_bytes"] == 4096
    assert summary["steps"][0]["transformer_layers"] == 1
    assert summary["steps"][0]["metal_gpu_interval_count"] == 1
    assert summary["aggregate"]["metal_gpu_covered_steps"] == 1
    assert summary["steps"][0]["metrics"]["metal_dispatches"] == 7


def test_trace_rejects_invalid_decode_window(tmp_path):
    path = tmp_path / "decode.json"
    for start, steps in ((-1, 1), (0, 0)):
        try:
            DecodeTrace(path, decode_start=start, decode_steps=steps)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid trace window was accepted")
