import pytest

from glm53flash.chat_cli import build_parser, main
from glm53flash.runtime import DEFAULT_MODEL_DIR, TargetRuntime


def test_cli_keeps_basic_deepseek_user_surface():
    args = build_parser().parse_args(
        ["--chat", "--memory", "24", "--max-tokens", "1", "--thinking"]
    )
    assert args.chat
    assert args.memory == 24
    assert args.max_tokens == 1
    assert args.thinking
    assert not args.resident_bf16

    oracle = build_parser().parse_args(["--resident-bf16", "hello"])
    assert oracle.resident_bf16

    compact = build_parser().parse_args(["--resident-mxfp4", "hello"])
    assert compact.resident_mxfp4

    traced = build_parser().parse_args(
        [
            "--trace",
            "artifacts/decode.json",
            "--trace-decode-start",
            "2",
            "--trace-decode-steps",
            "24",
            "hello",
        ]
    )
    assert traced.trace.name == "decode.json"
    assert traced.trace_decode_start == 2
    assert traced.trace_decode_steps == 24


def test_cli_help_does_not_load_model(capsys):
    with pytest.raises(SystemExit) as stopped:
        build_parser().parse_args(["--help"])
    assert stopped.value.code == 0
    assert "ExpertSSD" in capsys.readouterr().out


def test_cli_rejects_conflicting_resident_formats(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--resident-bf16", "--resident-mxfp4", "--preflight"])
    assert stopped.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err


@pytest.mark.skipif(not (DEFAULT_MODEL_DIR / "chat_template.jinja").is_file(), reason="local tokenizer absent")
def test_official_chat_template_renders_with_runtime_dependencies():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_DIR, trust_remote_code=False)
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=True,
        return_dict=True,
        add_generation_prompt=True,
        clear_thinking=True,
        reasoning_effort="max",
    )
    assert encoded["input_ids"]

    runtime = object.__new__(TargetRuntime)
    runtime.tokenizer = tokenizer
    messages = [{"role": "user", "content": "hello"}]
    direct = runtime.encode_messages(messages, thinking=False)
    thinking = runtime.encode_messages(messages, thinking=True)
    assert direct[-2:] == [154841, 154842]
    assert thinking[-1] == 154841
