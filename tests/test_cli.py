import pytest

from glm53flash.chat_cli import build_parser
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


def test_cli_help_does_not_load_model(capsys):
    with pytest.raises(SystemExit) as stopped:
        build_parser().parse_args(["--help"])
    assert stopped.value.code == 0
    assert "ExpertSSD" in capsys.readouterr().out


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
