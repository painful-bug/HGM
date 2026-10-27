from types import SimpleNamespace
from unittest.mock import Mock

import llm
import llm_withtools


def response(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls), finish_reason="stop")], usage=None)


def test_missing_nvidia_key_uses_ollama(monkeypatch):
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)
    client, model = llm.create_client(llm.NIM_MODEL)
    assert model == llm.OLLAMA_FALLBACK
    assert str(client.base_url) == "http://localhost:11434/v1/"
    client.close()


def test_nvidia_failure_falls_back_for_tools(monkeypatch):
    primary, fallback = Mock(), Mock()
    primary.chat.completions.create.side_effect = ValueError("empty")
    call = SimpleNamespace(id="id", function=SimpleNamespace(name="read", arguments='{"x": 1}'))
    fallback.chat.completions.create.return_value = response(tool_calls=[call])
    monkeypatch.setattr(llm, "create_client", lambda model: (fallback, model))
    result = llm_withtools.get_response_withtools(primary, llm.NIM_MODEL, [{"role": "user", "content": "hi"}], [{"type": "function"}], "auto", logging=lambda _: None)
    assert llm_withtools.check_for_tool_use(result, llm.NIM_MODEL)["tool_input"] == {"x": 1}
    req = fallback.chat.completions.create.call_args.kwargs
    assert req["model"] == "qwen3.5:4b" and req["tools"] == [{"type": "function"}]
    assert "extra_body" not in req and "reasoning_effort" not in req
    fallback.close.assert_called_once()


def test_explicit_ollama_uses_chat_format(monkeypatch):
    client = Mock()
    client.chat.completions.create.return_value = response(content="done")
    text, history = llm.get_response_from_llm("hello", client, "ollama/qwen3.5:4b", "system")
    assert text == "done" and history[-1]["content"] == "done"
    assert client.chat.completions.create.call_args.kwargs["model"] == "qwen3.5:4b"
    assert llm_withtools.convert_msg_history([{"role": "user", "content": "hi"}], "ollama/qwen3.5:4b")[0]["content"] == "hi"
