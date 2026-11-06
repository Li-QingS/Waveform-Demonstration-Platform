import json
from urllib.error import HTTPError

import pytest

from waveform_sim.service.ai_log_analyzer import (
    DeepSeekConfig,
    analyze_log,
    build_analysis_prompt,
    get_api_key,
    prepare_log_text,
)


class _Response:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def getcode(self):
        return self.status


def test_prepare_log_text_keeps_latest_characters_when_truncated():
    text, truncated = prepare_log_text(["first", "second", "latest"], max_chars=9)

    assert text == "nd\nlatest"
    assert truncated is True


def test_prepare_log_text_handles_empty_entries():
    assert prepare_log_text([], max_chars=10) == ("", False)


def test_prepare_log_text_with_zero_limit_returns_empty_tail():
    assert prepare_log_text(["some log"], max_chars=0) == ("", True)


def test_build_analysis_prompt_mentions_evidence_sections_and_truncation():
    prompt = build_analysis_prompt("line one", truncated=True)

    assert "line one" in prompt
    assert "运行概况" in prompt
    assert "指标表现" in prompt
    assert "异常和证据" in prompt
    assert "可能原因" in prompt
    assert "建议动作" in prompt
    assert "截断" in prompt


def test_get_api_key_prefers_explicit_value_then_environment():
    assert get_api_key(" explicit-key ", {"DEEPSEEK_API_KEY": "env-key"}) == "explicit-key"
    assert get_api_key("", {"DEEPSEEK_API_KEY": " env-key "}) == "env-key"
    assert get_api_key("", {}) == ""


def test_analyze_log_builds_openai_compatible_request_and_parses_response():
    seen = {}

    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["headers"] = dict(request.headers)
        seen["timeout"] = timeout
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return _Response({"choices": [{"message": {"content": "诊断报告"}}]})

    result = analyze_log(
        "[DEBUG] link started",
        DeepSeekConfig("https://api.example.test/v1", "deepseek-chat", "secret", 12.5),
        opener=opener,
    )

    assert result == "诊断报告"
    assert seen["url"] == "https://api.example.test/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer secret"
    assert seen["headers"]["Content-type"] == "application/json"
    assert seen["timeout"] == 12.5
    assert seen["body"]["model"] == "deepseek-chat"
    assert "运行概况" in seen["body"]["messages"][-1]["content"]
    assert seen["body"]["messages"][-1]["content"].endswith("[DEBUG] link started")
    assert "secret" not in json.dumps(seen["body"])


def test_analyze_log_rejects_empty_log_without_network_call():
    called = False

    def opener(*args):
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="日志为空"):
        analyze_log("\n  ", DeepSeekConfig("https://api.example.test", "model", "key"), opener)
    assert called is False


def test_analyze_log_rejects_missing_api_key():
    with pytest.raises(ValueError, match="API Key"):
        analyze_log("log", DeepSeekConfig("https://api.example.test", "model", ""))


def test_analyze_log_converts_http_errors_to_displayable_runtime_error():
    def opener(request, timeout):
        raise HTTPError(request.full_url, 429, "Too Many Requests", {}, None)

    with pytest.raises(RuntimeError, match="HTTP 429"):
        analyze_log("log", DeepSeekConfig("https://api.example.test", "model", "key"), opener)


def test_analyze_log_rejects_invalid_response_shape():
    def opener(request, timeout):
        return _Response({"choices": []})

    with pytest.raises(RuntimeError, match="响应格式"):
        analyze_log("log", DeepSeekConfig("https://api.example.test", "model", "key"), opener)
