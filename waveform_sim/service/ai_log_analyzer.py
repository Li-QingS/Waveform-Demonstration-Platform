"""DeepSeek-compatible log analysis service."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Callable, Any
from urllib import error as urllib_error
from urllib import request as urllib_request


@dataclass(frozen=True)
class DeepSeekConfig:
    base_url: str
    model: str
    api_key: str
    timeout_s: float = 60.0


def prepare_log_text(entries: Iterable[str], max_chars: int = 24000) -> tuple[str, bool]:
    """Join log entries and retain the newest tail when over the limit."""
    if max_chars < 0:
        raise ValueError("max_chars must be non-negative")
    text = "\n".join(str(entry) for entry in entries)
    if len(text) <= max_chars:
        return text, False
    if max_chars == 0:
        return "", True
    return text[-max_chars:], True


def build_analysis_prompt(log_text: str, truncated: bool) -> str:
    truncation_note = (
        "日志已截断：超过长度上限，以下内容仅包含最新的尾部；请明确说明分析可能不完整。"
        if truncated
        else "日志未被截断。"
    )
    return (
        "你是 FDIDM 波形系统的诊断助手。只根据日志中可见的证据进行分析，"
        "不要臆造不存在的指标或事件。\n"
        "请用中文按以下结构输出：\n"
        "1. 运行概况\n"
        "2. 指标表现\n"
        "3. 异常和证据\n"
        "4. 可能原因\n"
        "5. 建议动作\n"
        "如果数据不足，请明确写出数据不足及需要补充的信息。\n"
        f"{truncation_note}\n\n"
        "日志内容：\n"
        f"{log_text}"
    )


def get_api_key(explicit: str = "", environ: Mapping[str, str] | None = None) -> str:
    """Return an explicitly configured key, falling back to the environment."""
    explicit_key = (explicit or "").strip()
    if explicit_key:
        return explicit_key
    env = os.environ if environ is None else environ
    return str(env.get("DEEPSEEK_API_KEY", "") or "").strip()


def _redact(message: str, api_key: str) -> str:
    return message.replace(api_key, "[REDACTED]") if api_key else message


def _endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


def analyze_log(
    log_text: str,
    config: DeepSeekConfig,
    opener: Callable[..., Any] | None = None,
) -> str:
    """Send a log to a DeepSeek-compatible endpoint and return model text."""
    if not str(log_text).strip():
        raise ValueError("日志为空，请先运行测试或选择日志文件")
    api_key = get_api_key(config.api_key)
    if not api_key:
        raise ValueError("未配置 DeepSeek API Key")
    if not config.base_url.strip():
        raise ValueError("未配置 API 地址")
    if not config.model.strip():
        raise ValueError("未配置模型名称")

    body = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": "你负责分析无线通信系统日志并给出基于证据的诊断。",
            },
            {"role": "user", "content": build_analysis_prompt(log_text, truncated=False)},
        ],
        "temperature": 0.2,
    }
    request = urllib_request.Request(
        _endpoint(config.base_url),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    open_fn = urllib_request.urlopen if opener is None else opener
    try:
        with open_fn(request, timeout=config.timeout_s) as response:
            raw = response.read()
    except urllib_error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            pass
        detail = _redact(detail, api_key)[:300]
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"DeepSeek HTTP {exc.code}{suffix}") from exc
    except (urllib_error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"DeepSeek 网络请求失败: {_redact(str(exc), api_key)}") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("DeepSeek 响应格式错误") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("DeepSeek 响应格式错误")
    return content.strip()
