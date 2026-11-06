"""Platform-owned AI configuration, outside the user-facing assistant UI.

Windows deployments store settings protected by DPAPI for the current user.
Environment overrides also support administrator-managed services and CI.
"""
from __future__ import annotations

import ctypes
import errno
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from .ai_log_analyzer import DeepSeekConfig

DEFAULT_AI_BASE_URL = "https://api.deepseek.com"
DEFAULT_AI_MODEL = "deepseek-flash"
_CONFIG_FILENAME = "ai-config.dpapi"
_CONFIG_ERROR = "AI 助手服务尚未就绪，请联系平台管理员。"


class AIConfigurationError(RuntimeError):
    """A configuration error that never includes credential material."""


def config_directory(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    root = env.get("LOCALAPPDATA") or env.get("XDG_CONFIG_HOME")
    base = Path(root) if root else Path.home() / ".config"
    return base / "WaveformDemonstrationPlatform"


def _crypt(data: bytes, *, decrypt: bool) -> bytes:
    if os.name != "nt":
        raise AIConfigurationError(_CONFIG_ERROR)
    from ctypes import wintypes

    class DataBlob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    buffer = ctypes.create_string_buffer(data)
    incoming = DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    outgoing = DataBlob()
    if decrypt:
        operation = crypt32.CryptUnprotectData
        description = None
    else:
        operation = crypt32.CryptProtectData
        description = "Waveform AI settings"
    operation.argtypes = [
        ctypes.POINTER(DataBlob), ctypes.c_void_p if decrypt else wintypes.LPCWSTR,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(DataBlob),
    ]
    operation.restype = wintypes.BOOL
    # UI forbidden; omit LOCAL_MACHINE so only the current Windows user can decrypt.
    if not operation(ctypes.byref(incoming), description, None, None, None, 1,
                     ctypes.byref(outgoing)):
        raise AIConfigurationError(_CONFIG_ERROR)
    try:
        return ctypes.string_at(outgoing.data, outgoing.size)
    finally:
        kernel32.LocalFree(ctypes.cast(outgoing.data, ctypes.c_void_p))


def _protect(data: bytes) -> bytes:
    return _crypt(data, decrypt=False)


def _unprotect(data: bytes) -> bytes:
    return _crypt(data, decrypt=True)


def _validated(config: DeepSeekConfig) -> DeepSeekConfig:
    try:
        values = (config.api_key, config.base_url, config.model)
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError
        base_url = config.base_url.strip()
        parsed = urlparse(base_url)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise ValueError
        timeout = float(config.timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError
        key = config.api_key.strip()
        if "\n" in key or "\r" in key:
            raise ValueError
        return DeepSeekConfig(base_url, config.model.strip(), key, timeout)
    except (AttributeError, ValueError, TypeError):
        raise AIConfigurationError(_CONFIG_ERROR) from None


def load_ai_config(*, config_dir: Path | None = None,
                   environ: Mapping[str, str] | None = None) -> DeepSeekConfig:
    """Read administrator-owned settings without creating any local files."""
    env = os.environ if environ is None else environ
    path = (Path(config_dir) if config_dir is not None else config_directory(env)) / _CONFIG_FILENAME
    saved = {}
    override_key = str(env.get("DEEPSEEK_API_KEY", "")).strip()
    try:
        if path.exists():
            saved = json.loads(_unprotect(path.read_bytes()).decode("utf-8"))
            if not isinstance(saved, dict):
                raise ValueError
    except Exception:
        if not override_key:
            raise AIConfigurationError(_CONFIG_ERROR) from None
        saved = {}

    config = DeepSeekConfig(
        env.get("DEEPSEEK_BASE_URL") or saved.get("base_url", DEFAULT_AI_BASE_URL),
        env.get("DEEPSEEK_MODEL") or saved.get("model", DEFAULT_AI_MODEL),
        override_key or saved.get("api_key", ""),
        saved.get("timeout_s", 60.0),
    )
    return _validated(config)


def save_ai_config(config: DeepSeekConfig, *, config_dir: Path | None = None) -> Path:
    """Save DPAPI-protected backend settings; never write plaintext."""
    config = _validated(config)
    try:
        raw = json.dumps({
            "base_url": config.base_url, "model": config.model,
            "api_key": config.api_key, "timeout_s": config.timeout_s,
        }, ensure_ascii=False).encode("utf-8")
        protected = _protect(raw)
        folder = Path(config_dir) if config_dir is not None else config_directory()
        folder.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".ai-config-", suffix=".tmp", dir=folder)
        temp = Path(name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(protected)
            path = folder / _CONFIG_FILENAME
            try:
                temp.replace(path)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                # Some mounted profiles disallow rename even within a folder.
                # The fallback still writes only the already-encrypted bytes.
                path.write_bytes(protected)
        finally:
            temp.unlink(missing_ok=True)
        return path
    except Exception:
        raise AIConfigurationError(_CONFIG_ERROR) from None
