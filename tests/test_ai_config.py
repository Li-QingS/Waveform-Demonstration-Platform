"""Platform-owned AI settings, isolated from any real operator credential."""
import json
import os
import errno
from pathlib import Path

import pytest

from waveform_sim.service.ai_log_analyzer import DeepSeekConfig


def test_unconfigured_backend_has_safe_error_and_does_not_create_files(tmp_path):
    from waveform_sim.service.ai_config import AIConfigurationError, load_ai_config

    folder = tmp_path / "not-created"
    with pytest.raises(AIConfigurationError, match="AI"):
        load_ai_config(config_dir=folder, environ={})
    assert not folder.exists()


def test_environment_settings_support_headless_admin_configuration(tmp_path):
    from waveform_sim.service.ai_config import load_ai_config

    config = load_ai_config(config_dir=tmp_path, environ={
        "DEEPSEEK_API_KEY": " test-only-admin-key ",
        "DEEPSEEK_MODEL": " operator-model ",
        "DEEPSEEK_BASE_URL": " https://example.test/v1 ",
    })
    assert config.api_key == "test-only-admin-key"
    assert config.model == "operator-model"
    assert config.base_url == "https://example.test/v1"
    assert "test-only-admin-key" not in repr(config)
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def protected_store(monkeypatch):
    from waveform_sim.service import ai_config

    # Replace the external OS encryption boundary for portable store tests.
    stored = {}

    def protect(raw):
        stored["payload"] = raw
        return b"opaque-os-protected-test-blob"

    def unprotect(blob):
        assert blob == b"opaque-os-protected-test-blob"
        return stored["payload"]

    monkeypatch.setattr(ai_config, "_protect", protect)
    monkeypatch.setattr(ai_config, "_unprotect", unprotect)
    return ai_config, stored


def test_saved_config_is_encrypted_and_recovers_after_reload(tmp_path, protected_store):
    ai_config, _ = protected_store
    config = DeepSeekConfig("https://example.test", "private-model", "test-only-admin-key")
    path = ai_config.save_ai_config(config, config_dir=tmp_path)
    assert path.parent == tmp_path
    assert path.read_bytes() == b"opaque-os-protected-test-blob"
    assert b"test-only-admin-key" not in path.read_bytes()
    assert not list(tmp_path.glob("*.tmp"))
    reloaded = ai_config.load_ai_config(config_dir=tmp_path, environ={})
    assert reloaded == config


def test_environment_overrides_saved_configuration(tmp_path, protected_store):
    ai_config, _ = protected_store
    ai_config.save_ai_config(DeepSeekConfig("https://saved.test", "saved", "stored-test-key"), config_dir=tmp_path)
    reloaded = ai_config.load_ai_config(config_dir=tmp_path, environ={
        "DEEPSEEK_API_KEY": "override-test-key",
        "DEEPSEEK_MODEL": "override",
        "DEEPSEEK_BASE_URL": "https://override.test",
    })
    assert reloaded.api_key == "override-test-key"
    assert reloaded.model == "override"
    assert reloaded.base_url == "https://override.test"


def test_decryption_failure_does_not_expose_underlying_data(tmp_path, monkeypatch):
    from waveform_sim.service import ai_config

    (tmp_path / "ai-config.dpapi").write_bytes(b"broken")

    def unprotect(_blob):
        raise RuntimeError("OS error with test-only-secret-in-details")

    monkeypatch.setattr(ai_config, "_unprotect", unprotect)
    with pytest.raises(ai_config.AIConfigurationError) as error:
        ai_config.load_ai_config(config_dir=tmp_path, environ={})
    assert "test-only-secret" not in str(error.value)


@pytest.mark.parametrize("payload", [[], {"api_key": ""}, {"api_key": 123}, {"api_key": "dummy", "model": []}])
def test_invalid_stored_payload_fails_safely(tmp_path, monkeypatch, payload):
    from waveform_sim.service import ai_config

    (tmp_path / "ai-config.dpapi").write_bytes(b"opaque")
    monkeypatch.setattr(ai_config, "_unprotect", lambda _blob: json.dumps(payload).encode())
    with pytest.raises(ai_config.AIConfigurationError):
        ai_config.load_ai_config(config_dir=tmp_path, environ={})


def test_environment_can_replace_unreadable_local_settings(tmp_path, monkeypatch):
    from waveform_sim.service import ai_config

    (tmp_path / "ai-config.dpapi").write_bytes(b"broken")
    monkeypatch.setattr(ai_config, "_unprotect", lambda _blob: b"invalid JSON")
    config = ai_config.load_ai_config(config_dir=tmp_path, environ={"DEEPSEEK_API_KEY": "env-test-key"})
    assert config.api_key == "env-test-key"
    assert config.model


def test_save_rejects_invalid_config_before_writing(tmp_path, protected_store):
    ai_config, _ = protected_store
    with pytest.raises(ai_config.AIConfigurationError):
        ai_config.save_ai_config(DeepSeekConfig("not-a-url", "model", "test-key"), config_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_encrypted_save_supports_cross_device_rename_fallback(tmp_path, protected_store, monkeypatch):
    ai_config, _ = protected_store

    def cross_device_replace(self, target):
        raise OSError(errno.EXDEV, "cross-device rename")

    monkeypatch.setattr(Path, "replace", cross_device_replace)
    config = DeepSeekConfig("https://example.test", "model", "dummy-credential")
    path = ai_config.save_ai_config(config, config_dir=tmp_path)
    assert b"dummy-credential" not in path.read_bytes()
    assert not list(tmp_path.glob("*.tmp"))
    assert ai_config.load_ai_config(config_dir=tmp_path, environ={}) == config


@pytest.mark.skipif(os.name != "nt", reason="Windows user-bound DPAPI integration")
def test_windows_dpapi_roundtrip_does_not_write_plaintext(tmp_path):
    from waveform_sim.service.ai_config import load_ai_config, save_ai_config

    config = DeepSeekConfig("https://api.deepseek.com", "deepseek-flash", "test-only-dpapi-credential")
    path = save_ai_config(config, config_dir=tmp_path)
    assert b"test-only-dpapi-credential" not in path.read_bytes()
    assert load_ai_config(config_dir=tmp_path, environ={}) == config
