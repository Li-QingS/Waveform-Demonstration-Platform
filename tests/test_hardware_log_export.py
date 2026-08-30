import importlib


def test_fdidm_default_log_export_uses_project_log_directory(tmp_path, monkeypatch):
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    expected_dir = tmp_path / "log"
    monkeypatch.setattr(mod, "project_log_directory", lambda: expected_dir)
    backend = object.__new__(mod._LegacyFDIDMHardwareTest)
    backend._debug_log = [
        {"seq": 1, "t": 0.25, "level": "INFO", "msg": "test entry"},
    ]

    default_saved = backend.export_debug_log()
    root_saved = backend.export_debug_log(tmp_path / "fdidm_debug_root.log")

    assert mod.Path(default_saved).parent == expected_dir
    assert mod.Path(root_saved).parent == expected_dir
    assert "test entry" in mod.Path(default_saved).read_text(encoding="utf-8")
