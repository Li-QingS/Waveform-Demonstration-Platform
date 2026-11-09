"""Backend replacement must wait for the previous radio flowgraph to stop."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from PyQt5.QtWidgets import QApplication

from waveform_sim.ui.hardware_test_tab import HardwareTestTab


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(app):
    return HardwareTestTab()


@pytest.mark.parametrize("failure", ["stop", "wait"])
def test_backend_replacement_aborts_if_previous_backend_cannot_stop_or_exit(
    page, monkeypatch, failure
):
    calls = []

    class PreviousBackend:
        def stop(self):
            calls.append("old.stop")
            if failure == "stop":
                raise RuntimeError("radio stop failed")

        def wait(self):
            calls.append("old.wait")
            if failure == "wait":
                raise RuntimeError("radio wait failed")

    class ReplacementBackend:
        def __init__(self, **_kwargs):
            calls.append("new.create")

    previous = PreviousBackend()
    page.backend = previous
    page.backend_waveform = "OFDM"
    page._backend_identity = page._current_identity()
    monkeypatch.setattr(page, "_load_backend_class", lambda _waveform: ReplacementBackend)
    monkeypatch.setattr(page, "_constructor_kwargs", lambda _waveform: {})
    monkeypatch.setattr(page, "_push_constellation_mode", lambda: None)
    monkeypatch.setattr(page, "_backend_summary", lambda: "fake backend")
    monkeypatch.setattr(page, "_refresh_plots_once", lambda: None)

    assert page._create_backend() is False

    assert page.backend is previous
    assert calls == (["old.stop"] if failure == "stop" else ["old.stop", "old.wait"])
    assert "取消创建新后端" in page.decode_status_label.text()


def test_backend_replacement_waits_and_releases_old_backend_before_creation(
    page, monkeypatch
):
    calls = []

    class PreviousBackend:
        def stop(self):
            calls.append("old.stop")

        def wait(self):
            calls.append("old.wait")

    class ReplacementBackend:
        def __init__(self, **_kwargs):
            calls.append(("new.create", page.backend))

    page.backend = PreviousBackend()
    page.backend_waveform = "OFDM"
    page._backend_identity = page._current_identity()
    monkeypatch.setattr(page, "_load_backend_class", lambda _waveform: ReplacementBackend)
    monkeypatch.setattr(page, "_constructor_kwargs", lambda _waveform: {})
    monkeypatch.setattr(page, "_push_constellation_mode", lambda: None)
    monkeypatch.setattr(page, "_backend_summary", lambda: "fake backend")
    monkeypatch.setattr(page, "_refresh_plots_once", lambda: None)

    assert page._create_backend() is True

    assert calls == ["old.stop", "old.wait", ("new.create", None)]
    assert isinstance(page.backend, ReplacementBackend)
    assert page.backend_waveform == "OFDM"
