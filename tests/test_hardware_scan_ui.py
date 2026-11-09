"""Regression tests for per-point OFDM hardware scan lifecycle."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from PyQt5.QtWidgets import QApplication

from waveform_sim.hardware.hardware_scan import NTNScanPlan
from waveform_sim.ui.hardware_test_tab import HardwareTestTab


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(app):
    tab = HardwareTestTab()
    yield tab
    tab.close()


class _ScanBackend:
    def __init__(self):
        self.frame_id = 17
        self.frame_limit = None
        self.started = False
        self.stopped = False

    def get_status(self):
        return {"frame_id": self.frame_id}

    def set_frame_limit(self, frame_limit):
        self.frame_limit = frame_limit

    def start(self):
        self.frame_id = 0
        self.started = True
        self.stopped = False

    def stop(self):
        self.stopped = True
        self.started = False

    def wait(self):
        return None

    def get_decode_stats(self):
        return {
            "frame_id": self.frame_id,
            "bit_errors": 2,
            "bits_total": 300,
            "cumulative_ber": 2 / 300,
            "decode_ok": False,
        }


class _RunRecorder:
    run_dir = "run-artifact"

    def __init__(self):
        self.statuses = []
        self.summary = None
        self.error = None

    def record_status(self, status):
        self.statuses.append(dict(status))
        return True

    def finish(self, summary):
        self.summary = dict(summary)

    def fail(self, error):
        self.error = error


def test_scan_start_uses_reset_per_run_counter_and_applies_exact_frame_limit(page, monkeypatch):
    backend = _ScanBackend()
    page.backend = backend
    page.backend_waveform = "OFDM"
    page._scan_active = True
    page._scan_plan = NTNScanPlan(
        parameter="ntn_doppler_spread_hz",
        start=200,
        stop=400,
        points=2,
        frames_per_point=3,
    )
    monkeypatch.setattr(page, "_begin_hardware_run", lambda **_kwargs: True)
    monkeypatch.setattr(page, "_reset_runtime_plots", lambda: None)

    assert page._start_recorded_backend(ntn_kwargs={"ntn_doppler_spread_hz": 200})

    assert page._scan_start_frame_id == 0
    assert backend.frame_limit == 3
    assert backend.frame_id == 0


def test_scan_completion_records_post_stop_final_metrics(page):
    backend = _ScanBackend()
    backend.frame_id = 3
    page.backend = backend
    page.backend_waveform = "OFDM"
    page.test_running = True
    page._scan_active = True
    page._scan_values = [200.0]
    page._scan_index = 0
    page._scan_plan = NTNScanPlan(
        parameter="ntn_doppler_spread_hz",
        start=200,
        stop=200,
        points=2,
        frames_per_point=3,
    )
    recorder = _RunRecorder()
    page._hardware_recorder = recorder

    page._complete_scan_point(
        {"frame_id": 2, "bit_errors": 0, "bits_total": 200, "cumulative_ber": 0.0},
        {"frame_id": 2, "bit_errors": 0, "bits_total": 200, "cumulative_ber": 0.0},
    )

    assert backend.stopped is True
    assert recorder.statuses[0]["frame_id"] == 3
    assert recorder.summary["frame_id"] == 3
    assert recorder.summary["bit_errors"] == 2
    assert recorder.summary["bits_total"] == 300
    assert page._scan_active is False


def test_manual_stop_records_final_backend_snapshot(page):
    backend = _ScanBackend()
    backend.frame_id = 5
    page.backend = backend
    page.backend_waveform = "OFDM"
    page.test_running = True
    recorder = _RunRecorder()
    page._hardware_recorder = recorder

    page._on_stop_test_clicked()

    assert backend.stopped is True
    assert recorder.statuses[0]["frame_id"] == 5
    assert recorder.summary["bit_errors"] == 2
    assert recorder.summary["bits_total"] == 300


def test_leaving_hardware_page_records_final_backend_snapshot(page):
    backend = _ScanBackend()
    backend.frame_id = 6
    page.backend = backend
    page.backend_waveform = "OFDM"
    page.test_running = True
    recorder = _RunRecorder()
    page._hardware_recorder = recorder

    page.suspend_tasks()

    assert page.backend is None
    assert backend.stopped is True
    assert recorder.statuses[0]["frame_id"] == 6
    assert recorder.summary["bit_errors"] == 2
    assert recorder.summary["bits_total"] == 300
