"""Exercise the NTN OFDM page with real numerical frames and bounded workers."""

import json
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from waveform_sim.simulation.ofdm_ntn import run_study
from waveform_sim.ui import ofdm_tab
from waveform_sim.ui.main_window import MainWindow
from waveform_sim.ui.ofdm_tab import OfdmTab


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def tab(app):
    page = OfdmTab()
    yield page
    page.close()


@pytest.fixture
def platform(app):
    window = MainWindow()
    window.simulation_workspace.tabs.setCurrentIndex(1)
    window.open_workspace("simulation")
    yield window
    window.close()


def wait_until(predicate, timeout_ms=3000):
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate():
        assert time.monotonic() < deadline, "NTN worker did not reach the expected state"
        QTest.qWait(10)


def select_data(combo, value):
    index = combo.findData(value)
    assert index >= 0
    combo.setCurrentIndex(index)


def configure_awgn(page):
    select_data(page.preset_combo, "awgn")
    page.velocity_spin.setValue(0.0)


def test_default_ntn_configuration_and_profile_fft_updates_reach_summary(tab):
    config = tab._collect_config()
    assert config.profile == "NTN-TDLC5-200"
    assert (config.fft_len, config.cp_len, config.subcarrier_spacing_hz) == (128, 16, 30_000.0)
    assert config.carrier_frequency_hz == 2e9
    assert config.compensation_ratio == pytest.approx(0.999)
    assert config.estimator == "pilot_ls" and config.seed == 42
    assert "NTN-TDLC5-200" in tab.config_value_labels["波形"].text()

    select_data(tab.profile_combo, "NTN-TDLC5-1200")
    assert tab.doppler_spin.value() == 1200.0
    assert "NTN-TDLC5-1200" in tab.config_value_labels["波形"].text()
    tab.cp_spin.setValue(100)
    select_data(tab.fft_combo, 64)
    assert tab.cp_spin.maximum() == 63 and tab.cp_spin.value() == 63
    assert tab.config_value_labels["FFT/CP"].text() == "64 / 63"
    summary = json.loads(tab.status_text_label.toolTip())
    assert summary["sample_rate_hz"] == 64 * 30_000.0
    select_data(tab.spacing_combo, 60_000.0)
    summary = json.loads(tab.status_text_label.toolTip())
    assert summary["sample_rate_hz"] == 64 * 60_000.0
    select_data(tab.mod_combo, "64QAM")
    assert tab.config_value_labels["调制"].text() == "64QAM"

    select_data(tab.profile_combo, "NTN-TDLA100-200")
    tab.delay_scale_spin.setValue(1000.0)
    assert "CP 不足" in tab.status_text_label.text()
    select_data(tab.profile_combo, "AWGN")
    assert tab.doppler_spin.value() == 0.0
    np.testing.assert_array_equal(tab.channel_scatter.xData, [0.0])


@pytest.mark.parametrize("leave", ["stop", "navigation"])
def test_real_live_frames_stop_cleanly_and_navigation_keeps_results(platform, leave):
    page = platform.simulation_workspace.pages[1]
    configure_awgn(page)
    page.snr_spin.setValue(0.0)
    page.mod_combo.setCurrentText("16QAM")
    page._on_start_clicked()
    runner = page.tb
    assert runner is not None
    wait_until(lambda: runner.snapshot().get("frames", 0) >= 2)
    frame = runner.snapshot()
    assert frame["cumulative_bits"] == frame["frames"] * frame["total_bits"]
    assert frame["bit_errors"] == np.count_nonzero(frame["tx_bits"] != frame["rx_bits"])
    page._refresh_plots(force=True)
    spectrum = page.spectrum_curve.yData.copy()
    ber = page.ber_curve.yData.copy()
    if leave == "stop":
        page._on_stop_clicked()
        assert page.tb is None and not runner._thread.is_alive()
        np.testing.assert_array_equal(page.ber_curve.yData, ber)
    platform.show_home()
    assert page.tb is None and not runner._thread.is_alive()
    assert not page.update_timer.isActive()
    np.testing.assert_array_equal(page.spectrum_curve.yData, spectrum)
    np.testing.assert_array_equal(page.ber_curve.yData, ber)
    assert page.snr_spin.value() == 0.0 and page.mod_combo.currentText() == "16QAM"
    platform.open_workspace("simulation")
    assert platform.simulation_workspace.pages[1] is page
    QTest.qWait(200)
    assert page.tb is None and not page.update_timer.isActive()
    np.testing.assert_array_equal(page.ber_curve.yData, ber)


def test_two_point_study_shows_zero_error_bounds_and_exports_raw_snapshot(tab, tmp_path, monkeypatch):
    configure_awgn(tab)
    tab.snr_spin.setValue(50.0)
    tab.scan_values_edit.setText("40, 50")
    tab.scan_frames_spin.setValue(2)
    tab._start_auto_ber_snr_sweep()
    wait_until(lambda: not tab._auto_ber_snr_active)
    assert tab.result_table.rowCount() == 2
    assert len(tab._study_results) == 2 and tab.export_button.isEnabled()
    for row, point in enumerate(tab._study_results):
        assert set(point["strategies"]) == {"pilot_ls", "ideal", "uncompensated"}
        for key, stats in point["strategies"].items():
            assert stats["frames"] == 2
            assert stats["ber"] == stats["bit_errors"] == 0
            assert stats["total_bits"] > 0 and stats["ci_high"] > 0
            assert tab._study_zero_curves[key].yData[row] == pytest.approx(stats["ci_high"])
        assert tab.result_table.item(row, 1).text() == "0"
        stats = point["strategies"]["pilot_ls"]
        assert tab.result_table.item(row, 4).text() == f'0/{stats["total_bits"]}'
    assert "统计上界" in tab.study_note.text() or "统计上界" in tab.study_note.toolTip() or "95%" in tab.study_note.text()

    # Editing controls after completion must not rewrite the saved experiment.
    tab.snr_spin.setValue(5.0)
    tab.seed_spin.setValue(987)
    select_data(tab.profile_combo, "NTN-TDLA100-200")
    saved = tmp_path / "ofdm_study.json"
    monkeypatch.setattr(ofdm_tab.QFileDialog, "getSaveFileName", lambda *_args: (str(saved), "JSON"))
    tab._export_study()
    report = json.loads(saved.read_text(encoding="utf-8"))
    assert report["config"]["profile"] == "AWGN"
    assert report["config"]["ebn0_db"] == 50.0 and report["config"]["seed"] == 42
    assert report["axis"] == "ebn0" and report["values"] == [40.0, 50.0]
    assert len(report["results"]) == 2
    for point in report["results"]:
        for stats in point["strategies"].values():
            assert stats["ber"] == 0.0
            assert stats["bit_errors"] == 0 and stats["total_bits"] > 0


def test_leaving_study_cancels_worker_and_rejects_old_token_on_return(platform, monkeypatch):
    page = platform.simulation_workspace.pages[1]
    configure_awgn(page)
    page.scan_values_edit.setText("0, 5")
    page.scan_frames_spin.setValue(2)
    point = run_study(page._collect_config(), "ebn0", [0.0], frames=2)[0]
    started, ended = threading.Event(), threading.Event()
    calls = []

    def cancellable_study(config, axis, values, frames, stop_event=None, on_point=None):
        calls.append(True)
        on_point(point)
        started.set()
        assert stop_event.wait(2.0), "Navigation did not request cancellation"
        on_point(point)  # A late callback must not enqueue another result.
        ended.set()
        return [point]

    monkeypatch.setattr(ofdm_tab, "run_study", cancellable_study)
    page._start_auto_ber_snr_sweep()
    old_token = page._study_token
    worker = page._ber_snr_sweep_thread
    wait_until(lambda: started.is_set() and page.result_table.rowCount() == 1)
    platform.show_home()
    assert page._study_stop.is_set() and ended.is_set() and not worker.is_alive()
    assert not page._auto_ber_snr_active
    assert len(page._study_results) == page.result_table.rowCount() == 1
    platform.open_workspace("simulation")
    progress = page.study_progress.text()
    page.study_point.emit(old_token, point)
    page.study_finished.emit(old_token, "late completion")
    QTest.qWait(200)
    assert len(page._study_results) == page.result_table.rowCount() == 1
    assert page.study_progress.text() == progress
    assert calls == [True] and page.tb is None and not page.update_timer.isActive()


@pytest.mark.parametrize("axis,text", [
    ("ebn0", ""), ("ebn0", "0, 0"), ("ebn0", "nan, 4"),
    ("ebn0", "bad, 4"), ("ebn0", "1000"),
    ("spacing", "0, 15"), ("delay_scale", "-1, 2"),
])
def test_invalid_scan_values_show_feedback_without_starting_worker(tab, monkeypatch, axis, text):
    notices, calls = [], []
    monkeypatch.setattr(ofdm_tab.QMessageBox, "information", lambda *_args: notices.append(True))
    monkeypatch.setattr(ofdm_tab, "run_study", lambda *_args, **_kwargs: calls.append(True) or [])
    select_data(tab.axis_combo, axis)
    tab.scan_values_edit.setText(text)
    tab._start_auto_ber_snr_sweep()
    assert notices
    assert calls == [] and tab._ber_snr_sweep_thread is None
    assert not tab._auto_ber_snr_active
