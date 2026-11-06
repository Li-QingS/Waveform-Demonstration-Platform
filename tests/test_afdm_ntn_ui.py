"""Real AFDM frames, immutable studies and bounded ownership across Qt navigation."""

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

from waveform_sim.simulation.afdm_ntn import AfdmNtnConfig, run_study
from waveform_sim.ui import afdm_ntn_workers, afdm_tab
from waveform_sim.ui.main_window import MainWindow
from waveform_sim.ui.afdm_ntn_workers import AfdmNtnRunner
from waveform_sim.ui.afdm_tab import AfdmTab


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def tab(app):
    page = AfdmTab()
    yield page
    page.close()


@pytest.fixture
def platform(app):
    window = MainWindow()
    window.simulation_workspace.tabs.setCurrentIndex(3)
    window.open_workspace("simulation")
    yield window
    window.close()


def wait_until(predicate, timeout=4.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "AFDM worker did not reach the expected state"
        QTest.qWait(10)


def select_data(combo, value):
    index = combo.findData(value)
    assert index >= 0
    combo.setCurrentIndex(index)


def configure_awgn(page):
    select_data(page.preset_combo, "awgn")
    select_data(page.fft_combo, 32)
    page.blocks_spin.setValue(4)


def test_defaults_and_affine_controls_update_ntn_summary(tab):
    config = tab._collect_config()
    assert isinstance(config, AfdmNtnConfig)
    assert config.profile == "NTN-TDLC5-200"
    assert (config.fft_len, config.cp_len, config.symbols_per_frame) == (64, 16, 8)
    assert config.subcarrier_spacing_hz == 30_000.0
    assert config.estimator == "training_ls" and config.seed == 42
    assert config.prefix == "cpp"
    assert config.c1 == pytest.approx(3/(2*64), abs=1e-10)
    assert config.c2 == pytest.approx(1/(64*64*np.sqrt(2)), abs=1e-10)
    assert "AFDM" in tab.config_value_labels["波形"].text()
    select_data(tab.profile_combo, "NTN-TDLC5-1200")
    assert tab.doppler_spin.value() == 1200.0
    tab.cp_spin.setValue(100)
    select_data(tab.fft_combo, 32)
    assert tab.cp_spin.value() == tab.cp_spin.maximum() == 31
    tab.blocks_spin.setValue(6)
    assert tab._collect_config().symbols_per_frame == 6
    summary = json.loads(tab.status_text_label.toolTip())
    assert summary["sample_rate_hz"] == 32 * 30_000.0
    assert summary["payload_bits_per_frame"] == 32 * 6 * 2
    select_data(tab.profile_combo, "NTN-TDLA100-200")
    tab.delay_scale_spin.setValue(1000.0)
    assert "前缀不足" in tab.status_text_label.text()


def test_chirp_and_prefix_are_explicit_controls_without_silent_parameter_updates(tab):
    configure_awgn(tab)
    tab.c1_spin.setValue(0.01731)
    tab.c2_spin.setValue(0.001271)
    select_data(tab.prefix_combo, "cp")
    tab.doppler_spin.setValue(1200.0)
    cfg = tab._collect_config()
    assert (cfg.c1, cfg.c2, cfg.prefix) == (0.01731, 0.001271, "cp")
    # A channel edit must not silently change the experiment's chirp rates.
    tab.delay_scale_spin.setValue(5.0)
    assert tab.c1_spin.value() == 0.01731 and tab.c2_spin.value() == 0.001271
    tab.btn_suggest_chirp.click()
    suggested = tab._collect_config()
    assert suggested.c1 != cfg.c1
    assert suggested.c1 == tab.c1_spin.value() and suggested.c2 == tab.c2_spin.value()
    assert suggested.prefix == "cp"


def test_out_of_range_chirp_advice_is_rejected_without_silently_clamping_parameters(tab, monkeypatch):
    original = tab._collect_config()
    notices = []
    monkeypatch.setattr(afdm_tab, "suggested_chirp", lambda _cfg: {"c1":0.6,"c2":0.002})
    monkeypatch.setattr(afdm_tab.QMessageBox, "information", lambda *_args: notices.append(True))
    tab.btn_suggest_chirp.click()
    assert notices
    assert tab._collect_config().c1 == original.c1 and tab._collect_config().c2 == original.c2


def test_live_controls_lock_and_real_channel_image_stays_after_stop_and_navigation(platform):
    page = platform.simulation_workspace.pages[3]
    configure_awgn(page)
    page.snr_spin.setValue(0.0)
    page.mod_combo.setCurrentText("16QAM")
    page._on_start_clicked()
    runner = page.tb
    assert runner is not None
    wait_until(lambda: runner.snapshot().get("frames", 0) >= 2)
    for control in (page.blocks_spin, page.fft_combo, page.mod_combo, page.snr_spin,
                    page.cfo_spin, page.seed_spin, page.estimator_combo, page.scan_frames_spin,
                    page.c1_spin, page.c2_spin, page.prefix_combo, page.btn_suggest_chirp):
        assert not control.isEnabled()
    assert not page.btn_auto_ber_snr.isEnabled() and page.observe_combo.isEnabled()
    frame = runner.snapshot()
    assert frame["cumulative_bits"] == frame["frames"] * frame["total_bits"]
    assert frame["bit_errors"] == np.count_nonzero(frame["tx_bits"] != frame["rx_bits"])
    page._refresh_plots(force=True)
    assert page.channel_image.image is not None
    assert set(page.channel_image.image.shape) == {32}
    assert np.all(np.isfinite(page.channel_image.image))
    page._on_stop_clicked()
    assert page.tb is None and not runner._thread.is_alive()
    assert not page.update_timer.isActive()
    spectrum = page.spectrum_curve.yData.copy()
    image = page.channel_image.image.copy()
    points = page.constellation_scatter.getData()[0].copy()
    platform.show_home()
    np.testing.assert_array_equal(page.spectrum_curve.yData, spectrum)
    platform.open_workspace("simulation")
    QTest.qWait(150)
    assert platform.simulation_workspace.pages[3] is page
    assert page.tb is None and not page.update_timer.isActive()
    assert page.blocks_spin.value() == 4 and page.mod_combo.currentText() == "16QAM"
    np.testing.assert_array_equal(page.channel_image.image, image)
    np.testing.assert_array_equal(page.constellation_scatter.getData()[0], points)


def test_live_and_study_use_identical_seeded_counts_for_the_same_frame_budget(tab):
    configure_awgn(tab)
    tab.snr_spin.setValue(0.0)
    tab._on_start_clicked()
    runner = tab.tb
    wait_until(lambda: runner.snapshot().get("frames", 0) >= 1)
    runner.stop()
    runner.wait(timeout=1.5)
    snapshot = runner.snapshot()
    stats = run_study(runner.config, "ebn0", [runner.config.ebn0_db], frames=snapshot["frames"])[0]["strategies"]["training_ls"]
    assert snapshot["cumulative_errors"] == stats["bit_errors"]
    assert snapshot["cumulative_bits"] == stats["total_bits"]
    assert snapshot["cumulative_ber"] == stats["ber"]


def test_zero_error_markers_and_stopped_channel_coordinates_keep_the_original_experiment(tab):
    configure_awgn(tab)
    tab.snr_spin.setValue(100.0)
    tab._on_start_clicked()
    runner = tab.tb
    wait_until(lambda: runner.snapshot().get("frames", 0) >= 1)
    tab._on_stop_clicked()
    assert tab.tb is None
    assert tab.ber_curve.yData is None or not len(tab.ber_curve.yData)
    upper = tab.live_zero_curve.yData.copy()
    assert upper.size > 0 and np.all(upper > 0)
    image = tab.channel_image.image.copy()
    ranges = tab.channel_plot.getViewBox().viewRange()
    original = tab._live_config
    select_data(tab.fft_combo, 128)
    select_data(tab.spacing_combo, 60_000.0)
    tab.blocks_spin.setValue(12)
    tab.c1_spin.setValue(0.01731)
    select_data(tab.prefix_combo, "cp")
    tab.snr_spin.setValue(0.0)
    assert "上次实验" in tab.status_text_label.text()
    tab._refresh_plots(force=True)
    assert tab._live_config == original
    np.testing.assert_array_equal(tab.channel_image.image, image)
    np.testing.assert_array_equal(tab.live_zero_curve.yData, upper)
    np.testing.assert_allclose(tab.channel_plot.getViewBox().viewRange(), ranges)
    # Observation can still change using the saved frame, while the matrix
    # keeps its original dimensions and actual experiment configuration.
    select_data(tab.observe_combo, "pre")
    assert tab.channel_image.image.shape == image.shape
    np.testing.assert_allclose(tab.channel_plot.getViewBox().viewRange(), ranges)
    assert tab._last_live_snapshot["cumulative_ber"] == 0.0


def test_study_raw_zeros_bounds_and_export_use_frozen_config(tab, monkeypatch, tmp_path):
    configure_awgn(tab)
    tab.c1_spin.setValue(0.01731)
    tab.c2_spin.setValue(0.001271)
    select_data(tab.prefix_combo, "cpp")
    tab.snr_spin.setValue(50.0)
    tab.scan_values_edit.setText("40, 50")
    tab.scan_frames_spin.setValue(2)
    tab._start_auto_ber_snr_sweep()
    assert not tab.btn_start.isEnabled() and not tab.blocks_spin.isEnabled()
    wait_until(lambda: not tab._auto_ber_snr_active)
    assert tab.result_table.rowCount() == 2 and tab.export_button.isEnabled()
    for row, point in enumerate(tab._study_results):
        assert set(point["strategies"]) == {"training_ls", "ideal", "uncompensated"}
        for key, stats in point["strategies"].items():
            assert stats["frames"] == 2 and stats["total_bits"] == 512
            assert stats["ber"] == stats["bit_errors"] == 0
            assert stats["ci_high"] > 0
            assert tab._study_zero_curves[key].yData[row] == pytest.approx(stats["ci_high"])
        assert tab.result_table.item(row, 1).text() == "0"
        assert tab.result_table.item(row, 4).text() == "0/512"
    tab.snr_spin.setValue(5.0)
    tab.seed_spin.setValue(987)
    tab.blocks_spin.setValue(6)
    tab.c1_spin.setValue(0.031)
    tab.c2_spin.setValue(0.004)
    select_data(tab.prefix_combo, "cp")
    select_data(tab.profile_combo, "NTN-TDLA100-200")
    destination = tmp_path / "afdm_study.json"
    monkeypatch.setattr(afdm_tab.QFileDialog, "getSaveFileName", lambda *_args: (str(destination), "JSON"))
    tab._export_study()
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["config"]["profile"] == "AWGN"
    assert report["config"]["ebn0_db"] == 50.0 and report["config"]["seed"] == 42
    assert report["config"]["symbols_per_frame"] == 4
    assert report["config"]["c1"] == 0.01731 and report["config"]["c2"] == 0.001271
    assert report["config"]["prefix"] == "cpp"
    assert report["axis"] == "ebn0" and report["values"] == [40.0, 50.0]
    assert len(report["results"]) == 2


@pytest.mark.parametrize("axis,values", [
    ("doppler", "0, 1200"), ("blocks", "2, 4"),
    ("c1", "0, 0.01731"), ("c2", "0, 0.001271"),
])
def test_afdm_specific_studies_count_complete_frames(tab, axis, values):
    configure_awgn(tab)
    select_data(tab.profile_combo, "NTN-TDLA100-200")
    select_data(tab.axis_combo, axis)
    tab.scan_values_edit.setText(values)
    tab.scan_frames_spin.setValue(1)
    tab._start_auto_ber_snr_sweep()
    wait_until(lambda: not tab._auto_ber_snr_active)
    assert len(tab._study_results) == 2
    assert all(stats["frames"] == 1 for point in tab._study_results for stats in point["strategies"].values())
    if axis == "blocks":
        assert [p["strategies"]["training_ls"]["total_bits"] for p in tab._study_results] == [128, 256]


def test_navigation_cancels_study_and_rejects_late_queued_results(platform, monkeypatch):
    page = platform.simulation_workspace.pages[3]
    configure_awgn(page)
    page.scan_values_edit.setText("0, 5")
    page.scan_frames_spin.setValue(2)
    point = run_study(page._collect_config(), "ebn0", [0.0], frames=2)[0]
    entered, ended = threading.Event(), threading.Event()
    calls = []

    def cancellable_study(config, axis, values, frames, stop_event=None, on_point=None):
        calls.append(True)
        on_point(point)
        entered.set()
        assert stop_event.wait(3.0), "Navigation did not request cancellation"
        on_point(point)
        ended.set()
        return [point]

    monkeypatch.setattr(afdm_tab, "run_study", cancellable_study)
    page._start_auto_ber_snr_sweep()
    token = page._study_token
    worker = page._ber_snr_sweep_thread
    wait_until(lambda: entered.is_set() and page.result_table.rowCount() == 1)
    platform.show_home()
    assert ended.is_set() and not worker.is_alive()
    assert not page._auto_ber_snr_active
    platform.open_workspace("simulation")
    progress = page.study_progress.text()
    page.study_point.emit(token, point)
    page.study_finished.emit(token, "late completion")
    QTest.qWait(150)
    assert page.study_progress.text() == progress
    assert page.result_table.rowCount() == len(page._study_results) == 1
    assert calls == [True] and page.tb is None and not page.update_timer.isActive()


def test_runner_timeout_retains_thread_and_rejects_overlapping_restart(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def blocked_frame(config, frame_index=0, stop_event=None):
        entered.set()
        release.wait(5.0)
        raise InterruptedError("cancelled before publishing")

    monkeypatch.setattr(afdm_ntn_workers, "simulate_frame", blocked_frame)
    runner = AfdmNtnRunner(AfdmNtnConfig())
    runner.start()
    try:
        assert entered.wait(2.0)
        owned = runner._thread
        runner.stop()
        with pytest.raises((RuntimeError, TimeoutError)):
            runner.wait(timeout=0.01)
        assert runner._thread is owned and owned.is_alive()
        with pytest.raises(RuntimeError):
            runner.start()
        assert runner._thread is owned
    finally:
        release.set()
        runner.wait(timeout=2.0)


def test_page_stop_timeout_preserves_runner_and_blocks_navigation(tab):
    class SlowRunner:
        config = AfdmNtnConfig()
        stuck = True

        def stop(self):
            pass

        def wait(self, timeout=None):
            if self.stuck:
                raise TimeoutError("still computing")

        def snapshot(self):
            return {}

    runner = SlowRunner()
    tab.tb = runner
    tab.update_timer.start()
    tab._set_ui_enabled(False)
    try:
        tab._on_stop_clicked()
        assert tab.tb is runner and tab.update_timer.isActive()
        assert not tab.btn_start.isEnabled() and not tab.blocks_spin.isEnabled()
        with pytest.raises((RuntimeError, TimeoutError)):
            tab.suspend_tasks()
        assert tab.tb is runner
    finally:
        runner.stuck = False
        tab.suspend_tasks()
    assert tab.tb is None and not tab.update_timer.isActive()


def test_scan_timeout_retains_worker_and_blocks_new_tasks(tab):
    class SlowStudy:
        alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            pass

    worker = SlowStudy()
    tab._ber_snr_sweep_thread = worker
    tab._auto_ber_snr_active = True
    tab._set_ui_enabled(True)
    try:
        with pytest.raises((TimeoutError, RuntimeError)):
            tab.suspend_tasks()
        assert tab._ber_snr_sweep_thread is worker
        assert tab._auto_ber_snr_active and tab._study_stop.is_set()
        assert not tab.btn_start.isEnabled() and not tab.btn_auto_ber_snr.isEnabled()
        tab._on_start_clicked()
        assert tab.tb is None and tab._ber_snr_sweep_thread is worker
    finally:
        worker.alive = False
        tab.suspend_tasks()
    assert tab._ber_snr_sweep_thread is None and not tab._auto_ber_snr_active


def test_scan_preflights_uncompensated_aliasing_before_erasing_a_previous_study(tab, monkeypatch):
    # A compensated live experiment is representable at this sample rate,
    # but the extra study baseline disables compensation and would alias.
    configure_awgn(tab)
    select_data(tab.spacing_combo, 15_000.0)
    tab.fc_spin.setValue(40.0)
    tab.velocity_spin.setValue(40000.0)
    tab.radial_spin.setValue(1.0)
    tab.compensation_spin.setValue(100.0)
    assert tab._collect_config().compensation_ratio == 1.0
    notices, calls = [], []
    retained = [{"previous_completed_experiment": True}]
    tab._study_results = retained
    tab.result_table.setRowCount(1)
    monkeypatch.setattr(afdm_tab.QMessageBox, "information", lambda *_args: notices.append(True))
    monkeypatch.setattr(afdm_tab, "run_study", lambda *_args, **_kwargs: calls.append(True) or [])
    tab.scan_values_edit.setText("0, 5")
    tab._start_auto_ber_snr_sweep()
    assert notices and calls == []
    assert tab._ber_snr_sweep_thread is None and not tab._auto_ber_snr_active
    assert tab._study_results is retained and tab.result_table.rowCount() == 1


@pytest.mark.parametrize("axis,text", [
    ("ebn0", ""), ("ebn0", "0, 0"), ("ebn0", "nan, 4"), ("ebn0", "bad, 4"),
    ("ebn0", "1000"), ("spacing", "0, 15"), ("delay_scale", "-1, 2"),
    ("blocks", "4, 3.5"), ("doppler", "-1, 200"),
])
def test_invalid_studies_show_feedback_without_starting_or_erasing_results(tab, monkeypatch, axis, text):
    notices, calls = [], []
    monkeypatch.setattr(afdm_tab.QMessageBox, "information", lambda *_args: notices.append(True))
    monkeypatch.setattr(afdm_tab, "run_study", lambda *_args, **_kwargs: calls.append(True) or [])
    retained = [{"previous_completed_experiment": True}]
    tab._study_results = retained
    tab.result_table.setRowCount(1)
    select_data(tab.axis_combo, axis)
    tab.scan_values_edit.setText(text)
    tab._start_auto_ber_snr_sweep()
    assert notices and calls == []
    assert tab._ber_snr_sweep_thread is None and not tab._auto_ber_snr_active
    assert tab._study_results is retained and tab.result_table.rowCount() == 1
