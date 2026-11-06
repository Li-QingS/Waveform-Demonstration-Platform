"""Paired worker ownership, truthful plots and comparison-page navigation."""

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

from waveform_sim.ui import compare_workers, waveform_compare_tab
from waveform_sim.ui.compare_workers import _PairedWaveformRunner, _ScanWorker
from waveform_sim.ui.main_window import MainWindow
from waveform_sim.ui.waveform_compare_tab import WaveformCompareTab


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def tab(app):
    page = WaveformCompareTab()
    yield page
    page.close()


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "Paired comparison worker did not reach the expected state"
        QTest.qWait(10)


def scan_parameters(**overrides):
    params = dict(snr_db=12.0, mod_order="QPSK", frames=2, seed=42,
                  receiver_mode="full_mmse", cfo_axis=np.array([0.0]),
                  doppler_axis=np.array([0.0]), fixed_cfo=0.0, fixed_doppler=0.0)
    params.update(overrides)
    return params


def scan_store():
    return dict(cfo={}, doppler={}, done=False, status="")


class ZeroSimulator:
    """A tiny complete frame set used only to isolate UI/lifecycle behavior."""

    def __init__(self, config=None):
        self.config = config

    def simulate_frame_set(self, waveforms, frame_index=0, **_kwargs):
        return {
            name: dict(bit_errors=0, total_bits=100, frame_error=0, evm_percent=0.5,
                       rx_symbols=np.array([1 + 1j, -1 - 1j]))
            for name in dict.fromkeys(waveforms)
        }


def set_single_point_scan(page):
    for control in (page.scan_cfo_min_spin, page.scan_cfo_max_spin,
                    page.scan_doppler_min_spin, page.scan_doppler_max_spin,
                    page.scan_fixed_cfo_spin, page.scan_fixed_doppler_spin):
        control.setValue(0.0)
    page.scan_frames_spin.setValue(2)


def test_real_live_and_scan_workers_count_the_same_seeded_physical_frames():
    runner = _PairedWaveformRunner(frame_interval_s=10.0)
    params = dict(waveforms=("OFDM", "OTFS"), snr_db=12.0, cfo_hz=0.0,
                  doppler_hz=0.0, mod_order="QPSK", seed=42, receiver_mode="full_mmse")
    runner.start(**params)
    try:
        wait_until(lambda: runner.snapshot()["left"]["frames"] >= 1)
    finally:
        runner.stop()
        runner.wait()
    live = runner.snapshot()
    assert live["left"]["frames"] == live["right"]["frames"] == 1
    store = scan_store()
    worker = _ScanWorker(store, threading.Lock(), threading.Event(), scan_parameters(frames=1))
    worker.run()
    assert store["done"] and not store["error"] and not store["cancelled"]
    for side, name in (("left", "OFDM"), ("right", "OTFS")):
        for family in ("cfo", "doppler"):
            assert int(store[family][name]["bit_errors"][0]) == live[side]["bit_errors"]
            assert int(store[family][name]["total_bits"][0]) == live[side]["total_bits"]
            assert store[family][name]["ber"][0] == live[side]["ber"]


def test_duplicate_live_waveforms_are_identical_and_snapshots_do_not_alias(monkeypatch):
    monkeypatch.setattr(compare_workers, "WaveformScanSimulator", ZeroSimulator)
    runner = _PairedWaveformRunner(frame_interval_s=10.0)
    runner.start(("OFDM", "OFDM"), 12.0, 0.0, 0.0, "QPSK")
    try:
        wait_until(lambda: runner.snapshot()["left"]["frames"] == 1)
    finally:
        runner.stop()
        runner.wait()
    snapshot = runner.snapshot()
    for name in ("frames", "bit_errors", "total_bits", "ber"):
        assert snapshot["left"][name] == snapshot["right"][name]
    np.testing.assert_array_equal(snapshot["left"]["constellation"], snapshot["right"]["constellation"])
    assert snapshot["left"]["ber_v"][0] == 0.0
    assert snapshot["left"]["ber_upper"][0] > 0.0
    snapshot["left"]["constellation"][0] = 999
    snapshot["left"]["ber_v"][0] = 1.0
    retained = runner.snapshot()
    assert retained["left"]["constellation"][0] != 999
    assert retained["right"]["constellation"][0] != 999
    assert retained["left"]["ber_v"][0] == 0.0


def test_live_worker_timeout_keeps_ownership_and_refuses_overlapping_restart(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class BlockingSimulator(ZeroSimulator):
        def simulate_frame_set(self, *args, **kwargs):
            entered.set()
            release.wait(5.0)
            return super().simulate_frame_set(*args, **kwargs)

    monkeypatch.setattr(compare_workers, "WaveformScanSimulator", BlockingSimulator)
    runner = _PairedWaveformRunner()
    params = (("OFDM", "OTFS"), 12.0, 0.0, 0.0, "QPSK")
    runner.start(*params)
    try:
        assert entered.wait(2.0)
        owned = runner._thread
        runner.stop()
        with pytest.raises(RuntimeError):
            runner.wait(timeout=0.01)
        assert runner._thread is owned and owned.is_alive()
        with pytest.raises(RuntimeError):
            runner.start(*params)
        assert runner._thread is owned
    finally:
        release.set()
        runner.wait()
    snapshot = runner.snapshot()
    assert snapshot["done"] and snapshot["cancelled"]
    assert snapshot["left"]["frames"] == snapshot["right"]["frames"] == 0


def test_cancelled_scan_never_commits_a_partial_point(monkeypatch):
    entered, release, stop = threading.Event(), threading.Event(), threading.Event()

    class BlockingSimulator(ZeroSimulator):
        def simulate_frame_set(self, *args, **kwargs):
            entered.set()
            release.wait(5.0)
            return super().simulate_frame_set(*args, **kwargs)

    monkeypatch.setattr(compare_workers, "WaveformScanSimulator", BlockingSimulator)
    store = scan_store()
    worker = _ScanWorker(store, threading.Lock(), stop, scan_parameters())
    worker.start()
    try:
        assert entered.wait(2.0)
        stop.set()
    finally:
        release.set()
        worker.join(2.0)
    assert not worker.is_alive()
    assert store["done"] and store["cancelled"] and not store["error"]
    assert store["cfo"] == store["doppler"] == {}


@pytest.mark.parametrize("failure", ["raise", "unequal_bits", "incomplete_set"])
def test_bad_frame_sets_are_reported_as_errors_not_zero_ber(monkeypatch, failure):
    class BrokenSimulator(ZeroSimulator):
        def simulate_frame_set(self, *args, **kwargs):
            if failure == "raise":
                raise ValueError("receiver failed")
            result = super().simulate_frame_set(*args, **kwargs)
            if failure == "unequal_bits":
                result["OTFS"]["total_bits"] = 99
            else:
                result.pop("OTFS")
            return result

    monkeypatch.setattr(compare_workers, "WaveformScanSimulator", BrokenSimulator)
    store = scan_store()
    _ScanWorker(store, threading.Lock(), threading.Event(), scan_parameters()).run()
    assert store["done"] and store["error"] and not store["cancelled"]
    assert store["cfo"] == store["doppler"] == {}
    runner = _PairedWaveformRunner()
    runner.start(("OFDM", "OTFS"), 12.0, 0.0, 0.0, "QPSK")
    runner.wait()
    snapshot = runner.snapshot()
    assert snapshot["done"] and snapshot["error"]
    assert snapshot["left"]["total_bits"] == snapshot["right"]["total_bits"] == 0
    assert not len(snapshot["left"]["ber_v"])


def test_ui_locks_live_parameters_and_retained_titles_match_the_displayed_result(tab, monkeypatch):
    monkeypatch.setattr(compare_workers, "WaveformScanSimulator", ZeroSimulator)
    tab._on_start_clicked()
    wait_until(lambda: tab.live_runner.snapshot()["left"]["frames"] > 0)
    tab._refresh_live_plots()
    assert tab.left_panel["ber_curve"].yData is None or len(tab.left_panel["ber_curve"].yData) == 0
    assert np.all(tab.left_panel["zero_curve"].yData > 0)
    assert "错误/比特 0/" in tab.left_panel["summary"].text()
    for control in (tab.wave1_combo, tab.wave2_combo, tab.mod_combo, tab.snr_spin,
                    tab.cfo_spin, tab.doppler_spin, tab.seed_spin, tab.receiver_combo):
        assert not control.isEnabled()
    assert not tab.btn_scan_start.isEnabled()
    tab._on_stop_clicked()
    assert not tab.live_runner.is_running() and not tab.timer.isActive()
    assert tab.seed_spin.isEnabled() and tab.snr_spin.isEnabled()
    old_points = tab.left_panel["constellation_scatter"].getData()[0].copy()
    tab.wave1_combo.setCurrentText("AFDM")
    assert "OFDM" in tab.left_panel["box"].title()
    assert "AFDM" not in tab.left_panel["box"].title()
    np.testing.assert_array_equal(tab.left_panel["constellation_scatter"].getData()[0], old_points)
    assert "上次" in tab.note_label.text()


def test_ui_zero_error_table_triangles_and_export_keep_raw_counts_and_config(tab, monkeypatch, tmp_path):
    monkeypatch.setattr(compare_workers, "WaveformScanSimulator", ZeroSimulator)
    set_single_point_scan(tab)
    tab.scan_seed_spin.setValue(71)
    tab._on_scan_start_clicked()
    wait_until(lambda: tab._scan_results.get("done", False))
    tab._refresh_scan_plots()
    assert tab.result_table.rowCount() == 2 and tab.btn_export.isEnabled()
    for row in range(2):
        for column in (2, 3, 4):
            item = tab.result_table.item(row, column)
            assert item.text() == "0"
            assert "0/200" in item.toolTip() and "95%" in item.toolTip()
        assert tab.result_table.item(row, 5).text() == "200"
        assert tab.result_table.item(row, 6).text() == "2"
    for plot in (tab.scan_cfo_plot, tab.scan_doppler_plot):
        for name in ("ofdm", "otfs", "afdm"):
            raw = getattr(plot, name + "_curve").yData
            assert raw is None or len(raw) == 0
            upper = getattr(plot, name + "_zero_curve").yData
            assert len(upper) == 1 and 0 < upper[0] < 1
    tab.scan_seed_spin.setValue(999)
    saved = tmp_path / "comparison.json"
    monkeypatch.setattr(waveform_compare_tab.QFileDialog, "getSaveFileName", lambda *_args: (str(saved), "JSON"))
    tab._export_results()
    report = json.loads(saved.read_text(encoding="utf-8"))
    assert report["config"]["seed"] == 71
    assert report["config"]["receiver_mode"] == "full_mmse"
    for family in ("cfo", "doppler"):
        for stats in report[family].values():
            assert stats["ber"] == [0.0] and stats["bit_errors"] == [0]
            assert stats["total_bits"] == [200] and stats["ci_high"][0] > 0


def test_internal_pages_stop_the_previous_task_and_keep_live_scan_mutually_exclusive(tab, monkeypatch):
    entered = threading.Event()

    class CancellableSimulator(ZeroSimulator):
        def simulate_frame_set(self, *args, stop_event=None, **kwargs):
            entered.set()
            assert stop_event.wait(3.0)
            raise InterruptedError("cancelled frame")

    monkeypatch.setattr(compare_workers, "WaveformScanSimulator", ZeroSimulator)
    tab._on_start_clicked()
    wait_until(lambda: tab.live_runner.snapshot()["left"]["frames"] > 0)
    tab.tabs.setCurrentWidget(tab.scan_page)
    assert not tab.live_runner.is_running() and not tab.timer.isActive()
    monkeypatch.setattr(compare_workers, "WaveformScanSimulator", CancellableSimulator)
    set_single_point_scan(tab)
    tab._on_scan_start_clicked()
    worker = tab._scan_worker
    wait_until(entered.is_set)
    assert not tab.btn_start.isEnabled() and not tab.scan_seed_spin.isEnabled()
    tab.tabs.setCurrentWidget(tab.live_page)
    assert not worker.is_alive()
    assert tab._scan_worker is None and not tab.scan_timer.isActive()
    assert tab.btn_start.isEnabled() and tab.btn_scan_start.isEnabled()
    assert tab._scan_results["cancelled"]


def test_scan_stop_timeout_retains_the_worker_until_it_really_exits(tab, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class BlockingSimulator(ZeroSimulator):
        def simulate_frame_set(self, *args, **kwargs):
            entered.set()
            release.wait(5.0)
            return super().simulate_frame_set(*args, **kwargs)

    monkeypatch.setattr(compare_workers, "WaveformScanSimulator", BlockingSimulator)
    set_single_point_scan(tab)
    tab._on_scan_start_clicked()
    worker = tab._scan_worker
    try:
        assert entered.wait(2.0)
        with pytest.raises(RuntimeError):
            tab.suspend_tasks()
        assert tab._scan_worker is worker and worker.is_alive()
        assert not tab.btn_start.isEnabled() and not tab.btn_scan_start.isEnabled()
    finally:
        release.set()
        worker.join(2.0)
        tab.suspend_tasks()
    assert tab._scan_worker is None and not worker.is_alive()


def test_main_window_navigation_cancels_scan_without_automatic_restart(app, monkeypatch):
    entered = threading.Event()
    constructions = []

    class CancellableSimulator(ZeroSimulator):
        def __init__(self, config=None):
            super().__init__(config)
            constructions.append(True)

        def simulate_frame_set(self, *args, stop_event=None, **kwargs):
            entered.set()
            assert stop_event.wait(3.0)
            raise InterruptedError("navigation cancelled")

    monkeypatch.setattr(compare_workers, "WaveformScanSimulator", CancellableSimulator)
    platform = MainWindow()
    platform.simulation_workspace.tabs.setCurrentIndex(4)
    platform.open_workspace("simulation")
    page = platform.simulation_workspace.pages[4]
    try:
        set_single_point_scan(page)
        page._on_scan_start_clicked()
        worker = page._scan_worker
        wait_until(entered.is_set)
        platform.show_home()
        assert platform.current_mode is None and not worker.is_alive()
        assert not page.scan_timer.isActive()
        platform.open_workspace("simulation")
        QTest.qWait(200)
        assert platform.simulation_workspace.pages[4] is page
        assert page._scan_worker is None and not page.live_runner.is_running()
        assert constructions == [True]
    finally:
        platform.close()


def test_invalid_scan_range_does_not_start_a_worker_or_erase_previous_results(tab, monkeypatch):
    monkeypatch.setattr(compare_workers, "WaveformScanSimulator", ZeroSimulator)
    set_single_point_scan(tab)
    tab._on_scan_start_clicked()
    wait_until(lambda: tab._scan_results.get("done", False))
    tab._refresh_scan_plots()
    prior = tab._scan_results
    tab.scan_cfo_min_spin.setValue(100.0)
    tab.scan_cfo_max_spin.setValue(0.0)
    tab._on_scan_start_clicked()
    assert "参数有误" in tab.scan_note_label.text()
    assert tab._scan_results is prior and tab.result_table.rowCount() == 2
    assert tab._scan_worker is None
