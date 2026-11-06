"""Editable AFDM chirps must reach paired signals, workers and frozen results."""

from dataclasses import replace
import json
import math
import os
import threading
import time

import numpy as np
import pytest

from waveform_sim.simulation.compare_scan_backend import ScanConfig, WaveformScanSimulator
from waveform_sim.ui.compare_workers import _PairedWaveformRunner, _ScanWorker


WAVEFORMS = ("OFDM", "OTFS", "AFDM")


def small_config(**values):
    defaults = dict(fft_len=16, n_slots=3, cp_len=8, channel_model="multipath")
    defaults.update(values)
    return ScanConfig(**defaults)


def frame(simulator, **values):
    defaults = dict(snr_db=6.0, cfo_hz=2300.0, doppler_hz=200.0,
                    mod_order="16QAM", seed=73, frame_index=2)
    defaults.update(values)
    return simulator.simulate_frame_set(WAVEFORMS, **defaults)


def scan_parameters(**values):
    defaults = dict(snr_db=6.0, mod_order="16QAM", frames=1, seed=73,
                    receiver_mode="full_mmse", cfo_axis=np.array([2300.0]),
                    doppler_axis=np.array([200.0]), fixed_cfo=2300.0,
                    fixed_doppler=200.0, afdm_c1=0.01731, afdm_c2=0.001271)
    defaults.update(values)
    return defaults


def store():
    return dict(cfo={}, doppler={}, done=False, status="")


def wait_until(predicate, app=None, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "AFDM parameter experiment did not reach the expected state"
        if app is not None:
            app.processEvents()
        time.sleep(0.01)


@pytest.mark.parametrize("size", [16, 64])
def test_default_chirps_preserve_the_previous_normalized_waveform_at_each_fft(size):
    simulator = WaveformScanSimulator(small_config(fft_len=size))
    assert simulator.cfg.afdm_c1 == pytest.approx(0.05 / size)
    assert simulator.cfg.afdm_c2 == pytest.approx(0.05 / size)
    result = frame(simulator)["AFDM"]
    symbols = result["tx_symbols"].reshape(3, size)
    n = np.arange(size)
    old_time = np.fft.ifft(symbols * np.exp(2j*np.pi*0.05*n*n/size),
                           axis=1, norm="ortho") * np.exp(2j*np.pi*0.05*n*n/size)
    transmitted = np.concatenate((old_time[:, -8:], old_time), axis=1).reshape(-1)
    transmitted *= math.sqrt(simulator.frame_samples / np.vdot(transmitted, transmitted).real)
    np.testing.assert_allclose(result["tx_samples"], transmitted, atol=5e-13, rtol=5e-13)


def test_explicit_chirps_use_the_standard_daft_exponent_and_round_trip():
    size = 16
    c1, c2 = 0.01731, 0.001271
    simulator = WaveformScanSimulator(small_config(afdm_c1=c1, afdm_c2=c2))
    rng = np.random.default_rng(37)
    symbols = rng.normal(size=(3, size)) + 1j*rng.normal(size=(3, size))
    n, k = np.arange(size)[:, None], np.arange(size)[None, :]
    expected_matrix = np.exp(2j*np.pi*(c1*n*n + n*k/size + c2*k*k)) / math.sqrt(size)
    actual = simulator._transmit("AFDM", symbols)
    np.testing.assert_allclose(actual, symbols @ expected_matrix.T, atol=3e-13)
    np.testing.assert_allclose(simulator._receive("AFDM", actual), symbols, atol=3e-14)


def test_zero_chirps_reduce_to_ofdm_through_the_actual_paired_channel():
    simulator = WaveformScanSimulator(small_config(afdm_c1=0.0, afdm_c2=0.0))
    results = frame(simulator)
    for name in ("tx_samples", "rx_samples", "rx_clean_samples", "rx_symbols"):
        np.testing.assert_allclose(results["AFDM"][name], results["OFDM"][name], atol=3e-13)
    assert results["AFDM"]["bit_errors"] == results["OFDM"]["bit_errors"]


def test_manual_chirps_change_only_afdm_while_preserving_pairing_and_energy():
    original = frame(WaveformScanSimulator(small_config()))
    changed = frame(WaveformScanSimulator(small_config(afdm_c1=0.01731, afdm_c2=0.001271)))
    for name in ("OFDM", "OTFS"):
        for key in ("tx_samples", "rx_samples", "rx_symbols", "tx_bits", "rx_bits"):
            np.testing.assert_array_equal(original[name][key], changed[name][key])
    assert np.linalg.norm(original["AFDM"]["tx_samples"] - changed["AFDM"]["tx_samples"]) > 1.0
    for name in WAVEFORMS:
        np.testing.assert_array_equal(original[name]["tx_bits"], changed[name]["tx_bits"])
        np.testing.assert_array_equal(original[name]["noise_samples"], changed[name]["noise_samples"])
        for key in ("delays", "gains", "dopplers_hz", "matrices"):
            np.testing.assert_array_equal(original[name]["channel"][key], changed[name]["channel"][key])
        assert original[name]["tx_energy"] == pytest.approx(changed[name]["tx_energy"], rel=1e-13)


def test_frame_metadata_records_the_actual_coefficients_and_ordinary_cp_contract():
    cfg = small_config(afdm_c1=0.01731, afdm_c2=0.001271)
    results = frame(WaveformScanSimulator(cfg))
    for result in results.values():
        assert result["afdm_c1"] == cfg.afdm_c1 and result["afdm_c2"] == cfg.afdm_c2
        assert result["chirp_convention"] and "cp" in result["prefix"].lower()
        assert result["public_config"]["afdm_c1"] == cfg.afdm_c1
        assert result["public_config"]["afdm_c2"] == cfg.afdm_c2


@pytest.mark.parametrize("field", ["afdm_c1", "afdm_c2"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.001, 0.5001, True])
def test_invalid_chirp_configuration_is_rejected_before_simulation(field, value):
    with pytest.raises(ValueError):
        WaveformScanSimulator(replace(small_config(), **{field: value}))


@pytest.mark.parametrize("field", ["afdm_c1", "afdm_c2"])
def test_invalid_chirp_request_never_starts_a_live_worker_or_commits_scan_points(field):
    runner = _PairedWaveformRunner()
    with pytest.raises(ValueError):
        runner.start(("OFDM", "AFDM"), 6.0, 0.0, 0.0, "QPSK", **{field: float("nan")})
    assert runner._thread is None
    results = store()
    _ScanWorker(results, threading.Lock(), threading.Event(), scan_parameters(**{field: float("nan")})).run()
    assert results["done"] and results["error"]
    assert results["cfo"] == results["doppler"] == {}


def test_live_scan_and_direct_numerics_share_the_same_explicit_chirps_and_counts():
    params = scan_parameters()
    runner = _PairedWaveformRunner(frame_interval_s=10.0)
    runner.start(("OTFS", "AFDM"), params["snr_db"], params["fixed_cfo"],
                 params["fixed_doppler"], params["mod_order"], seed=params["seed"],
                 receiver_mode=params["receiver_mode"], afdm_c1=params["afdm_c1"], afdm_c2=params["afdm_c2"])
    try:
        wait_until(lambda: runner.snapshot()["left"]["frames"] == 1)
    finally:
        runner.stop()
        runner.wait()
    live = runner.snapshot()
    results = store()
    _ScanWorker(results, threading.Lock(), threading.Event(), params).run()
    assert results["done"] and not results["error"]
    reference = WaveformScanSimulator(ScanConfig(afdm_c1=params["afdm_c1"], afdm_c2=params["afdm_c2"]))
    actual = frame(reference, snr_db=params["snr_db"], cfo_hz=params["fixed_cfo"],
                   doppler_hz=params["fixed_doppler"], mod_order=params["mod_order"],
                   seed=params["seed"], frame_index=0)
    for side, name in (("left", "OTFS"), ("right", "AFDM")):
        assert live[side]["bit_errors"] == actual[name]["bit_errors"]
        for family in ("cfo", "doppler"):
            assert results[family][name]["bit_errors"][0] == live[side]["bit_errors"]
            assert results[family][name]["total_bits"][0] == live[side]["total_bits"]
    for config in (live["config"], results["config"]):
        assert config["afdm_c1"] == params["afdm_c1"] and config["afdm_c2"] == params["afdm_c2"]
        assert config["chirp_convention"] and "cp" in config["prefix"].lower()
    live["config"]["afdm_c1"] = 0.4
    assert runner.snapshot()["config"]["afdm_c1"] == params["afdm_c1"]


@pytest.fixture(scope="module")
def app():
    # Keep numerical and non-Qt worker checks active in headless NumPy-only CI.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt5")
    pytest.importorskip("pyqtgraph")
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def tab(app):
    from waveform_sim.ui.waveform_compare_tab import WaveformCompareTab
    page = WaveformCompareTab()
    yield page
    page.close()


def test_only_three_waveforms_remain_and_manual_chirps_reach_locked_live_session(tab, app):
    assert [tab.wave1_combo.itemText(i) for i in range(tab.wave1_combo.count())] == list(WAVEFORMS)
    assert [tab.wave2_combo.itemText(i) for i in range(tab.wave2_combo.count())] == list(WAVEFORMS)
    tab.wave2_combo.setCurrentText("AFDM")
    assert tab.c1_spin.isEnabled() and tab.c2_spin.isEnabled()
    tab.c1_spin.setValue(0.01731)
    tab.c2_spin.setValue(0.001271)
    tab._on_start_clicked()
    wait_until(lambda: tab.live_runner.snapshot()["left"]["frames"] >= 1, app)
    try:
        assert not tab.c1_spin.isEnabled() and not tab.c2_spin.isEnabled()
        assert not tab.scan_c1_spin.isEnabled() and not tab.scan_c2_spin.isEnabled()
        snapshot = tab.live_runner.snapshot()
        assert snapshot["config"]["afdm_c1"] == pytest.approx(0.01731)
        assert snapshot["config"]["afdm_c2"] == pytest.approx(0.001271)
    finally:
        tab._on_stop_clicked()
    assert tab.c1_spin.isEnabled() and tab.c2_spin.isEnabled()
    tab.c1_spin.setValue(0.2)
    tab.c2_spin.setValue(0.3)
    assert tab._live_config["afdm_c1"] == pytest.approx(0.01731)
    assert tab._live_config["afdm_c2"] == pytest.approx(0.001271)


def test_scan_export_keeps_the_chirp_configuration_used_by_the_completed_experiment(tab, app, monkeypatch, tmp_path):
    from waveform_sim.ui import waveform_compare_tab
    for control in (tab.scan_cfo_min_spin, tab.scan_cfo_max_spin,
                    tab.scan_doppler_min_spin, tab.scan_doppler_max_spin,
                    tab.scan_fixed_cfo_spin, tab.scan_fixed_doppler_spin):
        control.setValue(0.0)
    tab.scan_frames_spin.setValue(1)
    tab.scan_c1_spin.setValue(0.0137)
    tab.scan_c2_spin.setValue(0.0021)
    tab._on_scan_start_clicked()
    assert not tab.scan_c1_spin.isEnabled() and not tab.scan_c2_spin.isEnabled()
    wait_until(lambda: tab._scan_results.get("done", False), app)
    tab._refresh_scan_plots()
    assert tab.result_table.rowCount() == 2 and tab.btn_export.isEnabled()
    assert tab._scan_results["config"]["afdm_c1"] == pytest.approx(0.0137)
    assert tab._scan_results["config"]["afdm_c2"] == pytest.approx(0.0021)
    tab.scan_c1_spin.setValue(0.4)
    tab.scan_c2_spin.setValue(0.3)
    destination = tmp_path / "chirp_comparison.json"
    monkeypatch.setattr(waveform_compare_tab.QFileDialog, "getSaveFileName", lambda *_: (str(destination), "JSON"))
    tab._export_results()
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["config"]["afdm_c1"] == pytest.approx(0.0137)
    assert report["config"]["afdm_c2"] == pytest.approx(0.0021)
    assert report["config"]["chirp_convention"]
    for family in ("cfo", "doppler"):
        assert set(report[family]) == set(WAVEFORMS)
        assert all(stats["frames"] == [1] for stats in report[family].values())
