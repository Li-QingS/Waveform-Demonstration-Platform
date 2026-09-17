"""硬件后端模块导入基线：四个 hardtest 模块可导入且主类存在。"""
import importlib
import threading

import numpy as np
import pytest

HARDTEST_CLASSES = {
    "waveform_sim.hardware.fdidm_hardtest": "FDIDMHardwareTest",
    "waveform_sim.hardware.ofdm_hardtest": "OfdmHardwareTx",
    "waveform_sim.hardware.otfs_hardtest": "OTFSHardwareTest",
    "waveform_sim.hardware.afdm_hardtest": "AFDMHardwareTest",
}


def test_hardtest_modules_import():
    for mod_name, cls_name in HARDTEST_CLASSES.items():
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, cls_name), f"{mod_name} 缺少 {cls_name}"


def test_hardware_evidence_module_imports_without_runtime():
    mod = importlib.import_module("waveform_sim.hardware.evidence")
    assert hasattr(mod, "TxPowerContract")
    assert hasattr(mod, "classify_validation")


def test_fdidm_gain_guard_accepts_45_db_and_rejects_invalid_values():
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    assert mod.validate_fdidm_gain(45.0, name="TX gain") == 45.0
    assert mod.validate_fdidm_gain(45, name="RX gain") == 45.0
    for value in (-0.1, 45.01, float("nan"), float("inf"), "bad"):
        with pytest.raises(ValueError):
            mod.validate_fdidm_gain(value, name="TX gain")


def test_fdidm_constructor_and_runtime_gain_limits_are_reported(monkeypatch):
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    # This is a configuration-contract test, not a hardware smoke test.  UHD
    # may be installed on the developer machine (and a connected B210 may be
    # busy in the GUI), so keep construction offline and deterministic.
    monkeypatch.setattr(mod._LegacyFDIDMHardwareTest, "_build_top_block", lambda self: None)
    obj = mod._LegacyFDIDMHardwareTest(
        tx_gain=45.0, rx_gain=45.0, tx_frame_count=1,
        tx_min_waveform_duration_ms=1, log_to_stdout=False,
    )
    status = obj.get_status()
    assert status["tx_gain"] == 45.0 and status["rx_gain"] == 45.0
    assert status["gain_limits_db"] == {
        "tx_min": 0.0, "tx_max": 45.0, "rx_min": 0.0, "rx_max": 45.0,
    }
    # Geometry is known before RX starts; the initial status must not leak the
    # internal NaN reset sentinel into the operator-facing summary.
    assert np.isfinite(status["cfo_unambiguous_hz"])
    assert status["cfo_unambiguous_hz"] > 0.0
    with pytest.raises(ValueError):
        obj.set_tx_gain(45.1)
    with pytest.raises(ValueError):
        obj.configure(rx_gain=-1.0)


def test_fdidm_configure_validates_tx_and_rx_before_partial_update(monkeypatch):
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    monkeypatch.setattr(mod._LegacyFDIDMHardwareTest, "_build_top_block", lambda self: None)
    obj = mod._LegacyFDIDMHardwareTest(
        tx_gain=10.0, rx_gain=20.0, tx_frame_count=1,
        tx_min_waveform_duration_ms=1, log_to_stdout=False,
    )
    with pytest.raises(ValueError, match="RX gain"):
        obj.configure(tx_gain=30.0, rx_gain=45.1)
    assert obj.tx_gain == 10.0 and obj.rx_gain == 20.0


def test_probe_reanchor_does_not_report_whole_run_as_new_samples():
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")

    class Probe:
        def level(self):
            return np.ones(128, dtype=np.complex64)

    obj = object.__new__(mod._LegacyFDIDMHardwareTest)
    obj._rx_probe = Probe()
    obj._rx_probe_mode = "probe_signal_vc"
    obj._rx_probe_last_fp = None
    obj._rx_probe_reanchor_pending = True
    obj._rx_probe_start_t = 1.0
    obj._rx_samples_seen = 0
    obj.sample_rate = 500_000.0
    obj._debug = lambda *_args: None

    vec, absolute, available, newly = obj._read_rx_probe_window(128)
    assert vec.size == available == 128
    assert absolute == newly == 128


def test_refined_preamble_gate_rejects_noise_like_scores():
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    obj = object.__new__(mod._LegacyFDIDMHardwareTest)
    obj.preamble_refine_min_score = 0.20

    assert not obj._refined_preamble_is_reliable(float("nan"))
    assert not obj._refined_preamble_is_reliable(0.06)
    assert not obj._refined_preamble_is_reliable(0.1999)
    assert obj._refined_preamble_is_reliable(0.20)
    assert obj._refined_preamble_is_reliable(0.41)


def test_preamble_cfo_alias_extremes_are_finite_and_bounded():
    """Corrupt/overflowing phase estimates must not escape the CFO scan span."""
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    obj = object.__new__(mod._LegacyFDIDMHardwareTest)
    obj.sample_rate = 1_000_000.0
    obj.sync_half_len = 128
    obj.cfo_search_max_hz = 50_000.0
    obj.cfo_search_enable = True

    for alias in (float("nan"), float("inf"), float("-inf"), 1e300, -1e300):
        candidates = obj._preamble_cfo_candidates(alias)
        assert candidates
        assert all(np.isfinite(c) and abs(c) <= 50_000.0 + 1e-6 for c in candidates)
        cfo, score, _reason = obj._choose_cfo_from_scored_candidates(alias, [])
        assert np.isfinite(cfo)
        assert np.isfinite(score)


def test_choose_cfo_discards_nonfinite_and_out_of_range_scored_candidates():
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    obj = object.__new__(mod._LegacyFDIDMHardwareTest)
    obj.sample_rate = 1_000_000.0
    obj.sync_half_len = 128
    obj.cfo_search_max_hz = 50_000.0
    obj.cfo_search_enable = True
    cfo, score, reason = obj._choose_cfo_from_scored_candidates(
        0.0,
        [(1e300, 1.0), (float("nan"), 0.9), (1_000.0, 0.8)],
    )
    assert cfo == pytest.approx(1_000.0)
    assert score == pytest.approx(0.8)
    assert reason in {"scan_best", "alias_best"}


def test_live_waveform_sync_pauses_only_tx_graph():
    """A live TX swap must not stop the RX graph.

    Stopping both graphs made the UHD RX streamer report recurring
    "overflows occurred" for every alpha/beta candidate swap.
    """
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    events = []
    rx_events = []

    class TopBlock:
        def stop(self):
            events.append("stop")

        def wait(self):
            events.append("wait")

        def start(self):
            events.append("start")

    class RxTopBlock:
        def stop(self):
            rx_events.append("stop")

        def wait(self):
            rx_events.append("wait")

        def start(self):
            rx_events.append("start")

    class VectorSource:
        def set_data(self, data, tags):
            events.append("set_data")
            assert data == [complex(1, 2)]
            assert tags == []

        def rewind(self):
            events.append("rewind")

    obj = object.__new__(mod._LegacyFDIDMHardwareTest)
    obj._tb_tx = TopBlock()
    obj._tb_rx = RxTopBlock()
    obj._tb = obj._tb_rx
    obj._vector_source = VectorSource()
    obj._tx_waveform = np.array([1 + 2j], dtype=np.complex64)
    obj._tdl_channel_block = None
    obj._running = True
    obj._needs_top_block_rebuild = False
    obj._debug = lambda *_args: None
    obj._waveform_fingerprint = lambda: "test"
    obj._reset_rx_runtime_state = lambda **_kwargs: events.append("reset")

    obj._sync_waveform_to_top_block()

    assert events == ["stop", "wait", "set_data", "rewind", "start", "reset"]
    assert not obj._needs_top_block_rebuild
    # Only the TX graph was touched; the RX graph recorded nothing.
    assert rx_events == []


def test_live_waveform_sync_failure_is_not_reported_as_applied():
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    events = []

    class TopBlock:
        def stop(self):
            events.append("stop")

        def wait(self):
            events.append("wait")

        def start(self):
            events.append("start")

    class VectorSource:
        def set_data(self, *_args):
            events.append("set_data")
            raise ValueError("cannot replace vector")

    obj = object.__new__(mod._LegacyFDIDMHardwareTest)
    obj._tb_tx = TopBlock()
    obj._tb_rx = TopBlock()
    obj._tb = obj._tb_rx
    obj._vector_source = VectorSource()
    obj._tx_waveform = np.array([1 + 2j], dtype=np.complex64)
    obj._tdl_channel_block = None
    obj._running = True
    obj._needs_top_block_rebuild = False
    obj._debug = lambda *_args: None
    obj._waveform_fingerprint = lambda: "test"

    with pytest.raises(RuntimeError, match="live waveform sync failed"):
        obj._sync_waveform_to_top_block()

    assert events == ["stop", "wait", "set_data", "start"]
    assert obj._needs_top_block_rebuild


def test_live_validation_swap_can_reset_rx_state_under_rx_lock(monkeypatch):
    """The validation worker owns _lock while syncing a live waveform."""
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    monkeypatch.setattr(mod._LegacyFDIDMHardwareTest, "_build_top_block", lambda self: None)
    obj = mod._LegacyFDIDMHardwareTest(
        tx_frame_count=1, tx_min_waveform_duration_ms=1, log_to_stdout=False,
    )

    class TopBlock:
        def stop(self):
            pass

        def wait(self):
            pass

        def start(self):
            pass

    class VectorSource:
        def set_data(self, *_args):
            pass

        def rewind(self):
            pass

    obj._tb_tx = TopBlock()
    obj._tb_rx = TopBlock()
    obj._tb = obj._tb_rx
    obj._vector_source = VectorSource()
    obj._running = True
    errors = []

    def swap():
        try:
            with obj._lock:
                obj._sync_waveform_to_top_block()
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=swap, daemon=True)
    worker.start()
    worker.join(timeout=2.0)
    assert not worker.is_alive(), "live validation swap deadlocked while resetting RX state"
    assert not errors
