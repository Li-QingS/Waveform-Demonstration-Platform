"""Contracts for the optional pre-RF NTN impairment in the OFDM backend."""
from collections import deque
import threading

import numpy as np
import pytest

from waveform_sim.hardware import ofdm_hardtest


def _channel(**overrides):
    params = {
        "sample_rate_hz": 960_000.0,
        "carrier_frequency_hz": 2.4e9,
        "profile": "NTN-TDLC5-200",
        "enabled": True,
        "seed": 41,
    }
    params.update(overrides)
    return ofdm_hardtest.OfdmNtnChannel(**params)


def test_disabled_ntn_channel_is_an_exact_sample_passthrough():
    samples = (np.arange(129) * (0.25 + 0.125j)).astype(np.complex64)
    channel = _channel(enabled=False)

    result = channel.process(samples)

    assert result.dtype == samples.dtype
    np.testing.assert_array_equal(result, samples)
    assert channel.sample_index == 0


def test_ntn_channel_is_reproducible_for_a_fixed_seed():
    samples = np.exp(1j * np.arange(8192) * 0.037).astype(np.complex64)
    first = _channel(seed=7).process(samples)
    second = _channel(seed=7).process(samples)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, samples)


def test_ntn_channel_state_is_continuous_across_scheduler_chunks():
    samples = np.exp(1j * np.arange(8192) * 0.031).astype(np.complex64)
    whole = _channel(profile="NTN-TDLA100-200", seed=13).process(samples)
    chunked_channel = _channel(profile="NTN-TDLA100-200", seed=13)
    chunked = np.concatenate([
        chunked_channel.process(samples[start:start + 257])
        for start in range(0, samples.size, 257)
    ])

    np.testing.assert_allclose(chunked, whole, rtol=1e-6, atol=1e-6)
    assert chunked_channel.sample_index == samples.size


def test_reset_restarts_the_same_seeded_ntn_realization():
    samples = np.ones(1024, dtype=np.complex64)
    channel = _channel(profile="NTN-TDLA100-200", seed=23)
    first = channel.process(samples)
    channel.reset()
    second = channel.process(samples)

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("profile", [
    "NTN-TDLA100-200", "NTN-TDLC5-200", "NTN-TDLC5-1200",
])
def test_ntn_channel_accepts_the_existing_standard_profiles(profile):
    channel = _channel(profile=profile)
    summary = channel.summary()

    assert summary["profile"] == profile
    assert summary["enabled"] is True
    assert summary["maximum_delay_s"] >= 0.0


def test_ntn_channel_rejects_unknown_profiles_and_nonfinite_parameters():
    with pytest.raises(ValueError, match="profile"):
        _channel(profile="made-up-profile")
    with pytest.raises(ValueError, match="finite"):
        _channel(doppler_spread_hz=float("nan"))


def test_decoder_returns_integer_bit_error_count_and_denominator():
    backend = object.__new__(ofdm_hardtest._LegacyOfdmHardwareTx)
    backend.mod_order = "QPSK"
    backend._tx_frame_bits = np.array([0, 0, 1, 1, 0, 1, 1, 0], dtype=np.int8)
    backend._payload_repeat = 1
    backend._phi_locked = 0.0
    backend._frame_soft_history = deque(maxlen=12)
    backend._combine_frames = 0
    backend._parse_app_frame_exact = lambda _frame: (False, b"")

    received_bits = backend._tx_frame_bits.copy()
    received_bits[3] ^= 1
    pairs = received_bits.reshape(-1, 2)
    symbols = (1 - 2 * pairs[:, 1]) + 1j * (1 - 2 * pairs[:, 0])

    decoded = backend._recover_payload_from_symbols(symbols)

    assert decoded[1:3] == (1, len(backend._tx_frame_bits))
    assert decoded[0] == pytest.approx(1 / len(backend._tx_frame_bits))


def test_decode_stats_accumulate_only_integer_frame_counts():
    backend = object.__new__(ofdm_hardtest._LegacyOfdmHardwareTx)
    backend._lock = threading.Lock()
    backend._frame_id = 0
    backend._frame_bit_errors = 0
    backend._frame_bits_total = 0
    backend._bit_errors_total = 0
    backend._bits_total = 0
    backend._tx_payload = b""
    backend._match_bytes = 0
    backend._decode_ok = False

    backend._record_ber_counts(2, 128)
    backend._record_ber_counts(0, 128)
    stats = backend.get_decode_stats()

    assert stats["frame_id"] == 2
    assert stats["frame_bit_errors"] == 0
    assert stats["frame_bits_total"] == 128
    assert stats["frame_ber"] == 0.0
    assert stats["bit_errors"] == 2
    assert stats["bits_total"] == 256
    assert stats["cumulative_ber"] == pytest.approx(2 / 256)


def test_frame_limit_prevents_decoding_more_frames_in_the_current_run():
    backend = object.__new__(ofdm_hardtest._LegacyOfdmHardwareTx)
    backend._running = False
    backend._frame_id = 0
    backend._frame_limit = None
    backend._lock = threading.Lock()
    backend.set_frame_limit(2)
    backend._frame_id = 2
    backend._sync_metric = lambda _window: np.array([1.0])
    backend._find_sync_peaks = lambda *_args, **_kwargs: pytest.fail(
        "frame-limited runs must not decode another frame"
    )

    backend._try_process_rx_window(np.zeros(8, dtype=np.complex128), abs_seen=8)

    assert backend._frame_id == 2


def test_stop_keeps_a_timed_out_monitor_thread_and_requires_it_to_exit():
    calls = []

    class SlowMonitorThread:
        def join(self, timeout=None):
            calls.append(("monitor.join", timeout))

        def is_alive(self):
            return True

    class TopBlock:
        def stop(self):
            calls.append("top.stop")

        def wait(self):
            calls.append("top.wait")

        def start(self):
            calls.append("top.start")

    backend = object.__new__(ofdm_hardtest._LegacyOfdmHardwareTx)
    backend._running = True
    backend._status = "running"
    backend._monitor_stop = threading.Event()
    monitor_thread = SlowMonitorThread()
    backend._monitor_thread = monitor_thread
    backend._tb = TopBlock()

    with pytest.raises(RuntimeError, match="监控线程"):
        backend.stop()

    assert calls == ["top.stop", "top.wait", ("monitor.join", 1.5)]
    assert backend._monitor_thread is monitor_thread
    assert backend._running is False
    assert backend._monitor_stop.is_set()


def test_start_does_not_clear_stop_event_while_old_monitor_is_alive(monkeypatch):
    calls = []

    class SlowMonitorThread:
        def is_alive(self):
            return True

    class TopBlock:
        def start(self):
            calls.append("top.start")

    class NewMonitorThread:
        def __init__(self, **_kwargs):
            calls.append("monitor.create")

        def start(self):
            calls.append("monitor.start")

    backend = object.__new__(ofdm_hardtest._LegacyOfdmHardwareTx)
    backend._running = False
    backend._status = "stopped"
    backend._tb = TopBlock()
    backend._monitor_stop = threading.Event()
    backend._monitor_stop.set()
    backend._monitor_thread = SlowMonitorThread()
    backend._ntn_channel = type("Channel", (), {"reset": lambda _self: None})()
    backend._reset_run_metrics = lambda: None
    monkeypatch.setattr(ofdm_hardtest.threading, "Thread", NewMonitorThread)

    with pytest.raises(RuntimeError, match="监控线程"):
        backend.start()

    assert calls == []
    assert backend._monitor_stop.is_set()


def test_backend_configures_ntn_while_stopped_and_exposes_the_run_settings(monkeypatch):
    builds = []
    monkeypatch.setattr(
        ofdm_hardtest._LegacyOfdmHardwareTx, "_import_runtime", lambda self: None,
    )
    monkeypatch.setattr(
        ofdm_hardtest._LegacyOfdmHardwareTx,
        "_build_top_block",
        lambda self: builds.append(self.get_ntn_config()),
    )
    backend = ofdm_hardtest._LegacyOfdmHardwareTx(ntn_enabled=False)

    backend.configure(
        ntn_enabled=True,
        ntn_profile="NTN-TDLA100-200",
        ntn_seed=17,
        ntn_delay_scale=1.5,
    )
    status = backend.get_status()

    assert len(builds) == 2
    assert status["ntn"]["enabled"] is True
    assert status["ntn"]["profile"] == "NTN-TDLA100-200"
    assert status["ntn"]["seed"] == 17
    assert status["ntn"]["delay_scale"] == 1.5


def test_backend_rejects_invalid_ntn_configuration_before_partial_update(monkeypatch):
    monkeypatch.setattr(
        ofdm_hardtest._LegacyOfdmHardwareTx, "_import_runtime", lambda self: None,
    )
    monkeypatch.setattr(ofdm_hardtest._LegacyOfdmHardwareTx, "_build_top_block", lambda self: None)
    backend = ofdm_hardtest._LegacyOfdmHardwareTx(carrier_freq=2.4e9)

    with pytest.raises(ValueError, match="profile"):
        backend.configure(carrier_freq=2.3e9, ntn_profile="bad-profile")

    assert backend.carrier_freq == 2.4e9
