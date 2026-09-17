"""TX/RX graph split: a live TX swap must not stop the receive graph.

These tests need GNU Radio but no USRP: the UHD source/sink are replaced by
GNU Radio null blocks with no-op configuration setters, so the topology,
start/stop ordering and live-swap behaviour can be exercised offline.

Context: replacing the TX waveform used to stop the whole top_block.  The UHD
RX streamer kept receiving while no block drained it, so the console printed
"O" and "N overflows occurred" for every alpha/beta candidate swap.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip("gnuradio")

import gnuradio.blocks as blocks
import gnuradio.gr as gr
import gnuradio.uhd as uhd


def _fake_uhd(time_spec_cls=None):
    """Return a namespace-compatible stand-in for ``gnuradio.uhd``."""

    class _FakeSource(blocks.null_source):
        def __init__(self):
            blocks.null_source.__init__(self, gr.sizeof_gr_complex)

        def set_subdev_spec(self, *_a, **_k):
            pass

        def set_samp_rate(self, *_a, **_k):
            pass

        def set_time_unknown_pps(self, *_a, **_k):
            pass

        def set_center_freq(self, *_a, **_k):
            pass

        def set_antenna(self, *_a, **_k):
            pass

        def set_gain(self, *_a, **_k):
            pass

        def set_bandwidth(self, *_a, **_k):
            pass

        def get_bandwidth(self, *_a, **_k):
            return 0.0

    class _FakeSink(blocks.null_sink):
        def __init__(self):
            blocks.null_sink.__init__(self, gr.sizeof_gr_complex)

        set_subdev_spec = _FakeSource.set_subdev_spec
        set_samp_rate = _FakeSource.set_samp_rate
        set_time_unknown_pps = _FakeSource.set_time_unknown_pps
        set_center_freq = _FakeSource.set_center_freq
        set_antenna = _FakeSource.set_antenna
        set_gain = _FakeSource.set_gain
        set_bandwidth = _FakeSource.set_bandwidth
        get_bandwidth = _FakeSource.get_bandwidth

    class _TimeSpec:
        def __init__(self, *_a, **_k):
            pass

    class _FakeUhd:
        time_spec = _TimeSpec

        @staticmethod
        def stream_args(**_kwargs):
            return None

        @staticmethod
        def usrp_source(*_a, **_k):
            return _FakeSource()

        @staticmethod
        def usrp_sink(*_a, **_k):
            return _FakeSink()

    return _FakeUhd()


def _make_backend(monkeypatch, **kwargs):
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    # The constructor builds a real graph (which would need a device); keep it
    # offline, then run the real builder against the fake UHD namespace.
    real_build = mod._LegacyFDIDMHardwareTest._build_top_block
    monkeypatch.setattr(mod._LegacyFDIDMHardwareTest, "_build_top_block", lambda self: None)
    params = dict(
        carrier_freq=2.4e9, samp_rate=500_000.0, tx_gain=20.0, rx_gain=20.0,
        tx_text="GRAPH SPLIT TEST", mod_order="QPSK", equalizer="MMSE",
        channel_estimator="diag_tf", channel_mode="rf",
        tx_frame_count=1, tx_min_waveform_duration_ms=1.0,
        process_interval_ms=1000.0, adaptive_alpha_beta_enable=False,
        log_to_stdout=False,
    )
    params.update(kwargs)
    obj = mod._LegacyFDIDMHardwareTest(**params)
    obj._uhd = _fake_uhd()
    real_build(obj)
    return obj


def test_build_creates_separate_tx_and_rx_graphs(monkeypatch):
    obj = _make_backend(monkeypatch)
    assert obj._tb_tx is not None and obj._tb_rx is not None
    assert obj._tb_tx is not obj._tb_rx
    # ``_tb`` stays the receive graph for existing RX-side lookups.
    assert obj._tb is obj._tb_rx
    assert obj._tb_tx.name() == "FDIDM Hardware TX"
    assert obj._tb_rx.name() == "FDIDM Hardware RX"


def test_start_and_stop_cover_both_graphs(monkeypatch):
    obj = _make_backend(monkeypatch)
    calls = []
    for label, graph in (("tx", obj._tb_tx), ("rx", obj._tb_rx)):
        graph.start = (lambda _l: (lambda: calls.append((_l, "start"))))(label)
        graph.stop = (lambda _l: (lambda: calls.append((_l, "stop"))))(label)
        graph.wait = (lambda _l: (lambda: calls.append((_l, "wait"))))(label)
    try:
        obj.start()
        assert calls[:2] == [("rx", "start"), ("tx", "start")]
        calls.clear()
        obj.stop()
        assert calls[0] == ("tx", "stop")
        assert ("rx", "stop") in calls
    finally:
        obj._monitor_stop.set()
        if obj._monitor_thread is not None:
            obj._monitor_thread.join(timeout=3.0)
        obj._running = False


def test_live_swap_stops_only_the_tx_graph(monkeypatch):
    obj = _make_backend(monkeypatch)
    events = []
    obj._tb_tx.stop = lambda: events.append("tx.stop")
    obj._tb_tx.wait = lambda: events.append("tx.wait")
    obj._tb_tx.start = lambda: events.append("tx.start")
    obj._tb_rx.stop = lambda: events.append("rx.stop")
    obj._tb_rx.start = lambda: events.append("rx.start")
    obj._running = True

    obj._sync_waveform_to_top_block()

    assert events == ["tx.stop", "tx.wait", "tx.start"]
    assert "rx.stop" not in events
    obj._running = False


def test_live_swap_failure_restores_the_tx_graph(monkeypatch):
    obj = _make_backend(monkeypatch)
    events = []
    obj._tb_tx.stop = lambda: events.append("tx.stop")
    obj._tb_tx.wait = lambda: events.append("tx.wait")
    obj._tb_tx.start = lambda: events.append("tx.start")
    obj._tb_rx.stop = lambda: events.append("rx.stop")
    obj._running = True

    class _FailingVectorSource:
        def set_data(self, *_args):
            raise ValueError("cannot replace vector")

        def rewind(self):
            pass

    obj._vector_source = _FailingVectorSource()
    with pytest.raises(RuntimeError, match="live waveform sync failed"):
        obj._sync_waveform_to_top_block()

    assert events == ["tx.stop", "tx.wait", "tx.start"]
    assert obj._needs_top_block_rebuild
    assert "rx.stop" not in events
    obj._running = False
