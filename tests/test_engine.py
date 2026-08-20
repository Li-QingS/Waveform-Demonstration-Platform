"""统一引擎与 FDIDM 兼容壳测试。"""
import time

import pytest

from waveform_sim.core.config import WaveformConfig
from waveform_sim.core.engine import LinkSimulator
from waveform_sim.simulation.simple_fdidm_rx import (
    FDIDMTransceiver,
    _LegacyFDIDMTransceiver,
)


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_engine_fdidm_uses_legacy_backend():
    sim = LinkSimulator(WaveformConfig(waveform="FDIDM", snr_db=12.0, seed=42))
    assert sim._backend.__class__.__name__ == "_LegacyFDIDMTransceiver"
    sim.start()
    try:
        ok = _wait_until(lambda: sim.get_last_metrics().get("total_bits", 0) > 0)
        assert ok, "engine FDIDM 超时未出帧"
        m = sim.get_last_metrics()
        assert "ber" in m and "ser" in m and "evm_db" in m
        assert 0.0 <= m["ber"] <= 1.0
    finally:
        sim.stop()
        sim.wait(timeout=3.0)


def test_engine_config_aliases():
    sim = LinkSimulator(waveform="FDIDM", snr_db=12.0, channel_seed=7, decoder="MMSE", fc_hz=1e9)
    assert sim.config.snr_db == 12.0
    assert sim.config.seed == 7
    assert sim.config.detector == "MMSE"
    assert sim.config.center_freq_hz == 1e9
    sim.update_config(ebn0_db=9.0)
    assert sim.config.snr_db == 9.0
    sim.stop()


def test_fdidm_shell_is_link_simulator():
    tb = FDIDMTransceiver(alpha=0.2, beta=0.1, snr_db=12.0, channel_seed=42)
    assert isinstance(tb, LinkSimulator)
    assert tb._backend.__class__.__name__ == "_LegacyFDIDMTransceiver"
    assert callable(tb.get_debug_snapshot)
    pts, labels = FDIDMTransceiver._build_gray_qam(16)
    assert pts.shape == (16,) and labels.shape == (16, 4)


def test_fdidm_shell_matches_legacy_metrics_keys():
    kwargs = dict(snr_db=12.0, channel_seed=42)
    shell = FDIDMTransceiver(**kwargs)
    legacy = _LegacyFDIDMTransceiver(**kwargs)
    shell.start()
    legacy.start()
    try:
        ok = _wait_until(lambda: shell.get_last_metrics().get("total_bits", 0) > 0)
        assert ok
        ms = shell.get_last_metrics()
        ml = legacy.get_last_metrics()
        assert set(ms.keys()) == set(ml.keys())
        assert 0.0 <= ms["ber"] <= 1.0 and 0.0 <= ml["ber"] <= 1.0
    finally:
        shell.stop()
        legacy.stop()
        shell.wait(timeout=3.0)
        legacy.wait(timeout=3.0)


@pytest.mark.parametrize(
    "waveform,kwargs,legacy_cls",
    [
        ("OFDM", dict(snr_db=15.0), "_LegacyOfdmTransceiver"),
        ("OTFS", dict(snr_db=15.0), "_LegacyOTFSTransceiver"),
        ("AFDM", dict(snr_db=15.0), "_LegacyAFDMTransceiver"),
    ],
)
def test_engine_backend_smoke(waveform, kwargs, legacy_cls):
    sim = LinkSimulator(WaveformConfig(waveform=waveform, **kwargs))
    assert sim._backend.__class__.__name__ == legacy_cls
    sim.start()
    try:
        ok = _wait_until(lambda: sim.get_last_metrics().get("total_bits", 0) > 0)
        assert ok, f"{waveform} 超时未出帧"
        m = sim.get_last_metrics()
        assert "ber" in m and ("evm_db" in m or "evm_percent" in m)
        assert 0.0 <= m["ber"] <= 1.0
    finally:
        sim.stop()
        sim.wait(timeout=3.0)
