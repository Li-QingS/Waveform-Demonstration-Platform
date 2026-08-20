"""四波形收发链路冒烟基线：构造、启动、出帧、指标合理。"""
import time

from waveform_sim.simulation.simple_fdidm_rx import FDIDMTransceiver
from waveform_sim.simulation.simple_ofdm_rx import OfdmTransceiver
from waveform_sim.simulation.simple_otfs_rx import OTFSTransceiver
from waveform_sim.simulation.simple_afdm_rx import AFDMTransceiver


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _assert_smoke(transceiver, name: str) -> None:
    transceiver.start()
    try:
        ok = _wait_until(lambda: transceiver.get_last_metrics().get("total_bits", 0) > 0)
        assert ok, f"{name}: 超时未产生帧"
        metrics = transceiver.get_last_metrics()
        assert "ber" in metrics, f"{name}: 缺少 ber"
        assert "evm_db" in metrics or "evm_percent" in metrics, f"{name}: 缺少 evm"
        assert 0.0 <= metrics["ber"] <= 1.0, f"{name}: ber 越界 {metrics['ber']}"
    finally:
        transceiver.stop()
        transceiver.wait(timeout=3.0)


def test_fdidm_transceiver_smoke():
    _assert_smoke(FDIDMTransceiver(alpha=0.0, beta=0.0, snr_db=12.0, channel_seed=42), "FDIDM")


def test_ofdm_transceiver_smoke():
    _assert_smoke(OfdmTransceiver(snr_db=15.0), "OFDM")


def test_otfs_transceiver_smoke():
    _assert_smoke(OTFSTransceiver(snr_db=15.0), "OTFS")


def test_afdm_transceiver_smoke():
    _assert_smoke(AFDMTransceiver(snr_db=15.0), "AFDM")

