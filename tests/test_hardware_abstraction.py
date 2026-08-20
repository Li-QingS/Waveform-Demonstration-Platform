"""硬件抽象层测试：RF safety、transport、IQ replay、设备探测。"""
import numpy as np
import pytest

from waveform_sim.core.config import HardwareConfig
from waveform_sim.hardware.device_manager import probe_devices
from waveform_sim.hardware.iq_replay import IQReplaySource, save_iq
from waveform_sim.hardware.rf_safety import RFSafetyPolicy, RFSafetyViolation
from waveform_sim.hardware.transport import create_transport


def test_rf_safety_blocks_bad_loopback():
    cfg = HardwareConfig(mode="loopback", tx_gain_db=40.0, attenuator_db=0.0)
    with pytest.raises(RFSafetyViolation):
        RFSafetyPolicy(strict=True).validate(cfg)


def test_rf_safety_allows_good_loopback():
    cfg = HardwareConfig(mode="loopback", tx_gain_db=10.0, attenuator_db=30.0)
    decision = RFSafetyPolicy(strict=True).validate(cfg)
    assert decision.allowed
    assert decision.warnings == []


def test_rf_safety_blocks_ota_without_confirm():
    cfg = HardwareConfig(mode="ota", ota_confirmed=False, center_freq_hz=2.4e9)
    with pytest.raises(RFSafetyViolation):
        RFSafetyPolicy(strict=True).validate(cfg)


def test_rf_safety_blocks_out_of_whitelist():
    cfg = HardwareConfig(mode="ota", ota_confirmed=True, tx_gain_db=3.0, center_freq_hz=1.0e9)
    with pytest.raises(RFSafetyViolation):
        RFSafetyPolicy(strict=True).validate(cfg)


def test_transport_loopback_roundtrip():
    tx = create_transport(HardwareConfig(transport="simulated-loopback"))
    x = np.array([1 + 2j, 3 - 1j, 0.5 + 0.5j], dtype=np.complex128)
    tx.send(x)
    y = tx.recv(6)
    assert y.shape == (6,)
    assert np.allclose(y, np.tile(x, 2))


def test_transport_factory_unknown():
    with pytest.raises(ValueError):
        create_transport(HardwareConfig(transport="unknown"))


def test_uhd_transport_placeholder():
    try:
        t = create_transport(HardwareConfig(transport="uhd"))
    except RuntimeError as exc:
        assert "not available" in str(exc)
        return
    with pytest.raises(NotImplementedError):
        t.send(np.zeros(4, dtype=np.complex128))


def test_iq_replay_npy(tmp_path):
    path = tmp_path / "iq.npy"
    x = np.array([1 + 1j, 2 + 0j, 0 - 3j], dtype=np.complex128)
    save_iq(path, x)
    src = IQReplaySource(path, loop=True)
    assert np.allclose(src.read(3), x)
    assert np.allclose(src.read(6), np.tile(x, 2))


def test_iq_replay_raw_complex64(tmp_path):
    path = tmp_path / "iq.bin"
    x = np.array([1 + 1j, 2 + 0j], dtype=np.complex128)
    save_iq(path, x)
    src = IQReplaySource(path, loop=False)
    assert np.allclose(src.read(1), x[:1])
    assert np.allclose(src.read(5), x[1:])


def test_device_manager_probe():
    devices = probe_devices(timeout=5.0)
    assert isinstance(devices, list) and len(devices) >= 1
    assert hasattr(devices[0], "available")

