"""配置模型测试：归一化、序列化往返、JSON 存取。"""
from waveform_sim.core.config import (
    AdaptiveConfig,
    ExperimentConfig,
    HardwareConfig,
    WaveformConfig,
)


def test_waveform_config_roundtrip():
    cfg = WaveformConfig(waveform="fdidm", mod_order="qpsk", snr_db=10.0).normalized()
    data = cfg.to_dict()
    cfg2 = WaveformConfig(**data).normalized()
    assert cfg2.waveform == "FDIDM"
    assert cfg2.mod_order == "QPSK"
    assert cfg2.snr_db == 10.0


def test_experiment_config_roundtrip():
    cfg = ExperimentConfig(
        waveform=WaveformConfig(waveform="OFDM", fft_size=64),
        adaptive=AdaptiveConfig(objective="ber"),
        hardware=HardwareConfig(transport="simulated-loopback"),
        operator="测试员",
        tags=["demo"],
    ).normalized()
    data = cfg.to_dict()
    cfg2 = ExperimentConfig.from_dict(data)
    assert cfg2.waveform.waveform == "OFDM"
    assert cfg2.adaptive.objective == "ber"
    assert cfg2.operator == "测试员"
    assert cfg2.tags == ["demo"]


def test_experiment_config_save_load(tmp_path):
    cfg = ExperimentConfig(runs_dir=str(tmp_path)).normalized()
    path = tmp_path / "cfg.json"
    cfg.save(path)
    cfg2 = ExperimentConfig.load(path)
    assert cfg2.to_dict() == cfg.to_dict()

