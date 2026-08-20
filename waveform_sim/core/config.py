"""统一配置模型（阶段 1）。

只新增、不改旧代码。旧参数名到新字段的映射由后续引擎层
（waveform_sim/core/engine.py）的 update_config 负责。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class WaveformConfig:
    """四类波形（FDIDM/OFDM/OTFS/AFDM）的统一链路参数。"""

    waveform: str = "FDIDM"
    mod_order: str = "16QAM"
    snr_db: float = 20.0
    snr_definition: str = "Eb/N0"
    cfo_hz: float = 0.0
    center_freq_hz: float = 20e9
    payload_bits: int = 1024
    seed: int = 42
    # FDIDM / OTFS 网格
    m_subcarriers: int = 8
    n_symbols: int = 8
    subcarrier_spacing_hz: float = 300_000.0
    # OFDM / AFDM
    fft_size: int = 64
    cp_len: int = 16
    payload_symbols: int = 8
    frame_size: int = 128
    # OTFS
    n_subcarriers: int = 64
    sample_rate_hz: float = 960_000.0
    update_period: float = 0.08
    equalizer: str = "MMSE"
    delay_spread: int = 5
    doppler_spread_hz: float = 20.0
    # FDIDM 索引 / AFDM chirp
    alpha: float = 0.0
    beta: float = 0.0
    c1: float = 0.05
    c2: float = 0.05
    # 信道
    channel_model: str = "TDL-C"
    velocity_kmh: float = 0.0
    doppler_radial_factor: float = 0.10
    random_channel: bool = True
    dynamic_channel: bool = False
    channel_dynamics: str = "fixed"
    channel_coherence_frames: int = 20
    fast_channel_coherence_symbols: int = 1
    circular_channel: bool = True
    tf_notch_depth_db: float = 0.0
    tf_notch_count: int = 0
    # 接收机 / 搜索
    detector: str = "ZF"
    link_mode: str = "matrix"
    search_objective: str = "zf_theory_ser"
    optimize_indices: bool = False
    search_step: float = 0.1

    def normalized(self) -> "WaveformConfig":
        self.waveform = str(self.waveform or "FDIDM").upper()
        self.mod_order = str(self.mod_order or "16QAM").upper()
        self.detector = str(self.detector or "ZF").upper()
        self.equalizer = str(self.equalizer or "MMSE").upper()
        self.snr_definition = str(self.snr_definition or "Eb/N0")
        self.m_subcarriers = max(1, int(self.m_subcarriers))
        self.n_symbols = max(1, int(self.n_symbols))
        self.fft_size = max(1, int(self.fft_size))
        self.cp_len = max(0, int(self.cp_len))
        self.delay_spread = max(0, int(self.delay_spread))
        self.payload_bits = max(1, int(self.payload_bits))
        self.seed = int(max(1, self.seed))
        self.sample_rate_hz = float(max(1.0, self.sample_rate_hz))
        self.subcarrier_spacing_hz = float(max(1.0, self.subcarrier_spacing_hz))
        self.alpha = float(self.alpha)
        self.beta = float(self.beta)
        self.c1 = float(self.c1)
        self.c2 = float(self.c2)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveConfig:
    """alpha/beta 自适应调优参数。"""

    enabled: bool = True
    objective: str = "evm"
    coarse_step: float = 0.2
    fine_step: float = 0.05
    alpha_min: float = 0.0
    alpha_max: float = 2.0
    beta_min: float = 0.0
    beta_max: float = 2.0
    stability_evals: int = 2
    cooldown_frames: int = 20
    max_evaluations: int = 400
    min_improvement_db: float = 0.15
    apply_best: bool = True
    seed: int = 20260428

    def normalized(self) -> "AdaptiveConfig":
        self.objective = str(self.objective or "evm").lower()
        if self.objective not in {"evm", "ber", "ser"}:
            self.objective = "evm"
        self.coarse_step = float(max(0.01, self.coarse_step))
        self.fine_step = float(max(0.005, self.fine_step))
        self.alpha_min = float(self.alpha_min)
        self.alpha_max = float(max(self.alpha_min, self.alpha_max))
        self.beta_min = float(self.beta_min)
        self.beta_max = float(max(self.beta_min, self.beta_max))
        self.stability_evals = int(max(1, self.stability_evals))
        self.cooldown_frames = int(max(0, self.cooldown_frames))
        self.max_evaluations = int(max(1, self.max_evaluations))
        self.seed = int(max(1, self.seed))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HardwareConfig:
    """USRP / 仿真回环硬件参数与 RF 安全约束。"""

    transport: str = "simulated-loopback"
    device_args: str = ""
    device_hint: str = "B210"
    mode: str = "loopback"
    center_freq_hz: float = 2.4e9
    sample_rate_hz: float = 1_000_000.0
    tx_gain_db: float = 10.0
    rx_gain_db: float = 20.0
    tx_antenna: str = "TX/RX"
    rx_antenna: str = "RX2"
    attenuator_db: float = 30.0
    ota_confirmed: bool = False
    frequency_whitelist_hz: List[List[float]] = field(
        default_factory=lambda: [[2.30e9, 2.50e9], [5.70e9, 5.90e9]]
    )
    max_loopback_tx_gain_db: float = 20.0
    max_ota_tx_gain_db: float = 5.0
    process_interval_ms: int = 100
    usrp_buffer_frames: int = 256
    tx_min_waveform_duration_ms: int = 100
    startup_settle_ms: int = 500
    startup_settle_windows: int = 5
    cfo_search_max_hz: float = 25_000.0
    residual_cfo_max_hz: float = 2_000.0

    def normalized(self) -> "HardwareConfig":
        self.transport = str(self.transport or "simulated-loopback").lower()
        self.mode = str(self.mode or "loopback").lower()
        self.sample_rate_hz = float(max(1.0, self.sample_rate_hz))
        self.tx_gain_db = float(self.tx_gain_db)
        self.rx_gain_db = float(self.rx_gain_db)
        self.attenuator_db = float(max(0.0, self.attenuator_db))
        self.process_interval_ms = int(max(10, self.process_interval_ms))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """一次实验的完整配置：波形 + 自适应 + 硬件 + 记录。"""

    waveform: WaveformConfig = field(default_factory=WaveformConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    operator: str = ""
    lab: str = ""
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    record_iq: bool = False
    runs_dir: str = "runs"

    def normalized(self) -> "ExperimentConfig":
        self.waveform.normalized()
        self.adaptive.normalized()
        self.hardware.normalized()
        self.runs_dir = str(self.runs_dir or "runs")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        data = dict(data or {})
        return cls(
            waveform=WaveformConfig(**dict(data.get("waveform") or {})),
            adaptive=AdaptiveConfig(**dict(data.get("adaptive") or {})),
            hardware=HardwareConfig(**dict(data.get("hardware") or {})),
            operator=str(data.get("operator", "")),
            lab=str(data.get("lab", "")),
            notes=str(data.get("notes", "")),
            tags=list(data.get("tags", [])),
            record_iq=bool(data.get("record_iq", False)),
            runs_dir=str(data.get("runs_dir", "runs")),
        ).normalized()

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

