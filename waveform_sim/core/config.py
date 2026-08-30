"""Waveform platform configuration models.

The configuration keeps legacy field aliases for the existing waveform tabs while
adding explicit NTN Doppler, dynamic-channel and dual-timescale adaptation
parameters used by the upgraded FDIDM page.
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
    demo_frame_interval_s: float = 0.02
    equalizer: str = "MMSE"
    delay_spread: int = 5
    doppler_spread_hz: float = 20.0
    # FDIDM 索引 / AFDM chirp
    alpha: float = 0.0
    beta: float = 0.0
    c1: float = 0.05
    c2: float = 0.05
    # 信道
    channel_model: str = "NTN-TDLA100-200"
    # Explicit NR-NTN profile.  `channel_model` is kept for backward compatibility.
    ntn_profile: str = "NTN-TDLA100-200"
    velocity_kmh: float = 0.0
    doppler_radial_factor: float = 0.10
    # Large common orbital Doppler is predicted/compensated; the remaining
    # multipath time selectivity is controlled by this standard maximum Doppler.
    residual_doppler_spread_hz: float = 200.0
    doppler_compensation_ratio: float = 0.999
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
        self.ntn_profile = str(self.ntn_profile or self.channel_model or "NTN-TDLA100-200").upper().replace("_", "-")
        self.channel_model = str(self.channel_model or self.ntn_profile).upper().replace("_", "-")
        self.m_subcarriers = max(1, int(self.m_subcarriers))
        self.n_symbols = max(1, int(self.n_symbols))
        self.fft_size = max(1, int(self.fft_size))
        self.cp_len = max(0, int(self.cp_len))
        self.delay_spread = max(0, int(self.delay_spread))
        self.payload_bits = max(1, int(self.payload_bits))
        self.seed = int(max(1, self.seed))
        self.sample_rate_hz = float(max(1.0, self.sample_rate_hz))
        self.demo_frame_interval_s = float(max(0.0, self.demo_frame_interval_s))
        self.subcarrier_spacing_hz = float(max(1.0, self.subcarrier_spacing_hz))
        self.residual_doppler_spread_hz = float(max(0.0, self.residual_doppler_spread_hz))
        self.doppler_compensation_ratio = float(min(1.0, max(0.0, self.doppler_compensation_ratio)))
        self.alpha = float(self.alpha)
        self.beta = float(self.beta)
        self.c1 = float(self.c1)
        self.c2 = float(self.c2)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveConfig:
    """FDIDM 双时间尺度 alpha/beta 自适应参数。

    ``interval_frames``/``window_frames`` 控制慢时标参数优化；
    ``benchmark_interval_frames`` 与显示重采样参数只控制曲线更新，
    不改变物理信道或接收机的逐帧更新速度。
    """

    enabled: bool = True
    objective: str = "ser"
    # 慢时标 CSI 窗口与优化周期。
    interval_frames: int = 16
    window_frames: int = 32
    window_stride_frames: int = 1
    ensemble_snapshots: int = 3
    # 四波形指标采样与 UI 显示节流。
    benchmark_interval_frames: int = 4
    display_interval_s: float = 0.5
    display_ema_alpha: float = 0.28
    # alpha/beta 搜索与切换判决。
    coarse_step: float = 0.5
    fine_step: float = 0.1
    alpha_min: float = 0.0
    alpha_max: float = 2.0
    beta_min: float = 0.0
    beta_max: float = 2.0
    stability_evals: int = 2
    cooldown_frames: int = 32
    max_evaluations: int = 400
    min_improvement_db: float = 0.2
    decision_ser_floor: float = 1e-8
    apply_best: bool = True
    auto_apply: bool = True
    max_order: int = 512
    history_limit: int = 1000
    seed: int = 20260428

    def normalized(self) -> "AdaptiveConfig":
        self.objective = str(self.objective or "ser").lower()
        if self.objective not in {"evm", "ber", "ser"}:
            self.objective = "ser"
        self.interval_frames = int(max(1, self.interval_frames))
        self.window_frames = int(max(2, self.window_frames))
        self.window_stride_frames = int(max(1, self.window_stride_frames))
        self.ensemble_snapshots = int(max(1, min(self.window_frames, self.ensemble_snapshots)))
        self.benchmark_interval_frames = int(max(1, self.benchmark_interval_frames))
        self.display_interval_s = float(max(0.05, self.display_interval_s))
        self.display_ema_alpha = float(min(1.0, max(0.01, self.display_ema_alpha)))
        self.coarse_step = float(min(1.0, max(0.05, self.coarse_step)))
        self.fine_step = float(min(self.coarse_step, max(0.01, self.fine_step)))
        self.alpha_min = float(self.alpha_min)
        self.alpha_max = float(max(self.alpha_min, self.alpha_max))
        self.beta_min = float(self.beta_min)
        self.beta_max = float(max(self.beta_min, self.beta_max))
        self.stability_evals = int(max(1, self.stability_evals))
        self.cooldown_frames = int(max(0, self.cooldown_frames))
        self.max_evaluations = int(max(1, self.max_evaluations))
        self.min_improvement_db = float(max(0.0, self.min_improvement_db))
        self.decision_ser_floor = float(max(1e-15, self.decision_ser_floor))
        self.apply_best = bool(self.apply_best)
        self.auto_apply = bool(self.auto_apply and self.apply_best)
        self.max_order = int(max(4, self.max_order))
        self.history_limit = int(max(32, self.history_limit))
        self.seed = int(max(1, self.seed))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_backend_kwargs(self) -> Dict[str, Any]:
        """Map the unified config to the legacy-compatible FDIDM backend API."""
        self.normalized()
        return {
            "adaptive_enabled": bool(self.enabled),
            "adaptive_auto_apply": bool(self.auto_apply),
            "adaptive_interval_frames": int(self.interval_frames),
            "adaptive_window_frames": int(self.window_frames),
            "adaptive_window_stride_frames": int(self.window_stride_frames),
            "adaptive_ensemble_snapshots": int(self.ensemble_snapshots),
            "adaptive_benchmark_interval_frames": int(self.benchmark_interval_frames),
            "adaptive_display_interval_s": float(self.display_interval_s),
            "adaptive_display_ema_alpha": float(self.display_ema_alpha),
            "adaptive_coarse_step": float(self.coarse_step),
            "adaptive_fine_step": float(self.fine_step),
            "adaptive_stability_evals": int(self.stability_evals),
            "adaptive_cooldown_frames": int(self.cooldown_frames),
            "adaptive_min_improvement_db": float(self.min_improvement_db),
            "adaptive_decision_ser_floor": float(self.decision_ser_floor),
            "adaptive_max_order": int(self.max_order),
            "adaptive_history_limit": int(self.history_limit),
        }


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
