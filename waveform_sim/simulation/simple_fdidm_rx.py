"""Redesigned FDIDM demonstration backend.

The previous demo mixed a full time-chain, a predicted SER proxy and short-window
measured SER.  This file rebuilds the link around a single, debuggable matrix
model so that every displayed number can be traced:

    bits -> QAM x -> X -> IFDIT/TF channel/FDIT equivalent H -> y = Hx+n
         -> selected detector -> hard decision -> measured BER/SER

The displayed ZF-theory SER is computed from the SAME H and SAME noise variance
used by the Monte-Carlo frames.  For ZF it should agree with measured SER once
enough symbols are accumulated.  A debug snapshot exposes all critical
intermediate values: matrix consistency, unitarity errors, noise variance,
noise-gain statistics, expected errors, measured errors and confidence hints.

This version removes the default artificial TF notch and uses a paper-aligned TDL/CDL path model with fractional delay and fractional Doppler to construct the TF-domain channel matrix directly from the rectangular-pulse input-output relation. It remains a teaching/demo simulator, not a certified 3GPP NTN link-level simulator.
"""

from __future__ import annotations

import itertools
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from waveform_sim.core.engine import LinkSimulator
from waveform_sim.simulation.fdidm_adaptive import FDIDMSimAdaptiveMixin
from waveform_sim.simulation.leo_ntn_channel import (
    absolute_doppler_shift_hz,
    coherence_time_s,
    get_profile,
    normalize_profile_name,
    normalized_doppler,
    residual_common_cfo_hz,
    sample_path_dopplers,
    weighted_mean_and_rms,
)

C_LIGHT = 299_792_458.0
DEFAULT_FC_HZ = 20e9


@dataclass(frozen=True)
class FDIDMConfig:
    alpha: float = 0.0
    beta: float = 0.0
    m_subcarriers: int = 8
    n_symbols: int = 8
    subcarrier_spacing_hz: float = 300e3
    mod_order: str = "16QAM"
    # 3GPP NR-NTN SISO conformance profile.  `channel_model` is retained as a
    # compatibility alias because older UI/config files use that name.
    channel_model: str = "NTN-TDLA100-200"
    ntn_profile: str = "NTN-TDLA100-200"
    # Unit: km/h. LEO 7.8 km/s = 28080 km/h.
    velocity_kmh: float = 28080.0
    # Projection of orbital velocity on the propagation direction.  It controls
    # the predictable large common satellite Doppler shift, not fading spread.
    doppler_radial_factor: float = 0.10
    # The common orbital shift is assumed to be predicted/pre-compensated.  The
    # remaining multipath time selectivity is represented by the standardized
    # residual maximum Doppler (200 Hz or 1200 Hz in the included profiles).
    residual_doppler_spread_hz: float = 200.0
    doppler_compensation_ratio: float = 0.999
    decoder: str = "ZF"
    ebn0_db: float = 8.0
    # "Eb/N0" keeps the old GUI semantics; "Es/N0" is closer to the SER
    # equations in the paper.
    snr_definition: str = "Eb/N0"
    optimize_indices: bool = False
    search_step: float = 0.1
    fc_hz: float = DEFAULT_FC_HZ
    # matrix is the authoritative chain for the demo. full is kept only as a
    # consistency check path and uses the same linear matrices.
    link_mode: str = "matrix"
    search_objective: str = "zf_theory_ser"
    random_channel: bool = True
    channel_seed: int = 42
    # Channel dynamics for measured/live simulation.
    # fixed: keep the original static-H chain unchanged.
    # block: change the channel realization every channel_coherence_frames frames.
    # fast: change the realization every frame and additionally vary path gains
    #       across OFDM symbols inside one FDIDM frame.
    dynamic_channel: bool = False
    channel_coherence_frames: int = 20
    channel_dynamics: str = "fixed"
    fast_channel_coherence_symbols: int = 1
    # circular channel avoids boundary energy loss, so the displayed theory and
    # measured values refer to the same stationary matrix.
    circular_channel: bool = True
    tf_notch_depth_db: float = 0.0
    tf_notch_count: int = 0
    # Wall-clock pacing for the GUI demo only.  Physical channel time is derived
    # from N/Delta-f and remains independent of this display pacing value.
    demo_frame_interval_s: float = 0.02


class _MetricTracker:
    def __init__(self, history_len: int = 300, window_len: int = 80):
        self.history_len = int(max(8, history_len))
        self.window_len = int(max(4, window_len))
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.total_bits = 0
        self.total_bit_errors = 0
        self.total_symbols = 0
        self.total_symbol_errors = 0
        self.total_frames = 0
        self.frame_errors = deque(maxlen=self.window_len)
        self.recent_bit_errors = deque(maxlen=self.window_len)
        self.recent_bit_total = deque(maxlen=self.window_len)
        self.recent_symbol_errors = deque(maxlen=self.window_len)
        self.recent_symbol_total = deque(maxlen=self.window_len)
        self.ser_t = deque(maxlen=self.history_len)
        self.ser_v = deque(maxlen=self.history_len)
        self.ber_t = deque(maxlen=self.history_len)
        self.ber_v = deque(maxlen=self.history_len)
        self.last_bit_errors = 0
        self.last_symbol_errors = 0
        self.last_total_bits = 0
        self.last_total_symbols = 0

    def update(self, bit_errors: int, total_bits: int, symbol_errors: int, total_symbols: int):
        bit_errors = int(max(0, bit_errors))
        total_bits = int(max(1, total_bits))
        symbol_errors = int(max(0, symbol_errors))
        total_symbols = int(max(1, total_symbols))

        self.total_frames += 1
        self.total_bits += total_bits
        self.total_bit_errors += bit_errors
        self.total_symbols += total_symbols
        self.total_symbol_errors += symbol_errors
        self.last_bit_errors = bit_errors
        self.last_symbol_errors = symbol_errors
        self.last_total_bits = total_bits
        self.last_total_symbols = total_symbols

        self.frame_errors.append(1 if bit_errors > 0 else 0)
        self.recent_bit_errors.append(bit_errors)
        self.recent_bit_total.append(total_bits)
        self.recent_symbol_errors.append(symbol_errors)
        self.recent_symbol_total.append(total_symbols)

        t = time.time() - self.start_time
        # Plot uses a tiny positive floor only for log display.  Panel metrics
        # expose raw zero counts and upper-confidence hints.
        plot_floor = 1e-6
        self.ser_t.append(t)
        self.ser_v.append(max(self.get_window_ser(), plot_floor))
        self.ber_t.append(t)
        self.ber_v.append(max(self.get_window_ber(), plot_floor))

    @staticmethod
    def _ratio(errors, totals) -> float:
        total = int(np.sum(totals)) if totals else 0
        if total <= 0:
            return float("nan")
        return float(int(np.sum(errors)) / total)

    def get_window_ser(self) -> float:
        return self._ratio(self.recent_symbol_errors, self.recent_symbol_total)

    def get_window_ber(self) -> float:
        return self._ratio(self.recent_bit_errors, self.recent_bit_total)

    def get_cumulative_ser(self) -> float:
        return float(self.total_symbol_errors / self.total_symbols) if self.total_symbols else float("nan")

    def get_cumulative_ber(self) -> float:
        return float(self.total_bit_errors / self.total_bits) if self.total_bits else float("nan")

    def get_fer(self) -> float:
        return float(np.mean(self.frame_errors)) if self.frame_errors else float("nan")

    def get_ser_history(self):
        return np.asarray(self.ser_t, dtype=np.float64), np.asarray(self.ser_v, dtype=np.float64)

    def get_ber_history(self):
        return np.asarray(self.ber_t, dtype=np.float64), np.asarray(self.ber_v, dtype=np.float64)

    def zero_error_upper95_ser(self) -> float:
        # rule of three: if zero errors in N trials, 95% upper bound ~= 3/N.
        total = int(np.sum(self.recent_symbol_total)) if self.recent_symbol_total else 0
        err = int(np.sum(self.recent_symbol_errors)) if self.recent_symbol_errors else 0
        if total <= 0 or err != 0:
            return float("nan")
        return float(3.0 / total)


class _LegacyFDIDMTransceiver(FDIDMSimAdaptiveMixin, threading.Thread):
    _MOD_ORDERS = {"QPSK": 4, "16QAM": 16, "64QAM": 64}

    def __init__(
        self,
        alpha: float = 0.0,
        beta: float = 0.0,
        m_subcarriers: int = 8,
        n_symbols: int = 8,
        subcarrier_spacing_hz: float = 300e3,
        mod_order: str = "16QAM",
        channel_model: str = "NTN-TDLA100-200",
        ntn_profile: Optional[str] = None,
        velocity_kmh: float = 28080.0,
        doppler_radial_factor: float = 0.10,
        residual_doppler_spread_hz: float = 200.0,
        doppler_compensation_ratio: float = 0.999,
        decoder: str = "ZF",
        snr_db: float = 8.0,
        snr_definition: str = "Eb/N0",
        optimize_indices: bool = False,
        search_step: float = 0.1,
        fc_hz: float = DEFAULT_FC_HZ,
        link_mode: str = "matrix",
        search_objective: str = "zf_theory_ser",
        random_channel: bool = True,
        channel_seed: int = 42,
        dynamic_channel: bool = False,
        channel_coherence_frames: int = 20,
        channel_dynamics: str = "fixed",
        fast_channel_coherence_symbols: int = 1,
        circular_channel: bool = True,
        tf_notch_depth_db: float = 0.0,
        tf_notch_count: int = 0,
        demo_frame_interval_s: float = 0.02,
    ):
        super().__init__(daemon=True)
        channel_dynamics = str(channel_dynamics or ("block" if dynamic_channel else "fixed")).lower()
        if channel_dynamics not in ("fixed", "block", "fast", "cont"):
            channel_dynamics = "block" if dynamic_channel else "fixed"
        if channel_dynamics == "fixed" and dynamic_channel:
            channel_dynamics = "block"
        dynamic_channel = bool(channel_dynamics in ("block", "fast", "cont"))
        profile_name = normalize_profile_name(ntn_profile or channel_model)
        profile = get_profile(profile_name)
        residual_spread = float(residual_doppler_spread_hz)
        if not np.isfinite(residual_spread) or residual_spread <= 0.0:
            residual_spread = float(profile.maximum_doppler_hz)
        self.config = FDIDMConfig(
            alpha=float(alpha),
            beta=float(beta),
            m_subcarriers=int(m_subcarriers),
            n_symbols=int(n_symbols),
            subcarrier_spacing_hz=float(subcarrier_spacing_hz),
            mod_order=str(mod_order).upper(),
            channel_model=profile_name,
            ntn_profile=profile_name,
            velocity_kmh=float(velocity_kmh),
            doppler_radial_factor=float(np.clip(float(doppler_radial_factor), 0.0, 1.0)),
            residual_doppler_spread_hz=float(max(0.0, residual_spread)),
            doppler_compensation_ratio=float(np.clip(float(doppler_compensation_ratio), 0.0, 1.0)),
            decoder=str(decoder).upper(),
            ebn0_db=float(snr_db),
            snr_definition=str(snr_definition or "Eb/N0"),
            optimize_indices=bool(optimize_indices),
            search_step=float(search_step),
            fc_hz=float(fc_hz),
            link_mode=str(link_mode).lower(),
            search_objective=str(search_objective).lower(),
            random_channel=bool(random_channel),
            channel_seed=int(channel_seed),
            dynamic_channel=bool(dynamic_channel),
            channel_coherence_frames=int(max(1, channel_coherence_frames)),
            channel_dynamics=str(channel_dynamics),
            fast_channel_coherence_symbols=int(max(1, fast_channel_coherence_symbols)),
            circular_channel=bool(circular_channel),
            tf_notch_depth_db=float(tf_notch_depth_db),
            tf_notch_count=int(tf_notch_count),
            demo_frame_interval_s=float(max(0.0, demo_frame_interval_s)),
        )
        self._lock = threading.RLock()
        self._init_adaptive_state_locked()
        self._stop_event = threading.Event()
        self._rng = np.random.default_rng()
        self._metrics = _MetricTracker(history_len=300, window_len=80)
        self._gamma_cache: Dict[Tuple[int, float], np.ndarray] = {}
        self._constellation, self._bit_patterns = self._build_gray_qam(self.qam_order)

        self._cache_key = None
        self._H_tf = None
        self._H_cross = None
        self._G_time = None
        self._A_tx = None
        self._A_rx = None
        self._tx_fdit = None
        self._rx_fdit = None
        self._detector_cache = None
        self._fade_diag = None
        self._used_alpha = float(self.config.alpha)
        self._used_beta = float(self.config.beta)
        self._last_x = np.zeros(0, dtype=np.complex128)
        self._last_y = np.zeros(0, dtype=np.complex128)
        self._last_x_est = np.zeros(0, dtype=np.complex128)
        self._last_noise = np.zeros(0, dtype=np.complex128)
        self._last_equalized_noise = np.zeros(0, dtype=np.complex128)
        self._constellation_points = np.zeros(0, dtype=np.complex64)
        self._pre_eq_points = np.zeros(0, dtype=np.complex64)
        self._impulse_grid = np.zeros((self.M, self.N), dtype=np.complex64)
        self._impulse_peak_energy_ratio = float("nan")
        self._impulse_spread_cells_20db = 0
        self._impulse_max_sidelobe_db = float("nan")
        self._score = {}
        self._debug = {}
        self._last_metrics = self._empty_metrics()
        # Physical-channel evolution indicators used by the live status panel.
        # They are computed from H_TF, so an alpha/beta change is not mistaken for
        # a physical channel change.
        self._previous_h_tf_for_state = None
        self._channel_matrix_change_norm = float("nan")
        self._channel_matrix_correlation = float("nan")
        self._channel_power_db = float("nan")
        self._dynamic_base_seed = int(self.config.channel_seed)
        self._last_dynamic_block = None
        # 单调递增的仿真帧号：只随 _simulate_one_frame 增长，不随参数更新/BER 统计重置。
        # 信道块索引与自适应评估都以它为时间基准，避免"应用 α/β 后信道倒退"。
        self._sim_frame = 0
        # 连续多普勒模式（channel_dynamics="cont"）的持久化路径表与逐帧相位状态。
        self._cont_paths = None
        self._cont_phase = None
        self._cont_paths_key = None
        # 块衰落模式（channel_dynamics="block"）的持久化路径表与逐块相位随机游走状态。
        self._block_paths = None
        self._block_phase = None
        self._block_paths_key = None
        self._block_phase_block = -1

    # ----------------------------- properties -----------------------------
    @property
    def M(self) -> int:
        return int(self.config.m_subcarriers)

    @property
    def N(self) -> int:
        return int(self.config.n_symbols)

    @property
    def K(self) -> int:
        return int(self.M * self.N)

    @property
    def qam_order(self) -> int:
        return int(self._MOD_ORDERS.get(self.config.mod_order, 16))

    @property
    def bits_per_symbol(self) -> int:
        return int(round(math.log2(self.qam_order)))

    @property
    def sample_rate(self) -> float:
        return float(self.M * self.config.subcarrier_spacing_hz)

    # ----------------------------- thread API -----------------------------
    def stop(self):
        self._stop_event.set()

    def wait(self, timeout: Optional[float] = None):
        self.join(timeout=timeout)

    def run(self):
        while not self._stop_event.is_set():
            self._simulate_one_frame()
            # This is GUI pacing, not physical frame duration.  Curves are
            # separately resampled, therefore a faster simulation no longer
            # floods the time plot with points.
            delay = float(max(0.0, getattr(self.config, "demo_frame_interval_s", 0.02)))
            if delay > 0.0:
                time.sleep(delay)

    def step(self):
        """单帧仿真（供统一引擎调用）。"""
        self._simulate_one_frame()

    # ----------------------------- config API -----------------------------
    def _apply_config_update(self, **kwargs):
        old = self.config
        updates = dict(kwargs)
        # Normalize public/legacy aliases before constructing the frozen dataclass.
        # Unknown fields are intentionally ignored so LinkSimulator callers can
        # share one update dictionary across different waveform backends.
        aliases = {
            "snr_db": "ebn0_db",
            "modulation": "mod_order",
            "detector": "decoder",
            "center_freq_hz": "fc_hz",
            "seed": "channel_seed",
            "residual_doppler_hz": "residual_doppler_spread_hz",
            "doppler_spread_hz": "residual_doppler_spread_hz",
            "max_doppler_hz": "residual_doppler_spread_hz",
        }
        for source, target in aliases.items():
            if source in updates and target not in updates:
                updates[target] = updates[source]
        profile_was_explicit = "ntn_profile" in updates or "channel_model" in updates
        if "ntn_profile" not in updates and "channel_model" in updates:
            updates["ntn_profile"] = updates["channel_model"]
        if "channel_model" not in updates and "ntn_profile" in updates:
            updates["channel_model"] = updates["ntn_profile"]
        allowed = set(FDIDMConfig.__dataclass_fields__)
        updates = {key: value for key, value in updates.items() if key in allowed}
        data = old.__dict__.copy()
        data.update(updates)
        data["alpha"] = float(np.clip(float(data.get("alpha", old.alpha)), 0.0, 2.0))
        data["beta"] = float(np.clip(float(data.get("beta", old.beta)), 0.0, 2.0))
        data["m_subcarriers"] = int(max(2, int(data.get("m_subcarriers", old.m_subcarriers))))
        data["n_symbols"] = int(max(2, int(data.get("n_symbols", old.n_symbols))))
        data["subcarrier_spacing_hz"] = float(max(1.0, float(data.get("subcarrier_spacing_hz", old.subcarrier_spacing_hz))))
        data["mod_order"] = str(data.get("mod_order", old.mod_order)).upper()
        if data["mod_order"] not in self._MOD_ORDERS:
            data["mod_order"] = "16QAM"
        profile_source = (
            updates.get("ntn_profile")
            if "ntn_profile" in updates
            else updates.get("channel_model", getattr(old, "ntn_profile", old.channel_model))
        )
        profile_name = normalize_profile_name(profile_source)
        data["channel_model"] = profile_name
        data["ntn_profile"] = profile_name
        if profile_was_explicit and "residual_doppler_spread_hz" not in updates:
            data["residual_doppler_spread_hz"] = float(
                get_profile(profile_name).maximum_doppler_hz
            )
        data["velocity_kmh"] = float(data.get("velocity_kmh", old.velocity_kmh))
        data["doppler_radial_factor"] = float(np.clip(float(data.get("doppler_radial_factor", getattr(old, "doppler_radial_factor", 0.10))), 0.0, 1.0))
        spread = float(data.get("residual_doppler_spread_hz", getattr(old, "residual_doppler_spread_hz", get_profile(profile_name).maximum_doppler_hz)))
        if not np.isfinite(spread) or spread <= 0.0:
            spread = float(get_profile(profile_name).maximum_doppler_hz)
        data["residual_doppler_spread_hz"] = float(max(0.0, spread))
        data["doppler_compensation_ratio"] = float(np.clip(float(data.get("doppler_compensation_ratio", getattr(old, "doppler_compensation_ratio", 0.999))), 0.0, 1.0))
        data.pop("max_doppler_hz", None)
        data.pop("doppler_spread_hz", None)
        data.pop("residual_doppler_hz", None)
        data["decoder"] = str(data.get("decoder", old.decoder)).upper()
        data["ebn0_db"] = float(data.get("ebn0_db", old.ebn0_db))
        snr_def = str(data.get("snr_definition", getattr(old, "snr_definition", "Eb/N0")) or "Eb/N0").strip().upper().replace(" ", "")
        data["snr_definition"] = "Es/N0" if snr_def in ("ES/N0", "ESN0", "SNR") else "Eb/N0"
        data["optimize_indices"] = bool(data.get("optimize_indices", old.optimize_indices))
        data["search_step"] = float(data.get("search_step", old.search_step) or 0.1)
        data["fc_hz"] = float(data.get("fc_hz", old.fc_hz))
        data["link_mode"] = str(data.get("link_mode", old.link_mode)).lower()
        data["search_objective"] = str(data.get("search_objective", old.search_objective)).lower()
        data["random_channel"] = bool(data.get("random_channel", old.random_channel))
        data["channel_seed"] = int(data.get("channel_seed", old.channel_seed))
        data["dynamic_channel"] = bool(data.get("dynamic_channel", old.dynamic_channel))
        data["channel_coherence_frames"] = int(max(1, int(data.get("channel_coherence_frames", old.channel_coherence_frames))))
        mode = str(data.get("channel_dynamics", old.channel_dynamics) or "fixed").lower()
        if mode not in ("fixed", "block", "fast", "cont"):
            mode = "block" if data["dynamic_channel"] else "fixed"
        if mode == "fixed" and data["dynamic_channel"]:
            mode = "block"
        data["dynamic_channel"] = bool(mode in ("block", "fast", "cont"))
        data["channel_dynamics"] = mode
        data["fast_channel_coherence_symbols"] = int(max(1, int(data.get("fast_channel_coherence_symbols", old.fast_channel_coherence_symbols))))
        data["circular_channel"] = bool(data.get("circular_channel", old.circular_channel))
        data["tf_notch_depth_db"] = float(data.get("tf_notch_depth_db", old.tf_notch_depth_db))
        data["tf_notch_count"] = int(max(0, int(data.get("tf_notch_count", old.tf_notch_count))))
        data["demo_frame_interval_s"] = float(max(0.0, data.get("demo_frame_interval_s", getattr(old, "demo_frame_interval_s", 0.02))))
        if data["link_mode"] not in ("matrix", "full"):
            data["link_mode"] = "matrix"

        new = FDIDMConfig(**data)
        shape_changed = new.m_subcarriers != old.m_subcarriers or new.n_symbols != old.n_symbols
        mod_changed = new.mod_order != old.mod_order
        index_changed = (
            abs(new.alpha - old.alpha) > 1e-12
            or abs(new.beta - old.beta) > 1e-12
        )
        matrix_changed = (
            shape_changed
            or mod_changed
            or new.channel_model != old.channel_model
            or new.ntn_profile != getattr(old, "ntn_profile", old.channel_model)
            or abs(new.velocity_kmh - old.velocity_kmh) > 1e-12
            or abs(getattr(new, "doppler_radial_factor", 0.10) - getattr(old, "doppler_radial_factor", 0.10)) > 1e-12
            or abs(getattr(new, "residual_doppler_spread_hz", 200.0) - getattr(old, "residual_doppler_spread_hz", 200.0)) > 1e-12
            or abs(getattr(new, "doppler_compensation_ratio", 0.999) - getattr(old, "doppler_compensation_ratio", 0.999)) > 1e-12
            or abs(new.subcarrier_spacing_hz - old.subcarrier_spacing_hz) > 1e-9
            or abs(new.fc_hz - old.fc_hz) > 1e-3
            or new.random_channel != old.random_channel
            or new.channel_seed != old.channel_seed
            or new.dynamic_channel != old.dynamic_channel
            or new.channel_coherence_frames != old.channel_coherence_frames
            or new.channel_dynamics != old.channel_dynamics
            or new.fast_channel_coherence_symbols != old.fast_channel_coherence_symbols
            or new.circular_channel != old.circular_channel
            or abs(new.tf_notch_depth_db - old.tf_notch_depth_db) > 1e-12
            or new.tf_notch_count != old.tf_notch_count
            or abs(new.alpha - old.alpha) > 1e-12
            or abs(new.beta - old.beta) > 1e-12
            or new.optimize_indices != old.optimize_indices
            or abs(new.search_step - old.search_step) > 1e-12
        )
        self.config = new
        if index_changed and hasattr(self, "_adaptive_lock"):
            # A queued slow optimizer may have used the previous applied pair as
            # its baseline.  Advance the epoch and discard only stale decisions;
            # physical CSI snapshots remain valid and are retained.
            with self._adaptive_lock:
                self._adaptive_note_index_change_locked()
        if new.channel_seed != old.channel_seed:
            self._dynamic_base_seed = int(new.channel_seed)
            self._last_dynamic_block = None
        # 时变模式/相干参数变化时强制重新进入当前块。旧代码在 reset_ber_stats 里无条件
        # 重置 _last_dynamic_block，导致每次应用 α/β 都把信道重播种回第一个块。
        channel_ctx_changed = (
            new.random_channel != old.random_channel
            or new.dynamic_channel != old.dynamic_channel
            or new.channel_dynamics != old.channel_dynamics
            or new.channel_coherence_frames != old.channel_coherence_frames
            or new.fast_channel_coherence_symbols != old.fast_channel_coherence_symbols
            or new.ntn_profile != getattr(old, "ntn_profile", old.channel_model)
            or abs(getattr(new, "residual_doppler_spread_hz", 200.0) - getattr(old, "residual_doppler_spread_hz", 200.0)) > 1e-12
            or abs(getattr(new, "doppler_compensation_ratio", 0.999) - getattr(old, "doppler_compensation_ratio", 0.999)) > 1e-12
        )
        if channel_ctx_changed:
            self._last_dynamic_block = None
            self._previous_h_tf_for_state = None
            self._channel_matrix_change_norm = float("nan")
            self._channel_matrix_correlation = float("nan")
        if shape_changed:
            self._gamma_cache.clear()
            self._impulse_grid = np.zeros((self.M, self.N), dtype=np.complex64)
        if mod_changed:
            self._constellation, self._bit_patterns = self._build_gray_qam(self.qam_order)
        if matrix_changed:
            self._cache_key = None
            self._H_tf = None
            self._H_cross = None
            self._G_time = None
            self._detector_cache = None
        self.reset_ber_stats()
        self._last_metrics = self._empty_metrics()

    def set_snr_db(self, value: float):
        with self._lock:
            self._apply_config_update(ebn0_db=float(value))

    def set_indices(self, alpha: float, beta: float):
        with self._lock:
            self._apply_config_update(alpha=float(alpha), beta=float(beta), optimize_indices=False)

    def set_index_search_enabled(self, enabled: bool, step: float = 0.1):
        with self._lock:
            self._apply_config_update(optimize_indices=bool(enabled), search_step=float(step))

    def set_frame_shape(self, m_subcarriers: int, n_symbols: int):
        with self._lock:
            self._apply_config_update(m_subcarriers=int(m_subcarriers), n_symbols=int(n_symbols))

    def set_subcarrier_spacing_hz(self, value_hz: float):
        with self._lock:
            self._apply_config_update(subcarrier_spacing_hz=float(value_hz))

    def set_mod_order(self, mod_order: str):
        with self._lock:
            self._apply_config_update(mod_order=str(mod_order))

    def set_channel_model(self, channel_model: str):
        with self._lock:
            self._apply_config_update(channel_model=str(channel_model))

    def set_velocity_kmh(self, velocity_kmh: float):
        with self._lock:
            self._apply_config_update(velocity_kmh=float(velocity_kmh))

    def set_decoder(self, decoder: str):
        with self._lock:
            self._apply_config_update(decoder=str(decoder))

    def set_link_mode(self, mode: str):
        with self._lock:
            self._apply_config_update(link_mode=str(mode))

    def set_channel_randomization(self, enabled: bool, seed: int):
        with self._lock:
            self._apply_config_update(random_channel=bool(enabled), channel_seed=int(seed))

    def regenerate_channel(self, seed: Optional[int] = None):
        if seed is None:
            seed = int(time.time() * 1000) % 2_147_483_647
        with self._lock:
            self._apply_config_update(random_channel=True, channel_seed=int(seed))
        return int(seed)


    def update_runtime_parameters(self, **kwargs):
        with self._lock:
            self._apply_config_update(**kwargs)

    def reset_ber_stats(self):
        # 只重置 BER/SER 统计，不重置信道推进状态（_last_dynamic_block/_sim_frame），
        # 否则每次应用 α/β 都会让块索引回退、信道倒退到第一个实现。
        self._metrics.reset()

    @staticmethod
    def _valid_channel_seed(seed: int) -> int:
        seed = int(seed) % 2_147_483_647
        return seed if seed > 0 else 1

    def _dynamic_channel_seed_for_block(self, block_index: int) -> int:
        # Reproducible but decorrelated seed sequence for channel blocks/frames.
        return self._valid_channel_seed(int(self._dynamic_base_seed) + 104729 * int(block_index))

    def _channel_dynamics_mode_locked(self) -> str:
        cfg = self.config
        mode = str(getattr(cfg, "channel_dynamics", "block" if getattr(cfg, "dynamic_channel", False) else "fixed") or "fixed").lower()
        if mode not in ("fixed", "block", "fast", "cont"):
            mode = "block" if getattr(cfg, "dynamic_channel", False) else "fixed"
        if not getattr(cfg, "dynamic_channel", False):
            mode = "fixed"
        return mode

    def _maybe_update_dynamic_channel_locked(self):
        """Advance H only when the user enables a dynamic channel.

        fixed mode returns immediately and therefore preserves the original
        static-channel chain.  In block mode one H is reused for a configurable
        number of frames and adjacent blocks evolve by a small deterministic
        phase walk (correlated, SER changes smoothly).  In fast mode a fresh H
        is built for every frame, and _build_paper_tf_channel_matrix() also
        varies path gains across the N OFDM symbols inside that frame.
        """
        cfg = self.config
        mode = self._channel_dynamics_mode_locked()
        if mode == "fixed":
            return
        # 基于单调帧号的当前帧索引（0-based），不再受 BER 统计重置影响。
        frame_idx = max(0, int(getattr(self, "_sim_frame", 0)) - 1)
        if mode == "cont":
            # 连续多普勒：路径表固定，每帧按物理多普勒推进相位，信道连续演化。
            self._advance_cont_channel_locked()
            self._cache_key = None
            self._H_tf = None
            self._H_cross = None
            self._G_time = None
            self._detector_cache = None
            self._last_dynamic_block = ("cont", int(frame_idx))
            return
        if mode == "fast":
            block = int(frame_idx)
            seed_offset = 1_000_003
        else:
            coherence = int(max(1, getattr(cfg, "channel_coherence_frames", 1)))
            block = int(frame_idx // coherence)
            seed_offset = 0
        if self._last_dynamic_block == (mode, block):
            return
        if mode == "block":
            # 平滑块演化：只游走相位，不重新随机整条路径表。
            self._advance_block_phase_locked(block)
        new_seed = self._valid_channel_seed(self._dynamic_channel_seed_for_block(block) + seed_offset)
        data = cfg.__dict__.copy()
        data["channel_seed"] = int(new_seed)
        data["random_channel"] = True
        data["dynamic_channel"] = True
        data["channel_dynamics"] = mode
        self.config = FDIDMConfig(**data)
        self._cache_key = None
        self._H_tf = None
        self._H_cross = None
        self._G_time = None
        self._detector_cache = None
        self._last_dynamic_block = (mode, block)

    def calibrate_ebn0_for_target_ser(self, target_ser: float = 2e-2) -> float:
        """Choose Eb/N0 so OFDM-point ZF-theory SER is near target_ser."""
        target_ser = float(np.clip(target_ser, 1e-4, 0.3))
        with self._lock:
            old_alpha, old_beta = self.config.alpha, self.config.beta
            old_opt = self.config.optimize_indices
            self._apply_config_update(alpha=0.0, beta=0.0, optimize_indices=False)
            self._prepare_matrices_locked()
            H = self._H_cross.copy()
        # Binary search over Eb/N0.  SER decreases monotonically with Eb/N0.
        lo, hi = -6.0, 20.0
        best = self.config.ebn0_db
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            ser, _ = self._zf_theory_ser_for_channel(H, ebn0_db=mid)
            best = mid
            if ser > target_ser:
                lo = mid
            else:
                hi = mid
        with self._lock:
            self._apply_config_update(ebn0_db=best, alpha=old_alpha, beta=old_beta, optimize_indices=old_opt)
        return float(best)

    # ----------------------------- accessors -----------------------------
    def get_constellation(self):
        with self._lock:
            return self._constellation_points.copy()

    def get_pre_eq_constellation(self):
        with self._lock:
            return self._pre_eq_points.copy()

    def get_cross_domain_impulse_response(self):
        with self._lock:
            return self._impulse_grid.copy()

    def get_ser_history(self):
        with self._lock:
            return self._metrics.get_ser_history()

    def get_ber_history(self):
        with self._lock:
            return self._metrics.get_ber_history()

    def get_last_metrics(self):
        with self._lock:
            return dict(self._last_metrics)

    def get_debug_snapshot(self):
        with self._lock:
            self._prepare_matrices_locked()
            d = dict(self._debug)
            d.update({
                "last_metrics": dict(self._last_metrics),
                "score": dict(self._score),
                "config": self.config.__dict__.copy(),
            })
            return d

    # ----------------------------- core math -----------------------------
    @staticmethod
    def _unitary_dft(n: int) -> np.ndarray:
        k = np.arange(n).reshape(-1, 1)
        m = np.arange(n).reshape(1, -1)
        return np.exp(-1j * 2.0 * np.pi * k * m / n) / np.sqrt(n)

    def _gamma(self, n: int, eps: float) -> np.ndarray:
        eps = float(np.round(eps, 12))
        key = (int(n), eps)
        if key in self._gamma_cache:
            return self._gamma_cache[key]
        F = self._unitary_dft(n)
        G = np.zeros((n, n), dtype=np.complex128)
        Fp = np.eye(n, dtype=np.complex128)
        for p in range(4):
            if p == 0:
                Fp = np.eye(n, dtype=np.complex128)
            elif p == 1:
                Fp = F
            else:
                Fp = Fp @ F
            a = (
                np.cos((eps - p) * np.pi / 4.0)
                * np.cos(2.0 * (eps - p) * np.pi / 4.0)
                * np.exp(1j * 3.0 * (eps - p) * np.pi / 4.0)
            )
            G += a * Fp
        self._gamma_cache[key] = G
        return G

    def _fdit_matrices(self, alpha: float, beta: float):
        # vec(Gamma_M(alpha) X Gamma_N(-beta)) = (Gamma_N(-beta)^T ⊗ Gamma_M(alpha)) vec(X).
        # DFT powers are symmetric, so transpose equals itself for this construction.
        tx = np.kron(self._gamma(self.N, -beta), self._gamma(self.M, alpha))
        rx = np.kron(self._gamma(self.N, beta), self._gamma(self.M, -alpha))
        return tx, rx

    def _build_tf_transform_matrices(self):
        Fm = self._unitary_dft(self.M)
        A_tx = np.kron(np.eye(self.N, dtype=np.complex128), Fm.conj().T)
        A_rx = np.kron(np.eye(self.N, dtype=np.complex128), Fm)
        return A_tx, A_rx

    def _channel_profile(self):
        """Return the selected standardized NR-NTN delay/power profile."""
        profile = get_profile(getattr(self.config, "ntn_profile", self.config.channel_model))
        delays_ns = np.asarray(profile.delays_ns, dtype=np.float64)
        gains_db = np.asarray(profile.powers_db, dtype=np.float64)
        return delays_ns * 1e-9, gains_db

    def _absolute_doppler_shift_hz(self) -> float:
        return absolute_doppler_shift_hz(
            carrier_frequency_hz=float(self.config.fc_hz),
            velocity_kmh=float(self.config.velocity_kmh),
            radial_projection=float(getattr(self.config, "doppler_radial_factor", 0.10)),
        )

    def _velocity_doppler_hz(self) -> float:
        """Backward-compatible name for the predictable common orbital shift."""
        return self._absolute_doppler_shift_hz()

    def _residual_common_doppler_hz(self) -> float:
        return residual_common_cfo_hz(
            self._absolute_doppler_shift_hz(),
            float(getattr(self.config, "doppler_compensation_ratio", 0.999)),
        )

    def _residual_doppler_spread_hz(self) -> float:
        configured = float(getattr(self.config, "residual_doppler_spread_hz", 0.0))
        if np.isfinite(configured) and configured > 0.0:
            return configured
        profile = get_profile(getattr(self.config, "ntn_profile", self.config.channel_model))
        return float(profile.maximum_doppler_hz)

    def _channel_max_residual_doppler_hz(self) -> float:
        return float(abs(self._residual_common_doppler_hz()) + self._residual_doppler_spread_hz())

    def _fractional_circular_delay(self, delay_samples: float) -> np.ndarray:
        K = self.K
        F = self._unitary_dft(K)
        k = np.arange(K, dtype=np.float64)
        phase = np.exp(-1j * 2.0 * np.pi * k * float(delay_samples) / K)
        return F.conj().T @ (phase[:, None] * F)

    def _integer_linear_delay(self, delay_samples: float) -> np.ndarray:
        K = self.K
        d0 = int(np.floor(delay_samples))
        frac = float(delay_samples - d0)
        D = np.zeros((K, K), dtype=np.complex128)
        for q in range(K):
            src0 = q - d0
            src1 = q - d0 - 1
            if 0 <= src0 < K:
                D[q, src0] += 1.0 - frac
            if frac > 1e-12 and 0 <= src1 < K:
                D[q, src1] += frac
        return D

    def _build_time_channel_matrix(self):
        K = self.K
        fs = self.sample_rate
        delays_s, gains_db = self._channel_profile()
        gains_mag = 10.0 ** (gains_db / 20.0)
        rng = np.random.default_rng(int(self.config.channel_seed) if self.config.random_channel else 12345)
        phases = np.exp(1j * 2.0 * np.pi * rng.random(len(gains_mag))) if self.config.random_channel else np.ones(len(gains_mag), dtype=np.complex128)
        gains = gains_mag * phases
        gains /= np.sqrt(np.sum(np.abs(gains) ** 2) + 1e-15)
        fd_list = sample_path_dopplers(
            len(gains),
            maximum_spread_hz=self._residual_doppler_spread_hz(),
            common_residual_hz=self._residual_common_doppler_hz(),
            rng=rng,
            deterministic=not bool(self.config.random_channel),
        )
        t = np.arange(K, dtype=np.float64) / max(fs, 1e-12)
        G = np.zeros((K, K), dtype=np.complex128)
        path_table = []
        for tau, h, fd in zip(delays_s, gains, fd_list):
            delay_samples = float(tau * fs)
            D = self._fractional_circular_delay(delay_samples) if self.config.circular_channel else self._integer_linear_delay(delay_samples)
            Dop = np.diag(np.exp(1j * 2.0 * np.pi * float(fd) * (t - float(tau))))
            G += h * (Dop @ D)
            path_table.append((float(tau * 1e9), float(delay_samples), float(fd), float(np.abs(h))))
        return G, path_table

    @staticmethod
    def _rect_cross_ambiguity(freq_hz: float, time_shift_s: float, symbol_T: float) -> complex:
        """Cross-ambiguity of normalized rectangular Tx/Rx pulses.

        g(t)=1/sqrt(T) on [0,T].  The expression follows the Wigner/Heisenberg
        input-output relation used in the paper.  It keeps fractional delays and
        Dopplers; no rounding to delay or Doppler bins is performed.
        """
        T = float(symbol_T)
        t = float(time_shift_s)
        f = float(freq_hz)
        a = max(0.0, t)
        b = min(T, T + t)
        L = b - a
        if L <= 0.0:
            return 0.0 + 0.0j
        if abs(f) < 1e-12:
            return complex(L / T)
        return np.exp(1j * 2.0 * np.pi * f * t) * (
            np.exp(-1j * 2.0 * np.pi * f * b) - np.exp(-1j * 2.0 * np.pi * f * a)
        ) / (-1j * 2.0 * np.pi * f * T)

    def _sample_path_parameters(self, seed=None, random_phases=None, random_angles=None):
        """Draw one TDL/CDL path table: delays, complex gains, per-path Doppler.

        seed=None 使用当前 config.channel_seed，保持 block/fast 模式既有的逐实现
        可复现行为；连续多普勒模式显式传入独立种子以复用同一条路径表。
        """
        cfg = self.config
        if seed is None:
            seed = int(cfg.channel_seed)
        if random_phases is None:
            random_phases = bool(cfg.random_channel)
        if random_angles is None:
            random_angles = bool(cfg.random_channel)
        delays_s, gains_db = self._channel_profile()
        gains_mag = 10.0 ** (gains_db / 20.0)
        rng = np.random.default_rng(int(seed))
        if random_phases:
            phases = np.exp(1j * 2.0 * np.pi * rng.random(len(gains_mag)))
        else:
            phases = np.ones(len(gains_mag), dtype=np.complex128)
        gains = gains_mag * phases
        gains /= np.sqrt(np.sum(np.abs(gains) ** 2) + 1e-15)
        fd_list = sample_path_dopplers(
            len(gains),
            maximum_spread_hz=self._residual_doppler_spread_hz(),
            common_residual_hz=self._residual_common_doppler_hz(),
            rng=rng,
            deterministic=not bool(random_angles),
        )
        return delays_s, gains, fd_list

    def _cont_scenario_key_locked(self) -> tuple:
        """场景键：这些参数变化时重建连续多普勒模式的持久化路径表。"""
        cfg = self.config
        return (
            str(cfg.channel_model),
            str(getattr(cfg, "ntn_profile", cfg.channel_model)),
            float(cfg.velocity_kmh),
            float(getattr(cfg, "doppler_radial_factor", 0.10)),
            float(cfg.fc_hz),
            float(getattr(cfg, "residual_doppler_spread_hz", 200.0)),
            float(getattr(cfg, "doppler_compensation_ratio", 0.999)),
            float(cfg.subcarrier_spacing_hz),
            int(self._dynamic_base_seed),
        )

    def _ensure_cont_paths_locked(self):
        """惰性建立连续多普勒模式的固定路径表（同一场景只随机一次）。"""
        key = self._cont_scenario_key_locked()
        if self._cont_paths is not None and self._cont_paths_key == key:
            return
        # 独立于块模式的种子序列，保证同一场景可复现。
        seed = self._valid_channel_seed(int(self._dynamic_base_seed) + 777)
        delays_s, gains, dopplers_hz = self._sample_path_parameters(
            seed=int(seed), random_phases=True, random_angles=True
        )
        self._cont_paths = (delays_s, gains, dopplers_hz)
        self._cont_phase = np.ones(len(gains), dtype=np.complex128)
        self._cont_paths_key = key

    def _advance_cont_channel_locked(self):
        """每帧推进各径相位：exp(j2π fD T_frame)，T_frame = N/Δf。

        这里只推进补偿后残余 CFO 与多径 Doppler spread 对应的相位；可预测的
        大尺度轨道 Doppler 不直接作为随机快衰落注入信道。
        """
        self._ensure_cont_paths_locked()
        cfg = self.config
        T_frame = float(self.N) / max(float(cfg.subcarrier_spacing_hz), 1e-15)
        dopplers_hz = np.asarray(self._cont_paths[2], dtype=np.float64)
        step = np.exp(1j * 2.0 * np.pi * dopplers_hz * T_frame)
        self._cont_phase = np.asarray(self._cont_phase, dtype=np.complex128) * step

    def _ensure_block_paths_locked(self):
        """惰性建立块衰落模式的固定路径表（同一场景只随机一次）。"""
        key = self._cont_scenario_key_locked()
        if self._block_paths is not None and self._block_paths_key == key:
            return
        seed = self._valid_channel_seed(int(self._dynamic_base_seed) + 555)
        delays_s, gains, dopplers_hz = self._sample_path_parameters(
            seed=int(seed), random_phases=True, random_angles=True
        )
        self._block_paths = (delays_s, gains, dopplers_hz)
        # 初始相位取随机值（而非全 1），避免第一块到第二块的过渡出现异常大跳变。
        rng0 = np.random.default_rng(self._valid_channel_seed(int(seed) + 1))
        self._block_angles = 2.0 * np.pi * rng0.random(len(gains))
        self._block_phase = np.exp(1j * self._block_angles).astype(np.complex128)
        self._block_phase_block = -1
        self._block_paths_key = key

    def _advance_block_phase_locked(self, block):
        """块衰落：每个块对路径相位做一次小幅、确定性的随机游走。

        旧实现用独立种子 per-block 重新随机路径，相邻块信道完全不相关，导致
        SER 逐块大幅跳变。这里固定路径表（时延/幅度/多普勒），相位用均值回归的
        AR(1) 过程平滑演化（rho<1，相位在固定带内漂移而非无限随机游走），
        相邻块信道保持相关，SER 随时间平稳变化；游走步长由 (base_seed, block)
        决定，可复现。
        """
        self._ensure_block_paths_locked()
        n = len(self._block_paths[1])
        block = int(max(0, block))
        if self._block_phase is None or self._block_phase.size != n:
            rng0 = np.random.default_rng(self._valid_channel_seed(int(self._dynamic_base_seed) + 555 + 1))
            self._block_angles = 2.0 * np.pi * rng0.random(n)
            self._block_phase = np.exp(1j * self._block_angles).astype(np.complex128)
            self._block_phase_block = -1
        if block < self._block_phase_block:
            # 参数变化导致块索引回退时，重建到当前块，保证确定性。
            rng0 = np.random.default_rng(self._valid_channel_seed(int(self._dynamic_base_seed) + 555 + 1))
            self._block_angles = 2.0 * np.pi * rng0.random(n)
            self._block_phase = np.exp(1j * self._block_angles).astype(np.complex128)
            start = 0
        else:
            start = int(self._block_phase_block) + 1
        step_sigma = 0.25
        rho = 0.8
        for b in range(start, block + 1):
            rng = np.random.default_rng(
                self._valid_channel_seed(int(self._dynamic_base_seed) + 104729 * b + 987654)
            )
            innovation = rng.normal(0.0, step_sigma, n)
            self._block_angles = rho * np.asarray(self._block_angles, dtype=np.float64) + innovation
            self._block_phase = np.exp(1j * self._block_angles).astype(np.complex128)
        self._block_phase_block = int(block)

    def _generate_path_parameters(self):
        mode = self._channel_dynamics_mode_locked()
        if mode == "cont":
            # 连续多普勒：路径表固定，只随时间旋转相位（幅度/时延/多普勒不变）。
            self._ensure_cont_paths_locked()
            delays_s, gains, dopplers_hz = self._cont_paths
            gains = np.asarray(gains, dtype=np.complex128) * np.asarray(self._cont_phase, dtype=np.complex128)
            return delays_s, gains, dopplers_hz
        if mode == "block":
            # 块衰落：路径表固定，每个块只小幅随机游走相位，相邻块信道相关。
            frame_idx = max(0, int(getattr(self, "_sim_frame", 1)) - 1)
            coherence = int(max(1, int(getattr(self.config, "channel_coherence_frames", 1))))
            self._advance_block_phase_locked(frame_idx // coherence)
            delays_s, gains, dopplers_hz = self._block_paths
            gains = np.asarray(gains, dtype=np.complex128) * np.asarray(self._block_phase, dtype=np.complex128)
            return delays_s, gains, dopplers_hz
        return self._sample_path_parameters(
            seed=int(self.config.channel_seed),
            random_phases=bool(self.config.random_channel),
            random_angles=bool(self.config.random_channel),
        )

    def _build_fast_time_gain_profile(self, num_paths: int, n_symbols: int) -> np.ndarray:
        """Per-path gain evolution across OFDM symbols for fast-fading mode.

        This intentionally changes H inside one FDIDM frame.  The profile is an
        AR(1) complex process along the symbol index n; smaller coherence means
        stronger frame-internal variation.  Each path is RMS-normalized so the
        change primarily reflects time selectivity rather than a hidden average
        power offset.
        """
        mode = self._channel_dynamics_mode_locked()
        if mode != "fast":
            return np.ones((int(num_paths), int(n_symbols)), dtype=np.complex128)
        n_symbols = int(max(1, n_symbols))
        num_paths = int(max(1, num_paths))
        coh = float(max(1, getattr(self.config, "fast_channel_coherence_symbols", 1)))
        rho = float(np.exp(-1.0 / coh))
        innovation_scale = math.sqrt(max(0.0, 1.0 - rho * rho))
        rng = np.random.default_rng(self._valid_channel_seed(int(self.config.channel_seed) + 314159))
        profile = np.ones((num_paths, n_symbols), dtype=np.complex128)
        for pidx in range(num_paths):
            h = np.exp(1j * 2.0 * np.pi * rng.random())
            for n in range(n_symbols):
                innovation = (rng.standard_normal() + 1j * rng.standard_normal()) / math.sqrt(2.0)
                h = rho * h + innovation_scale * innovation
                profile[pidx, n] = h
            rms = math.sqrt(float(np.mean(np.abs(profile[pidx]) ** 2)) + 1e-15)
            profile[pidx] /= rms
        return profile

    def _build_paper_tf_channel_matrix(self):
        """Build H_TF from the paper's rectangular-pulse TF input-output relation.

        The matrix entry maps x_TF[n',m'] to y_TF[n,m] under a TDL/CDL-like
        SATCOM path profile with fractional delay and Doppler.  This replaces
        the earlier demo-only TF notch dominated model.
        """
        M, N = self.M, self.N
        df = float(self.config.subcarrier_spacing_hz)
        T = 1.0 / max(df, 1e-15)
        K = M * N
        delays_s, gains, dopplers_hz = self._generate_path_parameters()
        fast_gain_profile = self._build_fast_time_gain_profile(len(gains), N)
        H = np.zeros((K, K), dtype=np.complex128)
        path_table = []
        for pidx, (tau, gp, nu) in enumerate(zip(delays_s, gains, dopplers_hz)):
            tau = float(tau)
            nu = float(nu)
            # Only n'=n and n'=n-1 normally overlap for positive tau<T, but the
            # loop is kept general and clear for debugging.
            for n in range(N):
                row_base = n * M
                for np_ in range(N):
                    time_shift = (n - np_) * T - tau
                    if abs(time_shift) >= T:
                        continue
                    col_base = np_ * M
                    doppler_phase = np.exp(1j * 2.0 * np.pi * nu * np_ * T)
                    gp_eff = gp * fast_gain_profile[int(pidx), int(n)]
                    for m in range(M):
                        row = row_base + m
                        m_freq = m * df
                        for mp in range(M):
                            freq_shift = (m - mp) * df - nu
                            amb = self._rect_cross_ambiguity(freq_shift, time_shift, T)
                            if abs(amb) <= 1e-15:
                                continue
                            xi = (mp * df + nu) * time_shift
                            H[row, col_base + mp] += gp_eff * amb * np.exp(1j * 2.0 * np.pi * xi) * doppler_phase
            path_table.append((
                int(pidx),
                float(tau * 1e9),
                float(tau * self.sample_rate),
                float(nu),
                float(nu * N * T),
                float(np.abs(gp)),
            ))
        return H, path_table

    def _build_tf_fade_diagonal(self):
        K = self.K
        gains = np.ones(K, dtype=np.float64)
        count = int(max(0, min(K, self.config.tf_notch_count)))
        if count <= 0:
            return gains, []
        depth = 10.0 ** (float(self.config.tf_notch_depth_db) / 20.0)
        rng = np.random.default_rng(int(self.config.channel_seed) + 98173)
        # Avoid always placing notches at the same cells.  This is still fixed for
        # a given seed, so comparisons across alpha/beta use the same channel.
        idx = rng.choice(K, size=count, replace=False)
        for i in idx:
            gains[int(i)] = depth
        table = []
        for i in idx:
            m = int(i % self.M)
            n = int(i // self.M)
            table.append((int(i), m, n, float(self.config.tf_notch_depth_db)))
        return gains, table

    def _prepare_matrices_locked(self):
        cfg = self.config
        key = (
            cfg.alpha, cfg.beta, cfg.m_subcarriers, cfg.n_symbols, cfg.subcarrier_spacing_hz,
            cfg.mod_order, cfg.channel_model, getattr(cfg, "ntn_profile", cfg.channel_model),
            cfg.velocity_kmh, getattr(cfg, "doppler_radial_factor", 0.10),
            getattr(cfg, "residual_doppler_spread_hz", 200.0),
            getattr(cfg, "doppler_compensation_ratio", 0.999),
            cfg.decoder, cfg.ebn0_db, getattr(cfg, "snr_definition", "Eb/N0"),
            cfg.optimize_indices, cfg.search_step, cfg.fc_hz, cfg.link_mode, cfg.random_channel,
            cfg.channel_seed, cfg.dynamic_channel, cfg.channel_coherence_frames,
            cfg.channel_dynamics, cfg.fast_channel_coherence_symbols,
            cfg.circular_channel, cfg.tf_notch_depth_db, cfg.tf_notch_count,
        )
        if self._cache_key == key and self._H_cross is not None:
            return
        # Paper-aligned default: construct the TF-domain channel directly from
        # TDL/CDL-like paths with fractional delay and fractional Doppler.  The
        # previous artificial TF notch is disabled by default and only applied
        # when the user intentionally sets tf_notch_count > 0.
        self._H_tf, path_table = self._build_paper_tf_channel_matrix()
        self._A_tx, self._A_rx = self._build_tf_transform_matrices()
        self._G_time = None
        fade_diag, notch_table = self._build_tf_fade_diagonal()
        self._fade_diag = fade_diag.copy()
        if int(cfg.tf_notch_count) > 0:
            self._H_tf = fade_diag[:, None] * self._H_tf

        alpha, beta = cfg.alpha, cfg.beta
        search_score = float("nan")
        if cfg.optimize_indices:
            alpha, beta, search_score = self._search_indices_locked(step=cfg.search_step)
        self._used_alpha = float(alpha)
        self._used_beta = float(beta)
        self._tx_fdit, self._rx_fdit = self._fdit_matrices(self._used_alpha, self._used_beta)
        self._H_cross = self._rx_fdit @ self._H_tf @ self._tx_fdit
        self._detector_cache = None
        self._update_scores_locked(search_score=search_score, path_table=path_table, notch_table=notch_table)
        self._update_impulse_locked()
        self._cache_key = key

    def _noise_variance(self, ebn0_db: Optional[float] = None) -> float:
        if ebn0_db is None:
            ebn0_db = self.config.ebn0_db
        lin = 10.0 ** (float(ebn0_db) / 10.0)
        # QAM symbols are normalized to Es=1.  The old GUI label is Eb/N0, but
        # the paper's SER derivation is naturally written with Es/N0.  Keep both
        # modes explicit so curves are not silently shifted by log2(Q).
        snr_def = str(getattr(self.config, "snr_definition", "Eb/N0")).upper().replace(" ", "")
        if snr_def in ("ES/N0", "ESN0", "SNR"):
            return float(1.0 / max(lin, 1e-15))
        return float(1.0 / max(self.bits_per_symbol * lin, 1e-15))

    @staticmethod
    def _qfunc(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        flat = x.reshape(-1)
        out = np.fromiter((0.5 * math.erfc(float(v) / math.sqrt(2.0)) for v in flat), dtype=np.float64, count=flat.size)
        return out.reshape(x.shape)

    @classmethod
    def _qam_ser_from_complex_noise_var(cls, noise_var_eff: np.ndarray, order: int) -> np.ndarray:
        q = int(order)
        noise_var_eff = np.maximum(np.asarray(noise_var_eff, dtype=np.float64), 1e-15)
        kappa = 1.0 - 1.0 / np.sqrt(q)
        arg = np.sqrt(3.0 / ((q - 1.0) * noise_var_eff))
        qv = cls._qfunc(arg)
        ser = 4.0 * kappa * qv - 4.0 * (kappa ** 2) * (qv ** 2)
        return np.clip(ser, 0.0, 1.0)

    def _zf_noise_gain(self, H: np.ndarray):
        W = np.linalg.pinv(H, rcond=1e-10)
        row_norm2 = np.sum(np.abs(W) ** 2, axis=1)
        return W, row_norm2

    def _zf_theory_ser_for_channel(self, H: np.ndarray, ebn0_db: Optional[float] = None):
        W, row_norm2 = self._zf_noise_gain(H)
        noise_var = self._noise_variance(ebn0_db=ebn0_db)
        ser_each = self._qam_ser_from_complex_noise_var(noise_var * row_norm2, self.qam_order)
        return float(np.mean(ser_each)), {
            "noise_var": float(noise_var),
            "row_norm_mean": float(np.mean(np.sqrt(row_norm2))),
            "row_norm_max": float(np.max(np.sqrt(row_norm2))),
            "row_norm_min": float(np.min(np.sqrt(row_norm2))),
            "row_norm2_mean": float(np.mean(row_norm2)),
            "expected_sym_errors_per_frame": float(np.mean(ser_each) * self.K),
            "W": W,
        }

    def _mmse_theory_ser_for_channel(self, H: np.ndarray, ebn0_db: Optional[float] = None):
        noise_var = self._noise_variance(ebn0_db=ebn0_db)
        K = H.shape[1]
        A = H.conj().T @ H + noise_var * np.eye(K, dtype=np.complex128)
        W = np.linalg.solve(A, H.conj().T)
        WH = W @ H
        residual = WH - np.eye(K, dtype=np.complex128)
        # Eq. in the paper uses Gaussian approximation for QAM symbols; here Es=1.
        eff_var = np.sum(np.abs(residual) ** 2, axis=1) + noise_var * np.sum(np.abs(W) ** 2, axis=1)
        ser_each = self._qam_ser_from_complex_noise_var(eff_var, self.qam_order)
        return float(np.mean(ser_each)), {
            "noise_var": float(noise_var),
            "row_norm_mean": float(np.mean(np.sqrt(np.sum(np.abs(W) ** 2, axis=1)))),
            "row_norm_max": float(np.max(np.sqrt(np.sum(np.abs(W) ** 2, axis=1)))),
            "row_norm_min": float(np.min(np.sqrt(np.sum(np.abs(W) ** 2, axis=1)))),
            "row_norm2_mean": float(np.mean(np.sum(np.abs(W) ** 2, axis=1))),
            "expected_sym_errors_per_frame": float(np.mean(ser_each) * self.K),
            "W": W,
            "mmse_residual_mse_mean": float(np.mean(np.sum(np.abs(residual) ** 2, axis=1))),
        }

    def _theory_ser_for_channel(self, H: np.ndarray, ebn0_db: Optional[float] = None, decoder: Optional[str] = None):
        """Decoder-aware theory/proxy SER used by index search.

        ZF and MMSE use the closed-form style expressions.  For SIC/QTML the
        search falls back to the ZF noise-enhancement proxy because exact tree
        search inside every alpha/beta candidate would be too slow for the GUI.
        """
        dec = str(decoder or self.config.decoder or "ZF").upper()
        if dec == "MMSE":
            ser, aux = self._mmse_theory_ser_for_channel(H, ebn0_db=ebn0_db)
            aux["objective_label"] = "MMSE_theory_SER"
            return ser, aux
        ser, aux = self._zf_theory_ser_for_channel(H, ebn0_db=ebn0_db)
        aux["objective_label"] = "ZF_theory_SER" if dec == "ZF" else f"ZF_proxy_for_{dec}"
        return ser, aux

    def _update_scores_locked(self, search_score: float, path_table, notch_table):
        H = self._H_cross
        H_ofdm = self._H_tf  # alpha=beta=0 -> identity FDIT.
        zf_ser, zf_aux = self._zf_theory_ser_for_channel(H)
        mmse_ser, mmse_aux = self._mmse_theory_ser_for_channel(H)
        selected_ser, selected_aux = self._theory_ser_for_channel(H)
        ofdm_ser, ofdm_aux = self._theory_ser_for_channel(H_ofdm)
        try:
            svals = np.linalg.svd(H, compute_uv=False)
            cond = float(svals[0] / max(svals[-1], 1e-15))
            rank = int(np.linalg.matrix_rank(H, tol=1e-9))
            sv_min, sv_max = float(svals[-1]), float(svals[0])
        except Exception:
            cond, rank, sv_min, sv_max = float("nan"), 0, float("nan"), float("nan")
        K = H.shape[0]
        I = np.eye(K, dtype=np.complex128)
        tx_unit = float(np.linalg.norm(self._tx_fdit.conj().T @ self._tx_fdit - I) / max(np.linalg.norm(I), 1e-15))
        rx_unit = float(np.linalg.norm(self._rx_fdit.conj().T @ self._rx_fdit - I) / max(np.linalg.norm(I), 1e-15))
        H_tf_dbg = self._H_tf
        H_from_chain = self._rx_fdit @ H_tf_dbg @ self._tx_fdit
        chain_err = float(np.linalg.norm(H_from_chain - H) / max(np.linalg.norm(H), 1e-15))
        improvement = float(100.0 * (ofdm_ser - selected_ser) / ofdm_ser) if ofdm_ser > 1e-15 else float("nan")
        self._score = {
            "zf_theory_ser": zf_ser,
            "mmse_theory_ser": mmse_ser,
            "selected_theory_ser": selected_ser,
            "theory_objective_label": str(selected_aux.get("objective_label", "ZF_theory_SER")),
            "ofdm_zf_theory_ser": ofdm_ser,
            "theory_improvement_percent": improvement,
            "noise_var": float(selected_aux["noise_var"]),
            "row_norm_mean": float(selected_aux["row_norm_mean"]),
            "row_norm_max": float(selected_aux["row_norm_max"]),
            "row_norm_min": float(selected_aux["row_norm_min"]),
            "row_norm2_mean": float(selected_aux["row_norm2_mean"]),
            "ofdm_row_norm_max": float(ofdm_aux["row_norm_max"]),
            "expected_sym_errors_per_frame": float(selected_aux["expected_sym_errors_per_frame"]),
            "condition_number": cond,
            "rank": rank,
            "sv_min": sv_min,
            "sv_max": sv_max,
            "search_score": float(search_score),
        }
        self._debug = {
            "tx_unitarity_error": tx_unit,
            "rx_unitarity_error": rx_unit,
            "chain_matrix_relative_error": chain_err,
            "path_table_ns_samples_fd_abs_gain": path_table,
            "tf_notch_table_index_m_n_depthdb": notch_table,
            "tf_notch_depth_db": float(self.config.tf_notch_depth_db),
            "tf_notch_count": int(self.config.tf_notch_count),
            "avg_H_row_power": float(np.mean(np.sum(np.abs(H) ** 2, axis=1))),
            "avg_H_col_power": float(np.mean(np.sum(np.abs(H) ** 2, axis=0))),
        }

    def _update_impulse_locked(self):
        H = self._H_cross
        center = int((self.M // 2) + (self.N // 2) * self.M)
        center = int(np.clip(center, 0, H.shape[1] - 1))
        resp = H[:, center]
        self._impulse_grid = resp.reshape(self.M, self.N, order="F").astype(np.complex64, copy=True)
        mag = np.abs(resp)
        if mag.size and float(np.max(mag)) > 1e-15:
            peak = float(np.max(mag))
            energy = mag ** 2
            peak_idx = int(np.argmax(mag))
            self._impulse_peak_energy_ratio = float(energy[peak_idx] / max(float(np.sum(energy)), 1e-15))
            rel_db = 20.0 * np.log10(mag / peak + 1e-15)
            self._impulse_spread_cells_20db = int(np.count_nonzero(rel_db >= -20.0))
            self._impulse_max_sidelobe_db = float(np.max(np.delete(rel_db, peak_idx))) if mag.size > 1 else float("nan")
        else:
            self._impulse_peak_energy_ratio = float("nan")
            self._impulse_spread_cells_20db = 0
            self._impulse_max_sidelobe_db = float("nan")

    def _search_indices_locked(self, step: float = 0.1):
        values = np.round(np.arange(0.0, 2.0 + 0.5 * step, step), 10)
        best_a, best_b, best_s = 0.0, 0.0, float("inf")
        for a, b in itertools.product(values, values):
            tx, rx = self._fdit_matrices(float(a), float(b))
            H = rx @ self._H_tf @ tx
            ser, _ = self._theory_ser_for_channel(H)
            if ser < best_s:
                best_a, best_b, best_s = float(a), float(b), float(ser)
        return best_a, best_b, best_s

    # ----------------------------- simulation -----------------------------
    def _simulate_one_frame(self):
        # 整帧在锁内完成：比特生成、映射、解码、硬判、统计都使用同一份 config 快照，
        # 避免 GUI 线程在帧中途修改 mod_order/M/N/decoder 等参数导致
        # bits 长度与星座/比特表不匹配（如 256 bit vs 128 bit 广播错误）。
        with self._lock:
            self._sim_frame += 1
            self._maybe_update_dynamic_channel_locked()
            self._prepare_matrices_locked()
            H = self._H_cross.copy()
            H_tf = self._H_tf.copy()
            prev_h = self._previous_h_tf_for_state
            h_norm = float(np.linalg.norm(H_tf, "fro"))
            self._channel_power_db = float(
                10.0 * np.log10(max((h_norm * h_norm) / max(self.K, 1), 1e-15))
            )
            if prev_h is not None and np.shape(prev_h) == np.shape(H_tf):
                prev_norm = float(np.linalg.norm(prev_h, "fro"))
                denom = max(prev_norm * h_norm, 1e-15)
                self._channel_matrix_correlation = float(
                    np.clip(abs(np.vdot(prev_h, H_tf)) / denom, 0.0, 1.0)
                )
                self._channel_matrix_change_norm = float(
                    np.linalg.norm(H_tf - prev_h, "fro") / max(prev_norm, 1e-15)
                )
            else:
                self._channel_matrix_correlation = float("nan")
                self._channel_matrix_change_norm = float("nan")
            self._previous_h_tf_for_state = H_tf.copy()
            tx_fdit = self._tx_fdit.copy()
            rx_fdit = self._rx_fdit.copy()
            noise_var = self._noise_variance()
            cfg = self.config

            bits = self._rng.integers(0, 2, size=self.K * self.bits_per_symbol, dtype=np.uint8)
            bit_groups = bits.reshape(-1, self.bits_per_symbol)
            x = self._map_bits_to_symbols(bit_groups)
            noise = self._complex_awgn(self.K, noise_var)

            if cfg.link_mode == "full":
                # Paper-domain full chain: x -> IFDIT -> H_TF -> FDIT -> y.
                # Noise is added after FDIT so the theory and measured SER use the
                # same AWGN convention.
                y_clean = rx_fdit @ (H_tf @ (tx_fdit @ x))
                y = y_clean + noise
            else:
                y_clean = H @ x
                y = y_clean + noise

            x_soft = self._decode(y, H, noise_var)
            hard_idx = self._nearest_symbol_indices(x_soft)
            hard_symbols = self._constellation[hard_idx]
            hard_bits = self._bit_patterns[hard_idx].reshape(-1)
            bit_errors = int(np.count_nonzero(bits != hard_bits[: bits.size]))
            symbol_errors = int(np.count_nonzero(hard_idx != self._symbols_to_indices(x)))
            self._metrics.update(bit_errors, bits.size, symbol_errors, self.K)

        # Debug equalized noise only for ZF/MMSE linear detector.  For nonlinear
        # detectors this remains a useful linear reference.
        try:
            W = None
            if isinstance(self._detector_cache, dict) and "W" in self._detector_cache:
                W = self._detector_cache["W"]
            if W is None:
                W = self._linear_detector_matrix(H, noise_var, force_zf=(cfg.decoder.upper() == "ZF"))
            eq_noise = W @ noise
            mse_clean = float(np.mean(np.abs((W @ y_clean) - x) ** 2))
        except Exception:
            eq_noise = np.zeros_like(x)
            mse_clean = float("nan")
        evm = float(np.sqrt(np.mean(np.abs(x_soft - x) ** 2) / max(np.mean(np.abs(x) ** 2), 1e-15)) * 100.0)
        measured_noise_var = float(np.mean(np.abs(noise) ** 2))
        measured_eq_noise_var = float(np.mean(np.abs(eq_noise) ** 2))
        expected_eq_noise_var = float(noise_var * self._score.get("row_norm2_mean", float("nan")))

        with self._lock:
            self._last_x = x.copy()
            self._last_y = y.copy()
            self._last_x_est = x_soft.copy()
            self._last_noise = noise.copy()
            self._last_equalized_noise = eq_noise.copy()
            self._constellation_points = self._downsample_points(x_soft, 500)
            self._pre_eq_points = self._downsample_points(y, 500)
            self._last_metrics = self._make_metrics(bit_errors, bits.size, symbol_errors, self.K, evm, measured_noise_var, measured_eq_noise_var, expected_eq_noise_var, mse_clean)
        self._adaptive_note_frame_processed()

    def _complex_awgn(self, n: int, var: float) -> np.ndarray:
        return np.sqrt(max(var, 1e-15) / 2.0) * (self._rng.standard_normal(n) + 1j * self._rng.standard_normal(n))

    def _linear_detector_matrix(self, H: np.ndarray, noise_var: float, force_zf: bool = False):
        if force_zf or self.config.decoder.upper() == "ZF":
            return np.linalg.pinv(H, rcond=1e-10)
        A = H.conj().T @ H + noise_var * np.eye(H.shape[1], dtype=np.complex128)
        return np.linalg.solve(A, H.conj().T)

    def _decode(self, y: np.ndarray, H: np.ndarray, noise_var: float) -> np.ndarray:
        dec = self.config.decoder.upper()
        if "SIC" in dec:
            return self._decode_zf_sic(y, H)
        if "QTML" in dec:
            return self._decode_small_ml(y, H)
        cache_key = (dec, H.shape[0], round(noise_var, 14), round(self._used_alpha, 8), round(self._used_beta, 8))
        if self._detector_cache is None or self._detector_cache.get("key") != cache_key:
            W = self._linear_detector_matrix(H, noise_var, force_zf=(dec == "ZF"))
            self._detector_cache = {"key": cache_key, "W": W}
        return self._detector_cache["W"] @ y

    def _decode_zf_sic(self, y: np.ndarray, H: np.ndarray) -> np.ndarray:
        Qm, R = np.linalg.qr(H)
        yq = Qm.conj().T @ y
        K = H.shape[1]
        x = np.zeros(K, dtype=np.complex128)
        for k in range(K - 1, -1, -1):
            diag = R[k, k] if abs(R[k, k]) > 1e-12 else 1e-12
            x[k] = self._nearest_symbol((yq[k] - R[k, k + 1:] @ x[k + 1:]) / diag)
        return x

    def _decode_small_ml(self, y: np.ndarray, H: np.ndarray) -> np.ndarray:
        # Exact ML only for tiny toy cases; otherwise fall back to SIC so UI remains responsive.
        if self.K > (10 if self.qam_order == 4 else 6):
            return self._decode_zf_sic(y, H)
        best_x = None
        best_d = float("inf")
        for combo in itertools.product(self._constellation, repeat=self.K):
            cand = np.asarray(combo, dtype=np.complex128)
            d = float(np.linalg.norm(y - H @ cand) ** 2)
            if d < best_d:
                best_d = d
                best_x = cand
        return best_x if best_x is not None else self._decode_zf_sic(y, H)

    # ----------------------------- maps/debug -----------------------------
    def get_alpha_beta_score_map(self, step: float = 0.1):
        step = float(np.clip(step if step else 0.1, 0.05, 0.5))
        with self._lock:
            self._prepare_matrices_locked()
            H_tf = self._H_tf.copy()
        values = np.round(np.arange(0.0, 2.0 + 0.5 * step, step), 10)
        scores = np.zeros((len(values), len(values)), dtype=np.float64)
        rowmax = np.zeros_like(scores)
        best_a, best_b, best_s = 0.0, 0.0, float("inf")
        t0 = time.time()
        for ia, a in enumerate(values):
            for ib, b in enumerate(values):
                tx, rx = self._fdit_matrices(float(a), float(b))
                H = rx @ H_tf @ tx
                ser, aux = self._zf_theory_ser_for_channel(H)
                scores[ib, ia] = ser
                rowmax[ib, ia] = aux["row_norm_max"]
                if ser < best_s:
                    best_a, best_b, best_s = float(a), float(b), float(ser)
        return {
            "alphas": values,
            "betas": values,
            "scores": scores,
            "row_norm_max": rowmax,
            "best_alpha": best_a,
            "best_beta": best_b,
            "best_score": best_s,
            "elapsed_s": float(time.time() - t0),
            "channel_seed": int(self.config.channel_seed),
        }

    def estimate_ber(self, num_frames: int = 100, seed: int = None) -> float:
        old_rng = self._rng
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self.reset_ber_stats()
        frames = int(max(1, num_frames))
        for _ in range(frames):
            self._simulate_one_frame()
        ber = self._metrics.get_cumulative_ber()
        if seed is not None:
            self._rng = old_rng
        return float(ber)

    def _make_metrics(self, bit_errors, total_bits, sym_errors, total_syms, evm, measured_noise_var, measured_eq_noise_var, expected_eq_noise_var, mse_clean):
        window_ser = self._metrics.get_window_ser()
        window_ber = self._metrics.get_window_ber()
        cum_ser = self._metrics.get_cumulative_ser()
        cum_ber = self._metrics.get_cumulative_ber()
        z = self._metrics.zero_error_upper95_ser()
        score = dict(self._score)
        return {
            "ebn0_db": float(self.config.ebn0_db),
            "snr_db": float(self.config.ebn0_db),
            "ber": cum_ber,
            "ber_window": window_ber,
            "ser": window_ser,
            "ser_cumulative": cum_ser,
            "ser_frame": float(sym_errors / max(total_syms, 1)),
            "fer": self._metrics.get_fer(),
            "zero_error_upper95_ser": z,
            "zf_theory_ser": float(score.get("zf_theory_ser", float("nan"))),
            "mmse_theory_ser": float(score.get("mmse_theory_ser", float("nan"))),
            "selected_theory_ser": float(score.get("selected_theory_ser", float("nan"))),
            "theory_objective_label": str(score.get("theory_objective_label", "ZF_theory_SER")),
            "ofdm_zf_theory_ser": float(score.get("ofdm_zf_theory_ser", float("nan"))),
            "predicted_ser": float(score.get("selected_theory_ser", score.get("zf_theory_ser", float("nan")))),
            "ofdm_predicted_ser": float(score.get("ofdm_zf_theory_ser", float("nan"))),
            "predicted_ser_improvement_percent": float(score.get("theory_improvement_percent", float("nan"))),
            "row_norm_max": float(score.get("row_norm_max", float("nan"))),
            "row_norm_mean": float(score.get("row_norm_mean", float("nan"))),
            "ofdm_row_norm_max": float(score.get("ofdm_row_norm_max", float("nan"))),
            "expected_sym_errors_per_frame": float(score.get("expected_sym_errors_per_frame", float("nan"))),
            "condition_number": float(score.get("condition_number", float("nan"))),
            "rank": int(score.get("rank", 0)),
            "sv_min": float(score.get("sv_min", float("nan"))),
            "sv_max": float(score.get("sv_max", float("nan"))),
            "noise_var": float(score.get("noise_var", float("nan"))),
            "measured_noise_var": float(measured_noise_var),
            "measured_eq_noise_var": float(measured_eq_noise_var),
            "expected_eq_noise_var": float(expected_eq_noise_var),
            "clean_equalizer_mse": float(mse_clean),
            "tx_unitarity_error": float(self._debug.get("tx_unitarity_error", float("nan"))),
            "rx_unitarity_error": float(self._debug.get("rx_unitarity_error", float("nan"))),
            "chain_matrix_relative_error": float(self._debug.get("chain_matrix_relative_error", float("nan"))),
            "avg_H_row_power": float(self._debug.get("avg_H_row_power", float("nan"))),
            "channel_power_db": float(self._channel_power_db),
            "channel_matrix_change_norm": float(self._channel_matrix_change_norm),
            "channel_matrix_correlation": float(self._channel_matrix_correlation),
            "alpha": float(self.config.alpha),
            "beta": float(self.config.beta),
            "used_alpha": float(self._used_alpha),
            "used_beta": float(self._used_beta),
            "decoder": self.config.decoder,
            "decoder_actual": self.config.decoder,
            "mod_order": self.config.mod_order,
            "channel_model": self.config.channel_model,
            "ntn_profile": str(getattr(self.config, "ntn_profile", self.config.channel_model)),
            "velocity_kmh": float(self.config.velocity_kmh),
            # Keep doppler_hz as the residual channel limit for old UI readers.
            "doppler_hz": self._channel_max_residual_doppler_hz(),
            "absolute_doppler_shift_hz": self._absolute_doppler_shift_hz(),
            "residual_common_cfo_hz": self._residual_common_doppler_hz(),
            "residual_doppler_spread_hz": self._residual_doppler_spread_hz(),
            "coherence_time_s": coherence_time_s(self._residual_doppler_spread_hz()),
            "doppler_compensation_ratio": float(getattr(self.config, "doppler_compensation_ratio", 0.999)),
            "doppler_radial_factor": float(getattr(self.config, "doppler_radial_factor", 0.10)),
            "snr_definition": str(getattr(self.config, "snr_definition", "Eb/N0")),
            "random_channel": bool(self.config.random_channel),
            "channel_seed": int(self.config.channel_seed),
            "dynamic_channel": bool(self.config.dynamic_channel),
            "channel_dynamics": str(getattr(self.config, "channel_dynamics", "fixed")),
            "channel_coherence_frames": int(getattr(self.config, "channel_coherence_frames", 1)),
            "fast_channel_coherence_symbols": int(getattr(self.config, "fast_channel_coherence_symbols", 1)),
            "demo_frame_interval_s": float(getattr(self.config, "demo_frame_interval_s", 0.02)),
            "link_mode": self.config.link_mode,
            "circular_channel": bool(self.config.circular_channel),
            "tf_notch_depth_db": float(self.config.tf_notch_depth_db),
            "tf_notch_count": int(self.config.tf_notch_count),
            "bit_errors": int(bit_errors),
            "total_bits": int(total_bits),
            "symbol_errors": int(sym_errors),
            "total_symbols": int(total_syms),
            "window_symbol_errors": int(np.sum(self._metrics.recent_symbol_errors)) if self._metrics.recent_symbol_errors else 0,
            "window_symbol_total": int(np.sum(self._metrics.recent_symbol_total)) if self._metrics.recent_symbol_total else 0,
            "cumulative_symbol_errors": int(self._metrics.total_symbol_errors),
            "cumulative_symbol_total": int(self._metrics.total_symbols),
            "frames": int(getattr(self, "_sim_frame", self._metrics.total_frames)),
            "evm_percent": float(evm),
            "evm_db": float(20 * np.log10(max(evm / 100.0, 1e-12))),
            "impulse_peak_energy_ratio": float(self._impulse_peak_energy_ratio),
            "impulse_spread_cells_20db": int(self._impulse_spread_cells_20db),
            "impulse_max_sidelobe_db": float(self._impulse_max_sidelobe_db),
            "index_search_time_s": 0.0,
        }

    def _empty_metrics(self):
        return {
            "ebn0_db": float(self.config.ebn0_db),
            "ber": float("nan"),
            "ser": float("nan"),
            "fer": float("nan"),
            "zf_theory_ser": float("nan"),
            "selected_theory_ser": float("nan"),
            "theory_objective_label": "--",
            "condition_number": float("nan"),
            "channel_power_db": float("nan"),
            "channel_matrix_change_norm": float("nan"),
            "channel_matrix_correlation": float("nan"),
            "alpha": float(self.config.alpha),
            "beta": float(self.config.beta),
            "used_alpha": float(self.config.alpha),
            "used_beta": float(self.config.beta),
            "frames": int(getattr(self, "_sim_frame", 0)),
        }


    def estimate_ser(self, num_frames: int = 100, seed: int = None, stop_event=None, ser_display_floor: Optional[float] = None):
        """Monte-Carlo SER/BER for the current alpha/beta on the current channel.

        This method is intentionally deterministic when seed is provided.  It is
        used by the UI to draw measured SER-SNR curves for OFDM, OTFS, the
        manual point and the searched best point.  In fixed mode it uses the
        original same-H realization; in block/fast/cont modes it averages over
        the deterministic channel evolution selected in the parameter panel.  If
        zero errors occur, ``ser_display`` and ``ber_display``
        use the 95% rule-of-three upper bound instead of plotting an artificial
        zero on the log axis.
        """
        old_rng = self._rng
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self.reset_ber_stats()
        frames = int(max(1, num_frames))
        try:
            for _ in range(frames):
                if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                    break
                self._simulate_one_frame()
            total_symbols = int(self._metrics.total_symbols)
            symbol_errors = int(self._metrics.total_symbol_errors)
            total_bits = int(self._metrics.total_bits)
            bit_errors = int(self._metrics.total_bit_errors)
            raw_ser = float(self._metrics.get_cumulative_ser())
            raw_ber = float(self._metrics.get_cumulative_ber())
            zero_error_ser_upper = float(3.0 / max(total_symbols, 1))
            ser_floor = zero_error_ser_upper if ser_display_floor is None else float(ser_display_floor)
            if not np.isfinite(ser_floor) or ser_floor <= 0:
                ser_floor = zero_error_ser_upper
            # ser_display_floor allows the GUI to enforce a plot-specific lower
            # bound, e.g. the left-bottom α-SER curve must never be displayed
            # below 3/(configured_MC_frames*M*N), even when a theoretical value
            # or a measured 1-error estimate is lower.
            ser_display = max(raw_ser, ser_floor) if symbol_errors > 0 else max(zero_error_ser_upper, ser_floor)
            ber_display = raw_ber if bit_errors > 0 else float(3.0 / max(total_bits, 1))
            out = {
                "ser": raw_ser,
                "ber": raw_ber,
                "ser_display": float(ser_display),
                "ber_display": float(ber_display),
                "zero_error_ser_upper95": zero_error_ser_upper if symbol_errors == 0 else float("nan"),
                "ser_display_floor": float(ser_floor),
                "zero_error_ber_upper95": float(3.0 / max(total_bits, 1)) if bit_errors == 0 else float("nan"),
                "symbol_errors": symbol_errors,
                "total_symbols": total_symbols,
                "bit_errors": bit_errors,
                "total_bits": total_bits,
                "frames": int(getattr(self, "_sim_frame", self._metrics.total_frames)),
                "dynamic_channel": bool(self.config.dynamic_channel),
                "channel_dynamics": str(getattr(self.config, "channel_dynamics", "fixed")),
                "channel_coherence_frames": int(getattr(self.config, "channel_coherence_frames", 1)),
                "fast_channel_coherence_symbols": int(getattr(self.config, "fast_channel_coherence_symbols", 1)),
            }
        finally:
            if seed is not None:
                self._rng = old_rng
        return out

    def evaluate_theory_point(self, alpha: float, beta: float, ebn0_db: Optional[float] = None):
        """Evaluate decoder-aware theory/proxy SER for one FDIDM index pair.

        The physical/TF channel H_tf is fixed by the current LEO channel.  Only
        tx/rx FDIT matrices are changed to form H_eq(alpha,beta).  ZF and MMSE
        use closed-form/approximate theory; SIC-like decoders use the ZF proxy
        selected by _theory_ser_for_channel().
        """
        with self._lock:
            self._prepare_matrices_locked()
            H_tf = self._H_tf.copy()
        tx, rx = self._fdit_matrices(float(alpha), float(beta))
        H_eq = rx @ H_tf @ tx
        ser, aux = self._theory_ser_for_channel(H_eq, ebn0_db=ebn0_db)
        try:
            svals = np.linalg.svd(H_eq, compute_uv=False)
            cond = float(svals[0] / max(svals[-1], 1e-15))
            sv_min = float(svals[-1])
        except Exception:
            cond = float("nan")
            sv_min = float("nan")
        return {
            "alpha": float(alpha),
            "beta": float(beta),
            # Backward-compatible key kept for existing UI code.  For MMSE this
            # value is the selected decoder-aware theory/proxy SER, not ZF.
            "zf_theory_ser": float(ser),
            "selected_theory_ser": float(ser),
            "theory_objective_label": str(aux.get("objective_label", "ZF_theory_SER")),
            "row_norm_max": float(aux.get("row_norm_max", float("nan"))),
            "row_norm_mean": float(aux.get("row_norm_mean", float("nan"))),
            "expected_sym_errors_per_frame": float(aux.get("expected_sym_errors_per_frame", float("nan"))),
            "condition_number": cond,
            "sv_min": sv_min,
        }

    def evaluate_theory_curve(self, alpha: float, beta: float, snr_db_values):
        """Evaluate a decoder-aware theory/proxy SER curve for one index pair.

        This is retained for legacy theory-curve callers.  It evaluates many
        Eb/N0 points without Monte-Carlo transmission.  For ZF and SIC-like
        proxy modes, the expensive pseudo-inverse is computed once and reused
        across all SNR points; for MMSE, the equalizer genuinely depends on
        noise variance, so the MMSE theory expression is evaluated per SNR.
        """
        snr_values = [float(x) for x in snr_db_values]
        with self._lock:
            self._prepare_matrices_locked()
            H_tf = self._H_tf.copy()
        tx, rx = self._fdit_matrices(float(alpha), float(beta))
        H_eq = rx @ H_tf @ tx
        dec = str(self.config.decoder or "ZF").upper()

        ser_values = []
        label = "MMSE_theory_SER" if dec == "MMSE" else ("ZF_theory_SER" if dec == "ZF" else f"ZF_proxy_for_{dec}")
        if dec == "MMSE":
            for snr in snr_values:
                ser, _aux = self._mmse_theory_ser_for_channel(H_eq, ebn0_db=float(snr))
                ser_values.append(float(ser))
        else:
            # ZF and SIC/QTML proxy: H^{-1} noise gain is SNR-independent.
            _W, row_norm2 = self._zf_noise_gain(H_eq)
            for snr in snr_values:
                noise_var = self._noise_variance(ebn0_db=float(snr))
                ser_each = self._qam_ser_from_complex_noise_var(noise_var * row_norm2, self.qam_order)
                ser_values.append(float(np.mean(ser_each)))

        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "snr_db": [float(x) for x in snr_values],
            "selected_theory_ser": [float(x) for x in ser_values],
            "zf_theory_ser": [float(x) for x in ser_values],
            "theory_objective_label": label,
        }

    def _objective_snr_points(self, ebn0_db: Optional[float], objective_snr_points=None, objective_snr_offsets=None):
        """Build the SNR set used by alpha/beta search.

        A single-SNR optimum is often unstable and may not match the SER-SNR
        curve.  By default we optimize over a small local SNR window around the
        current operating Eb/N0, which is consistent with the legacy
        SER-SNR comparison plot.
        """
        if objective_snr_points is not None:
            pts = [float(x) for x in objective_snr_points]
        else:
            center = float(self.config.ebn0_db if ebn0_db is None else ebn0_db)
            # Do not optimize only at a very high operating SNR: many candidate
            # points can underflow to the same display/search floor, causing the
            # first grid point (0,0) to be selected by accident.  Include several
            # mid-SNR probes used by legacy SER-SNR callers so the ranking
            # is based on the curve shape, not on a saturated high-SNR tail.
            if objective_snr_offsets is not None:
                pts = [center + float(o) for o in objective_snr_offsets]
            else:
                pts = [center - 8.0, center - 4.0, center, 10.0, 15.0, 20.0]
        # Keep points in a numerically meaningful demo range and remove duplicates.
        pts = sorted({round(float(np.clip(x, -5.0, 30.0)), 6) for x in pts})
        return pts or [float(self.config.ebn0_db if ebn0_db is None else ebn0_db)]

    def _score_index_pair_on_snr_set(self, H_eq: np.ndarray, snr_points, score_floor: float = 1e-12):
        """Return robust search score for a fixed H_eq over several SNR points.

        score is the average log10(SER).  Lower is better.  The geometric-mean
        SER, 10**score, is easier to display and compare.
        """
        ser_list = []
        aux_last = {}
        floor = float(max(score_floor, 1e-15))
        for snr in snr_points:
            ser, aux = self._theory_ser_for_channel(H_eq, ebn0_db=float(snr))
            ser_list.append(float(ser))
            aux_last = aux
        safe = np.maximum(np.asarray(ser_list, dtype=np.float64), floor)
        score = float(np.mean(np.log10(safe)))
        score_ser = float(10.0 ** score)
        return score, score_ser, ser_list, aux_last

    def search_best_indices(
        self,
        step: float = 0.1,
        ebn0_db: Optional[float] = None,
        objective_snr_points=None,
        objective_snr_offsets=None,
        top_k: int = 5,
        significance_threshold_percent: float = 5.0,
        stop_event=None,
    ):
        """Robustly search alpha/beta by decoder-aware theory/proxy SER.

        The previous version optimized a single Eb/N0 point.  That can produce
        a fragile optimum that does not look optimal on the full SER-SNR curve.
        This version optimizes the mean log-SER over a local SNR set, returns
        top candidates, and reports whether the best point is meaningfully
        better than the best integer reference waveform (OFDM/OTFS).
        """
        step = float(np.clip(step if step else 0.1, 0.05, 0.5))
        work_ebn0 = float(self.config.ebn0_db if ebn0_db is None else ebn0_db)
        snr_points = self._objective_snr_points(work_ebn0, objective_snr_points, objective_snr_offsets)
        top_k = int(max(1, min(20, top_k)))

        with self._lock:
            self._prepare_matrices_locked()
            H_tf = self._H_tf.copy()

        values = np.round(np.arange(0.0, 2.0 + 0.5 * step, step), 10)
        candidates = []
        t0 = time.time()
        for a, b in itertools.product(values, values):
            if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                break
            tx, rx = self._fdit_matrices(float(a), float(b))
            H_eq = rx @ H_tf @ tx
            score, score_ser, ser_list, aux = self._score_index_pair_on_snr_set(H_eq, snr_points)
            ser_work, _aux_work = self._theory_ser_for_channel(H_eq, ebn0_db=work_ebn0)
            candidates.append({
                "alpha": float(a),
                "beta": float(b),
                "score": float(score),
                "score_ser_geomean": float(score_ser),
                "zf_theory_ser": float(ser_work),
                "ser_at_working_ebn0": float(ser_work),
                "ser_list": [float(x) for x in ser_list],
                "row_norm_max": float(aux.get("row_norm_max", float("nan"))),
                "row_norm_mean": float(aux.get("row_norm_mean", float("nan"))),
                "theory_objective_label": str(aux.get("objective_label", "ZF_theory_SER")),
            })

        candidates.sort(key=lambda x: x["score"])

        # If many candidates are numerically tied at the scoring floor, the raw
        # sort order would always pick the first grid point, i.e. (0,0).  This is
        # a ranking artifact rather than a physical conclusion.  Re-rank tied
        # candidates with a lower-SNR probe where theory SER is usually not
        # saturated.  OFDM is still selected if it is truly best at the probe.
        if candidates:
            best_score0 = float(candidates[0].get("score", float("nan")))
            tied = [c for c in candidates if np.isfinite(best_score0) and abs(float(c.get("score", 0.0)) - best_score0) <= 1e-10]
            if len(tied) > 1:
                probe_pts = [8.0, 12.0, 16.0, 20.0]
                for c in tied:
                    tx, rx = self._fdit_matrices(float(c["alpha"]), float(c["beta"]))
                    H_eq = rx @ H_tf @ tx
                    pscore, pscore_ser, _plist, _paux = self._score_index_pair_on_snr_set(H_eq, probe_pts, score_floor=1e-14)
                    c["tie_break_score"] = float(pscore)
                    c["tie_break_score_ser_geomean"] = float(pscore_ser)
                tied.sort(key=lambda x: (float(x.get("tie_break_score", x["score"])), float(x["alpha"]), float(x["beta"])))
                best_tie = tied[0]
                # Move the tie-broken best candidate to the front while keeping
                # the remaining full list available for diagnostics.
                candidates.sort(key=lambda x: 0 if x is best_tie else 1)

        best = candidates[0] if candidates else {
            "alpha": float(self.config.alpha), "beta": float(self.config.beta),
            "score": float("nan"), "score_ser_geomean": float("nan"),
            "zf_theory_ser": float("nan"), "ser_at_working_ebn0": float("nan"),
            "ser_list": [], "row_norm_max": float("nan"), "row_norm_mean": float("nan"),
        }
        second = candidates[1] if len(candidates) > 1 else None

        def eval_ref(a, b):
            tx, rx = self._fdit_matrices(float(a), float(b))
            H_eq = rx @ H_tf @ tx
            score, score_ser, ser_list, aux = self._score_index_pair_on_snr_set(H_eq, snr_points)
            work_item = self.evaluate_theory_point(float(a), float(b), ebn0_db=work_ebn0)
            work_item.update({
                "score": float(score),
                "score_ser_geomean": float(score_ser),
                "ser_list": [float(x) for x in ser_list],
                "ser_at_working_ebn0": float(work_item.get("zf_theory_ser", float("nan"))),
                "objective_snr_points": [float(x) for x in snr_points],
            })
            return work_item

        refs = {
            "OFDM(0,0)": eval_ref(0.0, 0.0),
            "OTFS(1,1)": eval_ref(1.0, 1.0),
            "当前手动点": eval_ref(self.config.alpha, self.config.beta),
            "理论最优": eval_ref(best["alpha"], best["beta"]),
        }

        ofdm_score_ser = refs["OFDM(0,0)"]["score_ser_geomean"]
        otfs_score_ser = refs["OTFS(1,1)"]["score_ser_geomean"]
        best_score_ser = refs["理论最优"]["score_ser_geomean"]
        integer_ref_score_ser = min(ofdm_score_ser, otfs_score_ser)

        def improvement(base, val):
            return float(100.0 * (base - val) / base) if np.isfinite(base) and base > 1e-15 else float("nan")

        imp_ofdm = improvement(ofdm_score_ser, best_score_ser)
        imp_otfs = improvement(otfs_score_ser, best_score_ser)
        imp_integer = improvement(integer_ref_score_ser, best_score_ser)
        second_gap = improvement(second["score_ser_geomean"], best_score_ser) if second is not None else float("nan")
        significant = bool(np.isfinite(imp_integer) and imp_integer >= float(significance_threshold_percent))
        if significant:
            note = "当前信道下分数索引相对最好的整数参考波形具有较明显理论优势。"
        elif np.isfinite(imp_integer) and imp_integer > 0:
            note = "当前信道下最优点仅略优于OFDM/OTFS，索引差异较弱，建议结合SER-SNR曲线判断。"
        else:
            note = "当前信道下OFDM或OTFS已接近/达到理论最优，分数索引收益不明显。"

        return {
            "best_alpha": float(best["alpha"]),
            "best_beta": float(best["beta"]),
            # SER at the current operating Eb/N0, kept for old UI compatibility.
            "best_ser": float(refs["理论最优"].get("zf_theory_ser", float("nan"))),
            "ofdm_ser": float(refs["OFDM(0,0)"].get("zf_theory_ser", float("nan"))),
            "otfs_ser": float(refs["OTFS(1,1)"].get("zf_theory_ser", float("nan"))),
            # Robust objective values used for actual ranking.
            "best_score": float(best["score"]),
            "best_score_ser_geomean": float(best_score_ser),
            "ofdm_score_ser_geomean": float(ofdm_score_ser),
            "otfs_score_ser_geomean": float(otfs_score_ser),
            "integer_reference_score_ser_geomean": float(integer_ref_score_ser),
            "improvement_vs_ofdm_percent": float(imp_ofdm),
            "improvement_vs_otfs_percent": float(imp_otfs),
            "improvement_vs_best_integer_percent": float(imp_integer),
            "gap_to_second_percent": float(second_gap),
            "significant": significant,
            "significance_threshold_percent": float(significance_threshold_percent),
            "significance_note": note,
            "elapsed_s": float(time.time() - t0),
            "step": float(step),
            "objective": str((candidates[0].get("theory_objective_label", "decoder_aware_theory_SER") if candidates else "decoder_aware_theory_SER")) + "_mean_log10_over_local_SNR_set",
            "decoder": str(self.config.decoder),
            "snr_definition": str(getattr(self.config, "snr_definition", "Eb/N0")),
            "objective_snr_points": [float(x) for x in snr_points],
            "working_ebn0_db": float(work_ebn0),
            "top_candidates": candidates[:top_k],
            "references": refs,
        }

    def get_channel_summary(self):
        """Return current NTN channel state with large/residual Doppler separated."""
        with self._lock:
            self._prepare_matrices_locked()
            debug = dict(self._debug)
            cfg = self.config
            frame = int(getattr(self, "_sim_frame", 0))
        table = []
        for row in debug.get("path_table_ns_samples_fd_abs_gain", []) or []:
            # row: pidx, tau_ns, tau_samples, residual_doppler_hz,
            #      normalized_doppler_index, |gain|
            try:
                pidx, tau_ns, tau_samples, fd, fd_idx, gain_abs = row
                table.append({
                    "path": int(pidx),
                    "delay_ns": float(tau_ns),
                    "delay_samples": float(tau_samples),
                    "doppler_hz": float(fd),
                    "doppler_index": float(fd_idx),
                    "gain_abs": float(gain_abs),
                })
            except Exception:
                continue

        delays = [float(p["delay_ns"]) for p in table]
        dopplers = [float(p["doppler_hz"]) for p in table]
        powers = [float(p["gain_abs"]) ** 2 for p in table]
        delay_mean_ns, delay_rms_ns = weighted_mean_and_rms(delays, powers)
        doppler_mean_hz, doppler_rms_hz = weighted_mean_and_rms(dopplers, powers)
        max_delay_ns = float(max(delays)) if delays else 0.0
        max_residual_hz = float(max((abs(x) for x in dopplers), default=self._channel_max_residual_doppler_hz()))
        bandwidth_hz = float(self.M * cfg.subcarrier_spacing_hz)
        physical_frame_duration_s = float(self.N / max(cfg.subcarrier_spacing_hz, 1e-15))
        absolute_shift = self._absolute_doppler_shift_hz()
        residual_common = self._residual_common_doppler_hz()
        residual_spread = self._residual_doppler_spread_hz()
        profile = get_profile(getattr(cfg, "ntn_profile", cfg.channel_model))

        return {
            "channel_model": profile.name,
            "ntn_profile": profile.name,
            "velocity_kmh": float(cfg.velocity_kmh),
            "velocity_kms": float(cfg.velocity_kmh) / 3600.0,
            "fc_hz": float(cfg.fc_hz),
            "subcarrier_spacing_hz": float(cfg.subcarrier_spacing_hz),
            "sample_rate_hz": float(self.sample_rate),
            "bandwidth_hz": bandwidth_hz,
            "btau_max": float(bandwidth_hz * max_delay_ns * 1e-9),
            "absolute_doppler_shift_hz": float(absolute_shift),
            "doppler_compensation_ratio": float(getattr(cfg, "doppler_compensation_ratio", 0.999)),
            "residual_common_cfo_hz": float(residual_common),
            "residual_doppler_spread_hz": float(residual_spread),
            "max_doppler_hz": max_residual_hz,
            "doppler_mean_hz": float(doppler_mean_hz),
            "doppler_spread_hz": float(doppler_rms_hz),
            "normalized_doppler": normalized_doppler(max_residual_hz, cfg.subcarrier_spacing_hz),
            "coherence_time_s": coherence_time_s(residual_spread),
            "doppler_radial_factor": float(getattr(cfg, "doppler_radial_factor", 0.10)),
            "doppler_warning": bool(max_residual_hz >= cfg.subcarrier_spacing_hz),
            "delay_mean_ns": float(delay_mean_ns),
            "delay_spread_ns": float(delay_rms_ns),
            "max_delay_ns": max_delay_ns,
            "nominal_rms_delay_ns": float(profile.nominal_rms_delay_ns),
            "snr_definition": str(getattr(cfg, "snr_definition", "Eb/N0")),
            "M": int(self.M),
            "N": int(self.N),
            "frame": frame,
            "physical_frame_duration_s": physical_frame_duration_s,
            "physical_time_s": float(frame * physical_frame_duration_s),
            "demo_update_period_s": float(getattr(cfg, "demo_frame_interval_s", 0.02)),
            "channel_power_db": float(self._channel_power_db),
            "channel_matrix_change_norm": float(self._channel_matrix_change_norm),
            "channel_matrix_correlation": float(self._channel_matrix_correlation),
            "seed": int(cfg.channel_seed),
            "base_seed": int(getattr(self, "_dynamic_base_seed", cfg.channel_seed)),
            "random_channel": bool(cfg.random_channel),
            "dynamic_channel": bool(cfg.dynamic_channel),
            "channel_dynamics": str(getattr(cfg, "channel_dynamics", "fixed")),
            "channel_coherence_frames": int(getattr(cfg, "channel_coherence_frames", 1)),
            "fast_channel_coherence_symbols": int(getattr(cfg, "fast_channel_coherence_symbols", 1)),
            "paths": table,
        }

    # ----------------------------- QAM helpers -----------------------------
    @classmethod
    def _build_gray_qam(cls, order: int) -> Tuple[np.ndarray, np.ndarray]:
        root = int(np.sqrt(order))
        if root * root != order:
            raise ValueError("Only square QAM is supported")
        bits_axis = int(np.log2(root))
        bits_total = 2 * bits_axis
        levels = np.arange(-(root - 1), root, 2, dtype=np.float64)
        constellation = np.zeros(order, dtype=np.complex128)
        bit_patterns = np.zeros((order, bits_total), dtype=np.uint8)
        for idx in range(order):
            bits = ((idx >> np.arange(bits_total - 1, -1, -1)) & 1).astype(np.uint8)
            i_gray = cls._bits_to_int(bits[:bits_axis])
            q_gray = cls._bits_to_int(bits[bits_axis:])
            i_bin = cls._gray_to_binary(i_gray)
            q_bin = cls._gray_to_binary(q_gray)
            constellation[idx] = levels[i_bin] + 1j * levels[q_bin]
            bit_patterns[idx] = bits
        constellation /= np.sqrt(np.mean(np.abs(constellation) ** 2) + 1e-15)
        return constellation, bit_patterns

    @staticmethod
    def _bits_to_int(bits: np.ndarray) -> int:
        out = 0
        for b in bits:
            out = (out << 1) | int(b)
        return int(out)

    @staticmethod
    def _gray_to_binary(gray: int) -> int:
        b = int(gray)
        while gray >> 1:
            gray >>= 1
            b ^= gray
        return int(b)

    def _map_bits_to_symbols(self, bit_groups: np.ndarray) -> np.ndarray:
        idx = np.zeros(bit_groups.shape[0], dtype=np.int64)
        for k in range(bit_groups.shape[1]):
            idx = (idx << 1) | bit_groups[:, k].astype(np.int64)
        return self._constellation[idx]

    def _symbols_to_indices(self, symbols: np.ndarray) -> np.ndarray:
        symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
        d = np.abs(symbols[:, None] - self._constellation[None, :])
        return np.argmin(d, axis=1)

    def _nearest_symbol_indices(self, symbols: np.ndarray) -> np.ndarray:
        return self._symbols_to_indices(symbols)

    def _nearest_symbol(self, z: complex) -> complex:
        return self._constellation[int(np.argmin(np.abs(z - self._constellation)))]

    def _downsample_points(self, x: np.ndarray, max_points: int):
        x = np.asarray(x, dtype=np.complex128).reshape(-1)
        if x.size <= max_points:
            return x.astype(np.complex64)
        idx = np.linspace(0, x.size - 1, max_points, dtype=np.int64)
        return x[idx].astype(np.complex64)


# ---------------------------------------------------------------------------
# Unified-engine compatibility facade
# ---------------------------------------------------------------------------
def _create_fdidm_backend(**kwargs):
    """Construct the established FDIDM backend for waveform_sim.core.engine."""
    return _LegacyFDIDMTransceiver(**kwargs)


class FDIDMTransceiver(LinkSimulator):
    """FDIDM compatibility facade backed by _LegacyFDIDMTransceiver."""

    def __init__(self, **kwargs):
        super().__init__(waveform="FDIDM", **kwargs)

    def __getattr__(self, name):
        backend = self.__dict__.get("_backend")
        if backend is not None and hasattr(backend, name):
            return getattr(backend, name)
        raise AttributeError(name)

    @staticmethod
    def _build_gray_qam(order):
        return _LegacyFDIDMTransceiver._build_gray_qam(order)
