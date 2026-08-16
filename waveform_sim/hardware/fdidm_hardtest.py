# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import math
import threading
import time
import zlib
from collections import deque
from typing import Any, Dict, Optional, Tuple, List

import numpy as np


class _SampleRing:
    """Bounded NumPy ring buffer for complex64 streams.

    Avoids GNU Radio vector_sink_c .data()/reset() and per-sample deque/list
    conversions in the live UHD path. Keeping the Python work bounded is
    essential for avoiding B210 U/O at modest sample rates on Windows.
    """

    def __init__(self, capacity: int):
        self.capacity = int(max(1024, capacity))
        self._buf = np.zeros(self.capacity, dtype=np.complex64)
        self._pos = 0
        self._count = 0
        self._total = 0
        self._lock = threading.Lock()

    def clear(self):
        with self._lock:
            self._pos = 0
            self._count = 0
            self._total = 0
            self._buf.fill(0)

    reset = clear

    def __len__(self):
        with self._lock:
            return int(self._count)

    @property
    def total_written(self) -> int:
        with self._lock:
            return int(self._total)

    def write(self, samples: np.ndarray):
        x = np.asarray(samples, dtype=np.complex64).reshape(-1)
        original_n = int(x.size)
        if original_n <= 0:
            return
        with self._lock:
            self._total += original_n
            if original_n >= self.capacity:
                self._buf[:] = x[-self.capacity:]
                self._pos = 0
                self._count = self.capacity
                return
            end = self._pos + original_n
            if end <= self.capacity:
                self._buf[self._pos:end] = x
            else:
                first = self.capacity - self._pos
                self._buf[self._pos:] = x[:first]
                self._buf[:end - self.capacity] = x[first:]
            self._pos = end % self.capacity
            self._count = min(self.capacity, self._count + original_n)

    def read_latest(self, n: int) -> tuple[np.ndarray, int, int]:
        n = int(max(0, n))
        with self._lock:
            k = min(n, self._count)
            total = int(self._total)
            count = int(self._count)
            if k <= 0:
                return np.zeros(0, dtype=np.complex64), total, count
            start = (self._pos - k) % self.capacity
            if start + k <= self.capacity:
                out = self._buf[start:start + k].copy()
            else:
                first = self.capacity - start
                out = np.concatenate((self._buf[start:].copy(), self._buf[:k - first].copy()))
            return out, total, count


class _NTNTDLChannel:
    """Small SISO NTN-TDL software channel for FDIDM bench testing.

    The tap tables follow the NTN-TDL-A/C/D profiles in 3GPP TR 38.811
    Tables 6.9.2-1, 6.9.2-3, and 6.9.2-4.  Delays in the tables are
    normalized; this class scales them to the requested RMS delay spread.

    This is intentionally a lightweight real-time GNU Radio helper:
    - fractional delays are implemented by vectorized linear interpolation;
    - Rayleigh taps use a sum-of-sinusoids approximation to the Doppler
      spectrum; LOS entries are deterministic specular components;
    - one common Doppler shift is applied to all taps, matching the NTN
      additional satellite Doppler statement in TR 38.811.
    """

    # (normalized_delay, power_dB, fading_kind)
    # LOS C/D are represented exactly as the table entries: a deterministic
    # LOS component plus a Rayleigh component at the same delay.
    TDL_PROFILES: Dict[str, List[Tuple[float, float, str]]] = {
        "tdl_a": [
            (0.0000, 0.000, "rayleigh"),
            (1.0811, -4.675, "rayleigh"),
            (2.8416, -6.482, "rayleigh"),
        ],
        "tdl_c": [
            (0.0000, -0.394, "los"),
            (0.0000, -10.618, "rayleigh"),
            (14.8124, -23.373, "rayleigh"),
        ],
        "tdl_d": [
            (0.0000, -0.284, "los"),
            (0.0000, -11.991, "rayleigh"),
            (0.5596, -9.887, "rayleigh"),
            (7.3340, -16.771, "rayleigh"),
        ],
    }

    DISPLAY_NAMES = {
        "tdl_a": "NTN-TDL-A (NLOS/Rayleigh)",
        "tdl_c": "NTN-TDL-C (LOS/Rician)",
        "tdl_d": "NTN-TDL-D (LOS/Rician)",
    }

    def __init__(
            self,
            sample_rate: float,
            model: str = "tdl_a",
            rms_delay_spread_ns: float = 1000.0,
            doppler_hz: float = 0.0,
            doppler_spread_hz: float = 0.0,
            snr_db: float = 35.0,
            seed: int = 0x38_811,
            normalize_power: bool = True,
            num_sinusoids: int = 8,
    ):
        self.sample_rate = float(max(sample_rate, 1.0))
        self.model = "tdl_a"
        self.rms_delay_spread_ns = 1000.0
        self.doppler_hz = 0.0
        self.doppler_spread_hz = 0.0
        self.snr_db = 35.0
        self.seed = int(seed) & 0xFFFFFFFF
        self.normalize_power = bool(normalize_power)
        self.num_sinusoids = int(max(4, min(int(num_sinusoids), 64)))
        self._rng = np.random.default_rng(self.seed)
        self._components: List[Dict[str, Any]] = []
        self._history = np.zeros(8, dtype=np.complex128)
        self._sample_index = 0
        self._delay_scale_s = 0.0
        self.configure(sample_rate=sample_rate, model=model,
                       rms_delay_spread_ns=rms_delay_spread_ns,
                       doppler_hz=doppler_hz,
                       doppler_spread_hz=doppler_spread_hz,
                       snr_db=snr_db,
                       seed=seed,
                       normalize_power=normalize_power,
                       num_sinusoids=num_sinusoids)

    @classmethod
    def normalize_model(cls, model: str) -> str:
        m = str(model or "tdl_a").strip().lower().replace("-", "_").replace(" ", "_")
        if m in ("a", "ntn_tdl_a", "tdla", "tdl_a"):
            return "tdl_a"
        if m in ("c", "ntn_tdl_c", "tdlc", "tdl_c"):
            return "tdl_c"
        if m in ("d", "ntn_tdl_d", "tdld", "tdl_d"):
            return "tdl_d"
        if m in cls.TDL_PROFILES:
            return m
        raise ValueError("software TDL model must be one of: tdl_a, tdl_c, tdl_d")

    @staticmethod
    def _db_to_linear(power_db: float) -> float:
        return float(10.0 ** (float(power_db) / 10.0))

    def configure(self, **kwargs: Any):
        if "sample_rate" in kwargs and kwargs["sample_rate"] is not None:
            self.sample_rate = float(max(float(kwargs["sample_rate"]), 1.0))
        if "model" in kwargs and kwargs["model"] is not None:
            self.model = self.normalize_model(kwargs["model"])
        if "rms_delay_spread_ns" in kwargs and kwargs["rms_delay_spread_ns"] is not None:
            self.rms_delay_spread_ns = float(max(0.0, float(kwargs["rms_delay_spread_ns"])))
        if "doppler_hz" in kwargs and kwargs["doppler_hz"] is not None:
            self.doppler_hz = float(kwargs["doppler_hz"])
        if "doppler_spread_hz" in kwargs and kwargs["doppler_spread_hz"] is not None:
            self.doppler_spread_hz = float(max(0.0, float(kwargs["doppler_spread_hz"])))
        if "snr_db" in kwargs and kwargs["snr_db"] is not None:
            self.snr_db = float(kwargs["snr_db"])
        if "seed" in kwargs and kwargs["seed"] is not None:
            self.seed = int(kwargs["seed"]) & 0xFFFFFFFF
        if "normalize_power" in kwargs and kwargs["normalize_power"] is not None:
            self.normalize_power = bool(kwargs["normalize_power"])
        if "num_sinusoids" in kwargs and kwargs["num_sinusoids"] is not None:
            self.num_sinusoids = int(max(4, min(int(kwargs["num_sinusoids"]), 64)))

        self._rng = np.random.default_rng(self.seed)
        raw = list(self.TDL_PROFILES[self.model])
        delays_norm = np.array([r[0] for r in raw], dtype=np.float64)
        powers = np.array([self._db_to_linear(r[1]) for r in raw], dtype=np.float64)
        if self.normalize_power:
            powers = powers / max(float(np.sum(powers)), 1e-12)
        mean_tau = float(np.sum(powers * delays_norm) / max(float(np.sum(powers)), 1e-12))
        rms_norm = float(np.sqrt(np.sum(powers * (delays_norm - mean_tau) ** 2) /
                                 max(float(np.sum(powers)), 1e-12)))
        desired_ds_s = float(self.rms_delay_spread_ns) * 1e-9
        self._delay_scale_s = desired_ds_s / max(rms_norm, 1e-12) if desired_ds_s > 0.0 else 0.0
        delays_samp = delays_norm * self._delay_scale_s * self.sample_rate
        max_delay_samp = float(np.max(delays_samp)) if delays_samp.size else 0.0
        hist_len = int(max(8, np.ceil(max_delay_samp) + 8))
        self._history = np.zeros(hist_len, dtype=np.complex128)
        self._sample_index = 0
        self._components = []
        for idx, (delay_norm, power_db, kind) in enumerate(raw):
            power_lin = self._db_to_linear(power_db)
            if self.normalize_power:
                power_lin = power_lin / max(float(np.sum([self._db_to_linear(r[1]) for r in raw])), 1e-12)
            comp: Dict[str, Any] = {
                "idx": int(idx),
                "delay_norm": float(delay_norm),
                "delay_samp": float(delay_norm) * self._delay_scale_s * self.sample_rate,
                "power_lin": float(power_lin),
                "sqrt_power": float(np.sqrt(max(power_lin, 0.0))),
                "kind": str(kind).lower(),
                "phase0": float(self._rng.uniform(0.0, 2.0 * np.pi)),
            }
            if comp["kind"] == "rayleigh":
                comp["static_rayleigh"] = ((self._rng.normal() + 1j * self._rng.normal()) / np.sqrt(2.0))
                # Deterministic-ish angle grid with a small random offset per tap.
                offset = float(self._rng.uniform(0.0, 1.0))
                comp["angles"] = 2.0 * np.pi * ((np.arange(self.num_sinusoids, dtype=np.float64) + offset)
                                                 / max(self.num_sinusoids, 1))
                comp["phases"] = self._rng.uniform(0.0, 2.0 * np.pi, size=self.num_sinusoids).astype(np.float64)
            self._components.append(comp)

    def reset(self):
        self._history[:] = 0.0
        self._sample_index = 0

    def summary(self) -> str:
        tap_desc = ", ".join(
            f"{c['kind']}@{c['delay_samp']:.3f} samp/{10*np.log10(max(c['power_lin'],1e-15)):.1f} dB"
            for c in self._components
        )
        return (f"{self.DISPLAY_NAMES.get(self.model, self.model)}, "
                f"DS={self.rms_delay_spread_ns:.1f} ns, fd={self.doppler_hz:.1f} Hz, "
                f"spread={self.doppler_spread_hz:.1f} Hz, SNR={self.snr_db:.1f} dB, taps=[{tap_desc}]")

    @staticmethod
    def _fractional_delay(ext: np.ndarray, hist_len: int, n: np.ndarray, delay_samp: float) -> np.ndarray:
        pos = float(hist_len) + n.astype(np.float64) - float(delay_samp)
        i0 = np.floor(pos).astype(np.int64)
        frac = pos - i0.astype(np.float64)
        out = np.zeros(n.size, dtype=np.complex128)
        valid = (i0 >= 0) & (i0 < ext.size)
        if np.any(valid):
            ii = i0[valid]
            ii1 = np.minimum(ii + 1, ext.size - 1)
            ff = frac[valid]
            out[valid] = (1.0 - ff) * ext[ii] + ff * ext[ii1]
        return out

    def _component_gain(self, comp: Dict[str, Any], t: np.ndarray) -> np.ndarray:
        sqrt_power = float(comp.get("sqrt_power", 0.0))
        common = float(self.doppler_hz)
        if comp.get("kind") == "los":
            return sqrt_power * np.exp(1j * (2.0 * np.pi * common * t + float(comp.get("phase0", 0.0))))
        spread = float(self.doppler_spread_hz)
        if spread <= 1e-9:
            return sqrt_power * complex(comp.get("static_rayleigh", 1.0 + 0.0j)) * np.exp(1j * 2.0 * np.pi * common * t)
        angles = np.asarray(comp.get("angles"), dtype=np.float64).reshape(-1)
        phases = np.asarray(comp.get("phases"), dtype=np.float64).reshape(-1)
        freqs = common + spread * np.cos(angles)
        # Shape: (num_sinusoids, num_samples).  The scale gives approximately
        # unit-power Rayleigh fading before multiplying by sqrt_power.
        ph = 2.0 * np.pi * freqs[:, None] * t[None, :] + phases[:, None]
        return sqrt_power * np.sum(np.exp(1j * ph), axis=0) / np.sqrt(max(freqs.size, 1))

    def process(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.complex64).reshape(-1)
        if x.size == 0:
            return np.zeros(0, dtype=np.complex64)
        # Denormal underflow from fractional-delay phasors / tiny gains and the
        # occasional overflow on a high-gain tap are numerically harmless here;
        # silence the spurious NumPy warnings around the synthesis math.
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            x128 = x.astype(np.complex128, copy=False)
            hist_len = int(self._history.size)
            ext = np.concatenate((self._history, x128))
            n = np.arange(x128.size, dtype=np.int64)
            t = (float(self._sample_index) + n.astype(np.float64)) / max(self.sample_rate, 1.0)
            y = np.zeros(x128.size, dtype=np.complex128)
            for comp in self._components:
                delayed = self._fractional_delay(ext, hist_len, n, float(comp.get("delay_samp", 0.0)))
                y += self._component_gain(comp, t) * delayed
            self._history = ext[-hist_len:].copy()
            self._sample_index += int(x128.size)
            if np.isfinite(self.snr_db) and self.snr_db < 200.0:
                sig_power = float(np.mean(np.abs(y) ** 2))
                if sig_power > 1e-18:
                    noise_power = sig_power / (10.0 ** (float(self.snr_db) / 10.0))
                    noise = np.sqrt(noise_power / 2.0) * (
                        self._rng.normal(size=y.size) + 1j * self._rng.normal(size=y.size)
                    )
                    y = y + noise
            return y.astype(np.complex64)


class FDIDMHardwareTest:
    APP_MAGIC = b"MTPK"
    PILOT_SEED = 0xFD1D_0017  # deterministic pilot generator
    DEFAULT_PILOT_SYMBOL = (1.0 + 1.0j) / np.sqrt(2.0)
    ALPHA_BETA_SIGNALING_MODE = "shared_memory"  # Phase 1 only; see class docstring

    def __init__(
            self,
            carrier_freq: float = 2.4e9,
            samp_rate: float = 1_000_000.0,  # v17: 1 MHz is an integer divisor of B210's 52 MHz MCR
            tx_gain: float = 20.0,  # v17: lower default; loopback safety
            rx_gain: float = 20.0,  # v17: lower default; loopback safety
            device_type: str = "USRP B210",
            serial: Optional[str] = None,
            tx_antenna: str = "TX/RX",
            rx_antenna: str = "RX2",
            tx_text: str = "FDIDM OK",
            mod_order: str = "QPSK",
            equalizer: str = "MMSE",
            alpha: float = 0.5,
            beta: float = 1.0,
            fdidm_alpha: Optional[float] = None,
            fdidm_beta: Optional[float] = None,
            fdidm_m: int = 16,
            fdidm_n: int = 16,
            cp_len: int = 4,
            tx_frame_count: int = 4,
            inter_frame_guard_len: int = 64,
            evm_average_frames: int = 8,
            training_amplitude: float = 1.0,
            training_probe_guard_len: int = 16,  # legacy; ignored in v17
            max_full_htf_order: int = 4096,  # maximum order for full paper H_TF estimation
            channel_estimator: str = "full_htf",  # "full_htf" = paper Eq.(20)/(29) matrix receiver; "tdl_param" = software-TDL basis; "diag_tf" = fast diagnostic mode
            full_htf_update_interval_frames: int = 10000,  # kept for backward compatibility; one-shot mode ignores it
            full_htf_once: bool = True,
            process_interval_ms: float = 200.0,      # throttle heavy Python decoding to avoid UHD RX overflow
            usrp_buffer_frames: int = 2048,          # UHD streamer buffer hint for Windows/B210 stability
            tx_min_waveform_duration_ms: float = 500.0,  # repeat the FDIDM super-cycle into a long UHD-friendly vector
            tx_max_waveform_samples: int = 1_048_576,    # cap the Python list handed to vector_source_c
            tx_prerender_tdl_before_rf: bool = True,     # kept for API compatibility; TDL->RF is always pre-rendered
            enable_realtime_scheduling: bool = True,
            cfo_search_enable: bool = True,          # resolve repeated-preamble CFO aliases for high Doppler
            cfo_search_max_hz: float = 50_000.0,     # scan range for common Doppler / oscillator offset
            residual_cfo_max_hz: float = 5_000.0,    # sanity limit for pilot residual CFO refinement
            startup_settle_ms: float = 800.0,        # drop dirty RX/probe windows after every USRP start
            startup_settle_windows: int = 3,         # additional fresh probe vectors to ignore after start/reconfigure
            cfo_scan_min_score: float = 0.55,        # non-alias CFO must score at least this well without a lock/hint
            cfo_scan_jump_guard_hz: float = 12_000.0,# reject low-score CFO jumps far from last good CFO/hint
            coding_scheme: str = "conv12",          # "none" or "conv12" (rate-1/2 convolutional + interleaver)
            coding_interleaver: bool = True,
            channel_mode: str = "rf",               # RF-only, RF->TDL, or TDL->RF; baseband-only loopback is disabled
            software_channel_model: Optional[str] = None,  # legacy alias; baseband-only TDL names map to RF->TDL
            log_to_stdout: bool = False,
            log_file_path: Optional[str] = None,
            debug_log_max_entries: int = 5000,
            tdl_rms_delay_spread_ns: float = 1000.0,
            tdl_doppler_hz: float = 0.0,             # common satellite Doppler shift applied to all taps
            tdl_doppler_spread_hz: float = 0.0,      # local scattering Doppler spread for Rayleigh taps
            tdl_snr_db: float = 35.0,
            tdl_seed: int = 0x38811,
            tdl_normalize_power: bool = True,
            # v28: parametric TDL receiver. It estimates path gains from one dense TF pilot,
            # then reconstructs H_TF from a known TDL-A/C/D delay/Doppler basis.
            tdl_param_num_sinusoids: int = 8,
            tdl_param_max_paths: int = 96,
            tdl_param_ridge: float = 1e-7,
            tdl_param_prune_db: float = -90.0,
            auto_tdl_param_for_software: bool = True,
            # Paper-guided, channel-adaptive alpha/beta control.  The optimizer
            # uses the measured H_TF and noise variance, evaluates the paper's
            # ZF/MMSE SER objective over alpha,beta in [0, 2], and publishes a
            # recommendation.  The UI applies it in its own thread so the RX
            # monitor never performs a live GNU Radio waveform swap.
            adaptive_alpha_beta_enable: bool = False,
            adaptive_alpha_beta_coarse_step: float = 0.25,
            adaptive_alpha_beta_fine_step: float = 0.05,
            adaptive_alpha_beta_interval_frames: int = 8,
            adaptive_alpha_beta_min_improvement_db: float = 0.5,
            adaptive_alpha_beta_stability_evals: int = 2,
            adaptive_alpha_beta_cooldown_frames: int = 16,
            adaptive_alpha_beta_integer_margin_db: float = 0.10,
            adaptive_alpha_beta_max_order: int = 512,
            adaptive_alpha_beta_min_sync_metric: float = 0.30,
            adaptive_alpha_beta_require_good_frame: bool = False,
            adaptive_alpha_beta_rcond: float = 1e-6,
            **_legacy_ignored: Any,
    ):
        self.carrier_freq = float(carrier_freq)
        self.sample_rate = float(samp_rate)
        self.samp_rate = self.sample_rate
        self.tx_gain = float(tx_gain)
        self.rx_gain = float(rx_gain)
        self.device_type = str(device_type)
        self.serial = serial
        self.tx_antenna = str(tx_antenna)
        self.rx_antenna = str(rx_antenna)

        self.M = int(max(4, min(int(fdidm_m), 64)))
        self.N = int(max(1, min(int(fdidm_n), 64)))
        self.cp_len = int(max(0, min(int(cp_len), max(self.M - 1, 0))))
        self.alpha = float(alpha if fdidm_alpha is None else fdidm_alpha)
        self.beta = float(beta if fdidm_beta is None else fdidm_beta)
        self.mod_order = str(mod_order).upper()
        self.equalizer = str(equalizer).upper()
        if self.mod_order not in ("QPSK", "16QAM", "64QAM"):
            raise ValueError(f"Unsupported modulation: {self.mod_order}")
        if self.equalizer not in ("ZF", "MMSE"):
            raise ValueError(f"Unsupported equalizer: {self.equalizer}")
        self.bits_per_symbol = self._get_bits_per_symbol(self.mod_order)
        self.subcarrier_spacing = self.sample_rate / max(self.M, 1)

        self.tx_frame_count = int(max(1, min(int(tx_frame_count), 32)))
        self.inter_frame_guard_len = int(max(0, min(int(inter_frame_guard_len), 8192)))
        self.evm_average_frames = int(max(1, min(int(evm_average_frames), 128)))
        self.training_amplitude = float(max(0.05, min(float(training_amplitude), 4.0)))

        # Legacy parameters kept only so old UI calls do not break.
        self.training_probe_guard_len = int(max(0, min(int(training_probe_guard_len), 8192)))
        self.max_full_htf_order = int(max(16, max_full_htf_order))
        # Requested and effective estimators are tracked separately.
        # self.channel_estimator is always the backend-effective source of truth.
        self.requested_channel_estimator = self._normalize_channel_estimator(channel_estimator or "full_htf")
        self.channel_estimator = self.requested_channel_estimator
        self.estimator_effective_reason = "as_requested"
        self._estimator_forced_reason = ""
        self.full_htf_update_interval_frames = int(max(1, min(int(full_htf_update_interval_frames), 10_000)))
        self.full_htf_once = bool(full_htf_once)
        self.process_interval_sec = max(0.03, float(process_interval_ms) / 1000.0)
        self.usrp_buffer_frames = int(max(32, min(int(usrp_buffer_frames), 4096)))
        self.tx_min_waveform_duration_ms = float(max(0.0, min(float(tx_min_waveform_duration_ms), 5000.0)))
        self.tx_max_waveform_samples = int(max(8192, min(int(tx_max_waveform_samples), 4_194_304)))
        self.tx_prerender_tdl_before_rf = True
        self.enable_realtime_scheduling = bool(enable_realtime_scheduling)
        self.cfo_search_enable = bool(cfo_search_enable)
        self.cfo_search_max_hz = float(max(0.0, min(float(cfo_search_max_hz), max(self.sample_rate / 2.0, 1.0))))
        self.residual_cfo_max_hz = float(max(0.0, min(float(residual_cfo_max_hz), max(self.sample_rate / 4.0, 1.0))))
        self._last_cfo_alias_hz = float("nan")
        self._last_cfo_scan_score = float("nan")
        self._last_cfo_unambiguous_hz = float("nan")
        self.startup_settle_sec = float(max(0.0, min(float(startup_settle_ms) / 1000.0, 5.0)))
        self.startup_settle_windows = int(max(0, min(int(startup_settle_windows), 32)))
        self.cfo_scan_min_score = float(max(0.0, min(float(cfo_scan_min_score), 1.0)))
        self.cfo_scan_jump_guard_hz = float(max(0.0, min(float(cfo_scan_jump_guard_hz), max(self.sample_rate / 2.0, 1.0))))
        self._rx_settle_until_wall = 0.0
        self._rx_settle_windows_remaining = 0
        self._run_id = 0
        self._last_good_cfo_hz = float("nan")
        self._last_good_cfo_wall = 0.0
        self._last_good_cfo_mode_key = ""
        self.log_to_stdout = bool(log_to_stdout)
        self.log_file_path = str(log_file_path or "")
        self.debug_log_max_entries = int(max(500, min(int(debug_log_max_entries), 100000)))
        self._channel_mode_note = ""
        self.coding_scheme = self._normalize_coding_scheme(coding_scheme)
        self.coding_interleaver = bool(coding_interleaver)
        self._tx_base_cycle_len = 0
        self._tx_uhd_repeats = 1
        self._tx_tdl_prerendered = False
        self._tx_coded_frame_bits = np.zeros(0, dtype=np.int8)
        self._tx_coded_bits_len = 0
        self._tx_uncoded_bits_len = 0
        self._last_fec_bit_ber = float("nan")
        self._last_raw_bit_ber = float("nan")

        # Channel path. Every supported mode now traverses the real USRP RF path.
        # Baseband-only TDL loopback was removed because it does not exercise
        # the actual hardware/channel used by this bench.
        self.channel_mode = self._normalize_channel_mode(software_channel_model or channel_mode)
        self.tdl_rms_delay_spread_ns = float(max(0.0, float(tdl_rms_delay_spread_ns)))
        self.tdl_doppler_hz = float(tdl_doppler_hz)
        self.tdl_doppler_spread_hz = float(max(0.0, float(tdl_doppler_spread_hz)))
        self.tdl_snr_db = float(tdl_snr_db)
        self.tdl_seed = int(tdl_seed) & 0xFFFFFFFF
        self.tdl_normalize_power = bool(tdl_normalize_power)
        self.auto_tdl_param_for_software = bool(auto_tdl_param_for_software)
        self.tdl_param_num_sinusoids = int(max(4, min(int(tdl_param_num_sinusoids), 64)))
        self.tdl_param_max_paths = int(max(1, min(int(tdl_param_max_paths), 512)))
        self.tdl_param_ridge = float(max(0.0, float(tdl_param_ridge)))
        self.tdl_param_prune_db = float(min(0.0, float(tdl_param_prune_db)))

        # Channel-adaptive alpha/beta optimizer configuration.  The search
        # resolution follows Section VI-B of the paper; coarse-to-fine search
        # keeps the real-time hardware path responsive while retaining a fine
        # step below 0.1 by default.
        self.adaptive_alpha_beta_enable = bool(adaptive_alpha_beta_enable)
        self.adaptive_alpha_beta_coarse_step = float(max(0.05, min(float(adaptive_alpha_beta_coarse_step), 1.0)))
        self.adaptive_alpha_beta_fine_step = float(max(0.01, min(float(adaptive_alpha_beta_fine_step), self.adaptive_alpha_beta_coarse_step)))
        self.adaptive_alpha_beta_interval_frames = int(max(1, min(int(adaptive_alpha_beta_interval_frames), 1024)))
        self.adaptive_alpha_beta_min_improvement_db = float(max(0.0, min(float(adaptive_alpha_beta_min_improvement_db), 30.0)))
        self.adaptive_alpha_beta_stability_evals = int(max(1, min(int(adaptive_alpha_beta_stability_evals), 16)))
        self.adaptive_alpha_beta_cooldown_frames = int(max(0, min(int(adaptive_alpha_beta_cooldown_frames), 4096)))
        self.adaptive_alpha_beta_integer_margin_db = float(max(0.0, min(float(adaptive_alpha_beta_integer_margin_db), 6.0)))
        self.adaptive_alpha_beta_max_order = int(max(16, min(int(adaptive_alpha_beta_max_order), 4096)))
        self.adaptive_alpha_beta_min_sync_metric = float(max(0.0, min(float(adaptive_alpha_beta_min_sync_metric), 1.0)))
        self.adaptive_alpha_beta_require_good_frame = bool(adaptive_alpha_beta_require_good_frame)
        self.adaptive_alpha_beta_rcond = float(max(1e-12, min(float(adaptive_alpha_beta_rcond), 1e-1)))

        self._estimator_auto_note = ""
        self._resolve_effective_channel_estimator()

        # Hardware/sync frame structure.
        # Frame = [pre_guard][sync_preamble][pilot_frame][data_frame][post_guard]
        self._recompute_strict_frame_timing()
        self.strict_chain_name = "FDIDM_HW_ONLY_v35_CHANNEL_ADAPTIVE_AB"

        # Tunables.
        self.sync_metric_threshold = 0.30
        self.update_period = 0.10
        self._rng_seed = 20260522

        # Cached transforms.
        self._gamma_cache: Dict[Tuple[int, float], np.ndarray] = {}
        self._tdl_param_basis_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._last_tdl_param_gains = np.zeros(0, dtype=np.complex128)
        self._last_tdl_param_paths: List[Dict[str, Any]] = []
        self._last_tdl_param_fit_nmse = float("nan")
        self.last_tdl_param_path_count = 0
        self.last_tdl_param_rank = 0
        self.last_tdl_param_cond = float("nan")

        # Cached random pilot grid X_pilot and its X_TF_pilot (deterministic via PILOT_SEED).
        self._pilot_X_cross: np.ndarray = np.zeros((self.M, self.N), dtype=np.complex128)
        self._pilot_X_tf: np.ndarray = np.zeros((self.M, self.N), dtype=np.complex128)
        self._rebuild_pilot_matrices()

        # Runtime buffers/state.
        self._lock = threading.Lock()
        self._status = "idle"
        self._last_error = ""
        self._last_info = ""
        self._running = False
        self._tb = None
        self._monitor_thread = None
        self._monitor_stop = threading.Event()
        self._usrp_args = self._build_device_args()
        self._gr = None
        self._blocks = None
        self._uhd = None
        self._import_runtime()

        self._buffer_keep = max(262144, 8 * self.frame_len)
        self._tx_buffer = _SampleRing(self._buffer_keep)
        # RX is captured through a GNU Radio C++ probe/vector chain in v20.
        # The Python ring remains only as a fallback for old GNU Radio builds.
        self._rx_buffer = _SampleRing(self._buffer_keep)
        self._rx_probe = None
        self._rx_stream_to_vector = None
        self._tdl_channel_block = None
        self._throttle_block = None
        self._tx_gain_block = None
        self._rx_probe_len = 0
        self._rx_probe_mode = "unconfigured"
        self._rx_probe_last_fp = None
        self._rx_probe_total_est = 0
        self._rx_probe_start_t = 0.0
        # Full-H_TF cache: first accepted CSI is reused until cleared or reconfigured.
        self._cached_htf_full: Optional[np.ndarray] = None
        self._cached_htf_leakage = float("nan")
        self._cached_htf_frame_counter = -10**18
        self._cached_H_cross: Optional[np.ndarray] = None
        self._cached_Hh_cross: Optional[np.ndarray] = None
        self._cached_HhH_cross: Optional[np.ndarray] = None
        self._cached_Phi: Optional[np.ndarray] = None
        self._cached_H_cond_proxy = float("nan")
        self._full_htf_estimates = 0
        self._last_process_t = 0.0
        self._tx_preview_start_t = time.time()
        self._tx_cycle_frame_count = int(self.tx_frame_count)
        # Only good frames are allowed to update plots/cache. This prevents a
        # momentary bad sync or noisy full-H estimate from making the constellation
        # appear stable for a while and then suddenly scatter.
        self.constellation_soft_ber_threshold = 5e-3
        self.constellation_soft_evm_threshold = 35.0
        self._latest_tx_samples = np.zeros(4096, dtype=np.complex64)
        self._latest_rx_samples = np.zeros(4096, dtype=np.complex64)
        self._latest_rx_frame_samples = np.zeros(0, dtype=np.complex64)
        self._latest_rx_data_samples = np.zeros(0, dtype=np.complex64)
        self._latest_rx_pilot_samples = np.zeros(0, dtype=np.complex64)
        # RX spectrum is intentionally driven by raw IQ samples, not by decoder success.
        # These counters make stale-but-still-visible spectrum traces explicit in the UI/status.
        self._rx_last_new_samples = 0
        self._rx_stream_updates = 0
        self._rx_latest_window_len = 0
        self._rx_last_update_wall = 0.0
        self._rx_spectrum_stale = True
        self._rx_spectrum_stale_sec = float("inf")
        # Constellation buffers.  _latest_constellation is POST-equalizer soft symbols
        # after residual gain/phase correction; _latest_constellation_pre_eq is the
        # cross-domain observation BEFORE equalization (Phi*y_TF or FDIT(Y_TF)).
        # Publish these for every decodable frame candidate, even when CRC fails,
        # so the plot can be used as a diagnostic instead of disappearing silently.
        self._latest_constellation = np.zeros(0, dtype=np.complex64)
        self._latest_constellation_post_eq = np.zeros(0, dtype=np.complex64)
        self._latest_constellation_post_eq_raw = np.zeros(0, dtype=np.complex64)
        self._latest_constellation_pre_eq = np.zeros(0, dtype=np.complex64)
        self._latest_constellation_tf = np.zeros(0, dtype=np.complex64)
        self._last_good_constellation = np.zeros(0, dtype=np.complex64)
        self._last_constellation_is_good = False
        self._last_constellation_source = "none"
        self.last_constellation_source = "none"
        self.last_constellation_points = 0
        self.last_constellation_quality = "no_frame_yet"
        self.constellation_display_mode = "post_equalized"

        self._tx_text = ""
        self._tx_payload = b""
        self._tx_frame = b""
        self._tx_frame_bits = np.zeros(0, dtype=np.int8)
        self._tx_bits_frame = np.zeros(0, dtype=np.int8)
        self._tx_bits_frames: List[np.ndarray] = []
        self._tx_x_cross = np.zeros((self.M, self.N), dtype=np.complex128)
        self._tx_x_tf = np.zeros((self.M, self.N), dtype=np.complex128)
        self._tx_waveform = np.zeros(1, dtype=np.complex64)
        self._rx_text = ""
        self._decode_ok = False
        self._match_bytes = 0
        self._last_good_rx_payload = b""
        self._last_raw_bytes = b""
        self._ber_estimate = float("nan")

        self.last_sync_index = 0
        self.last_payload_start = 0
        self.last_cfo_est_hz = 0.0
        self.last_cfo_preamble_hz = 0.0
        self.last_cfo_source = "preamble"
        self.last_sync_metric = 0.0
        self.last_frame_ok = False
        self.last_bad_reason = "init"
        self.last_htf_nmse = 0.0
        self.last_cond_h_cross = float("nan")
        self.last_equalizer_warning = ""
        self.last_evm_instant_percent = float("nan")
        self.last_evm_average_percent = float("nan")
        self.last_residual_gain_abs = float("nan")
        self.last_residual_phase_deg = float("nan")
        self.last_noise_var = float("nan")
        self._evm_history: deque = deque(maxlen=self.evm_average_frames)
        self._rx_samples_seen = 0
        self._last_processed_abs_start = -10 ** 18
        self._t0 = time.time()
        self._ber_hist_t: deque = deque(maxlen=200)
        self._ber_hist_v: deque = deque(maxlen=200)

        # Alpha/Beta performance surface data consumed by the UI.  It records
        # only metrics measured by the actual RX pipeline after each processed
        # frame; no synthetic sweep values are injected.  The context key below
        # excludes alpha/beta but includes modulation, channel, gain, coding,
        # estimator, and TDL settings so the surface is cleared automatically
        # when points would no longer be comparable.
        self._ab_surface_quant_digits = 3
        self._ab_surface_max_cells = 2048
        self._ab_surface_samples_per_cell = int(max(1, min(int(self.evm_average_frames), 128)))
        self._ab_metric_history: Dict[Tuple[float, float], Dict[str, Any]] = {}
        self._ab_surface_context_key = self._alpha_beta_surface_context_key()

        # Runtime diagnostics consumed by the UI and optional Python logging handlers.
        self._debug_log: deque = deque(maxlen=int(self.debug_log_max_entries))
        self._py_logger = self._create_python_logger()
        self._debug_seq = 0
        self._frames_processed = 0  # successful frame attempts (any sync, even bad ber)
        self._frames_decode_ok = 0  # CRC-passing frames
        self._monitor_cycles = 0  # alive counter for the worker thread
        self._monitor_last_log_t = 0.0
        self._needs_top_block_rebuild = False

        # Alpha/beta optimization is deliberately isolated from the GNU Radio
        # monitor thread.  The monitor only snapshots H_TF; a daemon worker
        # computes the paper SER search, and the UI later applies a stable
        # recommendation.
        self._adaptive_ab_lock = threading.RLock()
        self._adaptive_ab_event = threading.Event()
        self._adaptive_ab_stop = threading.Event()
        self._adaptive_ab_thread = None
        self._adaptive_ab_snapshot = None
        self._adaptive_ab_last_snapshot = None
        self._adaptive_ab_snapshot_seq = 0
        self._adaptive_ab_eval_seq = 0
        self._adaptive_ab_last_queued_frame = -10**18
        self._adaptive_ab_last_applied_frame = -10**18
        self._adaptive_ab_last_htf_identity = None
        self._adaptive_ab_force_next = False
        self._adaptive_ab_state = "idle" if self.adaptive_alpha_beta_enable else "disabled"
        self._adaptive_ab_last_error = ""
        self._adaptive_ab_stable_key = None
        self._adaptive_ab_stable_count = 0
        self._adaptive_ab_recommendation: Dict[str, Any] = {}

        self._set_tx_text_internal(tx_text)
        self._build_top_block()
        self._debug("INFO",
                    f"FDIDM backend v35 ready: chain={self.strict_chain_name}, "
                    f"estimator={self.channel_estimator} (requested={getattr(self, 'requested_channel_estimator', self.channel_estimator)}), use_full_htf={self.use_full_htf}, "
                    f"H_once={self.full_htf_once}, H_update_legacy={self.full_htf_update_interval_frames} frame(s), "
                    f"channel_mode={self.channel_mode}, "
                    f"TDL_DS={self.tdl_rms_delay_spread_ns:.1f} ns, fd={self.tdl_doppler_hz:.1f} Hz, "
                    f"spread={self.tdl_doppler_spread_hz:.1f} Hz, SNR={self.tdl_snr_db:.1f} dB, "
                    f"process_interval={self.process_interval_sec*1000:.0f} ms, "
                    f"M={self.M} N={self.N} CP={self.cp_len} alpha={self.alpha:.3f} beta={self.beta:.3f} "
                    f"mod={self.mod_order} eq={self.equalizer} coding={self._coding_summary()} "
                    f"Fs={self.sample_rate:.0f} Hz frame_len={self.frame_len} "
                    f"({self.frame_len / max(self.sample_rate, 1) * 1000.0:.2f} ms), "
                    f"tx_vector={self._tx_waveform.size} samp, base_cycle={self._tx_base_cycle_len}, "
                    f"repeats={self._tx_uhd_repeats}, prerender_tdl={self._tx_tdl_prerendered}")
        if getattr(self, "_estimator_auto_note", ""):
            self._debug("WARN", self._estimator_auto_note)
        if getattr(self, "_channel_mode_note", ""):
            self._debug("WARN", self._channel_mode_note)
        self._debug("INFO", self._format_link_limit_summary())
        if self.adaptive_alpha_beta_enable:
            self._ensure_alpha_beta_adaptation_worker()
            self._debug("INFO",
                        "channel-adaptive alpha/beta enabled: paper SER objective, "
                        f"coarse={self.adaptive_alpha_beta_coarse_step:.2f}, "
                        f"fine={self.adaptive_alpha_beta_fine_step:.2f}, "
                        f"interval={self.adaptive_alpha_beta_interval_frames} frame(s)")


    # =========================================================
    # Frame timing
    # =========================================================
    def _recompute_strict_frame_timing(self):
        self.pre_guard_len = max(16, self.M)
        self.sync_half_len = max(32, self.M)
        self.sync_len = 2 * self.sync_half_len
        self.block_len = self.M + self.cp_len
        self.data_frame_len = self.N * self.block_len

        # Receiver training modes:
        #   full_htf  : MN one-hot TF probes, directly estimates H_TF columns.
        #   tdl_param : one dense TF pilot, estimate P TDL path gains and synthesize H_TF.
        #   diag_tf   : one dense TF pilot, diagonal TF sanity-check.
        self.full_htf_order = int(self.M * self.N)
        self.use_tdl_param_htf = (self.channel_estimator == "tdl_param")
        self.use_full_htf = (self.channel_estimator == "full_htf" and
                             self.full_htf_order <= int(self.max_full_htf_order))
        if self.channel_estimator == "full_htf" and not self.use_full_htf:
            # Keep building so the UI can show a clear warning, but use the
            # fast one-frame diagonal estimator to avoid allocating a huge matrix.
            self.htf_training_blocks = 1
        elif self.use_full_htf:
            self.htf_training_blocks = self.full_htf_order
        else:
            # tdl_param and diag_tf both need one pilot frame only.
            self.htf_training_blocks = 1

        self.pilot_frame_len = self.htf_training_blocks * self.data_frame_len
        self.post_guard_len = max(self.cp_len + 32, self.M)

        self._off_sync = self.pre_guard_len
        self._off_pilot = self._off_sync + self.sync_len
        self._off_data = self._off_pilot + self.pilot_frame_len
        self._off_end = self._off_data + self.data_frame_len
        self.frame_len = self._off_end + self.post_guard_len

        # Build sync preamble once (deterministic Zadoff-Chu-like chirp, two halves).
        self.sync_preamble = self._build_sync_preamble(self.sync_half_len)
        self._sync_energy = float(np.vdot(self.sync_preamble, self.sync_preamble).real) + 1e-12

    def _rebuild_pilot_matrices(self):
        # v18: CONSTANT-MODULUS pilot defined DIRECTLY in the TF domain.
        # Rationale: the v17 pilot was a fixed QPSK grid in the CROSS domain;
        # after IFDIT it became non-constant-modulus in TF, so for OTFS /
        # fractional indices some TF cells had |X_TF| ~ 0.03. Per-cell channel
        # estimation Y_TF/X_TF then amplified noise on those weak cells, making
        # OTFS/fractional decode *worse* than OFDM on an otherwise flat link
        # (exactly the close-antenna OTA case here). Defining the pilot in TF
        # with unit modulus gives every cell the same estimation SNR,
        # independent of alpha/beta.
        rng = np.random.default_rng(self.PILOT_SEED ^ (self.M * 131 + self.N))
        # Random QPSK phases -> unit modulus in every TF cell.
        quad = rng.integers(0, 4, size=(self.M, self.N))
        phase = (np.pi / 4.0) + (np.pi / 2.0) * quad.astype(np.float64)
        x_tf = np.exp(1j * phase).astype(np.complex128)
        self._pilot_X_tf = (self.training_amplitude * x_tf).astype(np.complex128)
        # Cross-domain equivalent kept only for diagnostics / back-compat
        # (FDIT is the exact inverse of the IFDIT used on the data path).
        self._pilot_X_cross = self._fdit(self._pilot_X_tf)

    # =========================================================
    # Estimator selection helpers
    # =========================================================
    @staticmethod
    def _normalize_channel_estimator(estimator: str) -> str:
        e = str(estimator or "full_htf").strip().lower().replace("-", "_").replace(" ", "_")
        if e in ("tdl", "tdl_param", "tdl_path", "tdl_paths", "tdl_parametric", "tdl_param_est", "tdl_parameter", "tdl_parameters"):
            return "tdl_param"
        if e in ("full_h", "full_h_tf", "fullhtf", "full_htf", "paper", "paper_strict"):
            return "full_htf"
        if e in ("diag", "diag_tf", "diagonal", "diagonal_tf", "fast"):
            return "diag_tf"
        raise ValueError("channel_estimator must be 'full_htf', 'diag_tf', or 'tdl_param'")

    def _resolve_effective_channel_estimator(self) -> bool:
        """Resolve requested estimator into the executable backend mode.

        Grounding:
        - full_htf implements the paper Eq.(20)/(29) matrix receiver already
          present in this codebase;
        - tdl_param is a compact receiver for the known software-TDL basis and
          is not the paper's general H_TF measurement;
        - diag_tf is retained as the existing fast diagnostic fallback when a
          requested full H_TF order exceeds max_full_htf_order.
        """
        requested = self._normalize_channel_estimator(
            getattr(self, "requested_channel_estimator", getattr(self, "channel_estimator", "full_htf"))
        )
        old_effective = str(getattr(self, "channel_estimator", requested))
        effective = requested
        reason = "as_requested"
        K = int(max(1, int(getattr(self, "M", 1)) * int(getattr(self, "N", 1))))
        max_k = int(max(16, int(getattr(self, "max_full_htf_order", 4096))))
        rf_path = bool(self._rf_path_enabled()) if hasattr(self, "_rf_path_enabled") else False
        software_tdl = bool(self._software_channel_enabled()) if hasattr(self, "_software_channel_enabled") else False

        if requested == "full_htf" and K > max_k:
            effective = "diag_tf"
            reason = f"requested full_htf but M*N={K} exceeds max_full_htf_order={max_k}; effective estimator is diag_tf"
        elif requested == "tdl_param" and rf_path:
            # Every supported v33/v34 mode traverses real RF. A pure software-TDL
            # basis is incomplete for that composite channel; use the paper H_TF
            # measurement if the configured order allows it.
            if K <= max_k:
                effective = "full_htf"
                reason = (f"requested tdl_param on a path containing real RF; tdl_param models only the known software TDL basis "
                          f"(software_tdl={software_tdl}), while the paper Eq.(20)/(29) receiver is full_htf")
            else:
                effective = "diag_tf"
                reason = (f"requested tdl_param on real RF path but full H_TF order M*N={K} exceeds max_full_htf_order={max_k}; "
                          "effective estimator is diag_tf")
        elif requested == "tdl_param":
            reason = "tdl_param uses the configured software TDL basis; full_htf is the paper Eq.(20)/(29) matrix receiver"

        self.requested_channel_estimator = requested
        self.channel_estimator = effective
        self.estimator_effective_reason = reason
        self._estimator_forced_reason = "" if effective == requested else reason
        self._estimator_auto_note = "" if reason == "as_requested" else reason
        return old_effective != effective

    # =========================================================
    # Parameter limit / link budget helpers
    # =========================================================
    @staticmethod
    def _tdl_profile_rms_norm_and_ratio(model: str) -> Tuple[float, float]:
        raw = list(_NTNTDLChannel.TDL_PROFILES[_NTNTDLChannel.normalize_model(model)])
        delays = np.array([r[0] for r in raw], dtype=np.float64)
        powers = np.array([10.0 ** (float(r[1]) / 10.0) for r in raw], dtype=np.float64)
        powers = powers / max(float(np.sum(powers)), 1e-12)
        mean_tau = float(np.sum(powers * delays))
        rms_norm = float(np.sqrt(np.sum(powers * (delays - mean_tau) ** 2)))
        max_norm = float(np.max(delays)) if delays.size else 0.0
        ratio = max_norm / max(rms_norm, 1e-12) if max_norm > 0.0 else 0.0
        return rms_norm, ratio

    def compute_parameter_limits(self) -> Dict[str, Any]:
        """Compute analytic first-order limits for the current FDIDM settings.

        These are not guaranteed BER thresholds; they are engineering budgets
        derived from the actual frame timing and estimator assumptions:
          - common CFO/Doppler is limited by the repeated preamble alias period
            and by the configured v33 CFO scan window;
          - Doppler spread is limited by channel variation across the pilot/data
            observation time for diag-TF, and by the much longer MN-probe time
            for full-H_TF;
          - delay spread is limited by the CP relative to the selected TDL
            profile's maximum-delay/RMS-delay ratio.
        """
        Fs = float(max(self.sample_rate, 1.0))
        M = int(max(self.M, 1))
        N = int(max(self.N, 1))
        cp = int(max(self.cp_len, 0))
        block = int(max(self.block_len, M + cp))
        K = int(max(self.full_htf_order, M * N))
        t_block = float(block) / Fs
        t_data = float(self.data_frame_len) / Fs
        t_diag_obs = float(self.pilot_frame_len + self.data_frame_len) / Fs if not self.use_full_htf else 2.0 * t_data
        t_full_probe = float(K * self.data_frame_len + self.data_frame_len) / Fs
        cfo_unamb = self._preamble_cfo_unambiguous_hz()
        cfo_period = self._preamble_cfo_period_hz()
        cfo_scan = min(float(getattr(self, "cfo_search_max_hz", cfo_unamb)), 0.49 * Fs) if bool(getattr(self, "cfo_search_enable", True)) else cfo_unamb
        subcarrier_spacing = Fs / max(M, 1)
        # Strict/loose phase drift budgets across the relevant observation.
        theta_strict = 0.35  # rad, roughly <= -9 dB phase-error EVM contribution
        theta_loose = 0.75   # rad, FEC may still recover but EVM will be high
        diag_spread_strict = theta_strict / (2.0 * np.pi * max(t_diag_obs, 1e-12))
        diag_spread_loose = theta_loose / (2.0 * np.pi * max(t_diag_obs, 1e-12))
        full_spread_strict = theta_strict / (2.0 * np.pi * max(t_full_probe, 1e-12))
        full_spread_loose = theta_loose / (2.0 * np.pi * max(t_full_probe, 1e-12))
        t_cp = float(cp) / Fs
        ds_limits = {}
        for model in ("tdl_a", "tdl_c", "tdl_d"):
            _rms_norm, ratio = self._tdl_profile_rms_norm_and_ratio(model)
            if ratio <= 0.0:
                strict_ns = float("inf")
            else:
                strict_ns = (t_cp / ratio) * 1e9
            ds_limits[model] = {
                "max_rms_delay_spread_ns_cp": float(strict_ns),
                "recommended_rms_delay_spread_ns_80pct_cp": float(0.80 * strict_ns),
                "max_delay_over_rms_ratio": float(ratio),
            }
        return {
            "Fs": Fs,
            "M": M,
            "N": N,
            "CP": cp,
            "subcarrier_spacing_hz": float(subcarrier_spacing),
            "block_len_samples": int(block),
            "block_time_s": float(t_block),
            "data_time_s": float(t_data),
            "diag_observation_time_s": float(t_diag_obs),
            "full_htf_probe_time_s": float(t_full_probe),
            "sync_half_len": int(self.sync_half_len),
            "cfo_alias_period_hz": float(cfo_period),
            "cfo_unambiguous_hz": float(cfo_unamb),
            "cfo_scan_enabled": bool(getattr(self, "cfo_search_enable", True)),
            "cfo_scan_max_hz": float(cfo_scan),
            "common_doppler_practical_hz": float(0.80 * cfo_scan),
            "residual_cfo_target_hz": float(0.05 * subcarrier_spacing),
            "diag_doppler_spread_strict_hz": float(diag_spread_strict),
            "diag_doppler_spread_loose_hz": float(diag_spread_loose),
            "full_htf_doppler_spread_strict_hz": float(full_spread_strict),
            "full_htf_doppler_spread_loose_hz": float(full_spread_loose),
            "tdl_delay_spread_limits": ds_limits,
        }

    def _format_link_limit_summary(self) -> str:
        lim = self.compute_parameter_limits()
        ds = lim["tdl_delay_spread_limits"]
        return (
            "v33 limits: "
            f"CFO common unamb=±{lim['cfo_unambiguous_hz']:.1f}Hz, "
            f"scan=±{lim['cfo_scan_max_hz']:.1f}Hz, practical≈±{lim['common_doppler_practical_hz']:.1f}Hz; "
            f"residual target<{lim['residual_cfo_target_hz']:.1f}Hz; "
            f"Doppler spread diag strict/loose≈{lim['diag_doppler_spread_strict_hz']:.1f}/{lim['diag_doppler_spread_loose_hz']:.1f}Hz, "
            f"full-H strict/loose≈{lim['full_htf_doppler_spread_strict_hz']:.2f}/{lim['full_htf_doppler_spread_loose_hz']:.2f}Hz; "
            f"RMS-DS max by CP A/C/D≈{ds['tdl_a']['max_rms_delay_spread_ns_cp']:.0f}/"
            f"{ds['tdl_c']['max_rms_delay_spread_ns_cp']:.0f}/"
            f"{ds['tdl_d']['max_rms_delay_spread_ns_cp']:.0f}ns"
        )

    def _build_full_htf_training_waveform(self) -> np.ndarray:
        """Build MN TF-basis probe frames to estimate the full H_TF matrix.

        The kth probe has a single nonzero X_TF cell in column-wise vec order:
        k = m + n*M.  After Wigner transform at the receiver, the observed
        vec(Y_TF) is the kth column of H_TF times the known probe amplitude.
        This is the direct measurement form of the paper's Eq. (20).
        """
        blocks = []
        # A one-bin TF probe with amplitude sqrt(M) has approximately unit
        # time-domain magnitude in its active OFDM symbol, which gives usable
        # training SNR without changing the estimated H because we divide by it.
        amp = self.training_amplitude * np.sqrt(max(self.M, 1))
        for k in range(self.full_htf_order):
            x_vec = np.zeros(self.full_htf_order, dtype=np.complex128)
            x_vec[k] = amp
            x_tf = x_vec.reshape((self.M, self.N), order="F")
            blocks.append(self._heisenberg(x_tf))
        return np.concatenate(blocks).astype(np.complex128) if blocks else np.zeros(0, dtype=np.complex128)


    # =========================================================
    # PHY coding / interleaving helpers
    # =========================================================
    @staticmethod
    def _normalize_coding_scheme(scheme: str) -> str:
        s = str(scheme or "none").strip().lower().replace("-", "_").replace(" ", "_")
        if s in ("none", "off", "uncoded", "no", "false", "0"):
            return "none"
        if s in ("conv12", "conv_12", "conv1_2", "conv_1_2", "viterbi", "cc12", "rate12"):
            return "conv12"
        raise ValueError("coding_scheme must be 'none' or 'conv12'")

    @staticmethod
    def _parity_u32(x: int) -> int:
        return int(int(x).bit_count() & 1)

    @classmethod
    def _conv_next_state_output(cls, memory: int, bit: int) -> Tuple[int, Tuple[int, int]]:
        """K=7, rate-1/2 convolutional encoder trellis.

        The current input bit is bit0 of the shift register and older bits are
        shifted upward.  The Viterbi decoder below uses the exact same trellis,
        so the bit order is self-consistent even though different textbooks may
        print the NASA 171/133 polynomials in the opposite visual direction.
        """
        K = 7
        mask = (1 << (K - 1)) - 1
        reg = (int(bit) & 1) | ((int(memory) & mask) << 1)
        g0, g1 = 0o171, 0o133
        out = (cls._parity_u32(reg & g0), cls._parity_u32(reg & g1))
        return int(reg & mask), out

    @classmethod
    def _conv_encode_bits(cls, bits: np.ndarray, flush: bool = True) -> np.ndarray:
        b = np.asarray(bits, dtype=np.int8).reshape(-1)
        if flush:
            b = np.concatenate([b, np.zeros(6, dtype=np.int8)])
        memory = 0
        out = np.empty(b.size * 2, dtype=np.int8)
        j = 0
        for bit in b:
            memory, pair = cls._conv_next_state_output(memory, int(bit))
            out[j] = pair[0]
            out[j + 1] = pair[1]
            j += 2
        return out

    @classmethod
    def _conv_decode_bits(cls, coded_bits: np.ndarray, decoded_len: int, flushed: bool = True) -> np.ndarray:
        r = np.asarray(coded_bits, dtype=np.int8).reshape(-1)
        if r.size < 2:
            return np.zeros(0, dtype=np.int8)
        if r.size % 2:
            r = r[:-1]
        num_steps = r.size // 2
        num_states = 64
        inf = 10 ** 9
        metrics = np.full(num_states, inf, dtype=np.int32)
        metrics[0] = 0
        prev_state = np.zeros((num_steps, num_states), dtype=np.uint8)
        prev_bit = np.zeros((num_steps, num_states), dtype=np.uint8)
        next_state = np.zeros((num_states, 2), dtype=np.uint8)
        out_bits = np.zeros((num_states, 2, 2), dtype=np.uint8)
        for s in range(num_states):
            for b in (0, 1):
                ns, pair = cls._conv_next_state_output(s, b)
                next_state[s, b] = ns
                out_bits[s, b, 0] = pair[0]
                out_bits[s, b, 1] = pair[1]
        for t in range(num_steps):
            rx0 = int(r[2 * t])
            rx1 = int(r[2 * t + 1])
            new_metrics = np.full(num_states, inf, dtype=np.int32)
            for s in range(num_states):
                base = int(metrics[s])
                if base >= inf:
                    continue
                for b in (0, 1):
                    ns = int(next_state[s, b])
                    dist = int(out_bits[s, b, 0] != rx0) + int(out_bits[s, b, 1] != rx1)
                    cand = base + dist
                    if cand < int(new_metrics[ns]):
                        new_metrics[ns] = cand
                        prev_state[t, ns] = s
                        prev_bit[t, ns] = b
            metrics = new_metrics
        state = 0 if flushed else int(np.argmin(metrics))
        decoded = np.zeros(num_steps, dtype=np.int8)
        for t in range(num_steps - 1, -1, -1):
            b = int(prev_bit[t, state])
            decoded[t] = b
            state = int(prev_state[t, state])
        if flushed and decoded.size >= 6:
            decoded = decoded[:-6]
        decoded_len = int(max(0, decoded_len))
        if decoded.size >= decoded_len:
            return decoded[:decoded_len].astype(np.int8, copy=True)
        return np.concatenate([decoded.astype(np.int8), np.zeros(decoded_len - decoded.size, dtype=np.int8)])

    def _coding_permutation(self, n: int) -> np.ndarray:
        n = int(max(0, n))
        if n <= 1:
            return np.arange(n, dtype=np.int64)
        seed = (self.PILOT_SEED ^ 0xC0DEC0DE ^ (int(self.M) << 16) ^ (int(self.N) << 8) ^ n) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        return rng.permutation(n).astype(np.int64)

    def _apply_bit_interleaver(self, bits: np.ndarray) -> np.ndarray:
        b = np.asarray(bits, dtype=np.int8).reshape(-1)
        if not bool(getattr(self, "coding_interleaver", True)) or b.size <= 1:
            return b.astype(np.int8, copy=True)
        return b[self._coding_permutation(b.size)].astype(np.int8, copy=True)

    def _remove_bit_interleaver(self, bits: np.ndarray) -> np.ndarray:
        b = np.asarray(bits, dtype=np.int8).reshape(-1)
        if not bool(getattr(self, "coding_interleaver", True)) or b.size <= 1:
            return b.astype(np.int8, copy=True)
        perm = self._coding_permutation(b.size)
        out = np.zeros_like(b, dtype=np.int8)
        out[perm] = b
        return out

    def _encode_phy_bits(self, app_frame_bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(app_frame_bits, dtype=np.int8).reshape(-1)
        scheme = str(getattr(self, "coding_scheme", "none")).lower()
        if scheme == "none":
            return bits.astype(np.int8, copy=True)
        if scheme == "conv12":
            coded = self._conv_encode_bits(bits, flush=True)
            return self._apply_bit_interleaver(coded)
        raise ValueError(f"unsupported coding_scheme={scheme}")

    def _decode_phy_bits(self, hard_bits: np.ndarray, decoded_len: int) -> np.ndarray:
        bits = np.asarray(hard_bits, dtype=np.int8).reshape(-1)
        scheme = str(getattr(self, "coding_scheme", "none")).lower()
        if scheme == "none":
            decoded_len = int(max(0, decoded_len))
            if bits.size >= decoded_len:
                return bits[:decoded_len].astype(np.int8, copy=True)
            return np.concatenate([bits.astype(np.int8), np.zeros(decoded_len - bits.size, dtype=np.int8)])
        if scheme == "conv12":
            deint = self._remove_bit_interleaver(bits)
            return self._conv_decode_bits(deint, decoded_len=decoded_len, flushed=True)
        raise ValueError(f"unsupported coding_scheme={scheme}")

    def _coded_len_for_uncoded_len(self, uncoded_bits: int) -> int:
        uncoded_bits = int(max(0, uncoded_bits))
        scheme = str(getattr(self, "coding_scheme", "none")).lower()
        if scheme == "none":
            return uncoded_bits
        if scheme == "conv12":
            return 2 * (uncoded_bits + 6)
        raise ValueError(f"unsupported coding_scheme={scheme}")

    def _coding_summary(self) -> str:
        scheme = str(getattr(self, "coding_scheme", "none")).lower()
        if scheme == "none":
            return "none"
        if scheme == "conv12":
            return "conv12(rate=1/2,K=7,gens=171/133 octal" + (",interleaved" if self.coding_interleaver else "") + ")"
        return scheme

    def _max_payload_bytes_for_current_phy(self) -> int:
        """Maximum UTF-8 payload bytes that fit after PHY coding."""
        cap = self._max_data_bits_capacity()
        # APP frame adds magic(4)+length(4)+CRC32(4) = 12 bytes.
        best = 0
        for payload_bytes in range(0, max(1, cap // 8) + 1):
            uncoded = 8 * (payload_bytes + 12)
            if self._coded_len_for_uncoded_len(uncoded) <= cap:
                best = payload_bytes
            else:
                break
        return int(max(0, best))

    def _expand_waveform_for_uhd(self, cycle_wave: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Repeat one FDIDM super-cycle into a long vector_source payload.

        A very short repeat vector plus any Python block in the graph can make
        UHD's TX streamer see bursty scheduler service.  Feeding vector_source_c
        a few hundred milliseconds of already-normalized samples lowers wrap
        overhead and gives the B210 send buffers a steadier producer.
        """
        wave = np.asarray(cycle_wave, dtype=np.complex64).reshape(-1)
        if wave.size <= 0:
            return np.zeros(1, dtype=np.complex64), 1, 1
        min_by_time = int(np.ceil(max(self.sample_rate, 1.0) * max(self.tx_min_waveform_duration_ms, 0.0) / 1000.0))
        min_samples = max(65536, min_by_time)
        max_samples = int(max(wave.size, self.tx_max_waveform_samples))
        target = int(min(max_samples, max(min_samples, wave.size)))
        repeats = int(max(1, np.ceil(target / max(wave.size, 1))))
        # Do not exceed max_samples unless a single cycle is already longer.
        if wave.size * repeats > max_samples and wave.size < max_samples:
            repeats = max(1, max_samples // max(wave.size, 1))
        out = np.tile(wave, repeats).astype(np.complex64, copy=False)
        return out, int(wave.size), int(repeats)

    def _make_prerendered_tdl_tx(self, samples: np.ndarray) -> np.ndarray:
        """Apply the configured software TDL to TX samples offline.

        This is used for TDL->RF modes so the live GNU Radio TX chain remains
        vector_source -> multiply_const -> usrp_sink, i.e. no Python sync_block
        is allowed to starve the UHD sink.
        """
        model = self._tdl_model_for_current_mode()
        if model is None:
            return np.asarray(samples, dtype=np.complex64).reshape(-1)
        channel = _NTNTDLChannel(
            sample_rate=self.sample_rate,
            model=model,
            rms_delay_spread_ns=self.tdl_rms_delay_spread_ns,
            doppler_hz=self.tdl_doppler_hz,
            doppler_spread_hz=self.tdl_doppler_spread_hz,
            snr_db=self.tdl_snr_db,
            seed=self.tdl_seed,
            normalize_power=self.tdl_normalize_power,
            num_sinusoids=self.tdl_param_num_sinusoids,
        )
        # Process in chunks so a high max waveform length does not create a
        # single huge temporary inside the channel block.
        x = np.asarray(samples, dtype=np.complex64).reshape(-1)
        chunks = []
        step = 65536
        for start in range(0, x.size, step):
            chunks.append(channel.process(x[start:start + step]))
        y = np.concatenate(chunks).astype(np.complex64) if chunks else np.zeros(0, dtype=np.complex64)
        peak = float(np.max(np.abs(y)) + 1e-12) if y.size else 1.0
        if peak > 0:
            y = (0.9 / peak) * y
        return y.astype(np.complex64)

    # =========================================================
    # Application framing
    # =========================================================
    def _build_app_frame(self, payload: bytes) -> bytes:
        payload = payload or b" "
        length_bytes = len(payload).to_bytes(4, "big")
        header = self.APP_MAGIC + length_bytes
        crc = zlib.crc32(header + payload) & 0xFFFFFFFF
        return header + payload + crc.to_bytes(4, "big")

    def _parse_app_frame_exact(self, frame_bytes: bytes) -> Tuple[bool, bytes]:
        if len(frame_bytes) < 12:
            return False, b""
        if frame_bytes[:4] != self.APP_MAGIC:
            return False, b""
        payload_len = int.from_bytes(frame_bytes[4:8], "big")
        if len(frame_bytes) != payload_len + 12:
            return False, b""
        body = frame_bytes[:-4]
        crc_rx = int.from_bytes(frame_bytes[-4:], "big")
        crc_calc = zlib.crc32(body) & 0xFFFFFFFF
        if crc_rx != crc_calc:
            return False, b""
        return True, frame_bytes[8:-4]

    def _make_data_bits_for_frame(self, coded_frame_bits: np.ndarray, frame_idx: int) -> np.ndarray:
        """Build one payload-bearing data grid with random filler.

        The coded application header/payload/CRC are identical in every
        transmitted physical frame, so Viterbi/CRC recovery remains
        deterministic.  The unused capacity after the coded frame is randomized
        per physical frame, which prevents the USRP from replaying a completely
        identical data block forever and gives the constellation/decoder a more
        realistic symbol stream.
        """
        max_bits = self._max_data_bits_capacity()
        out = np.zeros(max_bits, dtype=np.int8)
        fb = np.asarray(coded_frame_bits, dtype=np.int8).reshape(-1)
        out[:fb.size] = fb[:max_bits]
        if fb.size < max_bits:
            seed = (int(self._rng_seed) + 0x9E3779B9 * (int(frame_idx) + 1) +
                    131 * int(self.M) + 17 * int(self.N)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            out[fb.size:] = rng.integers(0, 2, size=max_bits - fb.size, dtype=np.int8)
        return out

    def _build_one_physical_frame(self, data_bits: np.ndarray, pilot_block: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        qam = self._qam_modulate(data_bits, self.mod_order)
        x_cross = qam.reshape((self.M, self.N), order="F")
        x_tf = self._ifdit(x_cross)
        data_block = self._heisenberg(x_tf)
        one_frame = np.concatenate([
            np.zeros(self.pre_guard_len, dtype=np.complex128),
            self.sync_preamble.astype(np.complex128),
            pilot_block,
            data_block,
            np.zeros(self.post_guard_len, dtype=np.complex128),
        ]).astype(np.complex128)
        if one_frame.size != self.frame_len:
            raise RuntimeError(f"internal frame length mismatch: {one_frame.size} != {self.frame_len}")
        return one_frame, x_cross, x_tf

    def _set_tx_text_internal(self, text: str):
        if text is None or len(text) == 0:
            text = " "
        payload = text.encode("utf-8") or b" "
        frame = self._build_app_frame(payload)
        frame_bits = self._frame_to_bits(frame)
        coded_frame_bits = self._encode_phy_bits(frame_bits)
        max_bits = self._max_data_bits_capacity()
        if coded_frame_bits.size > max_bits:
            max_payload_bytes = self._max_payload_bytes_for_current_phy()
            raise ValueError(
                f"FDIDM strict frame capacity is too small after PHY coding: MxN={self.M}x{self.N}, "
                f"mod={self.mod_order}, coding={self._coding_summary()}, capacity={max_bits} coded bits, "
                f"needed={coded_frame_bits.size} coded bits, max payload about {max_payload_bytes} bytes, "
                f"current UTF-8 text is {len(payload)} bytes. Increase N/M, use higher-order QAM, or choose coding=none."
            )

        # The pilot/training part is shared by all repeated physical frames.
        # full_htf uses MN one-hot TF probes. diag_tf and tdl_param both use a
        # single dense constant-modulus TF pilot. In tdl_param mode this dense
        # pilot is used for LS estimation of the TDL path gains.
        if self.use_full_htf:
            pilot_block = self._build_full_htf_training_waveform()
        else:
            pilot_block = self._heisenberg(self._pilot_X_tf)

        guard = np.zeros(self.inter_frame_guard_len, dtype=np.complex128)
        frames = []
        bits_frames: List[np.ndarray] = []
        first_x_cross = None
        first_x_tf = None
        # Do not let the USRP repeat a single identical frame. Even if the UI
        # requests 1-2 physical frames, build a short pseudo-random super-cycle
        # whose application payload is the same but whose filler bits differ.
        self._tx_cycle_frame_count = int(max(self.tx_frame_count, 8))
        for frame_idx in range(self._tx_cycle_frame_count):
            tx_bits_i = self._make_data_bits_for_frame(coded_frame_bits, frame_idx)
            one_frame, x_cross_i, x_tf_i = self._build_one_physical_frame(tx_bits_i, pilot_block)
            frames.append(one_frame)
            bits_frames.append(tx_bits_i.astype(np.int8, copy=True))
            if first_x_cross is None:
                first_x_cross = x_cross_i.copy()
                first_x_tf = x_tf_i.copy()
            if self.inter_frame_guard_len > 0:
                frames.append(guard.copy())
        tx_cycle = np.concatenate(frames) if frames else np.zeros(1, dtype=np.complex128)
        # Peak-normalize one logical super-cycle; this preserves the
        # training/data ratio set above.
        peak = float(np.max(np.abs(tx_cycle)) + 1e-12)
        tx_cycle = (0.9 / peak) * tx_cycle
        tx_wave, base_cycle_len, uhd_repeats = self._expand_waveform_for_uhd(tx_cycle.astype(np.complex64))
        tdl_prerendered = False
        if self._tdl_before_rf_enabled():
            tx_wave = self._make_prerendered_tdl_tx(tx_wave)
            tdl_prerendered = True

        self._tx_text = text
        self._tx_payload = payload
        self._tx_frame = frame
        self._tx_frame_bits = frame_bits.astype(np.int8)
        self._tx_coded_frame_bits = coded_frame_bits.astype(np.int8)
        self._tx_coded_bits_len = int(coded_frame_bits.size)
        self._tx_uncoded_bits_len = int(frame_bits.size)
        self._tx_base_cycle_len = int(base_cycle_len)
        self._tx_uhd_repeats = int(uhd_repeats)
        self._tx_tdl_prerendered = bool(tdl_prerendered)
        self._tx_bits_frame = bits_frames[0].astype(np.int8) if bits_frames else np.zeros(max_bits, dtype=np.int8)
        self._tx_bits_frames = bits_frames
        self._tx_x_cross = first_x_cross if first_x_cross is not None else np.zeros((self.M, self.N), dtype=np.complex128)
        self._tx_x_tf = first_x_tf if first_x_tf is not None else np.zeros((self.M, self.N), dtype=np.complex128)
        self._tx_waveform = tx_wave.astype(np.complex64)
        if hasattr(self, "_debug_log"):
            self._debug(
                "INFO",
                f"TX waveform rebuilt: payload={len(payload)}B, uncoded={frame_bits.size}b, "
                f"coded={coded_frame_bits.size}b/{max_bits}b, coding={self._coding_summary()}, "
                f"cycle={base_cycle_len} samp, repeats={uhd_repeats}, txvec={self._tx_waveform.size} samp, "
                f"TDL_prerender={tdl_prerendered}, mode={self.channel_mode}"
            )
        self._rx_text = ""
        self._decode_ok = False
        self._match_bytes = 0
        self._last_good_rx_payload = b""
        self._last_raw_bytes = b""
        self._ber_estimate = float("nan")
        self._last_fec_bit_ber = float("nan")
        self._last_raw_bit_ber = float("nan")
        self._latest_rx_frame_samples = np.zeros(0, dtype=np.complex64)
        self._latest_rx_data_samples = np.zeros(0, dtype=np.complex64)
        self._latest_rx_pilot_samples = np.zeros(0, dtype=np.complex64)
        self._latest_constellation = np.zeros(0, dtype=np.complex64)
        self._latest_constellation_post_eq = np.zeros(0, dtype=np.complex64)
        self._latest_constellation_pre_eq = np.zeros(0, dtype=np.complex64)
        self._latest_constellation_tf = np.zeros(0, dtype=np.complex64)
        self._last_good_constellation = np.zeros(0, dtype=np.complex64)
        self._last_constellation_is_good = False
        self._last_constellation_source = "none"
        self.last_constellation_source = "none"
        self.last_constellation_points = 0
        self.last_constellation_quality = "reset"
        self._evm_history.clear()
        self.last_evm_instant_percent = float("nan")
        self.last_evm_average_percent = float("nan")

    # =========================================================
    # FSIT / FDIT (unchanged math; identical to paper Eqs. 1, 2, 6, 13)
    # =========================================================
    @staticmethod
    def _wrap_index(value: float) -> float:
        v = ((float(value) + 2.0) % 4.0) - 2.0
        if v <= -2.0 + 1e-12:
            v = 2.0
        return v

    @staticmethod
    def _ap_weight(p: int, eps: float) -> complex:
        d = float(eps) - float(p)
        return (np.cos(d * np.pi / 4.0)
                * np.cos(2.0 * d * np.pi / 4.0)
                * np.exp(1j * 3.0 * d * np.pi / 4.0))

    @staticmethod
    def _unitary_dft_matrix(order: int) -> np.ndarray:
        n = int(order)
        k = np.arange(n, dtype=np.float64)
        return np.exp(-1j * 2.0 * np.pi * np.outer(k, k) / max(n, 1)) / np.sqrt(max(n, 1))

    def _gamma(self, order: int, eps: float) -> np.ndarray:
        n = int(order)
        e = self._wrap_index(float(eps))
        key = (n, round(e, 12))
        if key in self._gamma_cache:
            return self._gamma_cache[key]
        F = self._unitary_dft_matrix(n)
        I = np.eye(n, dtype=np.complex128)
        powers = [I, F, F @ F, F.conj().T]
        G = np.zeros((n, n), dtype=np.complex128)
        for p in range(4):
            G += powers[p] * self._ap_weight(p, e)
        # Defensive: the DFT matrix this codebase uses is symmetric (F[k,l] = F[l,k]),
        # so all gamma powers are symmetric. The vec/kron simplification later relies
        # on this. Assert it now so a future basis change cannot break things silently.
        assert np.max(np.abs(G - G.T)) < 1e-9, "gamma must stay symmetric for kron simplification"
        self._gamma_cache[key] = G
        return G

    @staticmethod
    def _dft_power_apply_axis(arr: np.ndarray, power: int, axis: int) -> np.ndarray:
        """Apply the unitary DFT matrix power F^p along one matrix axis.

        This is the FFT-form counterpart of the dense matrix powers used in
        _gamma(). It follows the same unitary DFT convention as
        _unitary_dft_matrix(): F is fft/sqrt(n), F^2 is the DFT permutation,
        and F^3 is ifft*sqrt(n).
        """
        x = np.asarray(arr, dtype=np.complex128)
        axis = int(axis) % x.ndim
        n = int(x.shape[axis])
        p = int(power) % 4
        if p == 0 or n <= 1:
            return x.astype(np.complex128, copy=True)
        if p == 1:
            return (np.fft.fft(x, axis=axis) / np.sqrt(float(n))).astype(np.complex128)
        if p == 3:
            return (np.fft.ifft(x, axis=axis) * np.sqrt(float(n))).astype(np.complex128)
        idx = np.concatenate(([0], np.arange(n - 1, 0, -1))).astype(np.int64)
        return np.take(x, idx, axis=axis).astype(np.complex128)

    def _apply_gamma_axis(self, arr: np.ndarray, eps: float, axis: int) -> np.ndarray:
        """Apply Gamma^(eps) along one axis using the paper's 4-DFT sum.

        The paper's complexity comparison decomposes FDIT into groups of DFTs
        instead of dense Gamma matrix multiplication. This helper implements
        that decomposition for modulation/demodulation while _gamma() remains
        available for the paper Eq.(29) matrix receiver.
        """
        x = np.asarray(arr, dtype=np.complex128)
        e = self._wrap_index(float(eps))
        out = np.zeros_like(x, dtype=np.complex128)
        for p in range(4):
            w = self._ap_weight(p, e)
            if abs(w) > 1e-14:
                out += w * self._dft_power_apply_axis(x, p, axis=axis)
        return out.astype(np.complex128)

    def _ifdit(self, x_cross: np.ndarray) -> np.ndarray:
        # Paper Eq. 6: X_TF = Gamma_M(alpha) @ X @ Gamma_N(-beta).
        # Implemented through the paper's 4-DFT decomposition rather than
        # dense Gamma matrix multiplication.
        x = np.asarray(x_cross, dtype=np.complex128)
        return self._apply_gamma_axis(self._apply_gamma_axis(x, self.alpha, axis=0), -self.beta, axis=1)

    def _fdit(self, y_tf: np.ndarray) -> np.ndarray:
        # Paper Eq. 13: Y = Gamma_M(-alpha) @ Y_TF @ Gamma_N(beta).
        y = np.asarray(y_tf, dtype=np.complex128)
        return self._apply_gamma_axis(self._apply_gamma_axis(y, -self.alpha, axis=0), self.beta, axis=1)

    # =========================================================
    # Heisenberg / Wigner
    # =========================================================
    def _heisenberg(self, x_tf: np.ndarray) -> np.ndarray:
        x_tf = np.asarray(x_tf, dtype=np.complex128)
        parts = []
        for n in range(self.N):
            td = np.fft.ifft(x_tf[:, n]) * np.sqrt(self.M)
            cp = td[-self.cp_len:] if self.cp_len > 0 else np.zeros(0, dtype=np.complex128)
            parts.append(np.concatenate([cp, td]))
        return np.concatenate(parts).astype(np.complex128)

    def _wigner(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.complex128).reshape(-1)
        expected = self.N * self.block_len
        if samples.size < expected:
            raise ValueError("not enough samples for Wigner transform")
        blocks = samples[:expected].reshape(self.N, self.block_len)
        y = np.zeros((self.M, self.N), dtype=np.complex128)
        for n in range(self.N):
            td = blocks[n, self.cp_len:self.cp_len + self.M]
            y[:, n] = np.fft.fft(td) / np.sqrt(self.M)
        return y

    # =========================================================
    # Sync / CFO
    # =========================================================
    def _build_sync_preamble(self, half_len: int) -> np.ndarray:
        idx = np.arange(half_len, dtype=np.float64)
        half = np.exp(1j * 2.0 * np.pi * idx * (idx + 1.0) / max(2.0 * half_len, 1.0))
        half = half / np.sqrt(np.mean(np.abs(half) ** 2) + 1e-12)
        return np.concatenate([half, half]).astype(np.complex128)

    def _sync_metric(self, rx: np.ndarray) -> np.ndarray:
        """CFO-tolerant Schmidl-Cox-like sync metric.

        v31 used a sharp known-preamble cross-correlation gated by the repeated
        half autocorrelation.  That is excellent at small CFO, but at high
        common Doppler the cross-correlation can collapse before CFO is known.
        v33 therefore keeps the same sharp metric when it is available and
        falls back to the normalized repeated-half autocorrelation as a coarse,
        CFO-insensitive detector.  Timing is refined later with a CFO-scanned
        cross-correlation, so the broader autocorrelation plateau is safe.
        """
        rx = np.asarray(rx, dtype=np.complex128)
        Ls = int(self.sync_len)
        L = int(self.sync_half_len)
        if rx.size < Ls + 1:
            return np.zeros(1, dtype=np.float64)
        sync = self.sync_preamble.astype(np.complex128)
        rx_abs2 = np.abs(rx) ** 2
        cum = np.concatenate([[0.0], np.cumsum(rx_abs2)])

        # Known-preamble cross-correlation: sharp, but sensitive to large CFO.
        cross_corr = np.correlate(rx, sync, mode="valid")
        seg_energy = cum[Ls:] - cum[: rx.size - Ls + 1]
        m_cross = (np.abs(cross_corr) ** 2) / (self._sync_energy * (seg_energy + 1e-12))

        # Repeated-half autocorrelation: CFO-insensitive, but produces a plateau.
        prod = np.conj(rx[:-L]) * rx[L:]
        prod_cum = np.concatenate([[0.0 + 0.0j], np.cumsum(prod)])
        P = prod_cum[L:] - prod_cum[: prod.size - L + 1]
        R_a = cum[L: rx.size - L + 1] - cum[: rx.size - 2 * L + 1]
        R_b = cum[2 * L: rx.size + 1] - cum[L: rx.size - L + 1]
        m_auto = (np.abs(P) ** 2) / (R_a * R_b + 1e-12)

        n = min(m_cross.size, m_auto.size)
        m_cross = m_cross[:n]
        m_auto = m_auto[:n]
        gate = (m_auto > 0.18).astype(np.float64)
        # The 0.85 factor prevents pure autocorr plateaus from out-ranking a
        # valid sharp cross-corr peak at low CFO, while still letting high-CFO
        # frames pass the 0.30 detector threshold.
        return np.maximum(m_cross * gate, 0.85 * m_auto).astype(np.float64)

    def _preamble_cfo_period_hz(self) -> float:
        return float(self.sample_rate) / max(float(self.sync_half_len), 1.0)

    def _preamble_cfo_unambiguous_hz(self) -> float:
        return 0.5 * self._preamble_cfo_period_hz()

    def _cfo_hint_hz(self) -> Optional[float]:
        """Return a soft CFO prior for alias resolution, never a hard override."""
        key = self._current_cfo_mode_key() if hasattr(self, "_current_cfo_mode_key") else ""
        last = float(getattr(self, "_last_good_cfo_hz", float("nan")))
        if (np.isfinite(last) and str(getattr(self, "_last_good_cfo_mode_key", "")) == key):
            return float(last)
        # In software-TDL-involved tests the configured common Doppler is a useful
        # first-run prior.  Real RF oscillator offset is still measured by the
        # preamble, so this is only used to break low-score alias ties.
        if self._software_channel_enabled():
            fd = float(getattr(self, "tdl_doppler_hz", 0.0))
            if abs(fd) > 1e-9:
                return fd
        return None

    def _choose_cfo_from_scored_candidates(self, alias_hz: float, scored: List[Tuple[float, float]]) -> Tuple[float, float, str]:
        """Choose a CFO candidate without letting low-score scan aliases run away.

        The repeated-half phase gives an alias in ±Fs/(2L).  Wide scanning is
        only trusted when the compensated known-preamble score is strong or when
        a candidate agrees with a recent good CFO / configured Doppler hint.
        This directly suppresses the observed fd=0 false locks around ±47 kHz.
        """
        if not scored:
            return float(alias_hz), 0.0, "empty"
        scored = [(float(c), float(v)) for c, v in scored if np.isfinite(c) and np.isfinite(v)]
        if not scored:
            return float(alias_hz), 0.0, "empty_nonfinite"
        alias = float(alias_hz)
        unamb = float(self._preamble_cfo_unambiguous_hz())
        min_score = float(getattr(self, "cfo_scan_min_score", 0.55))
        jump_guard = float(getattr(self, "cfo_scan_jump_guard_hz", 12000.0))
        best_cfo, best_score = max(scored, key=lambda x: x[1])
        alias_cfo, alias_score = min(scored, key=lambda x: abs(x[0] - alias))
        hint = self._cfo_hint_hz()

        # Prefer the last-good/configured-Doppler neighborhood when it is nearly
        # as good as the absolute max.  This avoids jumping between equivalent
        # aliases on noisy transition windows.
        if hint is not None and np.isfinite(hint):
            hint_cfo, hint_score = min(scored, key=lambda x: abs(x[0] - float(hint)))
            if abs(best_cfo - float(hint)) > jump_guard and best_score < 0.75:
                if hint_score >= max(0.35, 0.78 * best_score):
                    return float(hint_cfo), float(hint_score), "hint_guard"
            if hint_score >= max(min_score, 0.90 * best_score):
                return float(hint_cfo), float(hint_score), "hint"

        # No reliable prior: do not trust a far alias unless the known-preamble
        # correlation after compensation is objectively strong.
        if abs(best_cfo - alias) > 0.5:
            if best_score < min_score:
                return float(alias_cfo), float(alias_score), "alias_low_scan_score"
            if abs(best_cfo) > max(unamb * 1.25, abs(alias) + unamb) and best_score < max(min_score, alias_score + 0.18):
                return float(alias_cfo), float(alias_score), "alias_jump_guard"

        return float(best_cfo), float(best_score), "scan_best" if abs(best_cfo - alias) > 0.5 else "alias_best"

    def _preamble_cfo_candidates(self, alias_hz: float) -> List[float]:
        """Return CFO candidates that share the same repeated-half phase.

        The repeated preamble estimates CFO modulo Fs/L.  Candidate values are
        alias_hz + k*Fs/L within ±cfo_search_max_hz, then the known preamble
        chooses the best one by compensated cross-correlation.
        """
        alias = float(alias_hz)
        period = max(self._preamble_cfo_period_hz(), 1e-12)
        if not bool(getattr(self, "cfo_search_enable", True)):
            return [alias]
        max_hz = min(float(getattr(self, "cfo_search_max_hz", 0.0)), 0.49 * float(self.sample_rate))
        if max_hz <= self._preamble_cfo_unambiguous_hz() + 1e-9:
            return [alias]
        k_min = int(np.ceil((-max_hz - alias) / period))
        k_max = int(np.floor((max_hz - alias) / period))
        cands = [alias + k * period for k in range(k_min, k_max + 1)]
        # Always include the raw alias first if numerical clipping removed it.
        if not any(abs(c - alias) < 1e-9 for c in cands):
            cands.insert(0, alias)
        return [float(c) for c in cands]

    def _score_preamble_cfo(self, rx: np.ndarray, sync_start: int, cfo_hz: float) -> float:
        sync = self.sync_preamble.astype(np.complex128)
        Ls = int(sync.size)
        s = int(sync_start)
        if s < 0 or s + Ls > rx.size:
            return 0.0
        seg = np.asarray(rx[s:s + Ls], dtype=np.complex128)
        n = np.arange(Ls, dtype=np.float64)
        seg = seg * np.exp(-1j * 2.0 * np.pi * float(cfo_hz) * n / max(float(self.sample_rate), 1e-12))
        return float((np.abs(np.vdot(sync, seg)) ** 2) / (self._sync_energy * (np.vdot(seg, seg).real + 1e-12)))

    def _refine_sync_and_cfo(self, rx: np.ndarray, coarse: int, search_radius: int = 24) -> Tuple[int, float, float, float]:
        """Refine sync timing and resolve high-Doppler CFO ambiguity.

        Returns (sync_start, cfo_hz, normalized_score, alias_hz).  v33 first
        scores every alias candidate for each timing hypothesis, then applies a
        conservative alias guard so dirty startup/reconfigure windows cannot
        select a low-correlation ±50 kHz CFO when the physical link is near 0 Hz.
        """
        Ls = int(self.sync_len)
        lo = max(0, int(coarse) - int(search_radius))
        hi = min(rx.size - Ls, int(coarse) + int(search_radius))
        best_idx = int(coarse)
        best_cfo = 0.0
        best_alias = 0.0
        best_score = -1.0
        best_reason = "init"
        for s in range(lo, hi + 1):
            alias = self._estimate_cfo_from_preamble(rx, s)
            scored = []
            for cand in self._preamble_cfo_candidates(alias):
                scored.append((float(cand), self._score_preamble_cfo(rx, s, cand)))
            chosen_cfo, chosen_score, reason = self._choose_cfo_from_scored_candidates(alias, scored)
            # Mildly prefer the alias/hint-guarded solution if scores tie; this
            # keeps CFO continuous across successive frames.
            tie_bonus = 1e-4 if reason.startswith(("alias", "hint")) else 0.0
            if chosen_score + tie_bonus > best_score:
                best_score = float(chosen_score)
                best_idx = int(s)
                best_cfo = float(chosen_cfo)
                best_alias = float(alias)
                best_reason = str(reason)
        self._last_cfo_alias_hz = float(best_alias)
        self._last_cfo_scan_score = float(best_score)
        self._last_cfo_unambiguous_hz = float(self._preamble_cfo_unambiguous_hz())
        if abs(best_cfo - best_alias) > 0.5 and (best_score < float(getattr(self, "cfo_scan_min_score", 0.55))):
            self._debug("DEBUG",
                        f"CFO scan guarded: reason={best_reason}, alias={best_alias:.1f}Hz, "
                        f"chosen={best_cfo:.1f}Hz, score={best_score:.3f}")
        return best_idx, best_cfo, best_score, best_alias

    def _find_sync_peaks(self, metric: np.ndarray, max_candidates: int = 3) -> List[int]:
        if metric.size <= 3:
            return []
        max_metric = float(np.max(metric))
        if max_metric < self.sync_metric_threshold:
            return []
        thr = max(self.sync_metric_threshold, 0.55 * max_metric)
        peaks: List[Tuple[float, int]] = []
        for i in range(1, metric.size - 1):
            if metric[i] >= thr and metric[i] >= metric[i - 1] and metric[i] >= metric[i + 1]:
                peaks.append((float(metric[i]), int(i)))
        if not peaks:
            idx = int(np.argmax(metric))
            return [idx]
        peaks.sort(key=lambda x: x[0], reverse=True)
        out: List[int] = []
        min_sep = max(1, self.frame_len // 2)
        for _, idx in peaks:
            if all(abs(idx - j) > min_sep for j in out):
                out.append(idx)
            if len(out) >= max_candidates:
                break
        return out

    def _refine_sync_start(self, rx: np.ndarray, coarse: int, search_radius: int = 24) -> int:
        sync = self.sync_preamble.astype(np.complex128)
        Ls = sync.size
        lo = max(0, int(coarse) - int(search_radius))
        hi = min(rx.size - Ls, int(coarse) + int(search_radius))
        best_idx = int(coarse)
        best_score = -1.0
        for s in range(lo, hi + 1):
            seg = rx[s:s + Ls]
            score = float((np.abs(np.vdot(sync, seg)) ** 2)
                          / (self._sync_energy * (np.vdot(seg, seg).real + 1e-12)))
            if score > best_score:
                best_score = score
                best_idx = int(s)
        return best_idx

    def _estimate_cfo_from_preamble(self, rx: np.ndarray, sync_start: int) -> float:
        L = int(self.sync_half_len)
        if sync_start + 2 * L > rx.size:
            return 0.0
        a = rx[sync_start:sync_start + L]
        b = rx[sync_start + L:sync_start + 2 * L]
        P = np.sum(np.conj(a) * b)
        phase = float(np.angle(P))
        return float(phase * self.sample_rate / (2.0 * np.pi * max(L, 1)))

    def _estimate_residual_cfo_from_pilot(self, pilot_samples: np.ndarray) -> float:
        """Refine residual CFO using all adjacent pilot OFDM symbols.

        The v31 implementation used only pilot symbols 0 and 1.  v33 keeps the averaged
        all adjacent pairs with magnitude weighting, which lowers estimator
        variance and makes the result less sensitive to one faded subcarrier.
        The output is sanity-limited because this estimator assumes a mostly
        diagonal, slowly varying TF channel; under strong TDL/Doppler it can be
        biased and should not override the preamble CFO by an implausible amount.
        """
        if self.N < 2:
            return 0.0
        try:
            y_tf = self._wigner(pilot_samples)
        except ValueError:
            return 0.0
        x_tf = self._pilot_X_tf
        safe_x = np.where(np.abs(x_tf) < 1e-10, 1e-10 + 0j, x_tf)
        h = y_tf / safe_x
        ratios = h[:, 1:] * np.conj(h[:, :-1])
        weights = np.abs(h[:, 1:]) * np.abs(h[:, :-1])
        good = np.isfinite(ratios) & np.isfinite(weights) & (weights > 1e-10)
        if not np.any(good):
            return 0.0
        # Robustly drop the weakest 20% pair products when enough samples exist.
        ww = weights[good]
        rr = ratios[good]
        if ww.size >= 8:
            floor = float(np.percentile(ww, 20.0))
            keep = ww >= floor
            ww = ww[keep]
            rr = rr[keep]
        phasor = np.sum(ww * rr / np.maximum(np.abs(rr), 1e-12))
        if not (np.isfinite(phasor.real) and np.isfinite(phasor.imag)) or abs(phasor) < 1e-12:
            return 0.0
        phase = float(np.angle(phasor))
        block_period = float(self.block_len) / max(self.sample_rate, 1e-12)
        est = float(phase / (2.0 * np.pi * max(block_period, 1e-12)))
        max_abs = min(float(getattr(self, "residual_cfo_max_hz", 5000.0)),
                      0.45 * float(self.sample_rate) / max(float(self.block_len), 1.0))
        if not np.isfinite(est):
            return 0.0
        if abs(est) > max_abs:
            self._debug("DEBUG", f"residual CFO rejected as out-of-range: {est:.1f} Hz > {max_abs:.1f} Hz")
            return 0.0
        return est

    def _fdidm_tx_matrix(self) -> np.ndarray:
        """A in vec(X_TF)=A vec(X), matching paper Eq. (25)."""
        return np.kron(self._gamma(self.N, -self.beta), self._gamma(self.M, self.alpha)).astype(np.complex128)

    def _fdidm_rx_matrix(self) -> np.ndarray:
        """Phi=A^H in y=Phi y_TF, matching paper Eqns. (27)-(29)."""
        return np.kron(self._gamma(self.N, self.beta), self._gamma(self.M, -self.alpha)).astype(np.complex128)

    def _clear_channel_cache(self):
        """Drop cached full-H_TF / TDL-param CSI and derived cross-domain matrices."""
        self._cached_htf_full = None
        self._cached_htf_leakage = float("nan")
        self._cached_htf_frame_counter = -10**18
        self._cached_H_cross = None
        self._cached_Hh_cross = None
        self._cached_HhH_cross = None
        self._cached_Phi = None
        self._cached_H_cond_proxy = float("nan")
        self._full_htf_estimates = 0
        if hasattr(self, "_tdl_param_basis_cache"):
            self._tdl_param_basis_cache.clear()
        self._last_tdl_param_gains = np.zeros(0, dtype=np.complex128)
        self._last_tdl_param_paths = []
        self._last_tdl_param_fit_nmse = float("nan")
        self.last_tdl_param_path_count = 0
        self.last_tdl_param_rank = 0
        self.last_tdl_param_cond = float("nan")

    def _snapshot_channel_cache(self) -> Dict[str, Any]:
        """Take a shallow/deep-enough snapshot so a bad full-H update can be rolled back."""
        return {
            "htf": None if self._cached_htf_full is None else self._cached_htf_full.copy(),
            "leakage": float(self._cached_htf_leakage),
            "frame_counter": int(self._cached_htf_frame_counter),
            "H": None if self._cached_H_cross is None else self._cached_H_cross.copy(),
            "Hh": None if self._cached_Hh_cross is None else self._cached_Hh_cross.copy(),
            "HhH": None if self._cached_HhH_cross is None else self._cached_HhH_cross.copy(),
            "Phi": None if self._cached_Phi is None else self._cached_Phi.copy(),
            "cond": float(self._cached_H_cond_proxy),
            "estimates": int(self._full_htf_estimates),
        }

    def _restore_channel_cache(self, snap: Optional[Dict[str, Any]]):
        if not snap:
            self._clear_channel_cache()
            return
        self._cached_htf_full = snap.get("htf")
        self._cached_htf_leakage = float(snap.get("leakage", float("nan")))
        self._cached_htf_frame_counter = int(snap.get("frame_counter", -10**18))
        self._cached_H_cross = snap.get("H")
        self._cached_Hh_cross = snap.get("Hh")
        self._cached_HhH_cross = snap.get("HhH")
        self._cached_Phi = snap.get("Phi")
        self._cached_H_cond_proxy = float(snap.get("cond", float("nan")))
        self._full_htf_estimates = int(snap.get("estimates", self._full_htf_estimates))

    def _should_refresh_full_htf(self) -> bool:
        if not self.use_full_htf:
            return False
        if self._cached_htf_full is None or self._cached_H_cross is None:
            return True
        # One-shot full-H_TF mode reuses cached CSI until explicitly cleared.
        # Once a good H_TF is cached, never re-estimate on later frames unless
        # reset_full_htf_cache() or a structural/channel reconfigure clears it.
        if bool(getattr(self, "full_htf_once", True)):
            return False
        # Backward-compatible periodic refresh can still be enabled explicitly
        # by setting full_htf_once=False.
        return (int(self._frames_decode_ok) - int(self._cached_htf_frame_counter)) >= int(self.full_htf_update_interval_frames)

    def _cache_full_htf(self, htf: np.ndarray, leakage: float):
        """Cache H_TF and the expensive H=Phi H_TF A products for reuse.

        This keeps the receiver paper-strict while avoiding repeated 256x256
        matrix construction on every identical loopback frame.
        """
        K = int(self.full_htf_order)
        Htf = np.asarray(htf, dtype=np.complex128)
        if Htf.shape != (K, K):
            raise ValueError(f"H_TF shape mismatch for cache: {Htf.shape} != {(K, K)}")
        A = self._fdidm_tx_matrix()
        Phi = self._fdidm_rx_matrix()
        H = Phi @ Htf @ A
        Hh = H.conj().T
        self._cached_htf_full = Htf.copy()
        self._cached_htf_leakage = float(leakage)
        self._cached_htf_frame_counter = int(self._frames_decode_ok)
        self._cached_H_cross = H
        self._cached_Hh_cross = Hh
        self._cached_HhH_cross = Hh @ H
        self._cached_Phi = Phi
        row_norm = np.linalg.norm(H, axis=1)
        nz = row_norm[row_norm > 1e-12]
        self._cached_H_cond_proxy = float(nz.max() / max(nz.min(), 1e-12)) if nz.size else float("inf")
        self._full_htf_estimates += 1

    # =========================================================
    # Channel estimation
    # =========================================================
    def _estimate_htf_full_from_pilot(self, pilot_samples: np.ndarray) -> Tuple[np.ndarray, float]:
        """Estimate the full H_TF matrix in paper Eq. (20).

        Returns H_TF with column-wise vec order.  The second return value is
        an off-diagonal energy ratio, useful as a diagnostic: it is close to 0
        only when the TF channel is nearly diagonal.
        """
        K = int(self.full_htf_order)
        expected = K * self.data_frame_len
        if pilot_samples.size < expected:
            raise ValueError(f"not enough full-H_TF pilot samples: {pilot_samples.size} < {expected}")
        Htf = np.zeros((K, K), dtype=np.complex128)
        amp = self.training_amplitude * np.sqrt(max(self.M, 1))
        amp = amp if abs(amp) > 1e-12 else 1.0
        for k in range(K):
            blk = pilot_samples[k * self.data_frame_len:(k + 1) * self.data_frame_len]
            Htf[:, k] = self._wigner(blk).reshape(-1, order="F") / amp
        total = float(np.linalg.norm(Htf, "fro") ** 2) + 1e-12
        diag = np.diag(np.diag(Htf))
        offdiag_ratio = float(np.linalg.norm(Htf - diag, "fro") ** 2 / total)
        return Htf, offdiag_ratio

    def _estimate_htf_diag_from_pilot(self, pilot_samples: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Robust per-subcarrier TF channel estimate from one dense pilot frame.

        For a channel that is time-invariant over a frame and whose delay spread
        fits inside the CP (short-range RF/cable loopback, or a *static* software
        TDL), the TF channel is diagonal and constant across the N OFDM symbols:
        Y_TF[m,n] = H[m] * X_TF[m,n] + w.  The naive per-cell estimate Y/X is the
        v18 behaviour and is what scattered the constellation on real hardware,
        because every data cell was equalised by a single noisy gain.

        Two estimator improvements:

          1. LS averaging across the N pilot symbols.  With a constant-modulus
             pilot this is the noise-optimal (MRC) estimate for a static channel
             and gives an ~N-fold estimation-noise reduction (~9 dB for N=8).

          2. Impulse-response denoising.  ifft(H) is the cyclic channel impulse
             response; for a CP-fitting channel its energy lives in the first
             few taps while estimation noise is spread over all M taps.  Taps
             that are both outside the CP-protected span *and* below a robust
             tail-noise threshold are zeroed before transforming back.  Genuine
             channel taps are always kept, so this never deletes channel energy.

        Returns (h_tf (M,N), selectivity, noise_var_cell).
        """
        y_tf = self._wigner(pilot_samples)                       # (M, N)
        x_tf = self._pilot_X_tf                                   # (M, N), |x|=const
        safe_x = np.where(np.abs(x_tf) < 1e-10, 1e-10 + 0j, x_tf)
        h_cell = (y_tf / safe_x).astype(np.complex128)           # raw per-cell

        # (1) average across the N pilot OFDM symbols -> per-subcarrier response.
        h_freq = np.mean(h_cell, axis=1)                         # (M,)

        # Per-cell noise variance directly from the pilot residual (this is the
        # quantity the diagonal MMSE load wants; far more reliable than the guard).
        resid = h_cell - h_freq[:, None]
        noise_var_cell = float(np.mean(np.abs(resid) ** 2)) if resid.size > h_freq.size else float("nan")

        # (2) impulse-response denoising of the averaged response.
        M = int(h_freq.size)
        if M >= 4:
            ir = np.fft.ifft(h_freq)
            mag = np.abs(ir)
            keep_span = min(M, int(self.cp_len) + max(2, M // 8))
            tail = mag[keep_span:]
            tail_floor = (float(np.median(tail)) * 3.0) if tail.size else 0.0
            keep = np.zeros(M, dtype=bool)
            keep[:keep_span] = True
            keep |= mag > max(tail_floor, 1e-12)
            h_freq = np.fft.fft(np.where(keep, ir, 0.0 + 0.0j))

        h_tf = np.repeat(h_freq[:, None], self.N, axis=1).astype(np.complex128)
        h_abs = np.abs(h_freq)
        mean_abs = float(np.mean(h_abs)) + 1e-12
        selectivity = float(np.std(h_abs) / mean_abs)
        return h_tf, selectivity, noise_var_cell


    # =========================================================
    # Parametric NTN-TDL channel estimation
    # =========================================================
    def _tdl_profile_components_for_estimator(self) -> List[Dict[str, Any]]:
        """Return the deterministic delay/Doppler basis used by tdl_param mode.

        The software TDL channel has known profile delays, RMS delay-spread scaling,
        common Doppler, and Doppler-spread construction.  The receiver estimates
        only the complex basis gains from the dense pilot; it does not sound all
        MN columns of H_TF.  When Doppler spread is enabled, each Rayleigh tap is
        expanded into a small number of sinusoidal Doppler subpaths matching the
        software-channel sum-of-sinusoids grid.
        """
        model = self._tdl_model_for_current_mode() or "tdl_a"
        tmp = _NTNTDLChannel(
            sample_rate=self.sample_rate,
            model=model,
            rms_delay_spread_ns=self.tdl_rms_delay_spread_ns,
            doppler_hz=self.tdl_doppler_hz,
            doppler_spread_hz=self.tdl_doppler_spread_hz,
            snr_db=300.0,
            seed=self.tdl_seed,
            normalize_power=self.tdl_normalize_power,
            num_sinusoids=self.tdl_param_num_sinusoids,
        )
        # The frame has already been CFO-corrected using the repeated preamble.
        # Therefore the common satellite Doppler is treated as removed here;
        # only Doppler spread remains in the compact basis.  This avoids double
        # counting a large common Doppler in the equalizer.
        common_hz = 0.0
        paths: List[Dict[str, Any]] = []
        for comp in list(getattr(tmp, "_components", [])):
            delay_samp = float(comp.get("delay_samp", 0.0))
            kind = str(comp.get("kind", "rayleigh")).lower()
            idx = int(comp.get("idx", len(paths)))
            base_name = f"tap{idx}:{kind}@{delay_samp:.3f}samp"
            if kind == "los":
                paths.append({
                    "name": base_name,
                    "delay_samp": delay_samp,
                    "doppler_hz": float(common_hz),
                    "kind": "los",
                })
            else:
                spread = float(self.tdl_doppler_spread_hz)
                if spread <= 1e-9:
                    paths.append({
                        "name": base_name,
                        "delay_samp": delay_samp,
                        "doppler_hz": float(common_hz),
                        "kind": "rayleigh",
                    })
                else:
                    angles = np.asarray(comp.get("angles"), dtype=np.float64).reshape(-1)
                    phases = np.asarray(comp.get("phases"), dtype=np.float64).reshape(-1)
                    if angles.size == 0:
                        angles = np.linspace(0.0, 2.0 * np.pi, self.tdl_param_num_sinusoids, endpoint=False)
                        phases = np.zeros_like(angles)
                    for q, ang in enumerate(angles[:int(self.tdl_param_num_sinusoids)]):
                        paths.append({
                            "name": f"{base_name}:sin{q}",
                            "delay_samp": delay_samp,
                            "doppler_hz": float(common_hz + spread * np.cos(float(ang))),
                            "kind": "rayleigh_sinusoid",
                            "phase0": float(phases[q]) if q < phases.size else 0.0,
                        })
        if len(paths) > int(self.tdl_param_max_paths):
            paths = paths[:int(self.tdl_param_max_paths)]
        return paths

    def _apply_parametric_path_time(self, samples: np.ndarray, delay_samp: float,
                                    doppler_hz: float, sample_offset: int = 0) -> np.ndarray:
        """Apply one unit-gain fractional-delay/Doppler path to a time block."""
        x = np.asarray(samples, dtype=np.complex128).reshape(-1)
        if x.size == 0:
            return np.zeros(0, dtype=np.complex128)
        hist_len = int(max(8, np.ceil(abs(float(delay_samp))) + 8))
        ext = np.concatenate((np.zeros(hist_len, dtype=np.complex128), x))
        n = np.arange(x.size, dtype=np.int64)
        delayed = _NTNTDLChannel._fractional_delay(ext, hist_len, n, float(delay_samp))
        t = (float(sample_offset) + n.astype(np.float64)) / max(float(self.sample_rate), 1.0)
        return delayed * np.exp(1j * 2.0 * np.pi * float(doppler_hz) * t)

    def _tdl_param_cache_key(self) -> Tuple[Any, ...]:
        return (
            "tdl_param_basis_v33", int(self.M), int(self.N), int(self.cp_len),
            round(float(self.sample_rate), 6), str(self.channel_mode), str(self._tdl_model_for_current_mode()),
            round(float(self.tdl_rms_delay_spread_ns), 6),
            round(float(self.tdl_doppler_hz), 6), round(float(self.tdl_doppler_spread_hz), 6),
            int(self.tdl_seed), bool(self.tdl_normalize_power),
            int(self.tdl_param_num_sinusoids), int(self.tdl_param_max_paths),
        )

    def _build_single_path_htf(self, delay_samp: float, doppler_hz: float,
                               sample_offset: int = 0) -> np.ndarray:
        """Build H_TF for one unit-gain parametric path.

        Column k is obtained by sending a unit TF basis vector at grid cell k,
        applying the fractional-delay/Doppler path in time, and Wigner-transforming
        the result. This is cached per grid / TDL setting.
        """
        K = int(self.full_htf_order)
        H = np.zeros((K, K), dtype=np.complex128)
        for k in range(K):
            x_vec = np.zeros(K, dtype=np.complex128)
            x_vec[k] = 1.0 + 0.0j
            x_tf = x_vec.reshape((self.M, self.N), order="F")
            tx = self._heisenberg(x_tf)
            rx = self._apply_parametric_path_time(tx, delay_samp, doppler_hz, sample_offset=sample_offset)
            H[:, k] = self._wigner(rx).reshape(-1, order="F")
        return H

    def _get_tdl_param_basis(self) -> Dict[str, Any]:
        """Return cached pilot/data basis matrices for the parametric TDL receiver."""
        key = self._tdl_param_cache_key()
        cached = self._tdl_param_basis_cache.get(key)
        if cached is not None:
            return cached
        if self.full_htf_order > int(self.max_full_htf_order):
            raise ValueError(
                f"tdl_param receiver refuses K={self.full_htf_order} > max_full_htf_order={self.max_full_htf_order}; "
                "increase max_full_htf_order or reduce M/N."
            )
        paths = self._tdl_profile_components_for_estimator()
        if not paths:
            raise ValueError("tdl_param receiver has no TDL paths; choose NTN-TDL-A/C/D software mode")
        K = int(self.full_htf_order)
        x_pilot_vec = np.asarray(self._pilot_X_tf, dtype=np.complex128).reshape(-1, order="F")[:K]
        B_pilot_cols = []
        B_data = []
        data_offset = int(self.pilot_frame_len)
        t0 = time.time()
        for path in paths:
            Hp = self._build_single_path_htf(float(path["delay_samp"]), float(path["doppler_hz"]), sample_offset=0)
            Hd = self._build_single_path_htf(float(path["delay_samp"]), float(path["doppler_hz"]), sample_offset=data_offset)
            B_pilot_cols.append(Hp @ x_pilot_vec)
            B_data.append(Hd)
        G = np.column_stack(B_pilot_cols).astype(np.complex128)
        col_norm = np.linalg.norm(G, axis=0)
        safe = np.where(col_norm < 1e-12, 1.0, col_norm)
        G_norm = G / safe[None, :]
        try:
            rank = int(np.linalg.matrix_rank(G_norm, tol=1e-9))
            svals = np.linalg.svd(G_norm, compute_uv=False)
            cond = float(svals[0] / max(svals[-1], 1e-12)) if svals.size else float("nan")
        except Exception:
            rank = 0
            cond = float("nan")
        out = {
            "paths": paths,
            "G": G,
            "G_norm": G_norm,
            "col_norm": safe.astype(np.float64),
            "B_data": B_data,
            "rank": rank,
            "cond": cond,
            "build_sec": float(time.time() - t0),
        }
        if len(self._tdl_param_basis_cache) > 4:
            self._tdl_param_basis_cache.clear()
        self._tdl_param_basis_cache[key] = out
        self._debug("INFO",
                    f"TDL-PARAM basis built: K={K}, paths={len(paths)}, rank={rank}, "
                    f"cond={cond:.2e}, build={out['build_sec']:.2f}s, mode={self.channel_mode}, "
                    f"DS={self.tdl_rms_delay_spread_ns:.1f}ns, fd={self.tdl_doppler_hz:.1f}Hz, "
                    f"spread={self.tdl_doppler_spread_hz:.1f}Hz")
        return out

    def _estimate_htf_tdl_param_from_pilot(self, pilot_samples: np.ndarray) -> Tuple[np.ndarray, float]:
        """Estimate path gains from one dense TF pilot and reconstruct H_TF.

        The pilot model is y_pilot = G g + noise. G is built from a small TDL
        delay/Doppler basis, and only the complex gains g are fitted. The data
        operator H_TF,data is reconstructed as sum_p g_p B_p,data.
        """
        y_tf = self._wigner(pilot_samples)
        y = y_tf.reshape(-1, order="F").astype(np.complex128)
        basis = self._get_tdl_param_basis()
        G_norm = np.asarray(basis["G_norm"], dtype=np.complex128)
        col_norm = np.asarray(basis["col_norm"], dtype=np.float64)
        if G_norm.shape[0] != y.size:
            raise ValueError(f"tdl_param G/y shape mismatch: {G_norm.shape} vs {y.shape}")
        # Tikhonov-loaded LS is noticeably more stable than bare lstsq when
        # the dense pilot basis columns are correlated (e.g. close delays or
        # many Doppler-spread sinusoids).  The load is relative to the average
        # diagonal of G^H G, so tdl_param_ridge is dimensionless.
        gram = G_norm.conj().T @ G_norm
        rhs = G_norm.conj().T @ y
        avg_diag = float(np.trace(gram).real) / max(int(gram.shape[0]), 1)
        load = float(self.tdl_param_ridge) * (avg_diag + 1e-12)
        try:
            g_scaled = np.linalg.solve(gram + load * np.eye(gram.shape[0], dtype=np.complex128), rhs)
            rank_lstsq = int(np.linalg.matrix_rank(G_norm, tol=1e-9))
        except np.linalg.LinAlgError:
            g_scaled, _residuals, rank_lstsq, _svals = np.linalg.lstsq(G_norm, y, rcond=None)
        g = g_scaled / np.where(col_norm < 1e-12, 1.0, col_norm)
        # Optional relative pruning.  This suppresses tiny noisy basis gains
        # without requiring knowledge of the true number of active paths.
        g_abs = np.abs(g)
        if g_abs.size and np.isfinite(g_abs.max()) and g_abs.max() > 0:
            floor = g_abs.max() * 10.0 ** (float(self.tdl_param_prune_db) / 20.0)
            g = np.where(g_abs >= floor, g, 0.0 + 0.0j)
        y_hat = np.asarray(basis["G"], dtype=np.complex128) @ g
        nmse = float(np.linalg.norm(y - y_hat) ** 2 / (np.linalg.norm(y) ** 2 + 1e-12))
        B_data = basis["B_data"]
        K = int(self.full_htf_order)
        Htf = np.zeros((K, K), dtype=np.complex128)
        for gp, Bp in zip(g, B_data):
            Htf += complex(gp) * np.asarray(Bp, dtype=np.complex128)
        total = float(np.linalg.norm(Htf, "fro") ** 2) + 1e-12
        diag = np.diag(np.diag(Htf))
        offdiag_ratio = float(np.linalg.norm(Htf - diag, "fro") ** 2 / total)
        self._last_tdl_param_gains = np.asarray(g, dtype=np.complex128).copy()
        self._last_tdl_param_paths = list(basis.get("paths", []))
        self._last_tdl_param_fit_nmse = float(nmse)
        self.last_tdl_param_path_count = int(len(B_data))
        self.last_tdl_param_rank = int(basis.get("rank", int(rank_lstsq)))
        self.last_tdl_param_cond = float(basis.get("cond", float("nan")))
        gain_abs = np.abs(g)
        self._debug("INFO",
                    f"TDL-PARAM estimate: paths={len(g)}, rank={self.last_tdl_param_rank}, "
                    f"Gcond={self.last_tdl_param_cond:.2e}, pilot_NMSE={nmse:.3e}, "
                    f"ridge={self.tdl_param_ridge:.1e}, prune={self.tdl_param_prune_db:.1f}dB, "
                    f"Hoffdiag={offdiag_ratio:.3e}, |g|mean={float(np.mean(gain_abs)):.3e}, "
                    f"|g|max={float(np.max(gain_abs)):.3e}")
        return Htf.astype(np.complex128), offdiag_ratio

    def _estimate_noise_var_from_guard(self, frame_samples: np.ndarray) -> float:
        """Estimate noise variance per time sample from the post-frame guard."""
        if frame_samples.size <= self._off_end:
            return float("nan")
        guard = frame_samples[self._off_end: self._off_end + self.post_guard_len]
        if guard.size == 0:
            return float("nan")
        return float(np.mean(np.abs(guard) ** 2))

    def _pre_equalized_cross_observation(self, y_tf_data: np.ndarray) -> np.ndarray:
        """Return the cross-domain received vector before channel equalization.

        For full-H_TF/TDL-param receivers this is y = Phi * vec(Y_TF), i.e.
        the left side of the paper cross-domain model y = Hx+n.  For diag-TF
        engineering mode it falls back to FDIT(Y_TF).  This diagnostic is not
        expected to form a tight constellation under multipath/Doppler; it is
        shown so users can see whether the equalizer actually improves the cloud.
        """
        y_tf = np.asarray(y_tf_data, dtype=np.complex128)
        if y_tf.size == 0:
            return np.zeros(0, dtype=np.complex128)
        try:
            if self.use_full_htf or self.use_tdl_param_htf:
                K = int(self.full_htf_order)
                y_tf_vec = y_tf.reshape(-1, order="F")[:K]
                Phi = self._cached_Phi if (self._cached_Phi is not None and self.use_full_htf) else self._fdidm_rx_matrix()
                return (Phi @ y_tf_vec).reshape(-1, order="F")
            return self._fdit(y_tf).reshape(-1, order="F")
        except Exception:
            return y_tf.reshape(-1, order="F")

    # =========================================================
    # Cross-domain equalization
    # =========================================================
    def _equalize_data_diag(self, y_tf_data: np.ndarray, h_tf_diag: np.ndarray,
                            noise_var: float) -> Tuple[np.ndarray, float, str]:
        """Fast engineering mode: diagonal TF equalization.

        This is valid for near-flat cabled/short-range tests.  It is not the
        general paper Eq. (29) receiver under fractional delay/Doppler.
        """
        H = np.asarray(h_tf_diag, dtype=np.complex128)
        Y = np.asarray(y_tf_data, dtype=np.complex128)
        warning = "diag_tf_equalizer_not_general_paper_H"
        h_abs = np.abs(H)
        if self.equalizer == "ZF":
            # Relative magnitude floor: on a deep frequency-selective fade a tiny
            # |H| would otherwise produce a huge 1/H and blow up the constellation
            # (a source of the downstream overflow).  Floor |H| at a small fraction
            # of the median response while preserving its phase.
            med = float(np.median(h_abs[h_abs > 0.0])) if np.any(h_abs > 0.0) else 0.0
            floor = max(med * 1e-3, 1e-10)
            small = h_abs < floor
            safe_H = H.copy()
            if np.any(small):
                phase = np.exp(1j * np.angle(H[small]))
                phase = np.where(np.abs(H[small]) < 1e-15, 1.0 + 0j, phase)
                safe_H[small] = floor * phase
                warning += ";zf_floor_applied"
            Z = Y / safe_H
        else:  # MMSE
            nv = max(float(noise_var) if np.isfinite(noise_var) else 0.0, 1e-12)
            W = np.conj(H) / (np.abs(H) ** 2 + nv)
            Z = Y * W
        x_hat = self._fdit(Z)
        # Sanitize: a bad/garbage frame must never propagate nan/inf or absurd
        # magnitudes into the abs()/**2 EVM and metric math.  Good frames are
        # untouched (their magnitudes are O(1)).
        x_hat = np.nan_to_num(x_hat, nan=0.0, posinf=0.0, neginf=0.0)
        mag = np.abs(x_hat)
        too_big = mag > 1e3
        if np.any(too_big):
            x_hat[too_big] = (x_hat[too_big] / mag[too_big]) * 1e3
        nz = h_abs > 1e-10
        cond_val = float(h_abs[nz].max() / max(h_abs[nz].min(), 1e-12)) if nz.any() else float("inf")
        return x_hat, cond_val, warning

    def _equalize_data_full_htf(self, y_tf_data: np.ndarray, h_tf: np.ndarray,
                                noise_var: float) -> Tuple[np.ndarray, float, str]:
        """Paper-strict receiver using H = Phi H_TF Phi^H (Eq. 29).

        Data path:
            x_TF = A x, A = Gamma_N(-beta) kron Gamma_M(alpha)
            y     = Phi y_TF, Phi = A^H
            H     = Phi H_TF A
        Then apply the paper's ZF/MMSE linear decoder to y = Hx+n.

        v21 uses cached H, H^H and H^H H whenever H_TF is reused.
        """
        K = int(self.full_htf_order)
        y_tf_vec = np.asarray(y_tf_data, dtype=np.complex128).reshape(-1, order="F")[:K]
        Htf = np.asarray(h_tf, dtype=np.complex128)
        if Htf.shape != (K, K):
            raise ValueError(f"H_TF shape mismatch: {Htf.shape} != {(K, K)}")

        use_cached_cross = (
            self._cached_htf_full is not None and
            Htf is self._cached_htf_full and
            self._cached_H_cross is not None and
            self._cached_Hh_cross is not None and
            self._cached_HhH_cross is not None and
            self._cached_Phi is not None
        )
        if use_cached_cross:
            H = self._cached_H_cross
            Hh = self._cached_Hh_cross
            HhH = self._cached_HhH_cross
            Phi = self._cached_Phi
            cond_val = self._cached_H_cond_proxy
        else:
            A = self._fdidm_tx_matrix()
            Phi = self._fdidm_rx_matrix()
            H = Phi @ Htf @ A
            Hh = H.conj().T
            HhH = Hh @ H
            row_norm = np.linalg.norm(H, axis=1)
            nz = row_norm[row_norm > 1e-12]
            cond_val = float(nz.max() / max(nz.min(), 1e-12)) if nz.size else float("inf")

        y = Phi @ y_tf_vec
        nv = max(float(noise_var) if np.isfinite(noise_var) else 0.0, 1e-12)
        warning = ""
        try:
            if self.equalizer == "ZF":
                x = np.linalg.solve(H, y)
            else:
                lhs = HhH + nv * np.eye(K, dtype=np.complex128)
                rhs = Hh @ y
                x = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            warning = "H_cross_singular_used_loaded_normal_eq"
            load = max(nv, 1e-6 * float(np.mean(np.abs(H) ** 2) + 1e-12))
            lhs = HhH + load * np.eye(K, dtype=np.complex128)
            rhs = Hh @ y
            x = np.linalg.solve(lhs, rhs)
        # An ill-conditioned full-H_TF solve can amplify noise into very large
        # values; keep them finite and bounded so the downstream EVM/abs/**2 math
        # cannot overflow.  Well-conditioned (good) frames stay O(1) and untouched.
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        xm = np.abs(x)
        big = xm > 1e3
        if np.any(big):
            x[big] = (x[big] / xm[big]) * 1e3
            warning = (warning + ";" if warning else "") + "x_clipped"
        return x.reshape((self.M, self.N), order="F"), float(cond_val), warning

    # =========================================================
    # Modem / payload recovery / EVM
    # =========================================================
    @staticmethod
    def _get_bits_per_symbol(mod_order: str) -> int:
        mod = str(mod_order).upper()
        if mod == "QPSK":  return 2
        if mod == "16QAM": return 4
        if mod == "64QAM": return 6
        raise ValueError(f"Unsupported modulation: {mod_order}")

    def _max_data_bits_capacity(self) -> int:
        return int(self.M * self.N * self.bits_per_symbol)

    def _frame_to_bits(self, frame: bytes) -> np.ndarray:
        arr = np.frombuffer(frame, dtype=np.uint8)
        bits = ((arr[:, None] >> np.arange(8, dtype=np.uint8)) & 1).astype(np.int8)
        return bits.reshape(-1)

    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        bits = np.asarray(bits, dtype=np.int8).reshape(-1)
        usable = (bits.size // 8) * 8
        if usable <= 0:
            return b""
        bits = bits[:usable].astype(np.uint8).reshape(-1, 8)
        vals = np.sum(bits << np.arange(8, dtype=np.uint8), axis=1).astype(np.uint8)
        return vals.tobytes()

    def _qam_modulate(self, bits: np.ndarray, mod_order: str) -> np.ndarray:
        mod = str(mod_order).upper()
        bits = np.asarray(bits, dtype=np.int8).reshape(-1)
        bps = self._get_bits_per_symbol(mod)
        if bits.size % bps != 0:
            bits = np.concatenate([bits, np.zeros(bps - bits.size % bps, dtype=np.int8)])
        if mod == "QPSK":
            b = bits.reshape(-1, 2)
            out = np.empty(b.shape[0], dtype=np.complex128)
            mask00 = (b[:, 0] == 0) & (b[:, 1] == 0)
            mask01 = (b[:, 0] == 0) & (b[:, 1] == 1)
            mask11 = (b[:, 0] == 1) & (b[:, 1] == 1)
            mask10 = (b[:, 0] == 1) & (b[:, 1] == 0)
            out[mask00] = 1 + 1j
            out[mask01] = -1 + 1j
            out[mask11] = -1 - 1j
            out[mask10] = 1 - 1j
            return out / np.sqrt(2.0)
        if mod == "16QAM":
            b = bits.reshape(-1, 4)
            lookup = np.array([3.0, 1.0, -3.0, -1.0], dtype=np.float64)
            i_idx = (b[:, 0].astype(np.int64) << 1) | b[:, 1].astype(np.int64)
            q_idx = (b[:, 2].astype(np.int64) << 1) | b[:, 3].astype(np.int64)
            return ((lookup[i_idx] + 1j * lookup[q_idx]) / np.sqrt(10.0)).astype(np.complex128)
        # 64QAM
        b = bits.reshape(-1, 6)
        table = {
            (0, 0, 0): 7, (0, 0, 1): 5, (0, 1, 1): 3, (0, 1, 0): 1,
            (1, 1, 0): -1, (1, 1, 1): -3, (1, 0, 1): -5, (1, 0, 0): -7,
        }
        i = np.array([table[tuple(row[:3].tolist())] for row in b], dtype=np.float64)
        q = np.array([table[tuple(row[3:].tolist())] for row in b], dtype=np.float64)
        return ((i + 1j * q) / np.sqrt(42.0)).astype(np.complex128)

    def _qam_demodulate(self, syms: np.ndarray, mod_order: str) -> np.ndarray:
        mod = str(mod_order).upper()
        syms = np.asarray(syms, dtype=np.complex128).reshape(-1)
        if mod == "QPSK":
            bits = np.zeros((len(syms), 2), dtype=np.int8)
            bits[:, 0] = (np.imag(syms) < 0).astype(np.int8)
            bits[:, 1] = (np.real(syms) < 0).astype(np.int8)
            return bits.reshape(-1)
        if mod == "16QAM":
            x = np.real(syms) * np.sqrt(10.0)
            y = np.imag(syms) * np.sqrt(10.0)
            bits = np.zeros((len(syms), 4), dtype=np.int8)
            bits[:, 0] = (x < 0).astype(np.int8)
            bits[:, 1] = (np.abs(x) < 2).astype(np.int8)
            bits[:, 2] = (y < 0).astype(np.int8)
            bits[:, 3] = (np.abs(y) < 2).astype(np.int8)
            return bits.reshape(-1)
        # 64QAM
        x = np.real(syms) * np.sqrt(42.0)
        y = np.imag(syms) * np.sqrt(42.0)

        def slicer(v: float):
            if v >= 6: return (0, 0, 0)
            if v >= 4: return (0, 0, 1)
            if v >= 2: return (0, 1, 1)
            if v >= 0: return (0, 1, 0)
            if v >= -2: return (1, 1, 0)
            if v >= -4: return (1, 1, 1)
            if v >= -6: return (1, 0, 1)
            return (1, 0, 0)

        out = np.zeros((len(syms), 6), dtype=np.int8)
        for k, (iv, qv) in enumerate(zip(x, y)):
            out[k, 0], out[k, 1], out[k, 2] = slicer(float(iv))
            out[k, 3], out[k, 4], out[k, 5] = slicer(float(qv))
        return out.reshape(-1)

    def _ideal_constellation_points(self) -> np.ndarray:
        mod = self.mod_order
        if mod == "QPSK":
            return np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex128) / np.sqrt(2.0)
        if mod == "16QAM":
            levels = np.array([-3, -1, 1, 3], dtype=np.float64)
            return np.array([i + 1j * q for i in levels for q in levels], dtype=np.complex128) / np.sqrt(10.0)
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float64)
        return np.array([i + 1j * q for i in levels for q in levels], dtype=np.complex128) / np.sqrt(42.0)

    def _estimate_evm_percent(self, syms: np.ndarray) -> float:
        syms = np.asarray(syms, dtype=np.complex128).reshape(-1)
        valid = np.isfinite(np.real(syms)) & np.isfinite(np.imag(syms))
        syms = syms[valid]
        if syms.size == 0:
            return float("nan")
        refs = self._ideal_constellation_points()
        idx = np.argmin(np.abs(syms[:, None] - refs[None, :]), axis=1)
        decisions = refs[idx]
        denom = np.vdot(decisions, decisions)
        if abs(denom) > 1e-12:
            gain = np.vdot(decisions, syms) / denom
            if np.isfinite(gain.real) and np.isfinite(gain.imag) and abs(gain) > 1e-8:
                syms = syms / gain
        idx = np.argmin(np.abs(syms[:, None] - refs[None, :]), axis=1)
        decisions = refs[idx]
        ref_power = float(np.mean(np.abs(decisions) ** 2)) + 1e-12
        return float(100.0 * np.sqrt(float(np.mean(np.abs(syms - decisions) ** 2)) / ref_power))

    def _update_evm_history(self, evm_inst: float):
        self.last_evm_instant_percent = float(evm_inst) if np.isfinite(evm_inst) else float("nan")
        if np.isfinite(evm_inst):
            self._evm_history.append(float(evm_inst))
            vals = np.asarray(self._evm_history, dtype=np.float64)
            self.last_evm_average_percent = float(np.sqrt(np.mean(vals ** 2))) if vals.size else float("nan")

    def _verification_ber(self, bits_all: np.ndarray) -> float:
        bits_all = np.asarray(bits_all, dtype=np.int8).reshape(-1)
        refs = getattr(self, "_tx_bits_frames", None) or [self._tx_bits_frame]
        vals = []
        for ref in refs:
            ref = np.asarray(ref, dtype=np.int8).reshape(-1)
            L = int(min(bits_all.size, ref.size))
            if L > 0:
                vals.append(float(np.mean(bits_all[:L] != ref[:L])))
        return min(vals) if vals else 1.0

    def _recover_payload_from_symbols(self, rx_syms: np.ndarray):
        """Demodulate, parse APP frame, compute diagnostics.

        Uses two stabilizers for real hardware tests:
        1) common gain/phase is estimated from the known application frame
           prefix rather than only the 4-byte magic, which reduces phase jitter;
        2) QPSK tries the four residual 90-degree rotations and chooses the
           candidate that passes CRC or has the lowest verification BER.
        """
        rx_syms = np.asarray(rx_syms, dtype=np.complex128).reshape(-1)
        need_syms = self.M * self.N
        if rx_syms.size < need_syms:
            return 1.0, b"", b"", "", 0, False, rx_syms.astype(np.complex64), float("nan")
        rx_syms = rx_syms[:need_syms]

        residual_gain = self._estimate_residual_data_gain(rx_syms)
        if not (np.isfinite(residual_gain.real) and np.isfinite(residual_gain.imag)) or abs(residual_gain) < 1e-8:
            residual_gain = 1.0 + 0.0j
        base_syms = rx_syms / residual_gain
        self.last_residual_gain_abs = float(abs(residual_gain))
        self.last_residual_phase_deg = float(np.angle(residual_gain) * 180.0 / np.pi)

        total_bits = self._max_data_bits_capacity()
        frame_bits_len = int(self._tx_frame_bits.size)
        coded_bits_len = int(getattr(self, "_tx_coded_bits_len", 0) or frame_bits_len)
        rotations = [1.0 + 0.0j]
        if self.mod_order == "QPSK":
            rotations = [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j]

        best = None
        for rot in rotations:
            syms = base_syms * rot
            bits_all = self._qam_demodulate(syms, self.mod_order)[:total_bits]
            raw_ber = self._verification_ber(bits_all)
            coded_rx = bits_all[:coded_bits_len]
            if coded_rx.size < coded_bits_len:
                coded_rx = np.concatenate([coded_rx, np.zeros(coded_bits_len - coded_rx.size, dtype=np.int8)])
            decoded_bits = self._decode_phy_bits(coded_rx, decoded_len=frame_bits_len)
            if frame_bits_len > 0:
                info_ber = float(np.mean(decoded_bits[:frame_bits_len] != self._tx_frame_bits[:frame_bits_len]))
            else:
                info_ber = float(raw_ber)
            frame_bytes = self._bits_to_bytes(decoded_bits[:frame_bits_len])
            ok, payload = self._parse_app_frame_exact(frame_bytes)
            text = payload.decode("utf-8", errors="replace") if ok else ""
            match = int(sum(int(a == b) for a, b in zip(payload, self._tx_payload))) if ok else 0
            evm = self._estimate_evm_percent(syms)
            score = (1200.0 * float(ok)
                     + 140.0 * (1.0 - min(info_ber, 1.0))
                     + 40.0 * (1.0 - min(raw_ber, 1.0))
                     - 0.2 * (evm if np.isfinite(evm) else 100.0))
            cand = (score, info_ber, raw_ber, frame_bytes, payload, text, match, bool(ok), syms.astype(np.complex64), evm)
            if best is None or cand[0] > best[0]:
                best = cand
            if ok:
                break
        _, ber, raw_ber, frame_bytes, payload, text, match, decode_ok, syms_best, evm = best
        self._last_fec_bit_ber = float(ber)
        self._last_raw_bit_ber = float(raw_ber)
        return float(ber), frame_bytes, payload, text, int(match), bool(decode_ok), syms_best, float(evm)

    def _known_preamble_ref_syms(self) -> np.ndarray:
        # Use all known application-frame bits before random filler.  This is a
        # hardware demonstration link where the transmitted text is known; using
        # the whole application frame makes the residual scalar estimate much
        # less noisy than using only the 4-byte magic.
        bps = max(self.bits_per_symbol, 1)
        # Use the coded physical bits actually mapped into the data grid.  In
        # The APP frame may be convolutionally encoded/interleaved before
        # modulation, so using the uncoded APP bits here would estimate a wrong
        # residual scalar and rotate/scatter the constellation.
        bits = np.asarray(getattr(self, "_tx_coded_frame_bits", self._tx_frame_bits), dtype=np.int8).reshape(-1)
        usable = (bits.size // bps) * bps
        if usable <= 0:
            return np.zeros(0, dtype=np.complex128)
        return self._qam_modulate(bits[:usable], self.mod_order).astype(np.complex128)

    def _estimate_residual_data_gain(self, rx_syms: np.ndarray) -> complex:
        rx = np.asarray(rx_syms, dtype=np.complex128).reshape(-1)
        ref = self._known_preamble_ref_syms()
        L = int(min(rx.size, ref.size))
        if L <= 0:
            return 1.0 + 0.0j
        denom = np.vdot(ref[:L], ref[:L])
        if abs(denom) < 1e-12:
            return 1.0 + 0.0j
        gain = np.vdot(ref[:L], rx[:L]) / denom
        if not (np.isfinite(gain.real) and np.isfinite(gain.imag)) or abs(gain) < 1e-8:
            return 1.0 + 0.0j
        return gain

    # =========================================================
    # Channel path configuration
    # =========================================================
    def _normalize_channel_mode(self, mode: str) -> str:
        """Normalize channel path mode.

        Supported hardware-facing values:
            rf          : USRP TX -> real RF/channel -> USRP RX
            rf_tdl_a/c/d: real RF first, then software NTN-TDL before decoding
            tdl_a/c/d_rf: software NTN-TDL pre-rendered into TX, then real RF

        Legacy baseband-only TDL names are mapped to RF->TDL so old saved UI settings
        still run through the USRP instead of silently using a baseband loopback.
        """
        raw = str(mode or "rf").strip().lower()
        m = (raw.replace("-", "_").replace(" ", "_")
                .replace("+", "_").replace("/", "_"))
        if m in ("rf", "usrp", "hardware", "hw", "off", "none", "no", "false", "0"):
            return "rf"
        legacy_baseband = {
            "tdl_a": "rf_tdl_a", "tdla": "rf_tdl_a", "ntn_tdl_a": "rf_tdl_a", "a": "rf_tdl_a",
            "tdl_c": "rf_tdl_c", "tdlc": "rf_tdl_c", "ntn_tdl_c": "rf_tdl_c", "c": "rf_tdl_c",
            "tdl_d": "rf_tdl_d", "tdld": "rf_tdl_d", "ntn_tdl_d": "rf_tdl_d", "d": "rf_tdl_d",
        }
        if m in legacy_baseband:
            mapped = legacy_baseband[m]
            self._channel_mode_note = (
                f"legacy baseband-only channel_mode={raw!r} mapped to {mapped!r}; "
                "baseband-only loopback is disabled in v33 so every run traverses the USRP RF path."
            )
            return mapped
        if m in ("rf_tdl_a", "rf_tdla", "hybrid_tdl_a", "hybrid_tdla", "usrp_tdl_a", "usrp_tdla"):
            return "rf_tdl_a"
        if m in ("rf_tdl_c", "rf_tdlc", "hybrid_tdl_c", "hybrid_tdlc", "usrp_tdl_c", "usrp_tdlc"):
            return "rf_tdl_c"
        if m in ("rf_tdl_d", "rf_tdld", "hybrid_tdl_d", "hybrid_tdld", "usrp_tdl_d", "usrp_tdld"):
            return "rf_tdl_d"
        if m in ("tdl_a_rf", "tdla_rf", "tdl_a_plus_rf", "tdl_a_and_rf"):
            return "tdl_a_rf"
        if m in ("tdl_c_rf", "tdlc_rf", "tdl_c_plus_rf", "tdl_c_and_rf"):
            return "tdl_c_rf"
        if m in ("tdl_d_rf", "tdld_rf", "tdl_d_plus_rf", "tdl_d_and_rf"):
            return "tdl_d_rf"
        raise ValueError("channel_mode must be one of: rf, rf_tdl_a/c/d, tdl_a/c/d_rf")

    def _tdl_model_for_current_mode(self) -> Optional[str]:
        mode = str(getattr(self, "channel_mode", "rf")).lower()
        if mode.startswith("rf_tdl_"):
            return mode[3:]
        if mode.endswith("_rf") and mode[:-3] in _NTNTDLChannel.TDL_PROFILES:
            return mode[:-3]
        return None

    def _rf_path_enabled(self) -> bool:
        mode = str(getattr(self, "channel_mode", "rf")).lower()
        return mode == "rf" or mode.startswith("rf_tdl_") or (mode.endswith("_rf") and mode[:-3] in _NTNTDLChannel.TDL_PROFILES)

    def _software_channel_enabled(self) -> bool:
        return self._tdl_model_for_current_mode() is not None

    def _tdl_before_rf_enabled(self) -> bool:
        mode = str(getattr(self, "channel_mode", "rf")).lower()
        return mode.endswith("_rf") and mode[:-3] in _NTNTDLChannel.TDL_PROFILES

    def _tdl_after_rf_enabled(self) -> bool:
        return str(getattr(self, "channel_mode", "rf")).lower().startswith("rf_tdl_")

    def _software_known_cfo_for_current_mode(self) -> Optional[float]:
        # Every supported v33 mode includes real RF hardware, so CFO must be
        # measured from the received preamble rather than assumed from a model.
        return None

    def _guard_estimator_for_software_tdl(self) -> bool:
        """Backward-compatible alias for estimator resolution."""
        return self._resolve_effective_channel_estimator()

    def _make_tdl_channel_block(self):
        """Create a GNU Radio Python sync_block wrapping _NTNTDLChannel."""
        gr = self._gr
        channel = _NTNTDLChannel(
            sample_rate=self.sample_rate,
            model=self._tdl_model_for_current_mode(),
            rms_delay_spread_ns=self.tdl_rms_delay_spread_ns,
            doppler_hz=self.tdl_doppler_hz,
            doppler_spread_hz=self.tdl_doppler_spread_hz,
            snr_db=self.tdl_snr_db,
            seed=self.tdl_seed,
            normalize_power=self.tdl_normalize_power,
            num_sinusoids=self.tdl_param_num_sinusoids,
        )

        class _TDLChannelBlock(gr.sync_block):
            def __init__(self):
                gr.sync_block.__init__(self, name="ntn_tdl_channel_v34", in_sig=[np.complex64], out_sig=[np.complex64])
                self.channel = channel
                self._channel_lock = threading.RLock()

            def work(self, input_items, output_items):
                # Runtime parameter updates can arrive from the UI thread.  Keep
                # channel.configure()/reset() mutually exclusive with process()
                # so a partial tap-table update cannot leak into one scheduler call.
                with self._channel_lock:
                    y = self.channel.process(input_items[0])
                output_items[0][:len(y)] = y
                return len(y)

            def reset_channel(self):
                with self._channel_lock:
                    self.channel.reset()

            def configure_channel(self, **kwargs: Any):
                with self._channel_lock:
                    self.channel.configure(**kwargs)
                    self.channel.reset()

            def channel_summary(self) -> str:
                with self._channel_lock:
                    return self.channel.summary()

        return _TDLChannelBlock()

    def reset_full_htf_cache(self):
        """Public API used by the UI: clear CSI and cached parametric TDL bases."""
        self._clear_channel_cache()
        self._invalidate_alpha_beta_adaptation(reason="csi_cache_reset", cooldown=False)
        try:
            self._tdl_param_basis_cache.clear()
        except Exception:
            pass
        self._debug("INFO", "CSI cache cleared; full-H will re-identify, TDL-param will rebuild basis/estimate on next frame")

    # =========================================================
    # GNU Radio / UHD lifecycle
    # =========================================================
    def _build_device_args(self) -> str:
        if self.device_type == "USRP B210":
            base = "type=b200,master_clock_rate=52e6"
        elif self.device_type == "USRP N210":
            base = "type=n200"
        elif self.device_type == "USRP X310":
            base = "type=x300"
        else:
            raise ValueError(f"Unsupported device_type: {self.device_type}")
        if self.serial:
            return f"serial={self.serial},{base}"
        return base

    def _rx_stream_args(self) -> str:
        # Larger UHD host buffers reduce occasional B210 RX overflow on Windows.
        return f"recv_frame_size=8192,num_recv_frames={int(self.usrp_buffer_frames)}"

    def _tx_stream_args(self) -> str:
        return f"send_frame_size=8192,num_send_frames={int(self.usrp_buffer_frames)}"

    def _import_runtime(self):
        try:
            from gnuradio import blocks, gr, uhd
            self._blocks = blocks
            self._gr = gr
            self._uhd = uhd
        except Exception as e:
            raise RuntimeError(
                "Cannot import GNU Radio/UHD. Please install gnuradio and gnuradio-uhd.\n"
                f"Original error: {e}"
            )

    def _make_rx_ring_sink(self):
        """Fallback bounded RX sink for old GNU Radio builds.

        This Python sync_block is only used if blocks.probe_signal_vc is not
        available.  The preferred v20 path below is entirely C++ in the live
        scheduler and therefore much less likely to overflow the B210 RX.
        """
        gr = self._gr
        outer = self

        class _RXRingSink(gr.sync_block):
            def __init__(self):
                gr.sync_block.__init__(self, name="rx_numpy_ring_sink_fallback", in_sig=[np.complex64], out_sig=[])

            def work(self, input_items, output_items):
                outer._rx_buffer.write(input_items[0])
                return len(input_items[0])

        return _RXRingSink()

    def _select_rx_probe_len(self) -> int:
        """Choose a contiguous RX window large enough for frame processing.

        RX probe length is rounded to a power of two. GNU Radio
        vector items are sizeof(gr_complex) * vector_len bytes; non-aligned
        sizes such as 499680 bytes trigger Windows double-mapped buffer warnings.
        Power-of-two lengths are also FFT/UI friendly and avoid the warning.
        """
        base = max(3 * (int(self.frame_len) + int(self.inter_frame_guard_len)), 8192)
        base = max(int(self.frame_len) + int(self.inter_frame_guard_len), int(base))
        # Next power of two, clamped.  Minimum 8192 keeps item bytes aligned to
        # 65536-byte allocation granularity because complex64 is 8 bytes.
        n = 1 << int(np.ceil(np.log2(max(8192, base))))
        n = min(int(n), int(self._buffer_keep))
        n = min(int(n), 524288)
        n = max(8192, int(n))
        # Final safety: multiple of 8192 complex samples -> 65536 bytes.
        if n % 8192 != 0:
            n = int(np.ceil(n / 8192.0) * 8192)
        return int(n)

    def _build_rx_probe_chain(self):
        """Return (vectorizer, sink, probe, mode) for RX capture.

        Preferred mode:
            usrp_source -> stream_to_vector -> probe_signal_vc

        Both blocks are GNU Radio C++ blocks.  The USRP source is therefore
        drained by the scheduler without Python being called for every RX
        chunk.  The monitor thread only polls the latest vector at UI rate.
        """
        gr = self._gr
        blocks = self._blocks
        self._rx_probe_len = self._select_rx_probe_len()
        if hasattr(blocks, "probe_signal_vc") and hasattr(blocks, "stream_to_vector"):
            try:
                rx_s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, self._rx_probe_len)
                rx_probe = blocks.probe_signal_vc(self._rx_probe_len)
                self._rx_probe_mode = "probe_signal_vc"
                return rx_s2v, rx_probe, rx_probe, self._rx_probe_mode
            except Exception as e:
                self._debug("WARN",
                            f"probe_signal_vc path unavailable; using Python ring fallback: {type(e).__name__}: {e}")
        rx_sink = self._make_rx_ring_sink()
        self._rx_probe_mode = "python_ring_fallback"
        return None, rx_sink, None, self._rx_probe_mode

    def _build_top_block(self):
        gr = self._gr
        blocks = self._blocks
        uhd = self._uhd

        old_tb = getattr(self, "_tb", None)
        if old_tb is not None:
            try:
                old_tb.stop()
                old_tb.wait()
                self._debug("INFO", "old top_block stopped during rebuild")
            except Exception as e:
                self._debug("WARN", f"old top_block.stop during rebuild: {type(e).__name__}: {e}")
        for attr in ("_tb", "_vector_source", "_usrp_source", "_usrp_sink",
                     "_tx_sink_vec", "_rx_sink_vec", "_rx_probe", "_rx_stream_to_vector",
                     "_tdl_channel_block", "_throttle_block", "_tx_gain_block"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
        import gc
        gc.collect()

        uses_rf = self._rf_path_enabled()
        uses_tdl = self._software_channel_enabled()
        tdl_model = self._tdl_model_for_current_mode()
        if not uses_rf:
            raise RuntimeError(f"channel_mode={self.channel_mode!r} does not traverse the USRP RF path; baseband-only loopback is disabled")
        time.sleep(0.25)

        if uses_tdl and self._tdl_before_rf_enabled():
            mode_text = f"software {tdl_model} pre-rendered -> USRP RF"
        elif uses_tdl:
            mode_text = f"USRP RF -> software {tdl_model}"
        else:
            mode_text = "USRP RF"
        self._debug("INFO",
                    f"building new top_block ({mode_text}): {self._waveform_fingerprint()}, "
                    f"alpha={self.alpha:.3f}, beta={self.beta:.3f}, mod={self.mod_order}, "
                    f"coding={self._coding_summary()}, UHDbuf={self.usrp_buffer_frames}")

        class _TopBlock(gr.top_block):
            pass

        tb = _TopBlock("FDIDM Hardware Test v33", catch_exceptions=True)
        vector_source = blocks.vector_source_c(self._tx_waveform.tolist(), True, 1, [])
        tx_gain_block = blocks.multiply_const_cc(1.0)
        tx_sink_vec = None
        rx_stream_to_vector, rx_sink_vec, rx_probe, rx_probe_mode = self._build_rx_probe_chain()

        usrp_source = None
        usrp_sink = None
        throttle_block = None
        live_tdl_needed = uses_tdl and not self._tdl_before_rf_enabled()
        tdl_channel_block = self._make_tdl_channel_block() if live_tdl_needed else None

        tx_min_buf = int(max(32768, min(max(self._tx_waveform.size // 4, 32768), 262144)))
        for blk in (vector_source, tx_gain_block):
            try:
                blk.set_min_output_buffer(tx_min_buf)
            except Exception:
                pass

        tb.connect((vector_source, 0), (tx_gain_block, 0))

        if uhd is None:
            raise RuntimeError("GNU Radio UHD module is unavailable but RF path requires USRP.")
        usrp_source = uhd.usrp_source(
            ",".join(("", self._usrp_args)),
            uhd.stream_args(cpu_format="fc32", args=self._rx_stream_args(), channels=list(range(0, 1))),
        )
        usrp_source.set_subdev_spec("A:A", 0)
        usrp_source.set_samp_rate(self.sample_rate)
        usrp_source.set_time_unknown_pps(uhd.time_spec(0))
        usrp_source.set_center_freq(self.carrier_freq, 0)
        usrp_source.set_antenna(self.rx_antenna, 0)
        usrp_source.set_gain(self.rx_gain, 0)
        try:
            usrp_source.set_min_output_buffer(max(32768, int(self._rx_probe_len)))
        except Exception:
            pass

        usrp_sink = uhd.usrp_sink(
            ",".join(("", self._usrp_args)),
            uhd.stream_args(cpu_format="fc32", args=self._tx_stream_args(), channels=list(range(0, 1))),
            "",
        )
        usrp_sink.set_subdev_spec("A:A", 0)
        usrp_sink.set_samp_rate(self.sample_rate)
        usrp_sink.set_time_unknown_pps(uhd.time_spec(0))
        usrp_sink.set_center_freq(self.carrier_freq, 0)
        usrp_sink.set_antenna(self.tx_antenna, 0)
        usrp_sink.set_gain(self.tx_gain, 0)

        tb.connect((tx_gain_block, 0), (usrp_sink, 0))
        rx_input = usrp_source
        log_label = "USRP RF only"
        if uses_tdl and self._tdl_before_rf_enabled():
            log_label = "software TDL pre-rendered into TX vector before USRP RF"
        elif uses_tdl:
            tb.connect((usrp_source, 0), (tdl_channel_block, 0))
            rx_input = tdl_channel_block
            log_label = "USRP RF before software TDL"

        if uses_tdl:
            try:
                if tdl_channel_block is not None:
                    self._debug("INFO", f"hybrid configured ({log_label}): " + tdl_channel_block.channel_summary())
                else:
                    self._debug("INFO", f"hybrid configured ({log_label}); live GNU Radio TX path has no Python TDL block")
            except Exception as e:
                self._debug("WARN", f"TDL summary unavailable: {type(e).__name__}: {e}")

        if rx_stream_to_vector is not None:
            tb.connect((rx_input, 0), (rx_stream_to_vector, 0))
            tb.connect((rx_stream_to_vector, 0), (rx_sink_vec, 0))
        else:
            tb.connect((rx_input, 0), (rx_sink_vec, 0))

        self._tb = tb
        self._usrp_source = usrp_source
        self._usrp_sink = usrp_sink
        self._tx_sink_vec = tx_sink_vec
        self._rx_sink_vec = rx_sink_vec
        self._rx_stream_to_vector = rx_stream_to_vector
        self._rx_probe = rx_probe
        self._vector_source = vector_source
        self._tdl_channel_block = tdl_channel_block
        self._throttle_block = throttle_block
        self._tx_gain_block = tx_gain_block
        self._debug("INFO",
                    f"new top_block assembled, path={mode_text}, rx_mode={rx_probe_mode}, "
                    f"rx_probe_len={self._rx_probe_len}")
        with self._lock:
            self._tx_buffer.clear()
            self._rx_buffer.clear()
            self._tx_buffer.write(self._tx_waveform.astype(np.complex64, copy=False))
            self._latest_tx_samples = self.get_tx_waveform_preview(4096)
            self._latest_rx_frame_samples = np.zeros(0, dtype=np.complex64)
            self._latest_rx_data_samples = np.zeros(0, dtype=np.complex64)
            self._latest_rx_pilot_samples = np.zeros(0, dtype=np.complex64)
            self._latest_constellation = np.zeros(0, dtype=np.complex64)
            self._latest_constellation_post_eq = np.zeros(0, dtype=np.complex64)
            self._latest_constellation_post_eq_raw = np.zeros(0, dtype=np.complex64)
            self._latest_constellation_pre_eq = np.zeros(0, dtype=np.complex64)
            self._latest_constellation_tf = np.zeros(0, dtype=np.complex64)
            self._last_good_constellation = np.zeros(0, dtype=np.complex64)
            self._last_constellation_is_good = False
            self._last_constellation_source = "none"
            self.last_constellation_source = "none"
            self.last_constellation_points = 0
            self.last_constellation_quality = "reset"
            self._rx_last_new_samples = 0
            self._rx_stream_updates = 0
            self._rx_latest_window_len = 0
            self._rx_last_update_wall = 0.0
            self._rx_spectrum_stale = True
            self._rx_spectrum_stale_sec = float("inf")
            self._rx_text = ""
            self._decode_ok = False
            self._match_bytes = 0
            self._ber_estimate = float("nan")
            self._last_error = ""
            self._last_info = ""
            self._status = "configured"
            self._rx_samples_seen = 0
            self._tx_preview_start_t = time.time()
            self._rx_probe_total_est = 0
            self._rx_probe_last_fp = None
            self._last_processed_abs_start = -10 ** 18
        self._needs_top_block_rebuild = False

    def _current_cfo_mode_key(self) -> str:
        return (
            f"mode={getattr(self, 'channel_mode', 'rf')}|Fs={float(getattr(self, 'sample_rate', 0.0)):.3f}|"
            f"M={int(getattr(self, 'M', 0))}|N={int(getattr(self, 'N', 0))}|"
            f"fc={float(getattr(self, 'carrier_freq', 0.0)):.3f}|"
            f"tdl_fd={float(getattr(self, 'tdl_doppler_hz', 0.0)):.3f}"
        )

    def _reset_rx_runtime_state(self, reason: str = "reset", reset_counters: bool = False):
        """Clear RX/probe/decoder state after any graph or waveform transition.

        This prevents stale probe vectors or half-old/half-new frames from being
        interpreted as valid FDIDM frames after a parameter switch.  TX preview
        data is intentionally left intact.
        """
        with self._lock:
            try:
                self._rx_buffer.clear()
            except Exception:
                pass
            self._latest_rx_samples = np.zeros(4096, dtype=np.complex64)
            self._latest_rx_frame_samples = np.zeros(0, dtype=np.complex64)
            self._latest_rx_data_samples = np.zeros(0, dtype=np.complex64)
            self._latest_rx_pilot_samples = np.zeros(0, dtype=np.complex64)
            self._latest_constellation = np.zeros(0, dtype=np.complex64)
            self._latest_constellation_post_eq = np.zeros(0, dtype=np.complex64)
            self._latest_constellation_post_eq_raw = np.zeros(0, dtype=np.complex64)
            self._latest_constellation_pre_eq = np.zeros(0, dtype=np.complex64)
            self._latest_constellation_tf = np.zeros(0, dtype=np.complex64)
            self._last_constellation_is_good = False
            self._last_constellation_source = "none"
            self.last_constellation_source = "none"
            self.last_constellation_points = 0
            self.last_constellation_quality = "reset"
            self._rx_text = ""
            self._decode_ok = False
            self._match_bytes = 0
            self._ber_estimate = float("nan")
            self._last_fec_bit_ber = float("nan")
            self._last_raw_bit_ber = float("nan")
            self.last_frame_ok = False
            self.last_bad_reason = reason
            self.last_sync_metric = 0.0
            self.last_cfo_est_hz = 0.0
            self.last_cfo_preamble_hz = 0.0
            self.last_cfo_source = "reset"
            self._last_cfo_alias_hz = float("nan")
            self._last_cfo_scan_score = float("nan")
            self.last_evm_instant_percent = float("nan")
            self.last_evm_average_percent = float("nan")
            self._evm_history.clear()
            self._rx_last_new_samples = 0
            self._rx_stream_updates = 0
            self._rx_latest_window_len = 0
            self._rx_last_update_wall = 0.0
            self._rx_spectrum_stale = True
            self._rx_spectrum_stale_sec = float("inf")
            self._rx_samples_seen = 0
            self._rx_probe_total_est = 0
            self._rx_probe_last_fp = None
            self._last_processed_abs_start = -10 ** 18
            self._last_process_t = 0.0
            if reset_counters:
                self._frames_processed = 0
                self._frames_decode_ok = 0
                self._ber_hist_t.clear()
                self._ber_hist_v.clear()
        self._debug("INFO", f"runtime RX state reset: reason={reason}, reset_counters={bool(reset_counters)}")

    def _arm_startup_settle(self):
        now = time.time()
        self._rx_settle_until_wall = now + float(getattr(self, "startup_settle_sec", 0.0))
        self._rx_settle_windows_remaining = int(max(0, getattr(self, "startup_settle_windows", 0)))
        self._debug("INFO",
                    f"startup settle armed: {self.startup_settle_sec*1000:.0f} ms + "
                    f"{self._rx_settle_windows_remaining} fresh RX vector(s)")

    def _in_startup_settle(self, rx_new_samples: int) -> bool:
        now = time.time()
        settling_time = now < float(getattr(self, "_rx_settle_until_wall", 0.0))
        if int(rx_new_samples) > 0 and int(getattr(self, "_rx_settle_windows_remaining", 0)) > 0:
            self._rx_settle_windows_remaining = max(0, int(self._rx_settle_windows_remaining) - 1)
        settling_windows = int(getattr(self, "_rx_settle_windows_remaining", 0)) > 0
        return bool(settling_time or settling_windows)

    def _waveform_fingerprint(self) -> str:

        wf = np.asarray(self._tx_waveform, dtype=np.complex64)
        if wf.size == 0:
            return "len=0"
        # Total L2 energy is invariant under exact unitary FSIT only up to
        # normalization; in practice it shifts measurably with alpha/beta
        # because the peak-norm step rescales differently for each X_TF.
        energy = float(np.sum(np.abs(wf) ** 2))
        # Stable lightweight hash for human diagnostics. Avoid uint64 summation
        # because NumPy may warn about intentional wraparound on Windows/Python.
        sample_n = int(min(4096, wf.size))
        if sample_n > 0:
            sample_idx = np.linspace(0, wf.size - 1, sample_n, dtype=np.int64)
            fold = int(zlib.crc32(wf[sample_idx].tobytes()) & 0xFFFFFFFF)
        else:
            fold = 0
        # Sample from the middle of the data segment (after pre_guard + sync + pilot),
        # which is the alpha/beta-dependent piece.
        mid_idx = self.pre_guard_len + self.sync_len + self.pilot_frame_len + (self.data_frame_len // 2)
        mid_idx = max(0, min(mid_idx, wf.size - 1))
        mid = complex(wf[mid_idx])
        return (f"len={wf.size}, base_cycle={int(getattr(self, '_tx_base_cycle_len', 0))}, "
                f"repeats={int(getattr(self, '_tx_uhd_repeats', 1))}, "
                f"coding={self._coding_summary()}, tdl_prerender={bool(getattr(self, '_tx_tdl_prerendered', False))}, "
                f"energy={energy:.3f}, hash=0x{fold:08x}, "
                f"data_mid_idx={mid_idx}, data_mid={mid:.4f}")

    def get_waveform_fingerprint(self) -> str:
        """Public API: read the current TX waveform fingerprint without
        triggering a configure(). Useful for the UI to add a quick
        "is the waveform actually different now?" check."""
        return self._waveform_fingerprint()

    def _sync_waveform_to_top_block(self):
        """Push the current TX vector into the GNU Radio source.

        v32/v33-pre rebuilt the entire top_block from this offline path.  That made one
        UI parameter application trigger two UHD teardown/build cycles: once
        here and once again in configure().  v33 keeps this method side-effect
        bounded: when stopped, it updates vector_source_c in place whenever
        possible and only queues a rebuild if the vector source cannot be
        updated or configure() already knows the graph structure changed.
        """

        if self._tb is None or self._vector_source is None:
            self._debug("WARN", "_sync_waveform_to_top_block: no top_block/vector_source yet; queued rebuild")
            self._needs_top_block_rebuild = True
            return

        fingerprint = self._waveform_fingerprint()
        data_list = self._tx_waveform.astype(np.complex64).tolist()
        path = "live swap" if self._running else "offline set_data"
        self._debug("INFO", f"sync ({path} path): {fingerprint}")
        try:
            try:
                self._vector_source.set_data(data_list, [])
            except TypeError:
                self._vector_source.set_data(data_list)
            try:
                self._vector_source.rewind()
            except Exception:
                pass
            if getattr(self, "_tdl_channel_block", None) is not None:
                try:
                    self._tdl_channel_block.reset_channel()
                except Exception:
                    pass
            self._needs_top_block_rebuild = False
            self._reset_rx_runtime_state(reason=f"waveform_sync_{path.replace(' ', '_')}", reset_counters=False)
            self._debug("INFO", f"{path}: set_data() + rewind ok")
        except Exception as e:
            self._debug("WARN",
                        f"{path}: set_data() failed ({type(e).__name__}: {e}); "
                        f"queued top_block rebuild for next start/configure")
            self._needs_top_block_rebuild = True

    # Legacy alias kept for any external caller.
    def _push_new_waveform_to_source(self):
        self._sync_waveform_to_top_block()

    def configure(
            self,
            carrier_freq: Optional[float] = None,
            samp_rate: Optional[float] = None,
            tx_gain: Optional[float] = None,
            rx_gain: Optional[float] = None,
            tx_text: Optional[str] = None,
            mod_order: Optional[str] = None,
            equalizer: Optional[str] = None,
            alpha: Optional[float] = None,
            beta: Optional[float] = None,
            fdidm_m: Optional[int] = None,
            fdidm_n: Optional[int] = None,
            cp_len: Optional[int] = None,
            tx_frame_count: Optional[int] = None,
            inter_frame_guard_len: Optional[int] = None,
            evm_average_frames: Optional[int] = None,
            training_amplitude: Optional[float] = None,
            training_probe_guard_len: Optional[int] = None,  # legacy; ignored
            max_full_htf_order: Optional[int] = None,  # legacy
            device_type: Optional[str] = None,
            channel_estimator: Optional[str] = None,
            full_htf_update_interval_frames: Optional[int] = None,
            full_htf_once: Optional[bool] = None,
            process_interval_ms: Optional[float] = None,
            usrp_buffer_frames: Optional[int] = None,
            tx_min_waveform_duration_ms: Optional[float] = None,
            tx_max_waveform_samples: Optional[int] = None,
            tx_prerender_tdl_before_rf: Optional[bool] = None,
            enable_realtime_scheduling: Optional[bool] = None,
            cfo_search_enable: Optional[bool] = None,
            cfo_search_max_hz: Optional[float] = None,
            residual_cfo_max_hz: Optional[float] = None,
            startup_settle_ms: Optional[float] = None,
            startup_settle_windows: Optional[int] = None,
            cfo_scan_min_score: Optional[float] = None,
            cfo_scan_jump_guard_hz: Optional[float] = None,
            coding_scheme: Optional[str] = None,
            coding_interleaver: Optional[bool] = None,
            channel_mode: Optional[str] = None,
            software_channel_model: Optional[str] = None,
            tdl_rms_delay_spread_ns: Optional[float] = None,
            tdl_doppler_hz: Optional[float] = None,
            tdl_doppler_spread_hz: Optional[float] = None,
            tdl_snr_db: Optional[float] = None,
            tdl_seed: Optional[int] = None,
            tdl_normalize_power: Optional[bool] = None,
            tdl_param_num_sinusoids: Optional[int] = None,
            tdl_param_max_paths: Optional[int] = None,
            tdl_param_ridge: Optional[float] = None,
            tdl_param_prune_db: Optional[float] = None,
            auto_tdl_param_for_software: Optional[bool] = None,
            adaptive_alpha_beta_enable: Optional[bool] = None,
            adaptive_alpha_beta_coarse_step: Optional[float] = None,
            adaptive_alpha_beta_fine_step: Optional[float] = None,
            adaptive_alpha_beta_interval_frames: Optional[int] = None,
            adaptive_alpha_beta_min_improvement_db: Optional[float] = None,
            adaptive_alpha_beta_stability_evals: Optional[int] = None,
            adaptive_alpha_beta_cooldown_frames: Optional[int] = None,
            adaptive_alpha_beta_integer_margin_db: Optional[float] = None,
            adaptive_alpha_beta_max_order: Optional[int] = None,
            adaptive_alpha_beta_min_sync_metric: Optional[float] = None,
            adaptive_alpha_beta_require_good_frame: Optional[bool] = None,
            adaptive_alpha_beta_rcond: Optional[float] = None,
            **_ignored: Any,
    ):

        adaptive_context_before = self._alpha_beta_adaptation_context_key()

        if self._running:
            # Reject only *changed* parameters that alter the GNU Radio/UHD graph
            # shape or the monitor thread's frame/probe geometry.  The UI passes
            # the full parameter set on every Apply, so checking only for None
            # would incorrectly force a restart for every harmless SNR/alpha edit.
            must_stop = False
            if samp_rate is not None and float(samp_rate) != self.sample_rate:
                must_stop = True
            if device_type is not None and str(device_type) != self.device_type:
                must_stop = True
            if fdidm_m is not None and int(max(4, min(int(fdidm_m), 64))) != self.M:
                must_stop = True
            if fdidm_n is not None and int(max(1, min(int(fdidm_n), 64))) != self.N:
                must_stop = True
            if cp_len is not None and int(max(0, min(int(cp_len), max(self.M - 1, 0)))) != self.cp_len:
                must_stop = True
            if inter_frame_guard_len is not None and int(max(0, min(int(inter_frame_guard_len), 8192))) != self.inter_frame_guard_len:
                must_stop = True
            if max_full_htf_order is not None and int(max(16, max_full_htf_order)) != self.max_full_htf_order:
                must_stop = True
            if usrp_buffer_frames is not None and int(max(32, min(int(usrp_buffer_frames), 4096))) != self.usrp_buffer_frames:
                must_stop = True
            if channel_estimator is not None:
                try:
                    must_stop = must_stop or (self._normalize_channel_estimator(channel_estimator) != self.requested_channel_estimator)
                except Exception:
                    must_stop = True
            requested_mode = None
            if software_channel_model is not None:
                requested_mode = self._normalize_channel_mode(software_channel_model)
            elif channel_mode is not None:
                requested_mode = self._normalize_channel_mode(channel_mode)
            if requested_mode is not None and requested_mode != self.channel_mode:
                must_stop = True
            if must_stop:
                raise RuntimeError("Cannot reconfigure structural/channel parameters while running; stop first.")

        rebuild_waveform = False
        rebuild_top_block = False
        alpha_beta_changed = False
        adaptive_config_changed = False
        tdl_live_reconfigure = False
        tdl_metadata_only_changed = False

        def _mark_tdl_parameter_changed(clear_cache: bool = True):
            """Route TDL parameter updates to the cheapest safe path.

            - RF-only: record the value for future software-TDL runs; do not
              touch UHD or the TX vector because no software channel is active.
            - TDL->RF: regenerate the pre-rendered TX vector and push it to the
              existing vector_source_c; no UHD graph rebuild is needed.
            - RF->TDL: reconfigure the live software channel block in place.
            """
            nonlocal rebuild_waveform, tdl_live_reconfigure, tdl_metadata_only_changed
            if clear_cache:
                self._clear_channel_cache()
            if self._tdl_before_rf_enabled():
                rebuild_waveform = True
            elif self._tdl_after_rf_enabled():
                tdl_live_reconfigure = True
            else:
                tdl_metadata_only_changed = True

        if device_type is not None and str(device_type) != self.device_type:
            self.device_type = str(device_type);
            rebuild_top_block = True
        if carrier_freq is not None and float(carrier_freq) != self.carrier_freq:
            self.carrier_freq = float(carrier_freq)
            if getattr(self, "_usrp_source", None) is not None and getattr(self, "_usrp_sink", None) is not None:
                try:
                    self._usrp_source.set_center_freq(self.carrier_freq, 0)
                    self._usrp_sink.set_center_freq(self.carrier_freq, 0)
                except Exception:
                    rebuild_top_block = True
            else:
                rebuild_top_block = True
        if samp_rate is not None and float(samp_rate) != self.sample_rate:
            self.sample_rate = float(samp_rate);
            self.samp_rate = self.sample_rate
            self.subcarrier_spacing = self.sample_rate / max(self.M, 1)
            rebuild_waveform = True
            self._clear_channel_cache()
            if getattr(self, "_usrp_source", None) is not None and getattr(self, "_usrp_sink", None) is not None:
                try:
                    self._usrp_source.set_samp_rate(self.sample_rate)
                    self._usrp_sink.set_samp_rate(self.sample_rate)
                except Exception:
                    rebuild_top_block = True
            else:
                rebuild_top_block = True
        if tx_gain is not None and float(tx_gain) != self.tx_gain:
            self.tx_gain = float(tx_gain)
            if getattr(self, "_usrp_sink", None) is not None:
                try:
                    self._usrp_sink.set_gain(self.tx_gain, 0)
                except Exception:
                    pass
        if rx_gain is not None and float(rx_gain) != self.rx_gain:
            self.rx_gain = float(rx_gain)
            if getattr(self, "_usrp_source", None) is not None:
                try:
                    self._usrp_source.set_gain(self.rx_gain, 0)
                except Exception:
                    pass
        if mod_order is not None and str(mod_order).upper() != self.mod_order:
            self.mod_order = str(mod_order).upper()
            self.bits_per_symbol = self._get_bits_per_symbol(self.mod_order)
            rebuild_waveform = True
        if equalizer is not None and str(equalizer).upper() != self.equalizer:
            self.equalizer = str(equalizer).upper()
            if self.equalizer not in ("ZF", "MMSE"):
                raise ValueError("Unsupported equalizer")
        if alpha is not None and float(alpha) != self.alpha:
            self.alpha = float(alpha)
            rebuild_waveform = True
            alpha_beta_changed = True
        if beta is not None and float(beta) != self.beta:
            self.beta = float(beta)
            rebuild_waveform = True
            alpha_beta_changed = True
        if alpha_beta_changed:
            self._note_alpha_beta_changed_for_adaptation(reason="configure_alpha_beta_changed")
        if fdidm_m is not None:
            new_m = int(max(4, min(int(fdidm_m), 64)))
            if new_m != self.M:
                self.M = new_m
                rebuild_waveform = True
                rebuild_top_block = True
        if fdidm_n is not None:
            new_n = int(max(1, min(int(fdidm_n), 64)))
            if new_n != self.N:
                self.N = new_n
                rebuild_waveform = True
                rebuild_top_block = True
        if cp_len is not None:
            new_cp = int(max(0, min(int(cp_len), max(self.M - 1, 0))))
            if new_cp != self.cp_len:
                self.cp_len = new_cp
                rebuild_waveform = True
                rebuild_top_block = True
        if tx_frame_count is not None:
            new_fc = int(max(1, min(int(tx_frame_count), 32)))
            if new_fc != self.tx_frame_count:
                self.tx_frame_count = new_fc;
                rebuild_waveform = True
        if inter_frame_guard_len is not None:
            new_ig = int(max(0, min(int(inter_frame_guard_len), 8192)))
            if new_ig != self.inter_frame_guard_len:
                self.inter_frame_guard_len = new_ig
                rebuild_waveform = True
                rebuild_top_block = True
        if evm_average_frames is not None:
            new_ev = int(max(1, min(int(evm_average_frames), 128)))
            if new_ev != self.evm_average_frames:
                self.evm_average_frames = new_ev
                self._evm_history = deque(maxlen=self.evm_average_frames)
                self._ab_surface_samples_per_cell = int(max(1, min(int(self.evm_average_frames), 128)))
        if training_amplitude is not None:
            new_ta = float(max(0.05, min(float(training_amplitude), 4.0)))
            if new_ta != self.training_amplitude:
                self.training_amplitude = new_ta;
                rebuild_waveform = True
        if training_probe_guard_len is not None:
            # legacy: accepted but unused in v17
            self.training_probe_guard_len = int(max(0, min(int(training_probe_guard_len), 8192)))
        if max_full_htf_order is not None:
            new_max = int(max(16, max_full_htf_order))
            if new_max != self.max_full_htf_order:
                self.max_full_htf_order = new_max
                rebuild_waveform = True
                rebuild_top_block = True
        if channel_estimator is not None:
            new_ce = self._normalize_channel_estimator(channel_estimator)
            self.requested_channel_estimator = new_ce
            if new_ce != self.channel_estimator:
                self.channel_estimator = new_ce
                rebuild_waveform = True
                rebuild_top_block = True
        if full_htf_update_interval_frames is not None:
            self.full_htf_update_interval_frames = int(max(1, min(int(full_htf_update_interval_frames), 10_000)))
        if process_interval_ms is not None:
            self.process_interval_sec = max(0.03, float(process_interval_ms) / 1000.0)
        if usrp_buffer_frames is not None:
            v = int(max(32, min(int(usrp_buffer_frames), 4096)))
            if v != self.usrp_buffer_frames:
                self.usrp_buffer_frames = v
                rebuild_top_block = True
        if tx_min_waveform_duration_ms is not None:
            v = float(max(0.0, min(float(tx_min_waveform_duration_ms), 5000.0)))
            if v != self.tx_min_waveform_duration_ms:
                self.tx_min_waveform_duration_ms = v
                rebuild_waveform = True
        if tx_max_waveform_samples is not None:
            v = int(max(8192, min(int(tx_max_waveform_samples), 4_194_304)))
            if v != self.tx_max_waveform_samples:
                self.tx_max_waveform_samples = v
                rebuild_waveform = True
        if tx_prerender_tdl_before_rf is not None and not bool(tx_prerender_tdl_before_rf):
            self._debug("WARN", "TDL->RF live Python channel was requested but is disabled in v33; TX-side TDL remains pre-rendered to protect UHD from underflow")
        self.tx_prerender_tdl_before_rf = True
        if enable_realtime_scheduling is not None:
            self.enable_realtime_scheduling = bool(enable_realtime_scheduling)
        if cfo_search_enable is not None:
            v = bool(cfo_search_enable)
            if v != self.cfo_search_enable:
                self.cfo_search_enable = v
        if cfo_search_max_hz is not None:
            v = float(max(0.0, min(float(cfo_search_max_hz), max(self.sample_rate / 2.0, 1.0))))
            if v != self.cfo_search_max_hz:
                self.cfo_search_max_hz = v
        if residual_cfo_max_hz is not None:
            v = float(max(0.0, min(float(residual_cfo_max_hz), max(self.sample_rate / 4.0, 1.0))))
            if v != self.residual_cfo_max_hz:
                self.residual_cfo_max_hz = v
        if startup_settle_ms is not None:
            self.startup_settle_sec = float(max(0.0, min(float(startup_settle_ms) / 1000.0, 5.0)))
        if startup_settle_windows is not None:
            self.startup_settle_windows = int(max(0, min(int(startup_settle_windows), 32)))
        if cfo_scan_min_score is not None:
            self.cfo_scan_min_score = float(max(0.0, min(float(cfo_scan_min_score), 1.0)))
        if cfo_scan_jump_guard_hz is not None:
            self.cfo_scan_jump_guard_hz = float(max(0.0, min(float(cfo_scan_jump_guard_hz), max(self.sample_rate / 2.0, 1.0))))
        if coding_scheme is not None:
            v = self._normalize_coding_scheme(coding_scheme)
            if v != self.coding_scheme:
                self.coding_scheme = v
                rebuild_waveform = True
        if coding_interleaver is not None:
            v = bool(coding_interleaver)
            if v != self.coding_interleaver:
                self.coding_interleaver = v
                rebuild_waveform = True
        if full_htf_once is not None:
            self.full_htf_once = bool(full_htf_once)
        if auto_tdl_param_for_software is not None:
            self.auto_tdl_param_for_software = bool(auto_tdl_param_for_software)

        if adaptive_alpha_beta_enable is not None:
            v = bool(adaptive_alpha_beta_enable)
            if v != self.adaptive_alpha_beta_enable:
                self.adaptive_alpha_beta_enable = v
                adaptive_config_changed = True
        if adaptive_alpha_beta_coarse_step is not None:
            v = float(max(0.05, min(float(adaptive_alpha_beta_coarse_step), 1.0)))
            if v != self.adaptive_alpha_beta_coarse_step:
                self.adaptive_alpha_beta_coarse_step = v; adaptive_config_changed = True
        if adaptive_alpha_beta_fine_step is not None:
            v = float(max(0.01, min(float(adaptive_alpha_beta_fine_step), self.adaptive_alpha_beta_coarse_step)))
            if v != self.adaptive_alpha_beta_fine_step:
                self.adaptive_alpha_beta_fine_step = v; adaptive_config_changed = True
        # Coarse may have been reduced after the fine value was read.
        self.adaptive_alpha_beta_fine_step = min(self.adaptive_alpha_beta_fine_step, self.adaptive_alpha_beta_coarse_step)
        if adaptive_alpha_beta_interval_frames is not None:
            v = int(max(1, min(int(adaptive_alpha_beta_interval_frames), 1024)))
            if v != self.adaptive_alpha_beta_interval_frames:
                self.adaptive_alpha_beta_interval_frames = v; adaptive_config_changed = True
        if adaptive_alpha_beta_min_improvement_db is not None:
            v = float(max(0.0, min(float(adaptive_alpha_beta_min_improvement_db), 30.0)))
            if v != self.adaptive_alpha_beta_min_improvement_db:
                self.adaptive_alpha_beta_min_improvement_db = v; adaptive_config_changed = True
        if adaptive_alpha_beta_stability_evals is not None:
            v = int(max(1, min(int(adaptive_alpha_beta_stability_evals), 16)))
            if v != self.adaptive_alpha_beta_stability_evals:
                self.adaptive_alpha_beta_stability_evals = v; adaptive_config_changed = True
        if adaptive_alpha_beta_cooldown_frames is not None:
            v = int(max(0, min(int(adaptive_alpha_beta_cooldown_frames), 4096)))
            if v != self.adaptive_alpha_beta_cooldown_frames:
                self.adaptive_alpha_beta_cooldown_frames = v; adaptive_config_changed = True
        if adaptive_alpha_beta_integer_margin_db is not None:
            v = float(max(0.0, min(float(adaptive_alpha_beta_integer_margin_db), 6.0)))
            if v != self.adaptive_alpha_beta_integer_margin_db:
                self.adaptive_alpha_beta_integer_margin_db = v; adaptive_config_changed = True
        if adaptive_alpha_beta_max_order is not None:
            v = int(max(16, min(int(adaptive_alpha_beta_max_order), 4096)))
            if v != self.adaptive_alpha_beta_max_order:
                self.adaptive_alpha_beta_max_order = v; adaptive_config_changed = True
        if adaptive_alpha_beta_min_sync_metric is not None:
            v = float(max(0.0, min(float(adaptive_alpha_beta_min_sync_metric), 1.0)))
            if v != self.adaptive_alpha_beta_min_sync_metric:
                self.adaptive_alpha_beta_min_sync_metric = v; adaptive_config_changed = True
        if adaptive_alpha_beta_require_good_frame is not None:
            v = bool(adaptive_alpha_beta_require_good_frame)
            if v != self.adaptive_alpha_beta_require_good_frame:
                self.adaptive_alpha_beta_require_good_frame = v; adaptive_config_changed = True
        if adaptive_alpha_beta_rcond is not None:
            v = float(max(1e-12, min(float(adaptive_alpha_beta_rcond), 1e-1)))
            if v != self.adaptive_alpha_beta_rcond:
                self.adaptive_alpha_beta_rcond = v; adaptive_config_changed = True

        if adaptive_config_changed:
            self._invalidate_alpha_beta_adaptation(reason="adaptive_config_changed", cooldown=False)
            if self.adaptive_alpha_beta_enable:
                self._ensure_alpha_beta_adaptation_worker()
            else:
                with self._adaptive_ab_lock:
                    self._adaptive_ab_state = "disabled"

        # Channel settings. Every mode traverses USRP RF; optional TDL stages either
        # run after RX or are pre-rendered before TX. Any change invalidates CSI.
        new_channel_mode = None
        if software_channel_model is not None or channel_mode is not None:
            self._channel_mode_note = ""
        if software_channel_model is not None:
            new_channel_mode = self._normalize_channel_mode(software_channel_model)
        elif channel_mode is not None:
            new_channel_mode = self._normalize_channel_mode(channel_mode)
        if getattr(self, "_channel_mode_note", ""):
            self._debug("WARN", self._channel_mode_note)
        if new_channel_mode is not None and new_channel_mode != self.channel_mode:
            old_channel_mode = str(self.channel_mode)
            self.channel_mode = new_channel_mode
            rebuild_top_block = True
            rebuild_waveform = True  # channel direction can change whether TX is TDL-pre-rendered
            self._clear_channel_cache()
            self._debug("INFO", f"channel_mode changed: {old_channel_mode} -> {self.channel_mode}; TX waveform/graph will be refreshed")

        if tdl_rms_delay_spread_ns is not None:
            v = float(max(0.0, float(tdl_rms_delay_spread_ns)))
            if v != self.tdl_rms_delay_spread_ns:
                self.tdl_rms_delay_spread_ns = v
                _mark_tdl_parameter_changed(clear_cache=True)
        if tdl_doppler_hz is not None:
            v = float(tdl_doppler_hz)
            if v != self.tdl_doppler_hz:
                self.tdl_doppler_hz = v
                _mark_tdl_parameter_changed(clear_cache=True)
        if tdl_doppler_spread_hz is not None:
            v = float(max(0.0, float(tdl_doppler_spread_hz)))
            if v != self.tdl_doppler_spread_hz:
                self.tdl_doppler_spread_hz = v
                _mark_tdl_parameter_changed(clear_cache=True)
        if tdl_snr_db is not None:
            v = float(tdl_snr_db)
            if v != self.tdl_snr_db:
                self.tdl_snr_db = v
                # SNR does not change deterministic CSI, but it does change the
                # active software channel if one is present.  In RF-only mode it
                # is only a stored value for the next software-TDL run.
                _mark_tdl_parameter_changed(clear_cache=False)
        if tdl_seed is not None:
            v = int(tdl_seed) & 0xFFFFFFFF
            if v != self.tdl_seed:
                self.tdl_seed = v
                _mark_tdl_parameter_changed(clear_cache=True)
        if tdl_normalize_power is not None:
            v = bool(tdl_normalize_power)
            if v != self.tdl_normalize_power:
                self.tdl_normalize_power = v
                _mark_tdl_parameter_changed(clear_cache=True)
        if tdl_param_num_sinusoids is not None:
            v = int(max(4, min(int(tdl_param_num_sinusoids), 64)))
            if v != self.tdl_param_num_sinusoids:
                self.tdl_param_num_sinusoids = v
                _mark_tdl_parameter_changed(clear_cache=True)
        if tdl_param_max_paths is not None:
            v = int(max(1, min(int(tdl_param_max_paths), 512)))
            if v != self.tdl_param_max_paths:
                self.tdl_param_max_paths = v
                self._clear_channel_cache()
        if tdl_param_ridge is not None:
            v = float(max(0.0, float(tdl_param_ridge)))
            if v != self.tdl_param_ridge:
                self.tdl_param_ridge = v
        if tdl_param_prune_db is not None:
            v = float(min(0.0, float(tdl_param_prune_db)))
            if v != self.tdl_param_prune_db:
                self.tdl_param_prune_db = v

        if rebuild_top_block and self._tdl_before_rf_enabled():
            # TDL->RF uses an RF-only live TX graph; TDL changes must regenerate
            # the pre-rendered TX vector as well as rebuild the RX graph.
            rebuild_waveform = True

        self._estimator_auto_note = ""
        if self._guard_estimator_for_software_tdl():
            rebuild_waveform = True
            rebuild_top_block = True
            self._clear_channel_cache()

        if tdl_live_reconfigure and not rebuild_waveform and not rebuild_top_block:
            block = getattr(self, "_tdl_channel_block", None)
            if block is not None and hasattr(block, "configure_channel"):
                block.configure_channel(
                    sample_rate=self.sample_rate,
                    model=self._tdl_model_for_current_mode(),
                    rms_delay_spread_ns=self.tdl_rms_delay_spread_ns,
                    doppler_hz=self.tdl_doppler_hz,
                    doppler_spread_hz=self.tdl_doppler_spread_hz,
                    snr_db=self.tdl_snr_db,
                    seed=self.tdl_seed,
                    normalize_power=self.tdl_normalize_power,
                    num_sinusoids=self.tdl_param_num_sinusoids,
                )
                self._clear_channel_cache()
                self._reset_rx_runtime_state(reason="tdl_live_reconfigure", reset_counters=False)
                if self._running:
                    self._arm_startup_settle()
                self._debug("INFO",
                            f"live software TDL channel updated without UHD rebuild: "
                            f"mode={self.channel_mode}, DS={self.tdl_rms_delay_spread_ns:.1f}ns, "
                            f"fd={self.tdl_doppler_hz:.1f}Hz, spread={self.tdl_doppler_spread_hz:.1f}Hz, "
                            f"SNR={self.tdl_snr_db:.1f}dB")
            elif self._tdl_after_rf_enabled():
                if self._running:
                    raise RuntimeError("Cannot live-update RF->TDL block because the TDL block is unavailable; stop first.")
                rebuild_top_block = True

        if tdl_metadata_only_changed and not (rebuild_waveform or rebuild_top_block or tdl_live_reconfigure):
            self._debug("INFO",
                        f"TDL parameter recorded for future software-channel use only: "
                        f"channel_mode={self.channel_mode}, SNR={self.tdl_snr_db:.1f}dB; "
                        "active RF-only graph/waveform left unchanged")

        adaptive_context_after = self._alpha_beta_adaptation_context_key()
        if adaptive_context_after != adaptive_context_before:
            self._invalidate_alpha_beta_adaptation(reason="objective_context_changed", cooldown=False)

        tx_text_changed = (tx_text is not None and str(tx_text) != self._tx_text)
        if rebuild_waveform or tx_text_changed:
            self._debug("INFO",
                        f"configure() applying changes: "
                        f"M={self.M} N={self.N} CP={self.cp_len} alpha={self.alpha:.3f} beta={self.beta:.3f} "
                        f"mod={self.mod_order} eq={self.equalizer} coding={self._coding_summary()} estimator={self.channel_estimator} "
                        f"H_once={self.full_htf_once}, H_update_legacy={self.full_htf_update_interval_frames}, "
                        f"channel_mode={self.channel_mode}, process_interval={self.process_interval_sec*1000:.0f}ms, tx_text_len="
                        f"{len(self._tx_text) if tx_text is None else len(str(tx_text))} bytes")
            self._gamma_cache.clear()
            self._recompute_strict_frame_timing()
            self._rebuild_pilot_matrices()
            self._clear_channel_cache()
            self._buffer_keep = max(262144, 8 * self.frame_len)
            with self._lock:
                self._tx_buffer = _SampleRing(self._buffer_keep)
                self._rx_buffer = _SampleRing(self._buffer_keep)
                self._rx_probe_last_fp = None
                self._rx_probe_total_est = 0
                self._tx_preview_start_t = time.time()
            if getattr(self, "_estimator_auto_note", ""):
                self._debug("WARN", self._estimator_auto_note)
            self._set_tx_text_internal(self._tx_text if tx_text is None else str(tx_text))
            self._tx_buffer.write(self._tx_waveform.astype(np.complex64, copy=False))
            # Always sync the new waveform to the GNU Radio vector source, even
            # while stopped, so the next start cannot replay stale samples.
            self._sync_waveform_to_top_block()
            if self._running:
                self._arm_startup_settle()
            self._debug("INFO", self._format_link_limit_summary())

        if rebuild_top_block and not self._running:
            self._debug("INFO",
                        f"configure() rebuilding top_block for path={self.channel_mode}, device_type={self.device_type}, "
                        f"TDL_DS={self.tdl_rms_delay_spread_ns:.1f}ns, fd={self.tdl_doppler_hz:.1f}Hz, "
                        f"spread={self.tdl_doppler_spread_hz:.1f}Hz, SNR={self.tdl_snr_db:.1f}dB")
            self._usrp_args = self._build_device_args()
            self._build_top_block()
            self._debug("INFO", self._format_link_limit_summary())

    # =========================================================
    # Runtime
    # =========================================================
    def start(self):
        if self._tb is None:
            raise RuntimeError("top_block not built")
        if self._running:
            self._debug("WARN", "start() called but already running; ignored")
            return
        if self._needs_top_block_rebuild:
            self._debug("INFO", "start(): applying queued top_block rebuild before launch")
            try:
                self._build_top_block()
                self._needs_top_block_rebuild = False
            except Exception as e:
                self._debug("ERROR", f"start(): queued rebuild failed: {type(e).__name__}: {e}")
                raise
        self._run_id = int(getattr(self, "_run_id", 0)) + 1
        self._invalidate_alpha_beta_adaptation(reason=f"start_run_{self._run_id}", cooldown=False)
        self._reset_rx_runtime_state(reason=f"start_run_{self._run_id}", reset_counters=True)
        if bool(getattr(self, "enable_realtime_scheduling", True)):
            try:
                rt = self._gr.enable_realtime_scheduling()
                self._debug("INFO", f"start(): GNU Radio realtime scheduling request returned {rt}")
            except Exception as e:
                self._debug("WARN", f"start(): realtime scheduling unavailable: {type(e).__name__}: {e}")
        self._debug("INFO",
                    f"start(): launching top_block, TX waveform len={self._tx_waveform.size}, "
                    f"base_cycle={self._tx_base_cycle_len}, repeats={self._tx_uhd_repeats}, "
                    f"frame_len={self.frame_len}, alpha={self.alpha:.3f}, beta={self.beta:.3f}, "
                    f"mod={self.mod_order}, eq={self.equalizer}, Fs={self.sample_rate:.0f} Hz, "
                    f"coding={self._coding_summary()}, H_once={self.full_htf_once}, channel_mode={self.channel_mode}, "
                    f"tdl_prerender={self._tx_tdl_prerendered}, "
                    f"TDL_fd={self.tdl_doppler_hz:.1f} Hz")
        self._tb.start()
        self._tx_preview_start_t = time.time()
        self._rx_probe_start_t = time.time()
        self._arm_startup_settle()
        self._last_process_t = 0.0
        self._rx_probe_total_est = 0
        self._rx_probe_last_fp = None
        self._running = True
        self._status = "running"
        self._monitor_stop.clear()
        self._monitor_cycles = 0
        self._monitor_last_log_t = time.time()
        self._monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        self._monitor_thread.start()
        self._debug("INFO", "start(): monitor thread launched")

    def stop(self):
        if not self._running:
            self._debug("INFO", "stop() called but not running; ignored")
            return
        self._debug("INFO",
                    f"stop(): tearing down, frames_processed={self._frames_processed}, "
                    f"frames_decode_ok={self._frames_decode_ok}, rx_samples_seen={self._rx_samples_seen}")
        # Order is important: stop the monitor before tearing down the GR sinks,
        # otherwise the monitor can call .data() on a half-destructed vector sink.
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=3.0)
            if self._monitor_thread.is_alive():
                self._debug("WARN", "stop(): monitor thread did not exit within 3 s")
            self._monitor_thread = None
        try:
            self._tb.stop()
            self._tb.wait()
        except Exception as e:
            self._debug("WARN", f"top_block stop error: {type(e).__name__}: {e}")
        # Give UHD a moment to actually release the USB endpoint before any restart.
        time.sleep(0.25)
        self._running = False
        self._status = "stopped"
        self._debug("INFO", "stop(): backend stopped cleanly")

    def wait(self, timeout: Optional[float] = 2.0):
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=timeout)

    def _read_rx_probe_window(self, desired_len: int) -> tuple[np.ndarray, int, int, int]:
        """Read the latest contiguous RX window.

        Returns (rx_window, abs_seen_est, rx_buf_size, newly_seen_est).
        In probe_signal_vc mode, abs_seen is wall-clock based because the C++
        probe intentionally does not expose a cumulative sample counter.  This
        is enough for de-duplicating frame attempts and for rate display; the
        actual samples used for decoding come from the probe vector.
        """
        desired_len = int(max(1, desired_len))
        if self._rx_probe is not None and self._rx_probe_mode == "probe_signal_vc":
            try:
                vec = np.asarray(self._rx_probe.level(), dtype=np.complex64).reshape(-1)
            except Exception as e:
                self._debug("WARN", f"rx_probe.level() failed: {type(e).__name__}: {e}")
                return np.zeros(0, dtype=np.complex64), int(self._rx_samples_seen), 0, 0
            if vec.size <= 0:
                return np.zeros(0, dtype=np.complex64), int(self._rx_samples_seen), 0, 0
            if vec.size > desired_len:
                vec = vec[-desired_len:].copy()
            else:
                vec = vec.copy()
            # Avoid reprocessing the same C++ probe vector multiple times when
            # the UI/monitor polls faster than stream_to_vector produces a new
            # block.  The actual RF waveform is noisy, so a small sampled CRC
            # is enough to detect genuine updates without hashing megabytes.
            sample_count = int(min(32, vec.size))
            fp_idx = np.linspace(0, vec.size - 1, sample_count, dtype=np.int64)
            fp = (int(vec.size), int(zlib.crc32(vec[fp_idx].tobytes()) & 0xFFFFFFFF))
            if fp == self._rx_probe_last_fp:
                return vec, int(self._rx_samples_seen), int(vec.size), 0
            self._rx_probe_last_fp = fp
            now = time.time()
            if self._rx_probe_start_t <= 0.0:
                self._rx_probe_start_t = now
            abs_est = int(max(0.0, (now - self._rx_probe_start_t) * max(self.sample_rate, 1.0)))
            if abs_est <= int(self._rx_samples_seen):
                abs_est = int(self._rx_samples_seen) + max(1, int(vec.size))
            newly = int(max(0, abs_est - int(self._rx_samples_seen)))
            return vec, abs_est, int(vec.size), newly

        vec, total, count = self._rx_buffer.read_latest(desired_len)
        newly = int(max(0, int(total) - int(self._rx_samples_seen)))
        return vec, int(total), int(count), newly

    def _monitor_worker(self):
        process_window_len = min(
            self._buffer_keep,
            max(3 * (self.frame_len + self.inter_frame_guard_len), 8192),
        )
        self._debug("INFO",
                    f"monitor: started, process_window_len={process_window_len}, "
                    f"buffer_keep={self._buffer_keep}, update_period={self.update_period * 1000:.0f} ms")
        while not self._monitor_stop.is_set():
            try:
                self._monitor_cycles += 1
                rx_window, abs_seen, rx_buf_size, rx_data_size = self._read_rx_probe_window(process_window_len)
                tx_data = self.get_tx_waveform_preview(min(8192, max(1, self._tx_waveform.size)))
                now = time.time()
                with self._lock:
                    self._rx_samples_seen = int(abs_seen)
                    self._latest_rx_samples = rx_window[-8192:].astype(np.complex64, copy=True)
                    self._latest_tx_samples = tx_data[-8192:].astype(np.complex64, copy=True)
                    self._rx_last_new_samples = int(rx_data_size)
                    self._rx_latest_window_len = int(rx_window.size)
                    if rx_data_size > 0:
                        self._rx_stream_updates = int(getattr(self, "_rx_stream_updates", 0)) + 1
                        self._rx_last_update_wall = now
                        self._rx_spectrum_stale = False
                        self._rx_spectrum_stale_sec = 0.0
                    else:
                        self._rx_spectrum_stale = True
                        if self._rx_last_update_wall > 0.0:
                            self._rx_spectrum_stale_sec = float(now - self._rx_last_update_wall)
                        else:
                            self._rx_spectrum_stale_sec = float("inf")
                    tx_buf_size = len(self._tx_buffer)
                # Heartbeat: log every ~2s so a static UI plot is easy to diagnose.
                if now - self._monitor_last_log_t > 2.0:
                    dt = now - self._monitor_last_log_t
                    rate = float(rx_data_size) / max(dt, 1e-6) if rx_data_size > 0 else 0.0
                    self._debug("INFO",
                                f"monitor heartbeat: cycle={self._monitor_cycles}, "
                                f"TX buf={tx_buf_size}, RX buf={rx_buf_size}, "
                                f"last_tx_chunk={tx_data.size}, last_rx_chunk={rx_data_size}, "
                                f"rx_seen_total={abs_seen}, last_rate~{rate / 1000:.1f} kS/s, "
                                f"rx_mode={self._rx_probe_mode}, probe_len={self._rx_probe_len}, "
                                f"frames_processed={self._frames_processed}, "
                                f"frames_decode_ok={self._frames_decode_ok}")
                    self._monitor_last_log_t = now
                settling = self._in_startup_settle(rx_data_size)
                if settling and rx_data_size > 0:
                    self._last_process_t = now
                    if self._monitor_cycles <= 8 or now - self._monitor_last_log_t > 1.0:
                        self._debug("DEBUG",
                                    f"startup settle: dropping RX vector, new={rx_data_size}, "
                                    f"remaining_windows={int(getattr(self, '_rx_settle_windows_remaining', 0))}, "
                                    f"time_left={max(0.0, float(getattr(self, '_rx_settle_until_wall', 0.0)) - now):.3f}s")
                if (not settling) and rx_data_size > 0 and rx_window.size >= self.frame_len:
                    if now - self._last_process_t >= float(self.process_interval_sec):
                        self._last_process_t = now
                        rx_window = rx_window[-process_window_len:]
                        try:
                            self._try_process_rx_window(rx_window.astype(np.complex128), abs_seen)
                        except Exception as e:
                            self._debug("WARN", f"frame processing exception: {type(e).__name__}: {e}")
                # Wake periodically; respond to stop quickly via Event.wait
                if self._monitor_stop.wait(timeout=max(self.update_period, 0.03)):
                    break
            except Exception as e:
                self._debug("ERROR", f"monitor cycle exception: {type(e).__name__}: {e}")
                if self._monitor_stop.wait(timeout=0.2):
                    break
        self._debug("INFO", f"monitor: exited after {self._monitor_cycles} cycles")

    def _try_process_rx_window(self, rx_window: np.ndarray, abs_seen: int):
        # DSP on real RF captures legitimately produces denormal underflow (tiny
        # values from FFT tails / phasor exponentials) and, on an ill-conditioned
        # channel, transient overflow before the equalizer output is sanitized.
        # These are numerically harmless here - correctness is enforced by the
        # explicit magnitude floors/clamps in the estimators and equalizers - so
        # silence the spurious NumPy warnings around the whole receive pipeline.
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            return self._try_process_rx_window_impl(rx_window, abs_seen)

    def _try_process_rx_window_impl(self, rx_window: np.ndarray, abs_seen: int):
        known_cfo = self._software_known_cfo_for_current_mode()
        sync_rx = np.asarray(rx_window, dtype=np.complex128)
        if known_cfo is not None and abs(float(known_cfo)) > 1e-12:
            n_sync = np.arange(sync_rx.size, dtype=np.float64)
            sync_rx = sync_rx * np.exp(-1j * 2.0 * np.pi * float(known_cfo) * n_sync / max(self.sample_rate, 1e-12))

        metric = self._sync_metric(sync_rx)
        if metric.size <= 1:
            self._debug("DEBUG", f"sync_metric too short: size={metric.size}, rx_window={rx_window.size}")
            return
        max_metric = float(np.max(metric))
        with self._lock:
            self.last_sync_metric = max_metric

        peaks = self._find_sync_peaks(metric, max_candidates=3)
        if not peaks:
            with self._lock:
                self.last_frame_ok = False
                self.last_bad_reason = f"sync_peak_not_found({max_metric:.3f})"
            cfo_note = "" if known_cfo is None else f", knownCFO={float(known_cfo):.1f}Hz"
            self._debug("DEBUG",
                        f"no sync peak above threshold: max_metric={max_metric:.3f}, "
                        f"threshold={self.sync_metric_threshold:.3f}, rx_window={rx_window.size}{cfo_note}")
            return

        self._debug("DEBUG",
                    f"sync peaks found: count={len(peaks)}, max_metric={max_metric:.3f}, "
                    f"first_peak_idx={peaks[0]}, peaks={peaks}")

        best = None
        attempts = 0
        for coarse in peaks:
            attempts += 1
            sync_start, cfo_scan_hz, cfo_scan_score, cfo_alias_hz = self._refine_sync_and_cfo(
                sync_rx, coarse, search_radius=max(16, self.M // 2)
            )
            frame_start = sync_start - self.pre_guard_len
            frame_end = frame_start + self.frame_len
            if frame_start < 0 or frame_end > rx_window.size:
                self._debug("DEBUG",
                            f"peak rejected: sync_start={sync_start}, frame[{frame_start}:{frame_end}] "
                            f"out of rx_window[0:{rx_window.size}]")
                continue
            abs_frame_start = abs_seen - rx_window.size + frame_start
            if abs_frame_start <= self._last_processed_abs_start + self.frame_len // 2:
                self._debug("DEBUG",
                            f"peak rejected: already processed near abs_frame_start={abs_frame_start} "
                            f"(last_processed={self._last_processed_abs_start})")
                continue

            cfo_hz_preamble = float(cfo_scan_hz)
            cfo_known = self._software_known_cfo_for_current_mode()
            if cfo_known is not None:
                cfo_hz = float(cfo_known)
                cfo_source = "software_known_common_doppler"
            else:
                cfo_hz = float(cfo_hz_preamble)
                cfo_source = "preamble_scan" if abs(float(cfo_scan_hz) - float(cfo_alias_hz)) > 0.5 else "preamble"
            frame_raw = rx_window[frame_start:frame_end].copy()
            t_idx = np.arange(frame_raw.size, dtype=np.float64)
            frame = frame_raw * np.exp(-1j * 2.0 * np.pi * cfo_hz * t_idx / max(self.sample_rate, 1e-12))

            pilot_samples = frame[self._off_pilot:self._off_data]
            data_samples = frame[self._off_data:self._off_end]
            if pilot_samples.size != self.pilot_frame_len or data_samples.size != self.data_frame_len:
                self._debug("WARN",
                            f"frame segment size mismatch: pilot={pilot_samples.size}/"
                            f"{self.pilot_frame_len}, data={data_samples.size}/{self.data_frame_len}")
                continue

            # Residual CFO refinement is meaningful for the one-frame diagonal pilot.
            # In full-H_TF mode, the MN probe frames are different basis vectors, so
            # a consecutive-symbol phase ratio would be invalid; rely on the sync CFO.
            res_cfo = 0.0
            if (not self.use_full_htf) and (not self.use_tdl_param_htf):
                # The single-pilot residual CFO estimator assumes a nearly diagonal
                # TF channel. It is useful for diag-TF hardware sanity checks, but
                # it can be biased under TDL multipath/Doppler.  In tdl_param mode
                # the repeated preamble CFO correction is used and the path model
                # handles the remaining TF coupling.
                res_cfo = self._estimate_residual_cfo_from_pilot(pilot_samples)
                if abs(res_cfo) > 0.0:
                    t_idx2 = np.arange(frame.size, dtype=np.float64)
                    frame = frame * np.exp(-1j * 2.0 * np.pi * res_cfo * t_idx2 / max(self.sample_rate, 1e-12))
                    pilot_samples = frame[self._off_pilot:self._off_data]
                    data_samples = frame[self._off_data:self._off_end]
                    cfo_hz = cfo_hz + res_cfo

            htf_cache_refreshing = False
            htf_old_snapshot = None
            try:
                if self.use_full_htf:
                    if self._should_refresh_full_htf():
                        htf_old_snapshot = self._snapshot_channel_cache()
                        h_tf_est, leakage = self._estimate_htf_full_from_pilot(pilot_samples)
                        # Tentatively cache so the equalizer can use the precomputed
                        # H, H^H, and H^H H. The cache is committed only after this
                        # candidate proves good; otherwise it is rolled back below.
                        self._cache_full_htf(h_tf_est, leakage)
                        htf_cache_refreshing = True
                        h_tf_est = self._cached_htf_full
                        h_abs = np.abs(h_tf_est)
                        nz_abs = h_abs[h_abs > 1e-12]
                        if nz_abs.size == 0:
                            nz_abs = np.array([0.0])
                        self._debug("INFO",
                                    f"FULL H_TF tentative update: order={self.full_htf_order}, "
                                    f"estimate_count={self._full_htf_estimates}, reuse_interval={self.full_htf_update_interval_frames}, "
                                    f"|H|_mean={float(np.mean(nz_abs)):.4f}, |H|_min={float(nz_abs.min()):.4e}, "
                                    f"|H|_max={float(nz_abs.max()):.4e}, offdiag_energy={leakage:.3e}, "
                                    f"cond_proxy={self._cached_H_cond_proxy:.2e}")
                    else:
                        h_tf_est = self._cached_htf_full
                        leakage = float(self._cached_htf_leakage)
                elif self.use_tdl_param_htf:
                    # One dense pilot -> small TDL path-gain LS -> reconstructed H_TF.
                    # Re-estimated each frame to follow software TDL time variation.
                    h_tf_est, leakage = self._estimate_htf_tdl_param_from_pilot(pilot_samples)
                else:
                    h_tf_est, leakage, noise_var_cell = self._estimate_htf_diag_from_pilot(pilot_samples)
                    h_abs = np.abs(h_tf_est)
                    self._debug("DEBUG",
                                f"DIAG H_TF stats: |H|_mean={float(np.mean(h_abs)):.4f}, "
                                f"|H|_std={float(np.std(h_abs)):.4f}, "
                                f"|H|_min={float(h_abs.min()):.4f}, |H|_max={float(h_abs.max()):.4f}, "
                                f"dynamic_range={float(h_abs.max() / max(h_abs.min(), 1e-12)):.2e}, "
                                f"H_selectivity(std/mean|H|)={leakage:.3e}, alpha={self.alpha:.3f}, beta={self.beta:.3f}")
            except Exception as e:
                self._debug("WARN", f"pilot estimation failed: {type(e).__name__}: {e}")
                continue

            noise_var = self._estimate_noise_var_from_guard(frame)
            if not np.isfinite(noise_var):
                ref_power = float(np.mean(np.abs(h_tf_est) ** 2)) if np.size(h_tf_est) else 1.0
                noise_var = 0.01 * max(ref_power, 1e-12)
                self._debug("DEBUG", f"noise_var fallback: {noise_var:.3e} (guard sample missing)")
            # In diag mode the pilot residual gives a direct, reliable per-cell noise
            # variance for the MMSE load.  Prefer it over the guard-based estimate,
            # which can be unavailable or biased on short hardware captures.
            if (not self.use_full_htf) and (not self.use_tdl_param_htf):
                nvc = locals().get("noise_var_cell", float("nan"))
                if nvc is not None and np.isfinite(nvc):
                    noise_var = max(float(nvc), 1e-12)
                    self._debug("DEBUG", f"noise_var from pilot residual: {noise_var:.3e}")

            try:
                y_tf_data = self._wigner(data_samples)
            except Exception as e:
                self._debug("WARN", f"wigner failed: {type(e).__name__}: {e}")
                continue

            pre_eq_syms = self._pre_equalized_cross_observation(y_tf_data)
            tf_syms = np.asarray(y_tf_data, dtype=np.complex128).reshape(-1, order="F")
            if self.use_full_htf or self.use_tdl_param_htf:
                x_hat, cond_val, warning = self._equalize_data_full_htf(y_tf_data, h_tf_est, noise_var)
                if self.use_tdl_param_htf:
                    warning = (warning + ";" if warning else "") + f"tdl_param_fit_nmse={self._last_tdl_param_fit_nmse:.2e}"
            else:
                x_hat, cond_val, warning = self._equalize_data_diag(y_tf_data, h_tf_est, noise_var)
            rx_syms = x_hat.reshape(-1, order="F")
            ber, raw_bytes, rx_payload, rx_text, match_bytes, decode_ok, rx_syms_best, evm_inst = (
                self._recover_payload_from_symbols(rx_syms)
            )

            sync_here = float(metric[min(max(int(coarse), 0), len(metric) - 1)])
            self._debug("DEBUG",
                        f"candidate #{attempts}: sync={sync_here:.3f}, CFO={cfo_hz:.1f} Hz({cfo_source}), rawCFO={cfo_hz_preamble:.1f} Hz, "
                        f"alias={float(cfo_alias_hz):.1f}Hz, scanScore={float(cfo_scan_score):.3f}, "
                        f"res_CFO={res_cfo:.1f} Hz, cond={cond_val:.2e}, noise_var={noise_var:.3e}, "
                        f"BER={ber:.3e}, EVM={evm_inst:.2f}%, decode_ok={decode_ok}, "
                        f"match={match_bytes}/{len(self._tx_payload)}, frame_start={frame_start}")

            score = (2000.0 * float(decode_ok)
                     + 200.0 * (1.0 - min(ber, 1.0))
                     + 40.0 * sync_here
                     - 0.05 * min(cond_val, 1e6)
                     - 0.5 * (evm_inst if np.isfinite(evm_inst) else 100.0))
            cand = dict(
                score=score, frame_start=int(frame_start),
                abs_frame_start=int(abs_frame_start),
                sync_metric=sync_here, cfo_hz=float(cfo_hz), cfo_hz_preamble=float(cfo_hz_preamble), cfo_source=str(cfo_source),
                cfo_alias_hz=float(cfo_alias_hz), cfo_scan_score=float(cfo_scan_score),
                htf_leakage=float(leakage), cond_h=float(cond_val),
                noise_var=float(noise_var),
                warning=warning, ber=float(ber),
                raw_bytes=raw_bytes, rx_payload=rx_payload,
                rx_text=rx_text, match_bytes=int(match_bytes),
                decode_ok=bool(decode_ok),
                rx_syms=rx_syms_best,
                rx_syms_post_raw=rx_syms.astype(np.complex64),
                rx_syms_pre_eq=pre_eq_syms.astype(np.complex64),
                rx_syms_tf=tf_syms.astype(np.complex64),
                frame_samples=frame.astype(np.complex64),
                pilot_samples=pilot_samples.astype(np.complex64),
                data_samples=data_samples.astype(np.complex64),
                evm_inst=float(evm_inst),
                adaptive_htf=h_tf_est,
                adaptive_htf_kind=("full" if (self.use_full_htf or self.use_tdl_param_htf) else "diag"),
                adaptive_htf_source=("full_htf" if self.use_full_htf else ("tdl_param" if self.use_tdl_param_htf else "diag_tf")),
                htf_cache_refreshing=bool(htf_cache_refreshing),
                htf_old_snapshot=htf_old_snapshot,
            )
            if best is None or cand["score"] > best["score"]:
                best = cand
            if decode_ok:
                break

        if best is None:
            with self._lock:
                self.last_frame_ok = False
                self.last_bad_reason = f"candidate_decode_fail({max_metric:.3f})"
            self._debug("WARN",
                        f"all {attempts} sync candidate(s) failed: max_metric={max_metric:.3f}, "
                        f"peaks={peaks}")
            return

        self._frames_processed += 1
        if best["decode_ok"]:
            self._frames_decode_ok += 1

        good_quality = bool(
            best["decode_ok"] or (
                best["ber"] < float(self.constellation_soft_ber_threshold) and
                np.isfinite(best["evm_inst"]) and
                best["evm_inst"] < float(self.constellation_soft_evm_threshold)
            )
        )
        # If a freshly estimated full-H_TF did not produce a good frame, roll it
        # back. This prevents one bad estimate from poisoning all later frames.
        if self.use_full_htf and best.get("htf_cache_refreshing") and not good_quality:
            self._restore_channel_cache(best.get("htf_old_snapshot"))
            self._debug("WARN",
                        f"FULL H_TF tentative update rejected: BER={best['ber']:.3e}, "
                        f"EVM={best['evm_inst']:.2f}%, decode_ok={best['decode_ok']}; restored previous cache")
        elif self.use_full_htf and best.get("htf_cache_refreshing"):
            # Mark refresh against the successful decode count after the frame has
            # been accepted, so the reuse interval is measured in good frames.
            self._cached_htf_frame_counter = int(self._frames_decode_ok)

        if np.isfinite(best.get("evm_inst", float("nan"))):
            self.last_evm_instant_percent = float(best["evm_inst"])
        if good_quality:
            self._update_evm_history(best["evm_inst"])
        # Always publish the best candidate constellation as a diagnostic,
        # even if CRC/BER quality gates fail.  The status field
        # last_constellation_source tells the UI whether it is a good frame or a
        # failed-frame diagnostic cloud.
        post_points = self._prepare_constellation_points(best["rx_syms"], display_mode="raw")
        pre_points = self._prepare_constellation_points(best.get("rx_syms_pre_eq", np.zeros(0, dtype=np.complex64)), display_mode="raw")
        tf_points = self._prepare_constellation_points(best.get("rx_syms_tf", np.zeros(0, dtype=np.complex64)), display_mode="raw")
        const_source = "post_eq_good" if good_quality else "post_eq_diagnostic_bad_frame"
        t_now = time.time() - self._t0
        with self._lock:
            self._last_processed_abs_start = best["abs_frame_start"]
            self.last_sync_index = int(best["frame_start"])
            self.last_payload_start = int(best["frame_start"] + self._off_data)
            self.last_sync_metric = float(best["sync_metric"])
            self.last_cfo_est_hz = float(best["cfo_hz"])
            self.last_cfo_preamble_hz = float(best.get("cfo_hz_preamble", best["cfo_hz"]))
            self.last_cfo_source = str(best.get("cfo_source", "preamble"))
            self._last_cfo_alias_hz = float(best.get("cfo_alias_hz", float("nan")))
            self._last_cfo_scan_score = float(best.get("cfo_scan_score", float("nan")))
            self._last_cfo_unambiguous_hz = float(self._preamble_cfo_unambiguous_hz())
            if good_quality and np.isfinite(float(best["cfo_hz"])):
                self._last_good_cfo_hz = float(best["cfo_hz"])
                self._last_good_cfo_wall = time.time()
                self._last_good_cfo_mode_key = self._current_cfo_mode_key()
            self.last_htf_nmse = float(best["htf_leakage"])
            self.last_cond_h_cross = float(best["cond_h"])
            self.last_noise_var = float(best["noise_var"])
            self.last_equalizer_warning = str(best["warning"])
            # v18: meaningful "frame ok". decode_ok is CRC-authoritative; the soft
            # tier now means "link essentially working" (BER < 2%), not the v17
            # BER < 0.45 which is indistinguishable from random bits.
            self.last_frame_ok = bool(best["decode_ok"] or best["ber"] < 0.02)
            self.last_bad_reason = "ok" if best["decode_ok"] else (
                "soft_ok" if best["ber"] < 0.02 else f"high_ber({best['ber']:.2f})"
            )
            self._latest_rx_frame_samples = np.asarray(best.get("frame_samples", []), dtype=np.complex64).copy()
            self._latest_rx_pilot_samples = np.asarray(best.get("pilot_samples", []), dtype=np.complex64).copy()
            self._latest_rx_data_samples = np.asarray(best.get("data_samples", []), dtype=np.complex64).copy()
            self._latest_constellation = post_points.astype(np.complex64)
            self._latest_constellation_post_eq = self._latest_constellation.copy()
            self._latest_constellation_post_eq_raw = np.asarray(best.get("rx_syms_post_raw", np.zeros(0, dtype=np.complex64)), dtype=np.complex64).copy()
            self._latest_constellation_pre_eq = pre_points.astype(np.complex64)
            self._latest_constellation_tf = tf_points.astype(np.complex64)
            self._last_constellation_is_good = bool(good_quality)
            self._last_constellation_source = const_source
            self.last_constellation_source = const_source
            self.last_constellation_points = int(self._latest_constellation.size)
            self.last_constellation_quality = "good" if good_quality else "bad_candidate"
            if good_quality:
                self._last_good_constellation = self._latest_constellation.copy()
            self._last_raw_bytes = best["raw_bytes"]
            self._ber_estimate = float(best["ber"])
            self._ber_hist_t.append(t_now)
            self._ber_hist_v.append(max(float(best["ber"]), 1e-6))
            if best["rx_payload"]:
                self._rx_text = best["rx_text"]
                self._last_good_rx_payload = best["rx_payload"]
                self._decode_ok = bool(best["decode_ok"])
                self._match_bytes = int(best["match_bytes"])
            self._status = "running"
            expected_payload = max(1, len(self._tx_payload))
            self._record_alpha_beta_performance_sample_locked({
                "evm_instant_percent": float(best.get("evm_inst", float("nan"))),
                "evm_average_percent": float(self.last_evm_average_percent),
                "evm_average_count": int(len(self._evm_history)),
                "ber": float(best.get("ber", float("nan"))),
                "fec_bit_ber": float(getattr(self, "_last_fec_bit_ber", float("nan"))),
                "raw_bit_ber": float(getattr(self, "_last_raw_bit_ber", float("nan"))),
                "sync_metric": float(best.get("sync_metric", float("nan"))),
                "cond_h_cross": float(best.get("cond_h", float("nan"))),
                "noise_var": float(best.get("noise_var", float("nan"))),
                "htf_leakage": float(best.get("htf_leakage", float("nan"))),
                "tdl_param_fit_nmse": float(getattr(self, "_last_tdl_param_fit_nmse", float("nan"))),
                "cfo_abs_hz": abs(float(best.get("cfo_hz", 0.0))) if np.isfinite(float(best.get("cfo_hz", 0.0))) else float("nan"),
                "decode_ok": bool(best.get("decode_ok", False)),
                "match_ratio": float(best.get("match_bytes", 0)) / float(expected_payload),
            })

        adaptive_channel_valid = not (
            self.use_full_htf and best.get("htf_cache_refreshing") and not good_quality
        )
        if adaptive_channel_valid:
            self._maybe_queue_alpha_beta_adaptation(
                h_tf_est=best.get("adaptive_htf"),
                htf_kind=str(best.get("adaptive_htf_kind", "full")),
                htf_source=str(best.get("adaptive_htf_source", self.channel_estimator)),
                noise_var=float(best.get("noise_var", float("nan"))),
                sync_metric=float(best.get("sync_metric", 0.0)),
                good_quality=bool(good_quality),
            )

        self._debug(
            "INFO",
            f"v33 frame: mode={self.channel_estimator}, full_cached={self._cached_htf_full is not None}, sync={best['sync_metric']:.3f}, CFO={best['cfo_hz']:.1f} Hz({best.get('cfo_source','preamble')}), rawCFO={best.get('cfo_hz_preamble', best['cfo_hz']):.1f} Hz, "
            f"alias={best.get('cfo_alias_hz', float('nan')):.1f}Hz, scanScore={best.get('cfo_scan_score', float('nan')):.3f}, "
            f"Hleak={best['htf_leakage']:.3f}, cond={best['cond_h']:.2e}, "
            f"BER={best['ber']:.3e}, rawBER={float(getattr(self, '_last_raw_bit_ber', float('nan'))):.3e}, FECBER={float(getattr(self, '_last_fec_bit_ber', float('nan'))):.3e}, EVM={best['evm_inst']:.2f}%, "
            f"noise_var={best['noise_var']:.2e}, "
            f"TDLfit={float(getattr(self, '_last_tdl_param_fit_nmse', float('nan'))):.2e}/"
            f"{int(getattr(self, 'last_tdl_param_path_count', 0))}p, "
            f"decode_ok={best['decode_ok']}, match={best['match_bytes']}/{len(self._tx_payload)}"
        )


    # =========================================================
    # Paper-guided channel-adaptive alpha/beta optimization
    # =========================================================
    def _alpha_beta_adaptation_context_key(self) -> Tuple[Any, ...]:
        """Context that changes the paper SER objective, excluding alpha/beta."""
        return (
            int(getattr(self, "M", 0)), int(getattr(self, "N", 0)),
            int(getattr(self, "cp_len", 0)), str(getattr(self, "mod_order", "")),
            str(getattr(self, "equalizer", "")), str(getattr(self, "channel_estimator", "")),
            str(getattr(self, "channel_mode", "")), round(float(getattr(self, "sample_rate", 0.0)), 3),
            round(float(getattr(self, "carrier_freq", 0.0)), 3),
            round(float(getattr(self, "tx_gain", 0.0)), 3),
            round(float(getattr(self, "rx_gain", 0.0)), 3),
            round(float(getattr(self, "training_amplitude", 0.0)), 6),
            round(float(getattr(self, "tdl_rms_delay_spread_ns", 0.0)), 6),
            round(float(getattr(self, "tdl_doppler_hz", 0.0)), 6),
            round(float(getattr(self, "tdl_doppler_spread_hz", 0.0)), 6),
            round(float(getattr(self, "tdl_snr_db", 0.0)), 6),
        )

    def _invalidate_alpha_beta_adaptation(self, reason: str = "context_change", cooldown: bool = False):
        lock = getattr(self, "_adaptive_ab_lock", None)
        if lock is None:
            return
        with lock:
            self._adaptive_ab_snapshot_seq = int(getattr(self, "_adaptive_ab_snapshot_seq", 0)) + 1
            self._adaptive_ab_snapshot = None
            self._adaptive_ab_recommendation = {}
            self._adaptive_ab_stable_key = None
            self._adaptive_ab_stable_count = 0
            self._adaptive_ab_last_htf_identity = None
            self._adaptive_ab_last_error = ""
            if cooldown:
                self._adaptive_ab_last_applied_frame = int(getattr(self, "_frames_processed", 0))
                self._adaptive_ab_state = "cooldown"
            else:
                self._adaptive_ab_state = "waiting_channel" if self.adaptive_alpha_beta_enable else "disabled"
        try:
            self._debug("DEBUG", f"alpha/beta adaptation invalidated: {reason}")
        except Exception:
            pass

    def _note_alpha_beta_changed_for_adaptation(self, reason: str = "alpha_beta_changed"):
        """Cancel stale recommendations after either manual or adaptive switching."""
        self._invalidate_alpha_beta_adaptation(reason=reason, cooldown=True)

    def _ensure_alpha_beta_adaptation_worker(self):
        if not bool(getattr(self, "adaptive_alpha_beta_enable", False)):
            return
        with self._adaptive_ab_lock:
            thread = getattr(self, "_adaptive_ab_thread", None)
            if thread is not None and thread.is_alive():
                return
            self._adaptive_ab_stop.clear()
            thread = threading.Thread(target=self._alpha_beta_adaptation_worker,
                                      name=f"fdidm-ab-opt-{id(self):x}", daemon=True)
            self._adaptive_ab_thread = thread
            thread.start()

    @staticmethod
    def _adaptive_qam_order(mod_order: str) -> int:
        return {"QPSK": 4, "16QAM": 16, "64QAM": 64}.get(str(mod_order).upper(), 4)

    @staticmethod
    def _adaptive_qfunc(values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        flat = x.reshape(-1)
        out = np.fromiter((0.5 * math.erfc(float(v) / math.sqrt(2.0)) for v in flat),
                          dtype=np.float64, count=flat.size)
        return out.reshape(x.shape)

    def _adaptive_ser_from_symbol_nsr(self, symbol_nsr: np.ndarray, mod_order: str) -> float:
        """Paper Eq. (40), averaged as in Eq. (44)/(47)."""
        xi = np.asarray(symbol_nsr, dtype=np.float64).reshape(-1)
        xi = np.where(np.isfinite(xi), np.maximum(xi, 1e-15), np.inf)
        Q = float(self._adaptive_qam_order(mod_order))
        kappa = 1.0 - 1.0 / math.sqrt(Q)
        argument = math.sqrt(3.0 / max(Q - 1.0, 1.0)) / xi
        q = self._adaptive_qfunc(argument)
        ser = 4.0 * kappa * q - 4.0 * (kappa ** 2) * (q ** 2)
        ser = np.clip(ser, 0.0, 1.0)
        return float(np.mean(ser)) if ser.size else float("inf")

    @staticmethod
    def _adaptive_grid_values(step: float) -> np.ndarray:
        step = float(max(0.01, min(float(step), 2.0)))
        count = int(math.floor(2.0 / step + 1e-9))
        vals = [min(2.0, k * step) for k in range(count + 1)]
        vals.extend([0.0, 1.0, 2.0])
        return np.asarray(sorted(set(round(float(v), 9) for v in vals if -1e-9 <= v <= 2.0 + 1e-9)),
                          dtype=np.float64)

    @staticmethod
    def _adaptive_canonical_index(value: float) -> float:
        v = ((float(value) + 2.0) % 4.0) - 2.0
        if v <= -2.0 + 1e-12:
            v = 2.0
        return float(abs(v))

    def _adaptive_prepare_base(self, snapshot: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Prepare the exact paper SER objective in a transform-efficient form.

        For H(alpha,beta)=Phi H_TF Phi^H, right multiplication by unitary Phi^H
        does not change row norms.  Eq. (41) and Eq. (46) therefore require only
        the left action of Phi.  A diagonal H_TF admits an even cheaper exact
        form: row powers are |Phi|^2 times a per-TF-cell NSR vector.
        """
        M = int(snapshot["M"]); N = int(snapshot["N"]); K = M * N
        htf_kind = str(snapshot.get("htf_kind", "full"))
        raw = np.asarray(snapshot["htf"], dtype=np.complex128)
        is_diag = htf_kind == "diag" or raw.shape == (M, N) or raw.ndim == 1

        noise_var = float(snapshot.get("noise_var", float("nan")))
        if is_diag:
            diag = raw.reshape(-1, order="F")[:K]
            if diag.size != K:
                raise ValueError(f"adaptive diagonal H_TF size mismatch: {diag.size} != {K}")
            if not np.all(np.isfinite(diag.real)) or not np.all(np.isfinite(diag.imag)):
                raise ValueError("adaptive diagonal H_TF contains NaN/Inf")
            signal_power = float(np.mean(np.abs(diag) ** 2))
            if not np.isfinite(noise_var) or noise_var <= 0.0:
                noise_var = max(signal_power * 1e-3, 1e-9)
            noise_var = max(noise_var, 1e-12)
            equalizer = str(snapshot.get("equalizer", "MMSE")).upper()
            if equalizer == "ZF":
                h_abs = np.abs(diag)
                nz = h_abs[h_abs > 0.0]
                med = float(np.median(nz)) if nz.size else 0.0
                floor = max(med * 1e-3, 1e-10)
                safe_abs = np.maximum(h_abs, floor)
                nsr_power = noise_var / np.maximum(safe_abs ** 2, 1e-20)
            else:
                W = np.conj(diag) / (np.abs(diag) ** 2 + noise_var)
                error = W * diag - 1.0
                nsr_power = np.abs(error) ** 2 + noise_var * np.abs(W) ** 2
            predicted_snr_db = 10.0 * math.log10(max(signal_power / noise_var, 1e-15))
            return {
                "kind": "diag_weights",
                "weights": np.asarray(nsr_power, dtype=np.float64),
            }, float(predicted_snr_db)

        Htf = raw.reshape((K, K)).astype(np.complex128, copy=False)
        if not np.all(np.isfinite(Htf.real)) or not np.all(np.isfinite(Htf.imag)):
            raise ValueError("adaptive H_TF contains NaN/Inf")
        signal_power = float(np.linalg.norm(Htf, "fro") ** 2 / max(K, 1))
        if not np.isfinite(noise_var) or noise_var <= 0.0:
            noise_var = max(signal_power * 1e-3, 1e-9)
        noise_var = max(noise_var, 1e-12)
        xi = math.sqrt(noise_var)  # normalized QAM symbols have Es=1
        equalizer = str(snapshot.get("equalizer", "MMSE")).upper()
        rcond = float(snapshot.get("rcond", 1e-6))
        I = np.eye(K, dtype=np.complex128)

        if equalizer == "ZF":
            try:
                inv_htf = np.linalg.solve(Htf, I)
            except np.linalg.LinAlgError:
                inv_htf = np.linalg.pinv(Htf, rcond=rcond)
            base = xi * inv_htf
        else:
            Hh = Htf.conj().T
            gram = Hh @ Htf
            try:
                Wtf = np.linalg.solve(gram + noise_var * I, Hh)
            except np.linalg.LinAlgError:
                load = max(noise_var, rcond * float(np.mean(np.abs(gram)) + 1e-12))
                Wtf = np.linalg.solve(gram + load * I, Hh)
            error_tf = Wtf @ Htf - I
            # Eq. (46): ||row(WH-I)||^2 + xi^2 ||row(W)||^2.
            base = np.concatenate((error_tf, xi * Wtf), axis=1)

        predicted_snr_db = 10.0 * math.log10(max(signal_power / noise_var, 1e-15))
        return {
            "kind": "matrix",
            "matrix": np.asarray(base, dtype=np.complex128),
        }, float(predicted_snr_db)

    def _adaptive_evaluate_candidates(self, prepared: Dict[str, Any], candidates: List[Tuple[float, float]],
                                      M: int, N: int, mod_order: str) -> List[Dict[str, float]]:
        """Evaluate candidate indices with exact paper ZF/MMSE row NSRs."""
        groups: Dict[float, List[float]] = {}
        for alpha, beta in candidates:
            a = round(float(alpha), 9); bb = round(float(beta), 9)
            groups.setdefault(a, []).append(bb)
        results: List[Dict[str, float]] = []

        if str(prepared.get("kind")) == "diag_weights":
            weights = np.asarray(prepared["weights"], dtype=np.float64).reshape((M, N), order="F")
            # For diagonal H_TF, diag(Phi C Phi^H) equals
            # |Gamma_M(-alpha)|^2 W |Gamma_N(beta)|^{2T} exactly.
            for alpha in sorted(groups):
                GM2 = np.abs(self._gamma(M, -float(alpha))) ** 2
                left = GM2 @ weights
                for beta in sorted(set(groups[alpha])):
                    GN2 = np.abs(self._gamma(N, float(beta))) ** 2
                    power = left @ GN2.T
                    symbol_nsr = np.sqrt(np.maximum(power.reshape(-1, order="F"), 0.0))
                    ser = self._adaptive_ser_from_symbol_nsr(symbol_nsr, mod_order)
                    results.append({"alpha": float(alpha), "beta": float(beta), "ser": float(ser)})
            return results

        K = int(M * N)
        base = np.asarray(prepared["matrix"], dtype=np.complex128)
        if base.shape[0] != K:
            raise ValueError(f"adaptive base rows mismatch: {base.shape[0]} != {K}")
        cube = base.reshape((M, N, base.shape[1]), order="F")
        for alpha in sorted(groups):
            after_alpha = self._apply_gamma_axis(cube, -float(alpha), axis=0)
            for beta in sorted(set(groups[alpha])):
                transformed = self._apply_gamma_axis(after_alpha, float(beta), axis=1)
                power = np.sum(transformed.real * transformed.real + transformed.imag * transformed.imag, axis=2)
                symbol_nsr = np.sqrt(np.maximum(power.reshape(-1, order="F"), 0.0))
                ser = self._adaptive_ser_from_symbol_nsr(symbol_nsr, mod_order)
                results.append({"alpha": float(alpha), "beta": float(beta), "ser": float(ser)})
        return results

    def _optimize_alpha_beta_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        M = int(snapshot["M"]); N = int(snapshot["N"]); K = M * N
        if K > int(snapshot.get("max_order", 512)):
            raise ValueError(f"adaptive search skipped: M*N={K} exceeds max_order={snapshot.get('max_order')}")
        prepared, predicted_snr_db = self._adaptive_prepare_base(snapshot)
        diagonal_fast_path = str(prepared.get("kind")) == "diag_weights"
        coarse_step = float(snapshot.get("coarse_step", 0.25))
        fine_step = float(snapshot.get("fine_step", 0.05))
        # Full H_TF searches are substantially heavier than diagonal-TF searches.
        # A 0.5 coarse grid plus coordinate refinement keeps the optimizer from
        # starving the hardware host while still finishing at the requested fine step.
        effective_coarse = coarse_step if diagonal_fast_path else max(coarse_step, 0.50)
        coarse_vals = self._adaptive_grid_values(effective_coarse)
        coarse_candidates = [(float(a), float(bb)) for a in coarse_vals for bb in coarse_vals]
        coarse_results = self._adaptive_evaluate_candidates(prepared, coarse_candidates, M, N, snapshot["mod_order"])
        coarse_results.sort(key=lambda r: (r["ser"], r["alpha"], r["beta"]))

        refine_candidates = set()
        if diagonal_fast_path:
            # Cheap exact two-dimensional refinement around three basins.
            radius = max(effective_coarse, 2.0 * fine_step)
            for seed in coarse_results[:3]:
                a0 = float(seed["alpha"]); b0 = float(seed["beta"])
                av = np.arange(max(0.0, a0 - radius), min(2.0, a0 + radius) + 0.5 * fine_step, fine_step)
                bv = np.arange(max(0.0, b0 - radius), min(2.0, b0 + radius) + 0.5 * fine_step, fine_step)
                for a in av:
                    for bb in bv:
                        refine_candidates.add((round(float(np.clip(a, 0.0, 2.0)), 9),
                                               round(float(np.clip(bb, 0.0, 2.0)), 9)))
        else:
            # Full-matrix coordinate refinement around the two best coarse basins.
            radius = max(0.5 * effective_coarse, 2.0 * fine_step)
            for seed in coarse_results[:2]:
                a0 = float(seed["alpha"]); b0 = float(seed["beta"])
                av = np.arange(max(0.0, a0 - radius), min(2.0, a0 + radius) + 0.5 * fine_step, fine_step)
                bv = np.arange(max(0.0, b0 - radius), min(2.0, b0 + radius) + 0.5 * fine_step, fine_step)
                for a in av:
                    refine_candidates.add((round(float(np.clip(a, 0.0, 2.0)), 9), round(b0, 9)))
                for bb in bv:
                    refine_candidates.add((round(a0, 9), round(float(np.clip(bb, 0.0, 2.0)), 9)))

        current_alpha = float(snapshot.get("alpha", 0.0))
        current_beta = float(snapshot.get("beta", 0.0))
        current_canonical = (round(self._adaptive_canonical_index(current_alpha), 9),
                             round(self._adaptive_canonical_index(current_beta), 9))
        refine_candidates.update({current_canonical, (0.0, 0.0), (1.0, 1.0), (0.5, 1.0)})
        fine_results = self._adaptive_evaluate_candidates(prepared, sorted(refine_candidates), M, N, snapshot["mod_order"])

        by_key: Dict[Tuple[float, float], Dict[str, float]] = {}
        for item in coarse_results + fine_results:
            key = (round(item["alpha"], 9), round(item["beta"], 9))
            old = by_key.get(key)
            if old is None or item["ser"] < old["ser"]:
                by_key[key] = item
        all_results = list(by_key.values())
        all_results.sort(key=lambda r: (r["ser"], r["alpha"], r["beta"]))
        raw_best = all_results[0]

        current_eval = self._adaptive_evaluate_candidates(
            prepared, [(current_alpha, current_beta)], M, N, snapshot["mod_order"]
        )[0]
        current_ser = float(current_eval["ser"])

        # Prefer the current point on a numerical tie; this prevents needless
        # switching on a flat/identity channel.
        tie_limit = float(raw_best["ser"]) * (1.0 + 1e-10) + 1e-15
        tied = [r for r in all_results if r["ser"] <= tie_limit]
        best = min(tied, key=lambda r: ((r["alpha"] - current_canonical[0]) ** 2 +
                                        (r["beta"] - current_canonical[1]) ** 2,
                                        r["alpha"], r["beta"]))

        # Paper-aligned complexity policy: if an integer-index waveform is within
        # a small SER margin of the fractional optimum, use the integer point.
        margin_db = float(snapshot.get("integer_margin_db", 0.0))
        integer_limit = float(best["ser"]) * (10.0 ** (margin_db / 10.0)) + 1e-15
        integer_results = [r for r in all_results
                           if abs(r["alpha"] - round(r["alpha"])) < 1e-9
                           and abs(r["beta"] - round(r["beta"])) < 1e-9
                           and r["ser"] <= integer_limit
                           and r["ser"] < current_ser * (1.0 - 1e-10)]
        if integer_results:
            best = min(integer_results, key=lambda r: (r["ser"],
                                                       (r["alpha"] - current_canonical[0]) ** 2 +
                                                       (r["beta"] - current_canonical[1]) ** 2))

        best_ser = float(best["ser"])
        improvement_db = 10.0 * math.log10(max(current_ser, 1e-15) / max(best_ser, 1e-15))
        ofdm = by_key.get((0.0, 0.0), {"ser": float("nan")})
        otfs = by_key.get((1.0, 1.0), {"ser": float("nan")})
        return {
            "recommended_alpha": float(best["alpha"]),
            "recommended_beta": float(best["beta"]),
            "predicted_ser_current": current_ser,
            "predicted_ser_best": best_ser,
            "predicted_ser_ofdm": float(ofdm.get("ser", float("nan"))),
            "predicted_ser_otfs": float(otfs.get("ser", float("nan"))),
            "predicted_improvement_db": float(improvement_db),
            "predicted_snr_db": float(predicted_snr_db),
            "candidate_count": int(len(all_results) + 1),
            "search_seconds": float(time.time() - t0),
            "search_mode": "diag_2d" if diagonal_fast_path else "full_coordinate",
            "htf_source": str(snapshot.get("htf_source", "unknown")),
            "htf_kind": str(snapshot.get("htf_kind", "unknown")),
            "equalizer": str(snapshot.get("equalizer", "")),
            "mod_order": str(snapshot.get("mod_order", "")),
            "frame_counter": int(snapshot.get("frame_counter", 0)),
            "snapshot_seq": int(snapshot.get("snapshot_seq", 0)),
        }

    def _alpha_beta_adaptation_worker(self):
        while not self._adaptive_ab_stop.is_set():
            if not self._adaptive_ab_event.wait(timeout=0.5):
                continue
            self._adaptive_ab_event.clear()
            with self._adaptive_ab_lock:
                snapshot = self._adaptive_ab_snapshot
                if snapshot is None or not self.adaptive_alpha_beta_enable:
                    continue
                snapshot = dict(snapshot)
                expected_seq = int(snapshot.get("snapshot_seq", -1))
                self._adaptive_ab_state = "optimizing"
                self._adaptive_ab_last_error = ""
            try:
                result = self._optimize_alpha_beta_snapshot(snapshot)
            except Exception as exc:
                with self._adaptive_ab_lock:
                    if expected_seq == int(self._adaptive_ab_snapshot_seq):
                        self._adaptive_ab_last_error = f"{type(exc).__name__}: {exc}"
                        self._adaptive_ab_state = "error"
                self._debug("WARN", f"alpha/beta optimizer failed: {type(exc).__name__}: {exc}")
                continue

            with self._adaptive_ab_lock:
                if expected_seq != int(self._adaptive_ab_snapshot_seq):
                    # Alpha/beta or channel context changed during the search.
                    continue
                fine = max(float(self.adaptive_alpha_beta_fine_step), 1e-9)
                key = (int(round(float(result["recommended_alpha"]) / fine)),
                       int(round(float(result["recommended_beta"]) / fine)))
                if key == self._adaptive_ab_stable_key:
                    self._adaptive_ab_stable_count += 1
                else:
                    self._adaptive_ab_stable_key = key
                    self._adaptive_ab_stable_count = 1
                self._adaptive_ab_eval_seq += 1
                result["recommendation_seq"] = int(self._adaptive_ab_eval_seq)
                result["stable_count"] = int(self._adaptive_ab_stable_count)
                result["stable_required"] = int(self.adaptive_alpha_beta_stability_evals)
                result["minimum_improvement_db"] = float(self.adaptive_alpha_beta_min_improvement_db)
                delta = abs(float(result["recommended_alpha"]) - float(snapshot.get("alpha", 0.0))) + \
                        abs(float(result["recommended_beta"]) - float(snapshot.get("beta", 0.0)))
                required_gain = max(float(self.adaptive_alpha_beta_min_improvement_db), 1e-6)
                ready = (float(result["predicted_improvement_db"]) >= required_gain
                         and int(self._adaptive_ab_stable_count) >= int(self.adaptive_alpha_beta_stability_evals)
                         and delta >= 0.5 * fine)
                result["ready"] = bool(ready)
                result["pending"] = bool(ready)
                self._adaptive_ab_recommendation = result
                self._adaptive_ab_state = "ready" if ready else "tracking"
            level = "INFO" if result.get("ready") else "DEBUG"
            self._debug(level,
                        "alpha/beta SER search: "
                        f"current={result['predicted_ser_current']:.3e}, best={result['predicted_ser_best']:.3e}, "
                        f"gain={result['predicted_improvement_db']:.2f}dB, "
                        f"rec=({result['recommended_alpha']:.2f},{result['recommended_beta']:.2f}), "
                        f"stable={result['stable_count']}/{result['stable_required']}, "
                        f"source={result['htf_source']}, time={result['search_seconds']:.3f}s")

    def _maybe_queue_alpha_beta_adaptation(self, h_tf_est: Any, htf_kind: str,
                                           htf_source: str, noise_var: float,
                                           sync_metric: float, good_quality: bool):
        if not bool(getattr(self, "adaptive_alpha_beta_enable", False)):
            return
        self._ensure_alpha_beta_adaptation_worker()
        K = int(self.M * self.N)
        if K > int(self.adaptive_alpha_beta_max_order):
            with self._adaptive_ab_lock:
                self._adaptive_ab_state = "order_limited"
                self._adaptive_ab_last_error = f"M*N={K} > adaptive max_order={self.adaptive_alpha_beta_max_order}"
            return
        if h_tf_est is None or not np.isfinite(float(sync_metric)):
            return
        if float(sync_metric) < float(self.adaptive_alpha_beta_min_sync_metric):
            return
        if self.adaptive_alpha_beta_require_good_frame and not bool(good_quality):
            return

        frame_counter = int(getattr(self, "_frames_processed", 0))
        htf_identity = (str(htf_source), id(h_tf_est))
        with self._adaptive_ab_lock:
            force = bool(self._adaptive_ab_force_next)
            if (not force and str(htf_source) == "full_htf" and bool(getattr(self, "full_htf_once", False))
                    and htf_identity == self._adaptive_ab_last_htf_identity):
                return
            if not force:
                if frame_counter - int(self._adaptive_ab_last_applied_frame) < int(self.adaptive_alpha_beta_cooldown_frames):
                    self._adaptive_ab_state = "cooldown"
                    return
                if frame_counter - int(self._adaptive_ab_last_queued_frame) < int(self.adaptive_alpha_beta_interval_frames):
                    return
            self._adaptive_ab_force_next = False

        raw = np.asarray(h_tf_est, dtype=np.complex128)
        if str(htf_kind) == "diag" or raw.shape == (self.M, self.N):
            htf_payload = raw.reshape(-1, order="F").copy()
            htf_kind = "diag"
        else:
            if raw.size != K * K:
                return
            htf_payload = raw.reshape((K, K)).copy()
            htf_kind = "full"
        if not np.all(np.isfinite(htf_payload.real)) or not np.all(np.isfinite(htf_payload.imag)):
            return

        with self._adaptive_ab_lock:
            self._adaptive_ab_snapshot_seq += 1
            snapshot = {
                "snapshot_seq": int(self._adaptive_ab_snapshot_seq),
                "frame_counter": frame_counter,
                "M": int(self.M), "N": int(self.N),
                "alpha": float(self.alpha), "beta": float(self.beta),
                "mod_order": str(self.mod_order), "equalizer": str(self.equalizer),
                "noise_var": float(noise_var), "sync_metric": float(sync_metric),
                "htf": htf_payload, "htf_kind": str(htf_kind), "htf_source": str(htf_source),
                "coarse_step": float(self.adaptive_alpha_beta_coarse_step),
                "fine_step": float(self.adaptive_alpha_beta_fine_step),
                "integer_margin_db": float(self.adaptive_alpha_beta_integer_margin_db),
                "max_order": int(self.adaptive_alpha_beta_max_order),
                "rcond": float(self.adaptive_alpha_beta_rcond),
            }
            self._adaptive_ab_snapshot = snapshot
            self._adaptive_ab_last_snapshot = snapshot
            self._adaptive_ab_last_queued_frame = frame_counter
            self._adaptive_ab_last_htf_identity = htf_identity
            self._adaptive_ab_state = "queued"
            self._adaptive_ab_last_error = ""
            self._adaptive_ab_event.set()

    def request_alpha_beta_adaptation(self) -> bool:
        """Request an immediate search using the latest H_TF, or the next valid frame."""
        if not bool(getattr(self, "adaptive_alpha_beta_enable", False)):
            return False
        self._ensure_alpha_beta_adaptation_worker()
        with self._adaptive_ab_lock:
            last = self._adaptive_ab_last_snapshot
            if last is None:
                self._adaptive_ab_force_next = True
                self._adaptive_ab_state = "waiting_channel"
                return False
            self._adaptive_ab_snapshot_seq += 1
            snap = dict(last)
            snap["snapshot_seq"] = int(self._adaptive_ab_snapshot_seq)
            snap["frame_counter"] = int(getattr(self, "_frames_processed", 0))
            snap["alpha"] = float(self.alpha); snap["beta"] = float(self.beta)
            snap["mod_order"] = str(self.mod_order); snap["equalizer"] = str(self.equalizer)
            snap["coarse_step"] = float(self.adaptive_alpha_beta_coarse_step)
            snap["fine_step"] = float(self.adaptive_alpha_beta_fine_step)
            snap["integer_margin_db"] = float(self.adaptive_alpha_beta_integer_margin_db)
            snap["max_order"] = int(self.adaptive_alpha_beta_max_order)
            snap["rcond"] = float(self.adaptive_alpha_beta_rcond)
            self._adaptive_ab_snapshot = snap
            self._adaptive_ab_last_snapshot = snap
            self._adaptive_ab_state = "queued"
            self._adaptive_ab_event.set()
            return True

    def get_alpha_beta_adaptation_status(self) -> Dict[str, Any]:
        lock = getattr(self, "_adaptive_ab_lock", None)
        if lock is None:
            return {"enabled": False, "state": "uninitialized", "ready": False, "pending": False}
        with lock:
            rec = dict(getattr(self, "_adaptive_ab_recommendation", {}) or {})
            return {
                "enabled": bool(getattr(self, "adaptive_alpha_beta_enable", False)),
                "state": str(getattr(self, "_adaptive_ab_state", "disabled")),
                "last_error": str(getattr(self, "_adaptive_ab_last_error", "")),
                "ready": bool(rec.get("ready", False)),
                "pending": bool(rec.get("pending", False)),
                "recommendation_seq": int(rec.get("recommendation_seq", 0)),
                "recommended_alpha": float(rec.get("recommended_alpha", float("nan"))),
                "recommended_beta": float(rec.get("recommended_beta", float("nan"))),
                "predicted_ser_current": float(rec.get("predicted_ser_current", float("nan"))),
                "predicted_ser_best": float(rec.get("predicted_ser_best", float("nan"))),
                "predicted_ser_ofdm": float(rec.get("predicted_ser_ofdm", float("nan"))),
                "predicted_ser_otfs": float(rec.get("predicted_ser_otfs", float("nan"))),
                "predicted_improvement_db": float(rec.get("predicted_improvement_db", float("nan"))),
                "predicted_snr_db": float(rec.get("predicted_snr_db", float("nan"))),
                "stable_count": int(rec.get("stable_count", 0)),
                "stable_required": int(rec.get("stable_required", self.adaptive_alpha_beta_stability_evals)),
                "candidate_count": int(rec.get("candidate_count", 0)),
                "search_seconds": float(rec.get("search_seconds", float("nan"))),
                "htf_source": str(rec.get("htf_source", "")),
                "htf_kind": str(rec.get("htf_kind", "")),
                "coarse_step": float(self.adaptive_alpha_beta_coarse_step),
                "fine_step": float(self.adaptive_alpha_beta_fine_step),
                "interval_frames": int(self.adaptive_alpha_beta_interval_frames),
                "minimum_improvement_db": float(self.adaptive_alpha_beta_min_improvement_db),
                "cooldown_frames": int(self.adaptive_alpha_beta_cooldown_frames),
                "integer_margin_db": float(self.adaptive_alpha_beta_integer_margin_db),
                "max_order": int(self.adaptive_alpha_beta_max_order),
                "signaling_mode": str(getattr(self, "ALPHA_BETA_SIGNALING_MODE", "shared_memory")),
            }

    # =========================================================
    # Display helpers
    # =========================================================
    def _prepare_constellation_points(self, arr: np.ndarray, display_mode: str = "raw") -> np.ndarray:
        arr = np.asarray(arr, dtype=np.complex64).reshape(-1)
        if arr.size == 0:
            return arr
        mag = np.abs(arr)
        med = float(np.median(mag))
        if med > 1e-6:
            keep = mag < 6.0 * med
            if np.any(keep):
                arr = arr[keep]
        return arr

    def _normalize_constellation_cloud_for_display(self, arr: np.ndarray) -> np.ndarray:
        """Scale diagnostic clouds to the visible constellation axes.

        Pre-equalized and raw TF clouds can have arbitrary gain; this display
        scaling does not affect BER/EVM/equalization, it only makes the cloud
        shape visible on the fixed [-2,2] plot.
        """
        arr = np.asarray(arr, dtype=np.complex64).reshape(-1)
        if arr.size == 0:
            return arr
        mag = np.abs(arr)
        finite = np.isfinite(mag)
        if not np.any(finite):
            return arr
        med = float(np.median(mag[finite]))
        if med <= 1e-8:
            return arr
        try:
            refs = self._ideal_constellation_points()
            target = float(np.median(np.abs(refs))) if refs.size else 1.0
        except Exception:
            target = 1.0
        return (arr * (target / med)).astype(np.complex64)

    def _apply_display_mode(self, arr: np.ndarray, mode: str) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.complex64).reshape(-1)
        if arr.size == 0:
            return arr
        mode = str(mode).lower()
        if mode in ("normalize", "normalized"):
            med = float(np.median(np.abs(arr)))
            if med > 1e-8 and np.isfinite(med):
                return (arr * ((1.0 / np.sqrt(2.0)) / med)).astype(np.complex64)
            return arr
        # Source-selection modes are handled in get_rx_constellation(); no extra shaping here.
        if mode in ("raw", "post_equalized", "last_good") or self.mod_order != "QPSK":
            return arr
        if mode not in ("dd_refined", "hard_decision"):
            return arr
        target_radius = 1.0 / np.sqrt(2.0)
        avg_mag = float(np.median(np.abs(arr)))
        if avg_mag > 1e-6:
            arr = arr * (target_radius / avg_mag)
        qpsk_points = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex64) * target_radius
        nearest = np.argmin(np.abs(arr[:, None] - qpsk_points[None, :]), axis=1)
        decisions = qpsk_points[nearest]
        if mode == "hard_decision":
            return decisions.copy()
        return (decisions + 0.25 * (arr - decisions)).astype(np.complex64)

    def set_constellation_display_mode(self, mode: str):
        mode = str(mode).lower()
        if mode not in ("raw", "post_equalized", "post_equalized_raw", "dd_refined", "hard_decision",
                        "pre_equalized", "tf_received", "last_good", "raw_iq"):
            raise ValueError("unsupported constellation display mode")
        with self._lock:
            self.constellation_display_mode = mode

    def set_tx_gain(self, value: float):
        changed = float(value) != float(getattr(self, "tx_gain", value))
        self.tx_gain = float(value)
        if changed:
            self._invalidate_alpha_beta_adaptation(reason="tx_gain_changed", cooldown=False)
        if getattr(self, "_usrp_sink", None) is not None:
            try:
                self._usrp_sink.set_gain(self.tx_gain, 0)
            except Exception:
                pass

    def set_rx_gain(self, value: float):
        changed = float(value) != float(getattr(self, "rx_gain", value))
        self.rx_gain = float(value)
        if changed:
            self._invalidate_alpha_beta_adaptation(reason="rx_gain_changed", cooldown=False)
        if getattr(self, "_usrp_source", None) is not None:
            try:
                self._usrp_source.set_gain(self.rx_gain, 0)
            except Exception:
                pass

    def set_mod_order(self, mod_order: str):
        was_running = bool(self._running)
        if was_running:
            self.stop()
        self.configure(mod_order=mod_order)
        if was_running:
            self.start()

    def set_alpha_beta(self, alpha: Optional[float] = None, beta: Optional[float] = None):
        was_running = bool(self._running)
        if was_running:
            self.stop()
        self.configure(alpha=self.alpha if alpha is None else alpha,
                       beta=self.beta if beta is None else beta)
        if was_running:
            self.start()

    def _create_python_logger(self) -> logging.Logger:
        """Create optional standard-logging handlers for backend diagnostics."""
        logger = logging.getLogger(f"fdidm.hardware.{id(self):x}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(name)s: %(message)s")
        if bool(getattr(self, "log_to_stdout", False)):
            sh = logging.StreamHandler()
            sh.setLevel(logging.DEBUG)
            sh.setFormatter(fmt)
            logger.addHandler(sh)
        path = str(getattr(self, "log_file_path", "") or "").strip()
        if path:
            try:
                fh = logging.FileHandler(path, encoding="utf-8")
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(fmt)
                logger.addHandler(fh)
            except Exception:
                # Keep the in-memory/UI log operational even if the file cannot be opened.
                pass
        return logger

    def export_debug_log(self, path: Optional[str] = None, max_entries: int = 2000, min_level: str = "DEBUG") -> str:
        """Write recent backend diagnostics to a text file and return the path."""
        if path is None or str(path).strip() == "":
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = f"fdidm_debug_{ts}.log"
        entries = self.get_debug_log(max_entries=max_entries, min_level=min_level)
        with open(str(path), "w", encoding="utf-8") as f:
            for e in entries:
                f.write(f"{e['seq']:06d} {e['t']:10.3f}s {e['level']:<5} {e['msg']}\n")
        return str(path)

    def _debug(self, level: str, msg: str):

        try:
            text = str(msg)
        except Exception:
            text = repr(msg)
        if len(text) > 1024:
            text = text[:1024] + "...<truncated>"
        # Important: we deliberately do NOT acquire self._lock here so that
        # the worker can log freely while it holds the lock for buffer copies.
        # collections.deque.append is atomic in CPython.
        self._debug_seq += 1
        self._debug_log.append({
            "seq": int(self._debug_seq),
            "t": float(time.time() - self._t0),
            "level": str(level).upper(),
            "msg": text,
        })
        if level == "ERROR" or level == "WARN":
            self._last_error = text
        elif level == "INFO":
            self._last_info = text
        logger = getattr(self, "_py_logger", None)
        if logger is not None and getattr(logger, "handlers", None):
            py_level = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARN": logging.WARNING, "ERROR": logging.ERROR}.get(str(level).upper(), logging.INFO)
            try:
                logger.log(py_level, text)
            except Exception:
                pass

    def get_debug_log(self, max_entries: int = 200, min_level: str = "INFO") -> List[Dict[str, Any]]:
        """Return the last `max_entries` log records at >= `min_level`.

        Use this for a one-shot dump (e.g. "give me the last 200 lines").
        For streaming, prefer drain_debug_log() which is sequence-aware.
        """
        priorities = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}
        min_p = priorities.get(str(min_level).upper(), 1)
        entries = list(self._debug_log)  # snapshot
        out = [e for e in entries if priorities.get(e["level"], 1) >= min_p]
        if max_entries > 0 and len(out) > max_entries:
            out = out[-max_entries:]
        return out

    def drain_debug_log(self, since_seq: int = 0, max_entries: int = 300,
                        min_level: str = "DEBUG") -> List[Dict[str, Any]]:

        priorities = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}
        min_p = priorities.get(str(min_level).upper(), 0)
        entries = list(self._debug_log)
        filtered = [e for e in entries
                    if e["seq"] > int(since_seq) and priorities.get(e["level"], 1) >= min_p]
        if max_entries > 0 and len(filtered) > max_entries:
            filtered = filtered[-max_entries:]
        return filtered

    # =========================================================
    # UI access
    # =========================================================
    def get_tx_text(self) -> str:
        return self._tx_text

    def get_rx_text(self) -> str:
        with self._lock:
            return self._rx_text

    def get_decode_stats(self) -> Dict[str, Any]:
        with self._lock:
            expected = len(self._tx_payload)
            ratio = (self._match_bytes / expected) if expected > 0 else 0.0
            return {
                "decode_ok": bool(self._decode_ok),
                "match_bytes": int(self._match_bytes),
                "expected_bytes": int(expected),
                "match_ratio": float(ratio),
            }

    def _tx_waveform_window(self, num_samples: int, animated: bool = True) -> np.ndarray:
        """Return a contiguous window from the known repeated TX waveform.

        The live TX stream is not mirrored into a Python sink
        because that can starve UHD. Instead, the TX signal is deterministic
        from vector_source_c(repeat=True), so the UI can show a time-advanced
        window based on the sample rate. This fixes the "TX plot looks frozen"
        issue without adding a heavy live TX tap.
        """
        n = max(1, int(num_samples))
        wave = np.asarray(self._tx_waveform, dtype=np.complex64).reshape(-1)
        if wave.size == 0:
            return np.zeros(0, dtype=np.complex64)
        if animated and self._running:
            t0 = float(self._tx_preview_start_t or self._t0)
            start = int(max(0.0, time.time() - t0) * max(self.sample_rate, 1.0)) % int(wave.size)
        else:
            start = 0
        if n <= wave.size:
            end = start + n
            if end <= wave.size:
                return wave[start:end].copy()
            return np.concatenate((wave[start:].copy(), wave[:end - wave.size].copy()))
        reps = int(np.ceil((start + n) / max(wave.size, 1))) + 1
        tiled = np.tile(wave, reps)
        return tiled[start:start + n].copy()

    def get_tx_samples(self, num_samples: int = 2048):
        """Return a time-advanced preview of the repeated TX stream."""
        return self._tx_waveform_window(num_samples, animated=True)

    def get_rx_samples(self, num_samples: int = 2048):
        """Return the most recent live RX samples from the USRP source.

        In v20, RX is normally captured by a C++ probe_signal_vc path.  The
        monitor thread copies the latest vector into _latest_rx_samples for UI
        plotting; fall back to the Python ring only on old GNU Radio builds.
        """
        n = max(1, int(num_samples))
        with self._lock:
            latest = np.asarray(self._latest_rx_samples, dtype=np.complex64).reshape(-1).copy()
        if latest.size > 0:
            return latest[-n:].copy() if latest.size >= n else np.pad(latest, (n - latest.size, 0))
        arr, _, _ = self._rx_buffer.read_latest(n)
        if arr.size == 0:
            return np.zeros(0, dtype=np.complex64)
        return arr[-n:].copy() if arr.size >= n else np.pad(arr, (n - arr.size, 0))

    def get_tx_spectrum_source(self, num_samples: int = 2048):
        return self.get_tx_samples(num_samples)

    def get_rx_spectrum_source(self, num_samples: int = 2048, source: str = "raw"):
        """Return samples for the RX spectrum plot.

        source="raw" is the live stream before synchronization/decoding.
        source="frame"/"pilot"/"data" uses the latest synchronized and CFO-corrected
        segments from the decoder.  A raw spectrum can look perfectly stable
        even while CRC fails, because it does not include timing, channel
        estimation, equalization, or payload parsing.
        """
        n = max(1, int(num_samples))
        src = str(source or "raw").lower()
        with self._lock:
            if src in ("frame", "sync_frame", "cfo_frame"):
                arr = np.asarray(self._latest_rx_frame_samples, dtype=np.complex64).reshape(-1).copy()
            elif src in ("pilot", "pilot_frame"):
                arr = np.asarray(self._latest_rx_pilot_samples, dtype=np.complex64).reshape(-1).copy()
            elif src in ("data", "data_frame"):
                arr = np.asarray(self._latest_rx_data_samples, dtype=np.complex64).reshape(-1).copy()
            else:
                arr = np.asarray(self._latest_rx_samples, dtype=np.complex64).reshape(-1).copy()
        if arr.size > 0:
            return arr[-n:].copy() if arr.size >= n else np.pad(arr, (n - arr.size, 0))
        return self.get_rx_samples(n)

    def get_tx_waveform_preview(self, num_samples: int = 4096):
        n = max(1, int(num_samples))
        wave = np.asarray(self._tx_waveform, dtype=np.complex64).reshape(-1)
        return wave[:n].copy() if wave.size >= n else np.pad(wave, (0, n - wave.size))

    def get_fdidm_preview_constellation(self, domain: str = "tf", max_points: int = 512):
        domain = str(domain).lower()
        arr = self._tx_x_cross if domain in ("cross", "x") else self._tx_x_tf
        pts = np.asarray(arr, dtype=np.complex128).reshape(-1, order="F")
        if pts.size > max_points:
            idx = np.linspace(0, pts.size - 1, max_points, dtype=np.int64)
            pts = pts[idx]
        return pts.astype(np.complex64)

    def get_rx_constellation(self, max_points: int = 256,
                             source: Optional[str] = None,
                             display_mode: Optional[str] = None):
        """Return constellation points for UI diagnostics.

        Default raw/post_equalized view is the latest processed candidate after
        channel equalization and residual scalar correction, regardless of CRC.
        This is intentional: a failed CRC should still let us see whether the
        equalizer output is rotating, noisy, collapsed, or simply bit-mapped
        incorrectly. Diagnostic views expose pre-EQ cross-domain points, raw
        received TF points, last-good post-EQ points, or raw RX IQ samples.
        """
        req = str(source or display_mode or self.constellation_display_mode or "post_equalized").lower()
        mode_to_apply = req
        scale_for_display = False
        with self._lock:
            if req in ("post_equalized_raw", "post_raw", "raw_post_eq"):
                raw = np.asarray(getattr(self, "_latest_constellation_post_eq_raw", np.zeros(0, dtype=np.complex64)), dtype=np.complex64).copy()
                mode_to_apply = "raw"
                view_source = "post_eq_raw_no_scalar"
            elif req in ("pre_equalized", "pre_eq"):
                raw = np.asarray(getattr(self, "_latest_constellation_pre_eq", np.zeros(0, dtype=np.complex64)), dtype=np.complex64).copy()
                mode_to_apply = "raw"
                view_source = "pre_eq_cross"
                scale_for_display = True
            elif req in ("tf_received", "tf", "y_tf"):
                raw = np.asarray(getattr(self, "_latest_constellation_tf", np.zeros(0, dtype=np.complex64)), dtype=np.complex64).copy()
                mode_to_apply = "raw"
                view_source = "raw_y_tf"
                scale_for_display = True
            elif req in ("last_good", "good"):
                raw = np.asarray(getattr(self, "_last_good_constellation", np.zeros(0, dtype=np.complex64)), dtype=np.complex64).copy()
                mode_to_apply = "raw"
                view_source = "last_good_post_eq"
            elif req in ("raw_iq", "iq"):
                raw = np.asarray(getattr(self, "_latest_rx_samples", np.zeros(0, dtype=np.complex64)), dtype=np.complex64).reshape(-1).copy()
                mode_to_apply = "raw"
                view_source = "raw_rx_iq"
                scale_for_display = True
            else:
                raw = np.asarray(getattr(self, "_latest_constellation_post_eq", np.zeros(0, dtype=np.complex64)), dtype=np.complex64).copy()
                if raw.size == 0:
                    raw = np.asarray(getattr(self, "_latest_constellation", np.zeros(0, dtype=np.complex64)), dtype=np.complex64).copy()
                if raw.size == 0 and getattr(self, "_last_good_constellation", np.zeros(0)).size > 0:
                    raw = np.asarray(self._last_good_constellation, dtype=np.complex64).copy()
                    view_source = "last_good_fallback"
                    mode_to_apply = "raw"
                else:
                    view_source = str(getattr(self, "_last_constellation_source", "post_eq"))
            self.last_constellation_source = view_source
        raw = np.asarray(raw, dtype=np.complex64).reshape(-1)
        if raw.size == 0:
            return raw
        if scale_for_display:
            raw = self._normalize_constellation_cloud_for_display(raw)
        pts = self._apply_display_mode(raw, mode_to_apply)
        if pts.size <= max_points:
            return pts
        idx = np.linspace(0, pts.size - 1, max_points, dtype=np.int64)
        return pts[idx].copy()

    def get_estimated_ber(self):
        with self._lock:
            return (np.array(self._ber_hist_t, dtype=np.float64),
                    np.array(self._ber_hist_v, dtype=np.float64))

    # =========================================================
    # Alpha/Beta performance surface for UI demonstration
    # =========================================================
    def _alpha_beta_surface_context_key(self) -> Tuple[Any, ...]:
        """Comparable-context key for the alpha/beta performance surface.

        The surface is meant to answer one question: under the same link,
        modulation, estimator, gain, coding and channel settings, which
        alpha/beta pair actually produced a better measured metric?  Alpha and
        beta themselves are intentionally excluded so changing them adds a new
        point; all other settings that would make points non-comparable are
        included and trigger an automatic clear.
        """
        def _r(value: Any, digits: int = 6) -> float:
            try:
                v = float(value)
                if not np.isfinite(v):
                    return float("nan")
                return round(v, int(digits))
            except Exception:
                return float("nan")

        return (
            str(getattr(self, "strict_chain_name", "")),
            str(getattr(self, "device_type", "")),
            str(getattr(self, "channel_mode", "")),
            str(getattr(self, "channel_estimator", "")),
            str(getattr(self, "requested_channel_estimator", "")),
            str(getattr(self, "mod_order", "")),
            str(getattr(self, "equalizer", "")),
            str(getattr(self, "coding_scheme", "")),
            bool(getattr(self, "coding_interleaver", False)),
            int(getattr(self, "M", 0)),
            int(getattr(self, "N", 0)),
            int(getattr(self, "cp_len", 0)),
            int(getattr(self, "max_full_htf_order", 0)),
            int(getattr(self, "tx_frame_count", 0)),
            int(getattr(self, "inter_frame_guard_len", 0)),
            int(getattr(self, "evm_average_frames", 0)),
            _r(getattr(self, "sample_rate", 0.0), 3),
            _r(getattr(self, "carrier_freq", 0.0), 3),
            _r(getattr(self, "tx_gain", 0.0), 3),
            _r(getattr(self, "rx_gain", 0.0), 3),
            _r(getattr(self, "training_amplitude", 0.0), 6),
            _r(getattr(self, "tdl_rms_delay_spread_ns", 0.0), 6),
            _r(getattr(self, "tdl_doppler_hz", 0.0), 6),
            _r(getattr(self, "tdl_doppler_spread_hz", 0.0), 6),
            _r(getattr(self, "tdl_snr_db", 0.0), 6),
            int(getattr(self, "tdl_seed", 0)),
            bool(getattr(self, "tdl_normalize_power", False)),
            int(getattr(self, "tdl_param_num_sinusoids", 0)),
            int(getattr(self, "tdl_param_max_paths", 0)),
            _r(getattr(self, "tdl_param_ridge", 0.0), 12),
            _r(getattr(self, "tdl_param_prune_db", 0.0), 6),
            int(getattr(self, "_tx_coded_bits_len", 0)),
            int(getattr(self, "_tx_uncoded_bits_len", 0)),
        )

    def _clear_alpha_beta_performance_surface_locked(self, reason: str = "context_change"):
        self._ab_metric_history = {}
        self._ab_surface_context_key = self._alpha_beta_surface_context_key()
        try:
            self._debug("INFO", f"alpha/beta performance surface cleared: {reason}")
        except Exception:
            pass

    def clear_alpha_beta_performance_surface(self, reason: str = "manual"):
        """Public UI hook: clear measured alpha/beta performance cells."""
        with self._lock:
            self._clear_alpha_beta_performance_surface_locked(reason=reason)

    def _ensure_alpha_beta_surface_context_locked(self):
        current_key = self._alpha_beta_surface_context_key()
        if current_key != getattr(self, "_ab_surface_context_key", None):
            self._clear_alpha_beta_performance_surface_locked(reason="non_alpha_beta_parameter_changed")

    @staticmethod
    def _finite_float_or_nan(value: Any) -> float:
        try:
            v = float(value)
            return v if np.isfinite(v) else float("nan")
        except Exception:
            return float("nan")

    def _alpha_beta_surface_target_samples_locked(self) -> int:
        """Number of real RX frames used before one α/β point is frozen.

        The UI asks for average metrics on the z-axis.  Therefore a point is
        not published while the current α/β value is still accumulating its
        averaging window.  Once this many processed frames have been collected,
        the cell is finalized and will never be changed again unless the user
        explicitly clears the surface or changes a non-α/β context parameter.
        """
        try:
            n = int(getattr(self, "_ab_surface_samples_per_cell", 0))
        except Exception:
            n = 0
        if n <= 0:
            try:
                n = int(getattr(self, "evm_average_frames", 1))
            except Exception:
                n = 1
        return int(max(1, min(n, 128)))

    def _finalize_alpha_beta_cell_locked(self, cell: Dict[str, Any]):
        """Freeze all z-axis metrics for one measured α/β cell.

        This is deliberately one-way: finalized cells are read-only.  It prevents
        a previously measured column from moving while the live link keeps
        processing repeated frames at the same α/β setting.
        """
        metric_store = cell.get("metrics", {}) if isinstance(cell, dict) else {}
        final_metrics: Dict[str, float] = {}
        final_counts: Dict[str, int] = {}

        for name, samples in dict(metric_store).items():
            try:
                arr = np.asarray(list(samples), dtype=np.float64).reshape(-1)
            except Exception:
                continue
            arr = arr[np.isfinite(arr)]
            if arr.size <= 0:
                continue
            if str(name) == "evm_instant_percent":
                # Keep the instantaneous diagnostic internally but do not expose it
                # in the UI selector.  It is also the mathematically correct source
                # for RMS average EVM over the finalized α/β window.
                final_metrics[str(name)] = float(arr[-1])
            elif str(name) == "evm_average_percent":
                # This value is overwritten below from evm_instant_percent when
                # available.  Keeping a fallback protects older call paths.
                final_metrics[str(name)] = float(arr[-1])
            else:
                final_metrics[str(name)] = float(np.mean(arr))
            final_counts[str(name)] = int(arr.size)

        try:
            inst = np.asarray(list(metric_store.get("evm_instant_percent", [])), dtype=np.float64).reshape(-1)
            inst = inst[np.isfinite(inst)]
        except Exception:
            inst = np.zeros(0, dtype=np.float64)
        if inst.size > 0:
            final_metrics["evm_average_percent"] = float(np.sqrt(np.mean(inst ** 2)))
            final_counts["evm_average_percent"] = int(inst.size)

        sample_count = int(max(0, cell.get("sample_count", 0)))
        decode_ok_count = int(max(0, cell.get("decode_ok_count", 0)))
        final_metrics["decode_success_ratio"] = float(decode_ok_count) / max(float(sample_count), 1.0)
        final_counts["decode_success_ratio"] = int(sample_count)

        # A frozen cell with no finite metrics is not useful; leave it partial so
        # a later valid frame under the same α/β can still complete it.
        if not final_metrics:
            return
        cell["final_metrics"] = final_metrics
        cell["final_metric_counts"] = final_counts
        cell["finalized"] = True
        cell["final_frame_counter"] = int(getattr(self, "_frames_processed", 0))
        cell["final_wall"] = float(time.time())

    def _record_alpha_beta_performance_sample_locked(self, metrics: Dict[str, Any]):
        """Record one measured frame into the alpha/beta surface.

        Caller must hold self._lock.  Values come from the same RX
        sync/equalizer/decoder path that updates BER/EVM status.  A cell is
        accumulated for a fixed averaging window and then frozen; finalized
        points are never updated by later frames, so already measured columns
        remain visually fixed while the operator moves to the next α/β setting.
        """
        self._ensure_alpha_beta_surface_context_locked()
        try:
            q_digits = int(getattr(self, "_ab_surface_quant_digits", 3))
            a = round(float(getattr(self, "alpha", 0.0)), q_digits)
            b = round(float(getattr(self, "beta", 0.0)), q_digits)
        except Exception:
            return
        key = (a, b)
        if key not in self._ab_metric_history and len(self._ab_metric_history) >= int(getattr(self, "_ab_surface_max_cells", 2048)):
            oldest = min(self._ab_metric_history.items(), key=lambda kv: float(kv[1].get("last_wall", 0.0)))[0]
            self._ab_metric_history.pop(oldest, None)

        target = self._alpha_beta_surface_target_samples_locked()
        cell = self._ab_metric_history.setdefault(key, {
            "alpha": float(a),
            "beta": float(b),
            "metrics": {},
            "sample_count": 0,
            "decode_ok_count": 0,
            "last_frame_counter": 0,
            "last_wall": 0.0,
            "target_sample_count": int(target),
            "finalized": False,
        })

        # Finalized means finalized.  Do not refresh sample_count, last_frame,
        # last_wall, or metrics; the visible bar and its metadata must remain
        # fixed for repeatability.
        if bool(cell.get("finalized", False)):
            return

        cell["target_sample_count"] = int(target)
        cell["sample_count"] = int(cell.get("sample_count", 0)) + 1
        if bool(metrics.get("decode_ok", False)):
            cell["decode_ok_count"] = int(cell.get("decode_ok_count", 0)) + 1
        cell["last_frame_counter"] = int(getattr(self, "_frames_processed", 0))
        cell["last_wall"] = float(time.time())

        metric_store = cell.setdefault("metrics", {})
        # Keep the full target window.  No rolling maxlen is used because a point
        # is frozen exactly once after target samples; losing early samples before
        # finalization would make the average ambiguous.
        for name, value in dict(metrics).items():
            if name == "decode_ok":
                value = 1.0 if bool(value) else 0.0
            if name == "evm_average_count":
                continue
            v = self._finite_float_or_nan(value)
            if not np.isfinite(v):
                continue
            if name not in metric_store:
                metric_store[name] = []
            metric_store[name].append(float(v))

        # Decode-rate is recomputed from the whole cell window.
        metric_store.setdefault("decode_success_ratio", [])
        metric_store["decode_success_ratio"].append(
            float(cell["decode_ok_count"]) / max(float(cell["sample_count"]), 1.0)
        )

        if int(cell.get("sample_count", 0)) >= int(target):
            self._finalize_alpha_beta_cell_locked(cell)
            if bool(cell.get("finalized", False)):
                try:
                    fm = cell.get("final_metrics", {})
                    evm = fm.get("evm_average_percent", float("nan"))
                    msg = (
                        f"alpha/beta point frozen: alpha={float(a):.3f}, beta={float(b):.3f}, "
                        f"samples={int(cell.get('sample_count', 0))}/{int(target)}"
                    )
                    if np.isfinite(float(evm)):
                        msg += f", EVMavg={float(evm):.3f}%"
                    self._debug("INFO", msg)
                except Exception:
                    pass

    def get_alpha_beta_performance_surface(self, metric: str = "evm_average_percent") -> Dict[str, Any]:
        """Return frozen measured alpha/beta surface points for one metric.

        Only finalized cells are returned as visible bars.  Partially measured
        cells are reported separately as progress metadata but are not drawn as
        columns, which eliminates fast-changing bars during the averaging window.
        """
        metric = str(metric or "evm_average_percent")
        with self._lock:
            self._ensure_alpha_beta_surface_context_locked()
            points: List[Dict[str, Any]] = []
            partial_count = 0
            active_key = None
            try:
                q_digits = int(getattr(self, "_ab_surface_quant_digits", 3))
                active_key = (round(float(getattr(self, "alpha", 0.0)), q_digits),
                              round(float(getattr(self, "beta", 0.0)), q_digits))
            except Exception:
                active_key = None
            active_progress = {"sample_count": 0, "target_sample_count": self._alpha_beta_surface_target_samples_locked(), "finalized": False}

            for key, cell in self._ab_metric_history.items():
                finalized = bool(cell.get("finalized", False))
                if key == active_key:
                    active_progress = {
                        "sample_count": int(cell.get("sample_count", 0)),
                        "target_sample_count": int(cell.get("target_sample_count", self._alpha_beta_surface_target_samples_locked())),
                        "finalized": finalized,
                    }
                if not finalized:
                    partial_count += 1
                    continue
                metrics_out = dict(cell.get("final_metrics", {}))
                counts_out = dict(cell.get("final_metric_counts", {}))
                if not metrics_out:
                    continue
                z = self._finite_float_or_nan(metrics_out.get(metric, float("nan")))
                points.append({
                    "alpha": float(cell.get("alpha", 0.0)),
                    "beta": float(cell.get("beta", 0.0)),
                    "z": z,
                    "metric": metric,
                    "metrics": metrics_out,
                    "metric_counts": counts_out,
                    "sample_count": int(cell.get("sample_count", 0)),
                    "decode_ok_count": int(cell.get("decode_ok_count", 0)),
                    "target_sample_count": int(cell.get("target_sample_count", self._alpha_beta_surface_target_samples_locked())),
                    "last_frame_counter": int(cell.get("final_frame_counter", cell.get("last_frame_counter", 0))),
                    "last_wall": float(cell.get("final_wall", cell.get("last_wall", 0.0))),
                    "finalized": True,
                })
            return {
                "metric": metric,
                "points": points,
                "point_count": int(len(points)),
                "partial_count": int(partial_count),
                "active_progress": active_progress,
                "current_alpha": float(getattr(self, "alpha", 0.0)),
                "current_beta": float(getattr(self, "beta", 0.0)),
                "context_key": repr(getattr(self, "_ab_surface_context_key", ())),
                "quant_digits": int(getattr(self, "_ab_surface_quant_digits", 3)),
                "samples_per_cell": int(self._alpha_beta_surface_target_samples_locked()),
            }

    def get_debug_snapshot(self) -> Dict[str, Any]:
        adaptive_ab = self.get_alpha_beta_adaptation_status()
        with self._lock:
            return {
                "frame_ok": bool(self.last_frame_ok),
                "reason": str(self.last_bad_reason),
                "sync_idx": int(self.last_sync_index),
                "payload_start": int(self.last_payload_start),
                "sync_metric": float(self.last_sync_metric),
                "cfo_est_hz": float(self.last_cfo_est_hz),
                "cfo_preamble_hz": float(getattr(self, "last_cfo_preamble_hz", self.last_cfo_est_hz)),
                "cfo_source": str(getattr(self, "last_cfo_source", "preamble")),
                "cfo_alias_hz": float(getattr(self, "_last_cfo_alias_hz", float("nan"))),
                "cfo_scan_score": float(getattr(self, "_last_cfo_scan_score", float("nan"))),
                "cfo_last_good_hz": float(getattr(self, "_last_good_cfo_hz", float("nan"))),
                "startup_settling": bool(time.time() < float(getattr(self, "_rx_settle_until_wall", 0.0)) or int(getattr(self, "_rx_settle_windows_remaining", 0)) > 0),
                "startup_settle_windows_remaining": int(getattr(self, "_rx_settle_windows_remaining", 0)),
                "ber": float(self._ber_estimate) if np.isfinite(self._ber_estimate) else float("nan"),
                "fec_bit_ber": float(getattr(self, "_last_fec_bit_ber", float("nan"))),
                "raw_bit_ber": float(getattr(self, "_last_raw_bit_ber", float("nan"))),
                "htf_leakage": float(self.last_htf_nmse),
                "cond_h_cross": float(self.last_cond_h_cross),
                "noise_var": float(self.last_noise_var),
                "evm_instant_percent": float(self.last_evm_instant_percent),
                "evm_average_percent": float(self.last_evm_average_percent),
                "evm_average_count": len(self._evm_history),
                "residual_gain_abs": float(self.last_residual_gain_abs),
                "residual_phase_deg": float(self.last_residual_phase_deg),
                "training_probe_guard_len": int(self.training_probe_guard_len),
                "evm_average_frames": int(self.evm_average_frames),
                "rx_samples_seen": int(self._rx_samples_seen),
                "rx_last_new_samples": int(getattr(self, "_rx_last_new_samples", 0)),
                "rx_stream_updates": int(getattr(self, "_rx_stream_updates", 0)),
                "rx_latest_window_len": int(getattr(self, "_rx_latest_window_len", 0)),
                "rx_spectrum_stale": bool(getattr(self, "_rx_spectrum_stale", True)),
                "rx_spectrum_stale_sec": float(getattr(self, "_rx_spectrum_stale_sec", float("inf"))),
                "constellation_source": str(getattr(self, "_last_constellation_source", "none")),
                "constellation_is_good": bool(getattr(self, "_last_constellation_is_good", False)),
                "constellation_good": bool(getattr(self, "_last_constellation_is_good", False)),
                "constellation_points": int(np.asarray(getattr(self, "_latest_constellation", np.zeros(0)), dtype=np.complex64).size),
                "constellation_post_eq_points": int(np.asarray(getattr(self, "_latest_constellation_post_eq", np.zeros(0)), dtype=np.complex64).size),
                "constellation_pre_eq_points": int(np.asarray(getattr(self, "_latest_constellation_pre_eq", np.zeros(0)), dtype=np.complex64).size),
                "constellation_tf_points": int(np.asarray(getattr(self, "_latest_constellation_tf", np.zeros(0)), dtype=np.complex64).size),
                # v17.1 visibility counters: these tick visibly while running
                # so the UI can tell the worker is alive even if a plot is
                # otherwise quiet.
                "frames_processed": int(self._frames_processed),
                "frames_decode_ok": int(self._frames_decode_ok),
                "monitor_cycles": int(self._monitor_cycles),
                "tx_buf_size": len(self._tx_buffer),
                "rx_buf_size": len(self._rx_buffer),
                "debug_seq": int(self._debug_seq),
                "alpha_beta_surface_points": int(sum(1 for _cell in getattr(self, "_ab_metric_history", {}).values() if bool(_cell.get("finalized", False)))),
                "alpha_beta_surface_partial_points": int(sum(1 for _cell in getattr(self, "_ab_metric_history", {}).values() if not bool(_cell.get("finalized", False)))),
                "alpha_beta_surface_context_key": repr(getattr(self, "_ab_surface_context_key", ())),
                "adaptive_alpha_beta": adaptive_ab,
                "adaptive_alpha_beta_enabled": bool(adaptive_ab.get("enabled", False)),
                "adaptive_alpha_beta_state": str(adaptive_ab.get("state", "disabled")),
                "adaptive_alpha_beta_ready": bool(adaptive_ab.get("ready", False)),
                "adaptive_alpha_beta_pending": bool(adaptive_ab.get("pending", False)),
                "adaptive_recommendation_seq": int(adaptive_ab.get("recommendation_seq", 0)),
                "adaptive_recommended_alpha": float(adaptive_ab.get("recommended_alpha", float("nan"))),
                "adaptive_recommended_beta": float(adaptive_ab.get("recommended_beta", float("nan"))),
                "adaptive_predicted_ser_current": float(adaptive_ab.get("predicted_ser_current", float("nan"))),
                "adaptive_predicted_ser_best": float(adaptive_ab.get("predicted_ser_best", float("nan"))),
                "adaptive_predicted_improvement_db": float(adaptive_ab.get("predicted_improvement_db", float("nan"))),
                "adaptive_predicted_snr_db": float(adaptive_ab.get("predicted_snr_db", float("nan"))),
                "adaptive_stable_count": int(adaptive_ab.get("stable_count", 0)),
                "adaptive_stable_required": int(adaptive_ab.get("stable_required", 0)),
                "adaptive_htf_source": str(adaptive_ab.get("htf_source", "")),
                "adaptive_search_seconds": float(adaptive_ab.get("search_seconds", float("nan"))),
                "adaptive_last_error": str(adaptive_ab.get("last_error", "")),
                "rx_probe_mode": str(self._rx_probe_mode),
                "rx_probe_len": int(self._rx_probe_len),
                "process_interval_ms": float(self.process_interval_sec * 1000.0),
                "tx_min_waveform_duration_ms": float(getattr(self, "tx_min_waveform_duration_ms", 0.0)),
                "tx_max_waveform_samples": int(getattr(self, "tx_max_waveform_samples", 0)),
                "tx_waveform_samples": int(np.asarray(getattr(self, "_tx_waveform", np.zeros(0)), dtype=np.complex64).size),
                "tx_base_cycle_len": int(getattr(self, "_tx_base_cycle_len", 0)),
                "tx_uhd_repeats": int(getattr(self, "_tx_uhd_repeats", 1)),
                "tx_tdl_prerendered": bool(getattr(self, "_tx_tdl_prerendered", False)),
                "tx_prerender_tdl_before_rf": bool(getattr(self, "tx_prerender_tdl_before_rf", True)),
                "coding_scheme": str(getattr(self, "coding_scheme", "none")),
                "coding_summary": self._coding_summary(),
                "coding_interleaver": bool(getattr(self, "coding_interleaver", False)),
                "tx_coded_bits_len": int(getattr(self, "_tx_coded_bits_len", 0)),
                "tx_uncoded_bits_len": int(getattr(self, "_tx_uncoded_bits_len", 0)),
                "requested_channel_estimator": str(getattr(self, "requested_channel_estimator", self.channel_estimator)),
                "effective_channel_estimator": str(getattr(self, "channel_estimator", "")),
                "estimator_effective_reason": str(getattr(self, "estimator_effective_reason", "")),
                "fdidm_transform_impl": "fft_4_dft_sum_for_fdit_ifdit",
                "full_htf_update_interval_frames": int(self.full_htf_update_interval_frames),
                "full_htf_once": bool(self.full_htf_once),
                "full_htf_cached": bool(self._cached_htf_full is not None),
                "full_htf_estimates": int(self._full_htf_estimates),
                "use_tdl_param_htf": bool(getattr(self, "use_tdl_param_htf", False)),
                "use_tdl_param": bool(getattr(self, "use_tdl_param_htf", False)),
                "tdl_param_fit_nmse": float(getattr(self, "_last_tdl_param_fit_nmse", float("nan"))),
                "tdl_param_nmse": float(getattr(self, "_last_tdl_param_fit_nmse", float("nan"))),
                "tdl_param_path_count": int(getattr(self, "last_tdl_param_path_count", 0)),
                "tdl_param_rank": int(getattr(self, "last_tdl_param_rank", 0)),
                "tdl_param_cond": float(getattr(self, "last_tdl_param_cond", float("nan"))),
                "tdl_param_ridge": float(getattr(self, "tdl_param_ridge", 0.0)),
                "tdl_param_prune_db": float(getattr(self, "tdl_param_prune_db", -90.0)),
                "tdl_param_num_sinusoids": int(getattr(self, "tdl_param_num_sinusoids", 8)),
                "tdl_param_max_paths": int(getattr(self, "tdl_param_max_paths", 96)),
                "constellation_quality": str(getattr(self, "last_constellation_quality", "unknown")),
                "constellation_good": bool(getattr(self, "_last_constellation_is_good", False)),
                "rx_frame_samples": int(np.asarray(getattr(self, "_latest_rx_frame_samples", []), dtype=np.complex64).size),
                "rx_data_samples": int(np.asarray(getattr(self, "_latest_rx_data_samples", []), dtype=np.complex64).size),
                "rx_pilot_samples": int(np.asarray(getattr(self, "_latest_rx_pilot_samples", []), dtype=np.complex64).size),
                "usrp_buffer_frames": int(self.usrp_buffer_frames),
            }

    def get_status(self) -> Dict[str, Any]:
        snap = self.get_debug_snapshot()
        stats = self.get_decode_stats()
        return {
            "status": self._status,
            "waveform": "FDIDM_STRICT_PAPER",
            "chain": self.strict_chain_name,
            "carrier_freq": self.carrier_freq,
            "sample_rate": self.sample_rate,
            "samp_rate": self.samp_rate,
            "subcarrier_spacing": self.subcarrier_spacing,
            "tx_gain": self.tx_gain,
            "rx_gain": self.rx_gain,
            "device_type": self.device_type,
            "device_args": self._usrp_args,
            "mod_order": self.mod_order,
            "equalizer": self.equalizer,
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "fdidm_alpha": float(self.alpha),
            "fdidm_beta": float(self.beta),
            "adaptive_alpha_beta": snap.get("adaptive_alpha_beta", {}),
            "adaptive_alpha_beta_enabled": bool(snap.get("adaptive_alpha_beta_enabled", False)),
            "adaptive_alpha_beta_state": str(snap.get("adaptive_alpha_beta_state", "disabled")),
            "adaptive_alpha_beta_ready": bool(snap.get("adaptive_alpha_beta_ready", False)),
            "adaptive_alpha_beta_pending": bool(snap.get("adaptive_alpha_beta_pending", False)),
            "adaptive_recommendation_seq": int(snap.get("adaptive_recommendation_seq", 0)),
            "adaptive_recommended_alpha": float(snap.get("adaptive_recommended_alpha", float("nan"))),
            "adaptive_recommended_beta": float(snap.get("adaptive_recommended_beta", float("nan"))),
            "adaptive_predicted_ser_current": float(snap.get("adaptive_predicted_ser_current", float("nan"))),
            "adaptive_predicted_ser_best": float(snap.get("adaptive_predicted_ser_best", float("nan"))),
            "adaptive_predicted_improvement_db": float(snap.get("adaptive_predicted_improvement_db", float("nan"))),
            "adaptive_predicted_snr_db": float(snap.get("adaptive_predicted_snr_db", float("nan"))),
            "adaptive_stable_count": int(snap.get("adaptive_stable_count", 0)),
            "adaptive_stable_required": int(snap.get("adaptive_stable_required", 0)),
            "adaptive_htf_source": str(snap.get("adaptive_htf_source", "")),
            "adaptive_search_seconds": float(snap.get("adaptive_search_seconds", float("nan"))),
            "adaptive_last_error": str(snap.get("adaptive_last_error", "")),
            "fdidm_m": int(self.M),
            "fdidm_n": int(self.N),
            "cp_len": int(self.cp_len),
            "frame_len": int(self.frame_len),
            "htf_training_blocks": int(self.htf_training_blocks),
            "full_htf_order": int(self.M * self.N),
            "channel_estimator": self.channel_estimator,
            "effective_channel_estimator": self.channel_estimator,
            "requested_channel_estimator": str(getattr(self, "requested_channel_estimator", self.channel_estimator)),
            "estimator_effective_reason": str(getattr(self, "estimator_effective_reason", "")),
            "fdidm_transform_impl": "fft_4_dft_sum_for_fdit_ifdit",
            "estimator_auto_note": str(getattr(self, "_estimator_auto_note", "")),
            "use_full_htf": bool(self.use_full_htf),
            "full_htf_update_interval_frames": int(self.full_htf_update_interval_frames),
            "full_htf_once": bool(self.full_htf_once),
            "full_htf_cached": bool(snap.get("full_htf_cached", False)),
            "full_htf_estimates": int(snap.get("full_htf_estimates", 0)),
            "use_tdl_param_htf": bool(snap.get("use_tdl_param_htf", False)),
            "use_tdl_param": bool(snap.get("use_tdl_param", snap.get("use_tdl_param_htf", False))),
            "tdl_param_fit_nmse": float(snap.get("tdl_param_fit_nmse", float("nan"))),
            "tdl_param_nmse": float(snap.get("tdl_param_nmse", snap.get("tdl_param_fit_nmse", float("nan")))),
            "tdl_param_path_count": int(snap.get("tdl_param_path_count", 0)),
            "tdl_param_rank": int(snap.get("tdl_param_rank", 0)),
            "tdl_param_cond": float(snap.get("tdl_param_cond", float("nan"))),
            "tdl_param_ridge": float(snap.get("tdl_param_ridge", getattr(self, "tdl_param_ridge", 0.0))),
            "tdl_param_prune_db": float(snap.get("tdl_param_prune_db", getattr(self, "tdl_param_prune_db", -90.0))),
            "tdl_param_num_sinusoids": int(snap.get("tdl_param_num_sinusoids", getattr(self, "tdl_param_num_sinusoids", 8))),
            "tdl_param_max_paths": int(snap.get("tdl_param_max_paths", getattr(self, "tdl_param_max_paths", 96))),
            "rx_last_new_samples": int(snap.get("rx_last_new_samples", 0)),
            "rx_spectrum_stale": bool(snap.get("rx_spectrum_stale", True)),
            "rx_spectrum_stale_sec": float(snap.get("rx_spectrum_stale_sec", float("inf"))),
            "constellation_source": str(snap.get("constellation_source", "none")),
            "constellation_is_good": bool(snap.get("constellation_is_good", False)),
            "constellation_points": int(snap.get("constellation_points", 0)),
            "constellation_pre_eq_points": int(snap.get("constellation_pre_eq_points", 0)),
            "constellation_quality": str(snap.get("constellation_quality", "unknown")),
            "rx_frame_samples": int(snap.get("rx_frame_samples", 0)),
            "rx_data_samples": int(snap.get("rx_data_samples", 0)),
            "rx_pilot_samples": int(snap.get("rx_pilot_samples", 0)),
            "process_interval_ms": float(snap.get("process_interval_ms", self.process_interval_sec * 1000.0)),
            "usrp_buffer_frames": int(snap.get("usrp_buffer_frames", self.usrp_buffer_frames)),
            "tx_min_waveform_duration_ms": float(snap.get("tx_min_waveform_duration_ms", self.tx_min_waveform_duration_ms)),
            "tx_max_waveform_samples": int(snap.get("tx_max_waveform_samples", self.tx_max_waveform_samples)),
            "tx_waveform_samples": int(snap.get("tx_waveform_samples", np.asarray(self._tx_waveform).size)),
            "tx_base_cycle_len": int(snap.get("tx_base_cycle_len", getattr(self, "_tx_base_cycle_len", 0))),
            "tx_uhd_repeats": int(snap.get("tx_uhd_repeats", getattr(self, "_tx_uhd_repeats", 1))),
            "tx_tdl_prerendered": bool(snap.get("tx_tdl_prerendered", getattr(self, "_tx_tdl_prerendered", False))),
            "tx_prerender_tdl_before_rf": bool(snap.get("tx_prerender_tdl_before_rf", getattr(self, "tx_prerender_tdl_before_rf", True))),
            "coding_scheme": str(snap.get("coding_scheme", getattr(self, "coding_scheme", "none"))),
            "coding_summary": str(snap.get("coding_summary", self._coding_summary())),
            "coding_interleaver": bool(snap.get("coding_interleaver", getattr(self, "coding_interleaver", False))),
            "tx_coded_bits_len": int(snap.get("tx_coded_bits_len", getattr(self, "_tx_coded_bits_len", 0))),
            "tx_uncoded_bits_len": int(snap.get("tx_uncoded_bits_len", getattr(self, "_tx_uncoded_bits_len", 0))),
            "channel_mode": str(self.channel_mode),
            "software_channel_enabled": bool(self._software_channel_enabled()),
            "rf_path_enabled": bool(self._rf_path_enabled()),
            "tdl_before_rf": bool(self._tdl_before_rf_enabled()),
            "tdl_after_rf": bool(self._tdl_after_rf_enabled()),
            "tdl_model": str(self._tdl_model_for_current_mode() or "off"),
            "tdl_rms_delay_spread_ns": float(self.tdl_rms_delay_spread_ns),
            "tdl_doppler_hz": float(self.tdl_doppler_hz),
            "tdl_doppler_spread_hz": float(self.tdl_doppler_spread_hz),
            "tdl_snr_db": float(self.tdl_snr_db),
            "tdl_seed": int(self.tdl_seed),
            "tdl_normalize_power": bool(self.tdl_normalize_power),
            "tx_frame_count": int(self.tx_frame_count),
            "tx_cycle_frame_count": int(getattr(self, "_tx_cycle_frame_count", self.tx_frame_count)),
            "inter_frame_guard_len": int(self.inter_frame_guard_len),
            "last_error": self._last_error,
            "last_info": self._last_info,
            "frame_ok": snap["frame_ok"],
            "reason": snap["reason"],
            "sync_metric": snap["sync_metric"],
            "cfo_est_hz": snap["cfo_est_hz"],
            "cfo_preamble_hz": snap.get("cfo_preamble_hz", snap["cfo_est_hz"]),
            "cfo_source": snap.get("cfo_source", "preamble"),
            "cfo_alias_hz": float(snap.get("cfo_alias_hz", getattr(self, "_last_cfo_alias_hz", float("nan")))),
            "cfo_scan_score": float(snap.get("cfo_scan_score", getattr(self, "_last_cfo_scan_score", float("nan")))),
            "cfo_unambiguous_hz": float(getattr(self, "_last_cfo_unambiguous_hz", self._preamble_cfo_unambiguous_hz())),
            "cfo_search_enable": bool(getattr(self, "cfo_search_enable", True)),
            "cfo_search_max_hz": float(getattr(self, "cfo_search_max_hz", 0.0)),
            "residual_cfo_max_hz": float(getattr(self, "residual_cfo_max_hz", 0.0)),
            "startup_settle_ms": float(getattr(self, "startup_settle_sec", 0.0) * 1000.0),
            "startup_settle_windows": int(getattr(self, "startup_settle_windows", 0)),
            "startup_settling": bool(snap.get("startup_settling", False)),
            "cfo_scan_min_score": float(getattr(self, "cfo_scan_min_score", 0.55)),
            "cfo_scan_jump_guard_hz": float(getattr(self, "cfo_scan_jump_guard_hz", 12000.0)),
            "cfo_last_good_hz": float(snap.get("cfo_last_good_hz", getattr(self, "_last_good_cfo_hz", float("nan")))),
            "parameter_limits": self.compute_parameter_limits(),
            "estimator_forced_reason": str(getattr(self, "_estimator_forced_reason", "")),
            "estimator_auto_note": str(getattr(self, "_estimator_auto_note", "")),
            "auto_tdl_param_for_software": bool(getattr(self, "auto_tdl_param_for_software", True)),
            "ber": snap["ber"],
            "fec_bit_ber": snap.get("fec_bit_ber", snap["ber"]),
            "raw_bit_ber": snap.get("raw_bit_ber", float("nan")),
            "htf_leakage": snap["htf_leakage"],
            "cond_h_cross": snap["cond_h_cross"],
            "noise_var": snap["noise_var"],
            "evm_percent": snap["evm_average_percent"],
            "evm_instant_percent": snap["evm_instant_percent"],
            "evm_average_percent": snap["evm_average_percent"],
            "evm_average_count": snap["evm_average_count"],
            "evm_average_frames": snap["evm_average_frames"],
            "rx_samples_seen": snap["rx_samples_seen"],
            "rx_last_new_samples": int(snap.get("rx_last_new_samples", 0)),
            "rx_stream_updates": int(snap.get("rx_stream_updates", 0)),
            "rx_latest_window_len": int(snap.get("rx_latest_window_len", 0)),
            "rx_spectrum_stale": bool(snap.get("rx_spectrum_stale", True)),
            "rx_spectrum_stale_sec": float(snap.get("rx_spectrum_stale_sec", float("inf"))),
            "constellation_source": str(snap.get("constellation_source", "none")),
            "constellation_is_good": bool(snap.get("constellation_is_good", False)),
            "constellation_good": bool(snap.get("constellation_good", snap.get("constellation_is_good", False))),
            "constellation_points": int(snap.get("constellation_points", 0)),
            "constellation_post_eq_points": int(snap.get("constellation_post_eq_points", 0)),
            "constellation_pre_eq_points": int(snap.get("constellation_pre_eq_points", 0)),
            "constellation_tf_points": int(snap.get("constellation_tf_points", 0)),
            "residual_gain_abs": snap["residual_gain_abs"],
            "residual_phase_deg": snap["residual_phase_deg"],
            "training_probe_guard_len": snap["training_probe_guard_len"],
            # v17.1 - visible "is the worker alive?" counters
            "frames_processed": snap["frames_processed"],
            "frames_decode_ok": snap["frames_decode_ok"],
            "monitor_cycles": snap["monitor_cycles"],
            "tx_buf_size": snap["tx_buf_size"],
            "rx_buf_size": snap["rx_buf_size"],
            "debug_seq": snap["debug_seq"],
            "alpha_beta_surface_points": int(snap.get("alpha_beta_surface_points", 0)),
            "alpha_beta_surface_partial_points": int(snap.get("alpha_beta_surface_partial_points", 0)),
            "alpha_beta_surface_context_key": str(snap.get("alpha_beta_surface_context_key", "")),
            "needs_top_block_rebuild": bool(self._needs_top_block_rebuild),
            "rx_probe_mode": snap.get("rx_probe_mode", str(self._rx_probe_mode)),
            "rx_probe_len": snap.get("rx_probe_len", int(self._rx_probe_len)),
            "decode_ok": stats["decode_ok"],
            "match_bytes": stats["match_bytes"],
            "expected_bytes": stats["expected_bytes"],
            "match_ratio": stats["match_ratio"],
            "equalizer_warning": self.last_equalizer_warning,
        }

    def get_last_error(self) -> str:
        return self._last_error


if __name__ == "__main__":
    tb = FDIDMHardwareTest(fdidm_m=16, fdidm_n=16, channel_mode="tdl_a_rf")
    st = tb.get_status()
    print(f"v35 channel-adaptive FDIDM ready: chain={st['chain']}, channel={st['channel_mode']}, "
          f"coding={st['coding_summary']}, tx_vector={st['tx_waveform_samples']} samples, "
          f"frame_len={st['frame_len']} samples "
          f"({st['frame_len'] / st['sample_rate'] * 1000:.2f} ms at {st['sample_rate'] / 1e6:.2f} MHz)")
