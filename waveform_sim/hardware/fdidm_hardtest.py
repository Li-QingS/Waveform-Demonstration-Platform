# -*- coding: utf-8 -*-

from __future__ import annotations

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
            channel_estimator: str = "full_htf",  # "full_htf" = paper-strict; "diag_tf" = fast loopback mode
            full_htf_update_interval_frames: int = 10000,  # kept for backward compatibility; one-shot mode ignores it
            full_htf_once: bool = True,              # v24: estimate full H_TF once, then reuse until cache is cleared
            process_interval_ms: float = 200.0,      # throttle heavy Python decoding to avoid UHD RX overflow
            usrp_buffer_frames: int = 512,           # UHD streamer buffer hint for Windows/B210 stability
            channel_mode: str = "rf",                # "rf" or "tdl_a"/"tdl_c"/"tdl_d" software loopback
            software_channel_model: Optional[str] = None,  # legacy/alias for channel_mode when set to a TDL model
            tdl_rms_delay_spread_ns: float = 1000.0,
            tdl_doppler_hz: float = 0.0,             # common satellite Doppler shift applied to all taps
            tdl_doppler_spread_hz: float = 0.0,      # local scattering Doppler spread for Rayleigh taps
            tdl_snr_db: float = 35.0,
            tdl_seed: int = 0x38811,
            tdl_normalize_power: bool = True,
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
        self.channel_estimator = str(channel_estimator or "full_htf").lower()
        if self.channel_estimator not in ("full_htf", "diag_tf"):
            raise ValueError("channel_estimator must be 'full_htf' or 'diag_tf'")
        self.full_htf_update_interval_frames = int(max(1, min(int(full_htf_update_interval_frames), 10_000)))
        self.full_htf_once = bool(full_htf_once)
        self.process_interval_sec = max(0.03, float(process_interval_ms) / 1000.0)
        self.usrp_buffer_frames = int(max(32, min(int(usrp_buffer_frames), 4096)))

        # Optional software NTN-TDL channel inserted between the FDIDM TX stream
        # and the RX decoder.  In TDL mode the flowgraph is pure baseband
        # loopback: vector_source -> throttle -> TDL channel -> RX probe.
        # In RF mode the original USRP TX/RX path is used.
        self.channel_mode = self._normalize_channel_mode(software_channel_model or channel_mode)
        self.tdl_rms_delay_spread_ns = float(max(0.0, float(tdl_rms_delay_spread_ns)))
        self.tdl_doppler_hz = float(tdl_doppler_hz)
        self.tdl_doppler_spread_hz = float(max(0.0, float(tdl_doppler_spread_hz)))
        self.tdl_snr_db = float(tdl_snr_db)
        self.tdl_seed = int(tdl_seed) & 0xFFFFFFFF
        self.tdl_normalize_power = bool(tdl_normalize_power)

        # Hardware/sync frame structure.
        # Frame = [pre_guard][sync_preamble][pilot_frame][data_frame][post_guard]
        self._recompute_strict_frame_timing()
        self.strict_chain_name = "FDIDM_FULL_HTF_ONESHOT_NTN_TDL_RXPROBE_STABLE_v24"

        # Tunables.
        self.sync_metric_threshold = 0.30
        self.update_period = 0.10
        self._rng_seed = 20260522

        # Cached transforms.
        self._gamma_cache: Dict[Tuple[int, float], np.ndarray] = {}

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
        # Full-H_TF cache: v24 performs one-shot channel identification.  The
        # first accepted full-H_TF estimate is reused for all later frames until
        # the cache is explicitly cleared or a structural/channel parameter is
        # reconfigured.  This matches the requested "identify once, do not
        # re-estimate every frame" behavior.
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
        self._latest_constellation = np.zeros(0, dtype=np.complex64)
        self._latest_constellation_pre_eq = np.zeros(0, dtype=np.complex64)
        self._last_good_constellation = np.zeros(0, dtype=np.complex64)
        self.constellation_display_mode = "raw"

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

        # v17.1 debug infrastructure
        self._debug_log: deque = deque(maxlen=2000)
        self._debug_seq = 0
        self._frames_processed = 0  # successful frame attempts (any sync, even bad ber)
        self._frames_decode_ok = 0  # CRC-passing frames
        self._monitor_cycles = 0  # alive counter for the worker thread
        self._monitor_last_log_t = 0.0
        self._needs_top_block_rebuild = False

        self._set_tx_text_internal(tx_text)
        self._build_top_block()
        self._debug("INFO",
                    f"FDIDM backend v23 rx-probe/cache ready: chain={self.strict_chain_name}, "
                    f"estimator={self.channel_estimator}, use_full_htf={self.use_full_htf}, "
                    f"H_once={self.full_htf_once}, H_update_legacy={self.full_htf_update_interval_frames} frame(s), "
                    f"channel_mode={self.channel_mode}, "
                    f"TDL_DS={self.tdl_rms_delay_spread_ns:.1f} ns, fd={self.tdl_doppler_hz:.1f} Hz, "
                    f"spread={self.tdl_doppler_spread_hz:.1f} Hz, SNR={self.tdl_snr_db:.1f} dB, "
                    f"process_interval={self.process_interval_sec*1000:.0f} ms, "
                    f"M={self.M} N={self.N} CP={self.cp_len} alpha={self.alpha:.3f} beta={self.beta:.3f} "
                    f"mod={self.mod_order} eq={self.equalizer} Fs={self.sample_rate:.0f} Hz "
                    f"frame_len={self.frame_len} ({self.frame_len / max(self.sample_rate, 1) * 1000.0:.2f} ms)")

    # =========================================================
    # Frame timing
    # =========================================================
    def _recompute_strict_frame_timing(self):
        self.pre_guard_len = max(16, self.M)
        self.sync_half_len = max(32, self.M)
        self.sync_len = 2 * self.sync_half_len
        self.block_len = self.M + self.cp_len
        self.data_frame_len = self.N * self.block_len

        # Paper-strict mode estimates the full H_TF in Eq. (20), then forms
        # H = Phi H_TF Phi^H in Eq. (29).  Estimating an MN x MN matrix needs
        # MN known TF-basis probe grids.  For large grids the code refuses to
        # silently fall back unless the user explicitly selects diag_tf.
        self.full_htf_order = int(self.M * self.N)
        self.use_full_htf = (self.channel_estimator == "full_htf" and
                             self.full_htf_order <= int(self.max_full_htf_order))
        if self.channel_estimator == "full_htf" and not self.use_full_htf:
            # Keep building so the UI can show a clear warning, but use the
            # fast one-frame diagonal estimator to avoid allocating a huge matrix.
            self.htf_training_blocks = 1
        else:
            self.htf_training_blocks = self.full_htf_order if self.use_full_htf else 1

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

    def _make_data_bits_for_frame(self, frame_bits: np.ndarray, frame_idx: int) -> np.ndarray:
        """Build one payload-bearing data grid with random filler.

        The application header/payload/CRC are identical in every transmitted
        physical frame, so CRC recovery remains deterministic.  The unused
        capacity after the application frame is randomized per physical frame,
        which prevents the USRP from replaying a completely identical data
        block forever and gives the constellation/decoder a more realistic
        symbol stream.
        """
        max_bits = self._max_data_bits_capacity()
        out = np.zeros(max_bits, dtype=np.int8)
        fb = np.asarray(frame_bits, dtype=np.int8).reshape(-1)
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
        max_bits = self._max_data_bits_capacity()
        if frame_bits.size > max_bits:
            max_payload_bytes = max(1, (max_bits // 8) - 12)
            raise ValueError(
                f"FDIDM strict frame capacity is too small: MxN={self.M}x{self.N}, "
                f"mod={self.mod_order}, max payload about {max_payload_bytes} bytes, "
                f"current UTF-8 text is {len(payload)} bytes."
            )

        # The pilot/training part is shared by all repeated physical frames.
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
            tx_bits_i = self._make_data_bits_for_frame(frame_bits, frame_idx)
            one_frame, x_cross_i, x_tf_i = self._build_one_physical_frame(tx_bits_i, pilot_block)
            frames.append(one_frame)
            bits_frames.append(tx_bits_i.astype(np.int8, copy=True))
            if first_x_cross is None:
                first_x_cross = x_cross_i.copy()
                first_x_tf = x_tf_i.copy()
            if self.inter_frame_guard_len > 0:
                frames.append(guard.copy())
        tx_wave = np.concatenate(frames) if frames else np.zeros(1, dtype=np.complex128)
        # Peak-normalize; this preserves the training/data ratio set above.
        peak = float(np.max(np.abs(tx_wave)) + 1e-12)
        tx_wave = (0.9 / peak) * tx_wave

        self._tx_text = text
        self._tx_payload = payload
        self._tx_frame = frame
        self._tx_frame_bits = frame_bits.astype(np.int8)
        self._tx_bits_frame = bits_frames[0].astype(np.int8) if bits_frames else np.zeros(max_bits, dtype=np.int8)
        self._tx_bits_frames = bits_frames
        self._tx_x_cross = first_x_cross if first_x_cross is not None else np.zeros((self.M, self.N), dtype=np.complex128)
        self._tx_x_tf = first_x_tf if first_x_tf is not None else np.zeros((self.M, self.N), dtype=np.complex128)
        self._tx_waveform = tx_wave.astype(np.complex64)
        self._rx_text = ""
        self._decode_ok = False
        self._match_bytes = 0
        self._last_good_rx_payload = b""
        self._last_raw_bytes = b""
        self._ber_estimate = float("nan")
        self._latest_constellation = np.zeros(0, dtype=np.complex64)
        self._latest_constellation_pre_eq = np.zeros(0, dtype=np.complex64)
        self._last_good_constellation = np.zeros(0, dtype=np.complex64)
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

    def _ifdit(self, x_cross: np.ndarray) -> np.ndarray:
        # Paper Eq. 6: X_TF = Gamma_M(alpha) @ X @ Gamma_N(-beta)
        gm = self._gamma(self.M, self.alpha)
        gn = self._gamma(self.N, -self.beta)
        return (gm @ np.asarray(x_cross, dtype=np.complex128) @ gn).astype(np.complex128)

    def _fdit(self, y_tf: np.ndarray) -> np.ndarray:
        # Paper Eq. 13: Y = Gamma_M(-alpha) @ Y_TF @ Gamma_N(beta)
        gm = self._gamma(self.M, -self.alpha)
        gn = self._gamma(self.N, self.beta)
        return (gm @ np.asarray(y_tf, dtype=np.complex128) @ gn).astype(np.complex128)

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
        """Schmidl-Cox-like metric: auto-corr for coarse plateau, cross-corr for fine timing.

        Returned metric uses the cross-correlation as the primary sharp metric,
        gated by the autocorrelation plateau as a coarse detector.
        """
        rx = np.asarray(rx, dtype=np.complex128)
        Ls = int(self.sync_len)
        L = int(self.sync_half_len)
        if rx.size < Ls + 1:
            return np.zeros(1, dtype=np.float64)
        sync = self.sync_preamble.astype(np.complex128)
        cross_corr = np.correlate(rx, sync, mode="valid")
        rx_abs2 = np.abs(rx) ** 2
        cum = np.concatenate([[0.0], np.cumsum(rx_abs2)])
        seg_energy = cum[Ls:] - cum[: rx.size - Ls + 1]
        m_cross = (np.abs(cross_corr) ** 2) / (self._sync_energy * (seg_energy + 1e-12))

        prod = np.conj(rx[:-L]) * rx[L:]
        prod_cum = np.concatenate([[0.0 + 0.0j], np.cumsum(prod)])
        P = prod_cum[L:] - prod_cum[: prod.size - L + 1]
        R_a = cum[L: rx.size - L + 1] - cum[: rx.size - 2 * L + 1]
        R_b = cum[2 * L: rx.size + 1] - cum[L: rx.size - L + 1]
        m_auto = (np.abs(P) ** 2) / (R_a * R_b + 1e-12)

        n = min(m_cross.size, m_auto.size)
        m_cross = m_cross[:n]
        m_auto = m_auto[:n]
        # Cross-corr peak gated by auto-corr plateau (>0.3) to suppress false alarms.
        gate = (m_auto > 0.30).astype(np.float64)
        return (m_cross * gate).astype(np.float64)

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
        """Refine residual CFO between sync and data using consecutive pilot OFDM symbols.

        Each Heisenberg block is M+CP samples; the M-point DFT of two consecutive
        blocks gives Y_TF[:, n] and Y_TF[:, n+1]. Their phase ratio per cell
        encodes the residual CFO (since the channel is approximately the same).
        """
        if self.N < 2:
            return 0.0
        try:
            y_tf = self._wigner(pilot_samples)
        except ValueError:
            return 0.0
        x_tf = self._pilot_X_tf
        h0 = y_tf[:, 0] / np.where(np.abs(x_tf[:, 0]) < 1e-10, 1e-10 + 0j, x_tf[:, 0])
        h1 = y_tf[:, 1] / np.where(np.abs(x_tf[:, 1]) < 1e-10, 1e-10 + 0j, x_tf[:, 1])
        ratios = h1 * np.conj(h0)
        # Drop weak / inf cells
        good = np.isfinite(ratios) & (np.abs(ratios) > 1e-6)
        if not np.any(good):
            return 0.0
        phase = float(np.angle(np.sum(ratios[good])))
        block_period = float(self.block_len) / max(self.sample_rate, 1e-12)
        return float(phase / (2.0 * np.pi * max(block_period, 1e-12)))

    def _fdidm_tx_matrix(self) -> np.ndarray:
        """A in vec(X_TF)=A vec(X), matching paper Eq. (25)."""
        return np.kron(self._gamma(self.N, -self.beta), self._gamma(self.M, self.alpha)).astype(np.complex128)

    def _fdidm_rx_matrix(self) -> np.ndarray:
        """Phi=A^H in y=Phi y_TF, matching paper Eqns. (27)-(29)."""
        return np.kron(self._gamma(self.N, self.beta), self._gamma(self.M, -self.alpha)).astype(np.complex128)

    def _clear_channel_cache(self):
        """Drop cached full-H_TF and derived cross-domain matrices."""
        self._cached_htf_full = None
        self._cached_htf_leakage = float("nan")
        self._cached_htf_frame_counter = -10**18
        self._cached_H_cross = None
        self._cached_Hh_cross = None
        self._cached_HhH_cross = None
        self._cached_Phi = None
        self._cached_H_cond_proxy = float("nan")
        self._full_htf_estimates = 0

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
        # v24 requested behavior: full-H_TF is one-shot channel identification.
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

    def _estimate_htf_diag_from_pilot(self, pilot_samples: np.ndarray) -> Tuple[np.ndarray, float]:
        """Per-cell H_TF estimate from a single known pilot frame.

        Returns:
            h_tf : (M, N) complex array, the diagonal of H_TF in TF coordinates.
            selectivity : coefficient of variation of |H_TF| across cells,
                std(|H|)/mean(|H|). For a flat link (cable or close-antenna OTA
                at this bandwidth) it is small; it grows as the channel becomes
                frequency-selective (resolvable multipath).
                v18 NOTE: this replaces the v17 "recon NMSE", which was
                tautologically ~0 (recon = (Y/X)*X == Y) and therefore could
                never detect a non-diagonal channel.
        """
        y_tf = self._wigner(pilot_samples)
        x_tf = self._pilot_X_tf
        # Safe per-cell division (pilot is now constant-modulus, so this rarely fires).
        safe_x = np.where(np.abs(x_tf) < 1e-10, 1e-10 + 0j, x_tf)
        h_tf = (y_tf / safe_x).astype(np.complex128)
        h_abs = np.abs(h_tf)
        mean_abs = float(np.mean(h_abs)) + 1e-12
        selectivity = float(np.std(h_abs) / mean_abs)
        return h_tf, selectivity

    def _estimate_noise_var_from_guard(self, frame_samples: np.ndarray) -> float:
        """Estimate noise variance per time sample from the post-frame guard."""
        if frame_samples.size <= self._off_end:
            return float("nan")
        guard = frame_samples[self._off_end: self._off_end + self.post_guard_len]
        if guard.size == 0:
            return float("nan")
        return float(np.mean(np.abs(guard) ** 2))

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
        if self.equalizer == "ZF":
            safe_H = np.where(np.abs(H) < 1e-10, 1e-10 + 0j, H)
            Z = Y / safe_H
        else:  # MMSE
            nv = max(float(noise_var) if np.isfinite(noise_var) else 0.0, 1e-12)
            W = np.conj(H) / (np.abs(H) ** 2 + nv)
            Z = Y * W
        x_hat = self._fdit(Z)
        h_abs = np.abs(H)
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

        v23 uses two stabilizers for real hardware tests:
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
        rotations = [1.0 + 0.0j]
        if self.mod_order == "QPSK":
            rotations = [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j]

        best = None
        for rot in rotations:
            syms = base_syms * rot
            bits_all = self._qam_demodulate(syms, self.mod_order)[:total_bits]
            ber = self._verification_ber(bits_all)
            frame_bits = bits_all[:frame_bits_len]
            frame_bytes = self._bits_to_bytes(frame_bits)
            ok, payload = self._parse_app_frame_exact(frame_bytes)
            text = payload.decode("utf-8", errors="replace") if ok else ""
            match = int(sum(int(a == b) for a, b in zip(payload, self._tx_payload))) if ok else 0
            evm = self._estimate_evm_percent(syms)
            score = (1000.0 * float(ok) + 100.0 * (1.0 - min(ber, 1.0))
                     - 0.2 * (evm if np.isfinite(evm) else 100.0))
            cand = (score, ber, frame_bytes, payload, text, match, bool(ok), syms.astype(np.complex64), evm)
            if best is None or cand[0] > best[0]:
                best = cand
            if ok:
                break
        _, ber, frame_bytes, payload, text, match, decode_ok, syms_best, evm = best
        return float(ber), frame_bytes, payload, text, int(match), bool(decode_ok), syms_best, float(evm)

    def _known_preamble_ref_syms(self) -> np.ndarray:
        # Use all known application-frame bits before random filler.  This is a
        # hardware demonstration link where the transmitted text is known; using
        # the whole application frame makes the residual scalar estimate much
        # less noisy than using only the 4-byte magic.
        bps = max(self.bits_per_symbol, 1)
        bits = np.asarray(self._tx_frame_bits, dtype=np.int8).reshape(-1)
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
    # Software NTN-TDL channel configuration
    # =========================================================
    def _normalize_channel_mode(self, mode: str) -> str:
        m = str(mode or "rf").strip().lower().replace("-", "_").replace(" ", "_")
        if m in ("rf", "usrp", "hardware", "off", "none", "no", "false", "0"):
            return "rf"
        return _NTNTDLChannel.normalize_model(m)

    def _software_channel_enabled(self) -> bool:
        return str(getattr(self, "channel_mode", "rf")).lower() in _NTNTDLChannel.TDL_PROFILES

    def _make_tdl_channel_block(self):
        """Create a GNU Radio Python sync_block wrapping _NTNTDLChannel."""
        gr = self._gr
        channel = _NTNTDLChannel(
            sample_rate=self.sample_rate,
            model=self.channel_mode,
            rms_delay_spread_ns=self.tdl_rms_delay_spread_ns,
            doppler_hz=self.tdl_doppler_hz,
            doppler_spread_hz=self.tdl_doppler_spread_hz,
            snr_db=self.tdl_snr_db,
            seed=self.tdl_seed,
            normalize_power=self.tdl_normalize_power,
        )

        class _TDLChannelBlock(gr.sync_block):
            def __init__(self):
                gr.sync_block.__init__(self, name="ntn_tdl_channel_v24", in_sig=[np.complex64], out_sig=[np.complex64])
                self.channel = channel

            def work(self, input_items, output_items):
                y = self.channel.process(input_items[0])
                output_items[0][:len(y)] = y
                return len(y)

            def reset_channel(self):
                self.channel.reset()

            def channel_summary(self) -> str:
                return self.channel.summary()

        return _TDLChannelBlock()

    def reset_full_htf_cache(self):
        """Public API used by the UI: force the next full-H_TF frame to re-identify the channel."""
        self._clear_channel_cache()
        self._debug("INFO", "full-H_TF cache cleared; next valid full-H frame will perform one-shot identification")

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

        v23 deliberately rounds the vector length to a power of two.  GNU Radio
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
        # Drop all strong references so old GR/UHD blocks can be finalized.
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
        # In RF mode, give libuhd a moment to release the USB endpoint before
        # re-opening.  In software-loopback mode this is harmless and brief.
        time.sleep(0.25 if not self._software_channel_enabled() else 0.05)

        mode_text = ("software " + str(self.channel_mode)) if self._software_channel_enabled() else "USRP RF"
        self._debug("INFO",
                    f"building new top_block ({mode_text}): {self._waveform_fingerprint()}, "
                    f"alpha={self.alpha:.3f}, beta={self.beta:.3f}, mod={self.mod_order}")

        class _TopBlock(gr.top_block):
            pass

        tb = _TopBlock("FDIDM Paper Strict Hardware Test v24 OneShot NTN-TDL", catch_exceptions=True)
        vector_source = blocks.vector_source_c(self._tx_waveform.tolist(), True, 1, [])
        tx_gain_block = blocks.multiply_const_cc(1.0)
        tx_sink_vec = None
        rx_stream_to_vector, rx_sink_vec, rx_probe, rx_probe_mode = self._build_rx_probe_chain()

        usrp_source = None
        usrp_sink = None
        throttle_block = None
        tdl_channel_block = None

        tb.connect((vector_source, 0), (tx_gain_block, 0))

        if self._software_channel_enabled():
            # Pure baseband loopback with a real-time throttle:
            # TX vector -> TDL channel -> RX probe.  This is the requested
            # software channel inserted between TX and RX.  It bypasses the RF
            # front-end so the impairment is exactly controlled and repeatable.
            throttle_block = blocks.throttle(gr.sizeof_gr_complex, float(self.sample_rate), True)
            tdl_channel_block = self._make_tdl_channel_block()
            tb.connect((tx_gain_block, 0), (throttle_block, 0))
            tb.connect((throttle_block, 0), (tdl_channel_block, 0))
            if rx_stream_to_vector is not None:
                tb.connect((tdl_channel_block, 0), (rx_stream_to_vector, 0))
                tb.connect((rx_stream_to_vector, 0), (rx_sink_vec, 0))
            else:
                tb.connect((tdl_channel_block, 0), (rx_sink_vec, 0))
            try:
                self._debug("INFO", "software channel configured: " + tdl_channel_block.channel_summary())
            except Exception:
                pass
        else:
            # Original hardware path: TX vector -> USRP sink, USRP source -> RX probe.
            if uhd is None:
                raise RuntimeError("GNU Radio UHD module is unavailable but channel_mode='rf' requires USRP.")
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
            if rx_stream_to_vector is not None:
                tb.connect((usrp_source, 0), (rx_stream_to_vector, 0))
                tb.connect((rx_stream_to_vector, 0), (rx_sink_vec, 0))
            else:
                tb.connect((usrp_source, 0), (rx_sink_vec, 0))

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
            self._latest_constellation = np.zeros(0, dtype=np.complex64)
            self._latest_constellation_pre_eq = np.zeros(0, dtype=np.complex64)
            self._last_good_constellation = np.zeros(0, dtype=np.complex64)
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
        return (f"len={wf.size}, energy={energy:.3f}, hash=0x{fold:08x}, "
                f"data_mid_idx={mid_idx}, data_mid={mid:.4f}")

    def get_waveform_fingerprint(self) -> str:
        """Public API: read the current TX waveform fingerprint without
        triggering a configure(). Useful for the UI to add a quick
        "is the waveform actually different now?" check."""
        return self._waveform_fingerprint()

    def _sync_waveform_to_top_block(self):

        if self._tb is None or self._vector_source is None:
            self._debug("WARN", "_sync_waveform_to_top_block: no top_block yet, skipping")
            return

        fingerprint = self._waveform_fingerprint()

        if not self._running:
            self._debug("INFO", f"sync (offline rebuild path): {fingerprint}")
            try:
                self._build_top_block()
                self._needs_top_block_rebuild = False
            except Exception as e:
                self._debug("ERROR", f"offline rebuild failed: {type(e).__name__}: {e}")
                self._needs_top_block_rebuild = True
            return

        # Running path: live swap.
        self._debug("INFO", f"sync (live swap path): {fingerprint}")
        try:
            data_list = self._tx_waveform.astype(np.complex64).tolist()
            # set_data() in modern GR (3.8+) takes (data, tags); older accept
            # just (data). Try both.
            try:
                self._vector_source.set_data(data_list, [])
            except TypeError:
                self._vector_source.set_data(data_list)
            # Force the source to restart from offset 0 so we don't emit a
            # half-old / half-new transient frame.
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
            self._debug("INFO", "live set_data() + rewind ok")
        except Exception as e:
            self._debug("WARN",
                        f"live set_data() failed ({type(e).__name__}: {e}); "
                        f"flagging rebuild for next stop/start")
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
            channel_mode: Optional[str] = None,
            software_channel_model: Optional[str] = None,
            tdl_rms_delay_spread_ns: Optional[float] = None,
            tdl_doppler_hz: Optional[float] = None,
            tdl_doppler_spread_hz: Optional[float] = None,
            tdl_snr_db: Optional[float] = None,
            tdl_seed: Optional[int] = None,
            tdl_normalize_power: Optional[bool] = None,
            **_ignored: Any,
    ):

        if self._running:
            # We allow hot-swapping a small subset of params while running.
            # Software-channel settings rebuild the top_block, so stop first.
            hot_swap_ok = all(v is None for v in (samp_rate, fdidm_m, fdidm_n, cp_len,
                                                  alpha, beta, mod_order, device_type, channel_estimator,
                                                  channel_mode, software_channel_model,
                                                  tdl_rms_delay_spread_ns, tdl_doppler_hz,
                                                  tdl_doppler_spread_hz, tdl_snr_db,
                                                  tdl_seed, tdl_normalize_power))
            if not hot_swap_ok:
                raise RuntimeError("Cannot reconfigure structural/channel parameters while running; stop first.")

        rebuild_waveform = False
        rebuild_top_block = False

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
            self.alpha = float(alpha);
            rebuild_waveform = True
        if beta is not None and float(beta) != self.beta:
            self.beta = float(beta);
            rebuild_waveform = True
        if fdidm_m is not None:
            new_m = int(max(4, min(int(fdidm_m), 64)))
            if new_m != self.M:
                self.M = new_m;
                rebuild_waveform = True
        if fdidm_n is not None:
            new_n = int(max(1, min(int(fdidm_n), 64)))
            if new_n != self.N:
                self.N = new_n;
                rebuild_waveform = True
        if cp_len is not None:
            new_cp = int(max(0, min(int(cp_len), max(self.M - 1, 0))))
            if new_cp != self.cp_len:
                self.cp_len = new_cp;
                rebuild_waveform = True
        if tx_frame_count is not None:
            new_fc = int(max(1, min(int(tx_frame_count), 32)))
            if new_fc != self.tx_frame_count:
                self.tx_frame_count = new_fc;
                rebuild_waveform = True
        if inter_frame_guard_len is not None:
            new_ig = int(max(0, min(int(inter_frame_guard_len), 8192)))
            if new_ig != self.inter_frame_guard_len:
                self.inter_frame_guard_len = new_ig;
                rebuild_waveform = True
        if evm_average_frames is not None:
            new_ev = int(max(1, min(int(evm_average_frames), 128)))
            if new_ev != self.evm_average_frames:
                self.evm_average_frames = new_ev
                self._evm_history = deque(maxlen=self.evm_average_frames)
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
                self.max_full_htf_order = new_max;
                rebuild_waveform = True
        if channel_estimator is not None:
            new_ce = str(channel_estimator).lower()
            if new_ce not in ("full_htf", "diag_tf"):
                raise ValueError("channel_estimator must be 'full_htf' or 'diag_tf'")
            if new_ce != self.channel_estimator:
                self.channel_estimator = new_ce
                rebuild_waveform = True
        if full_htf_update_interval_frames is not None:
            self.full_htf_update_interval_frames = int(max(1, min(int(full_htf_update_interval_frames), 10_000)))
        if process_interval_ms is not None:
            self.process_interval_sec = max(0.03, float(process_interval_ms) / 1000.0)
        if usrp_buffer_frames is not None:
            self.usrp_buffer_frames = int(max(32, min(int(usrp_buffer_frames), 4096)))
            rebuild_top_block = True
        if full_htf_once is not None:
            self.full_htf_once = bool(full_htf_once)

        # Software channel settings.  Any change invalidates one-shot CSI and
        # rebuilds the flowgraph because the channel is a live GNU Radio block.
        new_channel_mode = None
        if software_channel_model is not None:
            new_channel_mode = self._normalize_channel_mode(software_channel_model)
        elif channel_mode is not None:
            new_channel_mode = self._normalize_channel_mode(channel_mode)
        if new_channel_mode is not None and new_channel_mode != self.channel_mode:
            self.channel_mode = new_channel_mode
            rebuild_top_block = True
            self._clear_channel_cache()

        if tdl_rms_delay_spread_ns is not None:
            v = float(max(0.0, float(tdl_rms_delay_spread_ns)))
            if v != self.tdl_rms_delay_spread_ns:
                self.tdl_rms_delay_spread_ns = v
                rebuild_top_block = True
                self._clear_channel_cache()
        if tdl_doppler_hz is not None:
            v = float(tdl_doppler_hz)
            if v != self.tdl_doppler_hz:
                self.tdl_doppler_hz = v
                rebuild_top_block = True
                self._clear_channel_cache()
        if tdl_doppler_spread_hz is not None:
            v = float(max(0.0, float(tdl_doppler_spread_hz)))
            if v != self.tdl_doppler_spread_hz:
                self.tdl_doppler_spread_hz = v
                rebuild_top_block = True
                self._clear_channel_cache()
        if tdl_snr_db is not None:
            v = float(tdl_snr_db)
            if v != self.tdl_snr_db:
                self.tdl_snr_db = v
                rebuild_top_block = True
        if tdl_seed is not None:
            v = int(tdl_seed) & 0xFFFFFFFF
            if v != self.tdl_seed:
                self.tdl_seed = v
                rebuild_top_block = True
                self._clear_channel_cache()
        if tdl_normalize_power is not None:
            v = bool(tdl_normalize_power)
            if v != self.tdl_normalize_power:
                self.tdl_normalize_power = v
                rebuild_top_block = True
                self._clear_channel_cache()

        tx_text_changed = (tx_text is not None and str(tx_text) != self._tx_text)
        if rebuild_waveform or tx_text_changed:
            self._debug("INFO",
                        f"configure() applying changes: "
                        f"M={self.M} N={self.N} CP={self.cp_len} alpha={self.alpha:.3f} beta={self.beta:.3f} "
                        f"mod={self.mod_order} eq={self.equalizer} estimator={self.channel_estimator} "
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
            self._set_tx_text_internal(self._tx_text if tx_text is None else str(tx_text))
            self._tx_buffer.write(self._tx_waveform.astype(np.complex64, copy=False))
            # v17.1 fix: ALWAYS sync the new waveform to the GR top_block,
            # not just when running. When the UI does stop()->configure()->start()
            # we are *not* running between configure and start, so the old gating
            # `if self._running:` was the reason alpha/beta changes had no effect
            # on TX-side: the USRP just kept replaying the old vector_source data.
            self._sync_waveform_to_top_block()

        if rebuild_top_block and not self._running:
            self._debug("INFO",
                        f"configure() rebuilding top_block for path={self.channel_mode}, device_type={self.device_type}, "
                        f"TDL_DS={self.tdl_rms_delay_spread_ns:.1f}ns, fd={self.tdl_doppler_hz:.1f}Hz, "
                        f"spread={self.tdl_doppler_spread_hz:.1f}Hz, SNR={self.tdl_snr_db:.1f}dB")
            self._usrp_args = self._build_device_args()
            self._build_top_block()

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
        self._debug("INFO",
                    f"start(): launching top_block, TX waveform len={self._tx_waveform.size}, "
                    f"frame_len={self.frame_len}, alpha={self.alpha:.3f}, beta={self.beta:.3f}, "
                    f"mod={self.mod_order}, eq={self.equalizer}, Fs={self.sample_rate:.0f} Hz, "
                    f"H_once={self.full_htf_once}, channel_mode={self.channel_mode}, "
                    f"TDL_fd={self.tdl_doppler_hz:.1f} Hz")
        self._tb.start()
        self._tx_preview_start_t = time.time()
        self._rx_probe_start_t = time.time()
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
                with self._lock:
                    self._rx_samples_seen = int(abs_seen)
                    self._latest_rx_samples = rx_window[-8192:].astype(np.complex64, copy=True)
                    self._latest_tx_samples = tx_data[-8192:].astype(np.complex64, copy=True)
                    tx_buf_size = len(self._tx_buffer)
                # Heartbeat: log every ~2s so a static UI plot is easy to diagnose.
                now = time.time()
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
                if rx_data_size > 0 and rx_window.size >= self.frame_len:
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
        metric = self._sync_metric(rx_window)
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
            self._debug("DEBUG",
                        f"no sync peak above threshold: max_metric={max_metric:.3f}, "
                        f"threshold={self.sync_metric_threshold:.3f}, rx_window={rx_window.size}")
            return

        self._debug("DEBUG",
                    f"sync peaks found: count={len(peaks)}, max_metric={max_metric:.3f}, "
                    f"first_peak_idx={peaks[0]}, peaks={peaks}")

        best = None
        attempts = 0
        for coarse in peaks:
            attempts += 1
            sync_start = self._refine_sync_start(rx_window, coarse, search_radius=max(16, self.M // 2))
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

            cfo_hz = self._estimate_cfo_from_preamble(rx_window, sync_start)
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
            if not self.use_full_htf:
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
                else:
                    h_tf_est, leakage = self._estimate_htf_diag_from_pilot(pilot_samples)
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

            try:
                y_tf_data = self._wigner(data_samples)
            except Exception as e:
                self._debug("WARN", f"wigner failed: {type(e).__name__}: {e}")
                continue

            if self.use_full_htf:
                x_hat, cond_val, warning = self._equalize_data_full_htf(y_tf_data, h_tf_est, noise_var)
            else:
                x_hat, cond_val, warning = self._equalize_data_diag(y_tf_data, h_tf_est, noise_var)
            rx_syms = x_hat.reshape(-1, order="F")
            ber, raw_bytes, rx_payload, rx_text, match_bytes, decode_ok, rx_syms_best, evm_inst = (
                self._recover_payload_from_symbols(rx_syms)
            )

            sync_here = float(metric[min(max(int(coarse), 0), len(metric) - 1)])
            self._debug("DEBUG",
                        f"candidate #{attempts}: sync={sync_here:.3f}, CFO={cfo_hz:.1f} Hz, "
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
                sync_metric=sync_here, cfo_hz=float(cfo_hz),
                htf_leakage=float(leakage), cond_h=float(cond_val),
                noise_var=float(noise_var),
                warning=warning, ber=float(ber),
                raw_bytes=raw_bytes, rx_payload=rx_payload,
                rx_text=rx_text, match_bytes=int(match_bytes),
                decode_ok=bool(decode_ok),
                rx_syms=rx_syms_best, evm_inst=float(evm_inst),
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

        if good_quality:
            self._update_evm_history(best["evm_inst"])
        show_this_constellation = bool(good_quality)
        const_points = self._prepare_constellation_points(best["rx_syms"], display_mode="raw") if show_this_constellation else np.zeros(0, dtype=np.complex64)
        t_now = time.time() - self._t0
        with self._lock:
            self._last_processed_abs_start = best["abs_frame_start"]
            self.last_sync_index = int(best["frame_start"])
            self.last_payload_start = int(best["frame_start"] + self._off_data)
            self.last_sync_metric = float(best["sync_metric"])
            self.last_cfo_est_hz = float(best["cfo_hz"])
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
            if show_this_constellation:
                self._latest_constellation = const_points.astype(np.complex64)
                self._latest_constellation_pre_eq = const_points.astype(np.complex64)
                self._last_good_constellation = self._latest_constellation.copy()
            elif self._last_good_constellation.size > 0:
                self._latest_constellation = self._last_good_constellation.copy()
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
        self._debug(
            "INFO",
            f"v23 frame: mode={self.channel_estimator}, full_cached={self._cached_htf_full is not None}, sync={best['sync_metric']:.3f}, CFO={best['cfo_hz']:.1f} Hz, "
            f"Hleak={best['htf_leakage']:.3f}, cond={best['cond_h']:.2e}, "
            f"BER={best['ber']:.3e}, EVM={best['evm_inst']:.2f}%, "
            f"noise_var={best['noise_var']:.2e}, "
            f"decode_ok={best['decode_ok']}, match={best['match_bytes']}/{len(self._tx_payload)}"
        )

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
        return self._apply_display_mode(arr, display_mode)

    def _apply_display_mode(self, arr: np.ndarray, mode: str) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.complex64).reshape(-1)
        if arr.size == 0:
            return arr
        mode = str(mode).lower()
        if mode == "raw" or self.mod_order != "QPSK":
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
        if mode not in ("raw", "dd_refined", "hard_decision", "pre_equalized"):
            raise ValueError("unsupported constellation display mode")
        with self._lock:
            self.constellation_display_mode = mode

    def set_tx_gain(self, value: float):
        self.tx_gain = float(value)
        if getattr(self, "_usrp_sink", None) is not None:
            try:
                self._usrp_sink.set_gain(self.tx_gain, 0)
            except Exception:
                pass

    def set_rx_gain(self, value: float):
        self.rx_gain = float(value)
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

        v23 deliberately does not mirror the live TX stream into a Python sink
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

    def get_rx_spectrum_source(self, num_samples: int = 2048):
        return self.get_rx_samples(num_samples)

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
        with self._lock:
            mode = str(display_mode).lower() if display_mode is not None else self.constellation_display_mode
            raw = self._latest_constellation.copy()
            if raw.size == 0 and self._last_good_constellation.size > 0:
                raw = self._last_good_constellation.copy()
        pts = self._apply_display_mode(raw, mode)
        if pts.size <= max_points:
            return pts
        idx = np.linspace(0, pts.size - 1, max_points, dtype=np.int64)
        return pts[idx].copy()

    def get_estimated_ber(self):
        with self._lock:
            return (np.array(self._ber_hist_t, dtype=np.float64),
                    np.array(self._ber_hist_v, dtype=np.float64))

    def get_debug_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "frame_ok": bool(self.last_frame_ok),
                "reason": str(self.last_bad_reason),
                "sync_idx": int(self.last_sync_index),
                "payload_start": int(self.last_payload_start),
                "sync_metric": float(self.last_sync_metric),
                "cfo_est_hz": float(self.last_cfo_est_hz),
                "ber": float(self._ber_estimate) if np.isfinite(self._ber_estimate) else float("nan"),
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
                # v17.1 visibility counters: these tick visibly while running
                # so the UI can tell the worker is alive even if a plot is
                # otherwise quiet.
                "frames_processed": int(self._frames_processed),
                "frames_decode_ok": int(self._frames_decode_ok),
                "monitor_cycles": int(self._monitor_cycles),
                "tx_buf_size": len(self._tx_buffer),
                "rx_buf_size": len(self._rx_buffer),
                "debug_seq": int(self._debug_seq),
                "rx_probe_mode": str(self._rx_probe_mode),
                "rx_probe_len": int(self._rx_probe_len),
                "process_interval_ms": float(self.process_interval_sec * 1000.0),
                "full_htf_update_interval_frames": int(self.full_htf_update_interval_frames),
                "full_htf_once": bool(self.full_htf_once),
                "full_htf_cached": bool(self._cached_htf_full is not None),
                "full_htf_estimates": int(self._full_htf_estimates),
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
            "fdidm_m": int(self.M),
            "fdidm_n": int(self.N),
            "cp_len": int(self.cp_len),
            "frame_len": int(self.frame_len),
            "htf_training_blocks": int(self.htf_training_blocks),
            "full_htf_order": int(self.M * self.N),
            "channel_estimator": self.channel_estimator,
            "use_full_htf": bool(self.use_full_htf),
            "full_htf_update_interval_frames": int(self.full_htf_update_interval_frames),
            "full_htf_once": bool(self.full_htf_once),
            "full_htf_cached": bool(snap.get("full_htf_cached", False)),
            "full_htf_estimates": int(snap.get("full_htf_estimates", 0)),
            "process_interval_ms": float(snap.get("process_interval_ms", self.process_interval_sec * 1000.0)),
            "usrp_buffer_frames": int(snap.get("usrp_buffer_frames", self.usrp_buffer_frames)),
            "channel_mode": str(self.channel_mode),
            "software_channel_enabled": bool(self._software_channel_enabled()),
            "tdl_model": str(self.channel_mode) if self._software_channel_enabled() else "off",
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
            "ber": snap["ber"],
            "htf_leakage": snap["htf_leakage"],
            "cond_h_cross": snap["cond_h_cross"],
            "noise_var": snap["noise_var"],
            "evm_percent": snap["evm_average_percent"],
            "evm_instant_percent": snap["evm_instant_percent"],
            "evm_average_percent": snap["evm_average_percent"],
            "evm_average_count": snap["evm_average_count"],
            "evm_average_frames": snap["evm_average_frames"],
            "rx_samples_seen": snap["rx_samples_seen"],
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
    tb = FDIDMHardwareTest(fdidm_m=16, fdidm_n=16, channel_mode="tdl_a")
    st = tb.get_status()
    print(f"v24 one-shot full-H + NTN-TDL ready: chain={st['chain']}, channel={st['channel_mode']}, "
          f"frame_len={st['frame_len']} samples "
          f"({st['frame_len'] / st['sample_rate'] * 1000:.2f} ms at {st['sample_rate'] / 1e6:.2f} MHz)")
