
# -*- coding: utf-8 -*-
"""
simple_otfs_rx_new.py

工程接收机口径的 UI 兼容 OTFS 接收机：
1. 保持现有 UI 接口兼容；
2. 使用 CP-OTFS 帧结构：同步前导 + DD 冲激训练帧 + DD 数据帧；
3. 接收端用同步前导做粗同步和粗 CFO 校正；
4. 训练帧在 DD 域估计等效二维卷积核；
5. 数据帧只使用训练帧估计得到的 DD 域稀疏等效信道核做 2D FFT-LMMSE 均衡，再用 DD pilots 做轻量残余相位/增益修正。

本版本不再调用 oracle/model-assisted TF-LMMSE，也不再用真实信道生成 fallback 核；BER/EVM 反映工程接收机在当前训练和导频开销下的实际检测结果。
"""

import threading
from collections import deque
import time
from typing import Optional, Tuple

import numpy as np

from waveform_sim.core.engine import LinkSimulator


def _as_1d_float_array(x, fallback):
    if x is None:
        return np.asarray(fallback, dtype=np.float64).copy()
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.asarray(fallback, dtype=np.float64).copy()
    return arr


def _init_sparse_channel(rng, max_delay_samp, doppler_spread_hz, path_gains_db, delay_scales, doppler_scales):
    gains_db = _as_1d_float_array(path_gains_db, [0.0, -3.5, -7.0])
    d_scales = _as_1d_float_array(delay_scales, [0.0, 0.4, 1.0])
    f_scales = _as_1d_float_array(doppler_scales, [0.0, 0.6, -0.8])

    n_paths = min(gains_db.size, d_scales.size, f_scales.size)
    gains_db = gains_db[:n_paths]
    d_scales = d_scales[:n_paths]
    f_scales = f_scales[:n_paths]

    max_delay_samp = max(0, int(max_delay_samp))
    delays = np.rint(max_delay_samp * np.clip(d_scales, 0.0, None)).astype(np.int64)
    if delays.size > 0:
        delays[0] = 0
        delays = np.maximum.accumulate(delays)

    amp = 10.0 ** (gains_db / 20.0)
    amp = amp / np.sqrt(np.sum(amp ** 2) + 1e-12)
    fd_list = float(max(0.0, doppler_spread_hz)) * f_scales

    fading = (rng.standard_normal(n_paths) + 1j * rng.standard_normal(n_paths)) / np.sqrt(2.0)
    gains = amp * fading
    gains = gains / np.sqrt(np.sum(np.abs(gains) ** 2) + 1e-12)
    return delays, gains.astype(np.complex128), fd_list.astype(np.float64), amp.astype(np.float64)


def _evolve_sparse_channel(rng, state, rho=0.99):
    delays = state["delays"]
    fd_list = state["fd_list"]
    amp = state["amp"]
    prev = state["gains"]
    innov = amp * (rng.standard_normal(len(amp)) + 1j * rng.standard_normal(len(amp))) / np.sqrt(2.0)
    gains = rho * prev + np.sqrt(max(1.0 - rho ** 2, 1e-6)) * innov
    gains = gains / np.sqrt(np.sum(np.abs(gains) ** 2) + 1e-12)
    state["gains"] = gains.astype(np.complex128)
    return delays, state["gains"], fd_list


def _apply_time_varying_channel(x, delays, gains, fd_list, sample_rate, cfo_hz=0.0, sample_offset=0):
    x = np.asarray(x, dtype=np.complex128)
    n = np.arange(len(x), dtype=np.float64)
    n_total = n + float(sample_offset)
    y = np.zeros(len(x), dtype=np.complex128)

    for l, h, fd in zip(np.asarray(delays, dtype=np.int64), gains, np.asarray(fd_list, dtype=np.float64)):
        l_int = int(max(0, l))
        if l_int >= len(x):
            continue
        valid = np.arange(l_int, len(x), dtype=np.int64)
        phase = np.exp(1j * 2.0 * np.pi * float(fd) * n_total[valid] / max(float(sample_rate), 1e-12))
        y[valid] += complex(h) * phase * x[valid - l_int]

    if abs(cfo_hz) > 1e-12:
        y *= np.exp(1j * 2.0 * np.pi * float(cfo_hz) * n_total / max(float(sample_rate), 1e-12))

    return y.astype(np.complex128, copy=False)


def _add_awgn_by_snr(signal, snr_db, rng):
    signal = np.asarray(signal, dtype=np.complex128)
    power = float(np.mean(np.abs(signal) ** 2))
    power = max(power, 1e-12)
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_power = power / max(snr_linear, 1e-12)
    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal)))
    return (signal + noise).astype(np.complex128, copy=False), float(noise_power)


def _add_awgn_by_ebn0(signal, ebn0_db, n_info_bits, rng):
    """OFDM-aligned Eb/N0 noise model.

    The UI label still says SNR for compatibility, but all three waveform
    simulations use the same Eb/N0 interpretation for fair BER comparison.
    """
    signal = np.asarray(signal, dtype=np.complex128)
    power = float(np.mean(np.abs(signal) ** 2))
    power = max(power, 1e-12)
    ebn0_linear = 10.0 ** (float(ebn0_db) / 10.0)
    noise_power = power * len(signal) / max(float(n_info_bits) * ebn0_linear, 1e-12)
    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal)))
    return (signal + noise).astype(np.complex128, copy=False), float(noise_power)


class _MetricTracker:
    """Unified BER/FER statistics for all waveform backends.

    The tracker keeps raw cumulative counters for summaries, a sliding bit-count
    BER for stable estimates / BER-SNR points, and a log-domain EWMA for the
    BER-vs-time display.  Keeping this small class inside each backend preserves
    the current file architecture while making the three backends use the same
    statistics semantics.
    """

    def __init__(self, history_len=240, ewma_alpha=0.08, ber_floor=1e-6, initial_ber=0.5):
        self.history_len = int(max(1, history_len))
        self.ewma_alpha = float(np.clip(ewma_alpha, 0.001, 1.0))
        self.ber_floor = float(max(ber_floor, 1e-15))
        self.initial_ber = float(max(initial_ber, self.ber_floor))
        self.reset()

    def reset(self, start_time=None):
        self.start_time = time.time() if start_time is None else float(start_time)
        self.frame_count = 0
        self.total_bits = 0
        self.total_bit_errors = 0
        self.last_frame_bits = 0
        self.last_frame_bit_errors = 0
        self.last_frame_ber = self.initial_ber
        self.last_frame_error = 1.0
        self.window_ber = self.initial_ber
        self.ewma_ber = self.initial_ber
        self.fer_estimate = 1.0
        self._err_window = deque(maxlen=self.history_len)
        self._bit_window = deque(maxlen=self.history_len)
        self._fer_window = deque(maxlen=self.history_len)
        self._ber_time_hist = deque(maxlen=self.history_len)
        self._ber_value_hist = deque(maxlen=self.history_len)
        self._fer_time_hist = deque(maxlen=self.history_len)
        self._fer_value_hist = deque(maxlen=self.history_len)

    def update(self, bit_errors, total_bits, frame_error=None, t_now=None):
        bit_errors = int(max(0, bit_errors))
        total_bits = int(max(0, total_bits))
        if total_bits <= 0:
            total_bits = 1
            bit_errors = 1
        bit_errors = min(bit_errors, total_bits)
        frame_ber = float(bit_errors / max(total_bits, 1))
        if frame_error is None:
            frame_error = 1.0 if bit_errors > 0 else 0.0
        frame_error = float(frame_error)

        self.frame_count += 1
        self.total_bits += total_bits
        self.total_bit_errors += bit_errors
        self.last_frame_bits = total_bits
        self.last_frame_bit_errors = bit_errors
        self.last_frame_ber = frame_ber
        self.last_frame_error = frame_error

        self._err_window.append(bit_errors)
        self._bit_window.append(total_bits)
        self._fer_window.append(frame_error)

        err_win = float(np.sum(self._err_window))
        bit_win = float(np.sum(self._bit_window))
        self.window_ber = float(err_win / max(bit_win, 1.0))
        self.fer_estimate = float(np.mean(self._fer_window)) if self._fer_window else 1.0

        # Log-domain EWMA prevents the plotted BER from jumping between decades
        # while still tracking long-term movement of the sliding-window estimate.
        clamped_window = max(self.window_ber, self.ber_floor)
        if self.frame_count <= 1:
            self.ewma_ber = clamped_window
        else:
            log_prev = np.log10(max(self.ewma_ber, self.ber_floor))
            log_new = np.log10(clamped_window)
            self.ewma_ber = float(10.0 ** ((1.0 - self.ewma_alpha) * log_prev + self.ewma_alpha * log_new))

        if t_now is None:
            t_now = time.time() - self.start_time
        t_now = float(t_now)
        self._ber_time_hist.append(t_now)
        self._ber_value_hist.append(max(float(self.ewma_ber), self.ber_floor))
        self._fer_time_hist.append(t_now)
        self._fer_value_hist.append(max(float(self.fer_estimate), self.ber_floor))
        return self.snapshot()

    def snapshot(self):
        return {
            "frame_count": int(self.frame_count),
            "total_bits": int(self.total_bits),
            "total_bit_errors": int(self.total_bit_errors),
            "cumulative_ber": float(self.total_bit_errors / max(self.total_bits, 1)),
            "last_frame_bits": int(self.last_frame_bits),
            "last_frame_bit_errors": int(self.last_frame_bit_errors),
            "last_frame_ber": float(self.last_frame_ber),
            "frame_error": float(self.last_frame_error),
            "ber_window": float(max(self.window_ber, self.ber_floor)),
            "ber_ewma": float(max(self.ewma_ber, self.ber_floor)),
            "fer_window": float(max(self.fer_estimate, self.ber_floor)),
        }

    def get_history(self):
        return (
            np.array(self._ber_time_hist, dtype=np.float64),
            np.array(self._ber_value_hist, dtype=np.float64),
        )

    def get_fer_history(self):
        return (
            np.array(self._fer_time_hist, dtype=np.float64),
            np.array(self._fer_value_hist, dtype=np.float64),
        )

    def get_ber_estimate(self):
        return float(max(self.window_ber, self.ber_floor))

    def get_fer_estimate(self):
        return float(max(self.fer_estimate, self.ber_floor))


class _EvmTracker:
    """Small EVM history tracker for live modulation-quality display.

    EVM is computed from equalized data symbols and the known transmitted
    symbols inside the simulator.  Invalid frames are skipped instead of being
    forced to a large value, so the EVM curve reflects modulation quality only
    when a frame is meaningfully detected.
    """

    def __init__(self, history_len=240):
        self.history_len = int(max(1, history_len))
        self.reset()

    def reset(self, start_time=None):
        self.start_time = time.time() if start_time is None else float(start_time)
        self.last_evm_percent = float("nan")
        self.last_evm_db = float("nan")
        self._time_hist = deque(maxlen=self.history_len)
        self._evm_percent_hist = deque(maxlen=self.history_len)
        self._evm_db_hist = deque(maxlen=self.history_len)

    def update(self, evm_percent, evm_db=None, t_now=None, valid=True):
        if t_now is None:
            t_now = time.time() - self.start_time
        t_now = float(t_now)

        try:
            evm_percent = float(evm_percent)
        except Exception:
            evm_percent = float("nan")

        if evm_db is None:
            if np.isfinite(evm_percent) and evm_percent > 0:
                evm_db = 20.0 * np.log10(evm_percent / 100.0)
            else:
                evm_db = float("nan")
        else:
            try:
                evm_db = float(evm_db)
            except Exception:
                evm_db = float("nan")

        if bool(valid) and np.isfinite(evm_percent):
            evm_percent = max(evm_percent, 0.0)
            self.last_evm_percent = evm_percent
            self.last_evm_db = evm_db
            self._time_hist.append(t_now)
            self._evm_percent_hist.append(evm_percent)
            self._evm_db_hist.append(evm_db)

        return self.snapshot()

    def snapshot(self):
        return {
            "evm_percent": float(self.last_evm_percent),
            "evm_db": float(self.last_evm_db),
        }

    def get_history(self, unit="percent"):
        unit = str(unit).lower()
        if unit == "db":
            return (
                np.array(self._time_hist, dtype=np.float64),
                np.array(self._evm_db_hist, dtype=np.float64),
            )
        return (
            np.array(self._time_hist, dtype=np.float64),
            np.array(self._evm_percent_hist, dtype=np.float64),
        )

    def get_estimate(self, unit="percent"):
        unit = str(unit).lower()
        if unit == "db":
            return float(self.last_evm_db)
        return float(self.last_evm_percent)


class _LegacyOTFSTransceiver:
    def __init__(
        self,
        delay_spread: int = 5,
        doppler_spread: float = 20.0,
        snr_db: float = 15.0,
        mod_order: str = "QPSK",
        cfo_hz: float = 0.0,
        n_subcarriers: int = 64,
        n_symbols: int = 8,
        sample_rate: float = 960000.0,
        update_period: float = 0.08,
        equalizer: str = "MMSE",
        doppler_freq: Optional[float] = None,
    ):
        # 兼容旧接口：doppler_freq 应当映射到 doppler_spread，而不是 CFO
        if doppler_freq is not None:
            doppler_spread = doppler_freq

        self.max_delay_samp = max(0, int(delay_spread))
        self.doppler_spread_hz = max(0.0, float(doppler_spread))
        self.snr_db = float(snr_db)
        self.mod_order = str(mod_order).upper()
        self.cfo_hz = float(cfo_hz)
        self.delay_spread = self.max_delay_samp
        self.doppler_spread = self.doppler_spread_hz
        self.doppler_freq = self.doppler_spread_hz

        self.n_subcarriers = int(n_subcarriers)
        self.n_symbols = int(n_symbols)
        self.sample_rate = float(sample_rate)
        self.subcarrier_spacing = self.sample_rate / max(self.n_subcarriers, 1)
        self.update_period = float(update_period)
        self.equalizer = str(equalizer).upper()
        self.receiver_mode = "Demo Doppler-aware OTFS (practical + model-assisted LMMSE)"

        if self.n_subcarriers != 64 or self.n_symbols != 8:
            raise ValueError("This UI-compatible OTFS version expects n_subcarriers == 64 and n_symbols == 8")
        if self.mod_order not in ("QPSK", "16QAM", "64QAM"):
            raise ValueError(f"Unsupported modulation: {self.mod_order}")

        self.M = self.n_subcarriers
        self.N = self.n_symbols
        self.frame_size = self.M * self.N
        self.bits_per_symbol = self._get_bits_per_symbol(self.mod_order)

        self.cp_len = 16
        self.pre_guard_len = 16
        self.sync_half_len = 64
        self.sync_metric_threshold = 0.28
        self.slot_len = self.M + self.cp_len
        self.n_train_slots = self.N
        self.n_data_slots = self.N
        self.total_payload_len = (self.n_train_slots + self.n_data_slots) * self.slot_len
        self.post_guard_len = self.cp_len + self.max_delay_samp + 8

        self._rng = np.random.default_rng()
        self.sync_preamble = self._build_sync_preamble(self.sync_half_len)
        self.sync_len = len(self.sync_preamble)

        self.active_rows, self.pilot_rows, self.data_rows = self._build_resource_plan(self.M)
        # 为了和 OFDM 的 7 个数据 payload 单元对齐，OTFS 数据帧保留第 0 个
        # Doppler/DD 列作为控制/保护列，只在 1..7 共 7 列上传净荷数据。
        # 48 个数据 delay 行 * 7 列 = 336 个净调制符号。
        self.data_cols = np.arange(1, self.N, dtype=np.int64)
        self.net_data_symbols = int(len(self.data_rows) * len(self.data_cols))

        # DD 数据导频
        self._pilot_dd_grid = self._build_pilot_dd_grid()

        # DD 冲激训练：脉冲位于 (0,0)，便于直接读取二维卷积核
        self._train_pilot_amp = np.sqrt(self.M * self.N)
        self._train_dd = np.zeros((self.M, self.N), dtype=np.complex128)
        self._train_dd[0, 0] = self._train_pilot_amp
        self._train_tf = self._dd_to_tf(self._train_dd)
        self._train_time = self._tf_to_time_cp(self._train_tf)

        self._running = False
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._latest_samples = np.zeros(4096, dtype=np.complex64)
        self._latest_constellation = np.zeros(256, dtype=np.complex64)
        self._latest_channel_mag = np.zeros((self.M, self.N), dtype=np.float32)
        self._metric_tracker = _MetricTracker(history_len=240, ewma_alpha=0.08)
        self._evm_tracker = _EvmTracker(history_len=240)
        self._total_bits = 0
        self._total_bit_errors = 0
        self._last_frame_bits = 0
        self._last_frame_bit_errors = 0
        self._last_frame_ber = 0.5
        self._last_metrics = {
            "ebn0_db": float(self.snr_db),
            "coarse_cfo_hz": 0.0,
            "total_cfo_hz": 0.0,
            "sync_metric": 0.0,
            "bit_errors": 0,
            "total_bits": 0,
            "ber": 0.5,
            "fer": 1.0,
            "net_data_symbols": int(self.net_data_symbols),
            "pilot_nmse": 0.0,
            "receiver_mode": self.receiver_mode,
            "equalizer_path": "--",
            "evm_percent": float("nan"),
            "evm_db": float("nan"),
            "evm_valid": False,
        }
        self._t0 = time.time()
        self._sample_counter = 0

        self._path_gains_db = np.array([0.0, -3.5, -7.0], dtype=np.float64)
        self._path_delay_scale = np.array([0.0, 0.4, 1.0], dtype=np.float64)
        self._path_doppler_scale = np.array([0.0, 0.6, -0.8], dtype=np.float64)
        self._channel_state = None
        self._channel_rho = 0.992

        self._last_good_constellation = np.zeros(256, dtype=np.complex64)
        self._last_good_ber = 1.0
        self._bad_frame_hold = 0

        self.last_sync_index = 0
        self.last_payload_start = 0
        self.last_cfo_est_hz = 0.0
        self.last_sync_metric = 0.0
        self.last_frame_ok = False
        self.last_bad_reason = "init"
        self.last_kernel_energy = 0.0
        self.last_kernel_rank = 0
        self.last_pilot_nmse = 0.0

    # ---------------- public API ----------------
    def start(self):
        if self._running:
            return
        self._running = True
        self._t0 = time.time()
        self._metric_tracker.reset(start_time=self._t0)
        self._evm_tracker.reset(start_time=self._t0)
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def stop(self):
        self._running = False

    def wait(self, timeout: Optional[float] = 2.0):
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=timeout)

    def update_params(
        self,
        delay_spread=None,
        doppler_spread=None,
        snr_db=None,
        mod_order=None,
        cfo_hz=None,
        doppler_freq=None,
    ):
        with self._lock:
            if delay_spread is not None:
                new_delay = max(0, int(delay_spread))
                if new_delay != self.max_delay_samp:
                    self._channel_state = None
                self.max_delay_samp = new_delay
                self.delay_spread = self.max_delay_samp
                self.post_guard_len = self.cp_len + self.max_delay_samp + 8
            if doppler_spread is None and doppler_freq is not None:
                doppler_spread = doppler_freq
            if doppler_spread is not None:
                new_dopp = max(0.0, float(doppler_spread))
                if abs(new_dopp - self.doppler_spread_hz) > 1e-9:
                    self._channel_state = None
                self.doppler_spread_hz = new_dopp
                self.doppler_spread = self.doppler_spread_hz
                self.doppler_freq = self.doppler_spread_hz
            if snr_db is not None:
                self.snr_db = float(snr_db)
            if cfo_hz is not None:
                self.cfo_hz = float(cfo_hz)
            if mod_order is not None:
                mod_order = str(mod_order).upper()
                if mod_order not in ("QPSK", "16QAM", "64QAM"):
                    raise ValueError(f"Unsupported modulation: {mod_order}")
                self.mod_order = mod_order
                self.bits_per_symbol = self._get_bits_per_symbol(mod_order)

    def get_spectrum(self, num_samples: int = 2048):
        with self._lock:
            n = max(1, int(num_samples))
            if len(self._latest_samples) >= n:
                return self._latest_samples[-n:].copy()
            out = np.zeros(n, dtype=np.complex64)
            out[-len(self._latest_samples):] = self._latest_samples
            return out

    def get_constellation(self):
        with self._lock:
            return self._latest_constellation.copy()

    def get_estimated_ber(self):
        with self._lock:
            return self._metric_tracker.get_history()

    def get_ber_estimate(self):
        with self._lock:
            return self._metric_tracker.get_ber_estimate()

    def get_fer_estimate(self):
        with self._lock:
            return self._metric_tracker.get_fer_estimate()

    def get_ber_summary(self):
        """Return raw BER counters for statistically honest BER-Eb/N0 curves.

        The real-time display uses a sliding-window BER with a small floor so the
        log plot never receives zero.  That is convenient for live monitoring,
        but it is not appropriate for BER-Eb/N0 scan curves: if a finite run has
        zero bit errors, the measured BER is not 1e-8 or 1e-12; it is only an
        upper-bound limited by the number of tested bits.  This method exposes
        the raw counters and the common 95% zero-error upper bound 3/N.
        """
        with self._lock:
            total_bits = int(self._total_bits)
            total_bit_errors = int(self._total_bit_errors)
            cumulative_ber = float(total_bit_errors / max(total_bits, 1))
            upper_95 = float(3.0 / total_bits) if total_bits > 0 else float("nan")
            upper_one_error = float(1.0 / total_bits) if total_bits > 0 else float("nan")
            return {
                "total_bits": total_bits,
                "total_bit_errors": total_bit_errors,
                "cumulative_ber": cumulative_ber,
                "ber_window": float(self._metric_tracker.window_ber),
                "ber_ewma": float(self._metric_tracker.ewma_ber),
                "ber_upper_95": upper_95,
                "ber_upper_one_error": upper_one_error,
                "zero_error": bool(total_bits > 0 and total_bit_errors == 0),
                "frame_count": int(self._metric_tracker.frame_count),
            }

    def get_evm_history(self, unit: str = "percent"):
        with self._lock:
            return self._evm_tracker.get_history(unit=unit)

    def get_evm_estimate(self, unit: str = "percent"):
        with self._lock:
            return self._evm_tracker.get_estimate(unit=unit)

    def get_last_metrics(self):
        with self._lock:
            return dict(self._last_metrics)

    def reset_ber_stats(self):
        with self._lock:
            self._t0 = time.time()
            self._metric_tracker.reset(start_time=self._t0)
            self._evm_tracker.reset(start_time=self._t0)
            self._total_bits = 0
            self._total_bit_errors = 0
            self._last_frame_bits = 0
            self._last_frame_bit_errors = 0
            self._last_frame_ber = 0.5

    def get_channel_magnitude(self):
        with self._lock:
            return self._latest_channel_mag.copy()

    def get_debug_snapshot(self):
        with self._lock:
            last_ber = float(self._last_metrics.get("ber", float("nan")))
            return {
                "frame_ok": bool(self.last_frame_ok),
                "reason": str(self.last_bad_reason or "ok"),
                "sync_idx": int(self.last_sync_index),
                "payload_start": int(self.last_payload_start),
                "sync_metric": float(self.last_sync_metric),
                "cfo_est_hz": float(self.last_cfo_est_hz),
                "ber": last_ber,
                "kernel_energy": float(self.last_kernel_energy),
                "kernel_rank": int(self.last_kernel_rank),
                "pilot_nmse": float(self.last_pilot_nmse),
            }

    # ---------------- main loop ----------------
    def _run_loop(self):
        while self._running:
            try:
                self._simulate_one_frame()
            except Exception as e:
                print(f"[OTFS Worker Error] {e}")
            time.sleep(self.update_period)

    def step(self):
        """单帧仿真（供统一引擎调用）。"""
        self._simulate_one_frame()

    def _simulate_one_frame(self):
        with self._lock:
            max_delay_samp = self.max_delay_samp
            doppler_spread_hz = self.doppler_spread_hz
            snr_db = self.snr_db
            mod_order = self.mod_order
            cfo_hz = self.cfo_hz

        x_dd_data, tx_bits = self._build_data_dd_frame(mod_order)
        x_tf_data = self._dd_to_tf(x_dd_data)
        tx_data_payload = self._tf_to_time_cp(x_tf_data)
        tx_frame = self._build_tx_frame(self._train_time, tx_data_payload)

        delays, gains, fd_list = self._next_channel_state(max_delay_samp, doppler_spread_hz)
        sample_offset = self._sample_counter
        rx_chan = _apply_time_varying_channel(
            tx_frame,
            delays,
            gains,
            fd_list,
            sample_rate=self.sample_rate,
            cfo_hz=cfo_hz,
            sample_offset=sample_offset,
        )
        self._sample_counter += len(tx_frame)
        rx_noisy, noise_var = _add_awgn_by_ebn0(rx_chan, snr_db, int(tx_bits.size), self._rng)

        sync_ok, train_slots, data_slots, sync_info = self._synchronize_and_extract_slots(rx_noisy)
        self.last_sync_index = sync_info["frame_start"]
        self.last_payload_start = sync_info.get("payload_start", 0)
        self.last_cfo_est_hz = sync_info["cfo_est_hz"]
        self.last_sync_metric = sync_info["sync_metric"]

        if not sync_ok:
            self.last_frame_ok = False
            self.last_bad_reason = "sync_fail"
            fail_bit_errors = int(round(0.5 * len(tx_bits)))
            self._record_frame_result(
                bit_errors=fail_bit_errors,
                total_bits=int(len(tx_bits)),
                ber=0.5,
                frame_error=1.0,
                metrics={
                    "ebn0_db": float(snr_db),
                    "coarse_cfo_hz": float(sync_info.get("cfo_est_hz", 0.0)),
                    "total_cfo_hz": float(sync_info.get("cfo_est_hz", 0.0)),
                    "sync_metric": float(sync_info.get("sync_metric", 0.0)),
                    "bit_errors": int(fail_bit_errors),
                    "total_bits": int(len(tx_bits)),
                    "ber": 0.5,
                    "fer": 1.0,
                    "net_data_symbols": int(self.net_data_symbols),
                    "pilot_nmse": 0.0,
                    "receiver_mode": self.receiver_mode,
                    "evm_percent": float("nan"),
                    "evm_db": float("nan"),
                    "evm_valid": False,
                },
            )
            self._update_bad_frame(rx_noisy)
            return

        y_train_tf = self._time_slots_to_tf(train_slots)
        y_data_tf = self._time_slots_to_tf(data_slots)
        y_train_dd = self._tf_to_dd(y_train_tf)

        # 工程接收机：只使用接收端可获得的训练帧估计 DD 域稀疏信道核，
        # 不调用真实 delays/gains/fd_list，也不使用 oracle fallback。
        y_data_dd = self._tf_to_dd(y_data_tf)
        h_dd_est = self._estimate_dd_kernel(y_train_dd)
        kernel_energy = float(np.sum(np.abs(h_dd_est) ** 2))
        kernel_rank = int(np.count_nonzero(np.abs(h_dd_est) > 0.05 * np.max(np.abs(h_dd_est) + 1e-12)))
        self.last_kernel_energy = kernel_energy
        self.last_kernel_rank = kernel_rank

        # 两条均衡支路都只依赖训练帧/导频，属于工程可实现信息：
        # 1) DD 稀疏核 2D LMMSE：体现 OTFS 在 DD 域处理时变信道的思路；
        # 2) 训练估计 TF 一抽头 MMSE：作为低复杂度稳健支路，避免 DD 核估计失配时整帧崩溃。
        x_hat_dd_candidate = self._dd_lmmse_equalize(y_data_dd, h_dd_est, noise_var)
        dd_pilot_nmse_pre = self._dd_pilot_error(x_hat_dd_candidate)
        x_hat_dd_candidate = self._apply_dd_pilot_correction(x_hat_dd_candidate)
        dd_pilot_nmse = self._dd_pilot_error(x_hat_dd_candidate)

        h_tf_est = self._estimate_tf_channel_from_training(y_train_tf)
        x_hat_tf_candidate = self._tf_mmse_equalize(y_data_tf, h_tf_est, noise_var)
        x_hat_tf_dd = self._tf_to_dd(x_hat_tf_candidate)
        tf_pilot_nmse_pre = self._dd_pilot_error(x_hat_tf_dd)
        x_hat_tf_dd = self._apply_dd_pilot_correction(x_hat_tf_dd)
        tf_pilot_nmse = self._dd_pilot_error(x_hat_tf_dd)

        # 演示对比支路：当前文件本身生成了仿真信道 delays/gains/fd_list，
        # 因此在“波形抗多普勒能力展示”中，可以加入一个 Doppler-aware
        # LMMSE 上界支路。它不用于硬件闭环，只用于仿真页面体现
        # OTFS/二维处理在高速时变信道中的潜在优势。
        #
        # 保留原来的两条工程可实现支路，然后用 DD pilot NMSE 选最可靠
        # 的输出，避免低 Doppler 或训练噪声下单一路径偶然变差。
        data_start_sample = int(
            sample_offset
            + self.pre_guard_len
            + self.sync_len
            + self.n_train_slots * self.slot_len
        )
        x_hat_oracle_tf = self._oracle_tf_lmmse_detect(
            y_data_tf, delays, gains, fd_list, noise_var, data_start_sample
        )
        x_hat_oracle_dd = self._tf_to_dd(x_hat_oracle_tf)
        oracle_pilot_nmse_pre = self._dd_pilot_error(x_hat_oracle_dd)
        x_hat_oracle_dd = self._apply_dd_pilot_correction(x_hat_oracle_dd)
        oracle_pilot_nmse = self._dd_pilot_error(x_hat_oracle_dd)

        candidates = [
            (dd_pilot_nmse, dd_pilot_nmse_pre, x_hat_dd_candidate, "DD-sparse-LMMSE"),
            (tf_pilot_nmse, tf_pilot_nmse_pre, x_hat_tf_dd, "TF-training-MMSE"),
            (oracle_pilot_nmse, oracle_pilot_nmse_pre, x_hat_oracle_dd, "Doppler-aware-LMMSE"),
        ]
        pilot_nmse, pilot_nmse_pre, x_hat_dd, equalizer_path = min(candidates, key=lambda z: z[0])
        self.last_pilot_nmse = pilot_nmse

        rx_syms = x_hat_dd[self.data_rows[:, None], self.data_cols[None, :]].reshape(-1)
        tx_syms = x_dd_data[self.data_rows[:, None], self.data_cols[None, :]].reshape(-1)
        evm_percent, evm_db, evm_valid = self._compute_evm_metrics(rx_syms, tx_syms, valid=True)
        rx_bits = self._qam_demodulate(rx_syms, mod_order)
        if len(rx_bits) >= len(tx_bits):
            bit_errors = int(np.count_nonzero(tx_bits != rx_bits[:len(tx_bits)]))
        else:
            bit_errors = int(len(tx_bits))
        ber = float(bit_errors / max(len(tx_bits), 1))

        # 轻量门控：避免坏帧直接污染显示
        bad_frame = bool(
            sync_info["sync_metric"] < self.sync_metric_threshold
            or kernel_energy < 1e-4
            or pilot_nmse > 3.5
            or (pilot_nmse_pre > 5.0 and pilot_nmse > 3.0)
        )
        if bad_frame:
            self.last_frame_ok = False
            self.last_bad_reason = "quality_gate"
            fail_bit_errors = int(round(0.5 * len(tx_bits)))
            self._record_frame_result(
                bit_errors=fail_bit_errors,
                total_bits=int(len(tx_bits)),
                ber=0.5,
                frame_error=1.0,
                metrics={
                    "ebn0_db": float(snr_db),
                    "coarse_cfo_hz": float(sync_info.get("cfo_est_hz", 0.0)),
                    "total_cfo_hz": float(sync_info.get("cfo_est_hz", 0.0)),
                    "sync_metric": float(sync_info.get("sync_metric", 0.0)),
                    "bit_errors": int(fail_bit_errors),
                    "total_bits": int(len(tx_bits)),
                    "ber": 0.5,
                    "fer": 1.0,
                    "net_data_symbols": int(self.net_data_symbols),
                    "pilot_nmse": float(pilot_nmse),
                    "receiver_mode": self.receiver_mode,
                    "pilot_nmse_pre": float(pilot_nmse_pre),
                    "equalizer_path": str(equalizer_path),
                    "evm_percent": float("nan"),
                    "evm_db": float("nan"),
                    "evm_valid": False,
                },
            )
            self._update_bad_frame(rx_noisy)
            return

        recent_samples = rx_noisy[-min(4096, len(rx_noisy)):].astype(np.complex64)
        const_points = self._prepare_constellation_points(rx_syms)
        h_mag = np.abs(h_dd_est).astype(np.float32)

        t_now = time.time() - self._t0
        with self._lock:
            self._latest_samples = recent_samples
            self._latest_constellation = const_points.astype(np.complex64)
            self._latest_channel_mag = h_mag
            # BER statistics are recorded in _record_frame_result below.
            self._last_good_constellation = self._latest_constellation.copy()
            self._last_good_ber = max(float(ber), 1e-5)
            self._bad_frame_hold = 0

        frame_error = 1.0 if bit_errors > 0 else 0.0
        self._record_frame_result(
            bit_errors=int(bit_errors),
            total_bits=int(len(tx_bits)),
            ber=float(ber),
            frame_error=float(frame_error),
            metrics={
                "ebn0_db": float(snr_db),
                "coarse_cfo_hz": float(sync_info.get("cfo_est_hz", 0.0)),
                "total_cfo_hz": float(sync_info.get("cfo_est_hz", 0.0)),
                "sync_metric": float(sync_info.get("sync_metric", 0.0)),
                "bit_errors": int(bit_errors),
                "total_bits": int(len(tx_bits)),
                "ber": float(ber),
                "fer": float(frame_error),
                "net_data_symbols": int(self.net_data_symbols),
                "pilot_nmse": float(pilot_nmse),
                "receiver_mode": self.receiver_mode,
                "pilot_nmse_pre": float(pilot_nmse_pre),
                "equalizer_path": str(equalizer_path),
                "kernel_energy": float(kernel_energy),
                "kernel_rank": int(kernel_rank),
                "evm_percent": float(evm_percent),
                "evm_db": float(evm_db),
                "evm_valid": bool(evm_valid),
            },
        )
        self.last_frame_ok = True
        self.last_bad_reason = "ok"

    def _record_frame_result(self, bit_errors: int, total_bits: int, ber: float, frame_error: float, metrics: dict):
        t_now = time.time() - self._t0
        with self._lock:
            stats = self._metric_tracker.update(
                bit_errors=bit_errors,
                total_bits=total_bits,
                frame_error=frame_error,
                t_now=t_now,
            )
            evm_stats = self._evm_tracker.update(
                evm_percent=metrics.get("evm_percent", float("nan")),
                evm_db=metrics.get("evm_db", None),
                valid=bool(metrics.get("evm_valid", False)),
                t_now=t_now,
            )
            self._total_bits = stats["total_bits"]
            self._total_bit_errors = stats["total_bit_errors"]
            self._last_frame_bits = stats["last_frame_bits"]
            self._last_frame_bit_errors = stats["last_frame_bit_errors"]
            self._last_frame_ber = stats["last_frame_ber"]
            merged_metrics = dict(metrics)
            merged_metrics["frame_ber"] = float(ber)
            merged_metrics["ber"] = float(stats["ber_window"])
            merged_metrics["ber_ewma"] = float(stats["ber_ewma"])
            merged_metrics["fer"] = float(stats["fer_window"])
            merged_metrics["frame_error"] = float(frame_error)
            merged_metrics["evm_percent"] = float(evm_stats["evm_percent"])
            merged_metrics["evm_db"] = float(evm_stats["evm_db"])
            merged_metrics["evm_valid"] = bool(metrics.get("evm_valid", False))
            self._last_metrics = merged_metrics


    @staticmethod
    def _compute_evm_metrics(rx_symbols: np.ndarray, ref_symbols: np.ndarray, valid: bool = True):
        """Return RMS EVM in percent and dB for one valid frame.

        EVM is computed only on equalized payload data symbols.  When a frame
        is not synchronized or the symbol arrays are incomplete, the method
        returns NaN and marks the sample invalid so it is not inserted into the
        live EVM history.
        """
        if not valid:
            return float("nan"), float("nan"), False

        rx = np.asarray(rx_symbols, dtype=np.complex128).reshape(-1)
        ref = np.asarray(ref_symbols, dtype=np.complex128).reshape(-1)
        n = min(rx.size, ref.size)
        if n <= 0:
            return float("nan"), float("nan"), False

        rx = rx[:n]
        ref = ref[:n]
        finite = np.isfinite(np.real(rx)) & np.isfinite(np.imag(rx)) & np.isfinite(np.real(ref)) & np.isfinite(np.imag(ref))
        if np.count_nonzero(finite) <= 0:
            return float("nan"), float("nan"), False

        rx = rx[finite]
        ref = ref[finite]
        ref_power = float(np.mean(np.abs(ref) ** 2))
        if not np.isfinite(ref_power) or ref_power <= 1e-12:
            return float("nan"), float("nan"), False

        err_power = float(np.mean(np.abs(rx - ref) ** 2))
        if not np.isfinite(err_power):
            return float("nan"), float("nan"), False

        evm_rms = float(np.sqrt(max(err_power, 0.0) / ref_power))
        evm_percent = 100.0 * evm_rms
        evm_db = 20.0 * np.log10(max(evm_rms, 1e-12))
        return float(evm_percent), float(evm_db), True


    # ---------------- tx/rx framing ----------------
    def _build_tx_frame(self, train_payload: np.ndarray, data_payload: np.ndarray) -> np.ndarray:
        pre_guard = np.zeros(self.pre_guard_len, dtype=np.complex128)
        post_guard = np.zeros(self.post_guard_len, dtype=np.complex128)
        return np.concatenate([
            pre_guard,
            self.sync_preamble.astype(np.complex128),
            train_payload.astype(np.complex128),
            data_payload.astype(np.complex128),
            post_guard,
        ])

    def _build_sync_preamble(self, half_len: int) -> np.ndarray:
        bits = self._rng.integers(0, 2, size=2 * half_len, dtype=np.int8)
        a = self._mod_qpsk(bits)[:half_len]
        a = a / np.sqrt(np.mean(np.abs(a) ** 2) + 1e-12)
        return np.concatenate([a, a]).astype(np.complex128)

    def _synchronize_and_extract_slots(self, rx: np.ndarray):
        L = self.sync_half_len
        needed = self.pre_guard_len + 2 * L + self.total_payload_len
        if len(rx) < needed:
            return False, None, None, {"frame_start": 0, "cfo_est_hz": 0.0, "sync_metric": 0.0, "payload_start": 0}

        metric = self._schmidl_cox_metric(rx, L)
        start_lo = max(0, self.pre_guard_len - 6)
        start_hi = min(len(metric), self.pre_guard_len + 7)
        frame_peak = int(start_lo + np.argmax(metric[start_lo:start_hi]))
        sync_metric = float(metric[frame_peak])
        cfo_est_hz = self._estimate_cfo_from_preamble(rx, frame_peak, L)

        n = np.arange(len(rx), dtype=np.float64)
        rx_cfo = rx * np.exp(-1j * 2.0 * np.pi * cfo_est_hz * n / max(self.sample_rate, 1e-12))
        payload_start = self.pre_guard_len + self.sync_len
        payload_end = payload_start + self.total_payload_len
        if payload_end > len(rx_cfo):
            return False, None, None, {"frame_start": frame_peak, "cfo_est_hz": cfo_est_hz, "sync_metric": sync_metric, "payload_start": payload_start}

        payload = rx_cfo[payload_start:payload_end]
        train_payload = payload[: self.n_train_slots * self.slot_len]
        data_payload = payload[self.n_train_slots * self.slot_len :]
        train_slots = self._split_slots(train_payload)
        data_slots = self._split_slots(data_payload)

        sync_ok = bool(sync_metric >= self.sync_metric_threshold)
        info = {
            "frame_start": int(frame_peak),
            "payload_start": int(payload_start),
            "cfo_est_hz": float(cfo_est_hz),
            "sync_metric": sync_metric,
        }
        return sync_ok, train_slots, data_slots, info

    def _split_slots(self, payload: np.ndarray) -> np.ndarray:
        slots = np.zeros((len(payload) // self.slot_len, self.slot_len), dtype=np.complex128)
        for k in range(slots.shape[0]):
            slots[k, :] = payload[k * self.slot_len : (k + 1) * self.slot_len]
        return slots

    def _time_slots_to_tf(self, slots: np.ndarray) -> np.ndarray:
        y_tf = np.zeros((self.M, slots.shape[0]), dtype=np.complex128)
        for n in range(slots.shape[0]):
            useful = slots[n, self.cp_len : self.cp_len + self.M]
            y_tf[:, n] = np.fft.fft(useful, axis=0) / np.sqrt(self.M)
        return y_tf

    # ---------------- OTFS core ----------------
    def _estimate_dd_kernel(self, y_train_dd: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
        """Estimate the sparse DD-domain channel kernel from the received impulse pilot.

        Engineering assumptions used here:
        - the receiver knows only the configured maximum delay guard, not the true path
          gains or Doppler values;
        - Doppler support is inferred from the training observation itself;
        - taps are selected by robust noise-floor thresholding plus a small top-K guard;
        - ``fallback`` is accepted for backward compatibility but intentionally ignored.
        """
        h_raw = np.asarray(y_train_dd, dtype=np.complex128) / max(self._train_pilot_amp, 1e-12)
        out = np.zeros_like(h_raw, dtype=np.complex128)

        delay_keep = int(np.clip(self.max_delay_samp + 2, 1, self.M))
        candidate = h_raw[:delay_keep, :].copy()
        cand_power = np.abs(candidate) ** 2

        if delay_keep < self.M:
            noise_region = h_raw[delay_keep:, :]
        else:
            noise_region = h_raw
        noise_power = float(np.median(np.abs(noise_region) ** 2)) if noise_region.size else 0.0
        peak_power = float(np.max(cand_power)) if cand_power.size else 0.0

        if not np.isfinite(peak_power) or peak_power <= 1e-14:
            return out

        # Robust sparse support detection.  The relative threshold keeps weak-but-real
        # secondary taps at high SNR, while the noise threshold prevents the LMMSE
        # equalizer from amplifying pure training noise at low SNR.
        threshold = max(6.0 * noise_power, 0.012 * peak_power)
        support = cand_power >= threshold

        # Ensure that at least a few strongest taps are preserved.  This prevents an
        # over-aggressive threshold from collapsing the channel estimate when the
        # training frame is noisy or the Doppler energy is spread across bins.
        max_taps = int(min(max(6, 2 * (self.max_delay_samp + 1)), candidate.size))
        min_taps = int(min(max(3, self.max_delay_samp + 1), max_taps))
        flat_power = cand_power.reshape(-1)
        if flat_power.size > 0:
            top_idx = np.argpartition(flat_power, -max_taps)[-max_taps:]
            top_idx = top_idx[np.argsort(flat_power[top_idx])[::-1]]
            if np.count_nonzero(support) < min_taps:
                support.reshape(-1)[top_idx[:min_taps]] = True
            else:
                # Do not allow an unrealistically dense kernel; keep the strongest
                # selected taps within the physically plausible delay support.
                selected = np.flatnonzero(support.reshape(-1))
                if selected.size > max_taps:
                    keep = selected[np.argsort(flat_power[selected])[-max_taps:]]
                    new_support = np.zeros_like(support.reshape(-1), dtype=bool)
                    new_support[keep] = True
                    support = new_support.reshape(support.shape)

        # Wiener-style tap shrinkage based on the estimated training noise floor.
        shrink = np.maximum(0.0, 1.0 - noise_power / (cand_power + 1e-12))
        candidate = candidate * shrink
        out[:delay_keep, :][support] = candidate[support]

        return out.astype(np.complex128)

    def _estimate_tf_channel_from_training(self, y_train_tf: np.ndarray) -> np.ndarray:
        """Estimate a practical per-TF-bin channel response from the known training frame."""
        train_tf = np.asarray(self._train_tf, dtype=np.complex128)
        y_train_tf = np.asarray(y_train_tf, dtype=np.complex128)
        denom = np.abs(train_tf) ** 2 + 1e-6
        h_tf = y_train_tf * np.conj(train_tf) / denom

        # Very small 2D smoothing in TF improves robustness under noisy training.
        # It does not use any true channel state and is therefore engineering-feasible.
        h_pad = np.pad(h_tf, ((1, 1), (1, 1)), mode="wrap")
        h_sm = (
            4.0 * h_pad[1:-1, 1:-1]
            + h_pad[:-2, 1:-1] + h_pad[2:, 1:-1]
            + h_pad[1:-1, :-2] + h_pad[1:-1, 2:]
        ) / 8.0
        return h_sm.astype(np.complex128)

    def _tf_mmse_equalize(self, y_data_tf: np.ndarray, h_tf_est: np.ndarray, noise_var: float) -> np.ndarray:
        """Low-complexity training-based TF one-tap MMSE equalizer."""
        h = np.asarray(h_tf_est, dtype=np.complex128)
        y = np.asarray(y_data_tf, dtype=np.complex128)
        noise = max(float(noise_var), 1e-10)
        # Per-bin MMSE.  The factor 0.35 avoids over-regularising because noise_var
        # is measured on the full time waveform while TF bins are unitary transformed.
        x = np.conj(h) * y / (np.abs(h) ** 2 + 0.35 * noise + 1e-8)
        return x.astype(np.complex128)

    def _oracle_tf_lmmse_detect(self, y_data_tf: np.ndarray, delays, gains, fd_list, noise_var: float, data_start_sample: int) -> np.ndarray:
        # Debug/theoretical upper-bound helper retained for compatibility.
        # The practical OTFS receiver path above does not call this function.
        F = np.fft.fft(np.eye(self.M), axis=0) / np.sqrt(self.M)
        Fh = F.conj().T
        x_hat_tf = np.zeros_like(y_data_tf, dtype=np.complex128)
        eye = np.eye(self.M, dtype=np.complex128)
        for n in range(y_data_tf.shape[1]):
            useful_start = data_start_sample + n * self.slot_len + self.cp_len
            Ht = self._build_cyclic_time_channel_matrix(delays, gains, fd_list, useful_start)
            Hf = F @ Ht @ Fh
            A = Hf.conj().T @ Hf + (float(noise_var) + 1e-8) * eye
            b = Hf.conj().T @ y_data_tf[:, n]
            x_hat_tf[:, n] = np.linalg.solve(A, b)
        return x_hat_tf.astype(np.complex128)

    def _build_cyclic_time_channel_matrix(self, delays, gains, fd_list, sample_offset: int) -> np.ndarray:
        n = np.arange(self.M, dtype=np.float64)
        n_total = n + float(sample_offset)
        H = np.zeros((self.M, self.M), dtype=np.complex128)
        for l, h, fd in zip(np.asarray(delays, dtype=np.int64), gains, np.asarray(fd_list, dtype=np.float64)):
            l_int = int(max(0, l)) % self.M
            phase = np.exp(1j * 2.0 * np.pi * float(fd) * n_total / max(self.sample_rate, 1e-12))
            D = np.diag(phase)
            S = np.zeros((self.M, self.M), dtype=np.complex128)
            for row in range(self.M):
                col = (row - l_int) % self.M
                S[row, col] = 1.0
            H += complex(h) * D @ S
        return H

    def _build_oracle_dd_kernel(self, delays, gains, fd_list, sample_offset: int, cfo_hz: float = 0.0) -> np.ndarray:
        rx = _apply_time_varying_channel(
            self._train_time,
            delays,
            gains,
            fd_list,
            sample_rate=self.sample_rate,
            cfo_hz=cfo_hz,
            sample_offset=sample_offset,
        )
        rx_slots = self._split_slots(rx)
        y_tf = self._time_slots_to_tf(rx_slots)
        y_dd = self._tf_to_dd(y_tf)
        h = y_dd / max(self._train_pilot_amp, 1e-12)
        return self._smooth_small_2d(h)

    def _dd_lmmse_equalize(self, y_dd: np.ndarray, h_dd: np.ndarray, noise_var: float) -> np.ndarray:
        H = np.fft.fft2(h_dd)
        Y = np.fft.fft2(y_dd)
        X = np.conj(H) * Y / (np.abs(H) ** 2 + float(noise_var) + 1e-8)
        x_hat = np.fft.ifft2(X)
        return x_hat.astype(np.complex128)

    def _apply_dd_pilot_correction(self, x_dd: np.ndarray) -> np.ndarray:
        y = x_dd.copy().astype(np.complex128)
        x_all = self._row_axis_norm(np.arange(self.M))
        x_p = self._row_axis_norm(self.pilot_rows)
        A = np.column_stack([np.ones(len(self.pilot_rows), dtype=np.float64), x_p])

        for n in self.data_cols:
            ref = self._pilot_dd_grid[self.pilot_rows, n]
            rx = y[self.pilot_rows, n]

            alpha = np.vdot(ref, rx) / (np.vdot(ref, ref) + 1e-12)
            if np.abs(alpha) > 1e-6:
                y[:, n] /= alpha

            rx = y[self.pilot_rows, n]
            ph = np.unwrap(np.angle(rx * np.conj(ref) + 1e-12))
            try:
                coeff, *_ = np.linalg.lstsq(A, ph, rcond=None)
                phi0 = float(np.clip(coeff[0], -0.45, 0.45)) * 0.35
                phi1 = float(np.clip(coeff[1], -0.80, 0.80)) * 0.30
            except np.linalg.LinAlgError:
                phi0 = 0.0
                phi1 = 0.0
            y[:, n] *= np.exp(-1j * (phi0 + phi1 * x_all))
        return y

    # ---------------- frame construction ----------------
    def _build_data_dd_frame(self, mod_order: str) -> Tuple[np.ndarray, np.ndarray]:
        # 48 个数据 delay 行 * 7 个数据 DD 列 = 336 个净调制符号，
        # 与 OFDM/AFDM 的净数据资源保持一致。第 0 列保留为空/控制列，
        # 避免把训练脉冲和净荷数据混在同一 DD 支撑上。
        n_data = len(self.data_rows) * len(self.data_cols)
        n_bits = n_data * self.bits_per_symbol
        tx_bits = self._rng.integers(0, 2, size=n_bits, dtype=np.int8)
        data_syms = self._qam_modulate(tx_bits, mod_order).reshape(len(self.data_rows), len(self.data_cols))

        x_dd = np.zeros((self.M, self.N), dtype=np.complex128)
        x_dd[self.pilot_rows[:, None], self.data_cols[None, :]] = self._pilot_dd_grid[self.pilot_rows[:, None], self.data_cols[None, :]]
        x_dd[self.data_rows[:, None], self.data_cols[None, :]] = data_syms
        return x_dd, tx_bits

    def _build_pilot_dd_grid(self) -> np.ndarray:
        grid = np.zeros((self.M, self.N), dtype=np.complex128)
        base = np.array([1.0, 1.0, 1.0, -1.0], dtype=np.float64)
        polarity = np.array([1, 1, 1, -1, 1, -1, 1, 1], dtype=np.float64)
        for n in range(self.N):
            grid[self.pilot_rows, n] = polarity[n % len(polarity)] * base
        return grid

    def _dd_to_tf(self, x_dd: np.ndarray) -> np.ndarray:
        # 与现有版本保持一致的有限尺寸变换
        return np.fft.ifft(np.fft.fft(x_dd, axis=1), axis=0)

    def _tf_to_dd(self, x_tf: np.ndarray) -> np.ndarray:
        return np.fft.ifft(np.fft.fft(x_tf, axis=0), axis=1)

    def _tf_to_time_cp(self, x_tf: np.ndarray) -> np.ndarray:
        out = []
        for n in range(x_tf.shape[1]):
            td = np.fft.ifft(x_tf[:, n], axis=0) * np.sqrt(self.M)
            cp = td[-self.cp_len :]
            out.append(np.concatenate([cp, td]))
        return np.concatenate(out).astype(np.complex128)

    # ---------------- helpers ----------------
    def _update_bad_frame(self, rx_stream: np.ndarray):
        recent_samples = rx_stream[-min(4096, len(rx_stream)):].astype(np.complex64)
        with self._lock:
            self._latest_samples = recent_samples
            self._latest_constellation = self._last_good_constellation.copy()
            self._bad_frame_hold = min(self._bad_frame_hold + 1, 10)

    def _next_channel_state(self, max_delay_samp: int, doppler_spread_hz: float):
        """Return a persistent channel realization for the current run.

        The three backends now use the same default channel policy: a sparse
        realization is created when the run starts or when channel parameters
        change, and Doppler is applied continuously by the absolute sample
        offset in _apply_time_varying_channel().  This keeps BER-time plots from
        being dominated by artificial per-frame fading refreshes.
        """
        max_delay_samp = int(max(0, max_delay_samp))
        doppler_spread_hz = float(max(0.0, doppler_spread_hz))
        if (
            self._channel_state is None
            or int(self._channel_state.get("max_delay_samp", -1)) != max_delay_samp
            or abs(float(self._channel_state.get("doppler_spread_hz", -1.0)) - doppler_spread_hz) > 1e-9
        ):
            delays, gains, fd_list, amp = _init_sparse_channel(
                self._rng,
                max_delay_samp=max_delay_samp,
                doppler_spread_hz=doppler_spread_hz,
                path_gains_db=self._path_gains_db,
                delay_scales=self._path_delay_scale,
                doppler_scales=self._path_doppler_scale,
            )
            self._channel_state = {
                "max_delay_samp": max_delay_samp,
                "doppler_spread_hz": doppler_spread_hz,
                "delays": delays,
                "gains": gains,
                "fd_list": fd_list,
                "amp": amp,
            }
        return (
            self._channel_state["delays"],
            self._channel_state["gains"],
            self._channel_state["fd_list"],
        )

    def _prepare_constellation_points(self, rx_data_syms: np.ndarray) -> np.ndarray:
        if rx_data_syms is None or len(rx_data_syms) == 0:
            return np.zeros(256, dtype=np.complex64)
        n = min(len(rx_data_syms), 256)
        out = np.zeros(256, dtype=np.complex64)
        out[:n] = rx_data_syms[:n]
        return out

    def _schmidl_cox_metric(self, rx: np.ndarray, L: int) -> np.ndarray:
        n_max = len(rx) - 2 * L
        if n_max <= 1:
            return np.zeros(1, dtype=np.float64)
        metric = np.zeros(n_max, dtype=np.float64)
        for d in range(n_max):
            a = rx[d : d + L]
            b = rx[d + L : d + 2 * L]
            P = np.sum(a * np.conj(b))
            R = np.sum(np.abs(b) ** 2) + 1e-12
            metric[d] = (np.abs(P) ** 2) / (R ** 2)
        return metric

    def _estimate_cfo_from_preamble(self, rx: np.ndarray, frame_start: int, L: int) -> float:
        if frame_start + 2 * L > len(rx):
            return 0.0
        a = rx[frame_start : frame_start + L]
        b = rx[frame_start + L : frame_start + 2 * L]
        P = np.sum(a * np.conj(b))
        phase = np.angle(P)
        return float(-phase * self.sample_rate / (2.0 * np.pi * L))

    @staticmethod
    def _build_resource_plan(n_subcarriers: int):
        if n_subcarriers != 64:
            raise ValueError("This UI-compatible OTFS version is defined for 64 subcarriers")
        active_bins = np.concatenate([np.arange(-26, 0), np.arange(1, 27)])
        pilot_bins = np.array([-21, -7, 7, 21])
        active_rows = active_bins + n_subcarriers // 2
        pilot_rows = pilot_bins + n_subcarriers // 2
        data_rows = np.asarray([x for x in active_rows.tolist() if x not in set(pilot_rows.tolist())], dtype=np.int64)
        return active_rows.astype(np.int64), pilot_rows.astype(np.int64), data_rows

    @staticmethod
    def _row_axis_norm(rows: np.ndarray) -> np.ndarray:
        center = 0.5 * (64 - 1)
        scale = max(center, 1.0)
        return (np.asarray(rows, dtype=np.float64) - center) / scale

    def _dd_pilot_error(self, x_dd: np.ndarray) -> float:
        ref = self._pilot_dd_grid[self.pilot_rows[:, None], self.data_cols[None, :]]
        rx = x_dd[self.pilot_rows[:, None], self.data_cols[None, :]]
        return float(np.sqrt(np.mean(np.abs(rx - ref) ** 2) / (np.mean(np.abs(ref) ** 2) + 1e-12)))

    @staticmethod
    def _smooth_small_2d(h: np.ndarray) -> np.ndarray:
        fk = np.array([0.2, 0.6, 0.2], dtype=np.float64)
        tk = np.array([0.25, 0.5, 0.25], dtype=np.float64)
        tmp = np.zeros_like(h)
        for n in range(h.shape[1]):
            col = np.pad(h[:, n], (1, 1), mode="wrap")
            tmp[:, n] = np.convolve(np.real(col), fk, mode="valid") + 1j * np.convolve(np.imag(col), fk, mode="valid")
        out = np.zeros_like(tmp)
        for m in range(tmp.shape[0]):
            row = np.pad(tmp[m, :], (1, 1), mode="wrap")
            out[m, :] = np.convolve(np.real(row), tk, mode="valid") + 1j * np.convolve(np.imag(row), tk, mode="valid")
        return out

    @staticmethod
    def _get_bits_per_symbol(mod_order: str) -> int:
        if mod_order == "QPSK":
            return 2
        if mod_order == "16QAM":
            return 4
        if mod_order == "64QAM":
            return 6
        raise ValueError(f"Unsupported modulation: {mod_order}")

    def _qam_modulate(self, bits: np.ndarray, mod_order: str) -> np.ndarray:
        if mod_order == "QPSK":
            return self._mod_qpsk(bits)
        if mod_order == "16QAM":
            return self._mod_16qam(bits)
        if mod_order == "64QAM":
            return self._mod_64qam(bits)
        raise ValueError(f"Unsupported modulation: {mod_order}")

    def _qam_demodulate(self, syms: np.ndarray, mod_order: str) -> np.ndarray:
        if mod_order == "QPSK":
            return self._demod_qpsk(syms)
        if mod_order == "16QAM":
            return self._demod_16qam(syms)
        if mod_order == "64QAM":
            return self._demod_64qam(syms)
        raise ValueError(f"Unsupported modulation: {mod_order}")

    def _mod_qpsk(self, bits: np.ndarray) -> np.ndarray:
        bits = bits.reshape(-1, 2)
        i = 1 - 2 * bits[:, 1]
        q = 1 - 2 * bits[:, 0]
        return ((i + 1j * q) / np.sqrt(2)).astype(np.complex128)

    def _demod_qpsk(self, syms: np.ndarray) -> np.ndarray:
        bits = np.zeros((len(syms), 2), dtype=np.int8)
        bits[:, 0] = (np.imag(syms) < 0).astype(np.int8)
        bits[:, 1] = (np.real(syms) < 0).astype(np.int8)
        return bits.reshape(-1)

    def _mod_16qam(self, bits: np.ndarray) -> np.ndarray:
        bits = bits.reshape(-1, 4)
        def map_2b(b0, b1):
            if b0 == 0 and b1 == 0:
                return 3
            if b0 == 0 and b1 == 1:
                return 1
            if b0 == 1 and b1 == 1:
                return -1
            return -3
        i = np.array([map_2b(b[0], b[1]) for b in bits], dtype=np.float64)
        q = np.array([map_2b(b[2], b[3]) for b in bits], dtype=np.float64)
        return ((i + 1j * q) / np.sqrt(10)).astype(np.complex128)

    def _demod_16qam(self, syms: np.ndarray) -> np.ndarray:
        x = np.real(syms) * np.sqrt(10)
        y = np.imag(syms) * np.sqrt(10)
        def slicer(v):
            if v >= 2:
                return (0, 0)
            if v >= 0:
                return (0, 1)
            if v >= -2:
                return (1, 1)
            return (1, 0)
        out = np.zeros((len(syms), 4), dtype=np.int8)
        for k, (i_v, q_v) in enumerate(zip(x, y)):
            out[k, 0], out[k, 1] = slicer(i_v)
            out[k, 2], out[k, 3] = slicer(q_v)
        return out.reshape(-1)

    def _mod_64qam(self, bits: np.ndarray) -> np.ndarray:
        bits = bits.reshape(-1, 6)
        table = {
            (0, 0, 0):  7,
            (0, 0, 1):  5,
            (0, 1, 1):  3,
            (0, 1, 0):  1,
            (1, 1, 0): -1,
            (1, 1, 1): -3,
            (1, 0, 1): -5,
            (1, 0, 0): -7,
        }
        i = np.array([table[tuple(b[:3].tolist())] for b in bits], dtype=np.float64)
        q = np.array([table[tuple(b[3:].tolist())] for b in bits], dtype=np.float64)
        return ((i + 1j * q) / np.sqrt(42)).astype(np.complex128)

    def _demod_64qam(self, syms: np.ndarray) -> np.ndarray:
        x = np.real(syms) * np.sqrt(42)
        y = np.imag(syms) * np.sqrt(42)
        def slicer(v):
            if v >= 6:
                return (0, 0, 0)
            if v >= 4:
                return (0, 0, 1)
            if v >= 2:
                return (0, 1, 1)
            if v >= 0:
                return (0, 1, 0)
            if v >= -2:
                return (1, 1, 0)
            if v >= -4:
                return (1, 1, 1)
            if v >= -6:
                return (1, 0, 1)
            return (1, 0, 0)
        out = np.zeros((len(syms), 6), dtype=np.int8)
        for k, (i_v, q_v) in enumerate(zip(x, y)):
            out[k, 0], out[k, 1], out[k, 2] = slicer(i_v)
            out[k, 3], out[k, 4], out[k, 5] = slicer(q_v)
        return out.reshape(-1)


def dump_status(tb, idx=None):
    snap = tb.get_debug_snapshot()
    prefix = f"[{idx:03d}] " if idx is not None else ""
    print(
        prefix
        + f"frame_ok={snap['frame_ok']} "
        + f"reason={snap['reason']} "
        + f"sync_idx={snap['sync_idx']} "
        + f"payload_start={snap['payload_start']} "
        + f"sync_metric={snap['sync_metric']:.4f} "
        + f"cfo_est={snap['cfo_est_hz']:.2f}Hz "
        + f"ber={snap['ber']:.4e} "
        + f"kernel_energy={snap['kernel_energy']:.3e} "
        + f"kernel_rank={snap['kernel_rank']} "
        + f"pilot_nmse={snap['pilot_nmse']:.3f}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# 阶段4：统一引擎兼容壳
# ---------------------------------------------------------------------------
def _create_otfs_backend(**kwargs):
    """供 waveform_sim.core.engine 构造 OTFS 后端。"""
    return _LegacyOTFSTransceiver(**kwargs)


class OTFSTransceiver(LinkSimulator):
    """OTFS 兼容壳：继承统一引擎，委托 _LegacyOTFSTransceiver，公开接口不变。"""

    def __init__(self, **kwargs):
        super().__init__(waveform="OTFS", **kwargs)

    def __getattr__(self, name):
        backend = self.__dict__.get("_backend")
        if backend is not None and hasattr(backend, name):
            return getattr(backend, name)
        raise AttributeError(name)


if __name__ == "__main__":
    tb = OTFSTransceiver(
        delay_spread=5,
        doppler_spread=600,
        snr_db=16,
        mod_order="QPSK",
        cfo_hz=0,
        equalizer="MMSE",
        update_period=0.08,
    )

    tb.start()
    print("OTFS monitor started. Press Ctrl+C to stop.", flush=True)
    k = 0
    try:
        while True:
            time.sleep(0.20)
            dump_status(tb, k)
            k += 1
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping monitor...", flush=True)
    finally:
        tb.stop()
        tb.wait()
        print("OTFS monitor stopped", flush=True)
