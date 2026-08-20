import threading
import time
from collections import deque
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


def _generate_sparse_channel_realization(
    rng,
    max_delay_samp,
    doppler_spread_hz,
    path_gains_db,
    delay_scales,
    doppler_scales,
):
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

    gains_lin = 10.0 ** (gains_db / 20.0)
    fading = (
        rng.standard_normal(len(gains_lin)) + 1j * rng.standard_normal(len(gains_lin))
    ) / np.sqrt(2.0)

    gains = gains_lin * fading
    power = float(np.sum(np.abs(gains) ** 2))
    if power > 1e-12:
        gains = gains / np.sqrt(power)

    fd_list = float(max(0.0, doppler_spread_hz)) * f_scales
    return (
        delays.astype(np.int64, copy=False),
        gains.astype(np.complex128, copy=False),
        fd_list.astype(np.float64, copy=False),
    )


def _init_sparse_channel_state(
    rng,
    max_delay_samp,
    doppler_spread_hz,
    path_gains_db,
    delay_scales,
    doppler_scales,
):
    delays, gains, fd_list = _generate_sparse_channel_realization(
        rng,
        max_delay_samp=max_delay_samp,
        doppler_spread_hz=doppler_spread_hz,
        path_gains_db=path_gains_db,
        delay_scales=delay_scales,
        doppler_scales=doppler_scales,
    )

    gains_db = _as_1d_float_array(path_gains_db, [0.0, -3.5, -7.0])
    d_scales = _as_1d_float_array(delay_scales, [0.0, 0.4, 1.0])
    f_scales = _as_1d_float_array(doppler_scales, [0.0, 0.6, -0.8])
    n_paths = min(gains_db.size, d_scales.size, f_scales.size)
    amp = 10.0 ** (gains_db[:n_paths] / 20.0)
    amp = amp / np.sqrt(np.sum(amp ** 2) + 1e-12)

    return {
        "max_delay_samp": int(max(0, max_delay_samp)),
        "doppler_spread_hz": float(max(0.0, doppler_spread_hz)),
        "delays": delays.astype(np.int64, copy=False),
        "gains": gains.astype(np.complex128, copy=False),
        "fd_list": fd_list.astype(np.float64, copy=False),
        "amp": amp.astype(np.float64, copy=False),
    }


def _evolve_sparse_channel_state(rng, state, rho):
    rho = float(np.clip(rho, 0.0, 0.999999))
    amp = np.asarray(state["amp"], dtype=np.float64)
    prev = np.asarray(state["gains"], dtype=np.complex128)
    innovation = amp * (rng.standard_normal(len(amp)) + 1j * rng.standard_normal(len(amp))) / np.sqrt(2.0)
    gains = rho * prev + np.sqrt(max(1.0 - rho ** 2, 1e-12)) * innovation
    power = float(np.sum(np.abs(gains) ** 2))
    if power > 1e-12:
        gains = gains / np.sqrt(power)
    state["gains"] = gains.astype(np.complex128, copy=False)
    return state["delays"], state["gains"], state["fd_list"]


def _apply_time_varying_channel(
    x,
    delays,
    gains,
    fd_list,
    sample_rate,
    cfo_hz=0.0,
    sample_offset=0,
):
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


def _add_awgn_by_ebn0(signal, ebn0_db, n_info_bits, rng):
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
    """Small EVM history tracker for the real-time OFDM display.

    EVM is a modulation-quality metric computed from the equalized data
    symbols and the known transmitted symbols in the simulator.  Unlike BER,
    it is meaningful with a small number of symbols and is therefore better
    suited for a live time-domain display.  Invalid frames are not inserted
    into the EVM history, because EVM is undefined when synchronization fails.
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


class _LegacyOfdmTransceiver(threading.Thread):
    """
    公平性修正版 OFDM 收发链路仿真器。

    设计原则：
    1. 与 OTFS / AFDM 对齐为统一外层帧：前保护 + 重复半段同步前导 + 8 个 payload 单元；
    2. 统一按每帧净信息比特数定义噪声，外部 snr_db 参数按 Eb/N0 理解；
    3. 统一数据资源：7 个数据单元，每单元 48 个数据符号，共 336 个调制符号；
    4. 统一采用：1 个训练单元 + 7 个数据单元 + 4 个跟踪导频。
    """

    _MOD_ORDERS = {
        "QPSK": 4,
        "16QAM": 16,
        "64QAM": 64,
    }

    def __init__(
        self,
        fft_len: int = 64,
        cp_len: int = 16,
        snr_db: float = 15.0,
        doppler_spread_hz: float = 20.0,
        delay_spread: int = 5,
        cfo_hz: float = 0.0,
        mod_order: str = "QPSK",
        payload_symbols: int = 8,
        doppler_hz: Optional[float] = None,   # 兼容旧接口
    ):
        super().__init__(daemon=True)

        if doppler_hz is not None:
            doppler_spread_hz = doppler_hz

        self.fft_len = int(fft_len)
        self.cp_len = int(cp_len)
        self.snr_db = float(snr_db)  # 兼容旧 UI，内部按 Eb/N0 解释
        self.max_delay_samp = max(0, int(delay_spread))
        self.delay_spread = self.max_delay_samp
        self.doppler_spread_hz = max(0.0, float(doppler_spread_hz))
        self.cfo_hz = float(cfo_hz)
        self.mod_order = str(mod_order).upper()
        self.payload_symbols = int(payload_symbols)

        if self.mod_order not in self._MOD_ORDERS:
            raise ValueError(f"Unsupported modulation: {mod_order}")
        if self.cp_len <= 0 or self.cp_len >= self.fft_len:
            raise ValueError("cp_len must satisfy 0 < cp_len < fft_len")
        if self.payload_symbols != 8:
            raise ValueError("This fairness-aligned version expects payload_symbols == 8")

        self.subcarrier_spacing = 15e3
        self.samp_rate = float(self.fft_len * self.subcarrier_spacing)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._rng = np.random.default_rng()

        self._sample_counter = 0
        self._frame_counter = 0

        self._active_shift_idx, self._pilot_shift_idx, self._data_shift_idx = self._build_subcarrier_plan(
            self.fft_len
        )
        self._active_shift_idx = np.asarray(self._active_shift_idx, dtype=np.int64)
        self._pilot_shift_idx = np.asarray(self._pilot_shift_idx, dtype=np.int64)
        self._data_shift_idx = np.asarray(self._data_shift_idx, dtype=np.int64)

        self._pilot_active_pos = self._positions_in_parent(self._pilot_shift_idx, self._active_shift_idx)
        self._data_active_pos = self._positions_in_parent(self._data_shift_idx, self._active_shift_idx)

        self._n_active = int(len(self._active_shift_idx))
        self._n_pilots = int(len(self._pilot_shift_idx))
        self._n_data = int(len(self._data_shift_idx))

        self._qam_order = self._MOD_ORDERS[self.mod_order]
        self._bits_per_symbol = int(np.log2(self._qam_order))
        self._constellation, self._bit_patterns = self._build_gray_qam(self._qam_order)

        self.pre_guard_len = 16
        self.sync_half_len = self.fft_len
        self.sync_preamble = self._build_sync_preamble(self.sync_half_len)
        self.sync_len = len(self.sync_preamble)
        self._payload_symbol_len = self.cp_len + self.fft_len
        self._train_payload_symbols = 1
        self._data_payload_symbols = 7
        self._train_fd_active = self._build_training_active()
        self._frame_template_len = self.pre_guard_len + self.sync_len + self.payload_symbols * self._payload_symbol_len
        self.sync_metric_threshold = 0.35

        self._path_gains_db = np.array([0.0, -3.5, -7.0], dtype=np.float64)
        self._path_delay_scale = np.array([0.0, 0.4, 1.0], dtype=np.float64)
        self._path_doppler_scale = np.array([0.0, 0.6, -0.8], dtype=np.float64)

        self._spectrum_buffer = np.zeros(32768, dtype=np.complex64)
        self._pre_eq_constellation = np.zeros(0, dtype=np.complex64)
        self._post_eq_constellation = np.zeros(0, dtype=np.complex64)

        self._channel_state = None
        self._metric_tracker = _MetricTracker(history_len=240, ewma_alpha=0.08)
        self._evm_tracker = _EvmTracker(history_len=240)
        self._t0 = self._metric_tracker.start_time
        self._last_good_pre_eq_constellation = np.zeros(0, dtype=np.complex64)
        self._last_good_post_eq_constellation = np.zeros(0, dtype=np.complex64)
        self._last_metrics = {
            "ebn0_db": float(self.snr_db),
            "coarse_cfo_hz": 0.0,
            "total_cfo_hz": 0.0,
            "sync_metric": 0.0,
            "bit_errors": 0,
            "total_bits": 0,
            "ber": 0.5,
            "fer": 1.0,
            "evm_percent": float("nan"),
            "evm_db": float("nan"),
            "evm_valid": False,
        }

    def stop(self):
        self._stop_event.set()

    def wait(self, timeout: Optional[float] = None):
        self.join(timeout=timeout)

    def run(self):
        while not self._stop_event.is_set():
            self._simulate_one_frame()
            time.sleep(0.1)

    def step(self):
        """单帧仿真（供统一引擎调用）。"""
        self._simulate_one_frame()

    def set_snr_db(self, value: float):
        with self._lock:
            self.snr_db = float(value)
            self._reset_stats_locked()

    def set_doppler_spread_hz(self, value: float):
        with self._lock:
            self.doppler_spread_hz = max(0.0, float(value))
            self._channel_state = None
            self._reset_stats_locked()

    def set_delay_spread(self, value: int):
        with self._lock:
            self.max_delay_samp = max(0, int(value))
            self.delay_spread = self.max_delay_samp
            self._channel_state = None
            self._reset_stats_locked()

    def set_doppler_hz(self, value: float):
        # 兼容旧接口
        self.set_doppler_spread_hz(value)

    def set_cfo_hz(self, value: float):
        with self._lock:
            self.cfo_hz = float(value)
            self._reset_stats_locked()

    def get_spectrum(self, num_samples: int = 8192):
        with self._lock:
            n = max(1, int(num_samples))
            return self._spectrum_buffer[-n:].copy()

    def get_constellation(self):
        with self._lock:
            return self._post_eq_constellation.copy()

    def get_pre_eq_constellation(self):
        with self._lock:
            return self._pre_eq_constellation.copy()

    def get_fer_estimate(self):
        with self._lock:
            return self._metric_tracker.get_fer_estimate()

    def get_ber_estimate(self):
        with self._lock:
            return self._metric_tracker.get_ber_estimate()

    def get_estimated_ber(self):
        with self._lock:
            return self._metric_tracker.get_history()

    def get_evm_history(self, unit: str = "percent"):
        with self._lock:
            return self._evm_tracker.get_history(unit=unit)

    def get_evm_estimate(self, unit: str = "percent"):
        with self._lock:
            return self._evm_tracker.get_estimate(unit=unit)

    def get_estimated_fer(self):
        with self._lock:
            return self._metric_tracker.get_fer_history()

    def get_last_metrics(self):
        with self._lock:
            return dict(self._last_metrics)

    def _simulate_one_frame(self):
        with self._lock:
            ebn0_db = float(self.snr_db)
            max_delay_samp = int(self.max_delay_samp)
            doppler_spread_hz = float(self.doppler_spread_hz)
            cfo_hz = float(self.cfo_hz)

        sample_offset = self._sample_counter

        tx_bits = self._rng.integers(
            0,
            2,
            size=self._data_payload_symbols * self._n_data * self._bits_per_symbol,
            dtype=np.uint8,
        )
        tx_groups = tx_bits.reshape(-1, self._bits_per_symbol)
        tx_data_symbols = self._map_bits_to_symbols(tx_groups).reshape(self._data_payload_symbols, self._n_data)

        tx_frame = self._build_tx_frame(tx_data_symbols)
        delays, gains, fd_list = self._next_channel_state(max_delay_samp, doppler_spread_hz)
        rx_chan = _apply_time_varying_channel(
            tx_frame,
            delays,
            gains,
            fd_list,
            sample_rate=self.samp_rate,
            cfo_hz=cfo_hz,
            sample_offset=sample_offset,
        )
        self._sample_counter += len(tx_frame)
        rx_noisy, noise_var = _add_awgn_by_ebn0(rx_chan, ebn0_db, int(tx_bits.size), self._rng)

        payload_len = self.payload_symbols * self._payload_symbol_len
        sync_ok, rx_payload, sync_info = self._synchronize_and_extract(rx_noisy, payload_len)

        pre_eq_points = []
        post_eq_points = []
        rx_bit_groups = []

        if sync_ok:
            useful0 = rx_payload[self.cp_len: self.cp_len + self.fft_len]
            y0_shift = np.fft.fftshift(np.fft.fft(useful0) / np.sqrt(self.fft_len))
            y0_active = y0_shift[self._active_shift_idx]
            h_est_active = y0_active / np.where(np.abs(self._train_fd_active) < 1e-8, 1e-8 + 0j, self._train_fd_active)
            h_est_active = self._smooth_frequency_response(h_est_active)

            for data_sym_idx in range(self._data_payload_symbols):
                payload_sym_idx = data_sym_idx + 1
                sym_start = payload_sym_idx * self._payload_symbol_len
                useful = rx_payload[sym_start + self.cp_len: sym_start + self.cp_len + self.fft_len]
                y_shift = np.fft.fftshift(np.fft.fft(useful) / np.sqrt(self.fft_len))

                y_active = y_shift[self._active_shift_idx]
                y_data_raw = y_shift[self._data_shift_idx]
                w_active = np.conj(h_est_active) / (np.abs(h_est_active) ** 2 + noise_var + 1e-8)
                eq_active = y_active * w_active

                pilot_expected = self._pilot_values(data_sym_idx)
                pilot_rx = eq_active[self._pilot_active_pos]
                cpe = np.angle(np.vdot(pilot_expected, pilot_rx) + 1e-12)

                eq_active_cpe = eq_active * np.exp(-1j * cpe)
                eq_data = eq_active_cpe[self._data_active_pos]

                rx_groups = self._demap_symbols_to_bits(eq_data)
                rx_bit_groups.append(rx_groups)
                pre_eq_points.append(y_data_raw)
                post_eq_points.append(eq_data)

        if rx_bit_groups:
            rx_bit_groups = np.vstack(rx_bit_groups)
            rx_bits = rx_bit_groups.reshape(-1)
        else:
            rx_bits = np.zeros(0, dtype=np.uint8)

        total_bits = int(tx_bits.size)
        bit_errors = int(np.count_nonzero(tx_bits != rx_bits[:total_bits])) if total_bits > 0 and rx_bits.size >= total_bits else total_bits

        bad_frame = bool((not sync_ok) or (not np.isfinite(sync_info["sync_metric"])) or (sync_info["sync_metric"] < self.sync_metric_threshold))
        if bad_frame:
            bit_errors = total_bits

        frame_error = 1.0 if bit_errors > 0 else 0.0
        ber_frame = bit_errors / max(total_bits, 1)

        all_pre_eq = np.concatenate(pre_eq_points) if pre_eq_points else np.zeros(0, dtype=np.complex128)
        all_post_eq = np.concatenate(post_eq_points) if post_eq_points else np.zeros(0, dtype=np.complex128)
        pre_eq_points_ds = self._downsample_points(all_pre_eq, 320)
        post_eq_points_ds = self._downsample_points(all_post_eq, 320)

        tx_ref_symbols = tx_data_symbols.reshape(-1)
        evm_percent, evm_db, evm_valid = self._compute_evm_metrics(
            all_post_eq,
            tx_ref_symbols,
            valid=(not bad_frame),
        )
        t_now = time.time() - self._t0

        # Spectrum display should show the transmitted OFDM payload, not the
        # post-channel received waveform. The received waveform contains
        # multipath/CFO fading and the sync preamble, so its PSD can have
        # deep notches and a wavy envelope that looks unlike an OFDM transmit
        # spectrum. Use only the OFDM payload section for the UI spectrum.
        tx_payload_for_spectrum = tx_frame[self.pre_guard_len + self.sync_len:]

        with self._lock:
            self._append_spectrum_samples(tx_payload_for_spectrum)

            if bad_frame:
                self._pre_eq_constellation = self._last_good_pre_eq_constellation.copy()
                self._post_eq_constellation = self._last_good_post_eq_constellation.copy()
            else:
                self._pre_eq_constellation = pre_eq_points_ds
                self._post_eq_constellation = post_eq_points_ds
                self._last_good_pre_eq_constellation = pre_eq_points_ds.copy()
                self._last_good_post_eq_constellation = post_eq_points_ds.copy()

            stats = self._metric_tracker.update(
                bit_errors=bit_errors,
                total_bits=total_bits,
                frame_error=frame_error,
                t_now=t_now,
            )
            evm_stats = self._evm_tracker.update(
                evm_percent=evm_percent,
                evm_db=evm_db,
                valid=evm_valid,
                t_now=t_now,
            )

            self._last_metrics = {
                "ebn0_db": float(ebn0_db),
                "coarse_cfo_hz": float(sync_info["cfo_est_hz"]) if sync_ok else 0.0,
                "total_cfo_hz": float(sync_info["cfo_est_hz"]) if sync_ok else 0.0,
                "sync_metric": float(sync_info["sync_metric"]),
                "bit_errors": int(bit_errors),
                "total_bits": int(total_bits),
                "frame_ber": float(ber_frame),
                "ber": float(stats["ber_window"]),
                "ber_ewma": float(stats["ber_ewma"]),
                "fer": float(stats["fer_window"]),
                "frame_error": float(frame_error),
                "evm_percent": float(evm_stats["evm_percent"]),
                "evm_db": float(evm_stats["evm_db"]),
                "evm_valid": bool(evm_valid),
                "net_data_symbols": int(self._data_payload_symbols * self._n_data),
            }

        self._frame_counter += 1

    def _build_tx_frame(self, payload_data_symbols: np.ndarray) -> np.ndarray:
        parts = [
            np.zeros(self.pre_guard_len, dtype=np.complex128),
            self.sync_preamble.astype(np.complex128),
        ]

        fd_shift = np.zeros(self.fft_len, dtype=np.complex128)
        fd_shift[self._active_shift_idx] = self._train_fd_active
        td = np.fft.ifft(np.fft.ifftshift(fd_shift)) * np.sqrt(self.fft_len)
        parts.append(td[-self.cp_len:])
        parts.append(td)

        for sym_idx in range(self._data_payload_symbols):
            fd_shift = np.zeros(self.fft_len, dtype=np.complex128)
            fd_shift[self._pilot_shift_idx] = self._pilot_values(sym_idx)
            fd_shift[self._data_shift_idx] = payload_data_symbols[sym_idx]
            td = np.fft.ifft(np.fft.ifftshift(fd_shift)) * np.sqrt(self.fft_len)
            parts.append(td[-self.cp_len:])
            parts.append(td)

        return np.concatenate(parts).astype(np.complex128, copy=False)

    def _build_sync_preamble(self, half_len: int) -> np.ndarray:
        bits = self._rng.integers(0, 2, size=2 * half_len, dtype=np.uint8)
        groups = bits.reshape(-1, 2)
        a = self._map_bits_to_symbols(groups)[:half_len]
        a = a / np.sqrt(np.mean(np.abs(a) ** 2) + 1e-12)
        preamble = np.concatenate([a, a])
        return preamble.astype(np.complex128)

    def _build_training_active(self) -> np.ndarray:
        train = np.ones(self._n_active, dtype=np.complex128)
        train[1::2] = -1.0 + 0.0j
        return train

    def _synchronize_and_extract(self, rx: np.ndarray, payload_len: int):
        L = self.sync_half_len
        if len(rx) < self.pre_guard_len + 2 * L + payload_len:
            return False, np.zeros(payload_len, dtype=np.complex128), {
                "frame_start": 0,
                "cfo_est_hz": 0.0,
                "sync_metric": 0.0,
            }

        metric = self._schmidl_cox_metric(rx, L)
        start_lo = max(0, self.pre_guard_len - 4)
        start_hi = min(len(metric), self.pre_guard_len + 5)
        frame_peak = int(start_lo + np.argmax(metric[start_lo:start_hi]))
        frame_start = int(self.pre_guard_len)
        sync_metric = float(metric[frame_peak])
        cfo_est_hz = self._estimate_cfo_from_preamble(rx, frame_peak, L)

        n = np.arange(len(rx), dtype=np.float64)
        rx_cfo_corrected = rx * np.exp(-1j * 2 * np.pi * cfo_est_hz * n / self.samp_rate)

        payload_start = frame_start + 2 * L
        payload_end = payload_start + payload_len
        if payload_end > len(rx_cfo_corrected):
            return False, np.zeros(payload_len, dtype=np.complex128), {
                "frame_start": frame_start,
                "cfo_est_hz": cfo_est_hz,
                "sync_metric": sync_metric,
            }

        return True, rx_cfo_corrected[payload_start:payload_end].astype(np.complex128), {
            "frame_start": frame_start,
            "cfo_est_hz": float(cfo_est_hz),
            "sync_metric": sync_metric,
        }

    def _schmidl_cox_metric(self, rx: np.ndarray, L: int) -> np.ndarray:
        n_max = len(rx) - 2 * L
        if n_max <= 1:
            return np.zeros(1, dtype=np.float64)

        metric = np.zeros(n_max, dtype=np.float64)
        for d in range(n_max):
            a = rx[d:d + L]
            b = rx[d + L:d + 2 * L]
            P = np.sum(a * np.conj(b))
            R = np.sum(np.abs(b) ** 2) + 1e-12
            metric[d] = (np.abs(P) ** 2) / (R ** 2)
        return metric

    def _estimate_cfo_from_preamble(self, rx: np.ndarray, frame_start: int, L: int) -> float:
        if frame_start + 2 * L > len(rx):
            return 0.0
        a = rx[frame_start:frame_start + L]
        b = rx[frame_start + L:frame_start + 2 * L]
        P = np.sum(a * np.conj(b))
        phase = np.angle(P)
        return float(-phase * self.samp_rate / (2 * np.pi * L))

    @staticmethod
    def _smooth_frequency_response(h_est: np.ndarray) -> np.ndarray:
        if h_est.size < 3:
            return h_est.copy()
        kernel = np.array([0.2, 0.6, 0.2], dtype=np.float64)
        pad = np.pad(h_est, (1, 1), mode="edge")
        real = np.convolve(np.real(pad), kernel, mode="valid")
        imag = np.convolve(np.imag(pad), kernel, mode="valid")
        return real + 1j * imag

    def _generate_channel_realization(self, max_delay_samp: int, doppler_spread_hz: float):
        # Kept for compatibility/debugging.  The runtime path uses
        # _next_channel_state() so that BER-vs-time reflects a slow-varying
        # channel instead of an independent random channel on every frame.
        return _generate_sparse_channel_realization(
            self._rng,
            max_delay_samp=max_delay_samp,
            doppler_spread_hz=doppler_spread_hz,
            path_gains_db=self._path_gains_db,
            delay_scales=self._path_delay_scale,
            doppler_scales=self._path_doppler_scale,
        )

    def _channel_correlation(self, doppler_spread_hz: float) -> float:
        frame_len = self.pre_guard_len + self.sync_len + self.payload_symbols * self._payload_symbol_len
        t_frame = frame_len / max(float(self.samp_rate), 1e-12)
        fd = max(0.0, float(doppler_spread_hz))
        if fd <= 1e-9:
            return 0.99995
        rho = float(np.exp(-0.35 * (np.pi * fd * t_frame) ** 2))
        return float(min(max(rho, 0.94), 0.99995))

    def _next_channel_state(self, max_delay_samp: int, doppler_spread_hz: float):
        """Return a persistent channel realization for the current run.

        The previous slow-channel version still updated the complex path gains
        at every frame boundary.  Because _apply_time_varying_channel() already
        applies Doppler using the absolute sample offset, changing the gains at
        the frame boundary creates an artificial discontinuity in the received
        time-domain stream and can make the OFDM waveform/spectrum look wrong.

        For UI demonstration we keep one sparse realization fixed until a
        channel parameter changes.  Time variation is still represented by the
        continuous Doppler phase term inside _apply_time_varying_channel().
        """
        max_delay_samp = int(max(0, max_delay_samp))
        doppler_spread_hz = float(max(0.0, doppler_spread_hz))
        if (
            self._channel_state is None
            or int(self._channel_state.get("max_delay_samp", -1)) != max_delay_samp
            or abs(float(self._channel_state.get("doppler_spread_hz", -1.0)) - doppler_spread_hz) > 1e-9
        ):
            self._channel_state = _init_sparse_channel_state(
                self._rng,
                max_delay_samp=max_delay_samp,
                doppler_spread_hz=doppler_spread_hz,
                path_gains_db=self._path_gains_db,
                delay_scales=self._path_delay_scale,
                doppler_scales=self._path_doppler_scale,
            )
        return (
            self._channel_state["delays"],
            self._channel_state["gains"],
            self._channel_state["fd_list"],
        )

    def _pilot_values(self, sym_idx: int) -> np.ndarray:
        polarity_seq = np.array([1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1], dtype=np.float64)
        polarity = polarity_seq[sym_idx % len(polarity_seq)]
        base = np.array([1.0, 1.0, 1.0, -1.0], dtype=np.float64)
        return polarity * base

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
            i_amp = levels[i_bin]
            q_amp = levels[q_bin]
            constellation[idx] = i_amp + 1j * q_amp
            bit_patterns[idx] = bits

        constellation /= np.sqrt(np.mean(np.abs(constellation) ** 2) + 1e-12)
        return constellation, bit_patterns

    def _map_bits_to_symbols(self, bit_groups: np.ndarray) -> np.ndarray:
        bit_groups = np.asarray(bit_groups, dtype=np.uint8)
        idx = np.zeros(bit_groups.shape[0], dtype=np.int64)
        for k in range(bit_groups.shape[1]):
            idx = (idx << 1) | bit_groups[:, k].astype(np.int64)
        return self._constellation[idx]

    def _demap_symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        symbols = np.asarray(symbols, dtype=np.complex128)
        dist2 = np.abs(symbols[:, None] - self._constellation[None, :]) ** 2
        nearest = np.argmin(dist2, axis=1).astype(np.int64)
        return self._bit_patterns[nearest]

    @staticmethod
    def _compute_evm_metrics(rx_symbols: np.ndarray, ref_symbols: np.ndarray, valid: bool = True):
        """Return RMS EVM in percent and dB for one valid OFDM frame.

        EVM is computed on the equalized data symbols only.  If the frame is
        not synchronized or the symbol arrays are incomplete, EVM is reported
        as NaN and is not inserted into the live EVM history.
        """
        if not valid:
            return float("nan"), float("nan"), False

        rx_symbols = np.asarray(rx_symbols, dtype=np.complex128).reshape(-1)
        ref_symbols = np.asarray(ref_symbols, dtype=np.complex128).reshape(-1)
        n = min(rx_symbols.size, ref_symbols.size)
        if n <= 0:
            return float("nan"), float("nan"), False

        rx_symbols = rx_symbols[:n]
        ref_symbols = ref_symbols[:n]
        if not (np.all(np.isfinite(rx_symbols)) and np.all(np.isfinite(ref_symbols))):
            return float("nan"), float("nan"), False

        ref_power = float(np.mean(np.abs(ref_symbols) ** 2))
        if ref_power <= 1e-12:
            return float("nan"), float("nan"), False

        err_power = float(np.mean(np.abs(rx_symbols - ref_symbols) ** 2))
        evm_rms = float(np.sqrt(max(err_power, 0.0) / ref_power))
        evm_percent = 100.0 * evm_rms
        evm_db = 20.0 * np.log10(max(evm_rms, 1e-12))
        return float(evm_percent), float(evm_db), True

    def _reset_stats_locked(self):
        self._t0 = time.time()
        self._metric_tracker.reset(start_time=self._t0)
        self._evm_tracker.reset(start_time=self._t0)
        self._pre_eq_constellation = np.zeros(0, dtype=np.complex64)
        self._post_eq_constellation = np.zeros(0, dtype=np.complex64)
        self._last_good_pre_eq_constellation = np.zeros(0, dtype=np.complex64)
        self._last_good_post_eq_constellation = np.zeros(0, dtype=np.complex64)

    @staticmethod
    def _positions_in_parent(child: np.ndarray, parent: np.ndarray) -> np.ndarray:
        mapping = {int(v): i for i, v in enumerate(parent.tolist())}
        return np.asarray([mapping[int(v)] for v in child.tolist()], dtype=np.int64)

    @staticmethod
    def _bits_to_int(bits: np.ndarray) -> int:
        val = 0
        for b in bits.tolist():
            val = (val << 1) | int(b)
        return int(val)

    @staticmethod
    def _gray_to_binary(gray: int) -> int:
        binary = int(gray)
        shift = binary >> 1
        while shift > 0:
            binary ^= shift
            shift >>= 1
        return binary

    @staticmethod
    def _build_subcarrier_plan(fft_len: int):
        if fft_len != 64:
            raise ValueError("This fairness-aligned version is defined for fft_len == 64")
        active_bins = np.concatenate([np.arange(-26, 0), np.arange(1, 27)])
        pilot_bins = np.array([-21, -7, 7, 21])
        active_shift = active_bins + fft_len // 2
        pilot_shift = pilot_bins + fft_len // 2
        data_shift = np.asarray([x for x in active_shift.tolist() if x not in set(pilot_shift.tolist())], dtype=np.int64)
        return active_shift.astype(np.int64), pilot_shift.astype(np.int64), data_shift

    @staticmethod
    def _downsample_points(symbols: np.ndarray, max_points: int) -> np.ndarray:
        if symbols is None:
            return np.zeros(0, dtype=np.complex64)
        symbols = np.asarray(symbols, dtype=np.complex64)
        if symbols.size <= max_points:
            return symbols.copy()
        idx = np.linspace(0, symbols.size - 1, max_points, dtype=np.int64)
        return symbols[idx].copy()

    def _append_spectrum_samples(self, samples: np.ndarray):
        samples = np.asarray(samples, dtype=np.complex64)
        if samples.size == 0:
            return
        keep = 131072
        merged = np.concatenate([self._spectrum_buffer, samples])
        self._spectrum_buffer = merged[-keep:]


# ---------------------------------------------------------------------------
# 阶段4：统一引擎兼容壳
# ---------------------------------------------------------------------------
def _create_ofdm_backend(**kwargs):
    """供 waveform_sim.core.engine 构造 OFDM 后端。"""
    return _LegacyOfdmTransceiver(**kwargs)


class OfdmTransceiver(LinkSimulator):
    """OFDM 兼容壳：继承统一引擎，委托 _LegacyOfdmTransceiver，公开接口不变。"""

    def __init__(self, **kwargs):
        super().__init__(waveform="OFDM", **kwargs)

    def __getattr__(self, name):
        backend = self.__dict__.get("_backend")
        if backend is not None and hasattr(backend, name):
            return getattr(backend, name)
        raise AttributeError(name)


if __name__ == "__main__":
    tb = OfdmTransceiver(
        fft_len=64,
        cp_len=16,
        snr_db=15,
        doppler_spread_hz=20,
        cfo_hz=0.0,
        mod_order="QPSK",
        payload_symbols=8,
    )
    tb.start()
    time.sleep(1.0)
    print(tb.get_last_metrics())
    tb.stop()
    tb.wait()
