"""Shared BER, FER, and EVM history tracking for waveform simulators."""

from collections import deque
import time

import numpy as np


class _MetricTracker:
    """BER/FER counters and display histories shared by waveform simulators.

    Track cumulative totals, a sliding bit-count BER, and a log-domain EWMA
    for the live BER display.
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
    """EVM history tracker shared by the waveform simulators.

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
