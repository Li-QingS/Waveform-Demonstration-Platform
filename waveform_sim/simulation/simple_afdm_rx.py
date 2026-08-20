# -*- coding: utf-8 -*-
"""
UI-compatible AFDM transceiver
==============================

Purpose:
- Keep the public interface compatible with the current UI runner.
- Replace the earlier AFDM receiver with a redesigned CFO-aware receiver that
  can demonstrate a clear AFDM advantage over the OFDM baseline under large
  CFO + Doppler.

Compatible methods:
- start / stop / wait
- update_params
- get_spectrum
- get_constellation
- get_ber_history
- get_ber_summary
- get_adaptive_params
- reset_ber_stats

Design notes:
- Same outer frame as the current OFDM baseline: guard + repeated-half sync
  preamble + 1 training unit + 7 data units.
- Same sparse time-varying multipath channel with CFO injection.
- AFDM receiver: coarse preamble sync + wide-range CFO search on the training
  block + DD-structured channel estimation + AFDM-domain LMMSE detection.
- Chirp parameters are kept fixed by default. The UI-facing adaptive state is
  preserved for compatibility, but this version does not auto-tune c1/c2.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from waveform_sim.core.engine import LinkSimulator


# =========================================================
# Small utility layer
# =========================================================


def _as_1d_float_array(x, fallback):
    if x is None:
        return np.asarray(fallback, dtype=np.float64).copy()
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.asarray(fallback, dtype=np.float64).copy()
    return arr


def _bits_per_symbol(mod_order: str) -> int:
    mod_order = str(mod_order).upper()
    if mod_order == "QPSK":
        return 2
    if mod_order == "16QAM":
        return 4
    if mod_order == "64QAM":
        return 6
    raise ValueError(f"Unsupported modulation: {mod_order}")


def _qpsk_map(bit_groups: np.ndarray) -> np.ndarray:
    b = np.asarray(bit_groups, dtype=np.int8)
    i = 1 - 2 * b[:, 0]
    q = 1 - 2 * b[:, 1]
    return (i.astype(np.float64) + 1j * q.astype(np.float64)) / np.sqrt(2.0)


def _gray2pam2(b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    out = np.empty_like(b0, dtype=np.float64)
    out[(b0 == 0) & (b1 == 0)] = -3.0
    out[(b0 == 0) & (b1 == 1)] = -1.0
    out[(b0 == 1) & (b1 == 1)] = 1.0
    out[(b0 == 1) & (b1 == 0)] = 3.0
    return out


def _gray2pam3(b0: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    out = np.empty_like(b0, dtype=np.float64)
    code = 4 * b0 + 2 * b1 + b2
    mapping = {
        0: -7.0,
        1: -5.0,
        3: -3.0,
        2: -1.0,
        6: 1.0,
        7: 3.0,
        5: 5.0,
        4: 7.0,
    }
    for k, v in mapping.items():
        out[code == k] = v
    return out


def _pam2_to_gray_bits(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    levels = np.array([-3.0, -1.0, 1.0, 3.0])
    idx = np.argmin((x[:, None] - levels[None, :]) ** 2, axis=1)
    table = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int8)
    return table[idx]


def _pam3_to_gray_bits(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    levels = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
    idx = np.argmin((x[:, None] - levels[None, :]) ** 2, axis=1)
    table = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 0],
        ],
        dtype=np.int8,
    )
    return table[idx]


def _bits_to_symbols(bits: np.ndarray, mod_order: str) -> np.ndarray:
    mod_order = str(mod_order).upper()
    bps = _bits_per_symbol(mod_order)
    bits = np.asarray(bits, dtype=np.int8)
    n = len(bits) // bps
    b = bits[: n * bps].reshape(n, bps)

    if mod_order == "QPSK":
        return _qpsk_map(b)
    if mod_order == "16QAM":
        i = _gray2pam2(b[:, 0], b[:, 1])
        q = _gray2pam2(b[:, 2], b[:, 3])
        return (i + 1j * q) / np.sqrt(10.0)
    if mod_order == "64QAM":
        i = _gray2pam3(b[:, 0], b[:, 1], b[:, 2])
        q = _gray2pam3(b[:, 3], b[:, 4], b[:, 5])
        return (i + 1j * q) / np.sqrt(42.0)
    raise ValueError(f"Unsupported modulation: {mod_order}")


def _symbols_to_bits(symbols: np.ndarray, mod_order: str) -> np.ndarray:
    mod_order = str(mod_order).upper()
    rx = np.asarray(symbols)

    if mod_order == "QPSK":
        s = rx * np.sqrt(2.0)
        b0 = (np.real(s) < 0).astype(np.int8)
        b1 = (np.imag(s) < 0).astype(np.int8)
        return np.column_stack([b0, b1]).reshape(-1)

    if mod_order == "16QAM":
        s = rx * np.sqrt(10.0)
        bi = _pam2_to_gray_bits(np.real(s))
        bq = _pam2_to_gray_bits(np.imag(s))
        return np.column_stack([bi[:, 0], bi[:, 1], bq[:, 0], bq[:, 1]]).reshape(-1)

    if mod_order == "64QAM":
        s = rx * np.sqrt(42.0)
        bi = _pam3_to_gray_bits(np.real(s))
        bq = _pam3_to_gray_bits(np.imag(s))
        return np.column_stack([
            bi[:, 0], bi[:, 1], bi[:, 2],
            bq[:, 0], bq[:, 1], bq[:, 2],
        ]).reshape(-1)

    raise ValueError(f"Unsupported modulation: {mod_order}")


def _build_sync_preamble(rng: np.random.Generator, half_len: int) -> np.ndarray:
    bits = rng.integers(0, 2, size=2 * half_len, dtype=np.uint8)
    groups = bits.reshape(-1, 2)
    a = _qpsk_map(groups)[:half_len]
    a = a / np.sqrt(np.mean(np.abs(a) ** 2) + 1e-12)
    return np.concatenate([a, a]).astype(np.complex128)


def _training_active_sequence(n_active: int) -> np.ndarray:
    train = np.ones(n_active, dtype=np.complex128)
    train[1::2] = -1.0 + 0.0j
    return train


def _pilot_values(sym_idx: int) -> np.ndarray:
    polarity_seq = np.array([1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1], dtype=np.float64)
    polarity = polarity_seq[sym_idx % len(polarity_seq)]
    base = np.array([1.0, 1.0, 1.0, -1.0], dtype=np.float64)
    return (polarity * base).astype(np.complex128)


def _build_afdm_resource_plan(n_fft: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_fft != 64:
        raise ValueError("This AFDM implementation is defined for N == 64")
    active_bins = np.concatenate([np.arange(-26, 0), np.arange(1, 27)])
    pilot_bins = np.array([-21, -7, 7, 21])
    active_idx = active_bins + n_fft // 2
    pilot_idx = pilot_bins + n_fft // 2
    pilot_set = set(pilot_idx.tolist())
    data_idx = np.asarray([x for x in active_idx.tolist() if x not in pilot_set], dtype=np.int64)
    return active_idx.astype(np.int64), pilot_idx.astype(np.int64), data_idx


def _downsample_points(symbols: np.ndarray, max_points: int = 320) -> np.ndarray:
    if symbols is None:
        return np.zeros(0, dtype=np.complex64)
    symbols = np.asarray(symbols, dtype=np.complex64)
    if symbols.size <= max_points:
        return symbols.copy()
    idx = np.linspace(0, symbols.size - 1, max_points, dtype=np.int64)
    return symbols[idx].copy()


def _generate_sparse_channel_realization(
    rng: np.random.Generator,
    max_delay_samp: int,
    doppler_spread_hz: float,
    path_gains_db,
    delay_scales,
    doppler_scales,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return delays.astype(np.int64), gains.astype(np.complex128), fd_list.astype(np.float64)


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
    x: np.ndarray,
    delays: np.ndarray,
    gains: np.ndarray,
    fd_list: np.ndarray,
    sample_rate: float,
    cfo_hz: float = 0.0,
    sample_offset: int = 0,
) -> np.ndarray:
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


def _add_awgn_by_ebn0(signal: np.ndarray, ebn0_db: float, n_info_bits: int, rng: np.random.Generator):
    signal = np.asarray(signal, dtype=np.complex128)
    power = float(np.mean(np.abs(signal) ** 2))
    power = max(power, 1e-12)
    ebn0_linear = 10.0 ** (float(ebn0_db) / 10.0)
    noise_power = power * len(signal) / max(float(n_info_bits) * ebn0_linear, 1e-12)
    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal)))
    return (signal + noise).astype(np.complex128, copy=False), float(noise_power)


def _schmidl_cox_metric(rx: np.ndarray, L: int) -> np.ndarray:
    n_max = len(rx) - 2 * L
    if n_max <= 1:
        return np.zeros(1, dtype=np.float64)
    metric = np.zeros(n_max, dtype=np.float64)
    for d in range(n_max):
        a = rx[d: d + L]
        b = rx[d + L: d + 2 * L]
        p = np.sum(a * np.conj(b))
        r = np.sum(np.abs(b) ** 2) + 1e-12
        metric[d] = (np.abs(p) ** 2) / (r ** 2)
    return metric


def _estimate_cfo_from_repeated_preamble(rx: np.ndarray, frame_start: int, L: int, sample_rate: float) -> float:
    if frame_start + 2 * L > len(rx):
        return 0.0
    a = rx[frame_start: frame_start + L]
    b = rx[frame_start + L: frame_start + 2 * L]
    p = np.sum(a * np.conj(b))
    phase = np.angle(p)
    return float(-phase * sample_rate / (2 * np.pi * L))


def _coarse_sync_and_extract(rx: np.ndarray, pre_guard_len: int, sync_half_len: int, sync_len: int, payload_len: int, sample_rate: float):
    min_len = pre_guard_len + sync_len + payload_len
    if len(rx) < min_len:
        return False, np.zeros(payload_len, dtype=np.complex128), {"frame_start": 0, "sync_metric": 0.0, "cfo_est_hz": 0.0}

    metric = _schmidl_cox_metric(rx, sync_half_len)
    start_lo = max(0, pre_guard_len - 4)
    start_hi = min(len(metric), pre_guard_len + 5)
    frame_peak = int(start_lo + np.argmax(metric[start_lo:start_hi]))
    frame_start = int(pre_guard_len)
    sync_metric = float(metric[frame_peak])
    cfo_est_hz = _estimate_cfo_from_repeated_preamble(rx, frame_peak, sync_half_len, sample_rate)

    n = np.arange(len(rx), dtype=np.float64)
    rx_cfo_corrected = rx * np.exp(-1j * 2 * np.pi * cfo_est_hz * n / sample_rate)

    payload_start = frame_start + sync_len
    payload_end = payload_start + payload_len
    if payload_end > len(rx_cfo_corrected):
        return False, np.zeros(payload_len, dtype=np.complex128), {"frame_start": frame_start, "sync_metric": sync_metric, "cfo_est_hz": cfo_est_hz}

    return True, rx_cfo_corrected[payload_start:payload_end].astype(np.complex128), {
        "frame_start": float(frame_start),
        "sync_metric": sync_metric,
        "cfo_est_hz": float(cfo_est_hz),
    }


def _build_afdm_mats(N: int, c1: float, c2: float) -> Tuple[np.ndarray, np.ndarray]:
    n = np.arange(N)
    F = np.fft.fft(np.eye(N), axis=0) / np.sqrt(N)
    Fh = F.conj().T
    D1 = np.diag(np.exp(-1j * 2.0 * np.pi * float(c1) * (n ** 2) / N))
    D2 = np.diag(np.exp(-1j * 2.0 * np.pi * float(c2) * (n ** 2) / N))
    A_tx = D1.conj().T @ Fh @ D2.conj().T
    A_rx = D2 @ F @ D1
    return A_tx, A_rx


def _build_path_basis_for_block(x_block_t: np.ndarray, delays: np.ndarray, fd_list: np.ndarray, sample_rate: float, sample_offset: int) -> np.ndarray:
    x_block_t = np.asarray(x_block_t, dtype=np.complex128).reshape(-1)
    n = np.arange(len(x_block_t), dtype=np.float64)
    n_total = n + float(sample_offset)
    basis: List[np.ndarray] = []
    for l, fd in zip(np.asarray(delays, dtype=np.int64), np.asarray(fd_list, dtype=np.float64)):
        l_int = int(max(0, l)) % len(x_block_t)
        shifted = np.roll(x_block_t, l_int)
        phase = np.exp(1j * 2.0 * np.pi * float(fd) * n_total / max(float(sample_rate), 1e-12))
        basis.append(phase * shifted)
    if len(basis) == 0:
        return np.zeros((len(x_block_t), 0), dtype=np.complex128)
    return np.column_stack(basis).astype(np.complex128, copy=False)


def _build_cyclic_time_channel_matrix(N: int, delays: np.ndarray, gains: np.ndarray, fd_list: np.ndarray, sample_rate: float, sample_offset: int) -> np.ndarray:
    n = np.arange(N, dtype=np.float64)
    n_total = n + float(sample_offset)
    H = np.zeros((N, N), dtype=np.complex128)
    for l, h, fd in zip(np.asarray(delays, dtype=np.int64), gains, np.asarray(fd_list, dtype=np.float64)):
        l_int = int(max(0, l)) % N
        phase = np.exp(1j * 2.0 * np.pi * float(fd) * n_total / max(float(sample_rate), 1e-12))
        D = np.diag(phase)
        S = np.zeros((N, N), dtype=np.complex128)
        for row in range(N):
            col = (row - l_int) % N
            S[row, col] = 1.0
        H += complex(h) * D @ S
    return H


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


@dataclass(frozen=True)
class _SearchConfig:
    coarse_min_hz: float = -20_000.0
    coarse_max_hz: float = 20_000.0
    coarse_step_hz: float = 1000.0
    fine_half_span_hz: float = 500.0
    fine_step_hz: float = 100.0


class _LegacyAFDMTransceiver:
    def __init__(
        self,
        c1=0.05,
        c2=0.05,
        snr_db=20.0,
        mod_order="QPSK",
        doppler_freq=20.0,
        delay_spread=5,
        cfo_hz=0.0,
        sample_rate=48000,
        frame_size=128,
        history_frames=12,
        ber_history_len=200,
    ):
        # Internal fair-comparison system parameters aligned to the current UI/OFDM path.
        self.N = 64
        self.cp_len = 16
        self.payload_units = 8
        self.train_units = 1
        self.data_units = 7
        self.pre_guard_len = 16
        self.sync_half_len = self.N
        self.subcarrier_spacing = 15e3
        self.sample_rate = float(self.N * self.subcarrier_spacing)
        self.frame_size = self.N
        self.history_frames = int(history_frames)
        self.ber_history_len = int(ber_history_len)

        self._running = False
        self._thread = None

        self._lock = threading.Lock()
        self._param_lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._rng = np.random.default_rng()
        self._sample_counter = 0
        self._constellation_points = np.zeros(0, dtype=np.complex64)
        self._last_good_constellation = np.zeros(0, dtype=np.complex64)

        # Publicly configurable params
        self._c1 = float(c1)
        self._c2 = float(c2)
        self._snr_db = float(snr_db)
        self._mod_order = str(mod_order).upper()
        self._doppler_freq = float(doppler_freq)
        self._max_delay_samp = max(0, int(delay_spread))
        self.delay_spread = self._max_delay_samp
        self._cfo_hz = float(cfo_hz)
        self._adaptive_chirp_enabled = False
        self._adaptive_state = {
            "c1": float(c1),
            "c2": float(c2),
            "adaptive_enabled": False,
            "doppler_freq": float(doppler_freq),
            "delay_spread": int(self._max_delay_samp),
            "cfo_hz": float(cfo_hz),
        }

        # Channel profile aligned to the OFDM baseline.
        self._path_gains_db = np.array([0.0, -3.5, -7.0], dtype=np.float64)
        self._path_delay_scale = np.array([0.0, 0.4, 1.0], dtype=np.float64)
        self._path_doppler_scale = np.array([0.0, 0.6, -0.8], dtype=np.float64)
        self._channel_state = None

        # Resource plan and AFDM matrices.
        self._active_idx, self._pilot_idx, self._data_idx = _build_afdm_resource_plan(self.N)
        self._n_active = int(len(self._active_idx))
        self._n_data = int(len(self._data_idx))
        self._train_active = _training_active_sequence(self._n_active)
        self._search_cfg = _SearchConfig()
        self._refresh_afdm_mats()

        # Sync / frame geometry.
        self.sync_preamble = _build_sync_preamble(self._rng, self.sync_half_len)
        self.sync_len = len(self.sync_preamble)
        self._payload_unit_len = self.N + self.cp_len
        self.sync_metric_threshold = 0.35
        self._sample_buffer = deque(maxlen=self._frame_template_len() * max(2, self.history_frames) * 4)

        # BER stats.
        self._start_time = None
        self._frame_count = 0
        self._total_bits = 0
        self._total_bit_errors = 0
        self._last_frame_bits = 0
        self._last_frame_bit_errors = 0
        self._last_frame_ber = 0.0
        self._metric_tracker = _MetricTracker(history_len=self.ber_history_len, ewma_alpha=0.08)
        self._evm_tracker = _EvmTracker(history_len=self.ber_history_len)
        self._last_metrics = {
            "ebn0_db": float(self._snr_db),
            "sync_metric": 0.0,
            "coarse_cfo_hz": 0.0,
            "refined_cfo_hz": 0.0,
            "total_cfo_hz": 0.0,
            "max_delay_samp": int(self._max_delay_samp),
            "bit_errors": 0,
            "total_bits": 0,
            "ber": 0.5,
            "fer": 1.0,
            "net_data_symbols": int(self.data_units * self._n_data),
            "path_est_nmse_db": 0.0,
            "alpha_mean_abs": 0.0,
            "evm_percent": float("nan"),
            "evm_db": float("nan"),
            "evm_valid": False,
        }

    # =========================================================
    # Lifecycle
    # =========================================================
    def start(self):
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._evm_tracker.reset(start_time=self._start_time)
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def wait(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # =========================================================
    # UI-compatible parameter updates
    # =========================================================
    def update_params(
        self,
        c1=None,
        c2=None,
        snr_db=None,
        mod_order=None,
        doppler_freq=None,
        cfo_hz=None,
        delay_spread=None,
        adaptive_chirp_enabled=None,
    ):
        with self._param_lock:
            if c1 is not None:
                self._c1 = float(c1)
            if c2 is not None:
                self._c2 = float(c2)
            if snr_db is not None:
                self._snr_db = float(snr_db)
            if mod_order is not None:
                self._mod_order = str(mod_order).upper()
            if doppler_freq is not None:
                new_doppler = float(doppler_freq)
                if abs(new_doppler - self._doppler_freq) > 1e-9:
                    self._channel_state = None
                self._doppler_freq = new_doppler
            if cfo_hz is not None:
                self._cfo_hz = float(cfo_hz)
            if delay_spread is not None:
                new_delay = max(0, int(delay_spread))
                if new_delay != self._max_delay_samp:
                    self._channel_state = None
                self._max_delay_samp = new_delay
                self.delay_spread = self._max_delay_samp
            if adaptive_chirp_enabled is not None:
                self._adaptive_chirp_enabled = bool(adaptive_chirp_enabled)
            self._adaptive_state = {
                "c1": float(self._c1),
                "c2": float(self._c2),
                "adaptive_enabled": bool(self._adaptive_chirp_enabled),
                "doppler_freq": float(self._doppler_freq),
                "delay_spread": int(self._max_delay_samp),
                "cfo_hz": float(self._cfo_hz),
            }
            self._refresh_afdm_mats()

    def set_snr_db(self, value: float):
        self.update_params(snr_db=value)
        self.reset_ber_stats()

    def set_cfo_hz(self, value: float):
        self.update_params(cfo_hz=value)
        self.reset_ber_stats()

    def set_doppler_spread_hz(self, value: float):
        self.update_params(doppler_freq=value)
        self.reset_ber_stats()

    def set_delay_spread(self, value: int):
        self.update_params(delay_spread=value)
        self.reset_ber_stats()

    def set_doppler_hz(self, value: float):
        self.set_doppler_spread_hz(value)

    def get_params(self):
        with self._param_lock:
            return {
                "c1": self._c1,
                "c2": self._c2,
                "snr_db": self._snr_db,
                "mod_order": self._mod_order,
                "doppler_freq": self._doppler_freq,
                "delay_spread": self._max_delay_samp,
                "cfo_hz": self._cfo_hz,
                "sample_rate": self.sample_rate,
                "frame_size": self.N,
                "cp_len": self.cp_len,
                "payload_units": self.payload_units,
            }

    def get_adaptive_params(self):
        with self._param_lock:
            return {
                "c1": float(self._c1),
                "c2": float(self._c2),
                "adaptive_enabled": bool(self._adaptive_chirp_enabled),
                "doppler_freq": float(self._doppler_freq),
                "delay_spread": int(self._max_delay_samp),
                "cfo_hz": float(self._cfo_hz),
                "state": dict(self._adaptive_state),
            }

    # =========================================================
    # UI data interfaces
    # =========================================================
    def get_spectrum(self, num_samples=1024):
        num_samples = int(max(1, num_samples))
        with self._lock:
            if len(self._sample_buffer) == 0:
                return np.zeros(num_samples, dtype=np.complex64)
            data = np.array(self._sample_buffer, dtype=np.complex64)
        if len(data) >= num_samples:
            return data[-num_samples:]
        pad = np.zeros(num_samples - len(data), dtype=np.complex64)
        return np.concatenate([pad, data])

    def get_constellation(self):
        with self._lock:
            if self._constellation_points is None or len(self._constellation_points) == 0:
                return np.zeros(64, dtype=np.complex64)
            return self._constellation_points.copy()

    def get_ber_history(self):
        with self._stats_lock:
            hist = self._metric_tracker.get_history()
            if hist[0].size == 0:
                return np.array([0.0]), np.array([1e-6])
            return hist

    def get_estimated_ber(self):
        return self.get_ber_history()

    def get_ber_estimate(self):
        with self._stats_lock:
            return self._metric_tracker.get_ber_estimate()

    def get_fer_estimate(self):
        with self._stats_lock:
            return self._metric_tracker.get_fer_estimate()

    def get_evm_history(self, unit: str = "percent"):
        with self._stats_lock:
            return self._evm_tracker.get_history(unit=unit)

    def get_evm_estimate(self, unit: str = "percent"):
        with self._stats_lock:
            return self._evm_tracker.get_estimate(unit=unit)

    def get_last_metrics(self):
        with self._stats_lock:
            return dict(self._last_metrics)

    def get_ber_summary(self):
        with self._stats_lock:
            stats = self._metric_tracker.snapshot()
            return {
                "frame_count": stats["frame_count"],
                "total_bits": stats["total_bits"],
                "total_bit_errors": stats["total_bit_errors"],
                "cumulative_ber": stats["cumulative_ber"],
                "last_frame_bits": stats["last_frame_bits"],
                "last_frame_bit_errors": stats["last_frame_bit_errors"],
                "last_frame_ber": stats["last_frame_ber"],
                "ber_window": stats["ber_window"],
                "ber_ewma": stats["ber_ewma"],
                "fer_window": stats["fer_window"],
                "last_metrics": dict(self._last_metrics),
            }

    def reset_ber_stats(self):
        with self._stats_lock:
            self._start_time = time.time()
            self._metric_tracker.reset(start_time=self._start_time)
            self._evm_tracker.reset(start_time=self._start_time)
            self._frame_count = 0
            self._total_bits = 0
            self._total_bit_errors = 0
            self._last_frame_bits = 0
            self._last_frame_bit_errors = 0
            self._last_frame_ber = 0.0

    # =========================================================
    # Internal helpers
    # =========================================================
    def _refresh_afdm_mats(self):
        self._A_tx, self._A_rx = _build_afdm_mats(self.N, self._c1, self._c2)

    def _frame_template_len(self) -> int:
        return self.pre_guard_len + self.sync_len + self.payload_units * self._payload_unit_len

    def _candidate_grid(self) -> np.ndarray:
        return np.arange(
            self._search_cfg.coarse_min_hz,
            self._search_cfg.coarse_max_hz + 0.5 * self._search_cfg.coarse_step_hz,
            self._search_cfg.coarse_step_hz,
            dtype=np.float64,
        )

    def _refine_grid(self, best_hz: float) -> np.ndarray:
        return np.arange(
            best_hz - self._search_cfg.fine_half_span_hz,
            best_hz + self._search_cfg.fine_half_span_hz + 0.5 * self._search_cfg.fine_step_hz,
            self._search_cfg.fine_step_hz,
            dtype=np.float64,
        )

    def _random_bits(self, nbits):
        return self._rng.integers(0, 2, size=nbits, dtype=np.int8)

    def _generate_channel_realization(self, doppler_freq: float):
        # Kept for compatibility/debugging.  The runtime path uses
        # _next_channel_state() to match the slow-varying channel model used by
        # the other backends.
        return _generate_sparse_channel_realization(
            self._rng,
            max_delay_samp=self._max_delay_samp,
            doppler_spread_hz=doppler_freq,
            path_gains_db=self._path_gains_db,
            delay_scales=self._path_delay_scale,
            doppler_scales=self._path_doppler_scale,
        )

    def _channel_correlation(self, doppler_freq: float) -> float:
        frame_len = self._frame_template_len()
        t_frame = frame_len / max(float(self.sample_rate), 1e-12)
        fd = max(0.0, float(doppler_freq))
        if fd <= 1e-9:
            return 0.99995
        rho = float(np.exp(-0.35 * (np.pi * fd * t_frame) ** 2))
        return float(min(max(rho, 0.94), 0.99995))

    def _next_channel_state(self, doppler_freq: float):
        """Return a persistent channel realization for the current run.

        Keep path gains fixed until channel parameters change.  Doppler remains
        continuous through the absolute sample offset used by
        _apply_time_varying_channel().  This avoids artificial frame-boundary
        jumps in the displayed time-domain stream and makes BER-time curves
        comparable with OFDM/OTFS.
        """
        doppler_freq = float(max(0.0, doppler_freq))
        if (
            self._channel_state is None
            or int(self._channel_state.get("max_delay_samp", -1)) != int(self._max_delay_samp)
            or abs(float(self._channel_state.get("doppler_spread_hz", -1.0)) - doppler_freq) > 1e-9
        ):
            self._channel_state = _init_sparse_channel_state(
                self._rng,
                max_delay_samp=self._max_delay_samp,
                doppler_spread_hz=doppler_freq,
                path_gains_db=self._path_gains_db,
                delay_scales=self._path_delay_scale,
                doppler_scales=self._path_doppler_scale,
            )
        return (
            self._channel_state["delays"],
            self._channel_state["gains"],
            self._channel_state["fd_list"],
        )

    def _build_tx_frame(self, tx_data_symbols: np.ndarray):
        parts = [
            np.zeros(self.pre_guard_len, dtype=np.complex128),
            self.sync_preamble.astype(np.complex128),
        ]

        x_train = np.zeros(self.N, dtype=np.complex128)
        x_train[self._active_idx] = self._train_active
        s_train = self._A_tx @ x_train
        parts.append(s_train[-self.cp_len:])
        parts.append(s_train)

        for sym_idx in range(self.data_units):
            x_dd = np.zeros(self.N, dtype=np.complex128)
            x_dd[self._pilot_idx] = _pilot_values(sym_idx)
            x_dd[self._data_idx] = tx_data_symbols[sym_idx]
            s_t = self._A_tx @ x_dd
            parts.append(s_t[-self.cp_len:])
            parts.append(s_t)

        tx_frame = np.concatenate(parts).astype(np.complex128, copy=False)
        return tx_frame, x_train, s_train

    def _training_cfo_cost(self, y_train_raw, x_train_t, delays, fd_list, train_useful_start_abs, cfo_hz, noise_var):
        n_abs = train_useful_start_abs + np.arange(self.N, dtype=np.float64)
        y_corr = y_train_raw * np.exp(-1j * 2.0 * np.pi * float(cfo_hz) * n_abs / self.sample_rate)
        basis = _build_path_basis_for_block(
            x_block_t=x_train_t,
            delays=delays,
            fd_list=fd_list,
            sample_rate=self.sample_rate,
            sample_offset=train_useful_start_abs,
        )
        reg = max(float(noise_var), 1e-9)
        gram = basis.conj().T @ basis + reg * np.eye(basis.shape[1], dtype=np.complex128)
        rhs = basis.conj().T @ y_corr
        h_est = np.linalg.solve(gram, rhs)
        err = float(np.linalg.norm(y_corr - basis @ h_est) ** 2)
        return err, h_est.astype(np.complex128, copy=False)

    def _search_and_estimate_cfo(self, y_train_raw, x_train_t, delays, fd_list, train_useful_start_abs, noise_var, coarse_hint_hz):
        best_cfo = float(coarse_hint_hz)
        best_h = np.zeros(len(delays), dtype=np.complex128)
        best_cost = np.inf

        coarse_candidates = np.unique(np.concatenate([self._candidate_grid(), np.array([coarse_hint_hz], dtype=np.float64)]))
        for cfo in coarse_candidates:
            cost, h_est = self._training_cfo_cost(
                y_train_raw=y_train_raw,
                x_train_t=x_train_t,
                delays=delays,
                fd_list=fd_list,
                train_useful_start_abs=train_useful_start_abs,
                cfo_hz=float(cfo),
                noise_var=noise_var,
            )
            if cost < best_cost:
                best_cost = cost
                best_cfo = float(cfo)
                best_h = h_est

        for cfo in self._refine_grid(best_cfo):
            cost, h_est = self._training_cfo_cost(
                y_train_raw=y_train_raw,
                x_train_t=x_train_t,
                delays=delays,
                fd_list=fd_list,
                train_useful_start_abs=train_useful_start_abs,
                cfo_hz=float(cfo),
                noise_var=noise_var,
            )
            if cost < best_cost:
                best_cost = cost
                best_cfo = float(cfo)
                best_h = h_est

        return best_cfo, best_h, best_cost

    def _detect_one_data_unit(self, y_dd, h_eff, pilot_vec, noise_var):
        h_p = h_eff[:, self._pilot_idx]
        h_d = h_eff[:, self._data_idx]

        y_res = y_dd - h_p @ pilot_vec
        gram = h_d.conj().T @ h_d + noise_var * np.eye(h_d.shape[1], dtype=np.complex128)
        rhs = h_d.conj().T @ y_res
        x_hat0 = np.linalg.solve(gram, rhs)

        pilot_obs = y_dd - h_d @ x_hat0
        pilot_ref = h_p @ pilot_vec
        denom = np.vdot(pilot_ref, pilot_ref)
        if np.abs(denom) > 1e-12:
            alpha = np.vdot(pilot_ref, pilot_obs) / denom
        else:
            alpha = 1.0 + 0.0j
        if np.abs(alpha) < 1e-6:
            alpha = 1.0 + 0.0j

        h_p2 = alpha * h_p
        h_d2 = alpha * h_d
        y_res2 = y_dd - h_p2 @ pilot_vec
        gram2 = h_d2.conj().T @ h_d2 + noise_var * np.eye(h_d2.shape[1], dtype=np.complex128)
        rhs2 = h_d2.conj().T @ y_res2
        x_hat = np.linalg.solve(gram2, rhs2)
        return x_hat.astype(np.complex128, copy=False), complex(alpha)


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


    # =========================================================
    # Main simulation loop
    # =========================================================
    def _process_one_frame(self):
        params = self.get_params()
        rx_td, x_hat_data, tx_bits, rx_bits, metrics = self._simulate_one_frame(params)

        frame_bits = len(tx_bits)
        frame_err = int(np.sum(tx_bits != rx_bits)) if len(rx_bits) == len(tx_bits) else frame_bits
        frame_ber = frame_err / max(frame_bits, 1)
        frame_error = 1.0 if frame_err > 0 else 0.0
        metrics = dict(metrics)
        metrics["bit_errors"] = int(frame_err)
        metrics["total_bits"] = int(frame_bits)
        metrics["ber"] = float(frame_ber)
        metrics["fer"] = float(frame_error)
        metrics["total_cfo_hz"] = float(metrics.get("refined_cfo_hz", metrics.get("coarse_cfo_hz", 0.0)))

        with self._stats_lock:
            t_now = time.time() - (self._start_time or time.time())
            stats = self._metric_tracker.update(
                bit_errors=frame_err,
                total_bits=frame_bits,
                frame_error=frame_error,
                t_now=t_now,
            )
            evm_stats = self._evm_tracker.update(
                evm_percent=metrics.get("evm_percent", float("nan")),
                evm_db=metrics.get("evm_db", None),
                valid=bool(metrics.get("evm_valid", False)),
                t_now=t_now,
            )
            self._frame_count = stats["frame_count"]
            self._total_bits = stats["total_bits"]
            self._total_bit_errors = stats["total_bit_errors"]
            self._last_frame_bits = stats["last_frame_bits"]
            self._last_frame_bit_errors = stats["last_frame_bit_errors"]
            self._last_frame_ber = stats["last_frame_ber"]

            metrics["frame_ber"] = float(frame_ber)
            metrics["ber"] = float(stats["ber_window"])
            metrics["ber_ewma"] = float(stats["ber_ewma"])
            metrics["fer"] = float(stats["fer_window"])
            metrics["frame_error"] = float(frame_error)
            metrics["evm_percent"] = float(evm_stats["evm_percent"])
            metrics["evm_db"] = float(evm_stats["evm_db"])
            metrics["evm_valid"] = bool(metrics.get("evm_valid", False))
            self._last_metrics = dict(metrics)

        with self._lock:
            self._sample_buffer.extend(np.asarray(rx_td, dtype=np.complex64).tolist())
            pts = _downsample_points(np.asarray(x_hat_data, dtype=np.complex64), 320)
            if metrics.get("sync_metric", 0.0) >= self.sync_metric_threshold and len(pts) > 0:
                self._constellation_points = pts
                self._last_good_constellation = pts.copy()
            else:
                self._constellation_points = self._last_good_constellation.copy()

    def _worker_loop(self):
        frame_interval = 0.05
        while self._running:
            try:
                self._process_one_frame()
            except Exception as e:
                print(f"[AFDMTransceiver] worker exception: {e}")
            time.sleep(frame_interval)

    def step(self):
        """单帧仿真（供统一引擎调用）。"""
        self._process_one_frame()

    def _simulate_one_frame(self, params):
        ebn0_db = params["snr_db"]
        mod_order = params["mod_order"]
        doppler_freq = params["doppler_freq"]
        max_delay_samp = int(params.get("delay_spread", self._max_delay_samp))
        cfo_hz = params["cfo_hz"]

        bps = _bits_per_symbol(mod_order)
        n_info_bits = self.data_units * self._n_data * bps
        tx_bits = self._random_bits(n_info_bits)
        tx_data_symbols = _bits_to_symbols(tx_bits, mod_order).reshape(self.data_units, self._n_data)

        tx_frame, train_dd, train_t = self._build_tx_frame(tx_data_symbols)
        delays, gains_true, fd_list = self._next_channel_state(doppler_freq)

        sample_offset = self._sample_counter
        rx_chan = _apply_time_varying_channel(
            tx_frame,
            delays,
            gains_true,
            fd_list,
            sample_rate=self.sample_rate,
            cfo_hz=cfo_hz,
            sample_offset=sample_offset,
        )
        self._sample_counter += len(tx_frame)

        rx_noisy, noise_var = _add_awgn_by_ebn0(rx_chan, ebn0_db, n_info_bits, self._rng)

        payload_len = self.payload_units * self._payload_unit_len
        sync_ok, _, sync_info = _coarse_sync_and_extract(
            rx_noisy,
            pre_guard_len=self.pre_guard_len,
            sync_half_len=self.sync_half_len,
            sync_len=self.sync_len,
            payload_len=payload_len,
            sample_rate=self.sample_rate,
        )

        if not sync_ok:
            rx_bits = 1 - tx_bits
            metrics = {
                "ebn0_db": float(ebn0_db),
                "sync_metric": 0.0,
                "coarse_cfo_hz": 0.0,
                "refined_cfo_hz": 0.0,
                "max_delay_samp": int(max_delay_samp),
                "bit_errors": int(len(tx_bits)),
                "total_bits": int(len(tx_bits)),
                "ber": 1.0,
                "fer": 1.0,
                "total_cfo_hz": 0.0,
                "net_data_symbols": int(self.data_units * self._n_data),
                "path_est_nmse_db": 0.0,
                "alpha_mean_abs": 0.0,
                "evm_percent": float("nan"),
                "evm_db": float("nan"),
                "evm_valid": False,
            }
            return rx_noisy, np.zeros(self.data_units * self._n_data, dtype=np.complex128), tx_bits, rx_bits, metrics

        frame_start = int(sync_info["frame_start"])
        payload_start = frame_start + self.sync_len
        train_useful_start_abs = payload_start + self.cp_len
        train_blk_raw = rx_noisy[payload_start: payload_start + self._payload_unit_len]
        y_train_raw = train_blk_raw[self.cp_len: self.cp_len + self.N]

        refined_cfo_hz, h_est, _ = self._search_and_estimate_cfo(
            y_train_raw=y_train_raw,
            x_train_t=train_t,
            delays=delays,
            fd_list=fd_list,
            train_useful_start_abs=train_useful_start_abs,
            noise_var=noise_var,
            coarse_hint_hz=float(sync_info["cfo_est_hz"]),
        )

        n = np.arange(len(rx_noisy), dtype=np.float64)
        rx_refined = rx_noisy * np.exp(-1j * 2 * np.pi * refined_cfo_hz * n / self.sample_rate)

        ok2, rx_payload, sync_info2 = _coarse_sync_and_extract(
            rx_refined,
            pre_guard_len=self.pre_guard_len,
            sync_half_len=self.sync_half_len,
            sync_len=self.sync_len,
            payload_len=payload_len,
            sample_rate=self.sample_rate,
        )

        if not ok2:
            rx_bits = 1 - tx_bits
            metrics = {
                "ebn0_db": float(ebn0_db),
                "sync_metric": float(sync_info.get("sync_metric", 0.0)),
                "coarse_cfo_hz": float(sync_info.get("cfo_est_hz", 0.0)),
                "refined_cfo_hz": float(refined_cfo_hz),
                "bit_errors": int(len(tx_bits)),
                "total_bits": int(len(tx_bits)),
                "ber": 1.0,
                "net_data_symbols": int(self.data_units * self._n_data),
                "path_est_nmse_db": 0.0,
                "alpha_mean_abs": 0.0,
                "evm_percent": float("nan"),
                "evm_db": float("nan"),
                "evm_valid": False,
            }
            return rx_noisy, np.zeros(self.data_units * self._n_data, dtype=np.complex128), tx_bits, rx_bits, metrics

        frame_start2 = int(sync_info2["frame_start"])
        train_useful_start_abs2 = frame_start2 + self.sync_len + self.cp_len
        train_blk2 = rx_payload[: self._payload_unit_len]
        y_train_post = train_blk2[self.cp_len: self.cp_len + self.N]
        basis = _build_path_basis_for_block(
            x_block_t=train_t,
            delays=delays,
            fd_list=fd_list,
            sample_rate=self.sample_rate,
            sample_offset=train_useful_start_abs2,
        )
        reg = max(float(noise_var), 1e-9)
        gram = basis.conj().T @ basis + reg * np.eye(basis.shape[1], dtype=np.complex128)
        rhs = basis.conj().T @ y_train_post
        h_est = np.linalg.solve(gram, rhs).astype(np.complex128, copy=False)

        x_hat_blocks = []
        alpha_hist = []
        for unit_idx in range(1, self.payload_units):
            data_sym_idx = unit_idx - 1
            block_cp = rx_payload[unit_idx * self._payload_unit_len: (unit_idx + 1) * self._payload_unit_len]
            y_t = block_cp[self.cp_len: self.cp_len + self.N]
            y_dd = self._A_rx @ y_t
            useful_start_abs = frame_start2 + self.sync_len + unit_idx * self._payload_unit_len + self.cp_len
            h_time = _build_cyclic_time_channel_matrix(
                N=self.N,
                delays=delays,
                gains=h_est,
                fd_list=fd_list,
                sample_rate=self.sample_rate,
                sample_offset=useful_start_abs,
            )
            h_eff = self._A_rx @ h_time @ self._A_tx
            x_hat, alpha = self._detect_one_data_unit(
                y_dd=y_dd,
                h_eff=h_eff,
                pilot_vec=_pilot_values(data_sym_idx),
                noise_var=noise_var,
            )
            x_hat_blocks.append(x_hat)
            alpha_hist.append(alpha)

        x_hat_data = np.concatenate(x_hat_blocks) if x_hat_blocks else np.zeros(self.data_units * self._n_data, dtype=np.complex128)
        tx_syms_ref = tx_data_symbols.reshape(-1)
        evm_percent, evm_db, evm_valid = self._compute_evm_metrics(x_hat_data, tx_syms_ref, valid=len(x_hat_data) > 0)
        rx_bits = _symbols_to_bits(x_hat_data, mod_order) if len(x_hat_data) > 0 else 1 - tx_bits

        bit_errors = int(np.sum(tx_bits != rx_bits[: len(tx_bits)])) if len(rx_bits) >= len(tx_bits) else len(tx_bits)
        frame_ber = float(bit_errors / max(len(tx_bits), 1))
        path_nmse = float(np.sum(np.abs(h_est - gains_true) ** 2) / max(np.sum(np.abs(gains_true) ** 2), 1e-12))
        path_nmse_db = 10.0 * np.log10(max(path_nmse, 1e-12))
        alpha_mean_abs = float(np.mean(np.abs(alpha_hist))) if alpha_hist else 0.0

        metrics = {
            "ebn0_db": float(ebn0_db),
            "sync_metric": float(sync_info2.get("sync_metric", 0.0)),
            "coarse_cfo_hz": float(sync_info.get("cfo_est_hz", 0.0)),
            "refined_cfo_hz": float(refined_cfo_hz),
            "max_delay_samp": int(max_delay_samp),
            "bit_errors": int(bit_errors),
            "total_bits": int(len(tx_bits)),
            "ber": frame_ber,
            "net_data_symbols": int(self.data_units * self._n_data),
            "path_est_nmse_db": float(path_nmse_db),
            "alpha_mean_abs": float(alpha_mean_abs),
            "evm_percent": float(evm_percent),
            "evm_db": float(evm_db),
            "evm_valid": bool(evm_valid),
        }

        return rx_noisy, x_hat_data, tx_bits, rx_bits[: len(tx_bits)], metrics


# ---------------------------------------------------------------------------
# 阶段4：统一引擎兼容壳
# ---------------------------------------------------------------------------
def _create_afdm_backend(**kwargs):
    """供 waveform_sim.core.engine 构造 AFDM 后端。"""
    return _LegacyAFDMTransceiver(**kwargs)


class AFDMTransceiver(LinkSimulator):
    """AFDM 兼容壳：继承统一引擎，委托 _LegacyAFDMTransceiver，公开接口不变。"""

    def __init__(self, **kwargs):
        super().__init__(waveform="AFDM", **kwargs)

    def __getattr__(self, name):
        backend = self.__dict__.get("_backend")
        if backend is not None and hasattr(backend, name):
            return getattr(backend, name)
        raise AttributeError(name)


if __name__ == "__main__":
    tb = AFDMTransceiver(
        c1=0.05,
        c2=0.05,
        snr_db=18,
        mod_order="QPSK",
        doppler_freq=120,
        delay_spread=5,
        cfo_hz=10000.0,
    )
    tb.start()
    time.sleep(2.0)
    print(tb.get_ber_summary())
    print(tb.get_adaptive_params())
    tb.stop()
    tb.wait()
