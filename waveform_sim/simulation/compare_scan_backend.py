
# -*- coding: utf-8 -*-
"""
compare_scan_backend.py

独立的三波形扫描仿真后端。

这个文件不调用 simple_ofdm_rx/simple_otfs_rx/simple_afdm_rx 中的 BER。
它专门用于“波形对比/自动扫描”页面，按统一物理信道和统一 Monte-Carlo
统计方式生成 BER-CFO 与 BER-Doppler 曲线。

建模口径：
1. 统一 64 点基带离散系统，子载波间隔 15 kHz，采样率 960 kSa/s；
2. 统一 7 个数据单元，每单元 48 个数据符号；
3. 统一 QAM 映射、统一 3 径时变多径、统一 CFO、统一 Doppler；
4. OFDM 接收机只做单抽头频域均衡，CFO/Doppler 产生的 ICI 不被完全消除；
5. OTFS 接收机用 DD 域完整 LMMSE 检测；
6. AFDM 接收机用 DAFT 域完整 LMMSE 检测；
7. 结果来自符号级 Monte-Carlo 误码统计，而不是人工 penalty 曲线。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


MATLAB_COLORS = {
    "OFDM": (0, 114, 189),
    "OTFS": (217, 83, 25),
    "AFDM": (119, 172, 48),
}


def _bits_per_symbol(mod_order: str) -> int:
    mod_order = str(mod_order).upper()
    if mod_order == "QPSK":
        return 2
    if mod_order == "16QAM":
        return 4
    if mod_order == "64QAM":
        return 6
    raise ValueError(f"unsupported modulation: {mod_order}")


def _qam_mod(bits: np.ndarray, mod_order: str) -> np.ndarray:
    mod_order = str(mod_order).upper()
    bits = np.asarray(bits, dtype=np.int8).reshape(-1)
    bps = _bits_per_symbol(mod_order)
    n = len(bits) // bps
    b = bits[: n * bps].reshape(n, bps)

    if mod_order == "QPSK":
        i = 1 - 2 * b[:, 0]
        q = 1 - 2 * b[:, 1]
        return (i + 1j * q) / np.sqrt(2.0)

    if mod_order == "16QAM":
        def pam2(x0, x1):
            code = 2 * x0 + x1
            table = np.array([-3.0, -1.0, 3.0, 1.0])
            return table[code]
        i = pam2(b[:, 0], b[:, 1])
        q = pam2(b[:, 2], b[:, 3])
        return (i + 1j * q) / np.sqrt(10.0)

    if mod_order == "64QAM":
        # 3-bit Gray PAM: 000,-7; 001,-5; 011,-3; 010,-1; 110,1; 111,3; 101,5; 100,7
        table = {
            0: -7.0, 1: -5.0, 3: -3.0, 2: -1.0,
            6: 1.0, 7: 3.0, 5: 5.0, 4: 7.0,
        }
        ci = 4 * b[:, 0] + 2 * b[:, 1] + b[:, 2]
        cq = 4 * b[:, 3] + 2 * b[:, 4] + b[:, 5]
        i = np.array([table[int(x)] for x in ci], dtype=np.float64)
        q = np.array([table[int(x)] for x in cq], dtype=np.float64)
        return (i + 1j * q) / np.sqrt(42.0)

    raise ValueError(f"unsupported modulation: {mod_order}")


def _qam_demod(symbols: np.ndarray, mod_order: str) -> np.ndarray:
    mod_order = str(mod_order).upper()
    s = np.asarray(symbols, dtype=np.complex128).reshape(-1)

    if mod_order == "QPSK":
        out = np.empty((len(s), 2), dtype=np.int8)
        z = s * np.sqrt(2.0)
        out[:, 0] = (np.real(z) < 0).astype(np.int8)
        out[:, 1] = (np.imag(z) < 0).astype(np.int8)
        return out.reshape(-1)

    if mod_order == "16QAM":
        z = s * np.sqrt(10.0)
        levels = np.array([-3.0, -1.0, 1.0, 3.0])
        table = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int8)

        idx_i = np.argmin((np.real(z)[:, None] - levels[None, :]) ** 2, axis=1)
        idx_q = np.argmin((np.imag(z)[:, None] - levels[None, :]) ** 2, axis=1)
        bi = table[idx_i]
        bq = table[idx_q]
        return np.column_stack([bi[:, 0], bi[:, 1], bq[:, 0], bq[:, 1]]).reshape(-1)

    if mod_order == "64QAM":
        z = s * np.sqrt(42.0)
        levels = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
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
        idx_i = np.argmin((np.real(z)[:, None] - levels[None, :]) ** 2, axis=1)
        idx_q = np.argmin((np.imag(z)[:, None] - levels[None, :]) ** 2, axis=1)
        bi = table[idx_i]
        bq = table[idx_q]
        return np.column_stack([bi[:, 0], bi[:, 1], bi[:, 2], bq[:, 0], bq[:, 1], bq[:, 2]]).reshape(-1)

    raise ValueError(f"unsupported modulation: {mod_order}")


def _normalized_sparse_channel(rng: np.random.Generator, doppler_hz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    delays = np.array([0, 2, 5], dtype=np.int64)
    gain_db = np.array([0.0, -3.5, -7.0], dtype=np.float64)
    amp = 10.0 ** (gain_db / 20.0)
    fading = (rng.standard_normal(3) + 1j * rng.standard_normal(3)) / np.sqrt(2.0)
    gains = amp * fading
    gains = gains / np.sqrt(np.sum(np.abs(gains) ** 2) + 1e-12)
    fd = float(max(0.0, doppler_hz)) * np.array([0.0, 0.6, -0.8], dtype=np.float64)
    return delays, gains.astype(np.complex128), fd.astype(np.float64)


@dataclass
class ScanConfig:
    fft_len: int = 64
    n_slots: int = 7
    cp_len: int = 16
    subcarrier_spacing: float = 15e3
    lmmse_diag_loading: float = 1e-8


class WaveformScanSimulator:
    """独立扫描仿真器：输入波形和信道参数，返回 BER。"""

    def __init__(self, config: ScanConfig | None = None):
        self.cfg = config or ScanConfig()
        self.M = int(self.cfg.fft_len)
        self.T = int(self.cfg.n_slots)
        self.sample_rate = self.M * float(self.cfg.subcarrier_spacing)

        self.F = np.fft.fft(np.eye(self.M), axis=0) / np.sqrt(self.M)
        self.Fh = self.F.conj().T

        # shifted 坐标中的 52 个 active、4 个 pilot、48 个 data
        active_bins = np.concatenate([np.arange(-26, 0), np.arange(1, 27)])
        pilot_bins = np.array([-21, -7, 7, 21])
        active_shift_idx = active_bins + self.M // 2
        pilot_shift_idx = pilot_bins + self.M // 2
        pilot_set = set(pilot_shift_idx.tolist())
        data_shift_idx = np.array([x for x in active_shift_idx if x not in pilot_set], dtype=np.int64)

        self.data_shift_idx = data_shift_idx
        self.data_nat_idx = (self.data_shift_idx + self.M // 2) % self.M
        self.n_data = len(self.data_shift_idx)
        self.net_symbols = self.n_data * self.T

        self.A_tx, self.A_rx = self._build_afdm_mats(c1=0.05, c2=0.05)
        self.U_dd_to_tf, self.U_tf_to_dd = self._build_otfs_mats()

    def _build_afdm_mats(self, c1: float, c2: float):
        n = np.arange(self.M)
        D1 = np.diag(np.exp(-1j * 2.0 * np.pi * c1 * (n ** 2) / self.M))
        D2 = np.diag(np.exp(-1j * 2.0 * np.pi * c2 * (n ** 2) / self.M))
        A_tx = D1.conj().T @ self.Fh @ D2.conj().T
        A_rx = D2 @ self.F @ D1
        return A_tx.astype(np.complex128), A_rx.astype(np.complex128)

    def _build_otfs_mats(self):
        size = self.M * self.T

        def dd_to_tf_vec(v):
            x = np.asarray(v, dtype=np.complex128).reshape(self.M, self.T, order="F")
            y = np.fft.ifft(np.fft.fft(x, axis=1, norm="ortho"), axis=0, norm="ortho")
            return y.reshape(-1, order="F")

        def tf_to_dd_vec(v):
            x = np.asarray(v, dtype=np.complex128).reshape(self.M, self.T, order="F")
            y = np.fft.ifft(np.fft.fft(x, axis=0, norm="ortho"), axis=1, norm="ortho")
            return y.reshape(-1, order="F")

        eye = np.eye(size, dtype=np.complex128)
        U = np.column_stack([dd_to_tf_vec(eye[:, k]) for k in range(size)])
        Uh = np.column_stack([tf_to_dd_vec(eye[:, k]) for k in range(size)])
        return U.astype(np.complex128), Uh.astype(np.complex128)

    def _channel_matrix(self, delays, gains, fd_list, cfo_hz: float, slot_idx: int) -> np.ndarray:
        n = np.arange(self.M, dtype=np.float64)
        # 绝对采样偏移让 Doppler/CFO 相位随 slot 连续变化
        n_abs = slot_idx * (self.M + self.cfg.cp_len) + n
        H = np.zeros((self.M, self.M), dtype=np.complex128)

        for delay, gain, fd in zip(delays, gains, fd_list):
            d = int(delay) % self.M
            phase = np.exp(1j * 2.0 * np.pi * (float(fd) + float(cfo_hz)) * n_abs / self.sample_rate)
            D = np.diag(phase)
            S = np.zeros((self.M, self.M), dtype=np.complex128)
            for row in range(self.M):
                S[row, (row - d) % self.M] = 1.0
            H += complex(gain) * D @ S
        return H

    @staticmethod
    def _add_awgn(y: np.ndarray, ebn0_db: float, bits_per_symbol: int, rng: np.random.Generator):
        p = float(np.mean(np.abs(y) ** 2))
        ebn0 = 10.0 ** (float(ebn0_db) / 10.0)
        noise_var = max(p / max(bits_per_symbol * ebn0, 1e-12), 1e-12)
        noise = np.sqrt(noise_var / 2.0) * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape))
        return y + noise, noise_var

    def simulate_ber(
        self,
        waveform: str,
        snr_db: float,
        cfo_hz: float,
        doppler_hz: float,
        mod_order: str = "QPSK",
        frames: int = 20,
        seed: int | None = None,
    ) -> float:
        waveform = str(waveform).upper()
        mod_order = str(mod_order).upper()
        frames = int(max(1, frames))
        rng = np.random.default_rng(seed)
        bps = _bits_per_symbol(mod_order)

        total_errors = 0
        total_bits = 0

        for _frame in range(frames):
            delays, gains, fd_list = _normalized_sparse_channel(rng, doppler_hz)

            if waveform == "OFDM":
                e, b = self._simulate_ofdm_frame(rng, delays, gains, fd_list, snr_db, cfo_hz, mod_order, bps)
            elif waveform == "AFDM":
                e, b = self._simulate_afdm_frame(rng, delays, gains, fd_list, snr_db, cfo_hz, mod_order, bps)
            elif waveform == "OTFS":
                e, b = self._simulate_otfs_frame(rng, delays, gains, fd_list, snr_db, cfo_hz, mod_order, bps)
            else:
                raise ValueError(f"unsupported waveform: {waveform}")

            total_errors += int(e)
            total_bits += int(b)

        return float(total_errors / max(total_bits, 1))

    def _simulate_ofdm_frame(self, rng, delays, gains, fd_list, snr_db, cfo_hz, mod_order, bps):
        bits = rng.integers(0, 2, size=self.net_symbols * bps, dtype=np.int8)
        tx_symbols = _qam_mod(bits, mod_order).reshape(self.T, self.n_data)
        rx_symbols = []

        for t in range(self.T):
            x_nat = np.zeros(self.M, dtype=np.complex128)
            x_nat[self.data_nat_idx] = tx_symbols[t]
            tx_time = self.Fh @ x_nat

            Ht = self._channel_matrix(delays, gains, fd_list, cfo_hz=cfo_hz, slot_idx=t)
            y_time = Ht @ tx_time
            y_time, noise_var = self._add_awgn(y_time, snr_db, bps, rng)

            y_nat = self.F @ y_time
            Hf = self.F @ Ht @ self.Fh

            # OFDM baseline: 只做单抽头频域均衡，ICI 作为干扰保留。
            diag = np.diag(Hf)[self.data_nat_idx]
            x_hat = y_nat[self.data_nat_idx] * np.conj(diag) / (np.abs(diag) ** 2 + noise_var + 1e-8)
            rx_symbols.append(x_hat)

        rx_bits = _qam_demod(np.concatenate(rx_symbols), mod_order)
        n = min(len(bits), len(rx_bits))
        return int(np.count_nonzero(bits[:n] != rx_bits[:n])), int(n)

    def _simulate_afdm_frame(self, rng, delays, gains, fd_list, snr_db, cfo_hz, mod_order, bps):
        bits = rng.integers(0, 2, size=self.net_symbols * bps, dtype=np.int8)
        tx_symbols = _qam_mod(bits, mod_order).reshape(self.T, self.n_data)
        rx_symbols = []

        for t in range(self.T):
            x = np.zeros(self.M, dtype=np.complex128)
            x[self.data_shift_idx] = tx_symbols[t]
            tx_time = self.A_tx @ x

            Ht = self._channel_matrix(delays, gains, fd_list, cfo_hz=cfo_hz, slot_idx=t)
            y_time = Ht @ tx_time
            y_time, noise_var = self._add_awgn(y_time, snr_db, bps, rng)

            y_daft = self.A_rx @ y_time
            Heff = self.A_rx @ Ht @ self.A_tx
            Hd = Heff[:, self.data_shift_idx]

            gram = Hd.conj().T @ Hd + (noise_var + self.cfg.lmmse_diag_loading) * np.eye(self.n_data)
            rhs = Hd.conj().T @ y_daft
            x_hat = np.linalg.solve(gram, rhs)
            rx_symbols.append(x_hat)

        rx_bits = _qam_demod(np.concatenate(rx_symbols), mod_order)
        n = min(len(bits), len(rx_bits))
        return int(np.count_nonzero(bits[:n] != rx_bits[:n])), int(n)

    def _simulate_otfs_frame(self, rng, delays, gains, fd_list, snr_db, cfo_hz, mod_order, bps):
        bits = rng.integers(0, 2, size=self.net_symbols * bps, dtype=np.int8)
        tx_data = _qam_mod(bits, mod_order)

        dd = np.zeros((self.M, self.T), dtype=np.complex128)
        dd[self.data_shift_idx[:, None], np.arange(self.T)[None, :]] = tx_data.reshape(self.n_data, self.T, order="F")
        x_dd_vec = dd.reshape(-1, order="F")

        x_tf_vec = self.U_dd_to_tf @ x_dd_vec
        y_tf_vec = np.zeros_like(x_tf_vec)
        Htf = np.zeros((self.M * self.T, self.M * self.T), dtype=np.complex128)

        noise_vars = []
        for t in range(self.T):
            x_tf_slot = x_tf_vec[t * self.M:(t + 1) * self.M]
            tx_time = self.Fh @ x_tf_slot
            Ht = self._channel_matrix(delays, gains, fd_list, cfo_hz=cfo_hz, slot_idx=t)
            y_time = Ht @ tx_time
            y_time, noise_var = self._add_awgn(y_time, snr_db, bps, rng)
            noise_vars.append(noise_var)
            y_tf_vec[t * self.M:(t + 1) * self.M] = self.F @ y_time
            Htf[t * self.M:(t + 1) * self.M, t * self.M:(t + 1) * self.M] = self.F @ Ht @ self.Fh

        y_dd_vec = self.U_tf_to_dd @ y_tf_vec
        Heff = self.U_tf_to_dd @ Htf @ self.U_dd_to_tf
        data_flat_idx = []
        for t in range(self.T):
            for row in self.data_shift_idx:
                data_flat_idx.append(int(row + t * self.M))
        data_flat_idx = np.array(data_flat_idx, dtype=np.int64)

        Hd = Heff[:, data_flat_idx]
        noise_var = float(np.mean(noise_vars))
        gram = Hd.conj().T @ Hd + (noise_var + self.cfg.lmmse_diag_loading) * np.eye(len(data_flat_idx))
        rhs = Hd.conj().T @ y_dd_vec
        x_hat = np.linalg.solve(gram, rhs)

        rx_bits = _qam_demod(x_hat, mod_order)
        n = min(len(bits), len(rx_bits))
        return int(np.count_nonzero(bits[:n] != rx_bits[:n])), int(n)


def scan_axis(
    axis_name: str,
    axis_values: np.ndarray,
    snr_db: float,
    mod_order: str,
    frames: int,
    fixed_cfo_hz: float,
    fixed_doppler_hz: float,
    seed_base: int = 20260502,
) -> Dict[str, np.ndarray]:
    """一次性扫描某个横轴，返回三种波形 BER 数组。"""
    axis_name = str(axis_name).lower()
    sim = WaveformScanSimulator()
    out: Dict[str, np.ndarray] = {}

    for w_idx, waveform in enumerate(["OFDM", "OTFS", "AFDM"]):
        vals = []
        for p_idx, val in enumerate(np.asarray(axis_values, dtype=np.float64)):
            if axis_name == "cfo":
                cfo_hz = float(val)
                doppler_hz = float(fixed_doppler_hz)
            elif axis_name == "doppler":
                cfo_hz = float(fixed_cfo_hz)
                doppler_hz = float(val)
            else:
                raise ValueError("axis_name must be 'cfo' or 'doppler'")

            ber = sim.simulate_ber(
                waveform=waveform,
                snr_db=snr_db,
                cfo_hz=cfo_hz,
                doppler_hz=doppler_hz,
                mod_order=mod_order,
                frames=frames,
                seed=seed_base + 10000 * w_idx + 100 * p_idx,
            )
            vals.append(max(float(ber), 1e-8))
        out[waveform] = np.array(vals, dtype=np.float64)

    return out
