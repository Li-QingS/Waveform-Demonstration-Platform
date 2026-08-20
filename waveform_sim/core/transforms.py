"""四类波形统一变换（移植自现有 simple_*_rx.py 的数学，算法不变）。"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# FDIDM（分数阶傅里叶型索引调制）
# --------------------------------------------------------------------------
def unitary_dft(n: int) -> np.ndarray:
    k = np.arange(n).reshape(-1, 1)
    m = np.arange(n).reshape(1, -1)
    return np.exp(-1j * 2.0 * np.pi * k * m / n) / np.sqrt(n)


def gamma_matrix(n: int, eps: float) -> np.ndarray:
    eps = float(np.round(eps, 12))
    F = unitary_dft(n)
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
    return G


def fdidm_modulate(
    symbols, m: int, n: int, alpha: float = 0.0, beta: float = 0.0
) -> np.ndarray:
    x = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    tx = np.kron(gamma_matrix(n, -beta), gamma_matrix(m, alpha))
    return tx @ x


def fdidm_demodulate(
    samples, m: int, n: int, alpha: float = 0.0, beta: float = 0.0
) -> np.ndarray:
    y = np.asarray(samples, dtype=np.complex128).reshape(-1)
    rx = np.kron(gamma_matrix(n, beta), gamma_matrix(m, -alpha))
    return rx @ y


# --------------------------------------------------------------------------
# OFDM（IFFT + CP）
# --------------------------------------------------------------------------
def ofdm_modulate(symbols, fft_size: int = 64, cp_len: int = 16) -> np.ndarray:
    arr = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    fft_size = int(max(1, fft_size))
    cp_len = int(max(0, min(cp_len, fft_size)))
    pad = (-arr.size) % fft_size
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.complex128)])
    grid = arr.reshape(-1, fft_size)
    td = np.fft.ifft(grid, axis=1) * np.sqrt(fft_size)
    if cp_len:
        td = np.concatenate([td[:, -cp_len:], td], axis=1)
    return td.reshape(-1).astype(np.complex128, copy=False)


def ofdm_demodulate(
    samples, fft_size: int = 64, cp_len: int = 16, n_symbols: Optional[int] = None
) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.complex128).reshape(-1)
    fft_size = int(max(1, fft_size))
    cp_len = int(max(0, min(cp_len, fft_size)))
    block = fft_size + cp_len
    usable = (arr.size // block) * block
    if usable == 0:
        return np.zeros(0, dtype=np.complex128)
    grid = arr[:usable].reshape(-1, block)
    if cp_len:
        grid = grid[:, cp_len:]
    freq = np.fft.fft(grid, axis=1) / np.sqrt(fft_size)
    out = freq.reshape(-1)
    if n_symbols is not None:
        return out[: int(n_symbols)]
    return out


# --------------------------------------------------------------------------
# OTFS（DD ↔ TF + 每列 IFFT/CP）
# --------------------------------------------------------------------------
def otfs_modulate(
    symbols, m: int, n: int, fft_size: Optional[int] = None, cp_len: int = 0
) -> np.ndarray:
    m = int(m)
    n = int(n)
    grid = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    total = m * n
    if grid.size < total:
        grid = np.concatenate([grid, np.zeros(total - grid.size, dtype=np.complex128)])
    dd = grid[:total].reshape(m, n)
    tf = np.fft.ifft(np.fft.fft(dd, axis=1), axis=0)
    cp_len = int(max(0, min(cp_len, m)))
    out = []
    for col in range(tf.shape[1]):
        td = np.fft.ifft(tf[:, col], axis=0) * np.sqrt(m)
        if cp_len:
            td = np.concatenate([td[-cp_len:], td])
        out.append(td)
    return np.concatenate(out).astype(np.complex128, copy=False)


def otfs_demodulate(
    samples, m: int, n: int, fft_size: Optional[int] = None, cp_len: int = 0
) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.complex128).reshape(-1)
    m = int(m)
    n = int(n)
    cp_len = int(max(0, min(cp_len, m)))
    block = m + cp_len
    need = n * block
    arr = arr[:need]
    if arr.size < block:
        return np.zeros(m * n, dtype=np.complex128)
    slots = arr.reshape(n, block)
    if cp_len:
        slots = slots[:, cp_len:]
    tf = np.fft.fft(slots, axis=1) / np.sqrt(m)  # (n, m)
    dd = np.fft.ifft(np.fft.fft(tf.T, axis=0), axis=1)  # (m, n)
    return dd.reshape(-1)


# --------------------------------------------------------------------------
# AFDM（chirp + DFT 矩阵）
# --------------------------------------------------------------------------
def afdm_matrices(n: int, c1: float, c2: float) -> Tuple[np.ndarray, np.ndarray]:
    n = int(n)
    k = np.arange(n)
    F = np.fft.fft(np.eye(n), axis=0) / np.sqrt(n)
    Fh = F.conj().T
    D1 = np.diag(np.exp(-1j * 2.0 * np.pi * float(c1) * (k ** 2) / n))
    D2 = np.diag(np.exp(-1j * 2.0 * np.pi * float(c2) * (k ** 2) / n))
    A_tx = D1.conj().T @ Fh @ D2.conj().T
    A_rx = D2 @ F @ D1
    return A_tx, A_rx


def afdm_modulate(
    symbols, fft_size: int = 64, cp_len: int = 16, c1: float = 0.05, c2: float = 0.05
) -> np.ndarray:
    arr = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    fft_size = int(max(1, fft_size))
    cp_len = int(max(0, min(cp_len, fft_size)))
    A_tx, _ = afdm_matrices(fft_size, c1, c2)
    pad = (-arr.size) % fft_size
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.complex128)])
    grid = arr.reshape(-1, fft_size)
    td = (A_tx @ grid.T).T
    if cp_len:
        td = np.concatenate([td[:, -cp_len:], td], axis=1)
    return td.reshape(-1).astype(np.complex128, copy=False)


def afdm_demodulate(
    samples,
    fft_size: int = 64,
    cp_len: int = 16,
    c1: float = 0.05,
    c2: float = 0.05,
    n_symbols: Optional[int] = None,
) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.complex128).reshape(-1)
    fft_size = int(max(1, fft_size))
    cp_len = int(max(0, min(cp_len, fft_size)))
    block = fft_size + cp_len
    usable = (arr.size // block) * block
    if usable == 0:
        return np.zeros(0, dtype=np.complex128)
    grid = arr[:usable].reshape(-1, block)
    if cp_len:
        grid = grid[:, cp_len:]
    _, A_rx = afdm_matrices(fft_size, c1, c2)
    freq = (A_rx @ grid.T).T
    out = freq.reshape(-1)
    if n_symbols is not None:
        return out[: int(n_symbols)]
    return out

