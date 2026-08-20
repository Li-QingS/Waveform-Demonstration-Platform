"""统一 QAM 调制/解调（移植自 simple_fdidm_rx.py 的 Gray QAM，数学不变）。"""
from __future__ import annotations

from typing import Tuple

import numpy as np


_MOD_ORDER = {"QPSK": 4, "16QAM": 16, "64QAM": 64}


def bits_per_symbol(mod_order: str) -> int:
    text = str(mod_order or "QPSK").upper()
    if text not in _MOD_ORDER:
        raise ValueError(f"Unsupported modulation order: {mod_order}")
    return int(np.log2(_MOD_ORDER[text]))


def _bits_to_int(bits) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return int(out)


def _gray_to_binary(gray: int) -> int:
    b = int(gray)
    while gray >> 1:
        gray >>= 1
        b ^= gray
    return int(b)


def constellation(mod_order: str) -> Tuple[np.ndarray, np.ndarray]:
    """返回（星座点, 位标签），与 FDIDMTransceiver._build_gray_qam 一致。"""
    order = _MOD_ORDER[str(mod_order).upper()]
    root = int(np.sqrt(order))
    if root * root != order:
        raise ValueError("Only square QAM is supported")
    bits_axis = int(np.log2(root))
    bits_total = 2 * bits_axis
    levels = np.arange(-(root - 1), root, 2, dtype=np.float64)
    points = np.zeros(order, dtype=np.complex128)
    labels = np.zeros((order, bits_total), dtype=np.uint8)
    for idx in range(order):
        bits = ((idx >> np.arange(bits_total - 1, -1, -1)) & 1).astype(np.uint8)
        i_gray = _bits_to_int(bits[:bits_axis])
        q_gray = _bits_to_int(bits[bits_axis:])
        i_bin = _gray_to_binary(i_gray)
        q_bin = _gray_to_binary(q_gray)
        points[idx] = levels[i_bin] + 1j * levels[q_bin]
        labels[idx] = bits
    points /= np.sqrt(np.mean(np.abs(points) ** 2) + 1e-15)
    return points, labels


def qam_modulate(bits, mod_order: str = "16QAM") -> np.ndarray:
    bits_arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    bps = bits_per_symbol(mod_order)
    if bits_arr.size == 0:
        return np.zeros(0, dtype=np.complex128)
    pad = (-bits_arr.size) % bps
    if pad:
        bits_arr = np.concatenate([bits_arr, np.zeros(pad, dtype=np.uint8)])
    groups = bits_arr.reshape(-1, bps)
    idx = np.zeros(groups.shape[0], dtype=np.int64)
    for k in range(groups.shape[1]):
        idx = (idx << 1) | groups[:, k].astype(np.int64)
    points, _ = constellation(mod_order)
    return points[idx]


def qam_demodulate(symbols, mod_order: str = "16QAM") -> np.ndarray:
    sym = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if sym.size == 0:
        return np.zeros(0, dtype=np.uint8)
    points, labels = constellation(mod_order)
    dist = np.abs(sym[:, None] - points[None, :])
    idx = np.argmin(dist, axis=1)
    return labels[idx].reshape(-1).astype(np.uint8)


def hard_decision_symbols(symbols, mod_order: str = "16QAM") -> np.ndarray:
    sym = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if sym.size == 0:
        return sym
    points, _ = constellation(mod_order)
    idx = np.argmin(np.abs(sym[:, None] - points[None, :]), axis=1)
    return points[idx]

