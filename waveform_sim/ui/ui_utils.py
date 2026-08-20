"""UI 通用静态助手（从 hardware_test_tab.py 搬移，阶段 8）。"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def has_signal(samples: np.ndarray) -> bool:
    if samples.size == 0:
        return False
    finite = np.isfinite(np.real(samples)) & np.isfinite(np.imag(samples))
    if not np.any(finite):
        return False
    return bool(np.max(np.abs(samples[finite])) > 1e-10)


def compute_spectrum(
    samples: np.ndarray, samp_rate: float, segment_len: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=np.complex128).reshape(-1)
    if samples.size == 0:
        return np.zeros(0), np.zeros(0)

    segment_len = min(max(8, int(segment_len)), samples.size)
    n_segments = max(1, samples.size // segment_len)
    trimmed = samples[-n_segments * segment_len:]
    blocks = trimmed.reshape(n_segments, segment_len)
    window = np.hanning(segment_len).astype(np.float64)
    power = np.zeros(segment_len, dtype=np.float64)
    for block in blocks:
        spectrum = np.fft.fftshift(np.fft.fft(block * window))
        power += np.abs(spectrum) ** 2
    power /= float(n_segments)
    psd_db = 10.0 * np.log10(power + 1e-12)
    freq = np.linspace(
        -float(samp_rate) / 2.0,
        float(samp_rate) / 2.0,
        segment_len,
        endpoint=False,
    )
    return freq, psd_db


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result


def format_metric(value: Any, fmt: str = ".3f", fallback: str = "nan") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(number):
        return fallback
    return format(number, fmt)

