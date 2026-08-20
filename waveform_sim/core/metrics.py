"""链路指标：BER / SER / EVM 与 LinkMetrics。"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class LinkMetrics:
    frame_id: int = 0
    ber: float = 1.0
    fer: float = 1.0
    ser: float = 1.0
    evm_db: float = 0.0
    evm_rms: float = 1.0
    snr_db: float = 0.0
    sync_metric: float = 0.0
    cfo_est_hz: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    frames_processed: int = 0
    frames_decode_ok: int = 0
    bits_total: int = 0
    bit_errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def bit_error_rate(tx_bits, rx_bits) -> Tuple[float, int, int]:
    a = np.asarray(tx_bits, dtype=np.uint8).reshape(-1)
    b = np.asarray(rx_bits, dtype=np.uint8).reshape(-1)
    n = int(min(a.size, b.size))
    if n <= 0:
        return 1.0, 0, 0
    errors = int(np.count_nonzero(a[:n] != b[:n]))
    return float(errors / n), errors, n


def symbol_error_rate(tx_symbols, rx_symbols) -> float:
    a = np.asarray(tx_symbols).reshape(-1)
    b = np.asarray(rx_symbols).reshape(-1)
    n = int(min(a.size, b.size))
    if n <= 0:
        return 1.0
    return float(np.count_nonzero(a[:n] != b[:n]) / n)


def evm(reference, estimate) -> Tuple[float, float]:
    ref = np.asarray(reference, dtype=np.complex128).reshape(-1)
    est = np.asarray(estimate, dtype=np.complex128).reshape(-1)
    n = int(min(ref.size, est.size))
    if n <= 0:
        return 1.0, 0.0
    err = est[:n] - ref[:n]
    denom = float(np.mean(np.abs(ref[:n]) ** 2)) + 1e-12
    rms = float(np.sqrt(np.mean(np.abs(err) ** 2) / denom))
    db = float(20.0 * math.log10(max(rms, 1e-12)))
    return rms, db

