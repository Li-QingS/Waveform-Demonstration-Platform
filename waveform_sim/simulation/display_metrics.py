# -*- coding: utf-8 -*-
"""Display-rate decimation and log-domain smoothing for live link metrics."""
from __future__ import annotations

import math
import time
from typing import Dict, Mapping, Optional

import numpy as np


class LogMetricResampler:
    """Accept fast raw metrics and emit at a slower, stable display rate.

    SER/EVM-like positive metrics are smoothed in log10 space so a single large
    outlier does not dominate a line that spans several decades.  Simulation
    data remain untouched; only the UI history is decimated.
    """

    def __init__(self, interval_s: float = 0.5, ema_alpha: float = 0.28):
        self.interval_s = float(max(0.05, interval_s))
        self.ema_alpha = float(np.clip(float(ema_alpha), 0.01, 1.0))
        self._last_emit_monotonic = -1e30
        self._log_state: Dict[str, float] = {}

    def configure(self, interval_s: Optional[float] = None, ema_alpha: Optional[float] = None) -> None:
        if interval_s is not None:
            self.interval_s = float(max(0.05, interval_s))
        if ema_alpha is not None:
            self.ema_alpha = float(np.clip(float(ema_alpha), 0.01, 1.0))

    def reset(self) -> None:
        self._last_emit_monotonic = -1e30
        self._log_state.clear()

    def ready(self, now_monotonic: Optional[float] = None, force: bool = False) -> bool:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        return bool(force or now - self._last_emit_monotonic >= self.interval_s)

    def process(
        self,
        metrics: Mapping[str, float],
        now_monotonic: Optional[float] = None,
        force: bool = False,
    ) -> Optional[Dict[str, float]]:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if not self.ready(now, force=force):
            return None
        self._last_emit_monotonic = now
        out: Dict[str, float] = {}
        for key, value in dict(metrics).items():
            try:
                x = float(value)
            except Exception:
                out[str(key)] = value  # type: ignore[assignment]
                continue
            if not np.isfinite(x) or x <= 0.0:
                out[str(key)] = float("nan")
                continue
            logx = math.log10(max(x, 1e-300))
            old = self._log_state.get(str(key))
            smoothed = logx if old is None else self.ema_alpha * logx + (1.0 - self.ema_alpha) * old
            self._log_state[str(key)] = float(smoothed)
            out[str(key)] = float(10.0 ** smoothed)
        return out
