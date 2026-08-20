"""有界 NumPy 环形缓冲（从 fdidm_hardtest.py 搬移，阶段 6b）。"""
from __future__ import annotations

import threading

import numpy as np


class SampleRing:
    """Bounded NumPy ring buffer for complex64 streams.

    Avoids GNU Radio vector_sink_c .data()/reset() and per-sample deque/list
    conversions in the live UHD path. Keeping the Python work bounded is
    essential for avoiding B210 U/O at modest sample rates on Windows.
    """

    def __init__(self, capacity: int):
        self.capacity = int(max(1024, capacity))
        self._buf = np.zeros(self.capacity, dtype=np.complex64)
        self._pos = 0
        self._count = 0
        self._total = 0
        self._lock = threading.Lock()

    def clear(self):
        with self._lock:
            self._pos = 0
            self._count = 0
            self._total = 0
            self._buf.fill(0)

    reset = clear

    def __len__(self):
        with self._lock:
            return int(self._count)

    @property
    def total_written(self) -> int:
        with self._lock:
            return int(self._total)

    def write(self, samples: np.ndarray):
        x = np.asarray(samples, dtype=np.complex64).reshape(-1)
        original_n = int(x.size)
        if original_n <= 0:
            return
        with self._lock:
            self._total += original_n
            if original_n >= self.capacity:
                self._buf[:] = x[-self.capacity:]
                self._pos = 0
                self._count = self.capacity
                return
            end = self._pos + original_n
            if end <= self.capacity:
                self._buf[self._pos:end] = x
            else:
                first = self.capacity - self._pos
                self._buf[self._pos:] = x[:first]
                self._buf[:end - self.capacity] = x[first:]
            self._pos = end % self.capacity
            self._count = min(self.capacity, self._count + original_n)

    def read_latest(self, n: int) -> tuple[np.ndarray, int, int]:
        n = int(max(0, n))
        with self._lock:
            k = min(n, self._count)
            total = int(self._total)
            count = int(self._count)
            if k <= 0:
                return np.zeros(0, dtype=np.complex64), total, count
            start = (self._pos - k) % self.capacity
            if start + k <= self.capacity:
                out = self._buf[start:start + k].copy()
            else:
                first = self.capacity - start
                out = np.concatenate((self._buf[start:].copy(), self._buf[:k - first].copy()))
            return out, total, count

