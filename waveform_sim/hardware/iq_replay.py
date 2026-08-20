"""IQ 回放：.npy 与 complex64 原始文件（阶段 5）。"""
from __future__ import annotations

from pathlib import Path
import numpy as np


class IQReplaySource:
    def __init__(self, path: str | Path, loop: bool = True):
        self.path = Path(path)
        self.loop = bool(loop)
        self.samples = self._load(self.path)
        self.offset = 0

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            arr = np.load(path)
        else:
            arr = np.fromfile(path, dtype=np.complex64)
        return np.asarray(arr, dtype=np.complex128).reshape(-1)

    def read(self, count: int) -> np.ndarray:
        count = int(max(0, count))
        if count == 0 or self.samples.size == 0:
            return np.zeros(0, dtype=np.complex128)
        end = self.offset + count
        if end <= self.samples.size:
            out = self.samples[self.offset:end]
            self.offset = end
            return out.copy()
        if not self.loop:
            out = self.samples[self.offset:]
            self.offset = self.samples.size
            return out.copy()
        parts = [self.samples[self.offset:]]
        remaining = count - parts[0].size
        while remaining > 0:
            take = min(remaining, self.samples.size)
            parts.append(self.samples[:take])
            remaining -= take
        self.offset = count % self.samples.size
        return np.concatenate(parts).astype(np.complex128)


def save_iq(path: str | Path, samples) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(samples, dtype=np.complex64)
    if path.suffix.lower() == ".npy":
        np.save(path, arr)
    else:
        arr.tofile(path)

