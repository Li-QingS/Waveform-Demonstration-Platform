"""硬件传输抽象：simulated-loopback 与 UHD 占位（阶段 5，只增不改）。"""
from __future__ import annotations

import numpy as np

from ..core.config import HardwareConfig


class BaseTransport:
    def __init__(self, config: HardwareConfig):
        self.config = config.normalized()
        self.running = False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def send(self, samples):
        raise NotImplementedError

    def recv(self, count: int):
        raise NotImplementedError


class SimulatedLoopbackTransport(BaseTransport):
    def __init__(self, config: HardwareConfig):
        super().__init__(config)
        self._buffer = np.zeros(0, dtype=np.complex128)

    def send(self, samples):
        self._buffer = np.asarray(samples, dtype=np.complex128).reshape(-1).copy()

    def recv(self, count: int):
        count = int(max(0, count))
        if self._buffer.size == 0:
            return np.zeros(count, dtype=np.complex128)
        reps = int(np.ceil(count / self._buffer.size))
        return np.tile(self._buffer, reps)[:count]


class UHDTransport(BaseTransport):
    """UHD 真机占位：无 UHD 环境给出明确错误；流式逻辑在阶段 6 接入。"""

    def __init__(self, config: HardwareConfig):
        super().__init__(config)
        try:
            import uhd  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "UHD Python bindings are not available. Install UHD/GNU Radio or use simulated-loopback."
            ) from exc

    def send(self, samples):
        raise NotImplementedError("Real UHD streaming will be wired in phase 6")

    def recv(self, count: int):
        raise NotImplementedError("Real UHD streaming will be wired in phase 6")


def create_transport(config: HardwareConfig) -> BaseTransport:
    cfg = config.normalized()
    if cfg.transport in {"sim", "simulated", "simulated-loopback", "loopback"}:
        return SimulatedLoopbackTransport(cfg)
    if cfg.transport in {"uhd", "usrp"}:
        return UHDTransport(cfg)
    raise ValueError(f"Unsupported transport: {cfg.transport}")

