"""RF 安全策略：loopback / OTA 增益、衰减器、频段白名单校验（阶段 5）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..core.config import HardwareConfig


class RFSafetyViolation(ValueError):
    pass


@dataclass
class RFSafetyDecision:
    allowed: bool
    warnings: List[str]


class RFSafetyPolicy:
    def __init__(self, strict: bool = True):
        self.strict = bool(strict)

    def validate(self, config: HardwareConfig) -> RFSafetyDecision:
        cfg = config.normalized()
        warnings: List[str] = []
        if cfg.mode == "loopback":
            if cfg.tx_gain_db > cfg.max_loopback_tx_gain_db:
                warnings.append(
                    f"Loopback TX gain {cfg.tx_gain_db:.1f} dB exceeds limit {cfg.max_loopback_tx_gain_db:.1f} dB."
                )
            if cfg.attenuator_db < 20.0:
                warnings.append("Loopback mode requires at least 20 dB external attenuation.")
        elif cfg.mode == "ota":
            if not cfg.ota_confirmed:
                warnings.append("OTA mode requires explicit antenna/environment confirmation.")
            if cfg.tx_gain_db > cfg.max_ota_tx_gain_db:
                warnings.append(f"OTA TX gain {cfg.tx_gain_db:.1f} dB exceeds limit {cfg.max_ota_tx_gain_db:.1f} dB.")
            if not self._freq_allowed(cfg.center_freq_hz, cfg.frequency_whitelist_hz):
                warnings.append(f"Center frequency {cfg.center_freq_hz:.0f} Hz is outside configured whitelist.")
        else:
            warnings.append(f"Unknown RF mode: {cfg.mode}")
        if self.strict and warnings:
            raise RFSafetyViolation("RF safety policy blocked run: " + "; ".join(warnings))
        return RFSafetyDecision(allowed=not warnings, warnings=warnings)

    @staticmethod
    def _freq_allowed(freq_hz: float, whitelist: List[List[float]]) -> bool:
        f = float(freq_hz)
        for lo, hi in whitelist or []:
            if float(lo) <= f <= float(hi):
                return True
        return False

