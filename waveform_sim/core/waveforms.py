"""统一波形接口：四种波形通过 modulate()/demodulate() 运行。"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import transforms
from .config import WaveformConfig


class Waveform:
    name = "BASE"

    def __init__(self, config: WaveformConfig):
        self.config = config.normalized()

    @property
    def symbol_capacity(self) -> int:
        return int(self.config.m_subcarriers * self.config.n_symbols)

    def modulate(self, symbols) -> np.ndarray:
        raise NotImplementedError

    def demodulate(self, samples, n_symbols: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError


class FDIDMWaveform(Waveform):
    name = "FDIDM"

    def modulate(self, symbols) -> np.ndarray:
        return transforms.fdidm_modulate(
            symbols,
            self.config.m_subcarriers,
            self.config.n_symbols,
            self.config.alpha,
            self.config.beta,
        )

    def demodulate(self, samples, n_symbols: Optional[int] = None) -> np.ndarray:
        out = transforms.fdidm_demodulate(
            samples,
            self.config.m_subcarriers,
            self.config.n_symbols,
            self.config.alpha,
            self.config.beta,
        )
        return out if n_symbols is None else out[: int(n_symbols)]


class OFDMWaveform(Waveform):
    name = "OFDM"

    @property
    def symbol_capacity(self) -> int:
        return int(self.config.fft_size)

    def modulate(self, symbols) -> np.ndarray:
        return transforms.ofdm_modulate(symbols, self.config.fft_size, self.config.cp_len)

    def demodulate(self, samples, n_symbols: Optional[int] = None) -> np.ndarray:
        return transforms.ofdm_demodulate(samples, self.config.fft_size, self.config.cp_len, n_symbols)


class OTFSWaveform(Waveform):
    name = "OTFS"

    def modulate(self, symbols) -> np.ndarray:
        return transforms.otfs_modulate(
            symbols,
            self.config.m_subcarriers,
            self.config.n_symbols,
            self.config.fft_size,
            self.config.cp_len,
        )

    def demodulate(self, samples, n_symbols: Optional[int] = None) -> np.ndarray:
        out = transforms.otfs_demodulate(
            samples,
            self.config.m_subcarriers,
            self.config.n_symbols,
            self.config.fft_size,
            self.config.cp_len,
        )
        return out if n_symbols is None else out[: int(n_symbols)]


class AFDMWaveform(Waveform):
    name = "AFDM"

    @property
    def symbol_capacity(self) -> int:
        return int(self.config.fft_size)

    def modulate(self, symbols) -> np.ndarray:
        return transforms.afdm_modulate(
            symbols, self.config.fft_size, self.config.cp_len, self.config.c1, self.config.c2
        )

    def demodulate(self, samples, n_symbols: Optional[int] = None) -> np.ndarray:
        return transforms.afdm_demodulate(
            samples, self.config.fft_size, self.config.cp_len, self.config.c1, self.config.c2, n_symbols
        )


def create_waveform(config: WaveformConfig) -> Waveform:
    name = str(config.waveform or "FDIDM").upper()
    if name == "FDIDM":
        return FDIDMWaveform(config)
    if name == "OFDM":
        return OFDMWaveform(config)
    if name == "OTFS":
        return OTFSWaveform(config)
    if name == "AFDM":
        return AFDMWaveform(config)
    raise ValueError(f"Unsupported waveform: {config.waveform}")

