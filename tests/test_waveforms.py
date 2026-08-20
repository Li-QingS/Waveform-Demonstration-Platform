import numpy as np
import pytest

from waveform_sim.core.config import WaveformConfig
from waveform_sim.core.waveforms import create_waveform


CASES = {
    "FDIDM": dict(waveform="FDIDM", m_subcarriers=8, n_symbols=8, alpha=0.4, beta=-0.2),
    "OFDM": dict(waveform="OFDM", fft_size=64, cp_len=16),
    "OTFS": dict(waveform="OTFS", m_subcarriers=64, n_symbols=8, fft_size=64, cp_len=16),
    "AFDM": dict(waveform="AFDM", fft_size=64, cp_len=16, c1=0.05, c2=0.05),
}


@pytest.mark.parametrize("name", ["FDIDM", "OFDM", "OTFS", "AFDM"])
def test_waveform_roundtrip(name):
    cfg = WaveformConfig(**CASES[name]).normalized()
    wf = create_waveform(cfg)
    rng = np.random.default_rng(42)
    n = wf.symbol_capacity
    symbols = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    samples = wf.modulate(symbols)
    out = wf.demodulate(samples, n_symbols=n)
    assert np.allclose(out, symbols, atol=1e-9)


def test_create_waveform_unknown():
    with pytest.raises(ValueError):
        create_waveform(WaveformConfig(waveform="UNKNOWN"))

