import numpy as np

from waveform_sim.core import modem
from waveform_sim.simulation.simple_fdidm_rx import FDIDMTransceiver


def test_qam_roundtrip():
    rng = np.random.default_rng(123)
    for mod in ["QPSK", "16QAM", "64QAM"]:
        bits = rng.integers(0, 2, size=modem.bits_per_symbol(mod) * 32, dtype=np.uint8)
        syms = modem.qam_modulate(bits, mod)
        out = modem.qam_demodulate(syms, mod)[: bits.size]
        assert np.array_equal(bits, out)


def test_constellation_matches_fdidm():
    for order, mod in [(4, "QPSK"), (16, "16QAM"), (64, "64QAM")]:
        pts, labels = modem.constellation(mod)
        ref_pts, ref_labels = FDIDMTransceiver._build_gray_qam(order)
        assert np.allclose(pts, ref_pts, atol=1e-12)
        assert np.array_equal(labels, ref_labels)

