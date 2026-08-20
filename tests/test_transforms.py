import numpy as np

from waveform_sim.core import transforms
from waveform_sim.simulation.simple_fdidm_rx import FDIDMTransceiver
from waveform_sim.simulation.simple_otfs_rx import OTFSTransceiver
from waveform_sim.simulation.simple_afdm_rx import _build_afdm_mats


def test_fdidm_gamma_matches_legacy():
    tc = FDIDMTransceiver(m_subcarriers=8, n_symbols=8)
    for n, eps in [(4, 0.0), (8, 0.3), (16, -0.5)]:
        assert np.allclose(transforms.gamma_matrix(n, eps), tc._gamma(n, eps), atol=1e-12)


def test_fdidm_roundtrip():
    rng = np.random.default_rng(7)
    x = (rng.standard_normal(64) + 1j * rng.standard_normal(64)).astype(np.complex128)
    y = transforms.fdidm_modulate(x, 8, 8, 0.4, -0.2)
    x2 = transforms.fdidm_demodulate(y, 8, 8, 0.4, -0.2)
    assert np.allclose(x2, x, atol=1e-9)


def test_ofdm_roundtrip():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((7, 64)) + 1j * rng.standard_normal((7, 64))
    y = transforms.ofdm_modulate(x.reshape(-1), 64, 16)
    x2 = transforms.ofdm_demodulate(y, 64, 16, n_symbols=x.size)
    assert np.allclose(x2, x.reshape(-1), atol=1e-9)


def test_otfs_matches_legacy():
    tc = OTFSTransceiver()
    rng = np.random.default_rng(9)
    dd = rng.standard_normal((64, 8)) + 1j * rng.standard_normal((64, 8))
    ref = tc._tf_to_time_cp(tc._dd_to_tf(dd))
    new = transforms.otfs_modulate(dd.reshape(-1), 64, 8, 64, tc.cp_len)
    assert np.allclose(new, ref, atol=1e-9)


def test_otfs_roundtrip():
    rng = np.random.default_rng(10)
    dd = rng.standard_normal((64, 8)) + 1j * rng.standard_normal((64, 8))
    y = transforms.otfs_modulate(dd.reshape(-1), 64, 8, 64, 16)
    x2 = transforms.otfs_demodulate(y, 64, 8, 64, 16)
    assert np.allclose(x2, dd.reshape(-1), atol=1e-9)


def test_afdm_matrices_match_legacy():
    for N, c1, c2 in [(64, 0.05, 0.05), (32, 0.04, 0.03)]:
        at, ar = transforms.afdm_matrices(N, c1, c2)
        lt, lr = _build_afdm_mats(N, c1, c2)
        assert np.allclose(at, lt, atol=1e-12)
        assert np.allclose(ar, lr, atol=1e-12)


def test_afdm_roundtrip():
    rng = np.random.default_rng(11)
    x = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    y = transforms.afdm_modulate(x, 64, 16, 0.05, 0.05)
    x2 = transforms.afdm_demodulate(y, 64, 16, 0.05, 0.05)
    assert np.allclose(x2, x, atol=1e-9)

