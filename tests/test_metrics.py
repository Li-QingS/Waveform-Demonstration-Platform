import numpy as np

from waveform_sim.core.metrics import LinkMetrics, bit_error_rate, evm, symbol_error_rate


def test_bit_error_rate():
    a = np.array([0, 1, 0, 1], dtype=np.uint8)
    b = np.array([0, 1, 1, 1], dtype=np.uint8)
    ber, errors, total = bit_error_rate(a, b)
    assert ber == 0.25
    assert errors == 1
    assert total == 4


def test_symbol_error_rate():
    a = np.array([1 + 0j, 0 + 1j])
    b = np.array([1 + 0j, 1 + 0j])
    assert symbol_error_rate(a, b) == 0.5


def test_evm():
    ref = np.array([1.0 + 0.0j, 0.0 + 1.0j])
    est = ref + 0.1 * np.array([1.0 + 0.0j, 0.0 + 1.0j])
    rms, db = evm(ref, est)
    assert abs(rms - 0.1) < 1e-9
    assert abs(db - 20.0 * np.log10(0.1)) < 1e-9


def test_link_metrics_to_dict():
    m = LinkMetrics(ber=0.1, evm_db=-20.0, alpha=0.5, beta=0.25)
    d = m.to_dict()
    assert d["ber"] == 0.1
    assert d["alpha"] == 0.5

