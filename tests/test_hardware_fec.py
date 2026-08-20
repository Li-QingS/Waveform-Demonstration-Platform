"""FEC 测试（阶段 6b）。"""
import numpy as np

from waveform_sim.hardware.fec import FECMixin


def test_conv_roundtrip():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=100, dtype=np.int8)
    coded = FECMixin._conv_encode_bits(bits, flush=True)
    assert coded.size == 2 * (100 + 6)
    decoded = FECMixin._conv_decode_bits(coded, decoded_len=100, flushed=True)
    assert np.array_equal(decoded, bits)


def test_parity_u32():
    assert FECMixin._parity_u32(0) == 0
    assert FECMixin._parity_u32(1) == 1
    assert FECMixin._parity_u32(3) == 0
    assert FECMixin._parity_u32(7) == 1


def test_interleaver_roundtrip():
    obj = object.__new__(FECMixin)
    obj.coding_interleaver = True
    obj.PILOT_SEED = 12345
    obj.M = 8
    obj.N = 8
    bits = np.random.default_rng(2).integers(0, 2, size=200, dtype=np.int8)
    perm = obj._apply_bit_interleaver(bits)
    back = obj._remove_bit_interleaver(perm)
    assert np.array_equal(back, bits)

