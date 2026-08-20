"""环形缓冲测试（阶段 6b）。"""
import numpy as np

from waveform_sim.hardware.stream import SampleRing


def test_ring_write_read_latest():
    r = SampleRing(capacity=1024)
    x = np.array([1 + 1j, 2 + 0j], dtype=np.complex64)
    r.write(x)
    assert r.total_written == 2
    assert len(r) == 2
    data, total, count = r.read_latest(2)
    assert total == 2 and count == 2
    assert np.array_equal(data, x)


def test_ring_wraps():
    r = SampleRing(capacity=8)  # 内部钳制到 1024
    r.write(np.ones(1024, dtype=np.complex64))
    r.write(np.arange(0, 6, dtype=np.float32).astype(np.complex64))
    assert r.total_written == 1030
    assert len(r) == 1024
    data, _, _ = r.read_latest(6)
    assert np.array_equal(data, np.arange(0, 6, dtype=np.float32).astype(np.complex64))


def test_ring_clear():
    r = SampleRing(capacity=8)
    r.write(np.ones(4, dtype=np.complex64))
    r.clear()
    assert r.total_written == 0
    assert len(r) == 0

