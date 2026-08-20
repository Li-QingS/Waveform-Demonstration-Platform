"""波形对比扫描冒烟基线：scan_axis 返回三种波形的 BER 数组。"""
import numpy as np

from waveform_sim.simulation.compare_scan_backend import scan_axis


def test_scan_axis_smoke():
    result = scan_axis(
        axis_name="doppler",
        axis_values=np.array([0.0, 20.0]),
        snr_db=15.0,
        mod_order="QPSK",
        frames=2,
        fixed_cfo_hz=0.0,
        fixed_doppler_hz=20.0,
        seed_base=20260502,
    )
    assert set(result.keys()) == {"OFDM", "OTFS", "AFDM"}
    for name, arr in result.items():
        assert arr.shape == (2,), name
        assert np.all((arr >= 0.0) & (arr <= 1.0)), name

