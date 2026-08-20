"""NTN-TDL 软件信道测试（阶段 6b）。"""
import numpy as np

from waveform_sim.hardware.channel import NTNTDLChannel


def test_channel_configure_process():
    ch = NTNTDLChannel(
        sample_rate=1e6, model="tdl_a", rms_delay_spread_ns=100.0,
        doppler_hz=20.0, snr_db=30.0, seed=7,
    )
    y = ch.process(np.ones(256, dtype=np.complex64))
    assert y.shape == (256,)
    summary = ch.summary()
    assert "NTN-TDL-A" in summary


def test_channel_reset_and_profiles():
    ch = NTNTDLChannel(sample_rate=1e6, model="tdl_d")
    assert ch.model == "tdl_d"
    y1 = ch.process(np.ones(128, dtype=np.complex64))
    ch.reset()
    y2 = ch.process(np.ones(128, dtype=np.complex64))
    assert y1.shape == y2.shape == (128,)

