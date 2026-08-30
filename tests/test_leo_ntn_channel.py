import math

import numpy as np

from waveform_sim.simulation.leo_ntn_channel import (
    absolute_doppler_shift_hz,
    coherence_time_s,
    get_profile,
    residual_common_cfo_hz,
    sample_path_dopplers,
)
from waveform_sim.simulation.simple_fdidm_rx import _LegacyFDIDMTransceiver


def test_standard_ntn_profiles_are_encoded():
    tdla = get_profile("NTN-TDLA100-200")
    assert tdla.delays_ns == (0.0, 110.0, 285.0)
    assert tdla.powers_db == (0.0, -4.7, -6.5)
    assert tdla.maximum_doppler_hz == 200.0
    assert tdla.nominal_rms_delay_ns == 100.0

    tdlc = get_profile("NTN-TDLC5-1200")
    assert tdlc.delays_ns == (0.0, 0.0, 60.0)
    assert tdlc.powers_db == (-0.6, -8.9, -21.5)
    assert tdlc.maximum_doppler_hz == 1200.0
    assert tdlc.nominal_rms_delay_ns == 5.0


def test_orbital_shift_and_residual_cfo_are_separate():
    absolute = absolute_doppler_shift_hz(20e9, 28080.0, 0.10)
    assert absolute == pytest_approx(52_035.9988509, rel=1e-9)
    residual = residual_common_cfo_hz(absolute, 0.999)
    assert residual == pytest_approx(52.03599885, rel=1e-9)
    assert coherence_time_s(200.0) == pytest_approx(0.002115, rel=1e-12)
    assert coherence_time_s(1200.0) == pytest_approx(0.0003525, rel=1e-12)


def test_sampled_path_dopplers_stay_inside_residual_band():
    rng = np.random.default_rng(17)
    values = sample_path_dopplers(100, 200.0, 52.0, rng)
    assert values.shape == (100,)
    assert np.max(values) <= 252.0 + 1e-12
    assert np.min(values) >= -148.0 - 1e-12


def test_backend_reports_default_bandwidth_and_channel_change():
    tb = _LegacyFDIDMTransceiver(
        m_subcarriers=8,
        n_symbols=8,
        subcarrier_spacing_hz=300e3,
        channel_dynamics="cont",
        dynamic_channel=True,
        demo_frame_interval_s=0.0,
    )
    tb.step()
    tb.step()
    summary = tb.get_channel_summary()
    metrics = tb.get_last_metrics()
    assert summary["bandwidth_hz"] == pytest_approx(2.4e6)
    assert summary["absolute_doppler_shift_hz"] > 50e3
    assert summary["residual_doppler_spread_hz"] == pytest_approx(200.0)
    assert math.isfinite(metrics["channel_matrix_change_norm"])
    assert metrics["channel_matrix_change_norm"] > 0.0
    assert 0.0 <= metrics["channel_matrix_correlation"] <= 1.0


def pytest_approx(value, **kwargs):
    # Local helper avoids importing pytest at module import time in script-style
    # smoke runs while still returning the real pytest comparator when available.
    import pytest

    return pytest.approx(value, **kwargs)
