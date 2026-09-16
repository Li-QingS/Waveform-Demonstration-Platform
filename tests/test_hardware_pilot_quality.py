"""Offline regressions for candidate-local CSI and diagonal MMSE loading."""

import numpy as np
import pytest

import waveform_sim.hardware.fdidm_hardtest as hardtest_module


@pytest.fixture
def backend(monkeypatch):
    cls = hardtest_module._LegacyFDIDMHardwareTest
    monkeypatch.setattr(cls, "_import_runtime", lambda self: None)
    monkeypatch.setattr(cls, "_build_top_block", lambda self: None)
    return cls(
        channel_estimator="diag_tf",
        tx_min_waveform_duration_ms=0,
        tx_max_waveform_samples=65536,
    )


def test_failed_sync_candidates_leave_cross_frame_csi_unchanged(backend):
    pilot = backend._heisenberg(backend._pilot_X_tf)
    backend._estimate_htf_diag_from_pilot(pilot)
    prior = backend._diag_csi_smooth.copy()
    prior_adaptive = backend._latest_adaptive_diag_csi.copy()

    # All repeated frames retain a perfect sync preamble, but their pilot and
    # payload are erased.  Several sync candidates can therefore be attempted
    # in one window; none may alter the previously accepted channel estimate.
    rx = backend._tx_waveform[:2 * backend._tx_base_cycle_len].astype(np.complex128).copy()
    stride = backend.frame_len + backend.inter_frame_guard_len
    for frame in range(2 * backend._tx_cycle_frame_count):
        base = frame * stride
        rx[base + backend._off_pilot:base + backend._off_end] = 0.0

    backend._try_process_rx_window(rx, rx.size)

    assert backend._frames_processed == 1
    assert backend._frames_decode_ok == 0
    np.testing.assert_array_equal(backend._diag_csi_smooth, prior)
    np.testing.assert_array_equal(backend._latest_adaptive_diag_csi, prior_adaptive)


def test_good_frame_commits_one_csi_update(backend):
    backend._diag_csi_smooth_weight = 0.2  # Opt-in phase-coherent smoothing.
    rx = backend._tx_waveform[:2 * backend._tx_base_cycle_len].astype(np.complex128).copy()
    pilot = rx[backend._off_pilot:backend._off_data]
    backend._estimate_htf_diag_from_pilot(0.5 * pilot)
    prior = backend._diag_csi_smooth.copy()
    current = backend._wigner(pilot) / backend._pilot_X_tf
    expected = 0.8 * prior + 0.2 * current

    backend._try_process_rx_window(rx, rx.size)

    assert backend._frames_decode_ok == 1
    np.testing.assert_allclose(backend._diag_csi_smooth, expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(backend._latest_adaptive_diag_csi, expected, rtol=1e-6, atol=1e-9)


def test_default_csi_uses_current_rf_frame_phase(backend):
    pilot = backend._heisenberg(backend._pilot_X_tf)
    backend._estimate_htf_diag_from_pilot(pilot)

    h_tf, _, _, adaptive_csi = backend._estimate_htf_diag_from_pilot(
        1j * pilot, commit=False, return_csi=True
    )
    current = backend._wigner(1j * pilot) / backend._pilot_X_tf

    np.testing.assert_allclose(adaptive_csi, current, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h_tf, 1j * np.ones_like(h_tf), rtol=1e-12, atol=1e-12)


def test_graph_transition_clears_phase_sensitive_csi_even_with_smoothing(backend):
    backend._diag_csi_smooth_weight = 0.2
    pilot = backend._heisenberg(backend._pilot_X_tf)
    backend._estimate_htf_diag_from_pilot(pilot)
    assert backend._diag_csi_smooth is not None

    backend._reset_rx_runtime_state("alpha_beta_swap", reset_counters=False)
    assert backend._diag_csi_smooth is None
    assert not np.any(backend._latest_adaptive_diag_csi)

    _, _, _, candidate_csi = backend._estimate_htf_diag_from_pilot(
        1j * pilot, commit=False, return_csi=True
    )
    expected = backend._wigner(1j * pilot) / backend._pilot_X_tf
    np.testing.assert_allclose(candidate_csi, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("pilot_scale", [0.25, 1.0, 2.0])
def test_mmse_noise_load_uses_unsmoothed_current_pilot_residual(backend, pilot_scale):
    backend._pilot_X_tf *= pilot_scale
    pilot = backend._heisenberg(backend._pilot_X_tf)
    rng = np.random.default_rng(20260913)
    for _ in range(24):
        noise = 0.05 / np.sqrt(2.0) * (
            rng.standard_normal(pilot.size) + 1j * rng.standard_normal(pilot.size)
        )
        observation = pilot + noise
        raw = backend._wigner(observation) / backend._pilot_X_tf
        raw_residual = raw - np.mean(raw, axis=1)[:, None]
        expected = (float(np.mean(np.abs(raw_residual) ** 2))
                    * float(np.mean(np.abs(backend._pilot_X_tf) ** 2)))

        _, _, noise_var = backend._estimate_htf_diag_from_pilot(observation)

        assert noise_var == pytest.approx(expected, rel=1e-12, abs=1e-15)
