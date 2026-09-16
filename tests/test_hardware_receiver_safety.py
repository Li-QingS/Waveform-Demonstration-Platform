"""Focused numerical safety checks for the hardware receiver hot paths."""

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
        fdidm_m=16,
        fdidm_n=16,
        cp_len=4,
        samp_rate=500_000,
        tx_min_waveform_duration_ms=0,
        tx_max_waveform_samples=65536,
    )


def test_residual_cfo_accepts_coherent_pilot_and_publishes_confidence(backend):
    pilot = backend._heisenberg(backend._pilot_X_tf)
    true_cfo_hz = 120.0
    sample_idx = np.arange(pilot.size, dtype=np.float64)
    observed = pilot * np.exp(1j * 2.0 * np.pi * true_cfo_hz * sample_idx / backend.sample_rate)

    estimate = backend._estimate_residual_cfo_from_pilot(observed)

    assert estimate == pytest.approx(true_cfo_hz, abs=1.0)
    assert backend.last_residual_cfo_accepted is True
    assert backend.last_residual_cfo_reject_reason == "accepted"
    assert backend.last_residual_cfo_pair_count >= backend.N // 2
    assert backend.last_residual_cfo_confidence > 0.9
    assert backend.last_residual_cfo_phase_std_rad < 0.2


@pytest.mark.parametrize("pilot", [np.zeros(16 * 20, dtype=np.complex128), None])
def test_residual_cfo_rejects_unusable_pilot(backend, pilot):
    if pilot is None:
        rng = np.random.default_rng(20260913)
        pilot = (1e-3 / np.sqrt(2.0)) * (
            rng.standard_normal(backend.pilot_frame_len)
            + 1j * rng.standard_normal(backend.pilot_frame_len)
        )

    estimate = backend._estimate_residual_cfo_from_pilot(pilot)

    assert estimate == 0.0
    assert backend.last_residual_cfo_accepted is False
    assert backend.last_residual_cfo_reject_reason in {
        "no_finite_pairs", "too_few_pairs", "incoherent_pairs", "low_phase_coherence"
    }


@pytest.fixture
def full_backend(monkeypatch):
    cls = hardtest_module._LegacyFDIDMHardwareTest
    monkeypatch.setattr(cls, "_import_runtime", lambda self: None)
    monkeypatch.setattr(cls, "_build_top_block", lambda self: None)
    return cls(
        channel_estimator="full_htf",
        fdidm_m=4,
        fdidm_n=8,
        cp_len=1,
        alpha=0.0,
        beta=0.0,
        coding_scheme="none",
        mod_order="64QAM",
        tx_text="x",
        tx_min_waveform_duration_ms=0,
        tx_max_waveform_samples=65536,
    )


@pytest.mark.parametrize("equalizer", ["ZF", "MMSE"])
def test_full_h_svd_truncates_pathological_channel(full_backend, equalizer):
    full_backend.equalizer = equalizer
    K = full_backend.full_htf_order
    h_tf = np.diag(np.r_[1.0, 1e-12, np.ones(K - 2)]).astype(np.complex128)
    y_tf = np.ones((full_backend.M, full_backend.N), dtype=np.complex128)

    x_hat, cond_val, warning = full_backend._equalize_data_full_htf(y_tf, h_tf, 1e-8)

    assert x_hat.shape == (full_backend.M, full_backend.N)
    assert np.all(np.isfinite(x_hat))
    assert cond_val > 1e10
    assert "svd_truncated" in warning
    if equalizer == "MMSE":
        assert "mmse_loaded" in warning


def test_full_h_nonfinite_input_returns_finite_diagnostic(full_backend):
    K = full_backend.full_htf_order
    h_tf = np.eye(K, dtype=np.complex128)
    h_tf[0, 0] = np.nan + 0j
    y_tf = np.ones((full_backend.M, full_backend.N), dtype=np.complex128)

    x_hat, cond_val, warning = full_backend._equalize_data_full_htf(y_tf, h_tf, 1e-3)

    assert np.all(np.isfinite(x_hat))
    assert np.isinf(cond_val)
    assert "nonfinite_input" in warning


def test_full_h_identity_remains_well_conditioned(full_backend):
    K = full_backend.full_htf_order
    h_tf = np.eye(K, dtype=np.complex128)
    symbols = np.arange(1, K + 1, dtype=np.float64) + 1j * np.arange(K, dtype=np.float64)
    y_tf = symbols.reshape((full_backend.M, full_backend.N), order="F")

    x_hat, cond_val, warning = full_backend._equalize_data_full_htf(y_tf, h_tf, 1e-12)

    np.testing.assert_allclose(x_hat.reshape(-1, order="F"), symbols, atol=1e-8)
    assert cond_val == pytest.approx(1.0)
    assert warning in ("", "source=full_htf")
