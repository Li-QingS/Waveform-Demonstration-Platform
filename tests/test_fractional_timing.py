"""Sub-sample RX timing: estimator, corrector and backend wiring.

These tests are offline: they need neither GNU Radio, UHD, Qt nor a device.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from waveform_sim.hardware.timing import (
    DEFAULT_MIN_ABS_DELAY,
    delay_needs_correction,
    estimate_fractional_delay,
    fractional_delay_correct,
    fractional_delay_kernel,
)


def _bandlimited_reference(count: int = 320, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, count) * 2 - 1
    symbols = bits + 1j * (rng.integers(0, 2, count) * 2 - 1)
    spectrum = np.fft.fft(symbols)
    mask = np.zeros(count)
    half = count // 2 - 20
    mask[:half] = 1.0
    mask[-half:] = 1.0
    return np.fft.ifft(spectrum * mask).astype(np.complex128)


def _apply_delay(signal: np.ndarray, delay: float) -> np.ndarray:
    grid = np.fft.fftfreq(signal.size)
    return np.fft.ifft(np.fft.fft(signal) * np.exp(-1j * 2.0 * np.pi * grid * delay))


def _evm_percent(measured: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.mean(np.abs(reference) ** 2))
    error = float(np.mean(np.abs(measured - reference) ** 2))
    return 100.0 * float(np.sqrt(error / denominator))


def test_kernel_has_unit_gain_and_is_finite():
    for delay in (-0.4, -0.1, 0.0, 0.1, 0.4):
        kernel = fractional_delay_kernel(delay)
        assert np.all(np.isfinite(kernel))
        assert float(np.sum(kernel)) == pytest.approx(1.0, abs=1e-12)


def test_zero_delay_correction_is_an_exact_copy():
    signal = _bandlimited_reference(96)
    corrected = fractional_delay_correct(signal, 0.0)
    assert np.array_equal(corrected, signal)
    assert corrected.dtype == np.complex128


@pytest.mark.parametrize("delay", [-0.42, -0.20, -0.12, 0.12, 0.20, 0.42])
def test_estimator_recovers_a_known_fractional_delay(delay):
    reference = _bandlimited_reference()
    observed = _apply_delay(reference, delay)
    estimate = estimate_fractional_delay(observed, reference)
    assert estimate.reliable
    assert estimate.delay_samples == pytest.approx(delay, abs=0.1)
    assert estimate.integer_lag == 0
    assert abs(estimate.fractional_samples) <= 0.5


@pytest.mark.parametrize("delay", [-0.36, -0.20, 0.20, 0.36])
def test_correction_recovers_most_of_the_delay_loss(delay):
    reference = _bandlimited_reference()
    observed = _apply_delay(reference, delay)
    estimate = estimate_fractional_delay(observed, reference)
    corrected = fractional_delay_correct(observed, estimate.correction_samples())
    assert _evm_percent(corrected, reference) < 0.5 * _evm_percent(observed, reference)


def test_small_residual_is_left_untouched():
    reference = _bandlimited_reference()
    observed = _apply_delay(reference, 0.04)
    estimate = estimate_fractional_delay(observed, reference)
    assert estimate.correction_samples() == 0.0
    assert delay_needs_correction(0.05) is False
    assert delay_needs_correction(DEFAULT_MIN_ABS_DELAY) is True


def test_estimator_rejects_uncorrelated_input():
    reference = _bandlimited_reference(256, seed=3)
    gen = np.random.default_rng(5)
    noise = gen.normal(size=256) + 1j * gen.normal(size=256)
    estimate = estimate_fractional_delay(noise, reference)
    assert not estimate.reliable
    assert estimate.correction_samples() == 0.0


def test_estimator_handles_empty_and_degenerate_input():
    assert not estimate_fractional_delay(np.zeros(0), np.zeros(0)).reliable
    assert not estimate_fractional_delay(np.zeros(8), np.zeros(8)).reliable
    with pytest.raises(ValueError):
        fractional_delay_correct(np.ones(8, dtype=complex), float("nan"))


def test_backend_status_exposes_fractional_timing_fields(monkeypatch):
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    monkeypatch.setattr(mod._LegacyFDIDMHardwareTest, "_build_top_block", lambda self: None)
    obj = mod._LegacyFDIDMHardwareTest(
        tx_frame_count=1, tx_min_waveform_duration_ms=1, log_to_stdout=False,
    )
    status = obj.get_status()
    for key in (
        "frac_delay_samples",
        "frac_delay_fractional",
        "frac_delay_magnitude",
        "frac_delay_applied",
        "frac_delay_correction_count",
        "analog_bandwidth_hz",
    ):
        assert key in status
    assert status["frac_delay_applied"] is False
    assert status["frac_delay_correction_count"] == 0
    assert obj.enable_fractional_timing is True
    assert obj.fractional_timing_min_abs == pytest.approx(DEFAULT_MIN_ABS_DELAY)


def test_post_rf_tdl_mode_disables_the_known_pilot_reference(monkeypatch):
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    monkeypatch.setattr(mod._LegacyFDIDMHardwareTest, "_build_top_block", lambda self: None)
    after_rf = mod._LegacyFDIDMHardwareTest(
        channel_mode="rf_tdl_a", tx_frame_count=1, tx_min_waveform_duration_ms=1, log_to_stdout=False,
    )
    assert after_rf._tdl_after_rf_enabled() is True
    pre_rendered = mod._LegacyFDIDMHardwareTest(
        channel_mode="tdl_a_rf", tx_frame_count=1, tx_min_waveform_duration_ms=1, log_to_stdout=False,
    )
    assert pre_rendered._tdl_after_rf_enabled() is False


def test_analog_bandwidth_argument_is_validated(monkeypatch):
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    monkeypatch.setattr(mod._LegacyFDIDMHardwareTest, "_build_top_block", lambda self: None)
    explicit = mod._LegacyFDIDMHardwareTest(
        analog_bandwidth_hz=1_000_000.0, tx_frame_count=1, tx_min_waveform_duration_ms=1,
        log_to_stdout=False,
    )
    assert explicit.analog_bandwidth_hz == pytest.approx(1_000_000.0)
    default = mod._LegacyFDIDMHardwareTest(
        tx_frame_count=1, tx_min_waveform_duration_ms=1, log_to_stdout=False,
    )
    assert default.analog_bandwidth_hz is None
    with pytest.raises(ValueError):
        mod._LegacyFDIDMHardwareTest(analog_bandwidth_hz=-1.0)