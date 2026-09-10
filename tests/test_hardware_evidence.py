import json
import math

import numpy as np

from waveform_sim.hardware.evidence import (
    EvidenceWindow,
    TxPowerMetrics,
    analyze_unscaled_power,
    assess_power_comparability,
    classify_validation,
    create_power_contract,
    frame_structure_metrics,
    measure_known_symbols,
    measure_pilot_fit,
    scale_waveform_to_rms,
    signal_rms,
    wilson_interval,
)


QPSK = np.asarray([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex128) / np.sqrt(2.0)


def test_dataclasses_are_json_serializable_with_unavailable_values():
    metrics = TxPowerMetrics()
    data = metrics.as_dict()
    assert data["cycle_rms"] is None
    json.dumps(data, allow_nan=False)


def test_power_scaling_hits_target_without_exceeding_peak():
    cycle = np.asarray([1 + 0j, 1j, -1 + 0j, -1j])
    scaled, metrics = scale_waveform_to_rms(
        cycle, cycle[:2], target_rms=0.2, peak_limit=0.8, contract_id="p1"
    )
    assert signal_rms(scaled) == pytest_approx(0.2)
    assert metrics.cycle_rms == pytest_approx(0.2)
    assert metrics.data_rms == pytest_approx(0.2)
    assert metrics.peak <= 0.8
    assert metrics.backoff_db == pytest_approx(0.0, abs=1e-10)
    assert not metrics.peak_limited
    assert metrics.contract_id == "p1"


def test_high_papr_waveform_is_peak_limited_and_reports_backoff():
    cycle = np.ones(100, dtype=np.complex128)
    cycle[0] = 20.0
    preview = analyze_unscaled_power(cycle, cycle[20:80], target_rms=0.5, peak_limit=0.9)
    scaled, metrics = scale_waveform_to_rms(cycle, cycle[20:80], target_rms=0.5, peak_limit=0.9)
    assert metrics.peak == pytest_approx(0.9)
    assert metrics.cycle_rms < 0.5
    assert metrics.peak_limited
    assert metrics.backoff_db < 0.0
    assert preview.safe_rms == pytest_approx(metrics.cycle_rms)
    assert signal_rms(scaled) == pytest_approx(metrics.cycle_rms)


def test_empty_and_zero_power_inputs_are_stable():
    for cycle in (np.empty(0, dtype=np.complex64), np.zeros(8, dtype=np.complex64)):
        scaled, metrics = scale_waveform_to_rms(cycle, cycle, target_rms=0.2, peak_limit=0.9)
        assert scaled.size == cycle.size
        assert metrics.cycle_rms == 0.0
        assert metrics.safe_rms == 0.0
        assert not metrics.peak_limited


def test_power_contract_uses_common_safe_rms_and_is_stable():
    baseline = TxPowerMetrics(safe_rms=0.30)
    candidate = TxPowerMetrics(safe_rms=0.24)
    kwargs = dict(
        baseline_alpha=0.5,
        baseline_beta=1.0,
        candidate_alpha=0.75,
        candidate_beta=0.8,
        requested_rms=0.28,
        peak_limit=0.9,
    )
    first = create_power_contract(baseline, candidate, **kwargs)
    second = create_power_contract(baseline, candidate, **kwargs)
    assert first.comparable
    assert first.locked_rms == pytest_approx(0.24 * 0.98)
    assert first.locked_rms <= baseline.safe_rms
    assert first.locked_rms <= candidate.safe_rms
    assert first.contract_id == second.contract_id


def test_invalid_power_contract_and_comparability_reasons():
    invalid = create_power_contract(
        TxPowerMetrics(safe_rms=float("nan")),
        TxPowerMetrics(safe_rms=0.2),
        baseline_alpha=0.5,
        baseline_beta=1.0,
        candidate_alpha=0.75,
        candidate_beta=1.0,
        requested_rms=0.2,
        peak_limit=0.9,
    )
    assert not invalid.comparable
    assert invalid.reason
    assert not assess_power_comparability("", 0.2, "x", 0.2)[0]
    assert not assess_power_comparability("x", 0.2, "y", 0.2)[0]
    comparable, reason, difference = assess_power_comparability("x", 0.2, "x", 0.2)
    assert comparable and not reason and difference == pytest_approx(0.0)
    comparable, reason, difference = assess_power_comparability("x", 0.2, "x", 0.18)
    assert not comparable
    assert difference > 0.10
    assert "differs" in reason


def test_known_symbols_remove_common_gain_phase_and_count_exact_errors():
    reference = np.tile(QPSK, 40)
    rx = reference * (1.7 * np.exp(1j * 0.37))
    rx[3] = reference[2] * (1.7 * np.exp(1j * 0.37))
    metrics = measure_known_symbols(rx, reference, QPSK)
    assert metrics.reference_symbols == reference.size
    assert metrics.ser_symbols == reference.size
    assert metrics.ser_errors == 1
    assert metrics.ser == pytest_approx(1.0 / reference.size)
    assert metrics.residual_gain_abs == pytest_approx(1.7, rel=0.02)
    assert metrics.residual_phase_deg == pytest_approx(np.degrees(0.37), abs=0.5)
    assert metrics.data_aided_evm_percent > metrics.decision_directed_evm_percent


def test_known_symbol_evm_matches_constructed_orthogonal_noise():
    reference = np.tile(QPSK, 50)
    signs = np.tile(np.asarray([1, -1], dtype=float), reference.size // 2)
    noise = 0.1j * reference * signs
    metrics = measure_known_symbols(reference + noise, reference, QPSK)
    assert metrics.ser_errors == 0
    assert metrics.data_aided_evm_percent == pytest_approx(10.0, rel=1e-6)
    assert metrics.decision_directed_evm_percent == pytest_approx(10.0, rel=1e-6)


def test_known_symbol_empty_and_length_mismatch_are_safe():
    assert measure_known_symbols([], [], QPSK).ser_symbols == 0
    metrics = measure_known_symbols(QPSK[:2], QPSK, QPSK)
    assert metrics.ser_symbols == 2


def test_pilot_fit_reports_residual_sinr_and_nmse():
    fitted = np.ones(100, dtype=np.complex128)
    residual = np.tile(np.asarray([0.1, -0.1]), 50)
    observed = fitted + residual
    metrics = measure_pilot_fit(observed, fitted)
    assert metrics.residual_power == pytest_approx(0.01)
    assert metrics.fitted_power == pytest_approx(1.0)
    assert metrics.residual_sinr_db == pytest_approx(20.0)
    assert metrics.fit_nmse == pytest_approx(0.01 / 1.01)
    assert "snr_db" not in metrics.as_dict()


def test_pilot_fit_zero_residual_and_empty_inputs():
    exact = measure_pilot_fit(np.ones(4), np.ones(4))
    assert math.isinf(exact.residual_sinr_db)
    empty = measure_pilot_fit([], [])
    assert math.isnan(empty.fit_nmse)


def test_frame_efficiency_for_current_diag_tf_layout():
    metrics = frame_structure_metrics(
        data_samples=320,
        pilot_samples=320,
        sync_samples=64,
        guard_samples=116,
    )
    assert metrics["data_samples"] == 320
    assert metrics["pilot_samples"] == 320
    assert metrics["total_samples"] == 820
    assert metrics["training_data_ratio"] == pytest_approx(1.0)
    assert metrics["useful_data_ratio"] == pytest_approx(320 / 820)


def test_frame_efficiency_is_recomputed_from_each_actual_structure():
    compact = frame_structure_metrics(
        data_samples=320, pilot_samples=80, sync_samples=64, guard_samples=32)
    training_heavy = frame_structure_metrics(
        data_samples=320, pilot_samples=640, sync_samples=64, guard_samples=32)
    assert compact["training_data_ratio"] == pytest_approx(0.25)
    assert training_heavy["training_data_ratio"] == pytest_approx(2.0)
    assert compact["useful_data_ratio"] > training_heavy["useful_data_ratio"]
    assert compact["total_samples"] == 496
    assert training_heavy["total_samples"] == 1056


def _window(errors, symbols, frames=32, evm=10.0, crc_ratio=1.0, contract="p1", context=(1,)):
    window = EvidenceWindow(contract_id=contract, context_key=context)
    per_frame_symbols = symbols // frames
    per_frame_errors = errors // frames
    error_remainder = errors - per_frame_errors * frames
    symbol_remainder = symbols - per_frame_symbols * frames
    crc_ok_count = round(frames * crc_ratio)
    for index in range(frames):
        window.add_sample(
            ser_errors=per_frame_errors + (1 if index < error_remainder else 0),
            ser_symbols=per_frame_symbols + (1 if index < symbol_remainder else 0),
            data_aided_evm_percent=evm,
            crc_ok=index < crc_ok_count,
            raw_ber=0.01,
            fec_ber=0.0,
            power_rms=0.2,
        )
    return window


def test_evidence_window_aggregates_and_serializes():
    window = _window(320, 3200)
    window.drop("overflow")
    window.drop("sync")
    assert window.valid_frames == 32
    assert window.ser_errors == 320
    assert window.ser_symbols == 3200
    assert window.ser == pytest_approx(0.1)
    assert window.data_aided_evm_mean == pytest_approx(10.0)
    assert window.crc_success_ratio == pytest_approx(1.0)
    assert window.dropped_overflow == 1
    json.dumps(window.as_dict(), allow_nan=False)


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(100, 1000)
    assert low < 0.1 < high
    assert all(math.isnan(value) for value in wilson_interval(0, 0))


def test_validation_improved_when_exact_evidence_is_separated():
    baseline = _window(640, 6400, evm=12.0)
    candidate = _window(160, 6400, evm=9.0)
    result = classify_validation(baseline, candidate)
    assert result.outcome == "improved"
    assert result.measured_ser_gain_db > 0.25
    assert result.baseline_ser_interval[0] > result.candidate_ser_interval[1]


def test_validation_inconclusive_for_sample_error_or_power_shortage():
    short = _window(120, 2000, frames=20)
    enough = _window(120, 2000, frames=24)
    assert classify_validation(short, enough).outcome == "inconclusive"
    low_errors_a = _window(20, 6400)
    low_errors_b = _window(10, 6400)
    assert "insufficient SER errors" in classify_validation(low_errors_a, low_errors_b).reason
    result = classify_validation(enough, enough, comparable=False, comparability_reason="power mismatch")
    assert result.outcome == "inconclusive"
    assert result.reason == "power mismatch"


def test_validation_regressed_for_ser_evm_or_crc():
    good = _window(160, 6400, evm=8.0, crc_ratio=1.0)
    bad_ser = _window(640, 6400, evm=8.0, crc_ratio=1.0)
    assert classify_validation(good, bad_ser).outcome == "regressed"
    bad_evm = _window(160, 6400, evm=11.0, crc_ratio=1.0)
    assert classify_validation(good, bad_evm).outcome == "regressed"
    bad_crc = _window(160, 6400, evm=8.0, crc_ratio=0.75)
    assert classify_validation(good, bad_crc).outcome == "regressed"


def test_20260907_historical_power_unfair_no_improvement():
    baseline = _window(9480, 100000, frames=64, evm=45.952, contract="historical")
    candidate = _window(9645, 100000, frames=64, evm=45.445, contract="historical")
    comparable, reason, difference = assess_power_comparability(
        "historical", math.sqrt(26877.279), "historical", math.sqrt(22548.869)
    )
    assert not comparable
    assert difference > 0.7
    result = classify_validation(baseline, candidate, comparable=comparable, comparability_reason=reason)
    assert result.outcome in {"inconclusive", "regressed"}
    assert result.outcome != "improved"


def pytest_approx(value, **kwargs):
    # Keeps this module importable without exposing pytest in production code.
    import pytest

    return pytest.approx(value, **kwargs)
