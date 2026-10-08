import math

import pytest

from waveform_sim.hardware.evidence import (
    AcquisitionFingerprint,
    EvidenceWindow,
    classify_abba_validation,
    combine_abba_windows,
    fingerprints_stable,
)


def fingerprint(index=0, **overrides):
    values = dict(
        effective_cfo_hz=10.0 + 0.1 * index,
        sync_metric=0.95,
        residual_cfo_hz=0.2,
        pilot_nmse=0.02,
        pilot_fitted_power=0.1,
        rx_rms=0.2,
        rx_peak=0.5,
        continuity=True,
        overflow=False,
        acquisition_epoch=7,
        context_id="ctx",
        resource_id="res",
        contract_id="power",
    )
    values.update(overrides)
    return AcquisitionFingerprint(**values)


def window(evm, ser_errors, *, power=0.2, pilot_power=0.1, frames=8, epoch=1, contract="power"):
    result = EvidenceWindow(contract_id=contract, resource_id="res", context_id="ctx", acquisition_epoch=epoch, context_key=("ctx",), continuity=True)
    for _ in range(frames):
        result.add_sample(
            ser_errors=ser_errors,
            ser_symbols=1000,
            data_aided_evm_percent=evm,
            crc_ok=ser_errors == 0,
            power_rms=power,
            pilot_fit_nmse=0.02,
            pilot_fitted_power=pilot_power,
            gmi_nats_per_sample=0.1,
        )
    return result


def test_fingerprint_requires_consecutive_stable_epoch():
    samples = [fingerprint(index) for index in range(4)]
    result = fingerprints_stable(samples, "standard")
    assert result.stable
    changed = samples[:-1] + [fingerprint(3, acquisition_epoch=8)]
    assert fingerprints_stable(changed, "standard").reason == "acquisition_epoch_changed"


def test_fingerprint_overflow_and_gap_fail_closed():
    assert fingerprints_stable([fingerprint(i, overflow=(i == 3)) for i in range(4)]).reason == "overflow"
    assert fingerprints_stable([fingerprint(i, continuity=(i != 3)) for i in range(4)]).reason == "probe_gap"


def test_fingerprint_uses_effective_cfo_not_raw_alias():
    # There is deliberately no raw/coarse CFO field in the model.  Stable
    # effective CFO is sufficient even if an upstream alias would differ.
    result = fingerprints_stable([fingerprint(i, effective_cfo_hz=1.0) for i in range(4)])
    assert result.stable


def test_standard_fingerprint_rejects_real_cfo_or_residual_jump():
    cfo_jump = [fingerprint(i, effective_cfo_hz=(25.0 if i == 3 else 1.0)) for i in range(4)]
    assert fingerprints_stable(cfo_jump, "standard").reason == "effective_cfo_not_stable"
    residual_jump = [fingerprint(i, residual_cfo_hz=(150.0 if i == 3 else 0.0)) for i in range(4)]
    assert fingerprints_stable(residual_jump, "standard").reason == "residual_cfo_not_stable"


def test_fingerprint_settlement_recovers_after_old_reanchor_outlier():
    samples = [fingerprint(0, effective_cfo_hz=200.0)]
    samples.extend(fingerprint(index, effective_cfo_hz=2.0 + index) for index in range(4))

    result = fingerprints_stable(samples, "standard")

    assert result.stable
    assert result.sample_count == 4


def test_combine_abba_requires_equal_blocks_and_common_contract():
    evidence = combine_abba_windows(window(10, 2), window(9, 1), window(9, 1), window(10, 2))
    assert evidence.comparable
    assert evidence.baseline.valid_frames == evidence.candidate.valid_frames == 16
    bad = combine_abba_windows(window(10, 2), window(9, 1, frames=7), window(9, 1), window(10, 2))
    assert not bad.comparable
    assert "equal" in bad.reason
    changed = combine_abba_windows(window(10, 2), window(9, 1, contract="other"), window(9, 1), window(10, 2))
    assert not changed.comparable


def test_linear_drift_is_cancelled_by_symmetric_abba():
    # Linear pilot-power trend at t=0,1,2,3 has equal A/B means.
    evidence = combine_abba_windows(
        window(10, 1, pilot_power=0.10),
        window(10, 1, pilot_power=0.11),
        window(10, 1, pilot_power=0.12),
        window(10, 1, pilot_power=0.13),
    )
    decision = classify_abba_validation(evidence, gain_lower_nats=0.01, threshold_nats=0.001)
    assert decision.outcome == "improved"
    no_theory_gain = classify_abba_validation(evidence, gain_lower_nats=0.0, threshold_nats=0.001)
    assert no_theory_gain.outcome == "inconclusive"
    assert not no_theory_gain.gmi_confirmed


def test_nonlinear_drift_and_hardware_regression_do_not_commit():
    drift = combine_abba_windows(
        window(10, 1, pilot_power=0.10),
        window(10, 1, pilot_power=0.40),
        window(10, 1, pilot_power=0.40),
        window(10, 1, pilot_power=0.10),
    )
    assert classify_abba_validation(drift, gain_lower_nats=0.01, threshold_nats=0.001).outcome == "inconclusive"
    regression = combine_abba_windows(window(10, 0), window(15, 10), window(15, 10), window(10, 0))
    decision = classify_abba_validation(regression, gain_lower_nats=0.01, threshold_nats=0.001)
    assert decision.outcome == "regressed"
    assert not decision.hardware_non_regressing

    # Aggregated A/B means are equal here, but A1/A2 are not stationary.  The
    # endpoint guard must catch this pattern instead of letting averaging hide
    # a large within-waveform drift.
    endpoint_drift = combine_abba_windows(
        window(10, 1, pilot_power=0.10),
        window(10, 1, pilot_power=0.25),
        window(10, 1, pilot_power=0.25),
        window(10, 1, pilot_power=0.40),
    )
    endpoint_decision = classify_abba_validation(
        endpoint_drift,
        gain_lower_nats=0.01,
        threshold_nats=0.001,
    )
    assert endpoint_decision.outcome == "inconclusive"
    assert "A1/A2" in endpoint_decision.reason


def test_evm_only_improvement_is_not_gmi_advantage():
    evidence = combine_abba_windows(window(10, 2), window(5, 1), window(5, 1), window(10, 2))
    decision = classify_abba_validation(evidence, gain_lower_nats=0.0, threshold_nats=0.0)
    assert decision.outcome == "inconclusive"
    assert not decision.gmi_confirmed
