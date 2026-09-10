import copy
import json
import math

import numpy as np
import pytest

import waveform_sim.hardware.fdidm_hardtest as hardtest_module
from waveform_sim.hardware.evidence import power_difference_db, signal_rms


@pytest.fixture
def backend_factory(monkeypatch):
    cls = hardtest_module._LegacyFDIDMHardwareTest
    monkeypatch.setattr(cls, "_import_runtime", lambda self: None)
    monkeypatch.setattr(cls, "_build_top_block", lambda self: None)

    def build(**kwargs):
        defaults = {
            "channel_estimator": "diag_tf",
            "tx_min_waveform_duration_ms": 0,
            "tx_max_waveform_samples": 65536,
            "fdidm_m": 16,
            "fdidm_n": 16,
            "cp_len": 4,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    return build


def test_preview_is_side_effect_free(backend_factory):
    backend = backend_factory()
    before = {
        "alpha": backend.alpha,
        "beta": backend.beta,
        "waveform": backend._tx_waveform.copy(),
        "gamma_cache": copy.deepcopy(backend._gamma_cache),
        "debug_seq": backend._debug_seq,
        "frames_processed": backend._frames_processed,
        "power": backend._tx_power_metrics,
    }
    preview = backend._preview_waveform_build(0.75, 0.8)
    assert preview.tx_waveform.size > 1
    assert preview.power_metrics.safe_rms > 0.0
    assert backend.alpha == before["alpha"]
    assert backend.beta == before["beta"]
    np.testing.assert_array_equal(backend._tx_waveform, before["waveform"])
    assert backend._gamma_cache.keys() == before["gamma_cache"].keys()
    assert backend._debug_seq == before["debug_seq"]
    assert backend._frames_processed == before["frames_processed"]
    assert backend._tx_power_metrics == before["power"]


def test_committed_waveform_reports_actual_power_metrics(backend_factory):
    backend = backend_factory()
    metrics = backend._tx_power_metrics
    assert metrics.cycle_rms == pytest.approx(signal_rms(backend._tx_waveform), rel=1e-7)
    assert metrics.peak == pytest.approx(float(np.max(np.abs(backend._tx_waveform))), rel=1e-7)
    assert metrics.data_rms > 0.0
    assert metrics.papr_db > 0.0
    assert metrics.backoff_db <= 1e-9
    assert metrics.peak <= backend._tx_peak_limit + 1e-7


def test_common_contract_equalizes_alpha_candidates_without_peak_violation(backend_factory):
    backend = backend_factory(alpha=0.5, beta=1.0)
    contract = backend._prepare_power_contract(0.75, 0.8)
    baseline = backend._commit_waveform_build(
        0.5, 1.0, forced_rms=contract.locked_rms, contract_id=contract.contract_id
    )
    candidate = backend._commit_waveform_build(
        0.75, 0.8, forced_rms=contract.locked_rms, contract_id=contract.contract_id
    )
    assert contract.comparable
    assert baseline.power_metrics.contract_id == candidate.power_metrics.contract_id == contract.contract_id
    assert power_difference_db(
        baseline.power_metrics.cycle_rms, candidate.power_metrics.cycle_rms
    ) <= contract.tolerance_db
    assert baseline.power_metrics.peak <= contract.peak_limit + 1e-7
    assert candidate.power_metrics.peak <= contract.peak_limit + 1e-7
    assert not baseline.power_metrics.peak_limited
    assert not candidate.power_metrics.peak_limited


def test_known_reference_uses_only_common_coded_prefix(backend_factory):
    backend = backend_factory(tx_text="FDIDM OK")
    bps = backend.bits_per_symbol
    expected_count = backend._tx_coded_frame_bits.size // bps
    assert backend._tx_known_reference_symbols.size == expected_count
    np.testing.assert_allclose(
        backend._tx_known_reference_symbols,
        backend._known_preamble_ref_syms(),
    )
    assert expected_count < backend.M * backend.N
    recovered = backend._recover_payload_from_symbols(backend._tx_x_cross.reshape(-1, order="F"))
    assert recovered[5]
    assert backend._last_known_symbol_metrics.ser_errors == 0
    assert backend._last_known_symbol_metrics.ser_symbols == expected_count
    assert backend._last_known_symbol_metrics.data_aided_evm_percent == pytest.approx(0.0, abs=1e-8)


def test_exact_ser_counters_reset_only_with_runtime_counters(backend_factory):
    backend = backend_factory()
    backend._ser_errors_total = 12
    backend._ser_symbols_total = 120
    backend._reset_rx_runtime_state("live", reset_counters=False)
    assert backend._ser_errors_total == 12
    assert backend._ser_symbols_total == 120
    backend._reset_rx_runtime_state("restart", reset_counters=True)
    assert backend._ser_errors_total == 0
    assert backend._ser_symbols_total == 0


def test_pilot_fit_diagnostic_does_not_change_diag_estimator_result(backend_factory):
    backend = backend_factory()
    pilot = backend._heisenberg(backend._pilot_X_tf)
    y_tf = backend._wigner(pilot)
    raw_cell = y_tf / backend._pilot_X_tf
    expected_frequency_response = np.mean(raw_cell, axis=1)
    actual, selectivity, noise = backend._estimate_htf_diag_from_pilot(pilot)
    np.testing.assert_allclose(actual[:, 0], expected_frequency_response, atol=1e-10)
    np.testing.assert_allclose(actual, np.repeat(actual[:, :1], backend.N, axis=1), atol=1e-12)
    assert selectivity == pytest.approx(0.0, abs=1e-10)
    assert noise == pytest.approx(0.0, abs=1e-20)
    assert backend._last_pilot_fit_metrics.fit_nmse == pytest.approx(0.0, abs=1e-20)
    assert math.isinf(backend._last_pilot_fit_metrics.residual_sinr_db)


def test_status_exposes_power_known_symbol_pilot_and_frame_evidence(backend_factory):
    backend = backend_factory(tdl_snr_db=35.0)
    status = backend.get_status()
    assert status["tx_cycle_rms"] > 0.0
    assert status["tx_data_rms"] > 0.0
    assert status["tx_peak"] <= 0.9 + 1e-7
    assert "tx_papr_db" in status and "tx_power_backoff_db" in status
    assert status["tdl_injected_snr_db"] == pytest.approx(35.0)
    assert status["ser_errors_total"] == 0
    assert status["ser_symbols_total"] == 0
    assert "data_aided_evm_percent" in status
    assert "decision_directed_evm_percent" in status
    assert "pilot_residual_sinr_db" in status
    assert "pilot_fit_nmse" in status
    assert status["frame_structure"] == {
        "data_samples": 320,
        "pilot_samples": 320,
        "sync_samples": 64,
        "guard_samples": 116,
        "other_samples": 0,
        "total_samples": 820,
        "training_data_ratio": 1.0,
        "useful_data_ratio": pytest.approx(320 / 820),
    }
    json.dumps(status)


def test_20260907_old_scaling_was_not_power_comparable(backend_factory):
    backend = backend_factory(alpha=0.5, beta=1.0)
    old_baseline = backend._preview_waveform_build(0.5, 1.0).power_metrics
    old_candidate = backend._preview_waveform_build(0.75, 1.0).power_metrics
    assert power_difference_db(old_baseline.cycle_rms, old_candidate.cycle_rms) > 0.1
    contract = backend._prepare_power_contract(0.75, 1.0)
    baseline = backend._commit_waveform_build(0.5, 1.0, contract.locked_rms, contract.contract_id)
    candidate = backend._commit_waveform_build(0.75, 1.0, contract.locked_rms, contract.contract_id)
    assert power_difference_db(
        baseline.power_metrics.cycle_rms, candidate.power_metrics.cycle_rms
    ) <= 0.1
