"""Physical fairness, raw counters and cancellation for waveform comparison."""

from dataclasses import replace
import math
import threading

import numpy as np
import pytest

from waveform_sim.simulation.compare_scan_backend import (
    ScanConfig, WaveformScanSimulator, scan_axis,
)


WAVEFORMS = ("OFDM", "OTFS", "AFDM")


def small_simulator():
    return WaveformScanSimulator(ScanConfig(fft_len=16, n_slots=3, cp_len=4, channel_model="AWGN"))


def frame(simulator, waveforms=WAVEFORMS, **overrides):
    arguments = dict(snr_db=12.0, cfo_hz=0.0, doppler_hz=0.0,
                     mod_order="QPSK", seed=37, frame_index=0,
                     receiver_mode="full_mmse")
    arguments.update(overrides)
    return simulator.simulate_frame_set(waveforms, **arguments)


def test_default_waveforms_share_full_resource_bits_channel_noise_and_frame_energy():
    simulator = WaveformScanSimulator()
    results = frame(simulator, mod_order="16QAM")
    baseline = results["OFDM"]
    assert baseline["receiver_coefficients_per_row"] == 64
    assert baseline["receiver_complexity_proxy"] == 7 * 64 ** 3
    assert baseline["receiver_note"].endswith("完美 CSI 理想基准")
    assert set(results) == set(WAVEFORMS)
    for result in results.values():
        assert result["total_bits"] == 64 * 7 * 4
        assert len(result["tx_bits"]) == result["total_bits"]
        np.testing.assert_array_equal(result["tx_bits"], baseline["tx_bits"])
        np.testing.assert_array_equal(result["noise_samples"], baseline["noise_samples"])
        for name in ("delays", "gains", "dopplers_hz", "matrices"):
            np.testing.assert_array_equal(result["channel"][name], baseline["channel"][name])
        measured_energy = float(np.vdot(result["tx_samples"], result["tx_samples"]).real)
        assert measured_energy == pytest.approx(7 * (64 + 16), rel=1e-12)
        assert result["tx_energy"] == pytest.approx(measured_energy, rel=1e-12)
        expected_noise = measured_energy / (result["total_bits"] * 10.0 ** 1.2)
        assert result["noise_variance"] == pytest.approx(expected_noise, rel=1e-12)
        assert result["ber"] == result["bit_errors"] / result["total_bits"]


def test_random_realization_does_not_depend_on_waveform_order_or_duplicate_selection():
    simulator = small_simulator()
    normal = frame(simulator, frame_index=4)
    reverse = frame(simulator, tuple(reversed(WAVEFORMS)), frame_index=4)
    duplicate = frame(simulator, ("OFDM", "OFDM"), frame_index=4)
    assert list(duplicate) == ["OFDM"]
    for name in WAVEFORMS:
        for field in ("tx_bits", "tx_samples", "rx_samples", "noise_samples", "rx_symbols"):
            np.testing.assert_array_equal(normal[name][field], reverse[name][field])
        assert normal[name]["bit_errors"] == reverse[name]["bit_errors"]
    np.testing.assert_array_equal(duplicate["OFDM"]["rx_symbols"], normal["OFDM"]["rx_symbols"])
    next_frame = frame(simulator, frame_index=5)
    assert not np.array_equal(normal["OFDM"]["tx_bits"], next_frame["OFDM"]["tx_bits"])


def test_seed_replays_physical_realization_and_noise_scaling_across_snr_points():
    simulator = small_simulator()
    lower = frame(simulator, snr_db=0.0)
    higher = frame(simulator, snr_db=10.0)
    for name in WAVEFORMS:
        np.testing.assert_array_equal(lower[name]["tx_bits"], higher[name]["tx_bits"])
        np.testing.assert_array_equal(lower[name]["tx_samples"], higher[name]["tx_samples"])
        np.testing.assert_array_equal(lower[name]["channel"]["gains"], higher[name]["channel"]["gains"])
        np.testing.assert_allclose(
            lower[name]["noise_samples"] / math.sqrt(lower[name]["noise_variance"]),
            higher[name]["noise_samples"] / math.sqrt(higher[name]["noise_variance"]),
            rtol=1e-12, atol=1e-12,
        )


def test_receiver_channel_matrix_matches_the_actual_cp_protected_time_domain_channel():
    simulator = WaveformScanSimulator(ScanConfig(fft_len=16, n_slots=3, cp_len=8))
    results = frame(simulator, cfo_hz=7000.0, doppler_hz=500.0, mod_order="16QAM", frame_index=3)
    for result in results.values():
        transmitted_useful = result["tx_samples"].reshape(3, 24)[:, 8:]
        received_useful = result["rx_clean_samples"].reshape(3, 24)[:, 8:]
        predicted = np.einsum("tij,tj->ti", result["channel"]["matrices"], transmitted_useful)
        np.testing.assert_allclose(received_useful, predicted, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("modulation", ["QPSK", "16QAM", "64QAM"])
def test_each_waveform_recovers_bits_at_high_snr_without_impairments(modulation):
    results = frame(small_simulator(), snr_db=80.0, mod_order=modulation)
    for result in results.values():
        assert result["bit_errors"] == 0
        assert result["ber"] == 0.0 and result["frame_error"] == 0
        assert result["evm_percent"] < 0.1


def test_full_receiver_and_practical_receiver_are_explicitly_different_experiments():
    simulator = small_simulator()
    full = frame(simulator, snr_db=50.0, cfo_hz=7500.0,
                 receiver_mode="full_mmse", csi_mode="pilot")
    practical = frame(simulator, snr_db=50.0, cfo_hz=7500.0,
                      receiver_mode="practical", csi_mode="pilot")
    assert all(result["receiver_mode"] == "full_mmse" for result in full.values())
    assert all(result["receiver_mode"] == "practical" for result in practical.values())
    assert full["OFDM"]["receiver_complexity_proxy"] > practical["OFDM"]["receiver_complexity_proxy"]
    for name in ("OTFS", "AFDM"):
        assert full[name]["receiver_method"] == "full_lmmse_time_domain"
        assert practical[name]["receiver_method"] == "sparse_pcg_lmmse"
        assert practical[name]["receiver_complexity_proxy"] < full[name]["receiver_complexity_proxy"]


def test_shared_pilot_mode_uses_one_common_training_symbol_and_estimated_csi():
    results = frame(small_simulator(), csi_mode="pilot")
    baseline = results["OFDM"]

    assert baseline["csi_mode"] == "pilot"
    assert baseline["pilot_ofdm_symbols"] == 1
    assert baseline["pilot_overhead_fraction"] == pytest.approx(1 / 4)
    assert baseline["csi_nmse"] > 0.0
    assert baseline["pilot_symbols"] == 16
    assert "CFO/多普勒未知" in baseline["receiver_note"]
    for result in results.values():
        assert result["total_bits"] == baseline["total_bits"]
        assert result["tx_energy"] == pytest.approx(baseline["tx_energy"], rel=1e-12)
        assert result["noise_variance"] == pytest.approx(baseline["noise_variance"], rel=1e-12)
        assert result["csi_nmse"] == pytest.approx(baseline["csi_nmse"], rel=1e-12)
        np.testing.assert_array_equal(
            result["channel"]["estimated_matrices"],
            baseline["channel"]["estimated_matrices"],
        )
        np.testing.assert_array_equal(result["pilot_noise_samples"], baseline["pilot_noise_samples"])


def test_pilot_receiver_uses_static_estimate_without_oracle_cfo_or_doppler():
    simulator = WaveformScanSimulator(
        ScanConfig(fft_len=16, n_slots=3, cp_len=6, channel_model="multipath")
    )
    result = frame(
        simulator, snr_db=80.0, cfo_hz=1500.0, doppler_hz=500.0,
        receiver_mode="full_mmse", csi_mode="pilot",
    )["OFDM"]
    true_matrix = result["channel"]["matrices"][0]
    estimated_matrices = result["channel"]["estimated_matrices"]

    np.testing.assert_array_equal(np.argwhere(true_matrix != 0), np.argwhere(estimated_matrices[0] != 0))
    for estimated_matrix in estimated_matrices[1:]:
        np.testing.assert_allclose(estimated_matrix, estimated_matrices[0], atol=1e-12)
    assert result["csi_nmse"] > 1e-3
    clean_pilot = frame(
        simulator, snr_db=80.0, cfo_hz=0.0, doppler_hz=0.0,
        receiver_mode="full_mmse", csi_mode="pilot",
    )["OFDM"]
    assert clean_pilot["csi_nmse"] < 1e-4


@pytest.mark.parametrize(
    ("low_cfo", "low_doppler", "high_cfo", "high_doppler"),
    [(0.0, 0.0, 1500.0, 1200.0)],
)
def test_pilot_csi_scan_exposes_unestimated_cfo_and_doppler_sensitivity(
    low_cfo, low_doppler, high_cfo, high_doppler,
):
    simulator = WaveformScanSimulator(
        ScanConfig(fft_len=16, n_slots=3, cp_len=6, channel_model="multipath")
    )
    options = dict(waveforms=["OFDM"], snr_db=14.0, mod_order="QPSK", frames=12,
                   seed=42, receiver_mode="full_mmse", csi_mode="pilot")
    baseline = simulator.simulate_stats(
        **options, cfo_hz=low_cfo, doppler_hz=low_doppler,
    )["OFDM"]["ber"]
    cfo_stress = simulator.simulate_stats(
        **options, cfo_hz=high_cfo, doppler_hz=low_doppler,
    )["OFDM"]["ber"]
    doppler_stress = simulator.simulate_stats(
        **options, cfo_hz=low_cfo, doppler_hz=high_doppler,
    )["OFDM"]["ber"]

    assert cfo_stress > baseline + 0.02
    assert doppler_stress > baseline + 0.01


def test_ideal_csi_mode_has_no_training_overhead():
    result = frame(small_simulator())["OFDM"]

    assert result["csi_mode"] == "ideal"
    assert result["pilot_ofdm_symbols"] == 0
    assert result["pilot_overhead_fraction"] == 0.0
    assert result["csi_nmse"] == 0.0


def test_practical_receivers_use_the_shared_estimated_channel():
    results = frame(small_simulator(), receiver_mode="practical", csi_mode="ideal")

    assert results["OFDM"]["receiver_mode"] == "practical"
    assert results["OFDM"]["csi_mode"] == "pilot"
    assert results["OFDM"]["receiver_coefficients_per_row"] == 1
    for result in results.values():
        assert result["csi_nmse"] > 0.0
        assert result["pilot_overhead_fraction"] == pytest.approx(1 / 4)
        assert not np.array_equal(
            result["channel"]["estimated_matrices"], result["channel"]["matrices"]
        )


def test_practical_otfs_afdm_use_sparse_budgeted_lmmse():
    simulator = WaveformScanSimulator(
        ScanConfig(fft_len=16, n_slots=3, cp_len=6, channel_model="multipath")
    )
    results = frame(simulator, receiver_mode="practical", csi_mode="pilot")
    ofdm_proxy = results["OFDM"]["receiver_complexity_proxy"]

    assert results["OFDM"]["receiver_method"] == "ofdm_one_tap"
    assert results["OFDM"]["receiver_complexity_proxy"] == 3 * 16
    for name in ("OTFS", "AFDM"):
        assert results[name]["receiver_method"] == "sparse_pcg_lmmse"
        assert 1 <= results[name]["receiver_iterations_max"] <= 2
        assert results[name]["receiver_coefficients_per_row"] < 16 ** 2
        assert results[name]["receiver_complexity_proxy"] < 16 ** 3
        assert results[name]["receiver_complexity_proxy"] < ofdm_proxy * 20


def test_unified_baseline_reports_same_full_lmmse_method_for_all_waveforms():
    results = frame(small_simulator())

    assert {result["receiver_method"] for result in results.values()} == {
        "full_lmmse_time_domain"
    }


def test_awgn_bit_counts_match_qpsk_theory_including_actual_cp_energy():
    simulator = small_simulator()
    observed = {name: 0 for name in WAVEFORMS}
    expected = {name: 0.0 for name in WAVEFORMS}
    variances = {name: 0.0 for name in WAVEFORMS}
    for index in range(100):
        results = frame(simulator, snr_db=1.0, frame_index=index)
        for name, result in results.items():
            useful = result["tx_samples"].reshape(3, 16 + 4)[:, 4:]
            symbol_energy = float(np.mean(np.abs(useful) ** 2))
            probability = 0.5 * math.erfc(math.sqrt(symbol_energy / (2 * result["noise_variance"])))
            bits = result["total_bits"]
            observed[name] += result["bit_errors"]
            expected[name] += bits * probability
            variances[name] += bits * probability * (1 - probability)
    for name in WAVEFORMS:
        assert abs(observed[name] - expected[name]) <= 6 * math.sqrt(variances[name]) + 2


def test_cumulative_statistics_pool_exact_counts_and_keep_zero_ber_uncertainty():
    simulator = small_simulator()
    arguments = dict(snr_db=5.0, cfo_hz=1200.0, doppler_hz=0.0,
                     mod_order="16QAM", seed=9, receiver_mode="full_mmse")
    cumulative = simulator.simulate_stats(WAVEFORMS, frames=3, **arguments)
    frames = [simulator.simulate_frame_set(WAVEFORMS, frame_index=index, **arguments) for index in range(3)]
    for name, stats in cumulative.items():
        assert stats["frames"] == 3
        assert stats["bit_errors"] == sum(item[name]["bit_errors"] for item in frames)
        assert stats["total_bits"] == sum(item[name]["total_bits"] for item in frames)
        assert stats["ber"] == stats["bit_errors"] / stats["total_bits"]
        assert 0 <= stats["ci_low"] <= stats["ber"] <= stats["ci_high"] <= 1
        assert stats["ci_method"] in {
            "frame_cluster_bootstrap",
            "zero_error_frame_fer_upper_bound",
            "insufficient_frames_worst_case",
        }
        assert stats["independent_frames"] == 3
    zero = simulator.simulate_stats(WAVEFORMS, frames=2, snr_db=80.0, cfo_hz=0.0,
                                    doppler_hz=0.0, seed=9)
    for stats in zero.values():
        assert stats["bit_errors"] == 0 and stats["ber"] == 0.0
        assert stats["ci_low"] == 0.0 and 0 < stats["ci_high"] <= 1.0
        assert stats["ci_method"] == "zero_error_frame_fer_upper_bound"
    paired = zero["OFDM"]["paired_ber_difference_ci"]["OTFS"]
    assert paired["ci_low"] < 0.0 < paired["ci_high"]
    assert paired["ci_method"] == "zero_error_paired_fer_bounds"


def test_frame_cluster_interval_reflects_errors_concentrated_in_one_fade():
    from waveform_sim.simulation.compare_scan_backend import (
        _frame_cluster_bootstrap_interval,
        _wilson,
    )

    frame_errors = [34] + [0] * 9
    frame_bits = [896] * 10
    low, high, _method = _frame_cluster_bootstrap_interval(frame_errors, frame_bits)
    iid_high = _wilson(sum(frame_errors), sum(frame_bits))[1]

    assert low <= sum(frame_errors) / sum(frame_bits) <= high
    assert high > iid_high


def test_zero_error_frame_bound_respects_requested_confidence():
    from waveform_sim.simulation.compare_scan_backend import _frame_cluster_bootstrap_interval

    frame_errors = [0] * 10
    frame_bits = [896] * 10
    upper_90 = _frame_cluster_bootstrap_interval(
        frame_errors, frame_bits, confidence=0.90
    )[1]
    upper_99 = _frame_cluster_bootstrap_interval(
        frame_errors, frame_bits, confidence=0.99
    )[1]

    assert upper_90 < upper_99


def test_paired_frame_bootstrap_keeps_shared_fade_outcomes_together():
    from waveform_sim.simulation.compare_scan_backend import _paired_frame_difference_interval

    common_fade_errors = [34] + [0] * 9
    bits = [896] * 10
    low, high, _method = _paired_frame_difference_interval(
        common_fade_errors,
        bits,
        common_fade_errors,
        bits,
    )

    assert low == high == 0.0


def test_zero_error_paired_interval_does_not_claim_zero_uncertainty():
    from waveform_sim.simulation.compare_scan_backend import _paired_frame_difference_interval

    low, high, method = _paired_frame_difference_interval(
        [0] * 10, [896] * 10, [0] * 10, [896] * 10,
    )

    assert low < 0.0 < high
    assert method == "zero_error_paired_fer_bounds"


def test_legacy_scan_axis_returns_raw_zero_instead_of_a_fake_ber_floor():
    result = scan_axis("cfo", np.array([0.0]), snr_db=80.0, mod_order="QPSK",
                       frames=1, fixed_cfo_hz=0.0, fixed_doppler_hz=0.0, seed_base=42)
    assert set(result) == set(WAVEFORMS)
    for values in result.values():
        np.testing.assert_array_equal(values, [0.0])


def test_scan_point_iteration_order_does_not_change_its_random_experiment():
    config = ScanConfig(fft_len=16, n_slots=3, cp_len=4, channel_model="AWGN")
    values = np.array([-1200.0, 2200.0])
    parameters = dict(snr_db=6.0, mod_order="QPSK", frames=2, fixed_cfo_hz=0.0,
                      fixed_doppler_hz=0.0, seed_base=37, receiver_mode="practical", config=config)
    forward = scan_axis("cfo", values, **parameters)
    backward = scan_axis("cfo", values[::-1], **parameters)
    for name in WAVEFORMS:
        np.testing.assert_array_equal(forward[name], backward[name][::-1])


def test_cancellation_never_returns_a_zero_error_success():
    simulator = small_simulator()
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        frame(simulator, stop_event=stop)
    with pytest.raises(InterruptedError):
        simulator.simulate_stats(WAVEFORMS, snr_db=10.0, cfo_hz=0.0, doppler_hz=0.0,
                                 frames=2, stop_event=stop)


def test_cancellation_after_a_computed_frame_does_not_return_partial_statistics(monkeypatch):
    simulator = small_simulator()
    stop = threading.Event()
    calls = []
    original = simulator.simulate_frame_set

    def cancel_after_frame(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(True)
        stop.set()
        return result

    monkeypatch.setattr(simulator, "simulate_frame_set", cancel_after_frame)
    with pytest.raises(InterruptedError):
        simulator.simulate_stats(WAVEFORMS, snr_db=10.0, cfo_hz=0.0, doppler_hz=0.0,
                                 frames=10, stop_event=stop)
    assert calls == [True]


def test_frame_failure_propagates_instead_of_becoming_zero_ber(monkeypatch):
    simulator = small_simulator()

    def fail_frame(*_args, **_kwargs):
        raise ValueError("invalid physical frame")

    monkeypatch.setattr(simulator, "simulate_frame_set", fail_frame)
    with pytest.raises(ValueError, match="invalid physical frame"):
        simulator.simulate_stats(WAVEFORMS, snr_db=10.0, cfo_hz=0.0, doppler_hz=0.0, frames=2)


@pytest.mark.parametrize("changes", [
    {"fft_len": 7}, {"n_slots": 0}, {"cp_len": -1}, {"cp_len": 64},
    {"subcarrier_spacing": 0.0}, {"channel_model": "unknown"},
    {"cp_len": 4, "channel_model": "multipath"},
])
def test_invalid_resource_configuration_is_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        WaveformScanSimulator(replace(ScanConfig(), **changes))


@pytest.mark.parametrize("changes", [
    {"snr_db": float("nan")}, {"cfo_hz": float("inf")},
    {"doppler_hz": -1.0}, {"mod_order": "unsupported"},
    {"receiver_mode": "unstated_budget"}, {"csi_mode": "unknown"}, {"frame_index": -1},
])
def test_invalid_frame_arguments_are_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        frame(small_simulator(), **changes)
