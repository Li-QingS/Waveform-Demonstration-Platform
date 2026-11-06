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
    full = frame(simulator, snr_db=50.0, cfo_hz=7500.0, receiver_mode="full_mmse")
    practical = frame(simulator, snr_db=50.0, cfo_hz=7500.0, receiver_mode="practical")
    assert all(result["receiver_mode"] == "full_mmse" for result in full.values())
    assert all(result["receiver_mode"] == "practical" for result in practical.values())
    assert full["OFDM"]["bit_errors"] == 0
    assert practical["OFDM"]["bit_errors"] > 0
    assert practical["OFDM"]["evm_percent"] > 10 * full["OFDM"]["evm_percent"]
    for name in ("OTFS", "AFDM"):
        np.testing.assert_allclose(full[name]["rx_symbols"], practical[name]["rx_symbols"], atol=1e-12)


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
    zero = simulator.simulate_stats(WAVEFORMS, frames=2, snr_db=80.0, cfo_hz=0.0,
                                    doppler_hz=0.0, seed=9)
    for stats in zero.values():
        assert stats["bit_errors"] == 0 and stats["ber"] == 0.0
        assert stats["ci_low"] == 0.0 and 0 < stats["ci_high"] < 0.1


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
    {"receiver_mode": "unstated_budget"}, {"frame_index": -1},
])
def test_invalid_frame_arguments_are_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        frame(small_simulator(), **changes)
