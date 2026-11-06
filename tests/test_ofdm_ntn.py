"""Numerical checks for the independent, reproducible NTN OFDM experiment."""

from dataclasses import FrozenInstanceError, replace
import math
import threading

import numpy as np
import pytest

from waveform_sim.simulation import ofdm_ntn
from waveform_sim.simulation.ofdm_ntn import (
    OfdmNtnConfig, config_summary, run_study, simulate_frame,
)


def awgn_config(**overrides):
    values = dict(
        profile="AWGN", fft_len=128, cp_len=16, subcarrier_spacing_hz=30_000.0,
        symbols_per_frame=12, modulation="QPSK", ebn0_db=10.0,
        velocity_kmh=0.0, doppler_spread_hz=0.0, compensation_ratio=1.0,
        residual_cfo_hz=0.0, estimator="ideal", equalizer="zf", seed=23,
    )
    values.update(overrides)
    return OfdmNtnConfig(**values).validated()


@pytest.mark.parametrize("modulation", ["QPSK", "16QAM", "64QAM"])
@pytest.mark.parametrize("estimator", ["pilot_ls", "ideal"])
def test_high_snr_flat_link_decodes_actual_bits(modulation, estimator):
    result = simulate_frame(awgn_config(modulation=modulation, estimator=estimator, ebn0_db=100.0))
    assert result["bit_errors"] == 0
    assert result["frame_error"] == 0
    assert result["total_bits"] == len(result["tx_bits"])
    np.testing.assert_array_equal(result["rx_bits"], result["tx_bits"])
    assert result["evm_percent"] < 0.05


@pytest.mark.parametrize("ebn0_db", [0.0, 4.0])
def test_awgn_qpsk_bit_counts_match_theory_with_frame_energy_overhead(ebn0_db):
    config = awgn_config(ebn0_db=ebn0_db)
    errors = bits = 0
    expected_errors = variance = 0.0
    for frame_index in range(60):
        result = simulate_frame(config, frame_index=frame_index)
        count = result["total_bits"]
        errors += result["bit_errors"]
        bits += count
        # Es=1 and a unitary FFT give data-bit Eb/N0=1/(2*sigma_complex^2).
        # This includes the actual pilot/CP overhead rather than treating the
        # experiment's full-frame Eb/N0 as the payload-only theoretical Eb/N0.
        effective_ebn0 = 1.0 / (2.0 * result["noise_variance"])
        probability = 0.5 * math.erfc(math.sqrt(effective_ebn0))
        expected_errors += count * probability
        variance += count * probability * (1.0 - probability)
        assert result["bit_errors"] == np.count_nonzero(result["tx_bits"] != result["rx_bits"])
    assert bits >= 100_000
    assert abs(errors - expected_errors) <= 6.0 * math.sqrt(variance) + 2.0


def test_noise_is_referenced_to_transmitted_frame_energy_including_cp_and_pilot():
    for config in (awgn_config(ebn0_db=3.0), awgn_config(ebn0_db=3.0, cp_len=32)):
        result = simulate_frame(config, frame_index=7)
        tx_energy = float(np.vdot(result["tx_samples"], result["tx_samples"]).real)
        assert result["tx_energy"] == pytest.approx(tx_energy, rel=1e-12)
        expected = tx_energy / (result["total_bits"] * 10.0 ** (config.ebn0_db / 10.0))
        assert result["noise_variance"] == pytest.approx(expected, rel=1e-12)
    # Multipath fading must not redefine injected noise from received power.
    faded = simulate_frame(replace(awgn_config(ebn0_db=3.0), profile="NTN-TDLA100-200"))
    expected = faded["tx_energy"] / (faded["total_bits"] * 10.0 ** 0.3)
    assert faded["noise_variance"] == pytest.approx(expected, rel=1e-12)


def test_frame_reproducibility_and_receiver_comparisons_share_the_same_received_samples():
    config = awgn_config(ebn0_db=7.0)
    first = simulate_frame(config, frame_index=4)
    again = simulate_frame(config, frame_index=4)
    pilot = simulate_frame(replace(config, estimator="pilot_ls"), frame_index=4)
    for name in ("tx_bits", "tx_samples", "rx_samples", "channel_true"):
        np.testing.assert_array_equal(first[name], again[name])
        np.testing.assert_array_equal(first[name], pilot[name])
    other_frame = simulate_frame(config, frame_index=5)
    assert not np.array_equal(first["tx_bits"], other_frame["tx_bits"])


def test_snr_changes_scale_one_common_noise_realization_without_changing_bits_or_channel():
    config = awgn_config(ebn0_db=0.0)
    results = [simulate_frame(replace(config, ebn0_db=value), frame_index=9) for value in (0.0, 6.0, 12.0)]
    for item in results[1:]:
        np.testing.assert_array_equal(item["tx_bits"], results[0]["tx_bits"])
        np.testing.assert_array_equal(item["channel_true"], results[0]["channel_true"])
    common_noise = []
    for left, right in zip(results, results[1:]):
        scale = math.sqrt(left["noise_variance"]) - math.sqrt(right["noise_variance"])
        common_noise.append((left["rx_samples"] - right["rx_samples"]) / scale)
    np.testing.assert_allclose(common_noise[0], common_noise[1], rtol=1e-10, atol=1e-10)


def test_fractional_ntn_paths_do_not_collapse_to_zero_sample_delay():
    config = awgn_config(profile="NTN-TDLA100-200", fft_len=64, subcarrier_spacing_hz=15_000.0)
    summary = config_summary(config)
    # 285 ns at 0.96 Msps is 0.2736 samples: integer rounding erases this path.
    assert summary["max_delay_samples"] == pytest.approx(0.2736)
    result = simulate_frame(replace(config, ebn0_db=100.0), frame_index=2)
    response = np.asarray(result["channel_true"])
    assert np.max(np.abs(response - response.reshape(-1)[0])) > 1e-3


def test_insufficient_cp_has_real_isi_while_a_sufficient_cp_recovers_the_link():
    config = awgn_config(
        profile="NTN-TDLA100-200", modulation="64QAM", ebn0_db=100.0,
        delay_scale=10.0 / (285e-9 * 128 * 30_000.0),
    )
    short_errors = long_errors = 0
    short_evm = long_evm = 0.0
    for frame_index in range(12):
        short = simulate_frame(replace(config, cp_len=0), frame_index=frame_index)
        long = simulate_frame(replace(config, cp_len=16), frame_index=frame_index)
        np.testing.assert_array_equal(short["tx_bits"], long["tx_bits"])
        short_errors += short["bit_errors"]
        long_errors += long["bit_errors"]
        short_evm += short["evm_percent"]
        long_evm += long["evm_percent"]
    assert long_errors == 0
    assert short_errors > 0
    assert short_evm > 20.0 * long_evm


def test_common_doppler_precompensation_reduces_real_ofdm_impairment():
    config = awgn_config(
        ebn0_db=35.0, carrier_frequency_hz=2e9,
        velocity_kmh=28_080.0, radial_projection=0.5, compensation_ratio=0.999,
    )
    compensated = simulate_frame(config, frame_index=3)
    uncompensated = simulate_frame(replace(config, compensation_ratio=0.0), frame_index=3)
    np.testing.assert_array_equal(compensated["tx_bits"], uncompensated["tx_bits"])
    assert compensated["evm_percent"] < 0.2 * uncompensated["evm_percent"]
    assert compensated["bit_errors"] < uncompensated["bit_errors"]


def test_doppler_metadata_and_the_applied_sample_phase_use_the_same_units():
    config = awgn_config(
        carrier_frequency_hz=2e9, velocity_kmh=28_080.0,
        radial_projection=0.5, compensation_ratio=0.999, residual_cfo_hz=75.0,
    )
    summary = config_summary(config)
    orbit = (28_080.0 / 3.6) * 0.5 * 2e9 / 299_792_458.0
    expected_residual = orbit * 0.001 + 75.0
    assert summary["orbit_doppler_hz"] == pytest.approx(orbit)
    assert summary["residual_cfo_hz"] == pytest.approx(expected_residual)
    assert summary["normalized_cfo"] == pytest.approx(expected_residual / 30_000.0)
    assert summary["sample_rate_hz"] == 128 * 30_000.0
    result = simulate_frame(config)
    # Remove the known flat-channel transmit samples, leaving the actual
    # noiseless sample-to-sample carrier rotation, not a reported formula.
    rotation = result["rx_clean_samples"] * np.conj(result["tx_samples"])
    product = np.sum(rotation[1:] * np.conj(rotation[:-1]))
    measured_hz = np.angle(product) * result["sample_rate_hz"] / (2.0 * np.pi)
    assert measured_hz == pytest.approx(expected_residual, abs=1e-6)


def test_pilot_ls_estimation_error_is_real_and_improves_with_pilot_snr():
    config = awgn_config(estimator="pilot_ls", ebn0_db=0.0)
    low = simulate_frame(config, frame_index=6)
    high = simulate_frame(replace(config, ebn0_db=30.0), frame_index=6)
    ideal = simulate_frame(replace(config, estimator="ideal"), frame_index=6)
    assert low["channel_nmse"] > 0.0
    assert 0.0 < high["channel_nmse"] < 0.01 * low["channel_nmse"]
    assert ideal["channel_nmse"] == 0.0
    np.testing.assert_array_equal(low["rx_samples"], ideal["rx_samples"])
    assert not np.array_equal(low["channel_est"], ideal["channel_est"])


def test_zero_error_study_reports_zero_ber_with_a_nonzero_uncertainty_bound():
    point = run_study(awgn_config(ebn0_db=100.0), "ebn0", [100.0], frames=3)[0]
    assert point["axis_value"] == 100.0
    assert set(point["strategies"]) == {"pilot_ls", "ideal", "uncompensated"}
    for stats in point["strategies"].values():
        assert stats["frames"] == 3
        assert stats["bit_errors"] == 0 and stats["ber"] == 0.0
        assert stats["total_bits"] > 0
        assert stats["ci_low"] == pytest.approx(0.0, abs=1e-15)
        assert 0.0 < stats["ci_high"] < 0.01


def test_study_replays_frame_indices_for_each_snr_and_counts_raw_errors(monkeypatch):
    config = awgn_config()
    records = []
    original = ofdm_ntn.simulate_frame

    def record(config, frame_index=0):
        result = original(config, frame_index=frame_index)
        records.append((config.ebn0_db, config.estimator, frame_index, result))
        return result

    monkeypatch.setattr(ofdm_ntn, "simulate_frame", record)
    points = run_study(config, "ebn0", [0.0, 5.0], frames=2)
    assert len(points) == 2
    for point in points:
        for stats in point["strategies"].values():
            assert stats["ber"] == stats["bit_errors"] / stats["total_bits"]
            assert 0.0 <= stats["ci_low"] <= stats["ber"] <= stats["ci_high"] <= 1.0
    for index in (0, 1):
        samples = [result for _snr, _estimator, frame, result in records if frame == index]
        assert len(samples) == 6
        for result in samples[1:]:
            np.testing.assert_array_equal(result["tx_bits"], samples[0]["tx_bits"])
            np.testing.assert_array_equal(result["channel_true"], samples[0]["channel_true"])


def test_study_cancellation_before_start_or_after_a_completed_point():
    stop = threading.Event()
    stop.set()
    assert run_study(awgn_config(), "ebn0", [0.0, 5.0], frames=2, stop_event=stop) == []
    stop.clear()
    points_seen = []

    def on_point(point):
        points_seen.append(point)
        stop.set()

    points = run_study(awgn_config(), "ebn0", [0.0, 5.0], frames=2,
                       stop_event=stop, on_point=on_point)
    assert len(points_seen) == len(points) == 1
    assert points[0]["axis_value"] == 0.0


def test_cancellation_during_a_receiver_variant_discards_the_unmatched_frame(monkeypatch):
    stop = threading.Event()
    calls = []
    original = ofdm_ntn.simulate_frame

    def stop_after_first_variant(config, frame_index=0):
        calls.append(frame_index)
        result = original(config, frame_index=frame_index)
        stop.set()
        return result

    monkeypatch.setattr(ofdm_ntn, "simulate_frame", stop_after_first_variant)
    callbacks = []
    result = run_study(awgn_config(), "ebn0", [0.0, 5.0], frames=10,
                       stop_event=stop, on_point=callbacks.append)
    assert calls == [0]
    assert result == [] and callbacks == []


@pytest.mark.parametrize("axis,values", [
    ("residual_cfo", [-300.0, 300.0]),
    ("spacing", [15_000.0, 30_000.0]),
    ("delay_scale", [0.5, 2.0]),
])
def test_each_research_axis_runs_a_complete_equal_budget_comparison(axis, values):
    points = run_study(awgn_config(profile="NTN-TDLA100-200"), axis, values, frames=1)
    assert [point["axis_value"] for point in points] == values
    for point in points:
        stats = list(point["strategies"].values())
        assert all(item["frames"] == 1 for item in stats)
        assert len({item["total_bits"] for item in stats}) == 1


@pytest.mark.parametrize("changes", [
    {"fft_len": 63}, {"cp_len": -1}, {"cp_len": 128},
    {"subcarrier_spacing_hz": 0.0}, {"subcarrier_spacing_hz": float("nan")},
    {"symbols_per_frame": 0}, {"modulation": "unsupported"},
    {"ebn0_db": float("nan")}, {"compensation_ratio": -0.1},
    {"compensation_ratio": 1.1}, {"delay_scale": -1.0},
    {"estimator": "oracle_magic"}, {"equalizer": "unsupported"},
])
def test_invalid_model_configuration_is_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        replace(awgn_config(), **changes).validated()


def test_config_is_immutable_and_invalid_study_arguments_are_rejected():
    config = awgn_config()
    with pytest.raises(FrozenInstanceError):
        config.ebn0_db = 2.0
    with pytest.raises(ValueError):
        run_study(config, "not_an_axis", [1.0], frames=1)
    with pytest.raises(ValueError):
        run_study(config, "ebn0", [0.0], frames=0)
