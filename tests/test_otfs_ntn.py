"""Independent signal, energy and statistical checks for the NTN OTFS link."""

from dataclasses import FrozenInstanceError, replace
import math
import threading

import numpy as np
import pytest

from waveform_sim.simulation import otfs_ntn
from waveform_sim.simulation.otfs_ntn import (
    OtfsNtnConfig, config_summary, otfs_isfft, otfs_sfft, run_study,
    simulate_frame,
)


def awgn_config(**overrides):
    values = dict(
        profile="AWGN", fft_len=64, cp_len=8, subcarrier_spacing_hz=30_000.0,
        symbols_per_frame=8, modulation="QPSK", ebn0_db=10.0,
        velocity_kmh=0.0, doppler_spread_hz=0.0, compensation_ratio=1.0,
        residual_cfo_hz=0.0, estimator="ideal", equalizer="zf", seed=23,
    )
    values.update(overrides)
    return OtfsNtnConfig(**values).validated()


def test_otfs_transform_axes_match_the_explicit_symplectic_fourier_sum():
    # A non-square, non-power-of-two Doppler dimension exposes swapped axes,
    # transposes and a mistaken use of FFT in both directions.
    rng = np.random.default_rng(17)
    dd = rng.standard_normal((8, 5)) + 1j * rng.standard_normal((8, 5))
    delay = np.arange(8)
    doppler = np.arange(5)
    expected = (np.exp(2j * np.pi * doppler[:, None] * doppler[None, :] / 5)
                @ dd.T
                @ np.exp(-2j * np.pi * delay[:, None] * delay[None, :] / 8)) / math.sqrt(40)
    tf = otfs_isfft(dd)
    assert tf.shape == (5, 8)
    np.testing.assert_allclose(tf, expected, atol=1e-14)
    np.testing.assert_allclose(otfs_sfft(tf), dd, atol=1e-14)
    assert np.vdot(tf, tf).real == pytest.approx(np.vdot(dd, dd).real, rel=1e-13)


@pytest.mark.parametrize("modulation", ["QPSK", "16QAM", "64QAM"])
@pytest.mark.parametrize("estimator", ["training_ls", "ideal"])
def test_flat_high_snr_link_recovers_actual_dd_qam_bits(modulation, estimator):
    frame = simulate_frame(awgn_config(modulation=modulation, estimator=estimator, ebn0_db=100.0))
    assert frame["bit_errors"] == frame["frame_error"] == 0
    assert frame["total_bits"] == frame["tx_bits"].size
    np.testing.assert_array_equal(frame["tx_bits"], frame["rx_bits"])
    assert frame["tx_dd"].shape == frame["rx_dd"].shape == (64, 8)
    np.testing.assert_allclose(otfs_sfft(frame["tx_tf"]), frame["tx_dd"], atol=2e-14)
    assert frame["evm_percent"] < 0.05


@pytest.mark.parametrize("ebn0_db", [0.0, 4.0])
def test_awgn_bit_counts_match_qpsk_theory_after_training_and_cp_energy(ebn0_db):
    config = awgn_config(ebn0_db=ebn0_db)
    errors = bits = 0
    expected = variance = 0.0
    for index in range(100):
        result = simulate_frame(config, index)
        count = result["total_bits"]
        # Unitary transforms preserve complex Gaussian noise variance. Each
        # DD QPSK symbol has Es=1; the transmitted-frame Eb includes training
        # and CP and therefore is not the payload-only theoretical Eb.
        probability = 0.5 * math.erfc(math.sqrt(1.0 / (2.0 * result["noise_variance"])))
        errors += result["bit_errors"]
        bits += count
        expected += count * probability
        variance += count * probability * (1.0 - probability)
        assert result["bit_errors"] == np.count_nonzero(result["tx_bits"] != result["rx_bits"])
    assert bits >= 100_000
    assert abs(errors - expected) < 6.0 * math.sqrt(variance) + 2.0


def test_training_cp_and_guard_budget_matches_the_actual_transmitted_burst():
    config = awgn_config(ebn0_db=3.0)
    info = config_summary(config)
    frame = simulate_frame(config, 7)
    expected_length = (config.symbols_per_frame + 1) * (config.fft_len + config.cp_len) + 2 * config.cp_len
    assert info["frame_samples"] == expected_length == frame["tx_samples"].size
    assert frame["total_bits"] == 64 * 8 * 2
    np.testing.assert_array_equal(frame["tx_samples"][:config.cp_len], 0)
    np.testing.assert_array_equal(frame["tx_samples"][-config.cp_len:], 0)
    assert frame["tx_energy"] == pytest.approx(np.vdot(frame["tx_samples"], frame["tx_samples"]).real)
    assert frame["noise_variance"] == pytest.approx(
        frame["tx_energy"] / (frame["total_bits"] * 10.0 ** 0.3))
    # The training symbol occupies a real extra CP-OFDM slot and carries no
    # counted data bits, rather than being a free CSI side channel.
    training_time = frame["tx_samples"][2 * config.cp_len:2 * config.cp_len + config.fft_len]
    np.testing.assert_allclose(np.fft.fft(training_time, norm="ortho"), frame["training_tx_symbols"], atol=1e-13)
    assert np.sum(np.abs(training_time) ** 2) == pytest.approx(config.fft_len)


def test_dd_resolution_uses_the_physical_sample_rate_and_cp_extended_payload_duration():
    config = awgn_config()
    summary = config_summary(config)
    fs = config.fft_len * config.subcarrier_spacing_hz
    assert summary["delay_resolution_s"] == pytest.approx(1.0 / fs)
    assert summary["doppler_resolution_hz"] == pytest.approx(fs / (config.symbols_per_frame * (config.fft_len + config.cp_len)))
    doubled = config_summary(replace(config, symbols_per_frame=16))
    assert doubled["doppler_resolution_hz"] == pytest.approx(summary["doppler_resolution_hz"] / 2)
    assert doubled["delay_resolution_s"] == summary["delay_resolution_s"]
    assert doubled["training_fraction_of_tf_symbols"] < summary["training_fraction_of_tf_symbols"]


def test_channel_fading_never_redefines_the_injected_noise_power():
    config = awgn_config(ebn0_db=3.0)
    flat = simulate_frame(config, 3)
    faded = simulate_frame(replace(config, profile="NTN-TDLA100-200"), 3)
    assert faded["noise_variance"] == flat["noise_variance"]
    np.testing.assert_array_equal(faded["noise_samples"], flat["noise_samples"])
    assert not np.isclose(np.vdot(faded["rx_clean_samples"], faded["rx_clean_samples"]).real,
                          flat["tx_energy"])


def test_seeded_receiver_variants_have_identical_transmit_channel_and_noise_samples():
    config = awgn_config(profile="NTN-TDLA100-200", doppler_spread_hz=1200.0)
    first = simulate_frame(config, 4)
    again = simulate_frame(config, 4)
    training = simulate_frame(replace(config, estimator="training_ls"), 4)
    for field in ("tx_bits", "tx_samples", "rx_samples", "channel_true", "noise_samples"):
        np.testing.assert_array_equal(first[field], again[field])
        np.testing.assert_array_equal(first[field], training[field])
    next_frame = simulate_frame(config, 5)
    assert not np.array_equal(first["tx_bits"], next_frame["tx_bits"])
    assert not np.array_equal(first["noise_samples"], next_frame["noise_samples"])
    np.testing.assert_array_equal(first["path_gains"], next_frame["path_gains"])


def test_snr_scan_scales_one_common_noise_realization_and_keeps_the_waveform():
    config = awgn_config(ebn0_db=0.0)
    low = simulate_frame(config, 9)
    high = simulate_frame(replace(config, ebn0_db=10.0), 9)
    np.testing.assert_array_equal(low["tx_samples"], high["tx_samples"])
    np.testing.assert_array_equal(low["rx_clean_samples"], high["rx_clean_samples"])
    np.testing.assert_allclose(low["noise_samples"], math.sqrt(10.0) * high["noise_samples"], rtol=1e-13, atol=1e-13)


def test_training_ls_uses_only_the_received_known_symbol_not_payload_or_true_csi(monkeypatch):
    config = awgn_config(profile="NTN-TDLA100-200", estimator="training_ls", ebn0_db=15.0,
                         doppler_spread_hz=600.0)
    original = simulate_frame(config, 2)
    expected = original["training_rx_symbols"] / original["training_tx_symbols"]
    np.testing.assert_allclose(original["channel_est"], np.tile(expected, (config.symbols_per_frame, 1)))
    assert original["channel_nmse"] > 0.0
    assert not np.allclose(original["channel_est"], original["channel_true"])

    modulate = otfs_ntn.qam_modulate
    monkeypatch.setattr(otfs_ntn, "qam_modulate", lambda bits, modulation: -modulate(bits, modulation))
    opposite_payload = simulate_frame(config, 2)
    np.testing.assert_array_equal(opposite_payload["tx_dd"], -original["tx_dd"])
    # Global payload sign inversion leaves its energy, the preceding training
    # symbol, physical channel and generated noise unchanged.
    np.testing.assert_array_equal(opposite_payload["training_rx_symbols"], original["training_rx_symbols"])
    np.testing.assert_array_equal(opposite_payload["channel_est"], original["channel_est"])


def test_fractional_path_physical_fft_matches_static_diagonal_when_cp_is_sufficient():
    config = awgn_config(profile="NTN-TDLA100-200", fft_len=64, subcarrier_spacing_hz=15_000.0,
                         ebn0_db=100.0)
    assert config_summary(config)["max_delay_samples"] == pytest.approx(0.2736)
    frame = simulate_frame(config, 2)
    cp, size = config.cp_len, config.fft_len
    start = cp + (size + cp) + cp  # guard, training slot, first payload CP
    indices = start + np.arange(config.symbols_per_frame)[:, None] * (size + cp) + np.arange(size)
    clean_tf = np.fft.fft(frame["rx_clean_samples"][indices], axis=1, norm="ortho")
    np.testing.assert_allclose(clean_tf, frame["channel_true"] * frame["tx_tf"], atol=1e-12)
    assert np.max(np.abs(frame["channel_true"] - frame["channel_true"][0, 0])) > 1e-3


def test_out_of_cp_paths_create_real_cross_symbol_interference_instead_of_cyclic_wrap():
    config = awgn_config(profile="NTN-TDLA100-200", fft_len=64, modulation="64QAM", ebn0_db=100.0,
                         delay_scale=6.0 / (285e-9 * 64 * 30_000.0))
    short_errors = long_errors = 0
    short_evm = long_evm = 0.0
    for index in range(6):
        short = simulate_frame(replace(config, cp_len=0), index)
        long = simulate_frame(replace(config, cp_len=8), index)
        np.testing.assert_array_equal(short["tx_dd"], long["tx_dd"])
        short_errors += short["bit_errors"]
        long_errors += long["bit_errors"]
        short_evm += short["evm_percent"]
        long_evm += long["evm_percent"]
    assert long_errors == 0 and short_errors > 0
    assert short_evm > 100 * long_evm


def test_orbital_cfo_phase_and_frame_timeline_follow_the_reported_units():
    config = awgn_config(velocity_kmh=28080.0, radial_projection=0.5,
                         compensation_ratio=0.999, residual_cfo_hz=75.0)
    info = config_summary(config)
    orbit = (28080.0 / 3.6) * 0.5 * 2e9 / 299_792_458.0
    residual = orbit * 0.001 + 75.0
    assert info["orbit_doppler_hz"] == pytest.approx(orbit)
    assert info["residual_cfo_hz"] == pytest.approx(residual)
    frame = simulate_frame(config, 3)
    times = (np.arange(frame["tx_samples"].size) + 3 * frame["tx_samples"].size) / info["sample_rate_hz"]
    np.testing.assert_allclose(frame["rx_clean_samples"], frame["tx_samples"] * np.exp(2j * np.pi * residual * times), atol=1e-11)


def test_zero_error_study_retains_raw_zero_and_nonzero_wilson_uncertainty():
    point = run_study(awgn_config(ebn0_db=100.0), "ebn0", [100.0], frames=2)[0]
    assert set(point["strategies"]) == {"training_ls", "ideal", "uncompensated"}
    for stats in point["strategies"].values():
        assert stats["frames"] == 2
        assert stats["bit_errors"] == stats["ber"] == stats["fer"] == 0
        assert stats["total_bits"] == 2 * 64 * 8 * 2
        assert stats["ci_low"] == 0.0
        assert 0 < stats["ci_high"] < 0.01


@pytest.mark.parametrize("axis,values", [
    ("residual_cfo", [-300.0, 300.0]), ("spacing", [15_000.0, 30_000.0]),
    ("delay_scale", [0.5, 2.0]), ("doppler", [0.0, 1200.0]), ("slots", [4.0, 8.0]),
])
def test_research_axes_keep_the_same_budget_for_every_receiver(axis, values):
    points = run_study(awgn_config(profile="NTN-TDLA100-200"), axis, values, frames=1)
    assert [point["axis_value"] for point in points] == values
    for point in points:
        stats = list(point["strategies"].values())
        assert all(value["frames"] == 1 for value in stats)
        assert len({value["total_bits"] for value in stats}) == 1
        for value in stats:
            assert value["ber"] == value["bit_errors"] / value["total_bits"]
            assert 0 <= value["ci_low"] <= value["ber"] <= value["ci_high"] <= 1


def test_cancellation_discards_an_unfinished_point_even_after_complete_frames(monkeypatch):
    config = awgn_config()
    stop = threading.Event()
    original = otfs_ntn.simulate_frame
    calls = []

    def cancel_on_second_frame(config, frame_index=0, stop_event=None):
        calls.append(frame_index)
        result = original(config, frame_index)
        if frame_index == 1:
            stop.set()
        return result

    monkeypatch.setattr(otfs_ntn, "simulate_frame", cancel_on_second_frame)
    callbacks = []
    points = run_study(config, "ebn0", [0.0, 4.0], frames=3, stop_event=stop, on_point=callbacks.append)
    assert calls == [0, 0, 0, 1]
    assert points == callbacks == []


def test_cancellation_at_point_boundary_keeps_only_the_complete_paired_point():
    stop = threading.Event()
    seen = []

    def finish_one(point):
        seen.append(point)
        stop.set()

    points = run_study(awgn_config(), "ebn0", [0.0, 4.0], frames=2,
                       stop_event=stop, on_point=finish_one)
    assert len(points) == len(seen) == 1
    assert all(stats["frames"] == 2 for stats in points[0]["strategies"].values())
    assert run_study(awgn_config(), "ebn0", [0.0], frames=2, stop_event=stop) == []


@pytest.mark.parametrize("changes", [
    {"fft_len": 63}, {"cp_len": -1}, {"cp_len": 64}, {"subcarrier_spacing_hz": 0.0},
    {"symbols_per_frame": 1}, {"symbols_per_frame": 2.5}, {"symbols_per_frame": True},
    {"modulation": "unsupported"}, {"ebn0_db": float("nan")},
    {"compensation_ratio": -0.1}, {"compensation_ratio": 1.1},
    {"delay_scale": -1.0}, {"estimator": "oracle_magic"}, {"equalizer": "unsupported"},
    {"doppler_spread_hz": -1.0}, {"residual_cfo_hz": 1_000_000.0},
])
def test_invalid_model_configuration_is_rejected_before_simulation(changes):
    with pytest.raises((ValueError, TypeError)):
        replace(awgn_config(), **changes).validated()


@pytest.mark.parametrize("axis,values", [("ebn0", [0.0, float("nan")]), ("slots", [4.0, 3.5]),
                                         ("spacing", [30000.0, 0.0]), ("ebn0", []),
                                         ("ebn0", [0.0, 0.0])])
def test_entire_scan_is_validated_before_any_callback(axis, values):
    published = []
    with pytest.raises(ValueError):
        run_study(awgn_config(), axis, values, frames=1, on_point=published.append)
    assert published == []


def test_immutable_config_and_invalid_index_frames_and_axis():
    config = awgn_config()
    with pytest.raises(FrozenInstanceError):
        config.ebn0_db = 5.0
    for invalid in (-1, 1.5, True):
        with pytest.raises(ValueError):
            simulate_frame(config, invalid)
    with pytest.raises(ValueError):
        run_study(config, "invalid", [0.0], frames=1)
    with pytest.raises(ValueError):
        run_study(config, "ebn0", [0.0], frames=0)
