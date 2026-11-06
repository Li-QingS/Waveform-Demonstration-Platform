"""Independent affine-transform, prefix and physical-channel checks for AFDM NTN."""

from dataclasses import FrozenInstanceError, replace
import math
import threading

import numpy as np
import pytest

from waveform_sim.simulation import afdm_ntn
from waveform_sim.simulation.afdm_ntn import (
    AfdmNtnConfig, afdm_demodulate, afdm_modulate, chirp_prefix,
    config_summary, run_study, simulate_frame,
)


def awgn_config(**overrides):
    values = dict(
        profile="AWGN", fft_len=32, cp_len=8, subcarrier_spacing_hz=30_000.0,
        symbols_per_frame=3, modulation="QPSK", ebn0_db=10.0,
        velocity_kmh=0.0, doppler_spread_hz=0.0, compensation_ratio=1.0,
        residual_cfo_hz=0.0, estimator="ideal", equalizer="zf", seed=23,
        c1=0.01731, c2=0.001271, prefix="cpp",
    )
    values.update(overrides)
    return AfdmNtnConfig(**values).validated()


def explicit_idaft(symbols, c1, c2):
    size = symbols.shape[-1]
    n, k = np.arange(size)[:, None], np.arange(size)[None, :]
    matrix = np.exp(2j * np.pi * (c1 * n*n + n*k/size + c2 * k*k)) / math.sqrt(size)
    return symbols @ matrix.T


def test_idaft_matches_explicit_standard_chirp_sum_and_is_unitary():
    # Nontrivial coefficients expose the old convention's extra division by M.
    rng = np.random.default_rng(27)
    symbols = rng.normal(size=(3, 16)) + 1j * rng.normal(size=(3, 16))
    c1, c2 = 0.01731, 0.00673
    waveform = afdm_modulate(symbols, c1, c2)
    np.testing.assert_allclose(waveform, explicit_idaft(symbols, c1, c2), atol=2e-13)
    np.testing.assert_allclose(afdm_demodulate(waveform, c1, c2), symbols, atol=3e-14)
    assert np.vdot(waveform, waveform).real == pytest.approx(np.vdot(symbols, symbols).real, rel=1e-13)
    assert np.linalg.norm(waveform - explicit_idaft(symbols, c1/16, c2/16)) > 1.0


def test_zero_chirps_reduce_to_ordinary_fft_ofdm():
    symbols = np.arange(32) + 1j * np.arange(32)[::-1]
    expected = np.fft.ifft(symbols, norm="ortho")
    np.testing.assert_allclose(afdm_modulate(symbols, 0.0, 0.0), expected, atol=1e-13)
    np.testing.assert_allclose(afdm_demodulate(expected, 0.0, 0.0), symbols, atol=1e-13)


def test_cpp_uses_the_negative_time_chirp_phase_without_changing_payload_energy():
    size, cp, c1 = 32, 7, 0.01731
    rng = np.random.default_rng(19)
    payload = rng.normal(size=size) + 1j*rng.normal(size=size)
    prefix_time = np.arange(-cp, 0)
    factor = np.exp(-2j*np.pi*c1*(size*size + 2*size*prefix_time))
    corrected = chirp_prefix(payload, cp, c1, prefix="cpp")
    normal = chirp_prefix(payload, cp, c1, prefix="cp")
    assert corrected.shape == normal.shape == (size+cp,)
    np.testing.assert_allclose(corrected[:cp], payload[-cp:]*factor, atol=3e-13)
    np.testing.assert_array_equal(corrected[cp:], payload)
    np.testing.assert_array_equal(normal[:cp], payload[-cp:])
    assert np.linalg.norm(corrected[:cp]-normal[:cp]) > 1.0
    assert np.vdot(corrected, corrected).real == pytest.approx(np.vdot(normal, normal).real)
    np.testing.assert_array_equal(chirp_prefix(payload, 0, c1, prefix="cpp"), payload)


def test_even_fft_integer_twice_m_c1_cpp_special_case_reduces_to_cp():
    size, cp = 32, 7
    samples = np.arange(size) + 1j*np.arange(size)[::-1]
    np.testing.assert_allclose(chirp_prefix(samples, cp, 3/(2*size), prefix="cpp"),
                               chirp_prefix(samples, cp, 3/(2*size), prefix="cp"), atol=2e-11)


def test_identical_sampled_cpp_cp_bursts_have_identical_fractional_path_outputs():
    # For even M and integer 2*M*c1, CPP is exactly CP on the transmitted
    # sample grid. Their same band-limited samples cannot propagate as two
    # different continuous chirps simply because the prefix option changed.
    cfg = awgn_config(profile="NTN-TDLA100-200", c1=3/(2*32),
                      doppler_spread_hz=1200.0, residual_cfo_hz=321.0)
    corrected = simulate_frame(cfg, 2)
    ordinary = simulate_frame(replace(cfg, prefix="cp"), 2)
    assert 0 < config_summary(cfg)["max_delay_samples"] < 1
    np.testing.assert_allclose(corrected["tx_samples"], ordinary["tx_samples"], atol=3e-12)
    np.testing.assert_allclose(corrected["rx_clean_samples"], ordinary["rx_clean_samples"], atol=3e-11)
    np.testing.assert_allclose(corrected["effective_channel_true"], ordinary["effective_channel_true"], atol=3e-11)


@pytest.mark.parametrize("modulation", ["QPSK", "16QAM", "64QAM"])
@pytest.mark.parametrize("estimator", ["training_ls", "ideal"])
def test_high_snr_awgn_recovers_actual_afdm_payload_bits(modulation, estimator):
    result = simulate_frame(awgn_config(modulation=modulation, estimator=estimator, ebn0_db=100.0))
    assert result["bit_errors"] == result["frame_error"] == 0
    np.testing.assert_array_equal(result["tx_bits"], result["rx_bits"])
    assert result["bit_errors"] == np.count_nonzero(result["tx_bits"] != result["rx_bits"])
    assert result["evm_percent"] < 0.05


def test_full_ideal_matrix_can_invert_within_block_cfo_without_pretending_it_is_diagonal():
    cfg = awgn_config(residual_cfo_hz=3000.0, ebn0_db=100.0)
    result = simulate_frame(cfg, 2)
    channel = result["effective_channel_true"][0]
    off_diagonal = channel-np.diag(np.diag(channel))
    assert np.linalg.norm(off_diagonal) > 0.5
    assert result["bit_errors"] == 0
    assert result["evm_percent"] < 0.05


@pytest.mark.parametrize("ebn0_db", [0.0, 4.0])
def test_qpsk_awgn_matches_theory_using_actual_training_and_prefix_energy(ebn0_db):
    cfg = awgn_config(ebn0_db=ebn0_db, symbols_per_frame=6)
    errors = bits = 0
    expected = variance = 0.0
    for index in range(80):
        result = simulate_frame(cfg, index)
        probability = 0.5*math.erfc(math.sqrt(1/(2*result["noise_variance"])))
        count = result["total_bits"]
        errors += result["bit_errors"]
        bits += count
        expected += count*probability
        variance += count*probability*(1-probability)
    assert bits > 30_000
    assert abs(errors-expected) < 6*math.sqrt(variance)+2


def test_transmitted_burst_accounts_for_known_training_prefix_and_quiet_guards():
    cfg = awgn_config(ebn0_db=3.0)
    info = config_summary(cfg)
    result = simulate_frame(cfg, 7)
    size, cp, blocks = cfg.fft_len, cfg.cp_len, cfg.symbols_per_frame
    assert result["tx_samples"].size == info["frame_samples"] == (blocks+1)*(size+cp)+2*cp
    assert result["total_bits"] == size*blocks*2
    np.testing.assert_array_equal(result["tx_samples"][:cp], 0)
    np.testing.assert_array_equal(result["tx_samples"][-cp:], 0)
    assert result["tx_energy"] == pytest.approx(np.vdot(result["tx_samples"], result["tx_samples"]).real)
    assert result["noise_variance"] == pytest.approx(result["tx_energy"]/(result["total_bits"]*10**0.3))
    training_time = result["tx_samples"][2*cp:2*cp+size]
    np.testing.assert_allclose(np.fft.fft(training_time, norm="ortho"), result["training_tx_symbols"], atol=1e-13)
    data_symbols = np.asarray(result["tx_symbols"]).reshape(blocks, size)
    for block in range(blocks):
        begin = cp+(block+1)*(size+cp)
        expected = chirp_prefix(explicit_idaft(data_symbols[block], cfg.c1, cfg.c2), cp, cfg.c1, cfg.prefix)
        np.testing.assert_allclose(result["tx_samples"][begin:begin+size+cp], expected, atol=3e-13)


def test_channel_attenuation_never_redefines_noise_power():
    cfg = awgn_config()
    flat = simulate_frame(cfg, 2)
    faded = simulate_frame(replace(cfg, profile="NTN-TDLA100-200"), 2)
    assert flat["noise_variance"] == faded["noise_variance"]
    np.testing.assert_array_equal(flat["noise_samples"], faded["noise_samples"])
    assert not np.isclose(np.vdot(faded["rx_clean_samples"], faded["rx_clean_samples"]).real, flat["tx_energy"])


def test_seeded_receiver_variants_reuse_bits_physical_channel_and_noise():
    cfg = awgn_config(profile="NTN-TDLA100-200", doppler_spread_hz=1200.0)
    ideal = simulate_frame(cfg, 4)
    again = simulate_frame(cfg, 4)
    training = simulate_frame(replace(cfg, estimator="training_ls"), 4)
    for key in ("tx_bits", "tx_samples", "rx_samples", "noise_samples", "path_gains", "effective_channel_true"):
        np.testing.assert_array_equal(ideal[key], again[key])
        np.testing.assert_array_equal(ideal[key], training[key])
    future = simulate_frame(cfg, 5)
    assert not np.array_equal(ideal["tx_bits"], future["tx_bits"])
    assert not np.array_equal(ideal["noise_samples"], future["noise_samples"])
    np.testing.assert_array_equal(ideal["path_gains"], future["path_gains"])
    strong = simulate_frame(replace(cfg, ebn0_db=cfg.ebn0_db+10), 4)
    np.testing.assert_allclose(ideal["noise_samples"], math.sqrt(10)*strong["noise_samples"], atol=1e-13)


def test_training_channel_estimator_does_not_consult_payload_or_true_channel(monkeypatch):
    cfg = awgn_config(profile="NTN-TDLA100-200", estimator="training_ls", ebn0_db=15.0,
                      doppler_spread_hz=600.0)
    original = simulate_frame(cfg, 2)
    modulate = afdm_ntn.qam_modulate
    monkeypatch.setattr(afdm_ntn, "qam_modulate", lambda bits, modulation: -modulate(bits, modulation))
    opposite = simulate_frame(cfg, 2)
    np.testing.assert_allclose(opposite["tx_symbols"], -original["tx_symbols"], atol=0)
    np.testing.assert_array_equal(opposite["training_rx_symbols"], original["training_rx_symbols"])
    np.testing.assert_array_equal(opposite["effective_channel_est"], original["effective_channel_est"])
    assert not np.allclose(original["effective_channel_est"], original["effective_channel_true"])


def test_static_single_integer_path_training_estimate_uses_the_correct_cpp_source_phase(monkeypatch):
    cfg = awgn_config(profile="NTN-TDLA100-200", estimator="training_ls", ebn0_db=120.0,
                      c1=0.01731, c2=0.001271)
    fs = cfg.fft_len*cfg.subcarrier_spacing_hz
    monkeypatch.setattr(afdm_ntn, "_path_realization", lambda _cfg: (
        np.array([3.0/fs]), np.array([1.0+0j]), np.array([0.0])))
    result = simulate_frame(cfg, 1)
    # A delay-three path wraps actual tail samples with their CPP phases.
    # A static, covered integer-delay channel is exactly learnable from the
    # known OFDM training; noise is the only remaining estimation error.
    assert np.max(np.abs(result["effective_channel_est"]-result["effective_channel_true"])) < 2e-5
    assert result["bit_errors"] == 0
    assert result["evm_percent"] < 0.005


@pytest.mark.parametrize("prefix", ["cpp", "cp"])
def test_full_affine_channel_matrix_matches_real_fractional_time_varying_propagation(prefix):
    cfg = awgn_config(profile="NTN-TDLA100-200", residual_cfo_hz=1234.0,
                      doppler_spread_hz=1200.0, prefix=prefix, delay_scale=7.0)
    result = simulate_frame(cfg, 2)
    size, cp = cfg.fft_len, cfg.cp_len
    assert config_summary(cfg)["max_delay_samples"] < cp
    begin = cp+(size+cp)+cp
    indices = begin + np.arange(cfg.symbols_per_frame)[:, None]*(size+cp)+np.arange(size)
    physical = afdm_demodulate(result["rx_clean_samples"][indices], cfg.c1, cfg.c2)
    symbols = np.asarray(result["tx_symbols"]).reshape(cfg.symbols_per_frame, size)
    prediction = np.einsum("bij,bj->bi", result["effective_channel_true"], symbols)
    np.testing.assert_allclose(prediction, physical, atol=2e-11, rtol=2e-11)


def test_missing_prefix_creates_cross_block_interference_not_a_fake_cyclic_channel(monkeypatch):
    cfg = awgn_config(profile="NTN-TDLA100-200", modulation="64QAM", ebn0_db=100.0,
                      delay_scale=6/(285e-9*32*30_000.0))
    fs = cfg.fft_len*cfg.subcarrier_spacing_hz
    monkeypatch.setattr(afdm_ntn, "_path_realization", lambda _cfg: (
        np.array([6.0/fs]), np.array([1.0+0j]), np.array([0.0])))
    short_errors = long_errors = 0
    short_energy = long_energy = 0.0
    for index in range(4):
        short = simulate_frame(replace(cfg, cp_len=0), index)
        long = simulate_frame(replace(cfg, cp_len=8), index)
        np.testing.assert_array_equal(short["tx_bits"], long["tx_bits"])
        short_errors += short["bit_errors"]
        long_errors += long["bit_errors"]
        short_energy += short["error_energy"]
        long_energy += long["error_energy"]
    assert long_errors == 0 and short_errors > 0
    assert short_energy > 100*long_energy


def test_orbital_cfo_uses_absolute_physical_frame_time():
    cfg = awgn_config(velocity_kmh=28080.0, radial_projection=0.5,
                      compensation_ratio=0.999, residual_cfo_hz=75.0)
    info = config_summary(cfg)
    orbit = 28080/3.6*0.5*2e9/299_792_458.0
    residual = orbit*0.001+75
    assert info["orbit_doppler_hz"] == pytest.approx(orbit)
    assert info["residual_cfo_hz"] == pytest.approx(residual)
    result = simulate_frame(cfg, 3)
    count = result["tx_samples"].size
    time = (np.arange(count)+3*count)/info["sample_rate_hz"]
    np.testing.assert_allclose(result["rx_clean_samples"], result["tx_samples"]*np.exp(2j*np.pi*residual*time), atol=2e-11)


def test_zero_ber_is_raw_zero_with_finite_uncertainty_and_actual_energy_pooled_evm():
    point = run_study(awgn_config(ebn0_db=100), "ebn0", [100], frames=2)[0]
    for stats in point["strategies"].values():
        assert stats["bit_errors"] == stats["ber"] == stats["fer"] == 0
        assert stats["total_bits"] == 2*32*3*2
        assert stats["frames"] == 2 and stats["ci_low"] == 0
        assert 0 < stats["ci_high"] < 0.02
        assert stats["evm_percent"] == pytest.approx(100*math.sqrt(stats["error_energy"]/stats["reference_energy"]))


@pytest.mark.parametrize("axis,values", [
    ("residual_cfo", [-300.0, 300.0]), ("spacing", [15000.0, 30000.0]),
    ("delay_scale", [0.5, 2.0]), ("doppler", [0.0, 1200.0]),
    ("c1", [0.0, 0.01731]), ("c2", [0.0, 0.0031]), ("blocks", [2.0, 4.0]),
])
def test_scans_pair_the_same_information_budget_for_all_receivers(axis, values):
    points = run_study(awgn_config(profile="NTN-TDLA100-200"), axis, values, frames=1)
    assert [point["axis_value"] for point in points] == values
    for point in points:
        stats = list(point["strategies"].values())
        assert len({item["total_bits"] for item in stats}) == 1
        for item in stats:
            assert item["frames"] == 1
            assert item["ber"] == item["bit_errors"]/item["total_bits"]
            assert 0 <= item["ci_low"] <= item["ber"] <= item["ci_high"] <= 1


def test_cancel_drops_the_whole_unfinished_point_after_some_complete_frames(monkeypatch):
    stop, seen, callbacks = threading.Event(), [], []
    original = afdm_ntn.simulate_frame
    def cancel_on_second(config, frame_index=0, stop_event=None):
        seen.append(frame_index)
        result = original(config, frame_index)
        if frame_index == 1:
            stop.set()
        return result
    monkeypatch.setattr(afdm_ntn, "simulate_frame", cancel_on_second)
    points = run_study(awgn_config(), "ebn0", [0,4], frames=3,
                       stop_event=stop, on_point=callbacks.append)
    assert seen == [0,0,0,1]
    assert points == callbacks == []


def test_cancel_at_point_boundary_preserves_only_complete_paired_points():
    stop, seen = threading.Event(), []
    def finish(point):
        seen.append(point)
        stop.set()
    points = run_study(awgn_config(), "ebn0", [0,4], frames=2,
                       stop_event=stop, on_point=finish)
    assert len(points) == len(seen) == 1
    assert all(stats["frames"] == 2 for stats in points[0]["strategies"].values())
    assert run_study(awgn_config(), "ebn0", [0], frames=2, stop_event=stop) == []


@pytest.mark.parametrize("changes", [
    {"fft_len":63}, {"cp_len":-1}, {"cp_len":32}, {"subcarrier_spacing_hz":0},
    {"symbols_per_frame":0}, {"symbols_per_frame":2.5}, {"symbols_per_frame":True},
    {"modulation":"unsupported"}, {"ebn0_db":float("nan")},
    {"compensation_ratio":-0.1}, {"compensation_ratio":1.1}, {"delay_scale":-1},
    {"estimator":"oracle_magic"}, {"equalizer":"unsupported"}, {"prefix":"unsupported"},
    {"doppler_spread_hz":-1}, {"residual_cfo_hz":1e6},
    {"c1":float("nan")}, {"c2":float("inf")},
])
def test_invalid_config_is_rejected_before_simulation(changes):
    with pytest.raises((ValueError, TypeError)):
        replace(awgn_config(), **changes).validated()


@pytest.mark.parametrize("axis,values", [
    ("ebn0", [0,float("nan")]), ("blocks", [4,3.5]), ("spacing", [30000,0]),
    ("ebn0", []), ("ebn0", [0,0]), ("c1", [0,float("inf")]),
])
def test_entire_scan_is_validated_before_any_result_is_published(axis, values):
    seen = []
    with pytest.raises(ValueError):
        run_study(awgn_config(), axis, values, frames=1, on_point=seen.append)
    assert seen == []


def test_immutable_config_and_invalid_frame_index_budget_or_axis():
    cfg = awgn_config()
    with pytest.raises(FrozenInstanceError):
        cfg.c1 = 0.0
    for index in (-1,1.5,True):
        with pytest.raises(ValueError):
            simulate_frame(cfg,index)
    with pytest.raises(ValueError):
        run_study(cfg,"invalid",[0],frames=1)
    with pytest.raises(ValueError):
        run_study(cfg,"ebn0",[0],frames=0)
