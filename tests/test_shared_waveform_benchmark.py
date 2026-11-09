from dataclasses import replace

import numpy as np
import pytest

from waveform_sim.simulation.shared_waveform_benchmark import (
    BenchmarkContext,
    SharedWaveformBenchmark,
)
from waveform_sim.simulation.simple_fdidm_rx import _LegacyFDIDMTransceiver


def make_benchmark():
    tb = _LegacyFDIDMTransceiver(
        m_subcarriers=4,
        n_symbols=4,
        subcarrier_spacing_hz=300e3,
        channel_dynamics="fixed",
        dynamic_channel=False,
        channel_seed=42,
        decoder="MMSE",
        demo_frame_interval_s=0.0,
    )
    tb.step()
    return SharedWaveformBenchmark.from_backend(tb)


def test_four_waveforms_use_different_domain_channels_on_same_realization():
    bench = make_benchmark()
    matrices = {
        name: bench.equivalent_channel(name, 0.4, 0.9)
        for name in ("OFDM", "OTFS", "AFDM", "FDIDM")
    }
    for matrix in matrices.values():
        assert matrix.shape == (16, 16)
        assert np.all(np.isfinite(matrix))

    # They share one underlying physical channel but are not aliases for three
    # fixed FDIDM alpha/beta points.
    assert not np.allclose(matrices["OFDM"], matrices["OTFS"])
    assert not np.allclose(matrices["OFDM"], matrices["AFDM"])
    assert not np.allclose(matrices["AFDM"], matrices["FDIDM"])


def test_four_metrics_are_finite_without_forced_ordering():
    bench = make_benchmark()
    bench.context = replace(bench.context, maximum_delay_ns=1000.0)
    metrics = bench.evaluate_all(10.0, 0.4, 0.9)
    assert set(metrics) == {"OFDM", "OTFS", "AFDM", "FDIDM"}
    for item in metrics.values():
        assert 0.0 <= float(item["ser"]) <= 1.0
        assert np.isfinite(float(item["evm_percent"]))
        assert int(item["receiver_budget"]) >= 1
        assert item["guard_samples_synthesized"] is False
        assert item["receiver_policy"] == SharedWaveformBenchmark.RECEIVER_POLICY
        expected_support = (
            "fixed_diagonal_one_tap_model"
            if item["waveform"] == "OFDM"
            else "true_channel_oracle_strongest_per_row_with_diagonal_retention"
        )
        assert item["support_selection_source"] == expected_support
        assert item["ser_prediction_model"] == "gain_normalized_gaussian_residual"
        assert item["residual_covariance_source"] == "true_channel_oracle"
        assert item["guard_interval_budget_model"] == "excluded_from_uncoded_ser"
        assert item["guard_overhead_included"] is False

    expected_noise = bench.noise_variance(10.0)
    assert all(
        np.isclose(float(item["noise_variance"]), expected_noise)
        for item in metrics.values()
    )

    # The implementation intentionally does not hard-code an always-best curve.
    rounded = {round(float(item["ser"]), 12) for item in metrics.values()}
    assert len(rounded) >= 2


def test_backend_factory_keeps_snapshot_dimensions_after_backend_resize():
    tb = _LegacyFDIDMTransceiver(
        m_subcarriers=4,
        n_symbols=4,
        subcarrier_spacing_hz=300e3,
        channel_dynamics="fixed",
        dynamic_channel=False,
        channel_seed=42,
        decoder="MMSE",
        demo_frame_interval_s=0.0,
    )
    tb.step()
    bench = SharedWaveformBenchmark.from_backend(tb)

    # A live UI can change the backend frame shape while a theory search is
    # still running.  The benchmark must continue using its copied 4x4 frame.
    tb.set_frame_shape(8, 8)
    item = bench.evaluate("FDIDM", 10.0, alpha=0.4, beta=0.9)

    assert item["waveform"] == "FDIDM"
    assert np.isfinite(float(item["ser"]))


def test_backend_factory_keeps_qam_snapshot_when_live_config_changes(monkeypatch):
    tb = _LegacyFDIDMTransceiver(
        m_subcarriers=4,
        n_symbols=4,
        mod_order="16QAM",
        channel_dynamics="fixed",
        dynamic_channel=False,
        channel_seed=42,
        demo_frame_interval_s=0.0,
    )
    tb.step()
    original_summary = tb.get_channel_summary

    def change_live_modulation_after_snapshot():
        summary = original_summary()
        tb.update_runtime_parameters(mod_order="QPSK")
        return summary

    monkeypatch.setattr(tb, "get_channel_summary", change_live_modulation_after_snapshot)

    bench = SharedWaveformBenchmark.from_backend(tb)

    assert bench.context.qam_order == 16
    assert bench.context.bits_per_symbol == 4


def test_backend_context_preserves_total_residual_doppler_for_afdm_design():
    tb = _LegacyFDIDMTransceiver(
        m_subcarriers=4,
        n_symbols=4,
        subcarrier_spacing_hz=300e3,
        channel_dynamics="fixed",
        dynamic_channel=False,
        channel_seed=42,
        decoder="MMSE",
        demo_frame_interval_s=0.0,
    )
    tb.step()
    summary = tb.get_channel_summary()
    bench = SharedWaveformBenchmark.from_backend(tb)

    assert bench.context.residual_doppler_spread_hz == pytest.approx(
        summary["residual_doppler_spread_hz"]
    )
    assert bench.context.max_residual_doppler_hz == pytest.approx(
        summary["max_doppler_hz"]
    )
    assert bench.evaluate("AFDM", 10.0)["max_residual_doppler_hz"] == pytest.approx(
        summary["max_doppler_hz"]
    )


def test_benchmark_reports_its_fixed_receiver_policy_independent_of_decoder_selection():
    tb = _LegacyFDIDMTransceiver(
        m_subcarriers=4,
        n_symbols=4,
        subcarrier_spacing_hz=300e3,
        channel_dynamics="fixed",
        dynamic_channel=False,
        channel_seed=42,
        decoder="ZF",
        demo_frame_interval_s=0.0,
    )
    tb.step()
    bench = SharedWaveformBenchmark.from_backend(tb)

    assert bench.context.receiver_policy == SharedWaveformBenchmark.RECEIVER_POLICY
    assert "ZF" not in bench.context.receiver_policy
    assert bench.evaluate("OFDM", 10.0)["receiver_policy"] == SharedWaveformBenchmark.RECEIVER_POLICY


def test_fdidm_search_never_returns_worse_decision_score():
    bench = make_benchmark()
    result = SharedWaveformBenchmark.optimize_fdidm_over_ensemble(
        [bench],
        snr_db=10.0,
        current_alpha=0.0,
        current_beta=0.0,
        coarse_step=1.0,
        fine_step=0.5,
        decision_ser_floor=1e-12,
    )
    assert result["decision_ser_best"] <= result["decision_ser_current"] + 1e-15
    assert result["predicted_improvement_db"] >= -1e-9
    assert result["receiver_policy"] == SharedWaveformBenchmark.RECEIVER_POLICY
    assert result["search_mode"] == "shared_channel_coarse_fine_waveform_specific_lmmse"
    assert 0.0 <= result["recommended_alpha"] <= 2.0
    assert 0.0 <= result["recommended_beta"] <= 2.0


def test_high_snr_practical_floor_prevents_fake_switches():
    bench = make_benchmark()
    result = SharedWaveformBenchmark.optimize_fdidm_over_ensemble(
        [bench],
        snr_db=40.0,
        current_alpha=0.0,
        current_beta=0.0,
        coarse_step=1.0,
        fine_step=0.5,
        decision_ser_floor=1e-8,
    )
    assert result["recommended_alpha"] == 0.0
    assert result["recommended_beta"] == 0.0
    assert abs(result["predicted_improvement_db"]) < 1e-9
def make_benchmark_v5(m=4, n=4, seed=42):
    tb = _LegacyFDIDMTransceiver(
        m_subcarriers=m,
        n_symbols=n,
        subcarrier_spacing_hz=300e3,
        channel_dynamics="fixed",
        dynamic_channel=False,
        channel_seed=seed,
        decoder="MMSE",
        demo_frame_interval_s=0.0,
    )
    tb.step()
    return SharedWaveformBenchmark.from_backend(tb)


def test_interference_aware_receiver_is_not_worse_at_high_snr():
    bench = make_benchmark_v5()
    for name in ("OFDM", "OTFS", "AFDM", "FDIDM"):
        kwargs = {"alpha": 0.4, "beta": 0.9} if name == "FDIDM" else {}
        low = float(bench.evaluate(name, 0.0, **kwargs)["ser"])
        high = float(bench.evaluate(name, 20.0, **kwargs)["ser"])
        assert high <= low + 1e-12, (name, low, high)


def test_receiver_reports_explicit_model_mismatch_energy():
    bench = make_benchmark_v5()
    for name in ("OFDM", "OTFS", "AFDM", "FDIDM"):
        item = bench.evaluate(name, 10.0, alpha=0.4, beta=0.9)
        assert 0.0 <= float(item["retained_energy_ratio"]) <= 1.0 + 1e-12
        assert 0.0 <= float(item["omitted_energy_ratio"]) <= 1.0 + 1e-12
        assert np.isfinite(float(item["model_interference_power"]))
        assert abs(
            float(item["retained_energy_ratio"])
            + float(item["omitted_energy_ratio"])
            - 1.0
        ) < 1e-9


def test_sparse_receiver_keeps_desired_diagonal_when_interferer_is_stronger():
    channel = np.eye(4, dtype=np.complex128)
    channel[0, 1:] = 2.0
    model = SharedWaveformBenchmark._retain_strongest_per_row(channel, 3)

    assert model[0, 0] == channel[0, 0]
    assert np.count_nonzero(model[0]) == 3
    assert np.count_nonzero(model[0, 1:]) == 2


def test_sparse_receiver_does_not_spend_support_on_zero_diagonal():
    channel = np.zeros((4, 4), dtype=np.complex128)
    channel[0, 1:] = (5.0, 4.0, 3.0)
    model = SharedWaveformBenchmark._retain_strongest_per_row(channel, 3)

    assert model[0, 0] == 0.0
    assert np.count_nonzero(model[0]) == 3
    np.testing.assert_array_equal(model[0, 1:], channel[0, 1:])


def test_ofdm_one_tap_receiver_keeps_diagonal_when_ici_is_stronger(monkeypatch):
    bench = make_benchmark_v5()
    channel = np.eye(bench.K, dtype=np.complex128)
    channel[0, 2] = 2.0
    channel[1, 2] = 1.0
    monkeypatch.setattr(
        bench,
        "equivalent_channel",
        lambda _waveform, alpha=0.0, beta=0.0: channel,
    )

    captured = {}
    original = SharedWaveformBenchmark._interference_aware_lmmse

    def capture_model(model, true, noise_var, **kwargs):
        captured["model"] = model.copy()
        captured["receiver"] = original(model, true, noise_var, **kwargs)
        return captured["receiver"]

    monkeypatch.setattr(bench, "_interference_aware_lmmse", capture_model)
    bench.evaluate("OFDM", 10.0)

    np.testing.assert_array_equal(captured["model"], np.eye(bench.K))
    np.testing.assert_allclose(
        captured["receiver"] - np.diag(np.diag(captured["receiver"])),
        0.0,
        atol=1e-12,
    )


def test_mmse_ser_removes_equalizer_gain_before_qam_prediction():
    identity = np.eye(1, dtype=np.complex128)
    bench = SharedWaveformBenchmark(
        BenchmarkContext(
            m_subcarriers=1,
            n_symbols=1,
            subcarrier_spacing_hz=1.0,
            qam_order=4,
            bits_per_symbol=2,
        ),
        identity,
        lambda _alpha, _beta: (identity, identity),
    )

    predicted_ser = bench.evaluate("OFDM", 0.0)["ser"]

    # Unit-energy QPSK at Eb/N0=0 dB: 2Q(sqrt(2)) - Q(sqrt(2))^2.
    assert np.isclose(float(predicted_ser), 0.151113446915621, atol=1e-12)


def test_fdidm_search_does_not_trade_large_frame_regression_for_geometric_gain():
    class FrameBenchmark:
        def __init__(self, frame_ser):
            self.frame_ser = frame_ser

        def evaluate(self, waveform, snr_db, alpha=0.0, beta=0.0):
            assert waveform == "FDIDM"
            if alpha == 0.0 and beta == 0.0:
                ser = 0.1
            else:
                ser = self.frame_ser
            return {"ser": ser}

    result = SharedWaveformBenchmark.optimize_fdidm_over_ensemble(
        [FrameBenchmark(0.001), FrameBenchmark(0.9)],
        snr_db=10.0,
        current_alpha=0.0,
        current_beta=0.0,
        coarse_step=1.0,
        fine_step=0.5,
        decision_ser_floor=1e-12,
    )

    assert (result["recommended_alpha"], result["recommended_beta"]) == (0.0, 0.0)
    assert result["predicted_ser_best"] == result["predicted_ser_current"] == 0.1
    assert result["predicted_improvement_db"] == 0.0


def test_cancelled_fdidm_search_is_marked_cancelled_instead_of_published():
    import threading

    stop = threading.Event()

    class CancellingBenchmark:
        def evaluate(self, waveform, snr_db, alpha=0.0, beta=0.0):
            assert waveform == "FDIDM"
            stop.set()
            return {"ser": 0.25}

    result = SharedWaveformBenchmark.optimize_fdidm_over_ensemble(
        [CancellingBenchmark()],
        snr_db=10.0,
        current_alpha=0.0,
        current_beta=0.0,
        coarse_step=1.0,
        fine_step=0.5,
        stop_event=stop,
    )

    assert result["cancelled"] is True
    assert "recommended_alpha" not in result


def test_independent_snr_optimization_returns_valid_fdidm_points():
    bench = make_benchmark_v5(m=8, n=8)
    last = (0.0, 0.0)
    results = []
    for snr in (0.0, 10.0, 20.0, 30.0):
        result = SharedWaveformBenchmark.optimize_fdidm_over_ensemble(
            [bench],
            snr_db=snr,
            current_alpha=last[0],
            current_beta=last[1],
            coarse_step=0.5,
            fine_step=0.1,
            decision_ser_floor=1e-12,
        )
        a = float(result["recommended_alpha"])
        b = float(result["recommended_beta"])
        ser = float(result["predicted_ser_best"])
        assert 0.0 <= a <= 2.0
        assert 0.0 <= b <= 2.0
        assert 0.0 <= ser <= 1.0
        results.append((snr, a, b, ser))
        last = (a, b)
    assert all(results[i + 1][3] <= results[i][3] + 1e-12 for i in range(len(results) - 1))
