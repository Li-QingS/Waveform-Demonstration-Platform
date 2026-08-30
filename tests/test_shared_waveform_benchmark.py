import numpy as np

from waveform_sim.simulation.shared_waveform_benchmark import SharedWaveformBenchmark
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
    metrics = bench.evaluate_all(10.0, 0.4, 0.9)
    assert set(metrics) == {"OFDM", "OTFS", "AFDM", "FDIDM"}
    for item in metrics.values():
        assert 0.0 <= float(item["ser"]) <= 1.0
        assert np.isfinite(float(item["evm_percent"]))
        assert int(item["receiver_budget"]) >= 1

    # The implementation intentionally does not hard-code an always-best curve.
    rounded = {round(float(item["ser"]), 12) for item in metrics.values()}
    assert len(rounded) >= 2


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
    assert result["decision_geometric_ser_best"] <= result["decision_geometric_ser_current"] + 1e-15
    assert result["predicted_improvement_db"] >= -1e-9
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
