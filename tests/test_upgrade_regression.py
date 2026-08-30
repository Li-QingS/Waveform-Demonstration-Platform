import math
import threading

import pytest

from waveform_sim.simulation.shared_waveform_benchmark import SharedWaveformBenchmark
from waveform_sim.simulation.simple_fdidm_rx import _LegacyFDIDMTransceiver


def test_bandwidth_and_physical_frame_time_follow_grid_definition():
    tb = _LegacyFDIDMTransceiver(
        m_subcarriers=8,
        n_symbols=8,
        subcarrier_spacing_hz=300e3,
        dynamic_channel=False,
        channel_dynamics="fixed",
        demo_frame_interval_s=0.0,
    )
    summary = tb.get_channel_summary()
    assert summary["bandwidth_hz"] == pytest.approx(8 * 300e3)
    assert summary["sample_rate_hz"] == pytest.approx(8 * 300e3)
    assert summary["physical_frame_duration_s"] == pytest.approx(8 / 300e3)
    assert summary["physical_frame_duration_s"] != pytest.approx(
        summary["demo_update_period_s"]
    )


def test_index_switch_does_not_rewind_dynamic_channel_time():
    tb = _LegacyFDIDMTransceiver(
        m_subcarriers=4,
        n_symbols=4,
        subcarrier_spacing_hz=300e3,
        dynamic_channel=True,
        channel_dynamics="cont",
        residual_doppler_spread_hz=200.0,
        demo_frame_interval_s=0.0,
    )
    tb.step()
    tb.step()
    before = tb.get_channel_summary()
    epoch_before = int(tb._adaptive_index_epoch)
    paths_before = [
        (p["delay_ns"], p["doppler_hz"]) for p in before["paths"]
    ]

    tb.set_indices(0.5, 1.0)
    after_switch = tb.get_channel_summary()
    assert after_switch["frame"] == before["frame"]
    assert after_switch["physical_time_s"] == pytest.approx(before["physical_time_s"])
    assert int(tb._adaptive_index_epoch) == epoch_before + 1
    assert [
        (p["delay_ns"], p["doppler_hz"]) for p in after_switch["paths"]
    ] == pytest.approx(paths_before)

    tb.step()
    after_next_frame = tb.get_channel_summary()
    assert after_next_frame["frame"] == before["frame"] + 1
    assert after_next_frame["physical_time_s"] == pytest.approx(
        before["physical_time_s"] + before["physical_frame_duration_s"]
    )


def test_concurrent_optimizer_jobs_append_only_one_real_switch(monkeypatch):
    tb = _LegacyFDIDMTransceiver(
        m_subcarriers=4,
        n_symbols=4,
        snr_db=8.0,
        dynamic_channel=False,
        channel_dynamics="fixed",
        demo_frame_interval_s=0.0,
    )
    tb.step()
    with tb._adaptive_lock:
        tb.adaptive_enabled = True
        tb.adaptive_auto_apply = True
        tb.adaptive_stability_evals = 1
        tb.adaptive_cooldown_frames = 0
        tb.adaptive_min_improvement_db = 0.0
        tb.adaptive_fine_step = 0.5
        tb._adaptive_state = "collecting"
    with tb._lock:
        snapshot = tb._adaptive_snapshot_locked()
    snapshot["source_window_frames"] = 1

    barrier = threading.Barrier(2)

    def fake_optimize(
        cls,
        benchmarks,
        snr_db,
        current_alpha=0.0,
        current_beta=0.0,
        coarse_step=0.5,
        fine_step=0.1,
        decision_ser_floor=1e-8,
        stop_event=None,
    ):
        barrier.wait(timeout=5.0)
        return {
            "recommended_alpha": 1.0,
            "recommended_beta": 1.0,
            "predicted_ser_current": 0.2,
            "predicted_ser_best": 0.1,
            "predicted_geometric_ser_current": 0.2,
            "predicted_geometric_ser_best": 0.1,
            "decision_ser_floor": float(decision_ser_floor),
            "decision_geometric_ser_current": 0.2,
            "decision_geometric_ser_best": 0.1,
            "predicted_improvement_db": 3.0102999566,
            "candidate_count": 9,
            "search_seconds": 0.01,
            "top_candidates": [],
            "ensemble_size": len(benchmarks),
            "working_snr_db": float(snr_db),
            "search_mode": "test",
        }

    monkeypatch.setattr(
        SharedWaveformBenchmark,
        "optimize_fdidm_over_ensemble",
        classmethod(fake_optimize),
    )

    errors = []

    def run_job():
        try:
            tb._adaptive_process_ensemble([dict(snapshot)])
        except Exception as exc:  # pragma: no cover - assertion below surfaces it
            errors.append(exc)

    threads = [threading.Thread(target=run_job) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    switches = [
        item for item in tb.get_adaptive_history() if item.get("kind") == "switch"
    ]
    assert len(switches) == 1
    assert (float(tb.config.alpha), float(tb.config.beta)) == (1.0, 1.0)
    assert int(tb._adaptive_index_epoch) == 1
    assert switches[0]["from_alpha"] == 0.0
    assert switches[0]["from_beta"] == 0.0
    assert switches[0]["to_alpha"] == 1.0
    assert switches[0]["to_beta"] == 1.0


def test_runtime_aliases_and_profile_only_update_are_accepted():
    tb = _LegacyFDIDMTransceiver(
        ntn_profile="NTN-TDLA100-200",
        residual_doppler_spread_hz=200.0,
        demo_frame_interval_s=0.0,
    )
    tb.update_runtime_parameters(
        snr_db=12.5,
        channel_model="NTN-TDLC5-1200",
        doppler_spread_hz=1200.0,
        detector="MMSE",
        unused_cross_waveform_field="ignored",
    )
    assert tb.config.ebn0_db == pytest.approx(12.5)
    assert tb.config.ntn_profile == "NTN-TDLC5-1200"
    assert tb.config.channel_model == "NTN-TDLC5-1200"
    assert tb.config.residual_doppler_spread_hz == pytest.approx(1200.0)
    assert tb.config.decoder == "MMSE"
