import time

import numpy as np

from waveform_sim.core.config import AdaptiveConfig
from waveform_sim.simulation.simple_fdidm_rx import FDIDMTransceiver


def test_dual_timescale_history_separates_metric_eval_and_switch():
    tb = FDIDMTransceiver(
        m_subcarriers=4,
        n_symbols=4,
        snr_db=10.0,
        decoder="MMSE",
        dynamic_channel=True,
        channel_dynamics="cont",
        residual_doppler_spread_hz=200.0,
        demo_frame_interval_s=0.0,
    )
    cfg = AdaptiveConfig(
        interval_frames=2,
        window_frames=4,
        ensemble_snapshots=2,
        benchmark_interval_frames=1,
        display_interval_s=0.05,
        display_ema_alpha=0.5,
        coarse_step=1.0,
        fine_step=0.5,
        stability_evals=2,
        cooldown_frames=4,
        min_improvement_db=100.0,  # deliberately prevent a switch
        apply_best=True,
    )
    tb.start_adaptive_tuning(config=cfg)
    try:
        for _ in range(8):
            tb.step()
            time.sleep(0.06)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            kinds = [item.get("kind") for item in tb.get_adaptive_history()]
            if "metric" in kinds and "eval" in kinds:
                break
            time.sleep(0.05)
        history = tb.get_adaptive_history()
        kinds = [item.get("kind") for item in history]
        assert "metric" in kinds
        assert "eval" in kinds
        assert "switch" not in kinds

        metric = next(item for item in history if item.get("kind") == "metric")
        for key in ("ser_ofdm", "ser_otfs", "ser_afdm", "ser_fdidm"):
            assert np.isfinite(float(metric[key]))
            assert float(metric[key]) > 0.0

        status = tb.get_adaptive_status()
        assert status["window_fill"] == 4
        assert status["window_frames"] == 4
        assert status["ensemble_snapshots"] == 2
        assert status["display_interval_s"] == 0.05
        assert status["decision_ser_floor"] == 1e-8
    finally:
        tb.stop_adaptive_tuning()


def test_manual_evaluation_uses_accumulated_window_not_single_frame():
    tb = FDIDMTransceiver(
        m_subcarriers=4,
        n_symbols=4,
        snr_db=8.0,
        dynamic_channel=True,
        channel_dynamics="cont",
        demo_frame_interval_s=0.0,
    )
    tb.start_adaptive_tuning(
        adaptive_interval_frames=1000,
        adaptive_window_frames=6,
        adaptive_ensemble_snapshots=3,
        adaptive_benchmark_interval_frames=2,
        adaptive_display_interval_s=0.05,
        adaptive_coarse_step=1.0,
        adaptive_fine_step=0.5,
        adaptive_auto_apply=False,
    )
    try:
        for _ in range(6):
            tb.step()
        assert tb.request_adaptive_evaluation() is True
        deadline = time.time() + 3.0
        eval_event = None
        while time.time() < deadline:
            evals = [x for x in tb.get_adaptive_history() if x.get("kind") == "eval"]
            if evals:
                eval_event = evals[-1]
                break
            time.sleep(0.05)
        assert eval_event is not None
        assert int(eval_event["window_frames"]) == 3  # representative ensemble size
        assert int(eval_event["candidates"]) >= 9
    finally:
        tb.stop_adaptive_tuning()
