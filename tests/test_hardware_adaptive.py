"""FDIDM 自适应纯函数测试（阶段 6c）。"""
import numpy as np

from waveform_sim.hardware.fdidm_adaptive import FDIDMAdaptiveMixin


def test_adaptive_qam_order():
    assert FDIDMAdaptiveMixin._adaptive_qam_order("QPSK") == 4
    assert FDIDMAdaptiveMixin._adaptive_qam_order("16QAM") == 16
    assert FDIDMAdaptiveMixin._adaptive_qam_order("64QAM") == 64
    assert FDIDMAdaptiveMixin._adaptive_qam_order("UNKNOWN") == 4


def test_adaptive_qfunc():
    q0 = float(FDIDMAdaptiveMixin._adaptive_qfunc(np.array([0.0]))[0])
    assert abs(q0 - 0.5) < 1e-9
    q = float(FDIDMAdaptiveMixin._adaptive_qfunc(np.array([1.96]))[0])
    assert 0.02 < q < 0.03


def test_finite_float_or_nan():
    assert FDIDMAdaptiveMixin._finite_float_or_nan(3.0) == 3.0
    assert np.isnan(FDIDMAdaptiveMixin._finite_float_or_nan(float("nan")))
    assert np.isnan(FDIDMAdaptiveMixin._finite_float_or_nan(float("inf")))


def test_adaptive_ser_from_symbol_nsr():
    obj = object.__new__(FDIDMAdaptiveMixin)
    ser = obj._adaptive_ser_from_symbol_nsr(np.array([100.0]), "QPSK")
    assert 0.0 <= float(ser) <= 1.0


def test_context_invalidation_resets_run_frame_gates_and_snapshot():
    """A new run must not wait for the previous run's frame number."""
    import threading

    obj = object.__new__(FDIDMAdaptiveMixin)
    obj._adaptive_ab_lock = threading.RLock()
    obj.adaptive_alpha_beta_enable = True
    obj._frames_processed = 0
    obj._adaptive_ab_snapshot_seq = 7
    obj._adaptive_ab_snapshot = {"snapshot_seq": 7}
    obj._adaptive_ab_last_snapshot = {"snapshot_seq": 7}
    obj._adaptive_ab_recommendation = {"recommended_alpha": 1.0}
    obj._adaptive_ab_last_queued_frame = 350
    obj._adaptive_ab_last_applied_frame = 350
    obj._adaptive_ab_last_htf_identity = ("diag_tf", 123)
    obj._adaptive_ab_stable_key = (20, 20)
    obj._adaptive_ab_stable_count = 4
    obj._adaptive_ab_last_error = "old error"
    messages = []
    obj._debug = lambda level, message: messages.append((level, message))

    obj._invalidate_alpha_beta_adaptation(reason="start_run_2", cooldown=False)

    assert obj._adaptive_ab_last_queued_frame == -10**18
    assert obj._adaptive_ab_last_applied_frame == -10**18
    assert obj._adaptive_ab_snapshot is None
    assert obj._adaptive_ab_last_snapshot is None
    assert obj._adaptive_ab_recommendation == {}
    assert obj._adaptive_ab_state == "waiting_channel"
    assert any("start_run_2" in message for _, message in messages)
