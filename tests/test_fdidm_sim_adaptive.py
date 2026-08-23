# -*- coding: utf-8 -*-
"""FDIDM simulation-side predictive adaptive tests."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from waveform_sim.simulation.fdidm_adaptive import FDIDMSimAdaptiveMixin
from waveform_sim.simulation.simple_fdidm_rx import _LegacyFDIDMTransceiver


def _make_backend(**kwargs):
    params = dict(m_subcarriers=4, n_symbols=4, snr_db=12.0, dynamic_channel=False)
    params.update(kwargs)
    return _LegacyFDIDMTransceiver(**params)


def _snapshot(backend, alpha=0.0, beta=0.0, htf_kind="full", coarse=0.5):
    with backend._lock:
        htf = np.asarray(backend._H_tf, dtype=np.complex128).copy()
        noise_var = backend._noise_variance()
    if htf_kind == "diag":
        M = int(backend.config.m_subcarriers)
        N = int(backend.config.n_symbols)
        rng = np.random.default_rng(7)
        htf = (rng.standard_normal(M * N) + 1j * rng.standard_normal(M * N)).reshape(M, N, order="F")
    return {
        "M": int(backend.config.m_subcarriers),
        "N": int(backend.config.n_symbols),
        "htf": htf,
        "htf_kind": htf_kind,
        "noise_var": float(noise_var),
        "equalizer": str(backend.config.decoder),
        "mod_order": str(backend.config.mod_order),
        "alpha": float(alpha),
        "beta": float(beta),
        "coarse_step": float(coarse),
        "fine_step": 0.05,
        "integer_margin_db": 0.0,
        "max_order": 512,
        "rcond": 1e-6,
        "frame_counter": int(backend._metrics.total_frames),
        "snapshot_seq": 1,
    }


def _run_frames(backend, frames, sleep=0.015):
    for _ in range(int(frames)):
        backend.step()
        time.sleep(sleep)


# ---------------------------------------------------------------- kernel units
def test_kernel_qfunc_and_ser():
    obj = object.__new__(FDIDMSimAdaptiveMixin)
    q0 = float(FDIDMSimAdaptiveMixin._adaptive_qfunc(np.array([0.0]))[0])
    assert abs(q0 - 0.5) < 1e-9
    ser = obj._adaptive_ser_from_symbol_nsr(np.array([100.0]), "QPSK")
    assert 0.0 <= float(ser) <= 1.0
    assert FDIDMSimAdaptiveMixin._adaptive_qam_order("64QAM") == 64
    grid = FDIDMSimAdaptiveMixin._adaptive_grid_values(0.25)
    assert 0.0 in grid and 1.0 in grid and 2.0 in grid
    assert abs(FDIDMSimAdaptiveMixin._adaptive_canonical_index(1.7) - 1.7) < 1e-9


def test_kernel_optimize_snapshot_fields_full():
    tb = _make_backend()
    tb.step()
    res = tb._optimize_alpha_beta_snapshot(_snapshot(tb))
    assert 0.0 <= float(res["recommended_alpha"]) <= 2.0
    assert 0.0 <= float(res["recommended_beta"]) <= 2.0
    assert np.isfinite(float(res["predicted_ser_current"]))
    assert np.isfinite(float(res["predicted_ser_best"]))
    assert float(res["candidate_count"]) >= 1
    assert float(res["search_seconds"]) >= 0.0
    assert res["search_mode"] == "full_coordinate"


def test_kernel_optimize_snapshot_fields_diag():
    tb = _make_backend()
    tb.step()
    res = tb._optimize_alpha_beta_snapshot(_snapshot(tb, htf_kind="diag", coarse=0.25))
    assert 0.0 <= float(res["recommended_alpha"]) <= 2.0
    assert 0.0 <= float(res["recommended_beta"]) <= 2.0
    assert res["search_mode"] == "diag_2d"
    assert float(res["candidate_count"]) >= 1


def test_kernel_max_order_guard():
    tb = _make_backend(m_subcarriers=16, n_symbols=16)
    tb.step()
    snap = _snapshot(tb)
    snap["max_order"] = 128  # 16*16=256 > 128
    with pytest.raises(ValueError):
        tb._optimize_alpha_beta_snapshot(snap)


# ---------------------------------------------------------------- closed loop
def test_closed_loop_dynamic_channel_apply_and_cooldown():
    tb = _make_backend(dynamic_channel=True, channel_dynamics="block",
                       channel_coherence_frames=8)
    tb.start_adaptive_tuning(interval_frames=1, stability_evals=2,
                             min_improvement_db=0.0, auto_apply=True,
                             cooldown_frames=5, max_order=512)
    _run_frames(tb, 60)
    history = tb.get_adaptive_history()
    evals = [h for h in history if h["kind"] == "eval"]
    switches = [h for h in history if h["kind"] == "switch"]
    assert len(evals) >= 2
    assert len(switches) >= 1, "dynamic channel should eventually apply"
    frames = [int(s["frame"]) for s in switches]
    for a, b in zip(frames, frames[1:]):
        assert b - a >= 5, "applies must respect cooldown frames"
    applied = [h for h in evals if h["action"] == "apply"]
    assert all(int(h["stable_count"]) >= 2 for h in applied)
    tb.stop_adaptive_tuning()
    tb.stop()


def test_closed_loop_auto_apply_disabled():
    tb = _make_backend(dynamic_channel=True, channel_dynamics="block",
                       channel_coherence_frames=8)
    initial = (float(tb.config.alpha), float(tb.config.beta))
    tb.start_adaptive_tuning(interval_frames=1, stability_evals=1,
                             min_improvement_db=0.0, auto_apply=False,
                             cooldown_frames=0)
    _run_frames(tb, 25)
    status = tb.get_adaptive_status()
    assert status["recommendation_seq"] >= 1
    assert float(tb.config.alpha) == initial[0]
    assert float(tb.config.beta) == initial[1]
    tb.stop_adaptive_tuning()
    tb.stop()


def test_slow_search_does_not_starve_evals(monkeypatch):
    """回归：搜索慢于帧间隔时，结果不得被当作过期全部丢弃（CI 慢 runner 抖动根因）。"""
    from waveform_sim.simulation import fdidm_adaptive as fdidm_adaptive_mod
    orig_optimize = fdidm_adaptive_mod.FDIDMSimAdaptiveMixin._optimize_alpha_beta_snapshot

    def slow_optimize(self, snapshot):
        time.sleep(0.025)
        return orig_optimize(self, snapshot)

    monkeypatch.setattr(fdidm_adaptive_mod.FDIDMSimAdaptiveMixin,
                        "_optimize_alpha_beta_snapshot", slow_optimize)
    tb = _make_backend(dynamic_channel=True, channel_dynamics="block",
                       channel_coherence_frames=8)
    tb.start_adaptive_tuning(interval_frames=1, stability_evals=2,
                             min_improvement_db=0.0, auto_apply=True,
                             cooldown_frames=5, max_order=512)
    _run_frames(tb, 60)
    history = tb.get_adaptive_history()
    evals = [h for h in history if h["kind"] == "eval"]
    switches = [h for h in history if h["kind"] == "switch"]
    tb.stop_adaptive_tuning()
    tb.stop()
    assert len(evals) >= 2, "slow search must not starve evaluation records"
    assert len(switches) >= 1, "slow search must still allow a stable apply"


def test_closed_loop_context_invalidation():
    tb = _make_backend(dynamic_channel=True, channel_dynamics="block",
                       channel_coherence_frames=8)
    tb.start_adaptive_tuning(interval_frames=1, stability_evals=2,
                             min_improvement_db=0.0, auto_apply=True,
                             cooldown_frames=0)
    _run_frames(tb, 20)
    ctx_before = tb.get_adaptive_status()["context_key"]
    tb.update_runtime_parameters(mod_order="QPSK")
    _run_frames(tb, 10)
    status = tb.get_adaptive_status()
    assert status["context_key"] != ctx_before
    assert status["stable_count"] <= 2
    tb.stop_adaptive_tuning()
    tb.stop()


def test_history_capacity_limit():
    tb = _make_backend(dynamic_channel=True, channel_dynamics="block",
                       channel_coherence_frames=8)
    tb.start_adaptive_tuning(interval_frames=1, stability_evals=1,
                             min_improvement_db=0.0, auto_apply=False,
                             cooldown_frames=0, history_limit=5)
    _run_frames(tb, 40)
    history = tb.get_adaptive_history()
    assert 0 < len(history) <= 5
    assert len(tb.get_adaptive_history(limit=3)) <= 3
    tb.stop_adaptive_tuning()
    tb.stop()


def test_concurrent_start_stop_repeatable():
    tb = _make_backend()
    tb.start()
    try:
        for _ in range(3):
            tb.start_adaptive_tuning(interval_frames=1, min_improvement_db=0.0)
            _run_frames(tb, 5)
            tb.stop_adaptive_tuning()
            st = tb.get_adaptive_status()
            assert st["enabled"] is False
            assert st["state"] == "disabled"
    finally:
        tb.stop_adaptive_tuning()
        tb.stop()
        tb.wait(timeout=2.0)


# ---------------------------------------------------------------- engine
def test_engine_passthrough():
    from waveform_sim.core.engine import LinkSimulator

    sim = LinkSimulator(waveform="FDIDM", m_subcarriers=4, n_symbols=4, snr_db=12.0)
    assert sim.get_adaptive_history() == []
    st = sim.get_adaptive_status()
    assert st["enabled"] is False
    sim.step()
    sim.start_adaptive_tuning(interval_frames=1, stability_evals=1,
                              min_improvement_db=0.0, auto_apply=False)
    _run_frames(sim, 8)
    assert sim.get_adaptive_status()["recommendation_seq"] >= 1
    assert len(sim.get_adaptive_history()) >= 1
    sim.stop_adaptive_tuning()
    sim.stop()


# ---------------------------------------------------------------- UI smoke
def test_fdidm_tab_adaptive_panel_smoke():
    pytest.importorskip("PyQt5")
    pytest.importorskip("pyqtgraph")
    from PyQt5.QtWidgets import QApplication
    from waveform_sim.ui.fdidm_tab import FDIDMTab

    app = QApplication.instance() or QApplication([])
    tab = FDIDMTab()
    tab._search_worker = lambda *args, **kwargs: None
    try:
        assert tab.plot_tabs.count() == 2
        assert tab.plot_tabs.tabText(0) == "仿真图"
        assert tab.plot_tabs.tabText(1) == "自适应过程"
        assert tab.adaptive_controls is not None
        assert tab.adaptive_plots is not None
        tab._on_start_clicked()
        tab.adaptive_controls.enable_check.setChecked(True)
        tab._on_adaptive_config_changed()
        for _ in range(5):
            tab._refresh_adaptive_panel()
            time.sleep(0.05)
        status = tab.tb.get_adaptive_status()
        assert status["enabled"] is True
    finally:
        tab._on_stop_clicked()
        tab._on_qt_destroyed()
