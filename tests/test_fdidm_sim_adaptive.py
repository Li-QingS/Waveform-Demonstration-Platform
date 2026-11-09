# -*- coding: utf-8 -*-
"""FDIDM simulation-side predictive adaptive tests."""
import os
import threading
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


def _wait_for(predicate, timeout=3.0):
    """The optimizer runs off the frame thread; wait for its bounded result."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


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


def test_stability_vote_tolerates_local_drift_but_not_axis_flip_or_large_jump():
    same = FDIDMSimAdaptiveMixin._adaptive_same_search_region
    assert same((1.4, 0.0), (1.1, 0.0), (0.0, 0.0), 0.5)
    assert not same((1.4, 0.0), (0.7, 0.0), (0.0, 0.0), 0.5)
    assert not same((0.2, 0.0), (0.0, 0.2), (0.0, 0.0), 0.5)
    assert not same(None, (1.0, 0.0), (0.0, 0.0), 0.5)


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


def test_live_snapshot_benchmark_path_enforces_max_order_guard():
    tb = _make_backend(m_subcarriers=16, n_symbols=16)
    tb.update_adaptive_config(max_order=128)
    with tb._lock:
        snapshot = tb._adaptive_snapshot_locked()

    with pytest.raises(ValueError, match="exceeds max_order=128"):
        tb._adaptive_benchmark_from_snapshot(snapshot)


def test_snapshot_benchmark_freezes_fdidm_transform_dimensions_after_resize():
    tb = _make_backend(m_subcarriers=4, n_symbols=4)
    with tb._lock:
        snapshot = tb._adaptive_snapshot_locked()

    tb.set_frame_shape(8, 8)

    bench = tb._adaptive_benchmark_from_snapshot(snapshot)
    result = bench.evaluate("FDIDM", 10.0, alpha=0.2, beta=0.3)

    assert bench.K == 16
    assert np.isfinite(float(result["ser"]))


def test_dynamic_channel_keeps_reference_power_for_ser_plot():
    """Dynamic phase walks must not move every SER curve via total gain drift."""
    tb = _make_backend(
        mod_order="QPSK",
        snr_db=10.0,
        decoder="ZF",
        dynamic_channel=True,
        channel_dynamics="block",
        channel_coherence_frames=4,
        random_channel=True,
        channel_seed=42,
    )
    powers = []
    for _ in range(20):
        tb.step()
        powers.append(float(tb.get_last_metrics()["channel_power_db"]))
    # The dynamic channel still changes its matrix/condition, but its total
    # power is held at the first snapshot (floating-point tolerance only).
    assert max(powers) - min(powers) < 1e-9


# ---------------------------------------------------------------- closed loop
def test_closed_loop_dynamic_channel_apply_and_cooldown():
    tb = _make_backend(dynamic_channel=True, channel_dynamics="block",
                       channel_coherence_frames=8)
    tb.start_adaptive_tuning(interval_frames=1, stability_evals=2,
                             min_improvement_db=0.0, auto_apply=True,
                             cooldown_frames=5, max_order=512)
    _run_frames(tb, 60)
    assert _wait_for(lambda: sum(h["kind"] == "eval" for h in tb.get_adaptive_history()) >= 2
                     and any(h["kind"] == "switch" for h in tb.get_adaptive_history()))
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
                             cooldown_frames=0, window_frames=2,
                             ensemble_snapshots=2)
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
    assert _wait_for(lambda: sum(h["kind"] == "eval" for h in tb.get_adaptive_history()) >= 2
                     and any(h["kind"] == "switch" for h in tb.get_adaptive_history()))
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
    status_before = tb.get_adaptive_status()
    ctx_before = status_before["context_key"]
    switches_before = [h for h in tb.get_adaptive_history() if h["kind"] == "switch"]
    n_switches_before = len(switches_before)
    seq_before = max([int(s["seq"]) for s in switches_before], default=0)
    tb.update_runtime_parameters(mod_order="QPSK")
    _run_frames(tb, 10)
    status = tb.get_adaptive_status()
    assert status["context_key"] != ctx_before
    # 上下文失效后，任何应用都必须基于失效之后（seq 更大）的推荐，不能沿用旧信道的
    # 稳定结论；修复块推进后同一信道块内稳定计数可以重新累积，因此只约束其上限。
    switches = [h for h in tb.get_adaptive_history() if h["kind"] == "switch"]
    new_switches = switches[n_switches_before:]
    assert all(int(s["seq"]) > seq_before for s in new_switches)
    assert status["stable_count"] <= 10
    tb.stop_adaptive_tuning()
    tb.stop()


@pytest.mark.parametrize(
    "update",
    [
        {"channel_seed": 43},
        {"random_channel": False},
        {"circular_channel": False},
        {"tf_notch_depth_db": 12.0, "tf_notch_count": 1},
        {"fast_channel_coherence_symbols": 2},
    ],
)
def test_channel_scoring_configuration_changes_adaptive_context(update):
    tb = _make_backend()
    with tb._lock:
        before = tb._adaptive_context_key_locked()
    tb.update_runtime_parameters(**update)
    with tb._lock:
        after = tb._adaptive_context_key_locked()
    assert after != before


def test_adaptive_run_lifecycle_advances_context_generation():
    tb = _make_backend()
    tb.start_adaptive_tuning(auto_apply=False)
    first_generation = int(tb._adaptive_context_epoch)
    tb.stop_adaptive_tuning()
    stopped_generation = int(tb._adaptive_context_epoch)
    tb.start_adaptive_tuning(auto_apply=False)
    restarted_generation = int(tb._adaptive_context_epoch)
    tb.stop_adaptive_tuning()

    assert stopped_generation > first_generation
    assert restarted_generation > stopped_generation


def test_decision_config_change_clears_snapshots_from_previous_generation():
    tb = _make_backend()
    tb.start_adaptive_tuning(auto_apply=False, window_frames=8, ensemble_snapshots=3)
    try:
        with tb._adaptive_lock:
            snapshot = {
                "frame": 1,
                "context_key": tb._adaptive_context_key_locked(),
                "adaptive_context_epoch": int(tb._adaptive_context_epoch),
            }
            tb._adaptive_window.extend([dict(snapshot), dict(snapshot)])
            old_epoch = int(tb._adaptive_context_epoch)

        tb.update_adaptive_config(
            min_improvement_db=float(tb._adaptive_cfg("min_improvement_db")) + 0.1
        )

        with tb._adaptive_lock:
            assert int(tb._adaptive_context_epoch) > old_epoch
            assert len(tb._adaptive_window) == 0
            assert tb._adaptive_pending_ensemble is None
    finally:
        tb.stop_adaptive_tuning()


def test_ensemble_snapshot_count_is_clamped_to_two():
    tb = _make_backend()

    tb.update_adaptive_config(ensemble_snapshots=1)

    assert int(tb._adaptive_cfg("ensemble_snapshots")) == 2
    selected = tb._select_ensemble(
        [{"frame": index} for index in range(5)],
        int(tb._adaptive_cfg("ensemble_snapshots")),
    )
    assert [item["frame"] for item in selected] == [0, 4]


def test_restart_waits_for_stopping_worker_before_starting_new_worker():
    tb = _make_backend()
    entered = threading.Event()
    release = threading.Event()

    def block_search(_ensemble):
        entered.set()
        assert release.wait(5.0)

    tb._adaptive_process_ensemble = block_search
    try:
        tb.start_adaptive_tuning(auto_apply=False)
        with tb._adaptive_lock:
            tb._adaptive_pending_ensemble = [{}]
            tb._adaptive_search_event.set()
        assert entered.wait(2.0)
        old_worker = tb._adaptive_search_thread
        assert old_worker is not None

        tb.stop_adaptive_tuning()
        assert old_worker.is_alive()

        tb.start_adaptive_tuning(auto_apply=False)
        assert tb._adaptive_search_thread is old_worker

        release.set()
        assert _wait_for(
            lambda: tb._adaptive_search_thread is not None
            and tb._adaptive_search_thread is not old_worker
            and tb._adaptive_search_thread.is_alive(),
            timeout=5.0,
        )
        assert sum(
            thread.name == "fdidm-slow-search" and thread.is_alive()
            for thread in threading.enumerate()
        ) == 1
    finally:
        release.set()
        tb.stop_adaptive_tuning()


def test_inflight_search_is_discarded_after_snr_context_change(monkeypatch):
    from waveform_sim.simulation.shared_waveform_benchmark import SharedWaveformBenchmark

    tb = _make_backend(snr_db=8.0)
    tb.start_adaptive_tuning(
        interval_frames=1000,
        stability_evals=1,
        min_improvement_db=0.0,
        auto_apply=True,
        cooldown_frames=0,
        window_frames=2,
        ensemble_snapshots=2,
    )
    with tb._lock:
        snapshot = tb._adaptive_snapshot_locked()
    entered = threading.Event()
    release = threading.Event()

    def blocked_optimize(cls, benchmarks, snr_db, **kwargs):
        entered.set()
        assert release.wait(5.0)
        return {
            "recommended_alpha": 1.0,
            "recommended_beta": 0.0,
            "predicted_ser_current": 0.2,
            "predicted_ser_best": 0.1,
            "predicted_geometric_ser_current": 0.2,
            "predicted_geometric_ser_best": 0.1,
            "decision_ser_floor": 1e-8,
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
        classmethod(blocked_optimize),
    )
    errors = []

    def run_search():
        try:
            tb._adaptive_process_ensemble([snapshot])
        except Exception as exc:  # pragma: no cover - assertion below surfaces it
            errors.append(exc)

    worker = threading.Thread(target=run_search)
    worker.start()
    try:
        assert entered.wait(5.0)
        tb.update_runtime_parameters(snr_db=20.0)
        tb.step()
    finally:
        release.set()
        worker.join(timeout=10.0)
        tb.stop_adaptive_tuning()

    assert not worker.is_alive()
    assert not errors
    assert (float(tb.config.alpha), float(tb.config.beta)) == (0.0, 0.0)
    assert not [item for item in tb.get_adaptive_history() if item.get("kind") == "switch"]


def test_context_change_between_ready_decision_and_write_marks_eval_stale(monkeypatch):
    from waveform_sim.simulation.shared_waveform_benchmark import SharedWaveformBenchmark

    tb = _make_backend(snr_db=8.0)

    def change_context(_recommendation):
        tb.update_runtime_parameters(snr_db=20.0)
        tb.step()

    tb.start_adaptive_tuning(
        callback=change_context,
        stability_evals=1,
        min_improvement_db=0.0,
        auto_apply=True,
        cooldown_frames=0,
    )
    with tb._lock:
        snapshot = tb._adaptive_snapshot_locked()

    def optimized(cls, benchmarks, snr_db, **kwargs):
        return {
            "recommended_alpha": 1.0, "recommended_beta": 0.0,
            "predicted_ser_current": 0.2, "predicted_ser_best": 0.1,
            "predicted_geometric_ser_current": 0.2,
            "predicted_geometric_ser_best": 0.1,
            "decision_ser_floor": 1e-8,
            "decision_geometric_ser_current": 0.2,
            "decision_geometric_ser_best": 0.1,
            "predicted_improvement_db": 3.0102999566,
            "candidate_count": 9, "search_seconds": 0.01,
            "top_candidates": [], "ensemble_size": len(benchmarks),
            "working_snr_db": float(snr_db), "search_mode": "test",
        }

    monkeypatch.setattr(
        SharedWaveformBenchmark,
        "optimize_fdidm_over_ensemble",
        classmethod(optimized),
    )
    try:
        tb._adaptive_process_ensemble([snapshot])
    finally:
        tb.stop_adaptive_tuning()

    evaluations = [item for item in tb.get_adaptive_history() if item.get("kind") == "eval"]
    assert evaluations and evaluations[-1]["action"] == "stale"
    assert (float(tb.config.alpha), float(tb.config.beta)) == (0.0, 0.0)
    assert not [item for item in tb.get_adaptive_history() if item.get("kind") == "switch"]


def test_adaptive_benchmark_ignores_selected_decoder_for_theory_policy():
    from waveform_sim.simulation.shared_waveform_benchmark import SharedWaveformBenchmark

    tb = _make_backend(decoder="ZF")
    with tb._lock:
        snapshot = tb._adaptive_snapshot_locked()
    snapshot["equalizer"] = "ZF-SIC"

    bench = tb._adaptive_benchmark_from_snapshot(snapshot)

    assert bench.context.receiver_policy == SharedWaveformBenchmark.RECEIVER_POLICY
    assert bench.evaluate("FDIDM", 10.0, 0.0, 0.0)["receiver_policy"] == (
        SharedWaveformBenchmark.RECEIVER_POLICY
    )


def test_realtime_decoder_selection_does_not_invalidate_adaptive_context():
    from dataclasses import replace

    tb = _make_backend(decoder="ZF")
    with tb._lock:
        before = tb._adaptive_context_key_locked()
        tb.config = replace(tb.config, decoder="MMSE")
        after = tb._adaptive_context_key_locked()

    assert after == before


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


# ---------------------------------------------------------------- regressions
def test_dynamic_block_progression_monotonic_after_apply():
    """回归：应用 α/β 不得重置仿真帧计数/信道块索引。

    旧 bug：_apply_config_update -> reset_ber_stats() 把 total_frames 清零并把
    _last_dynamic_block 置 None，块索引随之回退，信道在 block0/1 之间循环。
    """
    tb = _make_backend(dynamic_channel=True, channel_dynamics="block",
                       channel_coherence_frames=8)
    tb.start_adaptive_tuning(interval_frames=1, stability_evals=1,
                             min_improvement_db=0.0, auto_apply=True,
                             cooldown_frames=1, max_order=512)
    frames_seen, seeds_seen = [], []
    for _ in range(40):
        tb.step()
        with tb._lock:
            frames_seen.append(int(tb._sim_frame))
            seeds_seen.append(int(tb.config.channel_seed))
    assert frames_seen == sorted(frames_seen)
    blocks = [(f - 1) // 8 for f in frames_seen]
    assert blocks == sorted(blocks)
    # 40 帧应推进到 block 4（至少 5 个不同的块种子），不能停留在 block0/1 循环。
    assert len(set(seeds_seen)) >= 5
    tb.stop_adaptive_tuning()
    tb.stop()


def test_continuous_doppler_mode_evolves_per_frame():
    """连续多普勒：每帧按路径多普勒旋转相位、信道连续演化，且不重播种。"""
    tb = _make_backend(dynamic_channel=True, channel_dynamics="cont")
    with tb._lock:
        tb._prepare_matrices_locked()
        H0 = np.asarray(tb._H_tf, dtype=np.complex128).copy()
        seed0 = int(tb.config.channel_seed)
        phase0 = np.asarray(tb._cont_phase, dtype=np.complex128).copy()
        dopplers = np.asarray(tb._cont_paths[2], dtype=np.float64).copy()
    tb.step()
    with tb._lock:
        H1 = np.asarray(tb._H_tf, dtype=np.complex128).copy()
        seed1 = int(tb.config.channel_seed)
        phase1 = np.asarray(tb._cont_phase, dtype=np.complex128).copy()
    assert seed0 == seed1, "cont 模式不应更换信道种子"
    assert not np.allclose(H0, H1), "每帧信道应随多普勒相位演化"
    cfg = tb.config
    T_frame = float(cfg.n_symbols) / max(float(cfg.subcarrier_spacing_hz), 1e-15)
    expected = np.exp(1j * 2.0 * np.pi * dopplers * T_frame)
    assert np.allclose(np.abs(phase1), np.abs(phase0), atol=1e-12)
    assert np.allclose(phase1, phase0 * expected, atol=1e-12)
    tb.stop_adaptive_tuning()
    tb.stop()


def test_continuous_doppler_deterministic_same_seed():
    """连续多普勒：相同种子产生相同的逐帧信道序列。"""
    a = _make_backend(dynamic_channel=True, channel_dynamics="cont")
    b = _make_backend(dynamic_channel=True, channel_dynamics="cont")
    a.step()
    b.step()
    with a._lock:
        Ha = np.asarray(a._H_tf, dtype=np.complex128).copy()
    with b._lock:
        Hb = np.asarray(b._H_tf, dtype=np.complex128).copy()
    assert np.allclose(Ha, Hb)
    a.stop_adaptive_tuning()
    b.stop_adaptive_tuning()
    a.stop()
    b.stop()


def test_runtime_param_change_during_sim_no_shape_mismatch():
    """回归：仿真线程运行中修改调制/解码参数不得引发 bits 与硬判长度不匹配。

    旧 bug：_simulate_one_frame 在锁外生成 bits，GUI 线程此时切换 mod_order
    会让 256 bit 与 128 bit 硬判结果广播失败（ValueError）。
    """
    tb = _make_backend(m_subcarriers=8, n_symbols=8, mod_order="16QAM",
                       dynamic_channel=True, channel_dynamics="block",
                       channel_coherence_frames=4)
    tb.start()
    mods = ["QPSK", "16QAM", "64QAM", "16QAM", "QPSK"]
    decs = ["ZF", "MMSE", "ZF-SIC"]
    t0 = time.time()
    i = 0
    last_frame = 0
    try:
        while time.time() - t0 < 2.5:
            tb.update_runtime_parameters(mod_order=mods[i % len(mods)],
                                         decoder=decs[i % len(decs)])
            i += 1
            time.sleep(0.02)
            with tb._lock:
                last_frame = int(tb._sim_frame)
    finally:
        tb.stop()
        tb.wait(timeout=2.0)
    assert last_frame >= 5, "仿真线程崩溃或停滞（帧号未推进）"


def test_block_mode_evolution_is_correlated_not_jumpy():
    """回归：块衰落相邻块应相关演化，SER 逐块平稳变化而非独立跳变。"""
    tb = _make_backend(m_subcarriers=8, n_symbols=8, snr_db=10.0,
                       dynamic_channel=True, channel_dynamics="block",
                       channel_coherence_frames=8)
    b = tb
    channels = []
    vals = []
    for _ in range(12):
        for _ in range(8):
            b.step()
        with b._lock:
            b._prepare_matrices_locked()
            channel = b._H_cross.copy()
            channels.append(channel)
            ser = b._zf_theory_ser_for_channel(channel, ebn0_db=10.0)[0]
        vals.append(float(ser))
    assert np.all(np.isfinite(vals))
    d = 1.0 - np.asarray([
        abs(np.vdot(left, right))
        / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-15)
        for left, right in zip(channels, channels[1:])
    ])
    assert np.median(d) < 0.2, "相邻块 SER 跳变过大"
    assert d.max() < 0.7, "存在异常大的逐块 SER 跳变"
    tb.stop_adaptive_tuning()
    tb.stop()


# ---------------------------------------------------------------- engine
def test_engine_passthrough():
    from waveform_sim.core.engine import LinkSimulator

    sim = LinkSimulator(waveform="FDIDM", m_subcarriers=4, n_symbols=4, snr_db=12.0)
    assert sim.get_adaptive_history() == []
    st = sim.get_adaptive_status()
    assert st["enabled"] is False
    sim.step()
    sim.start_adaptive_tuning(interval_frames=1, stability_evals=1,
                              min_improvement_db=0.0, auto_apply=False,
                              window_frames=2, ensemble_snapshots=2)
    _run_frames(sim, 8)
    assert _wait_for(lambda: sim.get_adaptive_status()["recommendation_seq"] >= 1)
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
