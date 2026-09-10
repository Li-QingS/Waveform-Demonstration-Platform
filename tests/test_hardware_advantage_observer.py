"""优势观测引擎离线单测：窗口汇总、瞬态/异常剔除、分级、配对与导出。

不依赖 PyQt / GNU Radio / USRP（spec N4）。
"""
import json
import math

import numpy as np
import pytest

from waveform_sim.ui.hardware_advantage_observer import (
    AdvantageObservationSession,
    MODE_OFF,
    MODE_ON,
    grade_from_k,
    ser_improvement_db,
)


class FakeClock:
    """0.1s 精度的整数刻度时钟，避免浮点累加漂移。"""

    def __init__(self):
        self.ticks = 0

    def __call__(self):
        return self.ticks * 0.1

    def advance(self, n=1):
        self.ticks += n


def make_status(frames, ok=0, ser=0.01, fec_ber=0.001, evm=8.0, overflow=0,
                alpha=0.5, beta=1.0, **overrides):
    base = {
        "frames_processed": frames,
        "frames_decode_ok": ok,
        "rx_overflow_count": overflow,
        "measured_ser": ser,
        "fec_bit_ber": fec_ber,
        "evm_average_percent": evm,
        "alpha": alpha,
        "beta": beta,
        "tx_uncoded_bits_len": 1200,
        "mod_order": 4,
        "tx_coded_bits_len": 2400,
        "channel_mode": "tdl_a_rf",
        "tdl_model": "tdl_a",
        "tdl_doppler_hz": 50.0,
        "tdl_rms_delay_spread_ns": 30.0,
        "tdl_snr_db": 20.0,
        "tdl_seed": 7,
    }
    base.update(overrides)
    return base


def feed(session, clock, n, ser, frames_per_sample=2, **overrides):
    frames = overrides.pop("_frames", None)
    if frames is None:
        frames = int(session._last_status_frames)
    ok = overrides.pop("_ok", None)
    if ok is None:
        ok = int(session._last_status_ok)
    overflow = overrides.pop("_overflow", None)
    if overflow is None:
        overflow = int(session._last_status_overflow)
    for _ in range(n):
        clock.advance()
        frames += frames_per_sample
        ok += frames_per_sample
        session.on_sample(make_status(frames, ok=ok, ser=ser, overflow=overflow, **overrides))
    return frames, ok, overflow


def run_pair(ser_before=0.05, ser_after=0.01):
    """标准双段序列：关闭 40 采样 → 开关 → 开启 40 采样（开启段保持未收尾）。"""
    clock = FakeClock()
    session = AdvantageObservationSession(clock=clock)
    session.start(False, {})
    frames, ok, overflow = feed(session, clock, 40, ser_before)
    session.on_toggle(True, alpha=0.75, beta=0.8)
    feed(session, clock, 40, ser_after, _frames=frames, _ok=ok, _overflow=overflow,
         alpha=0.75, beta=0.8, evm=6.0)
    return session


def test_grade_from_k_thresholds():
    assert grade_from_k(100) == "trusted"
    assert grade_from_k(1000) == "trusted"
    assert grade_from_k(10) == "reference"
    assert grade_from_k(99) == "reference"
    assert grade_from_k(9.9) == "insufficient"
    assert grade_from_k(0) == "insufficient"
    assert grade_from_k(float("nan")) == "insufficient"


def test_ser_improvement_db():
    assert abs(ser_improvement_db(0.05, 0.01) - 10 * math.log10(5)) < 1e-9
    assert ser_improvement_db(0.01, 0.05) < 0
    assert np.isnan(ser_improvement_db(0.0, 0.01))
    assert np.isnan(ser_improvement_db(float("nan"), 0.01))


def test_pair_improvement_db_and_grades():
    session = run_pair(ser_before=0.05, ser_after=0.01)
    pair = session.latest_pair()
    assert pair is not None and pair.comparable
    # 每帧 600 符号，两段各 31 个有效采样 × 2 帧 = 62 帧 → k_before=1860, k_after=372
    assert abs(pair.ser_improvement_db - 10 * math.log10(5)) < 0.01
    assert pair.before.grade == "trusted" and pair.after.grade == "trusted"
    assert pair.before.ser_k == 1860.0 and pair.before.ser_n == 37200
    assert pair.after.ser_k == 372.0
    assert abs(pair.before.ser - 0.05) < 1e-9 and abs(pair.after.ser - 0.01) < 1e-9
    # EVM: 8% → 6%，变化为负（更好）
    assert abs(pair.evm_delta_pp - (-2.0)) < 1e-6


def test_transient_samples_excluded_from_aggregate():
    clock = FakeClock()
    session = AdvantageObservationSession(clock=clock)
    session.start(False, {})
    feed(session, clock, 40, 0.05)
    session.on_toggle(True, 0.75, 0.8)
    feed(session, clock, 5, 0.01)  # 全部处于段首瞬态
    assert session.latest_pair() is None  # 开启段尚无有效数据，不产生结论
    feed(session, clock, 10, 0.01)  # 继续收集，出现非瞬态样本
    pair = session.latest_pair()
    # 段首 1.0s（前 9 个采样 t<1.0）为瞬态，不计入：40 采样中有效帧 = 31×2=62
    assert pair.before.frames == 62


def test_overflow_samples_dropped_and_reported():
    clock = FakeClock()
    session = AdvantageObservationSession(clock=clock)
    session.start(False, {})
    feed(session, clock, 20, 0.05)
    # 单次 overflow：该采样丢弃，窗口继续有效
    frames = int(session._last_status_frames)
    overflow = int(session._last_status_overflow) + 1
    clock.advance()
    session.on_sample(make_status(frames + 2, ok=frames + 2, ser=0.05, overflow=overflow))
    feed(session, clock, 19, 0.05, _frames=frames + 2, _overflow=overflow)
    session.on_toggle(True, 0.75, 0.8)
    win = [w for w in session.windows() if w.mode == MODE_OFF][0]
    assert win.dropped_samples == 1 and "rx_overflow" in win.anomalies
    assert not win.excluded and win.aggregate is not None


def test_anomaly_heavy_window_excluded():
    clock = FakeClock()
    session = AdvantageObservationSession(clock=clock)
    session.start(False, {})
    frames = ok = overflow = 0
    for i in range(20):
        clock.advance()
        frames += 2
        ok += 2
        session.on_sample(make_status(frames, ok=ok, ser=0.05, overflow=overflow))
    for i in range(20):  # 一半采样带 overflow 增量 → 占比超 20%
        clock.advance()
        overflow += 1
        frames += 2
        session.on_sample(make_status(frames, ok=ok, ser=0.05, overflow=overflow))
    session.on_toggle(True, 0.75, 0.8)
    win = [w for w in session.windows() if w.mode == MODE_OFF][0]
    assert win.excluded and "rx_overflow" in win.anomalies
    assert session.excluded_window_count() == 1
    assert session.latest_pair() is None  # 无可用 off 窗口，不伪造结论


def test_grade_levels_flow_into_aggregate():
    def grade_for(ser):
        clock = FakeClock()
        session = AdvantageObservationSession(clock=clock)
        session.start(False, {})
        feed(session, clock, 50, ser)
        session.on_toggle(True, 0.75, 0.8)
        return [w for w in session.windows() if w.mode == MODE_OFF][0].aggregate.grade

    # 100 帧 × 600 符号：k = ser × 60000
    assert grade_for(0.01) == "trusted"      # k=600
    assert grade_for(0.0005) == "reference"  # k=30
    assert grade_for(1e-5) == "insufficient" # k=0.6


def test_context_mismatch_marks_pair_not_comparable():
    clock = FakeClock()
    session = AdvantageObservationSession(clock=clock)
    session.start(False, {})
    feed(session, clock, 40, 0.05)
    session.on_toggle(True, 0.75, 0.8)
    # 开启段 TDL 种子变化 → 上下文不可比
    feed(session, clock, 40, 0.01, tdl_seed=99)
    pair = session.latest_pair()
    assert pair is not None and not pair.comparable
    assert "tdl_seed" in pair.comparability_note


def test_export_dict_is_json_serializable():
    session = run_pair()
    data = session.to_export_dict()
    text = json.dumps(data, ensure_ascii=False)
    assert data["schema_version"] == 2
    assert len(data["toggle_events"]) == 1
    assert data["toggle_events"][0]["enabled"] is True
    assert any(w["mode"] == MODE_OFF for w in data["windows"])
    assert data["latest_pair"]["before"]["ser"] is not None
    assert "excluded_window_count" in data


def feed_exact(session, clock, n, ser, *, state="", power_rms=0.2,
               contract_id="power-1", frames_per_sample=2, **overrides):
    frames = int(session._last_status_frames)
    ok = int(session._last_status_ok)
    errors_total = int(session._last_ser_errors_total)
    symbols_total = int(session._last_ser_symbols_total)
    symbols_step = 600 * frames_per_sample
    errors_step = int(round(ser * symbols_step))
    for _ in range(n):
        clock.advance()
        frames += frames_per_sample
        ok += frames_per_sample
        errors_total += errors_step
        symbols_total += symbols_step
        session.on_sample(make_status(
            frames,
            ok=ok,
            ser=ser,
            evm=overrides.get("data_aided_evm_percent", 8.0),
            ser_errors_total=errors_total,
            ser_symbols_total=symbols_total,
            data_aided_evm_percent=overrides.get("data_aided_evm_percent", 8.0),
            decision_directed_evm_percent=overrides.get("decision_directed_evm_percent", 6.0),
            raw_bit_ber=overrides.get("raw_bit_ber", ser / 2.0),
            tx_cycle_rms=power_rms,
            tx_power_contract_id=contract_id,
            adaptive_validation_state=state,
            adaptive_validation_reason=overrides.get("adaptive_validation_reason", ""),
            frame_structure=overrides.get("frame_structure", {"data_samples": 320, "pilot_samples": 320}),
            pilot_residual_sinr_db=overrides.get("pilot_residual_sinr_db", 12.0),
            pilot_fit_nmse=overrides.get("pilot_fit_nmse", 0.05),
            alpha=overrides.get("alpha", 0.5),
            beta=overrides.get("beta", 1.0),
        ))


def test_exact_cumulative_counts_drive_aggregate_and_improved_outcome():
    clock = FakeClock()
    session = AdvantageObservationSession(clock=clock)
    session.start(False, {})
    feed_exact(session, clock, 40, 0.05, data_aided_evm_percent=10.0)
    assert session.on_toggle(True, 0.75, 0.8) is not None
    feed_exact(session, clock, 40, 0.01, state="improved", data_aided_evm_percent=7.0,
               alpha=0.75, beta=0.8)
    pair = session.latest_pair()
    assert pair is not None
    assert not pair.before.estimated_k and not pair.after.estimated_k
    assert pair.before.ser_k == 1860
    assert pair.before.ser_n == 37200
    assert pair.after.ser_k == 372
    assert pair.outcome == "improved"
    assert pair.before.decision_evm_mean == pytest.approx(6.0)
    assert pair.before.raw_ber == pytest.approx(0.025)
    assert pair.before.wilson_low < pair.before.ser < pair.before.wilson_high


def test_legacy_counts_remain_compatible_but_cannot_claim_improved():
    pair = run_pair().latest_pair()
    assert pair.before.estimated_k and pair.after.estimated_k
    assert pair.outcome == "inconclusive"
    assert "估算" in pair.validation_reason


def test_power_contract_mismatch_marks_pair_inconclusive():
    clock = FakeClock()
    session = AdvantageObservationSession(clock=clock)
    session.start(False, {})
    feed_exact(session, clock, 40, 0.05, power_rms=0.2, contract_id="p1")
    session.on_toggle(True, 0.75, 0.8)
    feed_exact(session, clock, 40, 0.01, power_rms=0.18, contract_id="p1",
               alpha=0.75, beta=0.8)
    pair = session.latest_pair()
    assert not pair.comparable
    assert pair.outcome == "inconclusive"
    assert "功率不可比" in pair.comparability_note


def test_backend_validation_outcome_and_reason_are_preserved():
    clock = FakeClock()
    session = AdvantageObservationSession(clock=clock)
    session.start(False, {})
    feed_exact(session, clock, 40, 0.05)
    session.on_toggle(True, 0.75, 0.8)
    feed_exact(session, clock, 40, 0.01, state="regressed",
               adaptive_validation_reason="CRC success ratio regressed",
               alpha=0.75, beta=0.8)
    pair = session.latest_pair()
    assert pair.outcome == "regressed"
    assert pair.validation_reason == "CRC success ratio regressed"


def test_toggle_only_records_a_real_mode_change():
    session = AdvantageObservationSession(clock=FakeClock())
    assert session.on_toggle(True, 0.75, 0.8) is None
    session.start(False, {})
    assert session.on_toggle(False, 0.5, 1.0) is None
    assert session.events() == []
    assert session.on_toggle(True, 0.75, 0.8) is not None
    assert len(session.events()) == 1


def test_export_contains_exact_power_evm_validation_and_diagnostics():
    clock = FakeClock()
    session = AdvantageObservationSession(clock=clock)
    session.start(False, {})
    feed_exact(session, clock, 40, 0.05, data_aided_evm_percent=10.0)
    session.on_toggle(True, 0.75, 0.8)
    feed_exact(session, clock, 40, 0.01, state="improved",
               data_aided_evm_percent=7.0, alpha=0.75, beta=0.8)
    data = session.to_export_dict()
    pair = data["latest_pair"]
    assert pair["outcome"] == "improved"
    assert pair["before"]["estimated_counts"] is False
    assert pair["before"]["tx_power_contract_id"] == "power-1"
    assert pair["before"]["decision_evm_mean"] == pytest.approx(6.0)
    assert pair["before"]["wilson_low"] is not None
    assert data["context"]["frame_structure"]["pilot_samples"] == 320
    assert data["context"]["pilot_residual_sinr_db"] == 12.0
    json.dumps(data, ensure_ascii=False, allow_nan=False)


def test_stop_keeps_results_and_pair_usable():
    clock = FakeClock()
    session = AdvantageObservationSession(clock=clock)
    session.start(False, {})
    feed(session, clock, 40, 0.05)
    session.on_toggle(True, 0.75, 0.8)
    feed(session, clock, 40, 0.01)
    session.stop()
    assert not session.active
    pair = session.latest_pair()
    assert pair is not None and abs(pair.ser_improvement_db - 10 * math.log10(5)) < 0.01
    # stop 后结果保留，再次 start 清空（F1）
    session.start(False, {})
    assert session.windows() == [] and session.latest_pair() is None


def test_config_change_reopens_window():
    clock = FakeClock()
    session = AdvantageObservationSession(clock=clock)
    session.start(False, {})
    feed(session, clock, 20, 0.05)
    feed(session, clock, 20, 0.05, tx_uncoded_bits_len=2400)  # 配置变化 → 强制收尾重开
    session.on_toggle(True, 0.75, 0.8)  # 收尾重开的窗口
    off_windows = [w for w in session.windows() if w.mode == MODE_OFF]
    assert len(off_windows) == 2
    assert off_windows[0].aggregate.frames == 22  # 只有配置变化前的帧（11 个非瞬态采样）


def test_idle_samples_ignored_and_missing_fields_tolerated():
    clock = FakeClock()
    session = AdvantageObservationSession(clock=clock)
    session.start(False, {})
    feed(session, clock, 10, 0.05)
    frames = int(session._last_status_frames)
    clock.advance()
    session.on_sample({"frames_processed": frames})  # 链路空闲 + 大量字段缺失
    clock.advance()
    session.on_sample({"frames_processed": frames + 2, "frames_decode_ok": 2,
                       "rx_overflow_count": 0})  # 帧推进但无 SER → sync_loss 剔除
    session.on_toggle(True, 0.75, 0.8)
    win = session.windows()[0]
    assert win.dropped_samples == 1 and "sync_loss" in win.anomalies
    assert win.valid_samples == 10
