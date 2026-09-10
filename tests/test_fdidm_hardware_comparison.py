"""离屏验证 FDIDM 硬件优势观测面板（替代旧的定时对比会话测试）。

覆盖 spec：AC1 面板初始化与配对统计、AC2 异常窗口剔除显示、AC4 时间轴与
状态行、AC6 live 路径无 stop/start、AC8 导出报告。不依赖 USRP / GNU Radio。
"""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from PyQt5.QtWidgets import QApplication, QFileDialog

from waveform_sim.ui.fdidm_hardware_test_tab import FDIDMHardwareTestTab
from waveform_sim.ui.hardware_advantage_observer import AdvantageObservationSession


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class FakeClock:
    """0.1s 精度的整数刻度时钟，避免浮点累加漂移。"""

    def __init__(self):
        self.ticks = 0

    def __call__(self):
        return self.ticks * 0.1

    def advance(self, n=1):
        self.ticks += n


class FakeBackend:
    def __init__(self):
        self.frames = 0
        self.ok = 0
        self.overflow = 0
        self.start_calls = 0
        self.stop_calls = 0
        self.configures = []

    def start(self):
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1

    def wait(self):
        return None

    def configure(self, **kwargs):
        self.configures.append(kwargs)

    def get_status(self):
        return make_status(self.frames, self.ok, overflow=self.overflow)

    def get_decode_stats(self):
        return {"match_ratio": 0.9, "decode_ok": True}


def make_status(frames, ok, ser=0.05, evm=8.0, overflow=0, alpha=0.5, beta=1.0,
                contract_id="common-rms", tx_rms=0.2):
    symbols = int(frames) * 600
    return {
        "frames_processed": frames,
        "frames_decode_ok": ok,
        "rx_overflow_count": overflow,
        "measured_ser": ser,
        "fec_bit_ber": ser / 5,
        "evm_average_percent": evm,
        "evm_percent": evm,
        "data_aided_evm_percent": evm,
        "decision_directed_evm_percent": max(0.0, evm - 1.0),
        "ser_errors_total": int(round(symbols * ser)),
        "ser_symbols_total": symbols,
        "tx_power_contract_id": contract_id,
        "tx_cycle_rms": tx_rms,
        "tx_data_rms": tx_rms,
        "tx_peak": 0.75,
        "tx_papr_db": 7.5,
        "tx_power_backoff_db": 0.0,
        "pilot_residual_sinr_db": 18.0,
        "pilot_fit_nmse": 0.016,
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
        "tdl_injected_snr_db": 20.0,
        "tdl_seed": 7,
        "frame_structure": {
            "data_samples": 256,
            "pilot_samples": 256,
            "guard_samples": 64,
            "training_data_ratio": 1.0,
            "useful_data_ratio": 0.4,
        },
        "adaptive_alpha_beta_state": "monitoring",
        "adaptive_validation_state": "idle",
    }


def make_tab(app):
    tab = FDIDMHardwareTestTab()
    tab.backend = FakeBackend()
    tab.test_running = True
    clock = FakeClock()
    tab.observer = AdvantageObservationSession(clock=clock)
    return tab, clock


def feed(tab, clock, n, ser, evm=8.0, overflow_step=0, alpha=0.5, beta=1.0,
         contract_id="common-rms", tx_rms=0.2):
    frames = tab.observer._last_status_frames
    ok = tab.observer._last_status_ok
    overflow = tab.observer._last_status_overflow
    ser_errors = tab.observer._last_ser_errors_total
    ser_symbols = tab.observer._last_ser_symbols_total
    for _ in range(n):
        clock.advance()
        frames += 2
        ok += 2
        overflow += overflow_step
        ser_symbols += 1200
        ser_errors += int(round(float(ser) * 1200))
        status = make_status(frames, ok, ser=ser, evm=evm, overflow=overflow,
                             alpha=alpha, beta=beta, contract_id=contract_id,
                             tx_rms=tx_rms)
        status["ser_errors_total"] = ser_errors
        status["ser_symbols_total"] = ser_symbols
        tab.observer.on_sample(status)
        tab._append_timeline_sample(status)
    return frames, ok, overflow


def test_observation_controls_default_state(app):
    tab = FDIDMHardwareTestTab()
    assert not tab.btn_start_observation.isEnabled()
    assert not tab.btn_stop_observation.isEnabled()
    assert not tab.btn_export_observation.isEnabled()
    assert tab.btn_start_observation.text() == "重新开始观测"
    # 旧定时对比控件已彻底删除（F7）
    assert not hasattr(tab, "btn_start_comparison")
    assert not hasattr(tab, "comparison_session")
    assert "自动记录" in tab.observation_state_label.text()


def test_hardware_start_automatically_starts_observation_and_stop_preserves_it(app):
    tab = FDIDMHardwareTestTab()
    tab.backend = FakeBackend()
    tab._on_start_test_clicked()
    try:
        assert tab.test_running
        assert tab.observer.active
        assert tab.btn_start_observation.isEnabled()
        assert tab.btn_stop_observation.isEnabled()
        assert any("自动开始" in line for line in tab._ui_log_entries)
    finally:
        tab._on_stop_test_clicked()
    assert not tab.observer.active
    assert not tab.btn_start_observation.isEnabled()
    assert not tab.btn_stop_observation.isEnabled()


def test_automatic_end_to_end_observation_needs_no_extra_start_click(app):
    tab = FDIDMHardwareTestTab()
    tab.backend = FakeBackend()
    clock = FakeClock()
    tab.observer = AdvantageObservationSession(clock=clock)
    tab._on_start_test_clicked()
    feed(tab, clock, 40, ser=0.05, evm=8.0)
    tab.adaptive_enable_check.setChecked(True)
    feed(tab, clock, 40, ser=0.01, evm=6.0, alpha=0.75, beta=0.8)
    tab._on_stop_test_clicked()
    pair = tab.observer.latest_pair()
    assert pair is not None and pair.outcome == "improved"
    assert len(tab.observer.windows()) == 2
    assert tab.btn_export_observation.isEnabled()


def test_restart_observation_clears_old_windows(app):
    tab, clock = make_tab(app)
    tab._start_observation_clicked()
    feed(tab, clock, 20, ser=0.05)
    tab._stop_observation_clicked()
    assert tab.observer.windows()
    tab._start_observation_clicked()
    assert tab.observer.active
    assert tab.observer.windows() == []
    assert tab.observer.events() == []


def test_live_toggle_pending_is_initialized_and_keeps_runtime_curve(app):
    tab, clock = make_tab(app)
    tab._start_observation_clicked()
    tab._update_evm_plot(7.0)
    before = list(tab._evm_history)
    tab.adaptive_enable_check.setChecked(True)
    assert tab._adaptive_toggle_pending
    assert len(tab.observer.events()) == 1
    assert "窗口已切换" in tab._ui_log_entries[-1]
    tab._apply_params_to_backend()
    assert not tab._adaptive_toggle_pending
    assert list(tab._evm_history) == before


def test_toggle_feedback_is_truthful_when_observation_is_inactive(app):
    tab, _clock = make_tab(app)
    tab.observer.stop()
    tab._on_adaptive_enable_changed(2)
    assert "当前观测已停止" in tab._ui_log_entries[-1]
    assert "窗口已切换" not in tab._ui_log_entries[-1]


def test_observation_pair_updates_panel_and_timeline(app):
    tab, clock = make_tab(app)
    tab._start_observation_clicked()
    feed(tab, clock, 40, ser=0.05, evm=8.0)
    tab.adaptive_enable_check.setChecked(True)  # 评审手动开关 → 事件
    feed(tab, clock, 40, ser=0.01, evm=6.0, alpha=0.75, beta=0.8)
    tab._update_observation_display()

    pair = tab.observer.latest_pair()
    assert pair is not None and pair.comparable
    assert pair.outcome == "improved"
    assert "实测优势" in tab.observation_improvement_label.text()
    # v2：两页签各四宫格；观测页统计表出现窗口汇总行（AC6/AC7）
    assert tab.plot_tabs.count() == 2
    assert "开启前" in tab.observation_stats_label.text()
    assert "可信" in tab.observation_badge_label.text()
    assert "精确计数" in tab.observation_badge_label.text()
    assert "数据辅助EVM" in tab.observation_evm_label.text()
    stats_text = tab.observation_stats_label.text()
    assert "raw/FEC BER" in stats_text and "CRC" in stats_text
    assert "周期/数据RMS" in stats_text and "PAPR" in stats_text
    assert "合同 common-rms" in stats_text
    assert "SER" in tab.observation_before_label.text()
    # 时间轴：曲线有数据 + 一条开关事件竖线（AC4）
    assert tab.timeline_ser_curve.xData is not None and len(tab.timeline_ser_curve.xData) > 0
    assert len(tab._timeline_event_items) == 1
    # live 路径：观测全程未触发后端 start/stop（AC6）
    assert tab.backend.start_calls == 0 and tab.backend.stop_calls == 0


def test_anomaly_window_excluded_shown_in_panel(app):
    tab, clock = make_tab(app)
    tab._start_observation_clicked()
    feed(tab, clock, 20, ser=0.05)
    feed(tab, clock, 10, ser=0.05, overflow_step=1)  # 1/3 采样带 overflow → 整窗剔除
    tab.adaptive_enable_check.setChecked(True)
    feed(tab, clock, 20, ser=0.01, evm=6.0)
    tab._update_observation_display()
    # 剔除数在面板可见（AC2）
    assert "剔除 1 个异常窗口" in tab.observation_state_label.text()
    assert len(tab._timeline_anomaly_items) >= 1  # 时间轴异常剔除阴影（AC4）
    # 关闭期窗口被剔除，不得伪造结论（AC2/AC5）
    assert tab.observer.latest_pair() is None
    assert "—" in tab.observation_improvement_label.text()


def test_insufficient_samples_do_not_form_conclusion(app):
    tab, clock = make_tab(app)
    tab._start_observation_clicked()
    feed(tab, clock, 40, ser=0.05)
    tab.adaptive_enable_check.setChecked(True)
    feed(tab, clock, 40, ser=1e-5, evm=6.0)  # 开启段错误符号数 <10
    tab._update_observation_display()
    assert "样本不足" in tab.observation_improvement_label.text()
    assert "dB" not in tab.observation_improvement_label.text()
    assert "样本不足" in tab.observation_badge_label.text()


def test_adaptive_state_row_shows_observability(app):
    tab, clock = make_tab(app)
    status = make_status(10, 10)
    status["adaptive_alpha_observable"] = False
    status["adaptive_beta_observable"] = True
    tab._update_adaptive_state_row(status)
    txt = tab.adaptive_state_label.text()
    assert "α可观测:否" in txt and "β可观测:是" in txt
    assert "搜索:monitoring" in txt


def test_export_writes_json_report(app, tmp_path, monkeypatch):
    tab, clock = make_tab(app)
    tab._start_observation_clicked()
    feed(tab, clock, 40, ser=0.05)
    tab.adaptive_enable_check.setChecked(True)
    feed(tab, clock, 40, ser=0.01, evm=6.0)
    out = tmp_path / "advantage_report.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(out), ""))
    tab._export_observation_clicked()
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert data["windows"] and data["toggle_events"]
    assert "anomaly_drops" in data
    assert data["latest_pair"]["before"]["ser"] is not None


def test_stop_observation_keeps_results(app):
    tab, clock = make_tab(app)
    tab._start_observation_clicked()
    feed(tab, clock, 40, ser=0.05)
    tab.adaptive_enable_check.setChecked(True)
    feed(tab, clock, 40, ser=0.01, evm=6.0)
    tab._stop_observation_clicked()
    assert not tab.observer.active
    assert tab.btn_start_observation.isEnabled()
    assert tab.btn_export_observation.isEnabled()
    tab._update_observation_display()
    assert "实测优势" in tab.observation_improvement_label.text()  # 结果保留可查看


def test_power_mismatch_can_never_render_green_advantage(app):
    tab, clock = make_tab(app)
    tab._start_observation_clicked()
    feed(tab, clock, 40, ser=0.05, contract_id="baseline", tx_rms=0.20)
    tab.adaptive_enable_check.setChecked(True)
    feed(tab, clock, 40, ser=0.01, evm=6.0, contract_id="candidate", tx_rms=0.10)
    tab._update_observation_display()
    pair = tab.observer.latest_pair()
    assert pair is not None and not pair.comparable
    assert pair.outcome == "inconclusive"
    assert "无结论" in tab.observation_improvement_label.text()
    assert "#167c3a" not in tab.observation_improvement_label.styleSheet()
    assert "功率不可比" in tab.observation_note_label.text()


def test_regression_is_red_and_explicitly_reports_rollback(app):
    tab, clock = make_tab(app)
    tab._start_observation_clicked()
    feed(tab, clock, 40, ser=0.01, evm=5.0)
    tab.adaptive_enable_check.setChecked(True)
    feed(tab, clock, 40, ser=0.05, evm=9.0)
    tab._update_observation_display()
    assert tab.observer.latest_pair().outcome == "regressed"
    assert "实测退化" in tab.observation_improvement_label.text()
    assert "已回滚" in tab.observation_improvement_label.text()
    assert "#a33b3b" in tab.observation_improvement_label.styleSheet()


def test_decode_status_uses_explicit_evidence_names_and_missing_values(app):
    tab = FDIDMHardwareTestTab()
    legacy = make_status(1, 1)
    for key in ("data_aided_evm_percent", "decision_directed_evm_percent",
                "tx_cycle_rms", "tx_peak", "tx_papr_db", "tx_power_backoff_db",
                "pilot_residual_sinr_db", "pilot_fit_nmse", "frame_structure"):
        legacy.pop(key, None)
    tab._update_decode_status({"decode_ok": True}, legacy)
    detail = tab.decode_status_label.toolTip()
    assert "数据EVM=不可用" in detail
    assert "判决EVM=" in detail
    assert "TX RMS=不可用" in detail
    assert "残差SINR=不可用" in detail
    assert "数据/导频/保护=不可用" in detail


def test_injected_snr_and_frame_overhead_are_labeled_truthfully(app):
    tab = FDIDMHardwareTestTab()
    status = make_status(1, 1)
    status["tdl_injected_snr_db"] = 35.0
    tab._update_decode_status({"decode_ok": False}, status)
    assert "TDL注入设定SNR=35.0dB" in tab.decode_status_label.toolTip()
    tab._log_frame_structure_warning(status)
    assert "训练/数据=1.00" in tab._ui_log_entries[-1]
    assert "长导频本身" in tab._ui_log_entries[-1]


def test_predicted_metrics_are_visibly_separate_from_measured_metrics(app):
    tab = FDIDMHardwareTestTab()
    status = make_status(10, 8)
    status.update({
        "adaptive_alpha_beta_enabled": True,
        "adaptive_recommended_alpha": 0.75,
        "adaptive_recommended_beta": 0.8,
        "adaptive_predicted_ser_current": 0.08,
        "adaptive_predicted_ser_best": 0.04,
        "adaptive_predicted_improvement_db": 3.01,
        "adaptive_predicted_snr_db": 17.0,
        "adaptive_stable_count": 2,
        "adaptive_stable_required": 2,
        "adaptive_htf_source": "diag_tf",
    })
    tab._handle_alpha_beta_adaptation(status)
    adaptive = tab.adaptive_status_label.text()
    assert "预测SER=0.08→0.04" in adaptive
    assert "预测模型SNR=17.0dB" in adaptive
    tab._update_decode_status({"decode_ok": False}, status)
    measured = tab.decode_status_label.toolTip()
    assert "SER=0.05" in measured
    assert "TDL注入设定SNR=20.0dB" in measured
    assert "残差SINR=18.00dB" in measured


@pytest.mark.parametrize("width,height", [(1400, 800), (1400, 900), (1200, 780)])
def test_advantage_grid_four_cells_remain_balanced_with_long_text(app, width, height):
    tab = FDIDMHardwareTestTab()
    tab.resize(width, height)
    tab.plot_tabs.setCurrentIndex(1)
    tab.observation_stats_label.setText(("很长的证据说明：精确错误计数、Wilson区间、功率合同、"
                                         "数据辅助EVM和残差诊断。") * 8)
    tab.show()
    app.processEvents()
    try:
        cells = [tab.timeline_ser_cell, tab.timeline_evm_cell,
                 tab.timeline_ab_cell, tab.observation_stats_cell]
        widths = [cell.width() for cell in cells]
        heights = [cell.height() for cell in cells]
        assert min(widths) > 0 and max(widths) / min(widths) <= 1.08
        assert min(heights) > 0 and max(heights) / min(heights) <= 1.08
        assert tab.observation_stats_group.parent() is tab.observation_stats_cell
    finally:
        tab.close()
