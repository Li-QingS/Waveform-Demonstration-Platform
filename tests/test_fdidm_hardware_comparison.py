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


def make_status(frames, ok, ser=0.05, evm=8.0, overflow=0, alpha=0.5, beta=1.0):
    return {
        "frames_processed": frames,
        "frames_decode_ok": ok,
        "rx_overflow_count": overflow,
        "measured_ser": ser,
        "fec_bit_ber": ser / 5,
        "evm_average_percent": evm,
        "evm_percent": evm,
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


def feed(tab, clock, n, ser, evm=8.0, overflow_step=0, alpha=0.5, beta=1.0):
    frames = tab.observer._last_status_frames
    ok = tab.observer._last_status_ok
    overflow = tab.observer._last_status_overflow
    for _ in range(n):
        clock.advance()
        frames += 2
        ok += 2
        overflow += overflow_step
        status = make_status(frames, ok, ser=ser, evm=evm, overflow=overflow,
                             alpha=alpha, beta=beta)
        tab.observer.on_sample(status)
        tab._append_timeline_sample(status)
    return frames, ok, overflow


def test_observation_controls_default_state(app):
    tab = FDIDMHardwareTestTab()
    assert tab.btn_start_observation.isEnabled()
    assert not tab.btn_stop_observation.isEnabled()
    assert not tab.btn_export_observation.isEnabled()
    # 旧定时对比控件已彻底删除（F7）
    assert not hasattr(tab, "btn_start_comparison")
    assert not hasattr(tab, "comparison_session")
    assert "未开始" in tab.observation_state_label.text()


def test_observation_pair_updates_panel_and_timeline(app):
    tab, clock = make_tab(app)
    tab._start_observation_clicked()
    feed(tab, clock, 40, ser=0.05, evm=8.0)
    tab.adaptive_enable_check.setChecked(True)  # 评审手动开关 → 事件
    feed(tab, clock, 40, ser=0.01, evm=6.0, alpha=0.75, beta=0.8)
    tab._update_observation_display()

    pair = tab.observer.latest_pair()
    assert pair is not None and pair.comparable
    assert "dB" in tab.observation_improvement_label.text()
    assert "可信" in tab.observation_badge_label.text()
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
    assert data["schema_version"] == 1
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
    assert "dB" in tab.observation_improvement_label.text()  # 结果保留可查看
