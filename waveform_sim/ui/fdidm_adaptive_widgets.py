# -*- coding: utf-8 -*-
"""FDIDM dual-timescale adaptive controls and visualization."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

MATLAB_BLUE = (0, 114, 189)
MATLAB_ORANGE = (217, 83, 25)
MATLAB_GREEN = (119, 172, 48)
MATLAB_PURPLE = (126, 47, 142)
MATLAB_RED = (162, 20, 47)


class AdaptiveControlBox(QGroupBox):
    """Controls for the slow alpha/beta controller.

    Frame-by-frame CSI/equalizer tracking stays in the fast link path.  These
    controls only determine how a *window* of CSI snapshots is summarized and
    when the slower alpha/beta optimizer is allowed to switch waveform state.
    """

    config_changed = pyqtSignal()
    evaluate_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("自适应参数（慢时间尺度）", parent)
        form = QFormLayout(self)
        form.setContentsMargins(8, 8, 8, 8)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.enable_check = QCheckBox("启用慢时标自适应")
        self.enable_check.setChecked(True)
        self.auto_apply_check = QCheckBox("稳定后自动应用")
        self.auto_apply_check.setChecked(True)
        form.addRow(self.enable_check, self.auto_apply_check)

        self.window_frames_spin = QSpinBox()
        self.window_frames_spin.setRange(4, 4096)
        self.window_frames_spin.setValue(32)
        self.window_frames_spin.setToolTip("构造慢时标CSI统计量所使用的最近帧数。")

        self.ensemble_spin = QSpinBox()
        self.ensemble_spin.setRange(2, 32)
        self.ensemble_spin.setValue(3)
        self.ensemble_spin.setToolTip("从CSI窗口均匀抽取多少个代表性信道快照进行联合优化。")
        form.addRow("CSI窗口/快照", self._row(self.window_frames_spin, self.ensemble_spin))

        self.eval_interval_spin = QSpinBox()
        self.eval_interval_spin.setRange(4, 4096)
        self.eval_interval_spin.setValue(16)
        self.eval_interval_spin.setToolTip("每隔多少个仿真帧触发一次慢时标参数搜索。")

        self.benchmark_interval_spin = QSpinBox()
        self.benchmark_interval_spin.setRange(1, 256)
        self.benchmark_interval_spin.setValue(4)
        self.benchmark_interval_spin.setToolTip("每隔多少帧计算一次四波形共享信道性能样本。")
        form.addRow("优化/采样间隔", self._row(self.eval_interval_spin, self.benchmark_interval_spin))

        self.display_interval_spin = QDoubleSpinBox()
        self.display_interval_spin.setRange(0.1, 5.0)
        self.display_interval_spin.setDecimals(1)
        self.display_interval_spin.setSingleStep(0.1)
        self.display_interval_spin.setValue(0.5)
        self.display_interval_spin.setSuffix(" s")
        self.display_interval_spin.setToolTip("时间图最多按此周期增加一个显示点；不会降低后台仿真速率。")
        form.addRow("图形显示周期", self.display_interval_spin)

        self.coarse_spin = QDoubleSpinBox()
        self.coarse_spin.setRange(0.1, 1.0)
        self.coarse_spin.setDecimals(2)
        self.coarse_spin.setSingleStep(0.05)
        self.coarse_spin.setValue(0.50)

        self.fine_spin = QDoubleSpinBox()
        self.fine_spin.setRange(0.05, 0.5)
        self.fine_spin.setDecimals(2)
        self.fine_spin.setSingleStep(0.05)
        self.fine_spin.setValue(0.10)
        form.addRow("粗/细搜索步长", self._row(self.coarse_spin, self.fine_spin))

        self.stability_spin = QSpinBox()
        self.stability_spin.setRange(1, 16)
        self.stability_spin.setValue(2)
        self.stability_spin.setToolTip("连续多少次得到同一推荐区域后，参数才允许切换。")

        self.min_gain_spin = QDoubleSpinBox()
        self.min_gain_spin.setRange(0.0, 30.0)
        self.min_gain_spin.setDecimals(2)
        self.min_gain_spin.setSingleStep(0.1)
        self.min_gain_spin.setValue(0.20)
        self.min_gain_spin.setSuffix(" dB")
        self.min_gain_spin.setToolTip("推荐参数相对当前参数的预测增益至少达到此值。")
        form.addRow("稳定次数/最小增益", self._row(self.stability_spin, self.min_gain_spin))

        self.cooldown_spin = QSpinBox()
        self.cooldown_spin.setRange(0, 4096)
        self.cooldown_spin.setValue(32)
        self.cooldown_spin.setToolTip("一次真实切换后至少保持多少帧，防止参数来回振荡。")
        form.addRow("切换冷却帧", self.cooldown_spin)

        self.evaluate_btn = QPushButton("立即进行慢时标评估")
        self.evaluate_btn.setToolTip(
            "立即使用当前已积累的CSI窗口执行一次联合α/β搜索；"
            "不是拿单帧瞬时信道追踪快速衰落。"
        )
        form.addRow(self.evaluate_btn)

        note = QLabel("快速层逐帧更新信道/均衡器；本模块只跟踪信道统计结构并低频切换α/β。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#666666;font-size:9pt;")
        form.addRow(note)

        self.enable_check.stateChanged.connect(self.config_changed.emit)
        self.auto_apply_check.stateChanged.connect(self.config_changed.emit)
        for widget in (
            self.window_frames_spin,
            self.ensemble_spin,
            self.eval_interval_spin,
            self.benchmark_interval_spin,
            self.display_interval_spin,
            self.coarse_spin,
            self.fine_spin,
            self.stability_spin,
            self.min_gain_spin,
            self.cooldown_spin,
        ):
            widget.valueChanged.connect(self.config_changed.emit)
        self.evaluate_btn.clicked.connect(self.evaluate_requested.emit)

    @staticmethod
    def _row(w1, w2):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(w1)
        row.addWidget(w2)
        return row

    def collect_config(self) -> Dict[str, Any]:
        return {
            "adaptive_enabled": self.enable_check.isChecked(),
            "adaptive_auto_apply": self.auto_apply_check.isChecked(),
            "adaptive_window_frames": int(self.window_frames_spin.value()),
            "adaptive_ensemble_snapshots": int(self.ensemble_spin.value()),
            "adaptive_interval_frames": int(self.eval_interval_spin.value()),
            "adaptive_benchmark_interval_frames": int(self.benchmark_interval_spin.value()),
            "adaptive_display_interval_s": float(self.display_interval_spin.value()),
            "adaptive_display_ema_alpha": 0.28,
            "adaptive_coarse_step": float(self.coarse_spin.value()),
            "adaptive_fine_step": float(self.fine_spin.value()),
            "adaptive_stability_evals": int(self.stability_spin.value()),
            "adaptive_min_improvement_db": float(self.min_gain_spin.value()),
            "adaptive_cooldown_frames": int(self.cooldown_spin.value()),
        }


class AdaptiveProcessPlots(QWidget):
    """Shared-channel SNR-SER plot, link state and adaptive narration."""

    COLORS = {
        "OFDM": MATLAB_BLUE,
        "OTFS": MATLAB_ORANGE,
        "AFDM": MATLAB_GREEN,
        "FDIDM": MATLAB_PURPLE,
    }
    SYMBOLS = {"OFDM": "o", "OTFS": "s", "AFDM": "t", "FDIDM": "d"}

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        self.snr_plot = pg.PlotWidget(title="共享信道四波形稀疏LMMSE SER-SNR 对比")
        self.snr_plot.setBackground("w")
        self.snr_plot.showGrid(x=True, y=True, alpha=0.22)
        self.snr_plot.setLabel("bottom", "SNR", units="dB")
        self.snr_plot.setLabel("left", "SER")
        self.snr_plot.setLogMode(y=True)
        self.snr_plot.setXRange(0.0, 30.0, padding=0.02)
        self.snr_plot.addLegend(offset=(8, 8))

        self._snr_curves: Dict[str, Any] = {}
        # name -> SNR -> (SER, alpha, beta)
        self._snr_data: Dict[str, Dict[float, tuple]] = {}
        self._fdidm_labels: Dict[float, Any] = {}

        self.link_state_label = QLabel("实时链路状态：等待仿真数据")
        self.link_state_label.setWordWrap(True)
        self.link_state_label.setTextFormat(Qt.RichText)
        self.link_state_label.setStyleSheet("color:#333333;")

        self.narration_label = QLabel(
            "演示说明：快速层逐帧跟踪，慢速层基于CSI窗口进行稳定判决与参数切换。"
        )
        self.narration_label.setWordWrap(True)
        self.narration_label.setStyleSheet("color:#444444;")

        root.addWidget(self.snr_plot, stretch=5)
        root.addWidget(self.link_state_label, stretch=2)
        root.addWidget(self.narration_label, stretch=1)
        self._last_render_key = None

    def set_snr_title(self, title: str) -> None:
        self.snr_plot.setTitle(str(title))

    def clear_snr(self) -> None:
        self.snr_plot.clear()
        self.snr_plot.addLegend(offset=(8, 8))
        self.snr_plot.setXRange(0.0, 30.0, padding=0.02)
        self._snr_curves.clear()
        self._snr_data.clear()
        self._fdidm_labels.clear()

    def _remove_fdidm_label(self, snr: float) -> None:
        old = self._fdidm_labels.pop(float(snr), None)
        if old is not None:
            try:
                self.snr_plot.removeItem(old)
            except Exception:
                pass

    def add_snr_point(
        self,
        name: str,
        snr: float,
        ser: float,
        alpha: float,
        beta: float,
    ) -> None:
        """Add/replace one point; FDIDM labels are anchored beside their points."""
        name = str(name)
        snr = float(snr)
        ser = float(ser)
        alpha = float(alpha)
        beta = float(beta)

        if name not in self._snr_curves:
            color = self.COLORS.get(name, MATLAB_RED)
            pen = pg.mkPen(color, width=2.6 if name == "FDIDM" else 1.7)
            curve = self.snr_plot.plot(
                [], [], pen=pen, symbol=self.SYMBOLS.get(name, "o"),
                symbolSize=7, symbolPen=pen, symbolBrush="w", name=name,
            )
            self._snr_curves[name] = curve
            self._snr_data[name] = {}

        self._snr_data[name][snr] = (ser, alpha, beta)
        ordered = sorted(self._snr_data[name].items(), key=lambda item: item[0])
        xx = np.asarray([item[0] for item in ordered], dtype=float)
        yy = np.asarray([
            item[1][0] if np.isfinite(item[1][0]) and item[1][0] > 0 else np.nan
            for item in ordered
        ], dtype=float)
        self._snr_curves[name].setData(xx, yy)

        if name != "FDIDM":
            return
        self._remove_fdidm_label(snr)
        if not np.isfinite(ser) or ser <= 0:
            return

        # PlotDataItem applies log10 internally, TextItem does not.
        point_index = int(round(snr / 2.0))
        put_right = point_index % 2 == 0
        x_offset = 0.20 if put_right else -0.20
        y_offset = 0.05 if (point_index // 2) % 2 == 0 else -0.05
        anchor = (0.0, 0.5) if put_right else (1.0, 0.5)
        label = pg.TextItem(
            text=f"({alpha:g}, {beta:g})",
            anchor=anchor,
            color=MATLAB_PURPLE,
        )
        label.setFont(QFont("Microsoft YaHei", 8))
        label.setPos(snr + x_offset, float(np.log10(max(ser, 1e-300))) + y_offset)
        self.snr_plot.addItem(label)
        self._fdidm_labels[snr] = label

    @staticmethod
    def _fmt(value, suffix="", digits=3):
        try:
            number = float(value)
        except Exception:
            return "--"
        if not np.isfinite(number):
            return "--"
        if abs(number) >= 1e4 or (0 < abs(number) < 1e-3):
            return f"{number:.2e}{suffix}"
        return f"{number:.{digits}f}{suffix}"

    @staticmethod
    def _action_text(action: str) -> str:
        return {
            "apply": "满足条件，准备自动应用",
            "stable_pending": "推荐尚未连续稳定",
            "gain_below": "增益未达到门限",
            "cooldown": "仍在切换冷却期",
            "keep": "最优区域与当前参数一致",
            "ready_manual": "已就绪，等待手动应用",
            "stale": "搜索期间参数已变化，结果已丢弃",
        }.get(str(action), str(action or "--"))

    def refresh(
        self,
        status: Dict[str, Any],
        history: List[Dict[str, Any]],
        link_state: Dict[str, Any] | None = None,
    ) -> None:
        status = dict(status or {})
        history = list(history or [])
        link_state = dict(link_state or {})
        key = (
            int(status.get("recommendation_seq", 0)),
            len(history),
            tuple(sorted((str(k), str(v)) for k, v in link_state.items())),
        )
        if key == self._last_render_key:
            return
        self._last_render_key = key

        coherence_ms = float(link_state.get("coherence_time_s", np.nan)) * 1e3
        compensation_pct = float(link_state.get("doppler_compensation_ratio", np.nan)) * 100.0
        self.link_state_label.setText(
            "<b>实时链路状态</b><br>"
            f"SNR：配置 {self._fmt(link_state.get('configured_snr_db'), ' dB')}；"
            f"有效 {self._fmt(link_state.get('effective_snr_db'), ' dB')}<br>"
            f"轨道多普勒：预测公共频移 {self._fmt(link_state.get('absolute_doppler_shift_hz'), ' Hz')}；"
            f"补偿 {self._fmt(compensation_pct, ' %', 2)}；"
            f"残余公共CFO {self._fmt(link_state.get('residual_common_cfo_hz'), ' Hz')}<br>"
            f"小尺度时变：配置最大扩展 {self._fmt(link_state.get('residual_doppler_spread_hz'), ' Hz')}；"
            f"当前RMS {self._fmt(link_state.get('doppler_spread_hz'), ' Hz')}；"
            f"相干时间≈{self._fmt(coherence_ms, ' ms')}<br>"
            f"时延：均值 {self._fmt(link_state.get('delay_mean_ns'), ' ns')}；"
            f"RMS扩展 {self._fmt(link_state.get('delay_spread_ns'), ' ns')}；"
            f"最大 {self._fmt(link_state.get('max_delay_ns'), ' ns')}<br>"
            f"资源：带宽 {self._fmt(float(link_state.get('bandwidth_hz', np.nan))/1e6, ' MHz')}；"
            f"采样率 {self._fmt(float(link_state.get('sample_rate_hz', np.nan))/1e6, ' Msps')}；"
            f"物理帧长 {self._fmt(float(link_state.get('physical_frame_duration_s', np.nan))*1e6, ' μs')}；"
            f"归一化Doppler {self._fmt(link_state.get('normalized_doppler'))}<br>"
            f"噪声：目标 {self._fmt(link_state.get('noise_power'))}；"
            f"实测 {self._fmt(link_state.get('measured_noise_power'))}；"
            f"均衡后 {self._fmt(link_state.get('eq_noise_power'))}<br>"
            f"质量：EVM {self._fmt(link_state.get('evm_percent'), ' %')}；"
            f"cond(H) {self._fmt(link_state.get('condition_number'))}<br>"
            f"信道变化：H功率 {self._fmt(link_state.get('channel_power_db'), ' dB')}；"
            f"相邻帧相关 {self._fmt(link_state.get('channel_matrix_correlation'))}；"
            f"相对变化 {self._fmt(link_state.get('channel_matrix_change_norm'))}<br>"
            f"信道：{link_state.get('channel_type', '--')} / {link_state.get('channel_mode', '--')}；"
            f"frame={link_state.get('frame', '--')}；"
            f"窗口={status.get('window_fill', 0)}/{status.get('window_frames', '--')}；"
            f"α/β=({self._fmt(link_state.get('alpha'), digits=2)}, "
            f"{self._fmt(link_state.get('beta'), digits=2)})"
        )

        switches = [item for item in history if item.get("kind") == "switch"]
        evals = [item for item in history if item.get("kind") == "eval"]
        lines = []
        if switches:
            sw = switches[-1]
            lines.append(
                "最近一次真实自动切换："
                f"({float(sw.get('from_alpha', 0.0)):g}, {float(sw.get('from_beta', 0.0)):g}) → "
                f"({float(sw.get('to_alpha', 0.0)):g}, {float(sw.get('to_beta', 0.0)):g})，"
                f"基于{int(sw.get('window_frames', 0))}个代表快照，预测增益 "
                f"{float(sw.get('gain_db', 0.0)):.2f} dB。"
            )
        elif evals:
            ev = evals[-1]
            lines.append(
                f"慢时标评估：当前=({float(ev.get('alpha', 0.0)):g}, {float(ev.get('beta', 0.0)):g})；"
                f"推荐=({float(ev.get('rec_alpha', 0.0)):g}, {float(ev.get('rec_beta', 0.0)):g})；"
                f"当前SER={float(ev.get('ser_current', np.nan)):.3e}；"
                f"推荐SER={float(ev.get('ser_best', np.nan)):.3e}；"
                f"增益={float(ev.get('gain_db', 0.0)):.2f} dB；"
                f"稳定={int(ev.get('stable_count', 0))}/{int(ev.get('stable_required', 0))}；"
                f"{self._action_text(ev.get('action', '--'))}。"
            )
        else:
            lines.append("正在积累CSI窗口；不会用单帧瞬时衰落直接切换α/β。")

        if status.get("last_error"):
            lines.append("错误：" + str(status["last_error"]))
        self.narration_label.setText("演示说明：" + " ".join(lines))
