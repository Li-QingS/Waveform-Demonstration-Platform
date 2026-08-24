# -*- coding: utf-8 -*-
"""FDIDM adaptive controls and adaptive-process visualization."""
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
    config_changed = pyqtSignal()
    evaluate_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("自适应参数", parent)
        form = QFormLayout(self)
        form.setContentsMargins(8, 8, 8, 8)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.enable_check = QCheckBox("启用自适应")
        self.enable_check.setChecked(True)
        self.auto_apply_check = QCheckBox("自动应用")
        self.auto_apply_check.setChecked(True)
        form.addRow(self.enable_check, self.auto_apply_check)

        self.eval_interval_spin = QSpinBox()
        self.eval_interval_spin.setRange(1, 1024)
        self.eval_interval_spin.setValue(4)
        form.addRow("评估间隔/帧", self.eval_interval_spin)

        self.coarse_spin = QDoubleSpinBox()
        self.coarse_spin.setRange(0.01, 1.0)
        self.coarse_spin.setDecimals(2)
        self.coarse_spin.setSingleStep(0.05)
        self.coarse_spin.setValue(0.25)

        self.fine_spin = QDoubleSpinBox()
        self.fine_spin.setRange(0.005, 0.5)
        self.fine_spin.setDecimals(3)
        self.fine_spin.setSingleStep(0.005)
        self.fine_spin.setValue(0.05)
        form.addRow("粗/细步长", self._row(self.coarse_spin, self.fine_spin))

        self.stability_spin = QSpinBox()
        self.stability_spin.setRange(1, 16)
        self.stability_spin.setValue(1)

        self.min_gain_spin = QDoubleSpinBox()
        self.min_gain_spin.setRange(0.0, 30.0)
        self.min_gain_spin.setDecimals(2)
        self.min_gain_spin.setSingleStep(0.05)
        self.min_gain_spin.setValue(0.15)
        form.addRow("稳定/增益", self._row(self.stability_spin, self.min_gain_spin))

        self.cooldown_spin = QSpinBox()
        self.cooldown_spin.setRange(0, 4096)
        self.cooldown_spin.setValue(6)
        form.addRow("冷却帧", self.cooldown_spin)

        self.evaluate_btn = QPushButton("立即搜索最优参数")
        self.evaluate_btn.setToolTip(
            "不等待下一个评估周期，立即基于当前CSI和SNR搜索一次最优α/β。"
        )
        form.addRow(self.evaluate_btn)

        self.enable_check.stateChanged.connect(self.config_changed.emit)
        self.auto_apply_check.stateChanged.connect(self.config_changed.emit)
        for widget in (
            self.eval_interval_spin,
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
            "adaptive_interval_frames": int(self.eval_interval_spin.value()),
            "adaptive_coarse_step": float(self.coarse_spin.value()),
            "adaptive_fine_step": float(self.fine_spin.value()),
            "adaptive_stability_evals": int(self.stability_spin.value()),
            "adaptive_min_improvement_db": float(self.min_gain_spin.value()),
            "adaptive_cooldown_frames": int(self.cooldown_spin.value()),
        }


class AdaptiveProcessPlots(QWidget):
    """Exact-point SNR-SER plot plus current state and event narration."""

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

        self.snr_plot = pg.PlotWidget(title="SER-SNR 性能对比")
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
            "演示说明：启动后将解释当前参数、推荐参数和真实自动切换事件。"
        )
        self.narration_label.setWordWrap(True)
        self.narration_label.setStyleSheet("color:#444444;")

        root.addWidget(self.snr_plot, stretch=5)
        root.addWidget(self.link_state_label, stretch=1)
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
        """Add/replace one point and attach alpha/beta to its FDIDM marker."""
        name = str(name)
        snr = float(snr)
        ser = float(ser)
        alpha = float(alpha)
        beta = float(beta)

        if name not in self._snr_curves:
            color = self.COLORS.get(name, MATLAB_RED)
            pen = pg.mkPen(color, width=2.6 if name == "FDIDM" else 1.7)
            curve = self.snr_plot.plot(
                [],
                [],
                pen=pen,
                symbol=self.SYMBOLS.get(name, "o"),
                symbolSize=7,
                symbolPen=pen,
                symbolBrush="w",
                name=name,
            )
            self._snr_curves[name] = curve
            self._snr_data[name] = {}

        self._snr_data[name][snr] = (ser, alpha, beta)
        ordered = sorted(self._snr_data[name].items(), key=lambda item: item[0])
        xx = np.asarray([item[0] for item in ordered], dtype=float)
        yy = np.asarray(
            [
                item[1][0]
                if np.isfinite(item[1][0]) and item[1][0] > 0
                else np.nan
                for item in ordered
            ],
            dtype=float,
        )
        self._snr_curves[name].setData(xx, yy)

        if name != "FDIDM":
            return

        self._remove_fdidm_label(snr)
        if not np.isfinite(ser) or ser <= 0:
            return

        # PlotDataItem applies log10 internally, TextItem does not.  Therefore
        # its y coordinate must be log10(SER).  Alternate left/right offsets so
        # each label stays beside, rather than above, the corresponding point.
        point_index = int(round(snr / 2.0))
        put_right = point_index % 2 == 0
        x_offset = 0.22 if put_right else -0.22
        y_offset = 0.055 if (point_index // 2) % 2 == 0 else -0.055
        anchor = (0.0, 0.5) if put_right else (1.0, 0.5)
        label = pg.TextItem(
            text=f"({alpha:g}, {beta:g})",
            anchor=anchor,
            color=MATLAB_PURPLE,
        )
        label.setFont(QFont("Microsoft YaHei", 8))
        label.setPos(
            snr + x_offset,
            float(np.log10(max(ser, 1e-300))) + y_offset,
        )
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

        self.link_state_label.setText(
            "<b>实时链路状态</b><br>"
            f"SNR：配置 {self._fmt(link_state.get('configured_snr_db'), ' dB')}；"
            f"有效 {self._fmt(link_state.get('effective_snr_db'), ' dB')}<br>"
            f"多普勒：均值 {self._fmt(link_state.get('doppler_mean_hz'), ' Hz')}；"
            f"RMS扩展 {self._fmt(link_state.get('doppler_spread_hz'), ' Hz')}；"
            f"最大 {self._fmt(link_state.get('max_doppler_hz'), ' Hz')}<br>"
            f"时延：均值 {self._fmt(link_state.get('delay_mean_ns'), ' ns')}；"
            f"RMS扩展 {self._fmt(link_state.get('delay_spread_ns'), ' ns')}；"
            f"最大 {self._fmt(link_state.get('max_delay_ns'), ' ns')}<br>"
            f"噪声：目标 {self._fmt(link_state.get('noise_power'))}；"
            f"实测 {self._fmt(link_state.get('measured_noise_power'))}；"
            f"均衡后 {self._fmt(link_state.get('eq_noise_power'))}<br>"
            f"质量：EVM {self._fmt(link_state.get('evm_percent'), ' %')}；"
            f"cond(H) {self._fmt(link_state.get('condition_number'))}<br>"
            f"信道：{link_state.get('channel_type', '--')} / "
            f"{link_state.get('channel_mode', '--')}；"
            f"seed={link_state.get('channel_seed', '--')}；"
            f"frame={link_state.get('frame', '--')}；"
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
                f"({float(sw.get('from_alpha', 0.0)):g}, "
                f"{float(sw.get('from_beta', 0.0)):g}) → "
                f"({float(sw.get('to_alpha', 0.0)):g}, "
                f"{float(sw.get('to_beta', 0.0)):g})，"
                f"预测增益 {float(sw.get('gain_db', 0.0)):.2f} dB。"
            )
        elif evals:
            ev = evals[-1]
            lines.append(
                f"当前评估：实际α/β=({float(ev.get('alpha', 0.0)):g}, "
                f"{float(ev.get('beta', 0.0)):g})；"
                f"推荐=({float(ev.get('rec_alpha', 0.0)):g}, "
                f"{float(ev.get('rec_beta', 0.0)):g})；"
                f"当前SER={float(ev.get('ser_current', np.nan)):.3e}；"
                f"推荐SER={float(ev.get('ser_best', np.nan)):.3e}；"
                f"预测增益 {float(ev.get('gain_db', 0.0)):.2f} dB；"
                f"动作={ev.get('action', '--')}。"
            )
        else:
            lines.append("当前尚无自适应评估记录。")

        if status.get("last_error"):
            lines.append("错误：" + str(status["last_error"]))
        self.narration_label.setText("演示说明：" + " ".join(lines))
