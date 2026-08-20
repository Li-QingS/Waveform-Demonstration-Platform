# -*- coding: utf-8 -*-
"""FDIDM simulation adaptive-process UI pieces.

The adaptive closed loop lives in the FDIDM simulation backend.  The UI is
split into two lightweight pieces:

  * AdaptiveControlBox  - left-panel controls (enable / params / evaluate)
  * AdaptiveProcessPlots - right-panel "自适应过程" tab (trajectories, SER
    comparison, switch markers, status text)
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
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
MATLAB_GREEN = (0, 150, 80)
MATLAB_RED = (162, 20, 47)
MATLAB_PURPLE = (126, 47, 142)


class AdaptiveControlBox(QGroupBox):
    """Left-panel adaptive controls; only forwards config to the backend."""

    config_changed = pyqtSignal()
    evaluate_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("自适应", parent)
        form = QFormLayout(self)
        form.setContentsMargins(8, 8, 8, 8)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.enable_check = QCheckBox("启用自适应")
        self.auto_apply_check = QCheckBox("自动应用")
        self.auto_apply_check.setChecked(True)
        form.addRow(self.enable_check, self.auto_apply_check)

        self.eval_interval_spin = QSpinBox()
        self.eval_interval_spin.setRange(1, 1024)
        self.eval_interval_spin.setValue(8)
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
        self.stability_spin.setValue(2)
        self.min_gain_spin = QDoubleSpinBox()
        self.min_gain_spin.setRange(0.0, 30.0)
        self.min_gain_spin.setDecimals(2)
        self.min_gain_spin.setSingleStep(0.1)
        self.min_gain_spin.setValue(0.5)
        form.addRow("稳定/增益", self._row(self.stability_spin, self.min_gain_spin))

        self.cooldown_spin = QSpinBox()
        self.cooldown_spin.setRange(0, 4096)
        self.cooldown_spin.setValue(20)
        form.addRow("冷却帧", self.cooldown_spin)

        self.evaluate_btn = QPushButton("立即评估")
        form.addRow(self.evaluate_btn)

        self.enable_check.stateChanged.connect(self.config_changed.emit)
        self.auto_apply_check.stateChanged.connect(self.config_changed.emit)
        for w in (self.eval_interval_spin, self.coarse_spin, self.fine_spin,
                  self.stability_spin, self.min_gain_spin, self.cooldown_spin):
            w.valueChanged.connect(self.config_changed.emit)
        self.evaluate_btn.clicked.connect(self.evaluate_requested.emit)

    @staticmethod
    def _row(w1, w2) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(w1)
        row.addWidget(w2)
        return row

    def collect_config(self) -> Dict[str, Any]:
        """Return current panel configuration for backend start/update."""
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
    """Right-panel adaptive analysis view: trajectories + SER + status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        self.traj_plot = pg.PlotWidget(title="α/β 轨迹（实线=已应用，虚线=推荐，圆点=切换）")
        self.traj_plot.setBackground("w")
        self.traj_plot.showGrid(x=True, y=True, alpha=0.22)
        self.traj_plot.setXRange(0, 100, padding=0.02)
        self.traj_plot.setYRange(0, 2, padding=0.02)
        self.traj_plot.addLegend(offset=(8, 8))
        self.applied_alpha_curve = self.traj_plot.plot(pen=pg.mkPen(MATLAB_BLUE, width=1.6),
                                                       name="α 已应用", stepMode="center")
        self.applied_beta_curve = self.traj_plot.plot(
            pen=pg.mkPen(MATLAB_ORANGE, width=1.6, style=Qt.DashDotLine),
            name="β 已应用", stepMode="center")
        self.rec_alpha_curve = self.traj_plot.plot(pen=pg.mkPen(MATLAB_BLUE, width=1.2,
                                                                 style=Qt.DashLine),
                                                   name="α 推荐")
        self.rec_beta_curve = self.traj_plot.plot(pen=pg.mkPen(MATLAB_ORANGE, width=1.2,
                                                                style=Qt.DotLine),
                                                  name="β 推荐")
        self.switch_alpha_marker = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(*MATLAB_RED),
                                                      pen=pg.mkPen(None))
        self.switch_beta_marker = pg.ScatterPlotItem(size=10, symbol="t", brush=pg.mkBrush(*MATLAB_PURPLE),
                                                     pen=pg.mkPen(None))
        self.traj_plot.addItem(self.switch_alpha_marker)
        self.traj_plot.addItem(self.switch_beta_marker)

        self.ser_plot = pg.PlotWidget(title="预测 SER 对比（当前 vs 推荐，log）")
        self.ser_plot.setBackground("w")
        self.ser_plot.showGrid(x=True, y=True, alpha=0.22)
        self.ser_plot.setLogMode(y=True)
        self.ser_plot.setLabel("bottom", "帧号")
        self.ser_plot.setLabel("left", "预测 SER")
        self.ser_plot.addLegend(offset=(8, 8))
        self.ser_current_curve = self.ser_plot.plot(pen=pg.mkPen(MATLAB_BLUE, width=1.6),
                                                    name="当前参数")
        self.ser_best_curve = self.ser_plot.plot(pen=pg.mkPen(MATLAB_RED, width=1.6),
                                                 name="推荐参数")

        self.status_label = QLabel("自适应：关闭")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #444444;")

        root.addWidget(self.traj_plot, stretch=3)
        root.addWidget(self.ser_plot, stretch=2)
        root.addWidget(self.status_label)

        self._last_render_key: tuple = (0, 0)

    @staticmethod
    def _finite(values) -> List[float]:
        return [float(v) for v in values if np.isfinite(float(v))]

    @staticmethod
    def _step_pairs(evals: List[Dict[str, Any]], key: str):
        """Return step (x, y) arrays; pyqtgraph stepMode needs len(x)=len(y)+1."""
        out = [(int(h.get("frame", 0)), float(h.get(key, float("nan")))) for h in evals]
        out = [(x, y) for x, y in out if np.isfinite(y)]
        if not out:
            return [], []
        xs = [p[0] for p in out]
        ys = [p[1] for p in out]
        return xs + [xs[-1]], ys

    def refresh(self, status: Dict[str, Any], history: List[Dict[str, Any]]) -> None:
        """Redraw plots/status from backend status + history when data changed."""
        status = dict(status or {})
        history = list(history or [])
        render_key = (int(status.get("recommendation_seq", 0)), len(history))
        if render_key == self._last_render_key:
            return
        self._last_render_key = render_key

        evals = [h for h in history if h.get("kind") == "eval"]
        switches = [h for h in history if h.get("kind") == "switch"]
        x = [int(h.get("frame", 0)) for h in evals]

        if x:
            xa, applied_a = self._step_pairs(evals, "alpha")
            xb, applied_b = self._step_pairs(evals, "beta")
            rec_a = self._finite(h.get("rec_alpha", float("nan")) for h in evals)
            rec_b = self._finite(h.get("rec_beta", float("nan")) for h in evals)
            ser_cur = self._finite(h.get("ser_current", float("nan")) for h in evals)
            ser_best = self._finite(h.get("ser_best", float("nan")) for h in evals)
            self.applied_alpha_curve.setData(xa, applied_a)
            self.applied_beta_curve.setData(xb, applied_b)
            self.rec_alpha_curve.setData(x, rec_a)
            self.rec_beta_curve.setData(x, rec_b)
            self.ser_current_curve.setData(x, ser_cur)
            self.ser_best_curve.setData(x, ser_best)
            lo = max(0, int(min(x)) - 5)
            hi = int(max(x)) + 5
            self.traj_plot.setXRange(lo, hi, padding=0)
            self.ser_plot.setXRange(lo, hi, padding=0)
        else:
            for c in (self.applied_alpha_curve, self.applied_beta_curve,
                      self.rec_alpha_curve, self.rec_beta_curve,
                      self.ser_current_curve, self.ser_best_curve):
                c.setData([], [])

        sw_a_x = [int(s.get("frame", 0)) for s in switches]
        sw_a_y = [float(s.get("to_alpha", float("nan"))) for s in switches]
        sw_b_x = [int(s.get("frame", 0)) for s in switches]
        sw_b_y = [float(s.get("to_beta", float("nan"))) for s in switches]
        self.switch_alpha_marker.setData(sw_a_x, sw_a_y)
        self.switch_beta_marker.setData(sw_b_x, sw_b_y)

        enabled = bool(status.get("enabled", False))
        state = str(status.get("state", "disabled"))
        cur_a = status.get("current_alpha")
        cur_b = status.get("current_beta")
        rec_a = status.get("recommended_alpha")
        rec_b = status.get("recommended_beta")
        gain = status.get("predicted_improvement_db")
        stable = int(status.get("stable_count", 0))
        required = int(status.get("stable_required", 0))
        err = str(status.get("last_error", "") or "")
        line1 = "自适应：开启  状态：" + str(state) if enabled else "自适应：关闭  状态：" + str(state)
        lines = []
        if cur_a is not None and cur_b is not None:
            try:
                lines.append("当前 α=%.3f，β=%.3f" % (float(cur_a), float(cur_b)))
            except Exception:
                pass
        if rec_a is not None and rec_b is not None:
            try:
                lines.append("推荐 α=%.3f，β=%.3f，预测增益=%.3f dB，稳定 %d/%d"
                             % (float(rec_a), float(rec_b), float(gain), int(stable), int(required)))
            except Exception:
                pass
        if err:
            lines.append("错误：" + str(err))
        self.status_label.setText(line1 + (("\n" + "\n".join(lines)) if lines else ""))
