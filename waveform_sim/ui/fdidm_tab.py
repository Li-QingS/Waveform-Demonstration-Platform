"""FDIDM Mode-A soft waveform demonstration tab.

Goal:
    One LEO SATCOM channel -> search best fractional alpha/beta by paper-style
    ZF/MMSE-theory SER -> display the searched parameters as text -> validate with
    OFDM/OTFS/current/best theory SER-vs-SNR curves.

This page intentionally removes alpha-beta heatmap and paper-figure modes.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
)

from .base_waveform_tab import BaseWaveformTab

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))



from .fdidm_utils import _CurveSpec, alpha_ser_floor, copy_kwargs_with, merged_curve_specs

class FDIDMTab(BaseWaveformTab):
    """Mode-A FDIDM soft modulation page."""

    ser_snr_point = pyqtSignal(int, str, float, float)
    ser_snr_finished = pyqtSignal(int, str)
    search_finished = pyqtSignal(int, object)
    alpha_beta_point = pyqtSignal(int, str, float, float, float)
    alpha_beta_finished = pyqtSignal(int, str)

    # Right-bottom theory SER-vs-SNR points.
    SER_SNR_POINTS = [10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40 ]
    # Left-bottom alpha sweep: x-axis is alpha, each curve fixes one beta.
    ALPHA_SWEEP_BETAS = [round(0.2 * i, 1) for i in range(11)]
    # Right-bottom theory SER-vs-SNR uses raw positive theory values.
    # Non-positive/underflow values are not drawn on the log plot instead of
    # being clipped to an artificial display floor.

    def __init__(self):
        super().__init__("FDIDM")
        self.snr_label.setText("Eb/N0:")
        self.snr_spin.setValue(10.0)
        try:
            self.rename_info_rows({
                "同步度量": "SER指标",
                "CFO估计": "等效矩阵",
                "FFT/CP": "帧结构",
                "星座观察": "索引状态",
            })
        except Exception:
            pass

        self.tb = None
        self._best_alpha = None
        self._best_beta = None
        self._last_search_result = None
        self._auto_scan_token = 0
        self._auto_scan_reason = "初始化"
        self._next_alpha_curve_mode_override = None
        # Background refresh is split into independent workers:
        #   1) search worker: find best alpha/beta from the current H_tf,
        #   2) alpha worker: draw the left-bottom alpha-SER sweep,
        #   3) snr worker: draw the right-bottom theory SER-SNR curves.
        # This prevents a slow Monte-Carlo alpha sweep from blocking the fast
        # theory SER-SNR plot.
        self._search_thread = None
        self._alpha_thread = None
        self._snr_thread = None
        self._search_stop = threading.Event()
        self._alpha_stop = threading.Event()
        self._snr_stop = threading.Event()
        # Backward-compatible names used by older helper methods/buttons.
        self._ser_snr_thread = None
        self._ser_snr_stop = self._snr_stop
        self._search_active_token = None
        self._alpha_active_token = None
        self._snr_active_token = None
        self._last_completion_token = None
        self._refresh_context_by_token = {}
        self._ser_snr_curves = OrderedDict()
        self._ser_snr_data = {}
        self._alpha_beta_curves = OrderedDict()
        self._alpha_beta_data = {}
        self._alpha_ser_floor_by_token = {}
        self._qt_alive = True
        try:
            self.destroyed.connect(self._on_qt_destroyed)
        except Exception:
            pass

        self._add_fdidm_controls()
        self._connect_fdidm_signals()
        self._setup_plots()

        self.cfo_label.setVisible(False)
        self.cfo_spin.setVisible(False)
        self.doppler_label.setVisible(False)
        self.doppler_spin.setVisible(False)
        # 模式A中，右下角 SER-SNR 曲线不再由用户手动触发，
        # 而是在任意关键参数更新后自动重新搜索并绘制。
        self.btn_auto_ber_snr.setText("理论 SER-SNR 自动刷新已启用")
        self.btn_auto_ber_snr.hide()
        self.btn_stop_ber_snr.setText("停止理论 SER-SNR 刷新")
        self.btn_stop_ber_snr.setEnabled(False)

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._refresh_plots)
        self.ser_snr_point.connect(self._on_ser_snr_point)
        self.ser_snr_finished.connect(self._on_ser_snr_finished)
        self.search_finished.connect(self._on_search_finished)
        self.alpha_beta_point.connect(self._on_alpha_beta_point)
        self.alpha_beta_finished.connect(self._on_alpha_beta_finished)

        self._auto_refresh_timer = QTimer()
        self._auto_refresh_timer.setSingleShot(True)
        self._auto_refresh_timer.timeout.connect(self._start_auto_ser_snr_refresh)

        self._update_info_panel(running=False)
        QTimer.singleShot(400, lambda: self._schedule_auto_ser_snr("初始化"))

    # ------------------------------------------------------------------
    # UI controls
    # ------------------------------------------------------------------
    def _add_fdidm_controls(self):
        """添加 FDIDM 专用参数，并按功能分组整理左侧面板。"""

        def form_group(title):
            group = QGroupBox(title)
            form = QFormLayout(group)
            form.setContentsMargins(8, 8, 8, 8)
            form.setHorizontalSpacing(10)
            form.setVerticalSpacing(6)
            form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
            group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            return group, form

        def row2(w1, w2):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            row.addWidget(w1)
            row.addWidget(w2)
            return row

        # --- 索引与搜索 ---
        idx_group, idx_form = form_group("FDIDM索引 / 搜索")
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 2.0)
        self.alpha_spin.setDecimals(1)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(0.0)

        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.0, 2.0)
        self.beta_spin.setDecimals(1)
        self.beta_spin.setSingleStep(0.1)
        self.beta_spin.setValue(0.0)
        idx_form.addRow("α / β:", row2(self.alpha_spin, self.beta_spin))

        self.search_step_spin = QDoubleSpinBox()
        self.search_step_spin.setRange(0.1, 0.5)
        self.search_step_spin.setDecimals(1)
        self.search_step_spin.setSingleStep(0.1)
        self.search_step_spin.setValue(0.1)
        idx_form.addRow("搜索步长:", self.search_step_spin)

        self.decoder_combo = QComboBox()
        self.decoder_combo.addItems(["ZF", "MMSE", "ZF-SIC"])
        self.decoder_combo.setCurrentText("ZF")
        idx_form.addRow("检测器:", self.decoder_combo)

        # --- 帧结构与波形 ---
        frame_group, frame_form = form_group("帧结构 / 波形")
        self.snr_def_combo = QComboBox()
        self.snr_def_combo.addItems(["Eb/N0", "Es/N0"])
        self.snr_def_combo.setCurrentText("Eb/N0")
        frame_form.addRow("SNR定义:", self.snr_def_combo)

        self.m_spin = QSpinBox()
        self.m_spin.setRange(4, 16)
        self.m_spin.setSingleStep(4)
        self.m_spin.setValue(8)
        self.n_spin = QSpinBox()
        self.n_spin.setRange(4, 16)
        self.n_spin.setSingleStep(4)
        self.n_spin.setValue(8)
        frame_form.addRow("M / N:", row2(self.m_spin, self.n_spin))

        self.scs_spin = QDoubleSpinBox()
        self.scs_spin.setRange(1.0, 1000.0)
        self.scs_spin.setDecimals(1)
        self.scs_spin.setValue(300.0)
        self.scs_spin.setSuffix(" kHz")
        frame_form.addRow("子载波间隔:", self.scs_spin)

        self.fc_spin = QDoubleSpinBox()
        self.fc_spin.setRange(1.0, 80.0)
        self.fc_spin.setDecimals(1)
        self.fc_spin.setValue(20.0)
        self.fc_spin.setSuffix(" GHz")
        frame_form.addRow("载频:", self.fc_spin)

        self.mod_combo = QComboBox()
        self.mod_combo.addItems(["QPSK", "16QAM", "64QAM"])
        self.mod_combo.setCurrentText("16QAM")
        frame_form.addRow("调制:", self.mod_combo)

        # --- 信道 ---
        ch_group, ch_form = form_group("LEO信道")
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["TDL-A", "TDL-C", "TDL-D", "CDL"])
        self.channel_combo.setCurrentText("TDL-C")
        ch_form.addRow("模型:", self.channel_combo)

        self.velocity_combo = QComboBox()
        self.velocity_combo.addItems(["0", "120", "500", "28080"])
        self.velocity_combo.setCurrentText("28080")
        ch_form.addRow("速度(km/h):", self.velocity_combo)

        self.radial_factor_spin = QDoubleSpinBox()
        self.radial_factor_spin.setRange(0.0, 1.0)
        self.radial_factor_spin.setDecimals(2)
        self.radial_factor_spin.setSingleStep(0.01)
        self.radial_factor_spin.setValue(0.10)
        self.radial_factor_spin.setToolTip("用于计算最大多普勒：fDmax = v × 径向系数 × fc / c")
        ch_form.addRow("径向系数:", self.radial_factor_spin)

        self.random_channel_check = QCheckBox("随机路径相位/角度")
        self.random_channel_check.setChecked(True)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 2_147_483_647)
        self.seed_spin.setValue(42)
        ch_form.addRow("随机信道:", self.random_channel_check)
        ch_form.addRow("种子:", self.seed_spin)

        self.channel_dynamics_combo = QComboBox()
        self.channel_dynamics_combo.addItems(["固定信道", "动态块衰落", "帧内快时变"])
        self.channel_dynamics_combo.setCurrentText("固定信道")
        ch_form.addRow("时变模式:", self.channel_dynamics_combo)

        self.coherence_frames_spin = QSpinBox()
        self.coherence_frames_spin.setRange(1, 10000)
        self.coherence_frames_spin.setValue(20)
        self.fast_symbol_spin = QSpinBox()
        self.fast_symbol_spin.setRange(1, 64)
        self.fast_symbol_spin.setValue(1)
        ch_form.addRow("相干帧/符号:", row2(self.coherence_frames_spin, self.fast_symbol_spin))

        # --- 曲线统计 ---
        stat_group, stat_form = form_group("曲线统计")
        self.alpha_curve_mode_combo = QComboBox()
        self.alpha_curve_mode_combo.addItems(["ZF理论值"])
        self.alpha_curve_mode_combo.setCurrentText("ZF理论值")
        stat_form.addRow("α曲线默认:", self.alpha_curve_mode_combo)

        self.alpha_mc_frames_spin = QSpinBox()
        self.alpha_mc_frames_spin.setRange(10, 100000)
        self.alpha_mc_frames_spin.setValue(1000)
        self.alpha_mc_frames_spin.setSingleStep(100)
        stat_form.addRow("MC帧数:", self.alpha_mc_frames_spin)

        self.btn_alpha_mc_once = QPushButton("手动绘制MC α曲线")
        self.btn_alpha_mc_once.setMinimumHeight(28)
        self.btn_alpha_mc_once.setToolTip("仅下一次刷新使用Monte-Carlo实测；后续参数自动刷新仍默认使用ZF理论值。")
        stat_form.addRow("Monte-Carlo:", self.btn_alpha_mc_once)


        # --- 快捷按钮 ---
        quick_group = QGroupBox("快捷操作")
        quick_layout = QHBoxLayout(quick_group)
        quick_layout.setContentsMargins(8, 8, 8, 8)
        quick_layout.setSpacing(6)
        self.btn_search_best = QPushButton("理论搜索最优 α/β")
        self.btn_search_best.hide()
        self.btn_apply_best = QPushButton("应用最优")
        self.btn_apply_best.setEnabled(False)
        self.btn_regen_channel = QPushButton("重生成信道")
        self.btn_ofdm = QPushButton("OFDM")
        self.btn_otfs = QPushButton("OTFS")
        for btn in (self.btn_apply_best, self.btn_regen_channel, self.btn_ofdm, self.btn_otfs):
            btn.setMinimumHeight(28)
            quick_layout.addWidget(btn)

        for group in (idx_group, frame_group, ch_group, stat_group, quick_group):
            self.param_layout.addWidget(group)
    def _connect_fdidm_signals(self):
        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        try:
            self.btn_auto_ber_snr.clicked.disconnect()
        except TypeError:
            pass
        try:
            self.btn_stop_ber_snr.clicked.disconnect()
        except TypeError:
            pass
        # 手动绘图按钮已隐藏；停止按钮只用于中断正在运行的自动刷新。
        self.btn_auto_ber_snr.clicked.connect(lambda: self._schedule_auto_ser_snr("手动刷新"))
        self.btn_stop_ber_snr.clicked.connect(self._stop_ser_snr_sweep)
        self.btn_search_best.clicked.connect(lambda: self._schedule_auto_ser_snr("手动搜索"))
        self.btn_alpha_mc_once.clicked.connect(self._start_manual_alpha_mc_refresh)
        self.btn_apply_best.clicked.connect(self._apply_best_indices)
        self.btn_regen_channel.clicked.connect(self._regenerate_channel)
        self.btn_ofdm.clicked.connect(lambda: self._set_indices_ui(0.0, 0.0))
        self.btn_otfs.clicked.connect(lambda: self._set_indices_ui(1.0, 1.0))

        for widget in [
            self.alpha_spin, self.beta_spin, self.m_spin, self.n_spin,
            self.scs_spin, self.fc_spin, self.mod_combo, self.channel_combo,
            self.velocity_combo, self.radial_factor_spin,
            self.snr_def_combo, self.decoder_combo, self.random_channel_check,
            self.channel_dynamics_combo, self.coherence_frames_spin, self.fast_symbol_spin,
            self.alpha_curve_mode_combo, self.alpha_mc_frames_spin,
            self.seed_spin, self.snr_spin, self.search_step_spin,
        ]:
            try:
                widget.valueChanged.connect(self._hot_update_from_ui)
            except Exception:
                pass
            try:
                widget.currentIndexChanged.connect(self._hot_update_from_ui)
            except Exception:
                pass
            try:
                widget.stateChanged.connect(self._hot_update_from_ui)
            except Exception:
                pass

    def _setup_plots(self):
        """Use a MATLAB-like academic plotting style."""
        pg.setConfigOptions(antialias=True)
        matlab_colors = {
            "blue": (0, 114, 189),
            "orange": (217, 83, 25),
            "yellow": (237, 177, 32),
            "purple": (126, 47, 142),
            "green": (0, 150, 80),
            "cyan": (77, 190, 238),
            "red": (162, 20, 47),
        }
        axis_pen = pg.mkPen((40, 40, 40), width=1.1)
        grid_alpha = 0.22
        title_font = QFont("Times New Roman", 10)
        tick_font = QFont("Times New Roman", 9)

        for plot in [self.spectrum_plot, self.constellation_plot, self.ber_plot, self.ber_snr_plot]:
            plot.setBackground('w')
            plot.showGrid(x=True, y=True, alpha=grid_alpha)
            plot.getPlotItem().getViewBox().setBorder(pg.mkPen((210, 210, 210), width=0.8))
            plot.getPlotItem().titleLabel.item.setFont(title_font)
            for axis_name in ("left", "bottom"):
                axis = plot.getAxis(axis_name)
                axis.setPen(axis_pen)
                axis.setTextPen(axis_pen)
                axis.setStyle(tickFont=tick_font)

        self.spectrum_plot.clear()
        self.spectrum_plot.setTitle("LEO Delay-Doppler Paths")
        self.spectrum_plot.setLabel("bottom", "Delay", units="ns")
        self.spectrum_plot.setLabel("left", "Doppler", units="Hz")
        try:
            self.spectrum_plot.removeItem(self.spectrum_curve)
        except Exception:
            pass
        self.channel_scatter = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen(matlab_colors["blue"], width=1.2),
            brush=pg.mkBrush(91, 155, 213, 170),
        )
        self.spectrum_plot.addItem(self.channel_scatter)

        self.constellation_plot.setTitle("Detected Constellation")
        self.constellation_plot.setLabel("bottom", "I")
        self.constellation_plot.setLabel("left", "Q")
        self.constellation_plot.setAspectLocked(True)
        self.constellation_plot.setXRange(-4, 4, padding=0)
        self.constellation_plot.setYRange(-4, 4, padding=0)
        try:
            self.constellation_scatter.setPen(pg.mkPen(None))
            self.constellation_scatter.setBrush(pg.mkBrush(0, 150, 80, 180))
            self.constellation_scatter.setSize(5)
        except Exception:
            pass

        self.ber_plot.clear()
        try:
            self.ber_plot.addLegend(offset=(8, 8))
        except Exception:
            pass
        self.ber_plot.setTitle("SER vs α for β=0:0.2:2")
        self.ber_plot.setLabel("bottom", "α")
        self.ber_plot.setLabel("left", "SER")
        self.ber_plot.setLogMode(y=True)
        self.ber_plot.setXRange(0.0, 2.0, padding=0)
        self.ber_plot.setYRange(-6, 0, padding=0)

        self.ber_snr_plot.clear()
        self.ber_snr_plot.setTitle("SER vs Eb/N0")
        self.ber_snr_plot.setLabel("bottom", "Eb/N0", units="dB")
        self.ber_snr_plot.setLabel("left", "SER")
        self.ber_snr_plot.setLogMode(y=True)
        self.ber_snr_plot.enableAutoRange(axis='y', enable=True)
        self.ber_snr_plot.setXRange(min(self.SER_SNR_POINTS), max(self.SER_SNR_POINTS), padding=0)

    # ------------------------------------------------------------------
    # Backend construction and updates
    # ------------------------------------------------------------------
    def _selected_channel_dynamics(self):
        text = str(self.channel_dynamics_combo.currentText()) if hasattr(self, "channel_dynamics_combo") else "固定信道"
        if "快时变" in text:
            return "fast"
        if "动态" in text or "块" in text:
            return "block"
        return "fixed"

    def _dynamic_enabled(self):
        return self._selected_channel_dynamics() != "fixed"

    def _selected_alpha_curve_mode(self):
        text = str(self.alpha_curve_mode_combo.currentText()) if hasattr(self, "alpha_curve_mode_combo") else "ZF理论值"
        return "theory" if "理论" in text else "mc"

    def _alpha_curve_mode_label(self, mode=None):
        mode = str(mode or self._selected_alpha_curve_mode())
        return "ZF理论值" if mode == "theory" else "Monte-Carlo实测"

    def _current_alpha_ser_floor(self) -> float:
        return alpha_ser_floor(
            int(self.alpha_mc_frames_spin.value()),
            int(self.m_spin.value()),
            int(self.n_spin.value()),
        )

    def _backend_kwargs(self, alpha=None, beta=None, snr_db=None):
        return dict(
            alpha=float(self.alpha_spin.value() if alpha is None else alpha),
            beta=float(self.beta_spin.value() if beta is None else beta),
            m_subcarriers=int(self.m_spin.value()),
            n_symbols=int(self.n_spin.value()),
            subcarrier_spacing_hz=float(self.scs_spin.value()) * 1e3,
            mod_order=str(self.mod_combo.currentText()),
            channel_model=str(self.channel_combo.currentText()),
            velocity_kmh=float(self.velocity_combo.currentText()),
            doppler_radial_factor=float(self.radial_factor_spin.value()),
            decoder=str(self.decoder_combo.currentText()),
            snr_db=float(self.snr_spin.value() if snr_db is None else snr_db),
            snr_definition=str(self.snr_def_combo.currentText()),
            optimize_indices=False,
            search_step=float(self.search_step_spin.value()),
            fc_hz=float(self.fc_spin.value()) * 1e9,
            link_mode="matrix",
            random_channel=bool(self.random_channel_check.isChecked() or self._dynamic_enabled()),
            channel_seed=int(self.seed_spin.value()),
            dynamic_channel=bool(self._dynamic_enabled()),
            channel_dynamics=str(self._selected_channel_dynamics()),
            channel_coherence_frames=int(self.coherence_frames_spin.value()),
            fast_channel_coherence_symbols=int(self.fast_symbol_spin.value()),
            tf_notch_depth_db=0.0,
            tf_notch_count=0,
        )

    def _new_backend(self, alpha=None, beta=None, snr_db=None):
        from simulation.simple_fdidm_rx import FDIDMTransceiver
        return FDIDMTransceiver(**self._backend_kwargs(alpha=alpha, beta=beta, snr_db=snr_db))

    def _on_start_clicked(self):
        self._on_stop_clicked()
        try:
            self.tb = self._new_backend()
            self.tb.start()
            self.update_timer.start(250)
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self._update_info_panel(running=True)
            self._update_channel_plot()
            self._schedule_auto_ser_snr("开始仿真")
        except Exception as exc:
            self.update_info_panel(status_text=f"启动失败：{exc}")
            import traceback; traceback.print_exc()

    def _on_stop_clicked(self):
        self.update_timer.stop()
        if self.tb is not None:
            try:
                self.tb.stop()
                self.tb.wait(timeout=1.5)
            except Exception:
                pass
            self.tb = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._update_info_panel(running=False)

    def _hot_update_from_ui(self, *args):
        self._best_alpha = None
        self._best_beta = None
        self.btn_apply_best.setEnabled(False)
        if self.tb is not None:
            try:
                self.tb.update_runtime_parameters(
                    alpha=float(self.alpha_spin.value()),
                    beta=float(self.beta_spin.value()),
                    m_subcarriers=int(self.m_spin.value()),
                    n_symbols=int(self.n_spin.value()),
                    subcarrier_spacing_hz=float(self.scs_spin.value()) * 1e3,
                    mod_order=str(self.mod_combo.currentText()),
                    channel_model=str(self.channel_combo.currentText()),
                    velocity_kmh=float(self.velocity_combo.currentText()),
                    doppler_radial_factor=float(self.radial_factor_spin.value()),
                    decoder=str(self.decoder_combo.currentText()),
                    ebn0_db=float(self.snr_spin.value()),
                    snr_definition=str(self.snr_def_combo.currentText()),
                    fc_hz=float(self.fc_spin.value()) * 1e9,
                    random_channel=bool(self.random_channel_check.isChecked() or self._dynamic_enabled()),
                    channel_seed=int(self.seed_spin.value()),
                    dynamic_channel=bool(self._dynamic_enabled()),
                    channel_dynamics=str(self._selected_channel_dynamics()),
                    channel_coherence_frames=int(self.coherence_frames_spin.value()),
                    fast_channel_coherence_symbols=int(self.fast_symbol_spin.value()),
                    tf_notch_depth_db=0.0,
                    tf_notch_count=0,
                )
                self._update_channel_plot()
            except Exception:
                pass
        self._schedule_auto_ser_snr("参数更新")

    def _regenerate_channel(self):
        new_seed = int(time.time() * 1000) % 2_147_483_647
        self.seed_spin.blockSignals(True)
        self.seed_spin.setValue(max(1, new_seed))
        self.seed_spin.blockSignals(False)
        self._best_alpha = None
        self._best_beta = None
        self.btn_apply_best.setEnabled(False)
        if self.tb is not None:
            self.tb.update_runtime_parameters(
                channel_seed=int(self.seed_spin.value()),
                random_channel=True,
                dynamic_channel=bool(self._dynamic_enabled()),
                channel_dynamics=str(self._selected_channel_dynamics()),
                channel_coherence_frames=int(self.coherence_frames_spin.value()),
                fast_channel_coherence_symbols=int(self.fast_symbol_spin.value()),
            )
        self._update_channel_plot()
        self.update_info_panel(status_text="已重新生成当前LEO星地信道。系统将自动搜索最优 α/β，并按左侧选择刷新左下角 α-SER 多β曲线，同时刷新右下角理论 SER-SNR 曲线。")
        self._schedule_auto_ser_snr("重新生成信道")

    def _set_indices_ui(self, a, b):
        self.alpha_spin.blockSignals(True)
        self.beta_spin.blockSignals(True)
        self.alpha_spin.setValue(float(a))
        self.beta_spin.setValue(float(b))
        self.alpha_spin.blockSignals(False)
        self.beta_spin.blockSignals(False)
        if self.tb is not None:
            self.tb.set_indices(float(a), float(b))
        self._update_info_panel(running=self.tb is not None)
        self._schedule_auto_ser_snr("索引切换")

    def _apply_best_indices(self):
        if self._best_alpha is None or self._best_beta is None:
            return
        self._set_indices_ui(self._best_alpha, self._best_beta)

    # ------------------------------------------------------------------
    # Automatic theory search + SER-SNR curve refresh
    # ------------------------------------------------------------------
    def _schedule_auto_ser_snr(self, reason="参数更新"):
        """Debounced automatic refresh for the lower-right SER-SNR plot.

        Any key parameter update invalidates the previous best alpha/beta and
        schedules a fresh workflow:
            current H_tf -> theory search alpha*/beta* -> redraw SER-SNR curves.
        """
        if not hasattr(self, "_auto_refresh_timer"):
            return
        self._auto_scan_reason = str(reason)
        self._auto_refresh_timer.start(700)
        if self.status_text_label is not None:
            self.status_text_label.setText(
                f"参数已更新（{reason}），即将自动刷新：搜索 α*/β*、重绘 α-SER 与理论 SER-SNR。"
            )

    def _start_manual_alpha_mc_refresh(self):
        """Run the left-bottom α-SER sweep once with Monte-Carlo measurements.

        The automatic refresh mode remains theory-based after this one-shot run.
        """
        self._next_alpha_curve_mode_override = "mc"
        self._schedule_auto_ser_snr("手动Monte-Carlo α曲线")

    def _start_theory_search(self):
        # 兼容旧按钮；按钮已隐藏，实际刷新由 _schedule_auto_ser_snr 触发。
        self._schedule_auto_ser_snr("手动请求")

    def _start_ser_snr_sweep(self):
        # 兼容旧按钮；按钮已隐藏，实际刷新由 _schedule_auto_ser_snr 触发。
        self._schedule_auto_ser_snr("手动请求")

    def _set_all_refresh_stops(self):
        """Request every background refresh worker to stop."""
        for ev_name in ("_search_stop", "_alpha_stop", "_snr_stop"):
            try:
                getattr(self, ev_name).set()
            except Exception:
                pass
        # Keep old attribute in sync for compatibility with existing calls.
        try:
            self._ser_snr_stop = self._snr_stop
        except Exception:
            pass

    def _start_auto_ser_snr_refresh(self):
        """Start a new automatic refresh using independent search/plot workers."""
        try:
            base_kwargs = self._backend_kwargs()
            alpha_sweep_frames = int(self.alpha_mc_frames_spin.value())
            alpha_curve_mode = str(self._next_alpha_curve_mode_override or self._selected_alpha_curve_mode())
            self._next_alpha_curve_mode_override = None
        except Exception as exc:
            self.update_info_panel(status_text=f"自动刷新失败：无法读取参数：{exc}")
            return

        # Invalidate older workers.  Stale Qt signals are discarded by token.
        self._auto_scan_token += 1
        token = int(self._auto_scan_token)
        self._set_all_refresh_stops()
        self._search_stop = threading.Event()
        self._alpha_stop = threading.Event()
        self._snr_stop = threading.Event()
        self._ser_snr_stop = self._snr_stop

        self._best_alpha = None
        self._best_beta = None
        self._last_search_result = None
        self.btn_apply_best.setEnabled(False)
        self.btn_stop_ber_snr.setEnabled(True)
        self.btn_auto_ber_snr.setEnabled(False)
        self._search_active_token = token
        self._alpha_active_token = None
        self._snr_active_token = None
        self._last_completion_token = None

        alpha_ser_floor = alpha_ser_floor(
            alpha_sweep_frames,
            int(base_kwargs.get("m_subcarriers", 8)),
            int(base_kwargs.get("n_symbols", 8)),
        )
        self._alpha_ser_floor_by_token[token] = float(alpha_ser_floor)
        self._refresh_context_by_token[token] = {
            "base_kwargs": dict(base_kwargs),
            "alpha_sweep_frames": int(alpha_sweep_frames),
            "alpha_curve_mode": str(alpha_curve_mode),
            "alpha_ser_floor": float(alpha_ser_floor),
            "reason": str(self._auto_scan_reason),
        }

        self._clear_alpha_sweep_plot(alpha_curve_mode, floor=alpha_ser_floor)
        self._clear_ser_snr_plot()
        self._alpha_beta_curves.clear()
        self._alpha_beta_data.clear()
        self._ser_snr_curves.clear()
        self._ser_snr_data.clear()
        self.ber_snr_plot.setTitle("Theory SER vs Eb/N0 (waiting for α*/β*)")

        dyn_mode = str(base_kwargs.get("channel_dynamics", "fixed"))
        dyn_note = "固定信道" if dyn_mode == "fixed" else ("动态块衰落" if dyn_mode == "block" else "帧内快时变")
        alpha_mode_label = self._alpha_curve_mode_label(alpha_curve_mode)
        alpha_curve_text = (
            f"左下角：Monte-Carlo 实测 {alpha_sweep_frames} 帧/αβ点，α步长0.1，β步长0.2共11条"
            if alpha_curve_mode == "mc"
            else "左下角：理论值，α步长0.1，β步长0.2共11条"
        )
        self.update_info_panel(
            status_text=(
                f"自动刷新启动：{self._auto_scan_reason}。\n"
                f"信道统计模式：{dyn_note}；正在理论搜索 α*/β*。\n"
                f"搜索结束后将启动双线程：{alpha_curve_text}；右下角：理论 SER-SNR 绘制。\n"
                f"左下角显示下界=3/({alpha_sweep_frames}×{int(base_kwargs.get('m_subcarriers', 8))}×{int(base_kwargs.get('n_symbols', 8))})={alpha_ser_floor:.3e}；低于该值按下界连线显示。"
            )
        )

        self._search_thread = threading.Thread(
            target=self._search_worker,
            args=(token, dict(base_kwargs), self._search_stop, str(self._auto_scan_reason)),
            daemon=True,
        )
        self._search_thread.start()

    def _search_worker(self, token: int, base_kwargs: dict, stop_event: threading.Event, reason: str):
        """Worker 0: find best alpha/beta, then notify UI to start plot workers."""
        try:
            from simulation.simple_fdidm_rx import FDIDMTransceiver
            tb0 = FDIDMTransceiver(**base_kwargs)
            result = tb0.search_best_indices(
                step=float(base_kwargs.get("search_step", 0.1)),
                ebn0_db=float(base_kwargs.get("snr_db", 10.0)),
                stop_event=stop_event,
            )
            result["channel_summary"] = tb0.get_channel_summary()
            result["auto_reason"] = reason
            if stop_event.is_set() or token != self._auto_scan_token:
                return
            self._emit_signal_safe("search_finished", token, result)
        except Exception as exc:
            self._emit_signal_safe("search_finished", token, {"error": str(exc)})

    def _start_plot_workers_after_search(self, token: int, result: dict):
        """Start left and right plot workers concurrently after best α/β is known."""
        ctx = self._refresh_context_by_token.get(int(token), {})
        if not ctx:
            return
        base_kwargs = dict(ctx.get("base_kwargs", {}))
        alpha_sweep_frames = int(ctx.get("alpha_sweep_frames", self.alpha_mc_frames_spin.value()))
        alpha_curve_mode = str(ctx.get("alpha_curve_mode", self._selected_alpha_curve_mode()))
        best_a = float(result.get("best_alpha", base_kwargs.get("alpha", 0.0)))
        best_b = float(result.get("best_beta", base_kwargs.get("beta", 0.0)))

        # Replace older plot workers but keep the search result/token alive.
        try:
            self._alpha_stop.set()
            self._snr_stop.set()
        except Exception:
            pass
        self._alpha_stop = threading.Event()
        self._snr_stop = threading.Event()
        self._ser_snr_stop = self._snr_stop
        self._alpha_active_token = int(token)
        self._snr_active_token = int(token)

        self.ber_snr_plot.setTitle(
            f"Theory SER vs Eb/N0 (α*={best_a:.1f}, β*={best_b:.1f}; drawing)"
        )

        self._alpha_thread = threading.Thread(
            target=self._alpha_ser_worker,
            args=(int(token), dict(base_kwargs), self._alpha_stop, alpha_sweep_frames, alpha_curve_mode, best_a, best_b),
            daemon=True,
        )
        self._snr_thread = threading.Thread(
            target=self._theory_snr_worker,
            args=(int(token), dict(base_kwargs), self._snr_stop, best_a, best_b),
            daemon=True,
        )
        # Start both workers back-to-back.  They use separate backend instances,
        # so the fast right-bottom theory plot no longer waits for the slow
        # left-bottom Monte-Carlo alpha sweep.
        self._alpha_thread.start()
        self._snr_thread.start()

    def _alpha_ser_worker(self, token: int, base_kwargs: dict, stop_event: threading.Event,
                          alpha_sweep_frames: int, alpha_curve_mode: str, best_a: float, best_b: float):
        """Worker 1: draw left-bottom SER-vs-alpha curves with alpha step fixed at 0.1."""
        finish_reason = "完成"
        try:
            from simulation.simple_fdidm_rx import FDIDMTransceiver
            current_b = float(base_kwargs.get("beta", 0.0))
            # Requirement: alpha increases by 0.1 for every plotted point.
            alpha_values = np.round(np.arange(0.0, 2.0 + 0.05, 0.1), 10)
            beta_values = [round(float(b), 10) for b in self.ALPHA_SWEEP_BETAS]
            work_snr = float(base_kwargs.get("snr_db", 10.0))
            alpha_seed_base = int(base_kwargs.get("channel_seed", 42)) + 303_2026
            alpha_ser_floor = float(self._alpha_ser_floor_by_token.get(
                int(token),
                alpha_ser_floor(alpha_sweep_frames, int(base_kwargs.get("m_subcarriers", 8)), int(base_kwargs.get("n_symbols", 8))),
            ))
            mode = str(alpha_curve_mode or "mc").lower()
            for b in beta_values:
                curve_name = f"β={b:.1f}"
                for a in alpha_values:
                    if stop_event.is_set() or token != self._auto_scan_token:
                        finish_reason = "已停止"
                        self._emit_signal_safe("alpha_beta_finished", token, finish_reason)
                        return
                    tb_alpha = FDIDMTransceiver(**copy_kwargs_with(
                        base_kwargs, alpha=float(a), beta=float(b), snr_db=work_snr
                    ))
                    if mode == "theory":
                        item = tb_alpha.evaluate_theory_point(float(a), float(b), ebn0_db=work_snr)
                        ser_raw = float(item.get("selected_theory_ser", item.get("zf_theory_ser", np.nan)))
                    else:
                        item = tb_alpha.estimate_ser(
                            num_frames=int(alpha_sweep_frames),
                            seed=alpha_seed_base,  # common random numbers across all α/β points
                            stop_event=stop_event,
                            ser_display_floor=alpha_ser_floor,
                        )
                        ser_raw = float(item.get("ser", np.nan))
                    if not self._emit_signal_safe("alpha_beta_point", token, curve_name, float(a), float(ser_raw), float(alpha_ser_floor)):
                        return
        except Exception as exc:
            finish_reason = f"失败：{exc}"
        self._emit_signal_safe("alpha_beta_finished", token, finish_reason)

    def _theory_snr_worker(self, token: int, base_kwargs: dict, stop_event: threading.Event,
                           best_a: float, best_b: float):
        """Worker 2: draw right-bottom theory/proxy SER-SNR curves."""
        finish_reason = "完成"
        try:
            from simulation.simple_fdidm_rx import FDIDMTransceiver
            current_a = float(base_kwargs.get("alpha", 0.0))
            current_b = float(base_kwargs.get("beta", 0.0))
            raw_specs = [
                ("2D-OFDM α=0,β=0", 0.0, 0.0),
                ("OTFS α=1,β=1", 1.0, 1.0),
                (f"当前 α={current_a:.1f},β={current_b:.1f}", current_a, current_b),
                (f"理论最优 α={float(best_a):.1f},β={float(best_b):.1f}", float(best_a), float(best_b)),
            ]
            specs = merged_curve_specs(raw_specs)
            for name, a, b in specs:
                tb = FDIDMTransceiver(**copy_kwargs_with(base_kwargs, alpha=a, beta=b))
                for snr in self.SER_SNR_POINTS:
                    if stop_event.is_set() or token != self._auto_scan_token:
                        finish_reason = "已停止"
                        self._emit_signal_safe("ser_snr_finished", token, finish_reason)
                        return
                    item = tb.evaluate_theory_point(float(a), float(b), ebn0_db=float(snr))
                    ser_raw = float(item.get("selected_theory_ser", item.get("zf_theory_ser", np.nan)))
                    # No artificial display floor on the right-bottom theory plot.
                    # Values that are non-positive due to numerical underflow are
                    # passed as NaN and therefore not drawn on the log plot.
                    ser = float(ser_raw) if np.isfinite(ser_raw) and ser_raw > 0 else float("nan")
                    if not self._emit_signal_safe("ser_snr_point", token, name, float(snr), float(ser)):
                        return
        except Exception as exc:
            finish_reason = f"失败：{exc}"
        self._emit_signal_safe("ser_snr_finished", token, finish_reason)

    def _clear_alpha_sweep_plot(self, alpha_curve_mode=None, floor=None):
        self.ber_plot.clear()
        self._alpha_beta_curves.clear()
        self._alpha_beta_data.clear()
        try:
            self.ber_plot.addLegend(offset=(8, 8))
        except Exception:
            pass
        self.ber_plot.setLogMode(y=True)
        mode = str(alpha_curve_mode or self._selected_alpha_curve_mode())
        ylabel = "ZF Theory SER" if mode == "theory" else "Measured SER"
        title = "ZF Theory SER vs α for β=0:0.2:2" if mode == "theory" else "Monte-Carlo SER vs α for β=0:0.2:2"
        floor = self._current_alpha_ser_floor() if floor is None else float(floor)
        self.ber_plot.setLabel("bottom", "α")
        self.ber_plot.setLabel("left", ylabel)
        self.ber_plot.setTitle(f"{title} (display floor={floor:.2e})")
        # Log-axis lower limit follows the same rule-of-three floor rather than
        # a hard-coded 1e-12.  Values below the floor are plotted at the floor.
        ymin_exp = float(np.floor(np.log10(max(floor, 1e-300))))
        self.ber_plot.setYRange(ymin_exp - 0.15, 0, padding=0)
        self.ber_plot.setXRange(0.0, 2.0, padding=0)

    def _clear_ser_snr_plot(self):
        self.ber_snr_plot.clear()
        try:
            self.ber_snr_plot.addLegend(offset=(8, 8))
        except Exception:
            pass
        self.ber_snr_plot.setLogMode(y=True)
        self.ber_snr_plot.setLabel("bottom", "Eb/N0", units="dB")
        self.ber_snr_plot.setLabel("left", "Theory SER")
        self.ber_snr_plot.setTitle("Theory SER vs Eb/N0")
        self.ber_snr_plot.enableAutoRange(axis='y', enable=True)
        self.ber_snr_plot.setXRange(min(self.SER_SNR_POINTS), max(self.SER_SNR_POINTS), padding=0)

    def _on_qt_destroyed(self, *_args):
        """Stop background work when the Qt object is being destroyed."""
        self._qt_alive = False
        try:
            self._set_all_refresh_stops()
        except Exception:
            pass
        try:
            self._auto_scan_token += 1
        except Exception:
            pass

    def _emit_signal_safe(self, signal_name: str, *args) -> bool:
        """Emit a Qt signal from a worker thread only if the widget still exists.

        When the user closes the tab/window while the SER-SNR worker is still
        running, PyQt may delete the C++ QObject before the Python thread exits.
        Directly calling self.some_signal.emit(...) then raises
        RuntimeError: wrapped C/C++ object ... has been deleted.  This helper
        turns that race into a clean worker exit.
        """
        if not getattr(self, "_qt_alive", True):
            return False
        try:
            getattr(self, signal_name).emit(*args)
            return True
        except RuntimeError:
            self._qt_alive = False
            try:
                self._set_all_refresh_stops()
            except Exception:
                pass
            return False
        except Exception:
            return False

    def _stop_ser_snr_sweep(self):
        try:
            self._set_all_refresh_stops()
        except Exception:
            pass
        self._auto_scan_token += 1
        self._search_active_token = None
        self._alpha_active_token = None
        self._snr_active_token = None
        self.btn_stop_ber_snr.setEnabled(False)
        self.btn_auto_ber_snr.setEnabled(True)
        self.update_info_panel(status_text="已停止当前自动刷新。后续参数更新会再次自动触发搜索、左下角α-SER与右下角理论SER-SNR绘制。")

    def _on_search_finished(self, token, result):
        if int(token) != int(self._auto_scan_token):
            return
        self._search_active_token = None
        if not isinstance(result, dict) or "error" in result:
            self.update_info_panel(status_text=f"自动理论搜索失败：{result.get('error', 'unknown') if isinstance(result, dict) else result}")
            self._maybe_finish_auto_refresh(int(token), "搜索失败")
            return
        self._last_search_result = result
        self._best_alpha = float(result["best_alpha"])
        self._best_beta = float(result["best_beta"])
        self.btn_apply_best.setEnabled(True)
        self.update_info_panel(status_text=self._format_search_text(result))
        suffix = "显著" if result.get("significant", False) else "差异较弱"
        self.ber_snr_plot.setTitle(
            f"Theory SER vs Eb/N0 (α*={self._best_alpha:.1f}, β*={self._best_beta:.1f}, {suffix}; dual-thread drawing)"
        )
        self._start_plot_workers_after_search(int(token), result)

    def _maybe_finish_auto_refresh(self, token: int, reason: str = "完成"):
        if int(token) != int(self._auto_scan_token):
            return
        if self._search_active_token is not None or self._alpha_active_token is not None or self._snr_active_token is not None:
            return
        if self._last_completion_token == int(token):
            return
        self._last_completion_token = int(token)
        self.btn_auto_ber_snr.setEnabled(True)
        self.btn_stop_ber_snr.setEnabled(False)
        if self.status_text_label is not None:
            old = self.status_text_label.text()
            self.status_text_label.setText(
                old + f"\n\n自动刷新{reason}：左下角α-SER与右下角理论SER-SNR采用独立线程绘制，曲线已按当前token更新。"
            )

    def _format_search_text(self, r):
        """精简搜索结果文本，避免底部说明框过长影响可读性。"""
        ch = r.get("channel_summary", {})
        refs = r.get("references", {})
        best_a = float(r.get("best_alpha", np.nan))
        best_b = float(r.get("best_beta", np.nan))
        floor = self._current_alpha_ser_floor()
        norm_dop = ch.get("normalized_doppler", ch.get("max_doppler_hz", 0) / max(ch.get("subcarrier_spacing_hz", 1), 1))

        def ref_ser(name):
            item = refs.get(name, {})
            return item.get("score_ser_geomean", item.get("zf_theory_ser", np.nan))

        lines = [
            f"搜索完成：α*={best_a:.1f}, β*={best_b:.1f}；步长={r.get('step', np.nan):.2f}；耗时={r.get('elapsed_s', np.nan):.2f}s。",
            f"信道：{ch.get('channel_model','--')}，seed={ch.get('seed','--')}，fDmax≈{ch.get('max_doppler_hz',0):.0f}Hz，νmax/Δf≈{float(norm_dop):.3f}，径向系数={ch.get('doppler_radial_factor',0):.2f}。",
            f"搜索目标：{r.get('objective','decoder-aware SER')}；SNR定义={r.get('snr_definition','Eb/N0')}；显著性={('显著' if r.get('significant', False) else '较弱')}。",
            f"几何均值SER：OFDM={ref_ser('OFDM(0,0)'):.2e}，OTFS={ref_ser('OTFS(1,1)'):.2e}，最优={ref_ser('理论最优'):.2e}。",
            f"左下角：α-SER曲线，β=0:0.2:2共11条；SER<{floor:.2e}按floor连线显示；右下角：SER-SNR理论曲线使用原始正值，不再设置显示floor。",
            "说明：物理信道 H_tf 固定；α/β 只改变 H_eq=Rx_FDIT·H_tf·Tx_IFDIT 的等效表示。",
        ]
        note = str(r.get("significance_note", "")).strip()
        if note:
            lines.append(f"提示：{note}")
        return "\n".join(lines)

    def _curve_style(self, name):
        c = {
            "OFDM": (0, 114, 189),
            "OTFS": (217, 83, 25),
            "当前": (126, 47, 142),
            "理论最优": (0, 150, 80),
        }
        if "OFDM" in name:
            return pg.mkPen(c["OFDM"], width=2.2), 'o'
        if name.startswith("OTFS"):
            return pg.mkPen(c["OTFS"], width=2.2), 's'
        if name.startswith("理论最优"):
            return pg.mkPen(c["理论最优"], width=2.4), 'd'
        return pg.mkPen(c["当前"], width=2.2), 't'

    def _curve_style_alpha_beta(self, name):
        # Eleven visually distinct curves for β=0:0.2:2.
        colors = [
            (0, 114, 189),
            (217, 83, 25),
            (237, 177, 32),
            (126, 47, 142),
            (0, 150, 80),
            (77, 190, 238),
            (162, 20, 47),
            (120, 120, 120),
            (64, 64, 180),
            (180, 90, 20),
            (20, 130, 130),
        ]
        symbols = ['o', 's', 't', 'd', 'p', 'h', 'star', '+', 'x', 't2', 't3']
        idx = len(self._alpha_beta_curves)
        return pg.mkPen(colors[idx % len(colors)], width=2.0), symbols[idx % len(symbols)]

    def _on_alpha_beta_point(self, token, name, alpha, ser, ser_floor):
        if int(token) != int(self._auto_scan_token):
            return
        if name not in self._alpha_beta_curves:
            pen, symbol = self._curve_style_alpha_beta(name)
            try:
                fill_brush = pg.mkBrush(pen.color())
            except Exception:
                fill_brush = pg.mkBrush(60, 60, 60)
            curve = self.ber_plot.plot(
                [], [],
                pen=pen,
                symbol=symbol,
                symbolSize=6,
                symbolPen=pen,
                symbolBrush=fill_brush,
                name=name,
            )
            self._alpha_beta_curves[name] = curve
            self._alpha_beta_data[name] = ([], [])

        x, y = self._alpha_beta_data.setdefault(name, ([], []))
        floor = float(ser_floor)
        if not np.isfinite(floor) or floor <= 0:
            floor = self._current_alpha_ser_floor()
        val = float(ser)
        if not np.isfinite(val) or val <= 0:
            val = floor
        # Previous design: keep the curve continuous by plotting values below
        # the display floor at the floor instead of using separate censored markers.
        x.append(float(alpha))
        y.append(max(float(val), float(floor)))
        order = np.argsort(np.asarray(x))
        xx = np.asarray(x, dtype=float)[order]
        yy = np.asarray(y, dtype=float)[order]
        curve = self._alpha_beta_curves.get(name)
        if curve is not None:
            curve.setData(xx, yy)

    def _on_alpha_beta_finished(self, token, reason):
        if int(token) != int(self._auto_scan_token):
            return
        self._alpha_active_token = None
        ctx = self._refresh_context_by_token.get(int(token), {})
        mode = str(ctx.get("alpha_curve_mode", self._selected_alpha_curve_mode()))
        title = "ZF/MMSE Theory SER" if mode == "theory" else "Monte-Carlo SER"
        floor = float(self._alpha_ser_floor_by_token.get(int(token), self._current_alpha_ser_floor()))
        self.ber_plot.setTitle(f"{title} vs α for β=0:0.2:2 ({reason}, α step=0.1, display floor={floor:.2e})")
        self._maybe_finish_auto_refresh(int(token), str(reason))

    def _on_ser_snr_point(self, token, name, snr, ser):
        if int(token) != int(self._auto_scan_token):
            return
        if name not in self._ser_snr_curves:
            pen, symbol = self._curve_style(name)
            curve = self.ber_snr_plot.plot(
                [], [],
                pen=pen,
                symbol=symbol,
                symbolSize=7,
                symbolPen=pen,
                symbolBrush='w',
                name=name,
            )
            self._ser_snr_curves[name] = curve
            self._ser_snr_data[name] = ([], [])
        x, y = self._ser_snr_data.setdefault(name, ([], []))
        x.append(float(snr))
        val = float(ser)
        # Right-bottom plot no longer clips to a display floor.  Non-positive
        # values cannot be displayed on a log axis, so they break the curve.
        y.append(val if np.isfinite(val) and val > 0 else float("nan"))
        order = np.argsort(np.asarray(x))
        xx = np.asarray(x, dtype=float)[order]
        yy = np.asarray(y, dtype=float)[order]
        curve = self._ser_snr_curves.get(name)
        if curve is not None:
            curve.setData(xx, yy)

    def _on_ser_snr_finished(self, token, reason):
        if int(token) != int(self._auto_scan_token):
            return
        self._snr_active_token = None
        self.ber_snr_plot.setTitle(f"Theory SER vs Eb/N0 ({reason}; no display floor)")
        self._maybe_finish_auto_refresh(int(token), str(reason))

    # ------------------------------------------------------------------
    # Runtime plots and info
    # ------------------------------------------------------------------
    def _refresh_plots(self):
        if self.tb is None:
            return
        try:
            const = self.tb.get_constellation()
            if const is not None and len(const):
                self.constellation_scatter.setData(x=np.real(const), y=np.imag(const))
            # The left-bottom plot is now the alpha sweep.  Do not overwrite it
            # with sliding SER during runtime refresh.
            self._update_info_panel(running=True)
        except Exception:
            pass

    def _update_channel_plot(self):
        try:
            tb = self.tb if self.tb is not None else self._new_backend()
            summary = tb.get_channel_summary()
            paths = summary.get("paths", [])
            if not paths:
                self.channel_scatter.setData([], [])
                return
            x = np.asarray([p["delay_ns"] for p in paths], dtype=float)
            y = np.asarray([p["doppler_hz"] for p in paths], dtype=float)
            # 为了让 delay-Doppler 路径图更像学术散点图，默认不再用圆圈大小编码路径增益。
            # 路径增益仍在右下方“演示说明”文字区输出，避免大圆点遮挡坐标和误导视觉判断。
            marker_size = 13.0
            spots = [{"pos": (x[i], y[i]), "size": marker_size} for i in range(len(x))]
            self.channel_scatter.setData(spots)
            self.spectrum_plot.setTitle(
                f"LEO paths: {summary.get('channel_model')} seed={summary.get('seed')} "
                f"v={summary.get('velocity_kms', summary.get('velocity_kmh', 0)/3600):.2f} km/s "
                f"fD≈{summary.get('max_doppler_hz',0):.0f} Hz"
            )
        except Exception:
            pass

    @staticmethod
    def _fmt(v, digits=3):
        try:
            v = float(v)
        except Exception:
            return "--"
        if not np.isfinite(v):
            return "--"
        if abs(v) >= 1e3 or (0 < abs(v) < 1e-2):
            return f"{v:.{digits}e}"
        return f"{v:.{digits}f}"

    def _update_info_panel(self, running=False):
        m = {}
        if self.tb is not None:
            try:
                m = self.tb.get_last_metrics()
            except Exception:
                m = {}
        alpha = float(self.alpha_spin.value())
        beta = float(self.beta_spin.value())
        status = "运行中" if running else "未启动"
        self.update_info_panel(
            metrics={
                "运行状态": status,
                "Eb/N0": f"{self.snr_def_combo.currentText()}={self.snr_spin.value():.1f} dB",
                "BER": f"cum={self._fmt(m.get('ber'))}, win={self._fmt(m.get('ber_window'))}",
                "FER": self._fmt(m.get("fer")),
                "同步度量": f"SERwin={self._fmt(m.get('ser'))}, SERth={self._fmt(m.get('selected_theory_ser', m.get('zf_theory_ser')))}",
                "CFO估计": f"H_eq rowMax={self._fmt(m.get('row_norm_max'))}",
            },
            config={
                "波形": "FDIDM Mode-A",
                "FFT/CP": f"M×N={self.m_spin.value()}×{self.n_spin.value()}；无显式CP",
                "调制": self.mod_combo.currentText(),
                "采样率": f"{self.m_spin.value()*self.scs_spin.value()/1000:.2f} MHz",
                "数据符号": f"{self.m_spin.value()*self.n_spin.value()}",
                "星座观察": f"α={alpha:.1f}, β={beta:.1f}; v={float(self.velocity_combo.currentText())/3600:.2f} km/s；径向={self.radial_factor_spin.value():.2f}；信道={self.channel_dynamics_combo.currentText()}",
            },
        )
        if not running and self.status_text_label is not None and not self._last_search_result:
            self.status_text_label.setText(
                "模式A：自动搜索 α*/β*，左下角显示 α-SER 多β曲线，右下角显示理论 SER-SNR 对比。\n"
                "多普勒仅由速度、径向系数和载频计算；α-SER显示下界=3/(MC帧数×M×N)，右下角SER-SNR不再使用显示floor。"
            )

    def closeEvent(self, event):
        try:
            self._stop_ser_snr_sweep()
            self._on_stop_clicked()
        finally:
            super().closeEvent(event)
