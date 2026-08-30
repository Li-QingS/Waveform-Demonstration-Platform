"""FDIDM soft-waveform demonstration tab.

Goal:
    One standardized LEO/NTN channel -> separate predictable orbital Doppler from
    residual fading spread -> evaluate independent OFDM/OTFS/AFDM/FDIDM domains on
    the same channel -> adapt FDIDM alpha/beta on a slower CSI-statistics time scale.

This page intentionally removes alpha-beta heatmap and paper-figure modes.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import OrderedDict

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
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .base_waveform_tab import BaseWaveformTab
from .fdidm_adaptive_widgets import AdaptiveControlBox, AdaptiveProcessPlots

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))



from .fdidm_utils import alpha_ser_floor, copy_kwargs_with

class FDIDMTab(BaseWaveformTab):
    """FDIDM soft-waveform simulation and adaptation page."""

    ser_snr_point = pyqtSignal(int, str, float, float, float, float)
    ser_snr_finished = pyqtSignal(int, str)
    search_finished = pyqtSignal(int, object)
    alpha_beta_point = pyqtSignal(int, str, float, float, float)
    alpha_beta_finished = pyqtSignal(int, str)

    # Adaptive-process SER-vs-SNR sweep points.
    SER_SNR_POINTS = list(range(0, 31, 2))
    # Left-bottom alpha sweep: x-axis is alpha, each curve fixes one beta.
    ALPHA_SWEEP_BETAS = [round(0.2 * i, 1) for i in range(11)]
    # The adaptive-process SER-vs-SNR plot uses raw positive benchmark values.
    # Non-positive/underflow values are not drawn on the log plot instead of
    # being clipped to an artificial display floor.
    # 右下角"实时性能随时间变化"图的视图稳定策略：
    # X 轴使用固定宽度滚动时间窗（右边界按整步长滑动），Y 轴保持固定范围，
    # 避免新增噪声点导致坐标轴不断重算、画面抖动。
    TIME_WINDOW_SEC = 30.0             # 只显示最近30秒
    TIME_WINDOW_STEP_SEC = 5.0          # 每5秒平移一次视窗
    SER_Y_RANGE = (-6.0, 0.0)          # log10(SER) 显示范围：1e-6 ~ 1
    SER_Y_EXPAND_CLAMP = (-12.0, 2.0)  # 数据越界时仅扩展、不收缩，且限制在此范围内

    def __init__(self):
        super().__init__("FDIDM")
        self.snr_label.setText("Eb/N0:")
        self.snr_spin.setValue(10.0)
        try:
            self.rename_info_rows({
                "Eb/N0": "实时SNR",
                "BER": "瞬时多普勒",
                "FER": "时延状态",
                "同步度量": "多普勒扩展",
                "CFO估计": "噪声 / 信道",
                "FFT/CP": "帧结构",
                "星座观察": "索引状态",
            })
            metric_group = self.info_name_labels["运行状态"].parentWidget()
            if hasattr(metric_group, "setTitle"):
                metric_group.setTitle("实时链路状态")
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
        #   3) snr worker: draw the adaptive-process shared-channel SER-SNR curves.
        # This prevents a slow Monte-Carlo alpha sweep from blocking the
        # adaptive-process SER-SNR plot.
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

        self.adaptive_plots = AdaptiveProcessPlots()
        self.adaptive_controls.config_changed.connect(self._on_adaptive_config_changed)
        self.adaptive_controls.evaluate_requested.connect(self._on_adaptive_evaluate_clicked)

        # 右侧绘图区页签化：仿真图（2x2 + 信息面板）/ 自适应过程（轨迹 + SER + 状态）
        plot_layout = self.plot_panel.layout()
        grid_item = plot_layout.takeAt(0)
        info_item = plot_layout.takeAt(0)
        grid_layout = grid_item.layout()
        grid_layout.setParent(None)
        info_widget = info_item.widget()
        info_widget.setParent(None)

        sim_tab = QWidget()
        sim_layout = QVBoxLayout(sim_tab)
        sim_layout.setContentsMargins(6, 6, 6, 6)
        sim_layout.setSpacing(8)
        sim_layout.addLayout(grid_layout, stretch=4)
        sim_layout.addWidget(info_widget, stretch=1)

        self.plot_tabs = QTabWidget()
        self.plot_tabs.addTab(sim_tab, "仿真图")
        self.plot_tabs.addTab(self.adaptive_plots, "自适应过程")
        plot_layout.addWidget(self.plot_tabs)

        self.adaptive_timer = QTimer()
        self.adaptive_timer.setInterval(250)
        self.adaptive_timer.timeout.connect(self._refresh_adaptive_panel)
        self.adaptive_timer.start()

        self.cfo_label.setVisible(False)
        self.cfo_spin.setVisible(False)
        self.doppler_label.setVisible(False)
        self.doppler_spin.setVisible(False)
        # 自适应过程页的 SER-SNR 曲线由关键参数更新自动触发，
        # 旧版手动刷新按钮保留为隐藏的基类兼容控件。
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

        # Align residual Doppler, physical coherence and slow-controller window
        # before the first automatic SER-SNR refresh.
        self._sync_profile_defaults(self.channel_combo.currentText())
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

        self.decoder_combo = QComboBox()
        self.decoder_combo.addItems(["ZF", "MMSE", "ZF-SIC"])
        self.decoder_combo.setCurrentText("ZF")
        idx_form.addRow("检测器:", self.decoder_combo)

        # --- 帧结构与波形 ---
        frame_group, frame_form = form_group("帧结构 / 波形")
        self.snr_def_combo = QComboBox()
        self.snr_def_combo.addItems(["Eb/N0", "Es/N0"])
        self.snr_def_combo.setCurrentText("Eb/N0")

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
        self.channel_combo.addItems([
            "NTN-TDLA100-200",
            "NTN-TDLC5-200",
            "NTN-TDLC5-1200",
        ])
        self.channel_combo.setCurrentText("NTN-TDLA100-200")
        self.channel_combo.setToolTip(
            "标准化NTN延迟功率谱与残余最大Doppler组合；"
            "大尺度卫星公共频移在下方单独计算并补偿。"
        )
        ch_form.addRow("NTN模型:", self.channel_combo)
        # FDIDM 页面把通用 Eb/N0 控件移动到 LEO 信道模块中。
        self.snr_label.setParent(None)
        self.snr_spin.setParent(None)
        ch_form.addRow("Eb/N0:", self.snr_spin)

        self.velocity_combo = QComboBox()
        self.velocity_combo.addItems(["0", "120", "500", "28080"])
        self.velocity_combo.setCurrentText("28080")
        ch_form.addRow("速度(km/h):", self.velocity_combo)

        self.channel_dynamics_combo = QComboBox()
        self.channel_dynamics_combo.addItems(["固定信道", "动态块衰落", "帧内快时变", "连续多普勒"])
        self.channel_dynamics_combo.setCurrentText("连续多普勒")
        self.channel_dynamics_combo.setToolTip(
            "动态块衰落按相干帧数更新CSI；帧内快时变在一帧内部改变路径增益；"
            "连续多普勒只推进补偿后的残余CFO和多径Doppler扩展。"
            "可预测的轨道公共频移不会直接当成随机快衰落。"
        )
        ch_form.addRow("时变模式:", self.channel_dynamics_combo)

        # LEO 轨道公共频移与残余小尺度时变参数。
        self.radial_factor_spin = QDoubleSpinBox()
        self.radial_factor_spin.setRange(0.0, 1.0)
        self.radial_factor_spin.setDecimals(2)
        self.radial_factor_spin.setSingleStep(0.01)
        self.radial_factor_spin.setValue(0.10)
        self.radial_factor_spin.setToolTip(
            "用于计算可预测的轨道公共Doppler：f_orbit=v×径向投影×fc/c；"
            "它与小尺度Doppler扩展分开处理。"
        )
        ch_form.addRow("径向投影:", self.radial_factor_spin)

        self.residual_doppler_spin = QDoubleSpinBox()
        self.residual_doppler_spin.setRange(0.0, 5000.0)
        self.residual_doppler_spin.setDecimals(1)
        self.residual_doppler_spin.setSingleStep(100.0)
        self.residual_doppler_spin.setValue(200.0)
        self.residual_doppler_spin.setSuffix(" Hz")
        self.residual_doppler_spin.setToolTip(
            "公共轨道频移补偿后，真正决定小尺度信道相干时间的最大Doppler扩展。"
        )
        ch_form.addRow("残余Doppler扩展:", self.residual_doppler_spin)

        self.doppler_compensation_spin = QDoubleSpinBox()
        self.doppler_compensation_spin.setRange(0.0, 100.0)
        self.doppler_compensation_spin.setDecimals(3)
        self.doppler_compensation_spin.setSingleStep(0.1)
        self.doppler_compensation_spin.setValue(99.9)
        self.doppler_compensation_spin.setSuffix(" %")
        self.doppler_compensation_spin.setToolTip(
            "对可预测公共轨道Doppler的预补偿比例；剩余部分作为残余公共CFO。"
        )
        ch_form.addRow("公共Doppler补偿:", self.doppler_compensation_spin)

        self.random_channel_check = QCheckBox("随机路径相位/角度")
        self.random_channel_check.setChecked(True)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 2_147_483_647)
        self.seed_spin.setValue(42)

        self.coherence_frames_spin = QSpinBox()
        self.coherence_frames_spin.setRange(1, 10000)
        self.coherence_frames_spin.setValue(64)
        self.fast_symbol_spin = QSpinBox()
        self.fast_symbol_spin.setRange(1, 64)
        self.fast_symbol_spin.setValue(1)

        # --- 曲线统计 ---
        stat_group, stat_form = form_group("曲线统计")
        self.stat_group = stat_group  # 保留引用，避免控件被垃圾回收
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

        # 高级参数模块已移除。搜索步长、随机种子和相干参数仍保留为内部配置，
        # 由既有默认值驱动，避免破坏后端兼容性。

        # --- 自适应控制（左栏） ---
        self.adaptive_controls = AdaptiveControlBox()

        for group in (idx_group, frame_group, ch_group, self.adaptive_controls,
                      quick_group):
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
        self.channel_combo.currentTextChanged.connect(self._sync_profile_defaults)

        for widget in [
            self.alpha_spin, self.beta_spin, self.m_spin, self.n_spin,
            self.scs_spin, self.fc_spin, self.mod_combo, self.channel_combo,
            self.velocity_combo, self.radial_factor_spin,
            self.residual_doppler_spin, self.doppler_compensation_spin,
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
        self.ber_snr_plot.setTitle("实时波形性能随时间变化")
        self.ber_snr_plot.setLabel("bottom", "时间", units="s")
        self.ber_snr_plot.setLabel("left", "SER")
        self.ber_snr_plot.setLogMode(y=True)
        # 固定视图范围：X 轴为滚动时间窗，Y 轴固定为 log10(SER)∈[-6,0]（1e-6~1），
        # 关闭 auto-range，避免新增点导致坐标轴不断重算、画面抖动。
        self.ber_snr_plot.setXRange(0.0, self.TIME_WINDOW_SEC, padding=0)
        self.ber_snr_plot.setYRange(*self.SER_Y_RANGE, padding=0)
        self.ber_snr_plot.enableAutoRange(axis="x", enable=False)
        self.ber_snr_plot.enableAutoRange(axis="y", enable=False)
        self.ber_snr_plot.addLegend(offset=(8, 8))
        self._time_metric_curves = {}
        self._time_switch_labels = []
        self._time_switch_rendered = 0
        self._time_plot_eval_count = -1
        self._time_plot_x_view = None
        self._time_plot_y_fit = None

    # ------------------------------------------------------------------
    # Backend construction and updates
    # ------------------------------------------------------------------
    def _sync_profile_defaults(self, name):
        """Keep profile-dependent residual Doppler and block coherence defaults aligned.

        Slow-controller windows are intentionally scaled down from the physical
        coherence time (agile preset): the CSI window covers about half a
        coherence time and a switch can be voted after two consecutive
        recommendations, so parameter adaptation reacts within a few seconds
        while retaining a minimal anti-oscillation guard.
        """
        profile = str(name or "").upper()
        maximum_doppler = 1200.0 if "1200" in profile else 200.0
        # Tc≈0.423/fD and Tframe=N/Δf.  Round to a usable integer block length.
        frame_duration = float(self.n_spin.value()) / max(float(self.scs_spin.value()) * 1e3, 1e-15)
        coherence_frames = max(1, int(round((0.423 / maximum_doppler) / frame_duration)))
        slow_window = max(16, int(round(0.5 * coherence_frames)))
        slow_interval = max(8, int(round(0.5 * slow_window)))
        slow_ensemble = 3
        slow_cooldown = max(16, slow_window)
        controls = [
            self.residual_doppler_spin,
            self.coherence_frames_spin,
            self.adaptive_controls.window_frames_spin,
            self.adaptive_controls.eval_interval_spin,
            self.adaptive_controls.ensemble_spin,
            self.adaptive_controls.cooldown_spin,
        ]
        for widget in controls:
            widget.blockSignals(True)
        try:
            self.residual_doppler_spin.setValue(maximum_doppler)
            self.coherence_frames_spin.setValue(coherence_frames)
            self.adaptive_controls.window_frames_spin.setValue(slow_window)
            self.adaptive_controls.eval_interval_spin.setValue(slow_interval)
            self.adaptive_controls.ensemble_spin.setValue(slow_ensemble)
            self.adaptive_controls.cooldown_spin.setValue(slow_cooldown)
        finally:
            for widget in controls:
                widget.blockSignals(False)
        self._hot_update_from_ui()

    def _selected_channel_dynamics(self):
        text = str(self.channel_dynamics_combo.currentText()) if hasattr(self, "channel_dynamics_combo") else "固定信道"
        if "连续" in text or "多普勒" in text:
            return "cont"
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
            ntn_profile=str(self.channel_combo.currentText()),
            velocity_kmh=float(self.velocity_combo.currentText()),
            doppler_radial_factor=float(self.radial_factor_spin.value()),
            residual_doppler_spread_hz=float(self.residual_doppler_spin.value()),
            doppler_compensation_ratio=float(self.doppler_compensation_spin.value()) / 100.0,
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
            demo_frame_interval_s=0.02,
        )

    def _new_backend(self, alpha=None, beta=None, snr_db=None):
        from waveform_sim.simulation.simple_fdidm_rx import FDIDMTransceiver
        return FDIDMTransceiver(**self._backend_kwargs(alpha=alpha, beta=beta, snr_db=snr_db))

    def _on_start_clicked(self):
        self._on_stop_clicked()
        try:
            self.tb = self._new_backend()
            self.tb.start()
            if self.adaptive_controls.enable_check.isChecked():
                try:
                    self.tb.start_adaptive_tuning(**self.adaptive_controls.collect_config())
                except Exception:
                    pass
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
                self.tb.stop_adaptive_tuning()
            except Exception:
                pass
            try:
                self.tb.stop()
                self.tb.wait(timeout=1.5)
            except Exception:
                pass
            self.tb = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._update_info_panel(running=False)

    # ------------------------------------------------------------------
    # Adaptive process panel
    # ------------------------------------------------------------------
    def _on_adaptive_config_changed(self, *_args):
        """Forward panel configuration to the running backend."""
        tb = self.tb
        if tb is None:
            return
        try:
            cfg = self.adaptive_controls.collect_config()
            if cfg.get("adaptive_enabled"):
                tb.start_adaptive_tuning(**cfg)
            else:
                tb.stop_adaptive_tuning()
        except Exception:
            pass

    def _on_adaptive_evaluate_clicked(self):
        tb = self.tb
        if tb is not None:
            try:
                tb.request_adaptive_evaluation()
            except Exception:
                pass

    def _refresh_adaptive_panel(self):
        """Refresh live channel state and adaptive history.

        The previous page returned early when adaptation was disabled, so it
        could not display a changing channel.  This implementation always polls
        channel/metric state and also synchronizes the visible alpha/beta values
        after a real backend auto-apply event.
        """
        tb = self.tb
        if tb is None:
            return
        try:
            status = tb.get_adaptive_status()
        except Exception:
            status = {}
        try:
            history = tb.get_adaptive_history(limit=2000)
        except Exception:
            history = []
        try:
            channel = tb.get_channel_summary()
        except Exception:
            channel = {}
        try:
            metrics = tb.get_last_metrics()
        except Exception:
            metrics = {}

        current_a = status.get("current_alpha", metrics.get("alpha", np.nan))
        current_b = status.get("current_beta", metrics.get("beta", np.nan))
        try:
            current_a = float(current_a)
            current_b = float(current_b)
            if np.isfinite(current_a) and np.isfinite(current_b):
                if abs(self.alpha_spin.value() - current_a) > 1e-9:
                    self.alpha_spin.blockSignals(True)
                    self.alpha_spin.setValue(current_a)
                if abs(self.beta_spin.value() - current_b) > 1e-9:
                    self.beta_spin.blockSignals(True)
                    self.beta_spin.setValue(current_b)
        finally:
            self.alpha_spin.blockSignals(False)
            self.beta_spin.blockSignals(False)

        link_state = self._build_realtime_link_state(status, channel, metrics)
        self.adaptive_plots.refresh(status, history, link_state=link_state)
        self._refresh_time_metric_plot(history)

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
                    ntn_profile=str(self.channel_combo.currentText()),
                    velocity_kmh=float(self.velocity_combo.currentText()),
                    doppler_radial_factor=float(self.radial_factor_spin.value()),
                    residual_doppler_spread_hz=float(self.residual_doppler_spin.value()),
                    doppler_compensation_ratio=float(self.doppler_compensation_spin.value()) / 100.0,
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
        self.update_info_panel(status_text="已重新生成当前LEO星地信道。系统将自动搜索最优 α/β，并按左侧选择刷新左下角 α-SER 多β曲线，同时刷新自适应过程页的理论 SER-SNR 曲线。")
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

        ser_floor = alpha_ser_floor(
            alpha_sweep_frames,
            int(base_kwargs.get("m_subcarriers", 8)),
            int(base_kwargs.get("n_symbols", 8)),
        )
        self._alpha_ser_floor_by_token[token] = float(ser_floor)
        self._refresh_context_by_token[token] = {
            "base_kwargs": dict(base_kwargs),
            "alpha_sweep_frames": int(alpha_sweep_frames),
            "alpha_curve_mode": str(alpha_curve_mode),
            "alpha_ser_floor": float(ser_floor),
            "reason": str(self._auto_scan_reason),
        }

        self._clear_alpha_sweep_plot(alpha_curve_mode, floor=ser_floor)
        self._clear_ser_snr_plot()
        self._alpha_beta_curves.clear()
        self._alpha_beta_data.clear()
        self._ser_snr_curves.clear()
        self._ser_snr_data.clear()
        self.adaptive_plots.set_snr_title("SER-SNR 性能对比（等待动态参数搜索）")

        dyn_mode = str(base_kwargs.get("channel_dynamics", "fixed"))
        dyn_note = (
            "固定信道" if dyn_mode == "fixed"
            else ("连续多普勒" if dyn_mode == "cont"
                  else ("动态块衰落" if dyn_mode == "block" else "帧内快时变"))
        )
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
                f"搜索结束后将启动双线程：{alpha_curve_text}；自适应页：共享信道四波形 SER-SNR 绘制。\n"
                f"左下角显示下界=3/({alpha_sweep_frames}×{int(base_kwargs.get('m_subcarriers', 8))}×{int(base_kwargs.get('n_symbols', 8))})={ser_floor:.3e}；低于该值按下界连线显示。"
            )
        )

        self._search_thread = threading.Thread(
            target=self._search_worker,
            args=(token, dict(base_kwargs), self._search_stop, str(self._auto_scan_reason)),
            daemon=True,
        )
        self._search_thread.start()

    def _search_worker(self, token: int, base_kwargs: dict, stop_event: threading.Event, reason: str):
        """Find the current-SNR optimum with the same practical benchmark used by plots.

        The old page mixed two objectives: a dense perfect-CSI theory search for the
        button and a sparse shared-channel receiver for the comparison plots.  That
        could recommend parameters that were not actually optimal on the displayed
        curves.  This worker now freezes one physical channel snapshot and uses the
        independent-waveform benchmark throughout.
        """
        try:
            from waveform_sim.simulation.simple_fdidm_rx import FDIDMTransceiver
            from waveform_sim.simulation.shared_waveform_benchmark import SharedWaveformBenchmark

            fixed_kwargs = copy_kwargs_with(
                base_kwargs,
                dynamic_channel=False,
                channel_dynamics="fixed",
                channel_seed=int(base_kwargs.get("channel_seed", 42)),
            )
            tb0 = FDIDMTransceiver(**fixed_kwargs)
            benchmark = SharedWaveformBenchmark.from_backend(tb0)
            current_a = float(base_kwargs.get("alpha", 0.0))
            current_b = float(base_kwargs.get("beta", 0.0))
            working_snr = float(base_kwargs.get("snr_db", 10.0))
            coarse_step = max(0.25, float(base_kwargs.get("search_step", 0.1)))
            fine_step = min(0.10, max(0.05, coarse_step / 5.0))
            optimized = SharedWaveformBenchmark.optimize_fdidm_over_ensemble(
                [benchmark],
                snr_db=working_snr,
                current_alpha=current_a,
                current_beta=current_b,
                coarse_step=coarse_step,
                fine_step=fine_step,
                decision_ser_floor=1e-8,
                stop_event=stop_event,
            )
            refs = benchmark.evaluate_all(working_snr, current_a, current_b)
            best_a = float(optimized["recommended_alpha"])
            best_b = float(optimized["recommended_beta"])
            best_metric = benchmark.evaluate("FDIDM", working_snr, best_a, best_b)
            gain_db = float(optimized["predicted_improvement_db"])
            result = {
                **optimized,
                "best_alpha": best_a,
                "best_beta": best_b,
                "best_score": float(best_metric["ser"]),
                "current_score": float(optimized["predicted_ser_current"]),
                "step": float(coarse_step),
                "fine_step": float(fine_step),
                "elapsed_s": float(optimized["search_seconds"]),
                "objective": "共享物理信道 + 干扰感知稀疏LMMSE的SER",
                "snr_definition": str(base_kwargs.get("snr_definition", "Eb/N0")),
                "significant": bool(gain_db >= 0.5),
                "significance_note": (
                    f"预测增益{gain_db:.2f} dB；实时控制器仍需满足CSI窗口、"
                    "连续稳定判决和冷却时间才会真正切换。"
                ),
                "references": refs,
                "channel_summary": tb0.get_channel_summary(),
                "auto_reason": reason,
            }
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

        self.adaptive_plots.set_snr_title(
            "共享信道稀疏LMMSE SER-SNR（FDIDM每个SNR独立优化α/β）"
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
        # so the adaptive-process SNR plot no longer waits for the slow
        # left-bottom Monte-Carlo alpha sweep.
        self._alpha_thread.start()
        self._snr_thread.start()

    def _alpha_ser_worker(self, token: int, base_kwargs: dict, stop_event: threading.Event,
                          alpha_sweep_frames: int, alpha_curve_mode: str, best_a: float, best_b: float):
        """Worker 1: draw left-bottom SER-vs-alpha curves with alpha step fixed at 0.1."""
        finish_reason = "完成"
        try:
            from waveform_sim.simulation.simple_fdidm_rx import FDIDMTransceiver
            from waveform_sim.simulation.shared_waveform_benchmark import SharedWaveformBenchmark
            current_b = float(base_kwargs.get("beta", 0.0))
            fixed_tb = FDIDMTransceiver(**copy_kwargs_with(
                base_kwargs,
                dynamic_channel=False,
                channel_dynamics="fixed",
                channel_seed=int(base_kwargs.get("channel_seed", 42)),
            ))
            practical_benchmark = SharedWaveformBenchmark.from_backend(fixed_tb)
            # Requirement: alpha increases by 0.1 for every plotted point.
            alpha_values = np.round(np.arange(0.0, 2.0 + 0.05, 0.1), 10)
            beta_values = [round(float(b), 10) for b in self.ALPHA_SWEEP_BETAS]
            work_snr = float(base_kwargs.get("snr_db", 10.0))
            alpha_seed_base = int(base_kwargs.get("channel_seed", 42)) + 303_2026
            ser_floor = float(self._alpha_ser_floor_by_token.get(
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
                    if mode == "theory":
                        # Use the same fixed channel and practical receiver model as
                        # the SNR comparison and slow adaptive controller.
                        item = practical_benchmark.evaluate(
                            "FDIDM", work_snr, alpha=float(a), beta=float(b)
                        )
                        ser_raw = float(item.get("ser", np.nan))
                    else:
                        tb_alpha = FDIDMTransceiver(**copy_kwargs_with(
                            base_kwargs, alpha=float(a), beta=float(b), snr_db=work_snr
                        ))
                        item = tb_alpha.estimate_ser(
                            num_frames=int(alpha_sweep_frames),
                            seed=alpha_seed_base,  # common random numbers across all α/β points
                            stop_event=stop_event,
                            ser_display_floor=ser_floor,
                        )
                        ser_raw = float(item.get("ser", np.nan))
                    if not self._emit_signal_safe("alpha_beta_point", token, curve_name, float(a), float(ser_raw), float(ser_floor)):
                        return
        except Exception as exc:
            finish_reason = f"失败：{exc}"
        self._emit_signal_safe("alpha_beta_finished", token, finish_reason)

    def _theory_snr_worker(self, token: int, base_kwargs: dict, stop_event: threading.Event,
                           best_a: float, best_b: float):
        """Four waveform domains on one fixed physical channel realization.

        OFDM, OTFS and AFDM are no longer aliases for three FDIDM alpha/beta
        points.  A shared-channel evaluator builds their own transform-domain
        equivalent channels and uses waveform-appropriate complexity-limited
        receivers.  FDIDM is re-optimized independently at every SNR point.
        """
        finish_reason = "完成"
        try:
            from waveform_sim.simulation.simple_fdidm_rx import FDIDMTransceiver
            from waveform_sim.simulation.shared_waveform_benchmark import SharedWaveformBenchmark

            sweep_kwargs = copy_kwargs_with(
                base_kwargs,
                alpha=0.0,
                beta=0.0,
                dynamic_channel=False,
                channel_dynamics="fixed",
                # Keep the user-selected seed.  Do not cherry-pick a realization
                # merely to make one waveform look better.
                channel_seed=int(base_kwargs.get("channel_seed", 42)),
            )
            tb_model = FDIDMTransceiver(**sweep_kwargs)
            benchmark = SharedWaveformBenchmark.from_backend(tb_model)
            previous_a = float(best_a) if np.isfinite(float(best_a)) else 0.0
            previous_b = float(best_b) if np.isfinite(float(best_b)) else 0.0
            coarse_step = max(0.25, float(base_kwargs.get("search_step", 0.1)))
            fine_step = min(0.10, max(0.05, coarse_step / 5.0))

            for snr in self.SER_SNR_POINTS:
                if stop_event.is_set() or token != self._auto_scan_token:
                    finish_reason = "已停止"
                    self._emit_signal_safe("ser_snr_finished", token, finish_reason)
                    return

                for name in ("OFDM", "OTFS", "AFDM"):
                    item = benchmark.evaluate(name, float(snr))
                    ser = float(item.get("ser", np.nan))
                    if not np.isfinite(ser) or ser <= 0:
                        ser = float("nan")
                    if not self._emit_signal_safe(
                        "ser_snr_point", token, name, float(snr), ser,
                        float("nan"), float("nan"),
                    ):
                        return

                result = SharedWaveformBenchmark.optimize_fdidm_over_ensemble(
                    [benchmark],
                    snr_db=float(snr),
                    current_alpha=previous_a,
                    current_beta=previous_b,
                    coarse_step=coarse_step,
                    fine_step=fine_step,
                    # SER-SNR research plot keeps the raw theoretical ordering;
                    # the live controller uses a higher practical decision floor.
                    decision_ser_floor=1e-15,
                    stop_event=stop_event,
                )
                a_star = float(result["recommended_alpha"])
                b_star = float(result["recommended_beta"])
                ser = float(result["predicted_ser_best"])
                previous_a, previous_b = a_star, b_star
                if not np.isfinite(ser) or ser <= 0:
                    ser = float("nan")
                if not self._emit_signal_safe(
                    "ser_snr_point", token, "FDIDM", float(snr), ser,
                    a_star, b_star,
                ):
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
        """Clear only the adaptive-process SER-SNR chart."""
        if hasattr(self, "adaptive_plots"):
            self.adaptive_plots.clear_snr()


    def _on_qt_destroyed(self, *_args):
        """Stop background work when the Qt object is being destroyed."""
        self._qt_alive = False
        try:
            self.adaptive_timer.stop()
        except Exception:
            pass
        if getattr(self, "tb", None) is not None:
            try:
                self.tb.stop_adaptive_tuning()
            except Exception:
                pass
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
        self.update_info_panel(status_text="已停止当前自动刷新。后续参数更新会再次自动触发搜索、左下角α-SER与自适应过程页SER-SNR绘制。")

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
                old + f"\n\n自动刷新{reason}：左下角α-SER与自适应过程页SER-SNR采用独立线程绘制，曲线已按当前token更新。"
            )

    @staticmethod
    def _weighted_path_statistics(channel):
        """Calculate gain-weighted instantaneous delay/Doppler statistics."""
        paths = list((channel or {}).get("paths", []) or [])
        if not paths:
            return {
                "delay_mean_ns": float("nan"),
                "delay_spread_ns": float("nan"),
                "max_delay_ns": float("nan"),
                "doppler_mean_hz": float("nan"),
                "doppler_spread_hz": float("nan"),
                "max_doppler_hz": float((channel or {}).get("max_doppler_hz", np.nan)),
                "path_gain_power": float("nan"),
            }

        delays = np.asarray([float(p.get("delay_ns", 0.0)) for p in paths], dtype=float)
        dopplers = np.asarray([float(p.get("doppler_hz", 0.0)) for p in paths], dtype=float)
        gains = np.asarray([max(float(p.get("gain_abs", 0.0)), 0.0) for p in paths], dtype=float)
        powers = gains * gains
        power_sum = float(np.sum(powers))
        if not np.isfinite(power_sum) or power_sum <= 1e-20:
            weights = np.full(len(paths), 1.0 / max(len(paths), 1), dtype=float)
            power_sum = float("nan")
        else:
            weights = powers / power_sum

        delay_mean = float(np.sum(weights * delays))
        doppler_mean = float(np.sum(weights * dopplers))
        delay_rms = float(np.sqrt(max(np.sum(weights * (delays - delay_mean) ** 2), 0.0)))
        doppler_rms = float(np.sqrt(max(np.sum(weights * (dopplers - doppler_mean) ** 2), 0.0)))
        return {
            "delay_mean_ns": delay_mean,
            "delay_spread_ns": delay_rms,
            "max_delay_ns": float(np.max(delays)),
            "doppler_mean_hz": doppler_mean,
            "doppler_spread_hz": doppler_rms,
            "max_doppler_hz": float(np.max(np.abs(dopplers))),
            "path_gain_power": power_sum,
        }

    def _build_realtime_link_state(self, adaptive_status, channel, metrics):
        """Build current-frame state instead of repeating static configuration."""
        channel = dict(channel or {})
        metrics = dict(metrics or {})
        adaptive_status = dict(adaptive_status or {})
        path_stats = self._weighted_path_statistics(channel)

        signal_power = float(metrics.get("avg_H_row_power", np.nan))
        measured_noise = float(metrics.get("measured_noise_var", np.nan))
        if (
            np.isfinite(signal_power) and signal_power > 0
            and np.isfinite(measured_noise) and measured_noise > 0
        ):
            effective_snr_db = 10.0 * np.log10(signal_power / measured_noise)
        else:
            effective_snr_db = float("nan")

        return {
            "configured_snr_db": float(metrics.get("ebn0_db", self.snr_spin.value())),
            "effective_snr_db": float(effective_snr_db),
            **path_stats,
            "noise_power": float(metrics.get("noise_var", np.nan)),
            "measured_noise_power": float(metrics.get("measured_noise_var", np.nan)),
            "eq_noise_power": float(metrics.get("measured_eq_noise_var", np.nan)),
            "expected_eq_noise_power": float(metrics.get("expected_eq_noise_var", np.nan)),
            "evm_percent": float(metrics.get("evm_percent", np.nan)),
            "condition_number": float(metrics.get("condition_number", np.nan)),
            "channel_type": str(channel.get("ntn_profile", channel.get("channel_model", self.channel_combo.currentText()))),
            "channel_mode": str(self.channel_dynamics_combo.currentText()),
            "absolute_doppler_shift_hz": float(channel.get(
                "absolute_doppler_shift_hz", metrics.get("absolute_doppler_shift_hz", np.nan)
            )),
            "residual_common_cfo_hz": float(channel.get(
                "residual_common_cfo_hz", metrics.get("residual_common_cfo_hz", np.nan)
            )),
            "residual_doppler_spread_hz": float(channel.get(
                "residual_doppler_spread_hz", metrics.get("residual_doppler_spread_hz", np.nan)
            )),
            "doppler_compensation_ratio": float(channel.get(
                "doppler_compensation_ratio", metrics.get("doppler_compensation_ratio", np.nan)
            )),
            "coherence_time_s": float(channel.get(
                "coherence_time_s", metrics.get("coherence_time_s", np.nan)
            )),
            "channel_seed": int(channel.get(
                "seed", metrics.get("channel_seed", self.seed_spin.value())
            )),
            "frame": int(metrics.get("frames", 0)),
            "physical_time_s": float(channel.get("physical_time_s", np.nan)),
            "physical_frame_duration_s": float(channel.get("physical_frame_duration_s", np.nan)),
            "sample_rate_hz": float(channel.get("sample_rate_hz", np.nan)),
            "bandwidth_hz": float(channel.get(
                "bandwidth_hz", self.m_spin.value() * self.scs_spin.value() * 1e3
            )),
            "normalized_doppler": float(channel.get("normalized_doppler", np.nan)),
            "channel_power_db": float(metrics.get(
                "channel_power_db", channel.get("channel_power_db", np.nan)
            )),
            "channel_matrix_change_norm": float(metrics.get(
                "channel_matrix_change_norm", channel.get("channel_matrix_change_norm", np.nan)
            )),
            "channel_matrix_correlation": float(metrics.get(
                "channel_matrix_correlation", channel.get("channel_matrix_correlation", np.nan)
            )),
            "velocity_kmh": float(self.velocity_combo.currentText()),
            "radial_projection": float(self.radial_factor_spin.value()),
            "alpha": float(adaptive_status.get(
                "current_alpha", metrics.get("alpha", self.alpha_spin.value())
            )),
            "beta": float(adaptive_status.get(
                "current_beta", metrics.get("beta", self.beta_spin.value())
            )),
        }

    def _refresh_time_metric_plot(self, history):
        """Display decimated, log-domain-EMA waveform metrics over the last 30 s.

        The backend continues processing every frame.  It emits at most one
        display metric every configured display interval, so GUI point rate and
        physical channel update rate are independent.
        """
        metric_events = [dict(h) for h in (history or []) if h.get("kind") == "metric"]
        switches = [dict(h) for h in (history or []) if h.get("kind") == "switch"]
        if not metric_events:
            return

        if (len(metric_events) < self._time_plot_eval_count
                or len(switches) < self._time_switch_rendered):
            self._time_plot_eval_count = -1
            self._time_switch_rendered = 0
            self._time_plot_x_view = None
            self._time_plot_y_fit = None
            for item in self._time_switch_labels:
                try:
                    self.ber_snr_plot.removeItem(item)
                except Exception:
                    pass
            self._time_switch_labels = []
            for curve in self._time_metric_curves.values():
                curve.setData([], [])

        if (len(metric_events) == self._time_plot_eval_count
                and len(switches) == self._time_switch_rendered):
            return

        ts0 = min(float(h.get("ts", 0.0)) for h in metric_events)
        x = np.asarray([max(0.0, float(h.get("ts", 0.0)) - ts0) for h in metric_events], dtype=float)
        series = {
            "OFDM": [h.get("ser_ofdm", np.nan) for h in metric_events],
            "OTFS": [h.get("ser_otfs", np.nan) for h in metric_events],
            "AFDM": [h.get("ser_afdm", np.nan) for h in metric_events],
            "FDIDM": [h.get("ser_fdidm", np.nan) for h in metric_events],
        }
        styles = {
            "OFDM": ((0, 114, 189), "o"),
            "OTFS": ((217, 83, 25), "s"),
            "AFDM": ((119, 172, 48), "t"),
            "FDIDM": ((126, 47, 142), "d"),
        }
        self.ber_snr_plot.setTitle("共享动态信道实时SER（显示降采样 + log-EMA）")
        plotted = {}
        for name, values in series.items():
            y = np.asarray([
                float(v) if v is not None and np.isfinite(float(v)) and float(v) > 0 else np.nan
                for v in values
            ], dtype=float)
            plotted[name] = y
            curve = self._time_metric_curves.get(name)
            if curve is None:
                color, symbol = styles[name]
                pen = pg.mkPen(color, width=2.3 if name == "FDIDM" else 1.6)
                curve = self.ber_snr_plot.plot(
                    [], [], pen=pen, symbol=symbol, symbolSize=6,
                    symbolPen=pen, symbolBrush="w", name=name,
                )
                self._time_metric_curves[name] = curve
            curve.setData(x, y)

        window = float(self.TIME_WINDOW_SEC)
        step = float(self.TIME_WINDOW_STEP_SEC)
        t_max = float(x[-1]) if len(x) else 0.0
        if t_max <= window:
            view = (0.0, window)
        else:
            right = float(np.ceil(t_max / step) * step)
            view = (right - window, right)
        x_view_changed = (
            self._time_plot_x_view is None
            or abs(view[0] - self._time_plot_x_view[0]) > 1e-9
            or abs(view[1] - self._time_plot_x_view[1]) > 1e-9
        )
        if x_view_changed:
            self.ber_snr_plot.setXRange(*view, padding=0)
            self._time_plot_x_view = view

        try:
            mask = (x >= view[0]) & (x <= view[1])
            valid_parts = [np.asarray(y)[mask] for y in plotted.values()]
            valid = np.concatenate(valid_parts) if valid_parts else np.zeros(0)
            valid = valid[np.isfinite(valid) & (valid > 0)]
            if valid.size:
                lo = float(np.log10(np.min(valid)))
                hi = float(np.log10(np.max(valid)))
                fit_lo = max(lo - 0.30, self.SER_Y_EXPAND_CLAMP[0])
                fit_hi = min(hi + 0.30, self.SER_Y_EXPAND_CLAMP[1])
                if fit_hi - fit_lo < 0.7:
                    mid = 0.5 * (fit_lo + fit_hi)
                    fit_lo, fit_hi = mid - 0.35, mid + 0.35
                if x_view_changed or self._time_plot_y_fit is None:
                    self.ber_snr_plot.setYRange(fit_lo, fit_hi, padding=0)
                    self._time_plot_y_fit = (fit_lo, fit_hi)
                else:
                    old_lo, old_hi = self._time_plot_y_fit
                    if lo < old_lo - 0.25 or hi > old_hi + 0.25:
                        self.ber_snr_plot.setYRange(fit_lo, fit_hi, padding=0)
                        self._time_plot_y_fit = (fit_lo, fit_hi)
        except Exception:
            pass

        for switch_index in range(self._time_switch_rendered, len(switches)):
            sw = switches[switch_index]
            t = max(0.0, float(sw.get("ts", ts0)) - ts0)
            idx = int(np.argmin(np.abs(x - t))) if len(x) else 0
            y = float(plotted["FDIDM"][idx]) if len(x) else float("nan")
            if not np.isfinite(y) or y <= 0:
                continue
            label = pg.TextItem(
                text=f"({float(sw.get('to_alpha', 0.0)):g}, {float(sw.get('to_beta', 0.0)):g})",
                anchor=(0.0, 1.15 if switch_index % 2 == 0 else -0.15),
                color=(126, 47, 142),
            )
            label.setFont(QFont("Microsoft YaHei", 8))
            label.setPos(t, float(np.log10(max(y, 1e-300))))
            self.ber_snr_plot.addItem(label)
            self._time_switch_labels.append(label)
            self._time_switch_rendered = switch_index + 1

        self._time_plot_eval_count = len(metric_events)

    def _format_search_text(self, result):
        """Compact explanation for the fixed-snapshot practical search."""
        channel = dict(result.get("channel_summary", {}) or {})
        refs = dict(result.get("references", {}) or {})
        best_a = float(result.get("best_alpha", np.nan))
        best_b = float(result.get("best_beta", np.nan))

        def metric_ser(name):
            try:
                return float(refs.get(name, {}).get("ser", np.nan))
            except Exception:
                return float("nan")

        lines = [
            (
                f"当前SNR参数搜索完成：α*={best_a:g}, β*={best_b:g}；"
                f"粗/细步长={float(result.get('step', np.nan)):.2f}/"
                f"{float(result.get('fine_step', np.nan)):.2f}；"
                f"候选={int(result.get('candidate_count', 0))}；"
                f"耗时={float(result.get('elapsed_s', np.nan)):.2f}s。"
            ),
            (
                f"固定信道快照：{channel.get('ntn_profile', channel.get('channel_model', '--'))}，"
                f"seed={channel.get('seed', '--')}；带宽="
                f"{float(channel.get('bandwidth_hz', np.nan))/1e6:.3f} MHz；"
                f"轨道公共频移≈{float(channel.get('absolute_doppler_shift_hz', 0.0)):.0f} Hz；"
                f"残余扩展={float(channel.get('residual_doppler_spread_hz', 0.0)):.0f} Hz。"
            ),
            (
                f"同一物理信道、独立波形域SER：OFDM={metric_ser('OFDM'):.3e}，"
                f"OTFS={metric_ser('OTFS'):.3e}，AFDM={metric_ser('AFDM'):.3e}，"
                f"当前FDIDM={metric_ser('FDIDM'):.3e}；"
                f"搜索后FDIDM={float(result.get('predicted_ser_best', np.nan)):.3e}。"
            ),
            (
                f"预测增益={float(result.get('predicted_improvement_db', 0.0)):.2f} dB。"
                "此结果只用于当前快照预览；实时自适应还必须通过CSI窗口、"
                "连续稳定判决、增益门限与切换冷却。"
            ),
            (
                "自适应过程页：每个SNR独立搜索α/β并在FDIDM点旁标注；"
                "仿真图右下角：只显示当前已应用参数的时间SER，真实switch才标注。"
            ),
        ]
        note = str(result.get("significance_note", "")).strip()
        if note:
            lines.append("提示：" + note)
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

    def _on_ser_snr_point(self, token, name, snr, ser, alpha, beta):
        if int(token) != int(self._auto_scan_token):
            return
        self.adaptive_plots.add_snr_point(
            str(name), float(snr), float(ser), float(alpha), float(beta)
        )

    def _on_ser_snr_finished(self, token, reason):
        if int(token) != int(self._auto_scan_token):
            return
        self._snr_active_token = None
        self.adaptive_plots.set_snr_title(f"SER-SNR 性能对比（{reason}）")
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
            self._update_channel_plot()
            self._update_info_panel(running=True)
        except Exception:
            pass

    def _update_channel_plot(self):
        try:
            tb = self.tb if self.tb is not None else self._new_backend()
            summary = tb.get_channel_summary()
            paths = list(summary.get("paths", []) or [])
            if not paths:
                self.channel_scatter.setData([], [])
                return

            delays = np.asarray([float(p.get("delay_ns", 0.0)) for p in paths], dtype=float)
            dopplers = np.asarray([float(p.get("doppler_hz", 0.0)) for p in paths], dtype=float)
            gains = np.asarray([max(float(p.get("gain_abs", 0.0)), 0.0) for p in paths], dtype=float)
            gain_max = float(np.max(gains)) if gains.size else 0.0
            if gain_max > 1e-15:
                sizes = 7.0 + 15.0 * np.sqrt(gains / gain_max)
            else:
                sizes = np.full(len(paths), 10.0, dtype=float)

            spots = [
                {
                    "pos": (float(delays[i]), float(dopplers[i])),
                    "size": float(sizes[i]),
                    "data": {
                        "path": int(paths[i].get("path", i)),
                        "gain_abs": float(gains[i]),
                    },
                }
                for i in range(len(paths))
            ]
            self.channel_scatter.setData(spots)

            try:
                metrics = tb.get_last_metrics()
            except Exception:
                metrics = {}
            mode = str(summary.get("channel_dynamics", self._selected_channel_dynamics()))
            frame = int(metrics.get("frames", 0))
            self.spectrum_plot.setTitle(
                f"LEO Delay-Doppler：{summary.get('channel_model', '--')} / "
                f"{mode}；seed={summary.get('seed', '--')}；frame={frame}"
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

    @staticmethod
    def _trailing_mean(y, n):
        """尾部滑动平均：第 i 点只依赖 <=i 的数据，历史点不会随新点重算而漂移。"""
        y = np.asarray(y, dtype=np.float64)
        if n <= 1 or y.size <= 1:
            return y
        n = int(n)
        cum = np.cumsum(np.where(np.isfinite(y), y, 0.0))
        cnt = np.cumsum(np.isfinite(y))
        out = np.full_like(y, np.nan)
        for i in range(y.size):
            j0 = max(0, i - n + 1)
            s = cum[i] - (cum[j0 - 1] if j0 > 0 else 0.0)
            c = cnt[i] - (cnt[j0 - 1] if j0 > 0 else 0)
            if c > 0:
                out[i] = s / float(c)
        return out

    def _update_info_panel(self, running=False):
        metrics = {}
        channel = {}
        adaptive_status = {}
        if self.tb is not None:
            try:
                metrics = self.tb.get_last_metrics()
            except Exception:
                metrics = {}
            try:
                channel = self.tb.get_channel_summary()
            except Exception:
                channel = {}
            try:
                adaptive_status = self.tb.get_adaptive_status()
            except Exception:
                adaptive_status = {}

        state = self._build_realtime_link_state(adaptive_status, channel, metrics)
        alpha = float(state.get("alpha", self.alpha_spin.value()))
        beta = float(state.get("beta", self.beta_spin.value()))
        running_text = "运行中" if running else "未启动"
        if running and self._selected_channel_dynamics() == "fixed":
            running_text += "（固定信道）"
        elif running:
            running_text += "（信道变化中）"

        self.update_info_panel(
            metrics={
                "运行状态": (
                    f"{running_text}；frame={state.get('frame', 0)}；"
                    f"物理时间={self._fmt(float(state.get('physical_time_s', np.nan))*1e3)} ms；"
                    f"seed={state.get('channel_seed', '--')}"
                ),
                "Eb/N0": (
                    f"配置={self._fmt(state.get('configured_snr_db'))} dB；"
                    f"有效={self._fmt(state.get('effective_snr_db'))} dB"
                ),
                # Generic base-panel slots are reused as separate channel-state rows.
                "BER": (
                    f"轨道公共={self._fmt(state.get('absolute_doppler_shift_hz'))} Hz；"
                    f"补偿后CFO={self._fmt(state.get('residual_common_cfo_hz'))} Hz"
                ),
                "FER": (
                    f"均值={self._fmt(state.get('delay_mean_ns'))} ns；"
                    f"RMS={self._fmt(state.get('delay_spread_ns'))} ns；"
                    f"最大={self._fmt(state.get('max_delay_ns'))} ns"
                ),
                "同步度量": (
                    f"配置最大={self._fmt(state.get('residual_doppler_spread_hz'))} Hz；"
                    f"当前RMS={self._fmt(state.get('doppler_spread_hz'))} Hz；"
                    f"Tc≈{self._fmt(float(state.get('coherence_time_s', np.nan))*1e3)} ms"
                ),
                "CFO估计": (
                    f"目标噪声={self._fmt(state.get('noise_power'))}；"
                    f"实测={self._fmt(state.get('measured_noise_power'))}；"
                    f"均衡后={self._fmt(state.get('eq_noise_power'))}<br>"
                    f"H功率={self._fmt(state.get('channel_power_db'))} dB；"
                    f"相邻帧相关={self._fmt(state.get('channel_matrix_correlation'))}；"
                    f"相对变化={self._fmt(state.get('channel_matrix_change_norm'))}<br>"
                    f"{state.get('channel_type', '--')} / {state.get('channel_mode', '--')}"
                ),
            },
            config={
                "波形": "FDIDM",
                "FFT/CP": f"M×N={self.m_spin.value()}×{self.n_spin.value()}；无显式CP",
                "调制": self.mod_combo.currentText(),
                "采样率": f"{self.m_spin.value()*self.scs_spin.value()/1000:.2f} MHz",
                "数据符号": f"{self.m_spin.value()*self.n_spin.value()}",
                "星座观察": (
                    f"当前α/β=({alpha:g}, {beta:g})；"
                    f"v={float(self.velocity_combo.currentText())/3600:.2f} km/s"
                ),
            },
        )
        if not running and self.status_text_label is not None and not self._last_search_result:
            self.status_text_label.setText(
                "软波形仿真：连续多普勒默认开启。实时链路状态按帧显示有效SNR、"
                "增益加权多普勒、时延扩展、实测噪声和当前信道seed。\n"
                "时间图中的FDIDM使用当前已应用α/β的SER；只有真实switch事件才标注参数。"
            )

    def closeEvent(self, event):
        try:
            self._stop_ser_snr_sweep()
            self._on_stop_clicked()
        finally:
            super().closeEvent(event)
