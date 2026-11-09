# -*- coding: utf-8 -*-
"""
ui/hardware_test_tab.py

OFDM / OTFS / AFDM 通用 USRP 硬件测评页面。

本文件必须导出 ``HardwareTestTab``，因为 ``waveform_sim.ui.main_window`` 通过
``waveform_sim.ui.hardware_test_tab.HardwareTestTab`` 动态加载该页面。
后端在用户点击“连接 / 配置 USRP”时才按需导入和实例化，避免仅打开主界面
就占用 USRP，亦便于在缺少 GNU Radio / UHD 时给出清晰的界面错误信息。
"""

from __future__ import annotations

import gc
import importlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QLayout,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.config import ExperimentConfig, HardwareConfig, WaveformConfig
from ..hardware.hardware_scan import NTNScanPlan, NTN_SCAN_PARAMETERS
from ..service.hardware_run_recorder import HardwareRunRecorder
from ..simulation.leo_ntn_channel import get_profile


_BACKEND_SPECS: Dict[str, Tuple[str, str]] = {
    "OFDM": ("waveform_sim.hardware.ofdm_hardtest", "OfdmHardwareTx"),
    "OTFS": ("waveform_sim.hardware.otfs_hardtest", "OTFSHardwareTest"),
    "AFDM": ("waveform_sim.hardware.afdm_hardtest", "AFDMHardwareTest"),
}

_DEFAULT_TEXTS = {
    "OFDM": "Hello OFDM Hardware Test!",
    "OTFS": "Hello OTFS Hardware Test!",
    "AFDM": "Hello AFDM Hardware Test!",
}

_CONSTELLATION_MODES = [
    ("决策整形（推荐）", "dd_refined"),
    ("原始软符号", "raw"),
    ("硬判决", "hard_decision"),
    ("均衡前符号", "pre_equalized"),
]

MATLAB_BLUE = (0, 114, 189)
MATLAB_ORANGE = (217, 83, 25)
MATLAB_YELLOW = (237, 177, 32)
MATLAB_PURPLE = (126, 47, 142)
LIGHT_BG = (250, 250, 250)
AXIS_COLOR = (60, 60, 60)
BORDER_COLOR = (225, 225, 225)



from .ui_utils import compute_spectrum, format_metric, has_signal, safe_float


class _HardwareStatusText(QPlainTextEdit):
    """A bounded status viewport that keeps complete device errors readable."""

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(54)
        self.setText(text)

    def setText(self, text):
        value = str(text)
        # Preserve the operator's selection/scroll position during identical
        # 100 ms refreshes. New status starts at its first line.
        if self.toPlainText() != value:
            self.setPlainText(value)
            self.verticalScrollBar().setValue(0)
        self.setToolTip(value)

    def text(self):
        return self.toPlainText()


class HardwareTestTab(QWidget):
    """OFDM、OTFS、AFDM 共用的 USRP 真机测试页面。"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.backend = None
        self.backend_waveform: Optional[str] = None
        self._backend_identity: Optional[Tuple[str, str, str]] = None
        self.test_running = False
        self._backend_started_once = False
        self._configuration_dirty = False
        self._initializing = True
        self._last_status_error = ""
        self._last_runtime_log_time = 0.0
        self._last_plot_samp_rate: Optional[float] = None
        self._hardware_recorder: Optional[HardwareRunRecorder] = None
        self._last_run_status: Dict[str, Any] = {}
        self._scan_plan: Optional[NTNScanPlan] = None
        self._scan_values = []
        self._scan_index = 0
        self._scan_start_frame_id = 0
        self._scan_active = False
        self._scan_transitioning = False

        self._init_ui()
        self._init_plot_style()
        self._connect_signals()

        self.update_timer = QTimer(self)
        self.update_timer.setInterval(100)
        self.update_timer.timeout.connect(self._refresh_plots)

        self._on_waveform_changed(self.waveform_combo.currentText())
        self._initializing = False
        self._configuration_dirty = False
        self._update_button_states()

    # =========================================================
    # UI
    # =========================================================
    def _init_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.main_splitter = splitter
        splitter.addWidget(self._create_controls_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setSizes([360, 1000])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    def _create_controls_panel(self) -> QWidget:
        rail = QWidget()
        rail.setMinimumWidth(325)
        rail.setMaximumWidth(410)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(0, 0, 0, 0)
        rail_layout.setSpacing(6)
        panel = QWidget()
        self.controls_panel = panel
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # ---------- 波形与算法 ----------
        waveform_group = QGroupBox("测试波形")
        waveform_grid = QGridLayout(waveform_group)
        waveform_grid.setHorizontalSpacing(8)
        waveform_grid.setVerticalSpacing(8)

        self.waveform_combo = QComboBox()
        self.waveform_combo.addItems(list(_BACKEND_SPECS.keys()))

        self.mod_order_combo = QComboBox()
        self.mod_order_combo.addItems(["QPSK", "16QAM", "64QAM"])

        self.equalizer_combo = QComboBox()
        self.equalizer_combo.addItems(["MMSE", "ZF"])

        self.const_mode_combo = QComboBox()
        for label, _ in _CONSTELLATION_MODES:
            self.const_mode_combo.addItem(label)

        self.tx_plot_combo = QComboBox()
        self.tx_plot_combo.addItems(["TX 基带频谱", "TX 时域幅度"])

        waveform_grid.addWidget(QLabel("测试波形"), 0, 0)
        waveform_grid.addWidget(self.waveform_combo, 0, 1)
        layout.addWidget(waveform_group)

        self.config_tabs = QTabWidget()
        self.config_tabs.setMinimumHeight(160)
        self.config_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.config_tabs, 1)
        self.config_scrolls = []
        link_page = QWidget()
        link_layout = QVBoxLayout(link_page)
        waveform_page = QWidget()
        waveform_layout = QVBoxLayout(waveform_page)
        display_page = QWidget()
        display_layout = QVBoxLayout(display_page)
        for title, content, content_layout in (
            ("链路配置", link_page, link_layout),
            ("波形与文本", waveform_page, waveform_layout),
            ("显示", display_page, display_layout),
        ):
            content_layout.setContentsMargins(5, 5, 5, 5)
            content_layout.setSpacing(8)
            # A short tab viewport scrolls real content instead of squeezing
            # group frames and controls below their minimum content height.
            content_layout.setSizeConstraint(QLayout.SetMinAndMaxSize)
            content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setFrameShape(QScrollArea.NoFrame)
            scroll.setWidget(content)
            self.config_scrolls.append(scroll)
            self.config_tabs.addTab(scroll, title)
        self.controls_scroll = self.config_scrolls[0]
        self.config_tabs.currentChanged.connect(self._on_config_tab_changed)

        modem_group = QGroupBox("解调参数")
        modem_grid = QGridLayout(modem_group)
        modem_grid.addWidget(QLabel("调制方式"), 0, 0)
        modem_grid.addWidget(self.mod_order_combo, 0, 1)
        modem_grid.addWidget(QLabel("均衡器"), 1, 0)
        modem_grid.addWidget(self.equalizer_combo, 1, 1)
        waveform_layout.addWidget(modem_group)

        display_group = QGroupBox("图表显示")
        display_grid = QGridLayout(display_group)
        display_grid.addWidget(QLabel("星座显示"), 0, 0)
        display_grid.addWidget(self.const_mode_combo, 0, 1)
        display_grid.addWidget(QLabel("TX 图"), 1, 0)
        display_grid.addWidget(self.tx_plot_combo, 1, 1)
        display_layout.addWidget(display_group)
        display_layout.addStretch(1)

        # ---------- USRP ----------
        hw_group = QGroupBox("USRP 硬件配置")
        hw_grid = QGridLayout(hw_group)
        hw_grid.setHorizontalSpacing(8)
        hw_grid.setVerticalSpacing(8)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["USRP B210", "USRP N210", "USRP X310"])

        self.serial_edit = QLineEdit()
        self.serial_edit.setPlaceholderText("留空则自动选择设备")

        self.samp_rate_spin = self._dspin(100_000, 100_000_000, 960_000, 0, " Hz")
        self.fc_spin = self._dspin(1_000_000, 6_000_000_000, 2_400_000_000, 0, " Hz")
        self.tx_gain_spin = self._dspin(0, 100, 40, 1, " dB", step=1.0)
        self.rx_gain_spin = self._dspin(0, 100, 40, 1, " dB", step=1.0)

        self.tx_antenna_combo = QComboBox()
        self.tx_antenna_combo.setEditable(True)
        self.tx_antenna_combo.addItems(["TX/RX", "TX1", "TX2"])

        self.rx_antenna_combo = QComboBox()
        self.rx_antenna_combo.setEditable(True)
        self.rx_antenna_combo.addItems(["RX2", "TX/RX", "RX1"])

        hw_grid.addWidget(QLabel("设备类型"), 0, 0)
        hw_grid.addWidget(self.device_combo, 0, 1)
        hw_grid.addWidget(QLabel("序列号"), 1, 0)
        hw_grid.addWidget(self.serial_edit, 1, 1)
        hw_grid.addWidget(QLabel("采样率"), 2, 0)
        hw_grid.addWidget(self.samp_rate_spin, 2, 1)
        hw_grid.addWidget(QLabel("中心频率"), 3, 0)
        hw_grid.addWidget(self.fc_spin, 3, 1)
        hw_grid.addWidget(QLabel("发送增益"), 4, 0)
        hw_grid.addWidget(self.tx_gain_spin, 4, 1)
        hw_grid.addWidget(QLabel("接收增益"), 5, 0)
        hw_grid.addWidget(self.rx_gain_spin, 5, 1)
        hw_grid.addWidget(QLabel("TX 天线"), 6, 0)
        hw_grid.addWidget(self.tx_antenna_combo, 6, 1)
        hw_grid.addWidget(QLabel("RX 天线"), 7, 0)
        hw_grid.addWidget(self.rx_antenna_combo, 7, 1)
        link_layout.addWidget(hw_group)

        self.ntn_group = QGroupBox("软件 NTN 信道（发射前注入）")
        ntn_grid = QGridLayout(self.ntn_group)
        ntn_grid.setHorizontalSpacing(8)
        ntn_grid.setVerticalSpacing(6)
        self.ntn_enabled_check = QCheckBox("启用软件信道注入")
        self.ntn_profile_combo = QComboBox()
        self.ntn_profile_combo.addItems(
            ["NTN-TDLC5-200", "NTN-TDLA100-200", "NTN-TDLC5-1200"]
        )
        self.ntn_mode_combo = QComboBox()
        self.ntn_mode_combo.addItem("单点配置", "single")
        self.ntn_mode_combo.addItem("单参数扫描", "scan")
        self.ntn_velocity_spin = self._dspin(0, 40_000, 28_080, 0, " km/h")
        self.ntn_radial_spin = self._dspin(0, 1, 0.5, 3)
        self.ntn_compensation_spin = self._dspin(0, 100, 99.9, 2, " %")
        self.ntn_cfo_spin = self._dspin(-200_000, 200_000, 0, 1, " Hz", step=10)
        self.ntn_doppler_spin = self._dspin(0, 1200, 200, 1, " Hz", step=10)
        self.ntn_delay_scale_spin = self._dspin(0.01, 1000, 1, 2, " 倍", step=0.1)
        self.ntn_seed_spin = QSpinBox()
        self.ntn_seed_spin.setRange(0, 2_000_000_000)
        self.ntn_seed_spin.setValue(42)
        self.ntn_seed_spin.setMinimumHeight(30)

        ntn_grid.addWidget(self.ntn_enabled_check, 0, 0, 1, 2)
        ntn_grid.addWidget(QLabel("Profile"), 1, 0)
        ntn_grid.addWidget(self.ntn_profile_combo, 1, 1)
        ntn_grid.addWidget(QLabel("运行方式"), 2, 0)
        ntn_grid.addWidget(self.ntn_mode_combo, 2, 1)
        for row, label, control in (
            (3, "径向速度", self.ntn_velocity_spin),
            (4, "径向投影", self.ntn_radial_spin),
            (5, "预补偿比例", self.ntn_compensation_spin),
            (6, "额外残余 CFO", self.ntn_cfo_spin),
            (7, "散射多普勒", self.ntn_doppler_spin),
            (8, "时延缩放", self.ntn_delay_scale_spin),
            (9, "随机种子", self.ntn_seed_spin),
        ):
            ntn_grid.addWidget(QLabel(label), row, 0)
            ntn_grid.addWidget(control, row, 1)

        self.ntn_scan_widget = QWidget()
        scan_grid = QGridLayout(self.ntn_scan_widget)
        scan_grid.setContentsMargins(0, 4, 0, 0)
        self.ntn_scan_parameter_combo = QComboBox()
        for parameter, spec in NTN_SCAN_PARAMETERS.items():
            suffix = f" ({spec['unit']})" if spec["unit"] else ""
            self.ntn_scan_parameter_combo.addItem(f"{spec['label']}{suffix}", parameter)
        self.ntn_scan_start_spin = self._dspin(-40_000, 40_000, -200, 2)
        self.ntn_scan_stop_spin = self._dspin(-40_000, 40_000, 200, 2)
        self.ntn_scan_points_spin = QSpinBox()
        self.ntn_scan_points_spin.setRange(2, 101)
        self.ntn_scan_points_spin.setValue(6)
        self.ntn_scan_frames_spin = QSpinBox()
        self.ntn_scan_frames_spin.setRange(1, 10_000)
        self.ntn_scan_frames_spin.setValue(10)
        scan_grid.addWidget(QLabel("扫描参数"), 0, 0)
        scan_grid.addWidget(self.ntn_scan_parameter_combo, 0, 1)
        scan_grid.addWidget(QLabel("起始值"), 1, 0)
        scan_grid.addWidget(self.ntn_scan_start_spin, 1, 1)
        scan_grid.addWidget(QLabel("结束值"), 2, 0)
        scan_grid.addWidget(self.ntn_scan_stop_spin, 2, 1)
        scan_grid.addWidget(QLabel("扫描点数"), 3, 0)
        scan_grid.addWidget(self.ntn_scan_points_spin, 3, 1)
        scan_grid.addWidget(QLabel("每点接收帧数"), 4, 0)
        scan_grid.addWidget(self.ntn_scan_frames_spin, 4, 1)
        ntn_grid.addWidget(self.ntn_scan_widget, 10, 0, 1, 2)
        self.ntn_group.setToolTip(
            "OFDM 基带样点在送入 USRP 发射链之前经过软件 NTN 信道；参考 profile 为简化 SISO 工程模型。"
        )
        self.ntn_compensation_spin.setToolTip("预补偿公共轨道频移的比例；不消除散射多普勒扩展。")
        self.ntn_radial_spin.setToolTip("速度沿视线方向的比例，不代表卫星仰角。")
        self.ntn_delay_scale_spin.setToolTip("1 倍使用 profile 时延；其它值用于工程压力测试。")
        self._sync_ntn_scan_bounds()
        link_layout.addWidget(self.ntn_group)
        link_layout.addStretch(1)

        # ---------- AFDM 专用参数 ----------
        self.afdm_group = QGroupBox("AFDM 专用参数")
        afdm_grid = QGridLayout(self.afdm_group)
        self.c1_spin = self._dspin(-2.0, 2.0, 0.05, 4, "", step=0.01)
        self.c2_spin = self._dspin(-2.0, 2.0, 0.05, 4, "", step=0.01)
        afdm_grid.addWidget(QLabel("c1"), 0, 0)
        afdm_grid.addWidget(self.c1_spin, 0, 1)
        afdm_grid.addWidget(QLabel("c2"), 1, 0)
        afdm_grid.addWidget(self.c2_spin, 1, 1)
        afdm_note = QLabel("仅 AFDM 后端使用；OFDM / OTFS 会忽略该组参数。")
        afdm_note.setWordWrap(True)
        afdm_note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        afdm_grid.addWidget(afdm_note, 2, 0, 1, 2)
        waveform_layout.addWidget(self.afdm_group)

        # ---------- 文本 ----------
        text_group = QGroupBox("发送文本 / 文本文件")
        text_layout = QVBoxLayout(text_group)
        text_buttons = QHBoxLayout()
        self.btn_load_text = QPushButton("加载文本文件")
        self.btn_reset_text = QPushButton("恢复默认文本")
        text_buttons.addWidget(self.btn_load_text)
        text_buttons.addWidget(self.btn_reset_text)

        self.file_path_label = QLabel("可直接编辑下方文本，或加载 .txt 文件")
        self.file_path_label.setWordWrap(True)
        self.file_path_label.setMinimumWidth(0)
        self.file_path_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        self.tx_text_edit = QTextEdit()
        self.tx_text_edit.setPlainText(_DEFAULT_TEXTS["OFDM"])
        self.tx_text_edit.setMinimumHeight(90)
        self.tx_text_edit.setMaximumHeight(110)

        text_layout.addLayout(text_buttons)
        text_layout.addWidget(self.file_path_label)
        text_layout.addWidget(self.tx_text_edit)
        waveform_layout.addWidget(text_group)
        waveform_layout.addStretch(1)

        # ---------- 操作 ----------
        action_group = QGroupBox("运行控制")
        action_grid = QGridLayout(action_group)
        self.btn_connect = QPushButton("连接 / 配置 USRP")
        self.btn_apply = QPushButton("应用当前参数")
        self.btn_start_test = QPushButton("开始测试")
        self.btn_stop_test = QPushButton("停止测试")
        self.btn_disconnect = QPushButton("释放 USRP")
        self.btn_clear_log = QPushButton("清空日志")

        action_grid.addWidget(self.btn_connect, 0, 0)
        action_grid.addWidget(self.btn_apply, 0, 1)
        action_grid.addWidget(self.btn_start_test, 1, 0)
        action_grid.addWidget(self.btn_stop_test, 1, 1)
        action_grid.addWidget(self.btn_disconnect, 2, 0, 1, 2)
        action_grid.setContentsMargins(6, 6, 6, 6)
        action_grid.setSpacing(4)

        rail_layout.addWidget(panel, 1)
        rail_layout.addWidget(action_group)
        self._apply_control_style(rail)
        for group in rail.findChildren(QGroupBox):
            group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
            group.layout().setSizeConstraint(QLayout.SetMinimumSize)
        for control in rail.findChildren(QComboBox):
            control.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            control.setMinimumContentsLength(9)
            control.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for control in rail.findChildren(QLineEdit):
            control.setMinimumWidth(0)
            control.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return rail

    def _on_config_tab_changed(self, index):
        if 0 <= index < len(self.config_scrolls):
            self.controls_scroll = self.config_scrolls[index]

    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.decode_status_label = _HardwareStatusText("解调状态：未连接")
        layout.addWidget(self.decode_status_label)

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        self.results_splitter = splitter
        layout.addWidget(splitter)

        # ---------- 2x2 plots ----------
        plot_panel = QWidget()
        plot_panel.setMinimumSize(470, 330)
        grid = QGridLayout(plot_panel)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        for position in (0, 1):
            grid.setRowStretch(position, 1)
            grid.setColumnStretch(position, 1)

        self.tx_plot = pg.PlotWidget(title="发送端基带显示")
        self.tx_plot.setLabel("left", "幅度 / PSD")
        self.tx_plot.setLabel("bottom", "频率 / 样点")

        self.rx_spectrum_plot = pg.PlotWidget(title="USRP 接收基带频谱")
        self.rx_spectrum_plot.setLabel("left", "PSD", units="dB")
        self.rx_spectrum_plot.setLabel("bottom", "频率", units="Hz")

        self.ber_plot = pg.PlotWidget(title="BER 曲线")
        self.ber_plot.setLabel("left", "log10(BER)")
        self.ber_plot.setLabel("bottom", "运行时间", units="s")

        self.constellation_plot = pg.PlotWidget(title="接收星座图")
        self.constellation_plot.setLabel("left", "Q")
        self.constellation_plot.setLabel("bottom", "I")
        self.constellation_plot.setAspectLocked(True)

        from .fdidm_plot_widgets import _PlotGridCell
        for plot, row, column in (
            (self.tx_plot, 0, 0), (self.rx_spectrum_plot, 0, 1),
            (self.ber_plot, 1, 0), (self.constellation_plot, 1, 1),
        ):
            cell = _PlotGridCell(plot)
            cell.setMinimumSize(230, 160)
            grid.addWidget(cell, row, column)
        splitter.addWidget(plot_panel)

        # Text evidence and logs share one bounded area so the live charts
        # remain the primary view at the minimum supported window size.
        self.result_tabs = QTabWidget()
        self.result_tabs.setMinimumHeight(134)
        self.result_tabs.setMaximumHeight(190)
        splitter.addWidget(self.result_tabs)

        # ---------- 文本对照 ----------
        text_panel = QWidget()
        text_grid = QGridLayout(text_panel)
        text_grid.setContentsMargins(4, 4, 4, 4)
        text_grid.setSpacing(4)

        self.tx_text_view = QTextEdit()
        self.tx_text_view.setReadOnly(True)
        self.tx_text_view.setMaximumHeight(110)
        self.tx_text_view.setPlainText(self.tx_text_edit.toPlainText())

        self.rx_text_view = QTextEdit()
        self.rx_text_view.setReadOnly(True)
        self.rx_text_view.setMaximumHeight(110)

        text_grid.addWidget(QLabel("发送文本"), 0, 0)
        text_grid.addWidget(QLabel("接收解调文本"), 0, 1)
        text_grid.addWidget(self.tx_text_view, 1, 0)
        text_grid.addWidget(self.rx_text_view, 1, 1)
        self.result_tabs.addTab(text_panel, "文本对照")

        # ---------- 日志 ----------
        log_panel = QWidget()
        log_layout = QHBoxLayout(log_panel)
        log_layout.setContentsMargins(4, 4, 4, 4)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.document().setMaximumBlockCount(2000)
        log_layout.addWidget(self.log_text, 1)
        log_layout.addWidget(self.btn_clear_log, 0, Qt.AlignTop)
        self.result_tabs.addTab(log_panel, "测试日志")
        splitter.setSizes([650, 145])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        # curves/items
        self.tx_curve = self.tx_plot.plot(pen=pg.mkPen(MATLAB_BLUE, width=2))
        self.rx_curve = self.rx_spectrum_plot.plot(pen=pg.mkPen(MATLAB_ORANGE, width=2))
        self.ber_curve = self.ber_plot.plot(pen=pg.mkPen(MATLAB_PURPLE, width=2))
        self.constellation_scatter = pg.ScatterPlotItem(
            size=6,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(
                MATLAB_YELLOW[0], MATLAB_YELLOW[1], MATLAB_YELLOW[2], 170
            ),
        )
        self.constellation_plot.addItem(self.constellation_scatter)
        return panel

    def _dspin(
        self,
        lo: float,
        hi: float,
        value: float,
        decimals: int,
        suffix: str = "",
        step: Optional[float] = None,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(float(lo), float(hi))
        spin.setValue(float(value))
        spin.setDecimals(int(decimals))
        spin.setSuffix(str(suffix))
        if step is not None:
            spin.setSingleStep(float(step))
        spin.setMinimumHeight(30)
        return spin

    def _apply_control_style(self, widget: QWidget):
        widget.setStyleSheet(
            """
            QGroupBox {
                font-weight: 600;
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLabel { min-height: 24px; }
            QPushButton { min-height: 32px; padding: 5px 8px; }
            QComboBox, QLineEdit, QDoubleSpinBox { min-height: 28px; }
            """
        )

    def _init_plot_style(self):
        for plot in (
            self.tx_plot,
            self.rx_spectrum_plot,
            self.ber_plot,
            self.constellation_plot,
        ):
            plot.setBackground(LIGHT_BG)
            plot.showGrid(x=True, y=True, alpha=0.35)
            for axis_name in ("left", "bottom"):
                axis = plot.getAxis(axis_name)
                axis.setPen(pg.mkPen(AXIS_COLOR, width=1.0))
                axis.setTextPen(pg.mkPen(AXIS_COLOR, width=1.0))
            plot.getPlotItem().getViewBox().setBorder(
                pg.mkPen(BORDER_COLOR, width=1.0)
            )

        for plot in (self.tx_plot, self.rx_spectrum_plot, self.ber_plot):
            plot.disableAutoRange()
            plot.setMouseEnabled(x=False, y=False)

        self.constellation_plot.disableAutoRange()
        self.constellation_plot.setXRange(-2.0, 2.0, padding=0)
        self.constellation_plot.setYRange(-2.0, 2.0, padding=0)
        self.rx_spectrum_plot.setYRange(-120.0, 20.0, padding=0)
        self.ber_plot.setXRange(0.0, 60.0, padding=0)
        self.ber_plot.setYRange(-6.0, 0.0, padding=0)
        self._apply_stable_plot_ranges(self.samp_rate_spin.value())

    # =========================================================
    # Signals / control state
    # =========================================================
    def _connect_signals(self):
        self.btn_connect.clicked.connect(self._on_connect_clicked)
        self.btn_apply.clicked.connect(self._on_apply_clicked)
        self.btn_start_test.clicked.connect(self._on_start_test_clicked)
        self.btn_stop_test.clicked.connect(self._on_stop_test_clicked)
        self.btn_disconnect.clicked.connect(self._on_disconnect_clicked)
        self.btn_clear_log.clicked.connect(self.log_text.clear)
        self.btn_load_text.clicked.connect(self._on_load_text_clicked)
        self.btn_reset_text.clicked.connect(self._on_reset_text_clicked)

        self.waveform_combo.currentTextChanged.connect(self._on_waveform_changed)
        self.const_mode_combo.currentIndexChanged.connect(
            lambda _index: self._push_constellation_mode()
        )
        self.tx_plot_combo.currentIndexChanged.connect(
            lambda _index: self._refresh_tx_plot_only()
        )
        self.ntn_enabled_check.toggled.connect(self._on_ntn_configuration_changed)
        self.ntn_mode_combo.currentIndexChanged.connect(self._on_ntn_configuration_changed)
        self.ntn_profile_combo.currentTextChanged.connect(self._on_ntn_profile_changed)
        self.ntn_scan_parameter_combo.currentIndexChanged.connect(self._sync_ntn_scan_bounds)

        self.tx_gain_spin.valueChanged.connect(
            lambda value: self._hot_set_gain("tx", float(value))
        )
        self.rx_gain_spin.valueChanged.connect(
            lambda value: self._hot_set_gain("rx", float(value))
        )

        dirty_signals = [
            self.device_combo.currentTextChanged,
            self.serial_edit.textChanged,
            self.samp_rate_spin.valueChanged,
            self.fc_spin.valueChanged,
            self.tx_antenna_combo.currentTextChanged,
            self.rx_antenna_combo.currentTextChanged,
            self.mod_order_combo.currentTextChanged,
            self.equalizer_combo.currentTextChanged,
            self.c1_spin.valueChanged,
            self.c2_spin.valueChanged,
            self.tx_text_edit.textChanged,
            self.ntn_profile_combo.currentTextChanged,
            self.ntn_velocity_spin.valueChanged,
            self.ntn_radial_spin.valueChanged,
            self.ntn_compensation_spin.valueChanged,
            self.ntn_cfo_spin.valueChanged,
            self.ntn_doppler_spin.valueChanged,
            self.ntn_delay_scale_spin.valueChanged,
            self.ntn_seed_spin.valueChanged,
            self.ntn_mode_combo.currentIndexChanged,
            self.ntn_scan_parameter_combo.currentIndexChanged,
            self.ntn_scan_start_spin.valueChanged,
            self.ntn_scan_stop_spin.valueChanged,
            self.ntn_scan_points_spin.valueChanged,
            self.ntn_scan_frames_spin.valueChanged,
        ]
        for signal in dirty_signals:
            signal.connect(self._mark_configuration_dirty)

    def _mark_configuration_dirty(self, *_args):
        if self._initializing:
            return
        self._configuration_dirty = True
        if self.backend is not None and not self.test_running:
            self.decode_status_label.setText(
                "解调状态：参数已修改；点击“应用当前参数”或重新开始测试后生效。"
            )

    def _on_waveform_changed(self, waveform: str):
        waveform = str(waveform).upper()
        self.afdm_group.setVisible(waveform == "AFDM")
        self.ntn_group.setVisible(waveform == "OFDM")

        current_text = self.tx_text_edit.toPlainText().strip()
        known_defaults = {text.strip() for text in _DEFAULT_TEXTS.values()}
        if not current_text or current_text in known_defaults:
            self.tx_text_edit.setPlainText(_DEFAULT_TEXTS.get(waveform, "Hello Hardware Test!"))
            self.tx_text_view.setPlainText(self.tx_text_edit.toPlainText())

        if not self._initializing:
            self._mark_configuration_dirty()
        self._update_ntn_controls()

    def _on_ntn_profile_changed(self, profile: str):
        try:
            self.ntn_doppler_spin.setValue(float(get_profile(profile).maximum_doppler_hz))
        except Exception:
            pass
        self._on_ntn_configuration_changed()

    def _on_ntn_configuration_changed(self, *_args):
        if not self._initializing:
            self._mark_configuration_dirty()
        self._update_ntn_controls()

    def _sync_ntn_scan_bounds(self, *_args):
        parameter = self.ntn_scan_parameter_combo.currentData()
        spec = NTN_SCAN_PARAMETERS.get(parameter)
        if spec is None:
            return
        self.ntn_scan_start_spin.blockSignals(True)
        self.ntn_scan_stop_spin.blockSignals(True)
        self.ntn_scan_start_spin.setRange(spec["minimum"], spec["maximum"])
        self.ntn_scan_stop_spin.setRange(spec["minimum"], spec["maximum"])
        self.ntn_scan_start_spin.setValue(spec["start"])
        self.ntn_scan_stop_spin.setValue(spec["stop"])
        decimals = 3 if parameter == "ntn_radial_projection" else 2
        self.ntn_scan_start_spin.setDecimals(decimals)
        self.ntn_scan_stop_spin.setDecimals(decimals)
        self.ntn_scan_start_spin.setSuffix(f" {spec['unit']}" if spec["unit"] else "")
        self.ntn_scan_stop_spin.setSuffix(f" {spec['unit']}" if spec["unit"] else "")
        self.ntn_scan_start_spin.blockSignals(False)
        self.ntn_scan_stop_spin.blockSignals(False)
        self._update_ntn_controls()

    def _update_ntn_controls(self):
        is_ofdm = self._current_waveform() == "OFDM"
        enabled = is_ofdm and self.ntn_enabled_check.isChecked()
        scanning = enabled and self.ntn_mode_combo.currentData() == "scan"
        for control in (
            self.ntn_enabled_check,
            self.ntn_profile_combo,
            self.ntn_mode_combo,
            self.ntn_velocity_spin,
            self.ntn_radial_spin,
            self.ntn_compensation_spin,
            self.ntn_cfo_spin,
            self.ntn_doppler_spin,
            self.ntn_delay_scale_spin,
            self.ntn_seed_spin,
        ):
            control.setEnabled(is_ofdm and not self.test_running and (control is self.ntn_enabled_check or enabled))
        self.ntn_scan_widget.setVisible(self.ntn_mode_combo.currentData() == "scan")
        for control in (
            self.ntn_scan_parameter_combo,
            self.ntn_scan_start_spin,
            self.ntn_scan_stop_spin,
            self.ntn_scan_points_spin,
            self.ntn_scan_frames_spin,
        ):
            control.setEnabled(scanning and not self.test_running)

    def _update_button_states(self):
        connected = self.backend is not None
        running = bool(self.test_running)

        self.btn_connect.setEnabled(not running)
        self.btn_apply.setEnabled(not running)
        self.btn_start_test.setEnabled(connected and not running)
        self.btn_stop_test.setEnabled(connected and running)
        self.btn_disconnect.setEnabled(connected and not running)

        # 运行中禁止改变会触发重建的参数；增益和纯显示选项仍可热更新。
        for widget in (
            self.waveform_combo,
            self.device_combo,
            self.serial_edit,
            self.samp_rate_spin,
            self.fc_spin,
            self.tx_antenna_combo,
            self.rx_antenna_combo,
            self.mod_order_combo,
            self.equalizer_combo,
            self.c1_spin,
            self.c2_spin,
            self.tx_text_edit,
            self.btn_load_text,
            self.btn_reset_text,
        ):
            widget.setEnabled(not running)

        self.tx_gain_spin.setEnabled(True)
        self.rx_gain_spin.setEnabled(True)
        self.const_mode_combo.setEnabled(True)
        self.tx_plot_combo.setEnabled(True)
        self._update_ntn_controls()

    # =========================================================
    # Backend loading / lifecycle
    # =========================================================
    def _current_waveform(self) -> str:
        return self.waveform_combo.currentText().strip().upper()

    def _current_identity(self) -> Tuple[str, str, str]:
        return (
            self._current_waveform(),
            self.device_combo.currentText().strip(),
            self.serial_edit.text().strip(),
        )

    def _load_backend_class(self, waveform: str):
        try:
            module_name, class_name = _BACKEND_SPECS[waveform]
        except KeyError as exc:
            raise ValueError(f"不支持的硬件测试波形: {waveform}") from exc

        module = importlib.import_module(module_name)
        backend_class = getattr(module, class_name, None)
        if backend_class is None:
            raise AttributeError(
                f"模块 {module_name!r} 中未找到后端类 {class_name!r}"
            )
        return backend_class

    def _current_ntn_kwargs(self) -> Dict[str, Any]:
        return {
            "ntn_enabled": bool(self.ntn_enabled_check.isChecked()),
            "ntn_profile": self.ntn_profile_combo.currentText().strip(),
            "ntn_seed": int(self.ntn_seed_spin.value()),
            "ntn_velocity_kmh": float(self.ntn_velocity_spin.value()),
            "ntn_radial_projection": float(self.ntn_radial_spin.value()),
            "ntn_compensation_ratio": float(self.ntn_compensation_spin.value()) / 100.0,
            "ntn_residual_cfo_hz": float(self.ntn_cfo_spin.value()),
            "ntn_doppler_spread_hz": float(self.ntn_doppler_spin.value()),
            "ntn_delay_scale": float(self.ntn_delay_scale_spin.value()),
        }

    def _current_ntn_scan_plan(self) -> NTNScanPlan:
        return NTNScanPlan(
            parameter=str(self.ntn_scan_parameter_combo.currentData()),
            start=float(self.ntn_scan_start_spin.value()),
            stop=float(self.ntn_scan_stop_spin.value()),
            points=int(self.ntn_scan_points_spin.value()),
            frames_per_point=int(self.ntn_scan_frames_spin.value()),
        )

    def _constructor_kwargs(self, waveform: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "carrier_freq": float(self.fc_spin.value()),
            "samp_rate": float(self.samp_rate_spin.value()),
            "tx_gain": float(self.tx_gain_spin.value()),
            "rx_gain": float(self.rx_gain_spin.value()),
            "device_type": self.device_combo.currentText().strip(),
            "serial": self.serial_edit.text().strip() or None,
            "tx_antenna": self.tx_antenna_combo.currentText().strip() or "TX/RX",
            "rx_antenna": self.rx_antenna_combo.currentText().strip() or "RX2",
            "tx_text": self.tx_text_edit.toPlainText() or " ",
            "mod_order": self.mod_order_combo.currentText().strip(),
            "equalizer": self.equalizer_combo.currentText().strip(),
        }
        if waveform == "AFDM":
            kwargs.update(c1=float(self.c1_spin.value()), c2=float(self.c2_spin.value()))
        elif waveform == "OFDM":
            kwargs.update(self._current_ntn_kwargs())
        return kwargs

    def _build_hardware_run_config(self, ntn_kwargs: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
        ntn = dict(ntn_kwargs or (self._current_ntn_kwargs() if self._current_waveform() == "OFDM" else {}))
        waveform = self._current_waveform()
        wave_config = WaveformConfig(
            waveform=waveform,
            mod_order=self.mod_order_combo.currentText().strip(),
            center_freq_hz=float(self.fc_spin.value()),
            sample_rate_hz=float(self.samp_rate_spin.value()),
            seed=int(ntn.get("ntn_seed", 42)),
            ntn_profile=str(ntn.get("ntn_profile", "NTN-TDLA100-200")),
            cfo_hz=float(ntn.get("ntn_residual_cfo_hz", 0.0)),
            velocity_kmh=float(ntn.get("ntn_velocity_kmh", 0.0)),
            doppler_radial_factor=float(ntn.get("ntn_radial_projection", 0.0)),
            doppler_compensation_ratio=float(ntn.get("ntn_compensation_ratio", 1.0)),
            residual_doppler_spread_hz=float(ntn.get("ntn_doppler_spread_hz", 0.0)),
        )
        hardware_config = HardwareConfig(
            transport="uhd",
            device_args=self.serial_edit.text().strip(),
            device_hint=self.device_combo.currentText().strip(),
            mode="rf",
            center_freq_hz=float(self.fc_spin.value()),
            sample_rate_hz=float(self.samp_rate_spin.value()),
            tx_gain_db=float(self.tx_gain_spin.value()),
            rx_gain_db=float(self.rx_gain_spin.value()),
            tx_antenna=self.tx_antenna_combo.currentText().strip(),
            rx_antenna=self.rx_antenna_combo.currentText().strip(),
        )
        return ExperimentConfig(
            waveform=wave_config,
            hardware=hardware_config,
            notes="硬件收发验证",
            tags=["hardware", waveform.lower()],
            runs_dir="runs",
        ).normalized()

    def _begin_hardware_run(
        self,
        ntn_kwargs: Optional[Dict[str, Any]] = None,
        scan_point: Optional[Dict[str, Any]] = None,
    ) -> bool:
        ntn = dict(ntn_kwargs or {})
        metadata: Dict[str, Any] = {
            "waveform": self._current_waveform(),
            "modulation": self.mod_order_combo.currentText().strip(),
            "rf": {
                "device": self.device_combo.currentText().strip(),
                "serial": self.serial_edit.text().strip(),
                "center_freq_hz": float(self.fc_spin.value()),
                "sample_rate_hz": float(self.samp_rate_spin.value()),
                "tx_gain_db": float(self.tx_gain_spin.value()),
                "rx_gain_db": float(self.rx_gain_spin.value()),
                "tx_antenna": self.tx_antenna_combo.currentText().strip(),
                "rx_antenna": self.rx_antenna_combo.currentText().strip(),
            },
            "payload_utf8_bytes": len(self.tx_text_edit.toPlainText().encode("utf-8")),
        }
        if ntn:
            metadata["channel_injection"] = {
                "stage": "tx_baseband_before_rf",
                **ntn,
            }
        if scan_point:
            metadata["scan_point"] = dict(scan_point)
        try:
            recorder = HardwareRunRecorder(
                self._build_hardware_run_config(ntn), metadata=metadata
            )
            run_id = recorder.start()
            self._hardware_recorder = recorder
            self._last_run_status = {}
            self._log(f"硬件结果记录已创建：{recorder.run_dir.resolve()} ({run_id})")
            return True
        except Exception as exc:
            self._hardware_recorder = None
            self._log(f"创建硬件结果记录失败：{type(exc).__name__}: {exc}")
            self.decode_status_label.setText(
                f"解调状态：无法创建结果记录：{type(exc).__name__}: {exc}"
            )
            return False

    def _finish_hardware_run(self, status: Optional[Dict[str, Any]] = None, stats: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        recorder = self._hardware_recorder
        if recorder is None:
            return None
        combined = dict(status or self._last_run_status)
        combined.update(stats or {})
        run_dir = recorder.run_dir
        try:
            if error:
                recorder.fail(error)
            else:
                recorder.finish(
                    {
                        "frame_id": combined.get("frame_id"),
                        "bit_errors": combined.get("bit_errors"),
                        "bits_total": combined.get("bits_total"),
                        "ber": combined.get("cumulative_ber", combined.get("ber")),
                        "decode_ok": combined.get("decode_ok"),
                    }
                )
        except Exception as exc:
            self._log(f"写入硬件运行结束状态失败：{type(exc).__name__}: {exc}")
        finally:
            self._hardware_recorder = None
        return run_dir

    def _capture_final_hardware_status(
        self,
        fallback_status: Optional[Dict[str, Any]] = None,
        fallback_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        status = dict(fallback_status or {})
        stats = dict(fallback_stats or {})
        if self.backend is not None:
            status = dict(self.backend.get_status() or status)
            stats = dict(self.backend.get_decode_stats() or stats)
        status.update(stats)
        self._last_run_status = status
        if self._hardware_recorder is not None:
            self._hardware_recorder.record_status(status)
        return status

    def _start_recorded_backend(
        self,
        ntn_kwargs: Optional[Dict[str, Any]] = None,
        scan_point: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self._begin_hardware_run(ntn_kwargs=ntn_kwargs, scan_point=scan_point):
            return False
        try:
            if self.backend_waveform == "OFDM":
                frame_limit = (
                    self._scan_plan.frames_per_point
                    if self._scan_active and self._scan_plan is not None
                    else None
                )
                set_frame_limit = getattr(self.backend, "set_frame_limit", None)
                if frame_limit is not None and not callable(set_frame_limit):
                    raise RuntimeError("当前 OFDM 后端不支持按接收帧数结束扫描点")
                if callable(set_frame_limit):
                    set_frame_limit(frame_limit)
            self._reset_runtime_plots()
            self.rx_text_view.clear()
            self.backend.start()
            self._scan_start_frame_id = 0
            self.test_running = True
            self._backend_started_once = True
            self._last_runtime_log_time = 0.0
            self.update_timer.start()
            self.decode_status_label.setText(
                f"解调状态：{self.backend_waveform} 运行中，等待接收帧。"
            )
            return True
        except Exception as exc:
            self.test_running = False
            self.update_timer.stop()
            self._finish_hardware_run(error=f"{type(exc).__name__}: {exc}")
            self.decode_status_label.setText(
                f"解调状态：开始测试失败：{type(exc).__name__}: {exc}"
            )
            raise

    def _start_scan_point(self):
        if self.backend is None or self._scan_plan is None:
            return
        value = self._scan_values[self._scan_index]
        ntn_kwargs = self._current_ntn_kwargs()
        ntn_kwargs[self._scan_plan.parameter] = float(value)
        scan_point = {
            "parameter": self._scan_plan.parameter,
            "value": float(value),
            "index": self._scan_index + 1,
            "count": len(self._scan_values),
            "frames_per_point": self._scan_plan.frames_per_point,
        }
        try:
            self.backend.configure(**ntn_kwargs)
            if not self._start_recorded_backend(ntn_kwargs=ntn_kwargs, scan_point=scan_point):
                self._scan_active = False
                self._scan_plan = None
                self.test_running = False
                return
            if self.test_running:
                spec = NTN_SCAN_PARAMETERS[self._scan_plan.parameter]
                unit = f" {spec['unit']}" if spec["unit"] else ""
                self._log(
                    f"扫描点 {self._scan_index + 1}/{len(self._scan_values)}："
                    f"{spec['label']}={float(value):g}{unit}，"
                    f"每点 {self._scan_plan.frames_per_point} 帧。"
                )
        except Exception as exc:
            self._scan_active = False
            self._scan_transitioning = False
            self._finish_hardware_run(error=f"{type(exc).__name__}: {exc}")
            self.test_running = False
            self.update_timer.stop()
            self.decode_status_label.setText(
                f"解调状态：NTN 扫描中止：{type(exc).__name__}: {exc}"
            )
            self._log(f"NTN 扫描中止：{type(exc).__name__}: {exc}")
        finally:
            self._update_button_states()

    def _maybe_advance_scan(self, status: Dict[str, Any], stats: Dict[str, Any]):
        if not self._scan_active or self._scan_transitioning or self._scan_plan is None:
            return
        try:
            frame_id = int(status.get("frame_id", status.get("frames_processed", 0)) or 0)
        except (TypeError, ValueError, OverflowError):
            return
        if frame_id - self._scan_start_frame_id < self._scan_plan.frames_per_point:
            return
        self._scan_transitioning = True
        QTimer.singleShot(0, lambda: self._complete_scan_point(dict(status), dict(stats)))

    def _complete_scan_point(self, status: Dict[str, Any], stats: Dict[str, Any]):
        if not self._scan_active or self._scan_plan is None:
            self._scan_transitioning = False
            return
        try:
            self.update_timer.stop()
            self.backend.stop()
            if hasattr(self.backend, "wait"):
                self.backend.wait()
            self.test_running = False
            point_status = self._capture_final_hardware_status(status, stats)
            run_dir = self._finish_hardware_run(point_status)
            self._log(
                f"扫描点 {self._scan_index + 1}/{len(self._scan_values)} 完成："
                f"BER={point_status.get('cumulative_ber', point_status.get('ber'))}, "
                f"误码={point_status.get('bit_errors')}/{point_status.get('bits_total')}，"
                f"记录={run_dir}"
            )
            self._scan_index += 1
            if self._scan_index >= len(self._scan_values):
                self._scan_active = False
                self._scan_plan = None
                self.decode_status_label.setText(
                    f"解调状态：NTN 单参数扫描完成，共 {len(self._scan_values)} 点；"
                    "每点 BER 与误码计数已分别保存。"
                )
                self._log("NTN 单参数扫描完成。")
            else:
                self._start_scan_point()
        except Exception as exc:
            self._scan_active = False
            self._scan_plan = None
            if self.test_running:
                self.update_timer.start()
                self.decode_status_label.setText(
                    f"解调状态：扫描切换失败，当前后端保留：{type(exc).__name__}: {exc}"
                )
                self._log(f"NTN 扫描切换失败；当前结果记录保留：{type(exc).__name__}: {exc}")
            else:
                self.update_timer.stop()
                self._finish_hardware_run(error=f"{type(exc).__name__}: {exc}")
                self.decode_status_label.setText(
                    f"解调状态：扫描点停止后读取结果失败：{type(exc).__name__}: {exc}"
                )
                self._log(f"NTN 扫描点结果读取失败：{type(exc).__name__}: {exc}")
        finally:
            self._scan_transitioning = False
            self._update_button_states()

    def _create_backend(self) -> bool:
        waveform = self._current_waveform()
        identity = self._current_identity()
        try:
            self._dispose_backend(clear_plots=False, strict=True)
        except Exception as exc:
            self.decode_status_label.setText(
                f"解调状态：停止旧后端失败，已取消创建新后端：{type(exc).__name__}: {exc}"
            )
            self._log(f"停止旧后端失败，已取消创建新后端：{type(exc).__name__}: {exc}")
            self._update_button_states()
            return False

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            backend_class = self._load_backend_class(waveform)
            self._log(
                f"正在创建 {waveform} 后端：{backend_class.__module__}."
                f"{backend_class.__name__}"
            )
            backend = backend_class(**self._constructor_kwargs(waveform))
            self.backend = backend
            self.backend_waveform = waveform
            self._backend_started_once = False
            self._backend_identity = identity
            self._configuration_dirty = False
            self._last_status_error = ""
            self._push_constellation_mode()
            self.tx_text_view.setPlainText(self.tx_text_edit.toPlainText())
            self.rx_text_view.clear()
            self.decode_status_label.setText(
                f"解调状态：{waveform} 后端已配置，尚未开始测试。"
            )
            self._log("USRP 后端创建成功。")
            self._log(self._backend_summary())
            self._refresh_plots_once()
            return True
        except Exception as exc:
            self.backend = None
            self.backend_waveform = None
            self._backend_identity = None
            self.decode_status_label.setText(
                f"解调状态：连接 / 配置失败：{type(exc).__name__}: {exc}"
            )
            self._log(f"连接 / 配置失败：{type(exc).__name__}: {exc}")
            return False
        finally:
            QApplication.restoreOverrideCursor()
            self._update_button_states()

    def _dispose_backend(self, clear_plots: bool = True, *, strict: bool = False):
        self.update_timer.stop()
        backend = self.backend
        shutdown_complete = True

        if backend is not None:
            try:
                backend.stop()
            except Exception as exc:
                shutdown_complete = False
                self._log(f"停止后端时出现异常：{type(exc).__name__}: {exc}")
                if strict:
                    if self.test_running:
                        self.update_timer.start()
                    raise
            try:
                if hasattr(backend, "wait"):
                    backend.wait()
            except Exception as exc:
                shutdown_complete = False
                self._log(f"等待后端退出时出现异常：{type(exc).__name__}: {exc}")
                if strict:
                    if self.test_running:
                        self.update_timer.start()
                    raise
        if self._hardware_recorder is not None:
            if shutdown_complete:
                try:
                    final_status = self._capture_final_hardware_status()
                except Exception as exc:
                    self._finish_hardware_run(error=f"{type(exc).__name__}: {exc}")
                    self._log(f"读取硬件最终状态失败：{type(exc).__name__}: {exc}")
                else:
                    self._finish_hardware_run(final_status)
            elif not strict:
                self._finish_hardware_run(error="停止硬件后端失败，最终测量状态未知")
        self.backend = None
        self.backend_waveform = None
        self._backend_identity = None
        self.test_running = False
        self._backend_started_once = False
        if backend is not None:
            del backend
            gc.collect()

        if clear_plots:
            self._clear_plots()
            self.rx_text_view.clear()
        self._update_button_states()

    # =========================================================
    # Buttons
    # =========================================================
    def _on_connect_clicked(self):
        self._create_backend()

    def _on_apply_clicked(self):
        # 三个后端的 configure() 都可能重建含 UHD 句柄的 top_block。
        # 这里先完整释放旧对象再创建新对象，避免同一 USRP 被新旧 flowgraph
        # 同时持有，尤其适合 B210 的单设备闭环场景。
        self._create_backend()

    def _on_start_test_clicked(self):
        try:
            if (
                self.backend is None
                or self._backend_identity != self._current_identity()
                or self._configuration_dirty
                or self._backend_started_once
            ):
                if not self._create_backend():
                    return

            if (
                self.backend_waveform == "OFDM"
                and self.ntn_enabled_check.isChecked()
                and self.ntn_mode_combo.currentData() == "scan"
            ):
                self._scan_plan = self._current_ntn_scan_plan()
                self._scan_values = self._scan_plan.values()
                self._scan_index = 0
                self._scan_active = True
                self._scan_transitioning = False
                self._start_scan_point()
            else:
                self._scan_active = False
                self._scan_plan = None
                channel = self._current_ntn_kwargs() if self.backend_waveform == "OFDM" else None
                if self._start_recorded_backend(ntn_kwargs=channel):
                    self._log(f"{self.backend_waveform} 硬件测试已启动。")
        except Exception as exc:
            self.test_running = False
            self.update_timer.stop()
            self._scan_active = False
            self._scan_plan = None
            self._finish_hardware_run(error=f"{type(exc).__name__}: {exc}")
            self.decode_status_label.setText(
                f"解调状态：开始测试失败：{type(exc).__name__}: {exc}"
            )
            self._log(f"开始测试失败：{type(exc).__name__}: {exc}")
        finally:
            self._update_button_states()

    def _on_stop_test_clicked(self):
        self.update_timer.stop()
        if self.backend is None:
            self.test_running = False
            self._scan_active = False
            self._scan_plan = None
            self._finish_hardware_run()
            self._update_button_states()
            return

        try:
            self.backend.stop()
            if hasattr(self.backend, "wait"):
                self.backend.wait()
            self.test_running = False
            self._scan_active = False
            self._scan_plan = None
            final_status = self._capture_final_hardware_status()
            run_dir = self._finish_hardware_run(final_status)
            message = self._measurement_summary(self._last_run_status)
            self.decode_status_label.setText(
                f"解调状态：已停止。{message}" if message else "解调状态：已停止。"
            )
            self._log(f"硬件测试已停止；最后一次曲线与星座图予以保留。记录：{run_dir}")
        except Exception as exc:
            self._scan_active = False
            self._scan_plan = None
            if self.test_running:
                self.update_timer.start()
                self.decode_status_label.setText(
                    f"解调状态：停止失败，后端仍保留运行：{type(exc).__name__}: {exc}"
                )
                self._log(f"停止测试失败：{type(exc).__name__}: {exc}")
            else:
                self.update_timer.stop()
                self._finish_hardware_run(error=f"{type(exc).__name__}: {exc}")
                self.decode_status_label.setText(
                    f"解调状态：后端已停止，但读取最终结果失败：{type(exc).__name__}: {exc}"
                )
                self._log(f"读取最终硬件结果失败：{type(exc).__name__}: {exc}")
        finally:
            self._update_button_states()

    def _on_disconnect_clicked(self):
        self._dispose_backend(clear_plots=True)
        self.decode_status_label.setText("解调状态：USRP 已释放。")
        self._log("USRP 后端已释放。")

    def _on_load_text_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择文本文件",
            "",
            "Text Files (*.txt);;All Files (*)",
        )
        if not file_path:
            return
        try:
            text = self._read_text_file(file_path)
            self.tx_text_edit.setPlainText(text)
            self.tx_text_view.setPlainText(text)
            self.file_path_label.setText(file_path)
            self._mark_configuration_dirty()
            self._log(f"已加载文本文件：{file_path}")
        except Exception as exc:
            self._log(f"读取文本文件失败：{type(exc).__name__}: {exc}")

    def _on_reset_text_clicked(self):
        waveform = self._current_waveform()
        text = _DEFAULT_TEXTS.get(waveform, "Hello Hardware Test!")
        self.tx_text_edit.setPlainText(text)
        self.tx_text_view.setPlainText(text)
        self.file_path_label.setText("可直接编辑下方文本，或加载 .txt 文件")
        self._mark_configuration_dirty()

    # =========================================================
    # Live updates
    # =========================================================
    def _hot_set_gain(self, which: str, value: float):
        if self._initializing or self.backend is None:
            return
        try:
            if which == "tx":
                self.backend.set_tx_gain(float(value))
            else:
                self.backend.set_rx_gain(float(value))
            if self.test_running:
                self._log(
                    f"{'发送' if which == 'tx' else '接收'}增益热更新 -> {value:.1f} dB"
                )
        except Exception as exc:
            self._log(f"增益热更新失败：{type(exc).__name__}: {exc}")

    def _current_constellation_mode(self) -> str:
        index = self.const_mode_combo.currentIndex()
        if 0 <= index < len(_CONSTELLATION_MODES):
            return _CONSTELLATION_MODES[index][1]
        return "dd_refined"

    def _push_constellation_mode(self) -> bool:
        if self.backend is None:
            return False
        try:
            self.backend.set_constellation_display_mode(
                self._current_constellation_mode()
            )
            return True
        except Exception as exc:
            self._log(f"设置星座显示模式失败：{type(exc).__name__}: {exc}")
            return False

    # =========================================================
    # Refresh / plots
    # =========================================================
    def _refresh_plots_once(self):
        try:
            self._refresh_plots()
        except Exception:
            pass

    def _refresh_plots(self):
        if self.backend is None:
            return

        try:
            status = self.backend.get_status() or {}
            stats = self.backend.get_decode_stats() or {}
            frame_status = dict(status)
            frame_status.update(stats)
            self._last_run_status = frame_status
            if self.test_running and self._hardware_recorder is not None:
                self._hardware_recorder.record_status(frame_status)
            samp_rate = self._extract_samp_rate(status)
            self._apply_stable_plot_ranges(samp_rate)

            current_error = str(status.get("last_error", "") or "")
            if current_error and current_error != self._last_status_error:
                self._last_status_error = current_error
                self._log(f"后端状态异常：{current_error}")

            self._update_tx_plot(samp_rate)
            self._update_rx_spectrum(samp_rate)
            self._update_ber_plot(status)
            self._update_constellation_plot()

            self.tx_text_view.setPlainText(self.tx_text_edit.toPlainText())
            try:
                self.rx_text_view.setPlainText(self.backend.get_rx_text())
            except Exception:
                pass

            self._update_decode_status(stats, status)
            self._maybe_log_runtime(stats, status)
            self._maybe_advance_scan(status, stats)
        except Exception as exc:
            self._log(f"刷新失败：{type(exc).__name__}: {exc}")

    def _refresh_tx_plot_only(self):
        if self.backend is None:
            return
        try:
            status = self.backend.get_status() or {}
            self._update_tx_plot(self._extract_samp_rate(status))
        except Exception:
            pass

    def _update_tx_plot(self, samp_rate: float):
        if self.backend is None or not self.test_running:
            self.tx_curve.setData([], [])
            self.tx_plot.setTitle("发送端基带显示（尚无运行数据）")
            return

        samples = np.asarray(
            self.backend.get_tx_spectrum_source(4096), dtype=np.complex64
        ).reshape(-1)
        if not has_signal(samples):
            self.tx_curve.setData([], [])
            self.tx_plot.setTitle("发送端基带显示（暂未取得样点）")
            return

        if "时域" in self.tx_plot_combo.currentText():
            y = np.abs(samples).astype(np.float64)
            x = np.arange(y.size, dtype=np.float64)
            self.tx_curve.setData(x, y)
            self.tx_plot.setTitle(f"TX 时域幅度（{samples.size} 样点）")
            self.tx_plot.setXRange(0, max(10, samples.size), padding=0)
            ymax = max(0.25, float(np.nanmax(y)) * 1.15)
            self.tx_plot.setYRange(0, ymax, padding=0)
        else:
            freq, psd = compute_spectrum(samples, samp_rate, 1024)
            self.tx_curve.setData(freq, psd)
            self.tx_plot.setTitle(f"TX 基带频谱（{samples.size} 样点）")
            self.tx_plot.setXRange(-samp_rate / 2, samp_rate / 2, padding=0)
            self.tx_plot.setYRange(-120, 20, padding=0)

    def _update_rx_spectrum(self, samp_rate: float):
        if self.backend is None or not self.test_running:
            self.rx_curve.setData([], [])
            self.rx_spectrum_plot.setTitle("USRP 接收基带频谱（尚无运行数据）")
            return

        samples = np.asarray(
            self.backend.get_rx_spectrum_source(4096), dtype=np.complex64
        ).reshape(-1)
        if not has_signal(samples):
            self.rx_curve.setData([], [])
            self.rx_spectrum_plot.setTitle("USRP 接收基带频谱（暂未取得有效样点）")
            return

        freq, psd = compute_spectrum(samples, samp_rate, 1024)
        self.rx_curve.setData(freq, psd)
        self.rx_spectrum_plot.setTitle(
            f"USRP 接收基带频谱（{samples.size} 样点）"
        )

    def _update_ber_plot(self, status: Dict[str, Any]):
        if self.backend is None:
            self.ber_curve.setData([], [])
            return

        x = np.zeros(0, dtype=np.float64)
        ber = np.zeros(0, dtype=np.float64)
        try:
            x_raw, ber_raw = self.backend.get_estimated_ber()
            x = np.asarray(x_raw, dtype=np.float64).reshape(-1)
            ber = np.asarray(ber_raw, dtype=np.float64).reshape(-1)
        except Exception:
            pass

        n = min(x.size, ber.size)
        if n <= 0:
            self.ber_curve.setData([], [])
            current = safe_float(status.get("ber"), np.nan)
            title = "BER 曲线"
            if np.isfinite(current):
                title += f"（当前 {current:.3e}）"
            self.ber_plot.setTitle(title)
            return

        x = x[-n:]
        ber = ber[-n:]
        valid = np.isfinite(x) & np.isfinite(ber) & (ber >= 0)
        x = x[valid]
        ber = ber[valid]
        if x.size == 0:
            self.ber_curve.setData([], [])
            return

        y_log = np.log10(np.clip(ber, 1e-6, 1.0))
        self.ber_curve.setData(x, y_log)
        x_end = max(60.0, float(x[-1]) + 1.0)
        self.ber_plot.setXRange(max(0.0, x_end - 60.0), x_end, padding=0)
        self.ber_plot.setYRange(-6.0, 0.0, padding=0)
        self.ber_plot.setTitle(f"BER 曲线（当前 {ber[-1]:.3e}）")

    def _update_constellation_plot(self):
        if self.backend is None or not self.test_running:
            self.constellation_scatter.setData(x=[], y=[])
            self.constellation_plot.setTitle("接收星座图（尚无运行数据）")
            return

        try:
            points = np.asarray(
                self.backend.get_rx_constellation(1024), dtype=np.complex64
            ).reshape(-1)
        except Exception:
            points = np.zeros(0, dtype=np.complex64)

        finite = np.isfinite(np.real(points)) & np.isfinite(np.imag(points))
        points = points[finite]
        if points.size == 0:
            self.constellation_scatter.setData(x=[], y=[])
            self.constellation_plot.setTitle("接收星座图（等待有效解调符号）")
            return

        self.constellation_scatter.setData(x=np.real(points), y=np.imag(points))
        mode_label = self.const_mode_combo.currentText()
        self.constellation_plot.setTitle(f"接收星座图 — {mode_label}")

        # 使用稳健分位数避免少量外点把主星座簇压得过小。
        scale = float(np.nanpercentile(np.abs(points), 99.0))
        limit = min(8.0, max(1.5, 1.25 * scale))
        self.constellation_plot.setXRange(-limit, limit, padding=0)
        self.constellation_plot.setYRange(-limit, limit, padding=0)

    def _apply_stable_plot_ranges(self, samp_rate: float):
        samp_rate = float(samp_rate)
        if (
            self._last_plot_samp_rate is not None
            and abs(self._last_plot_samp_rate - samp_rate) < 1.0
        ):
            return
        self.rx_spectrum_plot.setXRange(-samp_rate / 2, samp_rate / 2, padding=0)
        self.rx_spectrum_plot.setYRange(-120, 20, padding=0)
        if "频谱" in self.tx_plot_combo.currentText():
            self.tx_plot.setXRange(-samp_rate / 2, samp_rate / 2, padding=0)
            self.tx_plot.setYRange(-120, 20, padding=0)
        self._last_plot_samp_rate = samp_rate

    def _reset_runtime_plots(self):
        self._last_runtime_log_time = 0.0
        self._last_status_error = ""
        self.tx_curve.setData([], [])
        self.rx_curve.setData([], [])
        self.ber_curve.setData([], [])
        self.constellation_scatter.setData(x=[], y=[])

    def _clear_plots(self):
        self._reset_runtime_plots()
        self.tx_plot.setTitle("发送端基带显示")
        self.rx_spectrum_plot.setTitle("USRP 接收基带频谱")
        self.ber_plot.setTitle("BER 曲线")
        self.constellation_plot.setTitle("接收星座图")

    # =========================================================
    # Status / log
    # =========================================================
    def _extract_samp_rate(self, status: Dict[str, Any]) -> float:
        return safe_float(
            status.get("samp_rate", status.get("sample_rate")),
            self.samp_rate_spin.value(),
        )

    def _update_decode_status(
        self, stats: Dict[str, Any], status: Dict[str, Any]
    ):
        waveform = self.backend_waveform or self._current_waveform()
        if not self.test_running:
            backend_state = str(status.get("status", "configured") or "configured")
            dirty_note = "，参数已修改、尚未应用" if self._configuration_dirty else ""
            self.decode_status_label.setText(
                f"解调状态：{waveform} 后端状态={backend_state}，尚未开始测试{dirty_note}。"
            )
            return

        decode_ok = bool(stats.get("decode_ok", status.get("decode_ok", False)))
        state_text = "CRC 通过，文本已恢复" if decode_ok else "尚未通过 CRC"
        match = int(stats.get("match_bytes", status.get("match_bytes", 0)) or 0)
        expected = int(
            stats.get("expected_bytes", status.get("expected_bytes", 0)) or 0
        )
        reason = str(status.get("reason", status.get("status", "")) or "")
        sync = format_metric(status.get("sync_metric"), ".3f")
        cfo = format_metric(status.get("cfo_est_hz"), ".1f")
        total_bits = status.get("bits_total", stats.get("bits_total"))
        bit_errors = status.get("bit_errors", stats.get("bit_errors"))
        ber_value = status.get("cumulative_ber", status.get("ber"))
        ber = format_metric(ber_value, ".3e")
        combine = int(status.get("combine_frames", 0) or 0)
        repeat = int(status.get("payload_repeat", 0) or 0)

        extras = []
        if waveform in ("OFDM", "AFDM"):
            extras.extend(
                [
                    f"TrainNMSE={format_metric(status.get('train_nmse'), '.3f')}",
                    f"PilotNMSE={format_metric(status.get('pilot_nmse'), '.3f')}",
                    f"α={format_metric(status.get('alpha_abs'), '.3f')}",
                ]
            )
        elif waveform == "OTFS":
            extras.extend(
                [
                    f"PilotNMSE={format_metric(status.get('pilot_nmse'), '.3f')}",
                    f"KernelE={format_metric(status.get('kernel_energy'), '.3e')}",
                    f"rank={int(status.get('kernel_rank', 0) or 0)}",
                ]
            )

        if waveform == "OFDM":
            extras.append(f"FFT偏移={int(status.get('fft_window_offset', 0) or 0):+d}")
        elif waveform == "AFDM":
            extras.extend(
                [
                    f"c1={format_metric(status.get('c1'), '.4f')}",
                    f"c2={format_metric(status.get('c2'), '.4f')}",
                ]
            )

        extra_text = " | " + ", ".join(extras) if extras else ""
        if total_bits is not None and bit_errors is not None:
            try:
                extras.append(f"误码={int(bit_errors)}/{int(total_bits)}")
                extra_text = " | " + ", ".join(extras)
            except (TypeError, ValueError, OverflowError):
                pass
        self.decode_status_label.setText(
            f"解调状态：{waveform} {state_text} ({match}/{expected} bytes) | "
            f"reason={reason}, Sync={sync}, CFO={cfo} Hz, BER={ber}, "
            f"合并帧={combine}, 帧内重复={repeat}{extra_text}"
        )

    @staticmethod
    def _measurement_summary(status: Dict[str, Any]) -> str:
        if not status:
            return ""
        total_bits = status.get("bits_total")
        bit_errors = status.get("bit_errors")
        ber = status.get("cumulative_ber", status.get("ber"))
        if total_bits is None and bit_errors is None and ber is None:
            return ""
        text = f"BER={format_metric(ber, '.3e')}"
        if total_bits is not None and bit_errors is not None:
            try:
                text += f"，误码={int(bit_errors)}/{int(total_bits)} bit"
            except (TypeError, ValueError, OverflowError):
                pass
        return text

    def _maybe_log_runtime(
        self, stats: Dict[str, Any], status: Dict[str, Any]
    ):
        if not self.test_running:
            return
        now = time.monotonic()
        if now - self._last_runtime_log_time < 5.0:
            return
        self._last_runtime_log_time = now
        self._log(
            f"runtime[{self.backend_waveform}]: "
            f"reason={status.get('reason', '')}, "
            f"sync={format_metric(status.get('sync_metric'), '.3f')}, "
            f"CFO={format_metric(status.get('cfo_est_hz'), '.1f')} Hz, "
            f"BER={format_metric(status.get('ber'), '.3e')}, "
            f"decode_ok={bool(stats.get('decode_ok', False))}, "
            f"match={int(stats.get('match_bytes', 0) or 0)}/"
            f"{int(stats.get('expected_bytes', 0) or 0)}"
        )

    def _backend_summary(self) -> str:
        if self.backend is None:
            return "未创建后端"
        try:
            status = self.backend.get_status() or {}
        except Exception:
            status = {}
        return (
            f"波形={self.backend_waveform}, 设备={status.get('device_type', self.device_combo.currentText())}, "
            f"Fc={self._extract_number(status, 'carrier_freq', self.fc_spin.value()):.0f} Hz, "
            f"Fs={self._extract_samp_rate(status):.0f} Hz, "
            f"TX/RX Gain={self._extract_number(status, 'tx_gain', self.tx_gain_spin.value()):.1f}/"
            f"{self._extract_number(status, 'rx_gain', self.rx_gain_spin.value()):.1f} dB, "
            f"调制={status.get('mod_order', self.mod_order_combo.currentText())}, "
            f"EQ={status.get('equalizer', self.equalizer_combo.currentText())}"
        )

    def _extract_number(
        self, status: Dict[str, Any], key: str, default: float
    ) -> float:
        return safe_float(status.get(key), default)

    @staticmethod
    def _read_text_file(file_path: str) -> str:
        for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            return file.read()

    def _log(self, message: str):
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    # =========================================================
    # Qt cleanup
    # =========================================================
    def suspend_tasks(self):
        """Release the radio when leaving this page, keeping the evidence."""
        had_backend = self.backend is not None
        run_dir = (
            self._hardware_recorder.run_dir
            if self._hardware_recorder is not None else None
        )
        self._scan_active = False
        self._scan_plan = None
        self._scan_transitioning = False
        self._dispose_backend(clear_plots=False, strict=True)
        if self._hardware_recorder is not None:
            run_dir = self._finish_hardware_run()
        if had_backend:
            self.decode_status_label.setText(
                f"解调状态：已停止并释放 USRP；{self._measurement_summary(self._last_run_status)}"
            )
            self._log(f"离开测试页面，已停止并释放 USRP；结果已保留：{run_dir}")

    def closeEvent(self, event):
        try:
            self.suspend_tasks()
        except Exception:
            event.ignore()
            return
        super().closeEvent(event)


# 明确导出，供 main_window.py 的动态加载器 getattr(module, "HardwareTestTab") 使用。
__all__ = ["HardwareTestTab"]
