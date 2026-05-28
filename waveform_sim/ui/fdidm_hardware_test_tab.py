# -*- coding: utf-8 -*-
"""
ui/fdidm_hardware_test_tab.py

FDIDM paper-strict hardware verification page - v17.

What changed in v17 vs v16 UI:
    1) _connect_signals: self.probe_guard_spin is now included in the spin
       tuple that connects to _on_params_changed; in v16 it was missing,
       so editing this spinbox did nothing.
    2) _connect_signals: the dead-code `if isinstance(QDoubleSpinBox) else`
       branch with identical bodies has been collapsed into a single line.
    3) Default samp_rate -> 1 MHz (was 960 kHz). 1 MHz is an integer divisor
       of the B210 master clock (52 MHz), making the rate exact.
    4) Default tx_gain / rx_gain -> 20 dB (was 40 dB). B210 self-loopback
       at 40+40 dB drives the RX ADC well into saturation; 20+20 dB is
       a safe starting point. The user can still raise them.
    5) The group title and the note next to the parameter block were
       updated to describe the v17 channel-estimation flow (single known
       pilot frame in place of the v16 M*N impulse probes).
    6) _apply_params_to_backend now also pushes the current TX text into
       tx_text_view so the displayed TX text doesn't lag the edit field.
    7) "探测保护" and "最大阶数" spinboxes are kept and still passed
       through configure() so old configs do not break, but they are
       silently ignored by the v17 backend.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QPushButton,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QTextEdit, QSplitter,
    QFileDialog, QScrollArea, QSizePolicy, QCheckBox
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from hardware.fdidm_hardtest import FDIDMHardwareTest

MATLAB_BLUE = (0, 114, 189)
MATLAB_ORANGE = (217, 83, 25)
MATLAB_YELLOW = (237, 177, 32)
MATLAB_PURPLE = (126, 47, 142)
LIGHT_BG = (250, 250, 250)
AXIS_COLOR = (60, 60, 60)
BORDER_COLOR = (225, 225, 225)


class FDIDMHardwareTestTab(QWidget):
    """FDIDM paper-strict USRP hardware verification UI (v17).

    Key properties of the v17 link:
      - no OFDM pilot holes inside the data grid;
      - full M x N FDIDM payload grid;
      - a single known random-QPSK pilot frame is transmitted directly
        before the data frame, replacing v16's M*N impulse probes;
      - cross-domain ZF / MMSE on a diagonal H_TF, mathematically
        equivalent to the paper's matrix-form equalization.
    """

    def __init__(self):
        super().__init__()
        self.backend = None
        self.test_running = False
        self.last_status_error = ""
        self._evm_history = deque(maxlen=300)
        self._evm_index = 0
        self._last_plot_samp_rate = None
        self._last_runtime_log_time = 0.0
        self._hot_update_in_progress = False
        # v17.1: cursor for incremental debug-log streaming from the backend.
        # Each refresh tick drains entries with seq > _last_debug_seq.
        self._last_debug_seq = 0
        # Level filter for the auto-stream; user can switch in code if desired.
        # Default INFO keeps the log readable; flip to "DEBUG" for full firehose.
        self._auto_debug_level = "INFO"

        self._init_ui()
        self._init_plot_style()
        self._connect_signals()
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._refresh_plots)

    # =========================================================
    # UI
    # =========================================================
    def _init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        splitter.addWidget(self._create_controls_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setSizes([520, 980])

    def _create_controls_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(500)
        scroll.setMaximumWidth(620)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        hw_group = QGroupBox("USRP 硬件配置")
        hw = QGridLayout(hw_group)
        hw.setHorizontalSpacing(8)
        hw.setVerticalSpacing(8)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["USRP B210", "USRP N210", "USRP X310"])
        # v17: default 1 MHz, exact integer divisor of B210 master clock 52 MHz.
        self.samp_rate_spin = self._dspin(1e5, 100e6, 1_000_000, 0, " Hz")
        self.fc_spin = self._dspin(70e6, 6e9, 2.4e9, 0, " Hz")
        hw.addWidget(QLabel("设备类型"), 0, 0)
        hw.addWidget(self.device_combo, 0, 1)
        hw.addWidget(QLabel("采样率"), 1, 0)
        hw.addWidget(self.samp_rate_spin, 1, 1)
        hw.addWidget(QLabel("中心频率"), 2, 0)
        hw.addWidget(self.fc_spin, 2, 1)
        layout.addWidget(hw_group)

        fd_group = QGroupBox("FDIDM 论文严格链路参数 v17（pilot 信道估计）")
        fd = QGridLayout(fd_group)
        fd.setHorizontalSpacing(8)
        fd.setVerticalSpacing(8)
        self.alpha_spin = self._dspin(-2.0, 2.0, 0.5, 1, "", step=0.1)
        self.beta_spin = self._dspin(-2.0, 2.0, 1.0, 1, "", step=0.1)
        self.m_spin = self._spin(4, 64, 16)
        self.n_spin = self._spin(1, 64, 16)
        self.cp_spin = self._spin(0, 63, 4)
        self.max_order_spin = self._spin(16, 4096, 1024)
        self.frame_count_spin = self._spin(1, 32, 4)
        self.guard_spin = self._spin(0, 8192, 64)
        self.probe_guard_spin = self._spin(0, 8192, 16)
        self.evm_avg_spin = self._spin(1, 128, 8)
        self.train_amp_spin = self._dspin(0.05, 4.0, 1.0, 2, "", step=0.05)
        fd.addWidget(QLabel("α"), 0, 0)
        fd.addWidget(self.alpha_spin, 0, 1)
        fd.addWidget(QLabel("β"), 0, 2)
        fd.addWidget(self.beta_spin, 0, 3)
        fd.addWidget(QLabel("M"), 1, 0)
        fd.addWidget(self.m_spin, 1, 1)
        fd.addWidget(QLabel("N"), 1, 2)
        fd.addWidget(self.n_spin, 1, 3)
        fd.addWidget(QLabel("CP"), 2, 0)
        fd.addWidget(self.cp_spin, 2, 1)
        fd.addWidget(QLabel("最大阶数*"), 2, 2)
        fd.addWidget(self.max_order_spin, 2, 3)
        fd.addWidget(QLabel("物理帧"), 3, 0)
        fd.addWidget(self.frame_count_spin, 3, 1)
        fd.addWidget(QLabel("帧间保护"), 3, 2)
        fd.addWidget(self.guard_spin, 3, 3)
        fd.addWidget(QLabel("探测保护*"), 4, 0)
        fd.addWidget(self.probe_guard_spin, 4, 1)
        fd.addWidget(QLabel("EVM平均"), 4, 2)
        fd.addWidget(self.evm_avg_spin, 4, 3)
        fd.addWidget(QLabel("Pilot 幅度"), 5, 0)
        fd.addWidget(self.train_amp_spin, 5, 1)
        self.btn_ofdm = QPushButton("OFDM 特例\nα=0 β=0")
        self.btn_otfs = QPushButton("OTFS 特例\nα=1 β=1")
        self.btn_reco = QPushButton("推荐初值\nα=0.5 β=1")
        self.btn_apply_params = QPushButton("应用 FDIDM 参数")
        fd.addWidget(self.btn_ofdm, 6, 0, 1, 2)
        fd.addWidget(self.btn_otfs, 6, 2, 1, 2)
        fd.addWidget(self.btn_reco, 7, 0, 1, 2)
        fd.addWidget(self.btn_apply_params, 7, 2, 1, 2)
        self.auto_apply_check = QCheckBox("FDIDM 参数改动后自动应用")
        self.auto_apply_check.setChecked(False)
        fd.addWidget(self.auto_apply_check, 8, 0, 1, 4)
        # v17 note: pilot-based estimation (single random QPSK pilot frame),
        # 269x -> 1x training overhead. The * marked legacy fields are kept
        # for config-file back-compat but the v17 backend silently ignores them.
        strict_note = QLabel(
            "严格模式 v17：完整 M×N 数据栅格 + 单帧已知随机 QPSK 导频信道估计 + 跨域均衡。\n"
            "Pilot 幅度 = TX 导频帧与数据帧的功率比；推荐 1.0。\n"
            "带 * 字段（最大阶数 / 探测保护）为 v16 兼容字段，v17 后端忽略。"
        )
        strict_note.setWordWrap(True)
        fd.addWidget(strict_note, 9, 0, 1, 4)
        layout.addWidget(fd_group)

        modem_group = QGroupBox("收发与解调配置")
        modem = QGridLayout(modem_group)
        # v17: lower default loopback gains. B210 internal loopback with
        # 40+40 dB saturates the RX ADC; 20+20 dB is the safer starting point.
        self.tx_gain_spin = self._dspin(0, 80, 20, 1, " dB")
        self.rx_gain_spin = self._dspin(0, 80, 20, 1, " dB")
        self.mod_order_combo = QComboBox()
        self.mod_order_combo.addItems(["QPSK", "16QAM", "64QAM"])
        self.equalizer_combo = QComboBox()
        self.equalizer_combo.addItems(["MMSE", "ZF"])
        self.const_mode_combo = QComboBox()
        self._const_mode_items = [
            ("决策整形（推荐）", "dd_refined"),
            ("原始软符号", "raw"),
            ("硬判决", "hard_decision"),
        ]
        for label, _ in self._const_mode_items:
            self.const_mode_combo.addItem(label)
        self.tx_plot_combo = QComboBox()
        self.tx_plot_combo.addItems(["TX 基带频谱", "TX 多帧时域幅度", "X_TF 栅格幅度", "X 跨域星座"])
        modem.addWidget(QLabel("发送增益"), 0, 0)
        modem.addWidget(self.tx_gain_spin, 0, 1)
        modem.addWidget(QLabel("接收增益"), 1, 0)
        modem.addWidget(self.rx_gain_spin, 1, 1)
        modem.addWidget(QLabel("调制方式"), 2, 0)
        modem.addWidget(self.mod_order_combo, 2, 1)
        modem.addWidget(QLabel("均衡器"), 3, 0)
        modem.addWidget(self.equalizer_combo, 3, 1)
        modem.addWidget(QLabel("星座图"), 4, 0)
        modem.addWidget(self.const_mode_combo, 4, 1)
        modem.addWidget(QLabel("TX 图"), 5, 0)
        modem.addWidget(self.tx_plot_combo, 5, 1)
        layout.addWidget(modem_group)

        text_group = QGroupBox("发送文本 / 文本文件")
        text_l = QVBoxLayout(text_group)
        self.btn_load_text = QPushButton("加载文本文件")
        self.btn_reset_text = QPushButton("恢复默认文本")
        self.file_path_label = QLabel("可直接编辑下方文本，或加载 .txt 文件")
        self.file_path_label.setWordWrap(True)
        self.tx_text_edit = QTextEdit()
        self.tx_text_edit.setPlainText("Hello FDIDM Paper Strict Test!")
        self.tx_text_edit.setMaximumHeight(110)
        text_l.addWidget(self.btn_load_text)
        text_l.addWidget(self.btn_reset_text)
        text_l.addWidget(self.file_path_label)
        text_l.addWidget(QLabel("待发送文本"))
        text_l.addWidget(self.tx_text_edit)
        layout.addWidget(text_group)

        btn_group = QGroupBox("操作控制")
        btn_l = QVBoxLayout(btn_group)
        self.btn_connect = QPushButton("连接 / 配置 USRP")
        self.btn_start_test = QPushButton("开始测试")
        self.btn_stop_test = QPushButton("停止测试")
        # v17.1: preview button removed per user request.
        # The TX plot now always reflects the live vector_source contents
        # (with a stopped-mode fallback to the cached _tx_waveform inside
        # the backend, so the plot keeps working even before the first start).
        self.btn_dump_debug = QPushButton("导出最近调试日志（200 条）到日志窗口")
        self.btn_start_test.setEnabled(False)
        self.btn_stop_test.setEnabled(False)
        btn_l.addWidget(self.btn_connect)
        btn_l.addWidget(self.btn_start_test)
        btn_l.addWidget(self.btn_stop_test)
        btn_l.addWidget(self.btn_dump_debug)
        layout.addWidget(btn_group)
        layout.addStretch()
        scroll.setWidget(panel)
        self._apply_control_style(scroll)
        return scroll

    def _create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        plot_panel = QWidget()
        grid = QGridLayout(plot_panel)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        self.tx_plot = pg.PlotWidget(title="FDIDM 发送端显示")
        self.tx_plot.setLabel("left", "幅度")
        self.tx_plot.setLabel("bottom", "频率/样点")
        self.tx_plot.showGrid(x=True, y=True)
        self.rx_spectrum_plot = pg.PlotWidget(title="USRP 接收基带频谱")
        self.rx_spectrum_plot.setLabel("left", "幅度", units="dB")
        self.rx_spectrum_plot.setLabel("bottom", "频率", units="Hz")
        self.rx_spectrum_plot.showGrid(x=True, y=True)
        self.evm_plot = pg.PlotWidget(title="EVM 曲线")
        self.evm_plot.setLabel("left", "EVM RMS", units="%")
        self.evm_plot.setLabel("bottom", "刷新次数")
        self.evm_plot.showGrid(x=True, y=True)
        self.constellation_plot = pg.PlotWidget(title="跨域接收星座图")
        self.constellation_plot.setLabel("left", "Q")
        self.constellation_plot.setLabel("bottom", "I")
        self.constellation_plot.setAspectLocked(True)
        self.constellation_plot.showGrid(x=True, y=True)
        self.constellation_plot.disableAutoRange()
        self.constellation_plot.setXRange(-2, 2, padding=0)
        self.constellation_plot.setYRange(-2, 2, padding=0)
        grid.addWidget(self.tx_plot, 0, 0)
        grid.addWidget(self.rx_spectrum_plot, 0, 1)
        grid.addWidget(self.evm_plot, 1, 0)
        grid.addWidget(self.constellation_plot, 1, 1)
        splitter.addWidget(plot_panel)

        text_panel = QWidget()
        text_grid = QGridLayout(text_panel)
        self.decode_status_label = QLabel("解调状态：未开始")
        self.decode_status_label.setMinimumHeight(28)
        self.decode_status_label.setMaximumHeight(42)
        self.decode_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.decode_status_label.setWordWrap(True)
        self.tx_text_view = QTextEdit(); self.tx_text_view.setReadOnly(True); self.tx_text_view.setMaximumHeight(120)
        self.rx_text_view = QTextEdit(); self.rx_text_view.setReadOnly(True); self.rx_text_view.setMaximumHeight(120)
        text_grid.addWidget(self.decode_status_label, 0, 0, 1, 2)
        text_grid.addWidget(QLabel("发送文本"), 1, 0)
        text_grid.addWidget(QLabel("接收解调文本"), 1, 1)
        text_grid.addWidget(self.tx_text_view, 2, 0)
        text_grid.addWidget(self.rx_text_view, 2, 1)
        splitter.addWidget(text_panel)

        log_panel = QWidget()
        log_l = QVBoxLayout(log_panel)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True); self.log_text.setMaximumHeight(220)
        log_l.addWidget(QLabel("测试日志"))
        log_l.addWidget(self.log_text)
        splitter.addWidget(log_panel)
        splitter.setSizes([560, 180, 220])

        self.tx_curve = self.tx_plot.plot(pen=pg.mkPen(MATLAB_BLUE, width=2))
        self.rx_curve = self.rx_spectrum_plot.plot(pen=pg.mkPen(MATLAB_ORANGE, width=2))
        self.evm_curve = self.evm_plot.plot(pen=pg.mkPen(MATLAB_PURPLE, width=2))
        self.constellation_scatter = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(MATLAB_YELLOW[0], MATLAB_YELLOW[1], MATLAB_YELLOW[2], 160))
        self.constellation_plot.addItem(self.constellation_scatter)
        return panel

    def _dspin(self, lo, hi, val, dec, suffix="", step=None):
        s = QDoubleSpinBox()
        s.setRange(float(lo), float(hi))
        s.setValue(float(val))
        s.setDecimals(int(dec))
        s.setSuffix(str(suffix))
        if step is not None:
            s.setSingleStep(float(step))
        s.setMinimumHeight(30)
        return s

    def _spin(self, lo, hi, val):
        s = QSpinBox()
        s.setRange(int(lo), int(hi))
        s.setValue(int(val))
        s.setMinimumHeight(30)
        return s

    def _apply_control_style(self, widget):
        widget.setStyleSheet("""
            QGroupBox { font-weight: 600; border: 1px solid #d0d0d0; border-radius: 8px; margin-top: 10px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QLabel { min-height: 24px; }
            QPushButton { min-height: 32px; padding: 5px 8px; }
            QComboBox, QSpinBox, QDoubleSpinBox { min-height: 28px; }
        """)

    def _init_plot_style(self):
        for p in (self.tx_plot, self.rx_spectrum_plot, self.evm_plot, self.constellation_plot):
            p.setBackground(LIGHT_BG)
            p.showGrid(x=True, y=True, alpha=0.35)
            for axis_name in ("left", "bottom"):
                axis = p.getAxis(axis_name)
                axis.setPen(pg.mkPen(AXIS_COLOR, width=1.0))
                axis.setTextPen(pg.mkPen(AXIS_COLOR, width=1.0))
            p.getPlotItem().getViewBox().setBorder(pg.mkPen(BORDER_COLOR, width=1.0))
        for p in (self.tx_plot, self.rx_spectrum_plot, self.evm_plot):
            p.disableAutoRange()
            p.setMouseEnabled(x=False, y=False)
        self.rx_spectrum_plot.setYRange(-120, 20, padding=0)
        self.evm_plot.setYRange(0, 100, padding=0)
        self.evm_plot.setXRange(0, 300, padding=0)
        self._apply_stable_plot_ranges(self.samp_rate_spin.value())

    # =========================================================
    # Signals
    # =========================================================
    def _connect_signals(self):
        self.btn_connect.clicked.connect(self._on_connect_clicked)
        self.btn_start_test.clicked.connect(self._on_start_test_clicked)
        self.btn_stop_test.clicked.connect(self._on_stop_test_clicked)
        # v17.1: preview button removed. Manual debug-log dump replaces it.
        self.btn_dump_debug.clicked.connect(self._on_dump_debug_clicked)
        self.btn_load_text.clicked.connect(self._on_load_text_clicked)
        self.btn_reset_text.clicked.connect(self._on_reset_text_clicked)
        self.btn_apply_params.clicked.connect(self._apply_params_to_backend)
        self.btn_ofdm.clicked.connect(lambda: self._set_indices(0.0, 0.0))
        self.btn_otfs.clicked.connect(lambda: self._set_indices(1.0, 1.0))
        self.btn_reco.clicked.connect(lambda: self._set_indices(0.5, 1.0))
        # v17 bug fix carried over: probe_guard_spin is in the tuple (was
        # missing in v16); the dead if/else with identical branches is gone.
        for w in (self.alpha_spin, self.beta_spin, self.m_spin, self.n_spin,
                  self.cp_spin, self.frame_count_spin, self.guard_spin,
                  self.probe_guard_spin, self.evm_avg_spin,
                  self.train_amp_spin, self.max_order_spin):
            w.valueChanged.connect(self._on_params_changed)
        self.mod_order_combo.currentTextChanged.connect(self._on_mod_or_eq_changed)
        self.equalizer_combo.currentTextChanged.connect(self._on_mod_or_eq_changed)
        self.tx_gain_spin.valueChanged.connect(lambda v: self._apply_gain("tx", float(v)))
        self.rx_gain_spin.valueChanged.connect(lambda v: self._apply_gain("rx", float(v)))
        self.const_mode_combo.currentIndexChanged.connect(lambda _: self._push_const_mode())
        # v17.1: TX plot mode changes just re-style; live data flows from the
        # backend buffers regardless of running state, no preview path needed.
        self.tx_plot_combo.currentIndexChanged.connect(lambda _: self._refresh_tx_plot_only())

    # =========================================================
    # Button logic
    # =========================================================
    def _on_connect_clicked(self):
        try:
            self._create_backend()
            self.tx_text_view.setPlainText(self.tx_text_edit.toPlainText())
            self.rx_text_view.clear()
            self._push_const_mode()
            self._log("FDIDM 严格论文链路后端已配置 v17.1（pilot 信道估计 + α/β 同步修复 + 调试日志流）。")
            self._log(self._backend_summary())
            self.btn_connect.setEnabled(False)
            self.btn_start_test.setEnabled(True)
            self._set_hw_controls_enabled(False)
            # Initial debug-log drain so the user sees what configure produced.
            self._drain_debug_to_log()
            self._refresh_tx_plot_only()
        except Exception as e:
            self.backend = None
            self._log(f"连接 / 配置 FDIDM 严格后端失败: {type(e).__name__}: {e}")

    def _on_start_test_clicked(self):
        try:
            if self.backend is None:
                self._create_backend()
            else:
                self._configure_backend(self.tx_text_edit.toPlainText())
            self.tx_text_view.setPlainText(self.tx_text_edit.toPlainText())
            self.rx_text_view.clear()
            self.decode_status_label.setText("解调状态：v17.1 运行中，等待 USRP 接收帧…")
            self._reset_runtime_curves()
            self.backend.start()
            self.test_running = True
            self.btn_start_test.setEnabled(False)
            self.btn_stop_test.setEnabled(True)
            self._set_test_controls_enabled(False)
            self.update_timer.start(100)
            self._log("v17.1 硬件测试已启动。")
            self._log(self._backend_summary())
        except Exception as e:
            self.test_running = False
            self._log(f"开始测试失败: {type(e).__name__}: {e}")

    def _on_stop_test_clicked(self):
        self.test_running = False
        self.update_timer.stop()
        if self.backend is not None:
            try:
                self.backend.stop()
                if hasattr(self.backend, "wait"):
                    self.backend.wait()
            except Exception as e:
                self._log(f"停止后端时出错: {e}")
        self.btn_start_test.setEnabled(True)
        self.btn_stop_test.setEnabled(False)
        self.btn_connect.setEnabled(True)
        self._set_hw_controls_enabled(True)
        self._set_test_controls_enabled(True)
        self._clear_plots()
        self.decode_status_label.setText("解调状态：已停止")
        self._log("停止严格 FDIDM 硬件测试。")

    def _on_load_text_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文本文件", "", "Text Files (*.txt);;All Files (*)")
        if not file_path:
            return
        text = self._read_text_file(file_path)
        self.tx_text_edit.setPlainText(text)
        self.file_path_label.setText(file_path)
        self.tx_text_view.setPlainText(text)
        self._log(f"已加载文本文件: {file_path}")

    def _on_reset_text_clicked(self):
        if self.test_running:
            self._log("运行中不能修改发送文本，请先停止测试。")
            return
        text = "Hello FDIDM Paper Strict Test!"
        self.tx_text_edit.setPlainText(text)
        self.tx_text_view.setPlainText(text)
        self.file_path_label.setText("可直接编辑下方文本，或加载 .txt 文件")
        self._log("已恢复严格 FDIDM 默认文本。")

    # =========================================================
    # Backend
    # =========================================================
    def _create_backend(self):
        # Backend is being recreated -> reset the debug-log cursor so we get
        # the boot-time INFO messages flowing into the visible log.
        self._last_debug_seq = 0
        self.backend = FDIDMHardwareTest(
            carrier_freq=self.fc_spin.value(),
            samp_rate=self.samp_rate_spin.value(),
            tx_gain=self.tx_gain_spin.value(),
            rx_gain=self.rx_gain_spin.value(),
            device_type=self.device_combo.currentText(),
            tx_text=self.tx_text_edit.toPlainText(),
            mod_order=self.mod_order_combo.currentText(),
            equalizer=self.equalizer_combo.currentText(),
            alpha=self.alpha_spin.value(),
            beta=self.beta_spin.value(),
            fdidm_m=self.m_spin.value(),
            fdidm_n=self.n_spin.value(),
            cp_len=self.cp_spin.value(),
            tx_frame_count=self.frame_count_spin.value(),
            inter_frame_guard_len=self.guard_spin.value(),
            evm_average_frames=self.evm_avg_spin.value(),
            training_amplitude=self.train_amp_spin.value(),
            # The next two are legacy fields. v17 accepts them for back-compat
            # but does not use them; they appear in get_status as-is so old
            # logs and configs keep working.
            training_probe_guard_len=self.probe_guard_spin.value(),
            max_full_htf_order=self.max_order_spin.value(),
        )

    def _configure_backend(self, tx_text: str):
        if self.backend is None:
            self._create_backend()
            return
        self.backend.configure(
            carrier_freq=self.fc_spin.value(),
            samp_rate=self.samp_rate_spin.value(),
            tx_gain=self.tx_gain_spin.value(),
            rx_gain=self.rx_gain_spin.value(),
            tx_text=tx_text,
            mod_order=self.mod_order_combo.currentText(),
            equalizer=self.equalizer_combo.currentText(),
            alpha=self.alpha_spin.value(),
            beta=self.beta_spin.value(),
            fdidm_m=self.m_spin.value(),
            fdidm_n=self.n_spin.value(),
            cp_len=self.cp_spin.value(),
            tx_frame_count=self.frame_count_spin.value(),
            inter_frame_guard_len=self.guard_spin.value(),
            evm_average_frames=self.evm_avg_spin.value(),
            training_amplitude=self.train_amp_spin.value(),
            training_probe_guard_len=self.probe_guard_spin.value(),
            max_full_htf_order=self.max_order_spin.value(),
        )
        self._push_const_mode()

    def _apply_params_to_backend(self):
        if self.backend is None:
            # Nothing to push to. Just leave the spinbox values for next connect().
            return
        was_running = self.test_running
        try:
            if was_running:
                self.update_timer.stop()
                self.backend.stop()
            self._configure_backend(self.tx_text_edit.toPlainText())
            # Sync the visible TX text view immediately so the user sees what
            # was actually pushed to the backend.
            self.tx_text_view.setPlainText(self.tx_text_edit.toPlainText())
            if was_running:
                self.backend.start()
                self.update_timer.start(100)
            self._reset_runtime_curves()
            self._refresh_tx_plot_only()
            self._log("严格 FDIDM 参数已应用。")
            self._log(self._backend_summary())
            # Drain any new debug entries (configure/build top block produce a few INFOs).
            self._drain_debug_to_log()
        except Exception as e:
            self._log(f"应用严格 FDIDM 参数失败: {type(e).__name__}: {e}")
        finally:
            self.test_running = was_running

    def _set_indices(self, alpha: float, beta: float):
        self.alpha_spin.setValue(float(alpha))
        self.beta_spin.setValue(float(beta))
        if not self.auto_apply_check.isChecked():
            self._apply_params_to_backend()

    def _on_params_changed(self, *_args):
        if self._hot_update_in_progress:
            return
        if self.auto_apply_check.isChecked():
            self._apply_params_to_backend()

    def _on_mod_or_eq_changed(self, *_args):
        if self.backend is None:
            return
        self._apply_params_to_backend()

    def _apply_gain(self, which: str, value: float):
        if self.backend is None or not self.test_running:
            return
        try:
            if which == "tx":
                self.backend.set_tx_gain(value)
                self._log(f"发送增益热更新 -> {value:.1f} dB")
            else:
                self.backend.set_rx_gain(value)
                self._log(f"接收增益热更新 -> {value:.1f} dB")
        except Exception as e:
            self._log(f"增益热更新失败: {e}")

    def _push_const_mode(self):
        if self.backend is None:
            return False
        idx = self.const_mode_combo.currentIndex()
        mode = self._const_mode_items[idx][1] if 0 <= idx < len(self._const_mode_items) else "dd_refined"
        try:
            self.backend.set_constellation_display_mode(mode)
            return True
        except Exception as e:
            self._log(f"设置星座图显示模式失败: {e}")
            return False

    # =========================================================
    # Refresh
    # =========================================================
    def _refresh_plots(self):
        if self.backend is None:
            return
        try:
            status = self.backend.get_status()
            stats = self.backend.get_decode_stats()
            samp_rate = self._extract_samp_rate(status)
            self._apply_stable_plot_ranges(samp_rate)
            current_error = status.get("last_error", "")
            if current_error and current_error != self.last_status_error:
                self.last_status_error = current_error
                self._log(f"后端状态异常: {current_error}")
            self._update_tx_plot(samp_rate)
            rx_signal = self.backend.get_rx_spectrum_source(4096)
            rx_freq, rx_psd = self._compute_spectrum(rx_signal, samp_rate, 1024)
            self.rx_curve.setData(rx_freq, rx_psd)
            evm = float(status.get("evm_percent", np.nan))
            self._update_evm_plot(evm)
            constellation = self.backend.get_rx_constellation(512)
            if constellation is not None and len(constellation) > 0:
                self.constellation_scatter.setData(x=np.real(constellation), y=np.imag(constellation))
            else:
                self.constellation_scatter.setData(x=[], y=[])
            self.tx_text_view.setPlainText(self.tx_text_edit.toPlainText())
            self.rx_text_view.setPlainText(self.backend.get_rx_text())
            self._update_decode_status(stats, status)
            self._maybe_log_runtime(status, stats)
            # v17.1: incremental debug-log streaming into the visible log panel.
            self._drain_debug_to_log()
        except Exception as e:
            self._log(f"刷新失败: {type(e).__name__}: {e}")

    def _refresh_tx_plot_only(self):
        """Repaint the TX plot once (used when the dropdown style changes).

        v17.1: there is no longer a 'preview' path. The backend's
        get_tx_spectrum_source / get_tx_samples both fall back to the
        cached self._tx_waveform when the runtime buffer is empty, so
        this works both before and during start()."""
        try:
            if self.backend is None:
                return
            samp_rate = float(self.backend.get_status().get("samp_rate", self.samp_rate_spin.value()))
            self._update_tx_plot(samp_rate)
        except Exception:
            pass

    def _update_tx_plot(self, samp_rate: float):
        if self.backend is None:
            return
        mode = self.tx_plot_combo.currentText()
        if "X_TF" in mode:
            pts = self.backend.get_fdidm_preview_constellation("tf", 4096)
            y = np.abs(pts).astype(np.float64)
            x = np.arange(y.size, dtype=np.float64)
            self.tx_curve.setData(x, y)
            self.tx_plot.setTitle("X_TF 栅格幅度（论文 IFDIT 输出）")
            self.tx_plot.setXRange(0, max(10, y.size), padding=0)
            self.tx_plot.setYRange(0, max(1.5, float(np.nanmax(y)) * 1.2 if y.size else 1.5), padding=0)
        elif "X 跨域" in mode:
            pts = self.backend.get_fdidm_preview_constellation("cross", 4096)
            self.tx_curve.setData(np.real(pts), np.imag(pts))
            self.tx_plot.setTitle("X 跨域发送星座（QAM 映射后）")
            self.tx_plot.setXRange(-2, 2, padding=0)
            self.tx_plot.setYRange(-2, 2, padding=0)
        elif "时域" in mode:
            # Live TX samples; backend falls back to cached waveform when no buffer.
            sig = self.backend.get_tx_spectrum_source(4096)
            y = np.abs(np.asarray(sig, dtype=np.complex64))
            x = np.arange(y.size, dtype=np.float64)
            self.tx_curve.setData(x, y)
            self.tx_plot.setTitle("TX 多帧时域幅度（实时）")
            self.tx_plot.setXRange(0, max(10, y.size), padding=0)
            self.tx_plot.setYRange(0, max(0.5, float(np.nanmax(y)) * 1.2 if y.size else 0.5), padding=0)
        else:
            sig = self.backend.get_tx_spectrum_source(4096)
            freq, psd = self._compute_spectrum(sig, samp_rate, 1024)
            self.tx_curve.setData(freq, psd)
            self.tx_plot.setTitle("TX 基带频谱（USRP 实时发射流）")
            self.tx_plot.setXRange(-samp_rate / 2, samp_rate / 2, padding=0)
            self.tx_plot.setYRange(-120, 20, padding=0)

    def _update_decode_status(self, stats, status):
        ok = bool(stats.get("decode_ok", False))
        head = "解调状态：v17.1 CRC 通过，文本已恢复" if ok else "解调状态：v17.1 尚未恢复"
        evm_avg = float(status.get("evm_average_percent", np.nan))
        evm_inst = float(status.get("evm_instant_percent", np.nan))
        evm_avg_text = "nan" if not np.isfinite(evm_avg) else f"{evm_avg:.2f}%"
        evm_inst_text = "nan" if not np.isfinite(evm_inst) else f"{evm_inst:.2f}%"
        fp = int(status.get("frames_processed", 0))
        fd = int(status.get("frames_decode_ok", 0))
        cyc = int(status.get("monitor_cycles", 0))
        self.decode_status_label.setText(
            f"{head} ({stats.get('match_bytes',0)}/{stats.get('expected_bytes',0)} bytes) | "
            f"frames {fd}/{fp}, cyc={cyc} | "
            f"Sync={float(status.get('sync_metric',0.0)):.3f}, CFO={float(status.get('cfo_est_hz',0.0)):.1f} Hz, "
            f"EVM(avg)={evm_avg_text}({int(status.get('evm_average_count',0))}/{int(status.get('evm_average_frames',1))}), "
            f"EVM(inst)={evm_inst_text}, Hleak={float(status.get('htf_leakage',0.0)):.3f}, "
            f"cond={float(status.get('cond_h_cross',np.nan)):.2e}, "
            f"resφ={float(status.get('residual_phase_deg',np.nan)):.1f}°, RX={int(status.get('rx_samples_seen',0))}"
        )

    def _maybe_log_runtime(self, status, stats):
        now = time.monotonic()
        if now - self._last_runtime_log_time < 2.0:
            return
        self._last_runtime_log_time = now
        self._log(
            "v17.1 runtime: "
            f"reason={status.get('reason','')}, "
            f"frames={int(status.get('frames_decode_ok',0))}/{int(status.get('frames_processed',0))} "
            f"(decode_ok/processed), monitor_cycles={int(status.get('monitor_cycles',0))}, "
            f"TX_buf={int(status.get('tx_buf_size',0))}, RX_buf={int(status.get('rx_buf_size',0))}, "
            f"rx_seen={int(status.get('rx_samples_seen',0))}, "
            f"Sync={float(status.get('sync_metric',0.0)):.3f}, "
            f"CFO={float(status.get('cfo_est_hz',0.0)):.1f} Hz, "
            f"EVMavg={float(status.get('evm_average_percent',np.nan)):.2f}%, "
            f"Hleak={float(status.get('htf_leakage',0.0)):.3f}, "
            f"cond={float(status.get('cond_h_cross',np.nan)):.2e}, "
            f"noise_var={float(status.get('noise_var',np.nan)):.2e}, "
            f"alpha/beta={float(status.get('alpha',0)):.3f}/{float(status.get('beta',0)):.3f}, "
            f"decode_ok={bool(stats.get('decode_ok',False))}, "
            f"match={int(stats.get('match_bytes',0))}/{int(stats.get('expected_bytes',0))}"
        )

    # =========================================================
    # v17.1 debug-log plumbing
    # =========================================================
    def _drain_debug_to_log(self):
        """Pull new backend debug entries (seq > cursor) into the visible log panel.

        Called from _refresh_plots on every UI tick. Filters by self._auto_debug_level
        so the panel doesn't drown in DEBUG-level chatter; users can dump everything
        on demand via _on_dump_debug_clicked.
        """
        if self.backend is None:
            return
        try:
            entries = self.backend.drain_debug_log(
                since_seq=int(self._last_debug_seq),
                max_entries=200,
                min_level=self._auto_debug_level,
            )
        except Exception:
            return
        if not entries:
            return
        for e in entries:
            self._log(
                f"BE[seq={e['seq']:04d} t={e['t']:7.3f}s {e['level']:<5}] {e['msg']}"
            )
            seq = int(e.get("seq", 0))
            if seq > self._last_debug_seq:
                self._last_debug_seq = seq

    def _on_dump_debug_clicked(self):
        """One-shot dump of the last 200 entries (all levels) into the log panel.

        Use this to copy a full diagnostic trace back to me when something
        misbehaves; the auto-stream above only shows INFO+ by default.
        """
        if self.backend is None:
            self._log("尚未连接后端，没有调试日志。")
            return
        try:
            entries = self.backend.get_debug_log(max_entries=200, min_level="DEBUG")
        except Exception as e:
            self._log(f"读取调试日志失败: {type(e).__name__}: {e}")
            return
        if not entries:
            self._log("调试日志为空。")
            return
        self._log(f"===== DUMP: 最近 {len(entries)} 条后端调试条目 (全级别) =====")
        for e in entries:
            self._log(
                f"BE[seq={e['seq']:04d} t={e['t']:7.3f}s {e['level']:<5}] {e['msg']}"
            )
        self._log(f"===== END DUMP ({len(entries)} entries) =====")
        last_seq = int(entries[-1]["seq"])
        if last_seq > self._last_debug_seq:
            self._last_debug_seq = last_seq

    # =========================================================
    # Utility
    # =========================================================
    def _compute_spectrum(self, samples, samp_rate, seg_len=1024):
        samples = np.asarray(samples, dtype=np.complex128)
        if samples.size == 0:
            return np.array([]), np.array([])
        seg_len = min(int(seg_len), len(samples))
        if seg_len <= 0:
            return np.array([]), np.array([])
        n_seg = max(1, len(samples) // seg_len)
        trimmed = samples[-n_seg * seg_len:]
        blocks = trimmed.reshape(n_seg, seg_len)
        window = np.hanning(seg_len).astype(np.float64)
        psd = np.zeros(seg_len, dtype=np.float64)
        for blk in blocks:
            spec = np.fft.fftshift(np.fft.fft(blk * window))
            psd += np.abs(spec) ** 2
        psd /= max(n_seg, 1)
        spectrum_db = 10.0 * np.log10(psd + 1e-12)
        freq_axis = np.linspace(-samp_rate / 2, samp_rate / 2, seg_len, endpoint=False)
        return freq_axis, spectrum_db

    def _update_evm_plot(self, evm_value):
        try:
            evm = float(evm_value)
        except Exception:
            evm = np.nan
        if np.isfinite(evm) and evm >= 0:
            self._evm_history.append((self._evm_index, evm))
            self._evm_index += 1
        if len(self._evm_history) == 0:
            self.evm_curve.setData([], [])
            return
        x = np.array([p[0] for p in self._evm_history], dtype=np.float64)
        y = np.array([p[1] for p in self._evm_history], dtype=np.float64)
        self.evm_curve.setData(x, y)
        x_max = max(300, int(x[-1]) + 5)
        self.evm_plot.setXRange(max(0, x_max - 300), x_max, padding=0)
        self.evm_plot.setYRange(0, max(20.0, min(100.0, float(np.nanmax(y)) * 1.25)), padding=0)

    def _apply_stable_plot_ranges(self, samp_rate):
        try:
            samp_rate = float(samp_rate)
        except Exception:
            samp_rate = float(self.samp_rate_spin.value())
        if self._last_plot_samp_rate is not None and abs(self._last_plot_samp_rate - samp_rate) < 1.0:
            return
        self.rx_spectrum_plot.setXRange(-samp_rate / 2, samp_rate / 2, padding=0)
        self.rx_spectrum_plot.setYRange(-120, 20, padding=0)
        self._last_plot_samp_rate = samp_rate

    def _reset_runtime_curves(self):
        self._evm_history.clear()
        self._evm_index = 0
        self.evm_curve.setData([], [])
        self.constellation_scatter.setData(x=[], y=[])
        self.last_status_error = ""
        self._last_runtime_log_time = 0.0

    def _clear_plots(self):
        self.tx_curve.setData([], [])
        self.rx_curve.setData([], [])
        self.evm_curve.setData([], [])
        self.constellation_scatter.setData(x=[], y=[])
        self._evm_history.clear()
        self._evm_index = 0

    def _extract_samp_rate(self, status):
        return float(status.get("samp_rate", status.get("sample_rate", self.samp_rate_spin.value())))

    def _set_hw_controls_enabled(self, enabled):
        self.device_combo.setEnabled(enabled)
        self.samp_rate_spin.setEnabled(enabled)
        self.fc_spin.setEnabled(enabled)

    def _set_test_controls_enabled(self, enabled):
        self.tx_gain_spin.setEnabled(True)
        self.rx_gain_spin.setEnabled(True)
        self.tx_text_edit.setEnabled(enabled)
        self.btn_load_text.setEnabled(enabled)

    def _backend_summary(self):
        if self.backend is None:
            return "未创建后端"
        st = self.backend.get_status()
        return (
            f"链路={st.get('chain')}, MxN={st.get('fdidm_m')}x{st.get('fdidm_n')}, "
            f"CP={st.get('cp_len')}, full-H_TF阶数={st.get('full_htf_order')}, "
            f"训练块={st.get('htf_training_blocks')}, Fs={st.get('samp_rate'):.0f} Hz, "
            f"调制={st.get('mod_order')}, EQ={st.get('equalizer')}, "
            f"α={st.get('alpha'):.2f}, β={st.get('beta'):.2f}, "
            f"帧长={st.get('frame_len')} 样点 "
            f"({(st.get('frame_len',0)/max(st.get('samp_rate',1),1))*1000.0:.2f} ms)"
        )

    def _read_text_file(self, file_path):
        for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except Exception:
                continue
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def _log(self, message):
        from datetime import datetime
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
