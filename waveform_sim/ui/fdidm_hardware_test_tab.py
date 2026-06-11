# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, QSignalBlocker
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QPushButton,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QTextEdit, QSplitter,
    QScrollArea, QSizePolicy, QCheckBox, QFileDialog,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from hardware.fdidm_hardtest import FDIDMHardwareTest

MATLAB_BLUE = (0, 114, 189)
MATLAB_ORANGE = (217, 83, 25)
MATLAB_PURPLE = (126, 47, 142)
LIGHT_BG = (250, 250, 250)
AXIS_COLOR = (60, 60, 60)
BORDER_COLOR = (225, 225, 225)


class FDIDMHardwareTestTab(QWidget):
    def __init__(self):
        super().__init__()
        self.backend = None
        self.test_running = False
        self.last_status_error = ""
        self._evm_history = deque(maxlen=300)
        self._evm_index = 0
        self._last_plot_samp_rate = None
        self._last_runtime_log_time = 0.0
        self._last_debug_seq = 0
        self._auto_debug_level = "INFO"
        self._applying_params = False
        self._pending_apply = False
        self._suppress_param_signals = False
        self._init_ui()
        self._init_plot_style()
        self._connect_signals()
        self._apply_debounce_timer = QTimer(self)
        self._apply_debounce_timer.setSingleShot(True)
        self._apply_debounce_timer.timeout.connect(self._apply_params_to_backend)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._refresh_plots)

    # ---------------- UI ----------------
    def _init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        splitter.addWidget(self._create_controls_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setSizes([400, 1100])

    def _create_controls_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(430)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        hw_group = QGroupBox("链路配置")
        hw = QGridLayout(hw_group)
        hw.setHorizontalSpacing(5)
        hw.setVerticalSpacing(5)
        self.device_combo = self._combo([("B210", "USRP B210"), ("N210", "USRP N210"), ("X310", "USRP X310")])
        self.samp_rate_spin = self._dspin(1e5, 100e6, 500_000, 0, " Hz")
        self.fc_spin = self._dspin(70e6, 6e9, 2.4e9, 0, " Hz")
        hw.addWidget(QLabel("设备"), 0, 0); hw.addWidget(self.device_combo, 0, 1)
        hw.addWidget(QLabel("采样率"), 1, 0); hw.addWidget(self.samp_rate_spin, 1, 1)
        hw.addWidget(QLabel("中心频率"), 2, 0); hw.addWidget(self.fc_spin, 2, 1)
        layout.addWidget(hw_group)

        fd_group = QGroupBox("FDIDM 参数")
        fd = QGridLayout(fd_group)
        fd.setHorizontalSpacing(5)
        fd.setVerticalSpacing(5)
        self.alpha_spin = self._dspin(-2.0, 2.0, 0.5, 1, "", 0.1)
        self.beta_spin = self._dspin(-2.0, 2.0, 1.0, 1, "", 0.1)
        self.m_spin = self._spin(4, 64, 16)
        # 默认 N=16：给 rate-1/2 卷积码留出容量，同时把物理帧拉长，
        # 减少超短 TX 向量反复 wrap 对 UHD 调度的压力。
        self.n_spin = self._spin(1, 64, 16)
        self.cp_spin = self._spin(0, 63, 4)
        self.max_order_spin = self._spin(16, 4096, 1024)
        self.frame_count_spin = self._spin(1, 32, 8)
        self.guard_spin = self._spin(0, 8192, 64)
        self.evm_avg_spin = self._spin(1, 128, 8)
        self.train_amp_spin = self._dspin(0.05, 4.0, 1.0, 2, "", 0.05)
        self.channel_estimator_combo = self._combo([
            ("TDL参数", "tdl_param"), ("full-H_TF", "full_htf"), ("diag-TF", "diag_tf")
        ], current=2, chars=10)
        self.htf_update_spin = self._spin(1, 10000, 10000)
        self.htf_once_check = QCheckBox("full-H一次辨识")
        self.htf_once_check.setChecked(True)
        self.process_interval_spin = self._spin(30, 2000, 300)
        self.coding_combo = self._combo([
            ("Conv1/2+交织", "conv12"), ("无编码", "none"),
        ], current=0, chars=10)
        self.coding_interleaver_check = QCheckBox("交织")
        self.coding_interleaver_check.setChecked(True)
        self.uhd_buf_spin = self._spin(32, 4096, 2048)
        self.tx_vec_ms_spin = self._spin(0, 5000, 500)
        self.prerender_tdl_check = QCheckBox("TDL→RF固定预渲染")
        self.prerender_tdl_check.setChecked(True)
        self.prerender_tdl_check.setEnabled(False)
        self.channel_mode_combo = self._combo([
            ("RF", "rf"),
            ("RF→A", "rf_tdl_a"), ("RF→C", "rf_tdl_c"), ("RF→D", "rf_tdl_d"),
            ("A→RF", "tdl_a_rf"), ("C→RF", "tdl_c_rf"), ("D→RF", "tdl_d_rf"),
        ], current=0, chars=8)
        self.tdl_ds_spin = self._dspin(0.0, 100000.0, 1000.0, 1, " ns", 100.0)
        self.tdl_fd_spin = self._dspin(-500000.0, 500000.0, 0.0, 1, " Hz", 100.0)
        self.tdl_spread_spin = self._dspin(0.0, 500000.0, 0.0, 1, " Hz", 100.0)
        self.tdl_snr_spin = self._dspin(-10.0, 100.0, 35.0, 1, " dB", 1.0)
        self.btn_reset_hcache = QPushButton("重置CSI缓存")
        self.btn_reset_hcache.setToolTip("清除接收端缓存的 full-H 矩阵、TDL 参数化基矩阵和相关 CSI 状态；不会清空发送文本。")

        fd.addWidget(QLabel("α"), 0, 0); fd.addWidget(self.alpha_spin, 0, 1)
        fd.addWidget(QLabel("β"), 0, 2); fd.addWidget(self.beta_spin, 0, 3)
        fd.addWidget(QLabel("M"), 1, 0); fd.addWidget(self.m_spin, 1, 1)
        fd.addWidget(QLabel("N"), 1, 2); fd.addWidget(self.n_spin, 1, 3)
        fd.addWidget(QLabel("CP"), 2, 0); fd.addWidget(self.cp_spin, 2, 1)
        fd.addWidget(QLabel("maxK"), 2, 2); fd.addWidget(self.max_order_spin, 2, 3)
        fd.addWidget(QLabel("帧数"), 3, 0); fd.addWidget(self.frame_count_spin, 3, 1)
        fd.addWidget(QLabel("保护"), 3, 2); fd.addWidget(self.guard_spin, 3, 3)
        fd.addWidget(QLabel("EVM均"), 4, 0); fd.addWidget(self.evm_avg_spin, 4, 1)
        fd.addWidget(QLabel("Pilot"), 4, 2); fd.addWidget(self.train_amp_spin, 4, 3)
        fd.addWidget(QLabel("估计"), 5, 0); fd.addWidget(self.channel_estimator_combo, 5, 1)
        fd.addWidget(QLabel("H间隔"), 5, 2); fd.addWidget(self.htf_update_spin, 5, 3)
        fd.addWidget(QLabel("处理ms"), 6, 0); fd.addWidget(self.process_interval_spin, 6, 1)
        fd.addWidget(self.htf_once_check, 6, 2, 1, 2)
        fd.addWidget(QLabel("编码"), 7, 0); fd.addWidget(self.coding_combo, 7, 1)
        fd.addWidget(self.coding_interleaver_check, 7, 2, 1, 2)
        fd.addWidget(QLabel("UHD帧"), 8, 0); fd.addWidget(self.uhd_buf_spin, 8, 1)
        fd.addWidget(QLabel("TX向量ms"), 8, 2); fd.addWidget(self.tx_vec_ms_spin, 8, 3)
        fd.addWidget(QLabel("链路"), 9, 0); fd.addWidget(self.channel_mode_combo, 9, 1, 1, 3)
        fd.addWidget(QLabel("RMS-DS"), 10, 0); fd.addWidget(self.tdl_ds_spin, 10, 1)
        fd.addWidget(QLabel("Doppler"), 10, 2); fd.addWidget(self.tdl_fd_spin, 10, 3)
        fd.addWidget(QLabel("扩展"), 11, 0); fd.addWidget(self.tdl_spread_spin, 11, 1)
        fd.addWidget(QLabel("SNR"), 11, 2); fd.addWidget(self.tdl_snr_spin, 11, 3)
        fd.addWidget(self.prerender_tdl_check, 12, 0, 1, 4)
        self.btn_ofdm = QPushButton("OFDM\n0/0")
        self.btn_otfs = QPushButton("OTFS\n1/1")
        self.btn_reco = QPushButton("推荐\n0.5/1")
        self.btn_apply_params = QPushButton("应用参数")
        fd.addWidget(self.btn_ofdm, 13, 0, 1, 2)
        fd.addWidget(self.btn_otfs, 13, 2, 1, 2)
        fd.addWidget(self.btn_reco, 14, 0, 1, 2)
        fd.addWidget(self.btn_apply_params, 14, 2, 1, 2)
        fd.addWidget(self.btn_reset_hcache, 15, 0, 1, 4)
        self.auto_apply_check = QCheckBox("参数改动自动应用")
        self.auto_apply_check.setChecked(False)
        fd.addWidget(self.auto_apply_check, 16, 0, 1, 4)
        note = QLabel("v33：所有模式都经过真实RF；TDL→RF固定离线预渲染。参数切换采用原子化应用、启动丢弃脏窗口，并对CFO别名扫描增加错锁保护。")
        note.setWordWrap(True)
        fd.addWidget(note, 17, 0, 1, 4)
        layout.addWidget(fd_group)

        modem_group = QGroupBox("收发/显示")
        modem = QGridLayout(modem_group)
        modem.setHorizontalSpacing(5)
        modem.setVerticalSpacing(5)
        self.tx_gain_spin = self._dspin(0, 80, 10, 1, " dB")
        self.rx_gain_spin = self._dspin(0, 80, 20, 1, " dB")
        self.mod_order_combo = self._combo([("QPSK", "QPSK"), ("16QAM", "16QAM"), ("64QAM", "64QAM")])
        self.equalizer_combo = self._combo([("MMSE", "MMSE"), ("ZF", "ZF")])
        self._const_mode_items = [
            ("均衡后校正", "post_equalized"), ("均衡后原始", "post_equalized_raw"),
            ("均衡前观测", "pre_equalized"), ("Y_TF散点", "tf_received"),
            ("最近好帧", "last_good"), ("原始IQ", "raw_iq"),
            ("QPSK整形(显示)", "dd_refined"), ("硬判决(显示)", "hard_decision"),
        ]
        self.const_mode_combo = self._combo(self._const_mode_items, chars=10)
        self.tx_plot_combo = self._combo([("TX频谱", "spec"), ("TX时域", "time"), ("X_TF幅度", "tf"), ("X星座", "cross")])
        self._rx_plot_items = [("RX原始", "raw"), ("整帧", "frame"), ("pilot", "pilot"), ("data", "data")]
        self.rx_plot_combo = self._combo(self._rx_plot_items)
        modem.addWidget(QLabel("TX增益"), 0, 0); modem.addWidget(self.tx_gain_spin, 0, 1)
        modem.addWidget(QLabel("RX增益"), 1, 0); modem.addWidget(self.rx_gain_spin, 1, 1)
        modem.addWidget(QLabel("调制"), 2, 0); modem.addWidget(self.mod_order_combo, 2, 1)
        modem.addWidget(QLabel("均衡"), 3, 0); modem.addWidget(self.equalizer_combo, 3, 1)
        modem.addWidget(QLabel("星座"), 4, 0); modem.addWidget(self.const_mode_combo, 4, 1)
        modem.addWidget(QLabel("TX图"), 5, 0); modem.addWidget(self.tx_plot_combo, 5, 1)
        modem.addWidget(QLabel("RX源"), 6, 0); modem.addWidget(self.rx_plot_combo, 6, 1)
        layout.addWidget(modem_group)

        text_group = QGroupBox("发送文本")
        text_l = QVBoxLayout(text_group)
        self.tx_text_edit = QTextEdit()
        self.tx_text_edit.setPlainText("FDIDM OK")
        self.tx_text_edit.setMaximumHeight(90)
        text_l.addWidget(QLabel("待发送文本"))
        text_l.addWidget(self.tx_text_edit)
        layout.addWidget(text_group)

        btn_group = QGroupBox("控制")
        btn_l = QVBoxLayout(btn_group)
        self.btn_connect = QPushButton("连接/配置")
        self.btn_start_test = QPushButton("开始测试")
        self.btn_stop_test = QPushButton("停止测试")
        self.btn_export_log = QPushButton("导出日志")
        self.btn_start_test.setEnabled(False)
        self.btn_stop_test.setEnabled(False)
        btn_l.addWidget(self.btn_connect)
        btn_l.addWidget(self.btn_start_test)
        btn_l.addWidget(self.btn_stop_test)
        btn_l.addWidget(self.btn_export_log)
        layout.addWidget(btn_group)
        layout.addStretch()
        self._compact_left_controls(panel)
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
        self.tx_plot = pg.PlotWidget(title="FDIDM 发送端")
        self.rx_spectrum_plot = pg.PlotWidget(title="RX频谱")
        self.evm_plot = pg.PlotWidget(title="EVM 曲线")
        self.constellation_plot = pg.PlotWidget(title="接收星座")
        for p in (self.tx_plot, self.rx_spectrum_plot, self.evm_plot, self.constellation_plot):
            p.showGrid(x=True, y=True)
        self.tx_plot.setLabel("left", "幅度")
        self.tx_plot.setLabel("bottom", "频率/样点")
        self.rx_spectrum_plot.setLabel("left", "幅度", units="dB")
        self.rx_spectrum_plot.setLabel("bottom", "频率", units="Hz")
        self.evm_plot.setLabel("left", "EVM RMS", units="%")
        self.evm_plot.setLabel("bottom", "刷新次数")
        self.constellation_plot.setLabel("left", "Q")
        self.constellation_plot.setLabel("bottom", "I")
        self.constellation_plot.setAspectLocked(True)
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
        self.decode_status_label.setMaximumHeight(48)
        self.decode_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.decode_status_label.setWordWrap(True)
        self.tx_text_view = QTextEdit(); self.tx_text_view.setReadOnly(True); self.tx_text_view.setMaximumHeight(115)
        self.rx_text_view = QTextEdit(); self.rx_text_view.setReadOnly(True); self.rx_text_view.setMaximumHeight(115)
        text_grid.addWidget(self.decode_status_label, 0, 0, 1, 2)
        text_grid.addWidget(QLabel("发送文本"), 1, 0)
        text_grid.addWidget(QLabel("接收文本"), 1, 1)
        text_grid.addWidget(self.tx_text_view, 2, 0)
        text_grid.addWidget(self.rx_text_view, 2, 1)
        splitter.addWidget(text_panel)

        log_panel = QWidget()
        log_l = QVBoxLayout(log_panel)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True); self.log_text.setMaximumHeight(220)
        log_l.addWidget(QLabel("测试日志"))
        log_l.addWidget(self.log_text)
        splitter.addWidget(log_panel)
        splitter.setSizes([560, 170, 220])

        self.tx_curve = self.tx_plot.plot(pen=pg.mkPen(MATLAB_BLUE, width=2))
        self.rx_curve = self.rx_spectrum_plot.plot(pen=pg.mkPen(MATLAB_ORANGE, width=2))
        self.evm_curve = self.evm_plot.plot(pen=pg.mkPen(MATLAB_PURPLE, width=2))
        self.constellation_scatter = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(237, 177, 32, 160))
        self.constellation_plot.addItem(self.constellation_scatter)
        return panel

    # ---------------- widget helpers ----------------
    def _combo(self, items, current=0, chars=8):
        c = QComboBox()
        for label, data in items:
            c.addItem(label, data)
        c.setCurrentIndex(int(current))
        return self._compact_combo(c, chars)

    def _compact_combo(self, combo: QComboBox, chars: int = 8):
        combo.setMinimumContentsLength(int(chars))
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        combo.setMaximumWidth(210)
        return combo

    def _compact_left_controls(self, panel):
        for combo in panel.findChildren(QComboBox):
            self._compact_combo(combo, 8)
        for edit in panel.findChildren((QSpinBox, QDoubleSpinBox)):
            edit.setMinimumWidth(68)
            edit.setMaximumWidth(116)

    def _dspin(self, lo, hi, val, dec, suffix="", step=None):
        s = QDoubleSpinBox()
        s.setRange(float(lo), float(hi))
        s.setValue(float(val))
        s.setDecimals(int(dec))
        s.setSuffix(str(suffix))
        if step is not None:
            s.setSingleStep(float(step))
        s.setMinimumHeight(26)
        s.setMaximumWidth(116)
        return s

    def _spin(self, lo, hi, val):
        s = QSpinBox()
        s.setRange(int(lo), int(hi))
        s.setValue(int(val))
        s.setMinimumHeight(26)
        s.setMaximumWidth(116)
        return s

    def _apply_control_style(self, widget):
        widget.setStyleSheet("""
            QGroupBox { font-weight: 600; border: 1px solid #d0d0d0; border-radius: 6px; margin-top: 6px; padding-top: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }
            QLabel { min-height: 20px; }
            QPushButton { min-height: 28px; padding: 3px 6px; }
            QComboBox, QSpinBox, QDoubleSpinBox { min-height: 24px; }
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

    # ---------------- signals ----------------
    def _connect_signals(self):
        self.btn_connect.clicked.connect(self._on_connect_clicked)
        self.btn_start_test.clicked.connect(self._on_start_test_clicked)
        self.btn_stop_test.clicked.connect(self._on_stop_test_clicked)
        self.btn_export_log.clicked.connect(self._on_export_log_clicked)
        self.btn_apply_params.clicked.connect(self._apply_params_to_backend)
        self.btn_ofdm.clicked.connect(lambda: self._set_indices(0.0, 0.0))
        self.btn_otfs.clicked.connect(lambda: self._set_indices(1.0, 1.0))
        self.btn_reco.clicked.connect(lambda: self._set_indices(0.5, 1.0))
        self.btn_reset_hcache.clicked.connect(self._on_reset_hcache_clicked)
        for w in (self.alpha_spin, self.beta_spin, self.m_spin, self.n_spin, self.cp_spin,
                  self.frame_count_spin, self.guard_spin, self.evm_avg_spin, self.train_amp_spin,
                  self.max_order_spin, self.htf_update_spin, self.process_interval_spin,
                  self.uhd_buf_spin, self.tx_vec_ms_spin,
                  self.tdl_ds_spin, self.tdl_fd_spin, self.tdl_spread_spin, self.tdl_snr_spin):
            w.valueChanged.connect(self._on_params_changed)
        self.htf_once_check.stateChanged.connect(lambda _: self._on_params_changed())
        self.coding_interleaver_check.stateChanged.connect(lambda _: self._on_params_changed())
        self.prerender_tdl_check.stateChanged.connect(lambda _: self._on_params_changed())
        self.coding_combo.currentIndexChanged.connect(lambda _: self._on_params_changed())
        self.channel_estimator_combo.currentIndexChanged.connect(lambda _: self._on_params_changed())
        self.channel_mode_combo.currentIndexChanged.connect(self._on_channel_mode_changed)
        self.mod_order_combo.currentTextChanged.connect(self._on_mod_or_eq_changed)
        self.equalizer_combo.currentTextChanged.connect(self._on_mod_or_eq_changed)
        self.tx_gain_spin.valueChanged.connect(lambda v: self._apply_gain("tx", float(v)))
        self.rx_gain_spin.valueChanged.connect(lambda v: self._apply_gain("rx", float(v)))
        self.const_mode_combo.currentIndexChanged.connect(lambda _: self._push_const_mode())
        self.tx_plot_combo.currentIndexChanged.connect(lambda _: self._refresh_tx_plot_only())
        self.rx_plot_combo.currentIndexChanged.connect(lambda _: self._refresh_plots())

    # ---------------- button handlers ----------------
    def _on_connect_clicked(self):
        try:
            self._create_backend()
            self.tx_text_view.setPlainText(self.tx_text_edit.toPlainText())
            self.rx_text_view.clear()
            self._push_const_mode()
            self._log("FDIDM 后端已配置 v33。")
            self._log(self._backend_summary())
            self.btn_connect.setEnabled(False)
            self.btn_start_test.setEnabled(True)
            self._set_hw_controls_enabled(False)
            self._drain_debug_to_log()
            self._refresh_tx_plot_only()
        except Exception as e:
            self.backend = None
            self._log(f"连接/配置失败: {type(e).__name__}: {e}")

    def _on_start_test_clicked(self):
        try:
            if self.backend is None:
                self._create_backend()
            else:
                self._configure_backend(self.tx_text_edit.toPlainText())
            self.tx_text_view.setPlainText(self.tx_text_edit.toPlainText())
            self.rx_text_view.clear()
            self.decode_status_label.setText("解调状态：运行中，等待接收帧…")
            self._reset_runtime_curves()
            self.backend.start()
            self.test_running = True
            self.btn_start_test.setEnabled(False)
            self.btn_stop_test.setEnabled(True)
            self._set_test_controls_enabled(False)
            self.update_timer.start(100)
            self._log("v33 测试已启动。")
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
        self._log("停止 FDIDM 测试。")

    def _on_reset_hcache_clicked(self):
        if self.backend is None:
            self._log("尚未创建后端。")
            return
        try:
            self.backend.reset_full_htf_cache()
            self._log("已重置 CSI/TDL 缓存。")
            self._drain_debug_to_log()
        except Exception as e:
            self._log(f"清空缓存失败: {type(e).__name__}: {e}")

    def _on_export_log_clicked(self):
        default_name = f"fdidm_debug_{time.strftime('%Y%m%d_%H%M%S')}.log"
        path, _ = QFileDialog.getSaveFileName(self, "导出 FDIDM 日志", default_name, "Log Files (*.log);;Text Files (*.txt);;All Files (*)")
        if not path:
            return
        try:
            if self.backend is not None and hasattr(self.backend, "export_debug_log"):
                saved = self.backend.export_debug_log(path, max_entries=5000, min_level="DEBUG")
                self._log(f"后端日志已导出：{saved}")
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.log_text.toPlainText())
                self._log(f"界面日志已导出：{path}")
        except Exception as e:
            self._log(f"导出日志失败: {type(e).__name__}: {e}")

    # ---------------- backend config ----------------
    def _current_data(self, combo: QComboBox, default: str):
        data = combo.currentData()
        return str(default if data is None else data)

    @staticmethod
    def _mode_involves_rf(mode: str) -> bool:
        """True for any link that traverses the real USRP RF path."""
        m = str(mode).lower()
        return m == "rf" or m.startswith("rf_tdl_") or m.endswith("_rf")

    def _selected_channel_estimator(self) -> str:
        est = self._current_data(self.channel_estimator_combo, "tdl_param")
        ch = self._selected_channel_mode()
        # v33 所有链路都经过真实 RF。真实 RF 响应不属于参数化 TDL 基，
        # RF+TDL 级联上的 full-H_TF 也容易病态，因此默认落到 diag-TF。
        if self._mode_involves_rf(ch) and est == "tdl_param":
            return "diag_tf"
        if self._mode_involves_rf(ch) and ch != "rf" and est == "full_htf":
            return "diag_tf"
        return est

    def _selected_channel_mode(self) -> str:
        return self._current_data(self.channel_mode_combo, "rf")

    def _selected_rx_spectrum_source(self) -> str:
        return self._current_data(self.rx_plot_combo, "raw")

    def _create_backend(self):
        self._last_debug_seq = 0
        self.backend = FDIDMHardwareTest(**self._backend_kwargs(self.tx_text_edit.toPlainText()))

    def _backend_kwargs(self, tx_text: str):
        return dict(
            carrier_freq=self.fc_spin.value(), samp_rate=self.samp_rate_spin.value(),
            tx_gain=self.tx_gain_spin.value(), rx_gain=self.rx_gain_spin.value(),
            device_type=self._current_data(self.device_combo, "USRP B210"), tx_text=tx_text,
            mod_order=self._current_data(self.mod_order_combo, "QPSK"),
            equalizer=self._current_data(self.equalizer_combo, "MMSE"),
            alpha=self.alpha_spin.value(), beta=self.beta_spin.value(),
            fdidm_m=self.m_spin.value(), fdidm_n=self.n_spin.value(), cp_len=self.cp_spin.value(),
            tx_frame_count=self.frame_count_spin.value(), inter_frame_guard_len=self.guard_spin.value(),
            evm_average_frames=self.evm_avg_spin.value(), training_amplitude=self.train_amp_spin.value(),
            training_probe_guard_len=16, max_full_htf_order=self.max_order_spin.value(),
            channel_estimator=self._selected_channel_estimator(),
            full_htf_update_interval_frames=self.htf_update_spin.value(),
            full_htf_once=self.htf_once_check.isChecked(), process_interval_ms=self.process_interval_spin.value(),
            usrp_buffer_frames=self.uhd_buf_spin.value(),
            tx_min_waveform_duration_ms=self.tx_vec_ms_spin.value(),
            tx_prerender_tdl_before_rf=True,
            coding_scheme=self._current_data(self.coding_combo, "conv12"),
            coding_interleaver=self.coding_interleaver_check.isChecked(),
            channel_mode=self._selected_channel_mode(),
            tdl_rms_delay_spread_ns=self.tdl_ds_spin.value(), tdl_doppler_hz=self.tdl_fd_spin.value(),
            tdl_doppler_spread_hz=self.tdl_spread_spin.value(), tdl_snr_db=self.tdl_snr_spin.value(),
            cfo_search_enable=True,
            cfo_search_max_hz=50_000.0,
            residual_cfo_max_hz=5_000.0,
            startup_settle_ms=800.0,
            startup_settle_windows=3,
            cfo_scan_min_score=0.55,
            cfo_scan_jump_guard_hz=12_000.0,
            auto_tdl_param_for_software=True,
        )

    def _configure_backend(self, tx_text: str):
        if self.backend is None:
            self._create_backend()
        else:
            self.backend.configure(**self._backend_kwargs(tx_text))
        self._push_const_mode()

    def _schedule_param_apply(self, delay_ms: int = 250):
        if self.backend is None or self._suppress_param_signals:
            return
        try:
            self._apply_debounce_timer.start(max(0, int(delay_ms)))
        except Exception:
            self._apply_params_to_backend()

    def _apply_params_to_backend(self):
        if self.backend is None:
            return
        if self._applying_params:
            self._pending_apply = True
            return
        self._applying_params = True
        self._pending_apply = False
        was_running = bool(self.test_running)
        try:
            self._apply_debounce_timer.stop()
        except Exception:
            pass
        try:
            if was_running:
                self.test_running = False
                self.update_timer.stop()
                self.backend.stop()
                if hasattr(self.backend, "wait"):
                    self.backend.wait()
            self._configure_backend(self.tx_text_edit.toPlainText())
            self.tx_text_view.setPlainText(self.tx_text_edit.toPlainText())
            if was_running:
                self.backend.start()
                self.test_running = True
                self.update_timer.start(100)
            self._reset_runtime_curves(); self._refresh_tx_plot_only()
            self._log("FDIDM 参数已原子化应用。")
            self._log(self._backend_summary())
            self._drain_debug_to_log()
        except Exception as e:
            self._log(f"应用参数失败: {type(e).__name__}: {e}")
            self.test_running = False
        finally:
            self._applying_params = False
            if self._pending_apply:
                self._pending_apply = False
                self._schedule_param_apply(300)

    def _set_indices(self, alpha: float, beta: float):
        self._suppress_param_signals = True
        blockers = [QSignalBlocker(self.alpha_spin), QSignalBlocker(self.beta_spin)]
        try:
            self.alpha_spin.setValue(float(alpha))
            self.beta_spin.setValue(float(beta))
        finally:
            del blockers
            self._suppress_param_signals = False
        if self.backend is not None:
            self._schedule_param_apply(0)

    def _on_params_changed(self, *_args):
        if self._suppress_param_signals:
            return
        if self.auto_apply_check.isChecked():
            self._schedule_param_apply(250)

    def _on_mod_or_eq_changed(self, *_args):
        if self.backend is not None and not self._suppress_param_signals:
            self._schedule_param_apply(0)

    def _on_channel_mode_changed(self, *_args):
        # All v33 channel modes traverse the USRP RF path, so diag-TF is the
        # safe default whenever the path selection changes.  Block the estimator
        # signal so one channel-mode click cannot produce two stop/config/start cycles.
        target = "diag_tf"
        blocker = QSignalBlocker(self.channel_estimator_combo)
        try:
            for i in range(self.channel_estimator_combo.count()):
                if self.channel_estimator_combo.itemData(i) == target:
                    self.channel_estimator_combo.setCurrentIndex(i)
                    break
        finally:
            del blocker
        if self.backend is not None:
            self._schedule_param_apply(250 if self.auto_apply_check.isChecked() else 0)

    def _apply_gain(self, which: str, value: float):
        if self.backend is None or not self.test_running:
            return
        try:
            (self.backend.set_tx_gain if which == "tx" else self.backend.set_rx_gain)(value)
        except Exception as e:
            self._log(f"增益更新失败: {e}")

    def _push_const_mode(self):
        if self.backend is None:
            return False
        try:
            self.backend.set_constellation_display_mode(self._current_data(self.const_mode_combo, "post_equalized"))
            return True
        except Exception as e:
            self._log(f"星座图模式设置失败: {e}")
            return False

    # ---------------- refresh ----------------
    def _refresh_plots(self):
        if self.backend is None:
            return
        try:
            status = self.backend.get_status(); stats = self.backend.get_decode_stats()
            samp_rate = self._extract_samp_rate(status)
            self._apply_stable_plot_ranges(samp_rate)
            err = status.get("last_error", "")
            if err and err != self.last_status_error:
                self.last_status_error = err; self._log(f"后端异常: {err}")
            self._update_tx_plot(samp_rate)
            rx_signal = self.backend.get_rx_spectrum_source(4096, source=self._selected_rx_spectrum_source())
            rx_freq, rx_psd = self._compute_spectrum(rx_signal, samp_rate, 1024)
            self.rx_curve.setData(rx_freq, rx_psd)
            stale = bool(status.get("rx_spectrum_stale", True)); age = float(status.get("rx_spectrum_stale_sec", np.nan))
            self.rx_spectrum_plot.setTitle(f"RX频谱[{self._selected_rx_spectrum_source()}] stale={stale} age={age:.1f}s")
            self._update_evm_plot(float(status.get("evm_percent", np.nan)))
            const = self.backend.get_rx_constellation(512, source=self._current_data(self.const_mode_combo, "post_equalized"))
            if const is not None and len(const) > 0:
                self.constellation_scatter.setData(x=np.real(const), y=np.imag(const))
            else:
                self.constellation_scatter.setData(x=[], y=[])
            self.constellation_plot.setTitle(
                f"星座[{self._current_data(self.const_mode_combo, 'post_equalized')}] "
                f"{status.get('constellation_source','none')}/{status.get('constellation_points',0)}pts"
            )
            self.tx_text_view.setPlainText(self.tx_text_edit.toPlainText())
            self.rx_text_view.setPlainText(self.backend.get_rx_text())
            self._update_decode_status(stats, status)
            self._maybe_log_runtime(status, stats)
            self._drain_debug_to_log()
        except Exception as e:
            self._log(f"刷新失败: {type(e).__name__}: {e}")

    def _refresh_tx_plot_only(self):
        if self.backend is None:
            return
        self._update_tx_plot(self._extract_samp_rate(self.backend.get_status()))

    def _update_tx_plot(self, samp_rate: float):
        if self.backend is None:
            return
        mode = self.tx_plot_combo.currentText()
        if "X_TF" in mode:
            pts = self.backend.get_fdidm_preview_constellation("tf", 4096)
            y = np.abs(pts).astype(np.float64); x = np.arange(y.size, dtype=np.float64)
            self.tx_curve.setData(x, y); self.tx_plot.setTitle("X_TF 幅度")
            self.tx_plot.setXRange(0, max(10, y.size), padding=0)
            self.tx_plot.setYRange(0, max(1.5, float(np.nanmax(y)) * 1.2 if y.size else 1.5), padding=0)
        elif "X星座" in mode:
            pts = self.backend.get_fdidm_preview_constellation("cross", 4096)
            self.tx_curve.setData(np.real(pts), np.imag(pts)); self.tx_plot.setTitle("X 跨域星座")
            self.tx_plot.setXRange(-2, 2, padding=0); self.tx_plot.setYRange(-2, 2, padding=0)
        elif "时域" in mode:
            sig = self.backend.get_tx_spectrum_source(4096)
            y = np.abs(np.asarray(sig, dtype=np.complex64)); x = np.arange(y.size, dtype=np.float64)
            self.tx_curve.setData(x, y); self.tx_plot.setTitle("TX 时域幅度")
            self.tx_plot.setXRange(0, max(10, y.size), padding=0)
            self.tx_plot.setYRange(0, max(0.5, float(np.nanmax(y)) * 1.2 if y.size else 0.5), padding=0)
        else:
            sig = self.backend.get_tx_spectrum_source(4096)
            freq, psd = self._compute_spectrum(sig, samp_rate, 1024)
            self.tx_curve.setData(freq, psd); self.tx_plot.setTitle("TX 基带频谱")
            self.tx_plot.setXRange(-samp_rate / 2, samp_rate / 2, padding=0)
            self.tx_plot.setYRange(-120, 20, padding=0)

    def _update_decode_status(self, stats, status):
        ok = bool(stats.get("decode_ok", False))
        evm = float(status.get("evm_average_percent", np.nan))
        evm_txt = "nan" if not np.isfinite(evm) else f"{evm:.2f}%"
        self.decode_status_label.setText(
            f"{'CRC通过' if ok else '未恢复'} | frames={int(status.get('frames_decode_ok',0))}/{int(status.get('frames_processed',0))}, "
            f"Sync={float(status.get('sync_metric',0.0)):.3f}, CFO={float(status.get('cfo_est_hz',0.0)):.1f}Hz/"
            f"{status.get('cfo_source','')}, raw={float(status.get('cfo_preamble_hz',0.0)):.1f}, "
            f"alias={float(status.get('cfo_alias_hz',np.nan)):.1f}, scan={float(status.get('cfo_scan_score',np.nan)):.2f}, "
            f"BER(FEC)={float(status.get('fec_bit_ber', status.get('ber',np.nan))):.3g}, "
            f"raw={float(status.get('raw_bit_ber',np.nan)):.3g}, EVM={evm_txt}, cond={float(status.get('cond_h_cross',np.nan)):.2e}, "
            f"mode={status.get('channel_estimator','')}, ch={status.get('channel_mode','')}, code={status.get('coding_scheme','')}, "
            f"TDLfit={float(status.get('tdl_param_fit_nmse',np.nan)):.2e}, const={status.get('constellation_source','none')}"
        )

    def _maybe_log_runtime(self, status, stats):
        now = time.monotonic()
        if now - self._last_runtime_log_time < 2.0:
            return
        self._last_runtime_log_time = now
        self._log(
            "v33 runtime: "
            f"reason={status.get('reason','')}, frames={int(status.get('frames_decode_ok',0))}/{int(status.get('frames_processed',0))}, "
            f"rx_new={int(status.get('rx_last_new_samples',0))}, stale={bool(status.get('rx_spectrum_stale',True))}, "
            f"Sync={float(status.get('sync_metric',0.0)):.3f}, CFO={float(status.get('cfo_est_hz',0.0)):.1f}/{status.get('cfo_source','')}, "
            f"alias={float(status.get('cfo_alias_hz',np.nan)):.1f}, scan={float(status.get('cfo_scan_score',np.nan)):.2f}, "
            f"BERfec={float(status.get('fec_bit_ber',status.get('ber',np.nan))):.3g}, raw={float(status.get('raw_bit_ber',np.nan)):.3g}, "
            f"EVM={float(status.get('evm_average_percent',np.nan)):.2f}%, "
            f"mode={status.get('channel_estimator','')}, ch={status.get('channel_mode','')}, fd={float(status.get('tdl_doppler_hz',0.0)):.1f}, "
            f"spread={float(status.get('tdl_doppler_spread_hz',0.0)):.1f}, TDLfit={float(status.get('tdl_param_fit_nmse',np.nan)):.2e}, "
            f"code={status.get('coding_scheme','')}, txvec={int(status.get('tx_waveform_samples',0))}, "
            f"prerender={bool(status.get('tx_tdl_prerendered',False))}, decode_ok={bool(stats.get('decode_ok',False))}"
        )

    def _drain_debug_to_log(self):
        if self.backend is None or not hasattr(self.backend, "drain_debug_log"):
            return
        try:
            entries = self.backend.drain_debug_log(since_seq=int(self._last_debug_seq), max_entries=150, min_level=self._auto_debug_level)
        except Exception:
            return
        for e in entries:
            self._log(f"BE[{e['seq']:04d} {e['t']:7.3f}s {e['level']:<5}] {e['msg']}")
            self._last_debug_seq = max(self._last_debug_seq, int(e.get("seq", 0)))

    # ---------------- utility ----------------
    def _compute_spectrum(self, samples, samp_rate, seg_len=1024):
        samples = np.asarray(samples, dtype=np.complex128).reshape(-1)
        if samples.size == 0:
            return np.array([]), np.array([])
        seg_len = min(int(seg_len), samples.size)
        n_seg = max(1, samples.size // seg_len)
        blocks = samples[-n_seg * seg_len:].reshape(n_seg, seg_len)
        window = np.hanning(seg_len).astype(np.float64)
        psd = np.zeros(seg_len, dtype=np.float64)
        for blk in blocks:
            psd += np.abs(np.fft.fftshift(np.fft.fft(blk * window))) ** 2
        psd /= max(n_seg, 1)
        return np.linspace(-samp_rate / 2, samp_rate / 2, seg_len, endpoint=False), 10.0 * np.log10(psd + 1e-12)

    def _update_evm_plot(self, evm_value):
        try:
            evm = float(evm_value)
        except Exception:
            evm = np.nan
        if np.isfinite(evm) and evm >= 0:
            self._evm_history.append((self._evm_index, evm)); self._evm_index += 1
        if not self._evm_history:
            self.evm_curve.setData([], []); return
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
        self._evm_history.clear(); self._evm_index = 0
        self.evm_curve.setData([], []); self.constellation_scatter.setData(x=[], y=[])
        self.last_status_error = ""; self._last_runtime_log_time = 0.0

    def _clear_plots(self):
        self.tx_curve.setData([], []); self.rx_curve.setData([], []); self.evm_curve.setData([], [])
        self.constellation_scatter.setData(x=[], y=[]); self._evm_history.clear(); self._evm_index = 0

    def _extract_samp_rate(self, status):
        return float(status.get("samp_rate", status.get("sample_rate", self.samp_rate_spin.value())))

    def _set_hw_controls_enabled(self, enabled):
        self.device_combo.setEnabled(enabled); self.samp_rate_spin.setEnabled(enabled); self.fc_spin.setEnabled(enabled)

    def _set_test_controls_enabled(self, enabled):
        self.tx_text_edit.setEnabled(enabled)

    def _backend_summary(self):
        if self.backend is None:
            return "未创建后端"
        st = self.backend.get_status()
        return (
            f"链路={st.get('chain')}, mode={st.get('channel_estimator')}, ch={st.get('channel_mode')}, "
            f"MxN={st.get('fdidm_m')}x{st.get('fdidm_n')}, CP={st.get('cp_len')}, "
            f"训练块={st.get('htf_training_blocks')}, Fs={st.get('samp_rate'):.0f}Hz, "
            f"调制={st.get('mod_order')}, EQ={st.get('equalizer')}, 编码={st.get('coding_summary')}, "
            f"α/β={st.get('alpha'):.2f}/{st.get('beta'):.2f}, frame={st.get('frame_len')} samples, "
            f"TX向量={st.get('tx_waveform_samples')} samples, UHD帧={st.get('usrp_buffer_frames')}, "
            f"CFO无歧义±{float(st.get('cfo_unambiguous_hz', np.nan)):.0f}Hz, "
            f"CFO扫描±{float(st.get('cfo_search_max_hz', np.nan)):.0f}Hz, "
            f"TDL预渲染={st.get('tx_tdl_prerendered')}"
        )

    def _log(self, message):
        from datetime import datetime
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
