# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pyqtgraph as pg
try:
    import pyqtgraph.opengl as gl
    _PG_OPENGL_AVAILABLE = True
except Exception:
    gl = None
    _PG_OPENGL_AVAILABLE = False
from PyQt5.QtCore import Qt, QTimer, QSignalBlocker, pyqtSignal, QEvent, QSize
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QPushButton, QTabWidget,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QTextEdit, QSplitter,
    QScrollArea, QSizePolicy, QCheckBox, QFileDialog, QDialog, QStackedLayout,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from hardware.fdidm_hardtest import FDIDMHardwareTest

MATLAB_BLUE = (0, 114, 189)
MATLAB_ORANGE = (217, 83, 25)
MATLAB_PURPLE = (126, 47, 142)
LIGHT_BG = (250, 250, 250)
AXIS_COLOR = (60, 60, 60)
BORDER_COLOR = (225, 225, 225)


from .fdidm_plot_widgets import _AlphaBetaSurfaceCanvas
from .hardware_advantage_observer import (
    CONTEXT_KEYS,
    AdvantageObservationSession,
    grade_from_k,
)

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
        self._last_adaptive_recommendation_seq = 0
        self.observer = AdvantageObservationSession()
        self._adaptive_toggle_markers = []
        self._timeline_event_items = []
        self._timeline_anomaly_items = []
        self._timeline_apply_points = []
        self._obs_t = deque(maxlen=1800)
        self._obs_ser = deque(maxlen=1800)
        self._obs_evm = deque(maxlen=1800)
        self._obs_alpha = deque(maxlen=1800)
        self._obs_beta = deque(maxlen=1800)
        self._obs_t0 = 0.0
        self._surface_metric_items = [
            ("EVM平均(%)", "evm_average_percent", "lower"),
            ("BER(FEC)", "fec_bit_ber", "lower"),
            ("BER(raw)", "raw_bit_ber", "lower"),
            ("同步度", "sync_metric", "higher"),
            ("cond(H)", "cond_h_cross", "lower"),
            ("噪声方差", "noise_var", "lower"),
            ("H泄漏", "htf_leakage", "lower"),
            ("TDL拟合NMSE", "tdl_param_fit_nmse", "lower"),
            ("CRC成功率", "decode_success_ratio", "higher"),
            ("文本匹配率", "match_ratio", "higher"),
            ("|CFO|", "cfo_abs_hz", "lower"),
        ]
        self._surface_metric_meta = {key: {"label": label, "direction": direction}
                                     for label, key, direction in self._surface_metric_items}
        self._ui_log_entries = deque(maxlen=5000)
        self._ab_surface_window = None
        self._ab_surface_window_canvas = None
        self._ab_surface_window_metric_combo = None
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
        splitter.setSizes([380, 1180])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        try:
            splitter.setCollapsible(1, False)
        except Exception:
            pass

    def _create_controls_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(470)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        hw_group = QGroupBox("链路配置")
        hw = QGridLayout(hw_group)
        hw.setHorizontalSpacing(6)
        hw.setVerticalSpacing(6)
        self.device_combo = self._combo([("B210", "USRP B210"), ("N210", "USRP N210"), ("X310", "USRP X310")])
        self.samp_rate_spin = self._dspin(1e5, 100e6, 500_000, 0, " Hz")
        self.fc_spin = self._dspin(70e6, 6e9, 2.4e9, 0, " Hz")
        hw.addWidget(QLabel("设备"), 0, 0); hw.addWidget(self.device_combo, 0, 1)
        hw.addWidget(QLabel("采样率"), 1, 0); hw.addWidget(self.samp_rate_spin, 1, 1)
        hw.addWidget(QLabel("中心频率"), 2, 0); hw.addWidget(self.fc_spin, 2, 1)
        layout.addWidget(hw_group)

        fd_group = QGroupBox("FDIDM 参数")
        fd = QGridLayout(fd_group)
        fd.setHorizontalSpacing(6)
        fd.setVerticalSpacing(6)
        self.alpha_spin = self._dspin(-2.0, 2.0, 0.5, 2, "", 0.05)
        self.beta_spin = self._dspin(-2.0, 2.0, 1.0, 2, "", 0.05)
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
        fd.addWidget(self.btn_reset_hcache, 15, 0, 1, 2)
        self.btn_clear_ab_surface = QPushButton("清空αβ性能面")
        self.btn_clear_ab_surface.setToolTip("清除已经冻结的 α/β 实测柱体；用于重新做一组可比实验。")
        fd.addWidget(self.btn_clear_ab_surface, 15, 2, 1, 2)
        self.auto_apply_check = QCheckBox("参数改动自动应用")
        self.auto_apply_check.setChecked(False)
        fd.addWidget(self.auto_apply_check, 16, 0, 1, 4)
        note = QLabel("v35：所有模式都经过真实RF；TDL→RF固定离线预渲染。α/β性能面按同一链路上下文记录，测够平均窗口后冻结该点。")
        note.setWordWrap(True)
        fd.addWidget(note, 17, 0, 1, 4)
        layout.addWidget(fd_group)


        adapt_group = QGroupBox("α/β 信道自适应（论文SER）")
        adapt = QGridLayout(adapt_group)
        adapt.setHorizontalSpacing(6)
        adapt.setVerticalSpacing(6)
        self.adaptive_enable_check = QCheckBox("启用信道自适应")
        # The demo starts in fixed mode so the operator can explicitly enable
        # adaptation and observe the before/after change.
        self.adaptive_enable_check.setChecked(False)
        self.adaptive_auto_apply_check = QCheckBox("自动应用稳定推荐")
        self.adaptive_auto_apply_check.setChecked(True)
        self.adaptive_coarse_spin = self._dspin(0.05, 1.0, 0.25, 2, "", 0.05)
        self.adaptive_fine_spin = self._dspin(0.01, 0.50, 0.05, 2, "", 0.01)
        self.adaptive_interval_spin = self._spin(1, 1024, 8)
        self.adaptive_stability_spin = self._spin(1, 16, 2)
        self.adaptive_min_gain_spin = self._dspin(0.0, 30.0, 0.5, 2, " dB", 0.1)
        self.adaptive_cooldown_spin = self._spin(0, 4096, 16)
        self.btn_adaptive_evaluate = QPushButton("立即评估")
        self.btn_adaptive_evaluate.setToolTip("使用最近一次有效 H_TF 立即运行论文SER搜索；尚无CSI时会等待下一有效帧。")
        self.adaptive_status_label = QLabel("自适应：关闭")
        self.adaptive_status_label.setWordWrap(True)
        self.adaptive_status_label.setStyleSheet("color: #444444;")
        adapt.addWidget(self.adaptive_enable_check, 0, 0, 1, 2)
        adapt.addWidget(self.adaptive_auto_apply_check, 0, 2, 1, 2)
        adapt.addWidget(QLabel("粗步长"), 1, 0); adapt.addWidget(self.adaptive_coarse_spin, 1, 1)
        adapt.addWidget(QLabel("细步长"), 1, 2); adapt.addWidget(self.adaptive_fine_spin, 1, 3)
        adapt.addWidget(QLabel("评估间隔/帧"), 2, 0); adapt.addWidget(self.adaptive_interval_spin, 2, 1)
        adapt.addWidget(QLabel("稳定次数"), 2, 2); adapt.addWidget(self.adaptive_stability_spin, 2, 3)
        adapt.addWidget(QLabel("最小预测增益"), 3, 0); adapt.addWidget(self.adaptive_min_gain_spin, 3, 1)
        adapt.addWidget(QLabel("切换冷却/帧"), 3, 2); adapt.addWidget(self.adaptive_cooldown_spin, 3, 3)
        adapt.addWidget(self.btn_adaptive_evaluate, 4, 0, 1, 2)
        adapt.addWidget(self.adaptive_status_label, 4, 2, 2, 2)
        adapt_note = QLabel("基于实测 H_TF 与噪声方差计算论文 Eq.(40)/(44)/(46) 的预测SER，在 [0,2] 内粗到细搜索；同一进程同时更新收发端 α/β。")
        adapt_note.setWordWrap(True)
        adapt.addWidget(adapt_note, 6, 0, 1, 4)
        layout.addWidget(adapt_group)

        modem_group = QGroupBox("收发/显示")
        modem = QGridLayout(modem_group)
        modem.setHorizontalSpacing(6)
        modem.setVerticalSpacing(6)
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
        self.ab_z_metric_combo = self._combo([(label, key) for label, key, _direction in self._surface_metric_items], chars=12)
        self._rx_plot_items = [("RX原始", "raw"), ("整帧", "frame"), ("pilot", "pilot"), ("data", "data")]
        self.rx_plot_combo = self._combo(self._rx_plot_items)
        modem.addWidget(QLabel("TX增益"), 0, 0); modem.addWidget(self.tx_gain_spin, 0, 1)
        modem.addWidget(QLabel("RX增益"), 1, 0); modem.addWidget(self.rx_gain_spin, 1, 1)
        modem.addWidget(QLabel("调制"), 2, 0); modem.addWidget(self.mod_order_combo, 2, 1)
        modem.addWidget(QLabel("均衡"), 3, 0); modem.addWidget(self.equalizer_combo, 3, 1)
        modem.addWidget(QLabel("星座"), 4, 0); modem.addWidget(self.const_mode_combo, 4, 1)
        modem.addWidget(QLabel("3D Z轴"), 5, 0); modem.addWidget(self.ab_z_metric_combo, 5, 1)
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
        self.log_status_label = QLabel("日志文本框已隐藏；需要时点击“导出日志”保存完整日志。")
        self.log_status_label.setWordWrap(True)
        self.log_status_label.setMaximumHeight(44)
        self.log_status_label.setStyleSheet("color: #555555;")
        btn_l.addWidget(self.btn_connect)
        btn_l.addWidget(self.btn_start_test)
        btn_l.addWidget(self.btn_stop_test)
        btn_l.addWidget(self.btn_export_log)
        btn_l.addWidget(self.log_status_label)
        layout.addWidget(btn_group)
        layout.addStretch()
        self._compact_left_controls(panel)
        scroll.setWidget(panel)
        self._apply_control_style(scroll)
        return scroll

    def _create_right_panel(self):
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 右栏四层：①优势观测横条（固定紧凑）②诊断图页签（主伸展，单幅满幅）
        # ③时间轴（左右并排）④文本区。页签替代旧 2×2 网格：每次只呈现一幅诊断图，
        # 幅面约为原四宫格单格的 4 倍，且彻底避免 OpenGL sizeHint 把网格挤变形。
        self.comparison_result_group = QGroupBox("优势观测（手动开关自适应 · 自动配对统计）")
        self.comparison_result_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        result_grid = QGridLayout(self.comparison_result_group)
        result_grid.setVerticalSpacing(2)
        result_grid.setHorizontalSpacing(12)
        self.btn_start_observation = QPushButton("开始观测")
        self.btn_stop_observation = QPushButton("结束观测")
        self.btn_export_observation = QPushButton("导出报告")
        for btn in (self.btn_start_observation, self.btn_stop_observation, self.btn_export_observation):
            btn.setMinimumWidth(96)
            btn.setMinimumHeight(28)
        self.btn_stop_observation.setEnabled(False)
        self.btn_export_observation.setEnabled(False)
        self.observation_state_label = QLabel("观测：未开始 | 开始观测后手动开关自适应即可积累前后窗口")
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)
        ctrl_row.addWidget(self.btn_start_observation)
        ctrl_row.addWidget(self.btn_stop_observation)
        ctrl_row.addWidget(self.btn_export_observation)
        ctrl_row.addWidget(self.observation_state_label, 1)
        result_grid.addLayout(ctrl_row, 0, 0, 1, 3)

        big_style = "font-size: 18px; font-weight: 700;"
        self.observation_before_label = QLabel("开启前：—")
        self.observation_after_label = QLabel("开启后：—")
        self.observation_improvement_label = QLabel("SER 改善：—")
        self.observation_before_label.setStyleSheet(big_style + " color:#333333;")
        self.observation_after_label.setStyleSheet(big_style + " color:#333333;")
        self.observation_improvement_label.setStyleSheet(big_style + " color:#167c3a;")
        # 单行省略：宽度不足时裁切而不是换行把横条撑高
        for lbl in (self.observation_before_label, self.observation_after_label,
                    self.observation_improvement_label):
            lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        result_grid.addWidget(self.observation_before_label, 1, 0)
        result_grid.addWidget(self.observation_after_label, 1, 1)
        result_grid.addWidget(self.observation_improvement_label, 1, 2)

        self.observation_badge_label = QLabel("可信度：—")
        self.observation_note_label = QLabel("")
        self.observation_note_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.adaptive_state_label = QLabel("α/β：— | 搜索：— | 验证：— | α可观测：— | β可观测：—")
        self.adaptive_state_label.setStyleSheet("color:#555555; font-size: 12px;")
        self.adaptive_state_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        result_grid.addWidget(self.observation_badge_label, 2, 0)
        result_grid.addWidget(self.observation_note_label, 2, 1, 1, 2)
        result_grid.addWidget(self.adaptive_state_label, 3, 0, 1, 3)
        layout.addWidget(self.comparison_result_group, 0)

        self.ab_surface_panel = self._create_ab_surface_panel()
        self.rx_spectrum_plot = pg.PlotWidget(title="RX 频谱")
        self.evm_plot = pg.PlotWidget(title="EVM 曲线")
        self.constellation_plot = pg.PlotWidget(title="接收星座")
        for w in (self.ab_surface_panel, self.rx_spectrum_plot, self.evm_plot, self.constellation_plot):
            w.setMinimumSize(0, 0)
            w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

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

        self.plot_tabs = QTabWidget()
        self.plot_tabs.addTab(self.ab_surface_panel, "α/β 性能面")
        self.plot_tabs.addTab(self.rx_spectrum_plot, "RX 频谱")
        self.plot_tabs.addTab(self.evm_plot, "EVM 曲线")
        self.plot_tabs.addTab(self.constellation_plot, "接收星座")
        self.plot_tabs.setCurrentIndex(0)
        layout.addWidget(self.plot_tabs, 1)

        # 因果时间轴：左图实测SER(log10)与EVM%信道代理，右图α/β轨迹；共享 x 轴联动。
        timeline_panel = QWidget()
        timeline_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        timeline_panel.setMinimumHeight(150)
        timeline_panel.setMaximumHeight(200)
        tl_layout = QHBoxLayout(timeline_panel)
        tl_layout.setContentsMargins(0, 0, 0, 0)
        tl_layout.setSpacing(8)
        self.timeline_perf_plot = pg.PlotWidget(title="观测时间轴：SER(log10) / EVM%（信道代理）")
        self.timeline_ab_plot = pg.PlotWidget(title="α/β 轨迹（▲=参数应用）")
        self.timeline_ab_plot.setXLink(self.timeline_perf_plot)
        self.timeline_perf_plot.showGrid(x=True, y=True)
        self.timeline_ab_plot.showGrid(x=True, y=True)
        self.timeline_perf_plot.setLabel("left", "log10(SER) / EVM %")
        self.timeline_ab_plot.setLabel("left", "α / β")
        self.timeline_perf_plot.setLabel("bottom", "观测时间 (s)")
        for w in (self.timeline_perf_plot, self.timeline_ab_plot):
            w.setMinimumSize(0, 0)
        tl_layout.addWidget(self.timeline_perf_plot, 3)
        tl_layout.addWidget(self.timeline_ab_plot, 2)
        layout.addWidget(timeline_panel, 0)

        text_panel = QWidget()
        text_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        text_panel.setMaximumHeight(120)
        text_grid = QGridLayout(text_panel)
        text_grid.setContentsMargins(0, 0, 0, 0)
        text_grid.setHorizontalSpacing(8)
        text_grid.setVerticalSpacing(2)
        self.decode_status_label = QLabel("解调状态：未开始")
        self.decode_status_label.setMinimumHeight(22)
        self.decode_status_label.setMaximumHeight(24)
        self.decode_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.tx_text_view = QTextEdit(); self.tx_text_view.setReadOnly(True); self.tx_text_view.setMaximumHeight(64)
        self.rx_text_view = QTextEdit(); self.rx_text_view.setReadOnly(True); self.rx_text_view.setMaximumHeight(64)
        text_grid.addWidget(self.decode_status_label, 0, 0, 1, 2)
        text_grid.addWidget(QLabel("发送文本"), 1, 0)
        text_grid.addWidget(QLabel("接收文本"), 1, 1)
        text_grid.addWidget(self.tx_text_view, 2, 0)
        text_grid.addWidget(self.rx_text_view, 2, 1)
        layout.addWidget(text_panel, 0)

        self.rx_curve = self.rx_spectrum_plot.plot(pen=pg.mkPen(MATLAB_ORANGE, width=2))
        self.evm_curve = self.evm_plot.plot(pen=pg.mkPen(MATLAB_PURPLE, width=2), name="current EVM")
        self.timeline_ser_curve = self.timeline_perf_plot.plot(
            pen=pg.mkPen(MATLAB_BLUE, width=2), name="log10(SER)", connect="finite")
        self.timeline_evm_curve = self.timeline_perf_plot.plot(
            pen=pg.mkPen(MATLAB_ORANGE, width=2), name="EVM%", connect="finite")
        self.timeline_alpha_curve = self.timeline_ab_plot.plot(
            pen=pg.mkPen(MATLAB_PURPLE, width=2), name="α", connect="finite")
        self.timeline_beta_curve = self.timeline_ab_plot.plot(
            pen=pg.mkPen(MATLAB_PURPLE, width=1, style=Qt.DashLine), name="β", connect="finite")
        self.timeline_apply_scatter = pg.ScatterPlotItem(
            size=12, symbol="t", pen=pg.mkPen(None), brush=pg.mkBrush(237, 177, 32, 220))
        self.timeline_ab_plot.addItem(self.timeline_apply_scatter)
        self.constellation_scatter = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(237, 177, 32, 160))
        self.constellation_plot.addItem(self.constellation_scatter)
        return panel

    def _create_ab_surface_panel(self):
        self.ab_surface_canvas = _AlphaBetaSurfaceCanvas(show_open_button=True, large=False, parent=self)
        self.ab_surface_canvas.openRequested.connect(self._open_ab_surface_window)
        self.ab_surface_canvas.viewModeChanged.connect(self._on_embedded_ab_view_changed)
        return self.ab_surface_canvas

    def _on_embedded_ab_view_changed(self, mode: str):
        canvas = getattr(self, "_ab_surface_window_canvas", None)
        if canvas is not None:
            try:
                canvas.set_view_mode(str(mode))
            except Exception:
                pass

    def _metric_combo_index_for_key(self, combo: QComboBox, key: str) -> int:
        for i in range(combo.count()):
            if str(combo.itemData(i)) == str(key):
                return i
        return -1

    def _sync_combo_to_key(self, combo: QComboBox, key: str):
        idx = self._metric_combo_index_for_key(combo, key)
        if idx >= 0 and combo.currentIndex() != idx:
            blocker = QSignalBlocker(combo)
            try:
                combo.setCurrentIndex(idx)
            finally:
                del blocker

    def _on_popup_metric_changed(self):
        popup_combo = getattr(self, "_ab_surface_window_metric_combo", None)
        if popup_combo is None:
            return
        key = str(popup_combo.currentData() or "evm_average_percent")
        idx = self._metric_combo_index_for_key(self.ab_z_metric_combo, key)
        if idx >= 0 and self.ab_z_metric_combo.currentIndex() != idx:
            self.ab_z_metric_combo.setCurrentIndex(idx)
        else:
            self._refresh_ab_surface_only()

    def _clear_ab_surface_window_refs(self, *_args):
        self._ab_surface_window = None
        self._ab_surface_window_canvas = None
        self._ab_surface_window_metric_combo = None

    def _open_ab_surface_window(self):
        if self._ab_surface_window is not None:
            try:
                self._ab_surface_window.raise_()
                self._ab_surface_window.activateWindow()
                self._refresh_ab_surface_only()
                return
            except Exception:
                self._clear_ab_surface_window_refs()

        win = QDialog(self)
        win.setWindowTitle("α-β 性能三维独立展示")
        win.setWindowModality(Qt.NonModal)
        win.setAttribute(Qt.WA_DeleteOnClose, True)
        win.resize(1080, 820)
        layout = QVBoxLayout(win)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top = QHBoxLayout()
        top.addWidget(QLabel("Z轴指标"))
        metric_combo = QComboBox()
        for label, key, _direction in self._surface_metric_items:
            metric_combo.addItem(label, key)
        metric_combo.setMinimumContentsLength(14)
        metric_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._sync_combo_to_key(metric_combo, self._selected_ab_surface_metric())
        metric_combo.currentIndexChanged.connect(lambda *_: self._on_popup_metric_changed())
        top.addWidget(metric_combo)
        hint = QLabel("三维视图：滚轮缩放、左键拖拽旋转、右键拖拽平移，刷新时不会重置视角；XZ/YZ/XY 使用二维投影视图，便于按 MATLAB 平面读数。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555555;")
        top.addWidget(hint, 1)
        layout.addLayout(top)

        canvas = _AlphaBetaSurfaceCanvas(show_open_button=False, large=True, parent=win)
        try:
            canvas.set_view_mode(getattr(self, "ab_surface_canvas", canvas).view_mode())
        except Exception:
            pass
        canvas.viewModeChanged.connect(lambda mode: getattr(self, "ab_surface_canvas", canvas).set_view_mode(str(mode)))
        layout.addWidget(canvas, 1)

        win.destroyed.connect(self._clear_ab_surface_window_refs)
        self._ab_surface_window = win
        self._ab_surface_window_canvas = canvas
        self._ab_surface_window_metric_combo = metric_combo
        self._refresh_ab_surface_only()
        win.show()
        win.raise_()
        win.activateWindow()

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
        combo.setMaximumWidth(260)
        return combo

    def _compact_left_controls(self, panel):
        for combo in panel.findChildren(QComboBox):
            self._compact_combo(combo, 8)
        for edit in panel.findChildren((QSpinBox, QDoubleSpinBox)):
            edit.setMinimumWidth(80)
            edit.setMaximumWidth(140)

    def _dspin(self, lo, hi, val, dec, suffix="", step=None):
        s = QDoubleSpinBox()
        s.setRange(float(lo), float(hi))
        s.setValue(float(val))
        s.setDecimals(int(dec))
        s.setSuffix(str(suffix))
        if step is not None:
            s.setSingleStep(float(step))
        s.setMinimumHeight(26)
        s.setMaximumWidth(140)
        return s

    def _spin(self, lo, hi, val):
        s = QSpinBox()
        s.setRange(int(lo), int(hi))
        s.setValue(int(val))
        s.setMinimumHeight(26)
        s.setMaximumWidth(140)
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
        plot_widgets = [self.rx_spectrum_plot, self.evm_plot, self.constellation_plot,
                        self.timeline_perf_plot, self.timeline_ab_plot]
        if getattr(self, "ab_surface_fallback_plot", None) is not None:
            plot_widgets.insert(0, self.ab_surface_fallback_plot)
        for p in plot_widgets:
            p.setBackground(LIGHT_BG)
            p.showGrid(x=True, y=True, alpha=0.35)
            for axis_name in ("left", "bottom"):
                axis = p.getAxis(axis_name)
                axis.setPen(pg.mkPen(AXIS_COLOR, width=1.0))
                axis.setTextPen(pg.mkPen(AXIS_COLOR, width=1.0))
            p.getPlotItem().getViewBox().setBorder(pg.mkPen(BORDER_COLOR, width=1.0))
        for p in (self.rx_spectrum_plot, self.evm_plot):
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
        self.btn_start_observation.clicked.connect(self._start_observation_clicked)
        self.btn_stop_observation.clicked.connect(self._stop_observation_clicked)
        self.btn_export_observation.clicked.connect(self._export_observation_clicked)
        self.btn_export_log.clicked.connect(self._on_export_log_clicked)
        self.btn_apply_params.clicked.connect(self._apply_params_to_backend)
        self.btn_ofdm.clicked.connect(lambda: self._set_indices(0.0, 0.0))
        self.btn_otfs.clicked.connect(lambda: self._set_indices(1.0, 1.0))
        self.btn_reco.clicked.connect(lambda: self._set_indices(0.5, 1.0))
        self.btn_reset_hcache.clicked.connect(self._on_reset_hcache_clicked)
        self.btn_clear_ab_surface.clicked.connect(self._on_clear_ab_surface_clicked)
        self.btn_adaptive_evaluate.clicked.connect(self._on_adaptive_evaluate_clicked)
        self.adaptive_enable_check.stateChanged.connect(self._on_adaptive_enable_changed)
        self.adaptive_auto_apply_check.stateChanged.connect(lambda _: self._handle_alpha_beta_adaptation(self.backend.get_status()) if self.backend is not None else None)
        for w in (self.adaptive_coarse_spin, self.adaptive_fine_spin, self.adaptive_interval_spin,
                  self.adaptive_stability_spin, self.adaptive_min_gain_spin, self.adaptive_cooldown_spin):
            w.valueChanged.connect(self._on_adaptive_config_changed)
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
        self.ab_z_metric_combo.currentIndexChanged.connect(lambda _: self._refresh_ab_surface_only())
        self.rx_plot_combo.currentIndexChanged.connect(lambda _: self._refresh_plots())

    # ---------------- button handlers ----------------
    def _on_connect_clicked(self):
        try:
            self._create_backend()
            self.tx_text_view.setPlainText(self.tx_text_edit.toPlainText())
            self.rx_text_view.clear()
            self._push_const_mode()
            self._log("FDIDM 后端已配置 v35（信道自适应α/β）。")
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
            self._log("v35 测试已启动。")
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
        if self.observer.active:
            self.observer.stop()
            self.btn_start_observation.setEnabled(True)
            self.btn_stop_observation.setEnabled(False)
            self.btn_export_observation.setEnabled(self.observer.windows() or self.observer.events())
            self._update_observation_display()
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

    def _on_clear_ab_surface_clicked(self):
        if self.backend is None:
            self._clear_ab_surface_plot()
            self._log("尚未创建后端；界面性能面已清空。")
            return
        try:
            if hasattr(self.backend, "clear_alpha_beta_performance_surface"):
                self.backend.clear_alpha_beta_performance_surface(reason="ui_manual_clear")
            self._clear_ab_surface_plot()
            self._log("已清空 α/β 性能面；后续 α/β 点将重新测量并冻结。")
            self._drain_debug_to_log()
        except Exception as e:
            self._log(f"清空 α/β 性能面失败: {type(e).__name__}: {e}")

    def _on_adaptive_evaluate_clicked(self):
        if self.backend is None:
            self._log("尚未创建后端，无法评估 α/β。")
            return
        try:
            queued = bool(self.backend.request_alpha_beta_adaptation()) if hasattr(self.backend, "request_alpha_beta_adaptation") else False
            self._log("已提交 α/β 立即评估。" if queued else "尚无有效CSI；将在下一有效帧评估 α/β。")
        except Exception as e:
            self._log(f"α/β 立即评估失败: {type(e).__name__}: {e}")

    def _on_export_log_clicked(self):
        log_dir = Path(__file__).resolve().parents[2] / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        default_name = str(log_dir / f"fdidm_debug_{time.strftime('%Y%m%d_%H%M%S')}.log")
        path, _ = QFileDialog.getSaveFileName(self, "导出 FDIDM 日志", default_name, "Log Files (*.log);;Text Files (*.txt);;All Files (*)")
        if not path:
            return
        try:
            ui_entries = list(getattr(self, "_ui_log_entries", []))
            if self.backend is not None and hasattr(self.backend, "export_debug_log"):
                saved = self.backend.export_debug_log(path, max_entries=5000, min_level="DEBUG")
                if ui_entries:
                    with open(saved, "a", encoding="utf-8") as f:
                        f.write("\n--- UI log, hidden from right panel ---\n")
                        for line in ui_entries:
                            f.write(str(line).rstrip("\n") + "\n")
                self._log(f"后端日志已导出：{saved}")
            else:
                with open(path, "w", encoding="utf-8") as f:
                    for line in ui_entries:
                        f.write(str(line).rstrip("\n") + "\n")
                self._log(f"界面日志已导出：{path}")
        except Exception as e:
            self._log(f"导出日志失败: {type(e).__name__}: {e}")

    # ---------------- 优势观测（手动开关 + 自动配对统计） ----------------
    def _observation_context(self):
        ctx = {}
        if self.backend is not None:
            try:
                st = self.backend.get_status()
                for key in CONTEXT_KEYS:
                    if key in st:
                        ctx[key] = st[key]
            except Exception:
                pass
        return ctx

    def _start_observation_clicked(self):
        if self.backend is None or not self.test_running:
            self._log("请先连接并启动测试，再开始优势观测。")
            return
        enabled = bool(self.adaptive_enable_check.isChecked())
        self.observer.start(enabled, self._observation_context())
        self._obs_t0 = self.observer.started_at()
        self._clear_timeline()
        self.btn_start_observation.setEnabled(False)
        self.btn_stop_observation.setEnabled(True)
        self.btn_export_observation.setEnabled(False)
        self._log(f"开始优势观测（自适应初始：{'开启' if enabled else '关闭'}）；开关自适应时自动记录前后窗口。")
        self._update_observation_display()

    def _stop_observation_clicked(self):
        self.observer.stop()
        self.btn_start_observation.setEnabled(True)
        self.btn_stop_observation.setEnabled(False)
        self.btn_export_observation.setEnabled(True)
        self._log("结束优势观测；结果保留，可导出报告。")
        self._update_observation_display()

    def _export_observation_clicked(self):
        data = self.observer.to_export_dict()
        if not data.get("windows") and not data.get("toggle_events"):
            self._log("尚无观测数据可导出；请先开始观测并开关自适应。")
            return
        log_dir = Path(__file__).resolve().parents[2] / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        default_name = str(log_dir / f"fdidm_advantage_{time.strftime('%Y%m%d_%H%M%S')}.json")
        path, _ = QFileDialog.getSaveFileName(self, "导出优势观测报告", default_name, "JSON (*.json);;All Files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=float)
            self._log(f"优势观测报告已导出：{path}")
        except Exception as e:
            self._log(f"导出观测报告失败: {type(e).__name__}: {e}")

    def _clear_timeline(self):
        self._obs_t.clear(); self._obs_ser.clear(); self._obs_evm.clear()
        self._obs_alpha.clear(); self._obs_beta.clear()
        self._timeline_apply_points = []
        self.timeline_apply_scatter.setData(x=[], y=[])
        for item in self._timeline_event_items + self._timeline_anomaly_items:
            try:
                self.timeline_perf_plot.removeItem(item)
            except Exception:
                pass
        self._timeline_event_items = []
        self._timeline_anomaly_items = []
        self._refresh_timeline_plot()

    def _append_timeline_sample(self, status):
        # 时间轴与观测会话共用引擎时钟，保证事件线/异常区与曲线同一时基
        now = self.observer.now()
        ser = float(status.get("measured_ser", np.nan))
        evm = float(status.get("evm_average_percent", status.get("evm_percent", np.nan)))
        self._obs_t.append(now - self._obs_t0)
        self._obs_ser.append(math.log10(ser) if np.isfinite(ser) and ser > 0 else np.nan)
        self._obs_evm.append(evm)
        self._obs_alpha.append(float(status.get("alpha", np.nan)))
        self._obs_beta.append(float(status.get("beta", np.nan)))
        self._refresh_timeline_plot()
        self._refresh_timeline_anomalies()

    def _refresh_timeline_plot(self):
        ts = list(self._obs_t)
        self.timeline_ser_curve.setData(ts, list(self._obs_ser))
        self.timeline_evm_curve.setData(ts, list(self._obs_evm))
        self.timeline_alpha_curve.setData(ts, list(self._obs_alpha))
        self.timeline_beta_curve.setData(ts, list(self._obs_beta))

    def _refresh_timeline_anomalies(self):
        drops = self.observer.anomaly_drops()
        if len(drops) == getattr(self, "_timeline_anomaly_count", -1):
            return
        self._timeline_anomaly_count = len(drops)
        for item in self._timeline_anomaly_items:
            try:
                self.timeline_perf_plot.removeItem(item)
            except Exception:
                pass
        self._timeline_anomaly_items = []
        t0 = getattr(self, "_obs_t0", 0.0)
        spans = []
        for t, _kind in drops:
            rel = t - t0
            if spans and rel - spans[-1][1] <= 0.35:
                spans[-1][1] = rel
            else:
                spans.append([rel, rel])
        for lo, hi in spans:
            region = pg.LinearRegionItem([lo - 0.15, hi + 0.15], movable=False,
                                         brush=pg.mkBrush(255, 0, 0, 35))
            region.setZValue(-10)
            self.timeline_perf_plot.addItem(region)
            self._timeline_anomaly_items.append(region)

    def _add_timeline_event_line(self, enabled, t_abs):
        if not self._obs_t:
            return
        line = pg.InfiniteLine(pos=t_abs - self._obs_t0, angle=90,
                               pen=pg.mkPen(MATLAB_BLUE if enabled else MATLAB_ORANGE,
                                            style=Qt.DashLine, width=2))
        self.timeline_perf_plot.addItem(line)
        self._timeline_event_items.append(line)

    def _mark_observation_apply(self, alpha, beta):
        """α/β 推荐被应用时在时间轴 α/β 图上打标记。"""
        if not self.observer.active or not self._obs_t or not np.isfinite(alpha):
            return
        self._timeline_apply_points.append((self.observer.now() - self._obs_t0, float(alpha)))
        xs = [p[0] for p in self._timeline_apply_points]
        ys = [p[1] for p in self._timeline_apply_points]
        self.timeline_apply_scatter.setData(x=xs, y=ys)

    def _update_adaptive_state_row(self, status):
        a = float(status.get("alpha", np.nan))
        b = float(status.get("beta", np.nan))
        search = str(status.get("adaptive_alpha_beta_state", "—"))
        valid = str(status.get("adaptive_validation_state", "—"))
        alpha_obs = status.get("adaptive_alpha_observable")
        beta_obs = status.get("adaptive_beta_observable")
        if alpha_obs is None or beta_obs is None:
            try:
                ab = self.backend.get_alpha_beta_adaptation_status()
                alpha_obs = ab.get("alpha_observable")
                beta_obs = ab.get("beta_observable")
            except Exception:
                pass
        def yn(v):
            return "—" if v is None else ("是" if bool(v) else "否")
        txt = f"α/β={a:.2f}/{b:.2f} | 搜索:{search} | 验证:{valid} | α可观测:{yn(alpha_obs)} | β可观测:{yn(beta_obs)}"
        reason = str(status.get("adaptive_validation_reason", "") or "")
        if reason:
            txt += f" | {reason[:60]}"
        self.adaptive_state_label.setText(txt)

    def _update_observation_display(self):
        started = bool(self.observer.windows() or self.observer.events())
        state_txt = "进行中" if self.observer.active else ("已结束" if started else "未开始")
        self.observation_state_label.setText(
            f"观测：{state_txt} | 窗口 {len(self.observer.windows())} | 开关事件 {len(self.observer.events())}"
            f" | 剔除 {self.observer.excluded_window_count()} 个异常窗口")
        pair = self.observer.latest_pair()
        if pair is None:
            self.observation_before_label.setText("开启前：—")
            self.observation_after_label.setText("开启后：—")
            self.observation_improvement_label.setText("SER 改善：—")
            self.observation_improvement_label.setStyleSheet("font-size: 15px; font-weight: 600; color:#167c3a;")
            self.observation_badge_label.setText("可信度：—")
            self.observation_badge_label.setStyleSheet("")
            self.observation_note_label.setText("开始观测后，手动关闭/开启自适应各积累一段窗口即可得到结论")
            return
        b, a = pair.before, pair.after
        def fmt(v, suffix=""):
            v = float(v)
            return "—" if not np.isfinite(v) else f"{v:.3g}{suffix}"
        self.observation_before_label.setText(f"开启前：SER {fmt(b.ser)} | EVM {fmt(b.evm_mean, '%')}")
        self.observation_after_label.setText(f"开启后：SER {fmt(a.ser)} | EVM {fmt(a.evm_mean, '%')}")
        # 样本不足的窗口不产生改善结论（F3/AC3）
        insufficient = a.grade == "insufficient" or b.grade == "insufficient"
        if insufficient:
            self.observation_improvement_label.setText("SER 改善：样本不足")
            self.observation_improvement_label.setStyleSheet("font-size: 15px; font-weight: 600; color:#b8860b;")
        elif np.isfinite(pair.ser_improvement_db):
            color = "#167c3a" if pair.ser_improvement_db >= 0 else "#a33b3b"
            self.observation_improvement_label.setText(f"SER 改善：{pair.ser_improvement_db:+.2f} dB")
            self.observation_improvement_label.setStyleSheet(f"font-size: 15px; font-weight: 600; color:{color};")
        else:
            self.observation_improvement_label.setText("SER 改善：—")
        badge_map = {"trusted": ("可信", "#167c3a"), "reference": ("参考", "#777777"),
                     "insufficient": ("样本不足", "#b8860b")}
        name, color = badge_map.get(a.grade, ("—", "#333333"))
        k_mark = "≈" if a.estimated_k else ""
        self.observation_badge_label.setText(
            f"开启后可信度：{name}（{k_mark}错误符号 {a.ser_k:.0f}/{a.ser_n}，{a.frames} 帧）")
        self.observation_badge_label.setStyleSheet(f"color:{color}; font-weight:600;")
        notes = []
        if insufficient:
            notes.append("每段需累计 ≥10 个错误符号才形成结论，≥100 为可信（继续运行积累）")
        if self.observer.excluded_window_count():
            notes.append(f"已剔除异常窗口 {self.observer.excluded_window_count()} 个")
        if not pair.comparable:
            notes.append(f"{pair.comparability_note}（可比性受限，结论仅供参考）")
        self.observation_note_label.setText("；".join(notes))

    def _on_adaptive_enable_changed(self, state):
        enabled = bool(state)
        if self.backend is not None and self.test_running:
            event = self.observer.on_toggle(enabled, self.alpha_spin.value(), self.beta_spin.value())
            if event is not None:
                self._add_timeline_event_line(enabled, event.t)
                try:
                    marker = pg.InfiniteLine(pos=max(0, self._evm_index), angle=90,
                                             pen=pg.mkPen(MATLAB_BLUE if enabled else MATLAB_ORANGE, style=Qt.DashLine, width=1))
                    self.evm_plot.addItem(marker)
                    self._adaptive_toggle_markers.append(marker)
                except Exception:
                    pass
            self._log(f"手动{'开启' if enabled else '关闭'} α/β 自适应，观测窗口已切换")
        self._on_adaptive_config_changed()


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
            adaptive_alpha_beta_enable=self.adaptive_enable_check.isChecked(),
            adaptive_alpha_beta_coarse_step=self.adaptive_coarse_spin.value(),
            adaptive_alpha_beta_fine_step=self.adaptive_fine_spin.value(),
            adaptive_alpha_beta_interval_frames=self.adaptive_interval_spin.value(),
            adaptive_alpha_beta_min_improvement_db=self.adaptive_min_gain_spin.value(),
            adaptive_alpha_beta_stability_evals=self.adaptive_stability_spin.value(),
            adaptive_alpha_beta_cooldown_frames=self.adaptive_cooldown_spin.value(),
            adaptive_alpha_beta_integer_margin_db=0.10,
            adaptive_alpha_beta_max_order=512,
            adaptive_alpha_beta_min_sync_metric=0.30,
            adaptive_alpha_beta_require_good_frame=False,
            adaptive_alpha_beta_rcond=1e-6,
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
        live_applied = False
        restarted = False
        try:
            self._apply_debounce_timer.stop()
        except Exception:
            pass
        try:
            if was_running:
                try:
                    # Preferred path: backend distinguishes live-safe changes
                    # from structural changes.  This avoids the old stop -> UHD
                    # rebuild -> start cycle for SNR, alpha/beta, text, coding,
                    # modulation, and TDL pre-rendered waveform updates.
                    self._configure_backend(self.tx_text_edit.toPlainText())
                    live_applied = True
                except RuntimeError as e:
                    msg = str(e)
                    if "stop first" not in msg and "Cannot reconfigure" not in msg:
                        raise
                    self._log("参数涉及帧结构/UHD图结构，执行一次受控停止-重启。")
                    self.test_running = False
                    self.update_timer.stop()
                    self.backend.stop()
                    if hasattr(self.backend, "wait"):
                        self.backend.wait()
                    self._configure_backend(self.tx_text_edit.toPlainText())
                    self.backend.start()
                    self.test_running = True
                    self.update_timer.start(100)
                    restarted = True
            else:
                self._configure_backend(self.tx_text_edit.toPlainText())
            self.tx_text_view.setPlainText(self.tx_text_edit.toPlainText())
            if not self._adaptive_toggle_pending:
                self._reset_runtime_curves()
            self._adaptive_toggle_pending = False
            self._refresh_tx_plot_only()
            if live_applied:
                self._log("FDIDM 参数已热更新，未重启 UHD。")
            elif restarted:
                self._log("FDIDM 参数已通过受控重启应用。")
            else:
                self._log("FDIDM 参数已应用。")
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

    def _on_adaptive_config_changed(self, *_args):
        # Keep the fine grid no coarser than the coarse grid.  These settings
        # are control-plane only and are safe to update while UHD is running.
        try:
            self.adaptive_fine_spin.setMaximum(max(0.01, self.adaptive_coarse_spin.value()))
        except Exception:
            pass
        if self.backend is not None and not self._suppress_param_signals:
            self._schedule_param_apply(0)

    def _on_params_changed(self, *_args):
        if self._suppress_param_signals:
            return
        if self.auto_apply_check.isChecked():
            self._schedule_param_apply(250)

    def _on_mod_or_eq_changed(self, *_args):
        if self.backend is not None and not self._suppress_param_signals and self.auto_apply_check.isChecked():
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
        if self.backend is not None and self.auto_apply_check.isChecked():
            self._schedule_param_apply(250)
        elif self.backend is not None:
            self._log("链路模式已选择；点击“应用参数”后生效。")

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
            self.observer.on_sample(status)
            if self.observer.active:
                self._append_timeline_sample(status)
                self._update_adaptive_state_row(status)
            self._update_observation_display()
            samp_rate = self._extract_samp_rate(status)
            self._apply_stable_plot_ranges(samp_rate)
            err = status.get("last_error", "")
            if err and err != self.last_status_error:
                self.last_status_error = err; self._log(f"后端异常: {err}")
            self._handle_alpha_beta_adaptation(status)
            self._update_ab_surface_plot(status)
            rx_signal = self.backend.get_rx_spectrum_source(4096, source=self._selected_rx_spectrum_source())
            rx_freq, rx_psd = self._compute_spectrum(rx_signal, samp_rate, 1024)
            self.rx_curve.setData(rx_freq, rx_psd)
            stale = bool(status.get("rx_spectrum_stale", True)); age = float(status.get("rx_spectrum_stale_sec", np.nan))
            self.rx_spectrum_plot.setTitle(f"RX频谱[{self._selected_rx_spectrum_source()}] stale={stale} age={age:.1f}s")
            self._update_evm_plot(float(status.get("evm_percent", np.nan)))
            self.evm_plot.setTitle("EVM 曲线（虚线标记自适应切换）" if self.observer.active else "EVM 曲线")
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

    def _refresh_ab_surface_only(self):
        if self.backend is None:
            return
        try:
            self._update_ab_surface_plot(self.backend.get_status())
        except Exception as e:
            self._log(f"αβ性能面刷新失败: {type(e).__name__}: {e}")

    # Backward-compatible alias: old code paths that refreshed the TX plot now
    # refresh the alpha/beta performance surface.
    def _refresh_tx_plot_only(self):
        self._refresh_ab_surface_only()

    def _selected_ab_surface_metric(self) -> str:
        return self._current_data(getattr(self, "ab_z_metric_combo", None), "evm_average_percent")

    def _surface_metric_label(self, metric: str) -> str:
        return self._surface_metric_meta.get(str(metric), {}).get("label", str(metric))

    def _surface_metric_direction(self, metric: str) -> str:
        return self._surface_metric_meta.get(str(metric), {}).get("direction", "lower")

    @staticmethod
    def _rgba(rgb, alpha=1.0):
        return (float(rgb[0]) / 255.0, float(rgb[1]) / 255.0, float(rgb[2]) / 255.0, float(alpha))

    def _format_metric_value(self, metric: str, value) -> str:
        try:
            v = float(value)
        except Exception:
            return "nan"
        if not np.isfinite(v):
            return "nan"
        metric = str(metric)
        if metric in ("evm_instant_percent", "evm_average_percent"):
            return f"{v:.2f}%"
        if metric in ("decode_success_ratio", "match_ratio"):
            return f"{100.0 * v:.1f}%"
        if metric in ("ber", "fec_bit_ber", "raw_bit_ber", "noise_var", "tdl_param_fit_nmse", "htf_leakage"):
            return f"{v:.2e}"
        if metric == "cond_h_cross":
            return f"{v:.2e}"
        if metric == "cfo_abs_hz":
            return f"{v:.1f} Hz"
        return f"{v:.3f}"

    def _fetch_alpha_beta_surface(self, metric: str, status=None):
        if self.backend is None:
            return {"metric": metric, "points": [], "point_count": 0}
        if hasattr(self.backend, "get_alpha_beta_performance_surface"):
            return self.backend.get_alpha_beta_performance_surface(metric)
        # Compatibility fallback for older backends: expose only the current
        # point, using the status dictionary already displayed elsewhere.
        st = status or self.backend.get_status()
        metrics = {
            "evm_instant_percent": float(st.get("evm_instant_percent", np.nan)),
            "evm_average_percent": float(st.get("evm_average_percent", np.nan)),
            "ber": float(st.get("ber", np.nan)),
            "fec_bit_ber": float(st.get("fec_bit_ber", st.get("ber", np.nan))),
            "raw_bit_ber": float(st.get("raw_bit_ber", np.nan)),
            "sync_metric": float(st.get("sync_metric", np.nan)),
            "cond_h_cross": float(st.get("cond_h_cross", np.nan)),
            "noise_var": float(st.get("noise_var", np.nan)),
            "htf_leakage": float(st.get("htf_leakage", np.nan)),
            "tdl_param_fit_nmse": float(st.get("tdl_param_fit_nmse", np.nan)),
            "decode_success_ratio": 1.0 if bool(st.get("decode_ok", False)) else 0.0,
            "match_ratio": float(st.get("match_ratio", np.nan)),
            "cfo_abs_hz": abs(float(st.get("cfo_est_hz", 0.0))),
        }
        z = metrics.get(metric, np.nan)
        point = {
            "alpha": float(st.get("alpha", 0.0)),
            "beta": float(st.get("beta", 0.0)),
            "z": z,
            "metric": metric,
            "metrics": metrics,
            "sample_count": 1,
        }
        return {"metric": metric, "points": [point], "point_count": 1}

    @staticmethod
    def _valid_surface_points(surface, metric: str):
        out = []
        for p in list((surface or {}).get("points", [])):
            metrics = p.get("metrics", {}) if isinstance(p, dict) else {}
            z = p.get("z", metrics.get(metric, np.nan)) if isinstance(p, dict) else np.nan
            try:
                a = float(p.get("alpha", np.nan)); b = float(p.get("beta", np.nan)); z = float(z)
            except Exception:
                continue
            if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(z)):
                continue
            q = dict(p)
            q["alpha"] = a; q["beta"] = b; q["z"] = z
            out.append(q)
        return out

    def _clear_ab_surface_plot(self):
        metric_label = self._surface_metric_label(self._selected_ab_surface_metric())
        title = f"α-β 性能三维网格柱状图：z={metric_label}，等待当前参数下的实测帧"
        axis = "x=α，y=β，z=所选平均性能指标；未测量的 α/β 不会被补值。"
        canvas = getattr(self, "ab_surface_canvas", None)
        if canvas is not None:
            canvas._metric_label = metric_label
            canvas.clear(title, axis)
        popup_canvas = getattr(self, "_ab_surface_window_canvas", None)
        if popup_canvas is not None:
            popup_canvas._metric_label = metric_label
            popup_canvas.clear(title, axis)

    def _update_ab_surface_plot(self, status=None):
        canvas = getattr(self, "ab_surface_canvas", None)
        if self.backend is None:
            self._clear_ab_surface_plot()
            return
        metric = self._selected_ab_surface_metric()
        label = self._surface_metric_label(metric)
        direction = self._surface_metric_direction(metric)
        surface = self._fetch_alpha_beta_surface(metric, status=status)
        points = self._valid_surface_points(surface, metric)
        popup_combo = getattr(self, "_ab_surface_window_metric_combo", None)
        if popup_combo is not None:
            self._sync_combo_to_key(popup_combo, metric)
        if not points:
            progress = dict((surface or {}).get("active_progress", {}) or {})
            done = int(progress.get("sample_count", 0) or 0)
            target = int(progress.get("target_sample_count", (surface or {}).get("samples_per_cell", 1)) or 1)
            if done > 0 and not bool(progress.get("finalized", False)):
                title = f"α-β 性能面：z={label}，正在平均当前点 {done}/{max(target, 1)}，冻结后显示柱体"
            else:
                title = f"α-β 性能三维网格柱状图：z={label}，等待当前参数下的平均实测帧"
            axis = "x=α，y=β，z=所选平均性能指标；每个 α/β 点测够平均窗口后冻结，未测点留空。"
            if canvas is not None:
                canvas.clear(title, axis)
            popup_canvas = getattr(self, "_ab_surface_window_canvas", None)
            if popup_canvas is not None:
                popup_canvas.clear(title, axis)
            return

        z_values = np.asarray([p["z"] for p in points], dtype=np.float64)
        best_idx = int(np.nanargmin(z_values) if direction != "higher" else np.nanargmax(z_values))
        best = points[best_idx]
        best_text = (f"最佳 α={best['alpha']:.3g}, β={best['beta']:.3g}, "
                     f"{label}={self._format_metric_value(metric, best['z'])}")
        if status is None:
            try:
                status = self.backend.get_status()
            except Exception:
                status = {}
        current_alpha = float((status or {}).get("alpha", surface.get("current_alpha", np.nan)))
        current_beta = float((status or {}).get("beta", surface.get("current_beta", np.nan)))
        current_text = ""
        if np.isfinite(current_alpha) and np.isfinite(current_beta):
            current_text = f"；当前 α={current_alpha:.3g}, β={current_beta:.3g}"
        min_text = self._format_metric_value(metric, float(np.nanmin(z_values)))
        max_text = self._format_metric_value(metric, float(np.nanmax(z_values)))
        better_text = "越高越好" if direction == "higher" else "越低越好"
        progress = dict((surface or {}).get("active_progress", {}) or {})
        done = int(progress.get("sample_count", 0) or 0)
        target = int(progress.get("target_sample_count", (surface or {}).get("samples_per_cell", 1)) or 1)
        partial = int((surface or {}).get("partial_count", 0) or 0)
        measuring_text = ""
        if done > 0 and not bool(progress.get("finalized", False)):
            measuring_text = f"；当前点平均中 {done}/{max(target, 1)}"
        elif partial > 0:
            measuring_text = f"；另有 {partial} 个点平均中"
        title = f"α-β 性能面：z={label}（{better_text}），冻结点={len(points)}；{best_text}{current_text}{measuring_text}"
        axis = (f"x=α，y=β，z={label}；冻结实测范围 {min_text} ~ {max_text}。"
                "每个柱体来自同一 α/β 平均窗口，冻结后不再被后续帧改写；XY/XZ/YZ 为未插值二维投影。")
        if canvas is not None:
            canvas.update_surface(points, metric, label, direction, best_idx, current_alpha, current_beta, title, axis)
        popup_canvas = getattr(self, "_ab_surface_window_canvas", None)
        if popup_canvas is not None:
            popup_canvas.update_surface(points, metric, label, direction, best_idx, current_alpha, current_beta, title, axis)


    def _handle_alpha_beta_adaptation(self, status):
        label = getattr(self, "adaptive_status_label", None)
        if status is None:
            return
        enabled = bool(status.get("adaptive_alpha_beta_enabled", False))
        state = str(status.get("adaptive_alpha_beta_state", "disabled"))
        err = str(status.get("adaptive_last_error", "") or "")
        rec_a = float(status.get("adaptive_recommended_alpha", np.nan))
        rec_b = float(status.get("adaptive_recommended_beta", np.nan))
        gain = float(status.get("adaptive_predicted_improvement_db", np.nan))
        snr = float(status.get("adaptive_predicted_snr_db", np.nan))
        step = float(status.get("adaptive_active_step", np.nan))
        direction = str(status.get("adaptive_selected_direction", "none"))
        stable = int(status.get("adaptive_stable_count", 0))
        required = int(status.get("adaptive_stable_required", 0))
        source = str(status.get("adaptive_htf_source", ""))
        ready = bool(status.get("adaptive_alpha_beta_ready", False))
        seq = int(status.get("adaptive_recommendation_seq", 0))

        if label is not None:
            if not enabled:
                label.setText("自适应：关闭")
            elif err:
                label.setText(f"自适应：{state}；{err}")
            elif np.isfinite(rec_a) and np.isfinite(rec_b):
                gain_txt = "nan" if not np.isfinite(gain) else f"{gain:.2f}dB"
                snr_txt = "nan" if not np.isfinite(snr) else f"{snr:.1f}dB"
                label.setText(
                    f"自适应：{state}；推荐 α/β={rec_a:.2f}/{rec_b:.2f}；"
                    f"预测增益={gain_txt}；SNR≈{snr_txt}；稳定={stable}/{required}；CSI={source}"
                )
            else:
                label.setText(f"自适应：{state}，等待有效 H_TF")

        if (not enabled or not ready or not self.adaptive_auto_apply_check.isChecked()
                or seq <= int(self._last_adaptive_recommendation_seq)):
            return
        self._last_adaptive_recommendation_seq = seq
        self._mark_observation_apply(rec_a, rec_b)
        if not (np.isfinite(rec_a) and np.isfinite(rec_b)):
            return
        delta = abs(rec_a - self.alpha_spin.value()) + abs(rec_b - self.beta_spin.value())
        if delta < 0.5 * max(self.adaptive_fine_spin.value(), 0.01):
            return
        self._log(
            f"信道自适应应用 α={rec_a:.2f}, β={rec_b:.2f}；"
            f"论文SER预测改善 {gain:.2f} dB，CSI={source}。"
        )
        if hasattr(self.backend, "apply_alpha_beta_candidate"):
            self.backend.apply_alpha_beta_candidate(
                rec_a, rec_b, predicted_improvement_db=gain,
                recommendation_seq=seq,
            )
        else:
            self._set_indices(rec_a, rec_b)

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
            f"raw={float(status.get('raw_bit_ber',np.nan)):.3g}, SER={float(status.get('measured_ser',np.nan)):.3g}, EVM={evm_txt}, cond={float(status.get('cond_h_cross',np.nan)):.2e}, "
            f"mode={status.get('channel_estimator','')}, ch={status.get('channel_mode','')}, code={status.get('coding_scheme','')}, "
            f"TDLfit={float(status.get('tdl_param_fit_nmse',np.nan)):.2e}, const={status.get('constellation_source','none')}, "
            f"ABauto={status.get('adaptive_alpha_beta_state','off')}, RXoverflow={int(status.get('rx_overflow_count',0))}"
        )

    def _maybe_log_runtime(self, status, stats):
        now = time.monotonic()
        if now - self._last_runtime_log_time < 2.0:
            return
        self._last_runtime_log_time = now
        self._log(
            "v35 runtime: "
            f"reason={status.get('reason','')}, frames={int(status.get('frames_decode_ok',0))}/{int(status.get('frames_processed',0))}, "
            f"rx_new={int(status.get('rx_last_new_samples',0))}, stale={bool(status.get('rx_spectrum_stale',True))}, "
            f"Sync={float(status.get('sync_metric',0.0)):.3f}, CFO={float(status.get('cfo_est_hz',0.0)):.1f}/{status.get('cfo_source','')}, "
            f"alias={float(status.get('cfo_alias_hz',np.nan)):.1f}, scan={float(status.get('cfo_scan_score',np.nan)):.2f}, "
            f"BERfec={float(status.get('fec_bit_ber',status.get('ber',np.nan))):.3g}, raw={float(status.get('raw_bit_ber',np.nan)):.3g}, "
            f"EVM={float(status.get('evm_average_percent',np.nan)):.2f}%, "
            f"mode={status.get('channel_estimator','')}, ch={status.get('channel_mode','')}, fd={float(status.get('tdl_doppler_hz',0.0)):.1f}, "
            f"spread={float(status.get('tdl_doppler_spread_hz',0.0)):.1f}, TDLfit={float(status.get('tdl_param_fit_nmse',np.nan)):.2e}, "
            f"code={status.get('coding_scheme','')}, txvec={int(status.get('tx_waveform_samples',0))}, "
            f"prerender={bool(status.get('tx_tdl_prerendered',False))}, decode_ok={bool(stats.get('decode_ok',False))}, "
            f"ABauto={status.get('adaptive_alpha_beta_state','off')}, "
            f"ABrec={float(status.get('adaptive_recommended_alpha',np.nan)):.2f}/"
            f"{float(status.get('adaptive_recommended_beta',np.nan)):.2f}, "
            f"ABgain={float(status.get('adaptive_predicted_improvement_db',np.nan)):.2f}dB"
            f", RXoverflow={int(status.get('rx_overflow_count',0))}"
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
        self.rx_curve.setData([], []); self.evm_curve.setData([], [])
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
            f"TDL预渲染={st.get('tx_tdl_prerendered')}, "
            f"αβ自适应={st.get('adaptive_alpha_beta_state','off')}"
        )

    def _log(self, message):
        from datetime import datetime
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        try:
            self._ui_log_entries.append(line)
        except Exception:
            pass
        label = getattr(self, "log_status_label", None)
        if label is not None:
            try:
                text = str(message)
                if len(text) > 150:
                    text = text[:150] + "…"
                label.setText("最近日志：" + text)
            except Exception:
                pass
