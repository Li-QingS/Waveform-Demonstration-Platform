# -*- coding: utf-8 -*-
from __future__ import annotations

import gc
import math
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
from PyQt5.QtCore import Qt, QTimer, QSignalBlocker, QEvent
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QPushButton, QTabWidget,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QTextEdit, QSplitter,
    QScrollArea, QSizePolicy, QCheckBox, QFileDialog, QDialog, QLayout,
)

from ..hardware.fdidm_hardtest import FDIDMHardwareTest, FDIDM_GAIN_MAX_DB

MATLAB_BLUE = (0, 114, 189)
MATLAB_ORANGE = (217, 83, 25)
MATLAB_PURPLE = (126, 47, 142)
LIGHT_BG = (250, 250, 250)
AXIS_COLOR = (60, 60, 60)
BORDER_COLOR = (225, 225, 225)


class _ControlScrollArea(QScrollArea):
    """Keep the form inside the viewport when the vertical bar appears.

    The first workspace show can otherwise leave the content at the old
    width for a layout pass. Horizontal scrolling is intentionally disabled,
    so that stale width would hide the right edges of the form controls.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fitting_width = False
        self.viewport().installEventFilter(self)
        self.verticalScrollBar().rangeChanged.connect(self._fit_width)

    def eventFilter(self, watched, event):
        if watched is self.viewport() and event.type() in (QEvent.Resize, QEvent.Show):
            self._fit_width()
        return super().eventFilter(watched, event)

    def _fit_width(self, *_args):
        content = self.widget()
        if content is None or self._fitting_width:
            return
        width = self.viewport().width()
        if width < max(1, content.minimumSizeHint().width()):
            return
        self._fitting_width = True
        try:
            content.setMaximumWidth(width)
            content.resize(width, content.height())
        finally:
            self._fitting_width = False


from .fdidm_plot_widgets import _AlphaBetaSurfaceCanvas
from .base_waveform_tab import ElidedValueLabel
from .hardware_advantage_observer import (
    MODE_OFF,
    MODE_ON,
    AdvantageObservationSession,
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
        self._last_surface_refresh_wall = 0.0
        self._last_debug_seq = 0
        self._auto_debug_level = "INFO"
        self._applying_params = False
        self._pending_apply = False
        self._suppress_param_signals = False
        self._last_adaptive_recommendation_seq = 0
        self._last_backend_alpha_beta = None
        self._last_status_snapshot = {}
        self._adaptive_toggle_pending = False
        self.observer = AdvantageObservationSession()
        self._timeline_event_items = []
        self._timeline_anomaly_items = []
        self._timeline_apply_points = []
        self._obs_t = deque(maxlen=1800)
        self._obs_ser = deque(maxlen=1800)
        self._obs_evm = deque(maxlen=1800)
        self._obs_ser_trend = deque(maxlen=1800)
        self._obs_evm_trend = deque(maxlen=1800)
        self._obs_alpha = deque(maxlen=1800)
        self._obs_beta = deque(maxlen=1800)
        self._obs_t0 = 0.0
        self._last_timeline_frame = -1
        # The effect chart has a fixed categorical x-axis.  Keep track of a
        # deliberate operator zoom so a newly arriving frame cannot silently
        # undo it by calling setYRange on every refresh.
        self._timeline_effect_user_y_zoom = False
        self._timeline_effect_range_initialized = False
        self._timeline_effect_setting_range = False
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
        self._ai_backend_log_snapshot = ""
        self._ai_analysis_dialog = None
        self._ab_surface_window = None
        self._ab_surface_window_canvas = None
        self._ab_surface_window_metric_combo = None
        self._init_ui()
        self._init_ai_controls()
        self._init_plot_style()
        self._connect_signals()
        self._apply_debounce_timer = QTimer(self)
        self._apply_debounce_timer.setSingleShot(True)
        self._apply_debounce_timer.timeout.connect(self._apply_params_to_backend)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._refresh_plots)

    # ---------------- UI ----------------
    def _init_ai_controls(self):
        from .fdidm_ai_analysis import FDIDMAIFloatingButton

        self.ai_floating_button = FDIDMAIFloatingButton(self._open_ai_analysis, self)
        self._position_ai_floating_button()

    def _position_ai_floating_button(self):
        button = getattr(self, "ai_floating_button", None)
        if button is None:
            return
        if getattr(self, "_global_ai_assistant", None) is not None:
            button.hide()
            return
        if not button.user_moved:
            button.move(max(0, self.width() - button.width() - 18),
                        max(0, self.height() - button.height() - 62))
        button.clamp_to_parent()
        button.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_ai_floating_button()

    def _open_ai_analysis(self):
        global_assistant = getattr(self, "_global_ai_assistant", None)
        if global_assistant is not None:
            return global_assistant.open()
        from .fdidm_ai_analysis import FDIDMAIAnalysisDialog

        if self._ai_analysis_dialog is None:
            self._ai_analysis_dialog = FDIDMAIAnalysisDialog(
                self._get_ai_log_snapshot, self, chart_context_provider=self._get_ai_chart_context)
        self._ai_analysis_dialog.show()
        self._ai_analysis_dialog.raise_()
        self._ai_analysis_dialog.activateWindow()

    def set_global_ai_assistant(self, assistant):
        self._global_ai_assistant = assistant
        self.ai_floating_button.hide()

    def _get_ai_chart_context(self):
        from .fdidm_ai_context import build_fdidm_chart_context

        return build_fdidm_chart_context(self)

    def _get_ai_log_snapshot(self):
        """Read diagnostics without exporting files or modifying the live receiver."""
        if self.backend is not None:
            # A newly configured backend must never inherit an old run's DEBUG log.
            self._ai_backend_log_snapshot = ""
            if hasattr(self.backend, "get_debug_log"):
                try:
                    entries = self.backend.get_debug_log(max_entries=5000, min_level="DEBUG")
                    self._ai_backend_log_snapshot = "\n".join(
                        f"{entry['seq']:06d} {entry['t']:10.3f}s "
                        f"{entry['level']:<5} {entry['msg']}" for entry in entries
                    )
                except Exception:
                    self._ai_backend_log_snapshot = "后端诊断日志读取失败；以下仅含可用的界面日志。"
        backend_text = self._ai_backend_log_snapshot
        ui_entries = list(self._ui_log_entries)
        if backend_text:
            # INFO backend entries are also echoed into the UI deque. Sending
            # them twice wastes context and can distort the model's counts.
            ui_entries = [entry for entry in ui_entries if "] BE[" not in entry]
        parts = []
        if backend_text:
            parts.append("--- 后端最近 DEBUG 日志（最多 5000 条） ---\n" + backend_text)
        if ui_entries:
            parts.append("--- 当前页面界面日志（最近记录，可能包含多次测试） ---\n" + "\n".join(ui_entries))
        return "\n\n".join(parts)

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.main_splitter = splitter
        layout.addWidget(splitter)
        splitter.addWidget(self._create_controls_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setSizes([360, 1180])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        try:
            splitter.setCollapsible(1, False)
        except Exception:
            pass

    def _create_controls_panel(self):
        rail = QWidget()
        rail.setMinimumWidth(340)
        rail.setMaximumWidth(400)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(0, 0, 0, 0)
        rail_layout.setSpacing(6)
        scroll = _ControlScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)

        panel = QWidget()
        self.controls_panel = panel
        self.controls_scroll = scroll
        layout = QVBoxLayout(panel)
        # Let the scroll area take excess height.  Compressing the groups
        # below their form contents makes their borders overlap the controls.
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)

        hw_group = QGroupBox("链路配置")
        hw = QGridLayout(hw_group)
        hw.setHorizontalSpacing(6)
        hw.setVerticalSpacing(6)
        self.device_combo = self._combo([("B210", "USRP B210"), ("N210", "USRP N210"), ("X310", "USRP X310")])
        self.samp_rate_spin = self._dspin(1e5, 100e6, 500_000, 0, " Hz")
        self.fc_spin = self._dspin(70e6, 6e9, 2.4e9, 0, " Hz")
        hw.addWidget(QLabel("设备"), 0, 0); hw.addWidget(self.device_combo, 0, 1)
        self.samp_rate_label = QLabel("采样率")
        hw.addWidget(self.samp_rate_label, 1, 0); hw.addWidget(self.samp_rate_spin, 1, 1)
        hw.addWidget(QLabel("中心频率"), 2, 0); hw.addWidget(self.fc_spin, 2, 1)
        self.rf_port_note = QLabel("B210：A:TX/RX → A:RX2；增益 ≤45 dB")
        self.rf_port_note.setToolTip("B210 端口建议：A:TX/RX，A:RX2；增益上限 45 dB")
        self.rf_port_note.setWordWrap(True)
        hw.addWidget(self.rf_port_note, 3, 0, 1, 2)
        self.btn_antenna_preset = QPushButton("B210 天线/增益预设")
        self.btn_antenna_preset.setToolTip("选择 B210 A:TX/RX + A:RX2，并将 TX/RX 增益设为 25/45 dB；45 dB 是硬件安全上限，外接衰减器仍不可省略")
        self.btn_antenna_preset.clicked.connect(self._on_antenna_preset)
        hw.addWidget(self.btn_antenna_preset, 4, 0, 1, 2)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.hw_group = hw_group
        layout.addWidget(hw_group)

        fd_group = QGroupBox("FDIDM 参数")
        fd = QGridLayout(fd_group)
        fd.setHorizontalSpacing(6)
        fd.setVerticalSpacing(6)
        self.alpha_spin = self._dspin(0.0, 2.0, 0.5, 2, "", 0.05)
        self.beta_spin = self._dspin(0.0, 2.0, 1.0, 2, "", 0.05)
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
        self.channel_estimator_label = QLabel("估计")
        fd.addWidget(self.channel_estimator_label, 5, 0); fd.addWidget(self.channel_estimator_combo, 5, 1)
        fd.addWidget(QLabel("H间隔"), 5, 2); fd.addWidget(self.htf_update_spin, 5, 3)
        fd.addWidget(QLabel("处理ms"), 6, 0); fd.addWidget(self.process_interval_spin, 6, 1)
        fd.addWidget(self.htf_once_check, 6, 2, 1, 2)
        fd.addWidget(QLabel("编码"), 7, 0); fd.addWidget(self.coding_combo, 7, 1)
        fd.addWidget(self.coding_interleaver_check, 7, 2, 1, 2)
        fd.addWidget(QLabel("UHD帧"), 8, 0); fd.addWidget(self.uhd_buf_spin, 8, 1)
        fd.addWidget(QLabel("TX向量ms"), 8, 2); fd.addWidget(self.tx_vec_ms_spin, 8, 3)
        fd.addWidget(QLabel("链路"), 9, 0); fd.addWidget(self.channel_mode_combo, 9, 1, 1, 3)
        self.tdl_ds_label = QLabel("RMS-DS")
        self.tdl_fd_label = QLabel("Doppler")
        self.tdl_spread_label = QLabel("扩展")
        self.tdl_snr_label = QLabel("TDL注入设定SNR")
        fd.addWidget(self.tdl_ds_label, 10, 0); fd.addWidget(self.tdl_ds_spin, 10, 1)
        fd.addWidget(self.tdl_fd_label, 10, 2); fd.addWidget(self.tdl_fd_spin, 10, 3)
        fd.addWidget(self.tdl_spread_label, 11, 0); fd.addWidget(self.tdl_spread_spin, 11, 1)
        fd.addWidget(self.tdl_snr_label, 11, 2); fd.addWidget(self.tdl_snr_spin, 11, 3)
        fd.addWidget(self.prerender_tdl_check, 12, 0, 1, 4)
        self.btn_ofdm = QPushButton("OFDM 0/0")
        self.btn_otfs = QPushButton("OTFS 1/1")
        self.btn_reco = QPushButton("推荐 0.5/1")
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
        # Keep a handle so the optional rows can be moved into the collapsible
        # advanced section below.  The alpha/beta row and link selector remain
        # in this compact group; the rest is still available on demand.
        self.fd_group = fd_group
        layout.addWidget(fd_group)
        # A lightweight visibility handle is kept separate from the always
        # visible FDIDM essentials (alpha/beta and channel path).  The actual
        # advanced widgets are toggled below so the operator never loses the
        # two parameters that explain the adaptive result.
        self.advanced_group = QGroupBox("高级参数")
        self.advanced_group.setMinimumHeight(0)
        self.advanced_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(self.advanced_group)


        adapt_group = QGroupBox("接收机相关 GMI 自适应")
        adapt = QGridLayout(adapt_group)
        adapt.setHorizontalSpacing(6)
        adapt.setVerticalSpacing(6)
        self.adaptive_enable_check = QCheckBox("启用 GMI 自适应")
        # The demo starts in fixed mode so the operator can explicitly enable
        # adaptation and observe the before/after change.
        self.adaptive_enable_check.setChecked(False)
        self.adaptive_enable_check.setToolTip("启用后，满足鲁棒门限的候选会自动进入 A-B-B-A 真链路验证。")
        self.gmi_receiver_combo = self._combo([
            ("对角接收机 K=0", "diagonal"),
            ("完整匹配（对照）", "full_matched"),
        ], chars=14)
        self.gmi_catalogue_combo = self._combo([
            ("4", 4), ("16（推荐）", 16), ("36", 36),
        ], chars=10)
        self.gmi_catalogue_combo.setCurrentIndex(1)
        self.gmi_robustness_combo = self._combo([
            ("标准", "standard"), ("快速", "fast"), ("保守", "conservative"),
        ], chars=8)
        self.btn_adaptive_evaluate = QPushButton("立即评估")
        self.btn_adaptive_evaluate.setToolTip("使用最近一次可信 CSI 对有限波形库计算接收机相关 GMI；没有可信 CSI 时等待下一帧。")
        self.adaptive_status_label = QLabel("自适应：关闭")
        self.adaptive_status_label.setWordWrap(True)
        self.adaptive_status_label.setStyleSheet("color: #444444;")
        adapt.addWidget(self.adaptive_enable_check, 0, 0, 1, 2)
        adapt.addWidget(self.btn_adaptive_evaluate, 0, 2, 1, 2)
        adapt.addWidget(QLabel("GMI 接收机"), 1, 0); adapt.addWidget(self.gmi_receiver_combo, 1, 1, 1, 3)
        adapt.addWidget(QLabel("波形库预算"), 2, 0); adapt.addWidget(self.gmi_catalogue_combo, 2, 1, 1, 3)
        adapt.addWidget(QLabel("鲁棒等级"), 3, 0); adapt.addWidget(self.gmi_robustness_combo, 3, 1, 1, 3)
        adapt.addWidget(self.adaptive_status_label, 4, 0, 1, 4)
        adapt_note = QLabel("净 GMI 负责选择；EVM/SER/CRC 只做 A-B-B-A 非回归验证。")
        adapt_note.setWordWrap(True)
        adapt_note.setToolTip(
            "候选只由公共 CSI 和接收机模型计算的净 GMI 下界产生；"
            "payload、EVM、SER、CRC 不参与在线选择，只用于真链路提交保护。"
        )
        adapt.addWidget(adapt_note, 5, 0, 1, 4)
        # Keep the default rail decision-oriented; the methodology remains on
        # hover and in exported evidence instead of occupying another row.
        adapt_note.setVisible(False)
        adapt_group.setToolTip(adapt_note.toolTip())
        self.adapt_group = adapt_group
        layout.addWidget(adapt_group)

        modem_group = QGroupBox("收发/显示")
        modem = QGridLayout(modem_group)
        modem.setHorizontalSpacing(6)
        modem.setVerticalSpacing(6)
        self.tx_gain_spin = self._dspin(0, FDIDM_GAIN_MAX_DB, 10, 1, " dB")
        self.rx_gain_spin = self._dspin(0, FDIDM_GAIN_MAX_DB, 20, 1, " dB")
        # TX/RX gain is part of normal RF bring-up and stays visible when
        # optional diagnostics are collapsed.
        self.tx_gain_label = QLabel("TX gain")
        self.rx_gain_label = QLabel("RX gain")
        gain_group = QGroupBox("RF 增益")
        gain = QGridLayout(gain_group)
        gain.setHorizontalSpacing(6)
        gain.setVerticalSpacing(6)
        gain.addWidget(self.tx_gain_label, 0, 0); gain.addWidget(self.tx_gain_spin, 0, 1)
        gain.addWidget(self.rx_gain_label, 1, 0); gain.addWidget(self.rx_gain_spin, 1, 1)
        layout.addWidget(gain_group)
        self.gain_group = gain_group
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
        # TX/RX gains are placed in the always-visible RF gain group above.
        modem.addWidget(QLabel("调制"), 2, 0); modem.addWidget(self.mod_order_combo, 2, 1)
        modem.addWidget(QLabel("均衡"), 3, 0); modem.addWidget(self.equalizer_combo, 3, 1)
        modem.addWidget(QLabel("星座"), 4, 0); modem.addWidget(self.const_mode_combo, 4, 1)
        modem.addWidget(QLabel("3D Z轴"), 5, 0); modem.addWidget(self.ab_z_metric_combo, 5, 1)
        modem.addWidget(QLabel("RX源"), 6, 0); modem.addWidget(self.rx_plot_combo, 6, 1)
        layout.addWidget(modem_group)
        self.modem_group = modem_group

        text_group = QGroupBox("发送文本")
        text_l = QVBoxLayout(text_group)
        self.tx_text_edit = QTextEdit()
        self.tx_text_edit.setPlainText("FDIDM OK")
        self.tx_text_edit.setMaximumHeight(90)
        text_l.addWidget(QLabel("待发送文本"))
        text_l.addWidget(self.tx_text_edit)
        layout.addWidget(text_group)
        self.text_group = text_group

        # The default operator view should expose only the controls needed to
        # bring up the link and explain an adaptive result.  Keep the complete
        # lab configuration, including the modem/display and text editors,
        # inside one genuinely collapsed container.  Merely hiding children
        # leaves their size hints in QLayouts on some Qt versions and makes the
        # left panel needlessly tall.
        self._advanced_layout = QVBoxLayout(self.advanced_group)
        self._advanced_layout.setContentsMargins(6, 8, 6, 6)
        self._advanced_layout.setSpacing(6)
        self._advanced_layout.setSizeConstraint(QLayout.SetMinimumSize)

        def _move_grid_rows(source_group, rows, title):
            """Move a group of form rows without inheriting four-column widths.

            Read each layout item once: itemAtPosition on a spanning item can
            otherwise lose its original span when the widget is reinserted.
            The lab forms use a label/value pair per row, with standalone
            notes/checks spanning the width and pairs of action buttons.
            """
            source_layout = source_group.layout()
            moved_group = QGroupBox(title)
            moved_layout = QGridLayout(moved_group)
            moved_layout.setHorizontalSpacing(6)
            moved_layout.setVerticalSpacing(4)
            moved_layout.setColumnStretch(1, 1)
            row_widgets = {}
            for index in range(source_layout.count()):
                row, column, _row_span, _column_span = source_layout.getItemPosition(index)
                item = source_layout.itemAt(index)
                if row in rows and item.widget() is not None:
                    row_widgets.setdefault(row, []).append((column, item.widget()))
            for items in row_widgets.values():
                for _column, widget in items:
                    source_layout.removeWidget(widget)
            destination_row = 0
            for source_row in rows:
                widgets = [widget for _column, widget in sorted(row_widgets.get(source_row, []))]
                cursor = 0
                while cursor < len(widgets):
                    widget = widgets[cursor]
                    following = widgets[cursor + 1] if cursor + 1 < len(widgets) else None
                    if isinstance(widget, QLabel) and following is not None:
                        moved_layout.addWidget(widget, destination_row, 0)
                        moved_layout.addWidget(following, destination_row, 1)
                        cursor += 2
                    elif isinstance(widget, QPushButton) and isinstance(following, QPushButton):
                        moved_layout.addWidget(widget, destination_row, 0)
                        moved_layout.addWidget(following, destination_row, 1)
                        cursor += 2
                    else:
                        moved_layout.addWidget(widget, destination_row, 0, 1, 2)
                        cursor += 1
                    destination_row += 1
            self._advanced_layout.addWidget(moved_group)
            return moved_group

        # Essential rows retained in the compact groups:
        #   hardware: device/frequency/antenna; FDIDM: α/β + link; adaptive:
        #   enable/evaluate/status.  All tuning knobs stay one click away.
        self._advanced_hw_group = _move_grid_rows(self.hw_group, [1], "采样率")
        self._advanced_fd_group = _move_grid_rows(
            self.fd_group,
            [1, 2, 3, 4, 5, 6, 7, 8], "波形与处理")
        self._advanced_channel_group = _move_grid_rows(
            self.fd_group, [10, 11, 12], "注入信道")
        self._advanced_actions_group = _move_grid_rows(
            self.fd_group, [13, 14, 15, 16, 17], "参数应用与性能面")
        # Remove optional editor groups from the main layout before parenting
        # them under the collapsed group.  They are reinserted automatically
        # when the operator checks “显示高级参数”.
        layout.removeWidget(self.modem_group)
        layout.removeWidget(self.text_group)
        self._advanced_layout.addWidget(self.modem_group)
        self._advanced_layout.addWidget(self.text_group)
        layout.removeWidget(self.advanced_group)

        self.btn_toggle_advanced = QCheckBox("显示高级参数")
        # Keep RF bring-up together; optional lab controls follow the common
        # link, waveform and adaptive settings instead of splitting them.
        layout.removeWidget(self.gain_group)
        layout.insertWidget(1, self.gain_group)
        layout.addWidget(self.btn_toggle_advanced)
        self.btn_toggle_advanced.toggled.connect(self._toggle_advanced_controls)
        self.advanced_group.setVisible(False)
        self.modem_group.setVisible(False)
        self.text_group.setVisible(False)
        self.samp_rate_spin.setVisible(False)
        self.channel_estimator_combo.setVisible(False)
        self._toggle_advanced_controls(False)

        btn_group = QGroupBox("运行控制")
        # A two-column command pad keeps the always-visible controls compact;
        # the previous vertical stack consumed an entire screen on the narrow
        # side panel even though each action is only one click.
        btn_l = QGridLayout(btn_group)
        btn_l.setContentsMargins(5, 5, 5, 5)
        btn_l.setHorizontalSpacing(4)
        btn_l.setVerticalSpacing(3)
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
        btn_l.addWidget(self.btn_connect, 0, 0)
        btn_l.addWidget(self.btn_start_test, 0, 1)
        btn_l.addWidget(self.btn_stop_test, 1, 0)
        btn_l.addWidget(self.btn_export_log, 1, 1)
        btn_l.addWidget(self.log_status_label, 2, 0, 1, 2)
        # Export feedback remains available through this label and the button
        # tooltip, but it does not consume scarce height in the default
        # operator rail.  It is made visible only after an export action.
        self.log_status_label.setVisible(False)
        layout.addStretch()
        for compact_layout in (hw, fd, adapt, gain, btn_l):
            compact_layout.setContentsMargins(5, 6, 5, 5)
            compact_layout.setVerticalSpacing(3)
        self._compact_left_controls(panel)
        for group in panel.findChildren(QGroupBox):
            group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._on_channel_mode_changed()
        self._on_device_changed()
        scroll.setWidget(panel)
        # Commands stay outside the scrolling configuration area, including
        # while the advanced drawer is expanded.
        rail_layout.addWidget(scroll, 1)
        rail_layout.addWidget(btn_group)
        self._apply_control_style(rail)
        return rail

    def _create_right_panel(self):
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 观测页签的三张图各自回答一个问题：
        # 1) 效果图：同一局部信道窗口内，FDIDM ON 相对 OFF 的变化；
        # 2) 质量图：EVM 随时间是否稳定；
        # 3) 动作图：α/β 何时改变、是否真的应用。
        # 三个 ViewBox 独立缩放，避免操作一张图改变其它图的坐标。
        self.timeline_effect_plot = pg.PlotWidget(title="局部 OFF→ON 效果")
        self.timeline_effect_plot.setToolTip("FDIDM 效果：相邻 OFF→ON 局部窗口；OFF 基线为 100%，ON 越低越好。")
        # Compatibility alias for callers/tests written before the effect
        # chart was given its explicit name.
        self.timeline_ser_plot = self.timeline_effect_plot
        self.timeline_evm_plot = pg.PlotWidget(title="数据辅助 EVM 趋势")
        self.timeline_evm_plot.setToolTip("解调误差趋势：使用已知发送数据辅助计算的 EVM，而非判决参考 EVM。")
        self.timeline_ab_plot = pg.PlotWidget(title="自适应动作 α / β")
        self.timeline_ab_plot.setToolTip("自适应动作：α/β 随时间变化；三角标记表示已实际应用。")
        # Each observation plot owns its ViewBox and interaction range.
        for _plot in (self.timeline_effect_plot, self.timeline_evm_plot, self.timeline_ab_plot):
            _plot.setMouseEnabled(x=True, y=True)
            # Never inherit a link left by a previous plot layout.  Each
            # observation chart owns its horizontal and vertical range so a
            # wheel/drag operation cannot unexpectedly rescale its neighbours.
            _view_box = _plot.getPlotItem().getViewBox()
            _view_box.setXLink(None)
            _view_box.setYLink(None)
        self.timeline_effect_plot.setLabel("left", "相对 OFF", units="%")
        self.timeline_evm_plot.setLabel("left", "数据辅助 EVM", units="%")
        self.timeline_ab_plot.setLabel("left", "参数值")
        self.timeline_effect_plot.setLabel("bottom", "指标")
        self.timeline_evm_plot.setLabel("bottom", "观测时间 (s)")
        self.timeline_ab_plot.setLabel("bottom", "观测时间 (s)")

        self.observation_stats_group = QGroupBox("窗口汇总")
        stats_layout = QVBoxLayout(self.observation_stats_group)
        self.observation_stats_label = QLabel("硬件测试启动后自动记录各窗口汇总")
        self.observation_stats_label.setWordWrap(True)
        self.observation_stats_label.setMinimumSize(0, 0)
        self.observation_stats_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        self.observation_stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.observation_stats_label.setStyleSheet("color:#555555; font-size: 12px;")
        stats_layout.setContentsMargins(6, 6, 6, 6)
        self.observation_stats_scroll = QScrollArea()
        self.observation_stats_scroll.setFrameShape(QScrollArea.NoFrame)
        self.observation_stats_scroll.setWidgetResizable(True)
        self.observation_stats_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        stats_content = QWidget()
        stats_content_layout = QVBoxLayout(stats_content)
        stats_content_layout.setContentsMargins(0, 0, 0, 0)
        stats_content_layout.addWidget(self.observation_stats_label)
        stats_content_layout.addStretch(1)
        self.observation_stats_scroll.setWidget(stats_content)
        stats_layout.addWidget(self.observation_stats_scroll, 1)

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

        # 两个页签、每页 2×2 四宫格（v2 方向修订）。恢复 _PlotGridCell 包装：
        # 它刻意压小 sizeHint，让网格拉伸规则主导，防止 OpenGL/pyqtgraph
        # 组件的大 sizeHint 把四宫格挤变形。
        from .fdidm_plot_widgets import _PlotGridCell as _Cell
        self.ab_surface_cell = _Cell(self.ab_surface_panel)
        self.rx_spectrum_cell = _Cell(self.rx_spectrum_plot)
        self.evm_cell = _Cell(self.evm_plot)
        self.constellation_cell = _Cell(self.constellation_plot)

        diag_grid = QWidget()
        diag_grid.setMinimumSize(0, 0)
        diag_grid.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        dg = QGridLayout(diag_grid)
        dg.setContentsMargins(0, 0, 0, 0)
        dg.setSpacing(6)
        for r in (0, 1):
            dg.setRowStretch(r, 1)
        for c in (0, 1):
            dg.setColumnStretch(c, 1)
        dg.addWidget(self.ab_surface_cell, 0, 0)
        dg.addWidget(self.rx_spectrum_cell, 0, 1)
        dg.addWidget(self.evm_cell, 1, 0)
        dg.addWidget(self.constellation_cell, 1, 1)

        obs_grid = QWidget()
        obs_grid.setMinimumSize(0, 0)
        obs_grid.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        og = QGridLayout(obs_grid)
        og.setContentsMargins(0, 0, 0, 0)
        og.setSpacing(6)
        for r in (0, 1):
            og.setRowStretch(r, 1)
        for c in (0, 1):
            og.setColumnStretch(c, 1)
        self.timeline_ser_cell = _Cell(self.timeline_effect_plot)
        self.timeline_evm_cell = _Cell(self.timeline_evm_plot)
        self.timeline_ab_cell = _Cell(self.timeline_ab_plot)
        self.observation_stats_cell = _Cell(self.observation_stats_group)
        og.addWidget(self.timeline_ser_cell, 0, 0)
        og.addWidget(self.timeline_evm_cell, 0, 1)
        og.addWidget(self.timeline_ab_cell, 1, 0)
        og.addWidget(self.observation_stats_cell, 1, 1)

        self.plot_tabs = QTabWidget()
        self.plot_tabs.addTab(diag_grid, "链路诊断")
        self.plot_tabs.addTab(obs_grid, "优势观测")
        # The operator-facing observation result is the default landing page;
        # raw diagnostics remain one click away in the first tab.
        self.plot_tabs.setCurrentIndex(1)
        layout.addWidget(self.plot_tabs, 1)

        # Keep the real validation state visible on both chart pages. This
        # row is independent of the optional text comparison's height limit.
        self.adaptive_state_label = QLabel("自适应：等待硬件测试")
        self.adaptive_state_label.setWordWrap(True)
        self.adaptive_state_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        self.adaptive_state_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.adaptive_state_label.setStyleSheet("color:#555555; font-size:12px;")
        layout.addWidget(self.adaptive_state_label, 0)

        text_panel = QWidget()
        text_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        text_panel.setMaximumHeight(120)
        text_grid = QGridLayout(text_panel)
        text_grid.setContentsMargins(0, 0, 0, 0)
        text_grid.setHorizontalSpacing(8)
        text_grid.setVerticalSpacing(2)
        self.decode_status_label = ElidedValueLabel("解调状态：未开始")
        self.decode_status_label.setMinimumHeight(22)
        self.decode_status_label.setMaximumHeight(24)
        self.decode_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.btn_show_text = QCheckBox("显示文本对照")
        text_grid.addWidget(self.decode_status_label, 0, 0)
        text_grid.addWidget(self.btn_show_text, 0, 1, Qt.AlignRight)
        self.text_result_panel = QWidget()
        text_result_grid = QGridLayout(self.text_result_panel)
        text_result_grid.setContentsMargins(0, 0, 0, 0)
        text_result_grid.setSpacing(2)
        self.tx_text_view = QTextEdit(); self.tx_text_view.setReadOnly(True); self.tx_text_view.setMaximumHeight(64)
        self.rx_text_view = QTextEdit(); self.rx_text_view.setReadOnly(True); self.rx_text_view.setMaximumHeight(64)
        text_result_grid.addWidget(QLabel("发送文本"), 0, 0)
        text_result_grid.addWidget(QLabel("接收文本"), 0, 1)
        text_result_grid.addWidget(self.tx_text_view, 1, 0)
        text_result_grid.addWidget(self.rx_text_view, 1, 1)
        text_grid.addWidget(self.text_result_panel, 1, 0, 1, 2)
        self.text_result_panel.setVisible(False)
        self.btn_show_text.toggled.connect(self.text_result_panel.setVisible)
        layout.addWidget(text_panel, 0)

        self.rx_curve = self.rx_spectrum_plot.plot(pen=pg.mkPen(MATLAB_ORANGE, width=2))
        self.evm_curve = self.evm_plot.plot(pen=pg.mkPen(MATLAB_PURPLE, width=2), name="current EVM")
        # Keep the raw SER trend as a hidden compatibility/export trace on the
        # quality plot.  The first chart is deliberately an effect-only view:
        # it must contain only the OFF baseline and the ON candidate bars, so
        # an old hidden curve cannot be mistaken for a yellow event marker (or
        # affect its auto-range).
        # Keep the SER trend as a data-only compatibility trace.  It is no
        # longer attached to the EVM plot: log10(SER) and EVM have different
        # units and putting them on one axis made the quality chart misleading.
        self.timeline_ser_curve = pg.PlotDataItem(name="SER trend (data only)")
        # Effect-first bars (OFF baseline = 100%, ON candidate = local ratio).
        self.timeline_baseline_bars = pg.BarGraphItem(x=[0.0, 1.0], height=[0.0, 0.0], width=0.30,
                                                       brush=pg.mkBrush(150, 150, 150, 205),
                                                       pen=pg.mkPen((90, 90, 90), width=1))
        self.timeline_candidate_bars = pg.BarGraphItem(x=[0.38, 1.38], height=[0.0, 0.0], width=0.30,
                                                        brush=pg.mkBrush(35, 150, 85, 225),
                                                        pen=pg.mkPen((35, 150, 85), width=1))
        self.timeline_effect_plot.addItem(self.timeline_baseline_bars)
        self.timeline_effect_plot.addItem(self.timeline_candidate_bars)
        # The two groups are categorical (EVM and SER), not time samples.
        # Explicit ticks make the meaning obvious even when the chart has no
        # valid OFF→ON pair yet.  Keep a small horizontal margin around the
        # bars so labels never touch the plot frame.
        self.timeline_effect_plot.getAxis("bottom").setTicks([[
            (0.19, "EVM"), (1.19, "SER")
        ]])
        effect_view = self.timeline_effect_plot.getPlotItem().getViewBox()
        effect_view.disableAutoRange()
        effect_view.setXRange(-0.30, 1.70, padding=0)
        effect_view.setYRange(0.0, 120.0, padding=0)
        effect_view.sigRangeChangedManually.connect(self._on_effect_range_manually_changed)
        self.timeline_effect_labels = [
            pg.TextItem("", anchor=(0.5, 1), color=(55, 55, 55)),
            pg.TextItem("", anchor=(0.5, 1), color=(55, 55, 55)),
        ]
        for _item in self.timeline_effect_labels:
            # Labels are shown only while a valid pair exists.  In particular,
            # do not leave an old number on screen after a channel-context
            # change invalidates the local baseline.
            _item.setVisible(False)
            self.timeline_effect_plot.addItem(_item)
        # A real legend (rather than relying on bar colour alone) keeps the
        # comparison self-explanatory when the chart is exported or viewed in
        # grayscale.  It is anchored to the plot canvas and ignored by the
        # data bounds, so it cannot alter either the categorical range or the
        # operator's zoom.
        self.timeline_effect_legend = pg.LegendItem(
            offset=(8, 8),
            verSpacing=1,
            labelTextColor=(55, 55, 55),
            labelTextSize="8pt",
            brush=pg.mkBrush(250, 250, 250, 215),
            pen=pg.mkPen((190, 190, 190), width=1),
        )
        # Anchor inside the data view. PlotItem coordinates include its title
        # and axes, so the same offset there puts the legend over the title.
        self.timeline_effect_legend.setParentItem(self.timeline_effect_plot.getViewBox())
        self.timeline_effect_legend.addItem(self.timeline_baseline_bars, "OFF 基线 100%")
        self.timeline_effect_legend.addItem(self.timeline_candidate_bars, "ON 候选（越低越好）")
        self.timeline_ser_curve.setVisible(False)
        self.timeline_evm_curve = self.timeline_evm_plot.plot(
            pen=pg.mkPen(MATLAB_ORANGE, width=2), name="EVM%", connect="finite")
        self.timeline_alpha_curve = self.timeline_ab_plot.plot(
            pen=pg.mkPen(MATLAB_PURPLE, width=2), name="α", connect="finite")
        self.timeline_beta_curve = self.timeline_ab_plot.plot(
            pen=pg.mkPen(MATLAB_PURPLE, width=1, style=Qt.DashLine), name="β", connect="finite")
        self.timeline_apply_scatter = pg.ScatterPlotItem(
            # Blue marker is reserved for an applied action.  Yellow is not
            # used anywhere in the observation page, avoiding confusion with
            # the old (removed) vertical toggle marker in the first chart.
            size=12, symbol="t", pen=pg.mkPen(MATLAB_BLUE, width=1),
            brush=pg.mkBrush(0, 114, 189, 220))
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
            with QSignalBlocker(combo):
                combo.setCurrentIndex(idx)

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
            edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for button in panel.findChildren(QPushButton):
            # Native size hints include text and stylesheet padding. A hard
            # maximum below that hint clips the button's outline and label.
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for note in panel.findChildren(QLabel):
            if note.wordWrap():
                note.setMinimumWidth(0)
                note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)

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
                        self.timeline_effect_plot, self.timeline_evm_plot, self.timeline_ab_plot]
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
        self.btn_export_log.clicked.connect(self._on_export_log_clicked)
        self.btn_apply_params.clicked.connect(self._apply_params_to_backend)
        self.btn_ofdm.clicked.connect(lambda: self._set_indices(0.0, 0.0))
        self.btn_otfs.clicked.connect(lambda: self._set_indices(1.0, 1.0))
        self.btn_reco.clicked.connect(lambda: self._set_indices(0.5, 1.0))
        self.btn_reset_hcache.clicked.connect(self._on_reset_hcache_clicked)
        self.btn_clear_ab_surface.clicked.connect(self._on_clear_ab_surface_clicked)
        self.btn_adaptive_evaluate.clicked.connect(self._on_adaptive_evaluate_clicked)
        self.adaptive_enable_check.stateChanged.connect(self._on_adaptive_enable_changed)
        for w in (self.gmi_receiver_combo, self.gmi_catalogue_combo, self.gmi_robustness_combo):
            w.currentIndexChanged.connect(self._on_adaptive_config_changed)
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

    def _toggle_advanced_controls(self, checked: bool):
        """Keep the operator path compact while exposing the full lab controls on demand."""
        checked = bool(checked)
        panel_layout = self.controls_panel.layout()
        advanced_index = panel_layout.indexOf(self.advanced_group)
        if checked and advanced_index < 0:
            # Insert immediately after the toggle checkbox.  Keeping this
            # widget out of the collapsed layout is important: Qt otherwise
            # includes a hidden group's child sizeHint in the scroll area's
            # preferred height.
            panel_layout.insertWidget(panel_layout.indexOf(self.btn_toggle_advanced) + 1, self.advanced_group)
        elif not checked and advanced_index >= 0:
            panel_layout.removeWidget(self.advanced_group)
        self.advanced_group.setVisible(checked)
        self.modem_group.setVisible(checked)
        self.text_group.setVisible(checked)
        self.samp_rate_spin.setVisible(bool(checked))
        self.channel_estimator_combo.setVisible(bool(checked))
        self._advanced_channel_group.setVisible(checked and self._selected_channel_mode() != "rf")
        self.btn_toggle_advanced.setText("收起高级参数" if checked else "显示高级参数")
        panel_layout.invalidate()
        panel_layout.activate()

    def _on_antenna_preset(self):
        if self._current_data(self.device_combo, "USRP B210") != "USRP B210":
            return
        self._suppress_param_signals = True
        try:
            self.tx_gain_spin.setValue(25.0)
            self.rx_gain_spin.setValue(45.0)
        finally:
            self._suppress_param_signals = False
        self._log("已应用 B210 A:TX/RX + A:RX2 天线/衰减器预设：TX=25 dB，RX=45 dB")
        self._schedule_param_apply(0)

    def _on_device_changed(self, *_args):
        is_b210 = self._current_data(self.device_combo, "USRP B210") == "USRP B210"
        self.btn_antenna_preset.setEnabled(is_b210)
        self.rf_port_note.setVisible(is_b210)

    def _sync_alpha_beta_controls_from_status(self, status):
        """Reflect backend alpha/beta changes without overwriting an unsaved edit."""
        if not isinstance(status, dict):
            return
        try:
            current = (float(status.get("alpha")), float(status.get("beta")))
        except (TypeError, ValueError):
            return
        if not all(np.isfinite(v) for v in current):
            return
        previous = self._last_backend_alpha_beta
        # A user edit is preserved while the backend still reports the same
        # pair.  Once the backend pair itself changes (for example after a
        # validated candidate or rollback), it is authoritative and must be
        # reflected in the controls.
        backend_changed = previous is None or any(
            abs(current[i] - previous[i]) > 1e-9 for i in (0, 1)
        )
        if backend_changed:
            with QSignalBlocker(self.alpha_spin), QSignalBlocker(self.beta_spin):
                self.alpha_spin.setValue(current[0])
                self.beta_spin.setValue(current[1])
        self._last_backend_alpha_beta = current

    def _toggle_advanced_controls_from_mode(self):
        self._toggle_advanced_controls(self.btn_toggle_advanced.isChecked())

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
            initial_status = self.backend.get_status()
            self._log_frame_structure_warning(initial_status)
            self.test_running = True
            self.btn_start_test.setEnabled(False)
            self.btn_stop_test.setEnabled(True)
            self._set_test_controls_enabled(False)
            self.update_timer.start(100)
            self._begin_observation(initial_status)
            self._log("v35 测试已启动。")
            self._log(self._backend_summary())
        except Exception as e:
            if self.backend is not None:
                try:
                    self.backend.stop()
                    if hasattr(self.backend, "wait"):
                        self.backend.wait()
                except Exception:
                    pass
            self.test_running = False
            self.btn_start_test.setEnabled(True)
            self.btn_stop_test.setEnabled(False)
            self.btn_connect.setEnabled(True)
            self._log(f"开始测试失败: {type(e).__name__}: {e}")

    def _on_stop_test_clicked(self, _checked=False, *, preserve_plots=False, strict=False):
        was_running = self.test_running
        self.test_running = False
        self.update_timer.stop()
        self._apply_debounce_timer.stop()
        self._pending_apply = False
        self._adaptive_toggle_pending = False
        if self.backend is not None:
            try:
                self.backend.stop()
                if hasattr(self.backend, "wait"):
                    self.backend.wait()
            except Exception as e:
                self._log(f"停止后端时出错: {e}")
                if strict:
                    self.test_running = was_running
                    if was_running:
                        self.update_timer.start(100)
                    raise
        self.btn_start_test.setEnabled(True)
        self.btn_stop_test.setEnabled(False)
        self.btn_connect.setEnabled(True)
        self._set_hw_controls_enabled(True)
        self._set_test_controls_enabled(True)
        if not preserve_plots:
            self._clear_plots()
        if self.observer.active:
            self.observer.stop()
        self._update_observation_display()
        self.decode_status_label.setText("解调状态：已停止")
        self._log("停止 FDIDM 测试。")

    def suspend_tasks(self):
        """Stop pending work and release UHD without erasing the last result."""
        if self._ai_analysis_dialog is not None:
            self._ai_analysis_dialog.suspend_analysis()
        self._apply_debounce_timer.stop()
        self._pending_apply = False
        self._adaptive_toggle_pending = False
        if self.backend is not None or self.test_running or self.observer.active:
            self._on_stop_test_clicked(preserve_plots=True, strict=True)
            self._get_ai_log_snapshot()
            self.backend = None
            self._last_backend_alpha_beta = None
            self.btn_start_test.setEnabled(False)
            self.decode_status_label.setText("解调状态：已停止并释放 USRP；结果已保留。")
            self._log("离开测试页面，已释放 USRP；再次测试请连接设备。")
            gc.collect()
        else:
            self.update_timer.stop()
        if self._ab_surface_window is not None:
            self._ab_surface_window.close()
            self._clear_ab_surface_window_refs()

    def closeEvent(self, event):
        try:
            self.suspend_tasks()
        except Exception:
            event.ignore()
            return
        super().closeEvent(event)

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
            self._log("尚未创建后端，无法评估 GMI 波形库。")
            return
        try:
            queued = bool(self.backend.request_gmi_evaluation()) if hasattr(self.backend, "request_gmi_evaluation") else False
            self._log("已提交 GMI 立即评估。" if queued else "尚无可信 CSI；将在下一有效帧评估有限波形库。")
        except Exception as e:
            self._log(f"GMI 立即评估失败: {type(e).__name__}: {e}")

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

    # ---------------- 自动图表记录与局部窗口统计 ----------------
    def _begin_observation(self, status=None) -> None:
        """Start the chart session automatically with the hardware test."""
        # A new test must never inherit the previous run's cumulative frame
        # counters. The start handler supplies its fresh post-start snapshot.
        snapshot = dict(status or {})
        self._last_status_snapshot = snapshot
        enabled = bool(self.adaptive_enable_check.isChecked())
        self.observer.start(enabled, snapshot)
        self._obs_t0 = self.observer.started_at()
        self._clear_timeline()
        self._log(
            f"硬件测试启动，已自动开始图表记录（自适应初始：{'开启' if enabled else '关闭'}）；"
            "在线切换时自动分段并以精确错误计数配对。"
        )
        self._update_observation_display(snapshot)

    def _clear_timeline(self):
        self._obs_t.clear(); self._obs_ser.clear(); self._obs_evm.clear()
        self._obs_ser_trend.clear(); self._obs_evm_trend.clear(); self._last_timeline_frame = -1
        self._obs_alpha.clear(); self._obs_beta.clear()
        # A new observation session starts with a clean effect view.  This is
        # the one deliberate place where an operator's previous zoom is reset;
        # subsequent live refreshes preserve any manual y-axis adjustment.
        self._timeline_effect_user_y_zoom = False
        self._timeline_effect_range_initialized = False
        self._timeline_apply_points = []
        self.timeline_apply_scatter.setData(x=[], y=[])
        for item in self._timeline_event_items:
            for plot in (self.timeline_effect_plot, self.timeline_ab_plot):
                try: plot.removeItem(item)
                except Exception: pass
        for item in self._timeline_anomaly_items:
            for plot in (self.timeline_effect_plot, self.timeline_evm_plot):
                try: plot.removeItem(item)
                except Exception: pass
        self._timeline_event_items = []
        self._timeline_anomaly_items = []
        self._set_effect_y_range(120.0, force=True)
        self._refresh_timeline_plot()

    def _purge_effect_plot_overlays(self):
        """Keep the effect chart strictly categorical and overlay-free.

        Older builds put the adaptive-toggle ``InfiniteLine`` on the first
        chart.  A running Qt process can retain that item until the widget is
        rebuilt, and a stale line is especially confusing once the chart no
        longer has a time axis.  The effect view is deliberately limited to
        its two bar items and their value labels; event lines and anomaly
        regions belong to the parameter/quality charts only.  Removing any
        unexpected item here also makes hot-reload and restored UI states safe.
        """
        plot_item = self.timeline_effect_plot.getPlotItem()
        allowed = [self.timeline_baseline_bars, self.timeline_candidate_bars]
        allowed.extend(getattr(self, "timeline_effect_labels", ()))
        legend = getattr(self, "timeline_effect_legend", None)
        if legend is not None:
            allowed.append(legend)
        for item in list(getattr(plot_item, "items", ())):
            if any(item is keep for keep in allowed):
                continue
            try:
                plot_item.removeItem(item)
            except Exception:
                # A third-party pyqtgraph item may already have detached while
                # a queued refresh is running; it is harmless to ignore it.
                pass

    def _on_effect_range_manually_changed(self, axes):
        """Remember a user y-axis zoom on the categorical effect chart.

        ``ViewBox.sigRangeChangedManually`` emits the x/y mouse-enabled mask
        (rather than a single axis index).  Only suppress automatic y-range
        updates when y was actually part of the gesture; an x-only pan should
        not prevent the chart from expanding to show a large future ratio.
        """
        if self._timeline_effect_setting_range:
            return
        try:
            y_changed = bool(axes[1])
        except Exception:
            # Older pyqtgraph releases emitted no mask.  Treat the gesture as
            # affecting both axes, which is the safe choice for preserving it.
            y_changed = True
        if y_changed:
            self._timeline_effect_user_y_zoom = True

    def _set_effect_y_range(self, ymax: float, *, force: bool = False):
        """Set a sensible effect-chart range without fighting manual zoom."""
        try:
            target = max(120.0, float(ymax))
        except (TypeError, ValueError):
            target = 120.0
        if self._timeline_effect_user_y_zoom and not force:
            return
        view = self.timeline_effect_plot.getPlotItem().getViewBox()
        current = view.viewRange()[1]
        # Do not issue a range change for every 100 ms frame.  Besides being
        # expensive, that visibly jitters the chart and erases a user's zoom.
        if (self._timeline_effect_range_initialized and len(current) == 2
                and abs(float(current[1]) - target) <= max(1.0, target * 0.02)):
            return
        self._timeline_effect_setting_range = True
        try:
            self.timeline_effect_plot.setYRange(0.0, target, padding=0)
            self._timeline_effect_range_initialized = True
        finally:
            self._timeline_effect_setting_range = False

    def _append_timeline_sample(self, status):
        frame = int(status.get("frames_processed", 0) or 0)
        if frame <= 0 or frame == self._last_timeline_frame:
            return
        self._last_timeline_frame = frame
        # 时间轴与观测会话共用引擎时钟，保证事件线/异常区与曲线同一时基
        now = self.observer.now()
        ser = float(status.get("measured_ser", np.nan))
        evm = float(status.get("data_aided_evm_percent",
                               status.get("evm_average_percent", status.get("evm_percent", np.nan))))
        if not bool(status.get("evm_valid", True)):
            evm = float("nan")
        self._obs_t.append(now - self._obs_t0)
        self._obs_ser.append(math.log10(ser) if np.isfinite(ser) and ser > 0 else np.nan)
        self._obs_evm.append(evm)
        self._obs_ser_trend.append(self._robust_trend(self._obs_ser))
        self._obs_evm_trend.append(self._robust_trend(self._obs_evm))
        self._obs_alpha.append(float(status.get("alpha", np.nan)))
        self._obs_beta.append(float(status.get("beta", np.nan)))
        self._refresh_timeline_plot()
        self._refresh_timeline_anomalies()

    def _refresh_timeline_plot(self):
        # Defensive cleanup for sessions created by an older UI build.  It is
        # intentionally first so no stale yellow/event overlay survives even
        # when there is not yet a valid local OFF→ON pair.
        self._purge_effect_plot_overlays()
        ts = list(self._obs_t)
        pair = self.observer.latest_pair()
        if pair is not None and pair.comparable:
            b, a = pair.before, pair.after
            def ratio(x, y, metric):
                """Return ON/OFF percentage and a reason when it is unavailable.

                EVM and SER are intentionally evaluated independently.  In a
                clean link the exact SER count is often zero in both windows;
                treating ``0/0`` as an invalid *whole pair* used to hide a
                perfectly valid EVM comparison.  For SER we define the
                equal-zero case as 100% (no error regression), while a zero
                baseline with non-zero candidate errors remains explicitly
                unnormalisable.
                """
                try:
                    x, y = float(x), float(y)
                except (TypeError, ValueError):
                    return float("nan"), "unavailable"
                if not (np.isfinite(x) and np.isfinite(y)) or y < 0.0:
                    return float("nan"), "unavailable"
                if metric == "SER" and x == 0.0:
                    if y == 0.0:
                        return 100.0, "both_zero"
                    return float("nan"), "baseline_zero"
                if x <= 0.0:
                    return float("nan"), "unavailable"
                value = y / x * 100.0
                return (value, "valid") if np.isfinite(value) and value >= 0.0 else (float("nan"), "unavailable")

            metric_values = [
                ratio(b.evm_mean, a.evm_mean, "EVM"),
                ratio(b.ser, a.ser, "SER"),
            ]
            rel = [value for value, _reason in metric_values]
            valid = [np.isfinite(value) and value >= 0.0 for value in rel]
            if any(valid):
                color = ((35, 150, 85, 225) if pair.outcome == "improved"
                         else ((190, 70, 70, 225) if pair.outcome == "regressed"
                               else (145, 145, 145, 210)))
                # Set both brush and pen on every refresh.  A scalar brush is
                # required for BarGraphItem to keep a recoloured candidate
                # filled (per-bar ``brushes`` arrays otherwise retain stale
                # grey interiors on some pyqtgraph versions).
                self.timeline_baseline_bars.setOpts(
                    height=[100.0 if ok else 0.0 for ok in valid],
                    brush=pg.mkBrush(150, 150, 150, 205),
                    pen=pg.mkPen((90, 90, 90), width=1))
                self.timeline_candidate_bars.setOpts(
                    # Always provide one scalar brush and pen.  This keeps
                    # every visible candidate bar filled, including when the
                    # other metric is unavailable (rather than retaining a
                    # stale grey interior from a previous refresh).
                    height=[float(value) if ok else 0.0 for value, ok in zip(rel, valid)],
                    brush=pg.mkBrush(color),
                    pen=pg.mkPen(color[:3], width=1))
                for i, (item, (value, reason), ok) in enumerate(
                        zip(self.timeline_effect_labels, metric_values, valid)):
                    metric = "EVM" if i == 0 else "SER"
                    if not ok:
                        if reason == "baseline_zero":
                            text = f"{metric}: 基线为0，无法归一化"
                        else:
                            text = f"{metric}: 数据不可用"
                        # Keep an explicit reason visible below the 100% line;
                        # hiding the label made a valid EVM result look like a
                        # missing comparison whenever SER was exactly zero.
                        item.setText(text)
                        item.setPos(i + 0.19, 14.0)
                        item.setVisible(True)
                        continue
                    change = 100.0 - float(value)
                    if reason == "both_zero":
                        change_text = "双方0错误"
                    elif change > 0.05:
                        change_text = f"↓{change:.1f}%"
                    elif change < -0.05:
                        change_text = f"↑{abs(change):.1f}%"
                    else:
                        change_text = "≈0%"
                    # One compact label per metric states the complete
                    # comparison; the category ticks below identify the two
                    # metric groups.  Keep labels centred over the candidate
                    # bar so they remain readable at narrow window widths.
                    item.setText(f"{metric}: 100 → {value:.1f}% ({change_text})")
                    item.setPos(i + 0.19, max(100.0, value) + 4.0)
                    item.setVisible(True)
                valid_values = [float(value) for value, ok in zip(rel, valid) if ok]
                self._set_effect_y_range(max([100.0] + valid_values) * 1.18)
            else:
                # Neither metric can be normalised.  Keep an explicit status
                # label for each one (especially SER baseline=0) while leaving
                # the bars at zero; silently hiding the labels makes the chart
                # look as if the refresh failed.
                self.timeline_baseline_bars.setOpts(
                    height=[0.0, 0.0],
                    brush=pg.mkBrush(150, 150, 150, 205),
                    pen=pg.mkPen((90, 90, 90), width=1))
                self.timeline_candidate_bars.setOpts(
                    height=[0.0, 0.0],
                    brush=pg.mkBrush(145, 145, 145, 180),
                    pen=pg.mkPen((105, 105, 105), width=1))
                for i, (item, (_value, reason)) in enumerate(
                        zip(self.timeline_effect_labels, metric_values)):
                    metric = "EVM" if i == 0 else "SER"
                    text = (f"{metric}: 基线为0，无法归一化"
                            if reason == "baseline_zero"
                            else f"{metric}: 数据不可用")
                    item.setText(text)
                    item.setPos(i + 0.19, 14.0)
                    item.setVisible(True)
                self._set_effect_y_range(120.0)
        elif ts:
            # A changing RF channel makes a session-start sample an invalid
            # reference.  Wait for an adjacent local OFF window and do not
            # leave the previous pair's values on screen.
            self.timeline_baseline_bars.setOpts(height=[0.0, 0.0])
            self.timeline_candidate_bars.setOpts(
                height=[0.0, 0.0],
                brush=pg.mkBrush(145, 145, 145, 180),
                pen=pg.mkPen((105, 105, 105), width=1))
            for item in self.timeline_effect_labels:
                item.setVisible(False)
            self._set_effect_y_range(120.0)
        else:
            self.timeline_baseline_bars.setOpts(height=[0.0, 0.0])
            self.timeline_candidate_bars.setOpts(height=[0.0, 0.0])
            for item in self.timeline_effect_labels:
                item.setVisible(False)
        self.timeline_ser_curve.setData(ts, list(self._obs_ser_trend))
        self.timeline_evm_curve.setData(ts, list(self._obs_evm_trend))
        self.timeline_alpha_curve.setData(ts, list(self._obs_alpha), stepMode="left")
        self.timeline_beta_curve.setData(ts, list(self._obs_beta), stepMode="left")

    @staticmethod
    def _robust_trend(values, window=5):
        vals = list(values)
        if not vals or not np.isfinite(vals[-1]):
            return float("nan")
        recent = np.asarray(vals[-int(window):], dtype=float)
        recent = recent[np.isfinite(recent)]
        return float(np.median(recent)) if recent.size else float("nan")

    def _refresh_timeline_anomalies(self):
        drops = self.observer.anomaly_drops()
        if len(drops) == getattr(self, "_timeline_anomaly_count", -1):
            return
        self._timeline_anomaly_count = len(drops)
        for item in self._timeline_anomaly_items:
            for plot in (self.timeline_evm_plot, self.timeline_effect_plot):
                try: plot.removeItem(item)
                except Exception: pass
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
            self.timeline_evm_plot.addItem(region)
            self._timeline_anomaly_items.append(region)

    def _add_timeline_event_line(self, enabled, t_abs):
        if not self._obs_t:
            return
        line = pg.InfiniteLine(pos=t_abs - self._obs_t0, angle=90,
                               pen=pg.mkPen(MATLAB_BLUE if enabled else MATLAB_ORANGE,
                                            style=Qt.DashLine, width=2))
        self.timeline_ab_plot.addItem(line)
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
        gmi = dict(status.get("gmi_adaptation", {}) or {})
        if not gmi:
            self._update_legacy_adaptive_state_row(status)
            return
        receiver = str(gmi.get("receiver_mode", status.get("gmi_receiver_mode", "diagonal")))
        current_token = str(gmi.get("current_token_id", ""))
        candidate_token = str(gmi.get("candidate_token_id", ""))
        gain = float(gmi.get("gain_lower_nats", 0.0))
        threshold = float(gmi.get("threshold_nats", 0.0))
        search = str(gmi.get("state", "—"))
        validation = dict(status.get("adaptive_validation", {}) or {})
        valid = str(validation.get("state") or status.get("adaptive_validation_state") or "—")
        search_name = {
            "disabled": "关闭", "idle": "等待", "queued": "等待评分", "scoring": "GMI 评分中",
            "waiting_channel": "等待可信 CSI", "candidate_ready": "候选已就绪",
            "tracking": "保持当前 token", "error": "评分异常", "stopped": "已停止",
        }.get(search, search)
        valid_name = {
            "idle": "未验证", "prepare_contract": "准备公共功率合同",
            "a1_applying": "应用 A1", "a1_settling": "A1 重捕获", "a1_collecting": "采集 A1",
            "b1_applying": "应用 B1", "b1_settling": "B1 重捕获", "b1_collecting": "采集 B1",
            "b2_collecting": "采集 B2", "a2_applying": "应用 A2", "a2_settling": "A2 重捕获",
            "a2_collecting": "采集 A2", "commit_applying": "提交候选",
            "rollback_applying": "回滚基线",
            "improved": "已确认改善", "regressed": "已确认退化",
            "inconclusive": "暂不可归因", "rollback_failed": "回滚失败",
            "aborted": "验证中断",
        }.get(valid, valid)
        stage = valid_name if valid not in {"", "idle", "disabled"} else search_name
        receiver_name = "对角 K=0" if receiver == "diagonal" else "完整匹配"
        txt = (
            f"GMI：{receiver_name} · token {current_token[-6:] if current_token else '—'} · {stage} · "
            f"净增益下界 {gain:.3g}"
        )
        reason = str(validation.get("result_reason") or status.get("adaptive_validation_reason") or gmi.get("reason", ""))
        detail = (
            f"receiver={receiver}；current token={current_token or '—'}；candidate token={candidate_token or '—'}；"
            f"搜索={search_name}（{search}）；验证={valid_name}（{valid}）；"
            f"gain_lower={gain:.6g}；threshold={threshold:.6g}"
        )
        if reason:
            detail += f"；验证说明：{reason}"
        self._set_adaptive_state_display(txt, detail, valid, reason)

    def _update_legacy_adaptive_state_row(self, status):
        """Render historical SER-search status without granting it control.

        Old log fixtures and fixed-mode backends still expose the legacy
        fields.  Keeping them readable is useful for replay, while live GMI
        operation always takes the branch above.
        """
        a = float(status.get("alpha", np.nan))
        b = float(status.get("beta", np.nan))
        search = str(status.get("adaptive_alpha_beta_state", "—"))
        validation = dict(status.get("adaptive_validation", {}) or {})
        valid = str(validation.get("state") or status.get("adaptive_validation_state") or "—")
        alpha_obs = status.get("adaptive_alpha_observable")
        beta_obs = status.get("adaptive_beta_observable")

        def yn(value):
            return "—" if value is None else ("✓" if bool(value) else "×")

        search_name = {
            "disabled": "关闭", "idle": "等待有效信道", "monitoring": "监测中",
            "queued": "等待评估", "optimizing": "搜索中", "waiting_channel": "等待可信信道",
            "ready": "候选已就绪", "tracking": "持续跟踪", "cooldown": "冷却中", "error": "异常",
        }.get(search, search)
        valid_name = {
            "idle": "未验证", "prepare_contract": "准备对比", "baseline_applying": "切换 OFF 基线",
            "baseline_settling": "OFF 基线稳定", "baseline_collecting": "采集 OFF 基线",
            "candidate_applying": "切换 ON 候选", "candidate_settling": "ON 候选稳定",
            "candidate_collecting": "采集 ON 候选", "commit_applying": "提交候选",
            "rollback_applying": "回滚基线", "improved": "已确认改善", "regressed": "已确认退化",
            "inconclusive": "暂不可归因", "rollback_failed": "回滚失败", "aborted": "验证中断",
        }.get(valid, valid)
        index_text = "—" if not (np.isfinite(a) and np.isfinite(b)) else f"{a:.2f}/{b:.2f}"
        stage = valid_name if valid not in {"", "idle", "disabled"} else search_name
        txt = f"自适应：α/β={index_text} · {stage} · 可观测 α{yn(alpha_obs)} β{yn(beta_obs)}"
        reason = str(validation.get("result_reason") or status.get("adaptive_validation_reason") or "")
        detail = (
            f"legacy_ser；当前 α/β={index_text}；搜索状态={search_name}（{search}）；"
            f"验证状态={valid_name}（{valid}）；α可观测={yn(alpha_obs)}；β可观测={yn(beta_obs)}"
        )
        if reason:
            detail += f"；验证说明：{reason}"
        self._set_adaptive_state_display(txt, detail, valid, reason)

    def _set_adaptive_state_display(self, text, detail, validation_state, reason):
        """Keep validation warnings visible after removal of the result cards."""
        warning = {
            "rollback_failed": "回滚失败：请立即停止测试",
            "aborted": "验证中断：未确认回滚",
        }.get(validation_state, "")
        color = "#555555"
        if warning:
            text = f"{warning} · {text}"
            color = "#a33b3b"
        elif validation_state == "regressed":
            color = "#a33b3b"
        elif validation_state == "inconclusive" or validation_state.endswith(("_applying", "_settling", "_collecting")):
            color = "#b06a00"
        if reason and validation_state not in {"", "—", "idle", "disabled"}:
            compact_reason = reason if len(reason) <= 96 else f"{reason[:95]}…"
            text += f"\n原因：{compact_reason}"
        self.adaptive_state_label.setText(text)
        self.adaptive_state_label.setToolTip(detail)
        weight = "font-weight:600;" if warning else ""
        self.adaptive_state_label.setStyleSheet(f"color:{color}; font-size:12px; {weight}")

    def _update_observation_display(self, status=None):
        """Refresh chart evidence and safety state from one shared snapshot."""
        if status is not None:
            self._last_status_snapshot = dict(status)
        snapshot = self._last_status_snapshot
        self._update_adaptive_state_row(snapshot)
        validation = dict(snapshot.get("adaptive_validation", {}) or {})
        state = snapshot.get("adaptive_validation_state")
        if state not in {None, "", "—", "idle", "disabled"}:
            validation.setdefault("state", state)
        reason = snapshot.get("adaptive_validation_reason")
        if reason:
            validation.setdefault("result_reason", reason)
        self._update_observation_stats(validation)

    def _update_observation_stats(self, validation=None):
        """观测页签右下角统计表：最近窗口的汇总数字。"""
        validation = dict(validation or {})

        def finite_float_for_stats(value):
            try:
                number = float(value)
            except (TypeError, ValueError):
                return np.nan
            return number if np.isfinite(number) else np.nan

        name_map = {MODE_ON: "开启后", MODE_OFF: "开启前"}
        wins = self.observer.windows()
        if not wins and not validation:
            self.observation_stats_label.setText("硬件测试启动后自动记录各窗口汇总")
            self.observation_stats_label.setToolTip(
                "动态信道采用相邻 OFF→ON 局部基线；raw/FEC BER 与功率合同将在采样后显示")
            return
        state = "记录中" if self.observer.active else "已停止（保留结果）"
        lines = [
            f"图表{state}｜窗口 {len(wins)}｜开关事件 {len(self.observer.events())}｜"
            f"剔除 {self.observer.excluded_window_count()} 个异常窗口",
            "固定基线：否（动态信道采用相邻 OFF→ON 局部基线）",
        ]
        detail_lines = ["raw/FEC BER 与功率合同明细："]
        for w in wins[-4:]:
            agg = w.aggregate
            label = name_map.get(w.mode, w.mode)
            if agg is None:
                lines.append(f"{label}  已剔除（{w.exclusion_reason}）")
            else:
                ctx = dict(w.context or {})
                def ctx_num(key, pattern):
                    try:
                        value = float(ctx.get(key, np.nan))
                    except (TypeError, ValueError):
                        value = np.nan
                    return "不可用" if not np.isfinite(value) else format(value, pattern)
                lines.append(
                    f"{label}｜{agg.frames}帧｜SER {agg.ser:.3g} "
                    f"k/n={agg.ser_k:.0f}/{agg.ser_n}（{'估算' if agg.estimated_k else '精确'}）\n"
                    f"95%区间 [{agg.wilson_low:.3g}, {agg.wilson_high:.3g}]｜"
                    f"数据EVM {agg.evm_mean:.3g}%｜判决EVM {agg.decision_evm_mean:.3g}%｜"
                    f"CRC {100.0 * agg.crc_ok_ratio:.1f}%")
                detail_lines.append(
                    f"{label}: raw/FEC BER={agg.raw_ber:.3g}/{agg.fec_ber:.3g}; "
                    f"contract={agg.tx_power_contract_id or '不可用'}; "
                    f"周期/数据RMS={agg.tx_cycle_rms_mean:.4g}/{ctx_num('tx_data_rms', '.4g')}; "
                    f"peak={ctx_num('tx_peak', '.4g')}; PAPR={ctx_num('tx_papr_db', '.2f')} dB; "
                    f"回退={ctx_num('tx_power_backoff_db', '.2f')} dB")

        if validation:
            abba = dict(validation.get("abba_evidence", {}) or {})
            baseline = dict(validation.get("baseline_window") or abba.get("baseline") or {})
            candidate = dict(validation.get("candidate_window") or abba.get("candidate") or {})

            def window_line(label, window):
                if not window:
                    return f"{label}｜等待采集"
                k = finite_float_for_stats(window.get("ser_errors"))
                n = finite_float_for_stats(window.get("ser_symbols"))
                count = ("—/—" if not (np.isfinite(k) and np.isfinite(n))
                         else f"{k:.0f}/{n:.0f}")
                ser = finite_float_for_stats(window.get("ser"))
                evm = finite_float_for_stats(
                    window.get("data_aided_evm_mean", window.get("evm_mean")))
                frames = finite_float_for_stats(window.get("valid_frames"))
                return (
                    f"{label}｜{('—' if not np.isfinite(frames) else f'{frames:.0f}')}帧｜"
                    f"SER {('—' if not np.isfinite(ser) else f'{ser:.3g}')}｜k/n={count}｜"
                    f"数据EVM {('—' if not np.isfinite(evm) else f'{evm:.2f}%')}")

            lines.extend((window_line("验证 OFF", baseline), window_line("验证 ON", candidate)))
            reason = str(validation.get("result_reason", "") or "")
            if reason:
                lines.append(f"验证说明｜{reason}")
            contract_id = dict(validation.get("contract", {}) or {}).get("contract_id", "不可用")
            detail_lines.append(f"后端真实 A/B 窗口: raw/FEC BER=按帧累计; contract={contract_id}")

        pair = self.observer.latest_pair()
        if pair is not None:
            gain = "—" if not np.isfinite(pair.ser_improvement_db) else f"{pair.ser_improvement_db:+.2f} dB"
            lines.append(
                f"结论 {pair.outcome}｜SER变化 {gain}｜"
                f"{'功率/上下文可比' if pair.comparable else pair.comparability_note}｜"
                f"剔除窗口 {self.observer.excluded_window_count()}")
        self.observation_stats_label.setText("\n".join(lines))
        self.observation_stats_label.setToolTip("\n".join(lines + detail_lines))

    def _on_adaptive_enable_changed(self, state):
        enabled = bool(state)
        if self.backend is not None and self.test_running:
            self._adaptive_toggle_pending = True
            event = self.observer.on_toggle(enabled, self.alpha_spin.value(), self.beta_spin.value())
            if event is not None:
                self._add_timeline_event_line(enabled, event.t)
                self._log(f"手动{'开启' if enabled else '关闭'} α/β 自适应，观测窗口已切换。")
            elif self.observer.active:
                self._log(f"自适应已保持{'开启' if enabled else '关闭'}；状态未变化，未新建观测窗口。")
            else:
                self._log(
                    f"已{'开启' if enabled else '关闭'} α/β 自适应；当前观测已停止，"
                    "重新启动硬件测试后会自动记录新的图表会话。")
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
        self._ai_backend_log_snapshot = ""

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
            adaptive_alpha_beta_enable=False,
            gmi_adaptive_enable=self.adaptive_enable_check.isChecked(),
            gmi_receiver_mode=self._current_data(self.gmi_receiver_combo, "diagonal"),
            gmi_catalogue_cap=int(self._current_data(self.gmi_catalogue_combo, 16)),
            gmi_robustness_profile=self._current_data(self.gmi_robustness_combo, "standard"),
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
            try:
                self.backend.stop()
                if hasattr(self.backend, "wait"):
                    self.backend.wait()
            except Exception:
                pass
            self.btn_start_test.setEnabled(True)
            self.btn_stop_test.setEnabled(False)
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
        # Receiver/library/profile are control-plane settings.  Rebuilding the
        # finite library invalidates stale scores without touching RF gains.
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

    def _update_channel_param_visibility(self):
        """Show TDL parameters only for a link mode that actually uses TDL."""
        mode = self._selected_channel_mode()
        has_tdl = mode != "rf"
        self._advanced_channel_group.setVisible(has_tdl and self.btn_toggle_advanced.isChecked())
        for widget in (
            self.tdl_ds_label, self.tdl_ds_spin,
            self.tdl_fd_label, self.tdl_fd_spin,
            self.tdl_spread_label, self.tdl_spread_spin,
            self.tdl_snr_label, self.tdl_snr_spin,
        ):
            widget.setVisible(has_tdl)

        # The parametric estimator is the meaningful choice for the software
        # TDL leg.  RF-only keeps the fast diagonal estimator as its default.
        target = "tdl_param" if has_tdl else "diag_tf"
        blocker = QSignalBlocker(self.channel_estimator_combo)
        try:
            for i in range(self.channel_estimator_combo.count()):
                if self.channel_estimator_combo.itemData(i) == target:
                    self.channel_estimator_combo.setCurrentIndex(i)
                    break
        finally:
            del blocker

    def _on_channel_mode_changed(self, *_args):
        # Keep the controls and the estimator choice synchronized with the
        # selected RF/TDL path.  Block the estimator signal so one click does
        # not produce two stop/config/start cycles.
        self._update_channel_param_visibility()
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
            self._sync_alpha_beta_controls_from_status(status)
            self.observer.on_sample(status)
            if self.observer.active:
                self._append_timeline_sample(status)
            self._update_observation_display(status)
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
            self.rx_spectrum_plot.setTitle(f"RX 频谱 · {'等待新帧' if stale else '实时'}")
            self.rx_spectrum_plot.setToolTip(
                f"RX 频谱源：{self._selected_rx_spectrum_source()}；stale={stale}；距最近样本 {age:.1f}s")
            self._update_evm_plot(float(status.get("evm_percent", np.nan)))
            self.evm_plot.setTitle("EVM 曲线")
            const = self.backend.get_rx_constellation(512, source=self._current_data(self.const_mode_combo, "post_equalized"))
            if const is not None and len(const) > 0:
                self.constellation_scatter.setData(x=np.real(const), y=np.imag(const))
            else:
                self.constellation_scatter.setData(x=[], y=[])
            self.constellation_plot.setTitle(f"接收星座 · {status.get('constellation_points',0)} 点")
            self.constellation_plot.setToolTip(
                f"显示方式：{self.const_mode_combo.currentText()}；"
                f"数据源：{status.get('constellation_source','none')}；"
                f"原始模式：{self._current_data(self.const_mode_combo, 'post_equalized')}")
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
        st = self.backend.get_status() if status is None else status
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
        gmi = dict(status.get("gmi_adaptation", {}) or {})
        if not gmi:
            self._handle_legacy_adaptation_diagnostic(status, label)
            return
        enabled = bool(gmi.get("enabled", status.get("gmi_adaptive_enabled", False)))
        state = str(gmi.get("state", "disabled"))
        receiver = str(gmi.get("receiver_mode", "diagonal"))
        current_token = str(gmi.get("current_token_id", ""))
        candidate_token = str(gmi.get("candidate_token_id", ""))
        rec_a = float(gmi.get("candidate_alpha", np.nan))
        rec_b = float(gmi.get("candidate_beta", np.nan))
        gain = float(gmi.get("gain_lower_nats", 0.0))
        threshold = float(gmi.get("threshold_nats", 0.0))
        reason = str(gmi.get("reason", "") or "")
        ready = bool(gmi.get("switch_allowed", False))
        seq = int(gmi.get("sequence", 0))
        validation = dict(status.get("adaptive_validation", {}) or {})
        validation_state = str(validation.get("state") or status.get("adaptive_validation_state") or "")

        if label is not None:
            if not enabled:
                label.setText("GMI 自适应：关闭")
            elif receiver == "full_matched":
                label.setText("完整匹配接收机：GMI 对 α/β 不敏感，无需切换")
            elif validation_state in {"a1_collecting", "b1_collecting", "b2_collecting", "a2_collecting"}:
                label.setText(f"A-B-B-A 真链路验证：{validation_state[:2].upper()} 采集中")
            elif validation_state == "rollback_applying":
                label.setText("A-B-B-A 验证未通过：正在安全回滚")
            elif validation_state == "rollback_failed":
                label.setText("安全回滚失败：请立即停止测试")
            else:
                state_name = {
                    "idle": "等待", "queued": "排队", "scoring": "评分中",
                    "waiting_channel": "等待可信 CSI", "tracking": "保持当前 token",
                    "candidate_ready": "候选可验证", "error": "评分异常",
                }.get(state, state)
                token_text = current_token[-6:] if current_token else "—"
                candidate_text = candidate_token[-6:] if candidate_token else "—"
                label.setText(
                    f"GMI：{state_name} · 当前 {token_text} · 候选 {candidate_text} · "
                    f"保守增益 {gain:.3g} nats/sample"
                )
            label.setToolTip(
                f"receiver={receiver}；current={current_token or '—'}；candidate={candidate_token or '—'}；"
                f"gain_lower={gain:.6g}；threshold={threshold:.6g}；reason={reason or '—'}"
            )

        if validation_state in {
            "a1_applying", "a1_settling", "a1_collecting", "b1_applying", "b1_settling",
            "b1_collecting", "b2_collecting", "a2_applying", "a2_settling", "a2_collecting",
            "commit_applying", "rollback_applying", "rollback_failed", "aborted",
        }:
            return
        if (not enabled or not ready or seq <= int(self._last_adaptive_recommendation_seq)):
            return
        self._last_adaptive_recommendation_seq = seq
        self._mark_observation_apply(rec_a, rec_b)
        if not (np.isfinite(rec_a) and np.isfinite(rec_b)):
            return
        self._log(
            f"GMI 候选进入 A-B-B-A 验证：α={rec_a:.2f}, β={rec_b:.2f}；"
            f"保守增益下界 {gain:.4g} nats/sample。"
        )
        if hasattr(self.backend, "apply_gmi_candidate"):
            self.backend.apply_gmi_candidate()
        else:
            self._set_indices(rec_a, rec_b)

    def _handle_legacy_adaptation_diagnostic(self, status, label):
        """Display old SER recommendations as diagnostics only.

        Legacy status is intentionally non-executable here: only a versioned
        ``gmi_adaptation`` decision can start a new waveform transaction.
        This preserves historical log readability without letting the old SER
        objective regain control of the transmitter.
        """
        enabled = bool(status.get("adaptive_alpha_beta_enabled", False))
        state = str(status.get("adaptive_alpha_beta_state", "disabled"))
        error = str(status.get("adaptive_last_error", "") or "")
        rec_a = float(status.get("adaptive_recommended_alpha", np.nan))
        rec_b = float(status.get("adaptive_recommended_beta", np.nan))
        gain = float(status.get("adaptive_predicted_improvement_db", np.nan))
        snr = float(status.get("adaptive_predicted_snr_db", np.nan))
        predicted_current = float(status.get("adaptive_predicted_ser_current", np.nan))
        predicted_best = float(status.get("adaptive_predicted_ser_best", np.nan))
        stable = int(status.get("adaptive_stable_count", 0))
        required = int(status.get("adaptive_stable_required", 0))
        source = str(status.get("adaptive_htf_source", ""))
        validation_state = str(status.get("adaptive_validation_state", "") or "")
        if label is None:
            return
        if not enabled:
            label.setText("自适应：关闭（历史 SER 诊断）")
        elif validation_state == "baseline_collecting":
            label.setText("真实链路验证：采集关闭基线；暂不应用新候选")
        elif validation_state == "candidate_collecting":
            label.setText("真实链路验证：采集候选窗口；暂不应用新候选")
        elif validation_state == "rollback_applying":
            label.setText("真实链路验证：正在回滚基线；暂不应用新候选")
        elif validation_state == "rollback_failed":
            label.setText("真实链路验证：回滚失败；请立即停止测试")
        elif error:
            state_name = {
                "idle": "等待有效信道", "monitoring": "监测中", "optimizing": "搜索中",
                "waiting_channel": "等待可信信道估计",
            }.get(state, state)
            label.setText(f"历史 SER 诊断：{state_name}；{error}")
        elif np.isfinite(rec_a) and np.isfinite(rec_b):
            gain_text = "nan" if not np.isfinite(gain) else f"{gain:.2f}dB"
            ser_text = (
                "不可用" if not (np.isfinite(predicted_current) and np.isfinite(predicted_best))
                else f"{predicted_current:.3g}→{predicted_best:.3g}"
            )
            state_name = {
                "idle": "等待有效信道", "monitoring": "监测中", "optimizing": "搜索中",
                "waiting_channel": "等待可信信道估计",
            }.get(state, state)
            label.setText(
                f"历史 SER 诊断：{state_name}；推荐 α/β={rec_a:.2f}/{rec_b:.2f}；"
                f"预测SER={ser_text}；预计改善 {gain_text}；稳定={stable}/{required}；CSI={source}"
            )
        else:
            state_name = {
                "idle": "等待有效信道", "monitoring": "监测中", "optimizing": "搜索中",
                "waiting_channel": "等待可信信道估计",
            }.get(state, state)
            label.setText(f"历史 SER 诊断：{state_name}，等待可信信道估计；检查 CRC")
        label.setToolTip(
            f"legacy_ser（仅诊断，不触发波形切换）；预测SER={predicted_current:.3g}→{predicted_best:.3g}；"
            f"预测SNR={snr:.1f}dB"
        )

    def _update_decode_status(self, stats, status):
        ok = bool(stats.get("decode_ok", False))
        def readable(value, pattern):
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = np.nan
            return "不可用" if not np.isfinite(value) else format(value, pattern)

        data_evm = readable(status.get("data_aided_evm_percent"), ".2f")
        decision_evm = readable(status.get("decision_directed_evm_percent",
                                           status.get("evm_average_percent")), ".2f")
        tx_cycle_rms = readable(status.get("tx_cycle_rms"), ".4g")
        tx_peak = readable(status.get("tx_peak"), ".4g")
        tx_papr = readable(status.get("tx_papr_db"), ".2f")
        tx_backoff = readable(status.get("tx_power_backoff_db"), ".2f")
        residual_sinr = readable(status.get("pilot_residual_sinr_db"), ".2f")
        pilot_nmse = readable(status.get("pilot_fit_nmse"), ".2e")
        injected_snr = readable(status.get("tdl_injected_snr_db", status.get("tdl_snr_db")), ".1f")
        frac_delay = readable(status.get("frac_delay_fractional"), "+.3f")
        frac_corr = readable(status.get("frac_delay_magnitude"), ".3f")
        frac_applied = "已补偿" if bool(status.get("frac_delay_applied", False)) else "未补偿"
        analog_bw = readable(status.get("analog_bandwidth_hz"), ".4g")
        frame = dict(status.get("frame_structure", {}) or {})
        frame_text = "不可用"
        useful_text = "不可用"
        if frame:
            frame_text = (f"{int(frame.get('data_samples', 0) or 0)}/"
                          f"{int(frame.get('pilot_samples', 0) or 0)}/"
                          f"{int(frame.get('guard_samples', 0) or 0)}")
            useful_text = readable(
                100.0 * float(frame.get("useful_data_ratio"))
                if frame.get("useful_data_ratio") is not None else None, ".1f")
        ser_k = int(status.get("ser_errors_total", 0) or 0)
        ser_n = int(status.get("ser_symbols_total", 0) or 0)
        rx_gap = int(status.get("rx_processing_gap_count", 0) or 0)
        rx_ovf = int(status.get("rx_overflow_count", 0) or 0)
        detail = (
            f"{'CRC通过' if ok else '未恢复'} | frames={int(status.get('frames_decode_ok',0))}/{int(status.get('frames_processed',0))}, "
            f"Sync={float(status.get('sync_metric',0.0)):.3f}, CFO={float(status.get('cfo_est_hz',0.0)):.1f}Hz/"
            f"{status.get('cfo_source','')}, raw={float(status.get('cfo_preamble_hz',0.0)):.1f}, "
            f"alias={float(status.get('cfo_alias_hz',np.nan)):.1f}, scan={float(status.get('cfo_scan_score',np.nan)):.2f}, "
            f"BER(FEC)={float(status.get('fec_bit_ber', status.get('ber',np.nan))):.3g}, "
            f"raw={float(status.get('raw_bit_ber',np.nan)):.3g}, "
            f"SER={float(status.get('measured_ser',np.nan)):.3g} (k/n={ser_k}/{ser_n}), "
            f"数据EVM={data_evm}%, 判决EVM={decision_evm}%, "
            f"TX RMS={tx_cycle_rms}, peak={tx_peak}, PAPR={tx_papr}dB, backoff={tx_backoff}dB, "
            f"残差SINR={residual_sinr}dB, fitNMSE={pilot_nmse}（含噪声/残余频偏/RF失真/模型失配）, "
            f"数据/导频/保护={frame_text}, 有效占比={useful_text}%, cond={float(status.get('cond_h_cross',np.nan)):.2e}, "
            f"mode={status.get('channel_estimator','')}, ch={status.get('channel_mode','')}, code={status.get('coding_scheme','')}, "
            f"TDL注入设定SNR={injected_snr}dB, "
            f"const={status.get('constellation_source','none')}, "
            f"计时残差={frac_delay}采样/{frac_applied}(相关={frac_corr}), 模拟带宽={analog_bw}Hz, "
            f"ABauto={status.get('adaptive_alpha_beta_state','off')}, "
            f"RX断流排除={rx_gap}窗口, RX溢出排除={rx_ovf}窗口（控制台 O 标记在此计数）"
        )
        if not bool(status.get("preamble_reliable", True)):
            detail = "未检测到可靠 FDIDM 前导；" + detail
        if not bool(status.get("evm_valid", True)):
            detail = detail.replace(f"数据EVM={data_evm}%", "数据EVM=不可用%")
        self.decode_status_label.setText(detail)
        self.decode_status_label.setToolTip(
            detail
            + "\n\nUHD 控制台的 \"O\" / \"overflows occurred\" 指接收流送器出现断样。"
              "这些窗口已被排除出 CRC、EVM 均值与自适应 A/B 证据，"
              "不会把坏窗口算成一次解调或一次改善。"
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
            f"dataEVM={float(status.get('data_aided_evm_percent',np.nan)):.2f}%, "
            f"decisionEVM={float(status.get('decision_directed_evm_percent',status.get('evm_average_percent',np.nan))):.2f}%, "
            f"SERk/n={int(status.get('ser_errors_total',0))}/{int(status.get('ser_symbols_total',0))}, "
            f"mode={status.get('channel_estimator','')}, ch={status.get('channel_mode','')}, fd={float(status.get('tdl_doppler_hz',0.0)):.1f}, "
            f"spread={float(status.get('tdl_doppler_spread_hz',0.0)):.1f}, "
            f"TDL_injected_SNR={float(status.get('tdl_injected_snr_db',status.get('tdl_snr_db',np.nan))):.1f}dB, "
            f"pilot_residual_SINR={float(status.get('pilot_residual_sinr_db',np.nan)):.2f}dB, "
            f"pilot_fit_NMSE={float(status.get('pilot_fit_nmse',np.nan)):.2e}, "
            f"TXrms={float(status.get('tx_cycle_rms',np.nan)):.4g}, peak={float(status.get('tx_peak',np.nan)):.4g}, "
            f"PAPR={float(status.get('tx_papr_db',np.nan)):.2f}dB, backoff={float(status.get('tx_power_backoff_db',np.nan)):.2f}dB, "
            f"code={status.get('coding_scheme','')}, txvec={int(status.get('tx_waveform_samples',0))}, "
            f"prerender={bool(status.get('tx_tdl_prerendered',False))}, decode_ok={bool(stats.get('decode_ok',False))}, "
            f"ABauto={status.get('adaptive_alpha_beta_state','off')}, "
            f"ABrec={float(status.get('adaptive_recommended_alpha',np.nan)):.2f}/"
            f"{float(status.get('adaptive_recommended_beta',np.nan)):.2f}, "
            f"ABgain={float(status.get('adaptive_predicted_improvement_db',np.nan)):.2f}dB"
            f", RXgap={int(status.get('rx_processing_gap_count',0))}"
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
        frame = dict(st.get("frame_structure", {}) or {})
        ratio = frame.get("training_data_ratio")
        useful = frame.get("useful_data_ratio")
        ratio_text = "—" if ratio is None else f"{float(ratio):.2f}"
        useful_text = "—" if useful is None else f"{100.0 * float(useful):.1f}%"
        return (
            f"链路={st.get('chain')}, mode={st.get('channel_estimator')}, ch={st.get('channel_mode')}, "
            f"MxN={st.get('fdidm_m')}x{st.get('fdidm_n')}, CP={st.get('cp_len')}, "
            f"训练块={st.get('htf_training_blocks')}, Fs={float(st.get('samp_rate', np.nan)):.0f}Hz, "
            f"调制={st.get('mod_order')}, EQ={st.get('equalizer')}, 编码={st.get('coding_summary')}, "
            f"α/β={st.get('alpha'):.2f}/{st.get('beta'):.2f}, frame={st.get('frame_len')} samples, "
            f"TX向量={st.get('tx_waveform_samples')} samples, UHD帧={st.get('usrp_buffer_frames')}, "
            f"数据/导频/保护={frame.get('data_samples','—')}/{frame.get('pilot_samples','—')}/{frame.get('guard_samples','—')}, "
            f"训练/数据={ratio_text}, 有效占比={useful_text}, "
            f"TX RMS={float(st.get('tx_cycle_rms',np.nan)):.4g}, peak={float(st.get('tx_peak',np.nan)):.4g}, "
            f"PAPR={float(st.get('tx_papr_db',np.nan)):.2f}dB, backoff={float(st.get('tx_power_backoff_db',np.nan)):.2f}dB, "
            f"CFO无歧义±{float(st.get('cfo_unambiguous_hz', np.nan)):.0f}Hz, "
            f"CFO扫描±{float(st.get('cfo_search_max_hz', np.nan)):.0f}Hz, "
            f"TDL预渲染={st.get('tx_tdl_prerendered')}, TDL注入SNR={float(st.get('tdl_injected_snr_db',st.get('tdl_snr_db',np.nan))):.1f}dB, "
            f"αβ自适应={st.get('adaptive_alpha_beta_state','off')}"
        )

    def _log_frame_structure_warning(self, status):
        frame = dict((status or {}).get("frame_structure", {}) or {})
        ratio = frame.get("training_data_ratio")
        if ratio is None:
            return
        ratio = float(ratio)
        if ratio >= 1.0:
            self._log(
                "开销提示：当前导频长度不小于数据长度，"
                f"训练/数据={ratio:.2f}，有效数据占比="
                f"{100.0 * float(frame.get('useful_data_ratio') or 0.0):.1f}%。"
                "这是信道估计开销，不应把长导频本身当作解调优势。"
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
