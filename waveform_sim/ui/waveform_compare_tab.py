# waveform_compare_tab.py
# -*- coding: utf-8 -*-
"""Paired waveform comparison with shared live/scan model and raw statistics.

Both modes use the same physical frame source. The receiver policy and random
seed are explicit, parameters are immutable during a run, and zero-error log
points are represented by statistical bounds rather than fabricated BER floors.
"""

import copy
import json
import threading
import numpy as np
import pyqtgraph as pg

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QGridLayout, QDoubleSpinBox,
    QSplitter, QTabWidget, QScrollArea, QSpinBox, QFileDialog,
    QHeaderView, QTableWidget, QTableWidgetItem, QSizePolicy,
)

MATLAB_BLUE = (0, 114, 189)
MATLAB_ORANGE = (217, 83, 25)
MATLAB_GREEN = (119, 172, 48)
MATLAB_YELLOW = (237, 177, 32)
MATLAB_PURPLE = (126, 47, 142)
LIGHT_PLOT_BACKGROUND = (250, 250, 250)
AXIS_COLOR = (60, 60, 60)
BORDER_COLOR = (225, 225, 225)
SCAN_CI_FLOOR_OFFSET_DECADES = 0.18
SCAN_CI_AXIS_PADDING_DECADES = 0.32



from .compare_workers import _ScanWorker, _PairedWaveformRunner
from ..simulation.compare_scan_backend import ScanConfig, validate_parameters

class WaveformCompareTab(QWidget):
    def __init__(self):
        super().__init__()

        self.live_runner = _PairedWaveformRunner()
        self._live_config = None
        self._has_live_result = False
        self._last_live_snapshot = {}
        self._mode_index = 0
        self._scan_signature = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_live_plots)

        self.scan_timer = QTimer(self)
        self.scan_timer.timeout.connect(self._refresh_scan_plots)
        self._scan_lock = threading.Lock()
        self._scan_stop_event = threading.Event()
        self._scan_worker = None
        self._scan_results = {
            "cfo": {},
            "doppler": {},
            "done": False,
            "status": "尚未开始扫描。",
        }

        self._init_ui()
        self._connect_signals()
        self._update_panel_titles()

    @staticmethod
    def _style_plot_widget(plot_widget: pg.PlotWidget):
        """统一浅色背景、灰色坐标轴和浅色网格。"""
        plot_widget.setBackground(LIGHT_PLOT_BACKGROUND)
        plot_widget.showGrid(x=True, y=True, alpha=0.35)
        for axis_name in ("left", "bottom"):
            axis = plot_widget.getAxis(axis_name)
            axis.setPen(pg.mkPen(AXIS_COLOR, width=1.0))
            axis.setTextPen(pg.mkPen(AXIS_COLOR, width=1.0))
        plot_widget.getPlotItem().getViewBox().setBorder(
            pg.mkPen(BORDER_COLOR, width=1.0)
        )

    def _init_ui(self):
        pg.setConfigOptions(antialias=True)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(8)

        # =======================
        # 左侧：运行期间锁定的公共实验参数
        # =======================
        ctrl_group = QGroupBox("波形对比设置")
        self.live_controls_group = ctrl_group
        ctrl_group.setMinimumWidth(270)
        ctrl_group.setMaximumWidth(330)
        ctrl_layout = QGridLayout(ctrl_group)

        ctrl_layout.addWidget(QLabel("左侧波形:"), 0, 0)
        self.wave1_combo = QComboBox()
        self.wave1_combo.addItems(["OFDM", "OTFS", "AFDM"])
        ctrl_layout.addWidget(self.wave1_combo, 0, 1)

        ctrl_layout.addWidget(QLabel("右侧波形:"), 1, 0)
        self.wave2_combo = QComboBox()
        self.wave2_combo.addItems(["OFDM", "OTFS", "AFDM"])
        self.wave2_combo.setCurrentText("OTFS")
        ctrl_layout.addWidget(self.wave2_combo, 1, 1)

        ctrl_layout.addWidget(QLabel("统一频偏 (CFO):"), 6, 0)
        self.cfo_spin = QDoubleSpinBox()
        self.cfo_spin.setRange(-400_000.0, 400_000.0)
        self.cfo_spin.setValue(0.0)
        self.cfo_spin.setSuffix(" Hz")
        ctrl_layout.addWidget(self.cfo_spin, 6, 1)

        ctrl_layout.addWidget(QLabel("多普勒扩展:"), 7, 0)
        self.doppler_spin = QDoubleSpinBox()
        self.doppler_spin.setRange(0.0, 10_000.0)
        self.doppler_spin.setValue(20.0)
        self.doppler_spin.setSuffix(" Hz")
        ctrl_layout.addWidget(self.doppler_spin, 7, 1)

        ctrl_layout.addWidget(QLabel("Eb/N0:"), 5, 0)
        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setRange(-20.0, 50.0)
        self.snr_spin.setValue(10.0)
        self.snr_spin.setSuffix(" dB")
        ctrl_layout.addWidget(self.snr_spin, 5, 1)

        ctrl_layout.addWidget(QLabel("调制方式:"), 4, 0)
        self.mod_combo = QComboBox()
        self.mod_combo.addItems(["QPSK", "16QAM", "64QAM"])
        ctrl_layout.addWidget(self.mod_combo, 4, 1)

        self.receiver_combo = self._receiver_combo()
        self.csi_combo = self._csi_combo()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        self.seed_spin.setValue(42)
        ctrl_layout.addWidget(QLabel("比较方式:"), 8, 0)
        ctrl_layout.addWidget(self.receiver_combo, 8, 1)
        ctrl_layout.addWidget(QLabel("CSI 条件:"), 9, 0)
        ctrl_layout.addWidget(self.csi_combo, 9, 1)
        ctrl_layout.addWidget(QLabel("随机种子:"), 10, 0)
        ctrl_layout.addWidget(self.seed_spin, 10, 1)

        self.c1_spin = self._chirp_spin()
        self.c2_spin = self._chirp_spin()
        ctrl_layout.addWidget(QLabel("AFDM c1:"), 2, 0)
        ctrl_layout.addWidget(self.c1_spin, 2, 1)
        ctrl_layout.addWidget(QLabel("AFDM c2:"), 3, 0)
        ctrl_layout.addWidget(self.c2_spin, 3, 1)
        self.btn_reset_chirp = QPushButton("恢复 AFDM 默认参数")
        ctrl_layout.addWidget(self.btn_reset_chirp, 11, 0, 1, 2)
        chirp_note = QLabel("c1/c2 仅作用于 AFDM，标准相位不除以 M。\n本页三波形均使用普通 CP。")
        chirp_note.setWordWrap(True)
        ctrl_layout.addWidget(chirp_note, 12, 0, 1, 2)

        self.btn_start = QPushButton("开始对比")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)

        for button in (self.btn_start, self.btn_stop):
            button.setMinimumHeight(30)

        self.note_label = QLabel(
            "左右按同一帧配对：相同比特、信道、噪声及发送能量。运行时锁定参数；停止后修改再运行。"
        )
        self.note_label.setWordWrap(True)
        self.note_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.model_note = QLabel(
            "64 × 7 全载波 / 15 kHz SCS / CP 16。\n可选理想 CSI 或共同导频估计 CSI。\n三角 = 零误码的 95% 统计上界。"
        )
        self.model_note.setWordWrap(True)
        self.model_note.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.model_note.setToolTip(
            "默认是理想 CSI 基准；共同导频模式每帧发送一个 OFDM 训练符号，"
            "接收端知道路径时延支撑和帧定时，但不知道残余 CFO/逐径多普勒，只用 LS 估计复增益。"
            "信道为通用三径模型，不是 NTN profile。"
        )
        ctrl_layout.setRowStretch(13, 1)
        ctrl_layout.setColumnStretch(1, 1)

        # 用分页避免 6 个图同时挤在一个页面里，但原实时 UI 完整保留。
        self.tabs = QTabWidget()
        self.live_page = QWidget()
        self.scan_page = QWidget()
        self.tabs.addTab(self.live_page, "实时对比")
        self.tabs.addTab(self.scan_page, "自动扫描")
        main_layout.addWidget(self.tabs, stretch=1)

        self._init_live_page()
        self._init_scan_page()

    @staticmethod
    def _chirp_spin():
        control = QDoubleSpinBox()
        control.setRange(0., .5)
        control.setDecimals(10)
        control.setSingleStep(.0001)
        control.setValue(.05 / 64.)
        control.setToolTip(
            '标准 DAFT 参数：exp(-j 2π c n²)，与独立 AFDM 仿真页相同。\n'
            '默认 0.05/64=0.00078125，保持原对比页波形；旧归一参数除以 M 后换算。\n'
            '仅改变 AFDM 变换；本页三种波形使用普通 CP。'
        )
        return control

    @staticmethod
    def _receiver_combo():
        combo = QComboBox()
        combo.addItem("统一接收机基准", "full_mmse")
        combo.addItem("实际接收方案对照", "practical")
        combo.setMinimumContentsLength(10)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setToolTip(
            "统一接收机基准：三种波形使用同一完整 LMMSE 检测器和相同每行系数预算。\n"
            "实际接收方案对照：OFDM 单抽头；OTFS/AFDM 使用最多两轮时域稀疏 PCG-LMMSE，再执行各自解调变换。"
            "复杂度代理只估算均衡器的信道系数运算，不含调制变换、向量更新和 CSI 估计；"
            "实际方案的计算预算仍不完全相等，结果反映波形与接收器的组合。"
        )
        return combo

    @staticmethod
    def _csi_combo(default="ideal"):
        combo = QComboBox()
        combo.addItem("理想 CSI（无导频）", "ideal")
        combo.addItem("共同导频估计 CSI", "pilot")
        combo.setToolTip(
            "共同导频模式每帧发送一个 OFDM 训练符号；接收端假设已知路径时延支撑和帧定时，"
            "用 LS 估计复增益，不知道扫描中的残余 CFO 和逐径多普勒。"
            "三种波形共享训练序列、导频开销和估计结果。"
        )
        combo.setCurrentIndex(combo.findData(default))
        return combo

    def _init_live_page(self):
        live_layout = QHBoxLayout(self.live_page)
        live_layout.setContentsMargins(6, 6, 6, 6)
        live_layout.setSpacing(8)
        rail = QWidget()
        rail.setMinimumWidth(290)
        rail.setMaximumWidth(330)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(0, 0, 0, 0)
        operations = QGroupBox('运行控制')
        buttons = QHBoxLayout(operations)
        buttons.addWidget(self.btn_start)
        buttons.addWidget(self.btn_stop)
        rail_layout.addWidget(operations)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.live_controls_group)
        rail_layout.addWidget(scroll, 1)
        rail_layout.addWidget(self.note_label)
        rail_layout.addWidget(self.model_note)
        live_layout.addWidget(rail)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.left_panel = self._create_live_plot_panel("左侧波形")
        self.right_panel = self._create_live_plot_panel("右侧波形")

        splitter.addWidget(self.left_panel["box"])
        splitter.addWidget(self.right_panel["box"])
        splitter.setSizes([640, 640])

        live_layout.addWidget(splitter, 1)

    def _init_scan_page(self):
        scan_layout = QHBoxLayout(self.scan_page)
        scan_layout.setContentsMargins(6, 6, 6, 6)
        scan_layout.setSpacing(8)

        rail = QWidget()
        rail.setMinimumWidth(290)
        rail.setMaximumWidth(330)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(0, 0, 0, 0)
        operation_group = QGroupBox("扫描控制")
        operations = QHBoxLayout(operation_group)
        self.btn_scan_start = QPushButton("开始自动扫描")
        self.btn_scan_stop = QPushButton("停止扫描")
        self.btn_export = QPushButton("导出结果")
        self.btn_export.setEnabled(False)
        self.btn_scan_stop.setEnabled(False)
        for button in (self.btn_scan_start, self.btn_scan_stop):
            button.setMinimumHeight(30)
            operations.addWidget(button)
        operations.addWidget(self.btn_export)
        rail_layout.addWidget(operation_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scan_group = QGroupBox("自动扫描设置")
        grid = QGridLayout(scan_group)
        grid.setColumnStretch(1, 1)

        def spin(low, high, value, suffix="", integer=False):
            control = QDoubleSpinBox()
            control.setRange(low, high)
            control.setValue(value)
            control.setSuffix(suffix)
            if integer:
                control.setDecimals(0)
            return control

        self.scan_snr_spin = spin(-20.0, 50.0, 14.0, " dB")
        self.scan_mod_combo = QComboBox()
        self.scan_mod_combo.addItems(["QPSK", "16QAM", "64QAM"])
        self.scan_frames_spin = spin(1, 500, 100, integer=True)
        self.scan_cfo_min_spin = spin(-400_000.0, 400_000.0, -300.0, " Hz")
        self.scan_cfo_max_spin = spin(-400_000.0, 400_000.0, 300.0, " Hz")
        self.scan_cfo_points_spin = spin(3, 41, 31, integer=True)
        self.scan_fixed_doppler_spin = spin(0.0, 10_000.0, 120.0, " Hz")
        self.scan_doppler_min_spin = spin(0.0, 10_000.0, 0.0, " Hz")
        self.scan_doppler_max_spin = spin(0.0, 10_000.0, 1200.0, " Hz")
        self.scan_doppler_points_spin = spin(3, 41, 31, integer=True)
        self.scan_fixed_cfo_spin = spin(-400_000.0, 400_000.0, 0.0, " Hz")
        self.scan_receiver_combo = self._receiver_combo()
        self.scan_csi_combo = self._csi_combo(default="pilot")
        self.scan_seed_spin = QSpinBox()
        self.scan_seed_spin.setRange(0, 2_000_000_000)
        self.scan_seed_spin.setValue(42)
        self.scan_c1_spin = self._chirp_spin()
        self.scan_c2_spin = self._chirp_spin()
        self.btn_scan_reset_chirp = QPushButton('恢复 AFDM 默认参数')

        rows = (
            ("扫描 Eb/N0:", self.scan_snr_spin),
            ("调制方式:", self.scan_mod_combo),
            ("每点统计帧数:", self.scan_frames_spin),
            ("比较方式:", self.scan_receiver_combo),
            ("CSI 条件:", self.scan_csi_combo),
            ("随机种子:", self.scan_seed_spin),
            ("AFDM c1:", self.scan_c1_spin),
            ("AFDM c2:", self.scan_c2_spin),
            ("CFO 起点:", self.scan_cfo_min_spin),
            ("CFO 终点:", self.scan_cfo_max_spin),
            ("CFO 点数:", self.scan_cfo_points_spin),
            ("固定多普勒:", self.scan_fixed_doppler_spin),
            ("多普勒起点:", self.scan_doppler_min_spin),
            ("多普勒终点:", self.scan_doppler_max_spin),
            ("多普勒点数:", self.scan_doppler_points_spin),
            ("固定 CFO:", self.scan_fixed_cfo_spin),
        )
        for row, (label, control) in enumerate(rows):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(control, row, 1)

        self.scan_note_label = QLabel(
            "每点三波形使用配对比特、信道与噪声。共同导频 CSI 不知道残余 CFO/多普勒。"
            "扫描 CFO 时固定多普勒，反之亦然。"
        )
        self.scan_note_label.setWordWrap(True)
        grid.addWidget(self.btn_scan_reset_chirp, len(rows), 0, 1, 2)
        grid.addWidget(self.scan_note_label, len(rows) + 1, 0, 1, 2)
        grid.setRowStretch(len(rows) + 2, 1)
        self.scan_cfo_min_spin.valueChanged.connect(self._update_scan_setup_note)
        self.scan_cfo_max_spin.valueChanged.connect(self._update_scan_setup_note)
        self.scan_doppler_max_spin.valueChanged.connect(self._update_scan_setup_note)
        self.scan_csi_combo.currentIndexChanged.connect(self._update_scan_setup_note)
        self._update_scan_setup_note()
        scroll.setWidget(scan_group)
        rail_layout.addWidget(scroll, 1)
        scan_layout.addWidget(rail)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.scan_cfo_plot = pg.PlotWidget(title="BER vs CFO")
        self.scan_doppler_plot = pg.PlotWidget(title="BER vs Doppler Spread")
        self._setup_scan_plot(self.scan_cfo_plot, "CFO (Hz)")
        self._setup_scan_plot(self.scan_doppler_plot, "Doppler Spread (Hz)")
        for plot in (self.scan_cfo_plot, self.scan_doppler_plot):
            plot.setMinimumSize(240, 180)
            splitter.addWidget(plot)
        splitter.setSizes([500, 500])
        results = QVBoxLayout()
        results.addWidget(splitter, 1)
        self.result_table = QTableWidget(0, 7)
        self.result_table.setHorizontalHeaderLabels(["扫描轴", "取值 Hz", "OFDM BER", "OTFS BER", "AFDM BER", "比特/波形", "帧数"])
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setMaximumHeight(165)
        self.result_table.verticalHeader().setDefaultSectionSize(24)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        results.addWidget(self.result_table)
        self.scan_summary = QLabel(
            "默认每轴 31 点、每点 100 帧；竖线为 95% 帧聚类区间。"
            "零误码三角标记显示保守上界；下界为 0 的区间以向下三角标出，并截在自适应图底线。"
            "图示截断不改变统计区间或导出结果。"
        )
        self.scan_summary.setWordWrap(True)
        self.scan_summary.setMinimumWidth(0)
        self.scan_summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        results.addWidget(self.scan_summary)
        scan_layout.addLayout(results, stretch=1)

    def _create_live_plot_panel(self, title: str):
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setSpacing(10)

        constellation_plot = pg.PlotWidget(title="接收端解调星座图")
        constellation_plot.setLabel("left", "Q")
        constellation_plot.setLabel("bottom", "I")
        constellation_plot.showGrid(x=True, y=True)
        constellation_plot.setAspectLocked(True)
        constellation_plot.disableAutoRange()
        constellation_plot.setXRange(-2.5, 2.5, padding=0)
        constellation_plot.setYRange(-2.5, 2.5, padding=0)

        constellation_scatter = pg.ScatterPlotItem(
            size=5,
            pen=pg.mkPen(None),
            brush=pg.mkBrush("g")
        )
        constellation_plot.addItem(constellation_scatter)

        ber_plot = pg.PlotWidget(title="误码率性能 (BER)")
        ber_plot.setLabel("left", "累计实测 BER")
        ber_plot.setLabel("bottom", "时间", units="s")
        ber_plot.setLogMode(y=True)
        ber_plot.showGrid(x=True, y=True)

        ber_curve = ber_plot.plot(
            pen=pg.mkPen(MATLAB_BLUE, width=2.0),
            symbol="o",
            symbolPen=pg.mkPen(MATLAB_BLUE, width=1.0),
            symbolBrush=MATLAB_BLUE,
            symbolSize=5
        )

        self._style_plot_widget(constellation_plot)
        self._style_plot_widget(ber_plot)
        constellation_scatter.setPen(pg.mkPen(MATLAB_ORANGE, width=0.8))
        constellation_scatter.setBrush(
            pg.mkBrush(MATLAB_ORANGE[0], MATLAB_ORANGE[1], MATLAB_ORANGE[2], 160)
        )

        layout.addWidget(constellation_plot, stretch=1)
        layout.addWidget(ber_plot, stretch=1)
        zero_curve = ber_plot.plot(pen=None, symbol="t", symbolSize=7, symbolBrush=MATLAB_BLUE)
        summary = QLabel("等待配对数据")
        summary.setWordWrap(True)
        summary.setMinimumHeight(48)
        summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(summary)

        return {
            "box": box,
            "constellation_scatter": constellation_scatter,
            "ber_curve": ber_curve,
            "zero_curve": zero_curve,
            "summary": summary,
            "ber_plot": ber_plot,
        }

    def _setup_scan_plot(self, plot_widget: pg.PlotWidget, x_label: str):
        plot_widget.setLabel("left", "BER")
        plot_widget.setLabel("bottom", x_label)
        plot_widget.setLogMode(y=True)
        self._style_plot_widget(plot_widget)
        plot_widget.addLegend(offset=(10, 10), labelTextColor=AXIS_COLOR)

        plot_widget.ofdm_curve = plot_widget.plot(
            [], [], name="OFDM",
            pen=pg.mkPen(MATLAB_BLUE, width=2),
            symbol="o", symbolSize=6,
            symbolBrush=MATLAB_BLUE,
            symbolPen=pg.mkPen(MATLAB_BLUE),
        )
        plot_widget.otfs_curve = plot_widget.plot(
            [], [], name="OTFS",
            pen=pg.mkPen(MATLAB_ORANGE, width=2),
            symbol="s", symbolSize=6,
            symbolBrush=MATLAB_ORANGE,
            symbolPen=pg.mkPen(MATLAB_ORANGE),
        )
        plot_widget.afdm_curve = plot_widget.plot(
            [], [], name="AFDM",
            pen=pg.mkPen(MATLAB_GREEN, width=2),
            symbol="d", symbolSize=7,
            symbolBrush=MATLAB_GREEN,
            symbolPen=pg.mkPen(MATLAB_GREEN),
        )
        for name, color in (("ofdm", MATLAB_BLUE), ("otfs", MATLAB_ORANGE), ("afdm", MATLAB_GREEN)):
            setattr(plot_widget, name + "_zero_curve", plot_widget.plot(
                pen=None, symbol="t", symbolSize=8, symbolBrush=color, symbolPen=color))
            interval_color = pg.mkColor(color)
            interval_color.setAlpha(175)
            ci_errorbar = pg.ErrorBarItem(beam=0.0, pen=pg.mkPen(interval_color, width=1.2))
            ci_errorbar.setToolTip(
                "95% 独立帧聚类区间。对数坐标下按端点绘制；下界为 0 时显示截断标记。"
            )
            plot_widget.addItem(ci_errorbar)
            setattr(plot_widget, name + "_ci_errorbar", ci_errorbar)
            ci_floor_curve = plot_widget.plot(
                pen=None, symbol="t1", symbolSize=8,
                symbolBrush=interval_color, symbolPen=pg.mkPen(interval_color),
            )
            ci_floor_curve.setToolTip(
                "该区间下界为 0；向下三角标记显示截在当前图表自适应底线，统计下界仍为 0。"
            )
            setattr(plot_widget, name + "_ci_floor_curve", ci_floor_curve)

    def _connect_signals(self):
        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_scan_start.clicked.connect(self._on_scan_start_clicked)
        self.btn_scan_stop.clicked.connect(self._on_scan_stop_clicked)
        self.btn_export.clicked.connect(self._export_results)
        self.btn_reset_chirp.clicked.connect(lambda: self._reset_chirp(False))
        self.btn_scan_reset_chirp.clicked.connect(lambda: self._reset_chirp(True))
        self.wave1_combo.currentTextChanged.connect(self._update_panel_titles)
        self.wave2_combo.currentTextChanged.connect(self._update_panel_titles)
        self.c1_spin.valueChanged.connect(self._on_chirp_changed)
        self.c2_spin.valueChanged.connect(self._on_chirp_changed)
        self.tabs.currentChanged.connect(self._on_mode_changed)
        self.receiver_combo.currentIndexChanged.connect(
            lambda *_: self._prefer_pilot_for_practical(self.receiver_combo, self.csi_combo)
        )
        self.scan_receiver_combo.currentIndexChanged.connect(
            lambda *_: self._prefer_pilot_for_practical(self.scan_receiver_combo, self.scan_csi_combo)
        )
        self.csi_combo.currentIndexChanged.connect(
            lambda *_: self._prefer_pilot_for_practical(self.receiver_combo, self.csi_combo)
        )
        self.scan_csi_combo.currentIndexChanged.connect(
            lambda *_: self._prefer_pilot_for_practical(self.scan_receiver_combo, self.scan_csi_combo)
        )

    @staticmethod
    def _prefer_pilot_for_practical(mode_combo, csi_combo):
        if mode_combo.currentData() == "practical" and csi_combo.currentData() == "ideal":
            csi_combo.setCurrentIndex(csi_combo.findData("pilot"))

    def _reset_chirp(self, scan=False):
        if self.live_runner.is_running() or (self._scan_worker is not None and self._scan_worker.is_alive()):
            return
        controls = (self.scan_c1_spin, self.scan_c2_spin) if scan else (self.c1_spin, self.c2_spin)
        for control in controls:
            control.setValue(.05 / 64.)

    def _on_chirp_changed(self, *_):
        if self._has_live_result:
            self._set_live_note('图表保留上次结果；AFDM 参数修改将在重新开始后生效。')

    def _update_panel_titles(self, *_):
        selected = (self.wave1_combo.currentText(), self.wave2_combo.currentText())
        shown = self._live_config['waveforms'] if self._has_live_result else selected
        self.left_panel['box'].setTitle(f'左侧：{shown[0]}')
        self.right_panel['box'].setTitle(f'右侧：{shown[1]}')
        if self._has_live_result and tuple(selected) != tuple(shown):
            self._set_live_note('图表保留上次波形的结果；当前选择将在重新开始后生效。')

    def _set_live_note(self, text):
        if self._live_config:
            cfg = self._live_config
            policy = self._receiver_text(cfg['receiver_mode'], cfg.get('csi_mode', 'ideal'))
            text += (f'\n结果：{cfg["snr_db"]:g} dB / CFO {cfg["cfo_hz"]:g} Hz / '
                     f'多普勒 {cfg["doppler_hz"]:g} Hz\nseed {cfg["seed"]} · {policy}'
                     f'\nSCS={cfg.get("subcarrier_spacing_hz", 15e3) / 1000:g} kHz · '
                     f'Fs={cfg.get("sample_rate_hz", 960e3) / 1e6:g} MHz'
                     f'\nAFDM c1/c2={cfg["afdm_c1"]:.7g}/{cfg["afdm_c2"]:.7g}')
            self.note_label.setToolTip(json.dumps(self._json_value(cfg), ensure_ascii=False, indent=2))
        self.note_label.setText(text)

    @staticmethod
    def _receiver_text(mode, csi_mode="ideal"):
        receiver = (
            '统一接收机基准：三者完整 LMMSE、每行系数预算相同'
            if mode == 'full_mmse' else
            '实际接收方案：OFDM 单抽头 / OTFS、AFDM 时域稀疏 PCG-LMMSE 后接各自解调；预算不完全相等'
        )
        csi = ('理想 CSI（无导频）' if csi_mode == 'ideal' else
               '共同导频 CSI（已知路径时延支撑；残余 CFO/多普勒未知）')
        return f'{receiver}；{csi}'

    def _set_controls_state(self, running=None):
        live = self.live_runner.is_running()
        scan = self._scan_worker is not None and self._scan_worker.is_alive()
        busy = live or scan
        for control in (self.wave1_combo, self.wave2_combo, self.mod_combo, self.snr_spin,
                        self.cfo_spin, self.doppler_spin, self.seed_spin, self.receiver_combo,
                        self.csi_combo,
                        self.c1_spin, self.c2_spin, self.btn_reset_chirp,
                        self.scan_snr_spin, self.scan_mod_combo, self.scan_frames_spin,
                        self.scan_cfo_min_spin, self.scan_cfo_max_spin, self.scan_cfo_points_spin,
                        self.scan_doppler_min_spin, self.scan_doppler_max_spin, self.scan_doppler_points_spin,
                        self.scan_fixed_cfo_spin, self.scan_fixed_doppler_spin,
                        self.scan_seed_spin, self.scan_receiver_combo, self.scan_csi_combo,
                        self.scan_c1_spin, self.scan_c2_spin, self.btn_scan_reset_chirp):
            control.setEnabled(not busy)
        self.btn_start.setEnabled(not busy)
        self.btn_stop.setEnabled(live)
        self.btn_scan_start.setEnabled(not busy)
        self.btn_scan_stop.setEnabled(scan)
        self.btn_export.setEnabled(not busy and self.result_table.rowCount() > 0)

    def _clear_panel(self, panel):
        panel['constellation_scatter'].setData(x=[], y=[])
        panel['ber_curve'].setData([], [])
        panel['zero_curve'].setData([], [])
        panel['summary'].setText('等待配对数据')

    def _clear_live_plots(self):
        self._has_live_result = False
        self._last_live_snapshot = {}
        self._clear_panel(self.left_panel)
        self._clear_panel(self.right_panel)

    def _clear_scan_plots(self):
        for plot in (self.scan_cfo_plot, self.scan_doppler_plot):
            for name in ('ofdm', 'otfs', 'afdm'):
                getattr(plot, name + '_curve').setData([], [])
                getattr(plot, name + '_zero_curve').setData([], [])
                getattr(plot, name + '_ci_errorbar').setData(
                    x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([])
                )
                getattr(plot, name + '_ci_floor_curve').setData([], [])
        self.result_table.setRowCount(0)
        self._scan_signature = None

    def _on_start_clicked(self):
        params = {
            'waveforms': (self.wave1_combo.currentText(), self.wave2_combo.currentText()),
            'snr_db': float(self.snr_spin.value()), 'cfo_hz': float(self.cfo_spin.value()),
            'doppler_hz': float(self.doppler_spin.value()), 'mod_order': self.mod_combo.currentText(),
            'seed': self.seed_spin.value(), 'receiver_mode': self.receiver_combo.currentData(),
            'csi_mode': self.csi_combo.currentData(),
            'afdm_c1': self.c1_spin.value(), 'afdm_c2': self.c2_spin.value(),
        }
        try:
            resource = ScanConfig(afdm_c1=params['afdm_c1'], afdm_c2=params['afdm_c2']).validated()
            validate_parameters(**{key: value for key, value in params.items()
                                   if key not in ('waveforms', 'afdm_c1', 'afdm_c2')}, config=resource)
        except ValueError as exc:
            self._set_live_note(f'对比参数有误：{exc}')
            return
        if not self._stop_live(silent=True) or not self._stop_scan(silent=True):
            return
        try:
            self.live_runner.start(**params)
        except Exception as exc:
            self.note_label.setText(f'对比未启动：{exc}')
            self._set_controls_state()
            return
        self._clear_live_plots()
        self._live_config = copy.deepcopy(self.live_runner.snapshot()['config'])
        self._update_panel_titles()
        self.tabs.setCurrentWidget(self.live_page)
        self.timer.start(160)
        self._set_live_note('配对对比运行中。')
        self._set_controls_state()

    def _on_stop_clicked(self, silent=False):
        self._stop_live(silent=silent)

    def _stop_live(self, *, silent=False, strict=False):
        self.live_runner.stop()
        try:
            self.live_runner.wait(timeout=1.5)
        except Exception as exc:
            self.note_label.setText(f'正在等待对比任务退出，请稍后重试：{exc}')
            self._set_controls_state()
            if strict:
                raise
            return False
        self.timer.stop()
        self._refresh_live_plots(finalize=False)
        self._set_controls_state()
        if not silent:
            self._set_live_note('已停止，保留结果；修改后重新开始。')
        return True

    @staticmethod
    def _build_axis(start_value, stop_value, count_value):
        start, end, count = float(start_value), float(stop_value), float(count_value)
        if not np.all(np.isfinite([start, end, count])):
            raise ValueError('扫描参数必须是有限数值。')
        if start > end:
            raise ValueError('扫描起点不能大于终点。')
        if not count.is_integer() or not 1 <= count <= 41:
            raise ValueError('扫描点数必须是 1–41 的整数。')
        if start == end:
            return np.array([start])
        if count < 2:
            raise ValueError('不同起止值至少需要两个扫描点。')
        return np.linspace(start, end, int(count))

    def _on_scan_start_clicked(self):
        try:
            params = {
                'snr_db': float(self.scan_snr_spin.value()), 'mod_order': self.scan_mod_combo.currentText(),
                'frames': int(self.scan_frames_spin.value()), 'seed': self.scan_seed_spin.value(),
                'receiver_mode': self.scan_receiver_combo.currentData(),
                'csi_mode': self.scan_csi_combo.currentData(),
                'afdm_c1': self.scan_c1_spin.value(), 'afdm_c2': self.scan_c2_spin.value(),
                'cfo_axis': self._build_axis(self.scan_cfo_min_spin.value(), self.scan_cfo_max_spin.value(), self.scan_cfo_points_spin.value()),
                'doppler_axis': self._build_axis(self.scan_doppler_min_spin.value(), self.scan_doppler_max_spin.value(), self.scan_doppler_points_spin.value()),
                'fixed_cfo': float(self.scan_fixed_cfo_spin.value()),
                'fixed_doppler': float(self.scan_fixed_doppler_spin.value()),
            }
            resource = ScanConfig(afdm_c1=params['afdm_c1'], afdm_c2=params['afdm_c2']).validated()
            for family in ('cfo', 'doppler'):
                for value in params[family + '_axis']:
                    validate_parameters(
                        params['snr_db'], value if family == 'cfo' else params['fixed_cfo'],
                        params['fixed_doppler'] if family == 'cfo' else value,
                        params['mod_order'], frames=params['frames'], seed=params['seed'],
                        receiver_mode=params['receiver_mode'], config=resource,
                        csi_mode=params['csi_mode'],
                    )
        except ValueError as exc:
            self.scan_note_label.setText(f'扫描参数有误：{exc}')
            return
        if not self._stop_live(silent=True) or not self._stop_scan(silent=True):
            return
        self._clear_scan_plots()
        self._scan_results = {
            'cfo': {}, 'doppler': {}, 'done': False, 'error': '', 'cancelled': False,
            'config': copy.deepcopy(params), 'status': '正在准备配对扫描…',
        }
        self._scan_stop_event = threading.Event()
        self._scan_worker = _ScanWorker(self._scan_results, self._scan_lock, self._scan_stop_event, copy.deepcopy(params))
        self.tabs.setCurrentWidget(self.scan_page)
        try:
            self._scan_worker.start()
        except Exception as exc:
            self._scan_worker = None
            self._scan_results.update(done=True, error=str(exc), status=f'扫描未启动：{exc}')
            self._refresh_scan_plots()
            return
        self.scan_timer.start(160)
        self.scan_note_label.setText('正在扫描配对样本。' + self._receiver_text(params['receiver_mode'], params['csi_mode']))
        self._set_controls_state()

    def _on_scan_stop_clicked(self, silent=False):
        self._stop_scan(silent=silent)

    def _stop_scan(self, *, silent=False, strict=False):
        worker = self._scan_worker
        if worker is not None:
            self._scan_stop_event.set()
            worker.join(timeout=1.5)
            if worker.is_alive():
                self.scan_note_label.setText('已请求停止，正在等待当前计算退出；退出前不能启动新任务。')
                self._set_controls_state()
                if strict:
                    raise RuntimeError('对比扫描仍在退出中')
                return False
            self._scan_worker = None
        self.scan_timer.stop()
        self._refresh_scan_plots(finalize=False)
        self._set_controls_state()
        if not silent:
            self.scan_note_label.setText('已停止，保留已完成的三波形配对点；未完成点不参与比较。')
        return True

    def _on_mode_changed(self, index):
        if index == self._mode_index:
            return
        success = self._stop_live(silent=True) if index == 1 else self._stop_scan(silent=True)
        if not success:
            self.tabs.blockSignals(True)
            self.tabs.setCurrentIndex(self._mode_index)
            self.tabs.blockSignals(False)
            return
        self._mode_index = index

    def _update_scan_setup_note(self, *_):
        if not hasattr(self, "scan_note_label"):
            return
        spacing_hz = ScanConfig().subcarrier_spacing
        cfo_fraction = max(abs(self.scan_cfo_min_spin.value()), abs(self.scan_cfo_max_spin.value())) / spacing_hz
        doppler_fraction = 0.8 * self.scan_doppler_max_spin.value() / spacing_hz
        sample_rate = 64 * spacing_hz
        csi_note = (
            "共同导频 CSI：假设已知路径时延支撑与帧定时，不向接收机提供残余 CFO/多普勒。"
            if self.scan_csi_combo.currentData() == "pilot" else
            "理想 CSI：接收机使用完整真实信道矩阵，包含 CFO/多普勒影响，不含导频开销。"
        )
        self.scan_note_label.setText(
            f"SCS={spacing_hz / 1000:g} kHz，采样率={sample_rate / 1e6:g} MHz；"
            f"当前 CFO 扫描端点={cfo_fraction:.3g}×SCS，最大逐径多普勒约 {doppler_fraction:.3g}×SCS。"
            "采样点数固定时，改变 SCS 会改变物理时延和 CP 时长，因此本对比固定 SCS。"
            f"{csi_note}"
            "两条扫描轴分别变化另一项固定。"
        )

    def suspend_tasks(self):
        self._stop_live(silent=True, strict=True)
        self._stop_scan(silent=True, strict=True)

    def resume_tasks(self):
        self._set_controls_state()

    def _refresh_live_plots(self, finalize=True):
        snapshot = self.live_runner.snapshot()
        if snapshot.get('left', {}).get('frames', 0) > 0:
            self._last_live_snapshot = snapshot
            self._has_live_result = True
            self._update_live_panel(self.left_panel, snapshot['left'])
            self._update_live_panel(self.right_panel, snapshot['right'])
            self._update_panel_titles()
        if snapshot.get('error'):
            self._set_live_note(f'对比失败：{snapshot["error"]}；保留已完成配对帧。')
        if finalize and snapshot.get('done') and not self.live_runner.is_running():
            self.timer.stop()
            self.live_runner.wait(timeout=.1)
            self._set_controls_state()

    def _update_live_panel(self, panel, data):
        constellation = np.asarray(data.get('constellation', []))
        panel['constellation_scatter'].setData(x=constellation.real, y=constellation.imag)
        x, y = np.asarray(data.get('ber_t', [])), np.asarray(data.get('ber_v', []))
        upper = np.asarray(data.get('ber_upper', []))
        if y.size and np.any(y > 0):
            panel['ber_curve'].setData(x, np.where(y > 0, y, np.nan), connect='finite')
        else:
            panel['ber_curve'].setData([], [])
        zero = y == 0
        panel['zero_curve'].setData(x[zero], upper[zero])
        panel['zero_curve'].setToolTip('三角标记=零误码样本的帧错误率保守上界；原始 BER 仍为 0。')
        errors, bits, frames = data['bit_errors'], data['total_bits'], data['frames']
        receiver_method = {
            'ofdm_one_tap': 'OFDM 单抽头',
            'sparse_pcg_lmmse': '稀疏 PCG-LMMSE',
            'full_lmmse_time_domain': '完整时域 LMMSE',
        }.get(data.get('receiver_method'), '接收器未运行')
        iterations = (
            f' · 最多 {data.get("receiver_iterations_max", 0)} 轮'
            if data.get('receiver_method') == 'sparse_pcg_lmmse' else ''
        )
        panel['summary'].setText(
            f'{frames} 配对帧 · 错误/比特 {errors}/{bits}\n'
            f'BER {errors/max(bits,1):.4g} · EVM {data["evm_percent"]:.2f}% · CSI-NMSE {data.get("csi_nmse", 0.0):.3g}\n'
            f'接收器 {receiver_method}{iterations}'
        )
        panel['summary'].setToolTip(
            f'95%帧聚类区间 [{data["ci_low"]:.6g}, {data["ci_high"]:.6g}]；'
            f'方法={data.get("ci_method", "unknown")}；FER={data["fer"]:.4g}\n'
            f'导频开销={data.get("pilot_overhead_fraction", 0.0):.1%}；'
            f'每行检测系数预算={data.get("receiver_coefficients_per_row", 0):.1f}；'
            f'均衡器复杂度代理={data.get("receiver_complexity_proxy", 0):.0f}；'
            f'{data.get("receiver_complexity_scope", "")}'
        )

    def _scan_snapshot(self):
        with self._scan_lock:
            return copy.deepcopy(self._scan_results)

    def _refresh_scan_plots(self, finalize=True):
        snapshot = self._scan_snapshot()
        for family, plot in (('cfo', self.scan_cfo_plot), ('doppler', self.scan_doppler_plot)):
            self._update_scan_plot(plot, snapshot.get(family, {}))
        self.scan_note_label.setText(snapshot.get('status', ''))
        self._update_result_table(snapshot)
        if finalize and snapshot.get('done') and self._scan_worker is not None and not self._scan_worker.is_alive():
            self._scan_worker.join(timeout=.1)
            self._scan_worker = None
            self.scan_timer.stop()
        self._set_controls_state()

    def _update_scan_plot(self, plot, family_data):
        y_range_values = []
        series = []
        for name in ('OFDM', 'OTFS', 'AFDM'):
            values = family_data.get(name, {})
            x = np.asarray(values.get('x', []), dtype=float).reshape(-1)
            y = np.asarray(values.get('ber', []), dtype=float).reshape(-1)
            low = np.asarray(values.get('ci_low', np.full(y.shape, np.nan)), dtype=float).reshape(-1)
            high = np.asarray(values.get('ci_high', np.full(y.shape, np.nan)), dtype=float).reshape(-1)
            ci_lengths_match = low.size == high.size == y.size
            data_lengths_match = x.size == y.size
            if low.size != y.size:
                low = np.full(y.shape, np.nan)
            if high.size != y.size:
                high = np.full(y.shape, np.nan)
            zero = (
                (y == 0) & np.isfinite(high) & (high > 0)
                if data_lengths_match and high.size == y.size else np.zeros(y.size, dtype=bool)
            )
            valid_ci = (
                (y > 0) & np.isfinite(y) & np.isfinite(low) & np.isfinite(high)
                & (low >= 0) & (high >= low) & (high > 0)
            ) if data_lengths_match and ci_lengths_match else np.zeros(y.size, dtype=bool)
            if data_lengths_match:
                positive_ber = y[np.isfinite(y) & (y > 0)]
                y_range_values.extend(positive_ber.tolist())
                if np.any(zero):
                    y_range_values.extend(high[zero].tolist())
                if np.any(valid_ci):
                    y_range_values.extend(low[valid_ci][low[valid_ci] > 0].tolist())
                    y_range_values.extend(high[valid_ci].tolist())
            series.append((name, x, y, low, high, zero, valid_ci))

        display_floor = None
        if y_range_values:
            log_values = np.log10(np.asarray(y_range_values, dtype=float))
            min_log = float(np.min(log_values))
            max_log = float(np.max(log_values))
            display_floor = 10 ** (min_log - SCAN_CI_FLOOR_OFFSET_DECADES)
            plot.setYRange(
                min_log - SCAN_CI_AXIS_PADDING_DECADES,
                max_log + SCAN_CI_FLOOR_OFFSET_DECADES,
                padding=0,
            )
        else:
            plot.enableAutoRange(axis=pg.ViewBox.YAxis)

        for name, x, y, low, high, zero, valid_ci in series:
            curve = getattr(plot, name.lower() + '_curve')
            zero_curve = getattr(plot, name.lower() + '_zero_curve')
            ci_errorbar = getattr(plot, name.lower() + '_ci_errorbar')
            ci_floor_curve = getattr(plot, name.lower() + '_ci_floor_curve')
            if x.size != y.size:
                curve.setData([], [])
                zero_curve.setData([], [])
                ci_errorbar.setData(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]))
                ci_floor_curve.setData([], [])
                continue
            if y.size and np.any(y > 0):
                curve.setData(x, np.where(y > 0, y, np.nan), connect='finite')
            else:
                curve.setData([], [])
            zero_curve.setData(x[zero], high[zero])
            if np.any(valid_ci) and display_floor is not None:
                display_low = np.where(low[valid_ci] == 0, display_floor, low[valid_ci])
                log_low = np.log10(display_low)
                log_high = np.log10(high[valid_ci])
            else:
                log_low = np.array([], dtype=float)
                log_high = np.array([], dtype=float)
            sorted_x = np.sort(np.unique(x[valid_ci]))
            steps = np.diff(sorted_x)
            positive_steps = steps[steps > 0]
            beam = float(np.min(positive_steps) * 0.18) if positive_steps.size else 0.0
            ci_errorbar.setData(
                x=x[valid_ci], y=log_low,
                top=log_high - log_low, bottom=np.zeros(log_low.size), beam=beam,
            )

            lower_clipped = valid_ci & (low == 0)
            if np.any(lower_clipped) and display_floor is not None:
                ci_floor_curve.setData(x[lower_clipped], np.full(np.count_nonzero(lower_clipped), display_floor))
                ci_floor_curve.setToolTip(
                    f"该区间下界为 0；向下三角标记截在当前图表底线 {display_floor:.3g}，统计下界仍为 0。"
                )
            else:
                ci_floor_curve.setData([], [])

    def _update_result_table(self, snapshot):
        signature = tuple(len(snapshot.get(f, {}).get('OFDM', {}).get('x', [])) for f in ('cfo', 'doppler'))
        if signature == self._scan_signature:
            return
        self._scan_signature = signature
        self.result_table.setRowCount(0)
        for family, label in (('cfo', 'CFO'), ('doppler', '多普勒')):
            values = snapshot.get(family, {})
            if not all(name in values for name in ('OFDM', 'OTFS', 'AFDM')):
                continue
            for i, x in enumerate(values['OFDM']['x']):
                row = self.result_table.rowCount()
                self.result_table.insertRow(row)
                data = [label, f'{x:g}'] + [f'{values[name]["ber"][i]:.4g}' for name in ('OFDM', 'OTFS', 'AFDM')]
                data += [str(int(values['OFDM']['total_bits'][i])), str(int(values['OFDM']['frames'][i]))]
                for column, text in enumerate(data):
                    item = QTableWidgetItem(text)
                    if 2 <= column <= 4:
                        stats = values[('OFDM', 'OTFS', 'AFDM')[column-2]]
                        paired = ""
                        if column in (3, 4):
                            waveform = "OTFS" if column == 3 else "AFDM"
                            paired = (
                                f'ΔBER({waveform}-OFDM)='
                                f'{stats["ber_diff_vs_OFDM"][i]:.4g} '
                                f'[{stats["ber_diff_vs_OFDM_ci_low"][i]:.4g}, '
                                f'{stats["ber_diff_vs_OFDM_ci_high"][i]:.4g}] '
                                f'({stats["ber_diff_vs_OFDM_ci_method"][i]})\n'
                            )
                        tooltip = (
                            f'错误/比特 {int(stats["bit_errors"][i])}/{int(stats["total_bits"][i])}\n'
                            f'95%按帧区间/上界 [{stats["ci_low"][i]:.6g}, {stats["ci_high"][i]:.6g}]\n'
                            f'方法={stats["ci_method"][i]}\n'
                            f'CSI-NMSE={stats["csi_nmse"][i]:.4g}；'
                            f'导频开销={stats["pilot_overhead_fraction"][i]:.1%}\n'
                            f'接收器={stats["receiver_method"][i]}；'
                            f'最多迭代={int(stats["receiver_iterations_max"][i])} 轮\n'
                            f'每行检测系数预算={stats["receiver_coefficients_per_row"][i]:.1f}；'
                            f'均衡器复杂度代理={stats["receiver_complexity_proxy"][i]:.0f}\n'
                            f'{stats["receiver_complexity_scope"][i]}\n'
                        )
                        item.setToolTip(
                            tooltip + paired + f'EVM={stats["evm_percent"][i]:.3f}%'
                        )
                    self.result_table.setItem(row, column, item)
        cfg = snapshot.get('config', {})
        mode = cfg.get('receiver_mode', 'full_mmse')
        csi_mode = cfg.get('csi_mode', 'ideal')
        spacing_hz = float(cfg.get('subcarrier_spacing_hz', 15e3))
        self.scan_summary.setText(
            f'已完成 CFO {signature[0]} 点、多普勒 {signature[1]} 点；'
            f'结果配置 Eb/N0={cfg.get("snr_db", "—")} dB，SCS={spacing_hz / 1000:g} kHz，'
            f'采样率={64 * spacing_hz / 1e6:g} MHz，seed={cfg.get("seed", "—")}；' + self._receiver_text(mode, csi_mode) +
            f'。AFDM c1/c2={cfg.get("afdm_c1", .05/64.):.7g}/{cfg.get("afdm_c2", .05/64.):.7g}（标准相位、普通 CP）' +
            f'。导频开销={1/(7+1):.1%}（若选择导频 CSI）；区间以独立帧为单位重采样。'
            '零误码三角标记为帧错误率保守上界。帧定时已知，不评估同步捕获算法。'
        )

    @staticmethod
    def _json_value(value):
        if isinstance(value, dict):
            return {k: WaveformCompareTab._json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [WaveformCompareTab._json_value(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _export_results(self):
        if not self.result_table.rowCount():
            return
        path, _ = QFileDialog.getSaveFileName(self, '导出配对波形对比结果', 'waveform_comparison.json', 'JSON (*.json)')
        if not path:
            return
        report = self._json_value(self._scan_snapshot())
        report['assumptions'] = {
            "channel": "Shared synthetic three-tap CP-protected baseband channel; not an NTN profile.",
            "pairing": "Paired bits, channel realization, data noise, and total transmit-frame energy per information bit.",
            "ideal_csi": "Perfect channel knowledge, zero pilot overhead; ideal baseline only.",
            "pilot_csi": "One shared OFDM training symbol per frame; LS estimates complex path gains with known delay support and frame timing. Residual CFO and per-path Doppler are unknown to the receiver.",
            "csi_estimation": {
                "estimated_parameters": ["complex_path_gains"],
                "assumed_known_parameters": ["path_delay_support", "frame_timing"],
                "unestimated_impairments": ["residual_cfo", "per_path_doppler"],
                "shared_across_waveforms": True,
            },
            "receiver_modes": {
                "unified_baseline": "Same full LMMSE detector family and per-row coefficient budget for all waveforms.",
                "practical_receiver": "OFDM one-tap; OTFS/AFDM use at most two sparse time-domain PCG-LMMSE iterations followed by their respective demodulation transforms. This is a receiver-structure comparison; equalizer budgets are not identical.",
            },
            "receiver_complexity_metric": {
                "kind": "algorithmic equalizer coefficient-application proxy, not wall-clock runtime",
                "excludes": ["waveform transforms", "vector updates", "CSI estimation", "memory traffic"],
            },
            "statistics": "Raw pooled BER with independent-frame cluster-bootstrap intervals; zero-error points use a conservative frame-error upper bound.",
            "afdm": "AFDM c1/c2 use exp(-j*2*pi*c*n^2) with no division by M; ordinary CP.",
            "synchronization": "Frame timing is assumed known. Residual CFO and per-path Doppler affect received pilot/data samples but are not supplied to the pilot CSI estimator or equalizer; acquisition algorithms are not evaluated.",
        }
        try:
            with open(path, 'w', encoding='utf-8') as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        except (OSError, ValueError) as exc:
            self.scan_note_label.setText(f'导出失败：{exc}')

    def closeEvent(self, event):
        try:
            self.suspend_tasks()
        except Exception:
            event.ignore()
            return
        super().closeEvent(event)
