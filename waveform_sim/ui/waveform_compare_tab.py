# waveform_compare_tab.py
# -*- coding: utf-8 -*-
"""
波形对比页面。

保留原有实时对比 UI：
- 顶部统一参数设置
- 左右两个波形实时对比面板
- 每个面板包含接收端星座图 + BER-Time 曲线

新增自动扫描 UI：
- 独立分页中增加两个 MATLAB 风格扫描图
- BER vs CFO
- BER vs Doppler Spread
- 每张图包含 OFDM / OTFS / AFDM 三条曲线
- 扫描结果来自 simulation.compare_scan_backend 中的独立 Monte-Carlo 后端，
  不读取当前实时后端的 BER。
"""

import os
import sys
import threading
import traceback
import numpy as np
import pyqtgraph as pg

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QGridLayout, QDoubleSpinBox,
    QSplitter, QTabWidget
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


MATLAB_BLUE = (0, 114, 189)
MATLAB_ORANGE = (217, 83, 25)
MATLAB_GREEN = (119, 172, 48)
MATLAB_YELLOW = (237, 177, 32)
MATLAB_PURPLE = (126, 47, 142)
LIGHT_PLOT_BACKGROUND = (250, 250, 250)
AXIS_COLOR = (60, 60, 60)
BORDER_COLOR = (225, 225, 225)



from .compare_workers import _ScanWorker, _WaveformRunner

class WaveformCompareTab(QWidget):
    def __init__(self):
        super().__init__()

        self.left_runner = _WaveformRunner()
        self.right_runner = _WaveformRunner()

        self.timer = QTimer()
        self.timer.timeout.connect(self._refresh_live_plots)

        self.scan_timer = QTimer()
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
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # =======================
        # 顶部：保留原实时对比参数区
        # =======================
        ctrl_group = QGroupBox("波形对比设置")
        ctrl_layout = QGridLayout(ctrl_group)

        ctrl_layout.addWidget(QLabel("左侧波形:"), 0, 0)
        self.wave1_combo = QComboBox()
        self.wave1_combo.addItems(["OFDM", "OTFS", "AFDM"])
        ctrl_layout.addWidget(self.wave1_combo, 0, 1)

        ctrl_layout.addWidget(QLabel("右侧波形:"), 0, 2)
        self.wave2_combo = QComboBox()
        self.wave2_combo.addItems(["OFDM", "OTFS", "AFDM"])
        self.wave2_combo.setCurrentText("OTFS")
        ctrl_layout.addWidget(self.wave2_combo, 0, 3)

        ctrl_layout.addWidget(QLabel("统一频偏 (CFO):"), 1, 0)
        self.cfo_spin = QDoubleSpinBox()
        self.cfo_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.cfo_spin.setValue(0.0)
        self.cfo_spin.setSuffix(" Hz")
        ctrl_layout.addWidget(self.cfo_spin, 1, 1)

        ctrl_layout.addWidget(QLabel("多普勒扩展:"), 1, 2)
        self.doppler_spin = QDoubleSpinBox()
        self.doppler_spin.setRange(0.0, 1_000_000.0)
        self.doppler_spin.setValue(20.0)
        self.doppler_spin.setSuffix(" Hz")
        ctrl_layout.addWidget(self.doppler_spin, 1, 3)

        ctrl_layout.addWidget(QLabel("Eb/N0:"), 2, 0)
        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setRange(-20.0, 50.0)
        self.snr_spin.setValue(10.0)
        self.snr_spin.setSuffix(" dB")
        ctrl_layout.addWidget(self.snr_spin, 2, 1)

        ctrl_layout.addWidget(QLabel("调制方式:"), 2, 2)
        self.mod_combo = QComboBox()
        self.mod_combo.addItems(["QPSK", "16QAM", "64QAM"])
        ctrl_layout.addWidget(self.mod_combo, 2, 3)

        self.btn_start = QPushButton("开始对比")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)

        ctrl_layout.addWidget(self.btn_start, 0, 4, 2, 1)
        ctrl_layout.addWidget(self.btn_stop, 2, 4, 1, 1)

        self.note_label = QLabel(
            "说明：实时对比页保留原来的左右波形对比；自动扫描页新增 BER-CFO "
            "和 BER-多普勒两个仿真框，扫描结果来自独立 Monte-Carlo 后端。"
        )
        self.note_label.setWordWrap(True)
        ctrl_layout.addWidget(self.note_label, 3, 0, 1, 5)

        main_layout.addWidget(ctrl_group)

        # 用分页避免 6 个图同时挤在一个页面里，但原实时 UI 完整保留。
        self.tabs = QTabWidget()
        self.live_page = QWidget()
        self.scan_page = QWidget()
        self.tabs.addTab(self.live_page, "实时对比")
        self.tabs.addTab(self.scan_page, "自动扫描")
        main_layout.addWidget(self.tabs, stretch=1)

        self._init_live_page()
        self._init_scan_page()

    def _init_live_page(self):
        live_layout = QVBoxLayout(self.live_page)
        live_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        self.left_panel = self._create_live_plot_panel("左侧波形")
        self.right_panel = self._create_live_plot_panel("右侧波形")

        splitter.addWidget(self.left_panel["box"])
        splitter.addWidget(self.right_panel["box"])
        splitter.setSizes([640, 640])

        live_layout.addWidget(splitter)

    def _init_scan_page(self):
        scan_layout = QVBoxLayout(self.scan_page)
        scan_layout.setContentsMargins(0, 0, 0, 0)
        scan_layout.setSpacing(10)

        scan_group = QGroupBox("自动扫描设置")
        grid = QGridLayout(scan_group)

        grid.addWidget(QLabel("扫描 Eb/N0:"), 0, 0)
        self.scan_snr_spin = QDoubleSpinBox()
        self.scan_snr_spin.setRange(-20.0, 50.0)
        self.scan_snr_spin.setValue(14.0)
        self.scan_snr_spin.setSuffix(" dB")
        grid.addWidget(self.scan_snr_spin, 0, 1)

        grid.addWidget(QLabel("扫描调制:"), 0, 2)
        self.scan_mod_combo = QComboBox()
        self.scan_mod_combo.addItems(["QPSK", "16QAM", "64QAM"])
        grid.addWidget(self.scan_mod_combo, 0, 3)

        grid.addWidget(QLabel("每点统计帧数:"), 0, 4)
        self.scan_frames_spin = QDoubleSpinBox()
        self.scan_frames_spin.setRange(1, 500)
        self.scan_frames_spin.setDecimals(0)
        self.scan_frames_spin.setValue(10)
        grid.addWidget(self.scan_frames_spin, 0, 5)

        grid.addWidget(QLabel("CFO 起点:"), 1, 0)
        self.scan_cfo_min_spin = QDoubleSpinBox()
        self.scan_cfo_min_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.scan_cfo_min_spin.setValue(-15000.0)
        self.scan_cfo_min_spin.setSuffix(" Hz")
        grid.addWidget(self.scan_cfo_min_spin, 1, 1)

        grid.addWidget(QLabel("CFO 终点:"), 1, 2)
        self.scan_cfo_max_spin = QDoubleSpinBox()
        self.scan_cfo_max_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.scan_cfo_max_spin.setValue(15000.0)
        self.scan_cfo_max_spin.setSuffix(" Hz")
        grid.addWidget(self.scan_cfo_max_spin, 1, 3)

        grid.addWidget(QLabel("CFO 点数:"), 1, 4)
        self.scan_cfo_points_spin = QDoubleSpinBox()
        self.scan_cfo_points_spin.setRange(3, 41)
        self.scan_cfo_points_spin.setDecimals(0)
        self.scan_cfo_points_spin.setValue(9)
        grid.addWidget(self.scan_cfo_points_spin, 1, 5)

        grid.addWidget(QLabel("CFO扫描固定多普勒:"), 2, 0)
        self.scan_fixed_doppler_spin = QDoubleSpinBox()
        self.scan_fixed_doppler_spin.setRange(0.0, 1_000_000.0)
        self.scan_fixed_doppler_spin.setValue(120.0)
        self.scan_fixed_doppler_spin.setSuffix(" Hz")
        grid.addWidget(self.scan_fixed_doppler_spin, 2, 1)

        grid.addWidget(QLabel("多普勒起点:"), 3, 0)
        self.scan_doppler_min_spin = QDoubleSpinBox()
        self.scan_doppler_min_spin.setRange(0.0, 1_000_000.0)
        self.scan_doppler_min_spin.setValue(0.0)
        self.scan_doppler_min_spin.setSuffix(" Hz")
        grid.addWidget(self.scan_doppler_min_spin, 3, 1)

        grid.addWidget(QLabel("多普勒终点:"), 3, 2)
        self.scan_doppler_max_spin = QDoubleSpinBox()
        self.scan_doppler_max_spin.setRange(0.0, 1_000_000.0)
        self.scan_doppler_max_spin.setValue(600.0)
        self.scan_doppler_max_spin.setSuffix(" Hz")
        grid.addWidget(self.scan_doppler_max_spin, 3, 3)

        grid.addWidget(QLabel("多普勒点数:"), 3, 4)
        self.scan_doppler_points_spin = QDoubleSpinBox()
        self.scan_doppler_points_spin.setRange(3, 41)
        self.scan_doppler_points_spin.setDecimals(0)
        self.scan_doppler_points_spin.setValue(9)
        grid.addWidget(self.scan_doppler_points_spin, 3, 5)

        grid.addWidget(QLabel("多普勒扫描固定 CFO:"), 4, 0)
        self.scan_fixed_cfo_spin = QDoubleSpinBox()
        self.scan_fixed_cfo_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.scan_fixed_cfo_spin.setValue(0.0)
        self.scan_fixed_cfo_spin.setSuffix(" Hz")
        grid.addWidget(self.scan_fixed_cfo_spin, 4, 1)

        self.btn_scan_start = QPushButton("开始自动扫描")
        self.btn_scan_stop = QPushButton("停止扫描")
        self.btn_scan_stop.setEnabled(False)
        grid.addWidget(self.btn_scan_start, 4, 4)
        grid.addWidget(self.btn_scan_stop, 4, 5)

        self.scan_note_label = QLabel(
            "扫描说明：BER-多普勒曲线默认固定 CFO=0 Hz，用来单独观察多普勒扩展；"
            "扫描后端显式建模 OFDM 残余 ICI 随多普勒增长更快，OTFS/AFDM 在 DD/Chirp 域增长更慢。"
        )
        self.scan_note_label.setWordWrap(True)
        grid.addWidget(self.scan_note_label, 5, 0, 1, 6)

        scan_layout.addWidget(scan_group)

        splitter = QSplitter(Qt.Horizontal)
        self.scan_cfo_plot = pg.PlotWidget(title="BER vs CFO")
        self.scan_doppler_plot = pg.PlotWidget(title="BER vs Doppler Spread")
        self._setup_scan_plot(self.scan_cfo_plot, "CFO (Hz)")
        self._setup_scan_plot(self.scan_doppler_plot, "Doppler Spread (Hz)")

        splitter.addWidget(self.scan_cfo_plot)
        splitter.addWidget(self.scan_doppler_plot)
        splitter.setSizes([640, 640])
        scan_layout.addWidget(splitter, stretch=1)

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
        ber_plot.setLabel("left", "Estimated BER")
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

        return {
            "box": box,
            "constellation_scatter": constellation_scatter,
            "ber_curve": ber_curve,
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
            symbol="t", symbolSize=7,
            symbolBrush=MATLAB_GREEN,
            symbolPen=pg.mkPen(MATLAB_GREEN),
        )

    def _connect_signals(self):
        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)

        self.btn_scan_start.clicked.connect(self._on_scan_start_clicked)
        self.btn_scan_stop.clicked.connect(self._on_scan_stop_clicked)

        self.wave1_combo.currentTextChanged.connect(self._update_panel_titles)
        self.wave2_combo.currentTextChanged.connect(self._update_panel_titles)

        self.snr_spin.valueChanged.connect(self._on_runtime_params_changed)
        self.cfo_spin.valueChanged.connect(self._on_runtime_params_changed)
        self.doppler_spin.valueChanged.connect(self._on_runtime_params_changed)

        self.mod_combo.currentTextChanged.connect(self._on_modulation_changed)

    def _update_panel_titles(self):
        self.left_panel["box"].setTitle(f"左侧：{self.wave1_combo.currentText()}")
        self.right_panel["box"].setTitle(f"右侧：{self.wave2_combo.currentText()}")

    def _set_controls_state(self, running: bool):
        self.wave1_combo.setEnabled(not running)
        self.wave2_combo.setEnabled(not running)
        self.mod_combo.setEnabled(not running)

        self.snr_spin.setEnabled(True)
        self.cfo_spin.setEnabled(True)
        self.doppler_spin.setEnabled(True)

        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    def _clear_panel(self, panel: dict):
        panel["constellation_scatter"].setData(x=[], y=[])
        panel["ber_curve"].setData([], [])

    def _clear_live_plots(self):
        self._clear_panel(self.left_panel)
        self._clear_panel(self.right_panel)

    def _clear_scan_plots(self):
        for plot in (self.scan_cfo_plot, self.scan_doppler_plot):
            plot.ofdm_curve.setData([], [])
            plot.otfs_curve.setData([], [])
            plot.afdm_curve.setData([], [])

    def _on_modulation_changed(self):
        if self.left_runner.is_running() or self.right_runner.is_running():
            self.note_label.setText(
                "说明：调制方式已修改，请先停止再重新开始对比，使新调制方式完整生效。"
            )

    def _on_runtime_params_changed(self, *args):
        if not (self.left_runner.is_running() or self.right_runner.is_running()):
            return

        snr_db = float(self.snr_spin.value())
        cfo_hz = float(self.cfo_spin.value())
        doppler_hz = float(self.doppler_spin.value())

        self.left_runner.update_runtime_params(snr_db, cfo_hz, doppler_hz)
        self.right_runner.update_runtime_params(snr_db, cfo_hz, doppler_hz)

    def _on_start_clicked(self):
        self._on_stop_clicked(silent=True)
        self._clear_live_plots()

        wave1 = self.wave1_combo.currentText()
        wave2 = self.wave2_combo.currentText()
        snr_db = float(self.snr_spin.value())
        cfo_hz = float(self.cfo_spin.value())
        doppler_hz = float(self.doppler_spin.value())
        mod_order = self.mod_combo.currentText()

        try:
            self.left_runner.start(
                wave_type=wave1,
                snr_db=snr_db,
                cfo_hz=cfo_hz,
                doppler_hz=doppler_hz,
                mod_order=mod_order,
            )

            self.right_runner.start(
                wave_type=wave2,
                snr_db=snr_db,
                cfo_hz=cfo_hz,
                doppler_hz=doppler_hz,
                mod_order=mod_order,
            )

            self._set_controls_state(True)
            self.timer.start(300)
            self.tabs.setCurrentWidget(self.live_page)

            self.note_label.setText(
                "对比已启动。你现在可以继续动态调整 CFO / 多普勒扩展 / Eb/N0；"
                "自动扫描页仍可单独运行独立扫描。"
            )

        except Exception as e:
            self._on_stop_clicked(silent=True)
            self.note_label.setText(f"启动失败：{e}")
            traceback.print_exc()

    def _on_stop_clicked(self, silent=False):
        self.timer.stop()
        self.left_runner.stop()
        self.right_runner.stop()
        self._set_controls_state(False)

        if not silent:
            self.note_label.setText("已停止实时对比。你可以修改波形或参数后重新开始。")

    @staticmethod
    def _build_axis(start_value: float, stop_value: float, count_value: float):
        count = max(2, int(count_value))
        if abs(float(stop_value) - float(start_value)) < 1e-12:
            return np.array([float(start_value)], dtype=np.float64)
        return np.linspace(float(start_value), float(stop_value), count, dtype=np.float64)

    def _on_scan_start_clicked(self):
        self._on_scan_stop_clicked(silent=True)
        self._clear_scan_plots()

        params = {
            "snr_db": float(self.scan_snr_spin.value()),
            "mod_order": self.scan_mod_combo.currentText(),
            "frames": int(self.scan_frames_spin.value()),
            "cfo_axis": self._build_axis(
                self.scan_cfo_min_spin.value(),
                self.scan_cfo_max_spin.value(),
                self.scan_cfo_points_spin.value(),
            ),
            "doppler_axis": self._build_axis(
                self.scan_doppler_min_spin.value(),
                self.scan_doppler_max_spin.value(),
                self.scan_doppler_points_spin.value(),
            ),
            "fixed_cfo": float(self.scan_fixed_cfo_spin.value()),
            "fixed_doppler": float(self.scan_fixed_doppler_spin.value()),
        }

        self._scan_results = {
            "cfo": {},
            "doppler": {},
            "done": False,
            "status": "正在准备扫描...",
        }
        self._scan_stop_event = threading.Event()
        self._scan_worker = _ScanWorker(
            result_store=self._scan_results,
            result_lock=self._scan_lock,
            stop_event=self._scan_stop_event,
            params=params,
        )
        self._scan_worker.start()
        self.scan_timer.start(300)

        self.tabs.setCurrentWidget(self.scan_page)
        self.btn_scan_start.setEnabled(False)
        self.btn_scan_stop.setEnabled(True)
        self.scan_note_label.setText(
            "自动扫描已启动：正在用独立后端生成 BER-CFO 和 BER-多普勒曲线。"
        )

    def _on_scan_stop_clicked(self, silent=False):
        self.scan_timer.stop()
        if self._scan_worker is not None:
            self._scan_stop_event.set()
            self._scan_worker.join(timeout=1.0)
            self._scan_worker = None

        self.btn_scan_start.setEnabled(True)
        self.btn_scan_stop.setEnabled(False)

        if not silent:
            self.scan_note_label.setText("自动扫描已停止。")

    def _refresh_live_plots(self):
        left_data = self.left_runner.snapshot()
        right_data = self.right_runner.snapshot()

        self._update_live_panel(self.left_panel, left_data)
        self._update_live_panel(self.right_panel, right_data)

    def _refresh_scan_plots(self):
        with self._scan_lock:
            scan_results = {
                "cfo": dict(self._scan_results.get("cfo", {})),
                "doppler": dict(self._scan_results.get("doppler", {})),
                "done": bool(self._scan_results.get("done", False)),
                "status": str(self._scan_results.get("status", "")),
            }

        self._update_scan_plot(self.scan_cfo_plot, scan_results["cfo"])
        self._update_scan_plot(self.scan_doppler_plot, scan_results["doppler"])
        self.scan_note_label.setText(scan_results["status"])

        if scan_results["done"]:
            self.scan_timer.stop()
            self.btn_scan_start.setEnabled(True)
            self.btn_scan_stop.setEnabled(False)
            self._scan_worker = None

    def _update_live_panel(self, panel: dict, data: dict):
        constellation = data["constellation"]
        if constellation is not None and len(constellation) > 0:
            panel["constellation_scatter"].setData(
                x=np.real(constellation),
                y=np.imag(constellation)
            )
        else:
            panel["constellation_scatter"].setData(x=[], y=[])

        ber_t = data["ber_t"]
        ber_v = data["ber_v"]
        if ber_t is not None and ber_v is not None and len(ber_t) > 0 and len(ber_v) > 0:
            n = min(len(ber_t), len(ber_v))
            panel["ber_curve"].setData(
                ber_t[:n],
                np.maximum(ber_v[:n], 1e-6)
            )
        else:
            panel["ber_curve"].setData([], [])

    def _update_scan_plot(self, plot_widget: pg.PlotWidget, family_data: dict):
        curves = {
            "OFDM": plot_widget.ofdm_curve,
            "OTFS": plot_widget.otfs_curve,
            "AFDM": plot_widget.afdm_curve,
        }
        for waveform, curve in curves.items():
            xy = family_data.get(waveform)
            if xy is None:
                curve.setData([], [])
                continue
            x, y = xy
            curve.setData(
                np.asarray(x, dtype=np.float64),
                np.maximum(np.asarray(y, dtype=np.float64), 1e-8),
            )

    def closeEvent(self, event):
        self._on_stop_clicked(silent=True)
        self._on_scan_stop_clicked(silent=True)
        super().closeEvent(event)
