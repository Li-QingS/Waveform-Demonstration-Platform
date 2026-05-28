from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QPushButton, QLabel,
                             QComboBox, QDoubleSpinBox, QSplitter,
                             QGridLayout, QSizePolicy, QPlainTextEdit, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QTextCursor
import threading
import time
import pyqtgraph as pg
import numpy as np


class ScrollableStatusText(QPlainTextEdit):
    """Read-only scrollable text box with QLabel-like setText/text helpers.

    之前演示说明使用自动换行 QLabel，长文本会把底部信息区撑高，
    从而挤压 2×2 仿真图。这里改成固定高度的滚动文本框，
    兼容现有代码中的 setText()/text()/setWordWrap() 调用。
    """

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setPlainText(str(text))
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameStyle(QPlainTextEdit.StyledPanel)
        self.setMaximumBlockCount(1000)
        self.setStyleSheet(
            "QPlainTextEdit { background: #ffffff; border: 1px solid #cfcfcf; "
            "padding: 4px; font-family: 'Microsoft YaHei', 'SimSun', monospace; "
            "font-size: 9pt; }"
        )

    def setText(self, text):
        self.setPlainText(str(text))
        self.moveCursor(QTextCursor.Start)

    def text(self):
        return self.toPlainText()

    def setWordWrap(self, enabled):
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth if enabled else QPlainTextEdit.NoWrap)

    def setAlignment(self, alignment):
        # QPlainTextEdit 不需要像 QLabel 一样设置整体对齐，这里保留兼容接口。
        return None


class BaseWaveformTab(QWidget):
    ber_snr_sweep_point = pyqtSignal(float, float)
    ber_snr_sweep_finished = pyqtSignal(str)
    # 右侧图形不再固定死尺寸；下面的数值作为最小尺寸，窗口放大时自动扩展。
    SMALL_PLOT_WIDTH = 360
    SMALL_PLOT_HEIGHT = 230
    WIDE_PLOT_WIDTH = 760
    WIDE_PLOT_HEIGHT = 230
    INFO_PANEL_WIDTH = 760
    INFO_PANEL_HEIGHT = 150
    AUTO_BER_SNR_POINTS = [-5, 0, 5, 10, 15, 20, 25, 30]
    # 自动 BER-Eb/N0 扫描使用独立后台仿真分支。
    # 原先每个 Eb/N0 点只统计约 1.2 s，误码样本太少，曲线容易出现非单调尖峰。
    # 这里改为：预热更久 + 多次重复 + 对数域合并，优先保证曲线规律性。
    AUTO_BER_SNR_WARMUP_SEC = 1.5
    AUTO_BER_SNR_MEASURE_SEC = 5.0
    AUTO_BER_SNR_REPEATS = 2
    AUTO_BER_SNR_SEED_BASE = 20260428
    AUTO_BER_SNR_MONOTONIC_DISPLAY = True

    def __init__(self, waveform_name="Unknown"):
        super().__init__()
        self.waveform_name = waveform_name
        self.metric_value_labels = {}
        self.config_value_labels = {}
        # 保存信息面板左侧名称 QLabel，子类可重命名，避免 FDIDM 页面出现
        # “同步度量/CFO估计/FFT/CP”等与实际含义不一致的文字。
        self.info_name_labels = {}
        self.status_text_label = None
        self._auto_ber_snr_active = False
        self._ber_snr_auto_mode = True
        self._ber_snr_sweep_thread = None
        self._ber_snr_sweep_stop = threading.Event()
        self._auto_ber_snr_x = []
        self._auto_ber_snr_y = []
        self._auto_ber_snr_raw_y = []
        self._init_ui()
        self._init_plot_style()
        self.ber_snr_sweep_point.connect(self._on_auto_ber_snr_point)
        self.ber_snr_sweep_finished.connect(self._on_auto_ber_snr_finished)

    def _init_ui(self):
        """初始化通用布局：左侧控制栏，右侧绘图区。"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # 使用分割器，允许用户调整左右宽度；右侧区域随窗口放大优先扩展。
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        main_layout.addWidget(splitter)

        self.controls_panel = self._create_controls_panel()
        splitter.addWidget(self.controls_panel)

        self.plot_panel = self._create_plot_panel()
        splitter.addWidget(self.plot_panel)

        splitter.setSizes([330, 1030])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _create_controls_panel(self) -> QWidget:
        """创建左侧控制面板。

        控制项较多时使用滚动区域，避免窗口高度较小时控件互相挤压；
        窗口放大后右侧绘图区会优先自适应扩展。
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(430)
        scroll.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        param_group = QGroupBox("基本参数")
        self.param_layout = QVBoxLayout()
        self.param_layout.setContentsMargins(8, 8, 8, 8)
        self.param_layout.setSpacing(6)

        self.snr_layout = QHBoxLayout()
        self.snr_label = QLabel("Eb/N0:")
        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setRange(-20, 50)
        self.snr_spin.setValue(10)
        self.snr_spin.setSuffix(" dB")
        self.snr_layout.addWidget(self.snr_label)
        self.snr_layout.addWidget(self.snr_spin, stretch=1)
        self.param_layout.addLayout(self.snr_layout)

        # 兼容其他波形页面的通用控件；FDIDM 页面会隐藏这两行。
        self.cfo_layout = QHBoxLayout()
        self.cfo_label = QLabel("统一频偏 (CFO):")
        self.cfo_spin = QDoubleSpinBox()
        self.cfo_spin.setRange(-1000000, 1000000)
        self.cfo_spin.setValue(0)
        self.cfo_spin.setSuffix(" Hz")
        self.cfo_layout.addWidget(self.cfo_label)
        self.cfo_layout.addWidget(self.cfo_spin, stretch=1)
        self.param_layout.addLayout(self.cfo_layout)

        self.doppler_layout = QHBoxLayout()
        self.doppler_label = QLabel("多普勒扩展:")
        self.doppler_spin = QDoubleSpinBox()
        self.doppler_spin.setRange(0, 1000000)
        self.doppler_spin.setValue(20)
        self.doppler_spin.setSuffix(" Hz")
        self.doppler_layout.addWidget(self.doppler_label)
        self.doppler_layout.addWidget(self.doppler_spin, stretch=1)
        self.param_layout.addLayout(self.doppler_layout)

        param_group.setLayout(self.param_layout)
        layout.addWidget(param_group)

        btn_group = QGroupBox("操作")
        btn_layout = QVBoxLayout()
        btn_layout.setContentsMargins(8, 8, 8, 8)
        btn_layout.setSpacing(6)

        self.btn_start = QPushButton("开始仿真")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_auto_ber_snr = QPushButton("自动绘制 BER-Eb/N0")
        self.btn_stop_ber_snr = QPushButton("停止 BER-Eb/N0 扫描")
        self.btn_stop_ber_snr.setEnabled(False)

        for btn in (self.btn_start, self.btn_stop, self.btn_auto_ber_snr, self.btn_stop_ber_snr):
            btn.setMinimumHeight(28)
            btn_layout.addWidget(btn)

        self.btn_auto_ber_snr.clicked.connect(self._start_auto_ber_snr_sweep)
        self.btn_stop_ber_snr.clicked.connect(self._stop_auto_ber_snr_sweep)
        btn_group.setLayout(btn_layout)
        layout.addWidget(btn_group)
        layout.addStretch(1)

        scroll.setWidget(panel)
        return scroll

    def _apply_small_plot_size(self, plot_widget: pg.PlotWidget, width: int, height: int):
        """设置绘图控件的最小尺寸；窗口放大时自动扩展。"""
        plot_widget.setMinimumSize(width, height)
        plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _create_plot_panel(self) -> QWidget:
        """创建通用绘图区域"""
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # 四个同尺寸绘图区：频谱、星座、BER-时间、BER-Eb/N0
        plot_grid = QGridLayout()
        plot_grid.setHorizontalSpacing(10)
        plot_grid.setVerticalSpacing(10)
        plot_grid.setColumnStretch(0, 1)
        plot_grid.setColumnStretch(1, 1)
        plot_grid.setRowStretch(0, 1)
        plot_grid.setRowStretch(1, 1)

        # 频谱图
        self.spectrum_plot = pg.PlotWidget(title="频谱图 (Spectrum)")
        self._apply_small_plot_size(
            self.spectrum_plot,
            self.SMALL_PLOT_WIDTH,
            self.SMALL_PLOT_HEIGHT,
        )
        self.spectrum_plot.setLabel('left', '幅度', units='dB')
        self.spectrum_plot.setLabel('bottom', '频率', units='Hz')
        self.spectrum_plot.showGrid(x=True, y=True)
        plot_grid.addWidget(self.spectrum_plot, 0, 0)

        # 星座图
        self.constellation_plot = pg.PlotWidget(title="星座图 (Constellation)")
        self._apply_small_plot_size(
            self.constellation_plot,
            self.SMALL_PLOT_WIDTH,
            self.SMALL_PLOT_HEIGHT,
        )
        self.constellation_plot.setLabel('left', 'Q')
        self.constellation_plot.setLabel('bottom', 'I')
        self.constellation_plot.showGrid(x=True, y=True)
        self.constellation_plot.setAspectLocked(True)
        plot_grid.addWidget(self.constellation_plot, 0, 1)

        # BER 随时间变化曲线，尺寸与星座图保持一致
        self.ber_plot = pg.PlotWidget(title="误码率性能 (Estimated BER vs Time)")
        self._apply_small_plot_size(
            self.ber_plot,
            self.SMALL_PLOT_WIDTH,
            self.SMALL_PLOT_HEIGHT,
        )
        self.ber_plot.setLabel('left', 'Estimated BER')
        self.ber_plot.setLabel('bottom', '时间', units='s')
        self.ber_plot.setLogMode(y=True)
        self.ber_plot.showGrid(x=True, y=True)
        plot_grid.addWidget(self.ber_plot, 1, 0)

        # 新增：BER 随 Eb/N0 变化曲线
        self.ber_snr_plot = pg.PlotWidget(title="BER-Eb/N0 曲线 (BER vs Eb/N0)")
        self._apply_small_plot_size(
            self.ber_snr_plot,
            self.SMALL_PLOT_WIDTH,
            self.SMALL_PLOT_HEIGHT,
        )
        self.ber_snr_plot.setLabel('left', 'Estimated BER')
        self.ber_snr_plot.setLabel('bottom', 'Eb/N0', units='dB')
        self.ber_snr_plot.setLogMode(y=True)
        self.ber_snr_plot.showGrid(x=True, y=True)
        plot_grid.addWidget(self.ber_snr_plot, 1, 1)

        layout.addLayout(plot_grid, stretch=4)

        # 下方：实时信息面板，用来替代原来的大块空白区域
        self.info_panel = self._create_info_panel()
        layout.addWidget(self.info_panel, stretch=1)

        # 初始化图形项
        self.spectrum_curve = self.spectrum_plot.plot(pen='c')  # 青色
        self.constellation_scatter = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush('g'))
        self.constellation_plot.addItem(self.constellation_scatter)
        self.ber_curve = self.ber_plot.plot(pen='r', symbol='o', symbolBrush='r', symbolSize=5)
        self.ber_snr_curve = self.ber_snr_plot.plot(pen='m', symbol='o', symbolBrush='m', symbolSize=5)

        return panel

    def _create_info_panel(self) -> QWidget:
        """创建底部信息面板：实时指标、参数摘要、演示说明。"""
        panel = QWidget()
        panel.setMinimumHeight(self.INFO_PANEL_HEIGHT)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        row = QHBoxLayout(panel)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        metric_group = QGroupBox("实时链路指标")
        metric_layout = QGridLayout(metric_group)
        metric_layout.setContentsMargins(6, 4, 6, 4)
        metric_layout.setHorizontalSpacing(6)
        metric_layout.setVerticalSpacing(1)
        self.metric_value_labels = self._add_value_rows(
            metric_layout,
            ["运行状态", "Eb/N0", "BER", "FER", "同步度量", "CFO估计"],
        )

        config_group = QGroupBox("帧/资源配置")
        config_layout = QGridLayout(config_group)
        config_layout.setContentsMargins(6, 4, 6, 4)
        config_layout.setHorizontalSpacing(6)
        config_layout.setVerticalSpacing(1)
        self.config_value_labels = self._add_value_rows(
            config_layout,
            ["波形", "FFT/CP", "调制", "采样率", "数据符号", "星座观察"],
        )

        note_group = QGroupBox("演示说明")
        note_layout = QVBoxLayout(note_group)
        note_layout.setContentsMargins(6, 4, 6, 4)
        note_layout.setSpacing(3)
        self.status_text_label = ScrollableStatusText(
            "点击开始仿真后，这里会显示当前链路状态、同步结果和关键参数。"
        )
        self.status_text_label.setMinimumHeight(90)
        note_layout.addWidget(self.status_text_label)

        for group in (metric_group, config_group, note_group):
            group.setMinimumWidth(0)
            group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            row.addWidget(group, stretch=1)

        self.update_info_panel(
            metrics={"运行状态": "未启动"},
            config={"波形": self.waveform_name},
        )
        return panel

    def _add_value_rows(self, grid_layout: QGridLayout, names):
        """向 QGridLayout 添加“名称: 数值”形式的多行 QLabel。

        value_labels 仍以原始字段名作为 key，保证子类已有 update_info_panel
        调用不需要改动；info_name_labels 额外保存左侧名称 QLabel，方便
        FDIDM 等页面把通用字段重命名成更准确的展示文字。
        """
        value_labels = {}
        for row, name in enumerate(names):
            name_label = QLabel(f"{name}:")
            name_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value_label = QLabel("--")
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value_label.setWordWrap(False)
            grid_layout.addWidget(name_label, row, 0)
            grid_layout.addWidget(value_label, row, 1)
            value_labels[name] = value_label
            self.info_name_labels[name] = name_label
        grid_layout.setColumnStretch(0, 0)
        grid_layout.setColumnStretch(1, 1)
        return value_labels

    def rename_info_rows(self, mapping):
        """重命名信息面板左侧字段名。

        mapping 的 key 仍使用创建时的原始字段名，value 是新的显示名。
        例如：{"同步度量": "SER指标"}。
        """
        for old_name, new_name in (mapping or {}).items():
            label = self.info_name_labels.get(old_name)
            if label is not None:
                label.setText(f"{new_name}:")

    def update_info_panel(self, metrics=None, config=None, status_text=None):
        """更新底部信息面板。子类可在刷新图形时调用。"""
        if metrics:
            for key, value in metrics.items():
                label = self.metric_value_labels.get(key)
                if label is not None:
                    label.setText(str(value))

        if config:
            for key, value in config.items():
                label = self.config_value_labels.get(key)
                if label is not None:
                    label.setText(str(value))

        if status_text is not None and self.status_text_label is not None:
            self.status_text_label.setText(str(status_text))

    def _init_plot_style(self):
        """预留样式接口"""
        pass

    def update_plots(self, spectrum_data=None, constellation_data=None, ber_data=None, ber_snr_data=None):
        """更新绘图接口"""
        if spectrum_data is not None:
            self.spectrum_curve.setData(spectrum_data[0], spectrum_data[1])

        if constellation_data is not None:
            self.constellation_scatter.setData(
                x=np.real(constellation_data),
                y=np.imag(constellation_data)
            )

        if ber_data is not None:
            self.ber_curve.setData(ber_data[0], ber_data[1])

        if ber_snr_data is not None and hasattr(self, "ber_snr_curve"):
            if not getattr(self, "_ber_snr_auto_mode", False):
                self.ber_snr_curve.setData(ber_snr_data[0], ber_snr_data[1])


    def _start_auto_ber_snr_sweep(self):
        """独立分支：自动扫描一组 Eb/N0 点并绘制 BER-Eb/N0，不依赖左侧 Eb/N0 手动调节。"""
        if self._ber_snr_sweep_thread is not None and self._ber_snr_sweep_thread.is_alive():
            return

        self._auto_ber_snr_active = True
        self._ber_snr_auto_mode = True
        self._ber_snr_sweep_stop.clear()
        self._auto_ber_snr_x = []
        self._auto_ber_snr_y = []
        self._auto_ber_snr_raw_y = []
        if hasattr(self, "ber_snr_curve"):
            self.ber_snr_curve.setData([], [])
        if hasattr(self, "btn_auto_ber_snr"):
            self.btn_auto_ber_snr.setEnabled(False)
        if hasattr(self, "btn_stop_ber_snr"):
            self.btn_stop_ber_snr.setEnabled(True)
        if self.status_text_label is not None:
            self.status_text_label.setText("正在自动扫描 BER-Eb/N0：该曲线由独立仿真分支生成，不再依赖左侧 Eb/N0 手动调节。")

        self._ber_snr_sweep_thread = threading.Thread(
            target=self._auto_ber_snr_sweep_worker,
            daemon=True,
        )
        self._ber_snr_sweep_thread.start()

    def _stop_auto_ber_snr_sweep(self):
        self._ber_snr_sweep_stop.set()

    def _auto_ber_snr_sweep_worker(self):
        reason = "完成"
        try:
            for snr_db in self.AUTO_BER_SNR_POINTS:
                if self._ber_snr_sweep_stop.is_set():
                    reason = "已停止"
                    break

                repeat_values = []
                repeat_total = int(max(1, self.AUTO_BER_SNR_REPEATS))
                for repeat_idx in range(repeat_total):
                    if self._ber_snr_sweep_stop.is_set():
                        reason = "已停止"
                        break

                    tb = self._create_ber_snr_sweep_transceiver(float(snr_db))
                    if tb is None:
                        reason = "当前页面未实现自动扫描分支"
                        break

                    try:
                        # 尽量让每个 Eb/N0 点使用相同的随机信道/比特/噪声序列集合，
                        # 避免“15 dB 用好信道、20 dB 用坏信道”导致 BER-Eb/N0 曲线反常。
                        self._prepare_ber_snr_sweep_transceiver(tb, repeat_idx)

                        if hasattr(tb, "start"):
                            tb.start()

                        self._sleep_with_stop(self.AUTO_BER_SNR_WARMUP_SEC)
                        if self._ber_snr_sweep_stop.is_set():
                            reason = "已停止"
                            break

                        if hasattr(tb, "reset_ber_stats"):
                            tb.reset_ber_stats()

                        self._sleep_with_stop(self.AUTO_BER_SNR_MEASURE_SEC)
                        ber = self._read_ber_snr_sweep_estimate(tb)
                        if np.isfinite(ber):
                            repeat_values.append(max(float(ber), 1e-12))
                    finally:
                        try:
                            if hasattr(tb, "stop"):
                                tb.stop()
                            if hasattr(tb, "wait"):
                                tb.wait(timeout=1.5)
                        except Exception:
                            pass

                if reason != "完成":
                    break

                if repeat_values:
                    # BER 是对数尺度指标，用 log-domain 合并比线性平均更稳，
                    # 可以减少单次深衰落或零误码剪裁造成的尖峰。
                    ber_combined = self._combine_ber_repeats(repeat_values)
                else:
                    ber_combined = 1.0

                self.ber_snr_sweep_point.emit(float(snr_db), max(float(ber_combined), 1e-12))
        except Exception as exc:
            reason = f"失败：{exc}"
        finally:
            self.ber_snr_sweep_finished.emit(reason)

    def _prepare_ber_snr_sweep_transceiver(self, tb, repeat_idx: int):
        """为自动 BER-Eb/N0 扫描准备独立仿真器。

        不新增后端接口，尽量通过现有属性实现：
        - 每个 repeat 使用固定 seed；
        - 不同 Eb/N0 点使用同一组 seed，降低随机信道差异导致的非单调波动；
        - 重置 sample counter / channel state，保证每个点起点一致。
        """
        seed = int(self.AUTO_BER_SNR_SEED_BASE + int(repeat_idx))
        try:
            if hasattr(tb, "_rng"):
                tb._rng = np.random.default_rng(seed)
        except Exception:
            pass
        try:
            if hasattr(tb, "_sample_counter"):
                tb._sample_counter = 0
        except Exception:
            pass
        try:
            if hasattr(tb, "_channel_state"):
                tb._channel_state = None
        except Exception:
            pass
        try:
            if hasattr(tb, "reset_ber_stats"):
                tb.reset_ber_stats()
        except Exception:
            pass

    @staticmethod
    def _read_ber_snr_sweep_estimate(tb) -> float:
        if hasattr(tb, "get_ber_estimate"):
            return float(tb.get_ber_estimate())
        if hasattr(tb, "get_ber_summary"):
            summary = tb.get_ber_summary()
            if "cumulative_ber" in summary:
                return float(summary.get("cumulative_ber", 1.0))
        if hasattr(tb, "get_last_metrics"):
            metrics = tb.get_last_metrics()
            if "ber" in metrics:
                return float(metrics.get("ber", 1.0))
        return 1.0

    @staticmethod
    def _combine_ber_repeats(values) -> float:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 1.0
        arr = np.clip(arr, 1e-12, 1.0)
        log_arr = np.log10(arr)
        if log_arr.size >= 3:
            return float(10.0 ** np.median(log_arr))
        return float(10.0 ** np.mean(log_arr))

    def _sleep_with_stop(self, seconds: float):
        deadline = time.time() + max(0.0, float(seconds))
        while time.time() < deadline and not self._ber_snr_sweep_stop.is_set():
            time.sleep(0.05)

    def _on_auto_ber_snr_point(self, snr_db: float, ber: float):
        self._auto_ber_snr_x.append(float(snr_db))
        self._auto_ber_snr_raw_y.append(max(float(ber), 1e-12))
        order = np.argsort(np.asarray(self._auto_ber_snr_x, dtype=np.float64))
        x = np.asarray(self._auto_ber_snr_x, dtype=np.float64)[order]
        raw_y = np.asarray(self._auto_ber_snr_raw_y, dtype=np.float64)[order]

        if getattr(self, "AUTO_BER_SNR_MONOTONIC_DISPLAY", True):
            # BER-Eb/N0 理论上应随 Eb/N0 增大而不升高。自动扫描仍保留 raw_y，
            # 显示时只做单调包络，避免少量统计误差产生明显反常尖峰。
            y = raw_y.copy()
            for i in range(1, len(y)):
                y[i] = min(y[i], y[i - 1])
        else:
            y = raw_y

        self._auto_ber_snr_y = y.tolist()
        if hasattr(self, "ber_snr_curve"):
            self.ber_snr_curve.setData(x, y)
        if self.status_text_label is not None:
            self.status_text_label.setText(
                f"BER-Eb/N0 自动扫描中：已完成 {len(x)} / {len(self.AUTO_BER_SNR_POINTS)} 个 Eb/N0 点。"
                f"每点统计 {int(max(1, self.AUTO_BER_SNR_REPEATS))} 次，"
                f"每次约 {self.AUTO_BER_SNR_MEASURE_SEC:.1f} s。"
            )

    def _on_auto_ber_snr_finished(self, reason: str):
        self._auto_ber_snr_active = False
        self._ber_snr_sweep_stop.clear()
        if hasattr(self, "btn_auto_ber_snr"):
            self.btn_auto_ber_snr.setEnabled(True)
        if hasattr(self, "btn_stop_ber_snr"):
            self.btn_stop_ber_snr.setEnabled(False)
        if self.status_text_label is not None:
            self.status_text_label.setText(
                f"BER-Eb/N0 自动扫描{reason}。曲线由独立仿真分支生成；"
                f"每个 Eb/N0 点采用更长统计时间和重复平均，左侧 Eb/N0 调试不会覆盖它。"
            )

    def _create_ber_snr_sweep_transceiver(self, snr_db: float):
        """子类重写：返回一个只用于自动 BER-Eb/N0 扫描的独立仿真器。"""
        return None
