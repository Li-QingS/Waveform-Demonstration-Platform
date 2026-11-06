"""AFDM in satellite-terrestrial channels: measured link and reproducible studies."""

from dataclasses import asdict, replace
import json
import threading

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, QRectF, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHeaderView, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget, QSizePolicy,
)

from .base_waveform_tab import BaseWaveformTab
from .afdm_ntn_workers import AfdmNtnRunner
from .ui_utils import compute_spectrum
from ..simulation.afdm_ntn import AfdmNtnConfig, config_summary, run_study, suggested_chirp
from ..simulation.leo_ntn_channel import absolute_doppler_shift_hz, get_profile


COLORS = {'training_ls': (0, 114, 189), 'ideal': (126, 47, 142),
          'uncompensated': (217, 83, 25)}
NAMES = {'training_ls': '帧首训练 LS', 'ideal': '理想块内 CSI', 'uncompensated': '关闭预补偿 + LS'}


class AfdmTab(BaseWaveformTab):
    study_point = pyqtSignal(int, object)
    study_finished = pyqtSignal(int, str)

    def __init__(self):
        super().__init__('AFDM')
        self.tb = None
        self._study_token = 0
        self._study_stop = threading.Event()
        self._study_results = []
        self._study_config = None
        self._study_axis = 'ebn0'
        self._study_values = []
        self._study_frames = 0
        self._study_status = '尚未运行'
        self._last_live_snapshot = None
        self._live_config = None
        self._last_frame = 0
        self._live_x, self._live_y, self._live_upper = [], [], []
        self._build_controls()
        self._build_plots()
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(160)
        self.update_timer.timeout.connect(self._refresh_plots)
        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.study_point.connect(self._on_study_point)
        self.study_finished.connect(self._on_study_finished)
        self._apply_preset(0)
        self._set_ui_enabled(True)
        self._update_afdm_info_panel(False)

    @staticmethod
    def _combo(items):
        box = QComboBox()
        for text, data in items:
            box.addItem(text, data)
        return box

    @staticmethod
    def _number(lo, hi, value, decimals=2, suffix=''):
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSuffix(suffix)
        return spin

    def _page(self, title):
        content = QWidget()
        form = QFormLayout(content)
        form.setContentsMargins(8, 8, 8, 8)
        form.setVerticalSpacing(7)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        self.parameter_tabs.addTab(scroll, title)
        return form

    def _build_controls(self):
        self.param_group.hide()
        self.controls_scroll.hide()
        self.parameter_tabs = QTabWidget()
        self.controls_rail_layout.addWidget(self.parameter_tabs, 1)
        self.controls_panel.setMinimumWidth(330)
        self.controls_panel.setMaximumWidth(390)
        link = self._page('星地场景')
        self.preset_combo = self._combo([
            ('LEO S 波段 · LOS', 's_los'), ('LEO S 波段 · NLOS', 's_nlos'),
            ('LEO Ka 波段 · 高动态', 'ka'), ('AWGN 基准', 'awgn'),
        ])
        self.profile_combo = self._combo([(name, name) for name in (
            'NTN-TDLC5-200', 'NTN-TDLA100-200', 'NTN-TDLC5-1200', 'AWGN')])
        self.fc_spin = self._number(.1, 40, 2, 2, ' GHz')
        self.velocity_spin = self._number(0, 40000, 28080, 0, ' km/h')
        self.radial_spin = self._number(0, 1, .5, 3)
        self.compensation_spin = self._number(0, 100, 99.9, 3, ' %')
        self.cfo_spin.setRange(-200000, 200000)
        self.cfo_spin.setValue(0)
        self.doppler_spin.setValue(200)
        self.doppler_spin.setMaximum(200000)
        self.delay_scale_spin = self._number(.01, 1000, 1, 2, ' 倍')
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        self.seed_spin.setValue(42)
        for text, widget in (
            ('场景预设', self.preset_combo), ('信道剖面', self.profile_combo),
            ('载频', self.fc_spin), ('相对速度', self.velocity_spin),
            ('径向投影', self.radial_spin), ('公共频移预补偿', self.compensation_spin),
            ('额外残余 CFO', self.cfo_spin), ('散射多普勒扩展', self.doppler_spin),
            ('时延缩放', self.delay_scale_spin), ('随机种子', self.seed_spin),
        ):
            link.addRow(text, widget)
        self.compensation_spin.setToolTip('补偿可预测公共轨道频移的比例；不消除散射多普勒扩展。')
        self.delay_scale_spin.setToolTip('1 倍使用标准剖面时延；其它倍数为工程压力测试，保留分数采样时延。')
        self.radial_spin.setToolTip('相对速度沿视线方向的比例，不是卫星仰角。')
        wave = self._page('波形与接收')
        self.fft_combo = self._combo([(str(n), n) for n in (32, 64, 128, 256)])
        self.fft_combo.setCurrentIndex(1)
        self.spacing_combo = self._combo([(f'{n} kHz', n * 1000.) for n in (15, 30, 60)])
        self.spacing_combo.setCurrentIndex(1)
        self.cp_spin = QSpinBox(); self.cp_spin.setRange(0, 63); self.cp_spin.setValue(16)
        self.mod_combo = self._combo([(s, s) for s in ('QPSK', '16QAM', '64QAM')])
        self.blocks_spin = QSpinBox(); self.blocks_spin.setRange(1, 32); self.blocks_spin.setValue(8)
        self.prefix_combo = self._combo([('Chirp 周期前缀 CPP', 'cpp'), ('普通循环前缀 CP', 'cp')])
        self.c1_spin = self._number(0, .5, 3/(2*64), 10)
        self.c2_spin = self._number(0, .5, 1/(64**2*np.sqrt(2)), 10)
        self.c1_spin.setSingleStep(.0001)
        self.c2_spin.setSingleStep(.00001)
        self.btn_suggest_chirp = QPushButton('应用场景建议 chirp')
        self.btn_suggest_chirp.setToolTip('按当前 FFT 与名义频移范围建议 c1/c2；点击后才应用，不是性能最优保证或在线自适应。')
        self.estimator_combo = self._combo([('训练 LS 静态矩阵', 'training_ls'), ('理想块内 CSI 参考', 'ideal')])
        self.equalizer_combo = self._combo([('完整矩阵 MMSE', 'mmse'), ('完整矩阵 ZF', 'zf')])
        self.observe_combo = self._combo([('均衡后', 'post'), ('均衡前', 'pre')])
        for text, widget in (
            ('Eb/N0', self.snr_spin), ('DAFT 点数 M', self.fft_combo),
            ('每帧数据块数', self.blocks_spin),
            ('子载波间隔', self.spacing_combo), ('前缀采样数', self.cp_spin),
            ('数据前缀', self.prefix_combo), ('时域 chirp c1', self.c1_spin),
            ('DAFT 域 chirp c2', self.c2_spin), ('参数建议', self.btn_suggest_chirp),
            ('调制', self.mod_combo), ('信道估计', self.estimator_combo),
            ('均衡器', self.equalizer_combo), ('星座观察', self.observe_combo),
        ):
            wave.addRow(text, widget)
        self.fft_combo.setToolTip('每个数据块的 DAFT 点数；完整矩阵均衡的计算量随点数增加。')
        self.blocks_spin.setToolTip('增大数据块数会延长帧长、改变训练占比，并可能加重帧首估计老化。')
        convention = ('本研究采用标准相位 exp(-j·2π·c·n²)，不再额外除以 M。\n'
                      '旧通用模型使用 exp(-j·2π·c_old·n²/M)，对应 c=c_old/M；两种约定的数值不能直接照搬。')
        self.c1_spin.setToolTip(convention)
        self.c2_spin.setToolTip(convention)
        self.prefix_combo.setToolTip('CPP 对复制的块尾添加 chirp 周期相位校正；普通 CP 供对照。帧首 OFDM 训练固定使用普通 CP。')
        self.estimator_combo.setToolTip('帧首全带宽训练 LS 构造静态循环信道近似矩阵并加入 CPP 包绕校正，整帧保持；分数时延或前缀不足时可能模型失配。理想 CSI 知道块内时变矩阵，仍不反演跨块 ISI。真实载荷只用于计分。')
        self.equalizer_combo.setToolTip('时域完整矩阵 ZF/MMSE，再作 DAFT；与 DAFT 域矩阵检测等价。MMSE 使用已知仿真噪声方差。')
        assumption = QLabel('SISO 未编码 AFDM；IDAFT 发送、DAFT 接收。已知帧定时，帧首训练估计整帧保持；理想 CSI 为块内矩阵参考。')
        assumption.setWordWrap(True)
        wave.addRow(assumption)
        study = self._page('性能研究')
        self.axis_combo = self._combo([
            ('Eb/N0 (dB)', 'ebn0'), ('额外残余 CFO (Hz)', 'residual_cfo'),
            ('子载波间隔 (kHz)', 'spacing'), ('时延缩放 (倍)', 'delay_scale'),
            ('散射多普勒扩展 (Hz)', 'doppler'), ('每帧数据块数', 'blocks'),
            ('时域 chirp c1', 'c1'), ('DAFT 域 chirp c2', 'c2'),
        ])
        self.scan_values_edit = QLineEdit('0, 4, 8, 12, 16, 20')
        self.scan_frames_spin = QSpinBox(); self.scan_frames_spin.setRange(1, 500); self.scan_frames_spin.setValue(40)
        self.export_button = QPushButton('导出研究结果 (JSON)')
        self.export_button.setEnabled(False)
        study.addRow('研究变量', self.axis_combo)
        study.addRow('扫描取值', self.scan_values_edit)
        study.addRow('每点帧数', self.scan_frames_spin)
        study.addRow(self.export_button)
        note = QLabel('三条曲线使用相同发送比特、信道和噪声样本。BER 保留原始计数；零误码用三角标记显示 95% 统计上界。')
        note.setWordWrap(True)
        study.addRow(note)
        self.study_progress = QLabel('尚未运行性能研究')
        self.study_progress.setWordWrap(True)
        study.addRow(self.study_progress)
        self.btn_auto_ber_snr.setText('开始性能扫描')
        self.btn_stop_ber_snr.setText('停止扫描')
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        self.fft_combo.currentIndexChanged.connect(self._on_fft_changed)
        self.spacing_combo.currentIndexChanged.connect(lambda *_: self._update_afdm_info_panel(self.tb is not None))
        self.mod_combo.currentIndexChanged.connect(lambda *_: self._update_afdm_info_panel(self.tb is not None))
        self.blocks_spin.valueChanged.connect(lambda *_: self._update_afdm_info_panel(self.tb is not None))
        self.prefix_combo.currentIndexChanged.connect(lambda *_: self._update_afdm_info_panel(self.tb is not None))
        self.estimator_combo.currentIndexChanged.connect(lambda *_: self._update_afdm_info_panel(self.tb is not None))
        self.equalizer_combo.currentIndexChanged.connect(lambda *_: self._update_afdm_info_panel(self.tb is not None))
        self.btn_suggest_chirp.clicked.connect(self._apply_suggested_chirp)
        self.axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        self.observe_combo.currentIndexChanged.connect(lambda: self._refresh_plots(force=True))
        self.export_button.clicked.connect(self._export_study)
        for widget in (self.snr_spin, self.fc_spin, self.velocity_spin, self.radial_spin,
                       self.compensation_spin, self.cfo_spin, self.doppler_spin,
                       self.delay_scale_spin, self.cp_spin, self.c1_spin, self.c2_spin, self.seed_spin):
            widget.valueChanged.connect(lambda *_: self._update_afdm_info_panel(self.tb is not None))

    def _style_plot(self, plot):
        plot.setBackground((250, 250, 250))
        plot.showGrid(x=True, y=True, alpha=.35)
        for name in ('left', 'bottom'):
            plot.getAxis(name).setPen(pg.mkPen(60, 60, 60))
            plot.getAxis(name).setTextPen(pg.mkPen(60, 60, 60))
        plot.getViewBox().setBorder(pg.mkPen(225, 225, 225))
        plot.setMinimumSize(230, 165)

    def _build_plots(self):
        layout = self.plot_panel.layout()
        grid = layout.takeAt(0).layout(); grid.setParent(None)
        grid.setVerticalSpacing(6)
        info = layout.takeAt(0).widget(); info.setParent(None)
        grid.removeWidget(self.ber_snr_plot)
        self.ber_snr_plot.setParent(None)
        self.channel_plot = pg.PlotWidget(title='估计 DAFT 信道 · 相对峰值 dB')
        self.channel_plot.setLabel('bottom', '发送 DAFT 格点')
        self.channel_plot.setLabel('left', '接收 DAFT 格点')
        self.channel_image = pg.ImageItem(axisOrder='row-major')
        self.channel_image.setLookupTable(pg.ColorMap(
            [0., .45, .75, 1.], [(250, 250, 250), (0, 114, 189), (237, 177, 32), (217, 83, 25)]
        ).getLookupTable(0., 1., 256))
        self.channel_plot.addItem(self.channel_image)
        self.channel_plot.setToolTip('第一数据块的有效 DAFT 信道矩阵功率，相对本矩阵峰值，范围 -40 到 0 dB。\n训练 LS 显示估计矩阵；理想 CSI 显示真实块内矩阵。坐标是 DAFT 输入/输出格点，不是物理时延—多普勒路径。')
        grid.addWidget(self.channel_plot, 1, 1)
        live = QWidget(); live_layout = QVBoxLayout(live)
        live_layout.setContentsMargins(4, 4, 4, 4)
        live_layout.addLayout(grid, 4); live_layout.addWidget(info, 1)
        self.plot_tabs = QTabWidget()
        self.plot_tabs.addTab(live, '实时星地链路')
        research = QWidget(); body = QVBoxLayout(research)
        body.setContentsMargins(4, 4, 4, 4)
        plots = QHBoxLayout()
        self.ber_snr_plot.setTitle('实测 BER 与统计上界')
        self.ber_snr_plot.setLabel('left', 'BER')
        self.ber_snr_plot.setLogMode(y=True)
        self.study_evm_plot = pg.PlotWidget(title='解调质量 (EVM)')
        self.study_evm_plot.setLabel('left', 'EVM RMS', units='%')
        self._study_curves, self._study_zero_curves, self._study_evm_curves = {}, {}, {}
        for plot in (self.ber_snr_plot, self.study_evm_plot):
            plot.addLegend(offset=(4, 4), labelTextSize='8pt', verSpacing=-3)
            plots.addWidget(plot)
        for key, color in COLORS.items():
            self._study_curves[key] = self.ber_snr_plot.plot(pen=pg.mkPen(color, width=2), symbol='o', symbolSize=5, name=NAMES[key])
            self._study_zero_curves[key] = self.ber_snr_plot.plot(pen=None, symbol='t', symbolSize=8, symbolBrush=color)
            self._study_evm_curves[key] = self.study_evm_plot.plot(pen=pg.mkPen(color, width=2), symbol='o', symbolSize=5, name=NAMES[key])
        body.addLayout(plots, 1)
        self.result_table = QTableWidget(0, 7)
        self.result_table.setHorizontalHeaderLabels(['扫描值', 'LS BER', '理想 CSI BER', '未补偿 BER', 'LS 错误/比特', 'LS 95% 区间', 'LS EVM (%)'])
        for column, explanation in enumerate((
            '研究变量的实际取值；子载波间隔以 kHz 显示。',
            '帧首训练 LS，估计整帧保持；原始 BER = 错误比特 / 总比特。',
            '理想块内时变 CSI + 完整矩阵均衡；不反演跨块 ISI。',
            '关闭公共轨道预补偿，仍使用训练 LS 和所选均衡器。',
            '训练 LS 的实际错误比特数 / 总信息比特数。',
            '训练 LS 的双侧 95% Wilson 计数区间，非独立衰落覆盖概率区间。',
            '训练 LS 的参考能量加权 RMS EVM。',
        )):
            self.result_table.horizontalHeaderItem(column).setToolTip(explanation)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setMaximumHeight(185)
        self.result_table.verticalHeader().setDefaultSectionSize(24)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        body.addWidget(self.result_table)
        self.study_note = QLabel('Eb/N0 计入前缀与训练能量。训练 LS 整帧保持；理想 CSI 使用块内时变矩阵，均不反演跨块 ISI。')
        self.study_note.setWordWrap(True)
        self.study_note.setMinimumWidth(0)
        self.study_note.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        body.addWidget(self.study_note)
        self.plot_tabs.addTab(research, '性能研究')
        layout.addWidget(self.plot_tabs)
        self.spectrum_plot.setTitle('接收频谱')
        self.constellation_plot.setTitle('接收星座')
        self.constellation_plot.setXRange(-2, 2); self.constellation_plot.setYRange(-2, 2)
        self.ber_plot.setTitle('累计实测 BER')
        self.ber_plot.setLabel('left', 'BER'); self.ber_plot.setLogMode(y=True)
        self.ber_curve.setPen(pg.mkPen(COLORS['training_ls'], width=2))
        self.ber_curve.setSymbolPen(pg.mkPen(COLORS['training_ls']))
        self.ber_curve.setSymbolBrush(pg.mkBrush(COLORS['training_ls']))
        self.spectrum_curve.setPen(pg.mkPen(COLORS['training_ls'], width=2))
        self.constellation_scatter.setPen(pg.mkPen(COLORS['uncompensated'], width=.8))
        self.constellation_scatter.setBrush(pg.mkBrush(*COLORS['uncompensated'], 160))
        self.live_zero_curve = self.ber_plot.plot(pen=None, symbol='t', symbolSize=7, symbolBrush=COLORS['training_ls'])
        for plot in (self.spectrum_plot, self.constellation_plot, self.ber_plot, self.channel_plot,
                     self.ber_snr_plot, self.study_evm_plot):
            self._style_plot(plot)
        self.rename_info_rows({'同步度量': 'EVM RMS', 'CFO估计': '残余 CFO',
                               'FFT/CP': 'DAFT / 前缀',
                               '数据符号': 'chirp c1 / c2', '星座观察': '前缀余量'})

    def _apply_preset(self, index):
        if not hasattr(self, 'fc_spin'):
            return
        key = self.preset_combo.itemData(index)
        profile, fc, doppler = {
            's_los': ('NTN-TDLC5-200', 2., 200.),
            's_nlos': ('NTN-TDLA100-200', 2., 200.),
            'ka': ('NTN-TDLC5-1200', 20., 1200.),
            'awgn': ('AWGN', 2., 0.),
        }[key]
        self.profile_combo.setCurrentText(profile)
        self.fc_spin.setValue(fc)
        self.velocity_spin.setValue(0 if key == 'awgn' else 28080)
        self.spacing_combo.setCurrentIndex(2 if key == 'ka' else 1)
        self.doppler_spin.setValue(doppler)
        self.compensation_spin.setValue(100. if key == 'awgn' else 99.9)
        self.cfo_spin.setValue(0)
        self.delay_scale_spin.setValue(1)
        self._update_afdm_info_panel(self.tb is not None)

    def _on_profile_changed(self):
        name = self.profile_combo.currentData()
        self.doppler_spin.setValue(0 if name == 'AWGN' else get_profile(name).maximum_doppler_hz)
        self._update_afdm_info_panel(self.tb is not None)

    def _on_fft_changed(self):
        self.cp_spin.setMaximum(int(self.fft_combo.currentData()) - 1)
        self._update_afdm_info_panel(self.tb is not None)

    def _apply_suggested_chirp(self):
        """Apply advice only after an explicit click, keeping experiments frozen."""
        if self.tb is not None or self._auto_ber_snr_active:
            return
        try:
            config = self._collect_config()
            advice = suggested_chirp(config)
            # Do not silently clamp out-of-range advice at the widget limits.
            replace(config, c1=advice['c1'], c2=advice['c2']).validated()
            self.c1_spin.setValue(advice['c1'])
            self.c2_spin.setValue(advice['c2'])
            self._update_afdm_info_panel(False)
        except ValueError as exc:
            QMessageBox.information(self, 'Chirp 参数建议', str(exc))

    def _on_axis_changed(self):
        axis = self.axis_combo.currentData()
        self.scan_values_edit.setText({
            'ebn0': '0, 4, 8, 12, 16, 20', 'residual_cfo': '0, 300, 1500, 3000, 6000, 15000',
            'spacing': '15, 30, 60', 'delay_scale': '1, 5, 10, 25, 50, 100',
            'doppler': '0, 200, 600, 1200, 2400, 4800', 'blocks': '1, 4, 8, 16, 32',
            'c1': '0, 0.0078125, 0.0234375, 0.05',
            'c2': '0, 0.0001726335, 0.001, 0.01',
        }[axis])

    def _collect_config(self):
        return AfdmNtnConfig(
            profile=self.profile_combo.currentData(), fft_len=int(self.fft_combo.currentData()),
            cp_len=self.cp_spin.value(), subcarrier_spacing_hz=float(self.spacing_combo.currentData()),
            modulation=self.mod_combo.currentText(), ebn0_db=self.snr_spin.value(),
            symbols_per_frame=self.blocks_spin.value(),
            carrier_frequency_hz=self.fc_spin.value() * 1e9, velocity_kmh=self.velocity_spin.value(),
            radial_projection=self.radial_spin.value(), compensation_ratio=self.compensation_spin.value() / 100.,
            residual_cfo_hz=self.cfo_spin.value(), doppler_spread_hz=self.doppler_spin.value(),
            delay_scale=self.delay_scale_spin.value(), estimator=self.estimator_combo.currentData(),
            equalizer=self.equalizer_combo.currentData(), seed=self.seed_spin.value(),
            c1=self.c1_spin.value(), c2=self.c2_spin.value(), prefix=self.prefix_combo.currentData(),
        ).validated()

    def _set_ui_enabled(self, controls_enabled):
        busy = not controls_enabled or self._auto_ber_snr_active or self.tb is not None
        self.btn_start.setEnabled(not busy)
        self.btn_stop.setEnabled(self.tb is not None)
        self.btn_auto_ber_snr.setEnabled(not busy)
        self.btn_stop_ber_snr.setEnabled(self._auto_ber_snr_active)
        if hasattr(self, 'parameter_tabs'):
            for widget in self.parameter_tabs.findChildren((QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QPushButton)):
                widget.setEnabled(not busy)
        self.observe_combo.setEnabled(True)
        self.export_button.setEnabled(bool(self._study_results) and not busy)

    def _on_start_clicked(self):
        try:
            config = self._collect_config()
            self._stop_auto_ber_snr_sweep(strict=True)
            self._stop_live(strict=True)
            self._tasks_suspended = False
            runner = AfdmNtnRunner(config)
            runner.start()
            self.tb = runner
            self._live_config = config
            self._last_live_snapshot = None
            self._last_frame = 0
            self._live_x, self._live_y, self._live_upper = [], [], []
            self.ber_curve.clear(); self.live_zero_curve.clear()
            self.spectrum_curve.clear(); self.constellation_scatter.clear(); self.channel_image.clear()
            self.update_timer.start()
            self.plot_tabs.setCurrentIndex(0)
            self._set_ui_enabled(False)
            self._update_afdm_info_panel(True)
        except Exception as exc:
            self._set_ui_enabled(self.tb is None)
            self.update_info_panel(status_text=f'仿真未启动：{exc}')

    def _on_stop_clicked(self):
        return self._stop_live()

    def _stop_live(self, strict=False):
        backend = self.tb
        if backend is not None:
            try:
                backend.stop()
                backend.wait(timeout=1.5)
            except Exception as exc:
                self._set_ui_enabled(False)
                self.update_info_panel(status_text=f'停止尚未完成：{exc}')
                if strict:
                    raise
                return False
            self._refresh_plots(force=True)
            self.tb = None
        self.update_timer.stop()
        self._set_ui_enabled(True)
        if backend is not None:
            self.update_info_panel(metrics={'运行状态': '已停止（保留结果）'})
        return True

    def suspend_tasks(self):
        """Navigation succeeds only after both workers have relinquished ownership."""
        self._stop_auto_ber_snr_sweep(strict=True)
        self._stop_live(strict=True)
        self._tasks_suspended = True

    def resume_tasks(self):
        self._tasks_suspended = False
        self._set_ui_enabled(self.tb is None)

    def closeEvent(self, event):
        try:
            self.suspend_tasks()
        except Exception:
            event.ignore()
            return
        event.accept()

    @staticmethod
    def _upper_limit(errors, bits):
        if bits <= 0:
            return float('nan')
        z = 1.95996398454; p = errors / bits
        return (p + z*z/(2*bits) + z*np.sqrt(p*(1-p)/bits+z*z/(4*bits*bits))) / (1+z*z/bits)

    def _refresh_plots(self, force=False):
        # A stopped experiment remains available for pre/post observation.
        frame = self.tb.snapshot() if self.tb is not None and hasattr(self.tb, 'snapshot') else self._last_live_snapshot
        if not frame:
            return
        if 'error' in frame:
            error = frame['error']
            if self.tb is not None:
                backend = self.tb
                try:
                    backend.stop(); backend.wait(timeout=1.5)
                except Exception:
                    self._set_ui_enabled(False)
                    self.update_info_panel(status_text=f'仿真失败，正在停止：{error}')
                    return
                self.tb = None
                self.update_timer.stop()
                self._set_ui_enabled(True)
            self._last_live_snapshot = {k: v for k, v in frame.items() if k != 'error'}
            self.update_info_panel(status_text=f'仿真失败：{frame["error"]}')
            return
        number = frame['frames']
        if number == self._last_frame and not force:
            return
        self._last_live_snapshot = frame
        config = self.tb.config if self.tb is not None else self._live_config
        samples = frame['rx_samples']
        fs = config.fft_len * config.subcarrier_spacing_hz
        freq, psd = compute_spectrum(samples, fs)
        self.spectrum_curve.setData(freq, psd)
        points = np.asarray(frame['pre_eq_symbols'] if self.observe_combo.currentData() == 'pre' else frame['rx_symbols']).reshape(-1)[:1024]
        self.constellation_scatter.setData(x=points.real, y=points.imag)
        self.constellation_plot.setTitle('均衡前 DAFT 星座' if self.observe_combo.currentData() == 'pre' else '均衡后 DAFT 星座')
        ideal = config.estimator == 'ideal'
        matrices = frame['effective_channel_true'] if ideal else frame['effective_channel_est']
        channel = np.asarray(matrices)[0]
        power = np.abs(channel) ** 2
        relative_db = 10 * np.log10(np.maximum(power / max(float(power.max()), 1e-30), 1e-4))
        self.channel_image.setImage(relative_db, autoLevels=False, levels=(-40., 0.))
        self.channel_image.setRect(QRectF(-.5, -.5, channel.shape[1], channel.shape[0]))
        self.channel_plot.setXRange(-.5, channel.shape[1]-.5, padding=0)
        self.channel_plot.setYRange(-.5, channel.shape[0]-.5, padding=0)
        self.channel_plot.setTitle(('真实块内' if ideal else '训练估计') + ' DAFT 信道 · 相对峰值 dB')
        if number != self._last_frame:
            self._last_frame = number
            self._live_x.append(frame['elapsed_s'])
            self._live_y.append(frame['cumulative_ber'])
            self._live_upper.append(self._upper_limit(frame['cumulative_errors'], frame['cumulative_bits']))
            self._live_x, self._live_y, self._live_upper = self._live_x[-240:], self._live_y[-240:], self._live_upper[-240:]
        x, y, upper = map(np.asarray, (self._live_x, self._live_y, self._live_upper))
        if np.any(y > 0):
            self.ber_curve.setData(x, np.where(y > 0, y, np.nan), connect='finite')
        else:
            self.ber_curve.setData([], [])
        zero = y == 0
        self.live_zero_curve.setData(x[zero], upper[zero])
        self.live_zero_curve.setToolTip('三角标记表示零误码样本的 95% Wilson 上界；原始 BER 仍为 0。')
        self.update_info_panel(metrics={
            '运行状态': f'{"运行中" if self.tb is not None else "已停止"} · {number} 帧', 'Eb/N0': f'{config.ebn0_db:.1f} dB',
            'BER': f'{frame["cumulative_ber"]:.3g} ({frame["cumulative_errors"]}/{frame["cumulative_bits"]})',
            'FER': f'{frame["cumulative_fer"]:.3g}', '同步度量': f'{frame["evm_percent"]:.2f} %',
        })

    def _update_afdm_info_panel(self, running=False):
        if not hasattr(self, 'profile_combo') or not hasattr(self, 'study_note'):
            return
        try:
            config = self._collect_config()
            summary = config_summary(config)
        except ValueError:
            return
        orbital = absolute_doppler_shift_hz(config.carrier_frequency_hz, config.velocity_kmh, config.radial_projection)
        residual = orbital * (1-config.compensation_ratio) + config.residual_cfo_hz
        cp_s = config.cp_len / (config.fft_len * config.subcarrier_spacing_hz)
        profile = None if config.profile == 'AWGN' else get_profile(config.profile)
        maximum_delay = 0. if profile is None else max(profile.delays_ns) * config.delay_scale * 1e-9
        margin = cp_s - maximum_delay
        stale = self._last_live_snapshot is not None and self._live_config != config
        self.update_info_panel(
            metrics={'运行状态': '运行中' if running else '待运行', 'Eb/N0': f'{config.ebn0_db:.1f} dB', 'CFO估计': f'{residual:.2f} Hz'},
            config={'波形': f'AFDM · {config.profile}', 'FFT/CP': f'{config.fft_len} / {config.prefix.upper()} {config.cp_len}',
                    '调制': config.modulation, '采样率': f'{config.fft_len*config.subcarrier_spacing_hz/1e6:.2f} MHz',
                    '数据符号': f'{config.c1:.5g} / {config.c2:.5g}', '星座观察': f'{margin*1e6:.3f} μs'},
            status_text=(f'公共轨道频移 {orbital/1000:.2f} kHz；预补偿后公共残余 {residual:.2f} Hz。\n'
                         f'散射扩展 {config.doppler_spread_hz:.0f} Hz；前缀 {cp_s*1e6:.3f} μs，最大超额时延 {maximum_delay*1e6:.3f} μs。\n'
                         + ('前缀不足，可能出现跨块 ISI。\n' if margin < 0 else '')
                         + ('待运行配置已变化，图中保留上次实验；请重新运行。\n' if stale else '')
                         + f'{config.symbols_per_frame} 个数据块 + 1 个训练块；完整矩阵 {config.equalizer.upper()}，已知定时。\n'
                         + '标准 chirp 相位不除以 M；参数建议需手动应用，不是全分集保证。'),
        )
        self.status_text_label.setToolTip(json.dumps(summary, ensure_ascii=False, indent=2))
        if self._last_live_snapshot is None:
            self.channel_plot.setXRange(-.5, config.fft_len-.5, padding=0)
            self.channel_plot.setYRange(-.5, config.fft_len-.5, padding=0)

    def _start_auto_ber_snr_sweep(self):
        if self._auto_ber_snr_active:
            return
        try:
            config = self._collect_config()
            text = self.scan_values_edit.text().replace('，', ',')
            values = [float(s.strip()) for s in text.split(',') if s.strip()]
            if not 1 <= len(values) <= 31 or not np.all(np.isfinite(values)):
                raise ValueError('请输入 1–31 个有限数值，以逗号分隔。')
            if len(set(values)) != len(values):
                raise ValueError('扫描取值不能重复。')
            axis = self.axis_combo.currentData()
            if axis == 'spacing':
                values = [v * 1000 for v in values]
            if axis in ('spacing', 'delay_scale') and min(values) <= 0:
                raise ValueError('子载波间隔和时延缩放必须大于 0。')
            if axis == 'blocks':
                if any(not value.is_integer() for value in values):
                    raise ValueError('每帧数据块数必须为 1–32 的整数。')
                values = [int(value) for value in values]
            field = {'ebn0': 'ebn0_db', 'residual_cfo': 'residual_cfo_hz',
                     'spacing': 'subcarrier_spacing_hz', 'delay_scale': 'delay_scale',
                     'doppler': 'doppler_spread_hz', 'blocks': 'symbols_per_frame',
                     'c1': 'c1', 'c2': 'c2'}[axis]
            for value in values:
                point = replace(config, **{field: value}).validated()
                # The third curve must be physically valid before clearing
                # any previous result, even if the selected live link uses
                # almost complete orbital pre-compensation.
                replace(point, compensation_ratio=0.0).validated()
        except ValueError as exc:
            QMessageBox.information(self, '扫描参数', str(exc))
            return
        try:
            self._stop_auto_ber_snr_sweep(strict=True)
            self._stop_live(strict=True)
        except Exception as exc:
            self.study_progress.setText(f'研究未启动：{exc}')
            return
        self._tasks_suspended = False
        self._study_token += 1
        token = self._study_token
        self._study_stop = threading.Event()
        stop = self._study_stop
        self._study_config, self._study_axis, self._study_values = config, axis, values
        self._study_frames = self.scan_frames_spin.value()
        self._study_status = 'running'
        self._study_results = []
        self.result_table.setRowCount(0)
        for curve in (*self._study_curves.values(), *self._study_zero_curves.values(), *self._study_evm_curves.values()):
            curve.clear()
        self._auto_ber_snr_active = True
        self._set_ui_enabled(True)
        self.plot_tabs.setCurrentIndex(1)
        self.study_progress.setText(f'性能研究进行中：0 / {len(values)} 点')
        label = self.axis_combo.currentText()
        for plot in (self.ber_snr_plot, self.study_evm_plot):
            plot.setLabel('bottom', label)
        self.study_note.setText(
            f'{config.profile} · DAFT {config.fft_len} × {config.symbols_per_frame} 块 / {config.prefix.upper()} {config.cp_len} · {config.modulation} · {self._study_frames} 帧/点 · seed {config.seed}。'
            '三策略使用相同样本；三角标记=零误码 95% 上界，非实测非零 BER。'
            + (' 固定 FFT 比较间隔时，带宽和 CP 时长也随之变化。' if axis == 'spacing' else '')
            + (' 数据块数改变帧长、有效比特数及训练开销。' if axis == 'blocks' else '')
            + (' Chirp 扫描会改变发射波形和实际前缀能量，单位噪声流相同。' if axis in ('c1', 'c2') else '')
        )
        frames = self.scan_frames_spin.value()
        def emit_point(point):
            if not stop.is_set() and not self._tasks_suspended:
                try:
                    self.study_point.emit(token, point)
                except RuntimeError:
                    stop.set()
        def work():
            message = '已完成'
            try:
                run_study(config, axis, values, frames, stop_event=stop, on_point=emit_point)
                if stop.is_set():
                    message = '已停止，保留已完成点'
            except InterruptedError:
                message = '已停止，保留已完成点'
            except Exception as exc:
                message = f'失败：{exc}'
            if not self._tasks_suspended:
                try:
                    self.study_finished.emit(token, message)
                except RuntimeError:
                    stop.set()
        self._ber_snr_sweep_thread = threading.Thread(target=work, daemon=True, name='afdm-ntn-study')
        try:
            self._ber_snr_sweep_thread.start()
        except Exception as exc:
            self._ber_snr_sweep_thread = None
            self._auto_ber_snr_active = False
            self._study_status = f'failed: {exc}'
            self._set_ui_enabled(True)
            self.study_progress.setText(f'研究未启动：{exc}')

    def _stop_auto_ber_snr_sweep(self, strict=False):
        self._study_stop.set()
        worker = self._ber_snr_sweep_thread
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=1.7)
        if worker is not None and worker.is_alive():
            self._set_ui_enabled(False)
            self.study_progress.setText('扫描仍在停止，请稍后重试；现有任务与结果已保留。')
            if strict:
                raise TimeoutError('AFDM 性能扫描尚未退出')
            return False
        self._study_token += 1
        self._ber_snr_sweep_thread = None
        was_running = self._auto_ber_snr_active
        self._auto_ber_snr_active = False
        if was_running:
            self._study_status = 'cancelled'
        if hasattr(self, 'export_button'):
            self._set_ui_enabled(self.tb is None)
            if was_running:
                self.study_progress.setText(f'已停止 · 保留 {len(self._study_results)} 个完整点')
        return True

    def _on_study_point(self, token, point):
        if token != self._study_token or self._tasks_suspended or not self._auto_ber_snr_active:
            return
        self._study_results.append(point)
        scale = 1000. if self._study_axis == 'spacing' else 1.
        x = np.array([p['axis_value']/scale for p in self._study_results])
        for key in COLORS:
            stats = [p['strategies'][key] for p in self._study_results]
            y = np.array([s['ber'] for s in stats])
            if np.any(y > 0):
                self._study_curves[key].setData(x, np.where(y > 0, y, np.nan), connect='finite')
            else:
                self._study_curves[key].setData([], [])
            zero = y == 0
            self._study_zero_curves[key].setData(x[zero], np.array([s['ci_high'] for s in stats])[zero])
            self._study_evm_curves[key].setData(x, [s.get('evm_percent', np.nan) for s in stats])
        values = point['strategies']; current = values['training_ls']
        row = self.result_table.rowCount(); self.result_table.insertRow(row)
        for col, text in enumerate((f'{point["axis_value"]/scale:g}', f'{current["ber"]:.3g}',
                f'{values["ideal"]["ber"]:.3g}', f'{values["uncompensated"]["ber"]:.3g}',
                f'{current["bit_errors"]}/{current["total_bits"]}',
                f'[{current["ci_low"]:.2g}, {current["ci_high"]:.2g}]', f'{current.get("evm_percent", np.nan):.2f}')):
            item = QTableWidgetItem(text); item.setToolTip(text)
            self.result_table.setItem(row, col, item)
        self.study_progress.setText(f'已完成 {len(self._study_results)} / {len(self._study_values)} 点')

    def _on_study_finished(self, token, message):
        if token != self._study_token or self._tasks_suspended:
            return
        worker = self._ber_snr_sweep_thread
        if worker is not None and worker.is_alive():
            QTimer.singleShot(20, lambda: self._on_study_finished(token, message))
            return
        self._study_token += 1
        self._ber_snr_sweep_thread = None
        self._auto_ber_snr_active = False
        self._study_status = message
        self._set_ui_enabled(self.tb is None)
        self.study_progress.setText(f'{message} · {len(self._study_results)} / {len(self._study_values)} 点')
        if self._study_results:
            point = max(self._study_results, key=lambda p: p['axis_value'])
            stats = point['strategies']
            ls = stats['training_ls']
            value = point['axis_value'] / (1000 if self._study_axis == 'spacing' else 1)
            self.study_note.setText(
                f'最大扫描值 {value:g}：训练 LS BER={ls["ber"]:.3g}，理想块内 CSI={stats["ideal"]["ber"]:.3g}，'
                f'关闭预补偿={stats["uncompensated"]["ber"]:.3g}；LS 共 {ls["bit_errors"]}/{ls["total_bits"]} 错误比特。'
                f'95% Wilson 区间 [{ls["ci_low"]:.2g}, {ls["ci_high"]:.2g}]；零误码受样本量限制。'
                + (' 间隔扫描固定 FFT，带宽和 CP 时长同时变化。' if self._study_axis == 'spacing' else '')
                + (' 数据块数扫描同时改变帧长、比特数及训练占比。' if self._study_axis == 'blocks' else '')
            )
            self.study_note.setToolTip(
                '原始统计未强制单调、未用 EVM 反推 BER。区间按二项分布计数估计；相关衰落样本并不等同于独立信道试验。'
                '每点采用固定帧数和同一随机路径实现，不代表所有信道平均性能。'
            )

    def _export_study(self):
        if not self._study_results:
            return
        path, _ = QFileDialog.getSaveFileName(self, '导出 AFDM 星地性能研究', 'afdm_ntn_study.json', 'JSON (*.json)')
        if not path:
            return
        report = {'model': 'SISO DAFT AFDM with CPP/CP, known frame-start training and full block-matrix equalization',
                  'config': asdict(self._study_config), 'axis': self._study_axis,
                  'values': self._study_values, 'results': self._study_results,
                  'frames_per_point': self._study_frames, 'status': self._study_status,
                  'assumptions': [
                      'SISO uncoded AFDM; assisted frame timing; no blind acquisition, coded NR protocol or link-budget model.',
                      'One known full-band training OFDM symbol per frame; LS held over the payload; payload bits used only for scoring.',
                      'Training LS builds a static circular-channel approximation with CPP wrap correction; fractional delays, fast fading and insufficient prefixes can cause model mismatch.',
                      'Full block-matrix ZF/MMSE then DAFT; ideal CSI knows the within-block time-varying channel, not a complete inter-block inverse.',
                      'Standard chirp phase exp(-j*2*pi*c*n^2); legacy normalized chirp coefficients must be divided by M.',
                      'Data CPP uses chirp-periodic correction, or CP when selected; the known OFDM training uses ordinary CP.',
                      'Transmit burst energy / payload bits defines Eb/N0 including prefixes and training; no receive-power normalization.',
                      'Raw BER and error counts; two-sided Wilson 95% intervals are conditional counting estimates, not independent fading-outage bounds.',
                      'One seed fixes path amplitudes and initial phases; continuous per-path Doppler, independent-burst tails.',
                      'LS, ideal CSI and uncompensated strategies share payload, paths and noise streams.',
                      'Spacing scans change bandwidth and prefix duration; block-count scans change frame duration, payload budget and training overhead.',
                      'Chirp advice is manually applied and is not a full-diversity or optimal-BER guarantee; chirp sweeps change the waveform and prefix energy.',
                  ]}
        try:
            with open(path, 'w', encoding='utf-8') as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, '导出失败', str(exc))
