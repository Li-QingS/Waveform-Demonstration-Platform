from .base_waveform_tab import BaseWaveformTab
from PyQt5.QtWidgets import QLabel, QComboBox, QSpinBox, QHBoxLayout
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class OfdmTab(BaseWaveformTab):
    def __init__(self):
        super().__init__("OFDM")
        self._add_ofdm_specific_controls()
        self._connect_signals()
        self.tb = None
        self._last_constellation = None
        self._last_constellation_time = 0.0

        self._sim_start_time = None
        self._evm_time_hist = []
        self._evm_value_hist = []
        self._ber_snr_sum = {}
        self._ber_snr_count = {}
        self._snr_settle_start_time = None

        # 默认观察均衡后星座；切换下拉框后会动态更新标题和数据源
        self.constellation_plot.setTitle("星座图 (Post-EQ Constellation)")

        # 与 sim.py 的 const sink 行为保持一致：固定显示范围，不自动缩放
        self.constellation_plot.disableAutoRange()
        self.constellation_plot.setXRange(-2.0, 2.0, padding=0)
        self.constellation_plot.setYRange(-2.0, 2.0, padding=0)

        # OFDM 实时曲线改为 EVM vs Time。底层仍复用 BaseWaveformTab
        # 里创建的 ber_plot / ber_curve 对象，避免改变现有文件架构和 UI 结构。
        self.ber_plot.setTitle("调制质量 (EVM vs Time)")
        self.ber_plot.setLabel('left', 'EVM RMS', units='%')
        self.ber_plot.setLabel('bottom', '时间', units='s')
        self.ber_plot.setLogMode(y=False)

        # OFDM 页面绘图样式：浅色背景 + MATLAB 风格配色
        self._apply_matlab_light_plot_style()

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._refresh_plots)
        self._update_constellation_title()
        self._update_ofdm_info_panel(running=False)

    def _apply_matlab_light_plot_style(self):
        """仅对 OFDM 页面应用浅色背景和 MATLAB 风格配色。"""
        matlab_blue = (0, 114, 189)
        matlab_orange = (217, 83, 25)
        matlab_yellow = (237, 177, 32)
        matlab_purple = (126, 47, 142)
        axis_color = (60, 60, 60)
        grid_color = (210, 210, 210)
        background_color = (250, 250, 250)

        for plot in (
            self.spectrum_plot,
            self.constellation_plot,
            self.ber_plot,
            self.ber_snr_plot,
        ):
            plot.setBackground(background_color)
            plot.showGrid(x=True, y=True, alpha=0.35)
            for axis_name in ("left", "bottom"):
                axis = plot.getAxis(axis_name)
                axis.setPen(pg.mkPen(axis_color, width=1.0))
                axis.setTextPen(pg.mkPen(axis_color, width=1.0))
            # 使用浅灰色网格，避免浅色背景下对比过弱或过强
            plot.getPlotItem().getViewBox().setBorder(pg.mkPen((225, 225, 225), width=1.0))

        # MATLAB 风格颜色映射
        self.spectrum_curve.setPen(pg.mkPen(matlab_blue, width=2.0))
        self.constellation_scatter.setPen(pg.mkPen(matlab_orange, width=0.8))
        self.constellation_scatter.setBrush(pg.mkBrush(matlab_orange[0], matlab_orange[1], matlab_orange[2], 160))
        self.ber_curve.setPen(pg.mkPen(matlab_yellow, width=2.0))
        self.ber_curve.setSymbolPen(pg.mkPen(matlab_yellow, width=1.0))
        self.ber_curve.setSymbolBrush(pg.mkBrush(matlab_yellow))
        self.ber_snr_curve.setPen(pg.mkPen(matlab_purple, width=2.0))
        self.ber_snr_curve.setSymbolPen(pg.mkPen(matlab_purple, width=1.0))
        self.ber_snr_curve.setSymbolBrush(pg.mkBrush(matlab_purple))

    def _add_ofdm_specific_controls(self):
        fft_layout = QHBoxLayout()
        fft_label = QLabel("FFT 点数:")
        self.fft_spin = QSpinBox()
        self.fft_spin.setRange(64, 4096)
        self.fft_spin.setValue(64)
        self.fft_spin.setEnabled(False)
        fft_layout.addWidget(fft_label)
        fft_layout.addWidget(self.fft_spin)

        delay_layout = QHBoxLayout()
        delay_label = QLabel("最大多径时延:")
        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(0, 64)
        self.delay_spin.setValue(5)
        self.delay_spin.setSuffix(" samp")
        delay_layout.addWidget(delay_label)
        delay_layout.addWidget(self.delay_spin)

        mod_layout = QHBoxLayout()
        mod_label = QLabel("调制方式:")
        self.mod_combo = QComboBox()
        self.mod_combo.addItems(["QPSK", "16QAM", "64QAM"])
        mod_layout.addWidget(mod_label)
        mod_layout.addWidget(self.mod_combo)

        observe_layout = QHBoxLayout()
        observe_label = QLabel("星座观察:")
        self.observe_combo = QComboBox()
        self.observe_combo.addItems(["均衡后", "均衡前"])
        observe_layout.addWidget(observe_label)
        observe_layout.addWidget(self.observe_combo)

        self.param_layout.insertLayout(0, observe_layout)
        self.param_layout.insertLayout(0, mod_layout)
        self.param_layout.insertLayout(0, delay_layout)
        self.param_layout.insertLayout(0, fft_layout)

    def _connect_signals(self):
        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)

        # 运行时联动：允许在仿真过程中动态修改信道参数
        self.snr_spin.valueChanged.connect(self._on_snr_changed)
        self.cfo_spin.valueChanged.connect(self._on_cfo_changed)
        self.doppler_spin.valueChanged.connect(self._on_doppler_changed)
        self.delay_spin.valueChanged.connect(self._on_delay_changed)
        self.observe_combo.currentIndexChanged.connect(self._on_observe_mode_changed)

    def _on_start_clicked(self):
        print("[OFDM] 正在启动 (基于GRC代码)...")
        self._on_stop_clicked()

        try:
            from simulation.simple_ofdm_rx import OfdmTransceiver
            self.tb = OfdmTransceiver(
                fft_len=64,
                cp_len=16,
                snr_db=self.snr_spin.value(),
                cfo_hz=self.cfo_spin.value(),
                doppler_spread_hz=self.doppler_spin.value(),
                delay_spread=self.delay_spin.value(),
                mod_order=self.mod_combo.currentText()
            )

            self.tb.start()
            self._sim_start_time = time.time()
            self._evm_time_hist = []
            self._evm_value_hist = []
            self._reset_ber_snr_curve()
            self._snr_settle_start_time = time.time()
            self.ber_curve.setData([], [])
            self._set_ui_enabled(False)
            self.update_timer.start(300)
            self._update_ofdm_info_panel(running=True)
            print("[OFDM] 启动成功。")

        except Exception as e:
            print(f"[Error] {e}")
            import traceback
            traceback.print_exc()

    def _on_stop_clicked(self):
        print("[OFDM] 停止中...")
        self.update_timer.stop()
        if self.tb:
            try:
                self.tb.stop()
                self.tb.wait()
            except Exception:
                pass
            finally:
                self.tb = None
        self._last_constellation = None
        self._last_constellation_time = 0.0
        self._sim_start_time = None
        self._evm_time_hist = []
        self._evm_value_hist = []
        self.constellation_scatter.setData(x=[], y=[])
        self.ber_curve.setData([], [])
        self._reset_ber_snr_curve()
        self._set_ui_enabled(True)
        self._update_ofdm_info_panel(running=False)

    def _set_ui_enabled(self, controls_enabled: bool):
        self.btn_start.setEnabled(controls_enabled)
        self.btn_stop.setEnabled(not controls_enabled)

        # Eb/N0 / CFO / 多普勒 / 观察模式 允许运行时修改；调制方式仍建议停机后再改。
        self.snr_spin.setEnabled(True)
        self.cfo_spin.setEnabled(True)
        self.doppler_spin.setEnabled(True)
        self.delay_spin.setEnabled(True)
        self.observe_combo.setEnabled(True)
        self.mod_combo.setEnabled(controls_enabled)

    def _update_constellation_title(self, has_fresh: bool = True):
        if self.observe_combo.currentText() == "均衡前":
            title = "星座图 (Pre-EQ Constellation)"
        else:
            title = "星座图 (Post-EQ Constellation)"
            if not has_fresh:
                title += " - 无新帧"
        self.constellation_plot.setTitle(title)

    def _on_observe_mode_changed(self, _index):
        self._last_constellation = None
        self._last_constellation_time = 0.0
        self._update_constellation_title()

    def _on_snr_changed(self, value):
        if self.tb:
            try:
                self.tb.set_snr_db(float(value))
                self._last_constellation = None
                self._last_constellation_time = 0.0
                self._evm_time_hist = []
                self._evm_value_hist = []
                self.constellation_scatter.setData(x=[], y=[])
                self.ber_curve.setData([], [])
                self._snr_settle_start_time = time.time()
                if self._sim_start_time is not None:
                    self._sim_start_time = time.time()
            except Exception:
                pass

    def _on_cfo_changed(self, value):
        if self.tb:
            try:
                self.tb.set_cfo_hz(float(value))
                self._last_constellation = None
                self._last_constellation_time = 0.0
                self._evm_time_hist = []
                self._evm_value_hist = []
                self.constellation_scatter.setData(x=[], y=[])
                self.ber_curve.setData([], [])
                self._reset_ber_snr_curve()
                self._snr_settle_start_time = time.time()
                if self._sim_start_time is not None:
                    self._sim_start_time = time.time()
            except Exception:
                pass

    def _reset_runtime_curves_after_channel_change(self):
        self._last_constellation = None
        self._last_constellation_time = 0.0
        self._evm_time_hist = []
        self._evm_value_hist = []
        self.constellation_scatter.setData(x=[], y=[])
        self.ber_curve.setData([], [])
        self._reset_ber_snr_curve()
        self._snr_settle_start_time = time.time()
        if self._sim_start_time is not None:
            self._sim_start_time = time.time()

    def _on_doppler_changed(self, value):
        if self.tb:
            try:
                self.tb.set_doppler_spread_hz(float(value))
                self._reset_runtime_curves_after_channel_change()
            except Exception:
                pass

    def _on_delay_changed(self, value):
        if self.tb:
            try:
                self.tb.set_delay_spread(int(value))
                self._reset_runtime_curves_after_channel_change()
            except Exception:
                pass

    def _refresh_plots(self):
        if not self.tb:
            return

        try:
            # 频谱用更长样本做分段平均，才能看出 OFDM 的带宽形状和 DC/保护子载波特征
            samples = self.tb.get_spectrum(num_samples=8192)

            if self.observe_combo.currentText() == "均衡前":
                constellation = self.tb.get_pre_eq_constellation()
            else:
                constellation = self.tb.get_constellation()

            evm_hist = self.tb.get_evm_history() if hasattr(self.tb, "get_evm_history") else None
            fer_est = self.tb.get_fer_estimate()
            ber_est = self.tb.get_ber_estimate() if hasattr(self.tb, "get_ber_estimate") else None
            last_metrics = self.tb.get_last_metrics() if hasattr(self.tb, "get_last_metrics") else {}
            evm_est = last_metrics.get("evm_percent", None)

            if samples is None or len(samples) == 0:
                return

            samp_rate = getattr(self.tb, "samp_rate", float(len(samples)))

            seg_len = 1024
            if len(samples) < seg_len:
                seg_len = len(samples)

            n_seg = max(1, len(samples) // seg_len)
            trimmed = samples[-n_seg * seg_len:]
            blocks_ = trimmed.reshape(n_seg, seg_len)
            window = np.hanning(seg_len).astype(np.float32)

            psd = np.zeros(seg_len, dtype=np.float64)
            for blk in blocks_:
                spec = np.fft.fftshift(np.fft.fft(blk * window))
                psd += np.abs(spec) ** 2

            psd /= max(n_seg, 1)
            spectrum_db = 10 * np.log10(psd + 1e-12)
            freq_axis = np.linspace(-samp_rate / 2, samp_rate / 2, seg_len, endpoint=False)

            self._update_ofdm_info_panel(
                running=True,
                metrics=last_metrics,
                ber_est=ber_est,
                fer_est=fer_est,
                evm_percent=evm_est,
                samp_rate=samp_rate,
            )

            self._record_ber_snr_point(last_metrics, ber_est)
            ber_snr_data = self._get_ber_snr_data()

            # 只显示“真正的均衡后星座”；如果本轮没新点，就沿用上一帧
            now = time.time()
            hold_sec = 0.35

            if constellation is not None and len(constellation) >= 10:
                self._last_constellation = constellation
                self._last_constellation_time = now
                has_fresh = True
            elif self._last_constellation is not None and (now - self._last_constellation_time) <= hold_sec:
                constellation = self._last_constellation
                has_fresh = True
            else:
                constellation = None
                has_fresh = False

            self._update_constellation_title(has_fresh=has_fresh)

            if evm_hist is not None and len(evm_hist) == 2:
                evm_t, evm_v = evm_hist
                evm_t = np.asarray(evm_t, dtype=np.float64)
                evm_v = np.asarray(evm_v, dtype=np.float64)
                finite_mask = np.isfinite(evm_t) & np.isfinite(evm_v)
                evm_t = evm_t[finite_mask]
                evm_v = evm_v[finite_mask]
                if evm_t.size > 0 and evm_v.size > 0:
                    if evm_t.size != evm_v.size:
                        n_hist = min(evm_t.size, evm_v.size)
                        evm_t = evm_t[:n_hist]
                        evm_v = evm_v[:n_hist]
                    recent_mask = (evm_t >= max(0.0, evm_t[-1] - 60.0))
                    self._evm_time_hist = evm_t[recent_mask].tolist()
                    self._evm_value_hist = np.maximum(evm_v[recent_mask], 0.0).tolist()
                elif self._sim_start_time is not None and evm_est is not None:
                    try:
                        evm_est_f = float(evm_est)
                    except Exception:
                        evm_est_f = float("nan")
                    if np.isfinite(evm_est_f):
                        elapsed = max(0.0, time.time() - self._sim_start_time)
                        self._evm_time_hist.append(elapsed)
                        self._evm_value_hist.append(max(evm_est_f, 0.0))
            elif self._sim_start_time is not None and evm_est is not None:
                try:
                    evm_est_f = float(evm_est)
                except Exception:
                    evm_est_f = float("nan")
                if np.isfinite(evm_est_f):
                    elapsed = max(0.0, time.time() - self._sim_start_time)
                    self._evm_time_hist.append(elapsed)
                    self._evm_value_hist.append(max(evm_est_f, 0.0))

            if self._evm_time_hist:
                latest_t = self._evm_time_hist[-1]
                while self._evm_time_hist and (latest_t - self._evm_time_hist[0]) > 60.0:
                    self._evm_time_hist.pop(0)
                    self._evm_value_hist.pop(0)

            if constellation is not None and len(constellation) > 0:
                self.update_plots(
                    spectrum_data=(freq_axis, spectrum_db),
                    constellation_data=constellation,
                    ber_data=(self._evm_time_hist, self._evm_value_hist),
                    ber_snr_data=ber_snr_data,
                )
            else:
                self.update_plots(
                    spectrum_data=(freq_axis, spectrum_db),
                    ber_data=(self._evm_time_hist, self._evm_value_hist),
                    ber_snr_data=ber_snr_data,
                )
                self.constellation_scatter.setData(x=[], y=[])

        except Exception as e:
            print(f"[OFDM][refresh error] {e}")
            import traceback
            traceback.print_exc()


    def _reset_ber_snr_curve(self):
        """清空 BER-SNR 曲线；用于重新开始或信道条件变化时。"""
        self._ber_snr_auto_mode = False
        self._auto_ber_snr_active = False
        self._ber_snr_sum = {}
        self._ber_snr_count = {}
        self._snr_settle_start_time = None
        if hasattr(self, "ber_snr_curve"):
            self.ber_snr_curve.setData([], [])

    def _record_ber_snr_point(self, metrics, ber_est):
        """把当前 SNR 下的 BER 估计累积到 BER-SNR 曲线。"""
        if getattr(self, "_ber_snr_auto_mode", False):
            return
        if ber_est is None:
            return

        # SNR 刚切换时，仿真器会重置统计量；稍等一下再记录，避免把瞬态误差画进去。
        if self._snr_settle_start_time is not None:
            if (time.time() - self._snr_settle_start_time) < 1.0:
                return
            self._snr_settle_start_time = None

        try:
            snr_db = float(metrics.get("ebn0_db", self.snr_spin.value()))
            ber_value = float(ber_est)
        except Exception:
            return

        if not (np.isfinite(snr_db) and np.isfinite(ber_value)):
            return

        snr_key = round(snr_db, 1)
        ber_value = max(ber_value, 1e-12)

        self._ber_snr_sum[snr_key] = self._ber_snr_sum.get(snr_key, 0.0) + ber_value
        self._ber_snr_count[snr_key] = self._ber_snr_count.get(snr_key, 0) + 1

    def _get_ber_snr_data(self):
        """返回按 SNR 排序后的 BER-SNR 曲线数据。"""
        if not self._ber_snr_sum:
            return ([], [])

        snr_values = sorted(self._ber_snr_sum.keys())
        ber_values = [
            max(self._ber_snr_sum[snr] / max(self._ber_snr_count.get(snr, 1), 1), 1e-12)
            for snr in snr_values
        ]
        return (snr_values, ber_values)


    @staticmethod
    def _format_scientific(value):
        """格式化 BER/FER，便于在信息面板中显示。"""
        try:
            value = float(value)
            if not np.isfinite(value):
                return "--"
            return f"{max(value, 1e-12):.2e}"
        except Exception:
            return "--"

    @staticmethod
    def _format_float(value, digits=2, suffix=""):
        """安全格式化普通浮点数。"""
        try:
            value = float(value)
            if not np.isfinite(value):
                return "--"
            return f"{value:.{digits}f}{suffix}"
        except Exception:
            return "--"

    @staticmethod
    def _format_percent(value, digits=2):
        """安全格式化 EVM 百分比。"""
        try:
            value = float(value)
            if not np.isfinite(value):
                return "--"
            return f"{value:.{digits}f}%"
        except Exception:
            return "--"

    def _update_ofdm_info_panel(
        self,
        running: bool,
        metrics=None,
        ber_est=None,
        fer_est=None,
        evm_percent=None,
        samp_rate=None,
    ):
        """刷新底部 OFDM 信息面板。"""
        metrics = metrics or {}

        elapsed_text = "--"
        if running and self._sim_start_time is not None:
            elapsed_text = f"{max(0.0, time.time() - self._sim_start_time):.1f} s"

        fft_len = int(self.fft_spin.value()) if hasattr(self, "fft_spin") else 64
        cp_len = int(getattr(self.tb, "cp_len", 16)) if self.tb is not None else 16
        data_symbols = metrics.get("net_data_symbols", 7 * 48)
        if samp_rate is None and self.tb is not None:
            samp_rate = getattr(self.tb, "samp_rate", None)

        if samp_rate is None:
            samp_rate_text = "--"
        elif float(samp_rate) >= 1e6:
            samp_rate_text = f"{float(samp_rate) / 1e6:.2f} MSa/s"
        else:
            samp_rate_text = f"{float(samp_rate) / 1e3:.1f} kSa/s"

        ebn0_db = metrics.get("ebn0_db", self.snr_spin.value())
        sync_metric = metrics.get("sync_metric", None)
        cfo_est = metrics.get("total_cfo_hz", metrics.get("coarse_cfo_hz", None))
        if evm_percent is None:
            evm_percent = metrics.get("evm_percent", None)
        evm_text = self._format_percent(evm_percent, digits=2)

        self.update_info_panel(
            metrics={
                "运行状态": "运行中" if running else "未启动",
                "Eb/N0": self._format_float(ebn0_db, digits=1, suffix=" dB"),
                "BER": self._format_scientific(ber_est),
                "FER": self._format_scientific(fer_est),
                "同步度量": self._format_float(sync_metric, digits=3),
                "CFO估计": self._format_float(cfo_est, digits=1, suffix=" Hz"),
            },
            config={
                "波形": "OFDM",
                "FFT/CP": f"{fft_len} / {cp_len}",
                "调制": self.mod_combo.currentText(),
                "采样率": samp_rate_text,
                "数据符号": str(data_symbols),
                "星座观察": self.observe_combo.currentText(),
            },
            status_text=(
                f"已运行 {elapsed_text}。EVM-时间曲线显示最近 60 s 的均衡后调制质量，当前 EVM={evm_text}；"
                "BER-SNR 曲线由左侧按钮启动独立自动扫描生成；"
                "频谱图采用分段平均，星座图可在均衡前/均衡后之间切换。"
                if running else
                "点击开始仿真后，左下角曲线显示 EVM vs Time，底部仍显示 Eb/N0、BER、FER、同步度量、CFO 估计和帧资源配置；"
                "点击“自动绘制 BER-SNR”后，右下角会独立扫描形成曲线。"
            ),
        )

    def _create_ber_snr_sweep_transceiver(self, snr_db: float):
        from simulation.simple_ofdm_rx import OfdmTransceiver
        return OfdmTransceiver(
            fft_len=64,
            cp_len=16,
            snr_db=float(snr_db),
            cfo_hz=self.cfo_spin.value(),
            doppler_spread_hz=self.doppler_spin.value(),
            delay_spread=self.delay_spin.value(),
            mod_order=self.mod_combo.currentText(),
        )
