from .base_waveform_tab import BaseWaveformTab
from PyQt5.QtWidgets import QLabel, QComboBox, QSpinBox, QHBoxLayout
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class OTFSTab(BaseWaveformTab):
    """OTFS 页面：控制项、统计口径、EVM 实时曲线和 BER-Eb/N0 扫描与 OFDM 页面对齐。"""

    # OTFS 的 BER-Eb/N0 扫描改为“固定帧数同步 Monte-Carlo”。
    # 不再用 start()+sleep() 的时间驱动方式，避免扫描曲线与实时链路
    # 因统计帧数、线程调度和滑动窗口口径不同而明显不一致。
    AUTO_BER_SNR_POINTS = [-10, -5, 0, 5, 10, 15, 20, 25, 30]
    AUTO_BER_SNR_REPEATS = 2
    OTFS_SWEEP_WARMUP_FRAMES = 10
    OTFS_SWEEP_MEASURE_FRAMES = 300

    def __init__(self):
        super().__init__("OTFS")
        self._add_otfs_specific_controls()
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

        self.constellation_plot.setTitle("星座图 (OTFS DD Equalized Constellation)")
        self.constellation_plot.disableAutoRange()
        self.constellation_plot.setXRange(-2.2, 2.2, padding=0)
        self.constellation_plot.setYRange(-2.2, 2.2, padding=0)

        # 实时曲线改为 EVM vs Time。底层仍复用 BaseWaveformTab
        # 里创建的 ber_plot / ber_curve 对象，避免改变现有文件架构和 UI 结构。
        self.ber_plot.setTitle("调制质量 (EVM vs Time)")
        self.ber_plot.setLabel('left', 'EVM RMS', units='%')
        self.ber_plot.setLabel('bottom', '时间', units='s')
        self.ber_plot.setLogMode(y=False)

        self.ber_snr_plot.setTitle("BER-Eb/N0 曲线（固定帧数扫描）")
        self.ber_snr_plot.setLabel('left', 'Estimated BER')
        self.ber_snr_plot.setLabel('bottom', 'Eb/N0', units='dB')

        # 页面绘图样式：浅色背景 + MATLAB 风格配色
        self._apply_matlab_light_plot_style()

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._refresh_plots)
        self._update_otfs_info_panel(running=False)

    def _apply_matlab_light_plot_style(self):
        """仅对当前波形页面应用浅色背景和 MATLAB 风格配色。"""
        matlab_blue = (0, 114, 189)
        matlab_orange = (217, 83, 25)
        matlab_yellow = (237, 177, 32)
        matlab_purple = (126, 47, 142)
        axis_color = (60, 60, 60)
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

            plot.getPlotItem().getViewBox().setBorder(
                pg.mkPen((225, 225, 225), width=1.0)
            )

        # MATLAB 默认配色风格
        self.spectrum_curve.setPen(pg.mkPen(matlab_blue, width=2.0))

        self.constellation_scatter.setPen(pg.mkPen(matlab_orange, width=0.8))
        self.constellation_scatter.setBrush(
            pg.mkBrush(matlab_orange[0], matlab_orange[1], matlab_orange[2], 160)
        )

        self.ber_curve.setPen(pg.mkPen(matlab_yellow, width=2.0))
        self.ber_curve.setSymbolPen(pg.mkPen(matlab_yellow, width=1.0))
        self.ber_curve.setSymbolBrush(pg.mkBrush(matlab_yellow))

        self.ber_snr_curve.setPen(pg.mkPen(matlab_purple, width=2.0))
        self.ber_snr_curve.setSymbolPen(pg.mkPen(matlab_purple, width=1.0))
        self.ber_snr_curve.setSymbolBrush(pg.mkBrush(matlab_purple))

    def _add_otfs_specific_controls(self):
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

        # 与 OFDM 一样，Eb/N0 / CFO / 多普勒扩展使用 BaseWaveformTab 的通用控件。
        self.param_layout.insertLayout(0, mod_layout)
        self.param_layout.insertLayout(0, delay_layout)

    def _connect_signals(self):
        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)

        self.snr_spin.valueChanged.connect(self._on_snr_changed)
        self.cfo_spin.valueChanged.connect(self._on_channel_changed)
        self.doppler_spin.valueChanged.connect(self._on_channel_changed)
        self.delay_spin.valueChanged.connect(self._on_channel_changed)
        self.mod_combo.currentTextChanged.connect(self._on_channel_changed)

    def _on_start_clicked(self):
        print("[OTFS] 正在启动仿真...")
        self._on_stop_clicked()

        try:
            from simulation.simple_otfs_rx import OTFSTransceiver

            self.tb = OTFSTransceiver(
                delay_spread=self.delay_spin.value(),
                doppler_spread=self.doppler_spin.value(),
                snr_db=self.snr_spin.value(),
                mod_order=self.mod_combo.currentText(),
                cfo_hz=self.cfo_spin.value(),
            )
            self.tb.reset_ber_stats()
            self.tb.start()

            self._sim_start_time = time.time()
            self._evm_time_hist = []
            self._evm_value_hist = []
            self._reset_ber_snr_curve()
            self._snr_settle_start_time = time.time()
            self.ber_curve.setData([], [])
            self.constellation_scatter.setData(x=[], y=[])
            self._set_ui_enabled(False)
            self.update_timer.start(300)
            self._update_otfs_info_panel(running=True)
            print("[OTFS] 仿真启动成功")
        except Exception as e:
            print(f"[OTFS Error] {e}")
            import traceback
            traceback.print_exc()

    def _on_stop_clicked(self):
        print("[OTFS] 停止仿真中...")
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
        self._update_otfs_info_panel(running=False)
        print("[OTFS] 仿真已停止")

    def _set_ui_enabled(self, is_enabled):
        self.btn_start.setEnabled(is_enabled)
        self.btn_stop.setEnabled(not is_enabled)
        self.snr_spin.setEnabled(True)
        self.cfo_spin.setEnabled(True)
        self.doppler_spin.setEnabled(True)
        self.delay_spin.setEnabled(True)
        self.mod_combo.setEnabled(True)

    def _on_snr_changed(self, value):
        if not self.tb:
            return
        try:
            self.tb.update_params(snr_db=float(value))
            if hasattr(self.tb, "reset_ber_stats"):
                self.tb.reset_ber_stats()
            self._last_constellation = None
            self._last_constellation_time = 0.0
            self._evm_time_hist = []
            self._evm_value_hist = []
            self.constellation_scatter.setData(x=[], y=[])
            self.ber_curve.setData([], [])
            self._snr_settle_start_time = time.time()
            if self._sim_start_time is not None:
                self._sim_start_time = time.time()
        except Exception as e:
            print(f"[OTFS SNR 更新失败] {e}")

    def _on_channel_changed(self, *args):
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
        if not self.tb:
            return
        try:
            self.tb.update_params(
                delay_spread=self.delay_spin.value(),
                doppler_spread=self.doppler_spin.value(),
                snr_db=self.snr_spin.value(),
                mod_order=self.mod_combo.currentText(),
                cfo_hz=self.cfo_spin.value(),
            )
            if hasattr(self.tb, "reset_ber_stats"):
                self.tb.reset_ber_stats()
        except Exception as e:
            print(f"[OTFS 参数更新失败] {e}")

    def _refresh_plots(self):
        if not self.tb:
            return

        try:
            samples = self.tb.get_spectrum(num_samples=8192)
            constellation = self.tb.get_constellation()
            evm_hist = self.tb.get_evm_history() if hasattr(self.tb, "get_evm_history") else None
            ber_est = self.tb.get_ber_estimate() if hasattr(self.tb, "get_ber_estimate") else None
            ber_summary = self.tb.get_ber_summary() if hasattr(self.tb, "get_ber_summary") else None
            ber_est = self._ber_display_value_from_summary(ber_summary, ber_est)
            fer_est = self.tb.get_fer_estimate() if hasattr(self.tb, "get_fer_estimate") else None
            last_metrics = self.tb.get_last_metrics() if hasattr(self.tb, "get_last_metrics") else {}
            evm_est = last_metrics.get("evm_percent", None)

            if samples is None or len(samples) == 0:
                return

            samp_rate = getattr(self.tb, "sample_rate", getattr(self.tb, "samp_rate", float(len(samples))))
            freq_axis, spectrum_db = self._compute_segmented_psd(samples, samp_rate)

            self._update_otfs_info_panel(
                running=True,
                metrics=last_metrics,
                ber_est=ber_est,
                fer_est=fer_est,
                evm_percent=evm_est,
                samp_rate=samp_rate,
                ber_summary=ber_summary,
            )
            self._record_ber_snr_point(last_metrics, ber_est)
            ber_snr_data = self._get_ber_snr_data()

            now = time.time()
            hold_sec = 0.55
            if constellation is not None and len(constellation) >= 10 and np.count_nonzero(np.abs(constellation) > 1e-8) >= 10:
                self._last_constellation = constellation
                self._last_constellation_time = now
                has_fresh = True
            elif self._last_constellation is not None and (now - self._last_constellation_time) <= hold_sec:
                constellation = self._last_constellation
                has_fresh = True
            else:
                constellation = None
                has_fresh = False

            if not has_fresh:
                self.constellation_plot.setTitle("星座图 (OTFS DD Equalized Constellation) - 无新帧")
            else:
                self.constellation_plot.setTitle("星座图 (OTFS DD Equalized Constellation)")

            if evm_hist is not None and len(evm_hist) == 2:
                evm_t, evm_v = evm_hist
                evm_t = np.asarray(evm_t, dtype=np.float64)
                evm_v = np.asarray(evm_v, dtype=np.float64)
                finite_mask = np.isfinite(evm_t) & np.isfinite(evm_v)
                evm_t = evm_t[finite_mask]
                evm_v = evm_v[finite_mask]
                if evm_t.size > 0 and evm_v.size > 0:
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
            print(f"[OTFS 绘图错误] {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def _compute_segmented_psd(samples, sample_rate):
        samples = np.asarray(samples, dtype=np.complex64)
        seg_len = 1024 if len(samples) >= 1024 else len(samples)
        seg_len = max(1, int(seg_len))
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
        freq_axis = np.linspace(-sample_rate / 2, sample_rate / 2, seg_len, endpoint=False)
        return freq_axis, spectrum_db

    def _reset_ber_snr_curve(self):
        self._ber_snr_auto_mode = False
        self._auto_ber_snr_active = False
        self._ber_snr_sum = {}
        self._ber_snr_count = {}
        self._snr_settle_start_time = None
        if hasattr(self, "ber_snr_curve"):
            self.ber_snr_curve.setData([], [])

    def _record_ber_snr_point(self, metrics, ber_est):
        if getattr(self, "_ber_snr_auto_mode", False):
            return
        if ber_est is None:
            return
        if self._snr_settle_start_time is not None:
            if (time.time() - self._snr_settle_start_time) < 1.0:
                return
            self._snr_settle_start_time = None
        try:
            snr_db = float(metrics.get("ebn0_db", self.snr_spin.value()))
            ber_summary = self.tb.get_ber_summary() if self.tb is not None and hasattr(self.tb, "get_ber_summary") else None
            ber_value = self._ber_display_value_from_summary(ber_summary, ber_est)
        except Exception:
            return
        if not (np.isfinite(snr_db) and np.isfinite(ber_value)):
            return
        snr_key = round(snr_db, 1)
        # For zero-error runs this value is a finite-sample upper bound (3/N),
        # not an arbitrary plotting floor.  This prevents the false 1e-8 plateau.
        ber_value = min(max(float(ber_value), 1e-12), 0.5)
        self._ber_snr_sum[snr_key] = self._ber_snr_sum.get(snr_key, 0.0) + ber_value
        self._ber_snr_count[snr_key] = self._ber_snr_count.get(snr_key, 0) + 1

    def _get_ber_snr_data(self):
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
        try:
            value = float(value)
            if not np.isfinite(value):
                return "--"
            return f"{max(value, 1e-12):.2e}"
        except Exception:
            return "--"

    @staticmethod
    def _format_float(value, digits=2, suffix=""):
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

    @staticmethod
    def _ber_display_value_from_summary(summary, fallback=None):
        """Return a statistically meaningful BER value for plots.

        When no bit error has been observed, a BER curve should not be pushed to
        an arbitrary floor such as 1e-8.  It should show the finite-sample upper
        bound.  We use the standard 95% rule-of-three bound, 3/N.
        """
        try:
            if isinstance(summary, dict):
                total_bits = int(summary.get("total_bits", 0))
                bit_errors = int(summary.get("total_bit_errors", 0))
                if total_bits > 0:
                    if bit_errors <= 0:
                        return min(max(3.0 / float(total_bits), 1e-12), 0.5)
                    return min(max(float(bit_errors) / float(total_bits), 1e-12), 0.5)
            if fallback is None:
                return float("nan")
            return min(max(float(fallback), 1e-12), 0.5)
        except Exception:
            return float("nan")

    def _format_ber_from_summary(self, summary, fallback=None):
        try:
            if isinstance(summary, dict):
                total_bits = int(summary.get("total_bits", 0))
                bit_errors = int(summary.get("total_bit_errors", 0))
                if total_bits > 0 and bit_errors <= 0:
                    return f"< {3.0 / float(total_bits):.2e}"
                if total_bits > 0:
                    return f"{float(bit_errors) / float(total_bits):.2e}"
            return self._format_scientific(fallback)
        except Exception:
            return "--"

    def _read_ber_snr_sweep_estimate(self, tb) -> float:
        """Use raw OTFS BER counters for BER-Eb/N0 scans.

        The base class reads a live sliding-window estimate.  For OTFS scans this
        caused zero-error points to be drawn at a hard floor, producing an
        unrealistic 1e-8 plateau.  This override reports cumulative BER, or the
        95% upper bound 3/N when the measurement has zero bit errors.
        """
        try:
            if hasattr(tb, "get_ber_summary"):
                return self._ber_display_value_from_summary(tb.get_ber_summary(), None)
        except Exception:
            pass
        return super()._read_ber_snr_sweep_estimate(tb)

    def _update_otfs_info_panel(self, running: bool, metrics=None, ber_est=None, fer_est=None, evm_percent=None, samp_rate=None, ber_summary=None):
        metrics = metrics or {}
        elapsed_text = "--"
        if running and self._sim_start_time is not None:
            elapsed_text = f"{max(0.0, time.time() - self._sim_start_time):.1f} s"

        cp_len = int(getattr(self.tb, "cp_len", 16)) if self.tb is not None else 16
        n_fft = int(getattr(self.tb, "M", 64)) if self.tb is not None else 64
        data_symbols = metrics.get("net_data_symbols", 7 * 48)
        if samp_rate is None and self.tb is not None:
            samp_rate = getattr(self.tb, "sample_rate", None)
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
        receiver_mode = str(metrics.get("receiver_mode", "Practical training/pilot-assisted OTFS"))
        equalizer_path = str(metrics.get("equalizer_path", "--"))

        self.update_info_panel(
            metrics={
                "运行状态": "运行中" if running else "未启动",
                "Eb/N0": self._format_float(ebn0_db, digits=1, suffix=" dB"),
                "BER": self._format_ber_from_summary(ber_summary, ber_est),
                "FER": self._format_scientific(fer_est),
                "同步度量": self._format_float(sync_metric, digits=3),
                "CFO估计": self._format_float(cfo_est, digits=1, suffix=" Hz"),
            },
            config={
                "波形": "OTFS",
                "FFT/CP": f"{n_fft} / {cp_len}",
                "调制": self.mod_combo.currentText(),
                "采样率": samp_rate_text,
                "数据符号": str(data_symbols),
                "星座观察": "工程DD-LMMSE后",
            },
            status_text=(
                f"已运行 {elapsed_text}。EVM-时间曲线显示最近 60 s 的工程 DD-LMMSE 均衡后调制质量，当前 EVM={evm_text}；"
                f"BER 若为零误码，将显示 95% 有限样本上界 <3/N，而不是绘图地板；"
                f"接收机模式：{receiver_mode}，当前支路：{equalizer_path}；只使用训练帧和 DD 导频，不调用真实信道 oracle；"
                f"OTFS 与 OFDM 保持相同采样率、CP、CFO/多普勒控件和 {data_symbols} 个净数据符号。"
                if running else
                "点击开始仿真后，左下角曲线显示 EVM vs Time；BER-Eb/N0 曲线的零误码点将按 95% 上界 <3/N 显示，不再落到 1e-8 假地板。"
            ),
        )

    def _auto_ber_snr_sweep_worker(self):
        """OTFS 专用 BER-Eb/N0 扫描：使用固定帧数同步仿真。

        BaseWaveformTab 的通用扫描逻辑是 start()+sleep()，适合轻量演示，
        但对 OTFS 工程接收机不够稳：不同 Eb/N0 点实际统计帧数可能不同，
        而且读到的可能是滑动窗口/显示 BER。这里直接调用同一个
        OTFSTransceiver._simulate_one_frame()，让扫描链路与实时链路使用同一套
        发射、信道、同步、训练、均衡和判决代码；每个 Eb/N0 点统计固定帧数。
        """
        reason = "完成"
        try:
            snr_points = list(getattr(self, "AUTO_BER_SNR_POINTS", []))
            repeat_total = int(max(1, getattr(self, "AUTO_BER_SNR_REPEATS", 1)))
            warmup_frames = int(max(0, getattr(self, "OTFS_SWEEP_WARMUP_FRAMES", 0)))
            measure_frames = int(max(1, getattr(self, "OTFS_SWEEP_MEASURE_FRAMES", 100)))

            for snr_idx, snr_db in enumerate(snr_points):
                if self._ber_snr_sweep_stop.is_set():
                    reason = "已停止"
                    break

                summaries = []
                for repeat_idx in range(repeat_total):
                    if self._ber_snr_sweep_stop.is_set():
                        reason = "已停止"
                        break

                    tb = self._create_ber_snr_sweep_transceiver(float(snr_db))
                    if tb is None:
                        reason = "当前页面未实现自动扫描分支"
                        break

                    try:
                        self._prepare_ber_snr_sweep_transceiver(tb, repeat_idx)

                        # 预热：让信道状态、同步和缓存进入稳定状态；预热帧不计入 BER。
                        for _ in range(warmup_frames):
                            if self._ber_snr_sweep_stop.is_set():
                                reason = "已停止"
                                break
                            tb._simulate_one_frame()
                        if reason != "完成":
                            break

                        if hasattr(tb, "reset_ber_stats"):
                            tb.reset_ber_stats()

                        # 固定帧数测量：每个 Eb/N0 点统计相同帧数，避免时间驱动导致统计量不同。
                        for frame_idx in range(measure_frames):
                            if self._ber_snr_sweep_stop.is_set():
                                reason = "已停止"
                                break
                            tb._simulate_one_frame()
                            if frame_idx % 25 == 0:
                                time.sleep(0.001)
                        if reason != "完成":
                            break

                        if hasattr(tb, "get_ber_summary"):
                            summaries.append(tb.get_ber_summary())
                    finally:
                        try:
                            if hasattr(tb, "stop"):
                                tb.stop()
                            if hasattr(tb, "wait"):
                                tb.wait(timeout=0.2)
                        except Exception:
                            pass

                if reason != "完成":
                    break

                ber_combined = self._ber_from_aggregate_summaries(summaries)
                if np.isfinite(ber_combined):
                    self.ber_snr_sweep_point.emit(float(snr_db), float(ber_combined))

            if reason == "完成":
                reason = (
                    f"完成：每点 {repeat_total} 次重复，每次 {measure_frames} 个有效统计帧，"
                    f"预热 {warmup_frames} 帧；零误码点按 95% 上界 3/N 显示"
                )
        except Exception as exc:
            reason = f"失败：{exc}"
        finally:
            self.ber_snr_sweep_finished.emit(reason)

    @staticmethod
    def _ber_from_aggregate_summaries(summaries) -> float:
        """把多次 repeat 的原始误码计数合并为一个 BER 显示值。"""
        total_bits = 0
        total_errors = 0
        for summary in summaries or []:
            if not isinstance(summary, dict):
                continue
            try:
                bits = int(summary.get("total_bits", 0))
                errs = int(summary.get("total_bit_errors", 0))
            except Exception:
                continue
            if bits <= 0:
                continue
            total_bits += bits
            total_errors += max(0, min(errs, bits))

        if total_bits <= 0:
            return float("nan")
        if total_errors <= 0:
            return min(max(3.0 / float(total_bits), 1e-12), 0.5)
        return min(max(float(total_errors) / float(total_bits), 1e-12), 0.5)

    def _on_auto_ber_snr_finished(self, reason: str):
        self._auto_ber_snr_active = False
        self._ber_snr_sweep_stop.clear()
        if hasattr(self, "btn_auto_ber_snr"):
            self.btn_auto_ber_snr.setEnabled(True)
        if hasattr(self, "btn_stop_ber_snr"):
            self.btn_stop_ber_snr.setEnabled(False)
        if self.status_text_label is not None:
            self.status_text_label.setText(
                f"BER-Eb/N0 自动扫描{reason}。该曲线使用 OTFS 实时链路同一接收机，"
                f"但每个 Eb/N0 点是独立 Monte-Carlo 统计；实时 EVM/BER 是当前单一运行链路状态。"
            )

    def _create_ber_snr_sweep_transceiver(self, snr_db: float):
        from simulation.simple_otfs_rx import OTFSTransceiver
        return OTFSTransceiver(
            delay_spread=self.delay_spin.value(),
            doppler_spread=self.doppler_spin.value(),
            snr_db=float(snr_db),
            mod_order=self.mod_combo.currentText(),
            cfo_hz=self.cfo_spin.value(),
        )
