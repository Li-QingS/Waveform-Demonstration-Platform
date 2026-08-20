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



MATLAB_BLUE = (0, 114, 189)
MATLAB_ORANGE = (217, 83, 25)
MATLAB_GREEN = (119, 172, 48)
MATLAB_YELLOW = (237, 177, 32)
MATLAB_PURPLE = (126, 47, 142)
LIGHT_PLOT_BACKGROUND = (250, 250, 250)
AXIS_COLOR = (60, 60, 60)
BORDER_COLOR = (225, 225, 225)



class _WaveformRunner:
    """统一封装 OFDM / OTFS / AFDM 的启动、停止、热更新和数据抓取。"""

    def __init__(self):
        self.wave_type = None
        self.tb = None

    def is_running(self):
        return self.tb is not None

    def start(self, wave_type: str, snr_db: float, cfo_hz: float,
              doppler_hz: float, mod_order: str):
        self.stop()
        self.wave_type = wave_type

        if wave_type == "OFDM":
            from simulation.simple_ofdm_rx import OfdmTransceiver
            self.tb = OfdmTransceiver(
                fft_len=64,
                cp_len=16,
                snr_db=snr_db,
                cfo_hz=cfo_hz,
                doppler_spread_hz=doppler_hz,
                mod_order=mod_order,
                payload_symbols=8,
            )

        elif wave_type == "OTFS":
            from simulation.simple_otfs_rx import OTFSTransceiver
            self.tb = OTFSTransceiver(
                delay_spread=5,
                doppler_spread=doppler_hz,
                snr_db=snr_db,
                mod_order=mod_order,
                cfo_hz=cfo_hz,
                n_subcarriers=64,
                n_symbols=8,
                sample_rate=960000.0,
                equalizer="MMSE",
            )

        elif wave_type == "AFDM":
            from simulation.simple_afdm_rx import AFDMTransceiver
            try:
                self.tb = AFDMTransceiver(
                    c1=0.05,
                    c2=0.05,
                    snr_db=snr_db,
                    mod_order=mod_order,
                    doppler_freq=doppler_hz,
                    cfo_hz=cfo_hz,
                )
            except TypeError:
                self.tb = AFDMTransceiver(
                    c1=0.05,
                    c2=0.05,
                    snr_db=snr_db,
                    mod_order=mod_order,
                    doppler_freq=doppler_hz,
                )
                if hasattr(self.tb, "_cfo_hz"):
                    self.tb._cfo_hz = float(cfo_hz)

            if hasattr(self.tb, "reset_ber_stats"):
                try:
                    self.tb.reset_ber_stats()
                except Exception:
                    pass
        else:
            raise ValueError(f"不支持的波形类型: {wave_type}")

        self.tb.start()

    def stop(self):
        if self.tb is not None:
            try:
                self.tb.stop()
            except Exception:
                pass

            try:
                self.tb.wait()
            except Exception:
                pass

            self.tb = None

    def update_runtime_params(self, snr_db: float, cfo_hz: float,
                              doppler_hz: float):
        """运行时热更新公共参数。调制方式不在运行中切换。"""
        if self.tb is None:
            return

        try:
            if self.wave_type == "OFDM":
                if hasattr(self.tb, "set_snr_db"):
                    self.tb.set_snr_db(float(snr_db))
                if hasattr(self.tb, "set_cfo_hz"):
                    self.tb.set_cfo_hz(float(cfo_hz))
                if hasattr(self.tb, "set_doppler_spread_hz"):
                    self.tb.set_doppler_spread_hz(float(doppler_hz))

            elif self.wave_type == "OTFS":
                if hasattr(self.tb, "update_params"):
                    self.tb.update_params(
                        snr_db=float(snr_db),
                        doppler_spread=float(doppler_hz),
                        cfo_hz=float(cfo_hz),
                    )

            elif self.wave_type == "AFDM":
                if hasattr(self.tb, "update_params"):
                    try:
                        self.tb.update_params(
                            snr_db=float(snr_db),
                            doppler_freq=float(doppler_hz),
                            cfo_hz=float(cfo_hz),
                            adaptive_chirp_enabled=True,
                        )
                    except TypeError:
                        self.tb.update_params(
                            snr_db=float(snr_db),
                            doppler_freq=float(doppler_hz),
                        )
                        if hasattr(self.tb, "_cfo_hz"):
                            self.tb._cfo_hz = float(cfo_hz)
        except Exception:
            pass

    def snapshot(self):
        """返回当前接收端解调星座图和 BER 数据。"""
        if self.tb is None:
            return {
                "constellation": np.zeros(0, dtype=np.complex64),
                "ber_t": np.array([], dtype=np.float64),
                "ber_v": np.array([], dtype=np.float64),
            }

        constellation = np.zeros(0, dtype=np.complex64)
        ber_t = np.array([], dtype=np.float64)
        ber_v = np.array([], dtype=np.float64)

        try:
            if hasattr(self.tb, "get_constellation"):
                constellation = np.asarray(
                    self.tb.get_constellation(),
                    dtype=np.complex64
                )
        except Exception:
            constellation = np.zeros(0, dtype=np.complex64)

        try:
            if hasattr(self.tb, "get_estimated_ber"):
                hist = self.tb.get_estimated_ber()
                if hist is not None and len(hist) == 2:
                    ber_t = np.asarray(hist[0], dtype=np.float64)
                    ber_v = np.asarray(hist[1], dtype=np.float64)
        except Exception:
            ber_t = np.array([], dtype=np.float64)
            ber_v = np.array([], dtype=np.float64)

        if ber_t.size == 0 or ber_v.size == 0:
            try:
                if hasattr(self.tb, "get_ber_history"):
                    hist = self.tb.get_ber_history()
                    if hist is not None and len(hist) == 2:
                        ber_t = np.asarray(hist[0], dtype=np.float64)
                        ber_v = np.asarray(hist[1], dtype=np.float64)
            except Exception:
                ber_t = np.array([], dtype=np.float64)
                ber_v = np.array([], dtype=np.float64)

        if ber_t.size > 0 and ber_v.size > 0:
            n = min(ber_t.size, ber_v.size)
            ber_t = ber_t[:n]
            ber_v = np.maximum(ber_v[:n], 1e-6)
        else:
            ber_t = np.array([], dtype=np.float64)
            ber_v = np.array([], dtype=np.float64)

        return {
            "constellation": constellation,
            "ber_t": ber_t,
            "ber_v": ber_v,
        }

class _ScanWorker(threading.Thread):
    """自动扫描后台线程，调用独立扫描后端，不读取实时后端 BER。"""

    def __init__(self, result_store: dict, result_lock: threading.Lock,
                 stop_event: threading.Event, params: dict):
        super().__init__(daemon=True)
        self.result_store = result_store
        self.result_lock = result_lock
        self.stop_event = stop_event
        self.params = params

    def _put_result(self, family: str, waveform: str, x, y, text: str):
        with self.result_lock:
            self.result_store[family][waveform] = (
                np.asarray(x, dtype=np.float64),
                np.maximum(np.asarray(y, dtype=np.float64), 1e-8),
            )
            self.result_store["status"] = text

    def run(self):
        try:
            from simulation.compare_scan_backend import WaveformScanSimulator

            sim = WaveformScanSimulator()
            waveforms = ["OFDM", "OTFS", "AFDM"]

            frames = int(self.params["frames"])
            snr_db = float(self.params["snr_db"])
            mod_order = str(self.params["mod_order"])
            fixed_cfo = float(self.params["fixed_cfo"])
            fixed_doppler = float(self.params["fixed_doppler"])
            cfo_axis = np.asarray(self.params["cfo_axis"], dtype=np.float64)
            doppler_axis = np.asarray(self.params["doppler_axis"], dtype=np.float64)

            for family, axis_values in (("cfo", cfo_axis), ("doppler", doppler_axis)):
                for w_idx, waveform in enumerate(waveforms):
                    y_values = []
                    for i, value in enumerate(axis_values):
                        if self.stop_event.is_set():
                            return

                        if family == "cfo":
                            cfo_hz = float(value)
                            doppler_hz = fixed_doppler
                        else:
                            cfo_hz = fixed_cfo
                            doppler_hz = float(value)

                        seed = (
                            20260503
                            + 100000 * (0 if family == "cfo" else 1)
                            + 10000 * w_idx
                            + i
                        )
                        ber = sim.simulate_ber(
                            waveform=waveform,
                            snr_db=snr_db,
                            cfo_hz=cfo_hz,
                            doppler_hz=doppler_hz,
                            mod_order=mod_order,
                            frames=frames,
                            seed=seed,
                        )
                        y_values.append(max(float(ber), 1e-8))

                        self._put_result(
                            family,
                            waveform,
                            axis_values[:i + 1],
                            np.array(y_values, dtype=np.float64),
                            f"扫描中：{family.upper()} / {waveform} / "
                            f"{i + 1}/{len(axis_values)} 点，每点 {frames} 帧 Monte-Carlo。"
                        )

            with self.result_lock:
                self.result_store["done"] = True
                self.result_store["status"] = (
                    "扫描完成：自动扫描结果来自独立 Monte-Carlo 后端，"
                    "实时对比图仍然保留在原页面中。"
                )

        except Exception as exc:
            traceback.print_exc()
            with self.result_lock:
                self.result_store["done"] = True
                self.result_store["status"] = f"扫描失败：{exc}"
