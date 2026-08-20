# -*- coding: utf-8 -*-
"""
hardware/afdm_hardtest.py

AFDM 真机硬件测试后端
====================

设计目标
--------
- 与现有 HardwareTestTab 的接口保持一致：
  start / stop / wait / configure /
  get_tx_spectrum_source / get_rx_spectrum_source /
  get_rx_constellation / get_rx_text / get_status / get_decode_stats
- 只面向 USRP 真机闭环文本收发，不包含仿真信道
- 沿用 OFDM/OTFS 的应用层文本帧：MAGIC + LEN + payload + CRC32
- 发射端用 Python 生成 AFDM 基带波形，GNU Radio / UHD 只负责持续发送与采样
- 接收端在 Python 中完成：同步、CFO 估计、AFDM 解调、训练估计、导频修正、文本恢复

说明
----
这是一份“工程可跑”的 AFDM 硬件后端，思路上继承了 simple_afdm_rx.py 的
AFDM 变换与资源映射，但去掉了仿真信道 / AWGN / 统计接口，并重构为真实硬件
收发链路。为了兼顾真机稳定性，本实现使用“训练块近似对角估计 + 导频逐块复数
标量修正 + 跨帧软合并”的简化接收机，而不是仿真版里依赖已知稀疏路径结构的
CFO-search + LMMSE 检测器。
"""

from __future__ import annotations

import threading
import time
import zlib
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class _LegacyAFDMHardwareTest:
    APP_MAGIC = b"MTPK"

    def __init__(
        self,
        carrier_freq: float = 2.4e9,
        samp_rate: float = 960000.0,
        tx_gain: float = 40.0,
        rx_gain: float = 40.0,
        device_type: str = "USRP B210",
        serial: Optional[str] = None,
        tx_antenna: str = "TX/RX",
        rx_antenna: str = "RX2",
        mod_order: str = "QPSK",
        equalizer: str = "MMSE",
        n_subcarriers: int = 64,
        n_symbols: int = 7,
        cp_len: int = 16,
        c1: float = 0.05,
        c2: float = 0.05,
        update_period: float = 0.08,
        tx_text: str = "Hello AFDM Hardware Test!",
        sample_rate: Optional[float] = None,
    ):
        if sample_rate is not None:
            samp_rate = sample_rate

        self.carrier_freq = float(carrier_freq)
        self.sample_rate = float(samp_rate)
        self.samp_rate = self.sample_rate
        self.tx_gain = float(tx_gain)
        self.rx_gain = float(rx_gain)
        self.device_type = str(device_type)
        self.serial = serial
        self.tx_antenna = str(tx_antenna)
        self.rx_antenna = str(rx_antenna)

        self.mod_order = str(mod_order).upper()
        self.equalizer = str(equalizer).upper()
        self.update_period = float(update_period)
        self.N = int(n_subcarriers)
        self.n_data_units = int(n_symbols)
        self.cp_len = int(cp_len)
        self.c1 = float(c1)
        self.c2 = float(c2)

        if self.N != 64:
            raise ValueError("This AFDM backend expects n_subcarriers == 64")
        if self.n_data_units <= 0:
            raise ValueError("n_symbols must be positive")
        if self.mod_order not in ("QPSK", "16QAM", "64QAM"):
            raise ValueError(f"Unsupported modulation: {self.mod_order}")
        if self.equalizer not in ("MMSE", "ZF"):
            raise ValueError(f"Unsupported equalizer: {self.equalizer}")

        self.bits_per_symbol = self._get_bits_per_symbol(self.mod_order)
        self.subcarrier_spacing = self.sample_rate / max(self.N, 1)

        # 帧结构: pre_guard | sync(repeated-half chirp) | train | data x K | post_guard
        self.pre_guard_len = 16
        self.sync_half_len = self.N
        self.sync_len = 2 * self.sync_half_len
        self.block_len = self.N + self.cp_len
        self.post_guard_len = self.cp_len + 16
        self.frame_len = (
            self.pre_guard_len
            + self.sync_len
            + (1 + self.n_data_units) * self.block_len
            + self.post_guard_len
        )
        self._off_sync = self.pre_guard_len
        self._off_train = self._off_sync + self.sync_len
        self._off_data = self._off_train + self.block_len
        self._off_end = self._off_data + self.n_data_units * self.block_len

        # 同步 / 检测参数
        self.sync_metric_threshold = 0.10
        self._noise_var_reg = 2e-2

        # 资源规划
        self.active_idx, self.pilot_idx, self.data_idx = self._build_afdm_resource_plan(self.N)
        self._n_active = int(len(self.active_idx))
        self._n_pilot = int(len(self.pilot_idx))
        self._n_data = int(len(self.data_idx))

        # AFDM 变换矩阵
        self._A_tx = None
        self._A_rx = None
        self._refresh_afdm_mats()

        # 同步 / 训练序列
        self.sync_preamble = self._build_sync_preamble(self.sync_half_len)
        self._sync_energy = float(np.vdot(self.sync_preamble, self.sync_preamble).real) + 1e-12
        self._train_dd_known = np.zeros(self.N, dtype=np.complex128)
        self._train_dd_known[self.active_idx] = self._training_active_sequence(self._n_active)
        self._train_time = self._afdm_symbol_to_time_cp(self._train_dd_known)

        # 文本 / 应用层状态
        self._tx_text = ""
        self._tx_payload = b""
        self._tx_frame = b""
        self._tx_frame_bits = np.zeros(0, dtype=np.int8)
        self._payload_repeat = 1
        self._tx_waveform = np.zeros(1, dtype=np.complex64)

        self._rx_text = ""
        self._decode_ok = False
        self._match_bytes = 0
        self._last_good_rx_payload = b""
        self._last_raw_bytes = b""

        # 运行态缓存
        self._lock = threading.Lock()
        self._status = "idle"
        self._last_error = ""
        self._running = False
        self._tb = None
        self._monitor_thread = None
        self._monitor_stop = threading.Event()

        self._buffer_keep = max(131072, 12 * self.frame_len)
        self._tx_buffer = deque(maxlen=self._buffer_keep)
        self._rx_buffer = deque(maxlen=self._buffer_keep)
        # 星座图显示模式：
        #   "raw"            —— 原始均衡后软符号
        #   "dd_refined"     —— 最近邻投影 + 残差压缩，视觉紧簇但保留 SNR
        #   "hard_decision"  —— 直接投影到理想 QPSK 点
        #   "pre_equalized"  —— AFDM 均衡前的 DD 域数据符号
        # 只影响显示，不影响判决 / BER / CRC。
        self._latest_constellation = np.zeros(0, dtype=np.complex64)
        self._latest_constellation_pre_eq = np.zeros(0, dtype=np.complex64)
        self._last_good_constellation = np.zeros(0, dtype=np.complex64)
        self.constellation_display_mode = "dd_refined"
        self._latest_rx_samples = np.zeros(4096, dtype=np.complex64)
        self._latest_tx_samples = np.zeros(4096, dtype=np.complex64)
        self._ber_hist_t: List[float] = []
        self._ber_hist_v: List[float] = []
        self._t0 = time.time()
        self._rx_samples_seen = 0
        self._last_processed_abs_start = -10**18
        self._frame_soft_history: deque = deque(maxlen=12)
        self._combine_frames = 0

        # 调试 / 状态
        self.last_sync_index = 0
        self.last_payload_start = 0
        self.last_cfo_est_hz = 0.0
        self.last_sync_metric = 0.0
        self.last_frame_ok = False
        self.last_bad_reason = "init"
        self.last_train_nmse = 0.0
        self.last_pilot_nmse = 0.0
        self.last_alpha_abs = 0.0

        self._usrp_args = self._build_device_args()
        self._gr = None
        self._blocks = None
        self._uhd = None
        self._import_runtime()

        self._set_tx_text_internal(tx_text)
        self._build_top_block()

    # =========================================================
    # 文本 / 应用层分帧
    # =========================================================
    def _build_app_frame(self, payload: bytes) -> bytes:
        payload = payload or b" "
        length_bytes = len(payload).to_bytes(4, "big")
        header = self.APP_MAGIC + length_bytes
        crc = zlib.crc32(header + payload) & 0xFFFFFFFF
        return header + payload + crc.to_bytes(4, "big")

    def _parse_app_frame_exact(self, frame_bytes: bytes) -> Tuple[bool, bytes]:
        if len(frame_bytes) < 12:
            return False, b""
        if frame_bytes[:4] != self.APP_MAGIC:
            return False, b""
        payload_len = int.from_bytes(frame_bytes[4:8], "big")
        if len(frame_bytes) != payload_len + 12:
            return False, b""
        body = frame_bytes[:-4]
        crc_rx = int.from_bytes(frame_bytes[-4:], "big")
        crc_calc = zlib.crc32(body) & 0xFFFFFFFF
        if crc_rx != crc_calc:
            return False, b""
        return True, frame_bytes[8:-4]

    def _set_tx_text_internal(self, text: str):
        if text is None or len(text) == 0:
            text = " "
        payload = text.encode("utf-8")
        if len(payload) == 0:
            payload = b" "

        frame = self._build_app_frame(payload)
        frame_bits = self._frame_to_bits(frame)
        max_bits = self._max_data_bits_capacity()
        if frame_bits.size > max_bits:
            max_payload_bytes = max(1, (max_bits // 8) - 12)
            raise ValueError(
                f"AFDM 数据区容量不足：当前最大净载荷约 {max_payload_bytes} bytes，"
                f"当前文本 UTF-8 编码后为 {len(payload)} bytes。"
            )

        payload_repeat = max(1, min(4, max_bits // max(frame_bits.size, 1)))
        tx_bits = np.tile(frame_bits, payload_repeat).astype(np.int8)
        if tx_bits.size < max_bits:
            remainder = max_bits - tx_bits.size
            reps_need = remainder // frame_bits.size + 1
            pad = np.tile(frame_bits, reps_need)[:remainder].astype(np.int8)
            tx_bits = np.concatenate([tx_bits, pad])
        elif tx_bits.size > max_bits:
            tx_bits = tx_bits[:max_bits]

        data_syms = self._qam_modulate(tx_bits, self.mod_order).reshape(self.n_data_units, self._n_data)
        tx_frame = self._build_tx_waveform(data_syms)

        rms = np.sqrt(np.mean(np.abs(tx_frame) ** 2) + 1e-12)
        tx_frame = 0.2 * tx_frame / rms

        self._tx_text = text
        self._tx_payload = payload
        self._tx_frame = frame
        self._tx_frame_bits = frame_bits.astype(np.int8)
        self._payload_repeat = int(payload_repeat)
        self._tx_waveform = tx_frame.astype(np.complex64)

        self._rx_text = ""
        self._decode_ok = False
        self._match_bytes = 0
        self._last_good_rx_payload = b""
        self._last_raw_bytes = b""
        self._latest_constellation = np.zeros(0, dtype=np.complex64)
        self._latest_constellation_pre_eq = np.zeros(0, dtype=np.complex64)
        self._last_good_constellation = np.zeros(0, dtype=np.complex64)
        self._frame_soft_history.clear()
        self._combine_frames = 0

    def get_tx_text(self) -> str:
        return self._tx_text

    def get_rx_text(self) -> str:
        with self._lock:
            return self._rx_text

    def get_decode_stats(self) -> Dict[str, Any]:
        with self._lock:
            expected = len(self._tx_payload)
            ratio = (self._match_bytes / expected) if expected > 0 else 0.0
            return {
                "decode_ok": bool(self._decode_ok),
                "match_bytes": int(self._match_bytes),
                "expected_bytes": int(expected),
                "match_ratio": float(ratio),
            }

    # =========================================================
    # 参数 / 生命周期
    # =========================================================
    def _build_device_args(self) -> str:
        if self.device_type == "USRP B210":
            base = "type=b200,master_clock_rate=52e6"
        elif self.device_type == "USRP N210":
            base = "type=n200"
        elif self.device_type == "USRP X310":
            base = "type=x300"
        else:
            raise ValueError(f"Unsupported device_type: {self.device_type}")
        if self.serial:
            return f"serial={self.serial},{base}"
        return base

    def _import_runtime(self):
        try:
            from gnuradio import blocks, gr, uhd
            self._blocks = blocks
            self._gr = gr
            self._uhd = uhd
        except Exception as e:
            raise RuntimeError(
                "无法导入 GNU Radio / UHD 运行环境，请确认已安装 gnuradio、gnuradio-uhd。\n"
                f"原始错误: {e}"
            )

    def configure(
        self,
        carrier_freq: Optional[float] = None,
        samp_rate: Optional[float] = None,
        tx_gain: Optional[float] = None,
        rx_gain: Optional[float] = None,
        tx_text: Optional[str] = None,
        mod_order: Optional[str] = None,
        equalizer: Optional[str] = None,
        tx_antenna: Optional[str] = None,
        rx_antenna: Optional[str] = None,
        c1: Optional[float] = None,
        c2: Optional[float] = None,
    ):
        if self._running:
            raise RuntimeError("运行中不能重新配置，请先 stop()")

        rebuild_top = False
        rebuild_waveform = False
        rebuild_mats = False

        if carrier_freq is not None:
            self.carrier_freq = float(carrier_freq)
            rebuild_top = True
        if samp_rate is not None:
            self.sample_rate = float(samp_rate)
            self.samp_rate = self.sample_rate
            self.subcarrier_spacing = self.sample_rate / max(self.N, 1)
            rebuild_top = True
            rebuild_waveform = True
        if tx_gain is not None:
            self.tx_gain = float(tx_gain)
            rebuild_top = True
        if rx_gain is not None:
            self.rx_gain = float(rx_gain)
            rebuild_top = True
        if tx_antenna is not None:
            self.tx_antenna = str(tx_antenna)
            rebuild_top = True
        if rx_antenna is not None:
            self.rx_antenna = str(rx_antenna)
            rebuild_top = True
        if c1 is not None:
            self.c1 = float(c1)
            rebuild_mats = True
            rebuild_waveform = True
        if c2 is not None:
            self.c2 = float(c2)
            rebuild_mats = True
            rebuild_waveform = True

        if mod_order is not None:
            mod_order = str(mod_order).upper()
            if mod_order not in ("QPSK", "16QAM", "64QAM"):
                raise ValueError(f"Unsupported modulation: {mod_order}")
            self.mod_order = mod_order
            self.bits_per_symbol = self._get_bits_per_symbol(mod_order)
            rebuild_waveform = True

        if equalizer is not None:
            equalizer = str(equalizer).upper()
            if equalizer not in ("MMSE", "ZF"):
                raise ValueError(f"Unsupported equalizer: {equalizer}")
            self.equalizer = equalizer

        if rebuild_mats:
            self._refresh_afdm_mats()
            self._train_time = self._afdm_symbol_to_time_cp(self._train_dd_known)

        if tx_text is not None:
            self._set_tx_text_internal(str(tx_text))
            rebuild_top = True
            rebuild_waveform = False
        elif rebuild_waveform:
            self._set_tx_text_internal(self._tx_text)
            rebuild_top = True
            rebuild_waveform = False

        if rebuild_top:
            self._usrp_args = self._build_device_args()
            self._build_top_block()

    def start(self):
        if self._tb is None:
            raise RuntimeError("top_block 未构建")
        if self._running:
            return
        self._tb.start()
        self._running = True
        self._status = "running"
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        self._monitor_thread.start()

    def stop(self):
        if not self._running:
            return
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=1.5)
            self._monitor_thread = None
        self._tb.stop()
        self._tb.wait()
        self._running = False
        self._status = "stopped"

    def wait(self, timeout: Optional[float] = 2.0):
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=timeout)

    # =========================================================
    # GNU Radio / UHD
    # =========================================================
    def _build_top_block(self):
        gr = self._gr
        blocks = self._blocks
        uhd = self._uhd

        class _TopBlock(gr.top_block):
            pass

        tb = _TopBlock("AFDM Hardware Text Test", catch_exceptions=True)

        vector_source = blocks.vector_source_c(self._tx_waveform.tolist(), True, 1, [])
        tx_gain_block = blocks.multiply_const_cc(1.0)
        tx_sink_vec = blocks.vector_sink_c()
        rx_sink_vec = blocks.vector_sink_c()

        usrp_source = uhd.usrp_source(
            ",".join(("", self._usrp_args)),
            uhd.stream_args(cpu_format="fc32", args="", channels=list(range(0, 1))),
        )
        usrp_source.set_subdev_spec("A:A", 0)
        usrp_source.set_samp_rate(self.sample_rate)
        usrp_source.set_time_unknown_pps(uhd.time_spec(0))
        usrp_source.set_center_freq(self.carrier_freq, 0)
        usrp_source.set_antenna(self.rx_antenna, 0)
        usrp_source.set_gain(self.rx_gain, 0)

        usrp_sink = uhd.usrp_sink(
            ",".join(("", self._usrp_args)),
            uhd.stream_args(cpu_format="fc32", args="", channels=list(range(0, 1))),
            "",
        )
        usrp_sink.set_subdev_spec("A:A", 0)
        usrp_sink.set_samp_rate(self.sample_rate)
        usrp_sink.set_time_unknown_pps(uhd.time_spec(0))
        usrp_sink.set_center_freq(self.carrier_freq, 0)
        usrp_sink.set_antenna(self.tx_antenna, 0)
        usrp_sink.set_gain(self.tx_gain, 0)

        tb.connect((vector_source, 0), (tx_gain_block, 0))
        tb.connect((tx_gain_block, 0), (usrp_sink, 0))
        tb.connect((tx_gain_block, 0), (tx_sink_vec, 0))
        tb.connect((usrp_source, 0), (rx_sink_vec, 0))

        self._tb = tb
        self._usrp_source = usrp_source
        self._usrp_sink = usrp_sink
        self._tx_sink_vec = tx_sink_vec
        self._rx_sink_vec = rx_sink_vec

        with self._lock:
            self._tx_buffer.clear()
            self._rx_buffer.clear()
            self._latest_constellation = np.zeros(0, dtype=np.complex64)
            self._latest_constellation_pre_eq = np.zeros(0, dtype=np.complex64)
            self._rx_text = ""
            self._decode_ok = False
            self._match_bytes = 0
            self._last_error = ""
            self._status = "configured"

        self._rx_samples_seen = 0
        self._last_processed_abs_start = -10**18

    # =========================================================
    # 监控线程
    # =========================================================
    def _monitor_worker(self):
        process_window_len = min(self._buffer_keep, max(4 * self.frame_len, 8192))
        while not self._monitor_stop.is_set():
            try:
                tx_data = np.asarray(self._tx_sink_vec.data(), dtype=np.complex64)
                rx_data = np.asarray(self._rx_sink_vec.data(), dtype=np.complex64)
                self._tx_sink_vec.reset()
                self._rx_sink_vec.reset()

                if tx_data.size > 0 or rx_data.size > 0:
                    with self._lock:
                        if tx_data.size > 0:
                            keep_tx = tx_data[-4096:]
                            self._latest_tx_samples = keep_tx.astype(np.complex64, copy=False)
                            for s in keep_tx:
                                self._tx_buffer.append(s)
                        if rx_data.size > 0:
                            keep_rx = rx_data[-8192:]
                            self._latest_rx_samples = keep_rx.astype(np.complex64, copy=False)
                            for s in keep_rx:
                                self._rx_buffer.append(s)
                            self._rx_samples_seen += int(rx_data.size)

                with self._lock:
                    rx_window = np.asarray(self._rx_buffer, dtype=np.complex64)
                    abs_seen = int(self._rx_samples_seen)

                if rx_window.size >= self.frame_len:
                    if rx_window.size > process_window_len:
                        rx_window = rx_window[-process_window_len:]
                    try:
                        self._try_process_rx_window(rx_window.astype(np.complex128), abs_seen)
                    except Exception as e:
                        self._last_error = f"process frame failed: {e}"

                time.sleep(max(self.update_period, 0.05))
            except Exception as e:
                self._last_error = f"monitor outer failure: {e}"
                time.sleep(0.2)

    # =========================================================
    # 接收处理
    # =========================================================
    def _try_process_rx_window(self, rx_window: np.ndarray, abs_seen: int):
        metric = self._sync_metric(rx_window)
        if metric.size <= 1:
            return
        max_metric = float(np.max(metric))
        self.last_sync_metric = max_metric

        peaks = self._find_sync_peaks(metric, max_candidates=2)
        if not peaks:
            self.last_frame_ok = False
            self.last_bad_reason = f"sync_peak_not_found({max_metric:.3f})"
            return

        best = None
        saw_pending = False

        for coarse_peak in peaks:
            sync_start = self._refine_sync_start(rx_window, int(coarse_peak), search_radius=max(12, self.cp_len))
            frame_start = sync_start - self.pre_guard_len
            if frame_start < 0:
                continue
            frame_end = frame_start + self.frame_len
            if frame_end > len(rx_window):
                saw_pending = True
                continue

            abs_frame_start = abs_seen - len(rx_window) + frame_start
            if abs_frame_start <= self._last_processed_abs_start + self.frame_len // 2:
                continue

            cfo_hz = self._estimate_cfo_from_preamble(rx_window, sync_start, self.sync_half_len)
            frame_raw = rx_window[frame_start:frame_end].copy()
            t_idx = np.arange(frame_raw.size, dtype=np.float64)
            frame = frame_raw * np.exp(-1j * 2.0 * np.pi * cfo_hz * t_idx / max(self.sample_rate, 1e-12))

            train_time = frame[self._off_train: self._off_train + self.block_len]
            data_time = frame[self._off_data: self._off_end]
            if len(data_time) != self.n_data_units * self.block_len:
                continue
            data_blocks = data_time.reshape(self.n_data_units, self.block_len)

            train_td = train_time[self.cp_len: self.cp_len + self.N]
            y_train_dd = self._A_rx @ train_td
            h_train = np.zeros(self.N, dtype=np.complex128)
            active = np.abs(self._train_dd_known) > 1e-8
            h_train[active] = y_train_dd[active] / self._train_dd_known[active]

            train_ref = self._train_dd_known[active]
            train_obs = y_train_dd[active]
            train_nmse = float(
                np.sqrt(
                    np.mean(np.abs(train_obs - h_train[active] * train_ref) ** 2)
                    / (np.mean(np.abs(train_obs) ** 2) + 1e-12)
                )
            )

            x_hat_units = []
            pre_eq_units = []
            alpha_hist = []
            pilot_err_hist = []
            for sym_idx, blk in enumerate(data_blocks):
                td = blk[self.cp_len: self.cp_len + self.N]
                y_dd = self._A_rx @ td
                pilots = self._pilot_values(sym_idx)
                pilot_h = h_train[self.pilot_idx]
                pilot_pred = pilot_h * pilots
                pilot_obs = y_dd[self.pilot_idx]
                denom = np.vdot(pilot_pred, pilot_pred)
                if abs(denom) > 1e-12:
                    alpha = np.vdot(pilot_pred, pilot_obs) / denom
                else:
                    alpha = 1.0 + 0.0j
                if abs(alpha) < 1e-6:
                    alpha = 1.0 + 0.0j

                pilot_err = float(
                    np.sqrt(
                        np.mean(np.abs(pilot_obs - alpha * pilot_pred) ** 2)
                        / (np.mean(np.abs(pilot_obs) ** 2) + 1e-12)
                    )
                )
                pilot_err_hist.append(pilot_err)
                alpha_hist.append(alpha)

                h_eff = alpha * h_train[self.data_idx]
                y_eff = y_dd[self.data_idx]
                pre_eq_units.append(y_eff.copy())
                x_hat = self._equalize_data(y_eff, h_eff)
                x_hat_units.append(x_hat)

            if not x_hat_units:
                continue
            rx_syms = np.concatenate(x_hat_units)
            pre_eq_syms = np.concatenate(pre_eq_units) if pre_eq_units else np.zeros(0, dtype=np.complex128)
            ber, raw_bytes, rx_payload, rx_text, match_bytes, decode_ok, rx_syms_best = (
                self._recover_payload_from_symbols(rx_syms)
            )

            sync_here = float(metric[min(max(int(coarse_peak), 0), len(metric) - 1)])
            pilot_nmse = float(np.mean(pilot_err_hist)) if pilot_err_hist else 1.0
            alpha_abs = float(np.mean(np.abs(alpha_hist))) if alpha_hist else 0.0
            quality_fail = bool(
                sync_here < self.sync_metric_threshold
                or not np.isfinite(pilot_nmse)
                or pilot_nmse > 2.5
                or not np.isfinite(alpha_abs)
                or alpha_abs < 1e-3
            )
            score = (
                1200.0 * float(decode_ok)
                + 140.0 * (1.0 - min(ber, 1.0))
                + 40.0 * max(sync_here, 0.0)
                + 12.0 * max(0.0, 2.0 - float(pilot_nmse))
                + 3.0 * min(alpha_abs, 3.0)
                - 60.0 * float(quality_fail)
            )

            cand = {
                "score": float(score),
                "sync_start": int(sync_start),
                "frame_start": int(frame_start),
                "abs_frame_start": int(abs_frame_start),
                "sync_metric": float(sync_here),
                "cfo_est_hz": float(cfo_hz),
                "train_nmse": float(train_nmse),
                "pilot_nmse": float(pilot_nmse),
                "alpha_abs": float(alpha_abs),
                "ber": float(ber),
                "raw_bytes": raw_bytes,
                "rx_payload": rx_payload,
                "rx_text": rx_text,
                "match_bytes": int(match_bytes),
                "decode_ok": bool(decode_ok),
                "quality_fail": bool(quality_fail),
                "rx_syms": rx_syms_best,
                "data_pre_eq_syms": pre_eq_syms,
            }
            if best is None or cand["score"] > best["score"]:
                best = cand
            if decode_ok and not quality_fail:
                break

        if best is None:
            self.last_frame_ok = False
            self.last_bad_reason = "frame_pending" if saw_pending else f"candidate_decode_fail({max_metric:.3f})"
            return

        self._last_processed_abs_start = best["abs_frame_start"]
        self.last_sync_index = int(best["frame_start"])
        self.last_payload_start = int(best["frame_start"] + self._off_train)
        self.last_sync_metric = float(best["sync_metric"])
        self.last_cfo_est_hz = float(best["cfo_est_hz"])
        self.last_train_nmse = float(best["train_nmse"])
        self.last_pilot_nmse = float(best["pilot_nmse"])
        self.last_alpha_abs = float(best["alpha_abs"])

        if best["quality_fail"] and not best["decode_ok"]:
            self.last_frame_ok = False
            self.last_bad_reason = "quality_gate"
        else:
            self.last_frame_ok = True
            self.last_bad_reason = "ok" if best["decode_ok"] else "soft_ok"

        const_points = self._prepare_constellation_points(best["rx_syms"], display_mode="raw")
        pre_eq_points = self._prepare_constellation_points(best.get("data_pre_eq_syms"), display_mode="raw")
        t_now = time.time() - self._t0
        with self._lock:
            self._latest_constellation = const_points.astype(np.complex64)
            self._latest_constellation_pre_eq = pre_eq_points.astype(np.complex64)
            self._ber_hist_t.append(t_now)
            self._ber_hist_v.append(max(float(best["ber"]), 1e-5))
            if len(self._ber_hist_t) > 200:
                self._ber_hist_t = self._ber_hist_t[-200:]
                self._ber_hist_v = self._ber_hist_v[-200:]
            self._last_good_constellation = self._latest_constellation.copy()
            self._last_raw_bytes = best["raw_bytes"]
            if best["rx_payload"]:
                self._rx_text = best["rx_text"]
                self._last_good_rx_payload = best["rx_payload"]
                self._decode_ok = bool(best["decode_ok"])
                self._match_bytes = int(best["match_bytes"])
        self._status = "running"

    # =========================================================
    # AFDM 生成 / 同步 / 检测
    # =========================================================
    def _refresh_afdm_mats(self):
        n = np.arange(self.N)
        F = np.fft.fft(np.eye(self.N), axis=0) / np.sqrt(self.N)
        Fh = F.conj().T
        D1 = np.diag(np.exp(-1j * 2.0 * np.pi * float(self.c1) * (n ** 2) / self.N))
        D2 = np.diag(np.exp(-1j * 2.0 * np.pi * float(self.c2) * (n ** 2) / self.N))
        self._A_tx = D1.conj().T @ Fh @ D2.conj().T
        self._A_rx = D2 @ F @ D1

    @staticmethod
    def _build_afdm_resource_plan(n_fft: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n_fft != 64:
            raise ValueError("This AFDM implementation is defined for N == 64")
        active_bins = np.concatenate([np.arange(-26, 0), np.arange(1, 27)])
        pilot_bins = np.array([-21, -7, 7, 21])
        active_idx = active_bins + n_fft // 2
        pilot_idx = pilot_bins + n_fft // 2
        pilot_set = set(pilot_idx.tolist())
        data_idx = np.asarray([x for x in active_idx.tolist() if x not in pilot_set], dtype=np.int64)
        return active_idx.astype(np.int64), pilot_idx.astype(np.int64), data_idx

    @staticmethod
    def _training_active_sequence(n_active: int) -> np.ndarray:
        train = np.ones(n_active, dtype=np.complex128)
        train[1::2] = -1.0 + 0.0j
        return train

    def _build_sync_preamble(self, half_len: int) -> np.ndarray:
        idx = np.arange(half_len, dtype=np.float64)
        tones = np.exp(1j * 2.0 * np.pi * idx * (idx + 1.0) / max(2.0 * half_len, 1.0))
        tones = tones / np.sqrt(np.mean(np.abs(tones) ** 2) + 1e-12)
        return np.concatenate([tones, tones]).astype(np.complex128)

    def _pilot_values(self, sym_idx: int) -> np.ndarray:
        polarity_seq = np.array([1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1], dtype=np.float64)
        polarity = polarity_seq[sym_idx % len(polarity_seq)]
        base = np.array([1.0, 1.0, 1.0, -1.0], dtype=np.float64)
        return (polarity * base).astype(np.complex128)

    def _afdm_symbol_to_time_cp(self, x_dd: np.ndarray) -> np.ndarray:
        td = self._A_tx @ np.asarray(x_dd, dtype=np.complex128).reshape(-1)
        cp = td[-self.cp_len:]
        return np.concatenate([cp, td]).astype(np.complex128)

    def _build_tx_waveform(self, data_syms: np.ndarray) -> np.ndarray:
        parts = [
            np.zeros(self.pre_guard_len, dtype=np.complex128),
            self.sync_preamble.astype(np.complex128),
            self._train_time.astype(np.complex128),
        ]
        for sym_idx in range(self.n_data_units):
            x_dd = np.zeros(self.N, dtype=np.complex128)
            x_dd[self.pilot_idx] = self._pilot_values(sym_idx)
            x_dd[self.data_idx] = data_syms[sym_idx]
            parts.append(self._afdm_symbol_to_time_cp(x_dd))
        parts.append(np.zeros(self.post_guard_len, dtype=np.complex128))
        return np.concatenate(parts).astype(np.complex128)

    def _sync_metric(self, rx: np.ndarray) -> np.ndarray:
        rx = np.asarray(rx, dtype=np.complex128)
        Ls = int(self.sync_len)
        L = int(self.sync_half_len)
        if rx.size < Ls + 1:
            return np.zeros(1, dtype=np.float64)
        n_out = rx.size - Ls + 1

        sync = self.sync_preamble.astype(np.complex128)
        cross_corr = np.correlate(rx, sync, mode="valid")
        cross_mag2 = np.abs(cross_corr) ** 2

        rx_abs2 = np.abs(rx) ** 2
        cum = np.concatenate([[0.0], np.cumsum(rx_abs2)])
        seg_energy = cum[Ls:] - cum[: rx.size - Ls + 1]
        seg_energy = seg_energy + 1e-12
        m_cross = cross_mag2 / (self._sync_energy * seg_energy)

        prod = np.conj(rx[:-L]) * rx[L:]
        prod_cum = np.concatenate([[0.0 + 0.0j], np.cumsum(prod)])
        P = prod_cum[L:] - prod_cum[: prod.size - L + 1]
        R_a = cum[L: rx.size - L + 1] - cum[: rx.size - 2 * L + 1]
        R_b = cum[2 * L: rx.size + 1] - cum[L: rx.size - L + 1]
        m_auto = (np.abs(P) ** 2) / (R_a * R_b + 1e-12)

        n = min(m_cross.size, m_auto.size)
        return 0.5 * m_cross[:n] + 0.5 * m_auto[:n]

    def _find_sync_peaks(self, metric: np.ndarray, max_candidates: int = 2) -> List[int]:
        if metric.size <= 3:
            return []
        max_metric = float(np.max(metric))
        thr = max(self.sync_metric_threshold, 0.55 * max_metric)
        peaks: List[Tuple[float, int]] = []
        for i in range(1, metric.size - 1):
            if metric[i] >= thr and metric[i] >= metric[i - 1] and metric[i] >= metric[i + 1]:
                peaks.append((float(metric[i]), int(i)))
        if not peaks:
            idx = int(np.argmax(metric))
            return [idx] if metric[idx] >= max(self.sync_metric_threshold, 0.10) else []
        peaks.sort(key=lambda x: x[0], reverse=True)
        min_sep = max(1, self.frame_len // 3)
        out: List[int] = []
        for _, idx in peaks:
            if all(abs(idx - j) > min_sep for j in out):
                out.append(idx)
            if len(out) >= max_candidates:
                break
        return out

    def _refine_sync_start(self, rx: np.ndarray, coarse: int, search_radius: int) -> int:
        sync = self.sync_preamble.astype(np.complex128)
        Ls = sync.size
        lo = max(0, int(coarse) - int(search_radius))
        hi = min(rx.size - Ls, int(coarse) + int(search_radius))
        if hi <= lo:
            return int(coarse)
        best_idx = int(coarse)
        best_score = -1.0
        for s in range(lo, hi + 1):
            seg = rx[s: s + Ls]
            seg_e = float(np.vdot(seg, seg).real) + 1e-12
            corr = np.vdot(sync, seg)
            score = float((np.abs(corr) ** 2) / (self._sync_energy * seg_e))
            if score > best_score:
                best_score = score
                best_idx = int(s)
        return best_idx

    def _estimate_cfo_from_preamble(self, rx: np.ndarray, sync_start: int, L: int) -> float:
        if sync_start + 2 * L > rx.size:
            return 0.0
        a = rx[sync_start: sync_start + L]
        b = rx[sync_start + L: sync_start + 2 * L]
        P = np.sum(a * np.conj(b))
        phase = float(np.angle(P))
        return float(-phase * self.sample_rate / (2.0 * np.pi * max(L, 1)))

    def _equalize_data(self, y_eff: np.ndarray, h_eff: np.ndarray) -> np.ndarray:
        y_eff = np.asarray(y_eff, dtype=np.complex128)
        h_eff = np.asarray(h_eff, dtype=np.complex128)
        if self.equalizer == "ZF":
            out = np.zeros_like(y_eff)
            mask = np.abs(h_eff) > 1e-8
            out[mask] = y_eff[mask] / h_eff[mask]
            return out
        h_power_mean = float(np.mean(np.abs(h_eff) ** 2)) + 1e-12
        noise_var = self._noise_var_reg * h_power_mean
        denom = np.abs(h_eff) ** 2 + noise_var + 1e-12
        return y_eff * np.conj(h_eff) / denom

    # =========================================================
    # 文本恢复
    # =========================================================
    def _recover_payload_from_symbols(self, rx_syms: np.ndarray):
        rx_syms = np.asarray(rx_syms, dtype=np.complex128).reshape(-1)
        frame_bits_len = int(self._tx_frame_bits.size)
        total_bits_need = int(frame_bits_len * self._payload_repeat)
        if frame_bits_len <= 0 or total_bits_need <= 0:
            return 1.0, b"", b"", "", 0, False, rx_syms

        rx_bits_all = self._qam_demodulate(rx_syms, self.mod_order)
        if rx_bits_all.size < total_bits_need:
            return 1.0, b"", b"", "", 0, False, rx_syms

        if self.mod_order == "QPSK":
            soft_vals = np.empty(rx_bits_all.size, dtype=np.float64)
            soft_vals[0::2] = np.imag(rx_syms)
            soft_vals[1::2] = np.real(rx_syms)
            soft_rep = soft_vals[:total_bits_need].reshape(self._payload_repeat, frame_bits_len)
            soft_sum = np.sum(soft_rep, axis=0)
            tail_start_bits = self._payload_repeat * frame_bits_len
            tail_soft = soft_vals[tail_start_bits:]
            tail_len = min(tail_soft.size, frame_bits_len)
            if tail_len > 0:
                soft_sum[:tail_len] += tail_soft[:tail_len]
            frame_bits_single = (soft_sum < 0.0).astype(np.int8)
            soft_sum_for_hist = soft_sum
        else:
            rx_bits_rep = rx_bits_all[:total_bits_need].reshape(self._payload_repeat, frame_bits_len)
            votes = np.sum(rx_bits_rep, axis=0).astype(np.int32)
            n_votes = np.full(frame_bits_len, self._payload_repeat, dtype=np.int32)
            tail_start = self._payload_repeat * frame_bits_len
            tail = rx_bits_all[tail_start:]
            tail_len = min(tail.size, frame_bits_len)
            if tail_len > 0:
                votes[:tail_len] += tail[:tail_len].astype(np.int32)
                n_votes[:tail_len] += 1
            frame_bits_single = (votes * 2 >= n_votes).astype(np.int8)
            soft_sum_for_hist = (1 - 2 * frame_bits_single).astype(np.float64)

        ber_single = float(np.mean(frame_bits_single != self._tx_frame_bits))

        soft_hist = list(self._frame_soft_history) + [soft_sum_for_hist]
        soft_hist_sum = np.sum(np.stack(soft_hist, axis=0), axis=0)
        frame_bits_comb = (soft_hist_sum < 0.0).astype(np.int8)
        ber_comb = float(np.mean(frame_bits_comb != self._tx_frame_bits))

        best = {
            "score": -1e18,
            "ber": 1.0,
            "raw_bytes": b"",
            "rx_payload": b"",
            "rx_text": "",
            "match_bytes": 0,
            "decode_ok": False,
            "syms": rx_syms,
            "soft_for_hist": None,
        }

        for bits_use, ber_use, mode_bonus in (
            (frame_bits_single, ber_single, 0.0),
            (frame_bits_comb, ber_comb, 0.2),
        ):
            frame_bytes = self._bits_to_bytes(bits_use)
            ok, rx_payload = self._parse_app_frame_exact(frame_bytes)
            rx_text = rx_payload.decode("utf-8", errors="replace") if ok else ""
            match_bytes = (
                int(sum(int(a == b) for a, b in zip(rx_payload, self._tx_payload)))
                if ok else 0
            )
            decode_ok = bool(ok and rx_payload == self._tx_payload)
            score = (
                2000.0 * float(decode_ok)
                + 200.0 * float(ok)
                + 20.0 * max(0.0, 1.0 - min(ber_use, 1.0))
                + mode_bonus
            )
            if score > best["score"]:
                best = {
                    "score": score,
                    "ber": ber_use,
                    "raw_bytes": frame_bytes,
                    "rx_payload": rx_payload,
                    "rx_text": rx_text,
                    "match_bytes": match_bytes,
                    "decode_ok": decode_ok,
                    "syms": rx_syms,
                    "soft_for_hist": soft_sum_for_hist.copy(),
                }
            if decode_ok:
                break

        if best.get("soft_for_hist") is not None and best["ber"] < 0.45:
            self._frame_soft_history.append(best["soft_for_hist"].copy())
        self._combine_frames = len(self._frame_soft_history)

        return (
            float(best["ber"]),
            best["raw_bytes"],
            best["rx_payload"],
            best["rx_text"],
            int(best["match_bytes"]),
            bool(best["decode_ok"]),
            best["syms"],
        )

    # =========================================================
    # 星座 / 比特 / QAM
    # =========================================================
    def _prepare_constellation_points(
        self,
        rx_data_syms: np.ndarray,
        display_mode: str = "raw",
    ) -> np.ndarray:
        """外点过滤 + 可选显示模式变换。仅影响 UI 显示。"""
        if rx_data_syms is None:
            return np.zeros(0, dtype=np.complex64)
        arr = np.asarray(rx_data_syms, dtype=np.complex64).reshape(-1)
        if arr.size == 0:
            return np.zeros(0, dtype=np.complex64)

        mag = np.abs(arr)
        median_mag = float(np.median(mag))
        if median_mag > 1e-6:
            threshold = 4.0 * median_mag
            keep = mag < threshold
            if np.any(keep):
                arr = arr[keep]

        if display_mode != "raw":
            arr = self._apply_display_mode(arr, display_mode)

        if arr.size > 1024:
            arr = arr[-1024:]
        return arr.copy()

    def _apply_display_mode(self, arr: np.ndarray, mode: str) -> np.ndarray:
        """星座图显示模式变换。只支持 QPSK；其它调制阶直接返回原值。"""
        arr = np.asarray(arr, dtype=np.complex64).reshape(-1)
        if arr.size == 0:
            return arr
        mode = str(mode).lower()
        if mode in ("raw", "pre_equalized") or self.mod_order != "QPSK":
            return arr
        if mode not in ("dd_refined", "hard_decision"):
            return arr

        target_radius = 1.0 / np.sqrt(2.0)
        avg_mag = float(np.median(np.abs(arr)))
        if avg_mag > 1e-6:
            arr = arr * (target_radius / avg_mag)

        qpsk_points = np.array(
            [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex64
        ) * target_radius
        dist = np.abs(arr[:, None] - qpsk_points[None, :])
        nearest = np.argmin(dist, axis=1)
        decisions = qpsk_points[nearest]

        if mode == "hard_decision":
            return decisions.copy()

        return (decisions + (arr - decisions) * 0.25).astype(np.complex64)

    def set_constellation_display_mode(self, mode: str):
        """运行中切换星座图显示模式。立即生效，不影响解调。"""
        mode = str(mode).lower()
        if mode not in ("raw", "dd_refined", "hard_decision", "pre_equalized"):
            raise ValueError(
                f"Unsupported constellation_display_mode: {mode}; "
                'expected "raw" | "dd_refined" | "hard_decision" | "pre_equalized"'
            )
        with self._lock:
            self.constellation_display_mode = mode

    def get_constellation_display_mode(self) -> str:
        with self._lock:
            return self.constellation_display_mode

    def _frame_to_bits(self, frame: bytes) -> np.ndarray:
        if len(frame) == 0:
            return np.zeros(0, dtype=np.int8)
        arr = np.frombuffer(frame, dtype=np.uint8)
        return np.unpackbits(arr, bitorder="big").astype(np.int8)

    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        usable = (bits.size // 8) * 8
        if usable <= 0:
            return b""
        arr = np.packbits(bits[:usable], bitorder="big")
        return bytes(arr.tolist())

    def _max_data_bits_capacity(self) -> int:
        return self.n_data_units * self._n_data * self.bits_per_symbol

    @staticmethod
    def _get_bits_per_symbol(mod_order: str) -> int:
        if mod_order == "QPSK":
            return 2
        if mod_order == "16QAM":
            return 4
        if mod_order == "64QAM":
            return 6
        raise ValueError(f"Unsupported modulation: {mod_order}")

    def _qam_modulate(self, bits: np.ndarray, mod_order: str) -> np.ndarray:
        mod_order = str(mod_order).upper()
        if mod_order == "QPSK":
            bits = np.asarray(bits, dtype=np.int8).reshape(-1, 2)
            i = 1 - 2 * bits[:, 1]
            q = 1 - 2 * bits[:, 0]
            return ((i + 1j * q) / np.sqrt(2)).astype(np.complex128)
        if mod_order == "16QAM":
            bits = np.asarray(bits, dtype=np.int8).reshape(-1, 4)
            lookup = np.array([3.0, 1.0, -3.0, -1.0], dtype=np.float64)
            i_idx = (bits[:, 0].astype(np.int64) << 1) | bits[:, 1].astype(np.int64)
            q_idx = (bits[:, 2].astype(np.int64) << 1) | bits[:, 3].astype(np.int64)
            i = lookup[i_idx]
            q = lookup[q_idx]
            return ((i + 1j * q) / np.sqrt(10)).astype(np.complex128)
        if mod_order == "64QAM":
            bits = np.asarray(bits, dtype=np.int8).reshape(-1, 6)
            table = {
                (0, 0, 0): 7, (0, 0, 1): 5, (0, 1, 1): 3, (0, 1, 0): 1,
                (1, 1, 0): -1, (1, 1, 1): -3, (1, 0, 1): -5, (1, 0, 0): -7,
            }
            i = np.array([table[tuple(b[:3].tolist())] for b in bits], dtype=np.float64)
            q = np.array([table[tuple(b[3:].tolist())] for b in bits], dtype=np.float64)
            return ((i + 1j * q) / np.sqrt(42)).astype(np.complex128)
        raise ValueError(f"Unsupported modulation: {mod_order}")

    def _qam_demodulate(self, syms: np.ndarray, mod_order: str) -> np.ndarray:
        mod_order = str(mod_order).upper()
        if mod_order == "QPSK":
            bits = np.zeros((len(syms), 2), dtype=np.int8)
            bits[:, 0] = (np.imag(syms) < 0).astype(np.int8)
            bits[:, 1] = (np.real(syms) < 0).astype(np.int8)
            return bits.reshape(-1)
        if mod_order == "16QAM":
            x = np.real(syms) * np.sqrt(10)
            y = np.imag(syms) * np.sqrt(10)
            bits = np.zeros((len(syms), 4), dtype=np.int8)
            bits[:, 0] = (x < 0).astype(np.int8)
            bits[:, 1] = (np.abs(x) < 2).astype(np.int8)
            bits[:, 2] = (y < 0).astype(np.int8)
            bits[:, 3] = (np.abs(y) < 2).astype(np.int8)
            return bits.reshape(-1)
        if mod_order == "64QAM":
            x = np.real(syms) * np.sqrt(42)
            y = np.imag(syms) * np.sqrt(42)
            def slicer(v: float):
                if v >= 6: return (0, 0, 0)
                if v >= 4: return (0, 0, 1)
                if v >= 2: return (0, 1, 1)
                if v >= 0: return (0, 1, 0)
                if v >= -2: return (1, 1, 0)
                if v >= -4: return (1, 1, 1)
                if v >= -6: return (1, 0, 1)
                return (1, 0, 0)
            out = np.zeros((len(syms), 6), dtype=np.int8)
            for k, (iv, qv) in enumerate(zip(x, y)):
                out[k, 0], out[k, 1], out[k, 2] = slicer(iv)
                out[k, 3], out[k, 4], out[k, 5] = slicer(qv)
            return out.reshape(-1)
        raise ValueError(f"Unsupported modulation: {mod_order}")

    # =========================================================
    # UI 接口
    # =========================================================
    def get_spectrum(self, num_samples: int = 2048):
        return self.get_rx_samples(num_samples)

    def get_tx_samples(self, num_samples: int = 2048):
        with self._lock:
            n = max(1, int(num_samples))
            arr = np.asarray(self._tx_buffer, dtype=np.complex64)
            if arr.size == 0:
                return np.zeros(n, dtype=np.complex64)
            return arr[-n:].copy() if arr.size >= n else np.pad(arr, (n - arr.size, 0))

    def get_rx_samples(self, num_samples: int = 2048):
        with self._lock:
            n = max(1, int(num_samples))
            arr = np.asarray(self._rx_buffer, dtype=np.complex64)
            if arr.size == 0:
                return np.zeros(n, dtype=np.complex64)
            return arr[-n:].copy() if arr.size >= n else np.pad(arr, (n - arr.size, 0))

    def get_constellation(self):
        """返回当前显示模式处理后的星座点（默认 dd_refined）。"""
        with self._lock:
            mode = self.constellation_display_mode
            if mode == "pre_equalized":
                raw = self._latest_constellation_pre_eq.copy()
            else:
                raw = self._latest_constellation.copy()
        if raw.size == 0:
            return raw
        return self._apply_display_mode(raw, mode)

    def get_rx_constellation(
        self,
        max_points: int = 256,
        source: Optional[str] = None,
        display_mode: Optional[str] = None,
    ):
        with self._lock:
            mode = (
                str(display_mode).lower()
                if display_mode is not None
                else self.constellation_display_mode
            )
            if mode == "pre_equalized":
                raw = self._latest_constellation_pre_eq.copy()
            else:
                raw = self._latest_constellation.copy()
        if raw.size == 0:
            return raw
        pts = self._apply_display_mode(raw, mode)
        if pts.size <= max_points:
            return pts
        idx = np.linspace(0, pts.size - 1, max_points, dtype=np.int64)
        return pts[idx].copy()

    def get_tx_spectrum_source(self, num_samples: int = 2048):
        return self.get_tx_samples(num_samples)

    def get_rx_spectrum_source(self, num_samples: int = 2048):
        return self.get_rx_samples(num_samples)

    def get_estimated_ber(self):
        with self._lock:
            return (
                np.array(self._ber_hist_t, dtype=np.float64),
                np.array(self._ber_hist_v, dtype=np.float64),
            )

    def get_debug_snapshot(self):
        with self._lock:
            last_ber = float(self._ber_hist_v[-1]) if len(self._ber_hist_v) > 0 else float("nan")
            return {
                "frame_ok": bool(self.last_frame_ok),
                "reason": str(self.last_bad_reason or "ok"),
                "sync_idx": int(self.last_sync_index),
                "payload_start": int(self.last_payload_start),
                "sync_metric": float(self.last_sync_metric),
                "cfo_est_hz": float(self.last_cfo_est_hz),
                "ber": last_ber,
                "train_nmse": float(self.last_train_nmse),
                "pilot_nmse": float(self.last_pilot_nmse),
                "alpha_abs": float(self.last_alpha_abs),
                "combine_frames": int(self._combine_frames),
                "payload_repeat": int(self._payload_repeat),
            }

    def get_status(self) -> Dict[str, Any]:
        snap = self.get_debug_snapshot()
        stats = self.get_decode_stats()
        return {
            "status": self._status,
            "carrier_freq": self.carrier_freq,
            "samp_rate": self.samp_rate,
            "sample_rate": self.sample_rate,
            "tx_gain": self.tx_gain,
            "rx_gain": self.rx_gain,
            "device_type": self.device_type,
            "device_args": self._usrp_args,
            "last_error": self._last_error,
            "mod_order": self.mod_order,
            "equalizer": self.equalizer,
            "constellation_display_mode": self.constellation_display_mode,
            "c1": self.c1,
            "c2": self.c2,
            "frame_ok": snap["frame_ok"],
            "reason": snap["reason"],
            "sync_metric": snap["sync_metric"],
            "cfo_est_hz": snap["cfo_est_hz"],
            "ber": snap["ber"],
            "train_nmse": snap["train_nmse"],
            "pilot_nmse": snap["pilot_nmse"],
            "alpha_abs": snap["alpha_abs"],
            "combine_frames": snap["combine_frames"],
            "payload_repeat": snap["payload_repeat"],
            "decode_ok": stats["decode_ok"],
            "match_bytes": stats["match_bytes"],
            "expected_bytes": stats["expected_bytes"],
            "match_ratio": stats["match_ratio"],
        }

    def get_last_error(self) -> str:
        return self._last_error

    def set_center_freq(self, value: float):
        self.carrier_freq = float(value)
        self._usrp_sink.set_center_freq(self.carrier_freq, 0)
        self._usrp_source.set_center_freq(self.carrier_freq, 0)

    def set_tx_gain(self, value: float):
        self.tx_gain = float(value)
        self._usrp_sink.set_gain(self.tx_gain, 0)

    def set_rx_gain(self, value: float):
        self.rx_gain = float(value)
        self._usrp_source.set_gain(self.rx_gain, 0)

    def set_samp_rate(self, value: float):
        self.sample_rate = float(value)
        self.samp_rate = self.sample_rate
        self.subcarrier_spacing = self.sample_rate / max(self.N, 1)
        self._usrp_sink.set_samp_rate(self.sample_rate)
        self._usrp_source.set_samp_rate(self.sample_rate)

    def set_mod_order(self, mod_order: str):
        mod_order = str(mod_order).upper()
        if mod_order not in ("QPSK", "16QAM", "64QAM"):
            raise ValueError(f"Unsupported modulation: {mod_order}")
        if mod_order == self.mod_order:
            return
        was_running = bool(self._running)
        if was_running:
            self.stop()
        self.configure(mod_order=mod_order)
        if was_running:
            self.start()


# ---------------------------------------------------------------------------
# 阶段6：统一硬件后端兼容壳
# ---------------------------------------------------------------------------
class AFDMHardwareTest:
    """兼容壳：委托 _LegacyAFDMHardwareTest，公开接口不变。"""

    def __init__(self, *args, backend=None, **kwargs):
        self._backend = backend if backend is not None else _LegacyAFDMHardwareTest(*args, **kwargs)

    def __getattr__(self, name):
        backend = self.__dict__.get("_backend")
        if backend is not None and hasattr(backend, name):
            return getattr(backend, name)
        raise AttributeError(name)


if __name__ == "__main__":
    tb = AFDMHardwareTest(
        carrier_freq=2.4e9,
        samp_rate=960000.0,
        tx_gain=40.0,
        rx_gain=40.0,
        device_type="USRP B210",
        mod_order="QPSK",
        equalizer="MMSE",
        tx_text="Hello AFDM Hardware Test!",
    )
    print("before start:", tb.get_status())
    tb.start()
    try:
        for _ in range(20):
            time.sleep(0.2)
            print(tb.get_status())
            print(tb.get_decode_stats(), tb.get_rx_text())
    finally:
        tb.stop()
        tb.wait()
        print("stopped:", tb.get_status())
