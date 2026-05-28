# -*- coding: utf-8 -*-
"""
hardware/ofdm_hardtest.py

OFDM 真机硬件测试后端（重构版）
================================

设计目标
--------
- 仅修改 OFDM 后端，保持 HardwareTestTab 现有调用接口不变
- 不依赖 GNU Radio 的 OFDM header/payload 门控来产出星座与 BER
- 发送端在 Python 中直接生成 OFDM 基带波形，GNU Radio / UHD 只负责持续发射与采样
- 接收端在 Python 中完成：同步、CFO 估计、训练信道估计、导频相位/幅度修正、QPSK 文本恢复
- 与 OTFS / AFDM 的运行风格对齐：每帧都更新状态、BER、星座图，即使 CRC 未通过也尽量给出“软结果”
"""

from __future__ import annotations

import threading
import time
import zlib
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class OfdmHardwareTx:
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
        tx_text: str = "Hello OFDM Hardware Test!",
        mod_order: str = "QPSK",
        equalizer: str = "MMSE",
    ):
        self.carrier_freq = float(carrier_freq)
        self.sample_rate = float(samp_rate)
        self.samp_rate = self.sample_rate
        self.tx_gain = float(tx_gain)
        self.rx_gain = float(rx_gain)
        self.device_type = str(device_type)
        self.serial = serial
        self.tx_antenna = str(tx_antenna)
        self.rx_antenna = str(rx_antenna)

        # ===== OFDM 配置 =====
        self.fft_len = 64
        self.cp_len = 16
        self.n_data_symbols = 8
        self.mod_order = str(mod_order).upper()
        self.equalizer = str(equalizer).upper()
        if self.mod_order not in ("QPSK", "16QAM", "64QAM"):
            raise ValueError(f"Unsupported modulation: {self.mod_order}")
        if self.equalizer not in ("MMSE", "ZF"):
            raise ValueError(f"Unsupported equalizer: {self.equalizer}")
        self.bits_per_symbol = self._get_bits_per_symbol(self.mod_order)
        self.subcarrier_spacing = self.sample_rate / max(self.fft_len, 1)

        # 资源分配：沿用 802.11a 风格的 52 个激活子载波，其中 4 个导频
        self.active_bins = np.concatenate([np.arange(-26, 0), np.arange(1, 27)]).astype(np.int64)
        self.pilot_bins = np.array([-21, -7, 7, 21], dtype=np.int64)
        pilot_set = set(self.pilot_bins.tolist())
        self.data_bins = np.array([b for b in self.active_bins.tolist() if b not in pilot_set], dtype=np.int64)
        self.active_idx = self._bins_to_indices(self.active_bins)
        self.pilot_idx = self._bins_to_indices(self.pilot_bins)
        self.data_idx = self._bins_to_indices(self.data_bins)
        self._n_active = int(self.active_idx.size)
        self._n_pilot = int(self.pilot_idx.size)
        self._n_data = int(self.data_idx.size)

        # 帧结构：pre_guard | sync(repeated half) | train | data x K | post_guard
        self.pre_guard_len = 16
        self.sync_half_len = 64
        self.sync_len = 2 * self.sync_half_len
        self.block_len = self.fft_len + self.cp_len
        self.post_guard_len = self.cp_len + 16
        self.frame_len = (
            self.pre_guard_len
            + self.sync_len
            + (1 + self.n_data_symbols) * self.block_len
            + self.post_guard_len
        )
        self._off_sync = self.pre_guard_len
        self._off_train = self._off_sync + self.sync_len
        self._off_data = self._off_train + self.block_len
        self._off_end = self._off_data + self.n_data_symbols * self.block_len

        # 同步 / 状态
        self.sync_metric_threshold = 0.10
        self._noise_var_reg = 1e-2
        self.update_period = 0.08

        self.last_sync_index = 0
        self.last_payload_start = 0
        self.last_cfo_est_hz = 0.0
        self.last_sync_metric = 0.0
        self.last_frame_ok = False
        self.last_bad_reason = "init"
        self.last_train_nmse = 0.0
        self.last_pilot_nmse = 0.0
        self.last_alpha_abs = 0.0
        self.last_fft_window_offset = 0
        self.last_train_denoise_used = False
        self.last_train_obs_median = 0.0
        self.last_pre_eq_median = 0.0
        self.last_post_eq_median = 0.0
        self.train_window_search_radius = min(8, max(2, self.cp_len // 2))
        self.enable_channel_denoise = True

        # 同步序列：双半段重复 chirp
        self.sync_preamble = self._build_sync_preamble(self.sync_half_len)
        self._sync_energy = float(np.vdot(self.sync_preamble, self.sync_preamble).real) + 1e-12

        # 训练 OFDM 符号：所有 active 子载波均为已知 BPSK，导频位置固定为 +1
        self._train_freq_known = np.zeros(self.fft_len, dtype=np.complex128)
        active_train = np.ones(self._n_active, dtype=np.complex128)
        active_train[1::2] = -1.0 + 0.0j
        self._train_freq_known[self.active_idx] = active_train
        self._train_freq_known[self.pilot_idx] = 1.0 + 0.0j
        self._train_time = self._ofdm_symbol_to_time_cp(self._train_freq_known)

        # 文本 / 应用层
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
        self._ber_estimate = float("nan")

        # 运行态
        self._lock = threading.Lock()
        self._status = "idle"
        self._last_error = ""
        self._last_info = ""
        self._running = False
        self._tb = None
        self._monitor_thread = None
        self._monitor_stop = threading.Event()

        self._buffer_keep = max(131072, 12 * self.frame_len)
        self._tx_buffer = deque(maxlen=self._buffer_keep)
        self._rx_buffer = deque(maxlen=self._buffer_keep)
        self._latest_constellation = np.zeros(0, dtype=np.complex64)
        self._latest_constellation_pre_eq = np.zeros(0, dtype=np.complex64)
        self._last_good_constellation = np.zeros(0, dtype=np.complex64)
        self._latest_tx_samples = np.zeros(4096, dtype=np.complex64)
        self._latest_rx_samples = np.zeros(4096, dtype=np.complex64)
        self.constellation_display_mode = "dd_refined"

        self._ber_hist_t: List[float] = []
        self._ber_hist_v: List[float] = []
        self._t0 = time.time()
        self._rx_samples_seen = 0
        self._last_processed_abs_start = -10**18
        self._frame_soft_history: deque[np.ndarray] = deque(maxlen=12)
        self._combine_frames = 0
        self._phi_locked = 0.0

        # GNU Radio / UHD
        self._usrp_args = self._build_device_args()
        self._gr = None
        self._blocks = None
        self._uhd = None
        self._import_runtime()

        self._set_tx_text_internal(tx_text)
        self._build_top_block()

    # =========================================================
    # 应用层分帧
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
                f"OFDM 数据区容量不足：当前最大净载荷约 {max_payload_bytes} bytes，"
                f"当前文本 UTF-8 编码后为 {len(payload)} bytes。"
            )

        payload_repeat = max(1, min(4, max_bits // max(frame_bits.size, 1)))
        tx_bits = np.tile(frame_bits, payload_repeat).astype(np.int8)
        if tx_bits.size < max_bits:
            remain = max_bits - tx_bits.size
            reps_need = remain // frame_bits.size + 1
            pad = np.tile(frame_bits, reps_need)[:remain].astype(np.int8)
            tx_bits = np.concatenate([tx_bits, pad])
        elif tx_bits.size > max_bits:
            tx_bits = tx_bits[:max_bits]

        data_syms = self._qam_modulate(tx_bits, self.mod_order).reshape(self.n_data_symbols, self._n_data)
        tx_frame = self._build_tx_waveform(data_syms)

        rms = np.sqrt(np.mean(np.abs(tx_frame) ** 2) + 1e-12)
        tx_frame = 0.20 * tx_frame / rms

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
        self._ber_estimate = float("nan")
        self._latest_constellation = np.zeros(0, dtype=np.complex64)
        self._latest_constellation_pre_eq = np.zeros(0, dtype=np.complex64)
        self._last_good_constellation = np.zeros(0, dtype=np.complex64)
        self._frame_soft_history.clear()
        self._combine_frames = 0
        self._phi_locked = 0.0
        self.last_fft_window_offset = 0
        self.last_train_denoise_used = False
        self.last_train_obs_median = 0.0
        self.last_pre_eq_median = 0.0
        self.last_post_eq_median = 0.0

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
    # 日志
    # =========================================================
    def _debug(self, level: str, msg: str):
        if level in ("ERROR", "WARN"):
            self._last_error = str(msg)
        elif level == "INFO":
            self._last_info = str(msg)

    # =========================================================
    # 配置 / 生命周期
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
        tx_antenna: Optional[str] = None,
        rx_antenna: Optional[str] = None,
        tx_text: Optional[str] = None,
        mod_order: Optional[str] = None,
        equalizer: Optional[str] = None,
    ):
        if self._running:
            raise RuntimeError("运行中不能重新配置，请先 stop()")

        rebuild_top = False
        rebuild_waveform = False

        if carrier_freq is not None:
            self.carrier_freq = float(carrier_freq)
            rebuild_top = True
        if samp_rate is not None:
            self.sample_rate = float(samp_rate)
            self.samp_rate = self.sample_rate
            self.subcarrier_spacing = self.sample_rate / max(self.fft_len, 1)
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

        if mod_order is not None:
            mod_order = str(mod_order).upper()
            if mod_order not in ("QPSK", "16QAM", "64QAM"):
                raise ValueError(f"Unsupported modulation: {mod_order}")
            if mod_order != self.mod_order:
                self.mod_order = mod_order
                self.bits_per_symbol = self._get_bits_per_symbol(mod_order)
                rebuild_waveform = True

        if equalizer is not None:
            equalizer = str(equalizer).upper()
            if equalizer not in ("MMSE", "ZF"):
                raise ValueError(f"Unsupported equalizer: {equalizer}")
            self.equalizer = equalizer

        if tx_text is not None:
            self._set_tx_text_internal(str(tx_text))
            rebuild_top = True
            rebuild_waveform = False
        elif rebuild_waveform:
            self._set_tx_text_internal(self._tx_text)
            rebuild_top = True

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
    # GNU Radio / UHD：只做连续发送与采样
    # =========================================================
    def _build_top_block(self):
        gr = self._gr
        blocks = self._blocks
        uhd = self._uhd

        class _TopBlock(gr.top_block):
            pass

        tb = _TopBlock("OFDM Hardware Text Test", catch_exceptions=True)

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
            self._last_good_constellation = np.zeros(0, dtype=np.complex64)
            self._rx_text = ""
            self._decode_ok = False
            self._match_bytes = 0
            self._ber_estimate = float("nan")
            self._last_error = ""
            self._last_info = ""
            self._status = "configured"
            self.last_fft_window_offset = 0
            self.last_train_denoise_used = False
            self.last_train_obs_median = 0.0
            self.last_pre_eq_median = 0.0
            self.last_post_eq_median = 0.0

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
                        self._debug("WARN", f"process frame failed: {e}")

                time.sleep(max(self.update_period, 0.05))
            except Exception as e:
                self._debug("ERROR", f"monitor outer failure: {e}")
                time.sleep(0.2)

    # =========================================================
    # OFDM 发射 / 接收主处理
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

            train_time = frame[self._off_train:self._off_train + self.block_len]
            data_time = frame[self._off_data:self._off_end]
            if len(data_time) != self.n_data_symbols * self.block_len:
                continue
            data_blocks = data_time.reshape(self.n_data_symbols, self.block_len)

            train_fit = self._estimate_train_channel(train_time)
            h_train = train_fit["h_train"]
            train_nmse = float(train_fit["train_nmse"])
            fft_offset = int(train_fit["fft_offset"])
            train_obs_median = float(train_fit["train_obs_median"])
            denoise_used = bool(train_fit["denoise_used"])

            x_hat_units = []
            pre_eq_units = []
            alpha_hist = []
            pilot_err_hist = []
            for sym_idx, blk in enumerate(data_blocks):
                td = self._extract_fft_td(blk, fft_offset)
                y_f = self._time_to_freq(td)
                pilots = self._pilot_values(sym_idx)
                pilot_h = h_train[self.pilot_idx]
                pilot_pred = pilot_h * pilots
                pilot_obs = y_f[self.pilot_idx]
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

                resid_corr_all = self._estimate_residual_correction(
                    pilot_obs,
                    alpha * pilot_pred,
                    self.active_bins,
                )
                resid_map = np.ones(self.fft_len, dtype=np.complex128)
                resid_map[self.active_idx] = resid_corr_all

                y_corr = y_f.copy()
                good_corr = np.abs(resid_map) > 1e-6
                y_corr[good_corr] = y_corr[good_corr] / resid_map[good_corr]

                h_eff = h_train[self.data_idx]
                y_eff = y_corr[self.data_idx]
                pre_eq_units.append(y_f[self.data_idx].copy())
                x_hat = self._equalize_data(y_eff, h_eff)
                x_hat_units.append(x_hat)

            if not x_hat_units:
                continue

            rx_syms = np.concatenate(x_hat_units)
            pre_eq_syms = np.concatenate(pre_eq_units) if pre_eq_units else np.zeros(0, dtype=np.complex128)
            ber, raw_bytes, rx_payload, rx_text, match_bytes, decode_ok, rx_syms_best, phi_best = (
                self._recover_payload_from_symbols(rx_syms)
            )

            sync_here = float(metric[min(max(int(coarse_peak), 0), len(metric) - 1)])
            pilot_nmse = float(np.mean(pilot_err_hist)) if pilot_err_hist else 1.0
            alpha_abs = float(np.mean(np.abs(alpha_hist))) if alpha_hist else 0.0
            pre_eq_median = float(np.median(np.abs(pre_eq_syms))) if pre_eq_syms.size > 0 else 0.0
            post_eq_median = float(np.median(np.abs(rx_syms_best))) if rx_syms_best.size > 0 else 0.0
            quality_fail = bool(
                sync_here < self.sync_metric_threshold
                or not np.isfinite(train_nmse)
                or train_nmse > 3.0
                or not np.isfinite(pilot_nmse)
                or pilot_nmse > 2.5
                or not np.isfinite(alpha_abs)
                or alpha_abs < 1e-2
            )
            score = (
                1200.0 * float(decode_ok)
                + 140.0 * (1.0 - min(ber, 1.0))
                + 40.0 * max(sync_here, 0.0)
                + 18.0 * max(0.0, 2.0 - min(train_nmse, 4.0))
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
                "phi_best": float(phi_best),
                "fft_offset": int(fft_offset),
                "denoise_used": bool(denoise_used),
                "train_obs_median": float(train_obs_median),
                "pre_eq_median": float(pre_eq_median),
                "post_eq_median": float(post_eq_median),
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
        self.last_fft_window_offset = int(best["fft_offset"])
        self.last_train_denoise_used = bool(best["denoise_used"])
        self.last_train_obs_median = float(best["train_obs_median"])
        self.last_pre_eq_median = float(best["pre_eq_median"])
        self.last_post_eq_median = float(best["post_eq_median"])

        if best["quality_fail"] and not best["decode_ok"]:
            self.last_frame_ok = False
            self.last_bad_reason = "quality_gate"
        else:
            self.last_frame_ok = True
            self.last_bad_reason = "ok" if best["decode_ok"] else "soft_ok"
            if best["decode_ok"]:
                self._phi_locked = float(best["phi_best"])

        self._debug(
            "INFO",
            "Frame select: "
            f"sync={best['sync_metric']:.3f}, CFO={best['cfo_est_hz']:.1f} Hz, "
            f"trainNMSE={best['train_nmse']:.3f}, pilotNMSE={best['pilot_nmse']:.3f}, "
            f"alpha={best['alpha_abs']:.3f}, BER={best['ber']:.4e}, "
            f"fft_offset={best['fft_offset']:+d}, denoise={best['denoise_used']}, "
            f"|train|~{best['train_obs_median']:.3f}, |pre|~{best['pre_eq_median']:.3f}, |post|~{best['post_eq_median']:.3f}, "
            f"reason={self.last_bad_reason}, decode_ok={best['decode_ok']}, "
            f"match={best['match_bytes']}/{len(self._tx_payload)}"
        )

        const_points = self._prepare_constellation_points(best["rx_syms"], display_mode="raw")
        pre_eq_points = self._prepare_constellation_points(best.get("data_pre_eq_syms"), display_mode="raw")
        t_now = time.time() - self._t0
        with self._lock:
            self._latest_constellation = const_points.astype(np.complex64)
            self._latest_constellation_pre_eq = pre_eq_points.astype(np.complex64)
            self._ber_hist_t.append(t_now)
            self._ber_hist_v.append(max(float(best["ber"]), 1e-6))
            if len(self._ber_hist_t) > 200:
                self._ber_hist_t = self._ber_hist_t[-200:]
                self._ber_hist_v = self._ber_hist_v[-200:]
            self._last_good_constellation = self._latest_constellation.copy()
            self._last_raw_bytes = best["raw_bytes"]
            self._ber_estimate = float(best["ber"])
            if best["rx_payload"]:
                self._rx_text = best["rx_text"]
                self._last_good_rx_payload = best["rx_payload"]
                self._decode_ok = bool(best["decode_ok"])
                self._match_bytes = int(best["match_bytes"])
        self._status = "running"

    # =========================================================
    # OFDM 生成 / 同步 / 检测
    # =========================================================
    def _bins_to_indices(self, bins: np.ndarray) -> np.ndarray:
        bins = np.asarray(bins, dtype=np.int64)
        return (bins % self.fft_len).astype(np.int64)

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

    def _ofdm_symbol_to_time_cp(self, x_freq: np.ndarray) -> np.ndarray:
        x = np.asarray(x_freq, dtype=np.complex128).reshape(-1)
        td = np.fft.ifft(x) * np.sqrt(self.fft_len)
        cp = td[-self.cp_len:]
        return np.concatenate([cp, td]).astype(np.complex128)

    def _time_to_freq(self, td: np.ndarray) -> np.ndarray:
        td = np.asarray(td, dtype=np.complex128).reshape(-1)
        return (np.fft.fft(td) / np.sqrt(self.fft_len)).astype(np.complex128)

    def _build_tx_waveform(self, data_syms: np.ndarray) -> np.ndarray:
        parts = [
            np.zeros(self.pre_guard_len, dtype=np.complex128),
            self.sync_preamble.astype(np.complex128),
            self._train_time.astype(np.complex128),
        ]
        for sym_idx in range(self.n_data_symbols):
            x_f = np.zeros(self.fft_len, dtype=np.complex128)
            x_f[self.pilot_idx] = self._pilot_values(sym_idx)
            x_f[self.data_idx] = data_syms[sym_idx]
            parts.append(self._ofdm_symbol_to_time_cp(x_f))
        parts.append(np.zeros(self.post_guard_len, dtype=np.complex128))
        return np.concatenate(parts).astype(np.complex128)

    def _sync_metric(self, rx: np.ndarray) -> np.ndarray:
        rx = np.asarray(rx, dtype=np.complex128)
        Ls = int(self.sync_len)
        L = int(self.sync_half_len)
        if rx.size < Ls + 1:
            return np.zeros(1, dtype=np.float64)

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
            return [idx] if metric[idx] >= max(self.sync_metric_threshold, 0.12) else []
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
            seg = rx[s:s + Ls]
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
        a = rx[sync_start:sync_start + L]
        b = rx[sync_start + L:sync_start + 2 * L]
        # 若 rx[n] = s[n] * exp(j*2*pi*df*n/Fs)，则 conj(a) * b 的相位为 +2*pi*df*L/Fs。
        # 旧实现使用 a * conj(b)，导致 CFO 符号反了，补偿后残余频偏反而加重。
        P = np.sum(np.conj(a) * b)
        phase = float(np.angle(P))
        return float(phase * self.sample_rate / (2.0 * np.pi * max(L, 1)))

    def _extract_fft_td(self, block: np.ndarray, fft_offset: int = 0) -> np.ndarray:
        block = np.asarray(block, dtype=np.complex128).reshape(-1)
        if block.size < self.fft_len:
            raise ValueError("OFDM block too short for FFT extraction")
        base = int(self.cp_len) + int(fft_offset)
        start = min(max(base, 0), max(0, block.size - self.fft_len))
        return block[start:start + self.fft_len].astype(np.complex128, copy=False)

    def _compute_train_nmse(self, y_train: np.ndarray, h_train: np.ndarray) -> float:
        active = np.abs(self._train_freq_known) > 1e-8
        train_ref = self._train_freq_known[active]
        train_obs = np.asarray(y_train, dtype=np.complex128)[active]
        train_pred = np.asarray(h_train, dtype=np.complex128)[active] * train_ref
        return float(
            np.sqrt(
                np.mean(np.abs(train_obs - train_pred) ** 2)
                / (np.mean(np.abs(train_obs) ** 2) + 1e-12)
            )
        )

    def _estimate_train_channel(self, train_block: np.ndarray) -> Dict[str, Any]:
        train_block = np.asarray(train_block, dtype=np.complex128).reshape(-1)
        active = np.abs(self._train_freq_known) > 1e-8
        search_radius = int(max(0, min(self.cp_len, self.train_window_search_radius)))
        best: Optional[Dict[str, Any]] = None

        for fft_offset in range(-search_radius, search_radius + 1):
            train_td = self._extract_fft_td(train_block, fft_offset)
            y_train = self._time_to_freq(train_td)
            h_ls = np.ones(self.fft_len, dtype=np.complex128)
            h_ls[active] = y_train[active] / self._train_freq_known[active]

            channel_candidates = [(False, h_ls)]
            if self.enable_channel_denoise:
                channel_candidates.append((True, self._denoise_channel_est(h_ls)))

            for denoise_used, h_cand in channel_candidates:
                train_nmse = self._compute_train_nmse(y_train, h_cand)
                if not np.isfinite(train_nmse):
                    continue
                score = float(train_nmse + 0.015 * abs(fft_offset) + (0.01 if denoise_used else 0.0))
                cand = {
                    "score": score,
                    "fft_offset": int(fft_offset),
                    "denoise_used": bool(denoise_used),
                    "y_train": y_train,
                    "h_train": h_cand,
                    "train_nmse": float(train_nmse),
                    "train_obs_median": float(np.median(np.abs(y_train[self.active_idx]))),
                }
                if best is None or cand["score"] < best["score"]:
                    best = cand

        if best is None:
            train_td = self._extract_fft_td(train_block, 0)
            y_train = self._time_to_freq(train_td)
            h_ls = np.ones(self.fft_len, dtype=np.complex128)
            h_ls[active] = y_train[active] / self._train_freq_known[active]
            best = {
                "score": 0.0,
                "fft_offset": 0,
                "denoise_used": False,
                "y_train": y_train,
                "h_train": h_ls,
                "train_nmse": float(self._compute_train_nmse(y_train, h_ls)),
                "train_obs_median": float(np.median(np.abs(y_train[self.active_idx]))),
            }
        return best

    def _denoise_channel_est(self, H: np.ndarray) -> np.ndarray:
        h = np.fft.ifft(np.asarray(H, dtype=np.complex128)) * np.sqrt(self.fft_len)
        keep = min(self.cp_len, self.fft_len)
        if keep < self.fft_len:
            h[keep:] = 0.0
        return (np.fft.fft(h) / np.sqrt(self.fft_len)).astype(np.complex128)

    def _estimate_residual_correction(self, pilot_obs: np.ndarray, pilot_pred: np.ndarray, target_bins: np.ndarray) -> np.ndarray:
        """用导频估计一个全带宽复数标量校正，而不是仅靠 4 个导频拟合线性幅相斜率。

        真机里当前 OFDM 主要是整体残余相位 + 整体幅度缩放；4 个导频去拟合
        频域一次多项式，容易把噪声当成 slope，导致边缘子载波被过补偿，表现为
        星座虽有 4 团但 BER 维持在 1e-1 左右。这里改成与 OTFS/AFDM 一致的
        最小二乘复标量 alpha：pilot_obs ≈ alpha * pilot_pred。
        """
        pilot_obs = np.asarray(pilot_obs, dtype=np.complex128).reshape(-1)
        pilot_pred = np.asarray(pilot_pred, dtype=np.complex128).reshape(-1)
        target_bins = np.asarray(target_bins, dtype=np.float64).reshape(-1)
        denom = np.vdot(pilot_pred, pilot_pred)
        if abs(denom) > 1e-12:
            alpha = np.vdot(pilot_pred, pilot_obs) / denom
        else:
            alpha = 1.0 + 0.0j
        if not np.isfinite(alpha.real) or not np.isfinite(alpha.imag) or abs(alpha) < 1e-6:
            alpha = 1.0 + 0.0j
        return np.full(target_bins.shape, alpha, dtype=np.complex128)

    def _equalize_data(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.complex128)
        h = np.asarray(h, dtype=np.complex128)
        if self.equalizer == "ZF":
            out = np.zeros_like(y)
            mask = np.abs(h) > 1e-8
            out[mask] = y[mask] / h[mask]
            return out
        h_power_mean = float(np.mean(np.abs(h) ** 2)) + 1e-12
        noise_var = self._noise_var_reg * h_power_mean
        denom = np.abs(h) ** 2 + noise_var + 1e-12
        return y * np.conj(h) / denom

    # =========================================================
    # QPSK / 比特恢复
    # =========================================================
    def _max_data_bits_capacity(self) -> int:
        return int(self.n_data_symbols * self._n_data * self.bits_per_symbol)

    def _frame_to_bits(self, frame: bytes) -> np.ndarray:
        arr = np.frombuffer(frame, dtype=np.uint8)
        bits = ((arr[:, None] >> np.arange(8, dtype=np.uint8)) & 1).astype(np.int8)
        return bits.reshape(-1)

    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        bits = np.asarray(bits, dtype=np.int8).reshape(-1)
        if bits.size == 0:
            return b""
        usable = (bits.size // 8) * 8
        if usable <= 0:
            return b""
        bits = bits[:usable].astype(np.uint8)
        byte_mat = bits.reshape(-1, 8)
        vals = np.sum(byte_mat << np.arange(8, dtype=np.uint8), axis=1).astype(np.uint8)
        return vals.tobytes()

    def _qam_modulate_qpsk(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.int8).reshape(-1)
        if bits.size % 2 != 0:
            bits = np.concatenate([bits, np.zeros(1, dtype=np.int8)])
        b = bits.reshape(-1, 2)
        out = np.empty(b.shape[0], dtype=np.complex128)
        for i, (b0, b1) in enumerate(b):
            if b0 == 0 and b1 == 0:
                out[i] = 1.0 + 1.0j
            elif b0 == 0 and b1 == 1:
                out[i] = -1.0 + 1.0j
            elif b0 == 1 and b1 == 1:
                out[i] = -1.0 - 1.0j
            else:
                out[i] = 1.0 - 1.0j
        return out / np.sqrt(2.0)

    @staticmethod
    def _get_bits_per_symbol(mod_order: str) -> int:
        mod_order = str(mod_order).upper()
        if mod_order == "QPSK":
            return 2
        if mod_order == "16QAM":
            return 4
        if mod_order == "64QAM":
            return 6
        raise ValueError(f"Unsupported modulation: {mod_order}")

    def _qam_modulate(self, bits: np.ndarray, mod_order: str) -> np.ndarray:
        mod_order = str(mod_order).upper()
        bits = np.asarray(bits, dtype=np.int8).reshape(-1)
        bps = self._get_bits_per_symbol(mod_order)
        if bits.size % bps != 0:
            bits = np.concatenate([bits, np.zeros(bps - bits.size % bps, dtype=np.int8)])
        if mod_order == "QPSK":
            return self._qam_modulate_qpsk(bits)
        if mod_order == "16QAM":
            b = bits.reshape(-1, 4)
            lookup = np.array([3.0, 1.0, -3.0, -1.0], dtype=np.float64)
            i_idx = (b[:, 0].astype(np.int64) << 1) | b[:, 1].astype(np.int64)
            q_idx = (b[:, 2].astype(np.int64) << 1) | b[:, 3].astype(np.int64)
            return ((lookup[i_idx] + 1j * lookup[q_idx]) / np.sqrt(10.0)).astype(np.complex128)
        if mod_order == "64QAM":
            b = bits.reshape(-1, 6)
            table = {
                (0, 0, 0): 7, (0, 0, 1): 5, (0, 1, 1): 3, (0, 1, 0): 1,
                (1, 1, 0): -1, (1, 1, 1): -3, (1, 0, 1): -5, (1, 0, 0): -7,
            }
            i = np.array([table[tuple(row[:3].tolist())] for row in b], dtype=np.float64)
            q = np.array([table[tuple(row[3:].tolist())] for row in b], dtype=np.float64)
            return ((i + 1j * q) / np.sqrt(42.0)).astype(np.complex128)
        raise ValueError(f"Unsupported modulation: {mod_order}")

    def _qam_demodulate(self, syms: np.ndarray, mod_order: str) -> np.ndarray:
        mod_order = str(mod_order).upper()
        syms = np.asarray(syms, dtype=np.complex128).reshape(-1)
        if mod_order == "QPSK":
            bits = np.zeros((len(syms), 2), dtype=np.int8)
            bits[:, 0] = (np.imag(syms) < 0).astype(np.int8)
            bits[:, 1] = (np.real(syms) < 0).astype(np.int8)
            return bits.reshape(-1)
        if mod_order == "16QAM":
            x = np.real(syms) * np.sqrt(10.0)
            y = np.imag(syms) * np.sqrt(10.0)
            bits = np.zeros((len(syms), 4), dtype=np.int8)
            bits[:, 0] = (x < 0).astype(np.int8)
            bits[:, 1] = (np.abs(x) < 2).astype(np.int8)
            bits[:, 2] = (y < 0).astype(np.int8)
            bits[:, 3] = (np.abs(y) < 2).astype(np.int8)
            return bits.reshape(-1)
        if mod_order == "64QAM":
            x = np.real(syms) * np.sqrt(42.0)
            y = np.imag(syms) * np.sqrt(42.0)

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
                out[k, 0], out[k, 1], out[k, 2] = slicer(float(iv))
                out[k, 3], out[k, 4], out[k, 5] = slicer(float(qv))
            return out.reshape(-1)
        raise ValueError(f"Unsupported modulation: {mod_order}")

    def _soft_bits_from_qpsk(self, symbols: np.ndarray) -> np.ndarray:
        s = np.asarray(symbols, dtype=np.complex128).reshape(-1)
        soft = np.empty(2 * s.size, dtype=np.float64)
        # bit0: 上半平面->0, 下半平面->1
        # bit1: 右半平面->0, 左半平面->1
        soft[0::2] = np.imag(s)
        soft[1::2] = np.real(s)
        return soft

    def _prepare_constellation_points(self, arr: Optional[np.ndarray], display_mode: str = "raw") -> np.ndarray:
        arr = np.asarray(arr if arr is not None else np.zeros(0, dtype=np.complex64), dtype=np.complex64).reshape(-1)
        if arr.size == 0:
            return arr
        mag = np.abs(arr)
        med = float(np.median(mag))
        if med > 1e-6:
            keep = mag < 6.0 * med
            if np.any(keep):
                arr = arr[keep]
        return self._apply_display_mode(arr, display_mode)

    def _recover_payload_from_symbols(
        self,
        rx_syms: np.ndarray,
    ) -> Tuple[float, bytes, bytes, str, int, bool, np.ndarray, float]:
        rx_syms = np.asarray(rx_syms, dtype=np.complex128).reshape(-1)
        frame_bits_len = int(self._tx_frame_bits.size)
        total_bits_need = int(frame_bits_len * self._payload_repeat)
        if frame_bits_len <= 0 or total_bits_need <= 0:
            return 1.0, b"", b"", "", 0, False, rx_syms.astype(np.complex64), 0.0

        # 相位候选：优先尝试最近一次成功锁定的相位
        base = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
        base.sort(key=lambda p: abs(((p - self._phi_locked + np.pi) % (2 * np.pi)) - np.pi))
        phase_candidates = base

        best = {
            "score": -1e18,
            "ber": 1.0,
            "ber_instant": 1.0,
            "raw_bytes": b"",
            "rx_payload": b"",
            "rx_text": "",
            "match_bytes": 0,
            "decode_ok": False,
            "syms": rx_syms.astype(np.complex64),
            "bits_frame": None,
            "soft_for_hist": None,
            "phi": 0.0,
        }

        for phi in phase_candidates:
            syms_rot = rx_syms * np.exp(-1j * phi)

            if self.mod_order == "QPSK":
                soft_vals = self._soft_bits_from_qpsk(syms_rot)
                if soft_vals.size < total_bits_need:
                    continue

                soft_rep = soft_vals[:total_bits_need].reshape(self._payload_repeat, frame_bits_len)
                soft_sum = np.sum(soft_rep, axis=0)

                tail_start_bits = self._payload_repeat * frame_bits_len
                tail_soft = soft_vals[tail_start_bits:]
                tail_len = min(tail_soft.size, frame_bits_len)
                if tail_len > 0:
                    soft_sum[:tail_len] += tail_soft[:tail_len]

                frame_bits_single = (soft_sum < 0.0).astype(np.int8)
            else:
                rx_bits_all = self._qam_demodulate(syms_rot, self.mod_order)
                if rx_bits_all.size < total_bits_need:
                    continue
                rx_bits_rep = rx_bits_all[:total_bits_need].reshape(self._payload_repeat, frame_bits_len)
                votes = np.sum(rx_bits_rep, axis=0).astype(np.int32)
                n_votes = np.full(frame_bits_len, self._payload_repeat, dtype=np.int32)
                tail_start_bits = self._payload_repeat * frame_bits_len
                tail = rx_bits_all[tail_start_bits:]
                tail_len = min(tail.size, frame_bits_len)
                if tail_len > 0:
                    votes[:tail_len] += tail[:tail_len].astype(np.int32)
                    n_votes[:tail_len] += 1
                frame_bits_single = (votes * 2 >= n_votes).astype(np.int8)
                soft_sum = (1 - 2 * frame_bits_single).astype(np.float64)

            ber_single = float(np.mean(frame_bits_single != self._tx_frame_bits))

            soft_hist = list(self._frame_soft_history) + [soft_sum]
            soft_hist_sum = np.sum(np.stack(soft_hist, axis=0), axis=0)
            frame_bits_comb = (soft_hist_sum < 0.0).astype(np.int8)
            ber_comb = float(np.mean(frame_bits_comb != self._tx_frame_bits))

            # 选择最优解码路径时仍可使用跨帧软合并；
            # 但对外上报的 BER 固定为“当前帧原始软符号 -> 单帧判决”的瞬时 BER。
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
                        "ber": ber_single,
                        "ber_instant": ber_single,
                        "raw_bytes": frame_bytes,
                        "rx_payload": rx_payload,
                        "rx_text": rx_text,
                        "match_bytes": match_bytes,
                        "decode_ok": decode_ok,
                        "syms": syms_rot.astype(np.complex64),
                        "bits_frame": frame_bits_single.copy(),
                        "soft_for_hist": soft_sum.copy(),
                        "phi": float(phi),
                    }
                if decode_ok:
                    break
            if best["decode_ok"]:
                break

        if best["bits_frame"] is not None and best["ber"] < 0.45:
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
            float(best["phi"]),
        )

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

    def get_tx_spectrum_source(self, num_samples: int = 2048):
        return self.get_tx_samples(num_samples)

    def get_rx_spectrum_source(self, num_samples: int = 2048):
        return self.get_rx_samples(num_samples)

    def get_constellation(self):
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
            mode = str(display_mode).lower() if display_mode is not None else self.constellation_display_mode
            use_pre_eq = (mode == "pre_equalized") or (source == "pre_equalized")
            if use_pre_eq:
                raw = self._latest_constellation_pre_eq.copy()
            else:
                raw = self._latest_constellation.copy()
                if raw.size == 0 and self._last_good_constellation.size > 0:
                    raw = self._last_good_constellation.copy()
        if raw.size == 0:
            return raw
        pts = self._prepare_pre_eq_display(raw) if use_pre_eq else self._apply_display_mode(raw, mode)
        if pts.size <= max_points:
            return pts
        idx = np.linspace(0, pts.size - 1, max_points, dtype=np.int64)
        return pts[idx].copy()

    def get_estimated_ber(self):
        with self._lock:
            return (
                np.array(self._ber_hist_t, dtype=np.float64),
                np.array(self._ber_hist_v, dtype=np.float64),
            )

    def _prepare_pre_eq_display(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.complex64).reshape(-1)
        if arr.size == 0:
            return arr
        mag = np.abs(arr)
        median_mag = float(np.median(mag))
        if median_mag > 1e-6:
            keep = mag < 6.0 * median_mag
            if np.any(keep):
                arr = arr[keep]
        target_radius = 1.0 / np.sqrt(2.0)
        avg_mag = float(np.median(np.abs(arr)))
        if avg_mag > 1e-6:
            arr = arr * (target_radius / avg_mag)
        return arr

    def _apply_display_mode(self, arr: np.ndarray, mode: str) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.complex64).reshape(-1)
        if arr.size == 0:
            return arr
        mode = str(mode).lower()
        if mode == "raw" or self.mod_order != "QPSK":
            return arr
        if mode not in ("dd_refined", "hard_decision"):
            return arr

        target_radius = 1.0 / np.sqrt(2.0)
        avg_mag = float(np.median(np.abs(arr)))
        if avg_mag > 1e-6:
            arr = arr * (target_radius / avg_mag)

        qpsk_points = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex64) * target_radius
        dist = np.abs(arr[:, None] - qpsk_points[None, :])
        nearest = np.argmin(dist, axis=1)
        decisions = qpsk_points[nearest]
        if mode == "hard_decision":
            return decisions.copy()
        return (decisions + (arr - decisions) * 0.25).astype(np.complex64)

    def set_constellation_display_mode(self, mode: str):
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

    def get_debug_snapshot(self) -> Dict[str, Any]:
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
                "fft_window_offset": int(self.last_fft_window_offset),
                "train_denoise_used": bool(self.last_train_denoise_used),
                "train_obs_median": float(self.last_train_obs_median),
                "pre_eq_median": float(self.last_pre_eq_median),
                "post_eq_median": float(self.last_post_eq_median),
                "combine_frames": int(self._combine_frames),
                "payload_repeat": int(self._payload_repeat),
            }

    def get_status(self) -> Dict[str, Any]:
        snap = self.get_debug_snapshot()
        stats = self.get_decode_stats()
        with self._lock:
            ber_est = self._ber_estimate
        return {
            "status": self._status,
            "carrier_freq": self.carrier_freq,
            "sample_rate": self.sample_rate,
            "samp_rate": self.samp_rate,
            "subcarrier_spacing": self.subcarrier_spacing,
            "tx_gain": self.tx_gain,
            "rx_gain": self.rx_gain,
            "device_type": self.device_type,
            "device_args": self._usrp_args,
            "mod_order": self.mod_order,
            "equalizer": self.equalizer,
            "constellation_display_mode": self.constellation_display_mode,
            "frame_ok": snap["frame_ok"],
            "reason": snap["reason"],
            "sync_metric": snap["sync_metric"],
            "cfo_est_hz": snap["cfo_est_hz"],
            "train_nmse": snap["train_nmse"],
            "pilot_nmse": snap["pilot_nmse"],
            "alpha_abs": snap["alpha_abs"],
            "fft_window_offset": snap["fft_window_offset"],
            "train_denoise_used": snap["train_denoise_used"],
            "train_obs_median": snap["train_obs_median"],
            "pre_eq_median": snap["pre_eq_median"],
            "post_eq_median": snap["post_eq_median"],
            "combine_frames": snap["combine_frames"],
            "payload_repeat": snap["payload_repeat"],
            "last_error": self._last_error,
            "last_info": self._last_info,
            "decode_ok": stats["decode_ok"],
            "match_bytes": stats["match_bytes"],
            "expected_bytes": stats["expected_bytes"],
            "match_ratio": stats["match_ratio"],
            "ber": float(ber_est) if np.isfinite(ber_est) else float("nan"),
        }

    def get_last_error(self) -> str:
        return self._last_error

    # =========================================================
    # 动态 setter
    # =========================================================
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
        self.subcarrier_spacing = self.sample_rate / max(self.fft_len, 1)
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


if __name__ == "__main__":
    tb = OfdmHardwareTx(
        carrier_freq=2.4e9,
        samp_rate=960000.0,
        tx_gain=40.0,
        rx_gain=40.0,
        device_type="USRP B210",
        tx_text="Hello OFDM Hardware Test!",
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
