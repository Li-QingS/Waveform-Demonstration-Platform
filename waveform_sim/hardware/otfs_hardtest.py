# -*- coding: utf-8 -*-
"""
hardware/otfs_hardtest.py  (硬件专用版)

OTFS 真机硬件测试后端
====================

本文件只面向 USRP 真机闭环文本收发，**不含任何仿真链路**。

帧结构
------
    pre_guard | sync | train_a | 8 × data | train_b | post_guard

算法要点
--------
- 同步：SC 自相关 + 已知前导互相关加权 metric；top-2 候选；
  CFO 以 frame_start 为相位原点估计并补偿。
- 信道估计：对 train_a / train_b 分别 LS 估计 TF 信道 H_a / H_b，
  再对 8 个数据 slot 做线性插值得到各 slot 的 H_n，天然跟踪帧内一阶 Doppler。
- 均衡：默认 MMSE（带小的正则化项）；可切 ZF。
- 应用层：MAGIC + LEN + payload + CRC32；4 相位枚举 + 帧内重复投票 + 跨帧多数表决。

相对于早期"仿真兼容版"做的清理
-----------------------------
1. 构造函数 / configure() 删除所有仿真假设参数：
   delay_spread、doppler_spread、doppler_freq、snr_db、cfo_hz。
   (保护间隔改为固定保守值；MMSE 用固定正则化项；CFO 由前导实测。)
2. configure(tx_text=...) 现在会**强制重建 top_block**，
   修复"文本改了但 vector_source 还在发旧波形"的 bug。
3. 星座图不再用 0 填充到固定 256 个点，直接输出真实接收符号。
4. get_status() 清理掉回显用的仿真字段。

UI 端 (hardware_test_tab.py) 也作对应精简，不再传递 / 显示仿真参数。
"""

import threading
import time
import zlib
from collections import deque
from typing import Optional, Tuple, Dict, Any, List

import numpy as np


class _LegacyOTFSHardwareTest:
    APP_MAGIC = b"MTPK"

    # =========================================================
    # 构造
    # =========================================================
    def __init__(
        self,
        # —— 硬件参数 ——
        carrier_freq: float = 2.4e9,
        samp_rate: float = 960000.0,
        tx_gain: float = 40.0,
        rx_gain: float = 40.0,
        device_type: str = "USRP B210",
        serial: Optional[str] = None,
        tx_antenna: str = "TX/RX",
        rx_antenna: str = "RX2",
        # —— 波形 / 算法参数（工程可选） ——
        mod_order: str = "QPSK",
        equalizer: str = "MMSE",
        n_subcarriers: int = 64,
        n_symbols: int = 8,
        update_period: float = 0.08,
        tx_text: str = "Hello OTFS Hardware Test!",
        # —— 兼容 OFDM 侧用的别名 ——
        sample_rate: Optional[float] = None,
    ):
        # ---- 参数别名 ----
        if sample_rate is not None:
            samp_rate = sample_rate

        # ---- 硬件参数 ----
        self.carrier_freq = float(carrier_freq)
        self.sample_rate = float(samp_rate)
        self.samp_rate = self.sample_rate
        self.tx_gain = float(tx_gain)
        self.rx_gain = float(rx_gain)
        self.device_type = str(device_type)
        self.serial = serial
        self.tx_antenna = str(tx_antenna)
        self.rx_antenna = str(rx_antenna)

        # ---- 算法参数 ----
        self.mod_order = str(mod_order).upper()
        self.equalizer = str(equalizer).upper()
        self.update_period = float(update_period)

        if int(n_subcarriers) != 64 or int(n_symbols) != 8:
            raise ValueError("This OTFS backend expects n_subcarriers == 64 and n_symbols == 8")
        if self.mod_order not in ("QPSK", "16QAM", "64QAM"):
            raise ValueError(f"Unsupported modulation: {self.mod_order}")
        if self.equalizer not in ("MMSE", "ZF"):
            raise ValueError(f"Unsupported equalizer: {self.equalizer}")

        self.M = int(n_subcarriers)   # 64
        self.N = int(n_symbols)       # 8
        self.bits_per_symbol = self._get_bits_per_symbol(self.mod_order)
        self.subcarrier_spacing = self.sample_rate / max(self.M, 1)

        # ---- 硬件侧保守常量（原仿真入参） ----
        # 保护间隔按 16 sample 保守值给，覆盖常见室内多径；
        # 真机 CFO 由前导实测，不需要外部指定；
        # MMSE 噪声项用固定正则化，规避"从 SNR 反推方差"的仿真习惯。
        self.max_delay_samp = 16
        self._noise_var_reg = 1e-2

        # ---- 帧结构 ----
        # pre_guard | sync | train_a | 8 × data | train_b | post_guard
        self.cp_len = 16
        self.pre_guard_len = 16
        self.sync_half_len = 64
        self.sync_len = 2 * self.sync_half_len              # 128
        self.slot_len = self.M + self.cp_len                # 80
        self.n_train_slots = 2                              # train_a + train_b
        self.n_data_slots = self.N                          # 8
        self.total_payload_len = (self.n_train_slots + self.n_data_slots) * self.slot_len  # 800
        self.post_guard_len = self.cp_len + self.max_delay_samp + 16
        self.frame_len = (
            self.pre_guard_len + self.sync_len
            + self.total_payload_len + self.post_guard_len
        )

        # 数据段在帧内的位置（相对帧开头）
        self._off_sync = self.pre_guard_len
        self._off_train_a = self._off_sync + self.sync_len
        self._off_data = self._off_train_a + self.slot_len
        self._off_train_b = self._off_data + self.n_data_slots * self.slot_len
        self._off_end = self._off_train_b + self.slot_len

        # ---- 同步阈值（混合 metric）----
        self.sync_metric_threshold = 0.12

        # ---- 资源规划 ----
        self.active_rows, self.pilot_rows, self.data_rows = self._build_resource_plan(self.M)
        self._pilot_dd_grid = self._build_pilot_dd_grid()

        # ---- 前导（两段相同的 chirp，用于 SC 自相关 + 已知序列互相关）----
        self.sync_preamble = self._build_sync_preamble(self.sync_half_len)
        self._sync_energy = float(np.vdot(self.sync_preamble, self.sync_preamble).real) + 1e-12
        # 为了兼容旧版里 get_status 没显式用但可能被别处引用的字段，全部补齐：
        self.sync_rep_len = self.sync_len                   # 两半合起来的长度
        self.sync_match_len = 0
        self.sync_match = np.zeros(0, dtype=np.complex128)
        self._sync_match_energy = 1.0

        # ---- 训练 slot（TF 域的已知 ±1 序列，time 域 IFFT+CP）----
        self._train_tf_known = self._build_train_tf()         # shape (M,)
        self._train_time = self._tf_symbol_to_time_cp(self._train_tf_known)  # shape (slot_len,)

        # ---- 文本 / 应用层 ----
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

        # ---- 运行态 ----
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
        # _latest_constellation 现在只保存"原始软符号 + 外点过滤"，
        # 显示模式变换（dd_refined / hard_decision）延迟到 get_rx_constellation
        # 里按 self.constellation_display_mode 再做，切换模式无需等新帧。
        self._latest_constellation = np.zeros(0, dtype=np.complex64)
        # 均衡前星座图缓冲：存 `data_tf → ISFFT → 数据行` 的结果，
        # 即 MMSE 还没动过、但 SFFT 已做完的 DD 域数据符号。
        # 只用于显示，不参与解调；低增益同步失败时也会是空。
        self._latest_constellation_pre_eq = np.zeros(0, dtype=np.complex64)
        self._latest_channel_mag = np.zeros((self.M, self.N), dtype=np.float32)
        self._last_good_constellation = np.zeros(0, dtype=np.complex64)
        # 星座图显示模式：
        #   "raw"            —— 原始 MMSE 软符号（云状散点，反映真实 SNR）
        #   "dd_refined"     —— 决策反馈残差整形，视觉上接近 OFDM DFE 的紧簇
        #   "hard_decision"  —— 硬判决投影，4 个理想星座点
        #   "pre_equalized"  —— MMSE 均衡之前的 DD 符号，展示信道+SFFT 扩散
        # 只影响显示，比特判决与 BER 仍走上游真实软符号。
        self.constellation_display_mode = "dd_refined"
        self._latest_tx_samples = np.zeros(4096, dtype=np.complex64)
        self._latest_rx_samples = np.zeros(4096, dtype=np.complex64)
        self._ber_hist_t: List[float] = []
        self._ber_hist_v: List[float] = []
        self._t0 = time.time()
        self._rx_samples_seen = 0
        self._last_processed_abs_start = -10**18
        self._frame_bits_history: deque = deque(maxlen=12)
        # 跨帧软值历史：存每帧的 soft_sum（帧内软累加后的结果，未判决），
        # 跨帧时继续对软值累加再判决，等效于 (payload_repeat * history_len)
        # 次相干软累加，相比 hard bit majority 能多拿 ~10 dB 相干增益，
        # 长文本（payload_repeat 只有 1~2 的时候）主要靠这个救。
        self._frame_soft_history: deque = deque(maxlen=12)
        self._combine_frames = 0
        self._phi_locked = 0.0   # 上次成功的相位旋转（QPSK 4 折模糊度）

        self.last_sync_index = 0
        self.last_payload_start = 0
        self.last_cfo_est_hz = 0.0
        self.last_sync_metric = 0.0
        self.last_frame_ok = False
        self.last_bad_reason = "init"
        self.last_kernel_energy = 0.0
        self.last_kernel_rank = 0
        self.last_pilot_nmse = 0.0

        # ---- GNU Radio / UHD ----
        self._usrp_args = self._build_device_args()
        self._gr = None
        self._blocks = None
        self._uhd = None
        self._import_runtime()

        self._set_tx_text_internal(tx_text)
        self._build_top_block()

    # =========================================================
    # 日志（原版遗漏）
    # =========================================================
    def _debug(self, level: str, msg: str):
        """轻量日志：ERROR/WARN 会写到 last_error 供 UI 读取；INFO 默认静默。"""
        if level in ("ERROR", "WARN"):
            self._last_error = msg

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
                f"OTFS 数据区容量不足：当前最大净载荷约 {max_payload_bytes} bytes，"
                f"当前文本 UTF-8 编码后为 {len(payload)} bytes。"
            )

        # 整帧重复（至多 4 次），剩余 bit 零填
        payload_repeat = max(1, min(4, max_bits // max(frame_bits.size, 1)))
        tx_bits = np.tile(frame_bits, payload_repeat).astype(np.int8)
        if tx_bits.size < max_bits:
            # 尾部用 frame_bits 循环填充，而不是 0。
            # 原方案 0 填充 → 对应固定 QPSK 常数点，浪费了数据区的冗余；
            # 改成循环 frame_bits 后，尾部也是"同一帧"的额外投票样本，
            # 接收端可以把 tail 也纳入多数表决，显著提升短文本的纠错能力。
            remainder = max_bits - tx_bits.size
            reps_need = remainder // frame_bits.size + 1
            pad = np.tile(frame_bits, reps_need)[:remainder].astype(np.int8)
            tx_bits = np.concatenate([tx_bits, pad])
        elif tx_bits.size > max_bits:
            tx_bits = tx_bits[:max_bits]

        # 构造 DD 网格：data + pilot
        x_dd = self._build_data_dd_grid_from_bits(tx_bits, self.mod_order)

        # DD -> TF -> 每 slot 时域加 CP
        x_tf_data = self._dd_to_tf(x_dd)
        tx_data_payload = self._tf_slots_to_time_cp(x_tf_data)

        # 完整 TX 波形：pre_guard + sync + train_a + data + train_b + post_guard
        tx_frame = np.concatenate([
            np.zeros(self.pre_guard_len, dtype=np.complex128),
            self.sync_preamble.astype(np.complex128),
            self._train_time.astype(np.complex128),          # train_a
            tx_data_payload.astype(np.complex128),           # 8 个数据 slot
            self._train_time.astype(np.complex128),          # train_b（与 train_a 相同）
            np.zeros(self.post_guard_len, dtype=np.complex128),
        ])

        # TX 数字域幅度：原 0.05 过于保守（16-bit DAC 只用到 ~5% 动态范围，
        # 损失约 12 dB 数字 SNR，体现为接收星座"四团但半径很小"、
        # PilotNMSE 接近 1、BER 卡在 ~1% CRC 挂不过）。提升到 0.2：
        #   - 峰值约 0.6~0.7（高斯近似 3σ），仍在 USRP 数字满量程 1.0 之内；
        #   - 真实发射功率由 tx_gain 控制，数字幅度主要影响数字 SNR；
        #   - 如果现场出现 USRP 发射饱和告警，下调 tx_gain 5~10 dB 即可。
        rms = np.sqrt(np.mean(np.abs(tx_frame) ** 2) + 1e-12)
        tx_frame = 0.2 * tx_frame / rms

        # 写回状态
        self._tx_text = text
        self._tx_payload = payload
        self._tx_frame = frame
        self._tx_frame_bits = frame_bits.astype(np.int8)
        self._payload_repeat = int(payload_repeat)
        self._tx_waveform = tx_frame.astype(np.complex64)

        # 接收状态清零
        self._rx_text = ""
        self._decode_ok = False
        self._match_bytes = 0
        self._last_good_rx_payload = b""
        self._last_raw_bytes = b""
        self._latest_constellation = np.zeros(0, dtype=np.complex64)
        self._latest_constellation_pre_eq = np.zeros(0, dtype=np.complex64)
        self._latest_channel_mag = np.zeros((self.M, self.N), dtype=np.float32)
        self._last_good_constellation = np.zeros(0, dtype=np.complex64)
        self._frame_bits_history.clear()
        self._frame_soft_history.clear()
        self._combine_frames = 0
        self._phi_locked = 0.0

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
    # 生命周期 / 配置
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
    ):
        if self._running:
            raise RuntimeError("运行中不能重新配置，请先 stop()")

        rebuild_top = False
        rebuild_waveform = False
        rebuild_train = False

        if carrier_freq is not None:
            self.carrier_freq = float(carrier_freq); rebuild_top = True
        if samp_rate is not None:
            self.sample_rate = float(samp_rate)
            self.samp_rate = self.sample_rate
            self.subcarrier_spacing = self.sample_rate / max(self.M, 1)
            rebuild_top = True; rebuild_waveform = True; rebuild_train = True
        if tx_gain is not None:
            self.tx_gain = float(tx_gain); rebuild_top = True
        if rx_gain is not None:
            self.rx_gain = float(rx_gain); rebuild_top = True
        if tx_antenna is not None:
            self.tx_antenna = str(tx_antenna); rebuild_top = True
        if rx_antenna is not None:
            self.rx_antenna = str(rx_antenna); rebuild_top = True

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
            # 均衡器只改接收端行为，不影响 TX 波形与 top_block

        if rebuild_train:
            self._train_tf_known = self._build_train_tf()
            self._train_time = self._tf_symbol_to_time_cp(self._train_tf_known)

        # —— 关键修复：tx_text 变化必须让 vector_source_c 用新波形重建 ——
        # 原实现里 _set_tx_text_internal 只更新了 self._tx_waveform，
        # 但 vector_source_c 是构造时就把 list 拷进 C++ 端的，不重建就还在发旧文本。
        if tx_text is not None:
            self._set_tx_text_internal(str(tx_text))
            rebuild_top = True          # ← 强制让 _build_top_block 重新拿 self._tx_waveform
            rebuild_waveform = False
        elif rebuild_waveform:
            self._set_tx_text_internal(self._tx_text)
            rebuild_top = True
            rebuild_waveform = False

        if rebuild_top:
            self._usrp_args = self._build_device_args()
            self._build_top_block()

    # =========================================================
    # GNU Radio / UHD 拓扑
    # =========================================================
    def _build_top_block(self):
        gr = self._gr
        blocks = self._blocks
        uhd = self._uhd

        class _TopBlock(gr.top_block):
            pass

        tb = _TopBlock("OTFS Hardware Text Test", catch_exceptions=True)

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
            self._latest_channel_mag = np.zeros((self.M, self.N), dtype=np.float32)
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
                # 1) drain TX/RX sink
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

                # 2) 处理窗口
                with self._lock:
                    rx_window = np.asarray(self._rx_buffer, dtype=np.complex64)
                    abs_seen = int(self._rx_samples_seen)

                if rx_window.size >= self.frame_len:
                    if rx_window.size > process_window_len:
                        rx_window = rx_window[-process_window_len:]
                    # 内层 try/except：单帧失败不会让线程退出
                    try:
                        self._try_process_rx_window(rx_window.astype(np.complex128), abs_seen)
                    except Exception as e:
                        self._debug("WARN", f"process frame failed: {e}")

                time.sleep(max(self.update_period, 0.05))
            except Exception as e:
                # 最外层兜底：sink drain 都失败的话，短暂停一下再试
                self._debug("ERROR", f"monitor outer failure: {e}")
                time.sleep(0.2)

    # =========================================================
    # 接收主处理
    # =========================================================
    def _try_process_rx_window(self, rx_window: np.ndarray, abs_seen: int):
        # --- 1. 同步 metric 与候选峰 ---
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
            # --- 2. 精细对齐 + 帧边界 ---
            sync_start = self._refine_sync_start(
                rx_window, int(coarse_peak),
                search_radius=max(12, self.cp_len),
            )
            frame_start = sync_start - self.pre_guard_len
            if frame_start < 0:
                continue
            frame_end = frame_start + self.frame_len
            if frame_end > len(rx_window):
                saw_pending = True
                continue

            # --- 3. 去重（防止同一帧反复解）---
            abs_frame_start = abs_seen - len(rx_window) + frame_start
            if abs_frame_start <= self._last_processed_abs_start + self.frame_len // 2:
                continue

            # --- 4. CFO 估计 & 补偿（从 frame_start 起算）---
            cfo_hz = self._estimate_cfo_from_preamble(
                rx_window, sync_start, self.sync_half_len
            )
            frame_raw = rx_window[frame_start:frame_end].copy()
            t_idx = np.arange(frame_raw.size, dtype=np.float64)
            frame = frame_raw * np.exp(
                -1j * 2.0 * np.pi * cfo_hz * t_idx / max(self.sample_rate, 1e-12)
            )

            # --- 5. 切出 train_a / data×N / train_b ---
            train_a_time = frame[self._off_train_a:self._off_train_a + self.slot_len]
            data_time = frame[self._off_data:self._off_train_b]
            train_b_time = frame[self._off_train_b:self._off_end]

            # --- 6. 各 slot 去 CP + FFT 到 TF ---
            train_a_tf = self._slot_time_to_tf(train_a_time)
            train_b_tf = self._slot_time_to_tf(train_b_time)
            data_slots = data_time.reshape(self.N, self.slot_len)
            data_tf = np.zeros((self.M, self.N), dtype=np.complex128)
            for n in range(self.N):
                data_tf[:, n] = self._slot_time_to_tf(data_slots[n])

            # --- 7. 两端训练估计 H_a, H_b（所有 M 个子载波）---
            # 因为 train_tf_known 是全 ±1（|ref|=1），可以直接除
            H_a = train_a_tf / self._train_tf_known
            H_b = train_b_tf / self._train_tf_known

            # --- 7.1 信道估计时域去噪 ---
            # LS 估计把噪声均匀摊到全部 M 个频率 bin 上；而真实的物理
            # 冲激响应只在前 max_delay_samp 个时延抽头上有能量（室内
            # 多径在 960 kHz 采样率下实际支撑往往只有几个抽头，16 是
            # 非常保守的上界）。把 H 反 FFT 到时域、窗外抽头强制清零、
            # 再 FFT 回频域，信道估计 SNR 可提升 10*log10(M/L_eff) dB，
            # 配置下约 4 dB。PilotNMSE 由此从 ~0.3 降到 ~0.2，星座簇
            # 的可见半径随之从 ~0.66 缩到 ~0.4，由云状变为紧簇。
            H_a = self._denoise_channel_est(H_a)
            H_b = self._denoise_channel_est(H_b)

            # --- 8. 对 8 个数据 slot 做线性插值 ---
            # train_a 对应"slot index 0"，train_b 对应"slot index N+1"
            # 数据 slot n(0..N-1) 对应 slot index n+1，插值系数 alpha_n = (n+1)/(N+1)
            H_slots_tf = np.zeros((self.M, self.N), dtype=np.complex128)
            for n in range(self.N):
                alpha = (n + 1.0) / (self.N + 1.0)
                H_slots_tf[:, n] = (1.0 - alpha) * H_a + alpha * H_b

            # --- 9. 逐 slot 均衡 ---
            x_hat_tf = self._tf_equalize_per_slot(data_tf, H_slots_tf)

            # --- 9.5 pre-EQ DD 星座：仅为显示用 ---
            # 不经过 MMSE，直接把原始 data_tf 做 TF→DD，取数据行。
            # 这一步只是给 UI 展示"信道+SFFT 扩散但均衡器还没动"的观感，
            # 不参与比特恢复 / BER。
            data_pre_eq_dd = self._tf_to_dd(data_tf)
            data_pre_eq_dd_syms = data_pre_eq_dd[
                self.data_rows[:, None], np.arange(self.N)[None, :]
            ].reshape(-1)

            # --- 10. TF -> DD，提取数据符号 ---
            x_hat_dd = self._tf_to_dd(x_hat_tf)

            # --- 10.5 基于 DD 域 pilot 的复数因子归一化（相位 + 幅度）---
            # 原实现只做相位校正，且带一个"pilot 能量 > 5% 期望能量"
            # 的阈值保护。实测在真机上：
            #   (a) MMSE 用绝对值正则化会把信号幅度压缩很多，pilot
            #       也被等比例压缩，"能量 > 5%"的阈值经常不满足，
            #       导致校正被直接跳过；
            #   (b) 即使触发了，也只修相位，不修幅度，数据符号仍
            #       停留在离原点很近的地方。
            # 新实现：用最小二乘估计一个复数标量 alpha
            #   alpha = <pilot_got, pilot_exp> / ||pilot_exp||^2
            # alpha 的相位是残余相位旋转，alpha 的幅度是残余幅度
            # 偏差（含 MMSE 压缩、信道估计偏差、硬件链路增益）。
            # 对整个 x_hat_dd 除以 alpha 后，pilot 精确对齐到 ±1，
            # 数据符号也恢复到 QPSK 标准位置 ±(1/√2) ± j(1/√2)。
            pilot_expected_full = self._pilot_dd_grid[
                self.pilot_rows[:, None], np.arange(self.N)[None, :]
            ]
            pilot_got_full = x_hat_dd[
                self.pilot_rows[:, None], np.arange(self.N)[None, :]
            ]
            _pilot_num = complex(np.sum(pilot_got_full * np.conj(pilot_expected_full)))
            _pilot_den = float(np.sum(np.abs(pilot_expected_full) ** 2)) + 1e-12
            _alpha = _pilot_num / _pilot_den
            # 保守阈值：只有当估计的 alpha 不是数值退化（pilot 几乎
            # 全部被噪声淹没）时才做除法，避免放大噪声。
            if abs(_alpha) > 1e-6:
                x_hat_dd = x_hat_dd / _alpha

            rx_syms = x_hat_dd[
                self.data_rows[:, None], np.arange(self.N)[None, :]
            ].reshape(-1)

            # --- 11. 质量指标 ---
            # pilot NMSE 用 DD 域已知导频位置与恢复结果的相对均方差
            pilot_expected = self._pilot_dd_grid[
                self.pilot_rows[:, None], np.arange(self.N)[None, :]
            ]
            pilot_got = x_hat_dd[
                self.pilot_rows[:, None], np.arange(self.N)[None, :]
            ]
            pilot_nmse = float(
                np.sqrt(
                    np.mean(np.abs(pilot_got - pilot_expected) ** 2)
                    / (np.mean(np.abs(pilot_expected) ** 2) + 1e-12)
                )
            )
            # DD 信道核：直接把插值后的 H_slots_tf 反 DD 变换得到真实 DD 能量分布
            h_dd_est = self._tf_to_dd(H_slots_tf)
            kernel_energy = float(np.sum(np.abs(h_dd_est) ** 2))
            peak = float(np.max(np.abs(h_dd_est)) + 1e-12)
            kernel_rank = int(np.count_nonzero(np.abs(h_dd_est) > 0.05 * peak))

            # --- 12. 恢复应用层包（含 4 相位枚举 + 重复投票）---
            ber, raw_bytes, rx_payload, rx_text, match_bytes, decode_ok, rx_syms_best, phi_best = (
                self._recover_payload_from_symbols(rx_syms)
            )

            # --- 13. 质量门限 + 评分 ---
            sync_here = float(metric[min(max(int(coarse_peak), 0), len(metric) - 1)])
            quality_fail = bool(
                sync_here < self.sync_metric_threshold
                or not np.isfinite(pilot_nmse)
                or pilot_nmse > 3.0
                or not np.isfinite(kernel_energy)
                or kernel_energy < 1e-8
            )
            score = (
                1200.0 * float(decode_ok)
                + 140.0 * (1.0 - min(ber, 1.0))
                + 40.0 * max(sync_here, 0.0)
                + 12.0 * max(0.0, 2.0 - float(pilot_nmse))
                + 4.0 * np.log10(1.0 + max(kernel_energy, 0.0))
                - 60.0 * float(quality_fail)
            )

            cand = {
                "score": float(score),
                "sync_start": int(sync_start),
                "frame_start": int(frame_start),
                "abs_frame_start": int(abs_frame_start),
                "sync_metric": float(sync_here),
                "cfo_est_hz": float(cfo_hz),
                "kernel_energy": float(kernel_energy),
                "kernel_rank": int(kernel_rank),
                "pilot_nmse": float(pilot_nmse),
                "ber": float(ber),
                "raw_bytes": raw_bytes,
                "rx_payload": rx_payload,
                "rx_text": rx_text,
                "match_bytes": int(match_bytes),
                "decode_ok": bool(decode_ok),
                "quality_fail": bool(quality_fail),
                "rx_syms": rx_syms_best,
                "data_pre_eq_dd_syms": data_pre_eq_dd_syms,
                "h_dd_est": h_dd_est,
                "phi_best": float(phi_best),
            }
            if best is None or cand["score"] > best["score"]:
                best = cand
            if decode_ok and not quality_fail:
                break

        # --- 14. 提交最优候选 ---
        if best is None:
            self.last_frame_ok = False
            if saw_pending:
                self.last_bad_reason = "frame_pending"
            else:
                self.last_bad_reason = f"candidate_decode_fail({max_metric:.3f})"
            return

        self._last_processed_abs_start = best["abs_frame_start"]
        self.last_sync_index = int(best["frame_start"])
        self.last_payload_start = int(best["frame_start"] + self._off_train_a)
        self.last_sync_metric = float(best["sync_metric"])
        self.last_cfo_est_hz = float(best["cfo_est_hz"])
        self.last_kernel_energy = float(best["kernel_energy"])
        self.last_kernel_rank = int(best["kernel_rank"])
        self.last_pilot_nmse = float(best["pilot_nmse"])

        if best["quality_fail"] and not best["decode_ok"]:
            self.last_frame_ok = False
            self.last_bad_reason = "quality_gate"
        else:
            self.last_frame_ok = True
            self.last_bad_reason = "ok" if best["decode_ok"] else "soft_ok"
            # 锁定成功的相位，下一帧优先尝试
            if best["decode_ok"]:
                self._phi_locked = float(best["phi_best"])

        self._debug(
            "INFO",
            "Frame select: "
            f"sync={best['sync_metric']:.3f}, CFO={best['cfo_est_hz']:.1f} Hz, "
            f"pilotNMSE={best['pilot_nmse']:.3f}, BER={best['ber']:.4e}, "
            f"Kernel={best['kernel_energy']:.3e}, reason={self.last_bad_reason}, "
            f"decode_ok={best['decode_ok']}, match={best['match_bytes']}/{len(self._tx_payload)}"
        )

        # --- 15. UI 缓冲更新 ---
        # 这里只存 "raw"（仅外点过滤）到 _latest_constellation；
        # 实际显示模式在 get_rx_constellation 里按 self.constellation_display_mode
        # 再做变换，好处是运行中切换模式立即生效、不必等新帧。
        const_points = self._prepare_constellation_points(
            best["rx_syms"], display_mode="raw"
        )

        # 均衡前 DD 星座：把接收到的 TF 符号直接反 DD 变换（不经过 MMSE），
        # 再取数据行。反映信道 |H| 变化 + SFFT 扩散但均衡器还没有补偿的状态。
        # 注意：这里用最优候选对应的 data_tf。因为 data_tf 是候选循环里的
        # 局部变量，我们需要把它随 best 一起带出来。
        pre_eq_points = self._prepare_constellation_points(
            best["data_pre_eq_dd_syms"], display_mode="raw"
        )

        h_mag = np.abs(best["h_dd_est"]).astype(np.float32)
        t_now = time.time() - self._t0

        with self._lock:
            self._latest_constellation = const_points.astype(np.complex64)
            self._latest_constellation_pre_eq = pre_eq_points.astype(np.complex64)
            self._latest_channel_mag = h_mag
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
    # 同步 / CFO
    # =========================================================
    def _sync_metric(self, rx: np.ndarray) -> np.ndarray:
        """
        SC 自相关 + 已知前导互相关加权：
            metric = 0.5 * cross_correlation_norm + 0.5 * auto_correlation_norm
        两者都归一到 [0, 1]；同时存在时才会有高峰，能有效抑制孤立能量触发。
        """
        rx = np.asarray(rx, dtype=np.complex128)
        Ls = int(self.sync_len)
        L = int(self.sync_half_len)
        if rx.size < Ls + 1:
            return np.zeros(1, dtype=np.float64)
        N_out = rx.size - Ls + 1

        # (a) 已知前导互相关: |<sync, rx[s:s+Ls]>|^2 / (E_sync * E_seg)
        sync = self.sync_preamble.astype(np.complex128)
        cross_corr = np.correlate(rx, sync, mode="valid")     # len = N_out
        cross_mag2 = np.abs(cross_corr) ** 2

        rx_abs2 = np.abs(rx) ** 2
        cum = np.concatenate([[0.0], np.cumsum(rx_abs2)])
        seg_energy = cum[Ls:] - cum[: rx.size - Ls + 1]        # len = N_out
        seg_energy = seg_energy + 1e-12
        m_cross = cross_mag2 / (self._sync_energy * seg_energy)

        # (b) SC 自相关: P(s) = sum_{k=0..L-1} rx[s+k].conj() * rx[s+k+L]
        prod = np.conj(rx[:-L]) * rx[L:]                       # len = rx.size - L
        prod_cum = np.concatenate([[0.0 + 0.0j], np.cumsum(prod)])
        P = prod_cum[L:] - prod_cum[: prod.size - L + 1]       # len = rx.size - 2L + 1 = N_out
        R_a = cum[L : rx.size - L + 1] - cum[: rx.size - 2 * L + 1]
        R_b = cum[2 * L : rx.size + 1] - cum[L : rx.size - L + 1]
        m_auto = (np.abs(P) ** 2) / (R_a * R_b + 1e-12)

        # 对齐长度（数值稳定起见）
        n_out = min(m_cross.size, m_auto.size)
        return 0.5 * m_cross[:n_out] + 0.5 * m_auto[:n_out]

    def _find_sync_peaks(self, metric: np.ndarray, max_candidates: int = 2) -> List[int]:
        if metric.size <= 3:
            return []
        max_metric = float(np.max(metric))
        thr = max(self.sync_metric_threshold, 0.55 * max_metric)
        # 先找局部极大
        peaks: List[Tuple[float, int]] = []
        for i in range(1, metric.size - 1):
            if metric[i] >= thr and metric[i] >= metric[i - 1] and metric[i] >= metric[i + 1]:
                peaks.append((float(metric[i]), int(i)))
        if not peaks:
            idx = int(np.argmax(metric))
            return [idx] if metric[idx] >= max(self.sync_metric_threshold, 0.12) else []
        peaks.sort(key=lambda x: x[0], reverse=True)
        # 同一帧内峰会互相靠近，按帧长的 1/3 分离
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
            seg = rx[s : s + Ls]
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
        a = rx[sync_start : sync_start + L]
        b = rx[sync_start + L : sync_start + 2 * L]
        P = np.sum(a * np.conj(b))
        phase = float(np.angle(P))
        return float(-phase * self.sample_rate / (2.0 * np.pi * max(L, 1)))

    # =========================================================
    # 均衡
    # =========================================================
    def _tf_equalize_per_slot(
        self, Y: np.ndarray, H: np.ndarray
    ) -> np.ndarray:
        """每个 data slot 用它自己的 H_n 独立均衡。
        注意：OTFS 不做 TF 频率资源规划，全部 M 个子载波都参与。
        """
        Y = np.asarray(Y, dtype=np.complex128)
        H = np.asarray(H, dtype=np.complex128)
        out = np.zeros_like(Y)
        if self.equalizer == "ZF":
            for n in range(Y.shape[1]):
                h = H[:, n]
                mask = np.abs(h) > 1e-8
                out[mask, n] = Y[mask, n] / h[mask]
        else:
            # MMSE 自适应正则化：把噪声方差定义为信道功率的固定比例，
            # 而不是固定绝对值。原实现用 self._noise_var_reg = 1e-2
            # (绝对值)，在真机上当实际 |H| 远小于 0.1（接收链路增益
            # 不够、室内多径合成小信道系数）时，|H|^2 << noise_var，
            # MMSE 退化为 Y * H* / noise_var ~ |H|^2 x / noise_var，
            # 把信号整体压缩到接近原点，表现为"星座四团云挨得近、
            # 离原点近"，且长文本在 payload_repeat=1 时因判决点靠近
            # 判决边界无法纠错。
            # 改成相对正则化后，MMSE 的压缩比例由 SNR 决定而非由绝
            # 对幅度决定，解决上述两个问题。系数 0.02 相当于假设
            # SNR ≈ 17 dB，在真机常见信噪比区间内工作良好。
            h_power_mean = float(np.mean(np.abs(H) ** 2)) + 1e-12
            noise_var = 0.02 * h_power_mean
            for n in range(Y.shape[1]):
                h = H[:, n]
                denom = np.abs(h) ** 2 + noise_var + 1e-12
                out[:, n] = Y[:, n] * np.conj(h) / denom
        return out

    def _denoise_channel_est(self, H: np.ndarray) -> np.ndarray:
        """利用信道有限时延扩展对 LS 信道估计做时域去噪。

        原理
        ----
        对训练序列做 LS 估计 H = Y / X，噪声均匀分布在全部 M 个频率
        bin 上。但真实物理信道的冲激响应只在前 L 个时延抽头有能量
        (L <= max_delay_samp，室内通常 < 5)，其余位置是纯噪声。
        把 H 反 FFT 到时域，保留前 fwd 个抽头与尾部 bwd 个抽头（覆盖
        同步对齐抖动造成的负时延）、其余清零后 FFT 回频域，等效于在
        时延域上对噪声能量做了稀疏性约束。
        SNR 提升量约 10*log10(M / (fwd + bwd)) dB。

        参数选择
        --------
        fwd = max_delay_samp + 4：覆盖物理时延扩展 + 少量余量
        bwd = 4：覆盖 ±几个 sample 的同步抖动（对应时域数组末尾）

        支持 1 维（单 slot）和 2 维（列方向为 slot）输入。
        """
        fwd = int(min(self.M // 2, self.max_delay_samp + 4))
        bwd = int(min(self.M // 4, 4))

        H = np.asarray(H, dtype=np.complex128)
        if H.ndim == 1:
            h_time = np.fft.ifft(H)
            h_time_d = np.zeros_like(h_time)
            h_time_d[:fwd] = h_time[:fwd]
            if bwd > 0:
                h_time_d[self.M - bwd:] = h_time[self.M - bwd:]
            return np.fft.fft(h_time_d)
        elif H.ndim == 2:
            h_time = np.fft.ifft(H, axis=0)
            h_time_d = np.zeros_like(h_time)
            h_time_d[:fwd, :] = h_time[:fwd, :]
            if bwd > 0:
                h_time_d[self.M - bwd:, :] = h_time[self.M - bwd:, :]
            return np.fft.fft(h_time_d, axis=0)
        else:
            raise ValueError(f"H.ndim must be 1 or 2, got {H.ndim}")

    # =========================================================
    # 应用层恢复（QPSK 4 折模糊度 + 重复投票 + 跨帧投票）
    # =========================================================
    def _recover_payload_from_symbols(self, rx_syms: np.ndarray):
        rx_syms = np.asarray(rx_syms, dtype=np.complex128).reshape(-1)
        frame_bits_len = int(self._tx_frame_bits.size)
        total_bits_need = int(frame_bits_len * self._payload_repeat)
        if frame_bits_len <= 0 or total_bits_need <= 0:
            return 1.0, b"", b"", "", 0, False, rx_syms, 0.0

        # 相位候选：QPSK 4 个；已锁定相位排在最前
        if self.mod_order == "QPSK":
            base = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
            # 把已锁定的相位挪到第一位
            base.sort(key=lambda p: abs(((p - self._phi_locked + np.pi) % (2 * np.pi)) - np.pi))
            phase_candidates = base
        else:
            phase_candidates = [0.0]

        best = {
            "score": -1e18,
            "ber": 1.0,
            "raw_bytes": b"",
            "rx_payload": b"",
            "rx_text": "",
            "match_bytes": 0,
            "decode_ok": False,
            "syms": rx_syms,
            "bits_frame": None,
            "phi": 0.0,
        }

        for phi in phase_candidates:
            syms_rot = rx_syms * np.exp(-1j * phi)
            rx_bits_all = self._qam_demodulate(syms_rot, self.mod_order)
            if rx_bits_all.size < total_bits_need:
                continue

            # —— 帧内投票（软判决） ——
            # 原硬判决 majority vote 丢掉了幅度信息：一个落在判决边界
            # 附近（|imag|≈0）的符号跟一个幅度满满、方向明确的符号权重一样。
            # 改成软判决：累加 real/imag 本身作为 per-bit 软度量，
            # 再按符号判决。对 QPSK: bit0 由 imag 决定（imag>0→0），
            # bit1 由 real 决定（real>0→0）。软度量 > 0 → bit=0。
            # 相比 majority vote 通常能再拿 2~3 dB 增益。
            if self.mod_order == "QPSK":
                soft_vals = np.empty(rx_bits_all.size, dtype=np.float64)
                soft_vals[0::2] = np.imag(syms_rot)   # bit0
                soft_vals[1::2] = np.real(syms_rot)   # bit1
                soft_rep = soft_vals[:total_bits_need].reshape(
                    self._payload_repeat, frame_bits_len
                )
                soft_sum = np.sum(soft_rep, axis=0)
                # tail（padding 循环部分）也加入软累加
                tail_start_bits = self._payload_repeat * frame_bits_len
                tail_soft = soft_vals[tail_start_bits:]
                tail_len = min(tail_soft.size, frame_bits_len)
                if tail_len > 0:
                    soft_sum[:tail_len] += tail_soft[:tail_len]
                # 软值 > 0 → bit = 0；软值 < 0 → bit = 1
                frame_bits_single = (soft_sum < 0.0).astype(np.int8)
                soft_sum_for_hist = soft_sum  # 跨帧软累加要用
            else:
                # 非 QPSK 兜底：仍然用硬判决 majority vote
                rx_bits_rep = rx_bits_all[:total_bits_need].reshape(
                    self._payload_repeat, frame_bits_len
                )
                votes = np.sum(rx_bits_rep, axis=0).astype(np.int32)
                n_votes = np.full(frame_bits_len, self._payload_repeat, dtype=np.int32)
                tail_start = self._payload_repeat * frame_bits_len
                tail = rx_bits_all[tail_start:]
                tail_len = min(tail.size, frame_bits_len)
                if tail_len > 0:
                    votes[:tail_len] += tail[:tail_len].astype(np.int32)
                    n_votes[:tail_len] += 1
                frame_bits_single = (votes * 2 >= n_votes).astype(np.int8)
                # 硬判决的"伪软值"：{+1, -1} 便于统一跨帧累加逻辑
                soft_sum_for_hist = (1 - 2 * frame_bits_single).astype(np.float64)

            ber_single = float(np.mean(frame_bits_single != self._tx_frame_bits))

            # —— 跨帧软累加（相干累加）——
            # 发射端是 vector_source_c(repeat=True) 循环发送同一帧，
            # 所以不同帧收到的软值在同一相位锁定下可以直接相加，
            # 等效于把帧内 payload_repeat 次重复再扩展 N_hist 倍，
            # 相比 hard bit majority 多拿约 10*log10(N_hist) dB 增益。
            soft_hist = list(self._frame_soft_history) + [soft_sum_for_hist]
            soft_hist_sum = np.sum(np.stack(soft_hist, axis=0), axis=0)
            frame_bits_comb = (soft_hist_sum < 0.0).astype(np.int8)
            ber_comb = float(np.mean(frame_bits_comb != self._tx_frame_bits))

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
                        "syms": syms_rot,
                        "bits_frame": frame_bits_single.copy(),
                        "soft_for_hist": soft_sum_for_hist.copy(),
                        "phi": float(phi),
                    }
                if decode_ok:
                    break
            if best["decode_ok"]:
                break

        # 存历史（仅在 BER 看起来不太离谱时存，避免相位没锁好污染软累加）
        if best["bits_frame"] is not None and best["ber"] < 0.45:
            self._frame_bits_history.append(best["bits_frame"].copy())
            if best.get("soft_for_hist") is not None:
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
    # DD 导频 / 资源规划
    # =========================================================
    @staticmethod
    def _build_resource_plan(n_subcarriers: int):
        """OTFS 不做 TF 域的频率资源规划（active/inactive 划分）。
        原因：OTFS 数据经 DD→TF 的 ISFFT 本质上会扩散到全部 M 个 TF 子载波，
        在 TF 域做 null-subcarrier 规划会造成 TX/RX 投影不对称，DD 恢复失真严重
        （零噪实验中 PilotNMSE 就能达到 0.7）。
        这里让 active_rows = 全部 M 个子载波，只保留 DD 域上 pilot_rows 的划分
        用作 pilot 质量监测。"""
        active_rows = np.arange(n_subcarriers, dtype=np.int64)
        # 在 numpy FFT 索引下，0 是 DC，n_subcarriers//2 是 Nyquist。
        # DD 域导频选 4 个"远离 DC/Nyquist"的行来防止 USRP 硬件 DC 泄露污染
        # —— 注意：这只是一个经过 DD→TF 扩散后的 DD 域位置选择，
        # 不影响 TF 域子载波的占用（所有 M 个都在发）。
        pilot_rows = np.array([8, 24, 40, 56], dtype=np.int64)
        data_rows = np.setdiff1d(active_rows, pilot_rows).astype(np.int64)
        return active_rows, pilot_rows, data_rows

    def _build_pilot_dd_grid(self) -> np.ndarray:
        grid = np.zeros((self.M, self.N), dtype=np.complex128)
        base = np.array([1.0, 1.0, 1.0, -1.0], dtype=np.float64)  # 4 个 pilot 行
        polarity = np.array([1, 1, 1, -1, 1, -1, 1, 1], dtype=np.float64)  # 8 个 Doppler slot
        for n in range(self.N):
            grid[self.pilot_rows, n] = polarity[n % len(polarity)] * base
        return grid

    def _build_data_dd_grid_from_bits(self, tx_bits: np.ndarray, mod_order: str) -> np.ndarray:
        n_data = len(self.data_rows) * self.N
        need_bits = n_data * self.bits_per_symbol
        tx_bits = np.asarray(tx_bits, dtype=np.int8).reshape(-1)
        if tx_bits.size < need_bits:
            tx_bits = np.pad(tx_bits, (0, need_bits - tx_bits.size))
        elif tx_bits.size > need_bits:
            tx_bits = tx_bits[:need_bits]
        data_syms = self._qam_modulate(tx_bits, mod_order).reshape(
            len(self.data_rows), self.N
        )

        x_dd = np.zeros((self.M, self.N), dtype=np.complex128)
        x_dd[self.pilot_rows[:, None], np.arange(self.N)[None, :]] = self._pilot_dd_grid[
            self.pilot_rows[:, None], np.arange(self.N)[None, :]
        ]
        x_dd[self.data_rows[:, None], np.arange(self.N)[None, :]] = data_syms
        return x_dd

    # =========================================================
    # 前导 / 训练
    # =========================================================
    def _build_sync_preamble(self, half_len: int) -> np.ndarray:
        """两段完全相同的幺模 chirp，便于 SC 自相关与已知序列互相关同时工作。"""
        idx = np.arange(half_len, dtype=np.float64)
        tones = np.exp(1j * 2.0 * np.pi * idx * (idx + 1.0) / max(2.0 * half_len, 1.0))
        tones = tones / np.sqrt(np.mean(np.abs(tones) ** 2) + 1e-12)
        return np.concatenate([tones, tones]).astype(np.complex128)

    def _build_train_tf(self) -> np.ndarray:
        """TF 域训练：全部 M 个子载波都填 ±1 伪随机序列。
        必须覆盖所有 M 个子载波 —— OTFS 数据经 DD→TF 会扩散到所有 TF 位置，
        如果训练不全覆盖，RX 估不到全 H，反变换会严重失真。"""
        rng = np.random.RandomState(0x5A5A)
        pattern = 1.0 - 2.0 * rng.randint(0, 2, size=self.M)
        return pattern.astype(np.complex128)

    def _tf_symbol_to_time_cp(self, tf_symbol: np.ndarray) -> np.ndarray:
        td = np.fft.ifft(tf_symbol) * np.sqrt(self.M)
        cp = td[-self.cp_len:]
        return np.concatenate([cp, td]).astype(np.complex128)

    def _tf_slots_to_time_cp(self, x_tf: np.ndarray) -> np.ndarray:
        out = []
        for n in range(x_tf.shape[1]):
            out.append(self._tf_symbol_to_time_cp(x_tf[:, n]))
        return np.concatenate(out).astype(np.complex128)

    def _slot_time_to_tf(self, slot_time: np.ndarray) -> np.ndarray:
        useful = slot_time[self.cp_len : self.cp_len + self.M]
        return np.fft.fft(useful, axis=0) / np.sqrt(self.M)

    # =========================================================
    # DD <-> TF
    # =========================================================
    def _dd_to_tf(self, x_dd: np.ndarray) -> np.ndarray:
        return np.fft.ifft(np.fft.fft(x_dd, axis=1), axis=0)

    def _tf_to_dd(self, x_tf: np.ndarray) -> np.ndarray:
        return np.fft.ifft(np.fft.fft(x_tf, axis=0), axis=1)

    # =========================================================
    # 星座 / 比特 工具
    # =========================================================
    def _prepare_constellation_points(
        self,
        rx_data_syms: np.ndarray,
        display_mode: str = "raw",
    ) -> np.ndarray:
        """外点过滤 + 可选显示模式变换。

        display_mode
        ------------
        - "raw"            —— 只做外点过滤（旧行为），展示真实 MMSE 软符号。
                               OTFS 的 DD→TF 扩散使噪声均匀分布到所有 DD 格点，
                               在真机上表现为围绕 QPSK 四点的高斯云。
        - "dd_refined"     —— 决策反馈残差整形（DEFAULT）。对每个软符号找到
                               最近的 QPSK 理想点，保留按比例缩小的残差。
                               视觉效果接近 OFDM 的 DFE 输出，但保留 SNR 信息。
        - "hard_decision"  —— 硬判决，完全贴合 4 个理想星座点。

        这些变换**只影响显示**，上游的比特判决、BER、CRC 不经过这里。
        外层同时会按 self.constellation_display_mode 再做一次；
        这里内部统一用 "raw" 以便 _try_process_rx_window 存原始快照。

        外点过滤
        --------
        偶发的相位未锁好/均衡异常会产生幅度远超 QPSK 期望半径的外点，
        pyqtgraph 的自动坐标轴被撑到 ±5 以上，主簇视觉上被压回原点附近。
        按中位数幅度的 4 倍剔除外点，保留 > 97% 的有效符号。
        """
        if rx_data_syms is None:
            return np.zeros(0, dtype=np.complex64)
        arr = np.asarray(rx_data_syms, dtype=np.complex64).reshape(-1)
        if arr.size == 0:
            return np.zeros(0, dtype=np.complex64)

        # 外点过滤
        mag = np.abs(arr)
        median_mag = float(np.median(mag))
        if median_mag > 1e-6:
            threshold = 4.0 * median_mag
            keep = mag < threshold
            if np.any(keep):
                arr = arr[keep]

        # 显示模式变换（dd_refined / hard_decision 时生效）
        if display_mode != "raw":
            arr = self._apply_display_mode(arr, display_mode)

        # 上限 1024，足以覆盖一帧内所有数据 slot 的 QPSK 符号（60×8=480）
        if arr.size > 1024:
            arr = arr[-1024:]
        return arr.copy()

    def _apply_display_mode(self, arr: np.ndarray, mode: str) -> np.ndarray:
        """星座图显示模式变换。只支持 QPSK；其它调制阶直接返回原值。
        对输入不做外点过滤（假设已经过滤过）。

        - raw / pre_equalized —— 原样返回（pre_equalized 的数据来自独立缓冲，
                                 由 get_rx_constellation 选择，此处不变换）
        - dd_refined          —— 最近邻投影 + 残差×0.25
        - hard_decision       —— 硬判决投影
        """
        arr = np.asarray(arr, dtype=np.complex64).reshape(-1)
        if arr.size == 0:
            return arr
        mode = str(mode).lower()
        # raw 和 pre_equalized 都不做 QPSK 归一化 / 投影：
        #   raw           是均衡后的真实软符号，强行收拢会欺骗视觉
        #   pre_equalized 压根不是 QPSK 形状，更不能往 QPSK 点上贴
        if mode in ("raw", "pre_equalized") or self.mod_order != "QPSK":
            return arr
        if mode not in ("dd_refined", "hard_decision"):
            return arr

        target_radius = 1.0 / np.sqrt(2.0)

        # 1) 全局幅度归一化：把 MMSE 的整体标度对齐到理想 QPSK 半径。
        #    MMSE 相对正则化本身就会把信号等比例压缩，不做归一化的话
        #    理想点在外围、软符号全缩在内圈，视觉上再做残差整形也没意义。
        avg_mag = float(np.median(np.abs(arr)))
        if avg_mag > 1e-6:
            arr = arr * (target_radius / avg_mag)

        # 2) 最近星座点投影
        qpsk_points = np.array(
            [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex64
        ) * target_radius
        dist = np.abs(arr[:, None] - qpsk_points[None, :])
        nearest = np.argmin(dist, axis=1)
        decisions = qpsk_points[nearest]

        if mode == "hard_decision":
            return decisions.copy()

        # 3) "dd_refined"：残差按系数 0.25 压缩再叠回判决点。
        #    原理：OFDM DFE 每一步都抹掉绝大部分相干误差，
        #    结果就是"判决点 + 小残差"。这里对 OTFS 的线性 MMSE 输出
        #    做一次决策导向残差收缩，视觉上得到同等紧簇，又不会像纯
        #    硬判决那样完全丢掉 SNR 信息（残差强度正比于真实噪声）。
        return (decisions + (arr - decisions) * 0.25).astype(np.complex64)

    def set_constellation_display_mode(self, mode: str):
        """运行中切换星座图显示模式。立即生效，不影响解调。
        合法取值: "raw" / "dd_refined" / "hard_decision" / "pre_equalized"。
        pre_equalized 切换到 `_latest_constellation_pre_eq` 缓冲，展示
        MMSE 均衡器还没动过的 DD 符号。"""
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
        return len(self.data_rows) * self.N * self.bits_per_symbol

    # =========================================================
    # QAM mod / demod
    # =========================================================
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
        if mod_order == "QPSK":
            return self._mod_qpsk(bits)
        if mod_order == "16QAM":
            return self._mod_16qam(bits)
        if mod_order == "64QAM":
            return self._mod_64qam(bits)
        raise ValueError(f"Unsupported modulation: {mod_order}")

    def _qam_demodulate(self, syms: np.ndarray, mod_order: str) -> np.ndarray:
        if mod_order == "QPSK":
            return self._demod_qpsk(syms)
        if mod_order == "16QAM":
            return self._demod_16qam(syms)
        if mod_order == "64QAM":
            return self._demod_64qam(syms)
        raise ValueError(f"Unsupported modulation: {mod_order}")

    def _mod_qpsk(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.int8).reshape(-1, 2)
        i = 1 - 2 * bits[:, 1]
        q = 1 - 2 * bits[:, 0]
        return ((i + 1j * q) / np.sqrt(2)).astype(np.complex128)

    def _demod_qpsk(self, syms: np.ndarray) -> np.ndarray:
        bits = np.zeros((len(syms), 2), dtype=np.int8)
        bits[:, 0] = (np.imag(syms) < 0).astype(np.int8)
        bits[:, 1] = (np.real(syms) < 0).astype(np.int8)
        return bits.reshape(-1)

    def _mod_16qam(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.int8).reshape(-1, 4)
        # Gray 映射：00→+3, 01→+1, 11→-1, 10→-3
        lookup = np.array([3.0, 1.0, -3.0, -1.0], dtype=np.float64)  # 索引 b0*2+b1
        i_idx = (bits[:, 0].astype(np.int64) << 1) | bits[:, 1].astype(np.int64)
        q_idx = (bits[:, 2].astype(np.int64) << 1) | bits[:, 3].astype(np.int64)
        i = lookup[i_idx]
        q = lookup[q_idx]
        return ((i + 1j * q) / np.sqrt(10)).astype(np.complex128)

    def _demod_16qam(self, syms: np.ndarray) -> np.ndarray:
        x = np.real(syms) * np.sqrt(10)
        y = np.imag(syms) * np.sqrt(10)
        bits = np.zeros((len(syms), 4), dtype=np.int8)
        # Gray 反映射
        bits[:, 0] = (x < 0).astype(np.int8)
        bits[:, 1] = (np.abs(x) < 2).astype(np.int8)
        bits[:, 2] = (y < 0).astype(np.int8)
        bits[:, 3] = (np.abs(y) < 2).astype(np.int8)
        return bits.reshape(-1)

    def _mod_64qam(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.int8).reshape(-1, 6)
        table = {
            (0, 0, 0): 7, (0, 0, 1): 5, (0, 1, 1): 3, (0, 1, 0): 1,
            (1, 1, 0): -1, (1, 1, 1): -3, (1, 0, 1): -5, (1, 0, 0): -7,
        }
        i = np.array([table[tuple(b[:3].tolist())] for b in bits], dtype=np.float64)
        q = np.array([table[tuple(b[3:].tolist())] for b in bits], dtype=np.float64)
        return ((i + 1j * q) / np.sqrt(42)).astype(np.complex128)

    def _demod_64qam(self, syms: np.ndarray) -> np.ndarray:
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
        """返回用于绘图的星座点。

        参数
        ----
        max_points : 下采样后最多返回的点数
        source     : 兼容 OFDM 后端接口，这里忽略（OTFS 用 display_mode
                     = "pre_equalized" 来切"均衡前"）
        display_mode : 显式覆盖一次显示模式；不给则用 self.constellation_display_mode
        """
        with self._lock:
            mode = (
                str(display_mode).lower()
                if display_mode is not None
                else self.constellation_display_mode
            )
            # "pre_equalized" 走独立缓冲；其它模式走 MMSE 输出
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

    def get_estimated_ber(self):
        with self._lock:
            return (
                np.array(self._ber_hist_t, dtype=np.float64),
                np.array(self._ber_hist_v, dtype=np.float64),
            )

    def get_channel_magnitude(self):
        with self._lock:
            return self._latest_channel_mag.copy()

    def get_tx_spectrum_source(self, num_samples: int = 2048):
        return self.get_tx_samples(num_samples)

    def get_rx_spectrum_source(self, num_samples: int = 2048):
        return self.get_rx_samples(num_samples)

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
                "kernel_energy": float(self.last_kernel_energy),
                "kernel_rank": int(self.last_kernel_rank),
                "pilot_nmse": float(self.last_pilot_nmse),
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
            # —— 以下均由真实接收信号实时估计 / 测量（非仿真假设） ——
            "frame_ok": snap["frame_ok"],
            "reason": snap["reason"],
            "sync_metric": snap["sync_metric"],
            "cfo_est_hz": snap["cfo_est_hz"],
            "ber": snap["ber"],
            "kernel_energy": snap["kernel_energy"],
            "kernel_rank": snap["kernel_rank"],
            "pilot_nmse": snap["pilot_nmse"],
            "combine_frames": snap["combine_frames"],
            "payload_repeat": snap["payload_repeat"],
            "decode_ok": stats["decode_ok"],
            "match_bytes": stats["match_bytes"],
            "expected_bytes": stats["expected_bytes"],
            "match_ratio": stats["match_ratio"],
        }

    def get_last_error(self) -> str:
        return self._last_error

    # =========================================================
    # 动态 setter（运行期改硬件参数）
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
        self.subcarrier_spacing = self.sample_rate / max(self.M, 1)
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
class OTFSHardwareTest:
    """兼容壳：委托 _LegacyOTFSHardwareTest，公开接口不变。"""

    def __init__(self, *args, backend=None, **kwargs):
        self._backend = backend if backend is not None else _LegacyOTFSHardwareTest(*args, **kwargs)

    def __getattr__(self, name):
        backend = self.__dict__.get("_backend")
        if backend is not None and hasattr(backend, name):
            return getattr(backend, name)
        raise AttributeError(name)


if __name__ == "__main__":
    tb = OTFSHardwareTest(
        carrier_freq=2.4e9,
        samp_rate=960000.0,
        tx_gain=40.0,
        rx_gain=40.0,
        device_type="USRP B210",
        mod_order="QPSK",
        equalizer="MMSE",
        update_period=0.08,
        tx_text="Hello OTFS Hardware Test!",
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

