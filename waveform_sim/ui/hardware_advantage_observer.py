# -*- coding: utf-8 -*-
"""FDIDM 硬件优势观测引擎：窗口汇总、错误计数分级、异常剔除、前后配对。

纯 Python + numpy，不依赖 PyQt（spec N4）。数据来源为硬件后端
``get_status()`` 的逐次快照，本模块只做测量方法学，不做任何呈现。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# 分级阈值（MATLAB ≥100 错误建议，与 UPGRADE_PLAN 问题 2 同源）
GRADE_TRUSTED_K = 100
GRADE_REFERENCE_K = 10
# 段首瞬态秒数：切换后前 TRANSIENT_SEC 的样本不计入汇总
TRANSIENT_SEC = 1.0
# 窗口内异常采样占比超过该值时整窗剔除，不参与配对
MAX_ANOMALY_SAMPLE_RATIO = 0.2

TRUSTED = "trusted"
REFERENCE = "reference"
INSUFFICIENT = "insufficient"

MODE_ON = "adaptive_on"
MODE_OFF = "adaptive_off"

# 配对可比性检查的上下文字段（TDL 注入参数 + 链路模式）
CONTEXT_KEYS = (
    "channel_mode",
    "tdl_model",
    "tdl_doppler_hz",
    "tdl_rms_delay_spread_ns",
    "tdl_snr_db",
    "tdl_seed",
)

SCHEMA_VERSION = 1


def grade_from_k(k: float) -> str:
    """按窗口内错误计数给出可信度分级。"""
    k = float(k)
    if not np.isfinite(k):
        return INSUFFICIENT
    if k >= GRADE_TRUSTED_K:
        return TRUSTED
    if k >= GRADE_REFERENCE_K:
        return REFERENCE
    return INSUFFICIENT


def ser_improvement_db(before: float, after: float) -> float:
    """SER 改善量（dB）：正值表示开启后 SER 更低。不可计算时返回 NaN。"""
    b, a = float(before), float(after)
    if not (np.isfinite(b) and np.isfinite(a)) or b <= 0.0 or a <= 0.0:
        return float("nan")
    return float(10.0 * math.log10(b / a))


def _f(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class WindowSample:
    """一次投喂产生的采样点。"""

    t: float
    frames_delta: int
    ok_delta: int
    ser: float
    fec_ber: float
    evm_percent: float
    alpha: float
    beta: float
    transient: bool = False


@dataclass
class WindowAggregate:
    """一个观测段的汇总（F3 的载体）。"""

    frames: int
    crc_ok_ratio: float
    ser: float
    ser_k: float
    ser_n: int
    fec_ber: float
    fec_k: float
    fec_n: int
    evm_mean: float
    alpha_mean: float
    beta_mean: float
    grade: str = INSUFFICIENT
    estimated_k: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ObservationWindow:
    """一个连续段（段内自适应开关状态不变）。"""

    mode: str
    start_t: float
    end_t: float = float("nan")
    context: Dict[str, Any] = field(default_factory=dict)
    anomalies: List[str] = field(default_factory=list)
    samples: List[WindowSample] = field(default_factory=list)
    valid_samples: int = 0
    dropped_samples: int = 0
    aggregate: Optional[WindowAggregate] = None
    excluded: bool = False
    exclusion_reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "mode": self.mode,
            "start_t": self.start_t,
            "end_t": self.end_t,
            "context": dict(self.context),
            "anomalies": list(self.anomalies),
            "valid_samples": self.valid_samples,
            "dropped_samples": self.dropped_samples,
            "excluded": self.excluded,
            "exclusion_reason": self.exclusion_reason,
            "aggregate": self.aggregate.as_dict() if self.aggregate is not None else None,
        }
        return out


@dataclass
class ToggleEvent:
    """评审手动开关自适应的事件（F1）。"""

    t: float
    enabled: bool
    alpha: float
    beta: float

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class PairComparison:
    """一组开启前/开启后的配对结论（F2+F5 的载体）。"""

    before: WindowAggregate
    after: WindowAggregate
    ser_improvement_db: float
    evm_delta_pp: float
    crc_delta: float
    comparable: bool = True
    comparability_note: str = ""


def _sanitize(value: Any) -> Any:
    """NaN/inf 转 None，便于 JSON 导出。"""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class AdvantageObservationSession:
    """观测会话：投喂 status 快照，产出窗口汇总与前后配对结论。

    生命周期：``start`` → 若干 ``on_sample`` / ``on_toggle`` → ``stop``。
    结果在 ``stop`` 后保留可查看；再次 ``start`` 清空全部旧结果（F1）。
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self.active = False
        self._started_wall = ""
        self._windows: List[ObservationWindow] = []
        self._events: List[ToggleEvent] = []
        self._current: Optional[ObservationWindow] = None
        self._current_enabled = False
        self._last_status_frames = 0
        self._last_status_ok = 0
        self._last_status_overflow = 0
        self._context: Dict[str, Any] = {}
        self._anomaly_drops: List[Any] = []
        self._symbols_per_frame = float("nan")
        self._coded_bits_per_frame = float("nan")
        self._last_config = (float("nan"), float("nan"))

    # ---------------- 生命周期 ----------------
    def start(self, enabled: bool, context: Optional[Dict[str, Any]] = None) -> None:
        self.active = True
        self._started_wall = time.strftime("%Y-%m-%d %H:%M:%S")
        self._started_clock = float(self._clock())
        self._windows = []
        self._events = []
        self._current = None
        self._context = dict(context or {})
        self._anomaly_drops = []
        self._last_status_frames = 0
        self._last_status_ok = 0
        self._last_status_overflow = 0
        self._open_window(bool(enabled))

    def now(self) -> float:
        """当前引擎时钟值；UI 时间轴与会话共用此时基。"""
        return float(self._clock())

    def started_at(self) -> float:
        return getattr(self, "_started_clock", float(self._clock()))

    def stop(self) -> None:
        if self._current is not None:
            self._finalize_window()
        self.active = False

    # ---------------- 投喂 ----------------
    def on_toggle(self, enabled: bool, alpha: float, beta: float) -> Optional[ToggleEvent]:
        if not self.active:
            return None
        event = ToggleEvent(t=float(self._clock()), enabled=bool(enabled),
                            alpha=_f(alpha), beta=_f(beta))
        self._events.append(event)
        if self._current is not None:
            self._finalize_window()
        self._open_window(bool(enabled))
        return event

    def on_sample(self, status: Dict[str, Any]) -> None:
        if not self.active or self._current is None:
            return
        self._context = {key: status.get(key) for key in CONTEXT_KEYS if key in status}
        self._current.context = dict(self._context)

        frames = _i(status.get("frames_processed"))
        ok = _i(status.get("frames_decode_ok"))
        overflow = _i(status.get("rx_overflow_count"))
        frames_delta = frames - self._last_status_frames
        ok_delta = ok - self._last_status_ok
        overflow_delta = overflow - self._last_status_overflow
        self._last_status_frames, self._last_status_ok = frames, ok
        self._last_status_overflow = overflow

        config = (_f(status.get("tx_uncoded_bits_len")), _f(status.get("mod_order")))
        # 字段缺失（NaN）不视为配置变化，保持上次口径
        if np.isfinite(config[0]) and np.isfinite(config[1]) and config != self._last_config:
            if np.isfinite(self._last_config[0]) and self._current is not None and self._current.valid_samples > 0:
                self._finalize_window()
                self._open_window(self._current_enabled)
            self._last_config = config
            self._update_symbol_counts(status)

        if frames_delta <= 0:
            return  # 链路空闲采样：既非数据也非异常
        if overflow_delta > 0:
            self._current.dropped_samples += 1
            self._anomaly_drops.append((float(self._clock()), "rx_overflow"))
            if "rx_overflow" not in self._current.anomalies:
                self._current.anomalies.append("rx_overflow")
            return
        ser = _f(status.get("measured_ser"))
        if not np.isfinite(ser):
            self._current.dropped_samples += 1
            self._anomaly_drops.append((float(self._clock()), "sync_loss"))
            if "sync_loss" not in self._current.anomalies:
                self._current.anomalies.append("sync_loss")
            return

        if not np.isfinite(self._symbols_per_frame):
            self._update_symbol_counts(status)
        transient = (float(self._clock()) - self._current.start_t) < TRANSIENT_SEC
        self._current.valid_samples += 1
        self._current.samples.append(
            WindowSample(
                t=float(self._clock()),
                frames_delta=frames_delta,
                ok_delta=ok_delta,
                ser=ser,
                fec_ber=_f(status.get("fec_bit_ber", status.get("ber"))),
                evm_percent=_f(status.get("evm_average_percent", status.get("evm_percent"))),
                alpha=_f(status.get("alpha")),
                beta=_f(status.get("beta")),
                transient=transient,
            )
        )

    # ---------------- 产出 ----------------
    def windows(self) -> List[ObservationWindow]:
        return list(self._windows)

    def anomaly_drops(self) -> List[Any]:
        """异常丢弃采样时刻列表 [(t, kind), ...]，供时间轴阴影使用。"""
        return list(self._anomaly_drops)

    def events(self) -> List[ToggleEvent]:
        return list(self._events)

    def excluded_window_count(self) -> int:
        return sum(1 for item in self._windows if item.excluded)

    def latest_pair(self) -> Optional[PairComparison]:
        before_win = self._last_finalized(MODE_OFF)
        after_win = self._last_finalized(MODE_ON)
        after_agg = after_win.aggregate if after_win is not None else None
        after_ctx = after_win.context if after_win is not None else None
        # 开启中的段也参与配对：结论面板需要实时"开启后"数字（F5）
        if after_agg is None and self._current is not None and self._current.mode == MODE_ON:
            prov = self._provisional_aggregate(self._current)
            if prov is not None:
                after_agg, after_ctx = prov, self._current.context
        if before_win is None or after_agg is None:
            return None
        comparable, note = self._comparability(before_win.context, after_ctx)
        return PairComparison(
            before=before_win.aggregate,
            after=after_agg,
            ser_improvement_db=ser_improvement_db(before_win.aggregate.ser, after_agg.ser),
            evm_delta_pp=after_agg.evm_mean - before_win.aggregate.evm_mean,
            crc_delta=after_agg.crc_ok_ratio - before_win.aggregate.crc_ok_ratio,
            comparable=comparable,
            comparability_note=note,
        )

    def to_export_dict(self) -> Dict[str, Any]:
        pair = self.latest_pair()
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_started_at": self._started_wall,
            "context": dict(self._context),
            "windows": [w.as_dict() for w in self._windows],
            "toggle_events": [e.as_dict() for e in self._events],
            "anomaly_drops": [{"t": t, "kind": kind} for t, kind in self._anomaly_drops],
            "latest_pair": self._pair_as_dict(pair),
            "excluded_window_count": self.excluded_window_count(),
        }

    # ---------------- 内部 ----------------
    def _open_window(self, enabled: bool) -> None:
        self._current_enabled = bool(enabled)
        self._current = ObservationWindow(
            mode=MODE_ON if enabled else MODE_OFF,
            start_t=float(self._clock()),
            context=dict(self._context),
        )

    def _finalize_window(self) -> None:
        win = self._current
        if win is None:
            return
        win.end_t = float(self._clock())
        self._current = None
        total_seen = win.valid_samples + win.dropped_samples
        if win.dropped_samples > 0 and total_seen > 0 and (win.dropped_samples / total_seen) > MAX_ANOMALY_SAMPLE_RATIO:
            win.excluded = True
            win.exclusion_reason = f"异常采样占比 {win.dropped_samples}/{total_seen}"
            self._windows.append(win)
            return
        win.aggregate = self._compute_aggregate(win)
        self._windows.append(win)

    def _last_finalized(self, mode: str) -> Optional[ObservationWindow]:
        return next((w for w in reversed(self._windows)
                     if w.mode == mode and not w.excluded and w.aggregate is not None), None)

    def _provisional_aggregate(self, win: ObservationWindow) -> Optional[WindowAggregate]:
        """为开启中的段计算临时汇总，规则与收尾一致。"""
        total_seen = win.valid_samples + win.dropped_samples
        if win.dropped_samples > 0 and total_seen > 0 and (win.dropped_samples / total_seen) > MAX_ANOMALY_SAMPLE_RATIO:
            return None
        return self._compute_aggregate(win)

    def _compute_aggregate(self, win: ObservationWindow) -> Optional[WindowAggregate]:
        samples = [s for s in win.samples if not s.transient]
        if not samples:
            return None
        frames = sum(s.frames_delta for s in samples)
        if frames <= 0:
            return None
        ok = sum(s.ok_delta for s in samples)
        weight = float(frames)
        ser_n = int(round(weight * self._symbols_per_frame)) if np.isfinite(self._symbols_per_frame) and self._symbols_per_frame > 0 else 0
        fec_n = int(round(weight * self._coded_bits_per_frame)) if np.isfinite(self._coded_bits_per_frame) and self._coded_bits_per_frame > 0 else 0
        ser_k = sum(s.ser * s.frames_delta for s in samples) * self._symbols_per_frame if ser_n > 0 else float("nan")
        fec_k = sum(s.fec_ber * s.frames_delta for s in samples) * self._coded_bits_per_frame if fec_n > 0 else float("nan")
        ser = (ser_k / ser_n) if ser_n > 0 else float(np.mean([s.ser for s in samples]))
        fec_valid = [s.fec_ber for s in samples if np.isfinite(s.fec_ber)]
        fec_ber = (fec_k / fec_n) if fec_n > 0 else (float(np.mean(fec_valid)) if fec_valid else float("nan"))
        evm_values = [(s.evm_percent, s.frames_delta) for s in samples if np.isfinite(s.evm_percent)]
        evm_mean = float(np.average([v for v, _ in evm_values], weights=[w for _, w in evm_values])) if evm_values else float("nan")
        alpha_values = [(s.alpha, s.frames_delta) for s in samples if np.isfinite(s.alpha)]
        beta_values = [(s.beta, s.frames_delta) for s in samples if np.isfinite(s.beta)]
        return WindowAggregate(
            frames=frames,
            crc_ok_ratio=(ok / frames) if frames > 0 else float("nan"),
            ser=float(ser),
            ser_k=float(np.round(ser_k, 1)) if np.isfinite(ser_k) else float("nan"),
            ser_n=ser_n,
            fec_ber=float(fec_ber),
            fec_k=float(np.round(fec_k, 1)) if np.isfinite(fec_k) else float("nan"),
            fec_n=fec_n,
            evm_mean=evm_mean,
            alpha_mean=float(np.average([v for v, _ in alpha_values], weights=[w for _, w in alpha_values])) if alpha_values else float("nan"),
            beta_mean=float(np.average([v for v, _ in beta_values], weights=[w for _, w in beta_values])) if beta_values else float("nan"),
            grade=grade_from_k(ser_k) if ser_n > 0 else INSUFFICIENT,
            estimated_k=any(s.frames_delta > 1 for s in samples),
        )

    def _update_symbol_counts(self, status: Dict[str, Any]) -> None:
        uncoded_bits = _f(status.get("tx_uncoded_bits_len"))
        mod_order = _f(status.get("mod_order"))
        if np.isfinite(uncoded_bits) and np.isfinite(mod_order) and mod_order > 1:
            self._symbols_per_frame = uncoded_bits / math.log2(mod_order)
        coded_bits = _f(status.get("tx_coded_bits_len"))
        if np.isfinite(coded_bits) and coded_bits > 0:
            self._coded_bits_per_frame = coded_bits

    @staticmethod
    def _comparability(before_ctx: Dict[str, Any], after_ctx: Dict[str, Any]):
        diffs = []
        for key in CONTEXT_KEYS:
            b, a = before_ctx.get(key), after_ctx.get(key)
            if b is None and a is None:
                continue
            try:
                if not math.isclose(float(b), float(a), rel_tol=1e-9, abs_tol=1e-9):
                    diffs.append(key)
            except (TypeError, ValueError):
                if b != a:
                    diffs.append(key)
        if diffs:
            return False, "信道上下文不可比: " + ", ".join(diffs)
        return True, ""

    @staticmethod
    def _pair_as_dict(pair: Optional[PairComparison]) -> Optional[Dict[str, Any]]:
        if pair is None:
            return None
        return {
            "before": pair.before.as_dict(),
            "after": pair.after.as_dict(),
            "ser_improvement_db": _sanitize(pair.ser_improvement_db),
            "evm_delta_pp": _sanitize(pair.evm_delta_pp),
            "crc_delta": _sanitize(pair.crc_delta),
            "comparable": pair.comparable,
            "comparability_note": pair.comparability_note,
        }
