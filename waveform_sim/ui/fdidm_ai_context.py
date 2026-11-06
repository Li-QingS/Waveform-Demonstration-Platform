"""Bounded descriptions of the FDIDM charts already displayed by the UI.

No receiver calls, IQ uploads, report exports or hardware operations occur here.
"""
from __future__ import annotations

import math
import numpy as np


def _scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        return value[:600]
    return value if isinstance(value, (str, int, float, bool, type(None))) else None


def _series(item, name):
    x, y = item.getData()
    x = np.asarray([] if x is None else x, dtype=float).reshape(-1)
    y = np.asarray([] if y is None else y, dtype=float).reshape(-1)
    count = min(x.size, y.size)
    valid = np.isfinite(x[:count]) & np.isfinite(y[:count])
    x, y = x[:count][valid], y[:count][valid]
    indices = np.linspace(0, max(0, y.size - 1), min(40, y.size), dtype=int)
    summary = {"name": name, "points_total": int(count), "finite_points": int(y.size),
               "samples": [[float(x[i]), float(y[i])] for i in indices]}
    if y.size:
        summary.update(y_first=float(y[0]), y_last=float(y[-1]),
                       y_min=float(y.min()), y_max=float(y.max()), y_mean=float(y.mean()))
    return summary


def _axis(plot, side):
    axis = plot.getAxis(side)
    label = axis.labelText
    return f"{label} ({axis.labelUnits})" if axis.labelUnits else str(label)


def _curve_chart(plot, title, group, curves, note=""):
    series = [_series(item, name) for name, item in curves]
    return {"title": title, "group": group, "x_axis": _axis(plot, "bottom"),
            "y_axis": _axis(plot, "left"), "series": series,
            "available": any(item["finite_points"] for item in series), "note": note}


def _effect_chart(tab):
    pair = tab.observer.latest_pair()
    available = bool(pair is not None and pair.comparable and
                     pair.before.frames > 0 and pair.after.frames > 0)
    chart = {"title": "局部 OFF→ON 效果", "group": "优势观测", "x_axis": "指标 (EVM / SER)",
             "y_axis": "相对 OFF (%)", "series": [], "available": available,
             "note": "OFF基线为100%，ON/OFF百分比越低越好；不是原始EVM或SER。无可比窗口时柱体为占位，不是零误码证据。"}
    if pair is not None:
        chart["comparison"] = {"comparable": bool(pair.comparable), "outcome": pair.outcome,
                               "note": pair.comparability_note}
    if available:
        baseline = list(tab.timeline_baseline_bars.opts["height"])
        candidate = list(tab.timeline_candidate_bars.opts["height"])
        chart["categories"] = ["EVM", "SER"]
        chart["baseline_percent"] = [_scalar(v) if v > 0 else None for v in baseline]
        chart["candidate_percent"] = [_scalar(v) if baseline[i] > 0 else None
                                      for i, v in enumerate(candidate)]
        chart["before"] = {key: _scalar(getattr(pair.before, key))
                           for key in ("frames", "ser", "ser_k", "ser_n", "evm_mean", "fec_ber")}
        chart["after"] = {key: _scalar(getattr(pair.after, key))
                          for key in ("frames", "ser", "ser_k", "ser_n", "evm_mean", "fec_ber")}
    return chart


def _surface_chart(tab):
    canvas = tab.ab_surface_canvas
    points = []
    for point in canvas._points:
        values = [_scalar(point.get(key)) for key in ("alpha", "beta", "z")]
        if all(isinstance(value, (int, float)) for value in values):
            points.append({"alpha": values[0], "beta": values[1], "z": values[2],
                           "sample_count": _scalar(point.get("sample_count")),
                           "finalized": bool(point.get("finalized", False))})
    metric = str(canvas._metric_label)
    chart = {"title": "α-β 性能面", "group": "链路诊断", "x_axis": "α", "y_axis": "β",
             "z_axis": metric, "direction": canvas._metric_direction,
             "available": bool(points), "series": [], "points_total": len(points),
             "note": "仅包含图中已显示的测量点；未测量区域不能补值。finalized=false的兼容参考点不保证经过冻结平均窗口。"}
    if points:
        direction = canvas._metric_direction
        best_idx = (max if direction == "higher" else min)(range(len(points)), key=lambda i: points[i]["z"])
        indices = list(np.linspace(0, len(points) - 1, min(40, len(points)), dtype=int))
        if best_idx not in indices:
            indices[len(indices) // 2] = best_idx
        chart.update(points=[points[int(i)] for i in sorted(set(indices))],
                     best=points[best_idx], z_min=min(p["z"] for p in points),
                     z_max=max(p["z"] for p in points))
    return chart


def build_fdidm_chart_context(tab):
    """Snapshot displayed plot data and whitelisted cached status only."""
    charts = [
        _effect_chart(tab),
        _curve_chart(tab.timeline_evm_plot, "数据辅助 EVM 趋势", "优势观测",
                     [("数据辅助EVM滚动中位数趋势", tab.timeline_evm_curve)],
                     "时间图使用最近5个样本的稳健中位数；与诊断页单次EVM不是完全相同口径。"),
        _curve_chart(tab.timeline_ab_plot, "自适应动作 α / β", "优势观测",
                     [("α", tab.timeline_alpha_curve), ("β", tab.timeline_beta_curve)],
                     "参数改变本身不代表链路收益；需结合可比窗口和实际验证结果。"),
        _surface_chart(tab),
        _curve_chart(tab.rx_spectrum_plot, "RX 频谱", "链路诊断", [("接收频谱", tab.rx_curve)]),
        _curve_chart(tab.evm_plot, "EVM 曲线", "链路诊断", [("单次EVM", tab.evm_curve)]),
        _curve_chart(tab.constellation_plot, "接收星座", "链路诊断", [("I/Q散点", tab.constellation_scatter)],
                     "横轴I、纵轴Q；仅给出有限采样的散点，不能仅靠包围范围判断误码率。"),
    ]
    keys = ("frames_processed", "frames_decode_ok", "measured_ser", "ser_errors_total", "ser_symbols_total",
            "fec_bit_ber", "raw_bit_ber", "data_aided_evm_percent", "decision_directed_evm_percent",
            "evm_percent", "alpha", "beta", "sync_metric", "cfo_est_hz", "channel_mode", "channel_estimator",
            "pilot_residual_sinr_db", "pilot_fit_nmse", "rx_overflow_count", "rx_processing_gap_count",
            "adaptive_validation_state", "adaptive_validation_reason", "gmi_gain_lower_nats",
            "gmi_threshold_nats", "samp_rate", "rx_spectrum_stale", "constellation_source")
    status = {key: _scalar(tab._last_status_snapshot[key]) for key in keys if key in tab._last_status_snapshot}
    selected_tab = tab.plot_tabs.tabText(tab.plot_tabs.currentIndex())
    # Keep the active page's evidence first if the provider's context cap is hit.
    charts.sort(key=lambda chart: chart["group"] != selected_tab)
    return {"source": "current_page_charts", "selected_tab": selected_tab,
            "has_data": any(chart["available"] for chart in charts), "charts": charts, "status": status,
            "summary": tab.observation_stats_label.text()[:1600],
            "scope": "图表来自当前页面保留的实际绘图数据；历史日志另有来源，二者不能默认为同次实验。优先解释当前页签。"}
