"""Read-only, bounded context snapshots for the platform-level assistant."""
from __future__ import annotations

import math
import re
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QLabel, QLineEdit, QPushButton,
    QSpinBox, QTabWidget,
)
import pyqtgraph as pg

from ..service.platform_guidance import get_page_guide


PAGE_IDS = {
    ("simulation", 0): "simulation.fdidm",
    ("simulation", 1): "simulation.ofdm",
    ("simulation", 2): "simulation.otfs",
    ("simulation", 3): "simulation.afdm",
    ("simulation", 4): "simulation.compare",
    ("hardware", 0): "hardware.fdidm",
    ("hardware", 1): "hardware.link",
}
PAGE_TITLES = {
    "simulation.fdidm": "FDIDM 软波形",
    "simulation.ofdm": "OFDM",
    "simulation.otfs": "OTFS",
    "simulation.afdm": "AFDM",
    "simulation.compare": "波形对比",
    "hardware.fdidm": "FDIDM 自适应验证",
    "hardware.link": "三波形链路测评",
}


def _safe(value: Any, depth: int = 0):
    if depth > 5:
        return "[已省略]"
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value if not isinstance(value, str) else value[:600]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if is_dataclass(value):
        return _safe(asdict(value), depth + 1)
    if isinstance(value, dict):
        return {str(k): _safe(v, depth + 1) for k, v in list(value.items())[:80]
                if not any(term in str(k).lower() for term in
                           ("raw_iq", "samples", "tx_bits", "rx_bits", "payload", "password", "api_key", "secret", "constellation", "h_tf", "h_eff"))}
    if isinstance(value, np.ndarray):
        return {"array_shape": list(value.shape), "note": "原始数组已省略"}
    if isinstance(value, (list, tuple)):
        return [_safe(v, depth + 1) for v in list(value)[:80]]
    return str(value)[:600]


_REFERENCE_TERMS = (
    "wilson", "upper", "upperbound", "reference", "baseline", "theory",
    "zero", "上界", "参考", "理论", "基线", "零误码",
)


def _item_name(item, fallback: str) -> str:
    try:
        name = item.name()
    except (AttributeError, TypeError):
        name = getattr(item, "name", "")
    return str(name or fallback)[:160]


def _series_kind(name: str, attr_name: str) -> str:
    source = f"{name} {attr_name}".lower()
    confidence_interval = bool(re.search(r"(?<![a-z0-9])ci(?:[_\s]|$)", source))
    return "reference" if confidence_interval or any(term in source for term in _REFERENCE_TERMS) else "measured"


def _series(item, name: str, attr_name: str = ""):
    x = y = None
    try:
        original = item.getOriginalDataset()
        if original is not None:
            x, y = original
    except Exception:
        pass
    if x is None or y is None:
        try:
            x, y = item.getData()
        except Exception:
            return None
    try:
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
    except Exception:
        return None
    count = min(x.size, y.size)
    x, y = x[:count], y[:count]
    valid = np.isfinite(x) & np.isfinite(y)
    finite_x, finite_y = x[valid], y[valid]
    if finite_y.size == 0:
        return {"name": name, "attr_name": attr_name, "kind": _series_kind(name, attr_name),
                "visible": bool(item.isVisible()), "points_total": int(count), "finite_points": 0, "samples": []}
    indices = np.linspace(0, finite_y.size - 1, min(40, finite_y.size), dtype=int)
    return {
        "name": name,
        "attr_name": attr_name,
        "kind": _series_kind(name, attr_name),
        "visible": bool(item.isVisible()),
        "points_total": int(count),
        "finite_points": int(finite_y.size),
        "y_min": float(finite_y.min()), "y_max": float(finite_y.max()),
        "y_first": float(finite_y[0]), "y_last": float(finite_y[-1]),
        "y_mean": float(finite_y.mean()),
        "samples": [[float(finite_x[i]), float(finite_y[i])] for i in indices],
    }


def _axis(plot, side):
    axis = plot.getAxis(side)
    label = axis.labelText or side
    return f"{label} ({axis.labelUnits})" if axis.labelUnits else str(label)


def _nested_item_names(page):
    """Name page-owned plot items without walking back into Qt ownership trees."""
    names = {}
    for attr_name, value in vars(page).items():
        if isinstance(value, (pg.PlotDataItem, pg.ScatterPlotItem, pg.ImageItem)):
            names[id(value)] = attr_name
        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (pg.PlotDataItem, pg.ScatterPlotItem, pg.ImageItem)):
                    names[id(item)] = f"{attr_name}.{key}"
    return names


def _owned_items(page):
    for attr_name, value in vars(page).items():
        if isinstance(value, (pg.PlotDataItem, pg.ScatterPlotItem, pg.ImageItem)):
            yield attr_name, value
        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (pg.PlotDataItem, pg.ScatterPlotItem, pg.ImageItem)):
                    yield f"{attr_name}.{key}", item


def _image_summary(item, attr_name: str):
    image = getattr(item, "image", None)
    if image is None:
        return None
    try:
        values = np.asarray(image, dtype=float)
    except Exception:
        return None
    finite = np.isfinite(values)
    count = int(finite.sum())
    if not count:
        return {"attr_name": attr_name, "shape": list(values.shape), "finite_cells": 0, "samples": []}
    finite_values = values[finite]
    flat_peak = int(np.nanargmax(np.where(finite, values, -np.inf)))
    peak = [int(index) for index in np.unravel_index(flat_peak, values.shape)]
    rows = np.linspace(0, max(0, values.shape[0] - 1), min(8, values.shape[0]), dtype=int)
    cols = np.linspace(0, max(0, values.shape[1] - 1), min(5, values.shape[1]), dtype=int) if values.ndim > 1 else np.array([0])
    samples = []
    if values.ndim == 1:
        samples = [[int(row), float(values[row])] for row in rows if math.isfinite(float(values[row]))]
    elif values.ndim == 2:
        samples = [[int(row), int(col), float(values[row, col])]
                   for row in rows for col in cols if math.isfinite(float(values[row, col]))][:40]
    return {"attr_name": attr_name, "shape": list(values.shape), "finite_cells": count,
            "min": float(finite_values.min()), "max": float(finite_values.max()),
            "peak_position": peak, "peak_value": float(values[tuple(peak)]), "samples": samples[:40]}


def _generic_charts(page):
    charts = []
    item_names = _nested_item_names(page)
    seen_images = set()
    for plot in page.findChildren(pg.PlotWidget):
        for attr_name, item in vars(plot).items():
            if isinstance(item, (pg.PlotDataItem, pg.ScatterPlotItem, pg.ImageItem)):
                item_names[id(item)] = f"{getattr(plot, 'objectName', lambda: '')() or 'plot'}.{attr_name}"
        title_item = getattr(plot.getPlotItem(), "titleLabel", None)
        title = str(title_item.text or "") if title_item is not None else ""
        if not title:
            title = plot.objectName() or "页面结果图"
        series = []
        for index, item in enumerate(plot.listDataItems()):
            attr_name = item_names.get(id(item), "")
            summary = _series(item, _item_name(item, attr_name or f"series-{index + 1}"), attr_name)
            if summary is not None:
                series.append(summary)
        ctrl = plot.getPlotItem().ctrl
        charts.append({
            "title": title,
            "group": "当前页面",
            "x_axis": _axis(plot, "bottom"),
            "y_axis": _axis(plot, "left"),
            "log_axes": {"x": bool(ctrl.logXCheck.isChecked()), "y": bool(ctrl.logYCheck.isChecked())},
            "visible": bool(plot.isVisible()),
            "series": series,
            "available": any(item.get("finite_points", 0) for item in series),
            "note": "曲线来自当前页面已显示的数据；空图仅表示尚未产生结果。",
        })
        for item in getattr(plot.getPlotItem(), "items", []):
            if isinstance(item, pg.ImageItem) and id(item) not in seen_images:
                seen_images.add(id(item))
                image = _image_summary(item, item_names.get(id(item), "image"))
                if image is not None:
                    charts.append({"title": title, "group": "当前页面", "x_axis": _axis(plot, "bottom"),
                                   "y_axis": _axis(plot, "left"), "log_axes": {"x": False, "y": False},
                                   "visible": bool(plot.isVisible()), "series": [], "image": image,
                                   "available": bool(image.get("finite_cells", 0)),
                                   "note": "二维图像为当前页面缓存的显示值；仅提供有界统计和采样单元。"})
    for attr_name, item in _owned_items(page):
        # ImageItems are often graphics items rather than QObject children; include
        # direct page attributes even when their plot cannot be recovered.
        if isinstance(item, pg.ImageItem) and id(item) not in seen_images:
            image = _image_summary(item, attr_name)
            if image is not None:
                charts.append({"title": attr_name, "group": "当前页面", "x_axis": "x", "y_axis": "y",
                               "log_axes": {"x": False, "y": False}, "visible": True, "series": [],
                               "image": image, "available": bool(image.get("finite_cells", 0)),
                               "note": "二维图像为当前页面缓存的显示值；仅提供有界统计和采样单元。"})
    return charts


def _controls(page):
    attr_names = {id(value): key for key, value in vars(page).items()
                  if isinstance(value, (QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton))}
    buttons = []
    for widget in page.findChildren(QPushButton):
        text = widget.text().strip()
        if text:
            buttons.append({"text": text[:120], "enabled": bool(widget.isEnabled()), "visible": bool(widget.isVisibleTo(page)),
                            "attr_name": attr_names.get(id(widget), widget.objectName())})
    parameters = []
    for widget in page.findChildren((QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox)):
        name = attr_names.get(id(widget), widget.objectName()) or widget.__class__.__name__
        if isinstance(widget, QComboBox):
            value = widget.currentText()
        elif isinstance(widget, QCheckBox):
            value = bool(widget.isChecked())
        else:
            value = widget.value()
        parameters.append({"attr_name": name, "value": _safe(value), "enabled": bool(widget.isEnabled()),
                           "visible": bool(widget.isVisibleTo(page))})
    return {"buttons": buttons[:100], "parameters": parameters[:120]}


def _scan_snapshot(page):
    getter = getattr(page, "_scan_snapshot", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return {}
    value = getattr(page, "_scan_results", {})
    return value if isinstance(value, dict) else {}


def _has_live_snapshot(page) -> bool:
    snapshot = getattr(page, "_last_live_snapshot", None)
    if not snapshot:
        return False
    if bool(getattr(page, "_has_live_result", False)):
        return True
    if isinstance(snapshot, dict):
        for key in ("frames", "left", "right", "cumulative_bits", "elapsed_s"):
            value = snapshot.get(key)
            if isinstance(value, dict) and value.get("frames", 0):
                return True
            if isinstance(value, (int, float)) and value > 0:
                return True
    return False


def _has_scan_result(snapshot) -> bool:
    if not isinstance(snapshot, dict):
        return False
    return any(bool(snapshot.get(axis, {}).get("OFDM", {}).get("x")) for axis in ("cfo", "doppler"))


def _result_configs(page, subview):
    configs = {}
    study_results = getattr(page, "_study_results", None)
    study_config = getattr(page, "_study_config", None)
    if study_config is not None and bool(study_results):
        configs["study"] = _safe(study_config)
    live_config = getattr(page, "_live_config", None)
    if live_config is not None and _has_live_snapshot(page):
        configs["live"] = _safe(live_config)
    scan = _scan_snapshot(page)
    if _has_scan_result(scan) and scan.get("config") is not None:
        configs["scan"] = _safe(scan["config"])
    subview_lower = subview.lower()
    if "性能研究" in subview or "研究" in subview_lower or subview_lower == "results":
        selected = configs.get("study") or configs.get("scan") or configs.get("live")
    elif "自动扫描" in subview or "scan" in subview_lower:
        selected = configs.get("scan") or configs.get("study") or configs.get("live")
    else:
        selected = configs.get("live") or configs.get("study") or configs.get("scan")
    return selected, configs


def _status_metrics(page):
    status = getattr(page, "_last_status_snapshot", None)
    if not isinstance(status, dict):
        return {}
    keys = ("frames_processed", "frames_decode_ok", "measured_ser", "ser_errors_total", "ser_symbols_total",
            "fec_bit_ber", "raw_bit_ber", "evm_percent", "data_aided_evm_percent", "sync_metric",
            "cfo_est_hz", "rx_overflow_count", "adaptive_validation_state", "adaptive_validation_reason")
    return {key: _safe(status[key]) for key in keys if key in status}


def _displayed_summary(page):
    """Short operational state rendered by the current page, never editable payload text."""
    names = ("decode_status_label", "status_text_label", "study_progress", "scan_summary",
             "scan_note_label", "note_label", "observation_stats_label")
    summary = {}
    for name in names:
        widget = getattr(page, name, None)
        text = getattr(widget, "text", None)
        if callable(text):
            try:
                value = str(text()).strip()
            except Exception:
                continue
            if value:
                summary[name] = value[:1200]
    labels = getattr(page, "metric_value_labels", None)
    if isinstance(labels, dict):
        metrics = {}
        for name, widget in list(labels.items())[:20]:
            text = getattr(widget, "text", None)
            if callable(text):
                try:
                    metrics[str(name)[:100]] = str(text())[:300]
                except Exception:
                    pass
        if metrics:
            summary["metrics"] = metrics
    return summary


def _log_snapshot(page):
    getter = getattr(page, "_get_ai_log_snapshot", None)
    if callable(getter):
        try:
            value = str(getter() or "")
            return ("[记录已截取最近 24000 字符]\n" + value[-24000:]) if len(value) > 24000 else value
        except Exception:
            return ""
    log_widget = getattr(page, "log_text", None)
    text = getattr(log_widget, "toPlainText", None)
    if callable(text):
        try:
            value = str(text() or "")
            return ("[记录已截取最近 24000 字符]\n" + value[-24000:]) if len(value) > 24000 else value
        except Exception:
            return ""
    return ""


def _running_state(page):
    running = bool(getattr(page, "test_running", False) or getattr(page, "_study_active", False)
                   or getattr(page, "_auto_ber_snr_active", False) or getattr(page, "_scan_active", False)
                   or getattr(page, "tb", None) is not None)
    runner = getattr(page, "live_runner", None)
    is_running = getattr(runner, "is_running", None)
    if callable(is_running):
        try:
            running = bool(is_running()) or running
        except Exception:
            pass
    worker = getattr(page, "_scan_worker", None)
    alive = getattr(worker, "is_alive", None)
    if callable(alive):
        try:
            running = bool(alive()) or running
        except Exception:
            pass
    timer = getattr(page, "update_timer", None)
    active = getattr(timer, "isActive", None)
    if callable(active):
        try:
            running = bool(active()) or running
        except Exception:
            pass
    return running


def _chart_has_measurement(charts):
    for chart in charts:
        image = chart.get("image", {})
        if image.get("finite_cells", 0):
            return True
        for series in chart.get("series", []):
            if series.get("kind") != "reference" and series.get("finite_points", 0):
                # An all-zero data curve is a valid display but needs frame/count
                # evidence before it can be treated as a completed experiment.
                if any(abs(float(series.get(key, 0.0))) > 0 for key in ("y_min", "y_max", "y_first", "y_last")):
                    return True
    return False


def _cached_results(page):
    candidates = []
    study_results = getattr(page, "_study_results", None)
    if study_results:
        candidates.append({"name": "study", "points": len(study_results), "value": _safe(study_results[:12])})
    if _has_live_snapshot(page):
        candidates.append({"name": "live", "value": _safe(getattr(page, "_last_live_snapshot", {}))})
    scan = _scan_snapshot(page)
    if _has_scan_result(scan):
        candidates.append({"name": "scan", "value": _safe({"status": scan.get("status"), "done": scan.get("done"),
                                                               "cfo": scan.get("cfo"), "doppler": scan.get("doppler")})})
    metrics = _status_metrics(page)
    if any(bool(metrics.get(key)) for key in ("frames_processed", "frames_decode_ok", "ser_symbols_total")):
        candidates.append({"name": "displayed_status", "value": metrics})
    return candidates[:12]


def _subview(page):
    result = []
    for name in ("tabs", "view_tabs", "plot_tabs", "mode_tabs", "result_tabs", "study_tabs", "config_tabs", "parameter_tabs"):
        tabs = getattr(page, name, None)
        if isinstance(tabs, QTabWidget):
            result.append(tabs.tabText(tabs.currentIndex()))
    waveform = getattr(page, "waveform_combo", None)
    if isinstance(waveform, QComboBox):
        result.append(waveform.currentText())
    return " / ".join(result)


def _page_snapshot(window):
    mode = getattr(window, "current_mode", None)
    if mode not in {"simulation", "hardware"}:
        return "home", "平台首页", "home", None
    workspace = window.workspaces.get(mode)
    page = getattr(workspace, "_current_page", None) if workspace is not None else None
    index = getattr(workspace, "_current_index", -1) if workspace is not None else -1
    page_id = PAGE_IDS.get((mode, index), f"{mode}.unknown")
    return page_id, PAGE_TITLES.get(page_id, "当前页面"), mode, page


def build_platform_context(window) -> dict:
    page_id, title, workspace_name, page = _page_snapshot(window)
    if page is None:
        guide = get_page_guide(page_id)
        landing = getattr(window, "landing_page", None)
        platform = {"guide": guide, "page_id": page_id, "page_title": title, "workspace": workspace_name,
                    "subview": "", "running": False, "running_state": False, "has_results": False,
                    "controls": _controls(landing) if landing is not None else {"buttons": [], "parameters": []},
                    "displayed_metrics": {}, "displayed_summary": {}, "cached_results": [],
                    "result_config": None, "result_configs": {}}
        return {"context_id": page_id, "source_description": title, "log_text": "",
                "chart_context": {"source": "current_page_charts", "selected_tab": "", "has_data": False, "charts": [], "scope": "当前页面"},
                "platform_context": platform}

    subview = _subview(page)
    context_id = f"{page_id}:{subview}" if subview else page_id
    charts = []
    if hasattr(page, "_get_ai_chart_context"):
        try:
            charts_context = page._get_ai_chart_context()
            charts = list(charts_context.get("charts", []))
        except Exception:
            charts = []
    log_text = _log_snapshot(page)
    if not charts:
        charts = _generic_charts(page)
    cached_results = _cached_results(page)
    result_config, result_configs = _result_configs(page, subview)
    running = _running_state(page)
    has_results = bool(cached_results) or _chart_has_measurement(charts)
    platform = {"guide": get_page_guide(page_id), "page_id": page_id, "page_title": title,
                "workspace": workspace_name, "subview": subview, "running": running,
                "running_state": running, "has_results": has_results, "controls": _controls(page),
                "displayed_metrics": _status_metrics(page), "displayed_summary": _displayed_summary(page),
                "cached_results": cached_results, "result_config": result_config,
                "result_configs": result_configs}
    chart_context = {"source": "current_page_charts", "selected_tab": subview,
                     "has_data": bool(any(chart.get("available") for chart in charts)),
                     "charts": charts[:30],
                     "scope": "当前页面已显示图表；空图和历史记录不自动混入。"}
    return {"context_id": context_id, "source_description": f"{title}{(' / ' + subview) if subview else ''}",
            "log_text": log_text, "chart_context": chart_context, "platform_context": platform}
