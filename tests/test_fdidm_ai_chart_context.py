"""AI explanations must use the actual plotted data rather than imagined charts."""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from PyQt5.QtWidgets import QApplication
from waveform_sim.ui.fdidm_hardware_test_tab import FDIDMHardwareTestTab


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(app):
    widget = FDIDMHardwareTestTab()
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


def _chart(context, title):
    return next(chart for chart in context["charts"] if chart["title"] == title)


def test_empty_placeholder_charts_are_not_reported_as_measurements(page):
    from waveform_sim.ui.fdidm_ai_context import build_fdidm_chart_context

    context = build_fdidm_chart_context(page)
    assert context["source"] == "current_page_charts"
    assert context["selected_tab"] == "优势观测"
    assert context["has_data"] is False
    assert not _chart(context, "局部 OFF→ON 效果")["available"]
    json.dumps(context, allow_nan=False)


def test_explanation_context_tracks_actual_curves_axes_and_samples(page):
    from waveform_sim.ui.fdidm_ai_context import build_fdidm_chart_context

    page.rx_curve.setData([-1000, 0, 1000], [-70, -10, -65])
    page.evm_curve.setData([0, 1, 2], [30, 20, 15])
    page.timeline_alpha_curve.setData([0, 1, 2], [0.5, 0.75, 1.0])
    page.plot_tabs.setCurrentIndex(0)
    context = build_fdidm_chart_context(page)
    assert context["has_data"] is True
    assert context["selected_tab"] == "链路诊断"
    spectrum = _chart(context, "RX 频谱")
    assert "Hz" in spectrum["x_axis"]
    assert "dB" in spectrum["y_axis"]
    series = spectrum["series"][0]
    assert series["points_total"] == 3
    assert series["y_max"] == -10
    assert series["samples"] == [[-1000.0, -70.0], [0.0, -10.0], [1000.0, -65.0]]
    evm = _chart(context, "EVM 曲线")["series"][0]
    assert evm["y_first"] == 30
    assert evm["y_last"] == 15
    assert evm["y_mean"] == pytest.approx(65 / 3)
    assert "后台原始日志" not in json.dumps(context, ensure_ascii=False)


def test_context_filters_nonfinite_numbers_and_caps_samples_without_losing_endpoints(page):
    from waveform_sim.ui.fdidm_ai_context import build_fdidm_chart_context

    page.timeline_evm_curve.setData([0, 1, 2], [np.nan, 6.0, np.inf])
    page.rx_curve.setData(np.arange(1000), np.arange(1000) / 10)
    context = build_fdidm_chart_context(page)
    json.dumps(context, allow_nan=False)
    evm = _chart(context, "数据辅助 EVM 趋势")["series"][0]
    assert evm["finite_points"] == 1
    assert evm["samples"] == [[1.0, 6.0]]
    spectrum = _chart(context, "RX 频谱")["series"][0]
    assert spectrum["points_total"] == 1000
    assert len(spectrum["samples"]) <= 40
    assert spectrum["samples"][0] == [0.0, 0.0]
    assert spectrum["samples"][-1] == [999.0, 99.9]


def test_context_uses_cached_status_and_survives_hardware_release(page):
    from waveform_sim.ui.fdidm_ai_context import build_fdidm_chart_context

    class ReadForbiddenBackend:
        def get_status(self):
            raise AssertionError("AI context must not read the hardware again")

        def stop(self):
            pass

        def wait(self):
            pass

    page.backend = ReadForbiddenBackend()
    page._last_status_snapshot = {"frames_processed": 12, "alpha": 0.75, "fec_bit_ber": 0.0}
    page.evm_curve.setData([0, 1], [20, 10])
    before = build_fdidm_chart_context(page)
    assert before["status"]["frames_processed"] == 12
    assert before["status"]["fec_bit_ber"] == 0.0
    page.suspend_tasks()
    assert page.backend is None
    after = build_fdidm_chart_context(page)
    assert _chart(after, "EVM 曲线")["series"][0]["y_last"] == 10
