import json
import os

import pytest
import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QLabel, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget
import pyqtgraph as pg


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _page(title="Test page"):
    page = QWidget()
    page.setWindowTitle(title)
    layout = QVBoxLayout(page)
    page.mode_tabs = QTabWidget(page)
    page.mode_tabs.addTab(QWidget(), "Results")
    layout.addWidget(page.mode_tabs)

    page.start_button = QPushButton("Start", page)
    layout.addWidget(page.start_button)
    page.modulation_combo = QComboBox(page)
    page.modulation_combo.addItems(["QPSK", "16QAM"])
    layout.addWidget(QLabel("Modulation", page))
    layout.addWidget(page.modulation_combo)
    page.snr_spin = QDoubleSpinBox(page)
    page.snr_spin.setValue(12.5)
    layout.addWidget(QLabel("Eb/N0", page))
    layout.addWidget(page.snr_spin)
    page.frames_spin = QSpinBox(page)
    page.frames_spin.setValue(10)
    layout.addWidget(page.frames_spin)
    page.enabled_check = QCheckBox("Adaptive", page)
    page.enabled_check.setChecked(True)
    layout.addWidget(page.enabled_check)
    return page


def _install_active_page(window, mode, index, page):
    workspace = window.workspaces[mode]
    workspace.pages[index] = page
    workspace._current_index = index
    workspace._current_page = page
    window.current_mode = mode
    return workspace


def test_home_and_all_platform_page_ids_are_routed_without_loading_pages(app):
    from waveform_sim.ui.main_window import MainWindow
    from waveform_sim.ui.platform_ai_context import build_platform_context

    window = MainWindow()
    try:
        home = build_platform_context(window)
        assert home["platform_context"]["page_id"] == "home"
        assert window.simulation_workspace.pages == {}
        assert window.hardware_workspace.pages == {}

        cases = (
            ("simulation", 0, "simulation.fdidm"),
            ("simulation", 1, "simulation.ofdm"),
            ("simulation", 2, "simulation.otfs"),
            ("simulation", 3, "simulation.afdm"),
            ("simulation", 4, "simulation.compare"),
            ("hardware", 0, "hardware.fdidm"),
            ("hardware", 1, "hardware.link"),
        )
        for mode, index, expected_id in cases:
            page = _page(expected_id)
            _install_active_page(window, mode, index, page)
            context = build_platform_context(window)
            assert context["platform_context"]["page_id"] == expected_id
            assert expected_id in context["context_id"]
            assert context["chart_context"]["source"] == "current_page_charts"
    finally:
        window.close()


def test_active_page_context_does_not_create_other_pages_or_start_backends(app):
    from waveform_sim.ui.main_window import MainWindow
    from waveform_sim.ui.platform_ai_context import build_platform_context

    window = MainWindow()
    try:
        window.open_workspace("simulation")
        active = window.simulation_workspace._current_page
        assert set(window.simulation_workspace.pages) == {0}
        assert window.hardware_workspace.pages == {}
        context = build_platform_context(window)
        assert context["platform_context"]["page_id"] == "simulation.fdidm"
        assert set(window.simulation_workspace.pages) == {0}
        assert window.hardware_workspace.pages == {}
        assert active is window.simulation_workspace._current_page
    finally:
        window.close()


def test_original_positive_ber_is_retained_on_log_plot_and_saved_result_config_wins(app):
    from waveform_sim.ui.main_window import MainWindow
    from waveform_sim.ui.platform_ai_context import build_platform_context

    window = MainWindow()
    page = _page("OFDM")
    plot = pg.PlotWidget(title="BER vs Eb/N0")
    plot.setLabel("bottom", "Eb/N0", units="dB")
    plot.setLabel("left", "BER")
    plot.setLogMode(y=True)
    curve = plot.plot([0.0, 5.0, 10.0], [1e-1, 1e-3, 1e-5], name="measured BER")
    page.layout().addWidget(plot)
    page._study_config = {"snr_db": 8.0, "frames": 40}
    page._study_results = [{"axis_value": 8.0, "ber": 0.001}]
    page.snr_spin.setValue(20.0)
    _install_active_page(window, "simulation", 1, page)
    try:
        context = build_platform_context(window)
        chart = next(chart for chart in context["chart_context"]["charts"] if chart["title"] == "BER vs Eb/N0")
        series = next(series for series in chart["series"] if series["name"] == "measured BER")
        assert chart["log_axes"]["y"] is True
        assert series["y_min"] == pytest.approx(1e-5)
        assert series["samples"][1][1] == pytest.approx(1e-3)
        assert context["platform_context"]["result_config"] == {"snr_db": 8.0, "frames": 40}
        assert any(item["attr_name"] == "snr_spin" and item["value"] == 20.0 for item in context["platform_context"]["controls"]["parameters"])
        json.dumps(context, allow_nan=False)
    finally:
        window.close()


def test_context_survives_home_and_revisit_with_bounded_cached_results(app):
    from waveform_sim.ui.main_window import MainWindow
    from waveform_sim.ui.platform_ai_context import build_platform_context

    window = MainWindow()
    page = _page("Compare")
    page._live_config = {"seed": 42, "snr_db": 12.0}
    page._has_live_result = True
    page._last_live_snapshot = {"left": {"frames": 5, "ber": 0.0}, "raw_iq": [1, 2, 3]}
    _install_active_page(window, "simulation", 4, page)
    try:
        before = build_platform_context(window)
        window.show_home()
        assert build_platform_context(window)["platform_context"]["page_id"] == "home"
        window.current_mode = "simulation"
        window.simulation_workspace._current_page = page
        window.simulation_workspace._current_index = 4
        after = build_platform_context(window)
        assert before["platform_context"]["result_config"] == after["platform_context"]["result_config"]
        assert "raw_iq" not in json.dumps(after, ensure_ascii=True).lower()
        assert len(after["platform_context"]["cached_results"]) <= 12
        json.dumps(after, allow_nan=False)
    finally:
        window.close()


def test_image_context_is_bounded_and_uses_actual_cached_cells(app):
    from waveform_sim.ui.main_window import MainWindow
    from waveform_sim.ui.platform_ai_context import build_platform_context

    window = MainWindow()
    page = _page("OTFS")
    plot = pg.PlotWidget(title="Delay Doppler")
    page.layout().addWidget(plot)
    page.dd_image = pg.ImageItem(axisOrder="row-major")
    page.dd_image.setImage(np.array([[0.0, -5.0, -10.0], [-2.0, -1.0, -8.0]]))
    plot.addItem(page.dd_image)
    _install_active_page(window, "simulation", 2, page)
    try:
        context = build_platform_context(window)
        image_chart = next(chart for chart in context["chart_context"]["charts"] if "image" in chart)
        image = image_chart["image"]
        assert image["shape"] == [2, 3]
        assert image["finite_cells"] == 6
        assert image["max"] == pytest.approx(0.0)
        assert len(image["samples"]) <= 40
        assert context["platform_context"]["has_results"] is True
        json.dumps(context, allow_nan=False)
    finally:
        window.close()


def test_unsaved_editable_live_config_is_not_reported_as_result_config(app):
    from waveform_sim.ui.main_window import MainWindow
    from waveform_sim.ui.platform_ai_context import build_platform_context

    window = MainWindow()
    page = _page("OTFS")
    page._study_config = {"frames": 99}
    page._study_results = []
    page._live_config = {"snr_db": 3.0}
    page._last_live_snapshot = {}
    _install_active_page(window, "simulation", 2, page)
    try:
        context = build_platform_context(window)
        assert context["platform_context"]["result_config"] is None
        assert context["platform_context"]["result_configs"] == {}
        assert context["platform_context"]["has_results"] is False
    finally:
        window.close()


def test_compare_scan_subview_selects_scan_config_and_marks_wilson_reference(app):
    from waveform_sim.ui.main_window import MainWindow
    from waveform_sim.ui.platform_ai_context import build_platform_context

    window = MainWindow()
    page = _page("Compare")
    page.tabs = QTabWidget(page)
    page.tabs.addTab(QWidget(), "实时对比")
    page.tabs.addTab(QWidget(), "自动扫描")
    page.tabs.setCurrentIndex(1)
    page._has_live_result = True
    page._live_config = {"source": "live"}
    page._last_live_snapshot = {"left": {"frames": 2}}
    scan = {"config": {"source": "scan"}, "cfo": {"OFDM": {"x": [0.0], "ber": [0.0]}}}
    page._scan_snapshot = lambda: scan
    page.scan_cfo_plot = pg.PlotWidget(title="BER vs CFO")
    page.scan_cfo_plot.ofdm_zero_curve = page.scan_cfo_plot.plot([0.0], [0.0038])
    page.scan_cfo_plot.ofdm_zero_curve.hide()
    page.scan_cfo_plot.decision_curve = page.scan_cfo_plot.plot([0.0], [0.1], name="decision EVM")
    page.layout().addWidget(page.scan_cfo_plot)
    _install_active_page(window, "simulation", 4, page)
    try:
        context = build_platform_context(window)
        assert "自动扫描" in context["context_id"]
        assert "自动扫描" in context["chart_context"]["selected_tab"]
        assert context["platform_context"]["result_config"] == {"source": "scan"}
        assert context["platform_context"]["result_configs"] == {"live": {"source": "live"}, "scan": {"source": "scan"}}
        chart = next(chart for chart in context["chart_context"]["charts"] if chart["title"] == "BER vs CFO")
        by_name = {series["name"]: series for series in chart["series"]}
        zero = next(series for series in chart["series"] if series["attr_name"].endswith("ofdm_zero_curve"))
        assert zero["kind"] == "reference"
        assert by_name["decision EVM"]["kind"] == "measured"
        assert "visible" in zero
        json.dumps(context, allow_nan=False)
    finally:
        window.close()


def test_real_pages_snapshot_without_starting_backends(app):
    from waveform_sim.ui.main_window import MainWindow
    from waveform_sim.ui.platform_ai_context import build_platform_context

    window = MainWindow()
    try:
        expected = []
        for mode, count in (("simulation", 5), ("hardware", 2)):
            window.open_workspace(mode)
            workspace = window.workspaces[mode]
            for index in range(count):
                workspace.tabs.setCurrentIndex(index)
                app.processEvents()
                context = build_platform_context(window)
                expected.append(context["platform_context"]["page_id"])
                assert context["platform_context"]["guide"]["workflows"]
                assert context["platform_context"]["running"] is False
                json.dumps(context, allow_nan=False)
        window.open_workspace("simulation")
        compare = window.workspaces["simulation"]
        compare.tabs.setCurrentIndex(4)
        app.processEvents()
        compare.pages[4].tabs.setCurrentIndex(1)
        app.processEvents()
        scan_context = build_platform_context(window)
        assert "自动扫描" in scan_context["context_id"]
        assert scan_context["chart_context"]["selected_tab"].startswith("自动扫描")
        assert len(set(expected)) == 7
    finally:
        window.close()
