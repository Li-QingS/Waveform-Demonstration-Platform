"""The assistant belongs to the platform, not to an individual experiment page."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QWidget
from waveform_sim.ui import main_window


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def platform(app, monkeypatch):
    class QuietPage(QWidget):
        def __init__(self):
            super().__init__()
            self.stops = 0

        def suspend_tasks(self):
            self.stops += 1

    monkeypatch.setattr(main_window, "_load_page", lambda *_: QuietPage())
    window = main_window.MainWindow()
    window.show()
    QTest.qWait(20)
    yield window
    window.close()
    window.deleteLater()
    app.processEvents()


def test_homepage_global_entry_does_not_load_experiment_pages(platform):
    controller = platform.ai_assistant
    assert controller.button.isVisible()
    assert not platform.simulation_workspace.pages
    assert not platform.hardware_workspace.pages
    controller.button.click()
    assert controller.dialog.isVisible()
    assert controller.dialog._platform_context["platform_context"]["page_id"] == "home"
    assert not platform.simulation_workspace.pages
    assert not platform.hardware_workspace.pages


def test_same_assistant_and_conversation_remain_open_across_all_navigation(platform):
    controller = platform.ai_assistant
    controller.open()
    dialog = controller.dialog
    dialog._add_message("user", "如何开始操作？")
    dialog.message_input.setPlainText("未发送的问题")
    dialog.set_ui_scale(110)
    platform.open_workspace("simulation")
    assert dialog.isVisible()
    assert controller.dialog is dialog
    assert "simulation" in dialog._context_id
    platform.simulation_workspace.tabs.setCurrentIndex(4)
    assert "compare" in dialog._context_id
    platform.open_workspace("hardware")
    assert "hardware" in dialog._context_id
    platform.show_home()
    assert dialog._platform_context["platform_context"]["page_id"] == "home"
    assert dialog.isVisible()
    assert dialog.message_input.toPlainText() == "未发送的问题"
    assert dialog.ui_scale_percent == 110
    assert any(message["content"] == "如何开始操作？" for message in dialog.messages)


def test_global_floating_button_fits_resizes_and_close_cancels_assistant(platform, app):
    platform.ai_assistant.open()
    for size in ((1100, 700), (1400, 900)):
        platform.resize(*size)
        app.processEvents()
        assert platform.rect().contains(platform.ai_assistant.button.geometry())
    dialog = platform.ai_assistant.dialog
    assert platform.close()
    assert not dialog.isVisible()
    assert not dialog._analysis_active


def test_fdidm_embedded_entry_is_hidden_and_routes_to_global_assistant(app):
    window = main_window.MainWindow()
    try:
        window.open_workspace("hardware")
        window.show()
        app.processEvents()
        page = window.hardware_workspace.pages[0]
        assert page._global_ai_assistant is window.ai_assistant
        assert page.ai_floating_button.isHidden()
        page._open_ai_analysis()
        assert window.ai_assistant.dialog.isVisible()
        assert page._ai_analysis_dialog is None
        window.show_home()
        assert window.ai_assistant.dialog.isVisible()
    finally:
        window.close()


def test_accepted_close_detaches_zoom_filter_and_ignores_queued_refresh(platform, app):
    from PyQt5.QtCore import Qt

    controller = platform.ai_assistant
    controller.open()
    dialog = controller.dialog
    scale_before = dialog.ui_scale_percent
    calls = []
    controller.snapshot = lambda: calls.append(True) or {}
    assert platform.close()
    controller.refresh_context()
    QTest.keyClick(dialog.message_input, Qt.Key_Plus, Qt.ControlModifier)
    app.processEvents()
    assert calls == []
    assert dialog.ui_scale_percent == scale_before
