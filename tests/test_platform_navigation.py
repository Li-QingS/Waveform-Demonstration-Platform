"""Exercise workspace navigation without opening a device or running a link."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QLineEdit, QPushButton, QWidget

from waveform_sim.ui import main_window
from waveform_sim.ui.landing_page import LandingPage


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def platform(app, monkeypatch):
    class FakePage(QWidget):
        def __init__(self):
            super().__init__()
            self.parameter = QLineEdit(self)
            self.running = False
            self.result = None
            self.resumed = 0
            self.stopped = 0

        def resume_tasks(self):
            self.resumed += 1

        def suspend_tasks(self):
            self.running = False
            self.stopped += 1

    monkeypatch.setattr(main_window, "_load_page", lambda *_: FakePage())
    window = main_window.MainWindow()
    window.show()
    app.processEvents()
    yield window
    window.close()


def test_entry_has_two_stacked_buttons_and_does_not_load_backends(platform):
    assert platform.current_mode is None
    assert platform.stack.currentWidget() is platform.landing_page
    assert not platform.simulation_workspace.pages
    assert not platform.hardware_workspace.pages
    buttons = platform.landing_page.findChildren(QPushButton)
    assert len(buttons) == 2
    assert buttons[0].geometry().bottom() < buttons[1].geometry().top()
    assert buttons[0].width() == buttons[1].width()
    assert buttons[0].height() >= 100


def test_homepage_uses_microsoft_yahei_and_distinct_base_colors(app):
    original_font = QFont(app.font())
    app.setFont(QFont("SimSun"))
    page = LandingPage()
    try:
        page.resize(900, 620)
        page.show()
        app.processEvents()

        assert page.font().family() == "Microsoft YaHei"
        assert page.simulation_button.font().family() == "Microsoft YaHei"
        assert page.hardware_button.font().family() == "Microsoft YaHei"

        image = page.grab().toImage()
        assert image.pixelColor(5, 5) == QColor("#F3F6FA")
        simulation = page.simulation_button.rect().translated(
            page.simulation_button.mapTo(page, page.simulation_button.rect().topLeft())
        )
        hardware = page.hardware_button.rect().translated(
            page.hardware_button.mapTo(page, page.hardware_button.rect().topLeft())
        )
        assert image.pixelColor(simulation.left() + 2, simulation.center().y()) == QColor("#0072BD")
        assert image.pixelColor(hardware.left() + 2, hardware.center().y()) == QColor("#D95319")
    finally:
        page.close()
        app.setFont(original_font)


def test_workspace_switch_stops_tasks_and_preserves_configuration(platform):
    QTest.mouseClick(platform.landing_page.simulation_button, Qt.LeftButton)
    assert platform.current_mode == "simulation"
    page = platform.fdidm_tab
    page.parameter.setText("custom alpha/beta")
    page.result = [0.2, 0.1]
    page.running = True
    QTest.mouseClick(platform.simulation_workspace.switch_button, Qt.LeftButton)
    assert platform.current_mode == "hardware"
    assert not page.running
    assert page.stopped == 1
    hardware_page = platform.hardware_workspace.pages[0]
    hardware_page.running = True
    QTest.mouseClick(platform.hardware_workspace.home_button, Qt.LeftButton)
    assert not hardware_page.running
    assert platform.current_mode is None
    QTest.mouseClick(platform.landing_page.simulation_button, Qt.LeftButton)
    assert platform.fdidm_tab is page
    assert page.parameter.text() == "custom alpha/beta"
    assert page.result == [0.2, 0.1]
    assert not page.running
    assert page.resumed == 2


def test_switching_inner_tabs_stops_previous_page_and_loads_only_selected(platform):
    platform.open_workspace("simulation")
    workspace = platform.simulation_workspace
    first = workspace.pages[0]
    first.running = True
    workspace.tabs.setCurrentIndex(4)
    assert not first.running
    assert set(workspace.pages) == {0, 4}
    assert "波形对比" in platform.status_bar.currentMessage()
    platform.close()
    assert workspace.pages[4].stopped == 1


def test_unavailable_page_can_be_retried_without_losing_workspace(app, monkeypatch):
    attempts = []

    def load(*_):
        attempts.append(True)
        if len(attempts) == 1:
            raise ImportError("missing optional dependency")
        return QWidget()

    monkeypatch.setattr(main_window, "_load_page", load)
    window = main_window.MainWindow()
    try:
        window.open_workspace("hardware")
        failed = window.hardware_workspace.pages[0]
        retry = failed.findChild(QPushButton)
        assert retry.text() == "重新加载"
        retry.click()
        assert len(attempts) == 2
        assert window.hardware_workspace.pages[0] is not failed
        assert window.current_mode == "hardware"
    finally:
        window.close()


def test_failed_shutdown_does_not_disable_signals_of_retained_pages(platform, monkeypatch):
    platform.open_workspace("simulation")
    first = platform.fdidm_tab
    destroyed = []
    first._on_qt_destroyed = lambda: destroyed.append(True)
    platform.open_workspace("hardware")
    later = platform.hardware_workspace.pages[0]
    original_stop = later.suspend_tasks

    def fail_stop():
        raise RuntimeError("device did not stop")

    monkeypatch.setattr(later, "suspend_tasks", fail_stop)
    monkeypatch.setattr(main_window.QMessageBox, "warning", lambda *_: None)
    assert not platform.close()
    assert not destroyed
    assert platform.isVisible()
    monkeypatch.setattr(later, "suspend_tasks", original_stop)
