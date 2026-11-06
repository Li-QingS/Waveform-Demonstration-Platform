"""Real-page navigation must stop work without losing results or radio handles."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from waveform_sim.ui.fdidm_hardware_test_tab import FDIDMHardwareTestTab
from waveform_sim.ui.hardware_test_tab import HardwareTestTab
from waveform_sim.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def platform(app, monkeypatch):
    def forbid_device_creation(*_args, **_kwargs):
        raise AssertionError("Lifecycle tests must never open a radio")

    for page_class in (FDIDMHardwareTestTab, HardwareTestTab):
        monkeypatch.setattr(page_class, "_create_backend", forbid_device_creation)
    window = MainWindow()
    yield window
    # Also permit deterministic cleanup after an assertion in a failure case.
    for workspace in window.workspaces.values():
        for page in workspace.pages.values():
            backend = getattr(page, "backend", None)
            if isinstance(backend, FakeBackend):
                backend.fail_stop = False
    window.close()


class FakeBackend:
    def __init__(self, *, fail_stop=False):
        self.fail_stop = fail_stop
        self.calls = []

    def stop(self):
        self.calls.append("stop")
        if self.fail_stop:
            raise RuntimeError("radio stop failed")

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))


def test_fdidm_initial_refresh_cannot_restart_after_returning_home(platform, monkeypatch):
    assert not platform.simulation_workspace.pages
    platform.open_workspace("simulation")
    page = platform.fdidm_tab
    assert page._initial_refresh_timer.isActive()
    scheduled = []
    searches = []
    original_schedule = page._schedule_auto_ser_snr

    def record_schedule(reason="parameter update"):
        scheduled.append(reason)
        return original_schedule(reason)

    monkeypatch.setattr(page, "_schedule_auto_ser_snr", record_schedule)
    monkeypatch.setattr(page, "_search_worker", lambda *_args: searches.append(True))

    platform.show_home()
    QTest.qWait(500)  # Cross the original 400 ms initialization deadline.
    assert platform.current_mode is None
    assert not page._initial_refresh_timer.isActive()
    assert not page._auto_refresh_timer.isActive()
    assert not page.adaptive_timer.isActive()
    assert not scheduled and not searches

    platform.open_workspace("simulation")
    assert platform.fdidm_tab is page
    assert page.adaptive_timer.isActive()
    assert page._auto_refresh_timer.isActive()
    assert scheduled == ["返回仿真页面"]
    assert page.tb is None  # Re-entry restores the UI, not a running link.


def test_ofdm_navigation_stops_backend_and_keeps_parameters_and_curves(platform):
    workspace = platform.simulation_workspace
    workspace.tabs.setCurrentIndex(1)
    platform.open_workspace("simulation")
    assert set(workspace.pages) == {1}
    page = workspace.pages[1]
    page.snr_spin.setValue(23.0)
    page.mod_combo.setCurrentText("16QAM")
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([12.0, 8.0, 6.0])
    page.ber_curve.setData(x, y)
    backend = FakeBackend()
    page.tb = backend
    page._set_ui_enabled(False)
    page.update_timer.start(60_000)

    platform.show_home()
    assert backend.calls == ["stop", ("wait", 1.5)]
    assert page.tb is None
    assert not page.update_timer.isActive()
    assert page.btn_start.isEnabled() and not page.btn_stop.isEnabled()
    assert page.snr_spin.value() == 23.0
    assert page.mod_combo.currentText() == "16QAM"
    np.testing.assert_array_equal(page.ber_curve.xData, x)
    np.testing.assert_array_equal(page.ber_curve.yData, y)

    platform.open_workspace("simulation")
    assert workspace.pages[1] is page
    assert page.tb is None and not page.update_timer.isActive()
    np.testing.assert_array_equal(page.ber_curve.yData, y)


@pytest.mark.parametrize("page_index", [0, 1], ids=["fdidm", "multi_waveform"])
def test_radio_stop_failure_blocks_navigation_and_retains_backend(
    platform, monkeypatch, page_index
):
    warnings = []
    monkeypatch.setattr(
        "waveform_sim.ui.main_window.QMessageBox.warning",
        lambda *_args: warnings.append(True),
    )
    workspace = platform.hardware_workspace
    workspace.tabs.setCurrentIndex(page_index)
    platform.open_workspace("hardware")
    assert set(workspace.pages) == {page_index}
    page = workspace.pages[page_index]
    backend = FakeBackend(fail_stop=True)
    page.backend = backend
    page.test_running = True
    page.update_timer.start(60_000)
    page.rx_curve.setData([0.0, 1.0], [-20.0, -15.0])

    with pytest.raises(RuntimeError, match="radio stop failed"):
        page.suspend_tasks()
    assert page.backend is backend
    assert page.test_running and page.update_timer.isActive()

    platform.show_home()
    assert platform.current_mode == "hardware"
    assert platform.stack.currentWidget() is workspace
    assert page.backend is backend and warnings
    assert not any(isinstance(call, tuple) for call in backend.calls)

    backend.fail_stop = False
    platform.show_home()
    assert platform.current_mode is None
    assert page.backend is None
    assert not page.test_running and not page.update_timer.isActive()
    assert backend.calls[-2:] == ["stop", ("wait", None)]
    np.testing.assert_array_equal(page.rx_curve.yData, [-20.0, -15.0])
