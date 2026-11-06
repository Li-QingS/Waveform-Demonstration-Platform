"""FDIDM page integration for the AI log analysis entry; no USRP is needed."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from waveform_sim.ui.fdidm_hardware_test_tab import FDIDMHardwareTestTab


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(app):
    widget = FDIDMHardwareTestTab()
    widget.resize(1100, 700)
    widget.show()
    QTest.qWait(20)
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_floating_entry_opens_single_analysis_window_and_tracks_resize(page, app):
    button = page.ai_floating_button
    assert button.isVisible()
    assert page.rect().contains(button.geometry())
    page._log("测试运行：同步正常，dataEVM=12.5%")
    QTest.mouseClick(button, Qt.LeftButton)
    app.processEvents()
    dialog = page._ai_analysis_dialog
    assert dialog.isVisible()
    assert "dataEVM=12.5%" in dialog._log_text
    dialog.close()
    QTest.mouseClick(button, Qt.LeftButton)
    assert page._ai_analysis_dialog is dialog
    assert dialog.isVisible()
    dialog.close()
    for size in ((1400, 900), (1100, 700)):
        page.resize(*size)
        app.processEvents()
        assert page.rect().contains(button.geometry())


class LogBackend:
    def __init__(self):
        self.stopped = False

    def get_debug_log(self, max_entries, min_level):
        assert max_entries == 5000
        assert min_level == "DEBUG"
        return [{"seq": 7, "t": 1.25, "level": "DEBUG", "msg": "pilot_fit_NMSE=0.012"}]

    def stop(self):
        self.stopped = True

    def wait(self):
        pass


def test_backend_debug_snapshot_survives_release_and_page_hides_analysis(page):
    backend = LogBackend()
    page.backend = backend
    page._log("TDL_injected_SNR=35.0dB")
    before = page._get_ai_log_snapshot()
    assert "pilot_fit_NMSE=0.012" in before
    assert "TDL_injected_SNR=35.0dB" in before
    page._open_ai_analysis()
    page.suspend_tasks()
    assert backend.stopped
    assert page.backend is None
    assert not page._ai_analysis_dialog.isVisible()
    after = page._get_ai_log_snapshot()
    assert "pilot_fit_NMSE=0.012" in after
    assert "TDL_injected_SNR=35.0dB" in after


def test_snapshot_skips_backend_echoes_and_does_not_reuse_old_backend_data(page):
    page.backend = LogBackend()
    page._log("BE[0007 1.250s DEBUG] pilot_fit_NMSE=0.012")
    snapshot = page._get_ai_log_snapshot()
    assert snapshot.count("pilot_fit_NMSE=0.012") == 1

    class FreshBackend(LogBackend):
        def get_debug_log(self, **kwargs):
            return []

    page.backend = FreshBackend()
    # The new backend's empty log must not be supplemented by the old cache.
    page._ui_log_entries.clear()
    assert "pilot_fit_NMSE=0.012" not in page._get_ai_log_snapshot()


def test_assistant_receives_current_chart_context_from_the_page(page):
    page.evm_curve.setData([0, 1], [25, 12])
    page._open_ai_analysis()
    context = page._ai_analysis_dialog._chart_context_provider()
    assert context["has_data"]
    evm = next(item for item in context["charts"] if item["title"] == "EVM 曲线")
    assert evm["series"][0]["y_last"] == 12
