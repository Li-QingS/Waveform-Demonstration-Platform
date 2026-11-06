"""Hardware configuration must scroll without cutting off controls or frames."""

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

pytest.importorskip('PyQt5')
pytest.importorskip('pyqtgraph')

from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QFont
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QLineEdit, QPushButton,
    QSpinBox, QTextEdit,
)

from waveform_sim.ui.main_window import MainWindow


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def platform(app):
    original = QFont(app.font())
    app.setFont(QFont('Microsoft YaHei UI', 9))
    window = MainWindow()
    window.resize(1100, 700)
    window.open_workspace('hardware')
    window.show()
    QTest.qWait(40)
    yield window
    window.close()
    app.setFont(original)


def assert_controls_fit(page):
    content = page.controls_scroll.widget()
    viewport = page.controls_scroll.viewport()
    assert content.width() <= viewport.width() + 1, 'Configuration content is wider than its scroll viewport'
    assert page.controls_scroll.horizontalScrollBar().maximum() == 0
    types = (QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QPushButton, QTextEdit)
    for control in content.findChildren(types):
        if not control.isVisibleTo(content):
            continue
        point = control.mapTo(content, QPoint(0, 0))
        assert point.x() >= 0
        assert point.x() + control.width() <= content.width() + 1, control.objectName()
        # Scrolling may hide a field vertically, but must not crush its hit area.
        assert control.height() >= 20


@pytest.mark.parametrize('page_index', [0, 1])
def test_first_entry_and_window_resize_fit_the_current_scroll_viewport(app, page_index):
    original = QFont(app.font())
    app.setFont(QFont('Microsoft YaHei UI', 9))
    window = MainWindow()
    try:
        window.show()
        window.open_workspace('hardware')
        window.hardware_workspace.tabs.setCurrentIndex(page_index)
        page = window.hardware_workspace.pages[page_index]
        for size in ((1400, 900), (1100, 700), (1400, 900)):
            window.resize(*size)
            QTest.qWait(150)
            assert_controls_fit(page)
        assert page.backend is None
    finally:
        window.close()
        app.setFont(original)


@pytest.mark.parametrize('font_size', [9, 11])
def test_fdidm_advanced_controls_fit_the_rail_after_expansion_and_channel_changes(platform, app, font_size):
    page = platform.hardware_workspace.pages[0]
    app.setFont(QFont('Microsoft YaHei UI', font_size))
    page.btn_toggle_advanced.setChecked(True)
    for mode in ('rf', 'rf_tdl_a', 'tdl_a_rf'):
        index = page.channel_mode_combo.findData(mode)
        assert index >= 0
        page.channel_mode_combo.setCurrentIndex(index)
        QTest.qWait(35)
        assert_controls_fit(page)
        page.controls_scroll.verticalScrollBar().setValue(page.controls_scroll.verticalScrollBar().maximum())
        QTest.qWait(20)
        # Commands are outside the scrolling parameter region.
        for button in (page.btn_connect, page.btn_start_test, page.btn_stop_test):
            assert not button.visibleRegion().isEmpty()
    page.btn_toggle_advanced.setChecked(False)
    QTest.qWait(30)
    assert_controls_fit(page)
    assert page.backend is None


@pytest.mark.parametrize('waveform', ['OFDM', 'OTFS', 'AFDM'])
def test_three_waveform_configuration_tabs_do_not_clip_fields_or_compress_text_editor(platform, waveform):
    platform.hardware_workspace.tabs.setCurrentIndex(1)
    page = platform.hardware_workspace.pages[1]
    page.waveform_combo.setCurrentText(waveform)
    for index in range(page.config_tabs.count()):
        page.config_tabs.setCurrentIndex(index)
        QTest.qWait(35)
        assert_controls_fit(page)
    page.config_tabs.setCurrentIndex(1)
    QTest.qWait(35)
    assert page.tx_text_edit.height() >= 60
    if page.controls_scroll.widget().minimumSizeHint().height() > page.controls_scroll.viewport().height():
        assert page.controls_scroll.verticalScrollBar().maximum() > 0
    for button in (page.btn_connect, page.btn_apply, page.btn_start_test, page.btn_stop_test, page.btn_disconnect):
        assert not button.visibleRegion().isEmpty()
    assert page.backend is None


def test_long_hardware_status_and_text_evidence_keep_usable_charts(platform):
    platform.hardware_workspace.tabs.setCurrentIndex(1)
    page = platform.hardware_workspace.pages[1]
    message = '连接诊断：设备序列号、采样率、同步结果和当前信道状态需要完整记录。' * 10
    page.decode_status_label.setText(message)
    page.tx_text_view.setPlainText('发送文本' * 300)
    page.rx_text_view.setPlainText('接收文本' * 300)
    for index in range(page.result_tabs.count()):
        page.result_tabs.setCurrentIndex(index)
        QTest.qWait(40)
        assert platform.size().width() == 1100 and platform.size().height() == 700
        for plot in (page.tx_plot, page.rx_spectrum_plot, page.ber_plot, page.constellation_plot):
            assert plot.width() >= 220 and plot.height() >= 120
        assert page.result_tabs.height() >= 100
        assert_controls_fit(page)
