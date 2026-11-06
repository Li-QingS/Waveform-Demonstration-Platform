"""Entry page and task-oriented workspaces for the desktop platform."""

import importlib
import logging

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAction, QApplication, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QStackedWidget, QStatusBar, QTabWidget, QVBoxLayout, QWidget, QMainWindow,
)

from .landing_page import LandingPage

LOGGER = logging.getLogger(__name__)

SIMULATION_PAGES = (
    ('waveform_sim.ui.fdidm_tab', 'FDIDMTab', 'FDIDM 软波形'),
    ('waveform_sim.ui.ofdm_tab', 'OfdmTab', 'OFDM'),
    ('waveform_sim.ui.otfs_tab', 'OTFSTab', 'OTFS'),
    ('waveform_sim.ui.afdm_tab', 'AfdmTab', 'AFDM'),
    ('waveform_sim.ui.waveform_compare_tab', 'WaveformCompareTab', '波形对比'),
)
HARDWARE_PAGES = (
    ('waveform_sim.ui.fdidm_hardware_test_tab', 'FDIDMHardwareTestTab', 'FDIDM 自适应验证'),
    ('waveform_sim.ui.hardware_test_tab', 'HardwareTestTab', '三波形链路测评'),
)


def _load_page(module_name, class_name):
    return getattr(importlib.import_module(module_name), class_name)()


class Workspace(QWidget):
    """Load a page only when visited, retaining its configuration on return."""

    home_requested = pyqtSignal()
    switch_requested = pyqtSignal()
    page_changed = pyqtSignal(str)

    def __init__(self, title, description, switch_title, specs, parent=None):
        super().__init__(parent)
        self.title = title
        self.specs = specs
        self.pages = {}
        self._active = False
        self._current_page = None
        self._current_index = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        header = QHBoxLayout()
        self.home_button = QPushButton('返回首页')
        self.home_button.setToolTip('停止当前任务并返回首页，保留参数与结果')
        self.home_button.clicked.connect(self.home_requested)
        header.addWidget(self.home_button)
        heading = QLabel(title)
        font = heading.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        heading.setFont(font)
        header.addWidget(heading)
        header.addWidget(QLabel(description))
        header.addStretch(1)
        self.switch_button = QPushButton(switch_title)
        self.switch_button.setToolTip('停止当前任务并切换工作区，保留参数与结果')
        self.switch_button.clicked.connect(self.switch_requested)
        header.addWidget(self.switch_button)
        layout.addLayout(header)
        self.tabs = QTabWidget()
        self.tabs.setMovable(False)
        self.tabs.setTabsClosable(False)
        self.tabs.setUsesScrollButtons(True)
        for _, _, label in specs:
            host = QWidget()
            body = QVBoxLayout(host)
            body.setContentsMargins(0, 0, 0, 0)
            self.tabs.addTab(host, label)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tabs, 1)

    def _ensure_page(self, index):
        if index in self.pages:
            return self.pages[index]
        module_name, class_name, label = self.specs[index]
        host = self.tabs.widget(index)
        try:
            page = _load_page(module_name, class_name)
        except Exception as exc:
            LOGGER.exception('Could not load %s', class_name)
            page = QWidget()
            body = QVBoxLayout(page)
            body.addStretch(1)
            message = QLabel(f'{label}暂时无法加载\n{type(exc).__name__}: {exc}')
            message.setWordWrap(True)
            message.setAlignment(Qt.AlignCenter)
            message.setTextInteractionFlags(Qt.TextSelectableByMouse)
            body.addWidget(message)
            retry = QPushButton('重新加载')
            retry.clicked.connect(lambda: self._retry_page(index))
            body.addWidget(retry, alignment=Qt.AlignHCenter)
            body.addStretch(1)
        host.layout().addWidget(page)
        self.pages[index] = page
        assistant = getattr(self, "_assistant_controller", None)
        if assistant is not None:
            assistant.bind_page(page)
        return page

    def _retry_page(self, index):
        old = self.pages.pop(index)
        self.tabs.widget(index).layout().removeWidget(old)
        old.deleteLater()
        self._current_page = self._ensure_page(index)
        if self._active:
            self._resume_page(self._current_page)

    @staticmethod
    def _resume_page(page):
        resume = getattr(page, 'resume_tasks', None)
        if resume is not None:
            resume()

    def activate(self):
        self._active = False
        self._current_index = self.tabs.currentIndex()
        self._current_page = self._ensure_page(self._current_index)
        self._resume_page(self._current_page)
        self._active = True
        self.page_changed.emit(self.tabs.tabText(self._current_index))

    def suspend(self):
        if self._current_page is not None:
            suspend = getattr(self._current_page, 'suspend_tasks', None)
            if suspend is not None:
                suspend()
        self._active = False

    def _on_tab_changed(self, index):
        if not self._active:
            return
        previous_index = self._current_index
        try:
            self.suspend()
        except Exception:
            self.tabs.blockSignals(True)
            self.tabs.setCurrentIndex(self._current_index)
            self.tabs.blockSignals(False)
            self._active = True
            LOGGER.exception('Could not stop the current page')
            QMessageBox.warning(self, '任务尚未停止', '当前任务停止失败，请先停止任务后再切换页面。')
            return
        try:
            self.activate()
        except Exception:
            LOGGER.exception('Could not activate the selected page')
            self.suspend()
            self.tabs.blockSignals(True)
            self.tabs.setCurrentIndex(previous_index)
            self.tabs.blockSignals(False)
            self.activate()
            QMessageBox.warning(self, '页面未就绪', '页面恢复失败，已返回之前的页面。')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('FDIDM软波形自适应演示平台 v6.0')
        self.setGeometry(80, 60, 1400, 900)
        self.setMinimumSize(1100, 700)
        self.current_mode = None
        self._init_ui()
        self._init_menu()
        self._init_ai_assistant()

    def _init_ai_assistant(self):
        from .platform_ai_assistant import PlatformAssistantController

        self.ai_assistant = PlatformAssistantController(self)
        for workspace in self.workspaces.values():
            workspace._assistant_controller = self.ai_assistant
        action = QAction("AI 助手", self)
        action.setShortcut("Ctrl+J")
        action.triggered.connect(self.ai_assistant.open)
        self.menuBar().addAction(action)
        self.ai_assistant_action = action

    def _init_ui(self):
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.landing_page = LandingPage()
        self.stack.addWidget(self.landing_page)
        self.simulation_workspace = Workspace(
            '仿真', '配置参数 → 运行仿真 → 分析结果', '切换至硬件验证', SIMULATION_PAGES
        )
        self.hardware_workspace = Workspace(
            '硬件验证', '配置链路 → 启动验证 → 观测与导出', '切换至仿真', HARDWARE_PAGES
        )
        self.workspaces = {
            'simulation': self.simulation_workspace,
            'hardware': self.hardware_workspace,
        }
        for workspace in self.workspaces.values():
            self.stack.addWidget(workspace)
            workspace.home_requested.connect(self.show_home)
            workspace.page_changed.connect(self._on_page_changed)
        self.simulation_workspace.switch_requested.connect(lambda: self.open_workspace('hardware'))
        self.hardware_workspace.switch_requested.connect(lambda: self.open_workspace('simulation'))
        self.landing_page.simulation_requested.connect(lambda: self.open_workspace('simulation'))
        self.landing_page.hardware_requested.connect(lambda: self.open_workspace('hardware'))
        self.tabs = self.simulation_workspace.tabs
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('系统就绪：请选择仿真或硬件验证')

    def _init_menu(self):
        file_menu = self.menuBar().addMenu('文件(&F)')
        home = QAction('返回首页', self)
        home.setShortcut('Alt+Home')
        home.triggered.connect(self.show_home)
        file_menu.addAction(home)
        file_menu.addSeparator()
        quit_action = QAction('退出', self)
        quit_action.setShortcut('Ctrl+Q')
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        view_menu = self.menuBar().addMenu('视图(&V)')
        for title, mode in (('仿真', 'simulation'), ('硬件验证', 'hardware')):
            action = QAction(title, self)
            action.triggered.connect(lambda checked=False, key=mode: self.open_workspace(key))
            view_menu.addAction(action)
        help_menu = self.menuBar().addMenu('帮助(&H)')
        about = QAction('关于平台', self)
        about.triggered.connect(lambda: QMessageBox.about(
            self, '关于平台', 'FDIDM软波形自适应演示平台 v6.0\n\n'
            '仿真：波形仿真、自适应分析与性能对比。\n'
            '硬件验证：USRP 链路验证、实时观测与结果导出。\n\n'
            '切换页面或返回首页时停止当前任务，保留配置与结果。'
        ))
        help_menu.addAction(about)

    def _suspend_current(self):
        if self.current_mode is not None:
            self.workspaces[self.current_mode].suspend()

    def open_workspace(self, mode):
        if mode not in self.workspaces:
            raise ValueError(f'Unknown workspace: {mode}')
        if mode == self.current_mode:
            return
        previous_mode = self.current_mode
        try:
            self._suspend_current()
        except Exception:
            LOGGER.exception('Could not leave the current workspace')
            QMessageBox.warning(self, '任务尚未停止', '当前任务停止失败，请先停止任务后再切换工作区。')
            return
        self.current_mode = mode
        workspace = self.workspaces[mode]
        self.stack.setCurrentWidget(workspace)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            workspace.activate()
        except Exception:
            LOGGER.exception('Could not activate the selected workspace')
            workspace.suspend()
            self.current_mode = previous_mode
            if previous_mode is None:
                self.stack.setCurrentWidget(self.landing_page)
                self.status_bar.showMessage('页面未就绪，请重试进入工作区')
            else:
                previous = self.workspaces[previous_mode]
                self.stack.setCurrentWidget(previous)
                previous.activate()
            QMessageBox.warning(self, '页面未就绪', '工作区恢复失败，已返回之前的页面。')
        finally:
            QApplication.restoreOverrideCursor()
        self.ai_assistant.refresh_context()

    def show_home(self):
        try:
            self._suspend_current()
        except Exception:
            LOGGER.exception('Could not return to the entry page')
            QMessageBox.warning(self, '任务尚未停止', '当前任务停止失败，请先停止任务后再返回首页。')
            return
        self.current_mode = None
        self.stack.setCurrentWidget(self.landing_page)
        self.status_bar.showMessage('系统就绪：请选择仿真或硬件验证；参数与结果已保留')
        self.ai_assistant.refresh_context()

    def _on_page_changed(self, title):
        if self.current_mode is not None:
            mode_title = self.workspaces[self.current_mode].title
            self.status_bar.showMessage(f'{mode_title} / {title}  ·  离开页面时停止任务并保留结果')
        assistant = getattr(self, "ai_assistant", None)
        if assistant is not None:
            assistant.refresh_context()

    @property
    def fdidm_tab(self):
        return self.simulation_workspace.pages.get(0)

    def closeEvent(self, event):
        try:
            for workspace in self.workspaces.values():
                for page in workspace.pages.values():
                    suspend = getattr(page, 'suspend_tasks', None)
                    if suspend is not None:
                        suspend()
        except Exception:
            LOGGER.exception('Could not stop platform tasks on exit')
            QMessageBox.warning(self, '任务尚未停止', '当前任务停止失败，请先停止任务后再退出平台。')
            event.ignore()
            return
        # Mark objects as destroyed only once all tasks have stopped.  If one
        # stop fails, the still-open window must retain working Qt signals.
        for workspace in self.workspaces.values():
            for page in workspace.pages.values():
                destroy = getattr(page, '_on_qt_destroyed', None)
                if destroy is not None:
                    destroy()
        self.ai_assistant.shutdown()
        super().closeEvent(event)
