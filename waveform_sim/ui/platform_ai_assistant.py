"""The platform owns one assistant independently of experiment page lifecycles."""
from __future__ import annotations

from PyQt5.QtCore import QEvent, QObject, Qt, QTimer
from PyQt5.QtWidgets import QApplication, QComboBox, QTabWidget

from .fdidm_ai_analysis import FDIDMAIFloatingButton, PlatformAIAssistantDialog


class PlatformAssistantController(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.dialog = None
        self._shutdown = False
        self.button = FDIDMAIFloatingButton(self.open, window)
        self.button.setToolTip("平台 AI 助手：结果分析与操作指导 (Ctrl+J)")
        self.button.setAccessibleName("平台 AI 助手")
        window.installEventFilter(self)
        self._position_button()

    def snapshot(self):
        from .platform_ai_context import build_platform_context

        return build_platform_context(self.window)

    def _position_button(self):
        if not self.button.user_moved:
            self.button.move(max(0, self.window.width() - self.button.width() - 20),
                             max(0, self.window.height() - self.button.height() - 50))
        self.button.clamp_to_parent()
        self.button.raise_()

    def eventFilter(self, watched, event):
        if (watched is getattr(self, "window", None) and hasattr(self, "button")
                and event.type() in {QEvent.Resize, QEvent.Show}):
            self._position_button()
        return super().eventFilter(watched, event)

    def open(self):
        if self._shutdown:
            return None
        if self.dialog is None:
            self.dialog = PlatformAIAssistantDialog(
                lambda: self.snapshot()["log_text"], self.window,
                chart_context_provider=lambda: self.snapshot()["chart_context"],
                context_provider=self.snapshot)
            # A non-modal tool stays above its owner while allowing the experiment
            # workspace to remain interactive. It does not own any radio task.
            self.dialog.setWindowFlag(Qt.Tool, True)
            self.dialog.setWindowModality(Qt.NonModal)
        self.refresh_context()
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
        return self.dialog

    def refresh_context(self):
        if self._shutdown:
            return
        if self.dialog is not None:
            self.dialog.set_platform_context(self.snapshot())
        self._position_button()

    def bind_page(self, page):
        """Follow subviews without changing any page action or run parameter."""
        if getattr(page, "_platform_ai_bound", False):
            return
        page._platform_ai_bound = True
        legacy = getattr(page, "set_global_ai_assistant", None)
        if legacy is not None:
            legacy(self)
        for tabs in page.findChildren(QTabWidget):
            tabs.currentChanged.connect(lambda *_args: QTimer.singleShot(0, self.refresh_context))
        waveform = getattr(page, "waveform_combo", None)
        if isinstance(waveform, QComboBox):
            waveform.currentIndexChanged.connect(lambda *_args: QTimer.singleShot(0, self.refresh_context))

    def shutdown(self):
        if self._shutdown:
            return
        self._shutdown = True
        if self.dialog is not None:
            self.dialog.suspend_analysis()
            # Its application-wide zoom filter outlives a hidden dialog unless
            # explicitly removed. Detach while both Qt/Python objects are valid.
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self.dialog)
        self.window.removeEventFilter(self)
