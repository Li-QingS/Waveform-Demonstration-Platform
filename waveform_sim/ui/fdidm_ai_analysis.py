"""Chat-oriented FDIDM AI assistant dialog and floating entry button."""
from __future__ import annotations

import html
import math
import threading
import weakref
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PyQt5.QtCore import QEvent, QObject, QPoint, QPointF, QRectF, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import (
    QColor, QFont, QFontDatabase, QFontMetrics, QLinearGradient, QPainter,
    QPainterPath, QPen, QRadialGradient, QTextBlockFormat, QTextCharFormat, QTextCursor,
)
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from waveform_sim.service import ai_log_analyzer


MAX_LOG_FILE_BYTES = 16 * 1024 * 1024
CHAT_INTENTS_WITH_LOGS = {"logs", "diagnosis", "suggestions"}
QUICK_ACTIONS = (
    ("日志分析", "logs"),
    ("运行结果图解释", "charts"),
    ("异常原因排查", "diagnosis"),
    ("优化建议", "suggestions"),
    ("操作指导", "guide"),
)


def _preferred_ui_font_family() -> str:
    available = set(QFontDatabase().families())
    for candidate in ("Microsoft YaHei UI", "Segoe UI", "Noto Sans CJK SC", "Noto Sans CJK", "Sans Serif"):
        if candidate in available:
            return candidate
    return QFont().defaultFamily()


@dataclass(frozen=True)
class _Typography:
    percent: int = 100
    family: str = ""
    dpi_scale: float = 1.0

    def __post_init__(self):
        if not self.family:
            object.__setattr__(self, "family", _preferred_ui_font_family())

    def px(self, base: int) -> int:
        return max(1, round(base * self.percent / 100 * self.dpi_scale))

    def font(self, base_px: int, weight: int = QFont.Normal) -> QFont:
        font = QFont(self.family)
        font.setStyleHint(QFont.SansSerif)
        font.setPixelSize(self.px(base_px))
        font.setWeight(weight)
        return font


def _format_markdown_document(document, typography: _Typography) -> None:
    """Keep Markdown semantics while applying a scale-aware document baseline."""
    document.setDefaultFont(typography.font(16))
    document.setDefaultStyleSheet(
        "body { line-height: 1.48; } "
        f"p {{ margin-top: {typography.px(4)}px; margin-bottom: {typography.px(6)}px; }} "
        f"li {{ margin-top: {typography.px(2)}px; margin-bottom: {typography.px(2)}px; }} "
        "pre, code { font-family: Consolas, 'Cascadia Mono', monospace; }"
    )
    cursor = QTextCursor(document)
    block = document.begin()
    while block.isValid():
        block_format = block.blockFormat()
        block_format.setLineHeight(148, QTextBlockFormat.ProportionalHeight)
        block_format.setTopMargin(typography.px(2))
        block_format.setBottomMargin(typography.px(6))
        cursor.setPosition(block.position())
        cursor.select(QTextCursor.BlockUnderCursor)
        cursor.setBlockFormat(block_format)
        block = block.next()

    block = document.begin()
    monospace_families = {"courier", "courier new", "consolas", "cascadia mono"}
    code_ranges = []
    while block.isValid():
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            char_format = fragment.charFormat()
            explicit_family = char_format.fontFamily()
            family = (explicit_family or char_format.font().family() or "").lower()
            if char_format.fontFixedPitch() or family in monospace_families or explicit_family == "Helvetica":
                code_ranges.append((fragment.position(), fragment.length(), char_format))
            iterator += 1
        block = block.next()

    for position, length, char_format in code_ranges:
        code_format = QTextCharFormat(char_format)
        code_font = char_format.font()
        code_font.setFamily(char_format.fontFamily() or code_font.family() or "Consolas")
        code_font.setPixelSize(typography.px(15))
        code_format.setFont(code_font)
        code_format.setFontFixedPitch(True)
        cursor.setPosition(position)
        cursor.setPosition(position + length, QTextCursor.KeepAnchor)
        cursor.setCharFormat(code_format)


class _AssistantBubble(QTextBrowser):
    """A markdown bubble that follows its actual document height as width changes."""

    def __init__(self, content: str, typography: _Typography, parent=None):
        super().__init__(parent)
        self._fit_queued = False
        self._fitting = False
        self._typography = typography
        self.setOpenExternalLinks(False)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setReadOnly(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFont(typography.font(16))
        self.document().setDefaultFont(typography.font(16))
        try:
            self.setMarkdown(content)
        except AttributeError:
            self.setPlainText(content)
        self.set_typography(typography)
        self.document().documentLayout().documentSizeChanged.connect(self._queue_fit)
        self._queue_fit()

    def set_typography(self, typography: _Typography) -> None:
        self._typography = typography
        self.setFont(typography.font(16))
        self.setMinimumWidth(typography.px(180))
        self.setMaximumWidth(typography.px(680))
        self.setStyleSheet(
            "QTextBrowser { background: #ffffff; color: #1a354d; border: 1px solid #d6e3ed; "
            f"border-radius: {typography.px(12)}px; padding: {typography.px(6)}px {typography.px(8)}px; "
            f"font-size: {typography.px(16)}px; }}"
        )
        _format_markdown_document(self.document(), typography)
        self._queue_fit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._queue_fit()

    def _queue_fit(self, *_args) -> None:
        if self._fitting:
            return
        if self._fit_queued:
            return
        self._fit_queued = True
        QTimer.singleShot(0, self._fit_to_document)

    def _fit_to_document(self) -> None:
        self._fit_queued = False
        self._fitting = True
        try:
            width = max(120, self.viewport().width())
            if abs(self.document().textWidth() - width) > 0.5:
                self.document().setTextWidth(width)
            document_height = math.ceil(self.document().size().height())
            target_height = max(self._typography.px(52), document_height + self._typography.px(24))
            if self.height() != target_height:
                self.setFixedHeight(target_height)
        finally:
            self._fitting = False


class _UserBubble(QLabel):
    def __init__(self, content: str, typography: _Typography, parent=None):
        super().__init__(parent)
        self._content = content
        self.setTextFormat(Qt.RichText)
        self.setWordWrap(True)
        self.setText(html.escape(content).replace("\n", "<br>"))
        self.set_typography(typography)

    def set_typography(self, typography: _Typography) -> None:
        self.setFont(typography.font(16))
        self.setMaximumWidth(typography.px(560))
        self.setStyleSheet(
            "QLabel { background: #dcedfa; color: #173d5e; border: 1px solid #bfd8ed; "
            f"border-radius: {typography.px(12)}px; padding: {typography.px(9)}px {typography.px(11)}px; "
            f"font-size: {typography.px(16)}px; }}"
        )


class _ChatInput(QPlainTextEdit):
    send_requested = pyqtSignal()

    def keyPressEvent(self, event) -> None:
        if event.key() in {Qt.Key_Return, Qt.Key_Enter} and not event.modifiers() & Qt.ShiftModifier:
            event.accept()
            self.send_requested.emit()
            return
        super().keyPressEvent(event)


class FDIDMAIAnalysisWorker(QObject):
    """Execute one chat request without blocking the Qt event loop."""

    succeeded = pyqtSignal(str)
    failed = pyqtSignal(str)
    completed = pyqtSignal()

    def __init__(self, message, config, *, log_text, chart_context, platform_context, history, intent):
        super().__init__()
        self.message = message
        self.config = config
        self.log_text = log_text
        self.chart_context = chart_context
        self.platform_context = platform_context
        self.history = history
        self.intent = intent
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            answer = ai_log_analyzer.chat_with_assistant(
                self.message,
                self.config,
                log_text=self.log_text,
                chart_context=self.chart_context,
                platform_context=self.platform_context,
                history=self.history,
                intent=self.intent,
            )
        except Exception as exc:
            if not self._cancel_event.is_set():
                self.failed.emit(str(exc))
        else:
            if not self._cancel_event.is_set():
                self.succeeded.emit(str(answer))
        finally:
            self.completed.emit()


class FDIDMAIAnalysisDialog(QDialog):
    """A chat UI that keeps logs and backend configuration private to requests."""

    def __init__(
        self,
        log_provider: Callable[[], str],
        parent=None,
        *,
        config_provider=None,
        chart_context_provider=None,
        context_provider=None,
    ):
        super().__init__(parent)
        self._log_provider = log_provider
        if config_provider is None:
            from waveform_sim.service.ai_config import load_ai_config

            config_provider = load_ai_config
        self._config_provider = config_provider
        self._chart_context_provider = chart_context_provider or (lambda: {})
        self._context_provider = context_provider
        self._platform_context = {}
        self._context_id = "fdidm"
        self._log_text = ""
        self._source_description = "当前测试"
        self._source_kind = "none"
        self._request_token = 0
        self._analysis_active = False
        self._worker = None
        self._worker_thread = None
        self._pending = None
        self._history = []
        self._chart_followups = False
        self._conversation_mode = None
        self._redaction_keys = set()
        self._private_contexts = deque(maxlen=12)
        self._auto_scroll_pending = False
        self.messages = []
        self._message_widgets = []
        screen = self.screen() or QApplication.primaryScreen()
        dpi_scale = max(0.8, float(screen.logicalDotsPerInch()) / 96.0) if screen is not None else 1.0
        self._typography = _Typography(dpi_scale=dpi_scale)
        self._ui_scale_percent = 100

        self.setWindowTitle("平台 AI 助手")
        self.resize(self._typography.px(900), self._typography.px(820))
        self.setMinimumSize(self._typography.px(560), self._typography.px(500))
        self._build_ui()
        self._reset_conversation()
        QApplication.instance().installEventFilter(self)

    def _build_ui(self) -> None:
        self.setStyleSheet("QDialog { background: #edf3f9; color: #18324a; }")
        layout = QVBoxLayout(self)
        self._main_layout = layout
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        header = QFrame()
        self._header_card = header
        header.setObjectName("assistantHeader")
        header.setStyleSheet(
            "QFrame#assistantHeader { background: #ffffff; border: 1px solid #d9e5f0; border-radius: 12px; }"
        )
        header_layout = QHBoxLayout(header)
        self._header_layout = header_layout
        header_layout.setContentsMargins(16, 13, 16, 13)
        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        self.header_title = QLabel("AI 助手")
        self.header_title.setObjectName("assistantTitle")
        self.header_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #163a5f;")
        self.header_subtitle = QLabel("实验诊断、结果解读与操作指导")
        self.header_subtitle.setObjectName("assistantSubtitle")
        self.header_subtitle.setStyleSheet("font-size: 12px; color: #637b91;")
        title_layout.addWidget(self.header_title)
        title_layout.addWidget(self.header_subtitle)
        header_layout.addLayout(title_layout)
        header_layout.addStretch(1)
        self.zoom_out_button = QPushButton("A−")
        self.zoom_out_button.setObjectName("zoomOutButton")
        self.zoom_reset_button = QPushButton("100%")
        self.zoom_reset_button.setObjectName("zoomResetButton")
        self.zoom_in_button = QPushButton("A+")
        self.zoom_in_button.setObjectName("zoomInButton")
        for button in (self.zoom_out_button, self.zoom_reset_button, self.zoom_in_button):
            button.setToolTip("调整助手界面字号")
            header_layout.addWidget(button)
        self.btn_clear = QPushButton("清空对话")
        self.btn_clear.setObjectName("clearConversationButton")
        self.btn_clear.setStyleSheet(
            "QPushButton { color: #46627e; background: #f5f8fb; border: 1px solid #d7e2ec; border-radius: 7px; padding: 7px 10px; }"
            "QPushButton:hover { background: #eaf2f9; }"
        )
        header_layout.addWidget(self.btn_clear)
        layout.addWidget(header)

        quick_card = QFrame()
        self._quick_card = quick_card
        quick_card.setObjectName("quickActionsCard")
        quick_card.setStyleSheet(
            "QFrame#quickActionsCard { background: #ffffff; border: 1px solid #d9e5f0; border-radius: 12px; }"
        )
        quick_layout = QGridLayout(quick_card)
        self._quick_layout = quick_layout
        quick_layout.setContentsMargins(12, 12, 12, 12)
        quick_layout.setHorizontalSpacing(10)
        quick_layout.setVerticalSpacing(10)
        self.quick_buttons = []
        for index, (label, intent) in enumerate(QUICK_ACTIONS):
            button = QPushButton(label)
            button.setObjectName(f"quickAction{index}")
            button.setMinimumHeight(38)
            button.setStyleSheet(
                "QPushButton { text-align: left; padding: 9px 12px; background: #f4f8fc; color: #1d4f7a; "
                "border: 1px solid #d9e7f2; border-radius: 8px; font-weight: 600; }"
                "QPushButton:hover { background: #e7f1fa; border-color: #a9c9e3; }"
            )
            button.clicked.connect(
                lambda _checked=False, message=label, request_intent=intent: self.send_message(message, request_intent)
            )
            quick_layout.addWidget(button, index // 3, index % 3)
            self.quick_buttons.append(button)
        layout.addWidget(quick_card)

        source_row = QHBoxLayout()
        self._source_row = source_row
        source_row.setSpacing(8)
        self.source_label = QLabel("来源：当前测试")
        self.source_label.setObjectName("sourceLabel")
        self.source_label.setWordWrap(False)
        self.source_label.setMinimumWidth(0)
        self.source_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.btn_current_log = QPushButton("当前测试")
        self.btn_current_log.setObjectName("currentLogButton")
        self.btn_open_log = QPushButton("选择历史记录")
        self.btn_open_log.setObjectName("openLogButton")
        for button in (self.btn_current_log, self.btn_open_log):
            button.setStyleSheet(
                "QPushButton { color: #2b618f; background: transparent; border: none; padding: 4px 6px; }"
                "QPushButton:hover { color: #123f68; text-decoration: underline; }"
            )
        source_row.addWidget(self.source_label, 1)
        source_row.addWidget(self.btn_current_log)
        source_row.addWidget(self.btn_open_log)
        layout.addLayout(source_row)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setObjectName("chatScroll")
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setFrameShape(QFrame.NoFrame)
        self.chat_scroll.setStyleSheet(
            "QScrollArea { background: transparent; } QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        self.chat_scroll.viewport().setStyleSheet("background: transparent;")
        self.chat_widget = QWidget()
        self.chat_widget.setObjectName("chatMessages")
        self.chat_widget.setStyleSheet("background: transparent;")
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.setContentsMargins(2, 4, 8, 4)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch(1)
        self.chat_scroll.setWidget(self.chat_widget)
        self._auto_scroll_timer = QTimer(self)
        self._auto_scroll_timer.setSingleShot(True)
        self._auto_scroll_timer.timeout.connect(self._release_auto_scroll)
        chat_bar = self.chat_scroll.verticalScrollBar()
        chat_bar.rangeChanged.connect(self._on_chat_range_changed)
        chat_bar.sliderPressed.connect(self._release_auto_scroll)
        layout.addWidget(self.chat_scroll, 1)

        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("statusLabel")

        composer = QFrame()
        self._composer = composer
        composer.setObjectName("chatComposer")
        composer.setStyleSheet(
            "QFrame#chatComposer { background: #ffffff; border: 1px solid #ceddea; border-radius: 12px; }"
        )
        composer_layout = QHBoxLayout(composer)
        self._composer_layout = composer_layout
        composer_layout.setContentsMargins(10, 9, 10, 9)
        composer_layout.setSpacing(10)
        self.message_input = _ChatInput()
        self.message_input.setObjectName("messageInput")
        self.message_input.setPlaceholderText("输入问题，Enter 发送，Shift+Enter 换行")
        self.btn_send = QPushButton("发送")
        self.btn_send.setObjectName("sendButton")
        composer_layout.addWidget(self.message_input, 1)
        composer_layout.addWidget(self.btn_send, 0, Qt.AlignBottom)
        layout.addWidget(composer)

        footer = QHBoxLayout()
        self._footer_layout = footer
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setObjectName("cancelButton")
        self.btn_cancel.setEnabled(False)
        self.btn_copy_report = QPushButton("复制对话")
        self.btn_copy_report.setObjectName("copyReportButton")
        self.btn_save_report = QPushButton("保存对话")
        self.btn_save_report.setObjectName("saveReportButton")
        for button in (self.btn_cancel, self.btn_copy_report, self.btn_save_report):
            button.setStyleSheet(
                "QPushButton { background: #ffffff; color: #45647e; border: 1px solid #d0deea; border-radius: 7px; padding: 7px 10px; }"
                "QPushButton:hover { background: #f3f8fc; }"
            )
        footer.addWidget(self.status_label, 1)
        footer.addWidget(self.btn_cancel)
        footer.addStretch(1)
        footer.addWidget(self.btn_copy_report)
        footer.addWidget(self.btn_save_report)
        layout.addLayout(footer)

        self.btn_current_log.clicked.connect(self.refresh_current_log)
        self.btn_open_log.clicked.connect(self.open_log_file)
        self.btn_send.clicked.connect(lambda _checked=False: self.send_message())
        self.message_input.send_requested.connect(self.send_message)
        self.btn_cancel.clicked.connect(self.suspend_analysis)
        self.btn_copy_report.clicked.connect(self.copy_report)
        self.btn_save_report.clicked.connect(self.save_report)
        self.btn_clear.clicked.connect(self._reset_conversation)
        self.zoom_out_button.clicked.connect(lambda: self.set_ui_scale(self.ui_scale_percent - 10))
        self.zoom_in_button.clicked.connect(lambda: self.set_ui_scale(self.ui_scale_percent + 10))
        self.zoom_reset_button.clicked.connect(lambda: self.set_ui_scale(100))
        self._apply_ui_scale()

    @property
    def ui_scale_percent(self) -> int:
        return self._ui_scale_percent

    def set_ui_scale(self, percent: int) -> None:
        snapped = int(round(int(percent) / 10.0) * 10)
        snapped = min(140, max(80, snapped))
        if snapped == self._ui_scale_percent:
            return
        bar = self.chat_scroll.verticalScrollBar()
        maximum = bar.maximum()
        was_at_bottom = maximum > 0 and bar.value() >= maximum - 2
        ratio = 1.0 if was_at_bottom else (bar.value() / maximum if maximum else 0.0)
        self._release_auto_scroll()
        self._ui_scale_percent = snapped
        self._typography = _Typography(percent=snapped, dpi_scale=self._typography.dpi_scale)
        self._apply_ui_scale(preserve_scroll=True)
        QTimer.singleShot(0, lambda: self._restore_chat_scroll(ratio, was_at_bottom))

    def _apply_ui_scale(self, preserve_scroll: bool = False) -> None:
        t = self._typography
        self.setFont(t.font(16))
        self.setStyleSheet("QDialog { background: #edf3f9; color: #18324a; }")
        self._main_layout.setContentsMargins(t.px(14), t.px(14), t.px(14), t.px(14))
        self._main_layout.setSpacing(t.px(8))

        self._header_card.setStyleSheet(
            "QFrame#assistantHeader { background: #ffffff; border: 1px solid #d9e5f0; "
            f"border-radius: {t.px(12)}px; }}"
        )
        self._header_layout.setContentsMargins(t.px(12), t.px(10), t.px(12), t.px(10))
        self._header_layout.setSpacing(t.px(6))
        self.header_title.setFont(t.font(24, QFont.Bold))
        self.header_title.setStyleSheet("color: #163a5f;")
        self.header_subtitle.setFont(t.font(14))
        self.header_subtitle.setStyleSheet("color: #637b91;")
        self.zoom_reset_button.setText(f"{self._ui_scale_percent}%")
        for button in (self.zoom_out_button, self.zoom_reset_button, self.zoom_in_button, self.btn_clear):
            button.setFont(t.font(15, QFont.DemiBold))
            button.setStyleSheet(
                "QPushButton { color: #46627e; background: #f5f8fb; border: 1px solid #d7e2ec; "
                f"border-radius: {t.px(7)}px; padding: {t.px(7)}px {t.px(10)}px; }}"
                "QPushButton:hover { background: #eaf2f9; }"
            )

        self._quick_card.setStyleSheet(
            "QFrame#quickActionsCard { background: #ffffff; border: 1px solid #d9e5f0; "
            f"border-radius: {t.px(12)}px; }}"
        )
        self._quick_layout.setContentsMargins(t.px(8), t.px(8), t.px(8), t.px(8))
        self._quick_layout.setHorizontalSpacing(t.px(8))
        self._quick_layout.setVerticalSpacing(t.px(8))
        for button in self.quick_buttons:
            button.setFont(t.font(16, QFont.DemiBold))
            button.setMinimumHeight(t.px(38))
            button.setStyleSheet(
                "QPushButton { text-align: left; background: #f4f8fc; color: #1d4f7a; "
                f"padding: {t.px(6)}px {t.px(10)}px; border: 1px solid #d9e7f2; "
                f"border-radius: {t.px(8)}px; }}"
                "QPushButton:hover { background: #e7f1fa; border-color: #a9c9e3; }"
            )

        self._source_row.setSpacing(t.px(8))
        self.source_label.setFont(t.font(14))
        self.source_label.setStyleSheet("color: #637b91;")
        for button in (self.btn_current_log, self.btn_open_log):
            button.setFont(t.font(15))
            button.setStyleSheet(
                "QPushButton { color: #2b618f; background: transparent; border: none; "
                f"padding: {t.px(4)}px {t.px(6)}px; }}"
                "QPushButton:hover { color: #123f68; text-decoration: underline; }"
            )

        self.chat_layout.setContentsMargins(t.px(2), t.px(2), t.px(8), t.px(2))
        self.chat_layout.setSpacing(t.px(8))
        self.status_label.setFont(t.font(14))
        self.status_label.setStyleSheet("color: #70869a;")
        self._composer.setStyleSheet(
            "QFrame#chatComposer { background: #ffffff; border: 1px solid #ceddea; "
            f"border-radius: {t.px(12)}px; }}"
        )
        self._composer_layout.setContentsMargins(t.px(8), t.px(7), t.px(8), t.px(7))
        self._composer_layout.setSpacing(t.px(8))
        self.message_input.setFont(t.font(16))
        self.message_input.setFixedHeight(t.px(68))
        self.message_input.setStyleSheet(
            "QPlainTextEdit { border: none; background: transparent; color: #18324a; "
            f"font-size: {t.px(16)}px; padding: {t.px(4)}px; }}"
        )
        self.btn_send.setFont(t.font(16, QFont.Bold))
        self.btn_send.setFixedSize(t.px(76), t.px(40))
        self.btn_send.setStyleSheet(
            "QPushButton { background: #2474ae; color: white; border: none; "
            f"border-radius: {t.px(8)}px; }}"
            "QPushButton:hover { background: #185d92; } QPushButton:disabled { background: #a9bfd1; }"
        )
        self._footer_layout.setSpacing(t.px(8))
        for button in (self.btn_cancel, self.btn_copy_report, self.btn_save_report):
            button.setFont(t.font(15))
            button.setStyleSheet(
                "QPushButton { background: #ffffff; color: #45647e; border: 1px solid #d0deea; "
                f"border-radius: {t.px(7)}px; padding: {t.px(7)}px {t.px(10)}px; }}"
                "QPushButton:hover { background: #f3f8fc; }"
            )

        for row in self._message_widgets:
            row_layout = row.layout()
            if row_layout is not None:
                row_layout.setContentsMargins(t.px(6), 0, t.px(6), 0)
            for bubble in row.findChildren(_AssistantBubble):
                bubble.set_typography(t)
            for bubble in row.findChildren(_UserBubble):
                bubble.set_typography(t)
        self._clamp_geometry_to_screen()
        QTimer.singleShot(0, self._update_source_label)
        if not preserve_scroll:
            self._request_auto_scroll()

    def _restore_chat_scroll(self, ratio: float, was_at_bottom: bool) -> None:
        bar = self.chat_scroll.verticalScrollBar()
        if was_at_bottom:
            bar.setValue(bar.maximum())
        else:
            bar.setValue(round(bar.maximum() * ratio))

    def _clamp_geometry_to_screen(self) -> None:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        dpi = self._typography.dpi_scale
        max_width = max(1, available.width() - round(40 * dpi))
        max_height = max(1, available.height() - round(80 * dpi))
        self.setMinimumSize(min(round(560 * dpi), max_width), min(round(500 * dpi), max_height))
        if self.width() > max_width or self.height() > max_height:
            self.resize(min(self.width(), max_width), min(self.height(), max_height))

    def eventFilter(self, watched, event) -> bool:
        if self._is_dialog_descendant(watched):
            if event.type() == QEvent.Wheel and event.modifiers() & Qt.ControlModifier:
                delta = event.angleDelta().y()
                if delta:
                    self.set_ui_scale(self.ui_scale_percent + (10 if delta > 0 else -10))
                event.accept()
                return True
            if event.type() == QEvent.KeyPress and event.modifiers() & Qt.ControlModifier:
                if event.key() in {Qt.Key_Plus, Qt.Key_Equal}:
                    self.set_ui_scale(self.ui_scale_percent + 10)
                    return True
                if event.key() == Qt.Key_Minus:
                    self.set_ui_scale(self.ui_scale_percent - 10)
                    return True
                if event.key() == Qt.Key_0:
                    self.set_ui_scale(100)
                    return True
        return super().eventFilter(watched, event)

    def _is_dialog_descendant(self, watched) -> bool:
        current = watched
        while current is not None:
            if current is self:
                return True
            current = current.parent()
        return False

    def _reset_conversation(self) -> None:
        if self._analysis_active:
            return
        for widget in self._message_widgets:
            self.chat_layout.removeWidget(widget)
            widget.deleteLater()
        self._message_widgets.clear()
        self.messages.clear()
        self._history.clear()
        self._chart_followups = False
        self._conversation_mode = None
        self._add_message("assistant", "你好，我可以帮你解读 FDIDM 实验现象、结果图和日志证据。")

    def _add_message(self, role: str, content: str) -> None:
        entry = {"role": role, "content": str(content)}
        self.messages.append(entry)
        bubble = self._message_bubble(role, entry["content"])
        self.chat_layout.insertWidget(max(0, self.chat_layout.count() - 1), bubble)
        self._message_widgets.append(bubble)
        self._request_auto_scroll()

    def _replace_pending(self, content: str) -> None:
        if self._pending is None:
            self._add_message("assistant", content)
            return
        index = self._pending["message_index"]
        old_widget = self._pending["widget"]
        self.chat_layout.removeWidget(old_widget)
        old_widget.deleteLater()
        try:
            self._message_widgets.remove(old_widget)
        except ValueError:
            pass
        self.messages.pop(index)
        self._add_message("assistant", content)
        self._pending = None

    def _message_bubble(self, role: str, content: str) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(6, 0, 6, 0)
        if role == "user":
            bubble = _UserBubble(content, self._typography)
            row_layout.addStretch(1)
            row_layout.addWidget(bubble)
        else:
            bubble = _AssistantBubble(content, self._typography)
            row_layout.addWidget(bubble, 1)
        return row

    def _scroll_to_latest(self) -> None:
        bar = self.chat_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _request_auto_scroll(self) -> None:
        self._auto_scroll_pending = True
        self._auto_scroll_timer.start(250)
        QTimer.singleShot(0, self._scroll_to_latest)

    def _on_chat_range_changed(self, *_args) -> None:
        if self._auto_scroll_pending:
            self._auto_scroll_timer.start(250)
            QTimer.singleShot(0, self._scroll_to_latest)

    def _release_auto_scroll(self) -> None:
        self._auto_scroll_pending = False

    def refresh_current_log(self, preserve_conversation: bool = False) -> None:
        if self._analysis_active:
            return
        if self._context_provider is not None:
            snapshot = self._fresh_platform_context()
            if "log_text" in snapshot:
                description = str(snapshot.get("source_description", "当前测试") or "当前测试")
                self._set_source_snapshot(
                    str(snapshot.get("log_text", "") or ""),
                    description,
                    "current",
                    preserve_conversation,
                )
                return
        try:
            text = self._log_provider()
        except Exception:
            self.status_label.setText("当前测试记录暂不可用。")
            return
        self._set_source_snapshot(str(text or ""), "当前测试", "current", preserve_conversation)

    def set_platform_context(self, snapshot: dict) -> None:
        """Adopt the controller-owned page snapshot without disturbing visible chat."""
        snapshot = dict(snapshot or {})
        next_id = str(snapshot.get("context_id", "") or "platform")
        if next_id != self._context_id:
            self._history.clear()
            self._chart_followups = False
            self._conversation_mode = None
        self._context_id = next_id
        self._platform_context = snapshot
        description = str(snapshot.get("source_description", "") or next_id)
        self.header_subtitle.setText(f"当前页面：{description}")
        if self._source_kind != "file":
            self._source_label_full_text = f"来源：{description}"
            self._update_source_label()

    def _fresh_platform_context(self) -> dict:
        if self._context_provider is None:
            return dict(self._platform_context)
        try:
            snapshot = self._context_provider() or {}
        except Exception:
            snapshot = {}
        self.set_platform_context(snapshot)
        return dict(self._platform_context)

    def open_log_file(self) -> None:
        if self._analysis_active:
            return
        initial_dir = Path(__file__).resolve().parents[2] / "log"
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择历史记录",
            str(initial_dir),
            "Log Files (*.log *.txt);;Text Files (*.txt);;All Files (*)",
        )
        if not path:
            return
        log_path = Path(path)
        try:
            if log_path.stat().st_size > MAX_LOG_FILE_BYTES:
                raise ValueError("记录文件超过 16 MiB")
            with log_path.open("rb") as handle:
                raw = handle.read(MAX_LOG_FILE_BYTES + 1)
            if len(raw) > MAX_LOG_FILE_BYTES:
                raise ValueError("记录文件超过 16 MiB")
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw.decode("gb18030")
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            self.status_label.setText(f"选择历史记录失败: {exc}")
            return
        self._set_source_snapshot(text, f"历史记录: {log_path.name}", "file")

    def _set_source_snapshot(
        self,
        text: str,
        source_description: str,
        source_kind: str,
        preserve_conversation: bool = False,
    ) -> None:
        changed = (
            source_kind != self._source_kind
            or source_description != self._source_description
            or text != self._log_text
        )
        self._log_text = text
        self._source_description = source_description
        self._source_kind = source_kind
        if changed and not preserve_conversation:
            self._history.clear()
            self._chart_followups = False
            self._conversation_mode = None
        self._source_label_full_text = f"来源：{source_description}"
        self._update_source_label()
        self.status_label.setText("已准备好。")

    def _update_source_label(self) -> None:
        width = self.source_label.width()
        if width <= 0:
            return
        elided = QFontMetrics(self.source_label.font()).elidedText(
            self._source_label_full_text, Qt.ElideRight, width
        )
        if self.source_label.text() != elided:
            self.source_label.setText(elided)

    def send_message(self, message=None, intent="chat") -> None:
        if self._analysis_active:
            return
        platform_snapshot = self._fresh_platform_context()
        if self._source_kind in {"none", "current"}:
            current_log = platform_snapshot.get("log_text") if platform_snapshot else None
            if current_log is not None:
                description = str(platform_snapshot.get("source_description", "当前测试") or "当前测试")
                self._set_source_snapshot(str(current_log or ""), description, "current", preserve_conversation=True)
            else:
                self.refresh_current_log(preserve_conversation=True)
        typed = message is None
        message = self.message_input.toPlainText() if typed else str(message)
        message = message.strip()
        if not message:
            return
        if typed:
            self.message_input.clear()
        self._add_message("user", message)

        chart_context = dict(platform_snapshot.get("chart_context", {}) or {}) if platform_snapshot else self._current_chart_context()
        if intent == "charts":
            if not bool(chart_context.get("has_data", False)):
                self._add_message("assistant", "当前还没有可解释的结果图，请先运行测试。")
                self.status_label.setText("等待可用结果图。")
                return
            self._chart_followups = True
        elif intent == "guide":
            self._chart_followups = True
        elif intent != "chat":
            self._chart_followups = False

        log_text = self._request_log_text(intent)
        if log_text:
            private_snapshot = log_text[-24000:]
            try:
                self._private_contexts.remove(private_snapshot)
            except ValueError:
                pass
            self._private_contexts.append(private_snapshot)
        request_platform_context = dict(platform_snapshot.get("platform_context", platform_snapshot) or {})
        has_results = (
            bool(chart_context.get("has_data", False))
            or bool(request_platform_context.get("has_results"))
            or bool(request_platform_context.get("results"))
            or bool(request_platform_context.get("current_run"))
            or bool(request_platform_context.get("displayed_metrics"))
        )
        if intent in CHAT_INTENTS_WITH_LOGS and not log_text.strip() and not has_results:
            self._add_message("assistant", "当前没有可用于分析的日志，请先运行测试或选择历史记录。")
            self.status_label.setText("等待可用日志。")
            return

        self._prepare_conversation_mode(intent)

        try:
            config = self._config_provider()
            api_key = str(config.api_key)
        except Exception:
            self._add_message("assistant", "AI 助手暂不可用，请联系平台管理员。")
            self.status_label.setText("AI 助手暂不可用，请联系平台管理员。")
            return
        if api_key:
            self._redaction_keys.add(api_key)

        self._request_token += 1
        token = self._request_token
        history = () if intent == "guide" else tuple(self._history)
        request_chart_context = self._request_chart_context(intent, chart_context)
        request_platform_context = self._request_platform_context(intent, request_platform_context)
        self._analysis_active = True
        self._set_request_controls(False)
        self.status_label.setText("正在思考…")
        self._add_message("assistant", "正在思考…")
        self._pending = {
            "message_index": len(self.messages) - 1,
            "widget": self._message_widgets[-1],
            "user_message": message,
        }
        worker = FDIDMAIAnalysisWorker(
            message,
            config,
            log_text=log_text,
            chart_context=request_chart_context,
            platform_context=request_platform_context,
            history=history,
            intent=intent,
        )
        self._worker = worker
        dialog_ref = weakref.ref(self)
        worker.succeeded.connect(
            lambda answer, request_token=token, request_message=message, request_context=self._context_id,
            request_origin=str(platform_snapshot.get("source_description", self._context_id) or self._context_id), ref=dialog_ref:
            FDIDMAIAnalysisDialog._deliver_success(ref, request_token, request_message, request_context, request_origin, answer)
        )
        worker.failed.connect(
            lambda error, request_token=token, ref=dialog_ref:
            FDIDMAIAnalysisDialog._deliver_failure(ref, request_token, error)
        )
        worker.completed.connect(
            lambda request_token=token, ref=dialog_ref:
            FDIDMAIAnalysisDialog._deliver_completed(ref, request_token)
        )
        thread = threading.Thread(target=worker.run, name="fdidm-ai-chat", daemon=True)
        self._worker_thread = thread
        thread.start()

    def _current_chart_context(self) -> dict:
        try:
            value = self._chart_context_provider() or {}
            return dict(value) if isinstance(value, dict) else {}
        except Exception:
            return {}

    def _request_log_text(self, intent: str) -> str:
        if intent == "guide":
            return ""
        if self._source_kind == "file" and (intent == "charts" or self._chart_followups):
            return ""
        return self._log_text

    def _request_chart_context(self, intent: str, chart_context: dict):
        if self._source_kind == "file" and intent not in {"charts", "guide"} and not self._chart_followups:
            return None
        if intent in {"charts", "guide"} or self._chart_followups:
            return chart_context
        if self._source_kind != "file" and intent in CHAT_INTENTS_WITH_LOGS and bool(chart_context.get("has_data", False)):
            return chart_context
        return None

    def _request_platform_context(self, intent: str, page_context: dict) -> dict:
        if self._source_kind != "file" or intent in {"charts", "guide"} or self._chart_followups:
            return page_context
        return {
            "source": "historical_log",
            "scope": "archive_independent",
            "source_name": self._source_description,
            "guide": page_context.get("guide", ""),
            "page_title": page_context.get("page_title", page_context.get("page_id", "")),
        }

    def _prepare_conversation_mode(self, intent: str) -> None:
        if intent in {"charts", "guide"}:
            mode = "current-charts"
        elif intent in CHAT_INTENTS_WITH_LOGS:
            mode = "historical-log" if self._source_kind == "file" else "current-log"
        else:
            if self._chart_followups:
                mode = "current-charts"
            elif self._source_kind == "file":
                mode = "historical-log"
            elif self._source_kind == "current":
                mode = "current-log"
            else:
                mode = self._conversation_mode or "chat"
        prior = self._conversation_mode
        if {prior, mode} == {"historical-log", "current-charts"}:
            self._history.clear()
        self._conversation_mode = mode

    def _handle_success(self, token: int, user_message: str, request_context: str, request_origin: str, answer: str) -> None:
        if token != self._request_token or not self._analysis_active:
            return
        answer = self._redact(answer)
        if request_context != self._context_id:
            answer = f"来自「{request_origin}」的回答：\n\n{answer}"
        self._replace_pending(answer)
        if request_context == self._context_id:
            self._history.extend((
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": answer},
            ))
        self.status_label.setText("已回复。")

    def _handle_failure(self, token: int, error: str) -> None:
        if token != self._request_token or not self._analysis_active:
            return
        self._replace_pending(f"分析失败: {self._redact(error)}")
        self.status_label.setText("请求失败。")

    def _handle_completed(self, token: int) -> None:
        if token != self._request_token:
            return
        self._analysis_active = False
        self._worker = None
        self._worker_thread = None
        self._set_request_controls(True)

    @staticmethod
    def _live_dialog(dialog_ref):
        dialog = dialog_ref()
        if dialog is None:
            return None
        try:
            dialog.objectName()
        except RuntimeError:
            return None
        return dialog

    @staticmethod
    def _deliver_success(dialog_ref, token: int, user_message: str, request_context: str, request_origin: str, answer: str) -> None:
        dialog = FDIDMAIAnalysisDialog._live_dialog(dialog_ref)
        if dialog is not None:
            dialog._handle_success(token, user_message, request_context, request_origin, answer)

    @staticmethod
    def _deliver_failure(dialog_ref, token: int, error: str) -> None:
        dialog = FDIDMAIAnalysisDialog._live_dialog(dialog_ref)
        if dialog is not None:
            dialog._handle_failure(token, error)

    @staticmethod
    def _deliver_completed(dialog_ref, token: int) -> None:
        dialog = FDIDMAIAnalysisDialog._live_dialog(dialog_ref)
        if dialog is not None:
            dialog._handle_completed(token)

    def _redact(self, text: str) -> str:
        safe = str(text)
        for key in self._redaction_keys:
            if key:
                safe = safe.replace(key, "[REDACTED]")
        for context in self._private_contexts:
            if context:
                safe = safe.replace(context, "[已隐藏的上下文]")
        return safe

    def _conversation_markdown(self) -> str:
        if not self.messages:
            return ""
        chunks = ["# FDIDM AI 助手对话\n"]
        for entry in self.messages:
            role = "我" if entry["role"] == "user" else "AI 助手"
            chunks.append(f"## {role}\n\n{self._redact(entry['content'])}\n")
        return "\n".join(chunks)

    def copy_report(self) -> None:
        content = self._conversation_markdown()
        if not content:
            return
        QApplication.clipboard().setText(content)
        self.status_label.setText("对话已复制。")

    def save_report(self) -> None:
        content = self._conversation_markdown()
        if not content:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 AI 助手对话",
            "fdidm_ai_conversation.md",
            "Markdown Files (*.md);;Text Files (*.txt)",
        )
        if not path:
            return
        try:
            Path(path).write_text(content, encoding="utf-8")
        except OSError:
            self.status_label.setText("保存对话失败。")
            return
        self.status_label.setText("对话已保存。")

    def suspend_analysis(self) -> None:
        self._request_token += 1
        if self._worker is not None:
            self._worker.cancel()
        if self._analysis_active:
            self._replace_pending("已取消本次请求。")
        self._worker = None
        self._worker_thread = None
        self._analysis_active = False
        self._set_request_controls(True)
        self.status_label.setText("已取消本次请求。")
        self.hide()

    def _set_request_controls(self, enabled: bool) -> None:
        self.btn_current_log.setEnabled(enabled)
        self.btn_open_log.setEnabled(enabled)
        self.btn_send.setEnabled(enabled)
        self.message_input.setReadOnly(not enabled)
        for button in self.quick_buttons:
            button.setEnabled(enabled)
        self.btn_cancel.setEnabled(not enabled)

    def closeEvent(self, event) -> None:
        self.suspend_analysis()
        event.accept()

    def reject(self) -> None:
        self.suspend_analysis()
        super().reject()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._source_kind in {"none", "current"}:
            self.refresh_current_log()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        for row in self._message_widgets:
            for bubble in row.findChildren(_AssistantBubble):
                bubble._queue_fit()


class FDIDMAIFloatingButton(QPushButton):
    """Vector assistant orb with an inset halo and the existing drag interaction."""

    def __init__(self, on_open: Callable[[], None], parent=None):
        super().__init__(parent)
        self._on_open = on_open
        self._drag_origin = None
        self._drag_start_global = None
        self._dragged = False
        self.user_moved = False
        self._hovered = False
        self.setObjectName("fdidmAiFloatingButton")
        self.setToolTip("平台 AI 助手 · 点击打开，拖动调整位置")
        self.setAccessibleName("平台 AI 助手")
        self.setAccessibleDescription("分析运行结果、解释图表并提供平台操作指导")
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAutoDefault(True)
        screen = self.screen() or QApplication.primaryScreen()
        dpi_scale = float(screen.logicalDotsPerInch()) / 96.0 if screen else 1.0
        side = max(48, round(60 * dpi_scale))
        self.setFixedSize(side, side)
        self.setStyleSheet(
            "QPushButton { background: transparent; border: none; padding: 0; }"
        )
        self.clicked.connect(self._on_open)

    @staticmethod
    def _sparkle(cx: float, cy: float, radius: float) -> QPainterPath:
        path = QPainterPath(QPointF(cx, cy - radius))
        path.cubicTo(cx + radius * .21, cy - radius * .21,
                     cx + radius * .21, cy - radius * .21, cx + radius, cy)
        path.cubicTo(cx + radius * .21, cy + radius * .21,
                     cx + radius * .21, cy + radius * .21, cx, cy + radius)
        path.cubicTo(cx - radius * .21, cy + radius * .21,
                     cx - radius * .21, cy + radius * .21, cx - radius, cy)
        path.cubicTo(cx - radius * .21, cy - radius * .21,
                     cx - radius * .21, cy - radius * .21, cx, cy - radius)
        path.closeSubpath()
        return path

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.scale(self.width() / 60.0, self.height() / 60.0)
        if not self.isEnabled():
            painter.setOpacity(.45)

        # All effects stay inside the widget, including at a clamped window edge.
        shadow = QRadialGradient(QPointF(30, 35), 28)
        shadow.setColorAt(0, QColor(61, 82, 171, 70 if self._hovered else 48))
        shadow.setColorAt(.72, QColor(70, 83, 165, 24))
        shadow.setColorAt(1, QColor(70, 83, 165, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(shadow)
        painter.drawEllipse(QRectF(2, 7, 56, 52))

        orb = QRectF(6, 4, 48, 48)
        if self.isDown():
            orb.adjust(1.0, 1.8, -1.0, -.2)
        elif self._hovered:
            orb.translate(0, -.7)

        rim = QLinearGradient(orb.topLeft(), orb.bottomRight())
        rim.setColorAt(0, QColor(214, 249, 255, 235))
        rim.setColorAt(.55, QColor(202, 214, 255, 200))
        rim.setColorAt(1, QColor(225, 217, 255, 225))
        painter.setBrush(rim)
        painter.drawEllipse(orb.adjusted(-1.2, -1.2, 1.2, 1.2))

        gradient = QLinearGradient(orb.topLeft(), orb.bottomRight())
        gradient.setColorAt(0, QColor("#1fb9dc" if self._hovered else "#28accf"))
        gradient.setColorAt(.48, QColor("#4287fa" if self._hovered else "#487de8"))
        gradient.setColorAt(1, QColor("#865ce8" if self._hovered else "#7955db"))
        painter.setBrush(gradient)
        painter.drawEllipse(orb)

        glow = QRadialGradient(orb.topLeft() + QPointF(13, 9), 38)
        glow.setColorAt(0, QColor(255, 255, 255, 82))
        glow.setColorAt(.6, QColor(255, 255, 255, 10))
        glow.setColorAt(1, QColor(255, 255, 255, 0))
        painter.setBrush(glow)
        painter.drawEllipse(orb)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 115), .65))
        painter.drawEllipse(orb.adjusted(.5, .5, -.5, -.5))

        # A speech outline and a four-point spark read as an assistant without
        # depending on platform fonts or a low-resolution bitmap icon.
        painter.save()
        if self.isDown():
            painter.translate(0, 1)
        bubble = QPainterPath(QPointF(22, 18))
        bubble.lineTo(36, 18)
        bubble.cubicTo(39.4, 18, 41, 20, 41, 23)
        bubble.lineTo(41, 30)
        bubble.cubicTo(41, 33.2, 39, 35, 36, 35)
        bubble.lineTo(29, 35)
        bubble.lineTo(23, 39)
        bubble.lineTo(23.8, 34.7)
        bubble.cubicTo(19.5, 34.7, 18, 32.4, 18, 29.5)
        bubble.lineTo(18, 23)
        bubble.cubicTo(18, 19.5, 19.6, 18, 22, 18)
        painter.setPen(QPen(QColor(255, 255, 255, 242), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPath(bubble)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#ffffff"))
        painter.drawPath(self._sparkle(29.4, 26.4, 6.2))
        painter.drawPath(self._sparkle(40.8, 15.2, 3.2))
        painter.setBrush(QColor(230, 248, 255, 220))
        painter.drawEllipse(QPointF(18.2, 39.1), 1.2, 1.2)
        painter.restore()

        if self.hasFocus():
            painter.setPen(QPen(QColor("#385eb9"), 1.2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(orb.adjusted(-3, -3, 3, 3))

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def clamp_to_parent(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        max_x = max(0, parent.width() - self.width())
        max_y = max(0, parent.height() - self.height())
        self.move(min(max(self.x(), 0), max_x), min(max(self.y(), 0), max_y))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_origin = event.globalPos() - self.frameGeometry().topLeft()
            self._drag_start_global = event.globalPos()
            self._dragged = False
            self.setDown(True)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_origin is not None and event.buttons() & Qt.LeftButton:
            new_pos = event.globalPos() - self._drag_origin
            if (event.globalPos() - self._drag_start_global).manhattanLength() >= QApplication.startDragDistance():
                self._dragged = True
                self.user_moved = True
                self.setDown(False)
            self.move(new_pos)
            self.clamp_to_parent()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._drag_origin is not None:
            was_dragged = self._dragged
            self._drag_origin = None
            self._drag_start_global = None
            self.setDown(False)
            self.clamp_to_parent()
            if not was_dragged:
                self.click()
            event.accept()
            return
        super().mouseReleaseEvent(event)


# The legacy module path remains stable for the FDIDM page while the same
# dialog is now used by the platform-wide assistant controller.
PlatformAIAssistantDialog = FDIDMAIAnalysisDialog
