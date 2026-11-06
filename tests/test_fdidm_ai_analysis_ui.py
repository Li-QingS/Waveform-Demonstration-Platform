import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtGui import QFont, QMouseEvent, QWheelEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QFileDialog, QTextBrowser, QWidget

from waveform_sim.service.ai_log_analyzer import DeepSeekConfig


APP = QApplication.instance() or QApplication([])
RAW_MARKER = "RAW_LOG_SECRET_MARKER"
CONFIG = DeepSeekConfig("https://config.example.invalid", "fdidm-assistant", "test-key")


def _wait_until(predicate, timeout_ms=1500):
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    QApplication.processEvents()
    return predicate()


@pytest.fixture(autouse=True)
def _cleanup_owned_assistant_dialogs():
    """Dispose only dialogs created by this module, leaving controller windows untouched."""
    from waveform_sim.ui.fdidm_ai_analysis import FDIDMAIAnalysisDialog

    existing_ids = {id(widget) for widget in APP.topLevelWidgets() if isinstance(widget, FDIDMAIAnalysisDialog)}
    yield
    for dialog in APP.topLevelWidgets():
        if not isinstance(dialog, FDIDMAIAnalysisDialog) or id(dialog) in existing_ids:
            continue
        try:
            dialog.close()
            dialog.deleteLater()
        except RuntimeError:
            pass
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    APP.processEvents()
    leaked = [
        widget for widget in APP.topLevelWidgets()
        if isinstance(widget, FDIDMAIAnalysisDialog) and id(widget) not in existing_ids
    ]
    assert leaked == []


def _process_events(duration_ms=50):
    deadline = time.monotonic() + duration_ms / 1000
    while time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)


def _dialog(log_provider=lambda: "current log", *, chart_context_provider=None, context_provider=None, config_provider=lambda: CONFIG):
    from waveform_sim.ui.fdidm_ai_analysis import FDIDMAIAnalysisDialog

    return FDIDMAIAnalysisDialog(
        log_provider,
        config_provider=config_provider,
        chart_context_provider=chart_context_provider,
        context_provider=context_provider,
    )


def _install_chat(monkeypatch, responses=None, calls=None):
    from waveform_sim.service import ai_log_analyzer

    calls = [] if calls is None else calls
    responses = responses or (lambda message, **_: f"答复: {message}")

    def fake_chat(message, config, **kwargs):
        calls.append({"message": message, "config": config, **kwargs})
        return responses(message, config=config, **kwargs)

    monkeypatch.setattr(ai_log_analyzer, "chat_with_assistant", fake_chat, raising=False)
    return calls


def _visible_text(dialog):
    return "\n".join(entry["content"] for entry in dialog.messages)


def test_chat_surface_has_shortcuts_and_never_displays_raw_log():
    """Reintroducing raw logs or config fields would defeat the assistant chat boundary."""
    dialog = _dialog(lambda: RAW_MARKER)
    dialog.show()
    _process_events()

    assert dialog.windowTitle() == "平台 AI 助手"
    assert dialog.header_title.text() == "AI 助手"
    assert "实验" in dialog.header_subtitle.text()
    assert [button.text() for button in dialog.quick_buttons] == [
        "日志分析", "运行结果图解释", "异常原因排查", "优化建议",
        "操作指导",
    ]
    assert dialog.btn_current_log.text() == "当前测试"
    assert dialog.btn_open_log.text() == "选择历史记录"
    assert not hasattr(dialog, "log_preview")
    assert not hasattr(dialog, "base_url_edit")
    assert RAW_MARKER not in _visible_text(dialog)
    assert RAW_MARKER not in "\n".join(widget.text() for widget in dialog.findChildren(type(dialog.source_label)))


def test_platform_guide_works_on_home_without_log_or_results(monkeypatch):
    """Operation guidance is available on pages with no run evidence."""
    calls = _install_chat(monkeypatch)
    context = {
        "context_id": "home",
        "source_description": "首页",
        "log_text": "",
        "chart_context": {"has_data": False},
        "platform_context": {"guide": "home controls"},
    }
    current = {"snapshot": context}
    dialog = _dialog(context_provider=lambda: current["snapshot"])
    dialog.quick_buttons[4].click()

    assert _wait_until(lambda: len(calls) == 1 and not dialog._analysis_active)
    assert calls[0]["intent"] == "guide"
    assert calls[0]["log_text"] == ""
    assert calls[0]["platform_context"] == context["platform_context"]


def test_platform_context_keeps_archive_caption_and_restores_current_page_caption(monkeypatch, tmp_path):
    """The source row must identify the active platform page until an explicit archive is chosen."""
    context = {
        "context_id": "home",
        "source_description": "首页",
        "log_text": "home log",
        "chart_context": {"has_data": False},
        "platform_context": {"page_id": "home"},
    }
    current = {"snapshot": context}
    dialog = _dialog(context_provider=lambda: current["snapshot"])
    dialog.set_platform_context(current["snapshot"])
    dialog.show()
    _process_events()
    assert "首页" in dialog.source_label.text()

    archive = tmp_path / "archive.log"
    archive.write_text("historical", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_: (str(archive), "Log Files (*.log)"))
    dialog.open_log_file()
    current["snapshot"] = {**context, "source_description": "仿真/OFDM"}
    dialog.set_platform_context(current["snapshot"])
    assert "历史记录" in dialog.source_label.text()

    dialog.btn_current_log.click()
    _process_events()
    assert "仿真/OFDM" in dialog.source_label.text()


def test_current_page_log_analysis_includes_chart_evidence_when_log_is_empty(monkeypatch):
    """Curve-only pages must send their actual chart summaries, not only a has-results flag."""
    calls = _install_chat(monkeypatch)
    chart = {"has_data": True, "source": "current_page_charts", "charts": [{"title": "SER", "series": [1, 2]}]}
    context = {
        "context_id": "ofdm",
        "source_description": "仿真/OFDM",
        "log_text": "",
        "chart_context": chart,
        "platform_context": {"page_id": "ofdm", "has_results": True},
    }
    dialog = _dialog(context_provider=lambda: context)
    dialog.quick_buttons[0].click()

    assert _wait_until(lambda: len(calls) == 1 and not dialog._analysis_active)
    assert calls[0]["log_text"] == ""
    assert calls[0]["chart_context"] == chart


def test_historical_log_analysis_strips_live_result_context(monkeypatch):
    """Archive analysis must not mix current run metrics or configuration with the selected record."""
    calls = _install_chat(monkeypatch)
    context = {
        "context_id": "ofdm",
        "source_description": "仿真/OFDM",
        "log_text": "current log",
        "chart_context": {"has_data": True, "charts": [{"title": "current"}]},
        "platform_context": {
            "page_id": "ofdm", "page_title": "OFDM", "guide": "controls",
            "cached_results": {"secret": "live"}, "result_config": {"snr": 10},
            "displayed_metrics": {"ser": 1e-3}, "current_run": {"running": True},
        },
    }
    dialog = _dialog(context_provider=lambda: context)
    dialog._set_source_snapshot("archive log", "历史记录: old.log", "file")
    dialog.quick_buttons[0].click()

    assert _wait_until(lambda: len(calls) == 1)
    payload = calls[0]["platform_context"]
    assert calls[0]["log_text"] == "archive log"
    assert calls[0]["chart_context"] is None
    assert payload["source"] == "historical_log"
    assert payload["scope"] == "archive_independent"
    assert payload["page_title"] == "OFDM"
    assert "cached_results" not in payload
    assert "result_config" not in payload
    assert "displayed_metrics" not in payload
    assert "current_run" not in payload


def test_guide_and_followup_ignore_selected_historical_log(monkeypatch):
    """Guidance remains grounded in the current platform page, never an archived record."""
    calls = _install_chat(monkeypatch)
    context = {
        "context_id": "home",
        "source_description": "首页",
        "log_text": "current page log",
        "chart_context": {"has_data": False},
        "platform_context": {"page_id": "home", "guide": "home controls"},
    }
    dialog = _dialog(context_provider=lambda: context)
    dialog._set_source_snapshot("archived only", "历史记录: old.log", "file")
    dialog.quick_buttons[4].click()
    assert _wait_until(lambda: len(calls) == 1 and not dialog._analysis_active)
    assert calls[0]["log_text"] == ""
    assert calls[0]["history"] == ()
    dialog.message_input.setPlainText("下一步怎么做")
    dialog.send_message()
    assert _wait_until(lambda: len(calls) == 2 and not dialog._analysis_active)
    assert calls[1]["log_text"] == ""
    assert calls[1]["chart_context"] == context["chart_context"]


def test_page_change_preserves_visible_chat_but_isolates_new_history(monkeypatch):
    """Controller page changes retain the transcript while preventing cross-page followup grounding."""
    calls = _install_chat(monkeypatch)
    current = {
        "context_id": "ofdm",
        "source_description": "仿真/OFDM",
        "log_text": "ofdm log",
        "chart_context": {"has_data": True},
        "platform_context": {"page": "ofdm"},
    }
    dialog = _dialog(context_provider=lambda: current)
    dialog.message_input.setPlainText("OFDM 怎么看")
    dialog.send_message()
    assert _wait_until(lambda: len(calls) == 1 and not dialog._analysis_active)
    dialog.message_input.setPlainText("草稿保留")
    dialog.set_ui_scale(120)
    old_messages = list(dialog.messages)

    current = {
        "context_id": "otfs",
        "source_description": "仿真/OTFS",
        "log_text": "otfs log",
        "chart_context": {"has_data": True},
        "platform_context": {"page": "otfs"},
    }
    dialog.set_platform_context(current)
    assert dialog.messages == old_messages
    assert dialog.message_input.toPlainText() == "草稿保留"
    assert dialog.ui_scale_percent == 120
    dialog.message_input.setPlainText("OTFS 怎么看")
    dialog.send_message()
    assert _wait_until(lambda: len(calls) == 2 and not dialog._analysis_active)
    assert calls[1]["history"] == ()
    assert calls[1]["log_text"] == "otfs log"


def test_late_prior_page_answer_is_labeled_and_excluded_from_new_history(monkeypatch, request):
    """An in-flight answer may finish visibly, but cannot be treated as a reply for the new page."""
    from waveform_sim.service import ai_log_analyzer

    release = threading.Event()
    started = threading.Event()
    request.addfinalizer(release.set)
    current = {
        "context_id": "ofdm",
        "source_description": "仿真/OFDM",
        "log_text": "ofdm",
        "chart_context": {"has_data": True},
        "platform_context": {"page": "ofdm"},
    }

    def staged(message, *_args, **_kwargs):
        if message == "old":
            started.set()
            release.wait(2)
            return "旧页面结论"
        return "新页面结论"

    monkeypatch.setattr(ai_log_analyzer, "chat_with_assistant", staged, raising=False)
    dialog = _dialog(context_provider=lambda: current)
    dialog.message_input.setPlainText("old")
    dialog.send_message()
    assert started.wait(1)
    current = {
        "context_id": "otfs",
        "source_description": "仿真/OTFS",
        "log_text": "otfs",
        "chart_context": {"has_data": True},
        "platform_context": {"page": "otfs"},
    }
    dialog.set_platform_context(current)
    release.set()
    assert _wait_until(lambda: "旧页面结论" in _visible_text(dialog))
    assert "来自「仿真/OFDM」的回答" in _visible_text(dialog)
    dialog.message_input.setPlainText("new")
    dialog.send_message()
    assert _wait_until(lambda: "新页面结论" in _visible_text(dialog))
    assert dialog._history[-2:] == [
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "新页面结论"},
    ]


def test_shortcut_adds_exact_user_message_and_sends_common_request(monkeypatch):
    """Shortcut clicks must use the same message pipeline as typed chat."""
    calls = _install_chat(monkeypatch)
    dialog = _dialog(lambda: RAW_MARKER)
    dialog.refresh_current_log()

    dialog.quick_buttons[0].click()

    assert _wait_until(lambda: any(entry["role"] == "assistant" and "答复" in entry["content"] for entry in dialog.messages))
    assert dialog.messages[-2] == {"role": "user", "content": "日志分析"}
    assert calls[0]["message"] == "日志分析"
    assert calls[0]["intent"] == "logs"
    assert calls[0]["log_text"] == RAW_MARKER
    assert RAW_MARKER not in _visible_text(dialog)


def test_current_source_refreshes_before_send_without_resetting_chat_mode(monkeypatch):
    """An already-open chat must use logs produced after it was opened without losing its followup context."""
    calls = _install_chat(monkeypatch)
    current = {"text": ""}
    chart_context = {"has_data": True, "source": "current_page_charts", "charts": [{"name": "SER"}]}
    dialog = _dialog(lambda: current["text"], chart_context_provider=lambda: chart_context)
    dialog.show()
    dialog.quick_buttons[1].click()
    assert _wait_until(lambda: len(calls) == 1 and not dialog._analysis_active)
    assert dialog._chart_followups is True

    current["text"] = "latest log"
    dialog.message_input.setPlainText("继续解释")
    dialog.send_message()
    assert _wait_until(lambda: len(calls) == 2 and not dialog._analysis_active)
    assert calls[1]["log_text"] == "latest log"
    assert dialog._chart_followups is True


def test_assistant_response_does_not_echo_private_log_context(monkeypatch):
    """A backend echo of the private context must not surface in chat or exports."""
    _install_chat(monkeypatch, responses=lambda *_args, **_kwargs: f"证据: {RAW_MARKER}")
    dialog = _dialog(lambda: RAW_MARKER)
    dialog.refresh_current_log()

    dialog.quick_buttons[0].click()

    assert _wait_until(lambda: dialog.messages[-1]["role"] == "assistant" and "证据" in dialog.messages[-1]["content"])
    assert RAW_MARKER not in _visible_text(dialog)
    assert RAW_MARKER not in dialog._conversation_markdown()


def test_typed_messages_use_enter_history_and_shift_enter_newline(monkeypatch):
    """Typing should retain conversational history while Shift+Enter only adds a newline."""
    calls = _install_chat(monkeypatch)
    dialog = _dialog()
    dialog.message_input.setPlainText("first question")
    dialog.btn_send.click()
    assert _wait_until(lambda: len(calls) == 1 and not dialog._analysis_active)

    dialog.message_input.setPlainText("second")
    QTest.keyClick(dialog.message_input, Qt.Key_Return, Qt.ShiftModifier)
    assert "\n" in dialog.message_input.toPlainText()
    assert len(calls) == 1
    dialog.message_input.setPlainText("second question")
    QTest.keyClick(dialog.message_input, Qt.Key_Return)
    assert _wait_until(lambda: len(calls) == 2 and not dialog._analysis_active)

    assert calls[0]["intent"] == "chat"
    assert calls[1]["history"] == (
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "答复: first question"},
    )


def test_source_and_grounding_changes_clear_backend_history_but_same_context_followup_keeps_it(monkeypatch):
    """Archived-log conclusions must not cross into current plots, while same-context followups retain context."""
    calls = _install_chat(monkeypatch)
    chart_context = {"has_data": True, "source": "current_page_charts", "charts": [{"name": "SER"}]}
    dialog = _dialog(lambda: "current", chart_context_provider=lambda: chart_context)

    dialog._set_source_snapshot("current", "当前测试", "current")
    dialog.quick_buttons[1].click()
    assert _wait_until(lambda: len(calls) == 1 and not dialog._analysis_active)
    dialog._set_source_snapshot("current", "当前测试", "current")
    assert dialog._chart_followups is True
    dialog.message_input.setPlainText("继续")
    dialog.send_message()
    assert _wait_until(lambda: len(calls) == 2 and not dialog._analysis_active)
    assert calls[1]["history"]

    dialog._set_source_snapshot("archived", "历史记录: old.log", "file")
    dialog.quick_buttons[0].click()
    assert _wait_until(lambda: len(calls) == 3 and not dialog._analysis_active)
    assert calls[2]["history"] == ()
    dialog.quick_buttons[1].click()
    assert _wait_until(lambda: len(calls) == 4 and not dialog._analysis_active)
    assert calls[3]["history"] == ()
    assert calls[3]["log_text"] == ""
    dialog.quick_buttons[0].click()
    assert _wait_until(lambda: len(calls) == 5 and not dialog._analysis_active)
    assert calls[4]["history"] == ()


def test_typed_historical_turn_isolated_before_current_chart_request(monkeypatch):
    """A generic typed question over an archive still grounds the next chart request in a different context."""
    calls = _install_chat(monkeypatch)
    chart_context = {"has_data": True, "source": "current_page_charts", "charts": [{"name": "SER"}]}
    dialog = _dialog(chart_context_provider=lambda: chart_context)
    dialog._set_source_snapshot("archived evidence", "历史记录: old.log", "file")

    dialog.message_input.setPlainText("这个归档记录说明什么")
    dialog.send_message()
    assert _wait_until(lambda: len(calls) == 1 and not dialog._analysis_active)
    dialog.quick_buttons[1].click()
    assert _wait_until(lambda: len(calls) == 2 and not dialog._analysis_active)

    assert calls[1]["intent"] == "charts"
    assert calls[1]["history"] == ()
    assert calls[1]["log_text"] == ""


def test_rejected_chart_action_keeps_archived_typed_followup_grounding(monkeypatch):
    """A no-data chart click is local UI feedback and must not mutate the active archive conversation."""
    calls = _install_chat(monkeypatch)
    dialog = _dialog(chart_context_provider=lambda: {"has_data": False})
    dialog._set_source_snapshot("archived evidence", "历史记录: old.log", "file")

    dialog.message_input.setPlainText("先看归档记录")
    dialog.send_message()
    assert _wait_until(lambda: len(calls) == 1 and not dialog._analysis_active)
    dialog.quick_buttons[1].click()
    assert len(calls) == 1
    dialog.message_input.setPlainText("继续解释")
    dialog.send_message()
    assert _wait_until(lambda: len(calls) == 2 and not dialog._analysis_active)

    assert calls[1]["intent"] == "chat"
    assert calls[1]["log_text"] == "archived evidence"
    assert calls[1]["history"]


def test_chart_shortcut_without_data_replies_locally_without_network(monkeypatch):
    """A chart explanation cannot call the service when no result graph exists."""
    calls = _install_chat(monkeypatch)
    dialog = _dialog(chart_context_provider=lambda: {"has_data": False})

    dialog.quick_buttons[1].click()

    assert calls == []
    assert dialog.messages[-2] == {"role": "user", "content": "运行结果图解释"}
    assert dialog.messages[-1] == {"role": "assistant", "content": "当前还没有可解释的结果图，请先运行测试。"}


def test_chart_request_excludes_historical_log_and_typed_followup(monkeypatch):
    """Current result graphs must never be combined with an archived log context."""
    calls = _install_chat(monkeypatch)
    chart_context = {"source": "current", "selected_tab": "SER", "has_data": True, "charts": [{"name": "SER"}]}
    dialog = _dialog(chart_context_provider=lambda: chart_context)
    dialog._set_source_snapshot(RAW_MARKER, "历史记录: archived.log", "file")

    dialog.quick_buttons[1].click()
    assert _wait_until(lambda: len(calls) == 1 and not dialog._analysis_active)
    assert calls[0]["intent"] == "charts"
    assert calls[0]["log_text"] == ""
    assert calls[0]["chart_context"] == chart_context

    dialog.message_input.setPlainText("继续解释这张图")
    dialog.send_message()
    assert _wait_until(lambda: len(calls) == 2 and not dialog._analysis_active)
    assert calls[1]["intent"] == "chat"
    assert calls[1]["log_text"] == ""


def test_missing_config_adds_friendly_assistant_message_without_network(monkeypatch):
    """Missing backend setup must preserve the user bubble without leaking an exception."""
    calls = _install_chat(monkeypatch)

    def unavailable():
        raise RuntimeError("secret backend failure")

    dialog = _dialog(config_provider=unavailable)
    dialog.message_input.setPlainText("hello")
    dialog.send_message()

    assert calls == []
    assert dialog.messages[-2] == {"role": "user", "content": "hello"}
    assert dialog.messages[-1] == {"role": "assistant", "content": "AI 助手暂不可用，请联系平台管理员。"}
    assert "secret backend" not in _visible_text(dialog)


def test_explicit_log_intents_without_logs_reply_locally(monkeypatch):
    """Log-dependent shortcuts must not send context-free requests."""
    calls = _install_chat(monkeypatch)
    dialog = _dialog(lambda: "")
    dialog.refresh_current_log()

    dialog.quick_buttons[0].click()
    assert calls == []
    assert dialog.messages[-1]["role"] == "assistant"
    assert "日志" in dialog.messages[-1]["content"]


def test_copy_save_and_clear_export_only_visible_conversation(monkeypatch, tmp_path):
    """Exports must be a readable transcript and never carry hidden request context."""
    calls = _install_chat(monkeypatch)
    target = tmp_path / "conversation.md"
    dialog = _dialog(lambda: RAW_MARKER)
    dialog.refresh_current_log()
    dialog.message_input.setPlainText("请总结")
    dialog.send_message()
    assert _wait_until(
        lambda: len(calls) == 1 and "答复: 请总结" in _visible_text(dialog) and not dialog._analysis_active
    )
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_: (str(target), "Markdown (*.md)"))

    dialog.btn_save_report.click()
    written = target.read_text(encoding="utf-8")
    assert "请总结" in written
    assert "答复: 请总结" in written
    assert RAW_MARKER not in written
    assert "test-key" not in written
    assert "config.example.invalid" not in written

    dialog.btn_copy_report.click()
    assert RAW_MARKER not in QApplication.clipboard().text()
    dialog.btn_clear.click()
    assert all(entry["content"] != "请总结" for entry in dialog.messages)


def test_assistant_bubbles_reflow_without_blank_space_or_clipping(monkeypatch):
    """Assistant bubbles must follow document height at desktop and narrow dialog widths."""
    long_answer = "\n\n".join(f"第 {index} 条解释：" + "这是较长的结果说明，用于验证消息气泡会根据实际宽度重新换行。" * 3 for index in range(10))
    _install_chat(monkeypatch, responses=lambda *_args, **_kwargs: long_answer)
    dialog = _dialog()
    dialog.resize(860, 720)
    dialog.show()
    dialog.message_input.setPlainText("解释结果")
    dialog.btn_send.click()
    assert _wait_until(lambda: long_answer in _visible_text(dialog))

    for width in (860, 620):
        dialog.resize(width, 720)
        _process_events(100)
        bubbles = dialog.findChildren(QTextBrowser)
        greeting, answer = bubbles[0], bubbles[-1]
        assert greeting.height() < 100
        assert answer.width() > 300
        assert answer.viewport().height() + 4 >= int(answer.document().size().height())


def test_new_long_assistant_message_scrolls_to_latest_after_layout_settles(monkeypatch):
    """Dynamic bubble growth must leave the newest response visible in a narrow dialog."""
    long_answer = "\n\n".join("长回复段落：" + "用于验证自动滚动会等待布局稳定。" * 6 for _ in range(15))
    _install_chat(monkeypatch, responses=lambda *_args, **_kwargs: long_answer)
    dialog = _dialog()
    dialog.resize(620, 650)
    dialog.show()
    QTest.qWait(80)
    dialog.message_input.setPlainText("显示长回复")
    dialog.btn_send.click()
    assert _wait_until(lambda: long_answer in _visible_text(dialog))
    QTest.qWait(150)
    bar = dialog.chat_scroll.verticalScrollBar()
    assert bar.value() == bar.maximum()


def test_dialog_scoped_typography_zoom_reflows_all_bubbles_and_keeps_draft(monkeypatch):
    """Ctrl zoom from an assistant bubble must scale the whole dialog, not just that text browser."""
    _install_chat(monkeypatch, responses=lambda *_args, **_kwargs: "**粗体结论**\n\n- 第一项\n- 第二项")
    dialog = _dialog()
    dialog.resize(900, 820)
    dialog.show()
    dialog.message_input.setPlainText("未发送草稿")
    dialog.message_input.setFocus()
    dialog.send_message("生成说明")
    assert _wait_until(lambda: len(dialog.findChildren(QTextBrowser)) >= 2 and not dialog._analysis_active)

    old_assistant, old_answer = dialog.findChildren(QTextBrowser)[0], dialog.findChildren(QTextBrowser)[-1]
    old_body_px = dialog.message_input.font().pixelSize()
    old_doc_px = old_answer.document().defaultFont().pixelSize()
    wheel = QWheelEvent(
        QPointF(8, 8), QPointF(8, 8), QPoint(), QPoint(0, 120),
        Qt.NoButton, Qt.ControlModifier, Qt.NoScrollPhase, False,
    )
    QApplication.sendEvent(old_answer.viewport(), wheel)
    _process_events(80)

    assert dialog.ui_scale_percent == 110
    assert dialog.message_input.font().pixelSize() > old_body_px
    assert old_answer.document().defaultFont().pixelSize() > old_doc_px
    assert dialog.message_input.toPlainText() == "未发送草稿"
    assert old_assistant.viewport().height() + 4 >= int(old_assistant.document().size().height())

    dialog._add_message("assistant", "新的 **Markdown** 消息")
    _process_events(80)
    new_answer = dialog.findChildren(QTextBrowser)[-1]
    assert new_answer.document().defaultFont().pixelSize() == old_answer.document().defaultFont().pixelSize()
    assert new_answer.viewport().height() + 4 >= int(new_answer.document().size().height())

    normal_wheel = QWheelEvent(
        QPointF(8, 8), QPointF(8, 8), QPoint(), QPoint(0, -120),
        Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False,
    )
    QApplication.sendEvent(old_answer.viewport(), normal_wheel)
    assert dialog.ui_scale_percent == 110
    QTest.keyClick(dialog.message_input, Qt.Key_0, Qt.ControlModifier)
    assert dialog.ui_scale_percent == 100
    dialog.zoom_in_button.click()
    assert dialog.ui_scale_percent == 110
    dialog.zoom_out_button.click()
    assert dialog.ui_scale_percent == 100
    QTest.keyClick(dialog.message_input, Qt.Key_Plus, Qt.ControlModifier)
    assert dialog.ui_scale_percent == 110
    QTest.keyClick(dialog.message_input, Qt.Key_Minus, Qt.ControlModifier)
    assert dialog.ui_scale_percent == 100


def test_first_show_and_new_assistant_bubbles_use_dialog_typography_not_app_default(monkeypatch):
    """Markdown must inherit the assistant font before its first parse, without requiring a zoom toggle."""
    _install_chat(monkeypatch, responses=lambda *_args, **_kwargs: "# 标题\n\n**结论** 与 `code`")
    app = QApplication.instance()
    original_font = QFont(app.font())
    app.setFont(QFont("SimSun"))
    try:
        dialog = _dialog()
        dialog.show()
        dialog.message_input.setPlainText("首条回复")
        dialog.btn_send.click()
        assert _wait_until(lambda: len(dialog.findChildren(QTextBrowser)) >= 2 and not dialog._analysis_active)
        expected_family = dialog.message_input.font().family()
        first_bubbles = dialog.findChildren(QTextBrowser)
        assert all(bubble.font().family() == expected_family for bubble in first_bubbles)
        assert all(bubble.document().defaultFont().family() == expected_family for bubble in first_bubbles)
        dialog._add_message("assistant", "## 新标题\n\n新的 `code`")
        _process_events(60)
        new_bubble = dialog.findChildren(QTextBrowser)[-1]
        assert new_bubble.font().family() == expected_family
        assert new_bubble.document().defaultFont().family() == expected_family
        assert new_bubble.viewport().height() + 4 >= int(new_bubble.document().size().height())
    finally:
        app.setFont(original_font)


def test_markdown_inline_and_fenced_code_scale_with_assistant_typography(monkeypatch):
    """Markdown parser-created code fragments must not stay at a fixed Courier point size."""
    response = "正文 `inline_code`。\n\n```text\ncode_value\n```"
    _install_chat(monkeypatch, responses=lambda *_args, **_kwargs: response)
    dialog = _dialog()
    dialog.show()
    dialog.send_message("检查代码排版")
    assert _wait_until(lambda: response in _visible_text(dialog) and not dialog._analysis_active)
    bubble = dialog.findChildren(QTextBrowser)[-1]

    def fragment_font(token):
        block = bubble.document().begin()
        while block.isValid():
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                if token in fragment.text():
                    return fragment.charFormat().font()
                iterator += 1
            block = block.next()
        raise AssertionError(f"missing fragment {token}")

    normal_before = fragment_font("正文")
    inline_before = fragment_font("inline_code")
    fenced_before = fragment_font("code_value")
    assert normal_before.fixedPitch() is False
    assert inline_before.fixedPitch() is True
    assert fenced_before.fixedPitch() is True
    dialog.set_ui_scale(120)
    _process_events(100)
    normal_after = fragment_font("正文")
    inline_after = fragment_font("inline_code")
    fenced_after = fragment_font("code_value")
    assert normal_after.fixedPitch() is False
    assert inline_after.pixelSize() > inline_before.pixelSize()
    assert fenced_after.pixelSize() > fenced_before.pixelSize()


def test_scale_change_preserves_reader_scroll_position(monkeypatch):
    """Typography changes must not force a reader from older messages to the newest answer."""
    dialog = _dialog()
    dialog.resize(900, 820)
    dialog.show()
    for index in range(12):
        dialog._add_message("assistant", f"消息 {index}\n\n" + "用于滚动位置测试的长文本。" * 10)
    _process_events(150)
    bar = dialog.chat_scroll.verticalScrollBar()
    assert bar.maximum() > 0
    bar.setValue(round(bar.maximum() * 0.33))
    before = bar.value() / bar.maximum()
    dialog.set_ui_scale(110)
    _process_events(150)
    after = bar.value() / bar.maximum()
    assert abs(after - before) < 0.08


def test_network_error_and_cancel_are_preserved_in_conversation(monkeypatch, request):
    """Errors and cancellation should appear as assistant messages while stale work stays hidden."""
    from waveform_sim.service import ai_log_analyzer

    started = threading.Event()
    release = threading.Event()
    request.addfinalizer(release.set)

    def slow_chat(*args, **kwargs):
        started.set()
        release.wait(2)
        return "stale answer"

    monkeypatch.setattr(ai_log_analyzer, "chat_with_assistant", slow_chat, raising=False)
    dialog = _dialog()
    dialog.message_input.setPlainText("cancel me")
    dialog.send_message()
    assert started.wait(1)
    assert "正在思考" in _visible_text(dialog)
    dialog.message_input.setPlainText("duplicate")
    dialog.send_message()
    assert "duplicate" not in _visible_text(dialog)
    dialog.suspend_analysis()
    assert dialog.messages[-1]["content"] == "已取消本次请求。"
    release.set()
    _process_events()
    assert "stale answer" not in _visible_text(dialog)

    monkeypatch.setattr(
        ai_log_analyzer,
        "chat_with_assistant",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("network failed")),
        raising=False,
    )
    dialog.show()
    dialog.message_input.setPlainText("fail me")
    dialog.send_message()
    assert _wait_until(lambda: "network failed" in _visible_text(dialog))


def test_escape_and_deleted_dialog_suppress_late_chat_result(monkeypatch, request):
    """Escape and deletion must invalidate a request without Qt thread lifetime failures."""
    from waveform_sim.service import ai_log_analyzer

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    request.addfinalizer(release.set)

    def slow_chat(*args, **kwargs):
        started.set()
        release.wait(2)
        finished.set()
        return "late answer"

    monkeypatch.setattr(ai_log_analyzer, "chat_with_assistant", slow_chat, raising=False)
    dialog = _dialog()
    dialog.show()
    dialog.message_input.setPlainText("escape me")
    dialog.send_message()
    assert started.wait(1)
    QTest.keyClick(dialog, Qt.Key_Escape)
    assert not dialog.isVisible()
    assert dialog.messages[-1]["content"] == "已取消本次请求。"
    release.set()
    assert finished.wait(1)
    _process_events()
    assert "late answer" not in _visible_text(dialog)

    second_started = threading.Event()
    second_release = threading.Event()
    request.addfinalizer(second_release.set)

    def delete_chat(*args, **kwargs):
        second_started.set()
        second_release.wait(2)
        return "deleted answer"

    monkeypatch.setattr(ai_log_analyzer, "chat_with_assistant", delete_chat, raising=False)
    dialog = _dialog()
    dialog.message_input.setPlainText("delete me")
    dialog.send_message()
    assert second_started.wait(1)
    dialog.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    second_release.set()
    _process_events()


def test_current_source_refreshes_on_reopen_and_history_source_stays_compact(monkeypatch, tmp_path):
    """Current source refreshes when shown; selected history retains only its visible file name."""
    current = {"text": "first"}
    dialog = _dialog(lambda: current["text"])
    dialog.show()
    dialog.hide()
    current["text"] = "second"
    dialog.show()
    assert dialog._log_text == "second"

    history = tmp_path / "a-very-long-history-name.log"
    history.write_text("historical", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_: (str(history), "Log Files (*.log)"))
    dialog.open_log_file()
    assert str(tmp_path) not in dialog.source_label.text()
    assert history.stem[:8] in dialog.source_label.text()


def test_floating_button_incremental_drag_marks_user_moved_without_opening():
    """Cumulative one-pixel moves must be a drag rather than an accidental click."""
    from waveform_sim.ui.fdidm_ai_analysis import FDIDMAIFloatingButton

    parent = QWidget()
    parent.resize(300, 200)
    parent.move(200, 100)
    opened = []
    button = FDIDMAIFloatingButton(lambda: opened.append(True), parent)
    button.move(100, 100)
    parent.show()
    button.show()
    QApplication.processEvents()
    press_global = button.mapToGlobal(QPoint(12, 12))

    def event(kind, global_pos, button_value, buttons):
        local = button.mapFromGlobal(global_pos)
        return QMouseEvent(kind, QPointF(local), QPointF(local), QPointF(global_pos), button_value, buttons, Qt.NoModifier)

    QApplication.sendEvent(button, event(QEvent.MouseButtonPress, press_global, Qt.LeftButton, Qt.LeftButton))
    for delta in range(1, 21):
        QApplication.sendEvent(button, event(QEvent.MouseMove, press_global + QPoint(delta, 0), Qt.NoButton, Qt.LeftButton))
    QApplication.sendEvent(button, event(QEvent.MouseButtonRelease, press_global + QPoint(20, 0), Qt.LeftButton, Qt.NoButton))

    assert button.user_moved is True
    assert opened == []
    assert button.pos() == QPoint(120, 100)
    parent.resize(64, 64)
    button.clamp_to_parent()
    assert parent.rect().contains(button.geometry())
