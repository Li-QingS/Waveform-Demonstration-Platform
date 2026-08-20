"""UI 运行路径回归测试（离屏；无 PyQt5 环境自动跳过）。"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from PyQt5.QtWidgets import QApplication

from waveform_sim.ui.fdidm_tab import FDIDMTab


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def test_fdidm_auto_refresh_floor_no_shadow(app):
    """回归：alpha_ser_floor 局部变量不得遮蔽模块函数（曾抛 UnboundLocalError）。"""
    tab = FDIDMTab()
    tab._search_worker = lambda *args, **kwargs: None
    try:
        tab._start_auto_ser_snr_refresh()
    finally:
        tab._search_stop.set()
        tab._on_qt_destroyed()


def test_fdidm_alpha_ser_worker_floor_no_shadow(app):
    """回归：_alpha_ser_worker 内 floor 计算不得抛 UnboundLocalError。"""
    tab = FDIDMTab()
    stop = threading.Event()
    stop.set()
    base_kwargs = tab._backend_kwargs()
    emitted = []
    tab._emit_signal_safe = lambda name, *args: emitted.append((name, args)) or True
    try:
        tab._alpha_ser_worker(
            token=0,
            base_kwargs=base_kwargs,
            stop_event=stop,
            alpha_sweep_frames=10,
            alpha_curve_mode="theory",
            best_a=0.0,
            best_b=0.0,
        )
    finally:
        tab._on_qt_destroyed()
    finished = [args for name, args in emitted if name == "alpha_beta_finished"]
    assert finished, "worker 未发出完成事件"
    reason = finished[-1][1]
    assert not str(reason).startswith("失败"), f"worker 失败：{reason}"
