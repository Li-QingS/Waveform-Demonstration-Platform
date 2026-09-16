"""UI 运行路径回归测试（离屏；无 PyQt5 环境自动跳过）。"""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading

import pytest
import numpy as np

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


def test_fdidm_time_plot_y_axis_fits_data(app):
    """回归：右下角时间图 Y 轴应按可见数据自动拟合，且无新数据时不重设。"""
    tab = FDIDMTab()
    tab._search_worker = lambda *args, **kwargs: None
    now = time.time()

    def hist(n, ser_ofdm=0.30, ser_otfs=0.84, ser_afdm=0.84, ser_cur=0.37):
        return [
            {
                "kind": "eval",
                "seq": i,
                "ts": now + i * 0.5,
                "ser_ofdm": ser_ofdm,
                "ser_otfs": ser_otfs,
                "ser_afdm": ser_afdm,
                "ser_current": ser_cur,
            }
            for i in range(n)
        ]

    try:
        tab._refresh_time_metric_plot(hist(5))
        y = tab.ber_snr_plot.viewRange()[1]
        # 高 SER 场景（0.30~0.84）应拟合到 -1 附近，而不是固定在 [-6,0] 顶部。
        assert y[0] > -2.0 and y[1] < 0.5

        # 无新数据时视图不重设。
        tab._refresh_time_metric_plot(hist(5))
        y2 = tab.ber_snr_plot.viewRange()[1]
        assert abs(y2[0] - y[0]) < 1e-9 and abs(y2[1] - y[1]) < 1e-9

        # 数据明显越界（SER 降至 1e-4 量级）时重新拟合。
        tab._refresh_time_metric_plot(hist(10, 1e-4, 3e-4, 2e-4, 1e-4))
        y4 = tab.ber_snr_plot.viewRange()[1]
        assert y4[1] < -2.5
    finally:
        tab._on_qt_destroyed()


def test_fdidm_time_plot_curves_are_smoothed(app):
    """回归：右下角时间图应显示滑动平均后的平稳曲线，而不是逐点剧烈抖动。"""
    tab = FDIDMTab()
    tab._search_worker = lambda *args, **kwargs: None
    now = time.time()
    # 构造一条逐点大幅跳动的原始 SER 序列（模拟块边界/瞬时信道变化）。
    ser_vals = [0.12, 0.78, 0.15, 0.82, 0.11, 0.79, 0.14, 0.81, 0.12, 0.80]
    history = [
        {
            "kind": "eval",
            "seq": i,
            "ts": now + i * 0.5,
            "ser_ofdm": ser_vals[i % len(ser_vals)],
            "ser_otfs": 0.84,
            "ser_afdm": 0.84,
            "ser_current": ser_vals[i % len(ser_vals)],
        }
        for i in range(14)
    ]
    try:
        tab._refresh_time_metric_plot(history)
        ys = np.asarray(tab._time_metric_curves["OFDM"].yData, dtype=float)
        ys = ys[np.isfinite(ys) & (ys > 0)]
        assert ys.size >= 6
        d = np.abs(np.diff(np.log10(ys)))
        # 平滑后曲线相邻点不应剧烈跳动（原始输入在 0.11~0.82 间反复横跳）。
        assert np.median(d) < 0.15
        assert d.max() < 0.6
    finally:
        tab._on_qt_destroyed()


def test_fdidm_sweep_preserves_selected_seed_and_fair_reference(app, monkeypatch):
    """The comparison keeps the selected realization, even if it is not flattering."""
    from waveform_sim.simulation import simple_fdidm_rx

    original = simple_fdidm_rx.FDIDMTransceiver
    created = []

    def recording_factory(**kwargs):
        created.append(dict(kwargs))
        return original(**kwargs)

    monkeypatch.setattr(simple_fdidm_rx, "FDIDMTransceiver", recording_factory)
    tab = FDIDMTab()
    tab._search_worker = lambda *args, **kwargs: None
    tab.SER_SNR_POINTS = (10.0,)
    emitted = []
    tab._emit_signal_safe = lambda name, *args: emitted.append((name, args)) or True
    base = dict(
        alpha=0.0, beta=0.0, m_subcarriers=4, n_symbols=4,
        subcarrier_spacing_hz=300e3, mod_order="16QAM", channel_model="TDL-C",
        velocity_kmh=28080, doppler_radial_factor=0.10, decoder="ZF", snr_db=10.0,
        snr_definition="Eb/N0", optimize_indices=False, search_step=0.1,
        fc_hz=20e9, link_mode="matrix", random_channel=True, channel_seed=73,
        dynamic_channel=True, channel_dynamics="block",
        channel_coherence_frames=8, fast_channel_coherence_symbols=1,
        tf_notch_depth_db=0.0, tf_notch_count=0,
    )
    try:
        tab._theory_snr_worker(tab._auto_scan_token, base, threading.Event(), 0.0, 0.0)
        assert created and created[0]["channel_seed"] == 73
        assert created[0]["dynamic_channel"] is False
        assert created[0]["channel_dynamics"] == "fixed"
        points = {args[1]: float(args[3]) for name, args in emitted if name == "ser_snr_point"}
        assert set(points) == {"OFDM", "OTFS", "AFDM", "FDIDM"}
        assert points["FDIDM"] <= points["OFDM"] + 1e-12
    finally:
        tab._on_qt_destroyed()
