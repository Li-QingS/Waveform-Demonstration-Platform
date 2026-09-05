"""离屏验证 FDIDM 单 USRP baseline/adaptive 对比会话。"""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from PyQt5.QtWidgets import QApplication

from waveform_sim.ui.fdidm_hardware_test_tab import FDIDMHardwareTestTab


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class FakeBackend:
    def __init__(self):
        self.frames_processed = 0
        self.frames_decode_ok = 0
        self.configures = []

    def start(self):
        return None

    def configure(self, **kwargs):
        self.configures.append(kwargs)

    def get_status(self):
        return {
            "frames_processed": self.frames_processed,
            "frames_decode_ok": self.frames_decode_ok,
            "match_ratio": 0.9,
            "measured_ser": 0.2,
            "fec_bit_ber": 0.01,
            "evm_average_percent": 8.0,
            "alpha": 0.5,
            "beta": 1.0,
        }

    def get_decode_stats(self):
        return {"match_ratio": 0.9, "decode_ok": True}


def test_comparison_controls_and_default_state(app):
    tab = FDIDMHardwareTestTab()
    assert tab.comparison_session.phase == "idle"
    assert tab.btn_start_comparison.isEnabled()
    assert not tab.btn_stop_comparison.isEnabled()


def test_comparison_phase_sequence_and_delta_metrics(app):
    tab = FDIDMHardwareTestTab()
    fake = FakeBackend()
    tab.backend = fake
    tab.comparison_duration_spin.setValue(2)
    tab.comparison_repeat_spin.setValue(1)
    tab._start_comparison_demo()
    assert tab.comparison_session.active
    assert tab.comparison_session.phase == "baseline"
    assert fake.configures[-1]["adaptive_alpha_beta_enable"] is False

    fake.frames_processed = 10
    fake.frames_decode_ok = 8
    tab._sample_comparison_metrics(fake.get_status(), fake.get_decode_stats())
    tab.comparison_session.phase_started_at = time.monotonic() - 3
    tab._comparison_tick(fake.get_status(), fake.get_decode_stats())
    assert tab.comparison_session.phase == "adaptive"
    assert tab.comparison_session.records[0].frames_processed == 10
    assert fake.configures[-1]["adaptive_alpha_beta_enable"] is True

    fake.frames_processed = 20
    fake.frames_decode_ok = 19
    tab._sample_comparison_metrics(fake.get_status(), fake.get_decode_stats())
    tab.comparison_session.phase_started_at = time.monotonic() - 3
    tab._comparison_tick(fake.get_status(), fake.get_decode_stats())
    assert tab.comparison_session.phase == "complete"
    assert len(tab.comparison_session.records) == 2
    assert tab.comparison_session.summary.conclusion_available
    assert "Baseline" in tab._comparison_summary_text()


def test_manual_adaptive_observation_records_before_after(app):
    tab = FDIDMHardwareTestTab()
    fake = FakeBackend()
    tab.backend = fake
    tab.test_running = True
    tab.adaptive_enable_check.setChecked(False)
    tab._reset_adaptive_observation(False)
    for _ in range(3):
        fake.frames_processed += 10
        fake.frames_decode_ok += 7
        tab._sample_adaptive_observation(fake.get_status(), fake.get_decode_stats())
    tab.adaptive_enable_check.setChecked(True)
    assert tab._adaptive_observation_state is True
    for _ in range(3):
        fake.frames_processed += 10
        fake.frames_decode_ok += 10
        status = fake.get_status()
        status.update(measured_ser=0.05, evm_average_percent=4.0, match_ratio=1.0)
        tab._sample_adaptive_observation(status, fake.get_decode_stats())
    tab._finalize_adaptive_observation_segment()
    records = tab.get_adaptive_observation_records()
    assert len(records) == 2
    assert records[0]["adaptive_enabled"] is False
    assert records[1]["adaptive_enabled"] is True
    assert records[1]["evm_average_percent"] < records[0]["evm_average_percent"]
