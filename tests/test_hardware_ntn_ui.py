import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from PyQt5.QtWidgets import QApplication

from waveform_sim.ui.hardware_test_tab import HardwareTestTab


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(app):
    tab = HardwareTestTab()
    yield tab
    tab.close()


def test_ofdm_ntn_controls_default_to_single_point_and_snapshot_parameters(page):
    assert page.ntn_enabled_check.isChecked() is False
    assert page.ntn_mode_combo.currentData() == "single"

    page.ntn_enabled_check.setChecked(True)
    page.ntn_profile_combo.setCurrentText("NTN-TDLC5-1200")
    page.ntn_velocity_spin.setValue(28_080)
    page.ntn_radial_spin.setValue(0.5)
    page.ntn_cfo_spin.setValue(300)
    page.ntn_delay_scale_spin.setValue(1.5)

    kwargs = page._constructor_kwargs("OFDM")
    assert kwargs["ntn_enabled"] is True
    assert kwargs["ntn_profile"] == "NTN-TDLC5-1200"
    assert kwargs["ntn_doppler_spread_hz"] == 1200
    assert kwargs["ntn_velocity_kmh"] == 28_080
    assert kwargs["ntn_radial_projection"] == pytest.approx(0.5)
    assert kwargs["ntn_residual_cfo_hz"] == pytest.approx(300)
    assert kwargs["ntn_delay_scale"] == pytest.approx(1.5)

    config = page._build_hardware_run_config(kwargs)
    assert config.waveform.waveform == "OFDM"
    assert config.waveform.ntn_profile == "NTN-TDLC5-1200"
    assert config.hardware.transport == "uhd"
    assert config.hardware.mode == "rf"


def test_ntn_scan_axis_uses_bounded_parameter_range_and_fixed_frame_count(page):
    page.ntn_enabled_check.setChecked(True)
    page.ntn_mode_combo.setCurrentIndex(page.ntn_mode_combo.findData("scan"))
    page.ntn_scan_parameter_combo.setCurrentIndex(
        page.ntn_scan_parameter_combo.findData("ntn_doppler_spread_hz")
    )
    page.ntn_scan_points_spin.setValue(5)
    page.ntn_scan_frames_spin.setValue(12)

    plan = page._current_ntn_scan_plan()
    assert plan.values() == [200.0, 450.0, 700.0, 950.0, 1200.0]
    assert plan.frames_per_point == 12


def test_other_waveforms_do_not_receive_ofdm_ntn_constructor_options(page):
    kwargs = page._constructor_kwargs("OTFS")
    assert not any(key.startswith("ntn_") for key in kwargs)
