import importlib
import json
import threading
import time

import numpy as np

from waveform_sim.gmi.model import ChannelState, ExperimentResources, ReceiverSpec
from waveform_sim.hardware.evidence import AcquisitionFingerprint, PilotFitMetrics
from waveform_sim.hardware.fdidm_adaptive import GMIAdaptiveController


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def resources(cap=4):
    return ExperimentResources(4, catalogue_cap=cap, grid_rows=4, grid_cols=1, sample_rate_hz=1e6, bandwidth_hz=1e6)


def channel(epoch=1, **overrides):
    values = dict(
        context_id="ctx",
        epoch=epoch,
        timestamp=float(epoch),
        h_tf=np.diag([1.9, 1.4, 1.0, 0.6]),
        noise_variance=1.0,
        sync_metric=0.9,
        pilot_nmse=0.02,
        pilot_power=0.1,
        continuity=True,
    )
    values.update(overrides)
    return ChannelState(**values)


def test_controller_consumes_latest_state_and_stops_cleanly():
    controller = GMIAdaptiveController(resources(), ReceiverSpec("diagonal"), current_alpha=0.0, current_beta=0.0)
    controller.start()
    for epoch in range(1, 8):
        controller.publish(channel(epoch))
    assert wait_until(lambda: controller.status()["state_epoch"] == 7)
    status = controller.status()
    assert status["library_size"] <= status["catalogue_cap"] == 4
    assert status["resource_id"]
    assert status["library_id"]
    controller.stop()
    assert controller.status()["state"] == "stopped"


def test_controller_wait_state_and_manual_reevaluation_reuse_last_csi():
    controller = GMIAdaptiveController(resources(), ReceiverSpec("diagonal"), current_alpha=0.0, current_beta=0.0)
    controller.start()
    time.sleep(0.3)
    assert controller.status()["state"] == "waiting_channel"
    controller.publish(channel(1))
    assert wait_until(lambda: controller.status()["state_epoch"] == 1)
    first_sequence = controller.status()["sequence"]
    assert controller.request_evaluation() is True
    assert wait_until(lambda: controller.status()["sequence"] > first_sequence and controller.status()["state"] == "tracking")
    controller.stop()


def test_newer_csi_supersedes_a_result_already_being_scored():
    controller = GMIAdaptiveController(resources(), ReceiverSpec("diagonal"), current_alpha=0.0, current_beta=0.0)
    entered = threading.Event()
    release = threading.Event()
    original = controller._selector.select

    def delayed(*args, **kwargs):
        entered.set()
        release.wait(2.0)
        return original(*args, **kwargs)

    controller._selector.select = delayed
    controller.start()
    controller.publish(channel(1))
    assert entered.wait(1.0)
    controller.publish(channel(2))
    release.set()
    assert wait_until(lambda: controller.status()["state_epoch"] == 2)
    controller.stop()


def test_stop_during_scoring_cannot_publish_a_late_candidate():
    controller = GMIAdaptiveController(resources(), ReceiverSpec("diagonal"), current_alpha=0.0, current_beta=0.0)
    entered = threading.Event()
    release = threading.Event()
    original = controller._selector.select

    def delayed(*args, **kwargs):
        entered.set()
        release.wait(2.0)
        return original(*args, **kwargs)

    controller._selector.select = delayed
    controller.start()
    controller.publish(channel(1))
    assert entered.wait(1.0)
    controller.stop(timeout=0.01)
    assert controller.status()["state"] == "stopped"
    release.set()
    time.sleep(0.1)
    assert controller.status()["state"] == "stopped"


def test_validation_public_state_recursively_serializes_fingerprints():
    fingerprint = AcquisitionFingerprint(
        effective_cfo_hz=1.0,
        sync_metric=0.9,
        residual_cfo_hz=0.1,
        pilot_nmse=0.02,
        pilot_fitted_power=0.1,
        rx_rms=0.2,
        rx_peak=0.4,
        continuity=True,
        overflow=False,
        acquisition_epoch=3,
    )
    mod = importlib.import_module("waveform_sim.hardware.fdidm_adaptive")
    payload = mod.FDIDMAdaptiveMixin._validation_public_state(
        {"acquisition_fingerprints": [fingerprint], "value": float("nan")}
    )
    assert payload["acquisition_fingerprints"][0]["acquisition_epoch"] == 3
    assert payload["value"] is None
    json.dumps(payload, allow_nan=False)


def test_full_matched_controller_is_reference_flat():
    controller = GMIAdaptiveController(resources(), ReceiverSpec("full_matched", None), current_alpha=0.0, current_beta=0.0)
    controller.start()
    controller.publish(channel())
    assert wait_until(lambda: controller.status()["state"] == "tracking")
    status = controller.status()
    assert status["reason"] == "reference_flat"
    assert not status["switch_allowed"]
    controller.stop()


def test_backend_exposes_separate_estimator_receiver_and_gmi_status(monkeypatch):
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    monkeypatch.setattr(mod._LegacyFDIDMHardwareTest, "_build_top_block", lambda self: None)
    backend = mod._LegacyFDIDMHardwareTest(
        fdidm_m=8,
        fdidm_n=8,
        mod_order="64QAM",
        coding_scheme="none",
        tx_frame_count=1,
        tx_min_waveform_duration_ms=1,
        channel_estimator="diag_tf",
        gmi_adaptive_enable=True,
        gmi_receiver_mode="diagonal",
        gmi_catalogue_cap=4,
        log_to_stdout=False,
    )
    backend._rx_probe_continuity = "contiguous"
    backend._rx_probe_generation = 9
    backend._rx_last_overflow_samples = 0
    backend.last_cfo_est_hz = 12.0
    backend._last_pilot_fit_metrics = PilotFitMetrics(fit_nmse=0.02, fitted_power=0.1)
    assert backend._publish_gmi_channel_state(
        h_tf_est=np.linspace(0.6, 1.9, 64),
        htf_kind="diag",
        noise_var=1.0,
        sync_metric=0.9,
        csi_trustworthy=True,
        processing_gap=False,
    )
    assert not backend._publish_gmi_channel_state(
        h_tf_est=np.linspace(0.6, 1.9, 64),
        htf_kind="diag",
        noise_var=1.0,
        sync_metric=0.9,
        csi_trustworthy=True,
        processing_gap=False,
    )
    assert wait_until(lambda: backend.get_gmi_adaptation_status().get("state_epoch") == 9)
    status = backend.get_status()
    assert status["effective_channel_estimator"] == "diag_tf"
    assert status["gmi_receiver_mode"] == "diagonal"
    assert status["gmi_resource_id"]
    assert status["gmi_library_id"]
    assert "scores" in status["gmi_adaptation"]
    assert status["gmi_evidence"]["theory"]["resource_id"] == status["gmi_resource_id"]
    assert status["gmi_evidence"]["selection"]["state_epoch"] == 9
    assert status["gmi_evidence"]["legacy_ser"]["diagnostic_only"] is True
    backend._gmi_controller.stop()


def test_fixed_m1_and_adaptive_m16_share_the_common_control_budget(monkeypatch):
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    monkeypatch.setattr(mod._LegacyFDIDMHardwareTest, "_build_top_block", lambda self: None)
    common = dict(
        fdidm_m=8,
        fdidm_n=8,
        mod_order="QPSK",
        coding_scheme="none",
        tx_frame_count=1,
        tx_min_waveform_duration_ms=1,
        gmi_receiver_mode="diagonal",
        gmi_catalogue_cap=16,
        tx_text="OK",
        log_to_stdout=False,
    )
    fixed = mod._LegacyFDIDMHardwareTest(gmi_adaptive_enable=False, **common)
    adaptive = mod._LegacyFDIDMHardwareTest(gmi_adaptive_enable=True, **common)
    fixed_status = fixed.get_status()["gmi_adaptation"]
    adaptive_status = adaptive.get_status()["gmi_adaptation"]
    assert fixed_status["active_library_size"] == 1
    assert adaptive_status["active_library_size"] == 16
    assert fixed_status["resource_id"] == adaptive_status["resource_id"]
    assert fixed._gmi_resources().control_bits == adaptive._gmi_resources().control_bits == 4
    fixed._gmi_controller.stop()
    adaptive._gmi_controller.stop()


def test_backend_processing_gap_state_fails_closed_without_false_overflow(monkeypatch):
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    monkeypatch.setattr(mod._LegacyFDIDMHardwareTest, "_build_top_block", lambda self: None)
    backend = mod._LegacyFDIDMHardwareTest(
        fdidm_m=8,
        fdidm_n=8,
        mod_order="64QAM",
        coding_scheme="none",
        tx_frame_count=1,
        tx_min_waveform_duration_ms=1,
        gmi_adaptive_enable=True,
        gmi_catalogue_cap=4,
        log_to_stdout=False,
    )
    backend._rx_probe_continuity = "gap"
    backend._rx_probe_generation = 2
    backend._rx_last_overflow_samples = 100
    backend.last_cfo_est_hz = 0.0
    backend._last_pilot_fit_metrics = PilotFitMetrics(fit_nmse=0.02, fitted_power=0.1)
    backend._publish_gmi_channel_state(
        h_tf_est=np.linspace(0.6, 1.9, 64),
        htf_kind="diag",
        noise_var=1.0,
        sync_metric=0.9,
        csi_trustworthy=False,
        processing_gap=True,
    )
    assert wait_until(lambda: backend.get_gmi_adaptation_status().get("state_epoch") == 2)
    status = backend.get_gmi_adaptation_status()
    assert not status["switch_allowed"]
    assert status["reason"] == "probe_gap"
    backend._gmi_controller.stop()


def test_active_library_resource_contract_is_immutable_during_runtime(monkeypatch):
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    monkeypatch.setattr(mod._LegacyFDIDMHardwareTest, "_build_top_block", lambda self: None)
    backend = mod._LegacyFDIDMHardwareTest(
        fdidm_m=8,
        fdidm_n=8,
        mod_order="64QAM",
        coding_scheme="none",
        tx_frame_count=1,
        tx_min_waveform_duration_ms=1,
        gmi_adaptive_enable=True,
        gmi_catalogue_cap=4,
        log_to_stdout=False,
    )
    initial = backend._gmi_resources()
    backend._analog_bandwidth_actual = initial.bandwidth_hz * 1.5

    assert backend._gmi_resources() is initial
    assert backend._gmi_resources().resource_id == backend._gmi_controller.resources.resource_id
    backend._gmi_controller.stop()


def test_historical_overflow_counter_does_not_latch_future_gmi_state(monkeypatch):
    mod = importlib.import_module("waveform_sim.hardware.fdidm_hardtest")
    monkeypatch.setattr(mod._LegacyFDIDMHardwareTest, "_build_top_block", lambda self: None)
    backend = mod._LegacyFDIDMHardwareTest(
        fdidm_m=8,
        fdidm_n=8,
        mod_order="64QAM",
        coding_scheme="none",
        tx_frame_count=1,
        tx_min_waveform_duration_ms=1,
        gmi_adaptive_enable=True,
        gmi_catalogue_cap=4,
        log_to_stdout=False,
    )
    backend._rx_probe_continuity = "contiguous"
    backend._rx_probe_generation = 3
    backend._rx_last_overflow_samples = 100  # previous window only
    backend.last_cfo_est_hz = 0.0
    backend._last_pilot_fit_metrics = PilotFitMetrics(fit_nmse=0.02, fitted_power=0.1)
    assert backend._publish_gmi_channel_state(
        h_tf_est=np.linspace(0.6, 1.9, 64),
        htf_kind="diag",
        noise_var=1.0,
        sync_metric=0.9,
        csi_trustworthy=True,
        processing_gap=False,
    )
    assert wait_until(lambda: backend.get_gmi_adaptation_status().get("state_epoch") == 3)
    assert backend.get_gmi_adaptation_status()["reason"] != "overflow"
    backend._gmi_controller.stop()
