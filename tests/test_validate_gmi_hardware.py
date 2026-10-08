from pathlib import Path

import pytest

from scripts.validate_gmi_hardware import HardwareValidationConfig, run_validation
from waveform_sim.hardware.device_manager import DeviceInfo


class Clock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += max(float(seconds), 0.01)


class FakeBackend:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.running = False
        self.stopped = False
        self.frames = 0
        self.applied = False
        self.__class__.instances.append(self)

    def start(self):
        self.running = True

    def stop(self):
        self.running = False
        self.stopped = True

    def wait(self, timeout=3.0):
        return None

    def apply_gmi_candidate(self):
        self.applied = True
        return {"state": "a1_applying"}

    def get_status(self):
        self.frames += 1
        enabled = bool(self.kwargs["gmi_adaptive_enable"])
        receiver = self.kwargs["gmi_receiver_mode"]
        return {
            "frames_processed": self.frames,
            "frames_decode_ok": self.frames,
            "crc_ok": True,
            "data_aided_evm_percent": 4.0,
            "evm_valid": True,
            "measured_ser": 0.0,
            "rx_overflow_count": 0,
            "rx_processing_gap_count": 0,
            "rx_probe_continuity": "contiguous",
            "gmi_resource_id": "res-common",
            "gmi_receiver_mode": receiver,
            "gmi_current_token_id": "token-a",
            "gmi_candidate_token_id": "token-b" if enabled else "",
            "gmi_gain_lower_nats": 0.03 if receiver == "diagonal" and enabled else 0.0,
            "gmi_adaptation": {
                "enabled": enabled,
                "receiver_mode": receiver,
                "state": "tracking" if enabled else "disabled",
                "state_epoch": self.frames if enabled else -1,
                "reason": "reference_flat" if receiver == "full_matched" else "current_token_is_robust_best",
                "scores": [{"valid": enabled}],
                "switch_allowed": receiver == "diagonal" and enabled and not self.applied,
            },
            "gmi_evidence": {},
            "adaptive_validation_state": "regressed" if self.applied else "idle",
            "adaptive_validation": {"outcome": "regressed"} if self.applied else {},
        }


def test_bounded_hardware_validation_stops_all_modes_and_keeps_common_resource(tmp_path):
    FakeBackend.instances = []
    clock = Clock()
    config = HardwareValidationConfig(
        serial="FAKE123",
        duration_sec=1.0,
        tx_gain_db=10.0,
        rx_gain_db=20.0,
        output=tmp_path / "evidence.json",
        poll_sec=0.25,
    )
    report = run_validation(
        config,
        backend_factory=FakeBackend,
        device_probe=lambda timeout: [DeviceInfo(True, "USRP B210", "serial: FAKE123")],
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert report["summary"]["passed"] is True
    assert report["summary"]["evidence_ready"] is True
    assert report["summary"]["selection_ready"] is True
    assert report["summary"]["closed_loop_complete"] is True
    assert report["summary"]["resource_ids"] == ["res-common"]
    assert [item["mode"] for item in report["segments"]] == ["fixed", "diagonal", "full_matched"]
    assert all(instance.stopped for instance in FakeBackend.instances)
    assert FakeBackend.instances[1].applied is True
    assert all(item["summary"]["overflow_delta"] == 0 for item in report["segments"])
    assert all(item["summary"]["evm_evidence_samples"] > 0 for item in report["segments"])


def test_gain_above_project_limit_is_rejected_before_device_probe(tmp_path):
    config = HardwareValidationConfig(
        serial="FAKE123",
        duration_sec=1.0,
        tx_gain_db=45.1,
        rx_gain_db=20.0,
        output=Path(tmp_path) / "evidence.json",
    )
    with pytest.raises(ValueError, match="TX gain"):
        run_validation(config, device_probe=lambda timeout: [])


def test_started_but_unfinished_abba_does_not_pass(tmp_path):
    class StuckValidationBackend(FakeBackend):
        def get_status(self):
            status = super().get_status()
            if self.applied:
                status["adaptive_validation_state"] = "a1_settling"
                status["adaptive_validation"] = {"state": "a1_settling"}
            return status

    StuckValidationBackend.instances = []
    clock = Clock()
    config = HardwareValidationConfig(
        serial="FAKE123",
        duration_sec=1.0,
        tx_gain_db=10.0,
        rx_gain_db=20.0,
        output=tmp_path / "evidence.json",
        poll_sec=0.25,
    )
    report = run_validation(
        config,
        backend_factory=StuckValidationBackend,
        device_probe=lambda timeout: [DeviceInfo(True, "USRP B210", "serial: FAKE123")],
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert report["summary"]["closed_loop_complete"] is False
    assert report["summary"]["passed"] is False


def test_terminal_live_swap_null_evm_keeps_prior_valid_evidence(tmp_path):
    class SwapPauseBackend(FakeBackend):
        def get_status(self):
            status = super().get_status()
            if self.applied:
                status["data_aided_evm_percent"] = None
                status["evm_valid"] = False
            else:
                status["evm_valid"] = True
            return status

    SwapPauseBackend.instances = []
    clock = Clock()
    config = HardwareValidationConfig(
        serial="FAKE123",
        duration_sec=1.0,
        tx_gain_db=10.0,
        rx_gain_db=20.0,
        output=tmp_path / "evidence.json",
        poll_sec=0.25,
    )
    report = run_validation(
        config,
        backend_factory=SwapPauseBackend,
        device_probe=lambda timeout: [DeviceInfo(True, "USRP B210", "serial: FAKE123")],
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    diagonal = report["segments"][1]["summary"]
    assert diagonal["data_aided_evm_percent"] == 4.0
    assert diagonal["evm_evidence_samples"] > 0
    assert report["summary"]["passed"] is True
