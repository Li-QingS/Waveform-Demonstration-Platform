#!/usr/bin/env python3
"""Run a bounded fixed/GMI/full-matched FDIDM hardware validation.

The script never chooses RF gains implicitly: both gains are required command
line inputs and are checked against the project's host-side 45 dB ceiling.
Every segment owns one backend and always stops it in ``finally``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from waveform_sim.hardware.device_manager import DeviceInfo, probe_devices
from waveform_sim.hardware.fdidm_hardtest import (
    FDIDM_GAIN_MAX_DB,
    FDIDMHardwareTest,
    validate_fdidm_gain,
)


SCHEMA_VERSION = 1
MODES = ("fixed", "diagonal", "full_matched")


@dataclass(frozen=True)
class HardwareValidationConfig:
    serial: str
    duration_sec: float
    tx_gain_db: float
    rx_gain_db: float
    output: Path
    carrier_freq_hz: float = 2.4e9
    sample_rate_hz: float = 500_000.0
    poll_sec: float = 0.5
    catalogue_cap: int = 16
    robustness_profile: str = "standard"
    validation_grace_sec: float = 75.0
    m: int = 16
    n: int = 16

    def validate(self) -> None:
        validate_fdidm_gain(self.tx_gain_db, name="TX gain")
        validate_fdidm_gain(self.rx_gain_db, name="RX gain")
        if not str(self.serial).strip():
            raise ValueError("serial must be non-empty")
        if not math.isfinite(self.duration_sec) or not 1.0 <= self.duration_sec <= 300.0:
            raise ValueError("duration must be in [1, 300] seconds per mode")
        if not math.isfinite(self.poll_sec) or not 0.1 <= self.poll_sec <= 5.0:
            raise ValueError("poll interval must be in [0.1, 5] seconds")
        if self.catalogue_cap not in {1, 4, 16, 25, 36, 49, 64}:
            raise ValueError("catalogue cap must be one of 1,4,16,25,36,49,64")
        if self.robustness_profile not in {"fast", "standard", "conservative"}:
            raise ValueError("robustness profile must be fast, standard, or conservative")
        if not math.isfinite(self.validation_grace_sec) or not 0.0 <= self.validation_grace_sec <= 180.0:
            raise ValueError("validation grace must be in [0, 180] seconds")
        if self.m < 4 or self.n < 1 or self.m * self.n > 4096:
            raise ValueError("invalid FDIDM M/N")


def _safe(value: Any) -> Any:
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _safe(value.as_dict())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _device_report(devices: Sequence[DeviceInfo], serial: str) -> Dict[str, Any]:
    rows = [item.to_dict() for item in devices]
    available = any(bool(item.available) for item in devices)
    serial_seen = any(str(serial) in str(item.detail) for item in devices if item.available)
    return {
        "available": available,
        "serial_requested": str(serial),
        "serial_seen": serial_seen,
        "devices": rows,
    }


def _snapshot(status: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
    return _safe({
        "elapsed_sec": float(elapsed),
        "frames_processed": int(status.get("frames_processed", 0) or 0),
        "frames_decode_ok": int(status.get("frames_decode_ok", 0) or 0),
        "crc_ok": bool(status.get("crc_ok", status.get("decode_ok", False))),
        "reason": str(status.get("reason", "")),
        "preamble_reliable": bool(status.get("preamble_reliable", False)),
        "sync_metric": status.get("sync_metric"),
        "cfo_est_hz": status.get("cfo_est_hz"),
        "data_aided_evm_percent": status.get("data_aided_evm_percent"),
        "evm_average_percent": status.get("evm_average_percent"),
        "evm_valid": bool(status.get("evm_valid", False)),
        "measured_ser": status.get("measured_ser"),
        "pilot_fit_nmse": status.get("pilot_fit_nmse"),
        "pilot_fitted_power": status.get("pilot_fitted_power"),
        "rx_input_rms": status.get("rx_input_rms"),
        "rx_input_peak": status.get("rx_input_peak"),
        "rx_last_new_samples": int(status.get("rx_last_new_samples", 0) or 0),
        "rx_stream_updates": int(status.get("rx_stream_updates", 0) or 0),
        "rx_spectrum_stale": bool(status.get("rx_spectrum_stale", True)),
        "rx_probe_generation": int(status.get("rx_probe_generation", 0) or 0),
        "rx_acquisition_epoch": int(status.get("rx_acquisition_epoch", 0) or 0),
        "rx_overflow_count": int(status.get("rx_overflow_count", 0) or 0),
        "rx_processing_gap_count": int(status.get("rx_processing_gap_count", 0) or 0),
        "rx_probe_continuity": str(status.get("rx_probe_continuity", "unknown")),
        "rx_probe_continuity_reason": str(status.get("rx_probe_continuity_reason", "")),
        "resource_id": str(status.get("gmi_resource_id", "")),
        "receiver_mode": str(status.get("gmi_receiver_mode", "")),
        "current_token_id": str(status.get("gmi_current_token_id", "")),
        "candidate_token_id": str(status.get("gmi_candidate_token_id", "")),
        "gain_lower_nats": status.get("gmi_gain_lower_nats"),
        "gmi_adaptation": dict(status.get("gmi_adaptation", {}) or {}),
        "gmi_evidence": dict(status.get("gmi_evidence", {}) or {}),
        "adaptive_validation_state": str(status.get("adaptive_validation_state", "")),
        "adaptive_validation": dict(status.get("adaptive_validation", {}) or {}),
    })


def _backend_kwargs(config: HardwareValidationConfig, mode: str) -> Dict[str, Any]:
    enabled = mode != "fixed"
    receiver = "diagonal" if mode == "fixed" else mode
    return {
        "serial": config.serial,
        "carrier_freq": config.carrier_freq_hz,
        "samp_rate": config.sample_rate_hz,
        "tx_gain": config.tx_gain_db,
        "rx_gain": config.rx_gain_db,
        "tx_antenna": "TX/RX",
        "rx_antenna": "RX2",
        "fdidm_m": config.m,
        "fdidm_n": config.n,
        "channel_estimator": "diag_tf",
        "adaptive_alpha_beta_enable": False,
        "gmi_adaptive_enable": enabled,
        "gmi_receiver_mode": receiver,
        "gmi_catalogue_cap": config.catalogue_cap,
        "gmi_robustness_profile": config.robustness_profile,
        "process_interval_ms": 300.0,
        "tx_min_waveform_duration_ms": 500.0,
        "log_to_stdout": False,
    }


def _run_segment(
    config: HardwareValidationConfig,
    mode: str,
    *,
    backend_factory: Callable[..., Any],
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> Dict[str, Any]:
    backend: Optional[Any] = None
    snapshots: List[Dict[str, Any]] = []
    apply_result: Optional[Dict[str, Any]] = None
    error = ""
    stopped = False
    started = monotonic()
    try:
        backend = backend_factory(**_backend_kwargs(config, mode))
        backend.start()
        started = monotonic()
        deadline = started + config.duration_sec
        validation_deadline = deadline + config.validation_grace_sec
        applied = False
        while True:
            now = monotonic()
            if now >= deadline:
                if not (mode == "diagonal" and applied and now < validation_deadline):
                    break
            status = dict(backend.get_status())
            snapshots.append(_snapshot(status, monotonic() - started))
            gmi = dict(status.get("gmi_adaptation", {}) or {})
            if mode == "diagonal" and not applied and bool(gmi.get("switch_allowed", False)):
                apply_result = _safe(dict(backend.apply_gmi_candidate()) or {})
                applied = True
            validation_state = str(status.get("adaptive_validation_state", ""))
            if (
                mode == "diagonal"
                and applied
                and monotonic() >= deadline
                and validation_state in {"improved", "regressed", "inconclusive", "aborted", "rollback_failed"}
            ):
                break
            active_deadline = validation_deadline if mode == "diagonal" and applied else deadline
            sleep(min(config.poll_sec, max(0.0, active_deadline - monotonic())))
        status = dict(backend.get_status())
        snapshots.append(_snapshot(status, monotonic() - started))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if backend is not None:
            try:
                backend.stop()
                backend.wait(timeout=3.0)
                stopped = True
            except Exception as exc:
                suffix = f"stop failed: {type(exc).__name__}: {exc}"
                error = f"{error}; {suffix}" if error else suffix
    first = snapshots[0] if snapshots else {}
    last = snapshots[-1] if snapshots else {}
    usable_selections = []
    for snapshot in snapshots:
        adaptation = dict(snapshot.get("gmi_adaptation", {}) or {})
        scores = list(adaptation.get("scores", []) or [])
        state = str(adaptation.get("state", ""))
        if (
            int(adaptation.get("state_epoch", -1)) >= 0
            and state not in {"", "waiting_channel", "queued", "scoring", "error"}
            and any(bool(score.get("valid", False)) for score in scores)
        ):
            usable_selections.append(adaptation)
    selection = usable_selections[-1] if usable_selections else dict(last.get("gmi_adaptation", {}) or {})
    evidence_snapshots = []
    for snapshot in snapshots:
        value = snapshot.get("data_aided_evm_percent")
        if (
            bool(snapshot.get("evm_valid", False))
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        ):
            evidence_snapshots.append(snapshot)
    evidence_snapshot = evidence_snapshots[-1] if evidence_snapshots else {}
    evm_values = np.asarray(
        [float(snapshot["data_aided_evm_percent"]) for snapshot in evidence_snapshots],
        dtype=np.float64,
    )
    overflow_delta = int(last.get("rx_overflow_count", 0) or 0) - int(first.get("rx_overflow_count", 0) or 0)
    gap_delta = int(last.get("rx_processing_gap_count", 0) or 0) - int(first.get("rx_processing_gap_count", 0) or 0)
    validation_state = str(last.get("adaptive_validation_state", ""))
    terminal_validation_states = {"improved", "regressed", "inconclusive", "aborted", "rollback_failed"}
    closed_loop_complete = bool(apply_result is None or validation_state in terminal_validation_states)
    return {
        "mode": mode,
        "receiver_mode": "diagonal" if mode == "fixed" else mode,
        "duration_sec": float(monotonic() - started),
        "stopped": stopped,
        "error": error,
        "apply_result": apply_result,
        "summary": {
            "resource_id": str(last.get("resource_id", "")),
            "frames_processed": int(last.get("frames_processed", 0) or 0),
            "frames_decode_ok": int(last.get("frames_decode_ok", 0) or 0),
            "overflow_delta": overflow_delta,
            "processing_gap_delta": gap_delta,
            "continuity_end": str(last.get("rx_probe_continuity", "unknown")),
            "validation_state": str(last.get("adaptive_validation_state", "")),
            "closed_loop_complete": closed_loop_complete,
            "gmi_state": str(selection.get("state", "")),
            "gmi_state_epoch": int(selection.get("state_epoch", -1)),
            "gmi_reason": str(selection.get("reason", "")),
            "selection_ready": bool(usable_selections),
            "data_aided_evm_percent": evidence_snapshot.get("data_aided_evm_percent"),
            "data_aided_evm_median_percent": (
                float(np.median(evm_values)) if evm_values.size else None
            ),
            "data_aided_evm_p95_percent": (
                float(np.percentile(evm_values, 95.0)) if evm_values.size else None
            ),
            "evm_evidence_samples": int(evm_values.size),
            "evm_evidence_elapsed_sec": evidence_snapshot.get("elapsed_sec"),
            "last_reason": str(last.get("reason", "")),
        },
        "snapshots": snapshots,
    }


def run_validation(
    config: HardwareValidationConfig,
    *,
    backend_factory: Callable[..., Any] = FDIDMHardwareTest,
    device_probe: Callable[..., Sequence[DeviceInfo]] = probe_devices,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Dict[str, Any]:
    config.validate()
    devices = list(device_probe(timeout=8.0))
    device = _device_report(devices, config.serial)
    if not device["available"] or not device["serial_seen"]:
        raise RuntimeError(f"requested USRP serial {config.serial!r} was not found")
    segments = [
        _run_segment(
            config,
            mode,
            backend_factory=backend_factory,
            sleep=sleep,
            monotonic=monotonic,
        )
        for mode in MODES
    ]
    resources = {segment["summary"]["resource_id"] for segment in segments if segment["summary"]["resource_id"]}
    common_resource = len(resources) == 1 and all(segment["summary"]["resource_id"] for segment in segments)
    clean_stream = all(
        segment["summary"]["overflow_delta"] == 0
        and segment["summary"]["processing_gap_delta"] == 0
        for segment in segments
    )
    clean_stop = all(segment["stopped"] for segment in segments)
    no_errors = all(not segment["error"] for segment in segments)
    evidence_ready = all(
        int(segment["summary"]["frames_processed"]) > 0
        and int(segment["summary"]["frames_decode_ok"]) > 0
        and segment["summary"]["data_aided_evm_percent"] is not None
        for segment in segments
    )
    selection_ready = all(
        segment["mode"] == "fixed"
        or bool(segment["summary"]["selection_ready"])
        for segment in segments
    )
    closed_loop_complete = all(
        segment["mode"] != "diagonal"
        or bool(segment["summary"]["closed_loop_complete"])
        for segment in segments
    )
    passed = bool(
        common_resource and clean_stream and clean_stop and no_errors
        and evidence_ready and selection_ready and closed_loop_complete
    )
    return _safe({
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(config),
        "device_scan": device,
        "segments": segments,
        "summary": {
            "passed": passed,
            "common_resource": common_resource,
            "resource_ids": sorted(resources),
            "clean_stream": clean_stream,
            "clean_stop": clean_stop,
            "no_errors": no_errors,
            "evidence_ready": evidence_ready,
            "selection_ready": selection_ready,
            "closed_loop_complete": closed_loop_complete,
        },
    })


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True, help="B210 serial shown by uhd_find_devices")
    parser.add_argument("--duration", type=float, default=20.0, help="seconds per mode (fixed/diagonal/full)")
    parser.add_argument("--tx-gain", type=float, required=True, help=f"explicit TX gain, 0..{FDIDM_GAIN_MAX_DB:.0f} dB")
    parser.add_argument("--rx-gain", type=float, required=True, help=f"explicit RX gain, 0..{FDIDM_GAIN_MAX_DB:.0f} dB")
    parser.add_argument("--frequency", type=float, default=2.4e9)
    parser.add_argument("--sample-rate", type=float, default=500_000.0)
    parser.add_argument("--catalogue-cap", type=int, default=16)
    parser.add_argument("--robustness", choices=("fast", "standard", "conservative"), default="standard")
    parser.add_argument("--validation-grace", type=float, default=75.0, help="bounded extra seconds for an in-flight ABBA closeout")
    parser.add_argument("--output", type=Path, default=Path("log/gmi_hardware_validation.json"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    config = HardwareValidationConfig(
        serial=args.serial,
        duration_sec=args.duration,
        tx_gain_db=args.tx_gain,
        rx_gain_db=args.rx_gain,
        output=args.output,
        carrier_freq_hz=args.frequency,
        sample_rate_hz=args.sample_rate,
        catalogue_cap=args.catalogue_cap,
        robustness_profile=args.robustness,
        validation_grace_sec=args.validation_grace,
    )
    try:
        report = run_validation(config)
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"device error: {exc}", file=sys.stderr)
        return 3
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    print(f"JSON: {config.output.resolve()}")
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
