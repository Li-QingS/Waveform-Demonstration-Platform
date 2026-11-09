"""Shared artifact lifecycle for live hardware waveform validation runs."""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..core.config import ExperimentConfig
from .experiment_service import ExperimentService


class HardwareRunRecorder:
    """Persist hardware configuration and sampled receive-status snapshots."""

    def __init__(self, config: ExperimentConfig, metadata: Optional[Dict[str, Any]] = None):
        self.service = ExperimentService(config)
        self.metadata = dict(metadata or {})
        self._last_frame_id: Optional[int] = None
        self._finished = False

    @property
    def run_id(self) -> str:
        return self.service.run_id

    @property
    def run_dir(self):
        return self.service.run_dir

    def start(self) -> str:
        run_id = self.service.start_run()
        if self.metadata:
            self.service.log_event(
                "HARDWARE_CONFIGURED", module="hardware", payload=self.metadata
            )
        return run_id

    def record_status(self, status: Dict[str, Any]) -> bool:
        if self._finished or not isinstance(status, dict):
            return False
        raw_frame_id = status.get("frame_id", status.get("frames_processed"))
        if raw_frame_id is None:
            return False
        try:
            frame_id = int(raw_frame_id)
        except (TypeError, ValueError, OverflowError):
            return False
        if frame_id <= 0:
            return False
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            return False
        skipped_frames = (
            max(0, frame_id - 1)
            if self._last_frame_id is None
            else max(0, frame_id - self._last_frame_id - 1)
        )
        self._last_frame_id = frame_id

        metrics = {
            "frame_id": frame_id,
            "recording_type": "polling_snapshot",
            "skipped_frames_since_last": skipped_frames,
            "ber": self._first(status, "cumulative_ber", "ber"),
            "frame_ber": self._first(status, "frame_ber", "ber"),
            "bit_errors": status.get("bit_errors"),
            "bits_total": status.get("bits_total"),
            "frame_bit_errors": status.get("frame_bit_errors"),
            "frame_bits_total": status.get("frame_bits_total"),
            "evm_db": status.get("evm_db"),
            "decode_ok": status.get("decode_ok"),
            "frames_processed": status.get("frames_processed", frame_id),
            "frames_decode_ok": status.get("frames_decode_ok"),
        }
        self.service.log_metrics(metrics)
        return True

    def finish(self, summary: Optional[Dict[str, Any]] = None) -> None:
        if self._finished:
            return
        self.service.finish_run(summary or {})
        self._finished = True

    def fail(self, error: str) -> None:
        if self._finished:
            return
        self.service.fail_run(error)
        self._finished = True

    @staticmethod
    def _first(status: Dict[str, Any], *keys: str):
        for key in keys:
            value = status.get(key)
            if value is not None:
                return value
        return None
