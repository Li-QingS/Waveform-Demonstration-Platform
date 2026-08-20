"""实验 run artifact 管理：runs/<run_id>/ 下的 config/events/metrics/report。"""
from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ..core.config import ExperimentConfig


class RunArtifactWriter:
    def __init__(self, config: ExperimentConfig):
        self.config = config.normalized()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{stamp}_{uuid.uuid4().hex[:8]}"
        self.run_dir = Path(self.config.runs_dir) / self.run_id
        self.figures_dir = self.run_dir / "figures"
        self.metrics_path = self.run_dir / "metrics.csv"
        self.events_path = self.run_dir / "events.jsonl"
        self.config_path = self.run_dir / "config.json"
        self.report_path = self.run_dir / "report.md"
        self._metric_fields = None

    def create(self) -> None:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(self.config.to_json(), encoding="utf-8")
        self.events_path.write_text("", encoding="utf-8")

    def write_metric(self, row: Dict) -> None:
        row = dict(row)
        if self._metric_fields is None:
            preferred = [
                "frame_id", "snr_db", "ber", "fer", "ser", "evm_db", "evm_percent",
                "alpha", "beta", "bits_total", "bit_errors",
                "frames_processed", "frames_decode_ok",
            ]
            keys = list(dict.fromkeys(preferred + list(row.keys())))
            self._metric_fields = keys
            with self.metrics_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._metric_fields)
                writer.writeheader()
        with self.metrics_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._metric_fields, extrasaction="ignore")
            writer.writerow(row)

    def write_report(self, summary: Optional[Dict] = None) -> None:
        summary = summary or {}
        lines = [
            f"# Experiment Report: {self.run_id}",
            "",
            "## Configuration",
            "",
            f"- Waveform: {self.config.waveform.waveform}",
            f"- Modulation: {self.config.waveform.mod_order}",
            f"- Alpha/Beta: {self.config.waveform.alpha} / {self.config.waveform.beta}",
            f"- Sample rate: {self.config.waveform.sample_rate_hz} Hz",
            f"- Hardware transport: {self.config.hardware.transport}",
            "",
            "## Artifacts",
            "",
            "- `config.json`",
            "- `events.jsonl`",
            "- `metrics.csv`",
            "- `figures/`",
            "",
            "## Summary",
            "",
        ]
        if summary:
            lines.append("```json")
            lines.append(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
            lines.append("```")
        else:
            lines.append("Run completed. See metrics.csv and events.jsonl for details.")
        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

