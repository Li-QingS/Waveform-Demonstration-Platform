"""把一次实验 run 目录导出为 Markdown 报告。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict


def export_run_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "report.md"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    last_metric: Dict[str, str] = {}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                last_metric = row
    lines = [
        f"# Experiment Report: {run_dir.name}",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Last Metrics",
        "",
        "```json",
        json.dumps(last_metric, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

