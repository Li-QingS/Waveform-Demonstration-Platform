#!/usr/bin/env python
"""环境健康检查（复用 waveform_sim.diagnostics.health_check）。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from waveform_sim.diagnostics.health_check import format_health_check, run_health_check


if __name__ == "__main__":
    print(format_health_check(run_health_check(include_usrp=True)))

