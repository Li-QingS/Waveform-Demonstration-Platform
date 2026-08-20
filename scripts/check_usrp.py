#!/usr/bin/env python
"""USRP 设备探测。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from waveform_sim.hardware.device_manager import probe_devices


if __name__ == "__main__":
    for dev in probe_devices():
        print(f"{dev.name}: {'PASS' if dev.available else 'WARN'} {dev.detail}")

