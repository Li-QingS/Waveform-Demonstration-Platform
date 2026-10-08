"""USRP 设备探测（uhd_find_devices 封装，阶段 5）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List
import shutil
import subprocess
import sys


@dataclass
class DeviceInfo:
    available: bool
    name: str = ""
    detail: str = ""

    def to_dict(self):
        return asdict(self)


def probe_devices(timeout: float = 5.0) -> List[DeviceInfo]:
    exe = shutil.which("uhd_find_devices")
    if not exe:
        # Radioconda/conda-style Windows installs keep UHD tools under the
        # base runtime even when the project uses a nested virtualenv whose
        # Scripts directory is the only entry on PATH.
        candidates = (
            Path(sys.base_prefix) / "Library" / "bin" / "uhd_find_devices.exe",
            Path(sys.prefix) / "Library" / "bin" / "uhd_find_devices.exe",
        )
        exe = next((str(path) for path in candidates if path.is_file()), None)
    if not exe:
        return [DeviceInfo(False, "UHD", "uhd_find_devices not found")]
    try:
        proc = subprocess.run([exe], text=True, capture_output=True, timeout=timeout)
        text = (proc.stdout or proc.stderr or "").strip()
        if proc.returncode != 0:
            return [DeviceInfo(False, "UHD", text)]
        if "No UHD Devices Found" in text or not text:
            return [DeviceInfo(False, "USRP", text or "No UHD devices found")]
        return [DeviceInfo(True, "USRP", text)]
    except Exception as exc:
        return [DeviceInfo(False, "UHD", str(exc))]

