"""USRP 设备探测（uhd_find_devices 封装，阶段 5）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List
import shutil
import subprocess


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

