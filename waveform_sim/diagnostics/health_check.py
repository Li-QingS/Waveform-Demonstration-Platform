"""环境健康检查 API（Python / NumPy / PyQt5 / pyqtgraph / pytest / GNU Radio / UHD / USRP）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
import platform
import shutil
import subprocess
import sys
from typing import List


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""

    def to_dict(self):
        return asdict(self)


def _import_check(name: str, module: str | None = None) -> CheckResult:
    module = module or name
    try:
        m = import_module(module)
        version = getattr(m, "__version__", "unknown")
        return CheckResult(name, "PASS", str(version))
    except Exception as exc:
        return CheckResult(name, "WARN", str(exc))


def _command_check(name: str, command: List[str]) -> CheckResult:
    exe = shutil.which(command[0])
    if not exe:
        return CheckResult(name, "WARN", f"{command[0]} not found")
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=5)
        detail = (proc.stdout or proc.stderr or "").strip().splitlines()[:3]
        return CheckResult(name, "PASS" if proc.returncode == 0 else "WARN", " | ".join(detail))
    except Exception as exc:
        return CheckResult(name, "WARN", str(exc))


def run_health_check(include_usrp: bool = True) -> List[CheckResult]:
    results = [
        CheckResult("Python", "PASS", f"{sys.version.split()[0]} on {platform.platform()}"),
        _import_check("NumPy", "numpy"),
        _import_check("PyQt5", "PyQt5"),
        _import_check("pyqtgraph", "pyqtgraph"),
        _import_check("pytest", "pytest"),
        _import_check("GNU Radio", "gnuradio"),
        _import_check("UHD Python", "uhd"),
        _command_check("uhd_find_devices", ["uhd_find_devices"]),
    ]
    if include_usrp:
        results.append(_command_check("USRP Probe", ["uhd_usrp_probe", "--args", "type=b200"]))
    return results


def format_health_check(results: List[CheckResult]) -> str:
    width = max(len(r.name) for r in results) if results else 12
    lines = []
    for r in results:
        lines.append(f"{r.name:<{width}}  {r.status:<5}  {r.detail}")
    return "\n".join(lines)

