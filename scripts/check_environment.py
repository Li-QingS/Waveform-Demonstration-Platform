#!/usr/bin/env python
"""环境健康检查：输出 Python/NumPy/PyQt5/pyqtgraph/pytest/GNU Radio/UHD/USRP 状态。"""
from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_check(name: str, module: str | None = None) -> str:
    module = module or name
    try:
        m = importlib.import_module(module)
        version = getattr(m, "__version__", "unknown")
        return f"PASS  {version}"
    except Exception as exc:
        return f"WARN  {exc}"


def _command_check(name: str, command: list[str]) -> str:
    exe = shutil.which(command[0])
    if not exe:
        return "WARN  not found"
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=5)
        detail = (proc.stdout or proc.stderr or "").strip().splitlines()
        first = detail[0] if detail else ""
        return f"PASS  {first}" if proc.returncode == 0 else f"WARN  {first}"
    except Exception as exc:
        return f"WARN  {exc}"


def main() -> None:
    rows = [
        ("Python", f"PASS  {sys.version.split()[0]} on {platform.platform()}"),
        ("NumPy", _import_check("NumPy", "numpy")),
        ("PyQt5", _import_check("PyQt5", "PyQt5")),
        ("pyqtgraph", _import_check("pyqtgraph", "pyqtgraph")),
        ("pytest", _import_check("pytest", "pytest")),
        ("GNU Radio", _import_check("GNU Radio", "gnuradio")),
        ("UHD Python", _import_check("UHD Python", "uhd")),
        ("uhd_find_devices", _command_check("uhd_find_devices", ["uhd_find_devices"])),
    ]
    width = max(len(name) for name, _ in rows)
    for name, status in rows:
        print(f"{name:<{width}}  {status}")


if __name__ == "__main__":
    main()
