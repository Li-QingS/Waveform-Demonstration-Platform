# 平台工程化重构 Tasks（阶段 9：诊断 / 脚本 / CI）

> 前置：阶段 8 已验收通过（pytest 63 passed）。
> 目标：新增诊断 API（health_check / report_exporter / snapshot）、两个运维脚本、GitHub Actions CI；`check_environment.py` 改为复用诊断 API。

## 文件清单

| 操作 | 文件 | 职责 |
|------|------|------|
| 新建 | `waveform_sim/diagnostics/__init__.py` | diagnostics 子包标记 |
| 新建 | `waveform_sim/diagnostics/health_check.py` | 环境健康检查 API |
| 新建 | `waveform_sim/diagnostics/report_exporter.py` | run 目录 Markdown 报告导出 |
| 新建 | `waveform_sim/diagnostics/snapshot.py` | 状态快照 dataclass |
| 新建 | `scripts/check_usrp.py` | USRP 设备探测脚本 |
| 新建 | `scripts/run_self_test.py` | 无显示环境自检（FDIDM 出帧 + artifact） |
| 修改 | `scripts/check_environment.py` | 复用 `diagnostics.health_check` |
| 新建 | `.github/workflows/ci.yml` | 纯 Python 核心 CI |
| 新建 | `tests/test_diagnostics.py` | 诊断 API 测试（3 用例） |

## 任务

### T9.1：新建 diagnostics 子包

**文件：** 新建 `waveform_sim/diagnostics/__init__.py`、`health_check.py`、`report_exporter.py`、`snapshot.py`
**依赖：** 无

**步骤：** 创建四个文件，内容如下。

`__init__.py`：

```python
"""诊断与可观测性：环境健康检查、报告导出、状态快照。"""
```

`health_check.py`：

```python
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
```

`report_exporter.py`：

```python
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
```

`snapshot.py`：

```python
"""状态快照 dataclass。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict


@dataclass
class Snapshot:
    timestamp: str
    state: str
    metrics: Dict
    config: Dict

    @classmethod
    def capture(cls, state: str, metrics: Dict, config: Dict) -> "Snapshot":
        return cls(datetime.now(timezone.utc).isoformat(), state, dict(metrics), dict(config))

    def to_dict(self):
        return asdict(self)
```

**验证：** `python -c "from waveform_sim.diagnostics.health_check import run_health_check, format_health_check; print(len(run_health_check(include_usrp=False)))"` 输出 `8`。

### T9.2：新增脚本与 check_environment 复用

**文件：** 新建 `scripts/check_usrp.py`、`scripts/run_self_test.py`；修改 `scripts/check_environment.py`
**依赖：** T9.1

**步骤：**

1. `scripts/check_usrp.py`：

```python
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
```

2. `scripts/run_self_test.py`（无显示环境自检；不含完整自适应调优，调优留 GUI/实验室）：

```python
#!/usr/bin/env python
"""无显示环境下自检：FDIDM 引擎出帧 + 指标 + 实验 artifact。"""
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from waveform_sim.core.config import ExperimentConfig, WaveformConfig
from waveform_sim.core.engine import LinkSimulator
from waveform_sim.service.experiment_service import ExperimentService


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = ExperimentConfig(
            waveform=WaveformConfig(waveform="FDIDM", snr_db=20.0, alpha=0.0, beta=0.0, seed=7),
            runs_dir=str(tmp),
        )
        svc = ExperimentService(cfg)
        svc.start_run()
        sim = LinkSimulator(cfg, experiment_service=svc)
        sim.step()
        sim.step()
        metrics = sim.get_last_metrics()
        svc.finish_run(metrics)
        print("smoke metrics:", {k: metrics.get(k) for k in ("ber", "ser", "evm_db")})
        print("run_id:", svc.run_id)
        print("artifacts:", sorted(p.name for p in svc.writer.run_dir.iterdir()))
        assert metrics.get("total_bits", 0) > 0
        assert (svc.writer.run_dir / "config.json").exists()
    print("self test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

3. `scripts/check_environment.py` 整体替换为复用诊断 API：

```python
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
```

**验证：** `python scripts/check_environment.py` 输出 9 行检查表；`python scripts/check_usrp.py` 无 traceback；`python scripts/run_self_test.py` 输出 `self test OK`。

### T9.3：新建 .github/workflows/ci.yml

**文件：** 新建 `.github/workflows/ci.yml`
**依赖：** 无

**步骤：** 创建文件，内容如下：

```yaml
name: core-ci

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: python -m pip install --upgrade pip
      - run: pip install numpy pytest
      - run: pytest -q
      - run: python -m compileall waveform_sim
```

> CI 只装 numpy + pytest：现有测试套件（收发冒烟、引擎、服务、硬件抽象、FEC/信道/环形缓冲、诊断）均不依赖 PyQt5；GUI 冒烟不在 pytest 中。

**验证：** 本地等价命令 `python -m pytest -q` 与 `python -m compileall waveform_sim` 通过。

### T9.4：新建 tests/test_diagnostics.py

**文件：** 新建 `tests/test_diagnostics.py`
**依赖：** T9.1

**步骤：** 创建文件，内容如下：

```python
"""诊断 API 测试（阶段 9）。"""
from waveform_sim.core.config import ExperimentConfig
from waveform_sim.diagnostics.health_check import format_health_check, run_health_check
from waveform_sim.diagnostics.report_exporter import export_run_report
from waveform_sim.diagnostics.snapshot import Snapshot
from waveform_sim.service.experiment_service import ExperimentService


def test_health_check_runs():
    results = run_health_check(include_usrp=False)
    assert len(results) == 8
    for r in results:
        assert r.status in ("PASS", "WARN")
    text = format_health_check(results)
    assert "NumPy" in text


def test_report_exporter(tmp_path):
    cfg = ExperimentConfig(runs_dir=str(tmp_path))
    svc = ExperimentService(cfg)
    svc.start_run()
    svc.log_metrics({"frame_id": 1, "ber": 0.0})
    svc.finish_run({})
    report = export_run_report(svc.writer.run_dir)
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert svc.run_id in text
    assert "## Config" in text


def test_snapshot():
    snap = Snapshot.capture("RUNNING", {"ber": 0.1}, {"waveform": "FDIDM"})
    d = snap.to_dict()
    assert d["state"] == "RUNNING"
    assert d["metrics"]["ber"] == 0.1
    assert d["config"]["waveform"] == "FDIDM"
```

**验证：** `python -m pytest tests/test_diagnostics.py -q` → `3 passed`。

### T9.5：全量验证与提交

**文件：** 本阶段全部新增/修改文件
**依赖：** T9.1~T9.4

**步骤：**

1. `python -m pytest -q` → 期望 `66 passed`（原 63 + 本阶段 3）。
2. `python -m compileall -q waveform_sim scripts` → 退出码 0、无输出。
3. 三个脚本运行验证：`check_environment.py`、`check_usrp.py`、`run_self_test.py`。
4. `git add waveform_sim/diagnostics scripts .github tests/test_diagnostics.py`
5. `git commit -m "feat: 新增诊断API、运维脚本与CI（阶段9）"`
6. `git status` → 工作区干净

**验证：** pytest `66 passed`；三个脚本可用；提交存在；工作区干净。

## 执行顺序

```
T9.1 → T9.2 → T9.3 → T9.4 → T9.5
```

## 阶段 checkpoint

- T9.5 后向用户报告 pytest / 脚本输出与提交号；确认后再拆解阶段 10（文档与收尾）。

