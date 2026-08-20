# 平台工程化重构 Tasks（阶段 7：服务层 · 实验记录）

> 前置：阶段 6c 已验收通过（pytest 60 passed）。
> 目标：新增实验记录服务（run artifact + JSONL 事件日志），并让 `LinkSimulator` 可挂接（默认关闭、可开关）。

## 文件清单

| 操作 | 文件 | 职责 |
|------|------|------|
| 新建 | `waveform_sim/service/__init__.py` | service 子包标记 |
| 新建 | `waveform_sim/service/run_state.py` | `RunState` 状态机枚举 |
| 新建 | `waveform_sim/service/artifact_writer.py` | `RunArtifactWriter`：`runs/<run_id>/{config.json,events.jsonl,metrics.csv,report.md}` |
| 新建 | `waveform_sim/service/event_logger.py` | `EventLogger`：追加式 JSONL 事件日志 |
| 新建 | `waveform_sim/service/experiment_service.py` | `ExperimentService`：run 生命周期编排 |
| 修改 | `waveform_sim/core/engine.py` | `LinkSimulator` 挂接实验服务（`auto_start_run` + 事件/指标钩子） |
| 新建 | `tests/test_service.py` | 服务层与引擎挂接测试（3 用例） |

## 任务

### T7.1：新建 service 子包四个文件

**文件：** 新建 `waveform_sim/service/__init__.py`、`run_state.py`、`artifact_writer.py`、`event_logger.py`、`experiment_service.py`
**依赖：** 无

**步骤：** 创建五个文件，内容如下。

`__init__.py`：

```python
"""实验记录服务：run artifact、事件日志与状态机。"""
```

`run_state.py`：

```python
from enum import Enum


class RunState(str, Enum):
    UNCONFIGURED = "UNCONFIGURED"
    CONFIGURED = "CONFIGURED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"
```

`artifact_writer.py`：

```python
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
```

`event_logger.py`：

```python
"""追加式 JSONL 事件日志。"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class EventLogger:
    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self._lock = threading.Lock()
        self._seq = 0

    def log(
        self,
        event: str,
        module: str = "platform",
        payload: Optional[Dict[str, Any]] = None,
        **context,
    ) -> Dict[str, Any]:
        with self._lock:
            self._seq += 1
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                "seq": self._seq,
                "module": module,
                "event": event,
                "payload": payload or {},
            }
            record.update({k: v for k, v in context.items() if v is not None})
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            return record
```

`experiment_service.py`：

```python
"""实验编排：run 生命周期与 artifact 记录。"""
from __future__ import annotations

from typing import Dict, Optional

from ..core.config import ExperimentConfig
from .artifact_writer import RunArtifactWriter
from .event_logger import EventLogger
from .run_state import RunState


class ExperimentService:
    def __init__(self, config: ExperimentConfig | None = None):
        self.config = (config or ExperimentConfig()).normalized()
        self.writer = RunArtifactWriter(self.config)
        self.logger: EventLogger | None = None
        self.state = RunState.CONFIGURED

    @property
    def run_id(self) -> str:
        return self.writer.run_id

    @property
    def run_dir(self):
        return self.writer.run_dir

    def start_run(self) -> str:
        self.state = RunState.STARTING
        self.writer.create()
        self.logger = EventLogger(self.writer.events_path, self.writer.run_id)
        self.log_event("RUN_STARTED", module="experiment_service", payload=self.config.to_dict())
        self.state = RunState.RUNNING
        return self.writer.run_id

    def log_event(
        self,
        event: str,
        module: str = "platform",
        payload: Optional[Dict] = None,
        **context,
    ) -> None:
        if self.logger is not None:
            self.logger.log(event=event, module=module, payload=payload or {}, **context)

    def log_metrics(self, metrics: Dict) -> None:
        self.writer.write_metric(metrics)

    def finish_run(self, summary: Optional[Dict] = None) -> None:
        self.state = RunState.STOPPING
        self.log_event("RUN_FINISHED", module="experiment_service", payload=summary or {})
        self.writer.write_report(summary or {})
        self.state = RunState.STOPPED

    def fail_run(self, error: str) -> None:
        self.state = RunState.ERROR
        self.log_event("RUN_FAILED", module="experiment_service", payload={"error": str(error)})
        self.writer.write_report({"error": str(error)})
```

**验证：** `python -c "from waveform_sim.service.experiment_service import ExperimentService; print(ExperimentService.__name__)"` 输出 `ExperimentService`。

### T7.2：LinkSimulator 挂接实验服务

**文件：** 修改 `waveform_sim/core/engine.py`
**依赖：** T7.1

**步骤：**

1. 文件头 import 区（`from .config import ...` 之后）加容错导入：

```python
try:
    from waveform_sim.service.experiment_service import ExperimentService
except Exception:  # pragma: no cover - 部分安装时容错
    ExperimentService = None
```

2. `__init__` 增加 `auto_start_run: bool = False` 参数，并在 `self.experiment_service = experiment_service` 之后加：

```python
        if auto_start_run and ExperimentService is not None and self.experiment_service is None:
            self.experiment_service = ExperimentService(self.experiment_config)
            self.experiment_service.start_run()
```

3. 生命周期方法挂钩：

```python
    def start(self) -> None:
        self._backend.start()
        self._log_event("LINK_STARTED", {"waveform": self.config.waveform})

    def stop(self) -> None:
        self._backend.stop()
        self._log_event("LINK_STOPPED", {"waveform": self.config.waveform})

    def step(self) -> None:
        self._backend.step()
        self._log_metric(self.get_last_metrics())
```

4. `set_indices` 末尾加 `self._log_event("INDICES_SET", {"alpha": float(alpha), "beta": float(beta)})`。

5. 类末尾新增两个钩子方法：

```python
    def _log_event(self, event: str, payload: Dict) -> None:
        svc = getattr(self, "experiment_service", None)
        if svc is not None:
            try:
                svc.log_event(event=event, module="link_simulator", payload=payload)
            except Exception:
                pass

    def _log_metric(self, metrics: Dict) -> None:
        svc = getattr(self, "experiment_service", None)
        if svc is not None:
            try:
                svc.log_metrics(metrics)
            except Exception:
                pass
```

**验证：** `python -m pytest tests/test_engine.py -q` → `4 passed`（既有引擎测试无回归）。

### T7.3：新建 tests/test_service.py

**文件：** 新建 `tests/test_service.py`
**依赖：** T7.1~T7.2

**步骤：** 创建文件，内容如下：

```python
"""实验记录服务与引擎挂接测试（阶段 7）。"""
import json
import time

from waveform_sim.core.config import ExperimentConfig, WaveformConfig
from waveform_sim.core.engine import LinkSimulator
from waveform_sim.service.experiment_service import ExperimentService


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_run_artifacts(tmp_path):
    cfg = ExperimentConfig(waveform=WaveformConfig(waveform="FDIDM"), runs_dir=str(tmp_path))
    svc = ExperimentService(cfg)
    run_id = svc.start_run()
    svc.log_event("TEST", module="pytest", payload={"ok": True})
    svc.log_metrics({"frame_id": 1, "ber": 0.0, "evm_db": -30.0})
    svc.finish_run({"ok": True})
    run_dir = tmp_path / run_id
    assert (run_dir / "config.json").exists()
    assert (run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "report.md").exists()
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[0]["event"] == "RUN_STARTED"
    assert events[-1]["event"] == "RUN_FINISHED"


def test_engine_experiment_hooks(tmp_path):
    cfg = ExperimentConfig(waveform=WaveformConfig(waveform="FDIDM", snr_db=12.0), runs_dir=str(tmp_path))
    svc = ExperimentService(cfg)
    sim = LinkSimulator(cfg, experiment_service=svc)
    svc.start_run()
    sim.step()
    sim.step()
    sim.start()
    try:
        ok = _wait_until(lambda: sim.get_last_metrics().get("total_bits", 0) > 0)
        assert ok
    finally:
        sim.stop()
        sim.wait(timeout=3.0)
    svc.finish_run(sim.get_last_metrics())
    events = (svc.writer.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert "LINK_STARTED" in events
    assert "LINK_STOPPED" in events
    rows = (svc.writer.metrics_path).read_text(encoding="utf-8").splitlines()
    assert len(rows) >= 3  # 表头 + 两次 step() 指标


def test_engine_auto_start_run(tmp_path):
    cfg = ExperimentConfig(waveform=WaveformConfig(waveform="FDIDM", snr_db=15.0), runs_dir=str(tmp_path))
    sim = LinkSimulator(cfg, auto_start_run=True)
    assert sim.experiment_service is not None
    assert sim.experiment_service.state.value == "RUNNING"
    sim.stop()
    sim.experiment_service.finish_run({})
    assert (sim.experiment_service.writer.run_dir / "config.json").exists()
```

**验证：** `python -m pytest tests/test_service.py -q` → `3 passed`。

### T7.4：全量验证与提交

**文件：** 本阶段全部新增/修改文件
**依赖：** T7.1~T7.3

**步骤：**

1. `python -m pytest -q` → 期望 `63 passed`（原 60 + 本阶段 3）。
2. `python -m compileall -q waveform_sim` → 退出码 0、无输出。
3. `git add waveform_sim/service tests/test_service.py waveform_sim/core/engine.py`
4. `git commit -m "feat: 新增实验记录服务与引擎挂接（阶段7）"`
5. `git status` → 工作区干净

**验证：** pytest `63 passed`；提交存在；工作区干净。

## 执行顺序

```
T7.1 → T7.2 → T7.3 → T7.4
```

## 阶段 checkpoint

- T7.4 后向用户报告 pytest / artifact 目录输出与提交号；确认后再拆解阶段 8（UI 收敛）。

