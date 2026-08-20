# 平台工程化重构 Tasks（阶段 0 ~ 1）

> 前置：`spec.md` 与 `plan.md` 已获批。
> 范围：仅拆解阶段 0（基线锁定）与阶段 1（统一配置）。阶段 2 及以后在阶段 0~1 验收通过后另行拆解。
> 执行说明：本仓库 `.git` 目录在沙箱内只读，`git add` / `git commit` 需使用已批准的执行方式；文件创建统一用 apply_patch，不用 shell 重定向写文件。

## 文件清单

### 阶段 0：基线锁定

| 操作 | 文件 | 职责 |
|------|------|------|
| 提交 | `markdown/06_platform_refactor_spec.md`、`markdown/07_platform_refactor_plan.md` | 固化已批准文档 |
| 新建 | `pyproject.toml` | 项目元数据、pytest 配置、依赖声明 |
| 新建 | `requirements.txt` | pip 依赖 |
| 新建 | `environment.yml` | conda 环境定义 |
| 修改 | `.gitignore` | 追加 `runs/`、`__pycache__/`、`.venv/`、`.pytest_cache/` |
| 新建 | `scripts/check_environment.py` | 环境健康检查脚本 |
| 新建 | `tests/test_transceivers_smoke.py` | 四波形收发链路冒烟基线 |
| 新建 | `tests/test_scan_backend_smoke.py` | 波形对比扫描冒烟基线 |
| 新建 | `tests/test_hardtest_import.py` | 硬件后端模块导入基线 |

### 阶段 1：统一配置

| 操作 | 文件 | 职责 |
|------|------|------|
| 新建 | `waveform_sim/__init__.py` | 使 `waveform_sim` 成为正式包 |
| 新建 | `waveform_sim/core/__init__.py` | core 子包标记 |
| 新建 | `waveform_sim/core/config.py` | 统一配置 dataclass（只增不改） |
| 新建 | `tests/test_config.py` | 配置序列化/归一化测试 |

## 任务

### T0.1：提交 spec/plan 文档

**文件：** `markdown/06_platform_refactor_spec.md`、`markdown/07_platform_refactor_plan.md`
**依赖：** 无

**步骤：**
1. `git add markdown/06_platform_refactor_spec.md markdown/07_platform_refactor_plan.md`
2. `git commit -m "docs: 添加平台工程化重构 spec 与 plan"`
3. `git log --oneline -1` 确认新提交在最前

**验证：** 新提交存在，提交信息正确。

### T0.2：新建 pyproject.toml

**文件：** 新建 `pyproject.toml`
**依赖：** 无

**步骤：** 创建文件，内容如下：

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "waveform-demonstration-platform"
version = "0.1.0"
description = "FDIDM/OFDM/OTFS/AFDM 波形原型验证演示平台"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.24,<3.0",
  "PyQt5>=5.15,<6.0",
  "pyqtgraph>=0.13,<0.14",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

**验证：** `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('toml ok')"` 输出 `toml ok`。

### T0.3：新建 requirements / environment / 更新 .gitignore

**文件：** 新建 `requirements.txt`、`environment.yml`；修改 `.gitignore`
**依赖：** 无

**步骤：**

1. 创建 `requirements.txt`：

```text
numpy>=1.24,<3.0
PyQt5>=5.15,<6.0
pyqtgraph>=0.13,<0.14
pytest>=8.0
```

2. 创建 `environment.yml`：

```yaml
name: waveform-demo-platform
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - pyqt
  - pyqtgraph
  - pytest
  - pip
```

3. 读取 `.gitignore`，若缺少以下条目则追加（已有则跳过）：

```text
runs/
.venv/
__pycache__/
*.pyc
.pytest_cache/
```

**验证：** 三个文件存在；`.gitignore` 包含上述条目（`Select-String -Path .gitignore -Pattern 'runs/'` 命中）。

### T0.4：新建 scripts/check_environment.py

**文件：** 新建 `scripts/check_environment.py`
**依赖：** 无

**步骤：** 创建文件，内容如下：

```python
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
```

**验证：** `python scripts/check_environment.py` 正常输出表格（NumPy 为 PASS，PyQt5/UHD 等按环境 WARN），无 traceback。

### T0.5：新建 tests/test_transceivers_smoke.py

**文件：** 新建 `tests/test_transceivers_smoke.py`
**依赖：** 无

**步骤：** 创建文件，内容如下：

```python
"""四波形收发链路冒烟基线：构造、启动、出帧、指标合理。"""
import time

from waveform_sim.simulation.simple_fdidm_rx import FDIDMTransceiver
from waveform_sim.simulation.simple_ofdm_rx import OfdmTransceiver
from waveform_sim.simulation.simple_otfs_rx import OTFSTransceiver
from waveform_sim.simulation.simple_afdm_rx import AFDMTransceiver


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _assert_smoke(transceiver, name: str) -> None:
    transceiver.start()
    try:
        ok = _wait_until(lambda: transceiver.get_last_metrics().get("total_bits", 0) > 0)
        assert ok, f"{name}: 超时未产生帧"
        metrics = transceiver.get_last_metrics()
        assert "ber" in metrics, f"{name}: 缺少 ber"
        assert "evm_db" in metrics or "evm_percent" in metrics, f"{name}: 缺少 evm"
        assert 0.0 <= metrics["ber"] <= 1.0, f"{name}: ber 越界 {metrics['ber']}"
    finally:
        transceiver.stop()
        transceiver.wait(timeout=3.0)


def test_fdidm_transceiver_smoke():
    _assert_smoke(FDIDMTransceiver(alpha=0.0, beta=0.0, snr_db=12.0, channel_seed=42), "FDIDM")


def test_ofdm_transceiver_smoke():
    _assert_smoke(OfdmTransceiver(snr_db=15.0), "OFDM")


def test_otfs_transceiver_smoke():
    _assert_smoke(OTFSTransceiver(snr_db=15.0), "OTFS")


def test_afdm_transceiver_smoke():
    _assert_smoke(AFDMTransceiver(snr_db=15.0), "AFDM")
```

**验证：** `python -m pytest tests/test_transceivers_smoke.py -q` → `4 passed`。

### T0.6：新建 tests/test_scan_backend_smoke.py

**文件：** 新建 `tests/test_scan_backend_smoke.py`
**依赖：** 无

**步骤：** 创建文件，内容如下：

```python
"""波形对比扫描冒烟基线：scan_axis 返回三种波形的 BER 数组。"""
import numpy as np

from waveform_sim.simulation.compare_scan_backend import scan_axis


def test_scan_axis_smoke():
    result = scan_axis(
        axis_name="doppler",
        axis_values=np.array([0.0, 20.0]),
        snr_db=15.0,
        mod_order="QPSK",
        frames=2,
        fixed_cfo_hz=0.0,
        fixed_doppler_hz=20.0,
        seed_base=20260502,
    )
    assert set(result.keys()) == {"OFDM", "OTFS", "AFDM"}
    for name, arr in result.items():
        assert arr.shape == (2,), name
        assert np.all((arr >= 0.0) & (arr <= 1.0)), name
```

**验证：** `python -m pytest tests/test_scan_backend_smoke.py -q` → `1 passed`。

### T0.7：新建 tests/test_hardtest_import.py

**文件：** 新建 `tests/test_hardtest_import.py`
**依赖：** 无

**步骤：** 创建文件，内容如下：

```python
"""硬件后端模块导入基线：四个 hardtest 模块可导入且主类存在。"""
import importlib

HARDTEST_CLASSES = {
    "waveform_sim.hardware.fdidm_hardtest": "FDIDMHardwareTest",
    "waveform_sim.hardware.ofdm_hardtest": "OfdmHardwareTx",
    "waveform_sim.hardware.otfs_hardtest": "OTFSHardwareTest",
    "waveform_sim.hardware.afdm_hardtest": "AFDMHardwareTest",
}


def test_hardtest_modules_import():
    for mod_name, cls_name in HARDTEST_CLASSES.items():
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, cls_name), f"{mod_name} 缺少 {cls_name}"
```

**验证：** `python -m pytest tests/test_hardtest_import.py -q` → `1 passed`。

### T0.8：阶段 0 全量验证

**文件：** 无新增
**依赖：** T0.2~T0.7

**步骤：**
1. `python -m pytest -q` → 期望 `6 passed`（4 收发 + 1 扫描 + 1 导入）。
2. `python -m compileall -q waveform_sim scripts` → 退出码 0、无输出。
3. `python scripts/check_environment.py` → 输出检查表，无 traceback。

**验证：** 三条命令全部通过；如有失败先修复再继续。

### T0.9：提交阶段 0

**文件：** `pyproject.toml`、`requirements.txt`、`environment.yml`、`.gitignore`、`scripts/`、`tests/`
**依赖：** T0.8

**步骤：**
1. `git add pyproject.toml requirements.txt environment.yml .gitignore scripts tests`
2. `git commit -m "chore: 添加测试基线与工程脚手架（阶段0）"`
3. `git status` → 工作区干净

**验证：** 提交存在；`python -m pytest -q` 仍为 `6 passed`。

---

### T1.1：新建 waveform_sim 包标记

**文件：** 新建 `waveform_sim/__init__.py`、`waveform_sim/core/__init__.py`
**依赖：** T0.9

**步骤：**

1. `waveform_sim/__init__.py` 内容：

```python
"""Waveform Demonstration Platform 顶层包。"""
```

2. `waveform_sim/core/__init__.py` 内容：

```python
"""工程化核心：配置、波形引擎、自适应调优与指标。"""
```

**验证：** `python -c "import waveform_sim.core; print(waveform_sim.__name__)"` 输出 `waveform_sim`；`python -m pytest -q` 仍为 `6 passed`。

### T1.2：新建 waveform_sim/core/config.py

**文件：** 新建 `waveform_sim/core/config.py`
**依赖：** T1.1

**步骤：** 创建文件，内容如下：

```python
"""统一配置模型（阶段 1）。

只新增、不改旧代码。旧参数名到新字段的映射由后续引擎层
（waveform_sim/core/engine.py）的 update_config 负责。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class WaveformConfig:
    """四类波形（FDIDM/OFDM/OTFS/AFDM）的统一链路参数。"""

    waveform: str = "FDIDM"
    mod_order: str = "16QAM"
    snr_db: float = 20.0
    snr_definition: str = "Eb/N0"
    cfo_hz: float = 0.0
    center_freq_hz: float = 20e9
    payload_bits: int = 1024
    seed: int = 42
    # FDIDM / OTFS 网格
    m_subcarriers: int = 8
    n_symbols: int = 8
    subcarrier_spacing_hz: float = 300_000.0
    # OFDM / AFDM
    fft_size: int = 64
    cp_len: int = 16
    payload_symbols: int = 8
    frame_size: int = 128
    # OTFS
    n_subcarriers: int = 64
    sample_rate_hz: float = 960_000.0
    update_period: float = 0.08
    equalizer: str = "MMSE"
    delay_spread: int = 5
    doppler_spread_hz: float = 20.0
    # FDIDM 索引 / AFDM chirp
    alpha: float = 0.0
    beta: float = 0.0
    c1: float = 0.05
    c2: float = 0.05
    # 信道
    channel_model: str = "TDL-C"
    velocity_kmh: float = 0.0
    doppler_radial_factor: float = 0.10
    random_channel: bool = True
    dynamic_channel: bool = False
    channel_dynamics: str = "fixed"
    channel_coherence_frames: int = 20
    fast_channel_coherence_symbols: int = 1
    circular_channel: bool = True
    tf_notch_depth_db: float = 0.0
    tf_notch_count: int = 0
    # 接收机 / 搜索
    detector: str = "ZF"
    link_mode: str = "matrix"
    search_objective: str = "zf_theory_ser"
    optimize_indices: bool = False
    search_step: float = 0.1

    def normalized(self) -> "WaveformConfig":
        self.waveform = str(self.waveform or "FDIDM").upper()
        self.mod_order = str(self.mod_order or "16QAM").upper()
        self.detector = str(self.detector or "ZF").upper()
        self.equalizer = str(self.equalizer or "MMSE").upper()
        self.snr_definition = str(self.snr_definition or "Eb/N0")
        self.m_subcarriers = max(1, int(self.m_subcarriers))
        self.n_symbols = max(1, int(self.n_symbols))
        self.fft_size = max(1, int(self.fft_size))
        self.cp_len = max(0, int(self.cp_len))
        self.delay_spread = max(0, int(self.delay_spread))
        self.payload_bits = max(1, int(self.payload_bits))
        self.seed = int(max(1, self.seed))
        self.sample_rate_hz = float(max(1.0, self.sample_rate_hz))
        self.subcarrier_spacing_hz = float(max(1.0, self.subcarrier_spacing_hz))
        self.alpha = float(self.alpha)
        self.beta = float(self.beta)
        self.c1 = float(self.c1)
        self.c2 = float(self.c2)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveConfig:
    """alpha/beta 自适应调优参数。"""

    enabled: bool = True
    objective: str = "evm"
    coarse_step: float = 0.2
    fine_step: float = 0.05
    alpha_min: float = 0.0
    alpha_max: float = 2.0
    beta_min: float = 0.0
    beta_max: float = 2.0
    stability_evals: int = 2
    cooldown_frames: int = 20
    max_evaluations: int = 400
    min_improvement_db: float = 0.15
    apply_best: bool = True
    seed: int = 20260428

    def normalized(self) -> "AdaptiveConfig":
        self.objective = str(self.objective or "evm").lower()
        if self.objective not in {"evm", "ber", "ser"}:
            self.objective = "evm"
        self.coarse_step = float(max(0.01, self.coarse_step))
        self.fine_step = float(max(0.005, self.fine_step))
        self.alpha_min = float(self.alpha_min)
        self.alpha_max = float(max(self.alpha_min, self.alpha_max))
        self.beta_min = float(self.beta_min)
        self.beta_max = float(max(self.beta_min, self.beta_max))
        self.stability_evals = int(max(1, self.stability_evals))
        self.cooldown_frames = int(max(0, self.cooldown_frames))
        self.max_evaluations = int(max(1, self.max_evaluations))
        self.seed = int(max(1, self.seed))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HardwareConfig:
    """USRP / 仿真回环硬件参数与 RF 安全约束。"""

    transport: str = "simulated-loopback"
    device_args: str = ""
    device_hint: str = "B210"
    mode: str = "loopback"
    center_freq_hz: float = 2.4e9
    sample_rate_hz: float = 1_000_000.0
    tx_gain_db: float = 10.0
    rx_gain_db: float = 20.0
    tx_antenna: str = "TX/RX"
    rx_antenna: str = "RX2"
    attenuator_db: float = 30.0
    ota_confirmed: bool = False
    frequency_whitelist_hz: List[List[float]] = field(
        default_factory=lambda: [[2.30e9, 2.50e9], [5.70e9, 5.90e9]]
    )
    max_loopback_tx_gain_db: float = 20.0
    max_ota_tx_gain_db: float = 5.0
    process_interval_ms: int = 100
    usrp_buffer_frames: int = 256
    tx_min_waveform_duration_ms: int = 100
    startup_settle_ms: int = 500
    startup_settle_windows: int = 5
    cfo_search_max_hz: float = 25_000.0
    residual_cfo_max_hz: float = 2_000.0

    def normalized(self) -> "HardwareConfig":
        self.transport = str(self.transport or "simulated-loopback").lower()
        self.mode = str(self.mode or "loopback").lower()
        self.sample_rate_hz = float(max(1.0, self.sample_rate_hz))
        self.tx_gain_db = float(self.tx_gain_db)
        self.rx_gain_db = float(self.rx_gain_db)
        self.attenuator_db = float(max(0.0, self.attenuator_db))
        self.process_interval_ms = int(max(10, self.process_interval_ms))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """一次实验的完整配置：波形 + 自适应 + 硬件 + 记录。"""

    waveform: WaveformConfig = field(default_factory=WaveformConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    operator: str = ""
    lab: str = ""
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    record_iq: bool = False
    runs_dir: str = "runs"

    def normalized(self) -> "ExperimentConfig":
        self.waveform.normalized()
        self.adaptive.normalized()
        self.hardware.normalized()
        self.runs_dir = str(self.runs_dir or "runs")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        data = dict(data or {})
        return cls(
            waveform=WaveformConfig(**dict(data.get("waveform") or {})),
            adaptive=AdaptiveConfig(**dict(data.get("adaptive") or {})),
            hardware=HardwareConfig(**dict(data.get("hardware") or {})),
            operator=str(data.get("operator", "")),
            lab=str(data.get("lab", "")),
            notes=str(data.get("notes", "")),
            tags=list(data.get("tags", [])),
            record_iq=bool(data.get("record_iq", False)),
            runs_dir=str(data.get("runs_dir", "runs")),
        ).normalized()

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
```

**验证：** `python -c "from waveform_sim.core.config import ExperimentConfig; print(ExperimentConfig().waveform.waveform)"` 输出 `FDIDM`。

### T1.3：新建 tests/test_config.py

**文件：** 新建 `tests/test_config.py`
**依赖：** T1.2

**步骤：** 创建文件，内容如下：

```python
"""配置模型测试：归一化、序列化往返、JSON 存取。"""
from waveform_sim.core.config import (
    AdaptiveConfig,
    ExperimentConfig,
    HardwareConfig,
    WaveformConfig,
)


def test_waveform_config_roundtrip():
    cfg = WaveformConfig(waveform="fdidm", mod_order="qpsk", snr_db=10.0).normalized()
    data = cfg.to_dict()
    cfg2 = WaveformConfig(**data).normalized()
    assert cfg2.waveform == "FDIDM"
    assert cfg2.mod_order == "QPSK"
    assert cfg2.snr_db == 10.0


def test_experiment_config_roundtrip():
    cfg = ExperimentConfig(
        waveform=WaveformConfig(waveform="OFDM", fft_size=64),
        adaptive=AdaptiveConfig(objective="ber"),
        hardware=HardwareConfig(transport="simulated-loopback"),
        operator="测试员",
        tags=["demo"],
    ).normalized()
    data = cfg.to_dict()
    cfg2 = ExperimentConfig.from_dict(data)
    assert cfg2.waveform.waveform == "OFDM"
    assert cfg2.adaptive.objective == "ber"
    assert cfg2.operator == "测试员"
    assert cfg2.tags == ["demo"]


def test_experiment_config_save_load(tmp_path):
    cfg = ExperimentConfig(runs_dir=str(tmp_path)).normalized()
    path = tmp_path / "cfg.json"
    cfg.save(path)
    cfg2 = ExperimentConfig.load(path)
    assert cfg2.to_dict() == cfg.to_dict()
```

**验证：** `python -m pytest tests/test_config.py -q` → `3 passed`。

### T1.4：阶段 1 全量验证与提交

**文件：** `waveform_sim/__init__.py`、`waveform_sim/core/`、`tests/test_config.py`
**依赖：** T1.1~T1.3

**步骤：**
1. `python -m pytest -q` → 期望 `9 passed`（6 + 3）。
2. `python -m compileall -q waveform_sim` → 退出码 0、无输出。
3. `git add waveform_sim/__init__.py waveform_sim/core tests/test_config.py`
4. `git commit -m "feat: 新增统一配置模型 waveform_sim/core/config.py（阶段1）"`
5. `git status` → 工作区干净

**验证：** pytest `9 passed`；提交存在；工作区干净。

## 执行顺序

```
T0.1 → T0.2 → T0.3 → T0.4 → T0.5 → T0.6 → T0.7 → T0.8 → T0.9
                                                              ↓
                                              T1.1 → T1.2 → T1.3 → T1.4
```

## 阶段 checkpoint

- 阶段 0 结束（T0.9 后）：向用户报告 pytest/compileall 输出与提交号，确认后再进入阶段 1。
- 阶段 1 结束（T1.4 后）：向用户报告验收结果，确认后再拆解阶段 2（公共 DSP 模块）。

