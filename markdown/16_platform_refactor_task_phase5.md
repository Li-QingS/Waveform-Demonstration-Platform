# 平台工程化重构 Tasks（阶段 5：硬件抽象层）

> 前置：阶段 4 已验收通过（pytest 34 passed，四波形统一引擎入口）。
> 原则：只新增、不改任何现有硬件文件（`waveform_sim/hardware/*_hardtest.py` 与 `__init__.py` 零改动）。

## 文件清单

| 操作 | 文件 | 职责 |
|------|------|------|
| 新建 | `waveform_sim/hardware/transport.py` | 传输抽象：simulated-loopback / UHD 占位 + 工厂 |
| 新建 | `waveform_sim/hardware/rf_safety.py` | loopback / OTA 增益、衰减器、频段白名单策略 |
| 新建 | `waveform_sim/hardware/device_manager.py` | `uhd_find_devices` 设备探测 |
| 新建 | `waveform_sim/hardware/iq_replay.py` | `.npy` / complex64 IQ 回放与保存 |
| 新建 | `tests/test_hardware_abstraction.py` | 硬件抽象层测试（10 个用例） |

## 任务

### T5.1：新建 waveform_sim/hardware/transport.py

**文件：** 新建 `waveform_sim/hardware/transport.py`
**依赖：** 无

**步骤：** 创建文件，内容如下：

```python
"""硬件传输抽象：simulated-loopback 与 UHD 占位（阶段 5，只增不改）。"""
from __future__ import annotations

import numpy as np

from ..core.config import HardwareConfig


class BaseTransport:
    def __init__(self, config: HardwareConfig):
        self.config = config.normalized()
        self.running = False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def send(self, samples):
        raise NotImplementedError

    def recv(self, count: int):
        raise NotImplementedError


class SimulatedLoopbackTransport(BaseTransport):
    def __init__(self, config: HardwareConfig):
        super().__init__(config)
        self._buffer = np.zeros(0, dtype=np.complex128)

    def send(self, samples):
        self._buffer = np.asarray(samples, dtype=np.complex128).reshape(-1).copy()

    def recv(self, count: int):
        count = int(max(0, count))
        if self._buffer.size == 0:
            return np.zeros(count, dtype=np.complex128)
        reps = int(np.ceil(count / self._buffer.size))
        return np.tile(self._buffer, reps)[:count]


class UHDTransport(BaseTransport):
    """UHD 真机占位：无 UHD 环境给出明确错误；流式逻辑在阶段 6 接入。"""

    def __init__(self, config: HardwareConfig):
        super().__init__(config)
        try:
            import uhd  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "UHD Python bindings are not available. Install UHD/GNU Radio or use simulated-loopback."
            ) from exc

    def send(self, samples):
        raise NotImplementedError("Real UHD streaming will be wired in phase 6")

    def recv(self, count: int):
        raise NotImplementedError("Real UHD streaming will be wired in phase 6")


def create_transport(config: HardwareConfig) -> BaseTransport:
    cfg = config.normalized()
    if cfg.transport in {"sim", "simulated", "simulated-loopback", "loopback"}:
        return SimulatedLoopbackTransport(cfg)
    if cfg.transport in {"uhd", "usrp"}:
        return UHDTransport(cfg)
    raise ValueError(f"Unsupported transport: {cfg.transport}")
```

**验证：** `python -c "from waveform_sim.hardware.transport import create_transport; from waveform_sim.core.config import HardwareConfig; t = create_transport(HardwareConfig()); print(type(t).__name__)"` 输出 `SimulatedLoopbackTransport`。

### T5.2：新建 waveform_sim/hardware/rf_safety.py

**文件：** 新建 `waveform_sim/hardware/rf_safety.py`
**依赖：** T5.1（无直接依赖，仅共用配置）

**步骤：** 创建文件，内容如下：

```python
"""RF 安全策略：loopback / OTA 增益、衰减器、频段白名单校验（阶段 5）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..core.config import HardwareConfig


class RFSafetyViolation(ValueError):
    pass


@dataclass
class RFSafetyDecision:
    allowed: bool
    warnings: List[str]


class RFSafetyPolicy:
    def __init__(self, strict: bool = True):
        self.strict = bool(strict)

    def validate(self, config: HardwareConfig) -> RFSafetyDecision:
        cfg = config.normalized()
        warnings: List[str] = []
        if cfg.mode == "loopback":
            if cfg.tx_gain_db > cfg.max_loopback_tx_gain_db:
                warnings.append(
                    f"Loopback TX gain {cfg.tx_gain_db:.1f} dB exceeds limit {cfg.max_loopback_tx_gain_db:.1f} dB."
                )
            if cfg.attenuator_db < 20.0:
                warnings.append("Loopback mode requires at least 20 dB external attenuation.")
        elif cfg.mode == "ota":
            if not cfg.ota_confirmed:
                warnings.append("OTA mode requires explicit antenna/environment confirmation.")
            if cfg.tx_gain_db > cfg.max_ota_tx_gain_db:
                warnings.append(f"OTA TX gain {cfg.tx_gain_db:.1f} dB exceeds limit {cfg.max_ota_tx_gain_db:.1f} dB.")
            if not self._freq_allowed(cfg.center_freq_hz, cfg.frequency_whitelist_hz):
                warnings.append(f"Center frequency {cfg.center_freq_hz:.0f} Hz is outside configured whitelist.")
        else:
            warnings.append(f"Unknown RF mode: {cfg.mode}")
        if self.strict and warnings:
            raise RFSafetyViolation("RF safety policy blocked run: " + "; ".join(warnings))
        return RFSafetyDecision(allowed=not warnings, warnings=warnings)

    @staticmethod
    def _freq_allowed(freq_hz: float, whitelist: List[List[float]]) -> bool:
        f = float(freq_hz)
        for lo, hi in whitelist or []:
            if float(lo) <= f <= float(hi):
                return True
        return False
```

**验证：** `python -c "from waveform_sim.hardware.rf_safety import RFSafetyPolicy, RFSafetyViolation; from waveform_sim.core.config import HardwareConfig; from pytest import raises; raises(RFSafetyViolation, RFSafetyPolicy().validate, HardwareConfig(mode='loopback', tx_gain_db=40.0, attenuator_db=0.0)); print('rf ok')"` 输出 `rf ok`。

### T5.3：新建 device_manager.py 与 iq_replay.py

**文件：** 新建 `waveform_sim/hardware/device_manager.py`、`waveform_sim/hardware/iq_replay.py`
**依赖：** 无

**步骤：** 创建两个文件，内容如下：

`device_manager.py`：

```python
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
```

`iq_replay.py`：

```python
"""IQ 回放：.npy 与 complex64 原始文件（阶段 5）。"""
from __future__ import annotations

from pathlib import Path
import numpy as np


class IQReplaySource:
    def __init__(self, path: str | Path, loop: bool = True):
        self.path = Path(path)
        self.loop = bool(loop)
        self.samples = self._load(self.path)
        self.offset = 0

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            arr = np.load(path)
        else:
            arr = np.fromfile(path, dtype=np.complex64)
        return np.asarray(arr, dtype=np.complex128).reshape(-1)

    def read(self, count: int) -> np.ndarray:
        count = int(max(0, count))
        if count == 0 or self.samples.size == 0:
            return np.zeros(0, dtype=np.complex128)
        end = self.offset + count
        if end <= self.samples.size:
            out = self.samples[self.offset:end]
            self.offset = end
            return out.copy()
        if not self.loop:
            out = self.samples[self.offset:]
            self.offset = self.samples.size
            return out.copy()
        parts = [self.samples[self.offset:]]
        remaining = count - parts[0].size
        while remaining > 0:
            take = min(remaining, self.samples.size)
            parts.append(self.samples[:take])
            remaining -= take
        self.offset = count % self.samples.size
        return np.concatenate(parts).astype(np.complex128)


def save_iq(path: str | Path, samples) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(samples, dtype=np.complex64)
    if path.suffix.lower() == ".npy":
        np.save(path, arr)
    else:
        arr.tofile(path)
```

**验证：** `python -c "from waveform_sim.hardware.device_manager import probe_devices; from waveform_sim.hardware.iq_replay import IQReplaySource, save_iq; import numpy as np, tempfile, os; p = os.path.join(tempfile.mkdtemp(), 'iq.npy'); save_iq(p, np.array([1+1j])); print(len(probe_devices()) > 0, IQReplaySource(p).read(1).size)"` 输出 `True 1`。

### T5.4：新建 tests/test_hardware_abstraction.py

**文件：** 新建 `tests/test_hardware_abstraction.py`
**依赖：** T5.1~T5.3

**步骤：** 创建文件，内容如下：

```python
"""硬件抽象层测试：RF safety、transport、IQ replay、设备探测。"""
import numpy as np
import pytest

from waveform_sim.core.config import HardwareConfig
from waveform_sim.hardware.device_manager import probe_devices
from waveform_sim.hardware.iq_replay import IQReplaySource, save_iq
from waveform_sim.hardware.rf_safety import RFSafetyPolicy, RFSafetyViolation
from waveform_sim.hardware.transport import create_transport


def test_rf_safety_blocks_bad_loopback():
    cfg = HardwareConfig(mode="loopback", tx_gain_db=40.0, attenuator_db=0.0)
    with pytest.raises(RFSafetyViolation):
        RFSafetyPolicy(strict=True).validate(cfg)


def test_rf_safety_allows_good_loopback():
    cfg = HardwareConfig(mode="loopback", tx_gain_db=10.0, attenuator_db=30.0)
    decision = RFSafetyPolicy(strict=True).validate(cfg)
    assert decision.allowed
    assert decision.warnings == []


def test_rf_safety_blocks_ota_without_confirm():
    cfg = HardwareConfig(mode="ota", ota_confirmed=False, center_freq_hz=2.4e9)
    with pytest.raises(RFSafetyViolation):
        RFSafetyPolicy(strict=True).validate(cfg)


def test_rf_safety_blocks_out_of_whitelist():
    cfg = HardwareConfig(mode="ota", ota_confirmed=True, tx_gain_db=3.0, center_freq_hz=1.0e9)
    with pytest.raises(RFSafetyViolation):
        RFSafetyPolicy(strict=True).validate(cfg)


def test_transport_loopback_roundtrip():
    tx = create_transport(HardwareConfig(transport="simulated-loopback"))
    x = np.array([1 + 2j, 3 - 1j, 0.5 + 0.5j], dtype=np.complex128)
    tx.send(x)
    y = tx.recv(6)
    assert y.shape == (6,)
    assert np.allclose(y, np.tile(x, 2))


def test_transport_factory_unknown():
    with pytest.raises(ValueError):
        create_transport(HardwareConfig(transport="unknown"))


def test_uhd_transport_placeholder():
    try:
        t = create_transport(HardwareConfig(transport="uhd"))
    except RuntimeError as exc:
        assert "not available" in str(exc)
        return
    with pytest.raises(NotImplementedError):
        t.send(np.zeros(4, dtype=np.complex128))


def test_iq_replay_npy(tmp_path):
    path = tmp_path / "iq.npy"
    x = np.array([1 + 1j, 2 + 0j, 0 - 3j], dtype=np.complex128)
    save_iq(path, x)
    src = IQReplaySource(path, loop=True)
    assert np.allclose(src.read(3), x)
    assert np.allclose(src.read(6), np.tile(x, 2))


def test_iq_replay_raw_complex64(tmp_path):
    path = tmp_path / "iq.bin"
    x = np.array([1 + 1j, 2 + 0j], dtype=np.complex128)
    save_iq(path, x)
    src = IQReplaySource(path, loop=False)
    assert np.allclose(src.read(1), x[:1])
    assert np.allclose(src.read(5), x[1:])


def test_device_manager_probe():
    devices = probe_devices(timeout=5.0)
    assert isinstance(devices, list) and len(devices) >= 1
    assert hasattr(devices[0], "available")
```

**验证：** `python -m pytest tests/test_hardware_abstraction.py -q` → `10 passed`。

### T5.5：全量验证与提交

**文件：** 本阶段全部新增文件
**依赖：** T5.1~T5.4

**步骤：**

1. `python -m pytest -q` → 期望 `44 passed`（原 34 + 本阶段 10）。
2. `python -m compileall -q waveform_sim` → 退出码 0、无输出。
3. `git add waveform_sim/hardware/transport.py waveform_sim/hardware/rf_safety.py waveform_sim/hardware/device_manager.py waveform_sim/hardware/iq_replay.py tests/test_hardware_abstraction.py`
4. `git commit -m "feat: 新增硬件抽象层 transport/rf_safety/device_manager/iq_replay（阶段5）"`
5. `git status` → 工作区干净

**验证：** pytest `44 passed`；提交存在；工作区干净。

## 执行顺序

```
T5.1 → T5.2 → T5.3 → T5.4 → T5.5
```

## 阶段 checkpoint

- T5.5 后向用户报告 pytest / compileall 输出与提交号；确认后再拆解阶段 6（硬件后端兼容壳，FDIDM 308KB 拆分）。

