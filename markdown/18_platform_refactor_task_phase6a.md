# 平台工程化重构 Tasks（阶段 6a：硬件后端兼容壳）

> 前置：阶段 5 已验收通过（pytest 44 passed）。
> 范围：四个 `*_hardtest.py` 逐个改为兼容壳（重命名旧类 + 公开同名壳类 + 委托），**零逻辑改动**。
> 重要事实：四个 hardtest 类在 `__init__` 就会尝试打开 USRP，本机无设备无法构造；因此本阶段验证采用"模块导入 + 桩后端委托 + GUI 冒烟"，不做真构造。
> 后续：阶段 6b 再做 GNU Radio 流式逻辑搬移（`gr_flow.py`）与 FDIDM 308KB 拆分（信道 / FEC / 自适应 / 性能面），在 6a 验收后单独拆解。

## 文件清单

| 操作 | 文件 | 职责 |
|------|------|------|
| 修改 | `waveform_sim/hardware/afdm_hardtest.py` | 旧类改名 + 公开壳 `AFDMHardwareTest`（T6.1） |
| 修改 | `waveform_sim/hardware/ofdm_hardtest.py` | 旧类改名 + 公开壳 `OfdmHardwareTx`（T6.2） |
| 修改 | `waveform_sim/hardware/otfs_hardtest.py` | 旧类改名 + 公开壳 `OTFSHardwareTest`（T6.3） |
| 修改 | `waveform_sim/hardware/fdidm_hardtest.py` | 旧类改名 + 公开壳 `FDIDMHardwareTest`（T6.4） |
| 新建 | `tests/test_hardware_shells.py` | 兼容壳委托测试（参数化 4 用例） |

## 兼容壳模板（四个文件通用）

将旧类 `Xxx` 改名为 `_LegacyXxx`，并在文件末尾 `if __name__ == "__main__":` 块**之前**插入：

```python
# ---------------------------------------------------------------------------
# 阶段6：统一硬件后端兼容壳
# ---------------------------------------------------------------------------
class Xxx:
    """兼容壳：委托 _LegacyXxx，公开接口不变。"""

    def __init__(self, *args, backend=None, **kwargs):
        self._backend = backend if backend is not None else _LegacyXxx(*args, **kwargs)

    def __getattr__(self, name):
        backend = self.__dict__.get("_backend")
        if backend is not None and hasattr(backend, name):
            return getattr(backend, name)
        raise AttributeError(name)
```

> `backend=` 为可选注入参数，仅用于测试（无 USRP 环境）；UI 与既有调用不传它，走透传构造。

## 任务

### T6.1：AFDM 轮

**文件：** 修改 `waveform_sim/hardware/afdm_hardtest.py`；新建 `tests/test_hardware_shells.py`
**依赖：** 无

**步骤：**

1. 将 `class AFDMHardwareTest:`（约 39 行）改为 `class _LegacyAFDMHardwareTest:`。
2. 在 `if __name__ == "__main__":`（约 1308 行）之前插入兼容壳（模板，类名 `AFDMHardwareTest`，委托 `_LegacyAFDMHardwareTest`）。
3. 新建 `tests/test_hardware_shells.py`：

```python
"""硬件后端兼容壳测试（阶段 6a）。"""
import importlib

import pytest


class _StubBackend:
    def get_status(self):
        return {"stub": True}


CASES = [
    ("waveform_sim.hardware.afdm_hardtest", "AFDMHardwareTest", "_LegacyAFDMHardwareTest"),
]


@pytest.mark.parametrize("mod_name,shell_cls,legacy_cls", CASES)
def test_shell_delegates_to_backend(mod_name, shell_cls, legacy_cls):
    mod = importlib.import_module(mod_name)
    shell_cls = getattr(mod, shell_cls)
    legacy_cls = getattr(mod, legacy_cls)
    assert shell_cls.__name__ != legacy_cls.__name__
    shell = shell_cls(backend=_StubBackend())
    assert shell.get_status() == {"stub": True}
```

**验证：** `python -m pytest tests/test_hardware_shells.py -q` → `1 passed`；`python -m compileall -q waveform_sim` 退出码 0。

**提交：** `git add waveform_sim/hardware/afdm_hardtest.py tests/test_hardware_shells.py` → `git commit -m "refactor: AFDM 硬件后端兼容壳（阶段6a/1）"`。

### T6.2：OFDM 轮

**文件：** 修改 `waveform_sim/hardware/ofdm_hardtest.py`、`tests/test_hardware_shells.py`
**依赖：** T6.1

**步骤：**

1. 将 `class OfdmHardwareTx:`（约 28 行）改为 `class _LegacyOfdmHardwareTx:`。
2. 在 `if __name__ == "__main__":`（约 1474 行）之前插入兼容壳（类名 `OfdmHardwareTx`，委托 `_LegacyOfdmHardwareTx`）。
3. `tests/test_hardware_shells.py` 的 `CASES` 追加 `("waveform_sim.hardware.ofdm_hardtest", "OfdmHardwareTx", "_LegacyOfdmHardwareTx")`。

**验证：** `python -m pytest tests/test_hardware_shells.py -q` → `2 passed`；compileall 退出码 0。

**提交：** `git commit -m "refactor: OFDM 硬件后端兼容壳（阶段6a/2）"`（文件同上两个）。

### T6.3：OTFS 轮

**文件：** 修改 `waveform_sim/hardware/otfs_hardtest.py`、`tests/test_hardware_shells.py`
**依赖：** T6.2

**步骤：**

1. 将 `class OTFSHardwareTest:`（约 45 行）改为 `class _LegacyOTFSHardwareTest:`。
2. 在 `if __name__ == "__main__":`（约 1693 行）之前插入兼容壳（类名 `OTFSHardwareTest`，委托 `_LegacyOTFSHardwareTest`）。
3. `CASES` 追加 `("waveform_sim.hardware.otfs_hardtest", "OTFSHardwareTest", "_LegacyOTFSHardwareTest")`。

**验证：** `python -m pytest tests/test_hardware_shells.py -q` → `3 passed`；compileall 退出码 0。

**提交：** `git commit -m "refactor: OTFS 硬件后端兼容壳（阶段6a/3）"`（文件同上两个）。

### T6.4：FDIDM 轮

**文件：** 修改 `waveform_sim/hardware/fdidm_hardtest.py`、`tests/test_hardware_shells.py`
**依赖：** T6.3

**步骤：**

1. 将 `class FDIDMHardwareTest:`（约 316 行）改为 `class _LegacyFDIDMHardwareTest:`。
2. 在 `if __name__ == "__main__":`（约 5720 行）之前插入兼容壳（类名 `FDIDMHardwareTest`，委托 `_LegacyFDIDMHardwareTest`）。
3. `CASES` 追加 `("waveform_sim.hardware.fdidm_hardtest", "FDIDMHardwareTest", "_LegacyFDIDMHardwareTest")`。

**验证：** `python -m pytest tests/test_hardware_shells.py -q` → `4 passed`。

**提交：** `git commit -m "refactor: FDIDM 硬件后端兼容壳（阶段6a/4）"`（文件同上两个）。

### T6.5：阶段 6a 全量验证

**文件：** 无新增
**依赖：** T6.1~T6.4

**步骤：**

1. `python -m pytest -q` → 期望 `48 passed`（原 44 + 壳委托 4）。
2. `python -m compileall -q waveform_sim` → 退出码 0。
3. GUI 离屏冒烟 `tabs: 7`。
4. `git status` → 工作区干净。

**验证：** 三条命令通过，工作区干净。

## 执行顺序

```
T6.1 → T6.2 → T6.3 → T6.4 → T6.5
```

## 阶段 checkpoint

- T6.5 后向用户报告 pytest / GUI 输出与 4 个提交号；确认后再拆解阶段 6b（gr_flow 搬移 + FDIDM 大文件拆分）。

