# 平台工程化重构 Tasks（阶段 6b：FDIDM 纯 Python 部分拆分）

> 前置：阶段 6a 已验收通过（pytest 48 passed，四个硬件后端已有兼容壳）。
> 范围：从 `fdidm_hardtest.py`（308KB）拆分**纯 Python、无 GNU Radio 依赖**的三个部分：环形缓冲、NTN-TDL 信道、FEC 编码。方法体原样搬移，零逻辑改动。
> 明确不做（留 6c）：自适应/性能面（约 800 行非连续块）与 GNU Radio 块（`_TopBlock` / `_TDLChannelBlock` / `_RXRingSink`）搬移——它们无法在本机无 USRP 环境验证，需要真机回归。

## 文件清单

| 操作 | 文件 | 职责 |
|------|------|------|
| 新建 | `waveform_sim/hardware/stream.py` | `SampleRing` 环形缓冲（T6b.1） |
| 修改 | `waveform_sim/hardware/fdidm_hardtest.py` | 删除原 `_SampleRing`，改为别名导入（T6b.1） |
| 新建 | `waveform_sim/hardware/channel.py` | `NTNTDLChannel` 软件信道（T6b.2） |
| 修改 | `waveform_sim/hardware/fdidm_hardtest.py` | 删除原 `_NTNTDLChannel`，改为别名导入（T6b.2） |
| 新建 | `waveform_sim/hardware/fec.py` | `FECMixin`：卷积编码/交织/校验（T6b.3） |
| 修改 | `waveform_sim/hardware/fdidm_hardtest.py` | 删除 FEC 方法块，类继承 `FECMixin`（T6b.3） |
| 新建 | `tests/test_hardware_stream.py` | 环形缓冲测试（3 用例） |
| 新建 | `tests/test_hardware_channel.py` | 信道测试（2 用例） |
| 新建 | `tests/test_hardware_fec.py` | FEC 测试（3 用例） |

## 任务

### T6b.1：stream.py 提取 `SampleRing`

**文件：** 新建 `waveform_sim/hardware/stream.py`；修改 `waveform_sim/hardware/fdidm_hardtest.py`
**依赖：** 无

**步骤：**

1. 新建 `waveform_sim/hardware/stream.py`，把 `fdidm_hardtest.py` 中 `_SampleRing` 类（约 16~87 行）整体搬入并改名 `SampleRing`，**方法体一字不改**。文件头：

```python
"""有界 NumPy 环形缓冲（从 fdidm_hardtest.py 搬移，阶段 6b）。"""
from __future__ import annotations

import threading

import numpy as np
```

2. `fdidm_hardtest.py`：删除原 `_SampleRing` 类定义（约 16~87 行），在头部 import 区加：

```python
from .stream import SampleRing as _SampleRing
```

其余所有 `_SampleRing(...)` 引用点（约 557/560/3638/3639 行）保持不变。

3. 新建 `tests/test_hardware_stream.py`：

```python
"""环形缓冲测试（阶段 6b）。"""
import numpy as np

from waveform_sim.hardware.stream import SampleRing


def test_ring_write_read_latest():
    r = SampleRing(capacity=1024)
    x = np.array([1 + 1j, 2 + 0j], dtype=np.complex64)
    r.write(x)
    assert r.total_written == 2
    assert len(r) == 2
    data, total, count = r.read_latest(2)
    assert total == 2 and count == 2
    assert np.array_equal(data, x)


def test_ring_wraps():
    r = SampleRing(capacity=8)  # 内部钳制到 1024
    r.write(np.ones(1024, dtype=np.complex64))
    r.write(np.arange(0, 6, dtype=np.float32).astype(np.complex64))
    assert r.total_written == 1030
    assert len(r) == 1024
    data, _, _ = r.read_latest(6)
    assert np.array_equal(data, np.arange(0, 6, dtype=np.float32).astype(np.complex64))


def test_ring_clear():
    r = SampleRing(capacity=8)
    r.write(np.ones(4, dtype=np.complex64))
    r.clear()
    assert r.total_written == 0
    assert len(r) == 0
```

**验证：** `python -m pytest tests/test_hardware_stream.py -q` → `3 passed`；`python -m pytest -q` → `51 passed`；`python -m compileall -q waveform_sim` 退出码 0。

**提交：** `git add waveform_sim/hardware/stream.py waveform_sim/hardware/fdidm_hardtest.py tests/test_hardware_stream.py` → `git commit -m "refactor: 提取 SampleRing 到 hardware/stream.py（阶段6b/1）"`。

### T6b.2：channel.py 提取 `NTNTDLChannel`

**文件：** 新建 `waveform_sim/hardware/channel.py`；修改 `waveform_sim/hardware/fdidm_hardtest.py`
**依赖：** T6b.1

**步骤：**

1. 新建 `waveform_sim/hardware/channel.py`，把 `fdidm_hardtest.py` 中 `_NTNTDLChannel` 类（约 89~314 行，含 docstring 与全部方法）整体搬入并改名 `NTNTDLChannel`，**方法体一字不改**。文件头：

```python
"""NTN-TDL 软件信道（从 fdidm_hardtest.py 搬移，阶段 6b）。"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
```

2. `fdidm_hardtest.py`：删除原 `_NTNTDLChannel` 类定义（约 89~314 行），在头部 import 区加：

```python
from .channel import NTNTDLChannel as _NTNTDLChannel
```

其余所有 `_NTNTDLChannel.X` 引用点（约 860/1185/1969/2034/2614/2620/2627/2644 行）保持不变。

3. 新建 `tests/test_hardware_channel.py`：

```python
"""NTN-TDL 软件信道测试（阶段 6b）。"""
import numpy as np

from waveform_sim.hardware.channel import NTNTDLChannel


def test_channel_configure_process():
    ch = NTNTDLChannel(
        sample_rate=1e6, model="tdl_a", rms_delay_spread_ns=100.0,
        doppler_hz=20.0, snr_db=30.0, seed=7,
    )
    y = ch.process(np.ones(256, dtype=np.complex64))
    assert y.shape == (256,)
    summary = ch.summary()
    assert "NTN-TDL-A" in summary


def test_channel_reset_and_profiles():
    ch = NTNTDLChannel(sample_rate=1e6, model="tdl_d")
    assert ch.model == "tdl_d"
    y1 = ch.process(np.ones(128, dtype=np.complex64))
    ch.reset()
    y2 = ch.process(np.ones(128, dtype=np.complex64))
    assert y1.shape == y2.shape == (128,)
```

**验证：** `python -m pytest tests/test_hardware_channel.py -q` → `2 passed`；`python -m pytest -q` → `53 passed`；compileall 退出码 0。

**提交：** `git commit -m "refactor: 提取 NTNTDLChannel 到 hardware/channel.py（阶段6b/2）"`。

### T6b.3：fec.py 提取 `FECMixin`

**文件：** 新建 `waveform_sim/hardware/fec.py`；修改 `waveform_sim/hardware/fdidm_hardtest.py`
**依赖：** T6b.2

**步骤：**

1. 新建 `waveform_sim/hardware/fec.py`，定义 `class FECMixin:`，把 `fdidm_hardtest.py` 中 FEC 连续方法块（约 991~1151 行，即 `_parity_u32` 到 `_max_payload_bytes_for_current_phy` 共 12 个方法）整体搬入，**方法体一字不改**（classmethod 保持 classmethod）。文件头：

```python
"""FDIDM FEC：卷积编码/交织/校验（从 fdidm_hardtest.py 搬移，阶段 6b）。"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class FECMixin:
```

2. `fdidm_hardtest.py`：删除该 FEC 方法块（约 991~1151 行）；类定义改为：

```python
class _LegacyFDIDMHardwareTest(FECMixin):
```

并在头部 import 区加：

```python
from .fec import FECMixin
```

3. 新建 `tests/test_hardware_fec.py`：

```python
"""FEC 测试（阶段 6b）。"""
import numpy as np

from waveform_sim.hardware.fec import FECMixin


def test_conv_roundtrip():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=100, dtype=np.int8)
    coded = FECMixin._conv_encode_bits(bits, flush=True)
    assert coded.size == 2 * (100 + 6)
    decoded = FECMixin._conv_decode_bits(coded, decoded_len=100, flushed=True)
    assert np.array_equal(decoded, bits)


def test_parity_u32():
    assert FECMixin._parity_u32(0) == 0
    assert FECMixin._parity_u32(1) == 1
    assert FECMixin._parity_u32(3) == 0
    assert FECMixin._parity_u32(7) == 1


def test_interleaver_roundtrip():
    obj = object.__new__(FECMixin)
    obj.coding_interleaver = True
    obj.PILOT_SEED = 12345
    obj.M = 8
    obj.N = 8
    bits = np.random.default_rng(2).integers(0, 2, size=200, dtype=np.int8)
    perm = obj._apply_bit_interleaver(bits)
    back = obj._remove_bit_interleaver(perm)
    assert np.array_equal(back, bits)
```

> 说明：`FECMixin` 的类方法（`_conv_*`）不依赖实例状态，可直接测试；实例方法（交织）通过 `object.__new__` 设置最小属性后测试，不触碰硬件路径。

**验证：** `python -m pytest tests/test_hardware_fec.py -q` → `3 passed`；`python -m pytest -q` → `56 passed`；compileall 退出码 0。

**提交：** `git add waveform_sim/hardware/fec.py waveform_sim/hardware/fdidm_hardtest.py tests/test_hardware_fec.py` → `git commit -m "refactor: 提取 FECMixin 到 hardware/fec.py（阶段6b/3）"`。

### T6b.4：全量验证

**文件：** 无新增
**依赖：** T6b.1~T6b.3

**步骤：**

1. `python -m pytest -q` → `56 passed`。
2. `python -m compileall -q waveform_sim` → 退出码 0。
3. GUI 离屏冒烟 `tabs: 7`。
4. `git status` → 工作区干净；`fdidm_hardtest.py` 行数应明显下降。

**验证：** 三条命令通过，工作区干净。

## 执行顺序

```
T6b.1 → T6b.2 → T6b.3 → T6b.4
```

## 阶段 checkpoint

- T6b.4 后向用户报告行数变化、pytest / GUI 输出与 3 个提交号；随后拆解阶段 6c（自适应/性能面与 GNU Radio 块搬移，需真机验证）。

