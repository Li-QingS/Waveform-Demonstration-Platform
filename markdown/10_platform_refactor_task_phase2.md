# 平台工程化重构 Tasks（阶段 2：公共 DSP 模块）

> 前置：阶段 0~1 已验收通过（pytest 9 passed）；`plan.md` 阶段 2 范围。
> 原则：只新增、不改旧代码；新模块从现有实现**原样移植数学**，不改算法；用一致性测试证明等价。

## 文件清单

| 操作 | 文件 | 职责 |
|------|------|------|
| 新建 | `waveform_sim/core/modem.py` | QPSK / 16QAM / 64QAM 统一映射与硬判决 |
| 新建 | `waveform_sim/core/metrics.py` | BER / SER / EVM 与 `LinkMetrics` |
| 新建 | `waveform_sim/core/transforms.py` | 四波形 modulate / demodulate 统一变换 |
| 新建 | `waveform_sim/core/waveforms.py` | `Waveform` 基类 + 四波形 + `create_waveform` |
| 新建 | `tests/test_modem.py` | 调制往返 + 与 FDIDM 现有星座一致性 |
| 新建 | `tests/test_metrics.py` | 指标计算测试 |
| 新建 | `tests/test_transforms.py` | 变换往返 + 与现有实现一致性 |
| 新建 | `tests/test_waveforms.py` | 波形接口往返测试 |

## 目标接口

### `waveform_sim/core/modem.py`

```python
def bits_per_symbol(mod_order: str) -> int          # QPSK→2, 16QAM→4, 64QAM→6
def constellation(mod_order: str) -> Tuple[np.ndarray, np.ndarray]  # (points, bit_labels)
def qam_modulate(bits: np.ndarray, mod_order: str = "16QAM") -> np.ndarray
def qam_demodulate(symbols: np.ndarray, mod_order: str = "16QAM") -> np.ndarray
def hard_decision_symbols(symbols: np.ndarray, mod_order: str = "16QAM") -> np.ndarray
```

### `waveform_sim/core/metrics.py`

```python
@dataclass
class LinkMetrics:
    frame_id: int = 0
    ber: float = 1.0
    fer: float = 1.0
    ser: float = 1.0
    evm_db: float = 0.0
    evm_rms: float = 1.0
    snr_db: float = 0.0
    sync_metric: float = 0.0
    cfo_est_hz: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    frames_processed: int = 0
    frames_decode_ok: int = 0
    bits_total: int = 0
    bit_errors: int = 0
    def to_dict(self) -> Dict[str, Any]: ...

def bit_error_rate(tx_bits, rx_bits) -> Tuple[float, int, int]   # (ber, errors, total)
def symbol_error_rate(tx_symbols, rx_symbols) -> float
def evm(reference, estimate) -> Tuple[float, float]              # (rms, db)
```

### `waveform_sim/core/transforms.py`

```python
def unitary_dft(n: int) -> np.ndarray
def gamma_matrix(n: int, eps: float) -> np.ndarray
def fdidm_modulate(symbols, m: int, n: int, alpha: float = 0.0, beta: float = 0.0) -> np.ndarray
def fdidm_demodulate(samples, m: int, n: int, alpha: float = 0.0, beta: float = 0.0) -> np.ndarray
def ofdm_modulate(symbols, fft_size: int = 64, cp_len: int = 16) -> np.ndarray
def ofdm_demodulate(samples, fft_size: int = 64, cp_len: int = 16, n_symbols: int | None = None) -> np.ndarray
def otfs_modulate(symbols, m: int, n: int, fft_size: int | None = None, cp_len: int = 0) -> np.ndarray
def otfs_demodulate(samples, m: int, n: int, fft_size: int | None = None, cp_len: int = 0) -> np.ndarray
def afdm_matrices(n: int, c1: float, c2: float) -> Tuple[np.ndarray, np.ndarray]   # (A_tx, A_rx)
def afdm_modulate(symbols, fft_size: int = 64, cp_len: int = 16, c1: float = 0.05, c2: float = 0.05) -> np.ndarray
def afdm_demodulate(samples, fft_size: int = 64, cp_len: int = 16, c1: float = 0.05, c2: float = 0.05, n_symbols: int | None = None) -> np.ndarray
```

### `waveform_sim/core/waveforms.py`

```python
class Waveform:  # 基类
    name = "BASE"
    @property
    def symbol_capacity(self) -> int: ...
    def modulate(self, symbols) -> np.ndarray: ...
    def demodulate(self, samples, n_symbols: int | None = None) -> np.ndarray: ...

class FDIDMWaveform(Waveform): ...   # symbol_capacity = m_subcarriers * n_symbols
class OFDMWaveform(Waveform): ...    # symbol_capacity = fft_size
class OTFSWaveform(Waveform): ...    # symbol_capacity = m_subcarriers * n_symbols
class AFDMWaveform(Waveform): ...    # symbol_capacity = fft_size

def create_waveform(config: WaveformConfig) -> Waveform   # 未知波形抛 ValueError
```

## 任务

### T2.1：新建 waveform_sim/core/modem.py

**文件：** 新建 `waveform_sim/core/modem.py`
**依赖：** 无

**步骤：**

1. 从 `waveform_sim/simulation/simple_fdidm_rx.py` 原样移植以下方法（数学不变，去掉 `self`/`cls`，改为模块级函数，`mod_order` 显式传参）：
   - `_build_gray_qam`（约 1673 行）→ `constellation(mod_order)`；
   - `_bits_to_int`（1694）、`_gray_to_binary`（1701）→ 模块级私有辅助；
   - `_map_bits_to_symbols`（1708）→ `qam_modulate(bits, mod_order)`（输入展平为位流，按 `bits_per_symbol` 分组）；
   - `_nearest_symbol_indices`（1719）→ `qam_demodulate` 的最近邻判决（argmin 距离，输出对应位标签）。
2. `bits_per_symbol`：`QPSK`→2、`16QAM`→4、`64QAM`→6，未知调制抛 `ValueError`；`mod_order` 统一大写化。
3. `hard_decision_symbols`：最近邻判决后返回星座点本身。
4. 模块只依赖 `numpy` 与标准库。

**验证：** `python -m pytest tests/test_modem.py -q` → `2 passed`。

### T2.2：新建 waveform_sim/core/metrics.py

**文件：** 新建 `waveform_sim/core/metrics.py`
**依赖：** 无

**步骤：** 按上述接口实现，内容如下：

```python
"""链路指标：BER / SER / EVM 与 LinkMetrics。"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class LinkMetrics:
    frame_id: int = 0
    ber: float = 1.0
    fer: float = 1.0
    ser: float = 1.0
    evm_db: float = 0.0
    evm_rms: float = 1.0
    snr_db: float = 0.0
    sync_metric: float = 0.0
    cfo_est_hz: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    frames_processed: int = 0
    frames_decode_ok: int = 0
    bits_total: int = 0
    bit_errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def bit_error_rate(tx_bits, rx_bits) -> Tuple[float, int, int]:
    a = np.asarray(tx_bits, dtype=np.uint8).reshape(-1)
    b = np.asarray(rx_bits, dtype=np.uint8).reshape(-1)
    n = int(min(a.size, b.size))
    if n <= 0:
        return 1.0, 0, 0
    errors = int(np.count_nonzero(a[:n] != b[:n]))
    return float(errors / n), errors, n


def symbol_error_rate(tx_symbols, rx_symbols) -> float:
    a = np.asarray(tx_symbols).reshape(-1)
    b = np.asarray(rx_symbols).reshape(-1)
    n = int(min(a.size, b.size))
    if n <= 0:
        return 1.0
    return float(np.count_nonzero(a[:n] != b[:n]) / n)


def evm(reference, estimate) -> Tuple[float, float]:
    ref = np.asarray(reference, dtype=np.complex128).reshape(-1)
    est = np.asarray(estimate, dtype=np.complex128).reshape(-1)
    n = int(min(ref.size, est.size))
    if n <= 0:
        return 1.0, 0.0
    err = est[:n] - ref[:n]
    denom = float(np.mean(np.abs(ref[:n]) ** 2)) + 1e-12
    rms = float(np.sqrt(np.mean(np.abs(err) ** 2) / denom))
    db = float(20.0 * math.log10(max(rms, 1e-12)))
    return rms, db
```

**验证：** `python -m pytest tests/test_metrics.py -q` → `4 passed`。

### T2.3：新建 waveform_sim/core/transforms.py

**文件：** 新建 `waveform_sim/core/transforms.py`
**依赖：** T2.1（无直接依赖，但保持同一风格）

**步骤：** 按目标接口实现，数学来源如下（原样移植，禁止改写）：

1. `unitary_dft(n)` ← `simple_fdidm_rx.py` 的 `_unitary_dft`（约 578 行）：`exp(-1j*2*pi*k*m/n)/sqrt(n)`。
2. `gamma_matrix(n, eps)` ← `_gamma`（约 583 行）：去掉缓存，`eps` 保留 `round(eps, 12)`，p=0..3 累加 `a_p * F^p`。
3. `fdidm_modulate/demodulate` ← `_fdit_matrices`（约 607 行）：
   - `tx = kron(gamma_matrix(n, -beta), gamma_matrix(m, alpha))`；
   - `rx = kron(gamma_matrix(n, beta), gamma_matrix(m, -alpha))`；
   - `modulate(x) = tx @ x.reshape(-1)`；`demodulate(y) = rx @ y.reshape(-1)`。
4. `ofdm_modulate/demodulate` ← `simple_ofdm_rx.py` `_build_tx_frame`（约 689 行）的**纯数据符号部分**（不含前导/训练/导频）：
   - 符号按 `(-1, fft_size)` 分块，每行 `ifft(row) * sqrt(fft_size)`，尾部追加 `cp_len` 个样本作 CP；
   - 解调：按 `(fft_size + cp_len)` 分块去 CP，每行 `fft(row) / sqrt(fft_size)`。
5. `otfs_modulate/demodulate` ← `simple_otfs_rx.py` 的 `_dd_to_tf`（约 1146 行）、`_tf_to_dd`（约 1150 行）、`_tf_to_time_cp`（约 1153 行）：
   - 调制：`tf = ifft(fft(dd, axis=1), axis=0)`；每列 `ifft(tf[:, k]) * sqrt(m)`，尾部追加 `cp_len`；
   - 解调：反向操作，`dd = ifft(fft(tf, axis=0), axis=1)`。
6. `afdm_matrices` ← `simple_afdm_rx.py` 的 `_build_afdm_mats`（约 395 行，已是模块级函数，原样复制）：
   - `F = fft(I)/sqrt(N)`；`D1/D2` 为对角 chirp；`A_tx = D1^H @ F^H @ D2^H`；`A_rx = D2 @ F @ D1`。
7. `afdm_modulate/demodulate`：符号按 `(-1, fft_size)` 分块，每块 `A_tx @ row`，追加 CP；解调反向，`A_rx @ row`。

**验证：** `python -m pytest tests/test_transforms.py -q` → `7 passed`。

### T2.4：新建 waveform_sim/core/waveforms.py

**文件：** 新建 `waveform_sim/core/waveforms.py`
**依赖：** T2.1~T2.3

**步骤：**

1. `Waveform` 基类：`name`、`symbol_capacity`（默认 `m_subcarriers * n_symbols`）、`modulate` / `demodulate`（未实现抛 `NotImplementedError`）。
2. `FDIDMWaveform`：`modulate` → `transforms.fdidm_modulate(symbols, m, n, alpha, beta)`；`demodulate` 同样参数；`symbol_capacity = m * n`。
3. `OFDMWaveform`：`symbol_capacity = fft_size`；`modulate/demodulate` 用 `fft_size`、`cp_len`。
4. `OTFSWaveform`：`symbol_capacity = m * n`；`modulate/demodulate` 用 `m_subcarriers`、`n_symbols`、`fft_size`、`cp_len`。
5. `AFDMWaveform`：`symbol_capacity = fft_size`；用 `fft_size`、`cp_len`、`c1`、`c2`。
6. `create_waveform(config)`：按 `config.waveform` 大写名分派，未知抛 `ValueError(f"Unsupported waveform: {config.waveform}")`。

**验证：** `python -m pytest tests/test_waveforms.py -q` → `5 passed`。

### T2.5：全量验证与提交

**文件：** 本阶段全部新增文件
**依赖：** T2.1~T2.4

**步骤：**

1. `python -m pytest -q` → 期望 `27 passed`（原 9 + 本阶段 18）。
2. `python -m compileall -q waveform_sim` → 退出码 0、无输出。
3. `python -c "from waveform_sim.core.waveforms import create_waveform; from waveform_sim.core.config import WaveformConfig; import numpy as np; [print(n, float(np.max(np.abs(w.demodulate(w.modulate(np.ones(w.symbol_capacity)), n_symbols=w.symbol_capacity) - np.ones(w.symbol_capacity))))) for n in ['FDIDM','OFDM','OTFS','AFDM'] for w in [create_waveform(WaveformConfig(waveform=n, m_subcarriers=8 if n!='OTFS' else 64, n_symbols=8, fft_size=64, cp_len=16).normalized())]]"` → 四行接近 0 的误差。
4. `git add waveform_sim/core tests/test_modem.py tests/test_metrics.py tests/test_transforms.py tests/test_waveforms.py`
5. `git commit -m "feat: 新增统一调制/变换/波形/指标模块（阶段2）"`
6. `git status` → 工作区干净

**验证：** pytest `27 passed`；提交存在；工作区干净。

## 测试代码（T2.1~T2.4 的验证依据）

### tests/test_modem.py

```python
import numpy as np

from waveform_sim.core import modem
from waveform_sim.simulation.simple_fdidm_rx import FDIDMTransceiver


def test_qam_roundtrip():
    rng = np.random.default_rng(123)
    for mod in ["QPSK", "16QAM", "64QAM"]:
        bits = rng.integers(0, 2, size=modem.bits_per_symbol(mod) * 32, dtype=np.uint8)
        syms = modem.qam_modulate(bits, mod)
        out = modem.qam_demodulate(syms, mod)[: bits.size]
        assert np.array_equal(bits, out)


def test_constellation_matches_fdidm():
    for order, mod in [(4, "QPSK"), (16, "16QAM"), (64, "64QAM")]:
        pts, labels = modem.constellation(mod)
        ref_pts, ref_labels = FDIDMTransceiver._build_gray_qam(order)
        assert np.allclose(pts, ref_pts, atol=1e-12)
        assert np.array_equal(labels, ref_labels)
```

### tests/test_metrics.py

```python
import numpy as np

from waveform_sim.core.metrics import LinkMetrics, bit_error_rate, evm, symbol_error_rate


def test_bit_error_rate():
    a = np.array([0, 1, 0, 1], dtype=np.uint8)
    b = np.array([0, 1, 1, 1], dtype=np.uint8)
    ber, errors, total = bit_error_rate(a, b)
    assert ber == 0.25
    assert errors == 1
    assert total == 4


def test_symbol_error_rate():
    a = np.array([1 + 0j, 0 + 1j])
    b = np.array([1 + 0j, 1 + 0j])
    assert symbol_error_rate(a, b) == 0.5


def test_evm():
    ref = np.array([1.0 + 0.0j, 0.0 + 1.0j])
    est = ref + 0.1 * np.array([1.0 + 0.0j, 0.0 + 1.0j])
    rms, db = evm(ref, est)
    assert abs(rms - 0.1) < 1e-9
    assert abs(db - 20.0 * np.log10(0.1)) < 1e-9


def test_link_metrics_to_dict():
    m = LinkMetrics(ber=0.1, evm_db=-20.0, alpha=0.5, beta=0.25)
    d = m.to_dict()
    assert d["ber"] == 0.1
    assert d["alpha"] == 0.5
```

### tests/test_transforms.py

```python
import numpy as np

from waveform_sim.core import transforms
from waveform_sim.simulation.simple_fdidm_rx import FDIDMTransceiver
from waveform_sim.simulation.simple_otfs_rx import OTFSTransceiver
from waveform_sim.simulation.simple_afdm_rx import _build_afdm_mats


def test_fdidm_gamma_matches_legacy():
    tc = FDIDMTransceiver(m_subcarriers=8, n_symbols=8)
    for n, eps in [(4, 0.0), (8, 0.3), (16, -0.5)]:
        assert np.allclose(transforms.gamma_matrix(n, eps), tc._gamma(n, eps), atol=1e-12)


def test_fdidm_roundtrip():
    rng = np.random.default_rng(7)
    x = (rng.standard_normal(64) + 1j * rng.standard_normal(64)).astype(np.complex128)
    y = transforms.fdidm_modulate(x, 8, 8, 0.4, -0.2)
    x2 = transforms.fdidm_demodulate(y, 8, 8, 0.4, -0.2)
    assert np.allclose(x2, x, atol=1e-9)


def test_ofdm_roundtrip():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((7, 64)) + 1j * rng.standard_normal((7, 64))
    y = transforms.ofdm_modulate(x.reshape(-1), 64, 16)
    x2 = transforms.ofdm_demodulate(y, 64, 16, n_symbols=x.size)
    assert np.allclose(x2, x.reshape(-1), atol=1e-9)


def test_otfs_matches_legacy():
    tc = OTFSTransceiver()
    rng = np.random.default_rng(9)
    dd = rng.standard_normal((64, 8)) + 1j * rng.standard_normal((64, 8))
    ref = tc._tf_to_time_cp(tc._dd_to_tf(dd))
    new = transforms.otfs_modulate(dd.reshape(-1), 64, 8, 64, tc.cp_len)
    assert np.allclose(new, ref, atol=1e-9)


def test_otfs_roundtrip():
    rng = np.random.default_rng(10)
    dd = rng.standard_normal((64, 8)) + 1j * rng.standard_normal((64, 8))
    y = transforms.otfs_modulate(dd.reshape(-1), 64, 8, 64, 16)
    x2 = transforms.otfs_demodulate(y, 64, 8, 64, 16)
    assert np.allclose(x2, dd.reshape(-1), atol=1e-9)


def test_afdm_matrices_match_legacy():
    for N, c1, c2 in [(64, 0.05, 0.05), (32, 0.04, 0.03)]:
        at, ar = transforms.afdm_matrices(N, c1, c2)
        lt, lr = _build_afdm_mats(N, c1, c2)
        assert np.allclose(at, lt, atol=1e-12)
        assert np.allclose(ar, lr, atol=1e-12)


def test_afdm_roundtrip():
    rng = np.random.default_rng(11)
    x = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    y = transforms.afdm_modulate(x, 64, 16, 0.05, 0.05)
    x2 = transforms.afdm_demodulate(y, 64, 16, 0.05, 0.05)
    assert np.allclose(x2, x, atol=1e-9)
```

### tests/test_waveforms.py

```python
import numpy as np
import pytest

from waveform_sim.core.config import WaveformConfig
from waveform_sim.core.waveforms import create_waveform


CASES = {
    "FDIDM": dict(waveform="FDIDM", m_subcarriers=8, n_symbols=8, alpha=0.4, beta=-0.2),
    "OFDM": dict(waveform="OFDM", fft_size=64, cp_len=16),
    "OTFS": dict(waveform="OTFS", m_subcarriers=64, n_symbols=8, fft_size=64, cp_len=16),
    "AFDM": dict(waveform="AFDM", fft_size=64, cp_len=16, c1=0.05, c2=0.05),
}


@pytest.mark.parametrize("name", ["FDIDM", "OFDM", "OTFS", "AFDM"])
def test_waveform_roundtrip(name):
    cfg = WaveformConfig(**CASES[name]).normalized()
    wf = create_waveform(cfg)
    rng = np.random.default_rng(42)
    n = wf.symbol_capacity
    symbols = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    samples = wf.modulate(symbols)
    out = wf.demodulate(samples, n_symbols=n)
    assert np.allclose(out, symbols, atol=1e-9)


def test_create_waveform_unknown():
    with pytest.raises(ValueError):
        create_waveform(WaveformConfig(waveform="UNKNOWN"))
```

## 执行顺序

```
T2.1 → T2.2 → T2.3 → T2.4 → T2.5
```

## 阶段 checkpoint

- T2.5 完成后向用户报告 pytest / compileall 输出与提交号，确认后再拆解阶段 3（FDIDM 接入统一引擎）。

