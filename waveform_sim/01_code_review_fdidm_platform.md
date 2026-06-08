# 代码审查模式：FDIDM 实验平台

## 审查重点：
- [安全性]
- [性能]
- [可读性]
- [最佳实践]
- [特定标准合规性]

## 项目上下文：[项目背景/编码规范]

本项目是基于论文 **Fractional Dual Index Division Multiplexing: A Soft Waveform Design Toward Integrated Satellite-Terrestrial Networks** 开发的 FDIDM 软件/硬件测试平台。论文核心是通过 FDIT/IFDIT 中两个可调指数 α、β，实现 OFDM、OTFS 及交叉域波形之间的软切换，并在分数时延/多普勒信道、TDL 信道和高动态卫星-地面融合场景下提升可靠性。

当前代码由两个主要文件组成：

- `fdidm_hardtest.py`：后端核心，包含 FDIDM 调制/解调、FDIT/IFDIT、Heisenberg/Wigner 变换、QAM 映射、同步/CFO 估计、信道估计、均衡、GNU Radio/UHD 运行链路、TDL 软件信道、状态与调试日志等。
- `fdidm_hardware_test_tab.py`：PyQt5/pyqtgraph 图形界面，包含 USRP 参数配置、FDIDM 参数配置、TDL 参数设置、运行控制、频谱/EVM/星座图显示、日志展示等。

建议编码规范：

- Python 3.9+，使用类型标注。
- 核心 DSP 算法与 GUI/硬件 I/O 分离。
- 数值算法应有单元测试和可复现实验参数。
- 硬件运行相关代码必须明确处理异常、停止流程和 RF 安全边界。
- 对实时链路避免无界缓存、频繁大矩阵重建、UI 线程重计算。

---

## 1. 关键问题（按严重程度排序）

### P0：硬件/RF 安全边界还不够显式

当前平台已经限制了部分输入范围，例如调制方式、均衡器、M/N、CP、训练幅度、USRP buffer 等都有边界裁剪。但是作为 B210/N210/X310 硬件测试平台，仍建议增加显式的 RF 安全策略：

- 启动前确认 TX/RX 频率、采样率、TX 增益、天线端口、是否接假负载/衰减器。
- 对 `tx_gain`、`carrier_freq`、`samp_rate` 建立硬件型号相关的推荐上限。
- “RF→TDL / TDL→RF / RF” 模式下，在 GUI 上给出明显的风险提示。
- 对未连接硬件、UHD 初始化失败、设备忙、时钟异常等错误进行分类显示，而不仅是字符串错误。

风险：误配置可能造成测试结果不可解释，严重时可能造成射频链路过载或违反实验室射频规范。

---

### P1：后端类职责过重，维护风险高

`FDIDMHardwareTest` 过于集中，承担了算法、帧结构、信道、硬件、运行状态、日志、图形数据缓存等多类职责。长类和长函数会导致：

- 修改一个算法参数时容易影响硬件运行链路。
- 难以为 FDIT、均衡、同步、信道估计分别编写独立测试。
- GUI 参数与后端参数的映射散落在多个位置。
- 后续实现 α/β 自动优化、实验记录、批量仿真会比较困难。

建议拆分为：

```text
fdidm/
  core/
    transforms.py       # FSIT/FDIT/IFDIT, gamma cache
    framing.py          # app frame, CRC, bit packing
    modem.py            # QAM mod/demod, EVM
    sync.py             # sync preamble, metric, CFO
    equalizers.py       # ZF/MMSE/full-H/diag-TF/TDL-param
  channel/
    ntn_tdl.py          # NTN-TDL software channel
  runtime/
    gnuradio_adapter.py # UHD/GNU Radio flowgraph
    buffers.py          # SampleRing and probes
  app/
    backend.py          # orchestrator
    config.py           # dataclass configs
  ui/
    fdidm_hardware_test_tab.py
```

---

### P1：实时路径中仍有较多 Python 级大数组复制

代码已经通过 `_SampleRing`、C++ probe/vector 链路、处理节流和 full-H 缓存降低了 B210 溢出风险，这是值得肯定的。但实时路径中仍存在以下潜在瓶颈：

- 软件 TDL `process()` 中使用 `np.concatenate((history, x))`，每个块都会分配新数组。
- full-H 模式需要构造和缓存 MN×MN 的复矩阵，M=N=64 时理论上非常重。
- `_try_process_rx_window_impl` 等函数逻辑较长，容易同时包含同步搜索、CFO、信道估计、均衡、CRC、图形缓存更新。
- UI 每 100 ms 刷新图形，若同时做 FFT、状态拼接、日志刷新，可能阻塞 Qt 主线程。

建议把实时接收、解码、图形显示分为不同节奏：

```text
UHD/GNU Radio 实时链路：尽量 C++ block / bounded probe
Python 解码线程：按 process_interval_ms 节流
GUI 刷新：只消费最近快照，不做重计算
日志/实验记录：低频批量写入
```

---

### P1：估计器选择逻辑合理，但需要形成“可测试策略对象”

GUI 中已经根据链路模式自动将某些 RF 场景从 `tdl_param` 或 `full_htf` 重定向到 `diag_tf`，这个机制符合工程直觉：真实 RF 链路包含模拟滤波、线缆、多径和硬件响应，纯 TDL 参数基不一定能表示。

问题是该规则目前主要写在 UI 层，后端也有类似防护逻辑。建议抽象为一个统一的策略函数，并在后端强制执行，避免 UI 与 backend 规则不一致。

建议接口：

```python
def resolve_estimator(requested: str, channel_mode: str, doppler_hz: float, doppler_spread_hz: float) -> tuple[str, str]:
    """
    Returns:
        resolved_estimator: 实际使用的估计器
        reason: 规则解释，用于 UI 展示和日志记录
    """
    involves_rf = channel_mode == "rf" or channel_mode.startswith("rf_tdl_") or channel_mode.endswith("_rf")

    if involves_rf and requested == "tdl_param":
        return "diag_tf", "RF 链路包含物理硬件响应，TDL 参数基不完整，自动切换到 diag_tf。"

    if involves_rf and channel_mode != "rf" and requested == "full_htf":
        return "diag_tf", "RF+TDL 级联下 full-H 估计不稳定，自动切换到 diag_tf。"

    if channel_mode != "rf" and requested == "full_htf" and (abs(doppler_hz) > 1e-9 or doppler_spread_hz > 1e-9):
        return "tdl_param", "动态软件 TDL 下 full-H 不适合作为默认估计器，自动切换到 tdl_param。"

    return requested, "使用用户指定估计器。"
```

---

### P2：异常处理过宽，错误分类不足

代码中有多处 `except Exception`，这对实验软件初期是常见写法，但后续会削弱可诊断性。建议至少区分：

- 参数错误：`ValueError`
- GNU Radio/UHD 导入或设备错误：`RuntimeError` 或自定义 `HardwareRuntimeError`
- 解码失败：`FrameDecodeError`
- 信道估计失败：`ChannelEstimationError`
- UI 操作错误：`UIStateError`

建议示例：

```python
class FDIDMError(Exception):
    pass

class HardwareRuntimeError(FDIDMError):
    pass

class ChannelEstimationError(FDIDMError):
    pass

class FrameDecodeError(FDIDMError):
    pass
```

---

### P2：GUI 与后端耦合偏紧

`fdidm_hardware_test_tab.py` 直接 `sys.path.append(...)` 后导入后端，这在原型阶段可用，但不利于包化部署和单元测试。建议改为标准包结构和相对/绝对导入。

改进前：

```python
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from hardware.fdidm_hardtest import FDIDMHardwareTest
```

改进后：

```python
from fdidm.runtime.backend import FDIDMHardwareTest
```

并在项目根目录提供：

```text
pyproject.toml
fdidm/
  __init__.py
  runtime/
    __init__.py
```

---

### P2：缺少可复现实验记录格式

平台已经输出 BER、EVM、CFO、同步度量、Hleak、条件数、TDL fit 等关键诊断指标。建议增加统一实验记录：

```json
{
  "timestamp": "2026-06-08T15:30:00",
  "hardware": {"device": "USRP B210", "sample_rate": 1000000, "carrier_freq": 2400000000},
  "fdidm": {"M": 16, "N": 16, "cp_len": 4, "alpha": 0.5, "beta": 1.0},
  "channel": {"mode": "tdl_a_rf", "tdl_ds_ns": 1000, "doppler_hz": 0, "snr_db": 35},
  "estimator": {"requested": "tdl_param", "resolved": "diag_tf"},
  "metrics": {"ber": 0.0, "evm_percent": 4.2, "cfo_hz": 12.3, "sync_metric": 0.91}
}
```

---

## 2. 改进建议（包括代码示例）

### 建议 1：使用 dataclass 管理配置

```python
from dataclasses import dataclass
from typing import Literal

ModOrder = Literal["QPSK", "16QAM", "64QAM"]
Equalizer = Literal["ZF", "MMSE"]
Estimator = Literal["full_htf", "diag_tf", "tdl_param"]

@dataclass(frozen=True)
class FDIDMConfig:
    carrier_freq: float = 2.4e9
    sample_rate: float = 1_000_000.0
    tx_gain: float = 10.0
    rx_gain: float = 20.0
    M: int = 16
    N: int = 16
    cp_len: int = 4
    alpha: float = 0.5
    beta: float = 1.0
    mod_order: ModOrder = "QPSK"
    equalizer: Equalizer = "MMSE"
    estimator: Estimator = "diag_tf"
    channel_mode: str = "tdl_a_rf"

    def validate(self) -> None:
        if not (-2.0 <= self.alpha <= 2.0):
            raise ValueError("alpha must be within [-2, 2].")
        if not (-2.0 <= self.beta <= 2.0):
            raise ValueError("beta must be within [-2, 2].")
        if not (4 <= self.M <= 64):
            raise ValueError("M must be within [4, 64].")
        if not (1 <= self.N <= 64):
            raise ValueError("N must be within [1, 64].")
        if not (0 <= self.cp_len <= self.M - 1):
            raise ValueError("cp_len must be within [0, M-1].")
```

---

### 建议 2：将 Gamma/FDIT 变换独立为纯算法模块

```python
class FDIDMTransform:
    def __init__(self):
        self._gamma_cache = {}

    @staticmethod
    def wrap_index(value: float) -> float:
        v = ((float(value) + 2.0) % 4.0) - 2.0
        return 2.0 if v <= -2.0 + 1e-12 else v

    @staticmethod
    def ap_weight(p: int, eps: float) -> complex:
        d = float(eps) - float(p)
        return (
            np.cos(d * np.pi / 4.0)
            * np.cos(2.0 * d * np.pi / 4.0)
            * np.exp(1j * 3.0 * d * np.pi / 4.0)
        )

    def gamma(self, order: int, eps: float) -> np.ndarray:
        key = (int(order), round(self.wrap_index(eps), 12))
        if key in self._gamma_cache:
            return self._gamma_cache[key]

        n = int(order)
        k = np.arange(n)
        F = np.exp(-1j * 2 * np.pi * np.outer(k, k) / n) / np.sqrt(n)
        powers = [np.eye(n), F, F @ F, F.conj().T]
        G = sum(powers[p] * self.ap_weight(p, key[1]) for p in range(4))
        self._gamma_cache[key] = G.astype(np.complex128)
        return self._gamma_cache[key]

    def ifdit(self, x_cross: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        M, N = x_cross.shape
        return self.gamma(M, alpha) @ x_cross @ self.gamma(N, -beta)

    def fdit(self, y_tf: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        M, N = y_tf.shape
        return self.gamma(M, -alpha) @ y_tf @ self.gamma(N, beta)
```

---

### 建议 3：为实验结果建立统一快照接口

```python
from dataclasses import dataclass, asdict
from datetime import datetime
import json

@dataclass
class RuntimeMetrics:
    timestamp: str
    frame_ok: bool
    ber: float
    evm_percent: float
    cfo_hz: float
    sync_metric: float
    cond_h: float
    estimator: str
    channel_mode: str
    alpha: float
    beta: float

def export_metrics(path: str, metrics: RuntimeMetrics) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(metrics), ensure_ascii=False) + "\n")
```

---

### 建议 4：UI 自动应用参数增加防抖

避免用户拖动 spinbox 时反复 stop/start backend。

```python
from PyQt5.QtCore import QTimer

def _init_debounce_timer(self):
    self._apply_debounce = QTimer(self)
    self._apply_debounce.setSingleShot(True)
    self._apply_debounce.setInterval(500)
    self._apply_debounce.timeout.connect(self._apply_params_to_backend)

def _on_params_changed(self, *_args):
    if self.auto_apply_check.isChecked():
        self._apply_debounce.start()
```

---

## 3. 值得称赞的部分

1. **论文机制映射比较完整**  
   代码中实现了 FSIT/FDIT/IFDIT、Heisenberg/Wigner、cross-domain 矩阵、TF 域信道估计、ZF/MMSE 均衡等核心链路，整体与论文中的发射机-信道-接收机结构相匹配。

2. **工程问题意识较强**  
   已经针对 B210 和 Windows 实时链路问题做了多处处理，例如 bounded ring buffer、C++ probe 优先、process interval 节流、power-of-two probe length、full-H 缓存等。

3. **实验可视化比较充分**  
   GUI 同时提供 TX 频谱/时域/X_TF、RX 频谱、EVM 曲线、星座图、发送/接收文本和日志，对硬件调试很有帮助。

4. **估计器选择体现真实链路理解**  
   UI/后端已经意识到 RF 链路不能简单用纯 TDL 参数基表示，并在部分场景下自动转向 `diag_tf`，这比机械套用论文模型更接近实验平台需求。

5. **参数边界和异常信息已有基础**  
   M/N/CP、调制方式、均衡方式、TDL 参数、buffer size 等均有一定边界控制，减少了明显的非法输入。

---

## 4. 整体质量评估

| 维度 | 评价 | 说明 |
|---|---:|---|
| 论文复现完整度 | 8/10 | 核心 FDIT/IFDIT、FDIDM 帧结构、TF/cross-domain 处理已覆盖，且加入硬件链路。 |
| 工程可运行性 | 7/10 | 已充分考虑 B210 实时性，但仍需更多环境自检、硬件安全提示和异常分类。 |
| 性能可控性 | 7/10 | 有缓存和节流设计，但 full-H、大矩阵、GUI刷新、软件TDL仍需分层优化。 |
| 可读性/可维护性 | 5/10 | 后端类过长、职责过多，建议模块化重构。 |
| 可测试性 | 5/10 | 纯算法部分可测试，但当前与运行状态/硬件耦合较强，需要拆分。 |
| 实验可复现性 | 6/10 | 有确定性 pilot/seed，但建议增加 JSONL/CSV 实验日志和配置快照。 |

**总体判断：** 该平台已经超过“简单论文复现脚本”，更接近一个可运行的 FDIDM 软硬件实验原型。下一阶段不应优先继续堆叠功能，而应进行模块化重构、实验记录标准化、硬件安全策略补强，并为 α/β 自动优化与批量实验预留接口。
