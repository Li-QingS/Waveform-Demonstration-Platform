# 系统设计模板：FDIDM 实验平台模块化设计

## 设计请求：[系统/模块/功能]设计

设计一个面向 **FDIDM 波形验证、软件 TDL 仿真、USRP 硬件环路测试、参数优化与实验记录** 的模块化软件平台。

目标是在保留现有可运行原型的基础上，将 `fdidm_hardtest.py` 和 `fdidm_hardware_test_tab.py` 拆分为可测试、可扩展、可复现实验的工程结构。

---

## 背景：[项目上下文和业务需求]

论文提出 FDIDM 软波形，通过 FDIT/IFDIT 中 α、β 两个可调指数，实现 OFDM、OTFS 及其他交叉域波形的统一表示。FDIDM 面向集成卫星-地面网络，核心挑战包括：

- 高动态 LEO 场景下的 CFO、多普勒扩展和时变信道。
- 分数时延和分数多普勒。
- 不同服务/链路状态下对波形复杂度与可靠性的差异化需求。
- 既要支持整数指数波形切换，又要支持分数指数性能优化。

当前平台已经实现了：

- FDIDM 发射/接收链路。
- FDIT/IFDIT、Heisenberg/Wigner。
- QPSK/16QAM/64QAM。
- ZF/MMSE 等均衡。
- TDL-A/C/D 软件信道。
- RF、pure TDL、RF→TDL、TDL→RF 等链路模式。
- USRP B210/N210/X310 配置。
- PyQt5 图形界面。
- TX/RX 频谱、EVM、星座图和日志显示。

现阶段的主要需求是工程化重构与功能扩展，而不是继续把所有功能堆在一个类里。

---

## 功能需求：

1. **核心 FDIDM 算法模块**
   - FSIT/FDIT/IFDIT。
   - Heisenberg/Wigner 变换。
   - cross-domain 与 TF-domain 矩阵变换。
   - OFDM、OTFS、FDIDM 特殊点切换。

2. **调制解调模块**
   - QPSK、16QAM、64QAM。
   - bit/byte/frame/CRC 处理。
   - EVM、BER、残余相位/增益估计。

3. **信道模型模块**
   - NTN-TDL-A/C/D。
   - 分数时延。
   - 多普勒频移与多普勒扩展。
   - AWGN。
   - 可扩展到 CDL、实测信道回放。

4. **信道估计与均衡模块**
   - diag-TF。
   - full-H_TF。
   - TDL-param。
   - ZF/MMSE。
   - 后续可扩展 QRD-SIC、MRC、QTML。

5. **硬件运行模块**
   - GNU Radio/UHD 适配。
   - USRP 参数配置。
   - bounded buffer。
   - 运行线程、停止恢复、异常分类。
   - RF 安全策略。

6. **GUI 模块**
   - 参数配置。
   - 运行控制。
   - 频谱/EVM/星座图。
   - 日志与状态诊断。
   - α/β 推荐结果显示。

7. **实验记录与复现模块**
   - 保存配置快照。
   - 保存 BER/EVM/CFO/同步度量/条件数。
   - CSV/JSONL 导出。
   - 扫描任务记录。

8. **α/β 参数优化模块**
   - 基线比较：OFDM、OTFS、默认 FDIDM。
   - 粗扫+细扫。
   - RF 安全模式。
   - 推荐理由和扫描表导出。

---

## 非功能需求：

- [可扩展性要求]  
  - 新增信道模型、估计器、均衡器不应修改 GUI 主体逻辑。
  - 新增硬件设备时只实现 HardwareAdapter。
  - α/β 优化器可在 pure simulation 和 hardware runtime 两种模式复用。

- [可用性要求]  
  - 用户可以通过 GUI 一键切换 OFDM、OTFS、推荐 FDIDM。
  - 出错时提示具体原因：参数错误、硬件错误、同步失败、估计失败、解码失败。
  - 图形显示必须标记 stale/last-good/current-frame 状态。

- [安全性要求]  
  - RF 模式启动前必须检查 TX/RX 增益、频率和设备模式。
  - 对高 TX 增益、天线直连、无衰减器场景给出确认提示。
  - 停止测试必须保证 GNU Radio top_block 停止并释放资源。
  - 参数扫描不能在未知状态下持续发射。

- [性能指标]  
  - B210 1 MS/s 默认配置下可稳定运行。
  - GUI 100 ms 刷新不阻塞用户操作。
  - 解码线程可独立节流。
  - M×N≤256 时支持较完整诊断；M×N>1024 时限制 full-H。
  - pure TDL 模式下 α/β 粗扫可在可接受时间内完成。

---

## 技术约束：

- [技术栈限制]  
  - Python + NumPy。
  - PyQt5 + pyqtgraph。
  - GNU Radio + UHD。
  - 不强制引入大型深度学习框架。
  - 优先保持现有代码可迁移。

- [集成要求]  
  - 与现有 `FDIDMHardwareTest` 行为兼容。
  - 与现有 GUI 控件兼容。
  - 保留当前链路模式命名：`rf`、`tdl_a`、`rf_tdl_a`、`tdl_a_rf` 等。
  - 保留现有状态字段或提供兼容层。

- [部署环境]  
  - Windows + B210 是重要目标环境。
  - Linux + GNU Radio 是推荐调试环境。
  - 硬件不存在时应支持 pure TDL / pure simulation 模式。

---

## 期望输出：

### 1. 高层架构设计

```text
+--------------------------------------------------------------+
|                         PyQt GUI                             |
|  Controls | Plots | Logs | Alpha/Beta Sweep | Experiment UI  |
+-------------------------+------------------------------------+
                          |
                          v
+--------------------------------------------------------------+
|                    Application Backend                       |
| Config Manager | Runtime State | Snapshot | Error Handling   |
+-------------+---------------+---------------+----------------+
              |               |               |
              v               v               v
+------------------+  +----------------+  +--------------------+
| FDIDM DSP Core   |  | Channel Models |  | Hardware Runtime    |
| FDIT/IFDIT       |  | NTN-TDL/CDL    |  | GNU Radio/UHD       |
| Modem/Framing    |  | AWGN/Doppler   |  | Probe/Buffer        |
| Sync/Equalizer   |  | Replay         |  | RF Safety           |
+------------------+  +----------------+  +--------------------+
              |
              v
+--------------------------------------------------------------+
|             Experiment Logger / Alpha-Beta Optimizer          |
| JSONL/CSV | Sweep Records | Metrics | Recommendation Reason  |
+--------------------------------------------------------------+
```

---

### 2. 关键组件和接口

#### 2.1 配置模型

```python
@dataclass(frozen=True)
class HardwareConfig:
    device_type: str
    carrier_freq: float
    sample_rate: float
    tx_gain: float
    rx_gain: float
    tx_antenna: str = "TX/RX"
    rx_antenna: str = "RX2"

@dataclass(frozen=True)
class FDIDMConfig:
    M: int
    N: int
    cp_len: int
    alpha: float
    beta: float
    mod_order: str
    equalizer: str

@dataclass(frozen=True)
class ChannelConfig:
    mode: str
    tdl_model: str
    rms_delay_spread_ns: float
    doppler_hz: float
    doppler_spread_hz: float
    snr_db: float
```

---

#### 2.2 DSP Core

```python
class FDIDMCore:
    def build_tx_waveform(self, bits: np.ndarray, config: FDIDMConfig) -> np.ndarray:
        ...

    def decode_rx_frame(self, samples: np.ndarray, config: FDIDMConfig) -> dict:
        ...

    def fdit(self, y_tf: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        ...

    def ifdit(self, x_cross: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        ...
```

---

#### 2.3 Channel Model

```python
class ChannelModel:
    def reset(self) -> None:
        ...

    def process(self, samples: np.ndarray) -> np.ndarray:
        ...

class NTNTDLChannel(ChannelModel):
    def configure(self, config: ChannelConfig) -> None:
        ...
```

---

#### 2.4 Hardware Adapter

```python
class HardwareAdapter:
    def configure(self, hardware: HardwareConfig) -> None:
        ...

    def start(self, tx_samples: np.ndarray) -> None:
        ...

    def stop(self) -> None:
        ...

    def read_latest_rx(self, n: int) -> np.ndarray:
        ...
```

---

#### 2.5 Backend Orchestrator

```python
class FDIDMBackend:
    def configure(self, fdidm: FDIDMConfig, channel: ChannelConfig, hardware: HardwareConfig) -> None:
        ...

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def get_status(self) -> dict:
        ...

    def get_rx_constellation(self, mode: str) -> np.ndarray:
        ...

    def reset_csi_cache(self) -> None:
        ...
```

---

### 3. 数据模型

#### 实验配置记录

```json
{
  "experiment_id": "20260608_153000_fdidm",
  "fdidm": {
    "M": 16,
    "N": 16,
    "cp_len": 4,
    "alpha": 0.5,
    "beta": 1.0,
    "mod_order": "QPSK",
    "equalizer": "MMSE"
  },
  "channel": {
    "mode": "tdl_a_rf",
    "rms_delay_spread_ns": 1000,
    "doppler_hz": 0,
    "snr_db": 35
  },
  "hardware": {
    "device_type": "USRP B210",
    "sample_rate": 1000000,
    "carrier_freq": 2400000000,
    "tx_gain": 10,
    "rx_gain": 20
  }
}
```

#### 运行指标记录

```json
{
  "timestamp": "2026-06-08T15:30:15",
  "frame_ok": true,
  "decode_ok": true,
  "ber": 0.0,
  "evm_percent": 5.2,
  "sync_metric": 0.86,
  "cfo_hz": 13.4,
  "cond_h": 15.7,
  "estimator": "diag_tf",
  "rx_spectrum_stale": false
}
```

#### α/β 扫描记录

```json
{
  "alpha": 0.45,
  "beta": 1.0,
  "score": 0.93,
  "ber": 0.0,
  "evm_percent": 4.8,
  "sync_metric": 0.88,
  "cond_h": 11.2,
  "recommendation": true
}
```

---

### 4. API设计

#### 后端 API

```python
backend.configure(fdidm_config, channel_config, hardware_config)
backend.start()
backend.stop()
status = backend.get_status()
constellation = backend.get_rx_constellation(mode="post_equalized")
backend.reset_csi_cache()
```

#### 优化器 API

```python
optimizer = AlphaBetaOptimizer(backend)
result = optimizer.run(
    alpha_range=(0.0, 2.0),
    beta_range=(0.0, 2.0),
    coarse_step=0.25,
    fine_step=0.05,
    safe_rf_mode=True,
)
backend.configure_alpha_beta(result.best.alpha, result.best.beta)
```

#### 实验记录 API

```python
logger = ExperimentLogger("runs/20260608_fdidm.jsonl")
logger.write_config(fdidm_config, channel_config, hardware_config)
logger.write_status(backend.get_status())
logger.write_sweep_result(result)
```

---

### 5. 潜在挑战和解决方案

| 挑战 | 风险 | 解决方案 |
|---|---|---|
| full-H_TF 计算开销大 | 大 M/N 下卡顿或内存爆炸 | 默认禁用大规模 full-H；提供 max_order；优先 diag_tf/tdl_param |
| RF 链路与理论 TDL 不一致 | 参数化估计器失配 | RF/RF+TDL 默认 diag_tf；记录自动切换理由 |
| GUI 与后端状态竞争 | 图形闪烁、状态不一致 | 后端发布不可变快照；GUI 只读快照 |
| 参数扫描影响硬件安全 | 频繁重启、未知状态持续发射 | 扫描前保存配置，失败恢复；RF 安全模式只扫少量点 |
| 论文公式与工程实现偏差 | 难以判断错误来自算法还是硬件 | 为 FDIT、frame、QAM、TDL、估计器分别建立 pure simulation 测试 |
| 结果不可复现 | 无法写论文/报告 | 每次实验保存完整配置、版本、seed、指标 |
| Windows B210 overflow | 实时调度不足 | C++ probe、bounded buffer、处理节流、降低 GUI 计算量 |
| 代码继续膨胀 | 难以维护 | 分层模块化，后端 orchestrator 不再承载所有算法细节 |

---

## 分阶段实施建议

### 阶段 1：低风险整理

- 新增 `config.py`，用 dataclass 管理参数。
- 新增 `transforms.py`，迁移 FSIT/FDIT/IFDIT。
- 新增 `modem.py`，迁移 QAM、frame、CRC、EVM。
- 保留旧 `FDIDMHardwareTest` 对外接口，内部逐步调用新模块。

### 阶段 2：可测试化

- 添加 pytest。
- 测试 `fdit(ifdit(X))≈X`。
- 测试 QAM 往返。
- 测试 app frame CRC。
- 测试 pure TDL 静态链路。

### 阶段 3：运行链路稳定化

- 统一估计器选择策略。
- GUI 自动应用防抖。
- 后端发布轻量状态快照。
- 增加实验 JSONL 日志。

### 阶段 4：功能扩展

- α/β 自动推荐。
- 批量扫描和结果导出。
- RF 安全启动向导。
- 可选添加 QRD-SIC/MRC/QTML 研究模式。
