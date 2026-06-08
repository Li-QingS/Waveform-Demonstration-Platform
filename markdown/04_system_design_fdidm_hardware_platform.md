# 04 系统设计模板：FDIDM 硬件验证平台专业落地设计

## 设计请求

设计一个 **面向真实 USRP/RF 链路的 FDIDM 硬件验证平台**。

平台目标不是单纯复现论文仿真，也不是做 alpha/beta 扫描推荐，而是把 FDIDM 波形、收发链路、硬件控制、接收诊断、日志系统和实验记录整合为一个可落地、可调试、可扩展的专业工程系统。

---

## 背景

论文提出 FDIDM，通过两个可调指数 alpha 和 beta 实现跨时域、频域、延迟域和多普勒域的切换与融合。整数指数可以退化为 OFDM、OTFS 等已有波形，分数指数可以在特定信道条件下获得更好的性能。

但在工程落地中，真实硬件链路包含大量论文仿真没有覆盖的因素：

- USRP 前端频响；
- 采样时钟误差；
- 本振频偏；
- ADC/DAC 动态范围；
- UHD/GNU Radio 缓冲；
- 操作系统调度；
- 线缆、衰减器、天线和环境多径；
- GUI 绘图和 Python 解码带来的实时性压力。

因此，本平台设计必须坚持：

1. 真实 RF 链路优先；
2. 软件 TDL 只作为辅助测试，不作为主结论；
3. alpha/beta 以手动配置和固定预设为主，不做实时扫描推荐；
4. 每个代码模块都有日志输出；
5. 每次实验都有配置、日志、指标和图表快照；
6. 代码分层清晰，后续便于维护和扩展。

---

# 功能需求

## 1. 核心功能 1：FDIDM 波形收发

- 支持 OFDM 预设：alpha=0, beta=0；
- 支持 OTFS 预设：alpha=1, beta=1；
- 支持 FDIDM 实验预设：alpha=0.5, beta=1.0；
- 支持手动设置 alpha/beta；
- 支持 QPSK、16QAM、64QAM；
- 支持应用层文本 payload；
- 支持 CRC 校验；
- 支持帧结构：guard、sync、pilot、data。

## 2. 核心功能 2：真实硬件链路控制

- 支持 USRP B210/N210/X310；
- 支持中心频率、采样率、TX/RX gain、天线口配置；
- 支持 RF 直连或 OTA；
- 支持启动、停止、重构 flowgraph；
- 支持 UHD overflow/underflow 监控。

## 3. 核心功能 3：接收机处理

- 同步检测；
- CFO 估计与补偿；
- pilot 提取；
- 信道估计；
- ZF/MMSE 均衡；
- 软/硬判决；
- BER/EVM/CRC 评估。

## 4. 核心功能 4：诊断可视化

- TX 频谱/时域；
- RX 原始频谱；
- RX frame/pilot/data 源切换；
- 星座图：均衡前、均衡后、Y_TF、最近好帧、原始 IQ；
- EVM 曲线；
- 解调状态；
- 运行日志。

## 5. 核心功能 5：结构化日志与实验导出

- 每次运行生成 run_id；
- 保存 `config.json`；
- 保存 `events.jsonl`；
- 保存 `metrics.csv`；
- 导出关键图；
- 生成 `report.md`。

---

# 非功能需求

## 可扩展性要求

- 核心数学模块不依赖 GUI 和硬件；
- 后端服务层不依赖 PyQt；
- GUI 可替换为 CLI 或 Web 控制台；
- 信道估计器可插拔；
- 日志格式稳定，后续可做自动分析。

## 可用性要求

- 默认参数能在 B210 上安全启动；
- 错误信息必须可读；
- 自动参数修正必须可见；
- 停止实验后能恢复到可重新配置状态；
- GUI 不应假死。

## 安全性要求

- 默认 TX gain 保守；
- 高 TX gain 或 OTA 模式给出警告；
- 非法频率、采样率、增益必须拒绝或显式警告；
- 运行中参数变更需要停止并重构链路；
- 所有异常必须记录。

## 性能指标

- 1 MHz 采样率下 B210 稳定运行；
- GUI 刷新不阻塞 RX 实时流；
- 接收处理耗时可记录；
- 日志写入异步化；
- 大数组不进入日志，只保存摘要或图像。

---

# 技术约束

## 技术栈限制

- Python + NumPy + GNU Radio + UHD + PyQt5 + pyqtgraph；
- 不引入复杂分布式框架；
- 不把核心算法绑定到 GUI 线程；
- 不依赖纯软件仿真作为主功能。

## 集成要求

- 保留现有 `FDIDMHardwareTest` 的主要能力，但拆分职责；
- 保留现有 GUI 的主要控件，但把核心策略迁移到 service/backend；
- 保留 `_SampleRing` 思路；
- 保留 CRC 和文本 payload 验证；
- 保留 diag_tf、full_htf、tdl_param，但重新定义适用边界。

## 部署环境

- 实验室 PC；
- Windows ；
- USRP B210/N210/X310；
- GNU Radio/UHD 已安装；
- 可选外部衰减器、线缆、天线、参考时钟。

---

# 期望输出

## 1. 高层架构设计

```text
┌────────────────────────────────────────────────────────────┐
│ GUI / CLI Layer                                             │
│ hardware_tab.py, plot_controller.py, log_viewer.py          │
└───────────────────────┬────────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────────┐
│ Service Layer                                               │
│ experiment_service.py, run_state.py, config_models.py       │
│ event_logger.py                                             │
└───────────────────────┬────────────────────────────────────┘
                        │
        ┌───────────────┴────────────────┐
        │                                │
┌───────▼────────┐              ┌────────▼─────────┐
│ Core DSP Layer │              │ Hardware Layer    │
│ fdit.py        │              │ uhd_flowgraph.py  │
│ modem.py       │              │ device_manager.py │
│ framing.py     │              │ rf_safety.py      │
│ metrics.py     │              │ stream_monitor.py │
└───────┬────────┘              └────────┬─────────┘
        │                                │
┌───────▼────────────────────────────────▼─────────┐
│ Receiver Layer                                    │
│ synchronizer.py, cfo.py, channel_estimator.py     │
│ equalizer.py, decoder.py                          │
└───────┬──────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────┐
│ Diagnostics Layer                                 │
│ snapshot.py, report_exporter.py, health_check.py  │
└──────────────────────────────────────────────────┘
```

---

## 2. 关键组件和接口

| 组件 | 职责 | 输入 | 输出 | 必须日志 |
|---|---|---|---|---|
| `ExperimentService` | 管理实验生命周期 | `ExperimentConfig` | 状态、backend 控制 | `RUN_CREATED`, `RUN_STARTED`, `RUN_STOPPED`, `RUN_FAILED` |
| `ConfigValidator` | 校验和修正配置 | config | validated config | `CONFIG_VALIDATED`, `CONFIG_REJECTED`, `ESTIMATOR_AUTO_SWITCH` |
| `EventLogger` | 统一结构化日志 | event | jsonl、GUI摘要 | `LOG_START`, `LOG_DROP_WARN`, `LOG_STOP` |
| `FditCore` | Gamma/FDIT/IFDIT | alpha/beta/M/N | matrices, transformed grid | `GAMMA_CACHE_HIT`, `GAMMA_BUILT`, `FDIT_DONE` |
| `FrameBuilder` | 应用帧和物理帧 | payload, waveform config | TX frame | `APP_FRAME_BUILT`, `PAYLOAD_TOO_LARGE`, `TX_FRAME_BUILT` |
| `UhdFlowgraph` | USRP/GNU Radio 流图 | hardware config, TX samples | RX samples | `FLOWGRAPH_BUILD`, `FLOWGRAPH_START`, `UHD_OVERFLOW`, `FLOWGRAPH_STOP` |
| `StreamMonitor` | RX 流健康检查 | sample counters | rate/stale status | `RX_RATE_SAMPLE`, `RX_STALE`, `RX_RATE_DROP` |
| `Synchronizer` | 同步 | RX window | sync index/metric | `SYNC_LOCKED`, `SYNC_LOW_METRIC`, `SYNC_LOST` |
| `CfoEstimator` | CFO 估计补偿 | preamble/pilot | CFO Hz | `CFO_ESTIMATED`, `CFO_COMP_APPLIED` |
| `ChannelEstimator` | CSI 估计 | pilot samples | channel matrix/vector | `CSI_EST_START`, `CSI_EST_DONE`, `CSI_COND_WARN` |
| `Equalizer` | 均衡 | symbols, CSI | equalized symbols | `EQ_START`, `EQ_DONE`, `EQ_WARN` |
| `Decoder` | 解码和 CRC | symbols | payload/result | `FRAME_DECODE_OK`, `FRAME_DECODE_FAIL`, `CRC_FAIL` |
| `ReportExporter` | 导出实验报告 | config, metrics, figures | report.md | `REPORT_EXPORT_START`, `REPORT_EXPORT_DONE` |

---

## 3. 数据模型

```python
@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    created_at: str
    operator: str = "unknown"
    project: str = "FDIDM hardware platform"
    notes: str = ""

@dataclass(frozen=True)
class ExperimentConfig:
    hardware: HardwareConfig
    waveform: WaveformConfig
    receiver: ReceiverConfig
    channel_mode: str
    tx_text: str

@dataclass
class RuntimeStatus:
    state: str
    frame_count: int
    decode_ok_count: int
    last_sync_metric: float
    last_cfo_hz: float
    last_evm_percent: float
    last_ber: float
    rx_samples_seen: int
    rx_stale: bool
    last_bad_reason: str

@dataclass
class DecodeResult:
    ok: bool
    payload: bytes
    text: str
    ber: float
    evm_percent: float
    sync_metric: float
    cfo_hz: float
    reason: str
```

---

## 4. API 设计

### ExperimentService API

```python
class ExperimentService:
    def configure(self, config: ExperimentConfig) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def apply_config(self, config: ExperimentConfig) -> None: ...
    def reset_csi(self) -> None: ...
    def get_status(self) -> RuntimeStatus: ...
    def export_report(self) -> str: ...
```

### Backend API

```python
class FdidmHardwareBackend:
    def build(self, config: ExperimentConfig) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def wait(self, timeout: float | None = None) -> None: ...
    def configure(self, config: ExperimentConfig) -> None: ...
    def reset_csi(self) -> None: ...
    def read_status(self) -> RuntimeStatus: ...
    def read_plot_data(self) -> dict: ...
```

### GUI 调用方式

```python
class HardwareTab(QWidget):
    def on_connect_clicked(self):
        cfg = self.read_config_from_widgets()
        self.service.configure(cfg)

    def on_start_clicked(self):
        self.service.start()

    def on_stop_clicked(self):
        self.service.stop()

    def refresh(self):
        status = self.service.get_status()
        self.update_status_view(status)
```

GUI 只调用 service，不直接控制 GNU Radio，不直接决定 estimator 策略。

---

## 5. 日志规范

### 日志等级

```text
DEBUG    高频细节，可关闭
INFO     主要流程
WARNING  可恢复异常、性能下降、自动修正
ERROR    操作失败
CRITICAL 必须停机
```

### 日志字段

```json
{
  "ts": "ISO-8601 UTC time",
  "run_id": "unique run id",
  "level": "INFO",
  "module": "channel_estimator",
  "event": "CSI_EST_DONE",
  "message": "CSI estimated from pilot",
  "frame_id": 42,
  "sample_index": 102400,
  "data": {
    "estimator": "diag_tf",
    "noise_var": 0.0021,
    "condition_proxy": 8.7
  }
}
```

### 日志输出边界

每个模块必须至少有：

```text
MODULE_INIT
CONFIG_APPLIED
START / STOP
核心成功事件
核心失败事件
PERF_SAMPLE
```

---

## 6. 状态机设计

```text
UNCONFIGURED
    ↓ configure
CONFIGURED
    ↓ start
STARTING
    ↓ flowgraph started
STREAMING
    ↓ stop
STOPPING
    ↓ stopped
STOPPED
```

异常路径：

```text
任意状态 -> ERROR -> STOPPING -> STOPPED / UNCONFIGURED
```

所有状态跳转必须写日志。

---

## 7. 潜在挑战和解决方案

### 挑战 1：B210 实时链路容易 overflow

解决方案：

- 默认 1 MHz 起步；
- RX 处理节流；
- GUI 绘图降采样；
- 日志异步写入；
- 大数组只做摘要；
- full-H_TF 不在运行中频繁重估。

### 挑战 2：真实 RF 链路与 TDL 模型不匹配

解决方案：

- RF 默认使用 `diag_tf`；
- `tdl_param` 只用于纯软件 TDL 或明确的 TDL 辅助验证；
- estimator 自动切换必须日志化；
- report 中记录实际生效 estimator。

### 挑战 3：alpha/beta 优化容易偏离工程主线

解决方案：

- 不做实时扫描推荐；
- 只保留手动和预设；
- 每次实验记录 alpha/beta 和结果；
- 后续基于真实实验日志做离线统计，而不是在线扫描。

### 挑战 4：GUI 和后端状态不一致

解决方案：

- 引入 `ExperimentService`；
- GUI 只读 service 状态；
- 所有参数应用经过状态机；
- 运行中结构性参数变更必须 stop -> rebuild -> start。

### 挑战 5：后续优化缺少证据链

解决方案：

- 每次实验自动生成 run 目录；
- 保存配置、日志、指标、图表；
- 所有异常和自动修正写入 events.jsonl；
- report.md 自动总结关键指标和失败原因。

---

## 8. 项目落地路线

### 第一阶段：工程地基

- 增加 config dataclass；
- 增加 EventLogger；
- 增加 run_id 和 run 目录；
- 增加状态机；
- 将 GUI 参数读取改为生成 `ExperimentConfig`。

### 第二阶段：拆分后端

- 拆出 `core/fdit.py`；
- 拆出 `core/framing.py`；
- 拆出 `receiver/synchronizer.py`；
- 拆出 `receiver/channel_estimator.py`；
- 拆出 `hardware/uhd_flowgraph.py`。

### 第三阶段：硬件稳定性

- 完成 RF safety checker；
- 完成 stream health monitor；
- 完成 RX processing profiler；
- 完成 overflow/stale/rate drop 日志；
- 固化 B210 推荐参数。

### 第四阶段：实验导出和报告

- 自动保存 config/status/metrics；
- 导出 figures；
- 生成 report.md；
- 建立硬件实验 checklist。

### 第五阶段：真实实验数据驱动优化

- 基于真实 RF run 日志分析 EVM/BER；
- 比较 OFDM、OTFS、FDIDM 预设；
- 分析 alpha/beta 在真实链路中的有效区间；
- 决定是否需要离线优化工具，而不是在线扫描推荐。

---

## 9. 结论

专业落地版本的 FDIDM 平台应当是：

```text
清晰分层 + 真实硬件主线 + 结构化日志 + 状态机 + 实验快照 + 可复现报告
```

当前最重要的不是继续增加理论功能，而是让每一次硬件运行都能被准确记录、准确复现、准确诊断。只有先把工程地基做好，后续的 FDIDM 指数选择、估计器优化和硬件性能对比才有可信依据。

