# 01 代码审查模式：FDIDM 硬件验证平台专业审查

## 审查重点

- [安全性]
- [性能]
- [可读性]
- [最佳实践]
- [特定标准合规性]

## 项目上下文

项目基于论文《Fractional Dual Index Division Multiplexing: A Soft Waveform Design Toward Integrated Satellite-Terrestrial Networks》开发，目标不是做“纯软件仿真演示”，而是形成可落地、可调试、可复现实验结果的 FDIDM 硬件验证平台。平台当前由两类核心代码组成：

1. `fdidm_hardtest.py`：后端核心，负责 FDIDM 波形生成、帧结构、FDIT/IFDIT、Heisenberg/Wigner 变换、同步、CFO 估计、信道估计、均衡、GNU Radio/UHD 硬件链路、软件 TDL 辅助信道、状态采集和调试信息。
2. `fdidm_hardware_test_tab.py`：PyQt5 图形界面，负责参数配置、开始/停止测试、绘图、文本收发显示、日志窗口和后端调用。

本次审查的原则：

- 不建议再做 alpha/beta 的扫描推荐功能。论文中的指数优化可以作为理论背景或离线研究内容，但当前平台要面向真实 USRP/RF 链路落地，纯软件扫描对工程调试价值有限。
- 软件 TDL 只能作为单元测试、回归测试、接口验证或对照工况，不能作为平台主线。
- 代码重构必须形成清晰层次结构，每一个模块、每一个关键流程、每一个异常分支都应有对应日志输出，便于后期定位 UHD 溢出、同步失败、CSI 失配、星座散点和 EVM 异常。
- 平台要从“实验脚本”升级为“专业硬件实验系统”：可配置、可追踪、可复现、可导出、可长期维护。

---

# 1. 关键问题（按严重程度排序）

## S1：后端类过于庞大，职责高度耦合

### 现象

`FDIDMHardwareTest` 同时承担以下职责：

- FDIDM 数学变换：FSIT/FDIT/IFDIT、Gamma 矩阵缓存、Heisenberg/Wigner；
- 应用层封包：payload、CRC、bit 映射；
- 物理帧结构：guard、sync、pilot、data；
- 调制解调：QPSK/16QAM/64QAM、ZF/MMSE；
- 信道估计：diag-TF、full-H_TF、TDL-param；
- GNU Radio/UHD 硬件链路构建；
- 软件 TDL 信道；
- 运行线程、状态快照、EVM/BER 历史、绘图缓存；
- 调试日志和异常处理。

### 风险

- 单个 bug 可能跨越数学、硬件、线程和 GUI 多个层面，定位困难。
- 后期若加入真实信道测量、实验记录导出、设备校准、报告生成，会导致主类继续膨胀。
- 每次修改都可能影响硬件链路稳定性，尤其在 UHD 实时路径中容易产生隐藏性能回退。

### 严重程度

高。

---

## S2：日志系统还不够工程化，缺少统一事件模型

### 现象

当前代码已有 `_debug_log`、`_debug_seq`、GUI 日志窗口和若干状态字段，这是一个很好的基础。但是日志目前更接近“调试文本队列”，还没有形成统一的工程日志规范：

- 缺少固定事件 ID；
- 缺少模块名、运行 ID、帧号、硬件设备、参数快照等结构化字段；
- 部分关键函数没有入口/出口日志；
- 异常日志与状态日志没有清晰分级；
- GUI 日志、后端日志和可导出实验日志没有统一格式；
- 不能直接支持后续自动分析，例如按 run_id 回放一次失败实验。

### 风险

在真实 B210/N210/X310 链路上，问题往往不是单一代码错误，而是参数、硬件状态、驱动、线程调度、采样率、增益和同步共同作用。如果没有结构化日志，后期优化会变成“靠记忆和截图”。

### 严重程度

高。

---

## S3：GUI 与后端参数逻辑耦合较深，不利于专业化维护

### 现象

`fdidm_hardware_test_tab.py` 中不仅负责 UI 控件，还包含了一部分工程判断，例如：

- RF 模式下将 `tdl_param` 自动转为 `diag_tf`；
- RF+TDL 级联下将 `full_htf` 转为 `diag_tf`；
- 动态软件 TDL 下将 `full_htf` 转为 `tdl_param`；
- 将 GUI 控件值直接拼装为后端构造参数。

这些判断有工程合理性，但放在 UI 层会导致“同一套规则在命令行、自动化测试、GUI、脚本调用中不一致”。

### 风险

- 后续若新增 CLI 或实验批处理脚本，可能绕过 GUI 的保护逻辑。
- 后端和 GUI 都有 estimator guard 时，规则可能重复或不一致。
- 用户看到的界面参数与实际运行参数可能不同，若日志不完整，会造成误判。

### 严重程度

高。

---

## S4：软件 TDL 功能在工程主线中权重过高，容易偏离落地目标

### 现象

当前代码支持纯 TDL、RF、RF->TDL、TDL->RF 等模式。软件 TDL 对验证算法链路有帮助，但用户明确指出：完全走软件仿真没有意义。对于要落地的硬件平台，应当把真实 RF 路径作为主线。

### 建议定位

- `rf`：主线模式，用于实际 USRP 链路验证。
- `tdl_a_rf / tdl_c_rf / tdl_d_rf`：辅助硬件应力测试模式，用于把软件信道扰动加到 RF 前端之前，观察硬件链路对已知扰动的鲁棒性。
- `rf_tdl_a / rf_tdl_c / rf_tdl_d`：辅助后处理/级联诊断模式，不能作为最终性能结论。
- `tdl_a / tdl_c / tdl_d`：仅保留为单元测试、CI、无硬件开发、数学链路回归，不应作为产品主功能入口。

### 严重程度

中高。

---

## S5：状态快照字段多，但缺少分层状态机

### 现象

当前状态包括 `idle`、`running`、`decode_ok`、`last_bad_reason`、`rx_spectrum_stale`、`needs_top_block_rebuild` 等，但没有明确的硬件实验状态机。例如：

```text
UNCONFIGURED -> CONFIGURED -> STARTING -> STREAMING -> SYNCED -> DECODING -> STOPPING -> STOPPED -> ERROR
```

### 风险

- GUI 控件启停逻辑容易出现边界状态问题。
- 后端正在 stop/rebuild 时，如果 UI 触发参数应用，可能造成状态竞态。
- 错误恢复路径不清晰，后期维护困难。

### 严重程度

中高。

---

## S6：参数配置缺少不可变快照和版本化实验记录

### 现象

当前配置通过构造函数和 `configure()` 传入，GUI 会直接读取控件值并传给后端。缺少统一的 `ExperimentConfig`、`RunMetadata`、`HardwareConfig`、`WaveformConfig`、`ReceiverConfig`。

### 风险

- 运行中参数是否改变不容易追溯。
- 一次实验失败后，很难完整复现实验。
- 图、日志和结果无法绑定到同一个配置快照。

### 严重程度

中。

---

## S7：RF 安全边界需要显式化

### 现象

代码中已有较低默认 TX/RX gain 和 loopback safety 注释，但建议进一步工程化：

- 限制默认 TX gain；
- 启动前检查是否 RF 模式；
- 记录天线、中心频率、采样率、增益；
- 高功率或外接天线时弹出确认；
- 对非法频段、异常采样率、设备未连接等情况给出明确错误。

### 严重程度

中。

---

## S8：代码注释中存在较多版本演进信息，建议迁移到 CHANGELOG

### 现象

当前代码含有 `v17`、`v20`、`v28`、`v29` 等版本说明。这些说明对作者很有帮助，但正式工程代码中应避免过多历史注释堆积。

### 建议

- 代码注释只保留“为什么这么做”的机制说明；
- 版本变化放到 `CHANGELOG.md`；
- 重要工程决策放到 `docs/adr/`，例如 `ADR-001-rf-estimator-default.md`。

### 严重程度

中。

---

# 2. 改进建议（包括代码示例）

## 2.1 推荐目录结构

当前最重要的重构不是新增算法，而是拆分职责、统一日志、建立硬件实验主线。

```text
fdidm_platform/
  app/
    main.py
    gui/
      hardware_tab.py
      widgets.py
      plot_controller.py
      log_viewer.py
  core/
    fdit.py
    transforms.py
    modem.py
    framing.py
    metrics.py
  hardware/
    uhd_flowgraph.py
    device_manager.py
    rf_safety.py
    stream_monitor.py
  receiver/
    synchronizer.py
    cfo.py
    channel_estimator.py
    equalizer.py
    decoder.py
  service/
    experiment_service.py
    run_state.py
    config_models.py
    event_logger.py
  diagnostics/
    snapshot.py
    report_exporter.py
    health_check.py
  tests/
    test_fdit.py
    test_framing_crc.py
    test_config_validation.py
    test_logger_schema.py
  docs/
    adr/
    operation_manual.md
    hardware_bringup_checklist.md
```

分层原则：

- `core/` 只做数学和信号处理，不依赖 PyQt、GNU Radio、UHD。
- `hardware/` 只做设备、流图、RF 安全、样本流监控。
- `receiver/` 只做接收机链路。
- `service/` 负责配置、状态机、运行生命周期、日志汇聚。
- `app/gui/` 只负责界面，不做核心工程策略判断。
- `diagnostics/` 负责实验快照、报告导出和健康检查。

---

## 2.2 统一日志事件模型

建议采用 JSON Lines 作为底层日志格式，同时把重要日志同步显示到 GUI。

### 日志字段规范

```python
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import json
import logging
import uuid

@dataclass(frozen=True)
class FdidmEvent:
    ts: str
    run_id: str
    level: str
    module: str
    event: str
    message: str
    frame_id: Optional[int] = None
    sample_index: Optional[int] = None
    data: Optional[Dict[str, Any]] = None

class EventLogger:
    def __init__(self, run_id: Optional[str] = None, path: Optional[str] = None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.path = path
        self._logger = logging.getLogger("fdidm")
        self._logger.setLevel(logging.INFO)

    def emit(self, level: str, module: str, event: str, message: str, **data: Any) -> None:
        record = FdidmEvent(
            ts=datetime.now(timezone.utc).isoformat(),
            run_id=self.run_id,
            level=level.upper(),
            module=module,
            event=event,
            message=message,
            data=data or None,
        )
        line = json.dumps(asdict(record), ensure_ascii=False)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        getattr(self._logger, level.lower(), self._logger.info)(line)
```

---

## 2.3 每个代码模块必须有日志输出

建议所有模块统一遵守以下日志边界：

```text
模块初始化：MODULE_INIT
配置变更：CONFIG_APPLIED / CONFIG_REJECTED
启动/停止：START / STOP / STOP_DONE
关键输入：INPUT_SUMMARY
关键输出：OUTPUT_SUMMARY
异常：ERROR / RECOVERABLE_ERROR
性能：PERF_SAMPLE
硬件状态：HW_STATUS
帧处理：FRAME_SYNC / FRAME_DECODE / FRAME_FAIL
```

### 代码模块与日志事件映射

| 模块 | 必须输出的日志 | 用途 |
|---|---|---|
| `config_models.py` | `CONFIG_CREATED`, `CONFIG_VALIDATED`, `CONFIG_REJECTED` | 保证实验参数可追溯 |
| `fdit.py` | `FDIT_MATRIX_CACHE_HIT`, `FDIT_MATRIX_BUILT`, `FDIT_PARAM_ERROR` | 定位 Gamma 矩阵和 alpha/beta 参数问题 |
| `framing.py` | `APP_FRAME_BUILT`, `CRC_PASS`, `CRC_FAIL`, `PAYLOAD_TOO_LARGE` | 定位收发文本和 CRC 问题 |
| `modem.py` | `MODULATE_DONE`, `DEMODULATE_DONE`, `UNSUPPORTED_MODULATION` | 定位调制解调问题 |
| `uhd_flowgraph.py` | `FLOWGRAPH_BUILD_START`, `USRP_CONFIG`, `FLOWGRAPH_START`, `UHD_OVERFLOW`, `FLOWGRAPH_STOP` | 定位 USRP 和 GNU Radio 问题 |
| `stream_monitor.py` | `RX_SAMPLES_UPDATE`, `RX_STALE`, `RX_RATE_DROP`, `BUFFER_LEVEL` | 定位数据流问题 |
| `synchronizer.py` | `SYNC_CANDIDATE`, `SYNC_LOCKED`, `SYNC_LOST`, `SYNC_LOW_METRIC` | 定位同步失败 |
| `cfo.py` | `CFO_ESTIMATED`, `CFO_OUT_OF_RANGE`, `CFO_COMP_APPLIED` | 定位频偏问题 |
| `channel_estimator.py` | `CSI_EST_START`, `CSI_EST_DONE`, `CSI_COND_WARN`, `ESTIMATOR_AUTO_SWITCH` | 定位信道估计选择和条件数问题 |
| `equalizer.py` | `EQ_START`, `EQ_DONE`, `EQ_SINGULAR_WARN`, `NOISE_VAR_EST` | 定位 ZF/MMSE 问题 |
| `decoder.py` | `FRAME_DECODE_OK`, `FRAME_DECODE_FAIL`, `BER_EVM_UPDATE` | 定位最终解码质量 |
| `experiment_service.py` | `RUN_CREATED`, `RUN_STARTED`, `RUN_STOPPED`, `RUN_FAILED`, `SNAPSHOT_EXPORTED` | 串联一次完整实验 |
| `hardware_tab.py` | `UI_CONNECT_CLICKED`, `UI_START_CLICKED`, `UI_PARAM_CHANGED`, `UI_ERROR_DISPLAYED` | 定位 GUI 操作与后端状态不一致 |

---

## 2.4 配置对象应从字典参数升级为 dataclass

```python
from dataclasses import dataclass
from typing import Literal, Optional

ModOrder = Literal["QPSK", "16QAM", "64QAM"]
Equalizer = Literal["ZF", "MMSE"]
ChannelMode = Literal["rf", "tdl_a_rf", "tdl_c_rf", "tdl_d_rf", "rf_tdl_a", "rf_tdl_c", "rf_tdl_d"]
Estimator = Literal["diag_tf", "full_htf", "tdl_param"]

@dataclass(frozen=True)
class HardwareConfig:
    device_type: str = "USRP B210"
    carrier_freq: float = 2.4e9
    sample_rate: float = 1_000_000.0
    tx_gain: float = 10.0
    rx_gain: float = 20.0
    tx_antenna: str = "TX/RX"
    rx_antenna: str = "RX2"
    serial: Optional[str] = None

@dataclass(frozen=True)
class WaveformConfig:
    m: int = 16
    n: int = 8
    cp_len: int = 4
    alpha: float = 0.5
    beta: float = 1.0
    mod_order: ModOrder = "QPSK"

@dataclass(frozen=True)
class ReceiverConfig:
    equalizer: Equalizer = "MMSE"
    estimator: Estimator = "diag_tf"
    evm_average_frames: int = 8
    process_interval_ms: int = 250

@dataclass(frozen=True)
class ExperimentConfig:
    hardware: HardwareConfig
    waveform: WaveformConfig
    receiver: ReceiverConfig
    channel_mode: ChannelMode = "rf"
    tx_text: str = "FDIDM OK"
```

### 日志化配置校验

```python
def validate_config(cfg: ExperimentConfig, logger: EventLogger) -> ExperimentConfig:
    logger.emit("info", "config", "CONFIG_VALIDATE_START", "Validating experiment configuration")

    if cfg.channel_mode == "rf" and cfg.receiver.estimator == "tdl_param":
        logger.emit(
            "warning",
            "config",
            "ESTIMATOR_AUTO_SWITCH",
            "RF path cannot be represented by pure TDL parameter basis; switching to diag_tf",
            from_estimator="tdl_param",
            to_estimator="diag_tf",
            channel_mode=cfg.channel_mode,
        )
        cfg = replace(cfg, receiver=replace(cfg.receiver, estimator="diag_tf"))

    if cfg.hardware.tx_gain > 30:
        logger.emit(
            "warning",
            "rf_safety",
            "TX_GAIN_HIGH",
            "TX gain is high for loopback; confirm attenuation and cabling",
            tx_gain=cfg.hardware.tx_gain,
        )

    logger.emit("info", "config", "CONFIG_VALIDATE_DONE", "Configuration accepted")
    return cfg
```

---

## 2.5 不再新增扫描推荐功能，改为“手动/预设/实验记录”机制

原来的“推荐 alpha/beta”按钮可以保留为固定预设，但不应误导为实时优化器。

建议改名：

```text
OFDM 预设：alpha=0, beta=0
OTFS 预设：alpha=1, beta=1
FDIDM 实验预设：alpha=0.5, beta=1.0
自定义：用户手动输入 alpha/beta
```

并在日志中记录来源：

```python
logger.emit(
    "info",
    "waveform",
    "ALPHA_BETA_APPLIED",
    "FDIDM indices applied",
    alpha=cfg.waveform.alpha,
    beta=cfg.waveform.beta,
    source="manual"  # manual / preset_ofdm / preset_otfs / preset_fdidm
)
```

---

## 2.6 接收机处理链建议拆为 Pipeline

```python
@dataclass
class DecodeResult:
    ok: bool
    text: str
    ber: float
    evm_percent: float
    reason: str
    metrics: dict

class ReceiverPipeline:
    def __init__(self, synchronizer, cfo_estimator, channel_estimator, equalizer, decoder, logger):
        self.sync = synchronizer
        self.cfo = cfo_estimator
        self.channel = channel_estimator
        self.eq = equalizer
        self.decoder = decoder
        self.logger = logger

    def process(self, samples: np.ndarray) -> DecodeResult:
        self.logger.emit("debug", "receiver", "RX_PROCESS_START", "Processing RX window", samples=len(samples))
        sync = self.sync.find(samples)
        if not sync.locked:
            self.logger.emit("warning", "synchronizer", "SYNC_LOW_METRIC", "No reliable sync candidate", metric=sync.metric)
            return DecodeResult(False, "", float("nan"), float("nan"), "sync_failed", {})

        corrected = self.cfo.correct(samples, sync)
        h = self.channel.estimate(corrected, sync)
        symbols = self.eq.apply(corrected, h)
        result = self.decoder.decode(symbols)

        self.logger.emit(
            "info" if result.ok else "warning",
            "decoder",
            "FRAME_DECODE_OK" if result.ok else "FRAME_DECODE_FAIL",
            result.reason,
            ber=result.ber,
            evm_percent=result.evm_percent,
        )
        return result
```

---

# 3. 值得称赞的部分

## 3.1 已经意识到 B210/UHD 实时路径的性能风险

`_SampleRing` 使用有界 NumPy ring buffer，避免实时路径中频繁使用 `vector_sink_c.data()/reset()` 和 Python list/deque 转换，这一点非常重要。对于 Windows + B210 这类容易出现 U/O 的环境，这是正确方向。

## 3.2 帧结构较完整

当前已经包含：

```text
pre_guard -> sync_preamble -> pilot_frame -> data_frame -> post_guard
```

这比只做裸 QAM/OFDM 波形更接近真实链路调试需求。

## 3.3 有 CRC 和应用层帧格式

`APP_MAGIC + length + payload + crc32` 的设计虽然简单，但非常适合硬件闭环验证。它能把“星座看起来还行”和“实际 payload 正确恢复”区分开。

## 3.4 GUI 已具备实验调试所需的核心观测量

已有 TX 图、RX 频谱、EVM 曲线、星座图、发送/接收文本、日志窗口等模块。对于后续专业化，重点不是推翻 GUI，而是把后端状态、日志和图形刷新做得更规范。

## 3.5 已经引入 estimator guard 的工程判断

RF 路径下自动避免 `tdl_param`、RF+TDL 下避免噪声脆弱的 full-H_TF 逆，这体现出对真实硬件链路的理解。后续需要把这些判断从散落逻辑升级成统一配置校验和日志化决策。

---

# 4. 整体质量评估

## 当前阶段判断

当前代码已经超过“论文公式复现脚本”，属于“可运行硬件实验原型”。它有较完整的发射、接收、同步、信道估计、均衡、GUI 和诊断状态。但距离“可落地项目”仍需要一次工程化重构。

## 评分

| 维度 | 评分 | 说明 |
|---|---:|---|
| 理论对应性 | 8/10 | FDIT/IFDIT、FDIDM 帧结构和多种估计器已有体现 |
| 硬件意识 | 7/10 | 已考虑 B210、UHD、buffer、RF estimator guard |
| 性能稳定性 | 6/10 | 有优化意识，但 Python 后端仍偏重，缺少完整 profiling |
| 可维护性 | 5/10 | 单类过大，职责耦合明显 |
| 可观测性 | 6/10 | 有 debug 队列和状态字段，但缺结构化日志体系 |
| 专业落地程度 | 5.5/10 | 作为实验原型可用，作为长期项目需重构 |

## 总体结论

建议下一阶段不要继续扩展“扫描推荐”“纯软件仿真”等功能，而是优先完成以下工程化任务：

1. 拆分模块层次；
2. 建立统一日志事件模型；
3. 建立配置 dataclass 和实验 run_id；
4. 建立硬件主线状态机；
5. 明确 RF 模式为主线，软件 TDL 仅为辅助；
6. 对每个模块建立日志输出、单元测试和异常边界；
7. 支持一次实验的配置、日志、图表、指标统一导出。

一句话评价：当前平台的算法和实验思路有基础，但要成为专业落地项目，必须从“单文件实验后端 + GUI 控件驱动”升级为“分层硬件实验系统 + 结构化日志 + 可复现实验记录”。
