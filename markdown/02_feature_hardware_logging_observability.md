# 02 功能实现模板：硬件链路观测与统一日志系统

## 任务

实现 **FDIDM 硬件链路观测与统一日志系统**。

本功能不做 alpha/beta 扫描推荐，不把纯软件 TDL 仿真作为主线。核心目标是让平台在真实 USRP/RF 链路上做到：

- 每次实验可复现；
- 每个模块有日志；
- 每个异常可定位；
- 每个参数变更可追踪；
- 每次运行可导出完整实验记录；
- 后续可根据日志分析 UHD overflow、同步失败、CSI 失配、EVM 发散、星座散点等问题。

---

## 技术栈

- 编程语言：Python 3.9+
- GUI：PyQt5
- 绘图：pyqtgraph
- 数值计算：NumPy
- SDR/硬件：GNU Radio、UHD、USRP B210/N210/X310
- 日志格式：JSON Lines + GUI 实时日志窗口
- 配置建模：`dataclasses`
- 测试：pytest

---

## 输入

### 1. 实验配置输入

```python
ExperimentConfig(
    hardware=HardwareConfig(
        device_type="USRP B210",
        carrier_freq=2.4e9,
        sample_rate=1_000_000,
        tx_gain=10,
        rx_gain=20,
        tx_antenna="TX/RX",
        rx_antenna="RX2",
    ),
    waveform=WaveformConfig(
        m=16,
        n=8,
        cp_len=4,
        alpha=0.5,
        beta=1.0,
        mod_order="QPSK",
    ),
    receiver=ReceiverConfig(
        estimator="diag_tf",
        equalizer="MMSE",
        process_interval_ms=250,
        evm_average_frames=8,
    ),
    channel_mode="rf",
    tx_text="FDIDM OK",
)
```

### 2. 运行时输入

- GUI 按钮事件：连接、开始、停止、应用参数、重置 CSI；
- UHD/GNU Radio 事件：启动、停止、overflow、underflow、设备错误；
- 接收样本窗口：complex64 IQ samples；
- 接收机中间结果：sync metric、CFO、CSI、EVM、BER、CRC；
- 用户手动选择的 alpha/beta 预设或自定义值。

---

## 输出

### 1. GUI 实时显示

- 当前运行状态；
- TX/RX 频谱；
- 接收星座；
- EVM 曲线；
- 发送/接收文本；
- 关键日志摘要；
- 当前配置摘要。

### 2. 结构化日志文件

输出路径示例：

```text
runs/20260608_153000_fd734a9e/events.jsonl
```

单条日志示例：

```json
{
  "ts": "2026-06-08T15:30:00.125+00:00",
  "run_id": "fd734a9e",
  "level": "INFO",
  "module": "uhd_flowgraph",
  "event": "FLOWGRAPH_START",
  "message": "USRP flowgraph started",
  "frame_id": null,
  "sample_index": null,
  "data": {
    "device_type": "USRP B210",
    "sample_rate": 1000000,
    "carrier_freq": 2400000000,
    "tx_gain": 10,
    "rx_gain": 20,
    "channel_mode": "rf"
  }
}
```

### 3. 实验快照

```text
runs/20260608_153000_fd734a9e/
  config.json
  events.jsonl
  status_last.json
  metrics.csv
  figures/
    tx_spectrum.png
    rx_spectrum.png
    constellation.png
    evm_curve.png
  report.md
```

---

## 约束条件

### 性能要求

- 日志写入不能阻塞 UHD 实时流；
- 高频日志必须限流，例如 RX sample 更新不超过 10 Hz；
- 大数组不直接写入日志，只记录摘要：长度、均值功率、峰值、窗口起点、CRC、EVM、BER；
- 日志系统异常不能导致主链路崩溃。

### 内存限制

- IQ buffer 使用有界 ring buffer；
- 日志队列设置上限，超限时丢弃 DEBUG 级别日志并记录一次 `LOG_DROP_WARN`；
- 每次实验导出的图和指标由后台低优先级任务生成。

### 代码风格/规范

- 所有配置使用 dataclass；
- GUI 不直接做后端工程策略判断；
- 后端不得依赖 PyQt；
- 每个模块必须注入 `EventLogger`；
- 所有 `except Exception` 必须写入结构化日志，并保留异常类型和上下文；
- 不允许新增“静默失败”。

### 错误处理要求

错误分级：

```text
INFO       正常流程
DEBUG      调试细节，默认可关闭
WARNING    可恢复异常或性能退化
ERROR      当前操作失败，但程序仍可运行
CRITICAL   硬件链路或状态机失控，需要停止实验
```

---

## 额外上下文

论文中的 FDIDM 强调通过两个可调指数在多个域之间切换和融合，并能覆盖 OFDM、OTFS 等特殊情形。但工程平台不能把“遍历搜索最优指数”作为核心功能，因为真实硬件链路下 alpha/beta 的变化会受到 RF 前端、时钟偏差、CFO、同步、增益、CSI 估计、线缆和环境影响。当前阶段更专业的做法是：

- 保留 OFDM、OTFS、FDIDM 实验预设；
- 支持用户手动设置 alpha/beta；
- 每次运行完整记录参数和结果；
- 后续基于真实实验日志做离线分析，而不是在平台里做实时扫描推荐。

---

# 设计方案

## 1. 新增模块

```text
service/event_logger.py
service/config_models.py
service/experiment_service.py
service/run_state.py
hardware/uhd_flowgraph.py
hardware/rf_safety.py
diagnostics/report_exporter.py
```

---

## 2. 配置模型

```python
from dataclasses import dataclass, asdict, replace
from typing import Literal, Optional

ModOrder = Literal["QPSK", "16QAM", "64QAM"]
Equalizer = Literal["ZF", "MMSE"]
Estimator = Literal["diag_tf", "full_htf", "tdl_param"]
ChannelMode = Literal[
    "rf",
    "tdl_a_rf", "tdl_c_rf", "tdl_d_rf",
    "rf_tdl_a", "rf_tdl_c", "rf_tdl_d",
]

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
    estimator: Estimator = "diag_tf"
    equalizer: Equalizer = "MMSE"
    process_interval_ms: int = 250
    evm_average_frames: int = 8

@dataclass(frozen=True)
class ExperimentConfig:
    hardware: HardwareConfig
    waveform: WaveformConfig
    receiver: ReceiverConfig
    channel_mode: ChannelMode = "rf"
    tx_text: str = "FDIDM OK"
```

---

## 3. 日志模块

```python
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue, Full
from typing import Any, Dict, Optional
import json
import threading
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
    def __init__(self, run_dir: Path, run_id: Optional[str] = None, queue_size: int = 5000):
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "events.jsonl"
        self._queue: Queue[str] = Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer, name="fdidm-log-writer", daemon=True)
        self._thread.start()

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
        try:
            self._queue.put_nowait(line)
        except Full:
            # 不阻塞实时链路。这里不递归写日志，只做轻量降级。
            pass

    def _writer(self) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    line = self._queue.get(timeout=0.2)
                except Exception:
                    continue
                f.write(line + "\n")
                f.flush()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
```

---

## 4. 实验服务层

```python
class ExperimentService:
    def __init__(self, backend_factory, logger: EventLogger):
        self.backend_factory = backend_factory
        self.logger = logger
        self.backend = None
        self.state = "UNCONFIGURED"
        self.config = None

    def configure(self, config: ExperimentConfig) -> None:
        self.logger.emit("info", "experiment", "CONFIGURE_START", "Configuring FDIDM experiment")
        config = validate_experiment_config(config, self.logger)
        self.backend = self.backend_factory(config=config, logger=self.logger)
        self.config = config
        self.state = "CONFIGURED"
        self.logger.emit("info", "experiment", "CONFIGURE_DONE", "Experiment configured")

    def start(self) -> None:
        if self.state != "CONFIGURED":
            self.logger.emit("error", "experiment", "START_REJECTED", "Experiment must be configured before start", state=self.state)
            raise RuntimeError(f"Invalid state for start: {self.state}")
        self.state = "STARTING"
        self.logger.emit("info", "experiment", "RUN_START", "Starting FDIDM hardware run")
        self.backend.start()
        self.state = "STREAMING"
        self.logger.emit("info", "experiment", "RUN_STREAMING", "FDIDM hardware run is streaming")

    def stop(self) -> None:
        if self.backend is None:
            return
        self.logger.emit("info", "experiment", "RUN_STOP", "Stopping FDIDM hardware run", state=self.state)
        self.state = "STOPPING"
        self.backend.stop()
        self.state = "STOPPED"
        self.logger.emit("info", "experiment", "RUN_STOPPED", "FDIDM hardware run stopped")
```

---

## 5. 每个代码模块的日志接入示例

### FDIT 模块

```python
class FditCore:
    def __init__(self, logger: EventLogger):
        self.logger = logger
        self._gamma_cache = {}
        self.logger.emit("info", "fdit", "MODULE_INIT", "FDIT core initialized")

    def gamma(self, order: int, eps: float):
        key = (order, round(float(eps), 12))
        if key in self._gamma_cache:
            self.logger.emit("debug", "fdit", "GAMMA_CACHE_HIT", "Gamma matrix cache hit", order=order, eps=eps)
            return self._gamma_cache[key]
        self.logger.emit("debug", "fdit", "GAMMA_BUILD_START", "Building Gamma matrix", order=order, eps=eps)
        # build matrix here
        gamma = build_gamma_matrix(order, eps)
        self._gamma_cache[key] = gamma
        self.logger.emit("debug", "fdit", "GAMMA_BUILD_DONE", "Gamma matrix built", order=order, eps=eps)
        return gamma
```

### UHD 流图模块

```python
class UhdFlowgraph:
    def __init__(self, config: HardwareConfig, logger: EventLogger):
        self.config = config
        self.logger = logger
        self.logger.emit("info", "uhd_flowgraph", "MODULE_INIT", "UHD flowgraph module initialized")

    def build(self):
        self.logger.emit(
            "info", "uhd_flowgraph", "FLOWGRAPH_BUILD_START",
            "Building UHD flowgraph",
            device_type=self.config.device_type,
            sample_rate=self.config.sample_rate,
            carrier_freq=self.config.carrier_freq,
            tx_gain=self.config.tx_gain,
            rx_gain=self.config.rx_gain,
        )
        # build GNU Radio top block here
        self.logger.emit("info", "uhd_flowgraph", "FLOWGRAPH_BUILD_DONE", "UHD flowgraph built")
```

### 同步模块

```python
class Synchronizer:
    def __init__(self, logger: EventLogger, threshold: float = 0.30):
        self.logger = logger
        self.threshold = threshold
        self.logger.emit("info", "synchronizer", "MODULE_INIT", "Synchronizer initialized", threshold=threshold)

    def find(self, rx):
        metric, index = compute_sync_metric(rx)
        if metric < self.threshold:
            self.logger.emit("warning", "synchronizer", "SYNC_LOW_METRIC", "Sync metric below threshold", metric=metric, threshold=self.threshold)
        else:
            self.logger.emit("debug", "synchronizer", "SYNC_LOCKED", "Sync candidate accepted", metric=metric, index=index)
        return SyncResult(metric=metric, index=index, locked=metric >= self.threshold)
```

---

## 6. 验收标准

### 功能验收

- 启动一次 RF 实验后，自动生成 run 目录；
- run 目录中包含 `config.json`、`events.jsonl`、`metrics.csv`；
- GUI 日志窗口能实时显示关键事件；
- 每个主要模块至少输出一次 `MODULE_INIT`；
- 每次开始、停止、参数应用、CSI 重置都能在日志中查到；
- 每次同步失败、CRC 失败、EVM 异常、UHD overflow 都有结构化日志；
- 日志中能查到实际生效的 alpha、beta、M、N、CP、采样率、中心频率、TX/RX gain、estimator、equalizer。

### 工程验收

- 纯软件 TDL 不作为默认启动模式；
- 默认模式改为 `rf` 或明确的 `tdl_a_rf` 硬件辅助模式；
- 不新增 alpha/beta 扫描推荐器；
- 所有自动 estimator 切换必须写入 `ESTIMATOR_AUTO_SWITCH`；
- 后端核心不依赖 PyQt；
- 单元测试覆盖 config validation、CRC、FDIT 矩阵缓存、logger schema。

---

## 7. 不做项

以下功能本阶段不做：

- 不做实时 alpha/beta 网格扫描；
- 不做完全软件仿真的性能结论展示；
- 不做复杂在线优化器；
- 不把软件 TDL 结果包装成硬件验证结果；
- 不在 GUI 层继续堆叠后端策略逻辑。

