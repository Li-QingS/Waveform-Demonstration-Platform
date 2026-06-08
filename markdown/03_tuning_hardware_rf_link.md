# 03 调优优化模板：FDIDM 硬件链路稳定性与可调试性优化

## 观察到的问题

当前平台面向 USRP B210/N210/X310 等硬件链路时，可能出现以下问题：

1. B210/UHD 出现 overflow 或 underflow；
2. RX 频谱有显示，但解码失败；
3. 同步 metric 偶发下降，帧起点漂移；
4. 星座图短时间稳定后突然发散；
5. EVM 曲线波动大，与 BER/CRC 不完全一致；
6. full-H_TF 或 tdl_param 在真实 RF 链路中效果不稳定；
7. GUI 显示的参数、后端实际生效参数、日志记录参数可能不完全一致；
8. 停止/重新应用参数后，缓存 CSI 或状态残留导致下一次实验结果异常。

本模板强调：调优必须围绕真实硬件链路展开。纯软件 TDL 只能用于单元测试或数学链路回归，不能作为系统调优的主依据。

---

## 环境信息

建议每次调试必须记录以下信息：

```text
操作系统：Windows / Ubuntu，具体版本
Python 版本：3.x
GNU Radio 版本：x.x
UHD 版本：x.x
USRP 型号：B210 / N210 / X310
连接方式：USB3 / 千兆网 / 10G 网
中心频率：例如 2.4 GHz
采样率：例如 1 MHz
TX/RX 增益：例如 TX=10 dB, RX=20 dB
链路方式：线缆直连 / 衰减器 / 天线 OTA
外部参考：internal / 10 MHz reference / PPS
FDIDM 参数：M, N, CP, alpha, beta, mod_order
接收机参数：estimator, equalizer, process_interval_ms
运行模式：rf / tdl_a_rf / rf_tdl_a
```

---

## 已尝试的解决方案

可记录：

```text
1. 降低采样率，例如从 2 MHz 降到 1 MHz。
2. 增大 process_interval_ms，减少 Python 解码压力。
3. 降低 TX gain，避免接收端饱和。
4. 切换 estimator：RF 模式优先使用 diag_tf。
5. 重置 CSI 缓存。
6. 增加 inter-frame guard。
7. 固定 alpha/beta 使用 OFDM、OTFS、FDIDM 预设对照。
8. 检查 USB3/网口带宽和 UHD 驱动。
```

---

# 1. 问题根本原因分析

## 1.1 UHD overflow 的核心原因

UHD overflow 通常不是单一算法错误，而是实时链路吞吐不足：

```text
USRP RX stream -> GNU Radio buffer -> Python probe/ring buffer -> Python decoder -> GUI plot
```

任何一环阻塞都会造成样本堆积。当前代码已经使用 `_SampleRing` 降低 Python 数据搬运压力，但如果解码、绘图、状态快照和日志都在同一刷新节奏中执行，仍可能造成实时路径被拖慢。

### 重点排查

- Python 解码是否过于频繁；
- GUI 绘图是否处理了过多点；
- 状态快照是否复制了大数组；
- full-H_TF 是否在运行中重复估计；
- 日志是否同步写文件导致阻塞；
- USB3 或网口带宽是否不稳定。

---

## 1.2 星座突然发散的可能原因

星座图发散常见原因：

1. 同步点错误，导致 pilot/data 切片错位；
2. CFO 估计短时失败；
3. CSI 缓存来自坏帧；
4. RF 增益过高导致 ADC 饱和；
5. estimator 与真实链路不匹配；
6. 运行中参数变更后缓存未清；
7. RX 图仍显示旧数据，误认为当前链路稳定。

---

## 1.3 full-H_TF / tdl_param 在真实 RF 链路中不稳定的原因

在真实 RF 链路中，信道不仅包含理论 TDL 路径，还包含：

- USRP 模拟前端频响；
- 线缆/连接器/衰减器响应；
- 时钟误差；
- 残余 CFO；
- IQ 不平衡；
- 增益压缩；
- 环境多径。

因此，纯 TDL 参数基不可能完整表示真实 RF 响应。对于 CP-fitting 的短链路，`diag_tf` 更稳定、更容易落地。`full-H_TF` 可以保留为高级静态研究工具，但不建议作为默认硬件调试方案。

---

## 1.4 “纯软件仿真有效”不代表硬件有效

纯软件 TDL 没有真实硬件中的以下因素：

- UHD 调度和缓冲压力；
- DAC/ADC 动态范围；
- 采样时钟误差；
- 本振频偏；
- RF 前端滤波；
- 线缆、衰减器和天线耦合；
- 操作系统调度抖动。

因此，本项目不能以纯软件 TDL 结果作为平台性能结论。它只能回答“数学链路是否自洽”，不能回答“硬件系统是否可落地”。

---

# 2. 修复代码

以下代码是建议重构片段，不要求一次性替换全部后端。建议先以最小侵入方式引入日志和状态机。

## 2.1 增加运行状态机

```python
from enum import Enum

class RunState(str, Enum):
    UNCONFIGURED = "UNCONFIGURED"
    CONFIGURED = "CONFIGURED"
    STARTING = "STARTING"
    STREAMING = "STREAMING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"

class StateMachine:
    ALLOWED = {
        RunState.UNCONFIGURED: {RunState.CONFIGURED, RunState.ERROR},
        RunState.CONFIGURED: {RunState.STARTING, RunState.UNCONFIGURED, RunState.ERROR},
        RunState.STARTING: {RunState.STREAMING, RunState.ERROR},
        RunState.STREAMING: {RunState.STOPPING, RunState.ERROR},
        RunState.STOPPING: {RunState.STOPPED, RunState.ERROR},
        RunState.STOPPED: {RunState.CONFIGURED, RunState.UNCONFIGURED, RunState.ERROR},
        RunState.ERROR: {RunState.STOPPING, RunState.UNCONFIGURED},
    }

    def __init__(self, logger):
        self.state = RunState.UNCONFIGURED
        self.logger = logger

    def transition(self, target: RunState, reason: str = "") -> None:
        if target not in self.ALLOWED[self.state]:
            self.logger.emit(
                "error", "state", "STATE_TRANSITION_REJECTED",
                "Invalid state transition",
                source=self.state.value,
                target=target.value,
                reason=reason,
            )
            raise RuntimeError(f"Invalid state transition: {self.state} -> {target}")
        self.logger.emit(
            "info", "state", "STATE_TRANSITION",
            "Run state changed",
            source=self.state.value,
            target=target.value,
            reason=reason,
        )
        self.state = target
```

---

## 2.2 对 RX 处理增加耗时日志和节流

```python
import time

class RxProcessor:
    def __init__(self, pipeline, logger, max_process_ms: float = 50.0):
        self.pipeline = pipeline
        self.logger = logger
        self.max_process_ms = max_process_ms
        self.last_process_wall = 0.0

    def process_window(self, samples, abs_start: int):
        t0 = time.perf_counter()
        self.logger.emit(
            "debug", "rx_processor", "RX_PROCESS_START",
            "RX processing started",
            samples=len(samples),
            abs_start=abs_start,
        )
        try:
            result = self.pipeline.process(samples)
            return result
        except Exception as exc:
            self.logger.emit(
                "error", "rx_processor", "RX_PROCESS_ERROR",
                "RX processing failed",
                error_type=type(exc).__name__,
                error=str(exc),
                samples=len(samples),
                abs_start=abs_start,
            )
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            level = "warning" if elapsed_ms > self.max_process_ms else "debug"
            self.logger.emit(
                level, "rx_processor", "RX_PROCESS_COST",
                "RX processing cost measured",
                elapsed_ms=elapsed_ms,
                max_process_ms=self.max_process_ms,
            )
```

---

## 2.3 增加 RF 安全检查

```python
class RfSafetyChecker:
    def __init__(self, logger):
        self.logger = logger

    def validate(self, cfg):
        hw = cfg.hardware
        self.logger.emit(
            "info", "rf_safety", "RF_SAFETY_CHECK_START",
            "Checking RF safety constraints",
            carrier_freq=hw.carrier_freq,
            sample_rate=hw.sample_rate,
            tx_gain=hw.tx_gain,
            rx_gain=hw.rx_gain,
            channel_mode=cfg.channel_mode,
        )

        if cfg.channel_mode == "rf" and hw.tx_gain > 30:
            self.logger.emit(
                "warning", "rf_safety", "TX_GAIN_HIGH",
                "TX gain is high for RF mode; use attenuator or reduce gain",
                tx_gain=hw.tx_gain,
            )

        if hw.sample_rate > 5_000_000 and hw.device_type == "USRP B210":
            self.logger.emit(
                "warning", "rf_safety", "B210_SAMPLE_RATE_HIGH",
                "High sample rate may increase overflow risk on B210",
                sample_rate=hw.sample_rate,
            )

        self.logger.emit("info", "rf_safety", "RF_SAFETY_CHECK_DONE", "RF safety check finished")
```

---

## 2.4 对 estimator 自动切换进行强制日志化

```python
def resolve_estimator(cfg, logger):
    mode = cfg.channel_mode
    est = cfg.receiver.estimator
    rf_path = mode == "rf" or mode.startswith("rf_tdl_") or mode.endswith("_rf")
    rf_cascade = rf_path and ("tdl" in mode)

    if rf_path and est == "tdl_param":
        logger.emit(
            "warning", "channel_estimator", "ESTIMATOR_AUTO_SWITCH",
            "tdl_param cannot represent real RF path; switching to diag_tf",
            channel_mode=mode,
            from_estimator=est,
            to_estimator="diag_tf",
        )
        return "diag_tf"

    if rf_cascade and est == "full_htf":
        logger.emit(
            "warning", "channel_estimator", "ESTIMATOR_AUTO_SWITCH",
            "full-H_TF is noise-fragile on RF cascade; switching to diag_tf",
            channel_mode=mode,
            from_estimator=est,
            to_estimator="diag_tf",
        )
        return "diag_tf"

    logger.emit(
        "info", "channel_estimator", "ESTIMATOR_RESOLVED",
        "Channel estimator resolved",
        channel_mode=mode,
        estimator=est,
    )
    return est
```

---

## 2.5 增加 RX stale 和样本速率诊断

```python
class StreamHealthMonitor:
    def __init__(self, logger, stale_sec: float = 1.0):
        self.logger = logger
        self.stale_sec = stale_sec
        self.last_total = 0
        self.last_wall = time.perf_counter()

    def update(self, total_samples: int, expected_rate: float):
        now = time.perf_counter()
        dt = now - self.last_wall
        ds = total_samples - self.last_total
        if dt <= 0:
            return

        measured_rate = ds / dt
        self.logger.emit(
            "debug", "stream_monitor", "RX_RATE_SAMPLE",
            "RX sample rate measured",
            total_samples=total_samples,
            new_samples=ds,
            measured_rate=measured_rate,
            expected_rate=expected_rate,
        )

        if ds == 0 and dt > self.stale_sec:
            self.logger.emit(
                "warning", "stream_monitor", "RX_STALE",
                "RX stream has no new samples",
                stale_sec=dt,
                total_samples=total_samples,
            )

        if expected_rate > 0 and measured_rate < 0.5 * expected_rate:
            self.logger.emit(
                "warning", "stream_monitor", "RX_RATE_DROP",
                "Measured RX rate is far below expected rate",
                measured_rate=measured_rate,
                expected_rate=expected_rate,
            )

        self.last_total = total_samples
        self.last_wall = now
```

---

## 2.6 参数应用必须清缓存并记录原因

```python
def apply_config(self, new_config):
    old_config = self.config
    changed = diff_config(old_config, new_config)

    self.logger.emit(
        "info", "experiment", "CONFIG_APPLY_START",
        "Applying new experiment configuration",
        changed_fields=changed,
    )

    structural_keys = {"m", "n", "cp_len", "alpha", "beta", "sample_rate", "channel_mode", "estimator"}
    if any(k in structural_keys for k in changed):
        self.backend.reset_full_htf_cache()
        self.logger.emit(
            "info", "channel_estimator", "CSI_CACHE_CLEARED",
            "CSI cache cleared because structural configuration changed",
            changed_fields=changed,
        )

    self.config = new_config
    self.logger.emit("info", "experiment", "CONFIG_APPLY_DONE", "Configuration applied")
```

---

# 3. 防止类似问题的最佳实践建议

## 3.1 建立硬件调试优先级

推荐顺序：

```text
1. RF 直连低增益 QPSK OFDM 预设
2. RF 直连低增益 QPSK OTFS 预设
3. RF 直连低增益 FDIDM 实验预设
4. 增加 16QAM / 64QAM
5. 增加 TDL->RF 或 RF->TDL 辅助扰动
6. 再考虑复杂 estimator 或 full-H_TF 静态研究
```

不建议一开始就做：

```text
alpha/beta 网格扫描
纯软件 TDL 性能结论
高阶 QAM + 高采样率 + full-H_TF
动态 TDL + RF 级联 + 自动优化
```

---

## 3.2 每次实验必须导出完整记录

至少包括：

```text
config.json
事件日志 events.jsonl
状态快照 status_last.json
EVM/BER/Sync/CFO 指标 metrics.csv
关键图 figures/*.png
实验报告 report.md
```

---

## 3.3 建立 RF bring-up checklist

```text
[ ] 确认 USRP 被 UHD 识别
[ ] 确认 USB3/网口带宽正常
[ ] 确认 TX/RX 天线口或线缆连接正确
[ ] 确认外接衰减器和安全功率
[ ] 采样率从 1 MHz 起步
[ ] 调制从 QPSK 起步
[ ] estimator 从 diag_tf 起步
[ ] equalizer 从 MMSE 起步
[ ] 先跑 OFDM/OTFS 基线，再跑 FDIDM
[ ] 每次修改只改一个变量
[ ] 每次实验保存 run 目录
```

---

## 3.4 GUI 只展示，不做关键策略判断

GUI 可以提供控件和提示，但 estimator 选择、RF 安全、配置合法性、状态机切换必须在 service/backend 层完成。这样以后增加 CLI、自动化测试和批量实验时，不会绕过保护逻辑。

---

## 3.5 所有“自动修正”都必须可见

例如：

- 自动把 `tdl_param` 改为 `diag_tf`；
- 自动清 CSI 缓存；
- 自动限制 TX gain；
- 自动关闭不适合的按钮；
- 自动跳过过期 RX 图。

都必须写日志，并在 GUI 中提示。

---

## 3.6 结论

当前调优重点不是继续叠加算法功能，而是把真实硬件链路的运行状态变得可观测、可复现、可回放。只有日志、状态机、配置快照和硬件调试流程建立起来，后续讨论 EVM、BER、alpha/beta 或 estimator 才有工程意义。

