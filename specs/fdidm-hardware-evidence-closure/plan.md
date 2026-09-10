# FDIDM 硬件实验证据闭环 Plan

## 架构概览

本轮在现有硬件后端、硬件自适应 mixin、优势观测引擎和 FDIDM 硬件页之间增加一层纯 Python 的“证据计算”模块。接收均衡算法保持不变；新增模块只负责功率合同、已知符号测量、窗口统计和结论分类。

```text
确定性发送帧
  ├─ 未缩放波形 ──→ 功率分析 ──→ 基线/候选共同 RMS 合同
  └─ 已知编码前缀 ──→ data-aided EVM / 精确 SER k,n

RX 有效帧
  ├─ 现有同步、CFO、diag-TF 估计、MMSE 均衡（行为不改）
  ├─ 导频原始拟合 ──→ residual SINR / fit NMSE
  ├─ 精确累计计数 ──→ status / 日志 / 优势观测
  └─ 验证状态机
       共同功率基线 → 切候选 → 候选窗口
       → improved / inconclusive / regressed
       → 保留候选或回滚

Qt 页面
  ├─ 硬件启动自动开始观测
  ├─ 四宫格统一容器
  ├─ 实测/预测/注入设定分栏显示
  └─ JSON 日志保留全部证据
```

## 核心数据结构与接口

### `TxPowerMetrics`

纯数据对象，描述最终送入 UHD 的一个发送周期：

- `cycle_rms`：含同步、导频、数据和保护间隔的实际 RMS；
- `data_rms`：仅数据段 RMS；
- `peak`：周期最大绝对值；
- `papr_db`：按完整周期计算的峰均比；
- `target_rms`：用户/后端期望 RMS；
- `safe_rms`：在给定未缩放波形和峰值上限下可达到的最大无削峰 RMS；
- `backoff_db`：实际 RMS 相对目标 RMS 的回退；
- `peak_limited`：是否发生二次峰值缩放；
- `contract_id`：当前功率合同标识。

### `TxPowerContract`

一次基线/候选比较共享的功率约束：

- `baseline_alpha/beta`、`candidate_alpha/beta`；
- `requested_rms`；
- `locked_rms = min(requested_rms, baseline_safe_rms, candidate_safe_rms) × 0.98`；
- `peak_limit`；
- `tolerance_db=0.10`；
- `contract_id`；
- `comparable` 与失败原因。

后端提供以下内部接口：

```python
_preview_waveform_build(alpha: float, beta: float) -> WaveformBuildResult
_prepare_power_contract(alpha: float, beta: float) -> TxPowerContract
_commit_waveform_build(alpha: float, beta: float, forced_rms: float | None,
                       contract_id: str = "") -> WaveformBuildResult
```

预览构建无外部副作用，不修改 UHD、运行状态或观测计数。提交构建才替换发送向量。共同 RMS 取两个候选的安全 RMS 下界，因此基线和候选都不需要关闭峰值保护，也不会发生候选特有的额外功率回退。

### `KnownSymbolMetrics`

每个有效帧在均衡后，使用所有物理帧都相同的“已知编码应用前缀”计算：

- `reference_symbols`；
- `ser_errors`、`ser_symbols`、`ser`；
- `data_aided_evm_percent`；
- `decision_directed_evm_percent`；
- `residual_gain_abs`、`residual_phase_deg`。

使用共同的最小二乘复增益完成幅相归一化。data-aided EVM 和精确 SER 只覆盖无歧义的已知编码前缀，不把每个 super-cycle 中不同的随机填充位作为参考，从而避免“从多个填充帧中取最小 BER”的乐观偏差进入优势结论。

纯函数接口：

```python
measure_known_symbols(rx_symbols: ndarray, reference_symbols: ndarray,
                      ideal_constellation: ndarray) -> KnownSymbolMetrics
```

### `PilotFitMetrics`

从当前帧未经跨帧平滑的密集导频估计中计算：

- `fit_nmse`：对角、准静态模型对原始导频网格的归一化拟合残差；
- `residual_sinr_db`：拟合信号能量与残差能量之比；
- `residual_power`、`fitted_power`。

该结构不使用 `snr_db` 名称，因为残差同时包含噪声、残余 CFO、RF 失真和信道模型失配。现有 TDL `snr_db` 统一命名为 `tdl_injected_snr_db`。

### `EvidenceWindow`

候选验证使用的精确多帧窗口：

- `valid_frames`、`ser_errors`、`ser_symbols`；
- `data_aided_evm_values`；
- `crc_ok_frames`、`raw_ber_values`、`fec_ber_values`；
- `power_rms_values`、`contract_id`、`context_key`；
- `dropped_overflow`、`dropped_sync`、`dropped_context`、`dropped_power`；
- `wilson_low/high`、平均 EVM 及其帧间标准误。

### `ValidationDecision`

```python
class ValidationDecision:
    outcome: Literal["improved", "inconclusive", "regressed"]
    measured_ser_gain_db: float
    baseline_ser_interval: tuple[float, float]
    candidate_ser_interval: tuple[float, float]
    evm_delta_pp: float
    reason: str
```

分类规则：

1. 每侧至少 24 个有效帧，最多等待 64 帧；每侧 SER 错误数达到 100 才能用 SER 形成“改善”结论。
2. `improved`：候选 95% Wilson 上界低于基线下界，且实测 SER 改善不低于配置的最小改善门槛，data-aided EVM 没有越过退化保护线。
3. `regressed`：基线 95% Wilson 上界低于候选下界，或 data-aided EVM/CRC 越过明确退化保护线。
4. 其余情况、样本达到最大帧数仍不足、上下文/功率不可比均为 `inconclusive`。
5. `regressed` 和 `inconclusive` 都回滚；只有 `improved` 保留候选。

### `AdvantageObservationSession` 扩展

观测引擎优先消费后端累计计数的增量：

```python
on_sample(status: dict) -> None
on_toggle(enabled: bool, alpha: float, beta: float) -> ToggleEvent | None
```

新增上下文包括 `tx_power_contract_id`、`tx_cycle_rms`、`tx_power_backoff_db` 和验证三态结果。旧后端缺失累计计数时保留原有 `SER × 帧数` 的兼容估算，但导出中标明 `estimated_counts=true`。

## 模块设计

### 模块 A：硬件证据计算模块

**文件：** `waveform_sim/hardware/evidence.py`

**职责：** 定义上述纯数据对象；计算功率指标、已知符号指标、Wilson 区间、验证分类和帧结构效率。不得 import Qt、GNU Radio 或 UHD。

**依赖：** 标准库、NumPy。

**对应需求：** F4、F5、F7、F8、F9，支撑 N3。

### 模块 B：FDIDM 硬件发送与接收诊断

**文件：** `waveform_sim/hardware/fdidm_hardtest.py`

**职责：**

1. 将“生成未缩放周期”和“按指定 RMS 提交周期”分离，支持无副作用预览。
2. 在候选试验前建立共同功率合同；提交后记录实际周期/数据 RMS、峰值、PAPR、回退和合同 ID。
3. 接收有效帧后计算 `KnownSymbolMetrics`，保留原判决导向 EVM 兼容字段，同时新增 data-aided EVM 与精确 SER 帧计数、运行累计计数。
4. 在对角导频估计中以原始导频网格计算 `PilotFitMetrics`，再执行现有平滑和均衡；不改变均衡输入和结果。
5. 状态、调试日志和帧结构摘要输出新增字段：各段采样数、有效数据占比、训练/数据比、注入 SNR 与残差 SINR。

**对应需求：** F4、F5、F6、F9、F10。

### 模块 C：候选验证状态机

**文件：** `waveform_sim/hardware/fdidm_adaptive.py`

**职责：** 将现有“瞬时基线 → 直接切候选 → 5 帧不退化即接受”替换为：

```text
prepare_contract
  → baseline_applying（必要时以共同 RMS 重发当前参数）
  → baseline_settling（3 帧）
  → baseline_collecting（24～64 帧）
  → candidate_applying
  → candidate_settling（3 帧）
  → candidate_collecting（24～64 帧）
  → improved / inconclusive / regressed
  → keep / rollback
```

阶段切换需要更新发送向量时，仍由独立线程执行，接收锁内只记录样本和设置 pending 状态，避免重入与阻塞。所有样本必须匹配同一个 `contract_id` 和信道上下文；overflow/失步帧不进入窗口。

**对应需求：** F7、F8、F10，满足 N2、N5。

### 模块 D：优势观测引擎

**文件：** `waveform_sim/ui/hardware_advantage_observer.py`

**职责：** 消费精确累计错误计数、data-aided EVM 和功率上下文；窗口配对时增加功率可比性判断；导出三态验证结论和诊断字段。保持纯 Python、无 Qt 依赖。

**对应需求：** F1、F4、F5、F7、F8、F10。

### 模块 E：FDIDM 硬件页面

**文件：** `waveform_sim/ui/fdidm_hardware_test_tab.py`

**职责：**

1. 初始化所有热更新状态；自适应开关发起的更新明确标记为 live-safe，不清空曲线。
2. 后端成功进入运行态时自动开始新观测；停止测试时自动收尾。“开始观测”改为“重新开始观测”的显式手动重置入口。
3. 只有观测引擎返回真实切换事件时才显示“窗口已切换”，否则显示未记录原因。
4. 右下角统计组与三张时间图统一使用 `_PlotGridCell`、`Ignored/Expanding` 尺寸策略、相同边距与 1:1 网格权重。
5. 结论区只对 `improved` 使用绿色优势文案；`inconclusive` 为中性提示，`regressed` 为红色回滚提示。
6. 诊断区显示 data-aided/判决导向 EVM、精确 SER k/n、实际功率/PAPR/回退、残差 SINR/拟合 NMSE、导频/数据/总开销。
7. 软件 TDL 的 35 dB 等字段明确标注“注入设定”，不使用“实测 SNR”措辞。

**对应需求：** F1、F2、F3、F5、F6、F8、F9、F10。

### 模块 F：测试

**文件：**

- `tests/test_hardware_evidence.py`：纯函数功率、EVM、错误计数、Wilson 分类、帧效率；
- `tests/test_simple_hardware_adaptation.py`：共同功率合同和三态验证状态机；
- `tests/test_hardware_advantage_observer.py`：精确累计计数、功率不可比、导出；
- `tests/test_fdidm_hardware_comparison.py`：自动观测、热更新真实路径、四宫格几何和页面文案；
- `tests/test_hardware_power_normalization.py`：现有波形功率回归与 2026-09-07 场景复现。

所有新增测试默认离线运行。

## 模块交互

### 启动与自动观测

```text
用户启动测试
 → backend.start() 成功
 → 页面读取首个 status 上下文
 → observer.start(adaptive_enabled, context)
 → 时间轴立即开始积累
```

### 候选试验

```text
自适应 worker 发布预测候选
 → apply candidate
 → 预览基线和候选未缩放波形
 → 锁定共同 RMS/contract_id
 → 共同功率下重新采集基线
 → live swap 候选，丢弃 3 帧瞬态
 → 采集精确 k/n 与 data-aided EVM
 → 证据分类
      improved     → 保留候选，记录实测增益
      inconclusive → 回滚，记录证据不足
      regressed    → 回滚，记录实测退化
```

### 每帧证据流

```text
均衡符号 + 已知编码前缀
 → KnownSymbolMetrics
 → 本帧状态 + 运行累计 k/n
 → 验证状态机窗口
 → status
 → 优势观测窗口 / 页面 / JSON
```

## 文件组织

```text
waveform_sim/
├── hardware/
│   ├── evidence.py                    新建：纯证据计算与数据结构
│   ├── fdidm_hardtest.py              修改：波形构建、功率、EVM、导频诊断、状态
│   └── fdidm_adaptive.py              修改：共同功率、多帧三态验证
└── ui/
    ├── hardware_advantage_observer.py 修改：精确计数与功率可比性
    └── fdidm_hardware_test_tab.py     修改：自动观测、布局、呈现、热更新状态
tests/
├── test_hardware_evidence.py          新建
├── test_hardware_power_normalization.py 修改或扩展
├── test_simple_hardware_adaptation.py 修改
├── test_hardware_advantage_observer.py 修改
└── test_fdidm_hardware_comparison.py  修改
```

## 技术决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 实施顺序 | 先证据闭环，后接收机重构 | 当前尚不能区分噪声、模型失配、功率回退；直接改估计器无法可靠验收 |
| 功率公平 | 对每个基线/候选对预览波形，锁定共同安全 RMS | 不需要扫描全部 α/β 网格；同时消除候选特有的峰值回退 |
| 安全余量 | 共同安全 RMS 再乘 0.98 | 避免浮点、数据段选择和最终转换造成峰值边界抖动 |
| 功率容差 | 0.10 dB | 明显小于本次 0.67 dB 预测收益，也远小于已发现的 0.76 dB 回退 |
| SER 计数 | 使用所有帧共有的已知编码前缀精确 k/n | 无随机填充帧索引歧义，不依赖 `SER×帧数` 估算 |
| EVM | data-aided 为实验主指标，decision-directed 保留为在线兼容指标 | 已知发送序列下应使用真实参考；两者分开可定位判决偏差 |
| 残差质量命名 | residual SINR / fit NMSE | 残差混有噪声与模型误差，不能冒充端到端 SNR |
| 统计判定 | 95% Wilson 区间 + 最小实测增益 + EVM/CRC退化保护 | 单个瞬时帧不能代表基线；明确区分改善与未发现退化 |
| 无结论处理 | 回滚并保留原因 | 演示平台不能用证据不足的候选替代基线并宣称优势 |
| 自动观测 | 测试启动即开始，手动按钮仅重置 | 消除当前必须额外点击但没有强提示的交互陷阱 |
| 四宫格 | 四个单元使用相同包装和 size policy | 仅设置 row/column stretch 不能抵消不同 widget 的 sizeHint |
| 接收算法 | 本轮不改 diag-TF 平滑与 MMSE 行为 | 先获得可解释的真实基线，符合 spec 边界 |

## spec 覆盖对照

| Spec | Plan 落点 |
|---|---|
| F1 自动观测 | 模块 D/E、启动交互 |
| F2 真实切换与热更新健壮 | 模块 E、页面集成测试 |
| F3 等大四宫格 | 模块 E、三档几何测试 |
| F4 功率合同 | `TxPowerMetrics`、`TxPowerContract`、模块 B/C |
| F5 两类 EVM/精确 k,n | `KnownSymbolMetrics`、模块 B/D/E |
| F6 残差指标与命名 | `PilotFitMetrics`、模块 B/E |
| F7 多帧可比窗口 | `EvidenceWindow`、模块 C/D |
| F8 三态结论 | `ValidationDecision`、模块 C/E |
| F9 导频效率 | 证据模块帧结构统计、模块 B/E |
| F10 可复核导出 | 模块 B/D/E |

## 自检结果

- F1–F10 均有明确模块归属和测试入口。
- 新增依赖单向：UI → 观测引擎/后端，后端与自适应 → evidence；`evidence.py` 不反向依赖 UI、UHD 或 GNU Radio。
- 功率公平、验证结论和观测窗口共享同一 `contract_id`，避免三套口径漂移。
- 方案不增加导频、不改变均衡器、不修改软件仿真，符合本周期边界。
