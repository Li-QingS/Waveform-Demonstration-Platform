# FDIDM 接收机相关 GMI 自适应 Plan

## 架构概览

本次改造采用“纯数学内核—有限库—在线策略—硬件验证—页面展示”五层结构。GMI 数学和库构造不依赖 Qt、GNU Radio 或 UHD；硬件后端只负责把可信 CSI、噪声和同步状态转换为选择器输入，并执行已经通过策略门限的 token 切换；EVM/SER/CRC 只在验证层使用。

```text
实测/仿真 CSI + 噪声 + 公共资源
              │
              ▼
    GMIExperimentBuilder
       Bη, Cη=ΠK Bη, Ση
              │
              ▼
      GMIScorer（nats/charged-sample）
              │
        ┌─────┴─────────┐
        ▼               ▼
  离线 LibraryBuilder   在线 RobustSelector
  曲率/覆盖/p-center    不确定性/实现误差/切换成本
        │               │
        ▼               ▼
  Versioned Library   token 切换请求
                          │
                          ▼
              稳定门控 + ABBA 真链路验证
                          │
                    提交或安全回滚
```

首期只把对角接收机 `K=0` 作为在线自适应主路径，把完整匹配接收机作为理论对照。循环带状接收机的数据结构和投影接口保留，但不进入首期自动切换路径。

## 核心数据结构

### `ExperimentResources`

不可变公共资源描述，字段包括：

- `signal_dimension`：一个有效数据块的复符号维数；
- `guard_samples`、`training_samples`、`control_samples`；
- `catalogue_cap`、`control_bits`、`control_bits_per_sample`；
- `sample_rate_hz`、`bandwidth_hz`；
- `target_rms`、`peak_limit`；
- `pilot_rule`、`receiver_rule`、`resource_id`。

`charged_samples` 由有效维数、保护、训练和控制样本共同计算。`resource_id` 是规范化字段的稳定摘要；不同 `resource_id` 的 token 不得进入同一可执行库或同一优势比较。

### `ReceiverSpec`

字段包括：

- `mode`：`diagonal`、`cyclic_band` 或 `full_matched`；
- `band_k`：循环带宽，首期对角模式固定为 0；
- `metric_covariance_mode`：首期为接收噪声协方差；
- `exponent_interval`：对角白噪声模式为 `[0, 1]`；
- `receiver_id`：稳定摘要。

信道估计器不属于 `ReceiverSpec`。`diag_tf/full_htf/tdl_param` 描述如何得到 CSI；`ReceiverSpec` 描述得到 CSI 后解码度量保留哪些矩阵项。

### `GMIExperiment`

字段包括：

- `B`：token 对应的真实有效信道；
- `C`：接收机度量使用的投影信道；
- `Sigma`：分析域噪声协方差；
- `symbol_power`；
- `charged_samples`；
- `receiver` 与 `resource_id`。

构造规则：

- 时间采样域信道使用论文的 `F(η)=S0V(α,β)` 与 `G(η)=F(η)^H`；
- 已在 TF 基下估计的 `H_TF` 使用 `Bη=V(η)^H H_TF V(η)`，避免重复应用公共 `S0`；
- `Cη=ΠK(Bη)`；完整匹配模式直接令 `Cη=Bη`。

### `GMIResult`

字段包括：

- `valid`、`reason`；
- `gmi_nats_per_sample`；
- `gmi_bits_per_sample`，仅由 nats 显式换算；
- `s_star`；
- `receiver_mode`、`token_id`、`resource_id`；
- 数值诊断：协方差最小特征值、条件数和求解路径。

无效实验返回 `valid=False` 和明确原因，不用 NaN/Inf 参与排序。

### `WaveformToken`

字段包括：

- `token_id`、规范化 `alpha`、`beta`；
- `receiver_id`、`resource_id`；
- `channel_estimator`、`equalizer`；
- `transform_version`、`implementation_mode`；
- `implementation_penalty_nats`；
- 功率、训练、保护、控制与数值实现摘要。

`token_id` 由规范化内容生成，不依赖列表位置。只包含 α/β 的对象不能通过可执行校验。

### `TransceiverLibrary`

字段包括：

- `schema_version=1`；
- `library_id`、`created_from`、`catalogue_cap`；
- 唯一的 `resources` 和 `receiver`；
- 有序 `tokens`；
- 构造信息：状态集摘要、候选集摘要、目标、最坏 regret、随机种子；
- 数字签名式内容摘要，用于检测文件内容变化。

读取时验证格式版本、容量、周期重复、token 上下文一致性和摘要；任何失败均拒绝执行。

### `ChannelState`

选择器可见的状态，字段包括：

- `h_tf` 或 `h_time`，二者必须明确标注基底；
- `noise_covariance` 或可展开为协方差的白噪声方差；
- `sync_metric`、有效 CFO、pilot NMSE、pilot power；
- `continuity`、`overflow`；
- `state_uncertainty_nats`；
- `context_id`、采集 epoch 和时间戳。

该结构不包含 payload 参考符号、解码字节、SER、BER、CRC 或数据辅助 EVM。

### `TokenScore` 与 `SelectionDecision`

`TokenScore` 包含估计 GMI、状态不确定性、实现误差、切换成本和上下界。候选相对当前 token 的保守净改善定义为：

```text
candidate_lower = J_candidate - uncertainty_candidate - implementation_penalty_candidate
current_upper   = J_current   + uncertainty_current   + implementation_penalty_current
gain_lower      = candidate_lower - current_upper - switch_cost
```

`SelectionDecision` 包含当前/候选 token、`gain_lower`、门限、`switch_allowed`、稳定原因、库/状态序号和完整评分列表。完整匹配模式始终返回对照结论，不发出 GMI 切换请求。

### `AcquisitionFingerprint`

切换后稳定性指纹，字段包括：

- 有效 CFO 与来源；
- 同步分数；
- residual CFO；
- pilot NMSE、pilot fitted power；
- RX 输入 RMS/峰值；
- probe continuity、overflow 和采集 epoch。

稳定判定使用连续窗口的有限值、绝对/相对漂移阈值和最少连续帧数。原始 CFO 别名不直接用于稳定判断，使用最终有效 CFO。

### `ABBAValidation`

由 `A1 → B1 → B2 → A2` 四个等帧块组成；A1+A2 与 B1+B2 的总帧数相等。字段包括：

- 四个证据窗口与四组指纹统计；
- 公共功率合同、资源与上下文 ID；
- 选择阶段的 GMI 决策快照；
- GMI 确认结果；
- EVM/SER/CRC 非回归保护结果；
- `improved/regressed/inconclusive/rollback_failed` 终态与原因。

只有 GMI 改善仍成立、A/B 两侧资源可比、A1/A2 与 B1/B2 内部稳定且硬件指标未显著回退时才提交 B。线性时间漂移通过对称块抵消；明显非线性或跨块漂移直接判为不确定。

## 模块设计

### `waveform_sim/gmi/model.py`

**职责：** 定义公共资源、接收机、实验、token、库、状态、评分和决策的不可变数据模型；提供规范化、摘要和严格校验。

**对外接口：**

- `ExperimentResources.charged_samples`
- `ReceiverSpec.validate(dimension)`
- `WaveformToken.create(...)`
- `TransceiverLibrary.validate()`
- `TransceiverLibrary.to_dict()/from_dict()`

**依赖：** 标准库、NumPy；不依赖硬件或 UI。

### `waveform_sim/gmi/scoring.py`

**职责：** 构造论文实验并计算 GMI。

**对外接口：**

- `canonical_fractional_index(value, period=2.0)`
- `fdidm_operator(m, n, alpha, beta, basis)`
- `cyclic_band_projection(matrix, band_k)`
- `build_experiment(channel_state, token, resources, receiver)`
- `score_gmi(experiment)`
- `score_library(channel_state, library)`

**算法：**

- 对角白噪声模式使用论文式 (13)/(14)，对导数做固定次数二分，`s∈[0,1]`；
- 完整匹配模式使用 log-det 闭式结果并返回 `s=1`；
- 矩阵求解使用 Cholesky/线性求解与 `slogdet`，不显式求逆；
- 内部统一使用 nats/charged-sample，页面层才换算 bits；
- 通用循环带模式只提供投影和离线实验构造，不作为首期硬件在线自动选择入口。

**依赖：** `model.py`、`waveform_sim/core/transforms.py`、NumPy。

### `waveform_sim/gmi/geometry.py`

**职责：** 实现接收机相关曲率、泄漏代理、方向分配和周期网格。

**对外接口：**

- `receiver_curvature(spectrum, generators, snr, receiver, charged_samples)`
- `leakage_curvature(...)`
- `finite_difference_curvature(score_field, optimum, step)`
- `allocate_axis_counts(curvature, capacity)`
- `anisotropic_periodic_grid(n_alpha, n_beta, period=2.0)`

**算法：**

- 论文 orbit 例子使用式 (18) 的谱权重 Gram 矩阵；
- 泄漏代理实现式 (28)–(31)，只用于筛选和边界诊断；
- 实际非理想信道允许使用离散 GMI 数值曲率，负小特征值在记录误差后投影到半正定锥；
- 首期 FDIT 方向分配按整数搜索最小化 `Q11/n_alpha² + Q22/n_beta²`，并满足 `n_alpha*n_beta<=capacity`。

**依赖：** `model.py`、`scoring.py`、NumPy。

### `waveform_sim/gmi/library.py`

**职责：** 从状态集和候选 token 构建、评估、序列化有限库。

**对外接口：**

- `evaluate_regret_matrix(states, candidates, resources, receiver)`
- `select_fixed_size_library(regret_matrix, capacity, seed)`
- `select_cover_library(regret_matrix, epsilon, capacity)`
- `build_transceiver_library(...)`
- `save_library(path, library)` / `load_library(path)`

**算法：**

- 固定规模路径使用确定性 greedy p-center 初始化，再做确定性单点交换改进；
- 覆盖路径按阈值形成集合覆盖，采用“覆盖最多未覆盖状态、再按最坏 regret 和 token_id 破平局”的确定性贪心；
- 任何构造都输出最坏 regret、未覆盖状态和容量饱和信息；失败不伪装为可行库。

**依赖：** `model.py`、`scoring.py`、`geometry.py`。

### `waveform_sim/gmi/selector.py`

**职责：** 对已加载库执行无 payload 的在线鲁棒选择。

**对外接口：**

- `RobustnessProfile`：`fast/standard/conservative` 三档固定策略；
- `score_tokens(state, library, current_token, profile)`
- `select_token(state, library, current_token, profile)`

**规则：**

- 状态无效、上下文不一致、完整匹配对照、当前验证未结束时不切换；
- 评分结果按保守下界、原始 GMI、切换距离、token_id 稳定排序；
- 只产生决策，不直接操作硬件；
- 不接收任何 payload 或解码质量参数。

**依赖：** `model.py`、`scoring.py`。

### `waveform_sim/hardware/evidence.py`

**职责变化：** 保留现有功率和已知符号证据；新增采集指纹、ABBA 窗口组合和 GMI/硬件双层验证纯函数。

**新增接口：**

- `fingerprints_stable(samples, profile)`
- `combine_abba_windows(a1, b1, b2, a2)`
- `classify_abba_validation(...)`

该模块仍无 Qt/GNU Radio/UHD 依赖。

### `waveform_sim/hardware/fdidm_adaptive.py`

**职责变化：** 从“连续 SER 优化器 + 顺序 A/B”重构为“库加载/生成 + GMI 选择工作线程 + ABBA 状态机”。原 SER 计算保留为诊断函数，但不再驱动 GMI 模式切换。

**状态机：**

```text
tracking
  → candidate_ready
  → A1_applying → A1_stabilizing → A1_collecting
  → B_applying  → B_stabilizing  → B1_collecting → B2_collecting
  → A2_applying → A2_stabilizing → A2_collecting
  → deciding
      ├─ improved → commit_applying → tracking
      ├─ regressed/inconclusive → rollback_applying → cooldown
      └─ rollback_failed → fail-closed
```

同一时刻只允许一个选择/验证事务。每次异步切换携带 generation 和 token_id，过期任务不得覆盖新配置。

### `waveform_sim/hardware/fdidm_hardtest.py`

**职责变化：**

- 初始化公共资源、接收机模式和 GMI 库；
- 将可信的 H_TF/噪声/同步信息发布为 `ChannelState`；
- 在非 RX 线程运行库评分；
- 将选择、token、GMI 和 ABBA 状态加入 status/debug/export；
- 使用现有安全功率合同和连续 live-swap 执行 token；
- 关闭 GMI 模式时保持固定波形链路不变。

首期默认库可由当前上下文确定性生成；显式加载的版本化库通过严格校验后覆盖默认库。

### `waveform_sim/ui/fdidm_hardware_test_tab.py`

**职责变化：**

- 默认操作区只保留：启用、接收机模式、库预算、鲁棒等级、立即评估；
- 原粗/细步长、稳定次数、最小 SER 增益和冷却帧移入兼容诊断区或移除无效连接；
- 状态行显示 token、接收机、净 GMI 改善下界和验证阶段；
- 四格观测区保持现有均衡布局，完整长原因进入 tooltip/导出。

### `waveform_sim/ui/hardware_advantage_observer.py`

**职责变化：** 导出时将 `theory`、`selection`、`hardware_validation` 三类字段分组；固定波形、自适应库和完整接收机使用同一 `resource_id` 才允许生成“可比较”结论。

### `scripts/reproduce_gmi_paper_example.py`

**职责：** 无硬件地复现论文第 VI 节核心数值：两组 SNR 的曲率方向、4×9 分配、等数量网格与曲率分配的 sampled regret。输出机器可读 JSON 和简洁控制台摘要，作为论文对齐证据。

## 模块交互

### 离线库构造

1. 状态生成器或历史硬件记录产生不含 payload 真值的 `ChannelState` 集合。
2. `geometry` 根据解析或数值曲率生成非均匀候选网格。
3. `library` 调用 `scoring` 形成状态×候选 GMI 与 regret 矩阵。
4. 固定规模或覆盖算法选择 token。
5. `TransceiverLibrary.validate()` 通过后保存版本化 JSON。

### 在线选择

1. 硬件解调链得到可信 CSI 和同步诊断。
2. 后端构造 `ChannelState`，放入单槽最新状态队列；旧状态可被新状态替换，不累积阻塞 RX。
3. GMI 工作线程对有限库评分并产生 `SelectionDecision`。
4. 若保守改善不足，继续 tracking；若允许切换，启动单一 ABBA 事务。
5. 页面只消费状态快照，不负责调用后端提交候选。

### ABBA 验证

1. 以共同功率合同依次应用 A/B/A token。
2. 每次实际换波形后等待指纹连续稳定；稳定前帧只计入诊断，不计入证据。
3. 采集 A1、B1、B2、A2 等帧块。
4. `evidence` 对称合并窗口，先检查资源/连续性/漂移，再检查 GMI，最后执行 EVM/SER/CRC 非回归保护。
5. 提交 B 时释放比较功率合同并恢复 token 的正常功率；失败则回滚 A，回滚失败保持 fail-closed。

## 文件组织

```text
waveform_sim/
├── gmi/
│   ├── __init__.py       — 稳定公开接口
│   ├── model.py          — 资源、接收机、token、库、状态与决策模型
│   ├── scoring.py        — 实验构造与 GMI 计算
│   ├── geometry.py       — 曲率、方向分配与周期候选网格
│   ├── library.py        — regret 矩阵、p-center、set-cover、JSON
│   └── selector.py       — 在线鲁棒有限库选择
├── hardware/
│   ├── evidence.py       — 稳定指纹和 ABBA 纯证据逻辑
│   ├── fdidm_adaptive.py — GMI 工作线程与 ABBA 状态机
│   └── fdidm_hardtest.py — CSI/status/live-swap 集成
└── ui/
    ├── fdidm_hardware_test_tab.py
    └── hardware_advantage_observer.py

scripts/
└── reproduce_gmi_paper_example.py

tests/
├── test_gmi_scoring.py
├── test_gmi_geometry.py
├── test_gmi_library.py
├── test_gmi_selector.py
├── test_gmi_abba_evidence.py
├── test_gmi_hardware_integration.py
└── 现有 FDIDM/硬件/UI 测试（回归）
```

## Spec 覆盖关系

| Spec | 设计归属 |
|---|---|
| F1–F3 | `model.py`、`scoring.py` |
| F4 | `geometry.py` |
| F5–F6 | `library.py`、版本化模型 |
| F7–F8 | `selector.py`、`ChannelState` 边界 |
| F9–F11 | `evidence.py`、ABBA 状态机 |
| F12–F13 | 硬件页面与状态 API |
| F14 | 公共 `resource_id`、完整接收机对照 |
| F15 | `fdidm_adaptive.py` 中的兼容诊断路径 |

## 技术决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 内部 GMI 单位 | nats/charged-sample | 与论文公式一致；UI 显式换算 bits，避免隐式混用 |
| 首期接收机 | 对角主路径 + 完整匹配对照 | 对应最清晰的“需要自适应/不需要自适应”实验；控制首期矩阵复杂度 |
| 默认库容量 | 16 | 4 bit 控制字段，在线评分开销可控，足以显示二维有限库效果 |
| token 域 | `[0,2)×[0,2)` | 利用论文对角 GMI 周期等价，消除 0/2 重复 |
| 指数优化 | 对角模式固定二分，完整模式闭式 | 复现论文方法，确定性强，不引入 SciPy |
| 库算法 | greedy p-center + 单点交换；greedy set-cover | 可重复、依赖少，并分别覆盖固定容量和固定误差两类目标 |
| 在线调度 | 单槽最新 CSI + 后台线程 | 不阻塞 RX，也避免排队处理过时信道状态 |
| 切换判据 | 候选下界减当前上界和切换成本 | 对状态/实现误差保守，收益不明确时保持当前 token |
| 真链路次序 | 等样本 ABBA | 抵消一阶时间漂移，并能检查两侧内部稳定性 |
| 页面控制 | 模式、容量、鲁棒等级 | 面向工程操作；算法步长不应由现场操作员调节 |
| 旧 SER 自适应 | 保留诊断、退出控制权 | 便于对照和回归，同时保证 GMI 模式只有一个决策源 |
| 版本与回溯 | schema + 内容摘要 + resource_id | 防止旧库静默运行在不同实验资源或算法版本上 |

## 兼容与迁移

- 新增 `adaptive_objective="gmi_library"` 作为 GMI 路径；旧配置缺少该字段时保持固定波形，不自动启用旧 SER 控制器。
- 现有 α/β 手动设置仍可用；手动设置会中止未完成的 GMI 验证事务并建立新的固定 token 上下文。
- 现有状态字段在一个过渡周期内保留；新增字段使用 `gmi_*` 与 `adaptive_token_*` 前缀，旧 `adaptive_predicted_ser_*` 明确标为 diagnostic。
- 现有日志与导出读取器遇到缺少 GMI 字段的历史记录时标记为 `legacy_ser`，不生成 GMI 优势结论。

## 性能与安全边界

- 默认 16×16、16 token 的对角 GMI 评分使用向量化对角公式和缓存变换，不构造每个 token 的重复大矩阵。
- 所有 GMI 工作线程均为可停止后台线程；`stop()` 必须等待/失效正在运行的 generation，禁止停止后重新应用波形。
- 完整匹配对照只计算一次信道 log-det，不为每个 token 重复矩阵求解。
- 任何协方差非正定、上下文不一致、非有限 CSI、overflow、probe gap 或资源摘要变化都会关闭本轮切换权限。
- 自动化测试使用纯数组和 mock，不创建 GNU Radio flowgraph；真机脚本始终在 `try/finally` 中停止和等待硬件。
