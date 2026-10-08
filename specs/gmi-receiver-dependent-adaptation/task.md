# FDIDM 接收机相关 GMI 自适应 Tasks

## 文件清单

| 操作 | 文件 | 职责 |
|---|---|---|
| 新建 | `waveform_sim/gmi/__init__.py` | 稳定公开接口 |
| 新建 | `waveform_sim/gmi/model.py` | 公共资源、接收机、实验、token、库、状态和决策模型 |
| 新建 | `waveform_sim/gmi/scoring.py` | GMI 实验构造、对角与完整接收机评分 |
| 新建 | `waveform_sim/gmi/geometry.py` | 曲率、方向分配和周期候选网格 |
| 新建 | `waveform_sim/gmi/library.py` | regret 矩阵、有限库选择与版本化序列化 |
| 新建 | `waveform_sim/gmi/selector.py` | 在线鲁棒有限库选择 |
| 修改 | `waveform_sim/hardware/evidence.py` | 采集指纹、ABBA 合并与双层验证 |
| 修改 | `waveform_sim/hardware/fdidm_adaptive.py` | GMI 工作线程、token 事务和 ABBA 状态机 |
| 修改 | `waveform_sim/hardware/fdidm_hardtest.py` | 配置、CSI 发布、状态、导出与安全切换集成 |
| 修改 | `waveform_sim/ui/fdidm_hardware_test_tab.py` | 精简 GMI 操作区和可读状态展示 |
| 修改 | `waveform_sim/ui/hardware_advantage_observer.py` | GMI/选择/硬件证据分组与公平性判定 |
| 新建 | `scripts/reproduce_gmi_paper_example.py` | 复现论文第 VI 节核心数值 |
| 新建 | `scripts/validate_gmi_hardware.py` | 有界、可终止的 GMI 真机短测与 JSON 证据 |
| 新建 | `tests/test_gmi_model.py` | 数据模型和格式校验 |
| 新建 | `tests/test_gmi_scoring.py` | GMI 公式、周期性和完整接收机对照 |
| 新建 | `tests/test_gmi_geometry.py` | 曲率、非均匀分配和基本域 |
| 新建 | `tests/test_gmi_library.py` | p-center、set-cover、确定性和序列化 |
| 新建 | `tests/test_gmi_selector.py` | 鲁棒门限、排序和无 payload 边界 |
| 新建 | `tests/test_gmi_abba_evidence.py` | 稳定指纹、漂移抵消和回滚判据 |
| 新建 | `tests/test_gmi_hardware_integration.py` | 后端线程、状态机、停止和状态 API |
| 修改 | `tests/test_hardtest_import.py` | 新配置默认值与固定模式回归 |
| 修改 | `tests/test_simple_hardware_adaptation.py` | token/ABBA 切换与回滚回归 |
| 修改 | `tests/test_fdidm_hardware_comparison.py` | GMI 页面、布局和导出回归 |
| 修改 | `tests/test_hardware_log_export.py` | 版本化 GMI 证据字段 |

## T1：建立 GMI 包和公共资源模型

**文件：** `waveform_sim/gmi/__init__.py`、`waveform_sim/gmi/model.py`

**依赖：** 无

**步骤：**

1. 建立 GMI 包，暂只导出稳定模型名称。
2. 定义不可变 `ExperimentResources`，校验库上限、控制位、样本数量、功率和采样参数。
3. 计算 `charged_samples`、规范化字典和稳定 `resource_id`。
4. 定义 `ReceiverSpec`，校验对角、循环带和完整匹配模式及 `band_k` 范围。

**验证：** 运行 `python -m pytest tests/test_gmi_model.py -q` 的资源/接收机用例，合法配置稳定生成相同 ID，非法配置明确报错。

## T2：建立实验、结果和在线状态模型

**文件：** `waveform_sim/gmi/model.py`、`tests/test_gmi_model.py`

**依赖：** T1

**步骤：**

1. 定义 `GMIExperiment` 和 `GMIResult`，矩阵维度、有限值、协方差和资源一致性可验证。
2. 定义不含 payload 真值的 `ChannelState`。
3. 定义 `TokenScore`、`SelectionDecision` 及显式 nats/bits 字段。
4. 添加无效结果的标准构造方式，禁止 NaN/Inf 进入可排序字段。

**验证：** `python -m pytest tests/test_gmi_model.py -q`，覆盖矩阵形状、协方差、非有限值和 ChannelState 禁止字段。

## T3：建立 token、有限库和版本化格式

**文件：** `waveform_sim/gmi/model.py`、`tests/test_gmi_model.py`

**依赖：** T1、T2

**步骤：**

1. 定义 `WaveformToken`，由规范化内容生成稳定 `token_id`。
2. 定义 `TransceiverLibrary`，强制唯一资源和接收机上下文。
3. 实现 schema 版本、内容摘要、规范化 JSON 字典和严格反序列化。
4. 校验容量、周期重复、未知版本、摘要变化和不完整 token。

**验证：** `python -m pytest tests/test_gmi_model.py -q`，同一库往返一致，篡改/重复/超容/未知版本均拒绝。

## T4：实现周期索引和 FDIT 实验算子

**文件：** `waveform_sim/gmi/scoring.py`、`tests/test_gmi_scoring.py`

**依赖：** T1–T3

**步骤：**

1. 实现 `[0,2)` 的 α/β 规范化，处理负值、端点和浮点容差。
2. 基于现有 `gamma_matrix` 构造 `V(α,β)`。
3. 区分 TF 基信道和时间采样基信道，分别构造有效 `Bη`。
4. 实现循环带投影 `ΠK` 和完整匹配投影。

**验证：** `python -m pytest tests/test_gmi_scoring.py -q`，单位信道、基底转换、投影带宽和 0/2 等价用例通过。

## T5：实现对角接收机 GMI

**文件：** `waveform_sim/gmi/scoring.py`、`tests/test_gmi_scoring.py`

**依赖：** T4

**步骤：**

1. 从 `B` 计算论文式 (13) 的 `x_i`、`y_i`。
2. 实现式 (14) 导数和固定 48 次二分的 `s∈[0,1]` 优化。
3. 用 `charged_samples` 归一化，内部返回 nats/charged-sample。
4. 对零信号、非白噪声误用、非有限矩阵和退化维度返回明确无效原因。

**验证：** `python -m pytest tests/test_gmi_scoring.py -q -k diagonal`，与直接公式样例一致，`s_star` 在闭区间内且结果有限。

## T6：实现完整匹配接收机和库批量评分

**文件：** `waveform_sim/gmi/scoring.py`、`tests/test_gmi_scoring.py`

**依赖：** T4、T5

**步骤：**

1. 使用 Cholesky/solve 与 `slogdet` 实现完整匹配 log-det GMI。
2. 禁止显式矩阵求逆；协方差非正定时返回无效结果。
3. 实现单状态对整库评分，缓存相同 M/N/token 的变换算子。
4. 完整模式只求一次信道 GMI并映射到全部 token。

**验证：** `python -m pytest tests/test_gmi_scoring.py -q`，完整接收机不同 α/β 得分在容差内相同，批量结果与逐项结果一致。

## T7：实现解析 GMI 曲率和泄漏代理

**文件：** `waveform_sim/gmi/geometry.py`、`tests/test_gmi_geometry.py`

**依赖：** T5

**步骤：**

1. 实现论文式 (18) 的 receiver-dependent 权重和 Gram 矩阵。
2. 实现循环带 discarded-pair 判定。
3. 实现式 (28)–(31) 的泄漏代理及 Loewner 边界诊断。
4. 对非简单谱、无丢弃方向和无效 SNR 输出明确状态。

**验证：** `python -m pytest tests/test_gmi_geometry.py -q -k curvature`，验证 `0.5U <= Q <= U`、完整接收机 `Q=0` 和 FDIT 交叉项。

## T8：实现数值曲率和方向分配

**文件：** `waveform_sim/gmi/geometry.py`、`tests/test_gmi_geometry.py`

**依赖：** T6、T7

**步骤：**

1. 实现周期域中心差分曲率，并记录离散误差。
2. 对数值小负特征值做可观测的半正定投影。
3. 通过整数搜索选择 `n_alpha*n_beta<=capacity` 的方向数量。
4. 生成不含周期重复端点的单元中心网格。

**验证：** `python -m pytest tests/test_gmi_geometry.py -q`，各向异性测试给敏感方向更多点，容量和周期唯一性满足约束。

## T9：复现论文 2×2 曲率与 4×9 分配

**文件：** `tests/test_gmi_geometry.py`

**依赖：** T7、T8

**步骤：**

1. 固定论文第 VI 节的谱、`r=0.04`、`rho=1/10` 和 `Nphys=18`。
2. 验证曲率两个对角元素的数量级、大小关系和 SNR 趋势。
3. 验证容量 36 时整数方向分配选择 4×9。
4. 验证完整接收机曲率为零。

**验证：** `python -m pytest tests/test_gmi_geometry.py -q -k paper_example`。

## T10：实现状态×候选 regret 矩阵

**文件：** `waveform_sim/gmi/library.py`、`tests/test_gmi_library.py`

**依赖：** T3、T6

**步骤：**

1. 批量计算每个状态、候选的 GMI。
2. 以状态内最佳 GMI 构造非负 regret。
3. 无效候选用显式不可用掩码表示，不使用 NaN 排序。
4. 输出状态/候选摘要与 oracle token 索引。

**验证：** `python -m pytest tests/test_gmi_library.py -q -k regret`，手算小矩阵与输出完全一致。

## T11：实现固定规模 p-center 选择

**文件：** `waveform_sim/gmi/library.py`、`tests/test_gmi_library.py`

**依赖：** T10

**步骤：**

1. 实现确定性 greedy 初始化。
2. 实现单 token 交换并只接受严格改善。
3. 固定 token_id 破平局，记录最坏状态和饱和原因。
4. 容量大于有效候选数时返回全部有效候选而不重复。

**验证：** `python -m pytest tests/test_gmi_library.py -q -k p_center`，多次运行返回相同 token 顺序和目标值。

## T12：实现 set-cover、建库和序列化

**文件：** `waveform_sim/gmi/library.py`、`tests/test_gmi_library.py`

**依赖：** T3、T10

**步骤：**

1. 按 regret 阈值生成覆盖矩阵。
2. 实现确定性贪心 set-cover 和容量失败信息。
3. 整合曲率候选、p-center/set-cover 与 `TransceiverLibrary`。
4. 实现原子式保存和严格加载，不覆盖输入状态数据。

**验证：** `python -m pytest tests/test_gmi_library.py -q`，可行覆盖覆盖所有状态，不可行覆盖明确失败，保存加载摘要不变。

## T13：实现在线鲁棒档位和评分

**文件：** `waveform_sim/gmi/selector.py`、`tests/test_gmi_selector.py`

**依赖：** T3、T6

**步骤：**

1. 定义 `fast/standard/conservative` 三档不确定性系数、最小增益和切换成本。
2. 计算候选下界、当前上界和 `gain_lower`。
3. 对上下文、连续性、overflow、无效 GMI 做 fail-closed 门控。
4. 评分按下界、原始 GMI、切换距离、token_id 确定性排序。

**验证：** `python -m pytest tests/test_gmi_selector.py -q -k score`。

## T14：实现选择决策和完整接收机对照

**文件：** `waveform_sim/gmi/selector.py`、`tests/test_gmi_selector.py`

**依赖：** T13

**步骤：**

1. 实现“保持/切换/不可判断”决策。
2. 改善下界等于门限时按“不切换”处理。
3. 完整匹配模式返回 `reference_flat` 且不发切换请求。
4. 决策记录状态 epoch、库摘要和每项惩罚。

**验证：** `python -m pytest tests/test_gmi_selector.py -q`，门限两侧、平局、完整模式和无效状态用例通过。

## T15：证明选择器没有 payload 依赖

**文件：** `tests/test_gmi_selector.py`

**依赖：** T14

**步骤：**

1. 检查 `ChannelState` 和选择接口不接受 payload、CRC、SER、BER、EVM 字段。
2. 用相同 CSI、不同外部 payload 验证选择结果不变。
3. 对未知附加字段执行拒绝或忽略策略并记录行为。

**验证：** `python -m pytest tests/test_gmi_selector.py -q -k no_payload`。

## T16：实现论文数值复现脚本

**文件：** `scripts/reproduce_gmi_paper_example.py`

**依赖：** T6、T8–T12

**步骤：**

1. 内置论文第 VI 节明确参数，不依赖硬件或 UI。
2. 计算 33×33 评估网格、等数量网格、曲率分配和有限贪心覆盖。
3. 输出 M=36 的 4×9 分配、两组 SNR sampled regret 和相对改善。
4. 支持可选 JSON 输出，配置和结果排序稳定。

**验证：** 运行 `python scripts/reproduce_gmi_paper_example.py`，得到论文同方向的 13.0%/15.4% 改善和可解析摘要。

## T17：锁定论文复现回归

**文件：** `tests/test_gmi_geometry.py`、`tests/test_gmi_library.py`

**依赖：** T16

**步骤：**

1. 将脚本核心计算抽为可测试入口。
2. 对论文给出的曲率、分配和 sampled regret 设置合理数值容差。
3. 对候选/状态顺序扰动验证结果确定性。

**验证：** `python -m pytest tests/test_gmi_geometry.py tests/test_gmi_library.py -q`。

## T18：实现采集稳定性指纹

**文件：** `waveform_sim/hardware/evidence.py`、`tests/test_gmi_abba_evidence.py`

**依赖：** T2

**步骤：**

1. 定义 `AcquisitionFingerprint` 和各字段有限性规则。
2. 实现连续帧 CFO、同步、pilot NMSE/power、RX 功率漂移判断。
3. overflow、probe gap、epoch 变化立即中断稳定序列。
4. 使用最终有效 CFO，不使用原始别名值。

**验证：** `python -m pytest tests/test_gmi_abba_evidence.py -q -k fingerprint`。

## T19：实现 ABBA 等样本窗口组合

**文件：** `waveform_sim/hardware/evidence.py`、`tests/test_gmi_abba_evidence.py`

**依赖：** T18

**步骤：**

1. 定义 A1/B1/B2/A2 窗口容器。
2. 强制四块目标帧数一致、A/B 总帧数一致。
3. 分别合并 A1+A2、B1+B2，同时保留每块统计。
4. 检查功率合同、资源 ID、上下文 ID 和采样连续性。

**验证：** `python -m pytest tests/test_gmi_abba_evidence.py -q -k combine`。

## T20：实现 GMI 主判据与硬件非回归保护

**文件：** `waveform_sim/hardware/evidence.py`、`tests/test_gmi_abba_evidence.py`

**依赖：** T19

**步骤：**

1. 先判断 GMI 改善与置信界，再检查 EVM/SER/CRC 非回归。
2. 明确 `improved/regressed/inconclusive` 原因和证据来源。
3. 构造线性漂移用例验证 ABBA 不产生假收益。
4. 构造非线性漂移、overflow、功率失配和指纹不稳定用例验证回滚。

**验证：** `python -m pytest tests/test_gmi_abba_evidence.py -q`。

## T21：在硬件后端初始化 GMI 配置和默认库

**文件：** `waveform_sim/hardware/fdidm_hardtest.py`、`tests/test_hardtest_import.py`

**依赖：** T3、T8、T12、T14

**步骤：**

1. 添加 objective、receiver mode、library capacity、robustness 和可选库路径配置。
2. 从当前 M/N/资源确定性生成默认库；显式库严格加载。
3. 将接收机配置与 channel estimator 分开保存和导出。
4. GMI 关闭时不启动工作线程、不改变固定 α/β。

**验证：** `python -m pytest tests/test_hardtest_import.py tests/test_gmi_hardware_integration.py -q -k init`。

## T22：建立最新 CSI 单槽和 GMI 工作线程

**文件：** `waveform_sim/hardware/fdidm_adaptive.py`、`waveform_sim/hardware/fdidm_hardtest.py`、`tests/test_gmi_hardware_integration.py`

**依赖：** T14、T21

**步骤：**

1. 可信解调帧只发布 `ChannelState`，不发布 payload 证据。
2. 使用单槽最新状态覆盖旧状态，后台线程异步评分。
3. 用 generation/context/library 摘要丢弃过期结果。
4. `stop()` 使线程退出并禁止晚到结果触发波形切换。

**验证：** `python -m pytest tests/test_gmi_hardware_integration.py -q -k worker`。

## T23：接入鲁棒选择和单事务门控

**文件：** `waveform_sim/hardware/fdidm_adaptive.py`、`tests/test_gmi_hardware_integration.py`

**依赖：** T22

**步骤：**

1. 将 `SelectionDecision` 发布到状态 API。
2. 改善不足保持 tracking；改善充分进入 candidate_ready。
3. 验证活动、手动修改、上下文变化和完整模式禁止新事务。
4. 同一 recommendation_seq 只允许启动一次验证。

**验证：** `python -m pytest tests/test_gmi_hardware_integration.py -q -k selection_gate`。

## T24：实现 A1/B1/B2/A2 状态推进

**文件：** `waveform_sim/hardware/fdidm_adaptive.py`、`tests/test_simple_hardware_adaptation.py`、`tests/test_gmi_hardware_integration.py`

**依赖：** T18–T20、T23

**步骤：**

1. 将验证窗口改为四块等样本 ABBA。
2. 每次实际 A↔B 切换后进入 stabilizing，指纹未稳定不计证据。
3. B1→B2 只切逻辑块，不重复应用相同波形。
4. watchdog 覆盖 applying、stabilizing、collecting 和 deciding。

**验证：** `python -m pytest tests/test_simple_hardware_adaptation.py tests/test_gmi_hardware_integration.py -q -k abba`。

## T25：实现 GMI 提交、正常功率恢复和安全回滚

**文件：** `waveform_sim/hardware/fdidm_adaptive.py`、`tests/test_simple_hardware_adaptation.py`

**依赖：** T24

**步骤：**

1. improved 后从 A2 应用 B，并释放临时比较功率合同。
2. regressed/inconclusive 后恢复 A 的正常功率。
3. rollback worker 失败进入 fail-closed，不被新推荐覆盖。
4. 手动修改和停止正确失效 generation，不在停止后重启 TX。

**验证：** `python -m pytest tests/test_simple_hardware_adaptation.py -q -k 'commit or rollback or stop'`。

## T26：发布硬件 GMI 状态、日志和导出字段

**文件：** `waveform_sim/hardware/fdidm_hardtest.py`、`tests/test_gmi_hardware_integration.py`、`tests/test_hardware_log_export.py`

**依赖：** T21–T25

**步骤：**

1. status/debug 增加 resource/library/token/receiver/GMI/penalty/ABBA 字段。
2. 理论、选择、硬件验证字段分组；旧 SER 字段标 diagnostic。
3. 日志记录每次保持/切换/提交/回滚的完整原因。
4. JSON 将非有限诊断转为 null，不转成有效零值。

**验证：** `python -m pytest tests/test_gmi_hardware_integration.py tests/test_hardware_log_export.py -q`。

## T27：验证固定模式兼容和硬件线程安全

**文件：** `tests/test_hardtest_import.py`、`tests/test_gmi_hardware_integration.py`

**依赖：** T26

**步骤：**

1. GMI 关闭时验证 α/β、波形构建、start/stop mock 行为不变。
2. 模拟 CSI 高频更新，验证 RX 调用不等待 GMI 评分。
3. 模拟 stop 与候选切换并发，验证无迟到提交。
4. 模拟上下文变化，验证旧库/旧评分不可执行。

**验证：** `python -m pytest tests/test_hardtest_import.py tests/test_gmi_hardware_integration.py -q`。

## T28：精简默认自适应操作区

**文件：** `waveform_sim/ui/fdidm_hardware_test_tab.py`、`tests/test_fdidm_hardware_comparison.py`

**依赖：** T21、T26

**步骤：**

1. 默认区提供启用、接收机模式、库预算、鲁棒等级和立即评估。
2. 移除默认区中的粗/细步长、稳定次数、最小 SER 增益和冷却帧。
3. 显式区分 estimator 和 receiver。
4. 参数变更通过后端配置入口失效旧评分/验证。

**验证：** `python -m pytest tests/test_fdidm_hardware_comparison.py -q -k controls`。

## T29：实现 GMI 选择与 ABBA 状态展示

**文件：** `waveform_sim/ui/fdidm_hardware_test_tab.py`、`tests/test_fdidm_hardware_comparison.py`

**依赖：** T28

**步骤：**

1. 状态行显示 receiver、token、GMI、gain lower 和验证阶段。
2. 完整模式显示“GMI 对 α/β 不敏感”的参考结论。
3. 四格区域用有限库 token 标记替代连续搜索叙述。
4. 可见文本保持短句，完整 reason 放 tooltip 和导出。

**验证：** `python -m pytest tests/test_fdidm_hardware_comparison.py -q -k 'gmi or adaptive_state or layout'`。

## T30：更新优势观测公平性和导出结构

**文件：** `waveform_sim/ui/hardware_advantage_observer.py`、`waveform_sim/ui/fdidm_hardware_test_tab.py`、`tests/test_fdidm_hardware_comparison.py`

**依赖：** T26、T29

**步骤：**

1. 按 `theory/selection/hardware_validation` 分组导出。
2. `resource_id` 或 receiver 不一致时标为不可比较。
3. 固定 M=1、自适应库和完整接收机对照共享控制预算。
4. 历史无 GMI 记录标 `legacy_ser`，不生成 GMI 优势徽章。

**验证：** `python -m pytest tests/test_fdidm_hardware_comparison.py tests/test_hardware_log_export.py -q`。

## T31：验证最小窗口布局和完整文案

**文件：** `tests/test_fdidm_hardware_comparison.py`、`tests/test_ui_regressions.py`

**依赖：** T28–T30

**步骤：**

1. 在项目支持的最小窗口渲染默认页。
2. 断言四个观测单元尺寸接近且不坍缩。
3. 断言状态行可读、长原因可通过 tooltip 获得。
4. 验证高级区折叠时默认控件无需滚动即可操作。

**验证：** `python -m pytest tests/test_fdidm_hardware_comparison.py tests/test_ui_regressions.py -q`，并保存一次离屏截图人工检查。

## T32：清理旧 SER 控制权和无效配置

**文件：** `waveform_sim/hardware/fdidm_adaptive.py`、`waveform_sim/hardware/fdidm_hardtest.py`、`waveform_sim/ui/fdidm_hardware_test_tab.py`、相关测试

**依赖：** T23、T28

**步骤：**

1. 将旧 SER 优化入口改名/标记为 diagnostic。
2. 删除已无调用方的 integer margin、canonical helper、recent metrics 等死字段。
3. 保留兼容状态字段但禁止其触发 GMI 波形切换。
4. 搜索并删除失效 UI signal 连接和重复状态映射。

**验证：** `rg -n "integer_margin|_adaptive_ab_recent_metrics|_adaptive_canonical_index" waveform_sim tests` 无生产调用；相关单测通过。

## T33：建立可重复真机短测脚本

**文件：** `scripts/validate_gmi_hardware.py`

**依赖：** T26

**步骤：**

1. 提供设备序列号、时长、TX/RX 增益和输出路径参数。
2. 启动前扫描设备，拒绝超过项目增益上限的输入。
3. 采集固定 token、选择结果、ABBA 状态、GMI、EVM、CRC、overflow 和 probe continuity。
4. 始终在 `finally` 中 stop/wait，输出版本化 JSON 和退出码。

**验证：** 无硬件时运行 `--help` 和 mock 测试；真机验收前人工核对默认增益不会自动覆盖用户值。

## T34：运行性能、确定性和完整回归

**文件：** 全部新增/修改测试

**依赖：** T1–T33

**步骤：**

1. 测量 16×16、16 token 对角评分耗时和 RX 非阻塞行为。
2. 连续运行论文复现和库构造两次，比较 JSON 摘要。
3. 运行所有 GMI 聚焦测试。
4. 运行完整 pytest、`git diff --check` 和导入检查。

**验证：** `python -m pytest -q` 全部通过；重复摘要一致；无空白错误。

## T35：执行受限真机验收并记录基线

**文件：** `log/` 下忽略的运行证据、Git 提交

**依赖：** T33、T34

**步骤：**

1. 使用用户已确认的 B210 端口和显式保守增益运行短测。
2. 先固定模式，再对角 GMI 模式，最后完整匹配对照；三者使用同一 `resource_id`。
3. 确认无遗留 flowgraph、无 overflow/probe gap，记录提交或回滚原因。
4. 清理本轮产生的缓存与无用临时文件，保留最终 JSON/日志证据。
5. 提交 GMI 实现并推送 GMI 分支；是否创建新的验收 tag 由用户在最终报告后决定。

**验证：** 远端 GMI 分支指向通过验收的提交；工作树干净；证据 JSON 可重新读取。

## 执行顺序

```text
T1 → T2 → T3
            ├→ T4 → T5 → T6 ───────────────┐
            └→ T7 → T8 → T9                │
                         │                  │
                         └→ T10 → T11 → T12│
                                           ├→ T13 → T14 → T15
                                           └→ T16 → T17

T2 → T18 → T19 → T20

T12 + T14 → T21 → T22 → T23
T20 + T23 → T24 → T25 → T26 → T27
T21 + T26 → T28 → T29 → T30 → T31
T23 + T28 → T32
T26 → T33

T1–T33 → T34 → T35
```

纯数学阶段 T1–T17 与纯证据阶段 T18–T20 逻辑独立，但实现时仍按提交组隔离，避免共享文件冲突。硬件/UI 阶段只在数学和证据测试稳定后开始。
