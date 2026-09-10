# FDIDM 硬件实验证据闭环 Tasks

## 执行约束

- 严格保持本周期边界：不修改软件仿真、不增加导频、不改变现有 diag-TF/MMSE 接收算法。
- 每个任务先写或更新测试，再完成最小实现，并立即运行该任务的验证命令。
- 所有测试默认离线运行；涉及 Qt 的测试使用 `QT_QPA_PLATFORM=offscreen`。
- 仓库当前存在与本功能无关的用户文件和删除记录。提交时只暂存本清单列出的文件，不使用批量暂存命令。
- T1～T6、T7～T11、T12～T15、T16～T17、T18～T21 分别形成一个逻辑提交；T22～T24 只修复本轮引入的回归，并按归属合并到对应提交或另建回归提交。

## 文件清单

| 操作 | 文件 | 职责 |
|---|---|---|
| 新建 | `waveform_sim/hardware/evidence.py` | 功率合同、已知符号测量、导频拟合、证据窗口与三态结论 |
| 修改 | `waveform_sim/hardware/fdidm_hardtest.py` | 波形预览/提交、功率核算、接收证据、状态与日志输出 |
| 修改 | `waveform_sim/hardware/fdidm_adaptive.py` | 共同功率下的基线/候选多帧验证状态机 |
| 修改 | `waveform_sim/ui/hardware_advantage_observer.py` | 精确计数、可比性、三态结果与导出 |
| 修改 | `waveform_sim/ui/fdidm_hardware_test_tab.py` | 自动观测、热更新、四宫格和证据展示 |
| 新建 | `tests/test_hardware_evidence.py` | 证据纯函数单元测试 |
| 新建 | `tests/test_hardware_power_normalization.py` | 实际波形功率合同和历史问题回归测试 |
| 修改 | `tests/test_simple_hardware_adaptation.py` | 验证状态机测试 |
| 修改 | `tests/test_hardware_advantage_observer.py` | 观测精确计数、功率可比性和导出测试 |
| 修改 | `tests/test_fdidm_hardware_comparison.py` | 页面生命周期、热更新、布局和文案测试 |
| 修改 | `tests/test_hardtest_import.py` | 新模块及无硬件导入回归 |

## T1：建立证据数据模型

**文件：** `waveform_sim/hardware/evidence.py`、`tests/test_hardware_evidence.py`
**依赖：** 无

**步骤：**

1. 定义 `TxPowerMetrics`、`TxPowerContract`、`KnownSymbolMetrics`、`PilotFitMetrics`、`EvidenceWindow` 和 `ValidationDecision` 数据对象。
2. 为可选数值统一约定有限值、空值和 JSON 可序列化形式，不让 `NaN` 参与真假判断。
3. 添加构造、默认值和序列化测试，确认模块只依赖标准库与 NumPy。

**验证：** 运行 `python -m pytest tests/test_hardware_evidence.py -q`，期望数据对象测试全部通过；运行 `python -c "import waveform_sim.hardware.evidence"`，期望无需 Qt、GNU Radio 或 UHD 即可导入。

## T2：实现单波形功率核算

**文件：** `waveform_sim/hardware/evidence.py`、`tests/test_hardware_evidence.py`
**依赖：** T1

**步骤：**

1. 实现对未缩放周期和数据片段的 RMS、峰值、PAPR 与安全 RMS 计算。
2. 实现按目标 RMS 缩放并保留峰值上限的计算，明确记录目标回退量和 `peak_limited`。
3. 覆盖零向量、空数据切片、普通波形和高 PAPR 波形边界。

**验证：** 运行 `python -m pytest tests/test_hardware_evidence.py -q -k power`，期望数值误差在测试容差内，高 PAPR 输入触发回退且不越过峰值上限。

## T3：实现基线/候选共同功率合同

**文件：** `waveform_sim/hardware/evidence.py`、`tests/test_hardware_evidence.py`
**依赖：** T2

**步骤：**

1. 用两侧 `safe_rms` 的较小值和 0.98 安全系数生成 `TxPowerContract`。
2. 生成稳定且可追踪的 `contract_id`，记录 α/β、请求 RMS、锁定 RMS、峰值限制和 0.10 dB 容差。
3. 实现功率可比性判断，覆盖同合同同功率、合同不一致、功率差超限和字段缺失。

**验证：** 运行 `python -m pytest tests/test_hardware_evidence.py -q -k "contract or comparable"`，期望共同 RMS 不高于任一侧安全 RMS，超限输入返回明确不可比原因。

## T4：实现已知符号 SER 与双 EVM

**文件：** `waveform_sim/hardware/evidence.py`、`tests/test_hardware_evidence.py`
**依赖：** T1

**步骤：**

1. 实现 `measure_known_symbols`，只比较接收符号与无歧义的已知参考前缀。
2. 用最小二乘复增益做共同幅相归一化，输出残余增益和相位。
3. 分别计算 data-aided EVM、decision-directed EVM 以及精确 `ser_errors/ser_symbols`。
4. 覆盖无误码、指定符号错误、整体幅相扰动、输入长度不同和空输入。

**验证：** 运行 `python -m pytest tests/test_hardware_evidence.py -q -k "known or evm or symbol"`，期望注入的错误数精确匹配，幅相归一化后的 data-aided EVM 符合构造值。

## T5：实现导频拟合和帧结构效率计算

**文件：** `waveform_sim/hardware/evidence.py`、`tests/test_hardware_evidence.py`
**依赖：** T1

**步骤：**

1. 实现原始导频网格相对当前对角准静态模型的拟合 NMSE、残差功率、拟合功率和 residual SINR。
2. 对零残差、零拟合功率和非有限输入定义稳定行为。
3. 实现同步、导频、数据、保护段、总周期、有效数据占比和训练/数据比统计。
4. 固化 `M=N=16、CP=4、diag-TF` 的 320 数据点、320 导频点回归样例。

**验证：** 运行 `python -m pytest tests/test_hardware_evidence.py -q -k "pilot or frame or efficiency"`，期望指标命名和样例计数准确，不产生端到端 `snr_db` 字段。

## T6：实现窗口统计与三态判定

**文件：** `waveform_sim/hardware/evidence.py`、`tests/test_hardware_evidence.py`
**依赖：** T3、T4

**步骤：**

1. 为 `EvidenceWindow` 实现有效帧累计、精确错误计数、EVM/BER/CRC 汇总及各类剔除计数。
2. 实现 95% Wilson 区间和基于精确 `k/n` 的实测 SER 改善 dB。
3. 实现 `ValidationDecision`：每侧至少 24 帧、最多 64 帧、改善结论每侧至少 100 个 SER 错误，并应用最小实测增益及 EVM/CRC 保护线。
4. 添加 improved、inconclusive、regressed、样本不足、功率不可比和上下文变化的确定性测试。
5. 用 2026-09-07 的 9.480%/9.645% SER、0.67 dB 预测改善和约 0.76 dB 功率差构造回归，断言不得判为 improved。

**验证：** 运行 `python -m pytest tests/test_hardware_evidence.py -q`，期望所有证据纯函数测试通过，历史场景只得到 inconclusive 或 regressed。

**提交检查点 A：** 仅暂存 `waveform_sim/hardware/evidence.py` 和 `tests/test_hardware_evidence.py`，检查 `git diff --cached` 后提交“硬件证据计算基座”。

## T7：拆分无副作用波形预览与提交

**文件：** `waveform_sim/hardware/fdidm_hardtest.py`、`tests/test_hardware_power_normalization.py`
**依赖：** T2

**步骤：**

1. 将现有帧构建逻辑拆成未缩放周期生成、无副作用预览和提交三个层次。
2. 定义 `WaveformBuildResult` 或等价内部结果，携带周期、数据切片和功率分析所需元数据。
3. 保证 `_preview_waveform_build` 不改变当前 α/β、发送源、运行计数、缓存或观测状态。
4. 保持正常启动时的发送内容、确定性 payload 和帧布局不变。

**验证：** 运行 `python -m pytest tests/test_hardware_power_normalization.py tests/test_hardtest_import.py -q -k "preview or import"`，期望预览前后后端状态一致且无硬件导入通过。

## T8：接入共同功率合同与实际发送指标

**文件：** `waveform_sim/hardware/fdidm_hardtest.py`、`tests/test_hardware_power_normalization.py`
**依赖：** T3、T7

**步骤：**

1. 实现 `_prepare_power_contract`，预览当前基线与候选 α/β 并生成共同功率合同。
2. 实现 `_commit_waveform_build` 的 `forced_rms` 与 `contract_id` 路径，提交后核算真实周期 RMS、数据 RMS、峰值、PAPR 和回退。
3. 正常非比较发送仍遵守当前目标 RMS 与峰值保护；比较发送遵守合同锁定 RMS。
4. 添加 α=0.5 与 α=0.75 高 PAPR 场景，断言共同合同下两侧功率差不超过 0.10 dB 且都不越峰值。

**验证：** 运行 `python -m pytest tests/test_hardware_power_normalization.py -q -k "contract or rms or peak"`，期望功率公平与射频峰值安全断言全部通过。

## T9：在接收帧接入已知符号证据

**文件：** `waveform_sim/hardware/fdidm_hardtest.py`、`tests/test_hardware_power_normalization.py`
**依赖：** T4、T7

**步骤：**

1. 在确定性发送帧构建时保存所有物理帧共有的已知编码应用前缀参考符号。
2. 在当前最佳均衡结果确定后计算 `KnownSymbolMetrics`，不改变候选评分、均衡结果或解码路径。
3. 保留原 `evm_percent` 兼容语义为 decision-directed EVM，新增明确命名的 data-aided/decision-directed 字段。
4. 累加运行期 `ser_errors_total` 和 `ser_symbols_total`，在 RX 状态重置时按既有计数语义处理。

**验证：** 运行 `python -m pytest tests/test_hardware_power_normalization.py -q -k "reference or data_aided or counter"`，期望参考前缀稳定、精确计数单调，兼容 EVM 字段仍可读取。

## T10：在现有导频估计旁路接入拟合诊断

**文件：** `waveform_sim/hardware/fdidm_hardtest.py`、`tests/test_hardware_power_normalization.py`
**依赖：** T5

**步骤：**

1. 在 `_estimate_htf_diag_from_pilot` 内使用未经跨帧平滑的当前帧导频网格计算 `PilotFitMetrics`。
2. 诊断完成后继续走原有平滑、缓存和 MMSE 输入，增加回归断言保证返回的信道估计与修改前算法等价。
3. 将软件 TDL 的配置值映射为 `tdl_injected_snr_db`，不删除旧日志读取所需的兼容入口。

**验证：** 运行 `python -m pytest tests/test_hardware_power_normalization.py -q -k "pilot_fit or estimator or injected"`，期望新增诊断可用且现有估计输出数值回归通过。

## T11：完善后端状态、日志与帧结构输出

**文件：** `waveform_sim/hardware/fdidm_hardtest.py`、`tests/test_hardware_power_normalization.py`、`tests/test_hardtest_import.py`
**依赖：** T8、T9、T10

**步骤：**

1. 在 debug snapshot、status 和逐帧日志中输出功率合同、双 EVM、精确 SER k/n、导频拟合与帧结构效率字段。
2. 对启动前、未同步和旧模拟路径给出稳定缺省值，不使状态读取异常。
3. 确认旧字段仍存在且含义未被悄然替换；新增字段使用一致单位和命名。
4. 增加 JSON 可序列化和无 GNU Radio/UHD 导入测试。

**验证：** 运行 `python -m pytest tests/test_hardware_power_normalization.py tests/test_hardtest_import.py -q`，期望状态键、数值单位、序列化及导入全部通过。

**提交检查点 B：** 仅暂存 `waveform_sim/hardware/fdidm_hardtest.py`、`tests/test_hardware_power_normalization.py` 和必要的 `tests/test_hardtest_import.py`，检查后提交“硬件收发证据与功率合同”。

## T12：定义多阶段验证状态与样本门禁

**文件：** `waveform_sim/hardware/fdidm_adaptive.py`、`tests/test_simple_hardware_adaptation.py`
**依赖：** T6、T11

**步骤：**

1. 用明确阶段替换旧的单帧基线与 5 帧 accepted/rejected 逻辑，初始化两个 `EvidenceWindow` 和当前合同上下文。
2. 实现帧门禁：只收相同 `contract_id`、相同信道上下文、同步有效、无 overflow 的帧。
3. 分别累计 dropped overflow、sync、context 和 power 原因。

**验证：** 运行 `python -m pytest tests/test_simple_hardware_adaptation.py -q -k "state or drop or context"`，期望阶段初始化正确且异常样本不增加有效帧数。

## T13：实现共同功率基线采集阶段

**文件：** `waveform_sim/hardware/fdidm_adaptive.py`、`tests/test_simple_hardware_adaptation.py`
**依赖：** T12

**步骤：**

1. 候选预测产生后先准备合同，并在共同 RMS 下重新提交当前基线波形。
2. 基线提交完成后丢弃 3 个切换瞬态帧，再采集 24～64 个有效帧。
3. 未达到最小帧数不得进入候选阶段；达到继续条件后才异步请求候选提交。

**验证：** 运行 `python -m pytest tests/test_simple_hardware_adaptation.py -q -k baseline`，期望前 3 帧被丢弃，第 24 个有效帧前不切换候选，功率合同沿阶段保持不变。

## T14：实现候选判定、保留与回滚

**文件：** `waveform_sim/hardware/fdidm_adaptive.py`、`tests/test_simple_hardware_adaptation.py`
**依赖：** T13

**步骤：**

1. 候选提交后丢弃 3 个瞬态帧，并按与基线相同的门禁采集 24～64 帧。
2. 调用统一三态分类器；预测收益只保留在元数据中，不直接影响 `outcome`。
3. improved 保留候选；inconclusive 和 regressed 异步恢复原 α/β 与合同发送设置。
4. 状态输出保留两侧窗口、区间、实测增益、EVM 差、结果和具体理由。

**验证：** 运行 `python -m pytest tests/test_simple_hardware_adaptation.py -q -k "improved or inconclusive or regressed or rollback"`，期望三种确定性场景结果与回滚行为正确。

## T15：保护异步更新、失效与重入路径

**文件：** `waveform_sim/hardware/fdidm_adaptive.py`、`tests/test_simple_hardware_adaptation.py`
**依赖：** T14

**步骤：**

1. 接收锁内只累计样本和设置 pending 标志，所有波形预览/提交由现有 worker 或独立线程完成。
2. 上下文改变、停止测试、手动参数改变和新请求重入时安全终止当前验证并记录原因。
3. 更新 `get_alpha_beta_adaptation_status`，兼容旧页面读取同时暴露三态结果和窗口进度。

**验证：** 运行 `python -m pytest tests/test_simple_hardware_adaptation.py -q`，期望原搜索测试与新增状态机测试全部通过，无重复提交或死锁模拟失败。

**提交检查点 C：** 仅暂存 `waveform_sim/hardware/fdidm_adaptive.py` 和 `tests/test_simple_hardware_adaptation.py`，检查后提交“多帧候选验证与安全回滚”。

## T16：让观测引擎消费精确累计计数

**文件：** `waveform_sim/ui/hardware_advantage_observer.py`、`tests/test_hardware_advantage_observer.py`
**依赖：** T11

**步骤：**

1. 优先使用后端 `ser_errors_total/ser_symbols_total` 的窗口增量，处理计数重置和非单调异常。
2. 使用 data-aided EVM 作为主 EVM，保留 decision-directed EVM 作为诊断字段。
3. 旧后端缺失精确计数时继续使用兼容估算，并在窗口与导出中设置 `estimated_counts=true`。

**验证：** 运行 `python -m pytest tests/test_hardware_advantage_observer.py -q -k "exact or counter or legacy or evm"`，期望精确 k/n 聚合正确，旧状态字典仍可工作且被标记为估算。

## T17：完善观测可比性、三态结果与导出

**文件：** `waveform_sim/ui/hardware_advantage_observer.py`、`tests/test_hardware_advantage_observer.py`
**依赖：** T16、T15

**步骤：**

1. 将 `contract_id`、实际周期 RMS 和功率回退加入窗口上下文与配对可比性检查。
2. 让 `on_toggle` 只在活动会话真实完成切换时返回事件；无活动会话返回可解释结果。
3. 配对结果消费后端三态验证证据；不可比或证据不足时不得形成绿色优势结论。
4. 导出精确计数、双 EVM、合同、剔除原因、Wilson 区间、三态结果、帧效率和导频拟合指标。

**验证：** 运行 `python -m pytest tests/test_hardware_advantage_observer.py -q`，期望原有窗口测试及新增功率、三态和 JSON 导出测试全部通过。

**提交检查点 D：** 仅暂存 `waveform_sim/ui/hardware_advantage_observer.py` 和 `tests/test_hardware_advantage_observer.py`，检查后提交“优势观测精确证据与可比性”。

## T18：修复页面热更新状态与真实切换反馈

**文件：** `waveform_sim/ui/fdidm_hardware_test_tab.py`、`tests/test_fdidm_hardware_comparison.py`
**依赖：** T17

**步骤：**

1. 在页面初始化中定义 `_adaptive_toggle_pending` 及同一路径所需全部状态。
2. 自适应开关触发的参数更新标记为 live-safe，避免清空已有曲线和观测窗口。
3. 只在 `observer.on_toggle` 返回真实事件时提示“窗口已切换”；否则展示会话未启动、状态未改变等具体原因。

**验证：** 运行 `python -m pytest tests/test_fdidm_hardware_comparison.py -q -k "toggle or pending or live"`，期望真实 UI 调用链不再出现属性异常，提示与事件一致。

## T19：接入硬件测试自动观测生命周期

**文件：** `waveform_sim/ui/fdidm_hardware_test_tab.py`、`tests/test_fdidm_hardware_comparison.py`
**依赖：** T18

**步骤：**

1. 后端成功进入运行态并获得当前上下文后自动调用观测会话 start。
2. 停止硬件测试时自动 stop 观测但保留窗口与结论。
3. 将原“开始优势观测”入口改成显式“重新开始观测”，仅用于用户主动清空重测。
4. 对启动失败、重复启动、停止后导出和旧模拟后端缺字段添加保护。

**验证：** 运行 `python -m pytest tests/test_fdidm_hardware_comparison.py -q -k "automatic or lifecycle or restart or stop"`，期望无需额外点击即可形成窗口，停止后结果仍可查看和导出。

## T20：统一优势页四宫格布局

**文件：** `waveform_sim/ui/fdidm_hardware_test_tab.py`、`tests/test_fdidm_hardware_comparison.py`
**依赖：** T18

**步骤：**

1. 将右下角窗口汇总放入与另外三张图一致的 `_PlotGridCell` 外层容器。
2. 四个单元使用相同的 `Ignored/Expanding` 尺寸策略、内容边距和 1:1 行列权重。
3. 汇总文本采用内部滚动或裁剪策略，不让 `sizeHint` 撑开网格。
4. 为 1400×800、1400×900、1200×780 三档离屏窗口添加宽高差容差测试，并在改变汇总文本长度后复测。

**验证：** 设置 `$env:QT_QPA_PLATFORM='offscreen'` 后运行 `python -m pytest tests/test_fdidm_hardware_comparison.py -q -k layout`，期望三档尺寸及长文本场景的 2×2 几何断言通过。

## T21：展示完整证据与三态文案

**文件：** `waveform_sim/ui/fdidm_hardware_test_tab.py`、`tests/test_fdidm_hardware_comparison.py`
**依赖：** T17、T19、T20

**步骤：**

1. 诊断区分别显示 data-aided EVM、decision-directed EVM、精确 SER k/n、raw/FEC BER 与 CRC。
2. 显示实际周期/数据 RMS、峰值、PAPR、回退、合同 ID 和功率可比性。
3. 显示 residual SINR、fit NMSE、导频/数据/同步保护点数、训练/数据比和有效数据占比。
4. 将软件 TDL 的 SNR 标成“注入设定”；预测值、实测值和注入值分栏或明确加前缀。
5. improved 使用绿色优势文案；inconclusive 使用中性证据不足文案；regressed 使用红色回滚文案。
6. 缺失新字段时显示“不可用”，刷新循环不中断。

**验证：** 运行 `python -m pytest tests/test_fdidm_hardware_comparison.py -q`，期望页面生命周期、几何、标签、缺字段兼容和三态颜色/文案测试全部通过。

**提交检查点 E：** 仅暂存 `waveform_sim/ui/fdidm_hardware_test_tab.py` 和 `tests/test_fdidm_hardware_comparison.py`，检查后提交“自动观测与四宫格证据展示”。

## T22：运行模块级集成回归

**文件：** 上述本轮文件
**依赖：** T15、T17、T21

**步骤：**

1. 联合运行证据、功率、状态机、观测、页面和硬件导入测试。
2. 修复字段名称、默认值、计数边界与模拟后端接口之间的不一致。
3. 确认所有修复仍符合已批准的 spec/plan，不借回归修复扩大范围。

**验证：** 设置 `$env:QT_QPA_PLATFORM='offscreen'` 后运行 `python -m pytest tests/test_hardware_evidence.py tests/test_hardware_power_normalization.py tests/test_simple_hardware_adaptation.py tests/test_hardware_advantage_observer.py tests/test_fdidm_hardware_comparison.py tests/test_hardtest_import.py -q`，期望专项测试全部通过。

## T23：运行相关硬件与 UI 回归

**文件：** 上述本轮文件；只有确认本轮回归时才修改对应既有测试或实现
**依赖：** T22

**步骤：**

1. 运行硬件抽象、硬件自适应、硬件流、日志导出、UI 回归和升级回归测试。
2. 对失败逐项判断是改动前基线失败还是本轮回归；只修复本轮回归。
3. 记录无法在无 USRP 环境执行的硬件在环项，不以跳过结果冒充通过。

**验证：** 设置 `$env:QT_QPA_PLATFORM='offscreen'` 后运行 `python -m pytest tests/test_hardware_abstraction.py tests/test_hardware_adaptive.py tests/test_hardware_stream.py tests/test_hardware_log_export.py tests/test_ui_regressions.py tests/test_upgrade_regression.py -q`，期望无本轮新增失败。

## T24：运行全量测试并审计改动范围

**文件：** 全部测试；不应产生新的范围外实现文件
**依赖：** T23

**步骤：**

1. 保存开发后 `git status --short`，与开发前用户已有改动清单对照。
2. 运行全量测试并记录通过、跳过、失败数量；失败项逐一与基线比较。
3. 运行 `git diff --check`，检查空白错误；检查 staged/unstaged diff，确保没有覆盖用户已有删除和未跟踪文件。
4. 如 T22～T24 产生了本轮回归修复，按所属模块明确暂存并提交；不暂存用户原有文件。

**验证：** 设置 `$env:QT_QPA_PLATFORM='offscreen'` 后运行 `python -m pytest -q` 和 `git diff --check`；期望全量测试无本轮新增失败、diff 无空白错误、改动范围只包含文件清单中的实现/测试以及本规格目录文档。

## 执行顺序

```text
T1 → T2 → T3 ───────────────┐
 │          └→ T6           │
 ├→ T4 ───────↗             │
 └→ T5                      │
                             ▼
T7 → T8 ─┐                 T12 → T13 → T14 → T15
 ├→ T9 ──┼→ T11                         │
 └→ T10 ─┘                              │
                 T11 → T16 → T17 ───────┤
                                         ▼
                              T18 → T19 ─┐
                                └→ T20 ──┼→ T21
                                         ▼
                              T22 → T23 → T24
```

## 覆盖自检

| Plan 组件 | 对应任务 |
|---|---|
| A：硬件证据计算 | T1～T6 |
| B：发送与接收诊断 | T7～T11 |
| C：候选验证状态机 | T12～T15 |
| D：优势观测引擎 | T16～T17 |
| E：FDIDM 硬件页面 | T18～T21 |
| F：离线、集成与全量测试 | T1～T24 各任务验证、T22～T24 汇总 |

- 所有任务均有明确文件、依赖、步骤和验证命令。
- 依赖图无循环；共同证据模块先于后端、状态机、观测与页面接入。
- F1～F10 和 AC1～AC10 均至少落入一个实现任务及一个专项或集成验证。
- 类型和接口名称与 `plan.md` 一致，没有扩大到接收算法重构、导频增长或软件仿真。
