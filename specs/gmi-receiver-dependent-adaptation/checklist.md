# FDIDM 接收机相关 GMI 自适应 Checklist

> 每一项必须通过运行代码、检查导出结果或观察真实页面/硬件行为来验证。代码存在但没有可重复证据，不视为通过。

## 实现完整性

- [x] GMI 核心包可在不导入 Qt、GNU Radio、UHD 的环境中独立导入（验证：在 `.venv` 中导入 `waveform_sim.gmi`，进程模块列表不出现硬件/UI 依赖）。
- [x] 公共资源、接收机、实验、token、库、状态和选择决策均使用显式模型传递（验证：运行 `tests/test_gmi_model.py`，合法模型往返一致，非法模型明确失败）。
- [x] GMI 库文件带有 schema 版本、内容摘要和唯一公共资源 ID（验证：保存后读取成功；修改内容、版本或资源字段后读取失败）。
- [x] GMI 自适应关闭时不启动选择线程、不加载可执行库、不改变手动 α/β（验证：运行固定模式后端测试并比较启用改造前的状态字段和波形摘要）。
- [x] 旧 SER 算法只作为 diagnostic 输出，不能在 GMI 模式触发切换（验证：构造 SER 推荐但 GMI 下界不足的状态，观察 token 保持不变）。

## 论文公式与数值正确性

- [x] 对角接收机 GMI 与论文式 (13) 的独立手算结果一致，`s_star∈[0,1]`（验证：运行 `tests/test_gmi_scoring.py -k diagonal`）。
- [x] 完整匹配接收机使用 log-det 结果，且不显式求矩阵逆（验证：完整模式数值测试通过，并检查评分路径诊断为 matched/logdet）。
- [x] α 或 β 平移 2 后，对角接收机 GMI 在数值容差内不变（验证：运行周期性参数化测试）。
- [x] `[0,2)×[0,2)` 基本域不包含重复的 0/2 端点 token（验证：生成多个容量的候选网格，token ID 均唯一）。
- [x] 完整匹配接收机对同一信道的不同 α/β token 给出相同 GMI（验证：运行 `tests/test_gmi_scoring.py -k full_matched`）。
- [x] 解析 GMI 曲率满足 `0.5U <= Q <= U`，完整接收机曲率为零（验证：运行 `tests/test_gmi_geometry.py -k curvature`）。
- [x] 论文第 VI 节 2×2 示例在 `rho=1` 与 `rho=10` 下给出正确的 α/β 敏感方向和数量级（验证：运行 `tests/test_gmi_geometry.py -k paper_example`）。
- [x] 容量 36 的论文示例选择 4×9 方向分配（验证：运行论文示例脚本并检查 JSON 的 `n_alpha=4`、`n_beta=9`）。
- [x] 论文示例中曲率分配相对 6×6 等数量网格的 sampled regret 改善与论文 13.0%/15.4% 在设定容差内一致（验证：运行 `scripts/reproduce_gmi_paper_example.py`）。

## 有限库与共同资源

- [x] 任意构造的库都不超过 `catalogue_cap`，且没有重复 token（验证：对容量 1、4、16、36 运行建库测试）。
- [x] 同一库中所有 token 的训练、保护、控制、功率和接收机资源摘要完全相同（验证：故意混入不同 `resource_id` 的 token，构造必须失败）。
- [x] `M=1` 与 `M=16` 比较使用同一个由 `bar_M=16` 决定的控制字段（验证：导出记录中的 `control_bits`、`control_samples` 和 `resource_id` 相同）。
- [x] 固定规模 p-center 在相同输入和随机种子下重复运行产生完全相同的 token 顺序和库摘要（验证：连续运行两次并比较 JSON/hash）。
- [x] set-cover 在可行阈值下覆盖所有状态，在不可行或超容量时明确报告失败（验证：运行 `tests/test_gmi_library.py -k cover`）。
- [x] 接收机相关非均匀库的测试集最大 regret 不高于同规模等数量网格（验证：固定状态集上比较 `max_regret`；没有改善时输出平局而不是正收益）。
- [x] 未知 schema、摘要不符、资源不一致或缺少执行字段的库不会被硬件后端加载（验证：运行模型和硬件初始化拒绝测试）。

## 在线鲁棒选择

- [x] 选择器输入模型中不存在 payload、解码字节、SER、BER、CRC 或数据辅助 EVM（验证：接口字段检查和非法输入测试通过）。
- [x] 相同 CSI/噪声状态配合不同 payload 外部数据产生相同选择结果（验证：运行 `tests/test_gmi_selector.py -k no_payload`）。
- [x] 候选改善下界小于或等于门限时保持当前 token（验证：测试门限下方、相等和浮点邻域三个用例）。
- [x] 候选改善下界超过门限时只产生一个带 sequence/context/library ID 的切换请求（验证：重复消费同一状态，不产生第二事务）。
- [x] `fast/standard/conservative` 三档使用固定、可导出的不确定性和切换成本规则（验证：同一状态下档位越保守，`gain_lower` 不增大）。
- [x] 完整匹配接收机返回 `reference_flat`，不发出 α/β 切换请求（验证：selector 和后端集成测试通过）。
- [x] 非有限 CSI、非正定协方差、overflow、probe gap 或上下文变化均禁止切换（验证：逐项注入异常并检查 fail-closed 原因）。
- [x] 16×16、16 token 的对角评分在一个正常后端处理周期内完成，且不阻塞 RX 调用线程（验证：性能测试记录评分时间和 RX mock 调用延迟）。

## 稳定指纹与 ABBA 验证

- [x] 实际波形切换后，未达到最少连续稳定指纹前不会向验证窗口加入样本（验证：模拟缓慢 CFO/同步收敛并检查计数保持为零）。
- [x] 稳定判断使用最终有效 CFO，而不是原始 CFO 别名（验证：原始 CFO 相差一个别名周期、有效 CFO 一致时仍可稳定）。
- [x] overflow、probe gap 或采集 epoch 变化会清空当前稳定序列并标记原因（验证：逐项注入异常）。
- [x] A1、B1、B2、A2 四块帧数相同，合并后 A/B 总帧数相同（验证：运行 `tests/test_gmi_abba_evidence.py -k combine`）。
- [x] B1→B2 不重复应用同一候选波形，A↔B 实际切换均使用新的 generation（验证：mock waveform publish 调用序列）。
- [x] 线性时间漂移不会被 ABBA 分类为候选收益（验证：构造单调改善但 token 无真实差异的序列，结果为平局或不确定）。
- [x] 明显非线性漂移、A1/A2 不一致或 B1/B2 不一致均得到 `inconclusive`（验证：运行漂移参数化测试）。
- [x] GMI 改善成立但 EVM/SER/CRC 显著回退时不提交候选（验证：构造 GMI 正、硬件指标回退的证据）。
- [x] EVM 改善但 GMI 改善不成立时不提交候选，也不显示 GMI 优势（验证：构造仅 EVM 改善的证据）。
- [x] improved 后候选恢复正常功率并释放比较合同；regressed/inconclusive 后基线恢复正常功率（验证：检查最终 α/β、RMS 和 contract ID）。
- [x] rollback 失败后系统保持 fail-closed，不接受新推荐（验证：模拟 waveform publish 异常）。
- [x] stop 与切换并发时不会在停止后重新启动发送或应用迟到 token（验证：并发 mock 测试通过且后台线程结束）。

## 硬件后端集成

- [x] channel estimator 与 receiver mode 是两个独立配置和状态字段（验证：改变 estimator 不会静默改变 receiver，反之亦然）。
- [x] 后端只把可信 CSI、噪声、同步和连续性信息发布给选择器（验证：捕获 `ChannelState`，确认不存在 payload 真值字段）。
- [x] 高频 CSI 更新只保留最新状态，不形成过期评分队列（验证：阻塞评分线程后发布多个 epoch，最终只消费最新 epoch）。
- [x] 手动修改 α/β 会中止当前事务、失效旧 generation，并建立新的固定 token 上下文（验证：硬件 mock 状态机测试）。
- [x] status/debug/export 包含 receiver、resource、library、token、GMI、penalty、decision 和 ABBA 字段（验证：运行 `tests/test_gmi_hardware_integration.py` 与日志导出测试）。
- [x] 非有限诊断在 JSON 中为 null 或明确 unavailable，不会变成零增益（验证：导出后重新解析并检查类型）。
- [x] 固定模式的构造、启动、停止和解调接口保持兼容（验证：现有 hardtest/import/receiver safety 测试全部通过）。

## 页面与可观测性

- [x] 默认自适应区只暴露启用、接收机模式、库预算、鲁棒等级和立即评估（验证：构造页面并枚举可见控件）。
- [x] 默认区不显示粗步长、细步长、稳定次数、最小 SER 增益和冷却帧（验证：最小窗口截图和控件可见性断言）。
- [x] 页面明确区分 CSI 估计器与 GMI 接收机模式（验证：标签、tooltip 和导出字段使用不同名称）。
- [x] 状态行能直接读出当前 receiver、token、净 GMI 下界和 ABBA 阶段（验证：分别注入 tracking、candidate、ABBA、rollback 状态）。
- [x] 完整接收机模式显示“GMI 对 α/β 不敏感/无需切换”的对照说明（验证：UI 状态测试）。
- [x] 页面不会把 EVM/SER/CRC 改善标注为 GMI 改善（验证：只注入硬件指标改善，GMI 徽章保持未确认）。
- [x] 四个观测区域在项目支持的最小窗口中尺寸接近、文字不重叠、右下角不挤压（验证：离屏渲染截图与尺寸断言）。
- [x] 长决策原因在可见区保持短句，完整内容可从 tooltip 和导出中获得（验证：注入超长 reason）。
- [x] 历史无 GMI 数据标记为 `legacy_ser`，不生成 GMI 优势结论（验证：加载历史记录 fixture）。

## 编译、测试与代码质量

- [x] `python -m pytest tests/test_gmi_model.py tests/test_gmi_scoring.py tests/test_gmi_geometry.py tests/test_gmi_library.py tests/test_gmi_selector.py -q` 全部通过。
- [x] `python -m pytest tests/test_gmi_abba_evidence.py tests/test_gmi_hardware_integration.py -q` 全部通过。
- [x] 现有 FDIDM 硬件、自适应、功率、日志和 UI 聚焦测试全部通过。
- [x] `python -m pytest -q` 完整测试全部通过，无跳过的 GMI 核心测试。
- [x] `git diff --check` 无空白错误，新增生产代码中没有未处理的 TODO/TBD。
- [x] 搜索确认硬件/UI 路径不再调用旧 integer margin、recent metrics、canonical helper；现有命中仅为独立仿真兼容实现、旧验证脚本和对应测试夹具。
- [x] 连续两次运行论文复现和默认建库得到相同 JSON 内容摘要。

## 端到端场景

- [x] 场景 1——论文复现：运行复现脚本，从论文参数生成曲率、4×9 分配和 sampled regret，对比结果在容差内一致。
- [x] 场景 2——软件选择：对一个对角接收机信道加载 16-token 库，选择器从固定 token 找到净 GMI 更高候选，并在置信不足时保持当前 token。
- [x] 场景 3——完整接收机对照：同一信道切换到完整匹配模式，所有 token GMI 近似相同，后端和页面均不发起切换。
- [x] 场景 4——漂移保护：向 ABBA 验证注入线性链路恢复，候选不被误判为优势；注入真实 token 收益且四块稳定时完成提交。
- [x] 场景 5——固定模式回归：关闭 GMI，启动、接收、停止固定 α/β mock 链路，所有既有状态和清理行为正常。
- [x] 场景 6——真机短测：设备扫描成功后，以显式保守增益运行固定/对角 GMI/完整对照三段，三者 `resource_id` 相同、无 overflow/probe gap、最终 flowgraph 已停止。
- [x] 场景 7——证据回放：重新读取真机 JSON，仅根据 token、公共资源、GMI 判据、功率合同和 ABBA 证据即可解释每次保持、提交或回滚。

## Git 与交付

- [x] 所有实现位于 `codex/gmi-adaptive` 分支，未改写 GMI 前基线 tag。
- [x] 数学内核、硬件集成、UI/验收按逻辑组提交，提交说明能对应任务清单。
- [x] 缓存、临时截图、临时论文渲染和无用运行文件未进入 Git。
- [x] 最终通过测试的提交已推送远端 GMI 分支，工作树干净。
- [x] 最终报告记录提交号、测试数量、论文复现结果、真机参数、GMI/EVM/CRC/overflow 结果和仍存在的限制。

## 2026-10-08 验收记录

- 软件：`.venv` 全量 `pytest` 为 `389 passed`；`compileall`、纯 GMI 包导入和 `git diff --check` 通过。
- 论文复现：连续两次 SHA-256 均为 `4C35C0A1380D87A0F1A7B38B724E141A69ACADBCA529DCC351E48E690AF69386`；容量 36 得到 4×9，regret 改善为 13.0348%/15.4439%。
- 真机：B210 `31896C6`，500 kS/s，2.4 GHz，TX/RX 增益 25/45 dB；固定、diagonal、full-matched 共用 `res-6d540944813a1f69`，三段均为 0 overflow、0 processing gap，并干净停止。
- 闭环：diagonal 从 `(0.5, 1.0)` 选择并提交 `(1.666666666667, 1.8)`；GMI 增益下界 0.05458 nat/sample，ABBA 合并后 SER 4.63%→3.05%、EVM 改善 4.05 个百分点、CRC 无回退。
- 证据：`log/gmi_hardware_validation_20261008.json`（运行证据受 `.gitignore` 保护，不进入版本库）。
- 逻辑提交：数学内核 `c7ad99d`、硬件闭环 `8011e02`、UI/可观测性 `9ce0446`。

## Spec 验收追踪

| Spec 验收标准 | Checklist 证据区域 |
|---|---|
| AC1 | 论文公式与数值正确性、端到端场景 1 |
| AC2 | 周期性和基本域唯一性检查 |
| AC3 | 完整匹配接收机检查、端到端场景 3 |
| AC4 | 有限库容量和共同资源检查 |
| AC5 | 非均匀库最大 regret 对比 |
| AC6 | 在线鲁棒选择门限与单事务检查 |
| AC7 | 无 payload 接口与不变性检查 |
| AC8 | 稳定指纹与 ABBA 漂移/异常检查 |
| AC9 | 默认页面、最小窗口和 tooltip 检查 |
| AC10 | 公共资源导出和 `legacy_ser` 检查 |
| AC11 | 固定模式兼容与完整 pytest 回归 |
| AC12 | 真机短测、停止清理和证据回放 |
