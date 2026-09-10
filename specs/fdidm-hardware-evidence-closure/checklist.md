# FDIDM 硬件实验证据闭环 Checklist

> 每一项都必须通过实际命令输出、离屏界面行为、导出文件或运行日志验证。验收时先收集证据，再勾选；没有 USRP 时不得把硬件在环项标记为通过。

## A. 实现完整性

- [x] A1. 证据计算模块可在没有 Qt、GNU Radio 和 UHD 的 Python 环境中独立导入，功率、双 EVM、精确 SER、导频拟合、窗口统计和三态判定均有可调用行为。（验证：运行 `python -c "import waveform_sim.hardware.evidence"` 以及 `python -m pytest tests/test_hardware_evidence.py -q`，期望导入成功且测试全通过）
- [x] A2. 波形预览不会修改正在使用的 α/β、发送向量、缓存或运行计数，只有正式提交才改变发送波形。（验证：运行 `python -m pytest tests/test_hardware_power_normalization.py -q -k preview`，期望预览前后状态快照相同）
- [x] A3. 硬件后端启动前、运行中和未同步状态下都能返回包含稳定缺省值的状态字典，新字段缺少有效测量时明确表示不可用。（验证：运行 `python -m pytest tests/test_hardware_power_normalization.py tests/test_hardtest_import.py -q -k "status or default or import"`，期望无属性异常或非序列化值）
- [x] A4. 接收诊断只旁路观察原始导频拟合，现有 diag-TF 平滑、缓存和 MMSE 均衡的数值行为未被改变。（验证：运行 `python -m pytest tests/test_hardware_power_normalization.py -q -k "pilot_fit or estimator"`，期望新增诊断前后的信道估计回归值一致）
- [x] A5. 候选验证状态可观察到基线应用、基线瞬态、基线采集、候选应用、候选瞬态、候选采集和最终结果，各阶段进度与原因可从状态获取。（验证：运行 `python -m pytest tests/test_simple_hardware_adaptation.py -q -k state`，期望阶段序列及窗口计数符合设计）

## B. 功率公平与射频安全（AC3）

- [x] B1. 任一已构建发送周期都报告实际周期 RMS、数据段 RMS、峰值、PAPR、相对目标回退、是否受峰值限制和合同标识。（验证：构造 α=0.5 与 α=0.75 波形并运行 `python -m pytest tests/test_hardware_power_normalization.py -q -k metrics`，期望字段齐全且与直接 NumPy 计算一致）
- [x] B2. 同一基线/候选比较使用共同安全 RMS，双方实际周期功率差不超过 0.10 dB，且峰值均不超过现有射频安全上限。（验证：运行 `python -m pytest tests/test_hardware_power_normalization.py -q -k "contract or peak"`，期望公平性和峰值断言通过）
- [x] B3. 模拟合同不一致、功率字段缺失或实际 RMS 差超过 0.10 dB 时，比较结果显示“功率不可比/无结论”，不会显示绿色优势。（验证：运行 `python -m pytest tests/test_hardware_evidence.py tests/test_hardware_advantage_observer.py tests/test_fdidm_hardware_comparison.py -q -k "power and (incomparable or inconclusive)"`，期望原因贯穿判定、观测和页面）
- [x] B4. 峰值保护始终有效，没有通过关闭保护或仅提高候选增益制造改善。（验证：检查功率专项测试输出及一次导出报告，期望两侧均记录峰值限制状态和真实回退，所有峰值不越限）

## C. 指标口径与高 EVM 诊断（AC4、AC5）

- [x] C1. 对已知注入错误的符号序列，SER 错误数 `k` 和总符号数 `n` 精确等于构造值，不再由 `SER×帧数` 反推。（验证：运行 `python -m pytest tests/test_hardware_evidence.py -q -k "known or symbol"`，期望精确计数断言通过）
- [x] C2. 对整体增益、相位和指定噪声扰动，data-aided EVM 与 decision-directed EVM 分别符合预期，页面和导出不会混用二者。（验证：运行 `python -m pytest tests/test_hardware_evidence.py tests/test_fdidm_hardware_comparison.py -q -k evm`，期望数值及两个独立标签断言通过）
- [x] C3. 页面同时显示精确 SER k/n、实测 SER、预测 SER、raw BER、FEC BER 和 CRC 成功率；FEC 后正确不会单独触发物理层优势结论。（验证：向离屏页面注入“FEC 成功但物理层退化”状态，期望页面显示各指标且结论不是 improved）
- [x] C4. 软件 TDL 配置为 35 dB 时，页面显示“TDL 注入设定 35 dB”或等价明确文案，不显示成端到端实测 SNR。（验证：运行 `python -m pytest tests/test_fdidm_hardware_comparison.py -q -k injected`，期望文案断言通过）
- [x] C5. 导频诊断显示 residual SINR 和 fit NMSE，并明确其包含噪声、残余频偏、RF 失真及模型失配，字段和页面中没有把它命名为端到端实测 SNR。（验证：运行 `python -m pytest tests/test_hardware_evidence.py tests/test_hardware_power_normalization.py tests/test_fdidm_hardware_comparison.py -q -k "pilot_fit or residual or snr"`，期望数值、字段和标签测试通过）

## D. 多帧验证与三态结论（AC6、AC7、AC9）

- [x] D1. 基线与候选切换后各丢弃 3 个瞬态帧，每侧少于 24 个有效帧时不得输出 improved；到 64 帧仍不足时输出 inconclusive。（验证：运行 `python -m pytest tests/test_simple_hardware_adaptation.py -q -k "settling or insufficient or max_frames"`，期望门槛边界全部通过）
- [x] D2. 候选 SER 显著优于基线、双方错误数均不少于 100、实测增益达到门槛且 EVM/CRC 未退化时，结果为 improved 并保留候选。（验证：运行 `python -m pytest tests/test_hardware_evidence.py tests/test_simple_hardware_adaptation.py -q -k improved`，期望 Wilson 区间分离且 α/β 保持候选值）
- [x] D3. 差异不显著、错误数不足、样本不足或功率不可比时，结果为 inconclusive，自动恢复原 α/β，并保留具体原因。（验证：运行 `python -m pytest tests/test_hardware_evidence.py tests/test_simple_hardware_adaptation.py -q -k inconclusive`，期望三态结果、回滚和原因断言通过）
- [x] D4. 候选 SER 显著恶化或 data-aided EVM/CRC 越过退化保护线时，结果为 regressed 并自动恢复原 α/β。（验证：运行 `python -m pytest tests/test_hardware_evidence.py tests/test_simple_hardware_adaptation.py -q -k regressed`，期望退化判定和回滚断言通过）
- [x] D5. 预测改善只能触发候选试验，不能将实测 inconclusive 或 regressed 改写成 improved。（验证：向状态机输入正预测收益和无改善实测窗口，期望最终结论仍由实测分类器决定）
- [x] D6. overflow、失步、上下文改变和功率超限样本都不会增加有效窗口帧数，页面、日志和导出分别列出各类剔除数量或原因。（验证：运行 `python -m pytest tests/test_simple_hardware_adaptation.py tests/test_hardware_advantage_observer.py tests/test_fdidm_hardware_comparison.py -q -k "overflow or sync or context or drop"`，期望端到端原因一致）
- [x] D7. 复现 2026-09-07 的“预测 +0.67 dB、候选功率约低 0.76 dB、长期 SER 9.480% 对 9.645%”输入时，只得到 inconclusive 或 regressed，绝不输出“实测优势已验证”。（验证：运行 `python -m pytest tests/test_hardware_evidence.py tests/test_hardware_power_normalization.py -q -k "20260907 or historical"`，期望历史回归断言通过）

## E. 优势观测生命周期与页面交互（AC1）

- [x] E1. 硬件后端成功进入运行态后自动开始观测，用户无需额外点击“开始优势观测”即可看到首段时间线积累。（验证：运行离屏端到端测试，执行启动并馈入固定参数帧，期望会话 active 且首窗口样本增加）
- [x] E2. 手动开启或关闭自适应时，只有活动会话真实切换窗口后才出现“观测窗口已切换”；会话不可用或状态未改变时显示可操作原因。（验证：运行 `python -m pytest tests/test_fdidm_hardware_comparison.py -q -k toggle`，期望事件数量与页面提示一一对应）
- [x] E3. 参数热更新完整路径不会出现 `_adaptive_toggle_pending` 或其他界面属性异常，也不会清空已有观测曲线。（验证：运行真实页面方法链的离屏测试并检查异常捕获和曲线样本数，期望无异常且样本保留）
- [x] E4. “重新开始观测”明确清空旧窗口并从当前自适应状态建立新会话；停止硬件测试会收尾但保留结果和导出能力。（验证：运行 `python -m pytest tests/test_fdidm_hardware_comparison.py -q -k "restart or stop or export"`，期望生命周期行为通过）
- [x] E5. 离屏端到端执行“启动 → 固定参数采样 → 手动开启自适应 → 候选采样 → 停止”，最终产生两段窗口和三态结论，全程不需要另点开始观测且无 UI 属性异常。（验证：运行 `python -m pytest tests/test_fdidm_hardware_comparison.py -q -k end_to_end`，期望完整流程测试通过）

## F. 四宫格与页面展示（AC2）

- [x] F1. 在 1400×800、1400×900 和 1200×780 三档离屏窗口中，优势页四个单元保持 2×2，同行同列宽高差均在测试定义的布局容差内。（验证：设置 `$env:QT_QPA_PLATFORM='offscreen'` 后运行 `python -m pytest tests/test_fdidm_hardware_comparison.py -q -k layout`，期望三档几何断言全部通过）
- [x] F2. 将右下角汇总文本从短内容改为长内容后，四宫格比例不发生超容差变化，文本在单元内部滚动或裁剪。（验证：运行长短文本离屏几何测试，期望四个外层单元几何保持等大）
- [x] F3. improved、inconclusive、regressed 分别使用优势绿色、中性提示、退化/回滚红色；功率不可比和样本不足不能使用优势绿色。（验证：向离屏页面依次注入四类状态并检查可见文案和样式属性）
- [x] F4. 旧模拟后端或旧日志缺失新增字段时，对应位置显示“不可用”，定时刷新仍继续运行。（验证：运行 `python -m pytest tests/test_fdidm_hardware_comparison.py tests/test_hardware_advantage_observer.py -q -k "missing or legacy"`，期望连续两次刷新和兼容聚合均通过）

## G. 导频与帧结构效率（AC8）

- [x] G1. `M=N=16、CP=4、diag-TF` 配置明确显示数据 320 点、导频 320 点，以及同步、保护间隔、总周期、训练/数据比和有效数据占比。（验证：运行 `python -m pytest tests/test_hardware_evidence.py tests/test_hardware_power_normalization.py tests/test_fdidm_hardware_comparison.py -q -k "frame or efficiency or overhead"`，期望后端与页面数字一致）
- [x] G2. 切换估计模式或帧结构后，训练/数据比由实际帧结构重新计算，不沿用固定常数。（验证：构造至少两种模式状态并刷新页面，期望显示值随输入变化）
- [x] G3. 高开销估计模式在启动前给出训练/数据开销提示，本轮没有通过增加导频长度来改善 EVM。（验证：离屏触发高开销配置并观察提示；比较改动前后默认帧段长度，期望导频长度未增长）

## H. 日志与导出证据链（AC10）

- [x] H1. 一次完整观测的 JSON 导出可被标准解析器读取，包含功率合同、实际功率、双 EVM、精确 SER k/n、raw/FEC BER、CRC、验证窗口、Wilson 区间、剔除原因、三态结论、帧效率和导频拟合指标。（验证：运行导出测试并用 `json.load` 读取，逐项断言所需字段存在且类型正确）
- [x] H2. 旧后端使用估算计数时导出明确包含 `estimated_counts=true`；使用新后端精确累计计数时为 false。（验证：分别导出兼容和精确场景，期望标志正确）
- [x] H3. 从导出报告可以复算双方 SER、实测改善 dB、功率差以及结论为何成立或不成立。（验证：测试中仅使用导出 JSON 的 k/n、RMS 和门槛重新计算，期望与报告结论一致）
- [x] H4. 调试日志明确区分 predicted、measured、injected 和 residual，不使用同一个 `SNR` 标签混合不同口径。（验证：生成一段离线调试日志并搜索字段/标签，期望四类来源可区分）

## I. 集成、性能与兼容性

- [x] I1. 接收锁内只执行有限的样本累计和状态更新，波形重建/提交由异步路径完成；快速连续请求不会重复提交或死锁。（验证：运行 `python -m pytest tests/test_simple_hardware_adaptation.py -q -k "async or reentry or lock"`，期望超时内完成且提交次数符合预期）
- [x] I2. 停止测试、手动改变 α/β 或信道上下文改变时，进行中的验证安全失效并记录原因，不在旧上下文继续下结论。（验证：运行状态机失效测试，期望验证停止、原因保留、候选不被错误接受）
- [x] I3. FDIDM 硬件模块在未安装 GNU Radio/UHD 时仍可导入，相关离线测试无需真实设备和显示器。（验证：设置离屏环境后运行 `python -m pytest tests/test_hardtest_import.py tests/test_hardware_evidence.py -q`，期望全部通过）
- [x] I4. 现有硬件状态消费者和旧模拟后端继续读取旧兼容字段，不因新增证据字段崩溃。（验证：运行硬件抽象、硬件流、日志导出和 UI 兼容测试，期望无新增失败）
- [x] I5. 修改仅影响 FDIDM 硬件发送、接收诊断、自适应验证、优势观测和对应测试，软件仿真及其他硬件页面无行为回归。（验证：运行相关回归与全量测试，并审查 `git diff --name-only`，期望无范围外实现改动）

## J. 自动化测试与代码质量（AC10）

- [x] J1. 专项测试全部通过。（验证：设置 `$env:QT_QPA_PLATFORM='offscreen'`，运行 `python -m pytest tests/test_hardware_evidence.py tests/test_hardware_power_normalization.py tests/test_simple_hardware_adaptation.py tests/test_hardware_advantage_observer.py tests/test_fdidm_hardware_comparison.py tests/test_hardtest_import.py -q`）
- [x] J2. 相关硬件和 UI 回归无本轮新增失败。（验证：运行 `python -m pytest tests/test_hardware_abstraction.py tests/test_hardware_adaptive.py tests/test_hardware_stream.py tests/test_hardware_log_export.py tests/test_ui_regressions.py tests/test_upgrade_regression.py -q`，逐项记录实际结果）
- [x] J3. 全量测试完成，报告实际通过、跳过和失败数量；任何失败均区分为改动前基线失败或本轮回归。（验证：运行 `python -m pytest -q`，与开发前基线结果对照）
- [x] J4. Python 文件可编译，改动不存在空白错误。（验证：运行 `python -m compileall -q waveform_sim tests` 和 `git diff --check`，期望退出码均为 0）
- [x] J5. Git 提交按证据模块、后端、状态机、观测引擎和页面五个逻辑组组织，每个提交只包含对应文件。（验证：运行 `git log --stat` 和 `git show --stat` 检查本轮提交）
- [x] J6. 开发前已有的删除记录和未跟踪文件未被覆盖或误提交。（验证：对照开发前后的 `git status --short`，确认 `_fdidm_v6_backup/`、`alpha_beta_validation.json`、图片、升级文档及既有 specs 状态保持为用户原状态，除非用户另有指示）

## K. 端到端场景

- [x] K1. 正常改善：启动硬件测试后自动观测，共同功率采集基线并切换候选；样本充分且显著改善时页面显示 improved，候选被保留，导出可复算结论。（验证：离屏 FakeBackend 完整流程测试）
- [x] K2. 证据不足：候选预测有收益但实测区间重叠或错误数不足；页面显示 inconclusive 和原因，候选回滚，导出不含“实测优势已验证”。（验证：离屏 FakeBackend 完整流程测试）
- [x] K3. 明显退化：候选 EVM/SER 显著恶化；页面显示 regressed/已回滚，原 α/β 恢复，导出保留两侧窗口。（验证：离屏 FakeBackend 完整流程测试）
- [x] K4. 异常链路：候选窗口发生 overflow、失步或功率不公平；异常帧被剔除，页面与导出给出具体原因，不产生优势结论。（验证：离屏 FakeBackend 注入异常的完整流程测试）
- [ ] K5. 可选硬件在环：连接单台 USRP 后执行 RF-only → 静态 TDL → 加噪声 → 加频偏/多普勒分层测试，记录功率、双 EVM、SER k/n、残差指标与三态结论。（验证：保存实际运行日志和 JSON 导出；未连接设备时标记“未执行”，不得标记通过）

## 验收覆盖

| Spec 验收标准 | Checklist 条目 |
|---|---|
| AC1 自动观测端到端 | E1～E5、K1～K4 |
| AC2 四宫格等大 | F1～F2 |
| AC3 功率公平 | B1～B4 |
| AC4 双 EVM 与精确 SER | C1～C3 |
| AC5 SNR/残差口径 | C4～C5 |
| AC6 三态验证与回滚 | D1～D5、K1～K3 |
| AC7 异常样本剔除 | D6、K4 |
| AC8 导频效率 | G1～G3 |
| AC9 历史日志场景 | D7 |
| AC10 专项与全量测试 | H1～H4、J1～J6 |

## 验收结果记录

验收日期：2026-09-10。

- 通过项：52/53；唯一未勾选项为 K5（真实 USRP 硬件在环），本机本轮未连接设备，因此按约束记录为“未执行”，不伪造通过结果。
- 专项测试：80 passed，命令覆盖证据、功率、硬件自适应、观测、硬件页面及无设备导入。
- 相关回归：25 passed、3 failed。3 项均属于未修改的软件仿真页 `FDIDMTab`，并已在改动前提交 `74378a5` 的独立工作树复现相同失败。
- 全量测试：190 passed、6 failed。6 项均在改动前提交独立工作树中复现：3 项为软件仿真自适应异步时序，3 项为软件仿真页面时间图/扫频接口；本轮没有新增失败。
- 静态检查：`python -m compileall -q waveform_sim tests` 与 `git diff --check` 均通过。
- 提交 A～E：`bb1a1dc`、`228d7e5`、`7283177`、`1d9018e`、`58f6c49`；补充验收测试提交：`6906518`。
- 2026-09-07 历史输入：实际功率差约 0.76 dB，超过 0.10 dB 容差，最终判定为 `inconclusive`，不会输出 improved。
- 改动范围审计：实现文件只涉及 FDIDM 硬件证据、硬件后端、硬件自适应、优势观测与硬件页面；开发前已有删除记录及未跟踪文件保持原状。
