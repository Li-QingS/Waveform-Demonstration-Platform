# FDIDM 硬件优势观测升级 Checklist

> 每一项通过运行代码或观察行为来验证，聚焦系统行为。括号内为验证方式。

## 实现完整性（spec AC → 检查项）

- [ ] AC1 离屏构造观测面板初始为空；投喂模拟窗口序列能算出开启前/开启后/变化及样本数；再次"开始观测"清空旧结果（验证：pytest 离屏用例断言面板文本与 `observer.windows`）
- [ ] AC2 模拟序列中配对窗口来自开关事件两侧正确时间区间；含 overflow 的窗口被剔除且结论面板显示剔除窗口数（验证：pytest 用例构造 overflow 采样，断言 `excluded_window_count` 与面板注记）
- [ ] AC3 错误数 <10 的窗口标"样本不足"且不进结论；≥100 带可信徽标；10–99 灰显"参考"（验证：pytest 三档 k 值驱动 `grade_from_k` 与面板徽标文本）
- [ ] AC4 时间轴图含 SER/EVM 代理/α/β 三组可区分曲线、开关事件竖线、异常剔除阴影；面板可见当前 α/β 与自适应搜索状态、轴可观测性（验证：pytest 断言曲线 item 数与状态行文本；真机上目视确认）
- [ ] AC5 结论面板改善量与窗口统计一致（dB 口径）；预测值（如有显示）明确标"预测"，不与实测混合（验证：pytest 用已知 SER 构造序列断言 dB 数值；grep 面板文案确认预测字样隔离）
- [ ] AC6 开关自适应走 live 路径：模拟后端证明观测功能未调用 stop/start、未触发结构重启（验证：pytest FakeBackend 记录 stop/start 调用次数为 0）
- [ ] AC7 旧定时对比代码全仓无残留（验证：`grep -rn "ComparisonSession\|comparison_session\|_start_comparison_demo\|btn_start_comparison\|_adaptive_observation_state" waveform_sim/ tests/` 零命中）
- [ ] AC8 观测会话导出 JSON 含窗口统计、事件序列、异常剔除记录与 schema_version（验证：pytest 导出到 tmp 路径并解析断言）

## 集成

- [ ] tab `_refresh_plots` 每 tick 调用 `observer.on_sample`，异常字段缺失（FakeBackend 极简 status）时不抛错不卡刷新（验证：离屏用例投喂缺字段 status）
- [ ] 开关自适应事件同时驱动：引擎建新窗、时间轴事件竖线、结论面板刷新，三者一致（验证：离屏用例 toggle 后三处断言）
- [ ] `tx_uncoded_bits_len`/`mod_order` 变化时窗口强制收尾重开，不出现混合口径窗口（验证：引擎单测）
- [ ] 所有新公开接口（引擎类/方法）至少被 tab 或测试真实调用（验证：编译 + 全量测试通过）

## 编译与测试

- [ ] `python -m pytest -q` 全量通过（原有 88+ 用例不回归 + 新增用例）（验证：CI 本地跑全量）
- [ ] `QT_QPA_PLATFORM=offscreen python -c "from waveform_sim.ui.fdidm_hardware_test_tab import FDIDMHardwareTestTab"` 导入成功（验证：命令退出码 0）
- [ ] `python -c "from waveform_sim.ui.hardware_advantage_observer import AdvantageObservationSession"` 无 PyQt 依赖成立（验证：在未装 PyQt 的最小环境中 import，或检查模块 import 区无 PyQt 字样）

## 端到端场景

- [ ] 场景 1（核心叙事）：离屏模拟——开始观测 → 关闭自适应跑 30 帧（SER 0.05）→ 手动开启自适应（SER 0.01）→ 跑 30 帧 → 结论面板出现"开启后 SER 改善 ≈7dB"+ 可信徽标，时间轴可见开关竖线与 α/β 轨迹移动（验证：pytest 端到端用例）
- [ ] 场景 2（异常健壮）：同上但中途注入 overflow 采样与失步帧 → 结论面板显示剔除窗口数 ≥1，改善量只由干净窗口计算（验证：pytest 端到端用例）
- [ ] 场景 3（真机彩排，有 USRP 时执行）：TDL 时变注入下开始观测，评审手动开关自适应 ≥2 次 → 时间轴实时滚动、结论面板实时更新、导出 JSON 可打开复核（验证：现场操作 + 打开导出文件）
