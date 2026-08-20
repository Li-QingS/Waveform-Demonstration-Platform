# FDIDM 仿真侧自适应 Checklist

> 本文档为软波谱 Spec 驱动开发流程的阶段四产物，基于已批准的 `32_fdidm_sim_adaptive_spec.md`、`33_fdidm_sim_adaptive_plan.md` 与 `34_fdidm_sim_adaptive_task.md`。
> 每一项通过运行代码或观察行为来验证，聚焦系统行为。

## 实现完整性

- [ ] `FDIDMSimAdaptiveMixin` 已实现并可被 FDIDM 仿真后端调用（验证：`python -m compileall waveform_sim` 通过；`from waveform_sim.simulation.fdidm_adaptive import FDIDMSimAdaptiveMixin` 成功）
- [ ] 搜索内核输出符合预期：推荐 α/β 在 [0,2]、预测 SER 有限、候选数≥1、耗时≥0（验证：`tests/test_fdidm_sim_adaptive.py` 单元测试通过）
- [ ] 评估事件字段完整：帧号、当前/推荐 α/β、ser_current/ser_best/ser_ofdm、gain_db、action、candidates、seconds 齐全（验证：单元测试断言 EvaluationRecord 字段）
- [ ] 决策策略符合预期：稳定计数、最小增益阈值、冷却、平局保持、auto_apply 开关均按 spec F2 生效（验证：对应单元测试通过）
- [ ] 并发安全：start / stop 可重复调用，搜索期间 step 循环与状态查询无异常、无数据竞争（验证：并发/集成测试通过）
- [ ] 自适应面板四类元素齐全：α/β 轨迹、预测 SER 对比、切换标记、状态文本（验证：offscreen 冒烟运行并观察控件/图元存在）
- [ ] 控制区控件即时生效：启用、自动应用、参数编辑、立即评估均透传后端（验证：UI 冒烟 + 集成测试断言）

## 集成

- [ ] `_LegacyFDIDMTransceiver` 在 `_simulate_one_frame` 末尾正确调用 `_adaptive_note_frame_processed()`，帧循环驱动快照调度（验证：真实后端 `step()` 若干帧后，启用自适应时状态进入 waiting/queued/optimizing 且 seq 递增）
- [ ] `LinkSimulator.get_adaptive_status` 正确透传后端状态，缺失时回退旧接口/默认值（验证：`tests/test_fdidm_sim_adaptive.py` 集成测试通过）
- [ ] 兼容壳 `FDIDMTransceiver` 经 `__getattr__` 可调用自适应接口（start / stop / request / status / history）（验证：集成测试调用全部接口无异常）
- [ ] UI 面板正确消费后端 status/history：仅当 recommendation_seq / 历史长度变化时重绘（验证：冒烟用假数据 refresh 断言曲线数量，再对真实后端轮询一轮）

## 编译与测试

- [ ] 项目编译无错误（验证：`python -m compileall waveform_sim` 退出码 0）
- [ ] 全部单元测试通过（验证：`python -m pytest -q` 全量通过，含新增 `tests/test_fdidm_sim_adaptive.py`）
- [ ] 不启用自适应时现有 FDIDM 仿真行为不变（验证：既有 `tests/test_engine.py`、`tests/test_waveforms.py` 等回归测试通过）

## 端到端场景

- [ ] 场景 1（动态信道自适应）：动态信道 + 启用自适应，运行 ≥ 200 帧 → 历史出现至少一次 apply 切换（或按策略全部合理 skip），α/β 轨迹出现阶梯与切换标记，状态文本显示推荐、预测增益、稳定计数（验证：集成测试 + offscreen 冒烟观察）
- [ ] 场景 2（静态信道不抖）：固定信道启用自适应运行若干帧 → 无 apply 或保持当前点，无频繁切换（验证：集成测试断言切换次数 ≤ 1 或为 0）
- [ ] 场景 3（自动应用关闭）：`adaptive_auto_apply=False` → 仿真 α/β 参数不变、推荐值仍可见（验证：集成测试断言参数未变、status 含推荐）
- [ ] 场景 4（上下文变化失效）：运行中修改调制阶数/网格/均衡器 → 旧推荐清空、状态回 waiting、下一帧重新评估（验证：集成测试断言 recommendation_seq 归零或重置）
- [ ] 场景 5（不阻塞）：自适应搜索期间仿真帧计数持续增长、UI 轮询正常无卡顿（验证：集成测试中 step 循环期间帧数递增；冒烟观察）
- [ ] 场景 6（历史容量）：连续评估超过容量上限后历史长度不超过 `adaptive_history_limit`，切换事件含旧值→新值、增益、帧号（验证：单元测试断言）
