# FDIDM 仿真侧自适应 Plan

> 本文档为软波谱 Spec 驱动开发流程的阶段二产物，基于已批准的 `32_fdidm_sim_adaptive_spec.md`，未获用户批准前不进入 task 阶段。

## 架构概述

- 自适应闭环放在真实仿真后端 `_LegacyFDIDMTransceiver` 内部，通过新增 mixin `FDIDMSimAdaptiveMixin`（新文件 `waveform_sim/simulation/fdidm_adaptive.py`）混入；兼容壳 `FDIDMTransceiver` 靠现有 `__getattr__` 自动透传接口，无需改壳。
- 搜索内核移植硬件端 `fdidm_adaptive.py` 的“粗搜 + 细搜”（含对角快速路径），在仿真侧自包含实现（补齐 `_apply_gamma_axis` 等辅助），不依赖硬件类。
- UI 新增“自适应过程”面板（新文件 `waveform_sim/ui/fdidm_adaptive_widgets.py`）：控制区 + 双图（α/β 轨迹、预测 SER 对比）+ 状态文本；`fdidm_tab.py` 只做接入和定时轮询。
- `core/engine.py` 小改：`get_adaptive_status` 优先调用后端新的 `get_adaptive_status`，保留对旧接口的兼容。

## 核心数据结构

### CSI 快照 `AdaptiveSnapshot`（dict）

- `M`、`N`、`htf`（H_TF 拷贝）、`htf_kind`（full / diag）、`noise_var`、`equalizer`
- `mod_order`、当前 `alpha` / `beta`、`frame_counter`、`snapshot_seq`、`context_key`
- `coarse_step`、`fine_step`、`max_order`、`rcond`

### 评估记录 `EvaluationRecord`（dict）

- `kind="eval"`、`seq`、`frame`
- 当前 `alpha` / `beta`、推荐 `rec_alpha` / `rec_beta`
- `ser_current`、`ser_best`、`ser_ofdm`、`gain_db`
- `action`（apply / skip / stable_pending / cooldown / gain_below）
- `stable_count`、`candidates`、`seconds`、`state`、`ts`

### 切换记录 `SwitchRecord`（dict）

- `kind="switch"`、`seq`、`frame`
- `from_alpha`、`from_beta`、`to_alpha`、`to_beta`
- `gain_db`、`reason`、`ts`

### 状态 `AdaptiveStatus`（`get_adaptive_status()` 返回）

- `enabled`、`auto_apply`、`state`、`last_error`、`ready`、`pending`
- `recommendation_seq`、`recommended_alpha` / `recommended_beta`
- `predicted_ser_current` / `predicted_ser_best` / `predicted_ser_ofdm`、`predicted_improvement_db`
- `stable_count`、`stable_required`、`candidate_count`、`search_seconds`
- `context_key`、`history_len`、当前配置字段

## 模块设计

### M1 仿真自适应内核（`FDIDMSimAdaptiveMixin` 内）

**职责：** 输入 CSI 快照，输出推荐 (α, β) 与预测 SER/增益。

**对外接口：**

- `_optimize_alpha_beta_snapshot(snapshot) -> dict`：移植硬件端粗搜+细搜实现（`_adaptive_prepare_base` / `_adaptive_evaluate_candidates` / 对角快速路径），返回推荐 α/β、预测 SER（当前/推荐/OFDM）、增益 dB、候选数、耗时、搜索模式。
- 自包含辅助：`_apply_gamma_axis`、SER 映射与 Q 函数等（不依赖硬件类）。

**依赖：** `self._gamma(n, eps)`（仿真后端已有）、`self._lock`（RLock，可重入）。

**说明：** htf 为全矩阵时走 full_coordinate 模式（粗步长下限 0.5），对角时走快速路径；K=M×N 超过 max_order 时跳过并记录错误状态。

### M2 闭环控制器（`FDIDMSimAdaptiveMixin` 内）

**职责：** 后台线程调度（threading.Event 唤醒）、按评估间隔取快照、上下文失效检测、稳定/增益/冷却决策、自动应用、历史记录。

**状态机：** disabled → waiting_channel → queued → optimizing → stable_pending → applied / cooldown → error。

**关键行为：**

- 应用动作在锁外调用已有 `set_indices(alpha, beta)`，避免锁序死锁；应用后记录切换事件并进入冷却。
- 历史存储为 `deque(maxlen=adaptive_history_limit)`，评估与切换事件按序追加。

### M3 控制/状态接口（mixin 对外）

- `start_adaptive_tuning(**cfg) -> bool`
- `stop_adaptive_tuning()`
- `request_adaptive_evaluation() -> bool`
- `update_adaptive_config(**cfg)`
- `get_adaptive_status() -> dict`
- `get_alpha_beta_adaptation_status() -> dict`（兼容别名）
- `get_adaptive_history(limit=None) -> list[dict]`
- 内部钩子 `_adaptive_note_frame_processed()`

**可配置字段：** `adaptive_enabled=False`、`adaptive_auto_apply=True`、`adaptive_interval_frames=8`、`adaptive_coarse_step=0.25`、`adaptive_fine_step=0.05`、`adaptive_stability_evals=2`、`adaptive_cooldown_frames=20`、`adaptive_min_improvement_db=0.5`、`adaptive_max_order=512`、`adaptive_rcond=1e-6`、`adaptive_history_limit=500`。

### M4 UI 自适应面板（`AdaptiveProcessWidget`，新文件 `waveform_sim/ui/fdidm_adaptive_widgets.py`）

**职责：** 控制区 + 双图 + 状态文本；只轮询与展示，不承载业务逻辑。

- 控制区：启用、自动应用、评估间隔/帧、粗步长、细步长、稳定次数、最小增益 dB、冷却帧、立即评估按钮、状态标签。
- 图 1 α/β 轨迹：实线 = 已应用值（阶梯），虚线 = 推荐值，圆点 = 切换标记。
- 图 2 预测 SER 对比：当前参数 SER 与推荐参数 SER 两条曲线（log 纵轴），与图 1 共用 x = 帧号。
- `refresh(status, history)`：仅当 `recommendation_seq` / 历史长度变化时重绘；数据全部来自后端历史，UI 不缓存业务状态。

### M5 引擎透传（`waveform_sim/core/engine.py` 小改）

- `get_adaptive_status` 优先调用后端 `get_adaptive_status`，保留 `get_alpha_beta_adaptation_status` 回退。
- `start_adaptive_tuning` / `stop_adaptive_tuning` 已有转发逻辑不动。

## 模块交互

1. 仿真循环：`_simulate_one_frame()` 末尾 → `_adaptive_note_frame_processed()` → 锁内拷贝 CSI/上下文；评估间隔到期或上下文变化时组装快照、seq 自增、唤醒 worker。
2. 后台 worker：`event.wait()` → 锁内取快照 → `_optimize_alpha_beta_snapshot` → 决策（稳定/增益/冷却）→ 记录 eval 事件 → 若满足条件：锁外 `set_indices` + 记录 switch 事件 + 置冷却。
3. UI：定时器（约 250ms）轮询 `get_adaptive_status()` / `get_adaptive_history()` → 刷新双图与状态文本；控件变更 → `start_adaptive_tuning` / `stop_adaptive_tuning` / `request_adaptive_evaluation` / `update_adaptive_config`。
4. 上下文变化：调制阶数、网格、均衡器、信道等更新时，mixin 检测 context_key 变化 → 重置推荐与稳定计数 → 下一帧重新快照。

## 文件组织

```text
waveform_sim/simulation/fdidm_adaptive.py        # 新建：FDIDMSimAdaptiveMixin（搜索内核 + 闭环 + 历史 + 状态）
waveform_sim/simulation/simple_fdidm_rx.py       # 修改：_LegacyFDIDMTransceiver 混入 mixin，_simulate_one_frame 末尾挂钩
waveform_sim/ui/fdidm_adaptive_widgets.py        # 新建：AdaptiveProcessWidget（控制区 + 双图 + 状态标签）
waveform_sim/ui/fdidm_tab.py                     # 修改：接入自适应面板、定时轮询、启停/参数透传
waveform_sim/core/engine.py                      # 小改：get_adaptive_status 优先调用新接口，保留旧接口回退
tests/test_fdidm_sim_adaptive.py                 # 新建：单元 + 集成 + UI 冒烟测试
markdown/33_fdidm_sim_adaptive_plan.md           # 本文档
```

## 技术决策

| 决策点 | 选择 | 理由 |
|------|------|------|
| 闭环位置 | 后端 mixin（`FDIDMSimAdaptiveMixin` 混入 `_LegacyFDIDMTransceiver`） | 方案一；与硬件端对称；兼容壳 `__getattr__` 自动透传，无需改壳 |
| 搜索内核 | 移植硬件端粗搜+细搜（含对角快速路径），仿真侧自包含实现 | spec F1 明确粗搜+细搜；与硬件行为一致；比逐候选 `rx@H@tx` 全矩阵更省 |
| 预测目标 | 论文 ZF/MMSE 理论 SER（Eq.40-47），噪声方差由 Eb/N0 换算，缺失时用硬件同款回退 | 预测式决策；与硬件端 `fdidm_adaptive.py` 一致 |
| 决策策略 | 稳定计数（容差=fine_step）+ 最小增益阈值 + 冷却 + 平局保持当前点 | 防抖；与硬件一致；满足 spec F2 |
| 线程模型 | 单后台线程 + threading.Event + RLock；应用时锁外调 `set_indices` | 不阻塞仿真（N1）；RLock 可重入 + 锁外调用避免死锁（N4） |
| 上下文失效 | context_key（调制/网格/均衡器/信道/SNR 等）变化即重置推荐与稳定计数 | 与硬件一致；满足 spec F3 |
| 历史存储 | `deque(maxlen=adaptive_history_limit)` 存 dict 事件 | 容量受限（N6）；UI 从历史派生曲线 |
| 可视化 | pyqtgraph 双图（α/β 轨迹+切换标记 / 预测 SER 对比）+ 状态文本，独立面板 widget | 用户选定的完整分析视图；面板独立模块便于维护（N5） |
| 默认状态 | `adaptive_enabled=False`，未启用时零行为变化 | 不改变现有仿真行为（N2） |
