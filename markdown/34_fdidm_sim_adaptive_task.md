# FDIDM 仿真侧自适应 Tasks

> 本文档为软波谱 Spec 驱动开发流程的阶段三产物，基于已批准的 `32_fdidm_sim_adaptive_spec.md` 与 `33_fdidm_sim_adaptive_plan.md`，未获用户批准前不进入实现阶段。

## 文件清单

| 操作 | 文件 | 职责 |
|------|------|------|
| 新建 | `waveform_sim/simulation/fdidm_adaptive.py` | `FDIDMSimAdaptiveMixin`：搜索内核 + 闭环控制器 + 历史 + 状态 |
| 修改 | `waveform_sim/simulation/simple_fdidm_rx.py` | `_LegacyFDIDMTransceiver` 混入 mixin，`_simulate_one_frame` 末尾挂钩 |
| 修改 | `waveform_sim/core/engine.py` | `get_adaptive_status` 优先调用新接口，保留旧接口回退 |
| 新建 | `waveform_sim/ui/fdidm_adaptive_widgets.py` | `AdaptiveProcessWidget`：控制区 + 双图 + 状态标签 |
| 修改 | `waveform_sim/ui/fdidm_tab.py` | 接入自适应面板、定时轮询、启停/参数透传、销毁清理 |
| 新建 | `tests/test_fdidm_sim_adaptive.py` | 单元 + 集成 + UI 冒烟测试 |
| 新建 | `markdown/34_fdidm_sim_adaptive_task.md` | 本文档 |

## T1: 创建 mixin 基础骨架

**文件：** `waveform_sim/simulation/fdidm_adaptive.py`
**依赖：** 无
**步骤：**
1. 定义 `FDIDMSimAdaptiveMixin` 类。
2. 定义默认配置常量与字段名（`adaptive_enabled=False`、`adaptive_auto_apply=True`、`adaptive_interval_frames=8`、`adaptive_coarse_step=0.25`、`adaptive_fine_step=0.05`、`adaptive_stability_evals=2`、`adaptive_cooldown_frames=20`、`adaptive_min_improvement_db=0.5`、`adaptive_max_order=512`、`adaptive_rcond=1e-6`、`adaptive_history_limit=500`）。
3. 实现 `__init_adaptive_state_locked()`：初始化 `_adaptive_lock`（RLock）、`_adaptive_thread`、`_adaptive_event`、`_adaptive_stop_event`、`_adaptive_snapshot_seq`、`_adaptive_last_snapshot`、`_adaptive_recommendation`、`_adaptive_stable_key/count`、`_adaptive_state`、`_adaptive_history`（deque）等状态字段。
4. 实现配置读取辅助 `_adaptive_cfg(key, default)`（从 `self` 读取 `adaptive_*` 属性或配置 dict）。

**验证：** `python -c "from waveform_sim.simulation.fdidm_adaptive import FDIDMSimAdaptiveMixin"` 通过；`compileall` 无语法错误。

## T2: 搜索内核（纯函数部分）

**文件：** `waveform_sim/simulation/fdidm_adaptive.py`
**依赖：** T1
**步骤：**
1. 实现 `_adaptive_qam_order`、`_adaptive_qfunc`、`_adaptive_ser_from_symbol_nsr`、`_adaptive_grid_values`、`_adaptive_canonical_index`（对照硬件端同名实现）。
2. 实现 `_apply_gamma_axis(cube, eps, axis)`：用 `self._gamma` 构造对角阵作用到指定轴（参考硬件端）。
3. 补充模块 docstring，注明移植来源与论文公式编号。

**验证：** 脚本断言：`_adaptive_qfunc([0])≈0.5`；SER 输出在 [0,1]；`_adaptive_grid_values(0.25)` 含 0/1/2；`_adaptive_canonical_index` 归一化正确。

## T3: 搜索内核（快照优化）

**文件：** `waveform_sim/simulation/fdidm_adaptive.py`
**依赖：** T2
**步骤：**
1. 实现 `_adaptive_prepare_base(snapshot)`：对角/全矩阵两条路径，噪声方差由快照提供或回退（signal_power×1e-3）。
2. 实现 `_adaptive_evaluate_candidates(prepared, candidates, M, N, mod_order)`。
3. 实现 `_optimize_alpha_beta_snapshot(snapshot)`：粗搜 + 细搜 + 平局保持当前点 + 整数点优先策略，返回 plan 第 2 段字段（推荐 α/β、ser_current/best/ofdm、gain_db、candidates、seconds、search_mode 等）；K=M×N 超 max_order 抛 ValueError。

**验证：** 用带 `_gamma` 的最小 stub（或真实后端实例，M=N=4）跑对角与全矩阵快照：返回字段齐全、推荐值在 [0,2]、SER 有限、candidates≥1、search_seconds≥0。

## T4: 上下文键与快照调度

**文件：** `waveform_sim/simulation/fdidm_adaptive.py`
**依赖：** T3
**步骤：**
1. 实现 `_adaptive_context_key()`：由 M、N、mod_order、decoder/equalizer、channel_model、velocity、ebn0_db、snr_definition、channel_seed、dynamic_channel、channel_coherence_frames、tf 参数等组成（对照硬件端 context key 思路）。
2. 实现 `_adaptive_invalidate(reason)`：清空推荐/稳定计数/历史相关缓存（保留历史事件），状态回 waiting_channel 或 disabled。
3. 实现 `_adaptive_note_frame_processed()`：锁内取 CSI（`self._H_tf` 拷贝）与上下文；间隔到期或上下文变化时组装 `AdaptiveSnapshot`、seq 自增、置 pending 并唤醒事件；上下文变化时先失效。

**验证：** 手动驱动 `step()` 循环：未启用时不产生快照；启用后间隔帧数到点出现 pending/queued 且 seq 递增；修改 mod_order 后状态回 waiting、推荐清空。

## T5: worker 与决策策略

**文件：** `waveform_sim/simulation/fdidm_adaptive.py`
**依赖：** T4
**步骤：**
1. 实现 `_adaptive_worker()`：`event.wait()` → 锁内取快照（校验 seq）→ 锁外跑 `_optimize_alpha_beta_snapshot` → 锁内校验 seq 后做决策。
2. 实现决策：推荐与上次一致（差值 ≤ fine_step）则 stable_count+1，否则重置；`stable_count ≥ stability_evals` 且 `gain_db ≥ min_improvement_db` 且不在冷却期且 auto_apply 开启 → 应用；否则记录对应 action（stable_pending / gain_below / cooldown / skip）。
3. 应用动作：锁外调用 `self.set_indices(alpha, beta)`；应用后记录 switch 事件、置 `_adaptive_cooldown_until_frame = frame + cooldown_frames`、状态 applied。
4. 每次评估后追加 `EvaluationRecord` 到历史 deque（超容量自动丢弃）。

**验证：** 固定快照脚本：连续相同推荐在第 stability_evals 次才 apply；增益不足时不 apply；apply 后冷却期内不再 apply；auto_apply=False 时参数不变但推荐可见。

## T6: 公开接口与历史查询

**文件：** `waveform_sim/simulation/fdidm_adaptive.py`
**依赖：** T5
**步骤：**
1. 实现 `start_adaptive_tuning(**cfg)`：设置配置、初始化状态、启动 worker 线程（幂等）。
2. 实现 `stop_adaptive_tuning()`：置停止事件、join 线程、状态 disabled。
3. 实现 `request_adaptive_evaluation()`：无最近 CSI 时置 `force_next`；有则立即组装快照并唤醒。
4. 实现 `update_adaptive_config(**cfg)`：运行时更新 `adaptive_*` 字段。
5. 实现 `get_adaptive_status()`、`get_alpha_beta_adaptation_status()`（别名）、`get_adaptive_history(limit=None)`。

**验证：** 脚本调用全部接口：返回结构符合 plan 第 2 段；start/stop 可重复；history 长度 ≤ limit；别名与主接口一致。

## T7: 仿真后端混入与挂钩

**文件：** `waveform_sim/simulation/simple_fdidm_rx.py`
**依赖：** T6
**步骤：**
1. `_LegacyFDIDMTransceiver` 混入 `FDIDMSimAdaptiveMixin`。
2. `__init__` 末尾调用 `self.__init_adaptive_state_locked()`（在 `self._lock` 已创建之后）。
3. `_simulate_one_frame()` 末尾调用 `self._adaptive_note_frame_processed()`（保持 RLock 可重入语义，不改变原返回值）。

**验证：** 真实后端 `step()` 20 帧无异常；未启用自适应时 `get_last_metrics()` 的帧数/指标与启用前一致（回归）；启用后各接口可用。

## T8: 引擎透传

**文件：** `waveform_sim/core/engine.py`
**依赖：** T7
**步骤：**
1. `get_adaptive_status` 优先调用 `backend.get_adaptive_status()`，失败/缺失时回退 `get_alpha_beta_adaptation_status()`，再回退 `{"active": False}`。
2. 补充 `get_adaptive_history` 透传（后端有则调用，否则返回空列表）。

**验证：** 构造 FDIDM `LinkSimulator`，`get_adaptive_status()` 返回包含 `enabled` 字段的 dict；未启用时 `history` 为空。

## T9: UI 自适应面板控件

**文件：** `waveform_sim/ui/fdidm_adaptive_widgets.py`（新建）
**依赖：** 无（纯 UI，可并行于 T2~T8）
**步骤：**
1. 定义 `AdaptiveProcessWidget(QGroupBox)`：控制区（启用/自动应用 checkbox、间隔/粗步长/细步长/稳定次数/最小增益/冷却帧 spin、立即评估按钮、状态标签）。
2. 双 pyqtgraph 图：图 1 α/β 轨迹（实线=已应用阶梯、虚线=推荐、散点=切换标记）；图 2 预测 SER 对比（log 纵轴，当前 vs 推荐）。
3. 实现 `refresh(status, history)`：仅当 `recommendation_seq` / 历史长度变化时重绘；提供 `collect_config() -> dict` 与信号/回调供外部透传。
4. 对外暴露控件变更信号（可 `valueChanged`/`stateChanged` 直接连到外部）。

**验证：** `QT_QPA_PLATFORM=offscreen` 下实例化不抛异常；用假 status/history 调 `refresh` 不抛异常且图曲线数量正确。

## T10: FDIDM 页接入

**文件：** `waveform_sim/ui/fdidm_tab.py`
**依赖：** T9
**步骤：**
1. 在 `plot_panel` 布局下方挂载 `AdaptiveProcessWidget`。
2. 后端创建/更新时应用面板配置（`start_adaptive_tuning(collect_config())` 或 `update_adaptive_config`）；页面启动/停止同步自适应启停。
3. 新增/复用定时器（约 250ms）轮询 `tb.get_adaptive_status()` / `tb.get_adaptive_history()` → `adaptive_widget.refresh(...)`；仅仿真运行且自适应启用时轮询。
4. 控件变更即时透传；页面销毁时 `stop_adaptive_tuning()` 并清理。

**验证：** `QT_QPA_PLATFORM=offscreen` 冒烟：创建 tab、启动仿真、启用自适应、多轮轮询无异常；关闭页面时线程退出、无残留。

## T11: 测试文件与回归

**文件：** `tests/test_fdidm_sim_adaptive.py`（新建）
**依赖：** T7、T8、T10
**步骤：**
1. 单元：搜索内核（T2/T3 断言场景）、决策策略（稳定/增益/冷却/平局/auto_apply）、历史容量、上下文失效。
2. 集成：真实后端启用自适应 `step()` 若干帧，动态信道下出现合理 apply 或 skip；`LinkSimulator` 透传正确。
3. 冒烟：offscreen 下 `FDIDMTab` 含自适应面板并完成一轮轮询。
4. 视情况补充 `tests/test_ui_regressions.py` 不冲突的回归断言。

**验证：** `python -m pytest tests/test_fdidm_sim_adaptive.py -q` 全过；`python -m pytest -q` 全量通过；`python -m compileall waveform_sim` 通过。

## 执行顺序

```text
T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9 → T10 → T11
```

（T9 与 T2~T8 无依赖，可并行；T11 依赖全部实现。）
