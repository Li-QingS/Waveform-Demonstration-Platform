# FDIDM 硬件优势观测升级 Tasks

## 文件清单

| 操作 | 文件 | 职责 |
|------|------|------|
| 新建 | `waveform_sim/ui/hardware_advantage_observer.py` | 观测引擎：数据结构、分级、会话、配对、导出 |
| 修改 | `waveform_sim/ui/fdidm_hardware_test_tab.py` | 删旧对比代码；接引擎；结论面板 + 时间轴图 + 控制组 |
| 新建 | `tests/test_hardware_advantage_observer.py` | 引擎离线单测 |
| 改写 | `tests/test_fdidm_hardware_comparison.py` | 旧定时对比测试 → 观测面板离屏测试 |

## T1: 引擎数据结构与纯函数

**文件：** `waveform_sim/ui/hardware_advantage_observer.py`
**依赖：** 无
**步骤：**
1. 定义 `WindowSample`、`WindowAggregate`、`ObservationWindow`、`ToggleEvent`、`PairComparison` dataclass（字段见 plan.md 核心数据结构）。
2. 定义常量：`GRADE_TRUSTED_K = 100`、`GRADE_REFERENCE_K = 10`、`TRANSIENT_SEC = 1.0`、`MAX_ANOMALY_SAMPLE_RATIO = 0.2`。
3. 实现 `grade_from_k(k)`：k≥100 → "trusted"；10≤k<100 → "reference"；k<10 → "insufficient"。
4. 实现 `ser_improvement_db(before, after)`：两 SER 均有限且 >0 时返回 `10*log10(before/after)`，否则 NaN。
5. 实现 `_aggregate_samples(samples, symbols_per_frame, coded_bits_per_frame)`：transient 样本除外，累加 frames/ok/ser_k(=Σ ser×S×Δframes)/ser_n/evm 加权均值/α/β 均值，产出 `WindowAggregate` 并附 `grade` 与 `estimated_k` 标记（任一采样 Δframes>1 时 k 为估计口径）。

**验证：** `python -c "from waveform_sim.ui.hardware_advantage_observer import AdvantageObservationSession, grade_from_k; assert grade_from_k(100)=='trusted' and grade_from_k(10)=='reference' and grade_from_k(9)=='insufficient'"` 通过。

## T2: AdvantageObservationSession 会话逻辑

**文件：** `waveform_sim/ui/hardware_advantage_observer.py`
**依赖：** T1
**步骤：**
1. `start(enabled, context)`：清空 windows/events/配对缓存，记录起始帧号/overflow 计数，开第一个窗口。
2. `on_sample(status)`：从 status 取 frames_processed/frames_decode_ok/measured_ser/fec_bit_ber/evm_average_percent/alpha/beta/rx_overflow_count/tx_uncoded_bits_len/mod_order/channel_mode/tdl_* 字段；计算 Δframes、Δok、Δoverflow；overflow 增量 >0 或 measured_ser 非有限（失步帧）→ 本采样丢弃并计入窗口异常采样；`tx_uncoded_bits_len` 或 `mod_order` 变化 → 强制收尾当前窗口重开。
3. `on_toggle(enabled, alpha, beta)`：收尾当前窗口（跳过段首 TRANSIENT_SEC 并判异常占比），append `ToggleEvent`，开新窗口。
4. 窗口收尾规则：总帧数为 0 → 丢弃；异常采样占比 >20% → 标记 anomalies 并计入排除窗口；其余生成 aggregate 入 windows。
5. `latest_pair()`：取最近一个完整 off 窗口与最近一个完整 on 窗口（均在本次 start 之后）构造 `PairComparison`；比较 context 字段（tdl_model/doppler/ds/snr/seed 一致）与 anomalies，不一致时 `comparable=False` 并写 `comparability_note`。
6. `stop()`：收尾当前窗口，保留结果；`excluded_window_count()` 返回被排除窗口数。
7. `to_export_dict()`：`schema_version`、started_at、context、windows（含 aggregate/anomalies）、toggle_events、latest_pair、excluded_window_count。

**验证：** `python -c` 冒烟：构造模拟 status 序列（30 帧 SER=0.05 → toggle → 30 帧 SER=0.01），断言 `latest_pair()` 存在、`ser_improvement_db≈7dB`、两个窗口均非 excluded。

## T3: 引擎离线单测

**文件：** `tests/test_hardware_advantage_observer.py`
**依赖：** T2
**步骤：**
1. `_make_status(ser, frames, overflow=0, **overrides)` 辅助构造模拟 status。
2. 用例：正常配对与 dB 改善量正确；段首瞬态样本不入汇总；overflow 采样丢弃；异常占比超限整窗排除并计入 excluded；k 三档分级进 aggregate.grade；TDL 上下文不一致 → comparable=False；`to_export_dict()` 含 schema_version/事件/窗口且 JSON 可序列化；stop 后 latest_pair 仍可用。
3. 全部用例不依赖 PyQt/USRP。

**验证：** `python -m pytest tests/test_hardware_advantage_observer.py -q` 全绿。

## T4: 观测控制组与会话接线

**文件：** `waveform_sim/ui/fdidm_hardware_test_tab.py`
**依赖：** T2
**步骤：**
1. import 引擎；`__init__` 建 `self.observer = AdvantageObservationSession()`。
2. 右栏"自适应效果观测"组内新增按钮行：开始观测 / 结束观测 / 导出报告（导出在 T8 接线，先置灰）。
3. 开始观测：`observer.start(当前自适应开关态, TDL上下文字典)`，清空时间轴 deques 与旧事件标记，按钮态切换。
4. `_refresh_plots()`：取得 status 后调用 `observer.on_sample(status)`；同时把 ser/evm/α/β 追加进时间轴 deques（T6 用）。
5. `_on_adaptive_enable_changed`：调 `observer.on_toggle(...)`，移除旧 `_adaptive_observation_*` 临时列表操作（保留 EVM 图竖线标记或迁移到时间轴）。
6. `_on_stop_test_clicked` 与"结束观测"：调 `observer.stop()`；再次开始测试不清空已结束的观测结果，仅"开始观测"清空（F1）。

**验证：** `QT_QPA_PLATFORM=offscreen python -c` 构造 tab + FakeBackend 投喂 5 个 status，断言 `tab.observer.windows` 非空、按钮态正确。

## T5: 一屏结论面板

**文件：** `waveform_sim/ui/fdidm_hardware_test_tab.py`
**依赖：** T4
**步骤：**
1. 重排"自适应效果观测"组：保留细栏高度约束改为自适应；SER 前后大字（`.3g`）、EVM 前后大字、改善量 dB 大字（绿色正/红色负）。
2. 徽标：`grade_from_k(after.ser_k)` → 绿"可信 n≥100" / 灰"参考 k=n" / 黄"样本不足"；旁注 `≈` 前缀当 `estimated_k`。
3. 次行：剔除窗口数、可比性注记（`comparable=False` 时黄字显示 note）、样本帧数。
4. `_update_adaptive_observation_display()` 改为从 `observer.latest_pair()` 读数；无配对时显示引导文案"开始观测后，手动开关自适应积累窗口"。

**验证：** 离屏投喂开/关两段模拟 status 后断言改善量文本包含 "dB" 且徽标文本正确。

## T6: 因果时间轴图

**文件：** `waveform_sim/ui/fdidm_hardware_test_tab.py`
**依赖：** T4
**步骤：**
1. 右栏新增 `pg.PlotWidget`（标题"优势观测时间轴"），复用 `_init_plot_style`；x 轴为观测会话内秒数。
2. 曲线：实测 SER（对数 y 轴，MATLAB_BLUE）、EVM%（信道代理，MATLAB_ORANGE）、α 与 β 轨迹（MATLAB_PURPLE 实/虚线）；图例可关。
3. 事件竖线：toggle 用蓝/橙 `InfiniteLine`；`adaptive_recommendation_seq` 变化沿打参数应用标记（三角散点）。
4. 异常区间阴影：`LinearRegionItem` 半透明红，覆盖被丢弃采样聚集区。
5. 仅观测会话内记录；"开始观测"清空。

**验证：** 离屏投喂含一次 toggle 的序列，断言时间轴 `listDataItems()` 曲线数 ≥3 且事件线 item 数随 toggle 增加。

## T7: 自适应状态行

**文件：** `waveform_sim/ui/fdidm_hardware_test_tab.py`
**依赖：** T4
**步骤：**
1. 时间轴图下方一行 QLabel：`α/β=x.x/x.x | 搜索:<adaptive_alpha_beta_state> | 验证:<adaptive_validation_state> | α可观测:是/否 | β可观测:是/否`。
2. 每次刷新从 status 更新；`adaptive_validation_reason` 非空时追加显示（截断 60 字符）。

**验证：** 离屏投喂带 `adaptive_alpha_observable=False` 的 status，断言文本含"否"。

## T8: 导出观测报告

**文件：** `waveform_sim/ui/fdidm_hardware_test_tab.py`
**依赖：** T4
**步骤：**
1. 解除导出按钮置灰；点击弹 `QFileDialog`，默认 `log/fdidm_advantage_YYYYmmdd_HHMMSS.json`。
2. 写入 `observer.to_export_dict()`（`ensure_ascii=False, indent=2`）；无观测数据时提示并返回。
3. 导出成功写 UI 日志一行（复用 `_log`）。

**验证：** 离屏测试调用导出处理函数到 tmp 路径，断言文件存在且含 `schema_version` 与 `windows`。

## T9: 删除旧对比代码

**文件：** `waveform_sim/ui/fdidm_hardware_test_tab.py`
**依赖：** T4–T8
**步骤：**
1. 删除 `ComparisonPhaseRecord` / `ComparisonSummary` / `ComparisonSession` 三个 dataclass 及 `comparison_session`、`_comparison_tick_busy`、`_comparison_evm_x` 属性。
2. 删除 `_start_comparison_demo` / `_stop_comparison_demo` / `_enter_comparison_phase` / `_finalize_comparison_phase` / `_comparison_tick` / `_sample_comparison_metrics` / `_comparison_summary_text` 及对比历史绘制分支（约 1118、1149 行处）。
3. 删除"单 USRP 优势对比"控件组、`btn_start_comparison` / `btn_stop_comparison` 及信号连接、`_on_start_test_clicked` / `_on_stop_test_clicked` 中对 comparison_session 的引用。
4. 删除 `_adaptive_observation_state/_samples/_records` 等旧临时观测属性与方法（已被引擎替代）；`get_adaptive_observation_records()` 若测试不再用则一并删除。
5. 全仓 grep 确认无残留符号。

**验证：** `grep -rn "ComparisonSession\|comparison_session\|_start_comparison_demo\|btn_start_comparison" waveform_sim/ tests/` 零命中；`QT_QPA_PLATFORM=offscreen python -c "from waveform_sim.ui.fdidm_hardware_test_tab import FDIDMHardwareTestTab"` 导入成功。

## T10: 观测面板离屏测试（改写旧测试文件）

**文件：** `tests/test_fdidm_hardware_comparison.py`
**依赖：** T9
**步骤：**
1. 保留 FakeBackend 模式，删除三个旧对比用例。
2. 新用例：观测按钮默认态；投喂 off/on 两段 status 后结论面板含 dB 改善量与可信徽标；toggle 事件竖线计数增加；导出写文件成功；停止测试不清空已结束观测结果。

**验证：** `python -m pytest tests/test_fdidm_hardware_comparison.py -q` 全绿。

## T11: 全量回归与残留扫描

**文件：** 无新改动（验证任务）
**依赖：** T3、T10
**步骤：**
1. `python -m pytest -q` 全量。
2. `grep -rn "comparison_session\|_adaptive_observation_state" waveform_sim/ tests/` 零命中。
3. 对照 plan.md「spec 覆盖对照」逐行核对 F1–F8 有落点。

**验证：** pytest 全绿（基线 88+ 用例 + 新增用例）；两个 grep 零命中。

## 执行顺序

```
T1 → T2 → T3 ──────────────────────────┐
            ↘ T4 → T5 → T7 → T8 → T9 → T10 → T11
                  ↘ T6 ↗
```

T5/T6/T7/T8 相互独立，可在 T4 后任意顺序执行；T9 必须在新面板全部接线后做；T10 依赖 T9（旧符号已删）。
