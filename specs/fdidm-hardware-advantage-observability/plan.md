# FDIDM 硬件优势观测升级 Plan

## 架构概览

```
backend.get_status()（每 100ms，已有，不改）
        │
        ▼
┌─────────────────────────────────────────────┐
│  hardware_advantage_observer.py（新建，纯 Python 无 Qt）│
│  AdvantageObservationSession 观测引擎：                  │
│   采样 → 窗口汇总 → 错误计数/分级 → 配对比较 → 导出字典    │
└─────────────────────────────────────────────┘
        │ latest_pair / windows / events
        ▼
fdidm_hardware_test_tab.py（修改）
  ├─ 观测控制组：开始观测 / 结束观测 / 导出报告（F1/F8）
  ├─ 一屏结论面板：替换现有三个文本标签（F5）
  ├─ 因果时间轴图：新增 pg.PlotWidget（F4）
  └─ 自适应状态行：α/β、搜索状态、轴可观测性（F6）
```

核心原则：测量方法学全部下沉到纯 Python 引擎，UI 只投喂数据和呈现——这是 N4（离线可测）的关键，也避免 1793 行的 tab 再膨胀。后端 `fdidm_hardtest.py` 零修改（所有需要的字段 `get_status()` 已具备：overflow 计数、自适应状态机、TDL 上下文、轴可观测性标志 `adaptive_alpha/beta_observable`）。

## 核心数据结构

### WindowSample
每次 100ms 刷新投喂的一个采样点：`t`（monotonic）、`frames_delta` / `ok_delta`（距上采样点新增处理帧数 / CRC 成功帧数）、`ser`（逐帧实测 SER，后端精确值）、`fec_ber`、`evm_percent`、`alpha` / `beta`、`overflow_delta`、`transient`（段首瞬态标记，不入汇总）。

### WindowAggregate
一个观测段的汇总（F3 的载体）：`frames`、`crc_ok_ratio`、`ser` / `ser_k` / `ser_n`（错误符号数 / 总符号数）、`fec_ber` / `fec_k` / `fec_n`、`evm_mean`、`alpha_mean` / `beta_mean`、`grade`（"trusted" / "reference" / "insufficient"，对应 k≥100 / 10–99 / <10）。

### ObservationWindow
一个连续段（段内自适应开关状态不变）：`mode`（"adaptive_on" / "adaptive_off"）、`start_t` / `end_t`、`context`（channel_mode / tdl_model / tdl_doppler_hz / tdl_ds_ns / tdl_snr_db / tdl_seed）、`anomalies`（["rx_overflow", "sync_loss", ...]）、`aggregate`。

### ToggleEvent
F1 的开关事件：`t`、`enabled`、`alpha`、`beta`。

### PairComparison
F2+F5 的结论载体：`before` / `after`（WindowAggregate）、`ser_improvement_db`（10·log10(SER_before/SER_after)）、`evm_delta_pp`、`crc_delta`、`comparable` / `comparability_note`（上下文不可比时标注）。

### AdvantageObservationSession（引擎接口）
```python
class AdvantageObservationSession:
    def start(self, enabled: bool, context: dict) -> None        # 清空旧结果（F1）
    def stop(self) -> None                                       # 收尾当前窗口，结果保留
    def on_toggle(self, enabled: bool, alpha: float, beta: float) -> None
    def on_sample(self, status: dict) -> None                    # 自行判 overflow 增量/失步/帧号
    def latest_pair(self) -> PairComparison | None               # 最近一次配对
    def excluded_window_count(self) -> int                       # 被剔除窗口数（F3 显示用）
    def to_export_dict(self) -> dict                             # F8：JSON 可序列化
```

## 模块设计

### 模块 A：ui/hardware_advantage_observer.py（新建，测量引擎）
**职责**：窗口汇总、错误计数与分级、overflow/失步剔除、段切换、配对比较、导出字典。全部纯 Python + numpy，不 import PyQt。
**对外接口**：`AdvantageObservationSession`、`PairComparison`、`grade_from_k(k)`。
**依赖**：仅标准库 + numpy。
**关键规则**：
- 段首 1.0s 样本标 `transient=True` 不入汇总（切换瞬态丢弃，与后端 live update 的瞬态丢弃互补）。
- 任一采样点 `overflow_delta>0` 或同步失锁 → 该采样点丢弃；一个窗口内被丢弃采样占比 >20% → 整窗标记异常并从配对中排除，计入 `excluded_window_count`。
- 每帧符号数 S = `tx_uncoded_bits_len / log2(mod_order)`（从 status 取）；配置变化时窗口强制收尾重开。

### 模块 B：fdidm_hardware_test_tab.py（修改，呈现层）
**职责**：观测控制（开始/结束/导出按钮）、结论面板、因果时间轴图、自适应状态行；把 `backend.get_status()` 每 100ms 投喂给引擎。
**改动点**：
1. "自适应效果观测"细栏重建为结论面板：SER/EVM 前后对比大字 + dB 改善量 + 可信度徽标（绿"可信" / 灰"参考" / 黄"样本不足"）+ 剔除窗口数 + 可比性注记（F5/F3）。
2. 新增因果时间轴 `pg.PlotWidget`：三组曲线（实测 SER 对数轴 / EVM% 信道代理 / α、β 轨迹双线）+ 开关事件竖线 + 参数应用标记（`adaptive_recommendation_seq` 变化沿）+ 异常剔除区间阴影（`LinearRegionItem`）（F4）。
3. 自适应状态行：`adaptive_alpha_beta_state`、`adaptive_validation_state`、`adaptive_alpha/beta_observable`、最近应用/回滚原因（F6）。
4. 观测控制组：开始/结束/导出三按钮；开始 = `session.start()` 清空旧结果并冻结时间轴旧数据（F1/F8）。
5. 删除：`ComparisonPhaseRecord` / `ComparisonSummary` / `ComparisonSession`、`_start/_stop_comparison_demo`、`_enter/_finalize_comparison_phase`、隐藏的"单 USRP 优势对比"控件组、`btn_start_comparison/btn_stop_comparison`、旧 EVM 对比历史绘制、`_adaptive_observation_*` 临时列表（由引擎替代）（F7）。

### 模块 C：tests/test_hardware_advantage_observer.py（新建）+ tests/test_fdidm_hardware_comparison.py（改写）
**职责**：离线引擎单测（窗口汇总/分级/剔除/配对/导出）+ 离屏 UI 测试（面板创建、投喂模拟 status 序列、验证结论文本）。

## 模块交互

```
开始观测点击 ──→ session.start(当前开关态, TDL上下文)
刷新定时器100ms ──→ status = backend.get_status()
                      ├─→ session.on_sample(status)          # 引擎建窗/判异常/累加k,n
                      ├─→ 时间轴 deques 追加 → 时间轴图增量刷新
                      └─→ session.latest_pair() → 结论面板刷新
开关自适应 ──→ session.on_toggle(...)（引擎收尾前窗、开新窗）──→ 时间轴打事件竖线
α/β 应用 ──→ adaptive_recommendation_seq 变化 ──→ 时间轴打应用标记
结束观测/停止测试 ──→ session.stop()（结果保留可查看）
导出点击 ──→ QFileDialog（默认 log/ 目录）──→ session.to_export_dict() 写 JSON
```

## 文件组织

```
waveform_sim/ui/hardware_advantage_observer.py   新建 — 观测引擎（数据结构+会话+分级）
waveform_sim/ui/fdidm_hardware_test_tab.py       修改 — 删旧对比代码、接引擎、结论面板+时间轴图+控制组
tests/test_hardware_advantage_observer.py        新建 — 引擎离线单测
tests/test_fdidm_hardware_comparison.py          改写 — 旧对比测试替换为观测面板离屏测试
```

## 技术决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 引擎位置 | 独立纯 Python 模块，不进 UI 文件 | N4 离线可测；tab 已 1793 行不宜再长 |
| 错误计数 k 来源 | UI 侧累加：Σ(逐帧SER × 每帧符号数 × Δframes)；Δframes>1 的采样按帧数加权，整窗 k 标"≈" | 后端 `measured_ser` 是逐帧精确值，累加即近似精确；后端零修改（不动接收机） |
| SER 改善量口径 | dB 比值 `10·log10(SER_before/SER_after)`；EVM 用 pp 差 | 专家口径一致；旧代码"SER 绝对差"无意义（SER 是量纲为 1 的概率） |
| 分级阈值 | k≥100 可信 / 10–99 参考 / <10 样本不足 | MATLAB ≥100 错误建议，与 UPGRADE_PLAN 问题 2 同源 |
| 信道代理量 | EVM 平均%（实测）为主曲线标签 | status 无实测每帧 SNR；`tdl_snr_db` 是注入设定值（常数）、`adaptive_predicted_snr_db` 是预测值，都不能当实测代理 |
| 时间轴载体 | tab 内新 `pg.PlotWidget`，复用 `_init_plot_style` | 与现有绘图风格统一；不新增页签（spec 边界） |
| 旧对比代码 | 连同其专用测试一并删除，不做兼容 | F7 全仓无残留；旧机制已被评审机制否决 |
| 导出格式 | JSON（含 `schema_version` 字段），默认存 `log/` | 机器可读可复核；与现有日志导出同路径惯例 |

## spec 覆盖对照

| spec 需求 | plan 归属 |
|-----------|-----------|
| F1 会话管理 | 引擎 `start/stop/on_toggle` + tab 观测控制组 |
| F2 配对统计 | 引擎窗口/`PairComparison` + 上下文可比性 |
| F3 分级/剔除 | 引擎 `WindowAggregate.grade`、`excluded_window_count` + 结论面板徽标 |
| F4 时间轴图 | tab 因果时间轴 `pg.PlotWidget` |
| F5 结论面板 | tab 结论面板 + 引擎 `latest_pair` |
| F6 状态可见 | tab 自适应状态行（消费现有 status 字段） |
| F7 清理旧代码 | tab 删除清单 + 测试改写 |
| F8 导出 | 引擎 `to_export_dict` + tab 导出按钮 |
