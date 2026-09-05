# FDIDM 硬件优势对比演示 Plan

## 架构概览

在现有 `FDIDMHardwareTestTab` 内增加一个轻量的“单 USRP 对比会话”状态机。
状态机只负责阶段编排和统计，不创建新的硬件后端。固定基线和自适应阶段都调用
现有 backend `configure()`/live αβ 更新路径，实时指标继续由原有 `get_status()`、
`get_decode_stats()` 和刷新定时器提供。

数据流如下：

```text
对比控件
  -> ComparisonSession 状态机
  -> 现有 FDIDM backend live configure
  -> get_status()/get_decode_stats()
  -> 阶段快照与计数差值
  -> 模式汇总 + 对比曲线 + 结论文本
```

## 核心数据结构

### ComparisonSession

- `active`: 是否正在对比
- `phase`: `idle`、`baseline`、`adaptive`、`complete`
- `phase_index`: 当前阶段序号
- `total_phases`: 总阶段数（轮数 × 2）
- `phase_started_at`: 单调时钟起点
- `phase_duration_sec`: 单阶段持续时间
- `baseline_alpha` / `baseline_beta`: 固定基线参数
- `repeat_count`: 交替轮数
- `phase_start_frames` / `phase_start_ok`: 阶段开始时的累计计数
- `phase_samples`: 当前阶段的瞬时指标样本
- `records`: 已完成阶段记录
- `history`: `baseline` / `adaptive` 两条时间序列

### PhaseRecord

- `mode`: `baseline` 或 `adaptive`
- `round_index`: 轮次
- `frames_processed`: 阶段内处理帧数
- `frames_decode_ok`: 阶段内 CRC 通过帧数
- `crc_success_ratio`: `frames_decode_ok / frames_processed`
- `match_ratio`: 阶段内文本匹配率均值
- `measured_ser`: 阶段内实测 SER 均值
- `fec_bit_ber`: 阶段内 FEC BER 均值
- `evm_average_percent`: 阶段内平均 EVM
- `valid_sample_count`: 有效指标采样数

### ComparisonSummary

按 `baseline` 和 `adaptive` 分别聚合 PhaseRecord，并计算：

- 各指标均值
- 自适应减基线或基线减自适应的改善量
- 有效阶段数与总帧数
- `conclusion_available` 标志

## 模块设计

### FDIDMHardwareTestTab 对比控制

职责：提供基线 α/β、阶段时长、轮数和开始/停止按钮，显示阶段状态和结论。

接口：

- `_start_comparison_demo()`
- `_stop_comparison_demo()`
- `_enter_comparison_phase(mode)`
- `_comparison_tick()`
- `_sample_comparison_metrics(status, stats)`
- `_finalize_comparison_phase()`
- `_comparison_summary_text()`

### 阶段配置适配

职责：在不重启 UHD 的前提下切换固定基线和自适应配置。

策略：

- 基线阶段设置指定 α/β，并关闭 `adaptive_alpha_beta_enable`。
- 自适应阶段设置同一基线 α/β 作为起点，并开启现有自适应控制器。
- 使用现有 `_configure_backend()`；若后端按既有规则拒绝 live 更新，则沿用原有
  受控重启错误处理，不新增特殊硬件路径。

### 指标采集与聚合

职责：将 backend 累计计数转换为阶段内计数，并对瞬时指标做有限样本均值。

规则：

- 帧数使用阶段结束计数减阶段开始计数。
- CRC 成功率使用阶段内 `frames_decode_ok / frames_processed`。
- SER、BER、EVM、文本匹配率只聚合有限值。
- 没有有效样本时返回 `None`，UI 显示不可用提示。

### 对比绘图

复用现有 EVM 图：普通运行显示当前 EVM；对比运行显示基线与自适应两条曲线。
曲线采用不同颜色，横轴为对比会话时间，阶段切换位置使用竖直标记或断点记录。
频谱、星座和 α/β 性能面保持原有用途。

## 模块交互

1. 用户点击“开始对比演示”。
2. 若 backend 未运行，复用现有连接/启动流程。
3. 初始化 session，保存当前设置，清空旧记录。
4. 进入 baseline 阶段，记录计数基线。
5. Qt 刷新周期采集瞬时指标并更新曲线。
6. 阶段时间到达后生成 PhaseRecord，进入 adaptive 阶段。
7. 完成指定轮数后生成 ComparisonSummary，保留结果并进入 complete。
8. 用户点击停止时结束当前阶段并保留已完成记录。

## 文件组织

```text
specs/fdidm-hardware-advantage-demo/
├── spec.md
├── plan.md
├── task.md
└── checklist.md
waveform_sim/ui/fdidm_hardware_test_tab.py  — 对比状态机、控件、绘图与结论
tests/test_fdidm_hardware_comparison.py     — 离屏状态机与汇总测试
```

## 技术决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 硬件资源 | 单 USRP 时分复用 | 符合现有实验条件，避免第二套收发机 |
| 基线类型 | 固定 α/β FDIDM | 不扩展废弃多波形页，且能直接隔离自适应收益 |
| 切换方式 | 现有 live configure | 保持连续收发，避免对比被重启空窗污染 |
| 主要结论指标 | CRC、实测 SER、EVM、文本匹配率 | 既有后端已提供，观众可直接理解 |
| 统计方式 | 阶段内计数差值 + 有限值均值 | 避免累计计数误读和 NaN 污染 |
| 图表位置 | 复用 EVM 图显示双曲线 | 控制改动小，保留其他诊断图 |
# 方向修订（2026-09-03）

实现重点改为连续真实链路上的手动自适应观测：复用现有开关，按切换事件分段统计，不再依赖定时 baseline/adaptive 状态机。
