# FDIDM 硬件页面板重排 Tasks（2026-09-07）

## 文件清单
| 操作 | 文件 | 职责 |
|------|------|------|
| 修改 | `waveform_sim/ui/fdidm_hardware_test_tab.py` | 右栏重构、样式补齐、左栏放宽 |

## T1 右栏重构
**文件：** `waveform_sim/ui/fdidm_hardware_test_tab.py`（`_create_right_panel`）
**步骤：** 结论横条收为三行紧凑（按钮行/大数字行/徽标注记行，max ~112px）；删 grid 与 `_PlotGridCell` 用法；建 `plot_tabs` 四页；时间轴左右并排（perf 3 : ab 2，保留 xLink，max ~195px）；文本区 max ~120px。
**验证：** 离屏构造成功；6 个观测 UI 测试不回归。

## T2 时间轴样式修复
**文件：** 同上（`_init_plot_style`）
**步骤：** timeline_perf_plot / timeline_ab_plot 加入样式循环。
**验证：** 代码检查 + 截图非黑底。

## T3 左栏放宽
**文件：** 同上（`_create_controls_panel`、`_compact_combo`、`_dspin`、`_spin`）
**步骤：** scroll 360/470；combo 260；spin/行编辑 140；spacing 6、边距 8。
**验证：** 离屏断言宽度属性；截图无换行。

## T4 渲染验收 + 回归
**步骤：** 1400×800 与 1400×900 离屏渲染 PNG 检查；全量 pytest 对照基线。
**验证：** AC1–AC5 截图证据 + pytest 与基线一致。

## 执行顺序
T1 → T2 → T3 → T4
