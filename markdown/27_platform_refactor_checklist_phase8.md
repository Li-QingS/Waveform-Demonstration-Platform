# 平台工程化重构 Checklist（阶段 8：UI 收敛）

> 范围：四个大 UI 文件提取自包含辅助类/静态助手到 `ui/` 子模块。
> 每一项通过运行命令或观察行为验证；真机交互留实验室。

## 实现完整性

- [ ] `ui/fdidm_plot_widgets.py` 提供 `_PlotGridCell`、`_AlphaBetaSurfaceCanvas`（验证：`python -c` 导入）
- [ ] `ui/compare_workers.py` 提供 `_WaveformRunner`、`_ScanWorker`（验证：`python -c` 导入）
- [ ] `ui/ui_utils.py` 提供 `safe_float` / `format_metric` / `compute_spectrum` / `has_signal` / `extract_samp_rate`（验证：`python -c` 导入）
- [ ] `ui/fdidm_utils.py` 提供 `_CurveSpec`、`_alpha_ser_floor`、`_merged_curve_specs`、`_copy_kwargs_with`（验证：`python -c` 导入）
- [ ] 四个原文件不再包含被搬走的定义（验证：`Select-String` 类名/方法名无命中）

## 行为保持

- [ ] 四个页签可 offscreen 直接构造（验证：分别 `import` 后 `QT_QPA_PLATFORM=offscreen` 构造 `FDIDMHardwareTestTab` / `WaveformCompareTab` / `HardwareTestTab` / `FDIDMTab`）
- [ ] GUI 离屏 `tabs: 7`（验证：GUI 冒烟命令）
- [ ] 既有 63 个测试无回归（验证：`python -m pytest -q` → `63 passed`）
- [ ] `compileall` 退出码 0（验证：运行命令）

## 边界

- [ ] 本阶段修改的 UI 文件仅 4 个（`fdidm_hardware_test_tab.py`、`waveform_compare_tab.py`、`hardware_test_tab.py`、`fdidm_tab.py`）（验证：4 轮提交文件清单）
- [ ] `waveform_sim/simulation/*`、`waveform_sim/hardware/*`、`waveform_sim/core/*` 零改动（验证：`git diff`）
- [ ] 提取均为逐字搬移，未改方法体（验证：`git diff` 中删除行与新增行一致）

## 编译与测试

- [ ] `python -m pytest -q` 为 `63 passed`（验证：运行命令看输出）
- [ ] `python -m compileall -q waveform_sim` 退出码 0、无输出（验证：运行命令）
- [ ] 4 轮提交后工作区干净（验证：`git status --short` 为空）

## 端到端场景

- [ ] 场景 1（页签可构造）：offscreen 直接构造四个页签无异常（验证：四条构造命令）
- [ ] 场景 2（UI 不受影响）：离屏启动 `MainWindow` 输出 `tabs: 7`（验证：GUI 冒烟命令）

