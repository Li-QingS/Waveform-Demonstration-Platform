# FDIDM 硬件优势对比演示 Tasks

## 文件清单

| 操作 | 文件 | 职责 |
|---|---|---|
| 新建 | `specs/fdidm-hardware-advantage-demo/spec.md` | 需求与边界 |
| 新建 | `specs/fdidm-hardware-advantage-demo/plan.md` | 技术设计 |
| 新建 | `specs/fdidm-hardware-advantage-demo/task.md` | 实现任务 |
| 新建 | `specs/fdidm-hardware-advantage-demo/checklist.md` | 验收清单 |
| 修改 | `waveform_sim/ui/fdidm_hardware_test_tab.py` | 对比会话 UI、状态机、统计和绘图 |
| 新建 | `tests/test_fdidm_hardware_comparison.py` | 离屏行为测试 |

## T1: 增加对比状态和控制控件

**文件：** `waveform_sim/ui/fdidm_hardware_test_tab.py`

**步骤：**
1. 增加基线 α/β、阶段时长、交替轮数控件。
2. 增加开始/停止按钮和阶段状态标签。
3. 初始化 session 字段和对比定时器。

**验证：** 离屏创建页面，控件存在、默认值有效、无硬件导入异常。

## T2: 实现阶段编排和 live 配置

**文件：** `waveform_sim/ui/fdidm_hardware_test_tab.py`

**步骤：**
1. 实现开始、停止、进入阶段和阶段计时。
2. 基线阶段关闭自适应，自适应阶段开启自适应。
3. 通过现有 configure/live 更新路径应用 α/β。

**验证：** 使用模拟 backend 记录 configure/stop/start 调用，阶段顺序正确且 α/β
切换不主动 stop/start。

## T3: 实现指标采样、汇总和结论

**文件：** `waveform_sim/ui/fdidm_hardware_test_tab.py`

**步骤：**
1. 在刷新周期采集阶段内指标和时间序列。
2. 阶段结束时按计数差值生成记录。
3. 计算两种模式的均值和改善量。
4. 更新状态标签和日志文本。

**验证：** 注入固定 status/stats 序列，得到正确 CRC、SER、EVM 和改善量；无样本时
显示不可用提示。

## T4: 接入双曲线展示

**文件：** `waveform_sim/ui/fdidm_hardware_test_tab.py`

**步骤：**
1. 增加 baseline/adaptive 两条 EVM 曲线。
2. 对比开始时清空旧曲线，对比结束后保留曲线。
3. 增加阶段边界和图标题说明。

**验证：** 离屏刷新后两条曲线均有数据，颜色/图例可区分，普通运行仍显示当前 EVM。

## T5: 增加自动化测试并回归

**文件：** `tests/test_fdidm_hardware_comparison.py`

**步骤：**
1. 测试控件和初始状态。
2. 测试阶段顺序和计数差值。
3. 测试汇总改善量、空样本处理和再次开始清理。
4. 运行原有硬件导入、UI 和全量测试。

**验证：** `pytest -q tests/test_fdidm_hardware_comparison.py tests/test_hardtest_import.py tests/test_ui_regressions.py` 通过。

## 执行顺序

```text
T1 → T2 → T3 → T4 → T5
```
# 方向修订（2026-09-03）

后续任务以手动开关前后观测为准；原定时交替对比任务保留为废弃设计记录，不作为验收前提。
