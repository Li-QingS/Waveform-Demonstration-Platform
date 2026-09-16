# FDIDM 硬件观测效果可视化 Tasks

## T1：新增自适应效果主视图

- 文件：`waveform_sim/ui/fdidm_hardware_test_tab.py`
- 用基线/候选窗口绘制 EVM、SER 并列对比和改善百分比。
- 根据 improved/inconclusive/regressed 切换颜色和标题。
- 验证：FakeBackend 三种结论均能读到正确的对比值。

## T2：重写质量趋势与参数动作图

- 为 EVM/SER 趋势增加滑动中位数和异常断点。
- α/β 改为阶梯线并标记应用/回滚事件，三图共享时间轴。
- 验证：离屏读取标题、图例、轴标签和事件标记。

## T3：重写结论文本框

- 将正文改为四行结论摘要，首行直接表达 FDIDM 自适应效果。
- 保留完整专业字段到 tooltip。
- 验证：基线、候选、改善量、样本数、结论和原因均可直接读取。

## T4：回归测试

- 扩展 `tests/test_fdidm_hardware_comparison.py` 覆盖趋势、异常断线和文案。
- 运行专项测试、全量测试和 compileall。
