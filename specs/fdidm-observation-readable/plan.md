# FDIDM 硬件观测效果可视化 Plan

## 架构

仅修改 `waveform_sim/ui/fdidm_hardware_test_tab.py` 的观测页显示层：后端继续提供逐帧 SER/EVM、有效性和异常计数；UI 维护窗口汇总、稳健趋势序列和异常事件。

## 关键设计

1. 效果主视图：使用 `observer.latest_pair()` 的基线/候选窗口，绘制两组并列柱（EVM、SER）和改善百分比文本；根据 outcome 使用绿/灰/红主题。
2. 链路质量趋势：保留 EVM/SER 原始缓存，增加窗口 5 的滑动中位数曲线；无效点写 NaN；在同一时间轴添加异常阴影。
3. 自适应动作：α/β 使用阶梯线，应用点显示 ▲，回滚点显示 ▼，图例和副标题说明“参数值，不是性能值”。
4. 结论文本：由 `outcome`、`baseline_window`、`candidate_window` 和 `result_reason` 生成四行短句；完整 contract、Wilson 区间、raw/FEC BER 放入 tooltip。
5. 三图 x 轴保持 link，事件线同时添加到三张图，避免用户在不同时间基之间来回换算。

## 兼容性

不删除 `_obs_ser/_obs_evm` 原始缓存和导出字段；只新增趋势缓存及格式化辅助函数。现有测试对原始缓存的断言继续有效。
