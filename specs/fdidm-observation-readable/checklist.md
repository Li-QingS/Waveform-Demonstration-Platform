# FDIDM 硬件观测效果可视化 Checklist

- [ ] 第一张效果主视图同时给出基线/候选 EVM、SER、改善量和颜色化结论。
- [ ] 第二张质量趋势采用滑动中位数，异常值以 NaN/阴影呈现，不伪造连续趋势。
- [ ] 第三张参数动作图使用阶梯线并标记应用/回滚事件，图例说明参数含义。
- [ ] 统计文本首行给出 FDIDM 自适应结论，随后给出基线→候选、改善量、样本可信度和原因。
- [ ] tooltip 保留 contract、Wilson 区间、raw/FEC BER 等详细证据。
- [ ] 现有 `tests/test_fdidm_hardware_comparison.py` 和全量 `pytest` 通过。
- [ ] `python -m compileall -q waveform_sim tests` 和 `git diff --check` 通过。
