# FDIDM 硬件链路稳定性与自适应闭环 Checklist

- [ ] RX 异常增量被计数并跳过（测试注入 `rx_new_samples` 超阈值）。
- [ ] 异常窗口不更新帧计数、CSI 或自适应快照。
- [ ] 候选应用后状态为 `validating`，并记录旧/新 αβ。
- [ ] 稳定有效帧达到目标后，实测指标满足门槛才为 `accepted`。
- [ ] 指标恶化、失步或超时触发 `rollback`，旧参数恢复。
- [ ] UI/日志分开显示预测改善和实测改善。
- [ ] `pytest -q tests/test_fdidm_hardware_comparison.py tests/test_hardtest_import.py tests/test_ui_regressions.py` 通过。
