# FDIDM 硬件链路稳定性与自适应闭环 Tasks

- T1: 在 `fdidm_hardtest.py` 增加 RX overflow 健康字段、动态异常阈值和窗口跳过逻辑；运行硬件导入测试。
- T2: 在 `fdidm_adaptive.py` 增加候选验证、接受/回滚和状态输出；运行自适应单元测试。
- T3: 修改 `fdidm_hardware_test_tab.py` 使用候选接口并区分预测/实测显示；运行 UI 回归测试。
- T4: 新增离屏测试覆盖异常 RX、验证接受和验证回滚；运行专项及全量测试。

执行顺序：T1 → T2 → T3 → T4。
