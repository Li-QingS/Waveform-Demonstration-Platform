# FDIDM 硬件链路稳定性与自适应闭环 Plan

## 架构

1. `fdidm_hardtest.py` 在 monitor 入口增加 RX 增量健康门控与计数器。
2. `fdidm_adaptive.py` 增加候选验证状态、窗口统计和回滚接口。
3. UI 只调用候选应用接口，并读取 predicted/measured 两类状态。

## 核心状态

- RX 健康：`rx_overflow_count`、`rx_last_overflow_samples`、`rx_overflow_reason`。
- 自适应验证：`idle|optimizing|ready|validating|accepted|rollback|cooldown`。
- 候选记录：旧/新 αβ、预测 SER、验证帧数、实测 SER/EVM/CRC、结果原因。

## 关键接口

- `begin_alpha_beta_candidate(alpha, beta, predicted_gain_db, recommendation_seq)`
- `_record_alpha_beta_validation_sample_locked(metrics)`
- `get_alpha_beta_adaptation_status()` 扩展验证字段。

## 决策

- RX 异常阈值按采样率和处理周期动态计算，异常窗口只推进游标不处理。
- 候选验证至少等待 3 个新鲜窗口，再累计 5 个有效帧。
- 接受门槛：CRC 不下降超过 5 个百分点，且 EVM/SER 不恶化超过 10%；否则回滚。
