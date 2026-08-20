# 平台工程化重构对照清单

## 总目标

在不改变算法行为、不破坏旧入口的前提下，把单体大文件重构为模块化工程架构：

- 保留 `cd waveform_sim && python main.py` 入口与全部公开类名；
- 统一配置、统一引擎、统一波形接口；
- 实验可复现（run artifact + JSONL 事件日志）；
- 硬件抽象与 RF 安全；
- 行为基线测试与 CI。

## 阶段完成状态

| 阶段 | 内容 | 状态 |
|------|------|------|
| 0 | 测试基线与工程脚手架 | 已完成 |
| 1 | 统一配置 `waveform_sim/core/config.py` | 已完成 |
| 2 | 公共 DSP 模块（modem/transforms/waveforms/metrics） | 已完成 |
| 3 | FDIDM 接入统一引擎 `LinkSimulator` | 已完成 |
| 4 | OFDM / OTFS / AFDM 接入统一引擎 | 已完成 |
| 5 | 硬件抽象层（transport/rf_safety/device_manager/iq_replay） | 已完成 |
| 6a | 硬件后端兼容壳（4 个 hardtest） | 已完成 |
| 6b | FDIDM 纯 Python 拆分（stream/channel/fec） | 已完成 |
| 6c | FDIDM 自适应/性能面与 GNU Radio 块拆分 | 已完成（真机回归待实验室） |
| 7 | 服务层（实验记录 artifact + 事件日志） | 已完成 |
| 8 | UI 收敛（画布/worker/助手提取） | 已完成 |
| 9 | 诊断 / 脚本 / CI | 已完成 |
| 10 | 文档与收尾 | 已完成 |

## 验收状态

- 行为基线：`python -m pytest -q` → 66 passed
- 编译：`python -m compileall -q waveform_sim scripts` 通过
- 自检：`python scripts/run_self_test.py` 输出 `self test OK`
- GUI：离屏 `MainWindow` 7 个页签可构造
- 待实验室：6c 真机回归（USRP 硬件链路 + 自适应调优 + 性能面）

## 当前工程结构

见 `README.md` 目录结构；核心变化是把约 5700 行的 `fdidm_hardtest.py` 拆分为
`stream / channel / fec / gr_flow / fdidm_adaptive` 等模块，四个 transceiver 与
四个 hardtest 均改为薄兼容壳，算法行为保持不变。

