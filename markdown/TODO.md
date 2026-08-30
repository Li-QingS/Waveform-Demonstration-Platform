# TODO 清单

> 创建日期：2026-08-17
> 状态说明：`- [ ]` 表示未开始，`- [x]` 表示已完成；所有条目带 `TODO` 标记，便于全局搜索。

## 1. 自适应方案（软件仿真 + 硬件）

**背景**：目前只有 FDIDM 硬件链路有一套 α/β 预测式自适应（`waveform_sim/hardware/fdidm_hardtest.py`），软件仿真和其他波形都没有统一的自适应方案，导致波形间对比缺少“各自在最优参数下”的公平性。

- [x] TODO: 软件仿真侧 FDIDM 预测式自适应——`waveform_sim/simulation/fdidm_adaptive.py` 实现“CSI 快照 → 理论 SER 搜索 → 稳定判定 → 冷却 → 自动应用”闭环，仿真页新增“自适应过程”面板（α/β 轨迹、预测 SER 对比、切换标记、状态文本）
- [ ] TODO: 定义自适应方案的验收指标（如增益下限、切换稳定性、开销上限）

## 2. 波形对比展示方式深入设计

**背景**：当前波形对比页（`waveform_sim/ui/waveform_compare_tab.py`）已有基础展示，但多维度、多场景的对比呈现方式还不够深入。

- [ ] TODO: 梳理对比维度：BER、EVM、吞吐量、复杂度、同步鲁棒性、时延等
- [ ] TODO: 设计同一信道条件下多波形并排 / 叠加对比的展示形式
- [ ] TODO: 增加时域、频域、时延-多普勒等多域可视化对比
- [ ] TODO: 扫描结果展示升级：曲线、热力图、3D 曲面（含 α/β 性能面）
- [ ] TODO: 对比报告自动生成与导出（图表 + 参数快照 + 结论）
- [ ] TODO: 对比页 UI 交互设计评审与原型验证

## 3. 平台架构优化

**背景**：核心文件过大（如 `fdidm_hardtest.py` 约 5700 行），各波形实现重复度高，需要模块化与统一接口。

- [x] TODO: 拆分 `fdidm_hardtest.py` 为可维护模块（信道估计、FEC、自适应、日志等）——已拆为 stream/channel/fec/gr_flow/fdidm_adaptive
- [x] TODO: 抽取 OFDM / OTFS / AFDM / FDIDM 硬件后端的公共基类与统一接口，消除重复代码——统一引擎 LinkSimulator + 四个 transceiver/hardtest 兼容壳
- [x] TODO: 统一配置定义与参数校验层——`waveform_sim/core/config.py`
- [x] TODO: 统一日志、状态快照与可观测性机制——`service/event_logger` + `diagnostics/*`
- [x] TODO: 补充单元测试与链路级回归测试——pytest 66 用例
- [ ] TODO: 优化运行时性能与稳定性（UHD overflow、实时调度、线程模型）
- [ ] TODO: 架构评审与重构后全链路验证

## 建议执行顺序

1. 先完成第 1 项（公平的自适应能力）；
2. 再做第 2 项（对比展示方式）；
3. 第 3 项架构优化可穿插进行，优先拆最大的 `fdidm_hardtest.py`。

各 TODO 的详细验收标准待后续讨论后补充。
