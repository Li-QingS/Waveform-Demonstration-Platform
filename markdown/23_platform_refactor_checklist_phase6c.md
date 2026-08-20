# 平台工程化重构 Checklist（阶段 6c：FDIDM 自适应/性能面与 GNU Radio 块）

> 范围：`fdidm_adaptive.py`（自适应+性能面 mixin）与 `gr_flow.py`（两个 GNU Radio 块工厂）。
> 每一项通过运行命令或观察行为验证；真机回归单独列在末尾，需实验室环境执行。

## 实现完整性

- [ ] `waveform_sim/hardware/fdidm_adaptive.py` 提供 `FDIDMAdaptiveMixin`，包含 Block1（自适应）与 Block2（性能面）全部方法（验证：`python -c` 导入 + `Select-String` 抽查方法名）
- [ ] `waveform_sim/hardware/gr_flow.py` 提供 `make_tdl_channel_block` 与 `make_rx_ring_sink`，且模块可在无 gnuradio 环境导入（验证：`python -c "import waveform_sim.hardware.gr_flow"`）
- [ ] `_LegacyFDIDMHardwareTest` 的 MRO 含 `FECMixin` 与 `FDIDMAdaptiveMixin`（验证：`__mro__` 输出）
- [ ] `fdidm_hardtest.py` 不再包含两个方法块与两个内嵌 GNU Radio 类（验证：`Select-String` 无命中）

## 行为保持（本机可验证部分）

- [ ] 自适应纯函数测试通过：`_adaptive_qam_order`、`_adaptive_qfunc`、`_finite_float_or_nan`、`_adaptive_ser_from_symbol_nsr`（验证：`test_hardware_adaptive.py` 4 用例 passed）
- [ ] 既有 56 个测试无回归（验证：`python -m pytest -q` → `60 passed`）
- [ ] 编译通过（验证：`compileall` 退出码 0）
- [ ] GUI 不受影响（验证：离屏 `MainWindow` `tabs: 7`）

## 边界

- [ ] 本阶段修改的业务文件仅 `fdidm_hardtest.py` 一个（验证：2 轮提交文件清单）
- [ ] `gr_flow.py` 不改变 gnuradio 导入时机（懒加载，验证：无 gnuradio 环境可导入模块的代码路径保持不变）
- [ ] 其余 `*_hardtest.py`、`ui/`、`simulation/`、`core/` 零改动（验证：`git diff`）

## 编译与测试

- [ ] `python -m pytest -q` 为 `60 passed`（验证：运行命令看输出）
- [ ] `python -m compileall -q waveform_sim` 退出码 0、无输出（验证：运行命令）
- [ ] 2 轮提交后工作区干净（验证：`git status --short` 为空）

## 端到端场景

- [ ] 场景 1（搬移完整性）：Block1/Block2 每个方法名在 `fdidm_adaptive.py` 且不在 `fdidm_hardtest.py`（验证：脚本/`Select-String` 对照）
- [ ] 场景 2（UI 不受影响）：离屏启动 `MainWindow` 输出 `tabs: 7`（验证：GUI 冒烟命令）

## 真机回归（实验室环境，本机无法执行）

- [ ] 有 USRP 的机器上启动 FDIDM 硬件验证页，运行一轮文本回环，确认自适应调优、性能面绘制与 GNU Radio 链路行为与 6b 前一致

