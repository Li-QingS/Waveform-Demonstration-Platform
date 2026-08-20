# 平台工程化重构 Checklist（阶段 3：FDIDM 接入统一引擎）

> 范围：新增 `waveform_sim/core/engine.py`；`simple_fdidm_rx.py` 改造为兼容壳。
> 每一项通过运行命令或观察行为验证。

## 实现完整性

- [ ] `waveform_sim/core/engine.py` 存在且提供 `LinkSimulator`（验证：`python -c "from waveform_sim.core.engine import LinkSimulator; print(LinkSimulator.__name__)"` 输出 `LinkSimulator`）
- [ ] `simple_fdidm_rx.py` 内旧实现类已更名为 `_LegacyFDIDMTransceiver`，模块末尾存在 `FDIDMTransceiver(LinkSimulator)` 兼容壳与 `_create_fdidm_backend` 工厂（验证：`python -c` 导入两个名字均成功）
- [ ] `tests/test_engine.py` 存在，含 4 个用例（验证：`python -m pytest tests/test_engine.py -q` → `4 passed`）

## 行为保持

- [ ] 既有四波形收发冒烟测试仍通过（含 FDIDM 兼容壳路径）（验证：`python -m pytest tests/test_transceivers_smoke.py -q` → `4 passed`）
- [ ] 阶段 2 的一致性测试仍通过：`_build_gray_qam` 桥接与 `_gamma` 委托不破坏 modem/transforms 测试（验证：`python -m pytest tests/test_modem.py tests/test_transforms.py -q` → `9 passed`）
- [ ] 兼容壳与旧实现产生相同指标键集合且 BER 在 [0,1]（验证：`tests/test_engine.py::test_fdidm_shell_matches_legacy_metrics_keys` passed）
- [ ] 引擎配置别名正确：`ebn0_db`→`snr_db`、`decoder`→`detector`、`fc_hz`→`center_freq_hz`、`channel_seed`→`seed`（验证：`tests/test_engine.py::test_engine_config_aliases` passed）
- [ ] GUI 离屏可启动且页签数为 7（验证：`QT_QPA_PLATFORM=offscreen` 下 `MainWindow().tabs.count()` 输出 `7`）

## 边界

- [ ] 本阶段只修改 `waveform_sim/simulation/simple_fdidm_rx.py` 一个业务文件；`waveform_sim/ui/*`、其余 `simulation/*`、`hardware/*` 零改动（验证：`git diff` 与提交文件清单）
- [ ] engine 对旧文件的依赖仅为 `simple_fdidm_rx._create_fdidm_backend` 一处，且带"阶段3过渡依赖"注释（验证：`Select-String` 检查 engine.py import）

## 编译与测试

- [ ] `python -m pytest -q` 为 `31 passed`（验证：运行命令看输出）
- [ ] `python -m compileall -q waveform_sim` 退出码 0、无输出（验证：运行命令）
- [ ] T3.5 提交后工作区干净（验证：`git status --short` 为空）

## 端到端场景

- [ ] 场景 1（引擎独立可用）：`LinkSimulator(WaveformConfig(waveform='FDIDM'))` 启动后出帧，指标含 ber/ser/evm_db（验证：`test_engine_fdidm_uses_legacy_backend` passed）
- [ ] 场景 2（UI 不受影响）：离屏启动 `MainWindow` 输出 `tabs: 7`，FDIDM 页签可构造（验证：GUI 冒烟命令）

