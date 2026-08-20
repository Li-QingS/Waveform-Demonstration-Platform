# 平台工程化重构 Checklist（阶段 4：OFDM / OTFS / AFDM 接入统一引擎）

> 范围：三个波形逐个接入 `LinkSimulator`；UI 与公开接口零改动。
> 每一项通过运行命令或观察行为验证。

## 实现完整性

- [ ] `engine._create_backend` 支持 OFDM / OTFS / AFDM 三个分支（验证：`python -c` 分别构造 `LinkSimulator(WaveformConfig(waveform=...))` 成功）
- [ ] `simple_ofdm_rx.py` / `simple_otfs_rx.py` / `simple_afdm_rx.py` 内旧实现类已分别更名为 `_LegacyOfdmTransceiver` / `_LegacyOTFSTransceiver` / `_LegacyAFDMTransceiver`，且公开名 `OfdmTransceiver` / `OTFSTransceiver` / `AFDMTransceiver` 仍是 `LinkSimulator` 子类（验证：`python -c` 导入并 isinstance 断言）
- [ ] `tests/test_engine.py` 的 `test_engine_backend_smoke` 参数化覆盖三个波形（验证：pytest 收集计数）

## 行为保持

- [ ] 四波形收发冒烟仍通过（三个新波形现在走兼容壳路径）（验证：`python -m pytest tests/test_transceivers_smoke.py -q` → `4 passed`）
- [ ] 引擎三个新后端均能启动出帧，指标含 ber 与 evm（验证：`test_engine_backend_smoke` 3 个参数化用例 passed）
- [ ] 阶段 2/3 一致性测试无回归（验证：`python -m pytest tests/test_modem.py tests/test_transforms.py tests/test_engine.py -q` 全绿）
- [ ] GUI 离屏启动 `tabs: 7`，三个仿真页签可构造（验证：GUI 冒烟命令）
- [ ] UI 构造参数经别名映射正确：`fft_len→fft_size`、`doppler_spread/doppler_freq/doppler_hz→doppler_spread_hz`、`sample_rate→sample_rate_hz`（验证：引擎配置断言）

## 边界

- [ ] 本阶段修改的业务文件仅 `simple_ofdm_rx.py`、`simple_otfs_rx.py`、`simple_afdm_rx.py` 三个（验证：三轮提交的文件清单）
- [ ] `waveform_sim/ui/*`、`waveform_sim/hardware/*`、`simple_fdidm_rx.py` 零改动（验证：`git diff` 或提交清单）

## 编译与测试

- [ ] `python -m pytest -q` 为 `34 passed`（验证：运行命令看输出）
- [ ] `python -m compileall -q waveform_sim` 退出码 0、无输出（验证：运行命令）
- [ ] 三轮提交后工作区干净（验证：`git status --short` 为空）

## 端到端场景

- [ ] 场景 1（四波形引擎统一入口）：`LinkSimulator(WaveformConfig(waveform=wf))` 对 FDIDM/OFDM/OTFS/AFDM 均能启动出帧（验证：`test_engine_backend_smoke` 参数化 + `test_engine_fdidm_uses_legacy_backend` 全绿）
- [ ] 场景 2（UI 不受影响）：离屏启动 `MainWindow` 输出 `tabs: 7`（验证：GUI 冒烟命令）

