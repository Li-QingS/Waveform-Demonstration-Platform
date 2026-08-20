# 平台工程化重构 Checklist（阶段 5：硬件抽象层）

> 范围：`waveform_sim/hardware/` 新增 transport / rf_safety / device_manager / iq_replay；只增不改。
> 每一项通过运行命令或观察行为验证。

## 实现完整性

- [ ] `waveform_sim/hardware/` 下新增 4 个模块，均可导入（验证：`python -c` 依次 import）
- [ ] `create_transport` 工厂存在，simulated-loopback 可创建、未知 transport 抛 `ValueError`（验证：`test_transport_factory_unknown` passed）
- [ ] `RFSafetyPolicy` 提供 `validate()`，返回 `RFSafetyDecision`，strict 模式违规抛 `RFSafetyViolation`（验证：rf_safety 4 个用例 passed）
- [ ] `probe_devices()` 返回 `DeviceInfo` 列表（验证：`test_device_manager_probe` passed）
- [ ] `IQReplaySource` / `save_iq` 支持 `.npy` 与 complex64 原始文件（验证：iq_replay 2 个用例 passed）

## 行为

- [ ] loopback 传输往返一致：send 后 recv 按需平铺（验证：`test_transport_loopback_roundtrip` passed）
- [ ] RF 安全策略能拦截违规 loopback（高增益/无衰减）、未确认 OTA、白名单外频段，放行合规 loopback（验证：4 个 rf_safety 用例 passed）
- [ ] UHD 占位行为正确：无 UHD 环境报可读 RuntimeError，有 UHD 环境 send/recv 抛 NotImplementedError（验证：`test_uhd_transport_placeholder` passed）

## 边界

- [ ] 现有硬件文件（`*_hardtest.py`、`hardware/__init__.py`）零改动（验证：`git diff` 只含新增文件）
- [ ] `waveform_sim/ui/*`、`waveform_sim/simulation/*`、`waveform_sim/core/*` 零改动（验证：提交文件清单）

## 编译与测试

- [ ] `python -m pytest -q` 为 `44 passed`（验证：运行命令看输出）
- [ ] `python -m compileall -q waveform_sim` 退出码 0、无输出（验证：运行命令）
- [ ] T5.5 提交后工作区干净（验证：`git status --short` 为空）

## 端到端场景

- [ ] 场景 1（安全策略 + 传输链路）：构造 loopback transport 发送 3 个符号并 recv 6 个，逐值一致；构造违规 loopback 配置被 RF 策略拒绝（验证：transport/rf_safety 用例）
- [ ] 场景 2（IQ 可复现）：`save_iq` 后 `IQReplaySource` 循环读取输出与输入一致（验证：`test_iq_replay_npy` passed）

