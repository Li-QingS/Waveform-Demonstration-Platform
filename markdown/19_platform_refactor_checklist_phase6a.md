# 平台工程化重构 Checklist（阶段 6a：硬件后端兼容壳）

> 范围：四个 `*_hardtest.py` 兼容壳改造（零逻辑改动）；无 USRP 环境下不做真构造。
> 每一项通过运行命令或观察行为验证。

## 实现完整性

- [ ] 四个文件均存在 `_LegacyAFDMHardwareTest` / `_LegacyOfdmHardwareTx` / `_LegacyOTFSHardwareTest` / `_LegacyFDIDMHardwareTest`（验证：`python -c` 导入四个模块并 getattr）
- [ ] 四个公开类名 `AFDMHardwareTest` / `OfdmHardwareTx` / `OTFSHardwareTest` / `FDIDMHardwareTest` 仍可导入，且与旧类名不同（验证：`test_hardware_shells.py` passed）
- [ ] 兼容壳支持 `backend=` 注入（验证：桩后端委托 `get_status` 返回 `{"stub": True}`）
- [ ] `tests/test_hardware_shells.py` 参数化覆盖 4 个波形（验证：pytest 收集计数）

## 行为保持

- [ ] `tests/test_hardtest_import.py` 仍通过：四个模块可导入且公开类存在（验证：`python -m pytest tests/test_hardtest_import.py -q` → `1 passed`）
- [ ] 既有 44 个测试无回归（验证：`python -m pytest -q` → `48 passed`）
- [ ] GUI 离屏启动 `tabs: 7`，硬件页签构造不触发 hardtest 实例化（验证：GUI 冒烟命令）
- [ ] UI 引用方式不变：`hardware_test_tab` / `fdidm_hardware_test_tab` 按类名动态构造仍可用（验证：代码零改动 + 导入冒烟）

## 边界

- [ ] 本阶段修改的业务文件仅 4 个 `*_hardtest.py`（验证：4 轮提交文件清单）
- [ ] `waveform_sim/ui/*`、`waveform_sim/simulation/*`、`waveform_sim/core/*`、阶段 5 的 4 个抽象模块零改动（验证：`git diff`）
- [ ] 无 USRP 环境下不构造真实后端，测试全部走桩注入（验证：测试文件无构造调用）

## 编译与测试

- [ ] `python -m pytest -q` 为 `48 passed`（验证：运行命令看输出）
- [ ] `python -m compileall -q waveform_sim` 退出码 0、无输出（验证：运行命令）
- [ ] 4 轮提交后工作区干净（验证：`git status --short` 为空）

## 端到端场景

- [ ] 场景 1（委托链）：`shell = AFDMHardwareTest(backend=stub)` → `shell.get_status()` 返回桩结果，证明公开接口委托到后端（验证：`test_shell_delegates_to_backend` passed）
- [ ] 场景 2（UI 不受影响）：离屏启动 `MainWindow` 输出 `tabs: 7`（验证：GUI 冒烟命令）

