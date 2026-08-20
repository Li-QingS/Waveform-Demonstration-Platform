# 平台工程化重构 Checklist（阶段 6b：FDIDM 纯 Python 部分拆分）

> 范围：`stream.py` / `channel.py` / `fec.py` 三个新模块 + `fdidm_hardtest.py` 对应删除。
> 每一项通过运行命令或观察行为验证。

## 实现完整性

- [ ] `waveform_sim/hardware/stream.py` 提供 `SampleRing`，`channel.py` 提供 `NTNTDLChannel`，`fec.py` 提供 `FECMixin`（验证：`python -c` 依次 import）
- [ ] `fdidm_hardtest.py` 仍可导入，`_SampleRing` / `_NTNTDLChannel` 通过别名导入可用，`_LegacyFDIDMHardwareTest` 继承 `FECMixin`（验证：`python -c` 导入模块并检查类 MRO）
- [ ] 三个新测试文件共 8 个用例（验证：`python -m pytest --collect-only -q` 计数）
- [ ] `fdidm_hardtest.py` 行数明显下降（三个块搬出，验证：`git diff --stat`）

## 行为保持

- [ ] 环形缓冲行为正确：写入/读取/回绕/清空（验证：`test_hardware_stream.py` 3 个用例 passed）
- [ ] 信道行为正确：构造/process 输出形状、summary 内容、reset（验证：`test_hardware_channel.py` 2 个用例 passed）
- [ ] FEC 行为正确：卷积编码/解码往返、奇偶校验、交织往返（验证：`test_hardware_fec.py` 3 个用例 passed）
- [ ] 既有 48 个测试无回归（验证：`python -m pytest -q` → `56 passed`）

## 边界

- [ ] 本阶段修改的业务文件仅 `fdidm_hardtest.py` 一个（验证：3 轮提交文件清单）
- [ ] 其余 `*_hardtest.py`、`waveform_sim/ui/*`、`waveform_sim/simulation/*`、`waveform_sim/core/*` 零改动（验证：`git diff`）
- [ ] 未搬移 GNU Radio 块与自适应/性能面（留 6c，验证：`fdidm_hardtest.py` 中仍存在 `gr.top_block` 引用与 `_alpha_beta_*` 方法）

## 编译与测试

- [ ] `python -m pytest -q` 为 `56 passed`（验证：运行命令看输出）
- [ ] `python -m compileall -q waveform_sim` 退出码 0、无输出（验证：运行命令）
- [ ] 3 轮提交后工作区干净（验证：`git status --short` 为空）

## 端到端场景

- [ ] 场景 1（新模块独立可用）：`SampleRing` 写入 1030 样本后 `read_latest(6)` 返回最后 6 个；`NTNTDLChannel` process 返回同长度输出；`FECMixin._conv_encode_bits/_conv_decode_bits` 往返一致（验证：三个测试文件）
- [ ] 场景 2（UI 不受影响）：离屏启动 `MainWindow` 输出 `tabs: 7`（验证：GUI 冒烟命令）

