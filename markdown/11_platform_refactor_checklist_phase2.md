# 平台工程化重构 Checklist（阶段 2：公共 DSP 模块）

> 范围：`waveform_sim/core` 新增 modem / metrics / transforms / waveforms，及一致性测试。
> 每一项通过运行命令或观察行为验证。

## 实现完整性

- [ ] `waveform_sim/core/` 下新增 `modem.py`、`metrics.py`、`transforms.py`、`waveforms.py` 四个模块（验证：`Test-Path` 均为 True）
- [ ] `tests/` 下新增 `test_modem.py`、`test_metrics.py`、`test_transforms.py`、`test_waveforms.py`，共 18 个用例（验证：`python -m pytest --collect-only -q` 计数）
- [ ] `modem.constellation(mod_order)` 返回（星座点, 位标签），支持 QPSK / 16QAM / 64QAM（验证：`test_modem.py` passed）
- [ ] `transforms` 提供 11 个目标接口（验证：`python -c "from waveform_sim.core import transforms; print(hasattr(transforms, 'fdidm_modulate'))"` 输出 True）
- [ ] `create_waveform` 对未知波形抛 `ValueError`（验证：`test_waveforms.py::test_create_waveform_unknown` passed）

## 行为一致（与现有实现等价）

- [ ] 新调制星座与 `simple_fdidm_rx.FDIDMTransceiver._build_gray_qam` 完全一致（验证：`test_modem.py::test_constellation_matches_fdidm` passed）
- [ ] `gamma_matrix` 与现有 `FDIDMTransceiver._gamma` 一致（验证：`test_transforms.py::test_fdidm_gamma_matches_legacy` passed）
- [ ] OTFS 时域输出与现有 `OTFSTransceiver._dd_to_tf + _tf_to_time_cp` 一致（验证：`test_transforms.py::test_otfs_matches_legacy` passed）
- [ ] AFDM 变换矩阵与现有 `simple_afdm_rx._build_afdm_mats` 一致（验证：`test_transforms.py::test_afdm_matrices_match_legacy` passed）
- [ ] 四波形无信道往返恢复误差 < 1e-9（验证：`test_waveforms.py` 4 个参数化用例 + transforms 往返用例 passed）

## 边界

- [ ] 旧业务文件零改动：`waveform_sim/simulation/*`、`waveform_sim/ui/*`、`waveform_sim/hardware/*` 无任何修改（验证：`git diff 532d0ca..HEAD --stat` 只含新增 core 与 tests 文件）
- [ ] `waveform_sim/core` 四个新模块只依赖 numpy / 标准库，不 import 旧的 `simulation` / `hardware` 模块（验证：`Select-String` 检查 import 语句）

## 编译与测试

- [ ] `python -m pytest -q` 为 `27 passed`（验证：运行命令看输出）
- [ ] `python -m compileall -q waveform_sim` 退出码 0、无输出（验证：运行命令）
- [ ] T2.5 提交后工作区干净（验证：`git status --short` 为空）

## 端到端场景

- [ ] 场景 1（单测全绿）：`python -m pytest -q` 输出 `27 passed`，其中一致性用例证明新模块与现有 DSP 等价
- [ ] 场景 2（接口独立可用）：`python -c "from waveform_sim.core.waveforms import create_waveform; from waveform_sim.core.config import WaveformConfig; import numpy as np; cfg = WaveformConfig(waveform='OFDM', fft_size=64, cp_len=16).normalized(); w = create_waveform(cfg); x = np.ones(w.symbol_capacity); print(float(np.max(np.abs(w.demodulate(w.modulate(x), n_symbols=x.size) - x))))"` 输出小于 1e-9 的数值

