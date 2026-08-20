# 平台工程化重构 Checklist（阶段 0 ~ 1）

> 范围：阶段 0（基线锁定）与阶段 1（统一配置）。
> 每一项通过运行命令或观察行为验证，不依赖逐行读代码。

## 实现完整性

- [ ] 四份前置文档（spec / plan / task）已提交入库（验证：`git log --oneline -5` 有 docs 提交）
- [ ] `pyproject.toml` 可解析，且 pytest 使用其中 `testpaths = ["tests"]` 与 `pythonpath = ["."]`（验证：`python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"` 无报错；`python -m pytest --collect-only -q` 从 `tests/` 收集）
- [ ] `requirements.txt` 与 `environment.yml` 存在（验证：`Test-Path` 两者均为 True）
- [ ] `scripts/check_environment.py` 可运行，输出包含 Python / NumPy / PyQt5 / pyqtgraph / pytest / GNU Radio / UHD Python / uhd_find_devices 共 8 行，无 traceback（验证：`python scripts/check_environment.py`）
- [ ] `tests/` 下三个测试文件存在，共 6 个用例（4 收发冒烟 + 1 扫描冒烟 + 1 导入）（验证：`python -m pytest --collect-only -q` 计数）
- [ ] `waveform_sim/__init__.py` 存在且 `import waveform_sim.core` 可用（验证：`python -c "import waveform_sim.core; print(waveform_sim.__name__)"` 输出 `waveform_sim`）
- [ ] `waveform_sim/core/config.py` 提供 `WaveformConfig` / `AdaptiveConfig` / `HardwareConfig` / `ExperimentConfig` 四个 dataclass，且各自有 `normalized()` 与 `to_dict()`（验证：`python -c` 依次构造并调用）

## 行为保持（本次重构的关键约束）

- [ ] 阶段 0 到阶段 1 前后，四波形收发链路冒烟行为一致（验证：同一 `tests/test_transceivers_smoke.py` 在 T0.9 与 T1.4 后均 `4 passed`）
- [ ] 波形对比扫描行为一致（验证：`tests/test_scan_backend_smoke.py` 前后均 passed）
- [ ] 四个 hardtest 模块可导入且主类存在（验证：`tests/test_hardtest_import.py` passed）
- [ ] 阶段 0~1 未修改任何现有业务文件（`waveform_sim/simulation/*`、`waveform_sim/ui/*`、`waveform_sim/hardware/*` 中已有文件零改动）（验证：`git diff` 只含新增文件与 `.gitignore` 修改）

## 配置模型行为

- [ ] `WaveformConfig` 归一化生效：小写 `waveform` / `mod_order` 转大写，数值钳制（验证：`tests/test_config.py::test_waveform_config_roundtrip` passed）
- [ ] `ExperimentConfig` 嵌套序列化往返一致（验证：`tests/test_config.py::test_experiment_config_roundtrip` passed）
- [ ] 配置 JSON 保存 / 加载往返一致（验证：`tests/test_config.py::test_experiment_config_save_load` passed）

## 编译与测试

- [ ] 阶段 0 结束时 `python -m pytest -q` 为 `6 passed`（验证：运行命令看输出）
- [ ] 阶段 1 结束时 `python -m pytest -q` 为 `9 passed`（验证：运行命令看输出）
- [ ] `python -m compileall -q waveform_sim scripts` 退出码 0、无输出（验证：运行命令）
- [ ] T0.9 与 T1.4 提交后工作区干净（验证：`git status --short` 为空）

## 端到端场景

- [ ] 场景 1（基线可用）：从仓库根目录运行 `python scripts/check_environment.py`，NumPy 行 PASS、无 traceback；随后 `python -m pytest -q` 输出 `6 passed`
- [ ] 场景 2（配置层独立可用）：运行 `python -c "from waveform_sim.core.config import ExperimentConfig, WaveformConfig; a = ExperimentConfig(waveform=WaveformConfig(waveform='OFDM')).to_dict(); b = ExperimentConfig.from_dict(a).to_dict(); assert a == b; print('ok')"`，输出 `ok`
- [ ] 场景 3（无回归）：阶段 1 提交后 `python -m pytest -q` 仍 `9 passed`，且 `python -m compileall -q waveform_sim` 覆盖 UI 入口无报错

