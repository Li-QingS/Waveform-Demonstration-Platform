# 平台工程化重构 Checklist（阶段 7：服务层 · 实验记录）

> 范围：`waveform_sim/service/` 四个模块 + 引擎挂接。
> 每一项通过运行命令或观察行为验证。

## 实现完整性

- [ ] `waveform_sim/service/` 下存在 `run_state.py`、`artifact_writer.py`、`event_logger.py`、`experiment_service.py`（验证：`Test-Path` 均为 True）
- [ ] `ExperimentService` 可创建，提供 `start_run` / `log_event` / `log_metrics` / `finish_run` / `fail_run`（验证：`python -c` 调用链）
- [ ] `LinkSimulator` 支持 `auto_start_run` 参数与 `experiment_service` 注入（验证：`test_service.py` 2 个引擎用例）
- [ ] `tests/test_service.py` 3 个用例（验证：pytest 收集计数）

## 行为

- [ ] 一次 run 生成四个 artifact：`config.json`、`events.jsonl`、`metrics.csv`、`report.md`（验证：`test_run_artifacts` passed）
- [ ] 事件日志为合法 JSONL，首事件 `RUN_STARTED`、末事件 `RUN_FINISHED`（验证：`test_run_artifacts` 断言）
- [ ] 引擎挂接后 `LINK_STARTED` / `LINK_STOPPED` 写入事件，`step()` 逐帧写 metrics.csv（验证：`test_engine_experiment_hooks` passed）
- [ ] `auto_start_run=True` 时自动创建服务并进入 `RUNNING`（验证：`test_engine_auto_start_run` passed）

## 边界

- [ ] 实验记录默认关闭：不传 `experiment_service` / `auto_start_run` 时引擎行为不变（验证：既有 `test_engine.py` 4 用例仍 passed）
- [ ] 本阶段修改的业务文件仅 `waveform_sim/core/engine.py` 一个（验证：提交文件清单）
- [ ] `waveform_sim/ui/*`、`waveform_sim/simulation/*`、`waveform_sim/hardware/*` 零改动（验证：`git diff`）

## 编译与测试

- [ ] `python -m pytest -q` 为 `63 passed`（验证：运行命令看输出）
- [ ] `python -m compileall -q waveform_sim` 退出码 0、无输出（验证：运行命令）
- [ ] T7.4 提交后工作区干净（验证：`git status --short` 为空）

## 端到端场景

- [ ] 场景 1（独立可用）：`ExperimentService` 走完 start→log→finish 后，`runs/<run_id>/` 下四个文件齐全且内容可解析（验证：`test_run_artifacts`）
- [ ] 场景 2（引擎挂接）：`LinkSimulator` 注入服务后启动/停止/出帧，事件与指标均落盘（验证：`test_engine_experiment_hooks`）

