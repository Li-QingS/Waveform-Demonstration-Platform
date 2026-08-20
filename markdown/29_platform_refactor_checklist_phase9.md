# 平台工程化重构 Checklist（阶段 9：诊断 / 脚本 / CI）

> 范围：diagnostics API、两个脚本、CI、check_environment 复用。
> 每一项通过运行命令或观察行为验证。

## 实现完整性

- [ ] `waveform_sim/diagnostics/` 下存在 `health_check.py`、`report_exporter.py`、`snapshot.py`（验证：`Test-Path` 均为 True）
- [ ] `run_health_check(include_usrp=False)` 返回 8 项检查，`format_health_check` 输出表格（验证：`test_health_check_runs` passed）
- [ ] `scripts/check_usrp.py`、`scripts/run_self_test.py` 存在且可运行（验证：运行脚本）
- [ ] `scripts/check_environment.py` 复用 `diagnostics.health_check`（验证：文件内容检查）
- [ ] `.github/workflows/ci.yml` 存在且只装 numpy+pytest（验证：文件内容检查）

## 行为

- [ ] `export_run_report` 从 run 目录生成 `report.md`，含 Config 与 Last Metrics（验证：`test_report_exporter` passed）
- [ ] `Snapshot.capture` 生成带时间戳的快照 dict（验证：`test_snapshot` passed）
- [ ] `run_self_test.py` 完成 FDIDM 出帧并生成实验 artifact，输出 `self test OK`（验证：运行脚本）
- [ ] `check_usrp.py` 输出设备探测结果、无 traceback（验证：运行脚本）

## 边界

- [ ] 本阶段修改的既有文件仅 `scripts/check_environment.py` 一个（验证：提交文件清单）
- [ ] `waveform_sim/ui/*`、`waveform_sim/simulation/*`、`waveform_sim/hardware/*`、`waveform_sim/core/*` 零改动（验证：`git diff`）
- [ ] CI 不引入 PyQt5 依赖，测试套件纯 numpy 可跑（验证：`pip install numpy pytest` 后 `pytest -q`）

## 编译与测试

- [ ] `python -m pytest -q` 为 `66 passed`（验证：运行命令看输出）
- [ ] `python -m compileall -q waveform_sim scripts` 退出码 0、无输出（验证：运行命令）
- [ ] T9.5 提交后工作区干净（验证：`git status --short` 为空）

## 端到端场景

- [ ] 场景 1（脚本链路）：`check_environment.py` → `check_usrp.py` → `run_self_test.py` 依次运行均无异常（验证：三条命令）
- [ ] 场景 2（CI 等价）：`pytest -q` + `compileall` 与 CI 步骤一致且通过（验证：运行命令）

