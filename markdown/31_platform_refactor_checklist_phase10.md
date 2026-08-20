# 平台工程化重构 Checklist（阶段 10：文档与收尾）

> 范围：README、重构对照清单、TODO 状态、全量回归。
> 每一项通过运行命令或观察行为验证。

## 实现完整性

- [ ] `README.md` 目录结构反映当前架构（core / service / diagnostics / hardware 抽象 / ui 公共组件 / scripts / tests / CI）（验证：`Select-String` 关键词）
- [ ] `markdown/REFACTOR_CHECKLIST.md` 存在，含阶段状态表（0~9 完成、6c 真机待实验室、10 本阶段）（验证：文件内容）
- [ ] `markdown/TODO.md` 第 3 节前五项已勾选、后两项未勾选（验证：文件内容）
- [ ] README 文档清单包含 06~31 与 REFACTOR_CHECKLIST、TODO（验证：文件内容）

## 行为保持

- [ ] `python -m pytest -q` 为 `66 passed`（验证：运行命令）
- [ ] `python -m compileall -q waveform_sim scripts` 退出码 0（验证：运行命令）
- [ ] `python scripts/run_self_test.py` 输出 `self test OK`（验证：运行命令）
- [ ] GUI 离屏 `tabs: 7`（验证：GUI 冒烟命令）

## 边界与清理

- [ ] 本阶段未修改任何业务代码（core/simulation/hardware/service/diagnostics/ui 零改动）（验证：提交文件清单）
- [ ] 仓库内无 `runs/` 残留目录（验证：`Get-ChildItem -Recurse -Directory -Filter runs` 为空）
- [ ] 工作区干净（验证：`git status --short` 为空）

## 端到端场景

- [ ] 场景 1（文档自洽）：按 README 目录结构能找到每个模块文件（验证：逐一 `Test-Path`）
- [ ] 场景 2（交付可验证）：`check_environment.py` → `run_self_test.py` → `pytest -q` 依次通过（验证：三条命令）

