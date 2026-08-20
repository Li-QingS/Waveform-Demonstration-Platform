# 平台工程化重构 Tasks（阶段 10：文档与收尾）

> 前置：阶段 9 已验收通过（pytest 66 passed）。
> 目标：更新 README 反映当前架构、新增重构对照清单、更新 TODO 状态，做全量回归并收尾提交。
> 说明：用户此前删除了 `markdown/05_platform_introduction.md`，本阶段不再重建独立平台介绍文档，由 README 承担介绍职责。

## 文件清单

| 操作 | 文件 | 职责 |
|------|------|------|
| 修改 | `README.md` | 目录结构、运行/验证说明、文档清单更新 |
| 新建 | `markdown/REFACTOR_CHECKLIST.md` | 重构对照清单（阶段完成状态 + 当前结构） |
| 修改 | `markdown/TODO.md` | 架构优化条目状态更新（已完成项勾选） |

## 任务

### T10.1：更新 README.md

**文件：** 修改 `README.md`
**依赖：** 无

**步骤：**

1. 目录结构替换为当前实际架构：

```text
waveform_sim/
  main.py                       # 程序入口（PyQt5 主窗口）
  core/                         # 工程化内核
    config.py                   #   统一配置（Waveform/Adaptive/Hardware/Experiment）
    engine.py                   #   统一链路引擎 LinkSimulator
    modem.py / transforms.py / waveforms.py / metrics.py
  simulation/                   # 软件仿真后端（transceiver 兼容壳）
    simple_fdidm_rx.py / simple_ofdm_rx.py / simple_otfs_rx.py / simple_afdm_rx.py
    compare_scan_backend.py     #   三波形 Monte-Carlo 对比扫描
  hardware/                     # 硬件后端（兼容壳 + 抽象层）
    fdidm_hardtest.py / ofdm_hardtest.py / otfs_hardtest.py / afdm_hardtest.py
    transport.py / rf_safety.py / device_manager.py / iq_replay.py
    stream.py / channel.py / fec.py / gr_flow.py / fdidm_adaptive.py
  service/                      # 实验记录：experiment_service / artifact_writer / event_logger / run_state
  diagnostics/                  # health_check / report_exporter / snapshot
  ui/                           # GUI 页签 + 公共组件（base_waveform_tab、plot_widgets、workers、ui_utils）
markdown/                       # spec/plan/task/checklist、REFACTOR_CHECKLIST、TODO
scripts/                        # check_environment / check_usrp / run_self_test
tests/                          # pytest 行为基线 / 回归测试
.github/workflows/ci.yml        # 纯 Python 核心 CI
```

2. “运行”小节补充验证命令：

```text
无显示环境验证：
  python scripts/check_environment.py   # 环境健康检查
  python scripts/run_self_test.py       # FDIDM 自检（出帧 + 实验 artifact）
  python -m pytest -q                   # 行为基线 / 回归测试
```

3. “文档”小节更新为当前清单（06~31 + REFACTOR_CHECKLIST + TODO）。

**验证：** `Get-Content README.md` 包含 `core/`、`service/`、`diagnostics/`、`REFACTOR_CHECKLIST` 等关键词。

### T10.2：新增 markdown/REFACTOR_CHECKLIST.md

**文件：** 新建 `markdown/REFACTOR_CHECKLIST.md`
**依赖：** 无

**步骤：** 创建文件，内容如下（对照清单）：

```markdown
# 平台工程化重构对照清单

## 总目标

在不改变算法行为、不破坏旧入口的前提下，把单体大文件重构为模块化工程架构：

- 保留 `cd waveform_sim && python main.py` 入口与全部公开类名；
- 统一配置、统一引擎、统一波形接口；
- 实验可复现（run artifact + JSONL 事件日志）；
- 硬件抽象与 RF 安全；
- 行为基线测试与 CI。

## 阶段完成状态

| 阶段 | 内容 | 状态 |
|------|------|------|
| 0 | 测试基线与工程脚手架 | 已完成 |
| 1 | 统一配置 `waveform_sim/core/config.py` | 已完成 |
| 2 | 公共 DSP 模块（modem/transforms/waveforms/metrics） | 已完成 |
| 3 | FDIDM 接入统一引擎 `LinkSimulator` | 已完成 |
| 4 | OFDM / OTFS / AFDM 接入统一引擎 | 已完成 |
| 5 | 硬件抽象层（transport/rf_safety/device_manager/iq_replay） | 已完成 |
| 6a | 硬件后端兼容壳（4 个 hardtest） | 已完成 |
| 6b | FDIDM 纯 Python 拆分（stream/channel/fec） | 已完成 |
| 6c | FDIDM 自适应/性能面与 GNU Radio 块拆分 | 已完成（真机回归待实验室） |
| 7 | 服务层（实验记录 artifact + 事件日志） | 已完成 |
| 8 | UI 收敛（画布/worker/助手提取） | 已完成 |
| 9 | 诊断 / 脚本 / CI | 已完成 |
| 10 | 文档与收尾 | 本阶段 |

## 验收状态

- 行为基线：`python -m pytest -q` → 66 passed
- 编译：`python -m compileall -q waveform_sim scripts` 通过
- 自检：`python scripts/run_self_test.py` 输出 `self test OK`
- GUI：离屏 `MainWindow` 7 个页签可构造
- 待实验室：6c 真机回归（USRP 硬件链路 + 自适应调优 + 性能面）

## 当前工程结构

见 `README.md` 目录结构；核心变化是把约 5700 行的 `fdidm_hardtest.py` 拆分为
`stream / channel / fec / gr_flow / fdidm_adaptive` 等模块，四个 transceiver 与
四个 hardtest 均改为薄兼容壳，算法行为保持不变。
```

**验证：** 文件存在且包含阶段状态表。

### T10.3：更新 markdown/TODO.md 架构优化状态

**文件：** 修改 `markdown/TODO.md`
**依赖：** 无

**步骤：** 第 3 节“平台架构优化”条目按实际完成情况勾选：

- `[x]` 拆分 `fdidm_hardtest.py` 为可维护模块（信道估计、FEC、自适应、日志等）——已拆为 stream/channel/fec/gr_flow/fdidm_adaptive；
- `[x]` 抽取 OFDM / OTFS / AFDM / FDIDM 硬件后端的公共基类与统一接口——engine + 四个 transceiver/hardtest 兼容壳；
- `[x]` 统一配置定义与参数校验层——`core/config.py`；
- `[x]` 统一日志、状态快照与可观测性机制——`service/event_logger` + `diagnostics/*`；
- `[x]` 补充单元测试与链路级回归测试——pytest 66 用例；
- `[ ]` 优化运行时性能与稳定性（UHD overflow、实时调度、线程模型）——未做，留待实验室；
- `[ ]` 架构评审与重构后全链路验证——未做，留待实验室。

**验证：** `Get-Content markdown/TODO.md` 第 3 节前五项为 `[x]`。

### T10.4：全量回归

**文件：** 无新增
**依赖：** T10.1~T10.3

**步骤：**

1. `python -m pytest -q` → `66 passed`。
2. `python -m compileall -q waveform_sim scripts` → 退出码 0。
3. `python scripts/run_self_test.py` → 输出 `self test OK`。
4. GUI 离屏冒烟 `tabs: 7`。
5. 临时文件检查：`git status --short` 为空；仓库内无 `runs/` 目录残留（测试均用临时目录）。

**验证：** 五条检查全部通过。

### T10.5：提交

**文件：** `README.md`、`markdown/REFACTOR_CHECKLIST.md`、`markdown/TODO.md`
**依赖：** T10.4

**步骤：**

1. `git add README.md markdown/REFACTOR_CHECKLIST.md markdown/TODO.md`
2. `git commit -m "docs: 重构收尾：更新README、新增对照清单、更新TODO（阶段10）"`
3. `git status` → 工作区干净

**验证：** 提交存在；工作区干净。

## 执行顺序

```
T10.1 → T10.2 → T10.3 → T10.4 → T10.5
```

## 阶段 checkpoint

- T10.5 后向用户提交**全项目重构验收报告**：阶段 0~10 状态、66 个测试、真机回归遗留项。

