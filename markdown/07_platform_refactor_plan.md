# 平台工程化重构 Plan

> 本文档为 Spec 驱动开发流程的阶段二产物，需在 spec.md 获批后生效，未获批前不进入 task 拆解。

## 架构概览

与交付版对齐的包结构，但**继续使用现有 `waveform_sim` 包名**，工程化内核作为其子包：

```
waveform_sim/
├── __init__.py       新增，使其成为正式包
├── main.py           入口保持 cd waveform_sim && python main.py
├── core/             config · modem · transforms · waveforms · engine · adaptive · metrics
├── hardware/         transport · rf_safety · device_manager · iq_replay（新增）
│                     + 现有 *_hardtest.py（逐步改为兼容壳）
├── service/          experiment_service · artifact_writer · event_logger · run_state
├── diagnostics/      health_check · report_exporter · snapshot
├── simulation/       现有文件，逐步改为薄兼容壳
└── ui/               现有文件，逐步收敛公共代码

tests/               行为基线 + 回归测试
scripts/             环境检查 / USRP 探测 / 自测
.github/workflows/   CI
pyproject.toml · requirements.txt · environment.yml
```

核心调用链（重构后）：

```
UI 页签 → Transceiver / HardwareTest（兼容壳）
        → waveform_sim.core.engine.LinkSimulator（统一引擎，后台线程）
        → Waveform / Transforms / Channel（沿用现有算法）
        → Metrics
        → waveform_sim.service.experiment_service（可选）→ runs/<run_id>/...
```

## 关键技术决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 新包命名 | 保持 `waveform_sim`，内核放 `waveform_sim/core`、`service`、`diagnostics` 等子包 | 沿用现有包名，避免与交付版命名混同；对旧 import 影响最小 |
| 算法处理 | 引擎内复用现有矩阵 / TDL 信道实现，不采用交付版简化模型 | 满足 N1 行为保持 |
| 旧类处理 | 保留类名，内部改为继承 / 委托新内核 | 满足 N2 兼容性 |
| 迁移节奏 | 一个波形 / 一个文件为一轮，每轮独立提交 | 满足 N3 增量、可回退 |
| 删除策略 | 旧文件仅在兼容壳验证通过后移除或归档 | 满足 N4 |
| 真机逻辑 | GNU Radio / UHD 流式代码先搬移、不删除 | 满足 N5 |
| 测试策略 | 阶段 0 先锁行为基线，之后每阶段跑全量 pytest | 用证据防止回归 |

## 分阶段实施清单

### 阶段 0：基线锁定（先有测试网）

**改动：**
- 新建 `tests/`，编写四波形固定种子行为锁定测试（单帧 BER/EVM 快照断言）。
- 新建 `pyproject.toml`（pytest 配置 + 依赖声明）、`requirements.txt`、`environment.yml`。
- 新建 `scripts/check_environment.py`。

**验证：** `pytest -q` 通过；`python -m compileall waveform_sim` 通过；现有入口不受影响。

**提交：** 1 个（chore）。

### 阶段 1：统一配置层（只增不改）

**改动：**
- 新增 `waveform_sim/__init__.py`，新建 `waveform_sim/core/config.py`：
  - `WaveformConfig`（覆盖四波形现有参数 + 旧名映射，如 `ebn0_db`→`snr_db`、`fc_hz`→`center_freq_hz`、`channel_seed`→`seed`、`decoder`→`detector`）；
  - `AdaptiveConfig`（现有搜索步长、目标、范围、重复评估等参数）；
  - `HardwareConfig`（现有 hardtest 硬件参数）；
  - `ExperimentConfig`（运行目录、operator、tags 等）。
- 提供 `normalized()` / `to_dict()` / `from_dict()` / JSON 存取。

**验证：** 配置序列化/反序列化单测；旧代码零改动、零影响。

**提交：** 1 个。

### 阶段 2：公共 DSP 模块（只增不改）

**改动：**
- 新建 `waveform_sim/core/modem.py`、`transforms.py`、`waveforms.py`、`metrics.py`。
- 实现以现有 `simple_*_rx.py` 的映射/变换为准；若各文件实现有细微差异，用一致性测试驱动并保留兼容开关。

**验证：** 固定输入下新旧 QAM 映射、变换输出一致（bit-exact 或误差 < 1e-9）。

**提交：** 1 个。

### 阶段 3：FDIDM 接入统一引擎

**改动：**
- 新建 `waveform_sim/core/engine.py`（`LinkSimulator`：单帧链路、后台线程、`update_config` / `set_indices` / `get_plot_data` / start / stop）。
- `simulation/simple_fdidm_rx.py` 的 `FDIDMTransceiver` 改为继承 / 委托 `LinkSimulator`，公开方法签名保持不变。

**验证：** FDIDM 固定种子单帧指标与阶段 0 基线一致；GUI FDIDM 页签冒烟。

**提交：** 1~2 个。

### 阶段 4：OFDM / OTFS / AFDM 接入引擎（每波形一轮）

**改动：** 依次将 `simple_ofdm_rx.py`、`simple_otfs_rx.py`、`simple_afdm_rx.py` 改为薄兼容壳。

**验证：** 每轮对应波形固定种子指标与基线一致；对应页签冒烟。

**提交：** 3 轮，每轮 1 个。

### 阶段 5：硬件抽象层（只增不改）

**改动：**
- 新建 `waveform_sim/hardware/transport.py`（simulated-loopback；UHD 占位并保留真机接入点）。
- 新建 `waveform_sim/hardware/rf_safety.py`（loopback/OTA 增益、衰减器、频段白名单策略）。
- 新建 `waveform_sim/hardware/device_manager.py`（`uhd_find_devices` 探测）。
- 新建 `waveform_sim/hardware/iq_replay.py`（`.npy` / complex64 回放）。

**验证：** rf_safety 单测、transport loopback 单测。

**提交：** 1 个。

### 阶段 6：硬件后端兼容壳（每文件一轮）

**改动：**
- 将 `hardware/*_hardtest.py` 逐个瘦身：GNU Radio 流式逻辑搬入 `waveform_sim/hardware/`（如 `gr_flow.py`，保留功能），类改为兼容壳，公开方法不变。
- 顺序建议：AFDM（54 KB）→ OFDM（65 KB）→ OTFS（79 KB）→ FDIDM（308 KB，拆分为信道/编解码/GNU Radio/自适应/性能面多个子步骤）。

**验证：** 每轮 simulated-loopback 冒烟 + 固定种子行为对比；FDIDM 按子步骤逐次提交。

**提交：** 4 轮，每轮 1~3 个。

### 阶段 7：服务层（实验记录）

**改动：**
- 新建 `waveform_sim/service/experiment_service.py`、`artifact_writer.py`、`event_logger.py`、`run_state.py`。
- `LinkSimulator` 挂接 `experiment_service`（默认关闭，UI 可开关）。

**验证：** 一次仿真生成 `runs/<run_id>/config.json`、`events.jsonl`、`metrics.csv`、`report.md`。

**提交：** 1 个。

### 阶段 8：UI 收敛（每文件一轮）

**改动：** 逐个瘦身 `ui/fdidm_tab.py`、`ui/hardware_test_tab.py`、`ui/waveform_compare_tab.py`、`ui/fdidm_hardware_test_tab.py`，公共代码收进 `base_waveform_tab.py`。

**验证：** 每轮 `compileall` 通过 + GUI 启动冒烟（有显示环境时）。

**提交：** 4 轮，每轮 1 个。

### 阶段 9：诊断 / 脚本 / CI

**改动：**
- 新建 `waveform_sim/diagnostics/health_check.py`、`report_exporter.py`、`snapshot.py`。
- 新建 `scripts/check_usrp.py`、`scripts/run_self_test.py`。
- 新建 `.github/workflows/ci.yml`。

**验证：** 三个脚本可运行；CI 本地等价命令（pytest + compileall）通过。

**提交：** 1 个。

### 阶段 10：文档与收尾

**改动：**
- 更新 README 与 `markdown/05_platform_introduction.md`；新增重构对照清单文档。
- 全量回归：pytest、compileall、自测、GUI 冒烟；确认可移除的临时文件。

**验证：** AC1~AC6 全部通过。

**提交：** 1 个。

## 模块交互与依赖

- `waveform_sim/ui/*` → `waveform_sim/simulation/*` 与 `waveform_sim/hardware/*`（兼容壳）→ `waveform_sim/core/engine.py`。
- `engine.py` → `waveforms.py` → `transforms.py` / `modem.py` → `metrics.py`。
- `engine.py` → `adaptive.py`（异步调优）→ 调优完成回调应用最优 alpha/beta。
- `engine.py` → `service/experiment_service.py`（可选）→ 写 run artifact。
- `hardware/*HardwareTest`（兼容壳）→ `hardware/transport.py` + `hardware/rf_safety.py` + `hardware/device_manager.py`。

依赖方向保持单向：`waveform_sim` 的 UI / 仿真 / 硬件层依赖 `core`、`service` 等子包；`core` 不反向依赖旧的 `simulation` / `hardware` 大文件（搬移时先复制后切换）。

## 风险与对策

| 风险 | 对策 |
|------|------|
| 308 KB 的 FDIDMHardwareTest 拆解难度大 | 阶段 0 先锁行为，阶段 6 按功能块拆成多个子步骤，每个子步骤可提交可回退 |
| 各文件 QAM 映射/信道实现存在细微差异 | 一致性测试驱动，必要时保留兼容开关，不强行统一数值 |
| 无显示环境无法冒烟 GUI | 以 compileall + 后端行为测试 + 离屏 Qt 测试兜底，GUI 冒烟留到有显示环境 |
| 中途需求变化 | 每阶段独立提交，回退粒度小；spec 变更需重新审批 |
