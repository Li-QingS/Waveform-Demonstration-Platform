# 平台工程化重构 Spec

> 本文档为软波谱 Spec 驱动开发流程的阶段一产物，未获用户批准前不进入 plan/task 阶段。

## 背景

- 当前仓库 `E:\USRP\Waveform-Demonstration-Platform` 是 FDIDM/OFDM/OTFS/AFDM 波形演示平台，功能完整，但代码以大型单体文件为主：
  - `waveform_sim/hardware/fdidm_hardtest.py` 约 308 KB / 5700 行；
  - `waveform_sim/ui/fdidm_hardware_test_tab.py` 约 101 KB；
  - `waveform_sim/simulation/simple_fdidm_rx.py` 约 84 KB；
  - 其余 hardtest / simulation / ui 文件多在 20~80 KB。
- 交付版 `E:\USRP\260820\waveform_platform_delivery\...` 提供了一套工程化内核参考架构：统一配置、统一波形引擎、实验 run artifact、结构化事件日志、RF safety、环境检查、IQ replay、自适应调优收敛，以及测试/CI/脚本。
- 本次目标：借鉴交付版的架构思路，但**保持本仓库算法行为不变**，以分阶段、低风险的方式完成结构性重构。

## 目标

- G1：在现有 `waveform_sim` 包内建立工程化内核子包（core / hardware / service / diagnostics），不引入新顶层包名。
- G2：现有运行入口与公开类名、方法签名保持兼容（`FDIDMTransceiver`、`OFDMTransceiver`、`OTFSTransceiver`、`AFDMTransceiver`、各 `*HardwareTest`、`cd waveform_sim && python main.py`）。
- G3：重构前后在相同随机种子下仿真指标（BER / EVM / SER）行为一致。
- G4：每个阶段独立提交、可运行、可回退，不一次性改完。
- G5：建立测试基线与 CI，防止回归。

## 功能需求

- F1：统一配置模型。将各 transceiver / hardtest 的构造与运行参数收敛为 dataclass，并保留旧参数名兼容映射。
- F2：统一波形接口。四类波形提供统一的 `modulate()` / `demodulate()` 入口，内部实现沿用现有算法（不做简化重写）。
- F3：统一引擎。提供可后台线程运行的链路引擎，供 GUI、仿真后端、硬件后端共享。
- F4：硬件抽象。transport（simulated-loopback / UHD）、RF safety 策略、设备探测、IQ replay。
- F5：实验记录。可选生成 `runs/<run_id>/config.json`、`events.jsonl`、`metrics.csv`、`report.md`。
- F6：自适应调优收敛。现有 alpha/beta 搜索逻辑统一为可复现优化器，保留现有评估方式与界面交互。
- F7：测试 / CI / 脚本。pytest 行为基线、环境检查脚本、自测脚本、GitHub Actions。
- F8：UI 收敛。把重复 UI 代码收进公共组件（BaseWaveformTab 等），不改视觉风格与页签结构。

## 非功能需求

- N1：行为保持。固定种子下，重构前后单帧 BER/EVM/SER 一致（bit-exact，或数值误差 < 1e-9 并附测试断言）。
- N2：兼容性。所有公开类名与常用方法签名不变；`cd waveform_sim && python main.py` 可启动。
- N3：增量。每阶段一个或多个提交，每阶段结束时项目可运行、可验证、可回退。
- N4：不提前删除。旧文件只在对应兼容壳验证通过后才移除或归档。
- N5：真机代码保留。UHD / GNU Radio 流式逻辑在早期阶段不删除，仅逐步搬移。

## 不做的事

- 不改写 / 简化 DSP 算法（不采用交付版的简化信道与变换模型）。
- 不新增业务功能（实验记录属于轻量新增，纳入本次范围）。
- 不做性能优化。
- 不改 UI 视觉风格与页签命名。

## 验收标准

- AC1：每个阶段有对应 git 提交，阶段结束时工作区干净。
- AC2：固定种子下四波形重构前后单帧指标一致（由测试断言）。
- AC3：公开类名与方法签名不变（由测试断言）。
- AC4：`pytest` 全绿；`python -m compileall` 通过；自测脚本通过。
- AC5：GUI 可启动、7 个页签可打开（在有显示环境时验证）。
- AC6：一次仿真可生成完整 run artifact。
