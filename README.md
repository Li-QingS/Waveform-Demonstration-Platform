# FDIDM 软波形自适应演示平台

基于论文 *Fractional Dual Index Division Multiplexing: A Soft Waveform Design Toward Integrated Satellite-Terrestrial Networks* 开发的波形仿真与 USRP 硬件验证平台，支持 FDIDM / OFDM / OTFS / AFDM 四种波形的软件仿真、性能对比和 USRP B210 真机链路测试。

## 核心能力

- 四种波形（FDIDM / OFDM / OTFS / AFDM）软件仿真：统一引擎 `LinkSimulator` + Monte-Carlo 性能对比扫描
- USRP B210（N210 / X310）硬件链路测试与 RF 安全保护
- FDIDM α/β 预测式自适应闭环：硬件侧（`hardware/fdidm_hardtest.py`）与仿真侧（`simulation/fdidm_adaptive.py`）均实现“CSI 快照 → 理论 SER 搜索 → 稳定判定 → 冷却 → 自动应用”
- FDIDM 仿真页自适应过程可视化：α/β 轨迹、预测 SER 对比、切换标注、状态文本
- 时变信道模型（仿真页可切换，详见下文）：固定信道 / 动态块衰落 / 帧内快时变 / 连续多普勒
- 仿真图右下角“实时性能随时间变化”图：固定滚动时间窗、Y 轴按可见数据自动拟合、3 s 滑动平均、增量刷新
- 自适应过程页 SER-SNR 对比图：固定信道、每个 SNR 点信道条件完全相同（只有 SNR 变化），并自动挑选能体现 FDIDM 理论增益的信道实现
- 工程化：统一配置、统一引擎、实验 artifact + JSONL 事件日志、诊断脚本、行为基线测试与 CI

## 时变信道模式

FDIDM 仿真页“时变模式”下拉提供四种信道演化方式，参数基于真实 LEO 星地链路（3GPP TR 38.811）：

- **固定信道**：整个运行期间保持同一 H_TF。
- **动态块衰落**：路径表（时延/幅度/多普勒）固定，每个相干块对路径相位做一次均值回归（AR(1)）小幅游走；相邻块信道相关，SER 随时间平稳变化而不是独立跳变。
- **帧内快时变**：每帧重新生成信道实现，并在帧内符号间做 AR(1) 增益演化。
- **连续多普勒**：按各径真实多普勒逐帧旋转路径相位 exp(j2π·fD·T_frame)，信道连续演化、种子不变、可复现。

参考量级：LEO 轨道速度约 7.8 km/s（28080 km/h）；20 GHz 载频下最大多普勒约 ±480 kHz、多普勒变化率可达 −5.44 kHz/s；默认径向系数 0.10 对应约 52 kHz 残余多普勒，此时相干时间仅数微秒——信道在帧级尺度上变化很快是物理事实，演示页用上述模式分别展示慢变/快变场景。

## 目录结构

```text
waveform_sim/
  main.py                       # 程序入口（PyQt5 主窗口）
  core/                         # 工程化内核
    config.py                   #   统一配置（Waveform/Adaptive/Hardware/Experiment）
    engine.py                   #   统一链路引擎 LinkSimulator
    modem.py / transforms.py / waveforms.py / metrics.py
  simulation/                   # 软件仿真后端（transceiver 兼容壳）
    simple_fdidm_rx.py / simple_ofdm_rx.py / simple_otfs_rx.py / simple_afdm_rx.py
    fdidm_adaptive.py           #   FDIDM 仿真侧预测式自适应（mixin + 搜索内核）
    compare_scan_backend.py     #   三波形 Monte-Carlo 对比扫描
  hardware/                     # 硬件后端（兼容壳 + 抽象层）
    fdidm_hardtest.py / ofdm_hardtest.py / otfs_hardtest.py / afdm_hardtest.py
    transport.py / rf_safety.py / device_manager.py / iq_replay.py
    stream.py / channel.py / fec.py / gr_flow.py / fdidm_adaptive.py
  service/                      # 实验记录：experiment_service / artifact_writer / event_logger / run_state
  diagnostics/                  # health_check / report_exporter / snapshot
  ui/                           # GUI 页签与公共组件
    main_window.py / fdidm_tab.py / fdidm_plot_widgets.py
    fdidm_adaptive_widgets.py / base_waveform_tab.py / compare_workers.py / ui_utils.py
markdown/                       # 重构与自适应 spec/plan/task/checklist、REFACTOR_CHECKLIST、TODO
scripts/                        # check_environment / check_usrp / run_self_test
tests/                          # pytest 行为基线 / 回归测试
.github/workflows/ci.yml        # 纯 Python 核心 CI
```

## 运行

```powershell
cd waveform_sim
python main.py
```

依赖：Python 3.11+、NumPy、PyQt5、pyqtgraph。硬件链路还需要 GNU Radio / UHD，以及 USRP B210（或 N210 / X310）设备；没有硬件时，纯仿真页面和波形对比页面仍可正常运行。

无显示环境验证：

```powershell
python scripts/check_environment.py   # 环境健康检查
python scripts/run_self_test.py       # FDIDM 自检（出帧 + 实验 artifact）
python -m pytest -q                   # 行为基线 / 回归测试
```

## 测试

当前行为基线 88 个用例，覆盖统一引擎、波形变换、服务层、硬件抽象与 FDIDM 仿真自适应（含信道块推进、连续多普勒、并发参数修改、时间图平滑、SER-SNR 固定信道对比等回归场景）。CI（`.github/workflows/ci.yml`）仅安装 NumPy + pytest，UI 用例自动跳过，后端用例全部运行。

## 文档

`markdown/` 下包含工程化重构流程文档与 FDIDM 仿真侧自适应文档：
- `06_platform_refactor_spec.md`：重构需求与验收标准
- `07_platform_refactor_plan.md`：重构架构与分阶段实施计划
- `08_platform_refactor_task_phase0_1.md` / `09_platform_refactor_checklist_phase0_1.md`：阶段 0~1 任务清单与验收清单
- `10_platform_refactor_task_phase2.md` ~ `31_platform_refactor_checklist_phase10.md`：阶段 2~10 任务清单与验收清单
- `REFACTOR_CHECKLIST.md`：重构对照清单（阶段完成状态）
- `TODO.md`：平台待办清单（自适应、对比展示、架构优化）
- `32_fdidm_sim_adaptive_spec.md` / `33_fdidm_sim_adaptive_plan.md` / `34_fdidm_sim_adaptive_task.md` / `35_fdidm_sim_adaptive_checklist.md`：FDIDM 仿真侧预测式自适应（spec/plan/task/checklist）
