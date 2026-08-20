# FDIDM 软波形自适应演示平台

基于论文 *Fractional Dual Index Division Multiplexing: A Soft Waveform Design Toward Integrated Satellite-Terrestrial Networks* 开发的波形仿真与 USRP 硬件验证平台，支持 FDIDM / OFDM / OTFS / AFDM 四种波形的软件仿真、性能对比和 USRP B210 真机链路测试。

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

## 文档

`markdown/` 下包含工程化重构流程文档：

- `06_platform_refactor_spec.md`：重构需求与验收标准
- `07_platform_refactor_plan.md`：重构架构与分阶段实施计划
- `08_platform_refactor_task_phase0_1.md` / `09_platform_refactor_checklist_phase0_1.md`：阶段 0~1 任务清单与验收清单
- `10_platform_refactor_task_phase2.md` ~ `31_platform_refactor_checklist_phase10.md`：阶段 2~10 任务清单与验收清单
- `REFACTOR_CHECKLIST.md`：重构对照清单（阶段完成状态）
- `TODO.md`：平台待办清单（自适应、对比展示、架构优化）
