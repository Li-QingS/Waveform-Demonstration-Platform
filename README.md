# FDIDM 软波形自适应演示平台

基于论文 *Fractional Dual Index Division Multiplexing: A Soft Waveform Design Toward Integrated Satellite-Terrestrial Networks* 开发的波形仿真与 USRP 硬件验证平台，支持 FDIDM / OFDM / OTFS / AFDM 四种波形的软件仿真、性能对比和 USRP B210 真机链路测试。

## 目录结构

```text
waveform_sim/
  main.py                       # 程序入口（PyQt5 主窗口）
  core/                         # 工程化内核（统一配置，后续加入引擎/自适应/指标）
  ui/                           # GUI：FDIDM 仿真页、OFDM/OTFS/AFDM 仿真页、
                                # FDIDM 硬件验证页、通用硬件测评页、波形对比页
  simulation/                   # 纯软件仿真后端
    simple_fdidm_rx.py          #   FDIDM 矩阵化演示链路
    simple_ofdm_rx.py           #   OFDM 接收链路
    simple_otfs_rx.py           #   OTFS 接收链路
    simple_afdm_rx.py           #   AFDM 接收链路
    compare_scan_backend.py     #   统一口径的三波形 Monte-Carlo 对比扫描
  hardware/                     # USRP 硬件后端
    fdidm_hardtest.py           #   FDIDM 硬件链路核心（同步/CFO/信道估计/均衡/FEC/alpha-beta自适应）
    ofdm_hardtest.py            #   OFDM 硬件测试
    otfs_hardtest.py            #   OTFS 硬件测试
    afdm_hardtest.py            #   AFDM 硬件测试
markdown/                       # 工程化重构与设计文档
scripts/                        # 环境检查等运维脚本
tests/                          # 行为基线 / 回归测试
```

## 运行

```powershell
cd waveform_sim
python main.py
```

依赖：Python 3.11+、NumPy、PyQt5、pyqtgraph。硬件链路还需要 GNU Radio / UHD，以及 USRP B210（或 N210 / X310）设备；没有硬件时，纯仿真页面和波形对比页面仍可正常运行。

## 文档

`markdown/` 下包含工程化重构流程文档：

- `06_platform_refactor_spec.md`：重构需求与验收标准
- `07_platform_refactor_plan.md`：重构架构与分阶段实施计划
- `08_platform_refactor_task_phase0_1.md` / `09_platform_refactor_checklist_phase0_1.md`：阶段 0~1 任务清单与验收清单
- `TODO.md`：平台待办清单（自适应、对比展示、架构优化）
