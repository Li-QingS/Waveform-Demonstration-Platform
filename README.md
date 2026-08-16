# FDIDM 软波形自适应演示平台

基于论文 *Fractional Dual Index Division Multiplexing: A Soft Waveform Design Toward Integrated Satellite-Terrestrial Networks* 开发的波形仿真与 USRP 硬件验证平台，支持 FDIDM / OFDM / OTFS / AFDM 四种波形的软件仿真、性能对比和 USRP B210 真机链路测试。

## 目录结构

```text
waveform_sim/
  main.py                       # 程序入口（PyQt5 主窗口）
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
markdown/                       # 设计、审查与调优文档
```

## 运行

```powershell
cd waveform_sim
python main.py
```

依赖：Python 3.11+、NumPy、PyQt5、pyqtgraph。硬件链路还需要 GNU Radio / UHD，以及 USRP B210（或 N210 / X310）设备；没有硬件时，纯仿真页面和波形对比页面仍可正常运行。

## 文档

`markdown/` 下包含面向真实硬件链路的专业文档：

- `01_code_review_fdidm_platform_professional.md`：代码审查与改进建议
- `02_feature_hardware_logging_observability.md`：硬件链路观测与统一日志设计
- `03_tuning_hardware_rf_link.md`：硬件链路稳定性与可调试性优化
- `04_system_design_fdidm_hardware_platform.md`：平台模块化重构设计
- `fdidm_v32_system_analysis.md`：多普勒/时延极限与 v32 收发流程分析
