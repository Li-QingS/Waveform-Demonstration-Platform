# FDIDM Markdown 文件说明

本目录包含针对当前 FDIDM 软件平台生成的四类 Markdown 文件：

1. `01_代码审查_FDIDM实验平台.md`  
   面向代码审查，覆盖安全性、性能、可读性、最佳实践和论文/系统一致性。

2. `02_功能实现_AlphaBeta自动推荐模块.md`  
   面向后续功能开发，建议实现 α/β 自动推荐与参数扫描模块。

3. `03_调优优化_B210溢出与星座发散.md`  
   面向运行调优，聚焦 USRP B210 溢出、星座图发散、GUI 卡顿和 full-H_TF 开销。

4. `04_系统设计_FDIDM实验平台模块化重构.md`  
   面向平台工程化设计，给出模块划分、数据模型、API 和分阶段重构路线。

这些文件不是泛化模板，而是已经结合以下材料进行定制：

- 论文：`Fractional_Dual_Index_Division_Multiplexing_A_Soft_Waveform_Design_Toward_Integrated_SatelliteTerrestrial_Networks.pdf`
- 后端代码：`fdidm_hardtest.py`
- GUI 代码：`fdidm_hardware_test_tab.py`

建议使用顺序：

1. 先看代码审查文件，确定优先级。
2. 再看系统设计文件，决定是否重构。
3. 如果要新增论文特色功能，使用 α/β 自动推荐文件。
4. 如果当前硬件运行不稳，优先使用调优优化文件。
