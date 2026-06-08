# FDIDM Markdown 文件说明

本目录包含重新生成的专业版 Markdown 文件，已根据以下要求调整：

1. 不再建议 alpha/beta 扫描推荐；
2. 不把纯软件仿真作为项目主线；
3. 强调真实 USRP/RF 硬件链路落地；
4. 强调代码重构层次结构；
5. 强调每个模块、每个关键流程、每个异常都要有日志输出；
6. 面向后期调试、优化和工程维护。

## 文件列表

- `01_code_review_fdidm_platform_professional.md`  
  FDIDM 硬件验证平台代码审查，包含关键问题、改进建议、代码示例、优点和整体质量评估。

- `02_feature_hardware_logging_observability.md`  
  功能实现模板，主题为“硬件链路观测与统一日志系统”。

- `03_tuning_hardware_rf_link.md`  
  调优优化模板，主题为“硬件链路稳定性与可调试性优化”。

- `04_system_design_fdidm_hardware_platform.md`  
  系统设计模板，主题为“FDIDM 硬件验证平台专业落地设计”。

## 使用建议

建议先看 `04_system_design_fdidm_hardware_platform.md`，确定整体架构；再看 `01_code_review_fdidm_platform_professional.md`，明确当前代码问题；随后用 `02_feature_hardware_logging_observability.md` 和 `03_tuning_hardware_rf_link.md` 指导下一步开发和调试。
