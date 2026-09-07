# FDIDM 硬件页面板重排 Plan（2026-09-07）

## 右栏结构
```
QVBoxLayout(spacing 6)
├─ comparison_result_group（max ~112px）
│   行0: [开始观测][结束观测][导出报告] + observation_state_label(stretch)
│   行1: before │ after │ improvement（18px 加粗，单行）
│   行2: badge │ note
├─ plot_tabs = QTabWidget（stretch 1）
│   ├ "α/β 性能面" → ab_surface_panel
│   ├ "RX 频谱" → rx_spectrum_plot
│   ├ "EVM 曲线" → evm_plot
│   └ "接收星座" → constellation_plot
├─ timeline_panel（max ~195px）：QHBoxLayout perf(3) | ab(2)，保留 xLink
└─ text_panel（max ~120px）：解调状态 + 发送/接收文本（~66px 高）
```

## 技术决策
| 决策点 | 选择 | 理由 |
|--------|------|------|
| 2×2 网格与 _PlotGridCell 包装 | 停用包装，图直接进页签；类保留不删 | 页签单幅满幅；避免影响他处导入 |
| 时间轴上下堆叠 → 左右并排 | perf 3 : ab 2 | 同高度下每图高度 ~90→190px |
| 时间轴黑底 | 纳入 _init_plot_style 循环 | 与主图同套样式 |
| OpenGL sizeHint 防御 | 保留 Ignored 策略 | 沿用现有手段 |
| 控件对象名 | 全部不变 | 测试与信号零改动（N2） |

## 左栏放宽
scroll min 360 / max 470；combo 上限 210→260；spin 上限 116→140；行编辑 68–116→80–140；spacing 5→6、外边距 6→8。

## 文件组织
```
waveform_sim/ui/fdidm_hardware_test_tab.py   修改
tests/                                       原则不改（对象名不变）
```
