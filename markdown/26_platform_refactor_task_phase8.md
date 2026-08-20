# 平台工程化重构 Tasks（阶段 8：UI 收敛）

> 前置：阶段 7 已验收通过（pytest 63 passed）。
> 目标：四个大 UI 文件逐个瘦身——把**自包含的辅助类与纯静态助手**逐字搬移到 `waveform_sim/ui/` 子模块，原文件改为 import 引用，公开行为不变。
> 验证口径：无显示环境，采用"offscreen 直接构造页签 + compileall + 全量测试 + GUI tabs:7"；真机交互（点击/运行）留实验室。

## 文件清单

| 操作 | 文件 | 职责 |
|------|------|------|
| 新建 | `waveform_sim/ui/fdidm_plot_widgets.py` | `_PlotGridCell`、`_AlphaBetaSurfaceCanvas`（T8.1） |
| 修改 | `waveform_sim/ui/fdidm_hardware_test_tab.py` | 删除两个内部类，改为 import（T8.1） |
| 新建 | `waveform_sim/ui/compare_workers.py` | `_WaveformRunner`、`_ScanWorker`（T8.2） |
| 修改 | `waveform_sim/ui/waveform_compare_tab.py` | 删除两个内部类，改为 import（T8.2） |
| 新建 | `waveform_sim/ui/ui_utils.py` | 通用静态助手：`safe_float` / `format_metric` / `compute_spectrum` / `has_signal` / `extract_samp_rate`（T8.3~T8.4） |
| 修改 | `waveform_sim/ui/hardware_test_tab.py` | 删除对应静态方法，改用模块函数（T8.3） |
| 修改 | `waveform_sim/ui/fdidm_tab.py` | `_CurveSpec` 与静态助手搬移到 `ui_utils.py` / `fdidm_utils.py`（T8.4） |

## 任务

### T8.1：fdidm_hardware_test_tab.py 提取画布类

**文件：** 新建 `waveform_sim/ui/fdidm_plot_widgets.py`；修改 `waveform_sim/ui/fdidm_hardware_test_tab.py`
**依赖：** 无

**步骤：**

1. 新建 `waveform_sim/ui/fdidm_plot_widgets.py`，把 `fdidm_hardware_test_tab.py` 中 `_PlotGridCell`（约 35~64 行）与 `_AlphaBetaSurfaceCanvas`（约 66~732 行）**逐字搬入**，保留类名、docstring、全部方法；所需 import（PyQt5、pyqtgraph、`pyqtgraph.opengl` 相关）从原文件头部复制到新模块。
2. `fdidm_hardware_test_tab.py`：删除这两个内部类；文件头部加：

```python
from .fdidm_plot_widgets import _AlphaBetaSurfaceCanvas, _PlotGridCell
```

3. 若原文件头 import 中只有这两个类在用（如 `pyqtgraph.opengl`、`Qt3D` 相关），同步清理；拿不准就保留（不动不算错）。

**验证：**
1. `python -c "import os; os.environ['QT_QPA_PLATFORM']='offscreen'; from PyQt5.QtWidgets import QApplication; from waveform_sim.ui.fdidm_hardware_test_tab import FDIDMHardwareTestTab; app=QApplication([]); t=FDIDMHardwareTestTab(); print('tab ok'); t.close()"` 输出 `tab ok`。
2. `python -m pytest -q` → `63 passed`；`python -m compileall -q waveform_sim` 退出码 0。
3. GUI 离屏冒烟 `tabs: 7`。

**提交：** `git add waveform_sim/ui/fdidm_plot_widgets.py waveform_sim/ui/fdidm_hardware_test_tab.py` → `git commit -m "refactor: 提取 FDIDM 硬件页画布到 fdidm_plot_widgets.py（阶段8/1）"`。

### T8.2：waveform_compare_tab.py 提取 worker

**文件：** 新建 `waveform_sim/ui/compare_workers.py`；修改 `waveform_sim/ui/waveform_compare_tab.py`
**依赖：** T8.1

**步骤：**

1. 新建 `waveform_sim/ui/compare_workers.py`，把 `waveform_compare_tab.py` 中 `_WaveformRunner`（约 47~233 行）与 `_ScanWorker`（约 234~321 行）**逐字搬入**；所需 import（threading、numpy、transceiver/scan_axis 等）从原文件头复制。
2. `waveform_compare_tab.py`：删除这两个内部类；文件头部加：

```python
from .compare_workers import _ScanWorker, _WaveformRunner
```

**验证：** 同 T8.1（用 `WaveformCompareTab` 直接构造）；pytest `63 passed`；compileall 退出码 0；GUI `tabs: 7`。

**提交：** `git commit -m "refactor: 提取波形对比 worker 到 compare_workers.py（阶段8/2）"`。

### T8.3：hardware_test_tab.py 提取通用助手

**文件：** 新建 `waveform_sim/ui/ui_utils.py`；修改 `waveform_sim/ui/hardware_test_tab.py`
**依赖：** T8.2

**步骤：**

1. 新建 `waveform_sim/ui/ui_utils.py`，把以下**无 self 状态的静态助手**逐字搬为模块级函数（去掉 `@staticmethod` 与 `self` 参数）：
   - `_safe_float(value, default=0.0)` → `safe_float`
   - `_format_metric(value, fmt=".3f", fallback="nan")` → `format_metric`
   - `_has_signal(samples)` → `has_signal`
   - `_extract_samp_rate(status)` → `extract_samp_rate`
   - `_compute_spectrum(...)` → `compute_spectrum`（保留原签名与实现）
   文件头 import 需要 numpy 与（如用到）pyqtgraph。
2. `hardware_test_tab.py`：删除这些方法；**调用点改为模块函数引用**（如 `self._safe_float(x)` → `safe_float(x)`），并在文件头加 `from .ui_utils import compute_spectrum, extract_samp_rate, format_metric, has_signal, safe_float`。
3. 用 `Select-String` 确认无残留 `self._safe_float` / `self._format_metric` / `self._compute_spectrum` / `self._has_signal` / `self._extract_samp_rate` 调用。

**验证：** 同 T8.1（用 `HardwareTestTab` 直接构造）；pytest `63 passed`；compileall 退出码 0；GUI `tabs: 7`。

**提交：** `git commit -m "refactor: 提取硬件测评页通用助手到 ui_utils.py（阶段8/3）"`。

### T8.4：fdidm_tab.py 提取数据类与静态助手

**文件：** 新建 `waveform_sim/ui/fdidm_utils.py`；修改 `waveform_sim/ui/fdidm_tab.py`
**依赖：** T8.3

**步骤：**

1. 新建 `waveform_sim/ui/fdidm_utils.py`，把 `fdidm_tab.py` 中以下内容逐字搬入（保留 `@dataclass` / `@staticmethod` 语义）：
   - `_CurveSpec` dataclass（约 42~49 行）
   - `_alpha_ser_floor(num_frames, m_subcarriers, n_symbols)`（约 457~470 行）
   - `_merged_curve_specs(raw_specs)`（约 842~861 行）
   - `_copy_kwargs_with(base_kwargs, **updates)`（约 933~937 行）
   所需 import（dataclass、numpy 等）复制到新模块。
2. `fdidm_tab.py`：删除上述定义；文件头加：

```python
from .fdidm_utils import _alpha_ser_floor, _copy_kwargs_with, _merged_curve_specs, _CurveSpec
```

3. 调用点保持不变（`self._alpha_ser_floor(...)` 等——注意 `_alpha_ser_floor` 原是静态方法，搬成模块函数后调用点需改为直接函数调用；用 `Select-String` 逐个核对 `_alpha_ser_floor`、`_merged_curve_specs`、`_copy_kwargs_with` 的调用方式并统一）。

**验证：** 同 T8.1（用 `FDIDMTab` 直接构造）；pytest `63 passed`；compileall 退出码 0；GUI `tabs: 7`。

**提交：** `git commit -m "refactor: 提取 FDIDM 页数据类与助手到 fdidm_utils.py（阶段8/4）"`。

## 执行顺序

```
T8.1 → T8.2 → T8.3 → T8.4
```

## 阶段 checkpoint

- T8.4 后向用户报告各文件行数变化、pytest / GUI 输出与 4 个提交号；确认后再拆解阶段 9（诊断 / 脚本 / CI）。

