# 平台工程化重构 Tasks（阶段 6c：FDIDM 自适应/性能面与 GNU Radio 块）

> 前置：阶段 6b 已验收通过（pytest 56 passed，`fdidm_hardtest.py` 5282 行）。
> 范围：
> 1. 自适应 + 性能面（约 780 行，两个非连续块）搬入 `fdidm_adaptive.py` 的 `FDIDMAdaptiveMixin`；
> 2. 两个 GNU Radio 块工厂（`TDLChannelBlock`、`RXRingSink`）搬入 `gr_flow.py`（保留懒加载，不破坏无 GNU Radio 环境的导入）。
> 说明：本机无 USRP，真机行为无法运行验证；搬移采用"逐字搬移 + 编译/导入/纯函数测试/MRO 检查"，真机回归留到实验室环境。

## 文件清单

| 操作 | 文件 | 职责 |
|------|------|------|
| 新建 | `waveform_sim/hardware/fdidm_adaptive.py` | `FDIDMAdaptiveMixin`（自适应 + 性能面，T6c.1） |
| 修改 | `waveform_sim/hardware/fdidm_hardtest.py` | 删除两个方法块；类继承 `FDIDMAdaptiveMixin`（T6c.1） |
| 新建 | `tests/test_hardware_adaptive.py` | 自适应纯函数测试（4 用例，T6c.1） |
| 新建 | `waveform_sim/hardware/gr_flow.py` | GNU Radio 块工厂（T6c.2） |
| 修改 | `waveform_sim/hardware/fdidm_hardtest.py` | `_make_tdl_channel_block` / `_make_rx_ring_sink` 改薄（T6c.2） |

## 任务

### T6c.1：搬移自适应/性能面到 FDIDMAdaptiveMixin

**文件：** 新建 `waveform_sim/hardware/fdidm_adaptive.py`；修改 `waveform_sim/hardware/fdidm_hardtest.py`；新建 `tests/test_hardware_adaptive.py`
**依赖：** 无

**步骤：**

1. 因两个方法块约 780 行且非连续，**用机械脚本搬移**（属于批量机械重写，不用手打 1500 行补丁以免抄错）。脚本逻辑：
   - 读 `fdidm_hardtest.py` 全部行；
   - Block1 = 段注释 `# Paper-guided channel-adaptive alpha/beta optimization` 所在的 `# ====...` 行起，到 `get_alpha_beta_adaptation_status` 方法结束（其下一个方法是 `_prepare_constellation_points`，往前去尾随空行）；
   - Block2 = 段注释 `# Alpha/Beta performance surface for UI demonstration` 所在的 `# ====...` 行起，到 `get_alpha_beta_performance_surface` 方法结束（其下一个方法是 `get_debug_snapshot`，往前去尾随空行）；
   - 生成 `fdidm_adaptive.py`：文件头（imports：`math`、`threading`、`time`、`numpy`、`typing` 的 `Any/Dict/List/Optional/Tuple`）+ `class FDIDMAdaptiveMixin:` + Block1 + Block2（方法体原样，装饰器保留，缩进不变——都是类内 4 空格）；
   - 从 `fdidm_hardtest.py` 删除 Block1、Block2（从下往上删，保持行号有效）。
2. 用 apply_patch 做两处小改动：
   - 头部加 `from .fdidm_adaptive import FDIDMAdaptiveMixin`；
   - 类定义 `class _LegacyFDIDMHardwareTest(FECMixin):` → `class _LegacyFDIDMHardwareTest(FECMixin, FDIDMAdaptiveMixin):`。
3. 新建 `tests/test_hardware_adaptive.py`：

```python
"""FDIDM 自适应纯函数测试（阶段 6c）。"""
import numpy as np

from waveform_sim.hardware.fdidm_adaptive import FDIDMAdaptiveMixin


def test_adaptive_qam_order():
    assert FDIDMAdaptiveMixin._adaptive_qam_order("QPSK") == 4
    assert FDIDMAdaptiveMixin._adaptive_qam_order("16QAM") == 16
    assert FDIDMAdaptiveMixin._adaptive_qam_order("64QAM") == 64
    assert FDIDMAdaptiveMixin._adaptive_qam_order("UNKNOWN") == 4


def test_adaptive_qfunc():
    q0 = float(FDIDMAdaptiveMixin._adaptive_qfunc(np.array([0.0]))[0])
    assert abs(q0 - 0.5) < 1e-9
    q = float(FDIDMAdaptiveMixin._adaptive_qfunc(np.array([1.96]))[0])
    assert 0.02 < q < 0.03


def test_finite_float_or_nan():
    assert FDIDMAdaptiveMixin._finite_float_or_nan(3.0) == 3.0
    assert np.isnan(FDIDMAdaptiveMixin._finite_float_or_nan(float("nan")))
    assert np.isnan(FDIDMAdaptiveMixin._finite_float_or_nan(float("inf")))


def test_adaptive_ser_from_symbol_nsr():
    obj = object.__new__(FDIDMAdaptiveMixin)
    ser = obj._adaptive_ser_from_symbol_nsr(np.array([100.0]), "QPSK")
    assert 0.0 <= float(ser) <= 1.0
```

> 说明：`_adaptive_ser_from_symbol_nsr` 内部经 `self._adaptive_qfunc`（staticmethod）调用，`object.__new__` 空实例即可解析；有状态方法（worker/性能面记录）不做运行测试，靠逐字搬移 + 编译/导入保证。

**验证：**
1. `python -m pytest tests/test_hardware_adaptive.py -q` → `4 passed`。
2. 方法归属检查：Block1/Block2 的全部方法名在 `fdidm_adaptive.py` 中，且不再出现在 `fdidm_hardtest.py`（用 `Select-String` 抽查 `_alpha_beta_adaptation_worker`、`get_alpha_beta_performance_surface` 等）。
3. `python -m pytest -q` → `60 passed`；`python -m compileall -q waveform_sim` 退出码 0；`python -c "import waveform_sim.hardware.fdidm_hardtest as m; print([c.__name__ for c in m._LegacyFDIDMHardwareTest.__mro__])"` 输出含 `FECMixin` 与 `FDIDMAdaptiveMixin`。

**提交：** `git add waveform_sim/hardware/fdidm_adaptive.py waveform_sim/hardware/fdidm_hardtest.py tests/test_hardware_adaptive.py` → `git commit -m "refactor: 提取自适应/性能面到 fdidm_adaptive.py（阶段6c/1）"`。

### T6c.2：gr_flow.py 提取 GNU Radio 块工厂

**文件：** 新建 `waveform_sim/hardware/gr_flow.py`；修改 `waveform_sim/hardware/fdidm_hardtest.py`
**依赖：** T6c.1

**步骤：**

1. 新建 `waveform_sim/hardware/gr_flow.py`（**懒加载 gnuradio**，保证无 GNU Radio 机器上导入不失败）：

```python
"""GNU Radio 流式块工厂（从 fdidm_hardtest.py 搬移，阶段 6c）。

gnuradio 在工厂函数内部懒加载，模块本身可在无 GNU Radio 环境导入。
"""
from __future__ import annotations

import threading

import numpy as np


def make_tdl_channel_block(channel):
    """Create a GNU Radio Python sync_block wrapping NTNTDLChannel."""
    from gnuradio import gr

    class _TDLChannelBlock(gr.sync_block):
        def __init__(self):
            gr.sync_block.__init__(self, name="ntn_tdl_channel_v34", in_sig=[np.complex64], out_sig=[np.complex64])
            self.channel = channel
            self._channel_lock = threading.RLock()

        def work(self, input_items, output_items):
            with self._channel_lock:
                y = self.channel.process(input_items[0])
            output_items[0][:len(y)] = y
            return len(y)

        def reset_channel(self):
            with self._channel_lock:
                self.channel.reset()

        def configure_channel(self, **kwargs):
            with self._channel_lock:
                self.channel.configure(**kwargs)
                self.channel.reset()

        def channel_summary(self) -> str:
            with self._channel_lock:
                return self.channel.summary()

    return _TDLChannelBlock()


def make_rx_ring_sink(rx_buffer):
    """Fallback bounded RX sink for old GNU Radio builds."""
    from gnuradio import gr

    class _RXRingSink(gr.sync_block):
        def __init__(self):
            gr.sync_block.__init__(self, name="rx_numpy_ring_sink_fallback", in_sig=[np.complex64], out_sig=[])

        def work(self, input_items, output_items):
            rx_buffer.write(input_items[0])
            return len(input_items[0])

    return _RXRingSink()
```

2. `fdidm_hardtest.py` 的 `_make_tdl_channel_block` 改为：

```python
    def _make_tdl_channel_block(self):
        """Create a GNU Radio Python sync_block wrapping _NTNTDLChannel."""
        from .gr_flow import make_tdl_channel_block

        channel = _NTNTDLChannel(
            sample_rate=self.sample_rate,
            model=self._tdl_model_for_current_mode(),
            rms_delay_spread_ns=self.tdl_rms_delay_spread_ns,
            doppler_hz=self.tdl_doppler_hz,
            doppler_spread_hz=self.tdl_doppler_spread_hz,
            snr_db=self.tdl_snr_db,
            seed=self.tdl_seed,
            normalize_power=self.tdl_normalize_power,
            num_sinusoids=self.tdl_param_num_sinusoids,
        )
        return make_tdl_channel_block(channel)
```

（删除原方法体内的 `gr = self._gr` 与内嵌 `_TDLChannelBlock` 类。）

3. `_make_rx_ring_sink` 改为：

```python
    def _make_rx_ring_sink(self):
        """Fallback bounded RX sink for old GNU Radio builds."""
        from .gr_flow import make_rx_ring_sink

        return make_rx_ring_sink(self._rx_buffer)
```

（删除原方法体内的 `gr` / `outer` 与内嵌 `_RXRingSink` 类。）

**验证：**
1. `python -c "import waveform_sim.hardware.gr_flow; print('gr_flow ok')"` 输出 `gr_flow ok`（无需 gnuradio）。
2. `python -m pytest -q` → `60 passed`；`python -m compileall -q waveform_sim` 退出码 0。
3. `Select-String` 确认 `fdidm_hardtest.py` 中不再有 `class _TDLChannelBlock` / `class _RXRingSink` 内嵌类定义。

**提交：** `git add waveform_sim/hardware/gr_flow.py waveform_sim/hardware/fdidm_hardtest.py` → `git commit -m "refactor: 提取 GNU Radio 块工厂到 gr_flow.py（阶段6c/2）"`。

### T6c.3：阶段 6c 全量验证

**文件：** 无新增
**依赖：** T6c.1~T6c.2

**步骤：**

1. `python -m pytest -q` → `60 passed`。
2. `python -m compileall -q waveform_sim` → 退出码 0。
3. GUI 离屏冒烟 `tabs: 7`。
4. `git status` → 工作区干净；`fdidm_hardtest.py` 行数应再降约 830 行。

**验证：** 三条命令通过，工作区干净。

## 执行顺序

```
T6c.1 → T6c.2 → T6c.3
```

## 阶段 checkpoint

- T6c.3 后向用户报告行数变化、pytest / GUI 输出与 2 个提交号；并**明确提示**：自适应/GNU Radio 相关逻辑需在实验室真机环境做一次回归（本机无 USRP 无法运行验证）。

