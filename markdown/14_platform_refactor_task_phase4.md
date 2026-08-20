# 平台工程化重构 Tasks（阶段 4：OFDM / OTFS / AFDM 接入统一引擎）

> 前置：阶段 3 已验收通过（pytest 31 passed）。
> 目标：三个波形逐个接入 `LinkSimulator`，每波形一轮、一轮一提交；UI 与公开接口零改动。

## 文件清单

| 操作 | 文件 | 职责 |
|------|------|------|
| 修改 | `waveform_sim/core/engine.py` | 新增 OFDM/OTFS/AFDM 后端分支与旧参数名别名 |
| 修改 | `waveform_sim/simulation/simple_ofdm_rx.py` | 重命名 + `step()` + 兼容壳（T4.2） |
| 修改 | `waveform_sim/simulation/simple_otfs_rx.py` | 重命名 + `step()` + 兼容壳（T4.3） |
| 修改 | `waveform_sim/simulation/simple_afdm_rx.py` | 重命名 + `step()` + 兼容壳（T4.4） |
| 修改 | `tests/test_engine.py` | 追加三个波形引擎冒烟用例 |

## 任务

### T4.1：engine.py 支持 OFDM / OTFS / AFDM 后端

**文件：** 修改 `waveform_sim/core/engine.py`
**依赖：** 无（阶段 3 产物）

**步骤：**

1. 扩展 `_ALIASES`（在现有 5 个映射后追加）：

```python
_ALIASES = {
    "ebn0_db": "snr_db",
    "modulation": "mod_order",
    "decoder": "detector",
    "fc_hz": "center_freq_hz",
    "channel_seed": "seed",
    "fft_len": "fft_size",
    "doppler_spread": "doppler_spread_hz",
    "doppler_freq": "doppler_spread_hz",
    "doppler_hz": "doppler_spread_hz",
    "sample_rate": "sample_rate_hz",
}
```

2. `_create_backend()` 增加三个分支（FDIDM 分支保持不变）：

```python
        if self.config.waveform == "OFDM":
            from waveform_sim.simulation.simple_ofdm_rx import _create_ofdm_backend

            return _create_ofdm_backend(**self._ofdm_legacy_kwargs())
        if self.config.waveform == "OTFS":
            from waveform_sim.simulation.simple_otfs_rx import _create_otfs_backend

            return _create_otfs_backend(**self._otfs_legacy_kwargs())
        if self.config.waveform == "AFDM":
            from waveform_sim.simulation.simple_afdm_rx import _create_afdm_backend

            return _create_afdm_backend(**self._afdm_legacy_kwargs())
```

3. 新增三个 legacy kwargs 映射方法：

```python
    def _ofdm_legacy_kwargs(self) -> Dict:
        c = self.config
        return dict(
            fft_len=int(c.fft_size),
            cp_len=int(c.cp_len),
            snr_db=float(c.snr_db),
            cfo_hz=float(c.cfo_hz),
            doppler_spread_hz=float(c.doppler_spread_hz),
            delay_spread=int(c.delay_spread),
            mod_order=str(c.mod_order),
            payload_symbols=int(c.payload_symbols),
        )

    def _otfs_legacy_kwargs(self) -> Dict:
        c = self.config
        return dict(
            delay_spread=int(c.delay_spread),
            doppler_spread=float(c.doppler_spread_hz),
            snr_db=float(c.snr_db),
            mod_order=str(c.mod_order),
            cfo_hz=float(c.cfo_hz),
            n_subcarriers=int(c.n_subcarriers),
            n_symbols=int(c.n_symbols),
            sample_rate=float(c.sample_rate_hz),
            update_period=float(c.update_period),
            equalizer=str(c.equalizer),
        )

    def _afdm_legacy_kwargs(self) -> Dict:
        c = self.config
        return dict(
            c1=float(c.c1),
            c2=float(c.c2),
            snr_db=float(c.snr_db),
            mod_order=str(c.mod_order),
            doppler_freq=float(c.doppler_spread_hz),
            delay_spread=int(c.delay_spread),
            cfo_hz=float(c.cfo_hz),
            sample_rate=float(c.sample_rate_hz),
            frame_size=int(c.frame_size),
        )
```

**验证：** `python -m pytest -q` 仍为 `31 passed`（引擎尚未被新波形调用，纯新增无回归）。

### T4.2：OFDM 轮

**文件：** 修改 `waveform_sim/simulation/simple_ofdm_rx.py`、`tests/test_engine.py`
**依赖：** T4.1

**步骤：**

1. `simple_ofdm_rx.py`：将 `class OfdmTransceiver(threading.Thread):` 改为 `class _LegacyOfdmTransceiver(threading.Thread):`。
2. 在 `run()` 后新增：

```python
    def step(self):
        """单帧仿真（供统一引擎调用）。"""
        self._simulate_one_frame()
```

3. 文件顶部 import 区加 `from waveform_sim.core.engine import LinkSimulator`；文件末尾追加：

```python

# ---------------------------------------------------------------------------
# 阶段4：统一引擎兼容壳
# ---------------------------------------------------------------------------
def _create_ofdm_backend(**kwargs):
    """供 waveform_sim.core.engine 构造 OFDM 后端。"""
    return _LegacyOfdmTransceiver(**kwargs)


class OfdmTransceiver(LinkSimulator):
    """OFDM 兼容壳：继承统一引擎，委托 _LegacyOfdmTransceiver，公开接口不变。"""

    def __init__(self, **kwargs):
        super().__init__(waveform="OFDM", **kwargs)

    def __getattr__(self, name):
        backend = self.__dict__.get("_backend")
        if backend is not None and hasattr(backend, name):
            return getattr(backend, name)
        raise AttributeError(name)
```

4. `tests/test_engine.py` 追加 OFDM 参数化用例（并入通用冒烟）：

```python
@pytest.mark.parametrize(
    "waveform,kwargs,legacy_cls",
    [
        ("OFDM", dict(snr_db=15.0), "_LegacyOfdmTransceiver"),
    ],
)
def test_engine_backend_smoke(waveform, kwargs, legacy_cls):
    sim = LinkSimulator(WaveformConfig(waveform=waveform, **kwargs))
    assert sim._backend.__class__.__name__ == legacy_cls
    sim.start()
    try:
        ok = _wait_until(lambda: sim.get_last_metrics().get("total_bits", 0) > 0)
        assert ok, f"{waveform} 超时未出帧"
        m = sim.get_last_metrics()
        assert "ber" in m and ("evm_db" in m or "evm_percent" in m)
        assert 0.0 <= m["ber"] <= 1.0
    finally:
        sim.stop()
        sim.wait(timeout=3.0)
```

（同时补 `import pytest`；后续 OTFS/AFDM 轮只需往参数列表追加一行。）

**验证：** `python -m pytest -q` → `32 passed`；GUI 离屏冒烟 `tabs: 7`。

**提交：** `git add waveform_sim/simulation/simple_ofdm_rx.py waveform_sim/core/engine.py tests/test_engine.py` → `git commit -m "refactor: OFDM 后端接入统一引擎（阶段4/4a）"`。

### T4.3：OTFS 轮

**文件：** 修改 `waveform_sim/simulation/simple_otfs_rx.py`、`tests/test_engine.py`
**依赖：** T4.2

**步骤：**

1. `simple_otfs_rx.py`：将 `class OTFSTransceiver:` 改为 `class _LegacyOTFSTransceiver:`。
2. 在 `_simulate_one_frame()` 附近新增：

```python
    def step(self):
        """单帧仿真（供统一引擎调用）。"""
        self._simulate_one_frame()
```

3. 顶部 import `from waveform_sim.core.engine import LinkSimulator`；末尾追加 `_create_otfs_backend` 工厂与 `class OTFSTransceiver(LinkSimulator)` 兼容壳（模式同 T4.2，waveform 名 `"OTFS"`）。
4. `tests/test_engine.py` 参数列表追加 `("OTFS", dict(snr_db=15.0), "_LegacyOTFSTransceiver")`。

**验证：** `python -m pytest -q` → `33 passed`；GUI 离屏冒烟 `tabs: 7`。

**提交：** `git commit -m "refactor: OTFS 后端接入统一引擎（阶段4/4b）"`（文件同上三个）。

### T4.4：AFDM 轮

**文件：** 修改 `waveform_sim/simulation/simple_afdm_rx.py`、`tests/test_engine.py`
**依赖：** T4.3

**步骤：**

1. `simple_afdm_rx.py`：将 `class AFDMTransceiver:` 改为 `class _LegacyAFDMTransceiver:`。
2. 在 `_simulate_one_frame()` 附近新增 `step()`（同上）。
3. 顶部 import `LinkSimulator`；末尾追加 `_create_afdm_backend` 工厂与 `class AFDMTransceiver(LinkSimulator)` 兼容壳（waveform 名 `"AFDM"`）。
4. `tests/test_engine.py` 参数列表追加 `("AFDM", dict(snr_db=15.0), "_LegacyAFDMTransceiver")`。

**验证：** `python -m pytest -q` → `34 passed`；GUI 离屏冒烟 `tabs: 7`。

**提交：** `git commit -m "refactor: AFDM 后端接入统一引擎（阶段4/4c）"`（文件同上三个）。

### T4.5：阶段 4 全量验证与收尾

**文件：** 无新增
**依赖：** T4.2~T4.4

**步骤：**

1. `python -m pytest -q` → `34 passed`。
2. `python -m compileall -q waveform_sim` → 退出码 0。
3. GUI 离屏冒烟 `tabs: 7`。
4. `git status` → 工作区干净（三轮已各自提交；若无额外改动则不产生空提交）。

**验证：** 三条命令通过，工作区干净。

## 执行顺序

```
T4.1 → T4.2 → T4.3 → T4.4 → T4.5
```

## 阶段 checkpoint

- T4.5 后向用户报告 pytest / GUI 冒烟输出与三个提交号；确认后再拆解阶段 5（硬件抽象层）。

