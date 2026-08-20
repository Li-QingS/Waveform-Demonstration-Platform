# 平台工程化重构 Tasks（阶段 3：FDIDM 接入统一引擎）

> 前置：阶段 2 已验收通过（pytest 27 passed）。
> 目标：新增 `LinkSimulator` 统一引擎；`simple_fdidm_rx.py` 的 `FDIDMTransceiver` 改为兼容壳，公开接口与 UI 零改动。
> 过渡说明：引擎的 FDIDM 后端当前**委托现有 `_LegacyFDIDMTransceiver`**（本阶段唯一允许的 core→旧文件依赖，代码中有注释标记），后续阶段再把算法内聚进 core。

## 文件清单

| 操作 | 文件 | 职责 |
|------|------|------|
| 新建 | `waveform_sim/core/engine.py` | `LinkSimulator`：统一配置、生命周期、指标、绘图、自适应入口 |
| 修改 | `waveform_sim/simulation/simple_fdidm_rx.py` | 重命名旧类为 `_LegacyFDIDMTransceiver`、补 `step()`、文件末尾追加兼容壳 `FDIDMTransceiver` |
| 新建 | `tests/test_engine.py` | 引擎后端、配置别名、兼容壳委托、指标键一致性测试 |

## 任务

### T3.1：新建 waveform_sim/core/engine.py

**文件：** 新建 `waveform_sim/core/engine.py`
**依赖：** 阶段 2 的 config（T1.2）

**步骤：** 创建文件，内容如下（完整代码）：

```python
"""统一链路引擎（阶段 3）。

LinkSimulator 提供统一的配置、生命周期、指标、绘图与自适应入口；
FDIDM 后端当前为过渡包装（委托 waveform_sim.simulation.simple_fdidm_rx
的 _LegacyFDIDMTransceiver），后续阶段将算法逐步内聚到 core。
"""
from __future__ import annotations

import threading
from typing import Callable, Dict, Optional

from .config import AdaptiveConfig, ExperimentConfig, WaveformConfig


_ALIASES = {
    "ebn0_db": "snr_db",
    "modulation": "mod_order",
    "decoder": "detector",
    "fc_hz": "center_freq_hz",
    "channel_seed": "seed",
}


class LinkSimulator:
    def __init__(
        self,
        config=None,
        *,
        adaptive: Optional[AdaptiveConfig] = None,
        experiment_service=None,
        backend=None,
        **kwargs,
    ):
        if isinstance(config, ExperimentConfig):
            self.experiment_config = config.normalized()
            self.config = self.experiment_config.waveform
            self.adaptive_config = self.experiment_config.adaptive
        else:
            if config is None:
                mapped = {_ALIASES.get(k, k): v for k, v in kwargs.items()}
                self.config = WaveformConfig(**mapped).normalized()
            elif isinstance(config, WaveformConfig):
                self.config = config.normalized()
            else:
                raise TypeError(f"Unsupported config type: {type(config)}")
            self.adaptive_config = (adaptive or AdaptiveConfig()).normalized()
            self.experiment_config = ExperimentConfig(
                waveform=self.config, adaptive=self.adaptive_config
            ).normalized()
        self.experiment_service = experiment_service
        self._backend = backend if backend is not None else self._create_backend()

    # ------------------------------------------------------------ backend
    def _create_backend(self):
        if self.config.waveform == "FDIDM":
            # 阶段3过渡依赖：后续阶段将算法内聚进 core 后移除
            from waveform_sim.simulation.simple_fdidm_rx import _create_fdidm_backend

            return _create_fdidm_backend(**self._fdidm_legacy_kwargs())
        raise NotImplementedError(
            f"LinkSimulator backend for {self.config.waveform} is not ready"
        )

    def _fdidm_legacy_kwargs(self) -> Dict:
        c = self.config
        return dict(
            alpha=float(c.alpha),
            beta=float(c.beta),
            m_subcarriers=int(c.m_subcarriers),
            n_symbols=int(c.n_symbols),
            subcarrier_spacing_hz=float(c.subcarrier_spacing_hz),
            mod_order=str(c.mod_order),
            channel_model=str(c.channel_model),
            velocity_kmh=float(c.velocity_kmh),
            doppler_radial_factor=float(c.doppler_radial_factor),
            decoder=str(c.detector),
            snr_db=float(c.snr_db),
            snr_definition=str(c.snr_definition),
            optimize_indices=bool(c.optimize_indices),
            search_step=float(c.search_step),
            fc_hz=float(c.center_freq_hz),
            link_mode=str(c.link_mode),
            search_objective=str(c.search_objective),
            random_channel=bool(c.random_channel),
            channel_seed=int(c.seed),
            dynamic_channel=bool(c.dynamic_channel),
            channel_coherence_frames=int(c.channel_coherence_frames),
            channel_dynamics=str(c.channel_dynamics),
            fast_channel_coherence_symbols=int(c.fast_channel_coherence_symbols),
            circular_channel=bool(c.circular_channel),
            tf_notch_depth_db=float(c.tf_notch_depth_db),
            tf_notch_count=int(c.tf_notch_count),
        )

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._backend.start()

    def stop(self) -> None:
        self._backend.stop()

    def wait(self, timeout: Optional[float] = None) -> None:
        self._backend.wait(timeout=timeout)

    def step(self) -> None:
        self._backend.step()

    # ------------------------------------------------------------ config
    def update_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            attr = _ALIASES.get(key, key)
            if hasattr(self.config, attr):
                setattr(self.config, attr, value)
        self.config.normalized()
        updater = getattr(self._backend, "update_runtime_parameters", None)
        if updater is not None:
            updater(**kwargs)

    def set_indices(self, alpha: float, beta: float) -> None:
        self.config.alpha = float(alpha)
        self.config.beta = float(beta)
        self._backend.set_indices(alpha, beta)

    def update_runtime_parameters(self, **kwargs) -> None:
        self.update_config(**kwargs)

    # ------------------------------------------------------------ metrics
    def get_last_metrics(self) -> Dict:
        return dict(self._backend.get_last_metrics())

    def get_ber_summary(self) -> Dict:
        summary = getattr(self._backend, "get_ber_summary", None)
        if summary is not None:
            return summary()
        m = self.get_last_metrics()
        return {
            "cumulative_ber": float(m.get("ber", 1.0)),
            "frames_processed": int(m.get("frames", 0)),
            "frames_decode_ok": int(m.get("frames_decode_ok", 0)),
            "frame_error_rate": float(m.get("fer", 1.0)),
            "bit_errors": int(m.get("bit_errors", 0)),
            "bits_total": int(m.get("total_bits", 0)),
        }

    def get_ber_estimate(self) -> float:
        m = self.get_last_metrics()
        return float(m.get("ber", m.get("ber_window", 1.0)))

    def reset_ber_stats(self) -> None:
        reset = getattr(self._backend, "reset_ber_stats", None)
        if reset is not None:
            reset()

    def get_plot_data(self) -> Dict:
        backend = self._backend
        data = {}
        for key, method in (
            ("constellation", "get_constellation"),
            ("pre_eq_constellation", "get_pre_eq_constellation"),
            ("ser", "get_ser_history"),
            ("ber", "get_ber_history"),
            ("impulse", "get_cross_domain_impulse_response"),
        ):
            fn = getattr(backend, method, None)
            if fn is not None:
                try:
                    data[key] = fn()
                except Exception:
                    pass
        return data

    # ------------------------------------------------------------ adaptive
    def start_adaptive_tuning(
        self,
        config: Optional[AdaptiveConfig] = None,
        callback: Optional[Callable] = None,
    ) -> None:
        if config is not None:
            self.adaptive_config = config.normalized()
        backend = self._backend
        if hasattr(backend, "start_adaptive_tuning"):
            backend.start_adaptive_tuning(callback=callback)
            return
        if hasattr(backend, "search_best_indices"):
            def worker():
                result = backend.search_best_indices()
                if callback is not None:
                    callback(result)

            threading.Thread(target=worker, daemon=True, name="engine-adaptive").start()
            return
        raise NotImplementedError(
            f"adaptive tuning not available for {self.config.waveform}"
        )

    def stop_adaptive_tuning(self) -> None:
        stop = getattr(self._backend, "stop_adaptive_tuning", None)
        if stop is not None:
            stop()

    def get_adaptive_status(self) -> Dict:
        status = getattr(self._backend, "get_alpha_beta_adaptation_status", None)
        if status is not None:
            return status()
        return {"active": False}
```

**验证：** `python -c "from waveform_sim.core.engine import LinkSimulator; print(LinkSimulator.__name__)"` 输出 `LinkSimulator`。

### T3.2：改造 waveform_sim/simulation/simple_fdidm_rx.py

**文件：** 修改 `waveform_sim/simulation/simple_fdidm_rx.py`
**依赖：** T3.1

**步骤：**

1. 将类定义行 `class FDIDMTransceiver(threading.Thread):` 改为 `class _LegacyFDIDMTransceiver(threading.Thread):`。
2. 在 `run()` 方法之后新增公开 `step()`：

```python
    def step(self):
        """单帧仿真（供统一引擎调用）。"""
        self._simulate_one_frame()
```

3. 在文件末尾追加（不改动任何既有方法体）：

```python

# ---------------------------------------------------------------------------
# 阶段3：统一引擎兼容壳
# ---------------------------------------------------------------------------
def _create_fdidm_backend(**kwargs):
    """供 waveform_sim.core.engine 构造 FDIDM 后端（过渡依赖）。"""
    return _LegacyFDIDMTransceiver(**kwargs)


class FDIDMTransceiver(LinkSimulator):
    """FDIDM 兼容壳：继承统一引擎，委托 _LegacyFDIDMTransceiver，公开接口不变。"""

    def __init__(self, **kwargs):
        super().__init__(waveform="FDIDM", **kwargs)

    def __getattr__(self, name):
        backend = self.__dict__.get("_backend")
        if backend is not None and hasattr(backend, name):
            return getattr(backend, name)
        raise AttributeError(name)

    @staticmethod
    def _build_gray_qam(order):
        return _LegacyFDIDMTransceiver._build_gray_qam(order)
```

4. 在文件顶部 import 区加入：

```python
from waveform_sim.core.engine import LinkSimulator
```

**验证：** `python -c "from waveform_sim.simulation.simple_fdidm_rx import FDIDMTransceiver, _LegacyFDIDMTransceiver; print(isinstance(FDIDMTransceiver(), object), _LegacyFDIDMTransceiver.__name__)"` 正常输出且不抛 ImportError。

### T3.3：新建 tests/test_engine.py

**文件：** 新建 `tests/test_engine.py`
**依赖：** T3.1~T3.2

**步骤：** 创建文件，内容如下：

```python
"""统一引擎与 FDIDM 兼容壳测试。"""
import time

from waveform_sim.core.config import WaveformConfig
from waveform_sim.core.engine import LinkSimulator
from waveform_sim.simulation.simple_fdidm_rx import (
    FDIDMTransceiver,
    _LegacyFDIDMTransceiver,
)


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_engine_fdidm_uses_legacy_backend():
    sim = LinkSimulator(WaveformConfig(waveform="FDIDM", snr_db=12.0, seed=42))
    assert sim._backend.__class__.__name__ == "_LegacyFDIDMTransceiver"
    sim.start()
    try:
        ok = _wait_until(lambda: sim.get_last_metrics().get("total_bits", 0) > 0)
        assert ok, "engine FDIDM 超时未出帧"
        m = sim.get_last_metrics()
        assert "ber" in m and "ser" in m and "evm_db" in m
        assert 0.0 <= m["ber"] <= 1.0
    finally:
        sim.stop()
        sim.wait(timeout=3.0)


def test_engine_config_aliases():
    sim = LinkSimulator(waveform="FDIDM", snr_db=12.0, channel_seed=7, decoder="MMSE", fc_hz=1e9)
    assert sim.config.snr_db == 12.0
    assert sim.config.seed == 7
    assert sim.config.detector == "MMSE"
    assert sim.config.center_freq_hz == 1e9
    sim.update_config(ebn0_db=9.0)
    assert sim.config.snr_db == 9.0
    sim.stop()


def test_fdidm_shell_is_link_simulator():
    tb = FDIDMTransceiver(alpha=0.2, beta=0.1, snr_db=12.0, channel_seed=42)
    assert isinstance(tb, LinkSimulator)
    assert tb._backend.__class__.__name__ == "_LegacyFDIDMTransceiver"
    assert callable(tb.get_debug_snapshot)
    pts, labels = FDIDMTransceiver._build_gray_qam(16)
    assert pts.shape == (16,) and labels.shape == (16, 4)
    tb.stop()
    tb.wait(timeout=2.0)


def test_fdidm_shell_matches_legacy_metrics_keys():
    kwargs = dict(snr_db=12.0, channel_seed=42)
    shell = FDIDMTransceiver(**kwargs)
    legacy = _LegacyFDIDMTransceiver(**kwargs)
    shell.start()
    legacy.start()
    try:
        ok = _wait_until(lambda: shell.get_last_metrics().get("total_bits", 0) > 0)
        assert ok
        ms = shell.get_last_metrics()
        ml = legacy.get_last_metrics()
        assert set(ms.keys()) == set(ml.keys())
        assert 0.0 <= ms["ber"] <= 1.0 and 0.0 <= ml["ber"] <= 1.0
    finally:
        shell.stop()
        legacy.stop()
        shell.wait(timeout=3.0)
        legacy.wait(timeout=3.0)
```

> 说明：旧实现的位流 RNG 未固定种子，因此本阶段断言"指标键一致 + 范围合法"；精确数值一致性将在算法内聚 core 并引入固定种子后验证。

**验证：** `python -m pytest tests/test_engine.py -q` → `4 passed`。

### T3.4：全量验证（含 GUI 离屏冒烟）

**文件：** 无新增
**依赖：** T3.1~T3.3

**步骤：**

1. `python -m pytest -q` → 期望 `31 passed`（原 27 + 本阶段 4）。
2. `python -m compileall -q waveform_sim` → 退出码 0、无输出。
3. GUI 离屏冒烟：

```powershell
python -c "import os; os.environ['QT_QPA_PLATFORM']='offscreen'; from PyQt5.QtWidgets import QApplication; from waveform_sim.ui.main_window import MainWindow; app = QApplication([]); w = MainWindow(); print('tabs:', w.tabs.count()); w.close()"
```

期望输出 `tabs: 7`。

**验证：** 三条命令全部通过。

### T3.5：提交

**文件：** `waveform_sim/core/engine.py`、`waveform_sim/simulation/simple_fdidm_rx.py`、`tests/test_engine.py`
**依赖：** T3.4

**步骤：**

1. `git add waveform_sim/core/engine.py waveform_sim/simulation/simple_fdidm_rx.py tests/test_engine.py`
2. `git commit -m "refactor: FDIDM 后端接入统一引擎 LinkSimulator（阶段3）"`
3. `git status` → 工作区干净

**验证：** 提交存在；`python -m pytest -q` 仍为 `31 passed`。

## 执行顺序

```
T3.1 → T3.2 → T3.3 → T3.4 → T3.5
```

## 阶段 checkpoint

- T3.5 后向用户报告 pytest / GUI 冒烟输出与提交号；确认后再拆解阶段 4（OFDM/OTFS/AFDM 逐个接入引擎）。

