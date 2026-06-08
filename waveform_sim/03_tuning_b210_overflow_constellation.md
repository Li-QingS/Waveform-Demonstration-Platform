# 调优优化模板：B210 溢出、星座发散与实时解码卡顿

## 观察到的问题：[详细描述错误现象/性能问题]

在 FDIDM 硬件测试平台运行过程中，可能出现以下现象：

1. **GNU Radio/UHD 出现 B210 RX overflow / underrun**
   - 运行一段时间后 RX 频谱更新变慢。
   - 日志中出现接收窗口没有新样点、spectrum stale 等现象。
   - BER/EVM 偶发恶化，但硬件链路本身不一定断开。

2. **星座图偶发发散或突然消失**
   - 同步成功时星座正常，某些帧后星座变散。
   - full-H 或 TDL 参数估计失败后，后续缓存可能导致图形短时间“看似稳定”。
   - RF+TDL 级联时，如果误用 TDL 参数基估计真实 RF 链路，均衡结果变差。

3. **GUI 卡顿**
   - 开启高采样率、大 M/N、full-H_TF 或高频刷新时，Qt 主线程响应变慢。
   - 参数改动自动应用时，频繁 stop/configure/start 导致卡顿。

4. **full-H_TF 模式计算开销过大**
   - M×N 增大时，需要估计和处理 MN×MN 矩阵。
   - 若每帧都更新 full-H，性能不可控。
   - 即使缓存 full-H，也需要严格判断缓存是否适用于当前参数和信道状态。

---

## 环境信息：[操作系统/运行时版本/依赖库]

建议在问题复现记录中填入：

```text
操作系统：Windows 10/11 或 Ubuntu 22.04
Python：3.x
NumPy：
PyQt5：
pyqtgraph：
GNU Radio：
UHD：
硬件：USRP B210 / N210 / X310
连接方式：loopback cable / antenna / attenuator
采样率：
中心频率：
TX/RX 增益：
FDIDM 参数：M, N, CP, alpha, beta
调制方式：QPSK / 16QAM / 64QAM
均衡器：ZF / MMSE
信道模式：rf / tdl_a / tdl_a_rf / rf_tdl_a ...
估计器：diag_tf / tdl_param / full_htf
```

---

## 已尝试的解决方案：[列出已尝试但失败的方法]

当前代码中已经尝试或具备以下优化：

- 使用 `_SampleRing` 替代无界 list/deque，避免实时路径中 Python 对每个样点处理。
- RX 优先使用 GNU Radio C++ probe/vector 链路，旧版本才退回 Python ring sink。
- `process_interval_ms` 对 Python 解码进行节流，避免每次 GUI 刷新都进行重解码。
- B210 默认采样率/增益相对保守，并设置 UHD buffer frame 参数。
- full-H_TF 支持一次辨识和缓存，避免每帧重建矩阵。
- RF 场景下自动将部分不合适的估计器切换到 `diag_tf`。
- 对星座图区分 post-equalized、pre-equalized、TF received、last good、raw IQ 等显示模式。

仍可能失败的原因：

- UI 刷新、FFT、日志拼接和状态读取仍在主线程。
- 软件 TDL 处理每块都有数组拼接/复制。
- 参数自动应用没有防抖时会频繁重启流图。
- full-H 大矩阵和 QR/solve 类操作在 M×N 较大时开销不可控。
- RF+TDL 级联场景中真实 RF 响应与理论 TDL 基不匹配。

---

## 请提供：

### 1. 问题根本原因分析

#### 根因 A：实时采集与 Python 解码节奏不一致

UHD/GNU Radio 采集是实时流，Python 解码不是严格实时。只要 Python 解码、矩阵计算或 GUI 刷新超过接收缓冲能力，就可能出现 overflow 或 stale spectrum。

#### 根因 B：大矩阵估计器不适合所有链路

论文中的 full-H_TF 输入输出关系适合理论和可控仿真，但硬件 RF 链路包含模拟滤波、IQ 失衡、线缆响应、非理想 CFO、采样时钟偏差等因素。RF+TDL 级联时，用纯 TDL 参数基去拟合整体链路会带来模型失配。

#### 根因 C：GUI 线程承担了过多工作

Qt 主线程应主要做显示和用户交互。当前如果在刷新中计算频谱、读取状态、更新大量日志、处理星座数据，会造成卡顿。

#### 根因 D：参数应用过于频繁

spinbox 每次 valueChanged 都可能触发配置更新。如果自动应用开启，连续改变 α、β、M/N、CP 或信道参数会造成重复 stop/start 和缓存重置。

#### 根因 E：星座图被“坏帧”污染

当同步或信道估计偶发失败时，如果图形缓存使用坏帧数据，用户会误以为链路突然劣化。反过来，如果只显示 last good，又可能掩盖实时问题。因此应同时显示“当前帧质量”和“last good 状态”。

---

### 2. 修复代码

#### 修复 1：参数自动应用增加防抖

```python
# 在 FDIDMHardwareTestTab.__init__ 或初始化 UI 后调用
def _init_apply_debounce(self):
    self._apply_debounce_timer = QTimer(self)
    self._apply_debounce_timer.setSingleShot(True)
    self._apply_debounce_timer.setInterval(500)
    self._apply_debounce_timer.timeout.connect(self._apply_params_to_backend)

def _on_params_changed(self, *_args):
    if self.auto_apply_check.isChecked():
        self._apply_debounce_timer.start()
```

---

#### 修复 2：GUI 刷新只消费快照，不做重型解码

```python
def _refresh_plots(self):
    if self.backend is None:
        return

    try:
        # get_status 应返回后端已经准备好的轻量快照。
        st = self.backend.get_status()
        self._update_decode_status(st)
        self._update_evm_plot(st)
        self._update_constellation_from_snapshot(st)
        self._maybe_log_runtime(st)
    except RuntimeError as e:
        self._log(f"刷新失败: {e}")
```

后端中应由工作线程维护 `latest_snapshot`：

```python
def _publish_snapshot(self) -> None:
    with self._lock:
        self._latest_snapshot = {
            "timestamp": time.time(),
            "decode_ok": self._decode_ok,
            "ber": self._ber_estimate,
            "evm": self.last_evm_average_percent,
            "rx_spectrum_stale": self._rx_spectrum_stale,
            "constellation_source": self.last_constellation_source,
        }
```

---

#### 修复 3：限制 full-H_TF 的默认使用范围

```python
def validate_estimator_budget(M: int, N: int, estimator: str, max_order: int = 1024) -> None:
    order = int(M) * int(N)
    if estimator == "full_htf" and order > max_order:
        raise ValueError(
            f"full-H_TF order={order} exceeds max_order={max_order}. "
            "请降低 M/N，或改用 diag_tf / tdl_param。"
        )
```

---

#### 修复 4：统一估计器选择策略

```python
def resolve_estimator(requested: str, channel_mode: str, doppler_hz: float, doppler_spread_hz: float) -> tuple[str, str]:
    involves_rf = channel_mode == "rf" or channel_mode.startswith("rf_tdl_") or channel_mode.endswith("_rf")

    if involves_rf and requested == "tdl_param":
        return "diag_tf", "真实 RF 链路无法由纯 TDL 参数基完整描述，切换到 diag_tf。"

    if involves_rf and channel_mode != "rf" and requested == "full_htf":
        return "diag_tf", "RF+TDL 级联下 full-H 默认不稳定，切换到 diag_tf。"

    if channel_mode != "rf" and requested == "full_htf" and (abs(doppler_hz) > 1e-9 or doppler_spread_hz > 1e-9):
        return "tdl_param", "动态软件 TDL 使用参数化估计器更稳。"

    return requested, "无需切换。"
```

---

#### 修复 5：为软件 TDL 减少临时数组分配

当前软件 TDL 中 `history + current block` 拼接直观但会频繁分配。若后续高采样率或大块处理时性能不足，可把历史缓冲改为预分配环形结构，或至少减少 copy 次数。

简化示例：

```python
class StreamingHistory:
    def __init__(self, hist_len: int):
        self.hist = np.zeros(hist_len, dtype=np.complex128)

    def extend(self, x: np.ndarray) -> np.ndarray:
        # 仍会生成 ext，但集中管理 history 更新，便于后续替换为无拷贝实现。
        ext = np.empty(self.hist.size + x.size, dtype=np.complex128)
        ext[:self.hist.size] = self.hist
        ext[self.hist.size:] = x
        self.hist[:] = ext[-self.hist.size:]
        return ext
```

---

#### 修复 6：增加实验日志 JSONL

```python
import json
from datetime import datetime

def append_experiment_log(path: str, config: dict, status: dict) -> None:
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "status": {
            "decode_ok": status.get("decode_ok"),
            "ber": status.get("ber_estimate"),
            "evm": status.get("evm_average_percent"),
            "cfo_hz": status.get("last_cfo_est_hz"),
            "sync_metric": status.get("sync_metric"),
            "cond_h": status.get("last_cond_h_cross"),
            "rx_spectrum_stale": status.get("rx_spectrum_stale"),
        },
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

---

### 3. 防止类似问题的最佳实践建议

1. **不要把 GUI 刷新频率等同于解码频率**  
   GUI 可以 100 ms 刷新，解码可以 200-500 ms 一次，日志可以 1 s 一次。

2. **每个链路模式有默认估计器**  
   - pure TDL static：`tdl_param` 或 `full_htf`
   - pure TDL dynamic：`tdl_param`
   - pure RF loopback：`diag_tf` 优先
   - RF+TDL cascade：`diag_tf` 优先，除非有专门级联模型

3. **对 full-H_TF 设置硬性预算**  
   full-H 适合论文严格验证和小规模矩阵，不应作为大 M/N 实时硬件默认项。

4. **把“当前帧”和“最近好帧”同时暴露给用户**  
   星座图可以显示 last good，但状态栏必须明确当前帧是否 stale、是否 CRC fail、是否使用旧 CSI。

5. **所有参数扫描必须可中断、可恢复**  
   α/β 优化不能让硬件链路停在未知状态。扫描前保存参数，失败时恢复。

6. **建立最小回归测试集**  
   - `gamma(eps) @ gamma(-eps) ≈ I`
   - `fdit(ifdit(X)) ≈ X`
   - QAM mod/demod round-trip
   - frame CRC parse/build round-trip
   - pure TDL 静态链路低噪声下 BER≈0
   - RF 模式下估计器自动切换规则稳定

7. **记录每次实验的完整配置**  
   没有配置快照的 BER/EVM 曲线很难复现实验结论。
