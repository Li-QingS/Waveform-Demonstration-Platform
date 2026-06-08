# 功能实现模板：α/β 自动推荐与参数扫描模块

## 任务：实现[具体功能]

实现 **FDIDM α/β 自动推荐模块**。该模块根据当前链路模式、信道估计结果、SNR/EVM/BER/条件数等指标，在有限搜索空间内自动给出推荐的 α、β，并支持将推荐结果回填到 GUI。

该功能的目标不是盲目追求全局最优，而是在实验平台中提供一个可解释、可复现、低风险的“参数推荐/扫描”能力：

- 在纯软件 TDL 模式下，可进行较完整的 α/β 扫描。
- 在 RF 或 RF+TDL 模式下，应采用安全的低频率扫描，避免频繁重建流图。
- 支持 OFDM `(0,0)`、OTFS `(1,1)`、论文推荐附近 `(0.5,1.0)` 作为基线。
- 输出推荐值、性能指标、扫描记录和推荐理由。

---

## 技术栈：[编程语言/框架/库]

- Python 3.9+
- NumPy
- PyQt5
- pyqtgraph
- GNU Radio / UHD
- 现有后端：`FDIDMHardwareTest`
- 现有 GUI：`FDIDMHardwareTestTab`

建议新增模块：

```text
fdidm/
  optimization/
    alpha_beta_optimizer.py
    score.py
    sweep_result.py
```

---

## 输入：[描述输入数据结构和示例]

### 输入 1：当前 FDIDM 配置

```python
config = {
    "M": 16,
    "N": 16,
    "cp_len": 4,
    "mod_order": "QPSK",
    "equalizer": "MMSE",
    "channel_mode": "tdl_a_rf",
    "channel_estimator": "diag_tf",
    "sample_rate": 1_000_000.0,
    "carrier_freq": 2.4e9,
    "tdl_rms_delay_spread_ns": 1000.0,
    "tdl_doppler_hz": 0.0,
    "tdl_doppler_spread_hz": 0.0,
    "tdl_snr_db": 35.0,
}
```

### 输入 2：候选 α/β 搜索空间

```python
search_config = {
    "alpha_range": [0.0, 2.0],
    "beta_range": [0.0, 2.0],
    "coarse_step": 0.25,
    "fine_step": 0.05,
    "max_candidates": 81,
    "baseline_points": [(0.0, 0.0), (1.0, 1.0), (0.5, 1.0)],
    "mode": "coarse_to_fine"
}
```

### 输入 3：评价指标快照

```python
status = {
    "decode_ok": True,
    "ber_estimate": 0.0,
    "evm_instant_percent": 5.2,
    "evm_average_percent": 6.1,
    "sync_metric": 0.83,
    "last_cfo_est_hz": 12.5,
    "last_cond_h_cross": 18.6,
    "last_htf_nmse": 0.012,
    "last_tdl_param_fit_nmse": 0.025,
    "frames_decode_ok": 37,
    "frames_processed": 40
}
```

---

## 输出：[描述期望输出和示例]

### 输出 1：推荐结果

```python
result = {
    "recommended_alpha": 0.45,
    "recommended_beta": 1.0,
    "score": 0.92,
    "reason": "在当前 TDL-C/RF 级联条件下，该组合相对 OFDM 和 OTFS 具有更低 EVM，并且 BER 为 0。",
    "baseline": {
        "ofdm": {"alpha": 0.0, "beta": 0.0, "evm": 12.4, "ber": 0.01},
        "otfs": {"alpha": 1.0, "beta": 1.0, "evm": 8.8, "ber": 0.002},
        "default": {"alpha": 0.5, "beta": 1.0, "evm": 5.4, "ber": 0.0}
    }
}
```

### 输出 2：扫描表格

```python
sweep_table = [
    {"alpha": 0.0, "beta": 0.0, "score": 0.55, "ber": 0.01, "evm": 12.4, "cond_h": 30.1},
    {"alpha": 0.5, "beta": 1.0, "score": 0.91, "ber": 0.0, "evm": 5.4, "cond_h": 12.7},
    {"alpha": 1.0, "beta": 1.0, "score": 0.79, "ber": 0.002, "evm": 8.8, "cond_h": 19.2}
]
```

### 输出 3：GUI 展示文本

```text
推荐 α/β = 0.45 / 1.00
依据：EVM 最低，BER 为 0，且条件数较 OFDM/OTFS 更小。
建议：可点击“应用推荐参数”写入当前 FDIDM 参数。
```

---

## 约束条件：

- [性能要求]  
  - 纯软件 TDL 模式下，粗扫不超过 5 秒，细扫不超过 15 秒。
  - RF 模式下默认只比较基线点和少量邻域点，不允许高频率重启 UHD/GNU Radio 流图。
  - 避免每个候选点都重新创建完整 backend；优先调用轻量配置更新。
  - 对 M×N 较大时，禁止默认使用 full-H 全矩阵扫描。

- [内存限制]  
  - M×N > 1024 时，不缓存所有候选点的 full-H 矩阵。
  - 单次扫描结果只保存指标、参数和必要诊断，不保存完整 IQ 流。
  - 对大型矩阵计算增加 `max_order` 和内存估算提示。

- [代码风格/规范]  
  - 使用 dataclass 描述输入输出。
  - 优化器不直接操作 PyQt 控件，只通过 backend/status 接口交互。
  - GUI 只负责触发任务、显示进度和接收结果。
  - 核心评分函数必须可单元测试。

- [错误处理要求]  
  - 如果没有稳定同步，返回 `status="no_sync"`。
  - 如果所有候选点 BER 都不可用，退化为 EVM/同步度量评分。
  - 如果 RF 模式下频繁失败，应自动停止扫描并恢复原参数。
  - 扫描结束后必须恢复或确认应用用户选择的参数。

---

## 额外上下文：[项目背景/现有代码结构]

论文指出，FDIDM 的 α、β 最优值依赖 SNR、星座映射、帧尺寸和 CSI 等因素，且搜索空间可限制在较小范围内。因此平台可以先做“工程可用”的粗细两阶段搜索，而不是一开始实现复杂全局优化。

现有平台已经具备：

- α、β 可调控件。
- OFDM `(0,0)`、OTFS `(1,1)`、推荐 `(0.5,1)` 快捷按钮。
- BER、EVM、CFO、同步度量、条件数、TDL 拟合误差等状态指标。
- `configure()` 参数更新能力。
- pure TDL、RF、RF→TDL、TDL→RF 等链路模式。
- `diag_tf`、`tdl_param`、`full_htf` 等估计器。

---

## 建议实现代码骨架

```python
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

@dataclass(frozen=True)
class AlphaBetaCandidate:
    alpha: float
    beta: float

@dataclass
class CandidateMetrics:
    alpha: float
    beta: float
    ber: float
    evm: float
    sync_metric: float
    cond_h: float
    decode_ok: bool
    score: float
    reason: str

@dataclass
class SweepResult:
    best: CandidateMetrics
    all_results: list[CandidateMetrics]
    restored_alpha: float
    restored_beta: float
    applied: bool = False

def score_candidate(status: dict) -> tuple[float, str]:
    """越大越好。兼顾 BER、EVM、同步质量和条件数。"""
    ber = float(status.get("ber_estimate", 1.0))
    evm = float(status.get("evm_average_percent", status.get("evm_instant_percent", 100.0)))
    sync = float(status.get("sync_metric", 0.0))
    cond = float(status.get("last_cond_h_cross", 1e6))

    score = 0.0
    score += 0.45 * max(0.0, 1.0 - min(ber, 0.1) / 0.1)
    score += 0.30 * max(0.0, 1.0 - min(evm, 50.0) / 50.0)
    score += 0.15 * min(sync, 1.0)
    score += 0.10 * max(0.0, 1.0 - min(cond, 1e3) / 1e3)

    reason = f"BER={ber:.2e}, EVM={evm:.2f}%, sync={sync:.3f}, cond={cond:.2e}"
    return score, reason

class AlphaBetaOptimizer:
    def __init__(self, backend, wait_seconds: float = 0.5):
        self.backend = backend
        self.wait_seconds = wait_seconds

    def generate_candidates(self, coarse_step: float = 0.25) -> list[AlphaBetaCandidate]:
        values = [round(i * coarse_step, 6) for i in range(int(2.0 / coarse_step) + 1)]
        base = {(0.0, 0.0), (1.0, 1.0), (0.5, 1.0)}
        grid = {(a, b) for a in values for b in values}
        return [AlphaBetaCandidate(a, b) for a, b in sorted(base | grid)]

    def evaluate(self, candidate: AlphaBetaCandidate) -> CandidateMetrics:
        self.backend.configure(alpha=candidate.alpha, beta=candidate.beta)
        self.backend.reset_full_htf_cache()
        # 实际实现中应等待若干帧稳定，而不是固定 sleep。
        import time
        time.sleep(self.wait_seconds)

        status = self.backend.get_status()
        score, reason = score_candidate(status)
        return CandidateMetrics(
            alpha=candidate.alpha,
            beta=candidate.beta,
            ber=float(status.get("ber_estimate", 1.0)),
            evm=float(status.get("evm_average_percent", 100.0)),
            sync_metric=float(status.get("sync_metric", 0.0)),
            cond_h=float(status.get("last_cond_h_cross", 1e6)),
            decode_ok=bool(status.get("decode_ok", False)),
            score=score,
            reason=reason,
        )

    def run(self, candidates: Optional[Iterable[AlphaBetaCandidate]] = None) -> SweepResult:
        original = self.backend.get_status()
        original_alpha = float(original.get("alpha", 0.5))
        original_beta = float(original.get("beta", 1.0))

        results = []
        try:
            for candidate in candidates or self.generate_candidates():
                results.append(self.evaluate(candidate))
            best = max(results, key=lambda x: x.score)
            return SweepResult(best=best, all_results=results,
                               restored_alpha=original_alpha, restored_beta=original_beta)
        finally:
            self.backend.configure(alpha=original_alpha, beta=original_beta)
```

---

## 验收标准

1. 在 pure TDL 模式下，可以完成 α/β 基线点 + 粗扫，并输出推荐值。
2. 在 RF 模式下，不因扫描导致 UHD/GNU Radio 流图频繁崩溃。
3. 扫描结果可导出为 CSV/JSONL。
4. GUI 中新增：
   - “扫描 α/β”
   - “停止扫描”
   - “应用推荐”
   - “导出扫描结果”
5. 推荐结果至少包含 OFDM、OTFS、默认推荐点和最优点对比。
6. 单元测试覆盖 `score_candidate()`、候选点生成、异常恢复。
