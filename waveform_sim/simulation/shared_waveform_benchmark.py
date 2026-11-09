# -*- coding: utf-8 -*-
"""Shared-channel OFDM/OTFS/AFDM/FDIDM performance benchmark.

All four waveforms are evaluated on the *same* physical channel realization and
with the same QAM/SNR convention.  This replaces the former shortcut that drew
OFDM, OTFS and AFDM as three alpha/beta points inside the FDIDM theory kernel.

Receiver model
--------------
A fully dense, perfect-CSI MMSE receiver is invariant to many unitary waveform
changes and therefore hides the practical difference between waveform domains.
The benchmark instead uses a support-limited channel model:

* OFDM: one-tap, per-tone MMSE equalization; off-diagonal ICI stays as
  per-tone interference power.
* OTFS: strongest-coefficient sparse 2-D channel model with LMMSE equalization.
* AFDM: strongest-coefficient sparse affine-domain channel model with LMMSE
  equalization.
* FDIDM: the same retained-coefficient budget as OTFS/AFDM, while alpha/beta
  select the fractional domain that best concentrates the current channel.

For a retained channel model H_d and the true equivalent channel H, the omitted
coupling H_o=H-H_d is treated as colored interference instead of being silently
ignored. The omitted-channel covariance is computed from the full true channel,
so it is an oracle covariance, not a practical channel-estimation result. The
support budget limits retained coefficients; the matrix solve itself is dense.
The linear MMSE filter before gain correction is

    W_0 = H_d^H (H_d H_d^H + H_o H_o^H + sigma^2 I)^-1.

Each stream is gain-normalized by the corresponding diagonal element of
W_0 H before QAM slicing. This removes MMSE shrinkage bias from the noise
variance used by the SER approximation.

The effective symbol error variance is then evaluated against the *true* channel:

    diag((W H-I)(W H-I)^H + sigma^2 W W^H).

The residual variance is passed to the same square-QAM SER expression for every
waveform using a Gaussian approximation for residual interference. The result
is an analytical prediction, not frame-level Monte Carlo. Guard-interval
throughput overhead is excluded because this compact matrix benchmark does not
synthesize prefix samples. No arbitrary curve offsets are used.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence, Tuple

import numpy as np

RECEIVER_POLICY = (
    "OFDM one-tap gain-normalized LMMSE; OTFS/AFDM/FDIDM support-limited "
    "LMMSE with true-channel residual covariance"
)
SUPPORT_SELECTION_SOURCE = (
    "true_channel_oracle_strongest_per_row_with_diagonal_retention"
)
OFDM_SUPPORT_SELECTION_SOURCE = "fixed_diagonal_one_tap_model"


@dataclass(frozen=True)
class BenchmarkContext:
    m_subcarriers: int
    n_symbols: int
    subcarrier_spacing_hz: float
    qam_order: int
    bits_per_symbol: int
    snr_definition: str = "Eb/N0"
    receiver_policy: str = field(default=RECEIVER_POLICY, init=False)
    support_selection_source: str = field(
        default=SUPPORT_SELECTION_SOURCE, init=False
    )
    path_count: int = 3
    maximum_delay_ns: float = 0.0
    residual_doppler_spread_hz: float = 200.0
    max_residual_doppler_hz: float | None = None

    @property
    def k(self) -> int:
        return int(self.m_subcarriers * self.n_symbols)

    @property
    def sample_rate_hz(self) -> float:
        return float(self.m_subcarriers * self.subcarrier_spacing_hz)


class SharedWaveformBenchmark:
    """Evaluate four waveform domains on one fixed H_TF realization."""

    WAVEFORMS = ("OFDM", "OTFS", "AFDM", "FDIDM")
    RECEIVER_POLICY = RECEIVER_POLICY

    def __init__(
        self,
        context: BenchmarkContext,
        h_tf: np.ndarray,
        fdit_factory: Callable[[float, float], Tuple[np.ndarray, np.ndarray]],
    ):
        self.context = context
        self.M = int(context.m_subcarriers)
        self.N = int(context.n_symbols)
        self.K = int(context.k)
        self.h_tf = np.asarray(h_tf, dtype=np.complex128).reshape(self.K, self.K)
        self._fdit_factory = fdit_factory
        self._Fm = self._unitary_dft(self.M)
        self._Fn = self._unitary_dft(self.N)
        self._Fk = self._unitary_dft(self.K)
        self._A_tx_ofdm = np.kron(np.eye(self.N, dtype=np.complex128), self._Fm.conj().T)
        self._A_rx_ofdm = self._A_tx_ofdm.conj().T
        # H_tf=A_rx G A_tx, hence G=A_rx^H H_tf A_tx^H.
        self.g_time = self._A_rx_ofdm.conj().T @ self.h_tf @ self._A_tx_ofdm.conj().T
        self._otfs_tx = self._build_otfs_isfft()
        self._afdm_analysis, self.afdm_c1, self.afdm_c2 = self._build_afdm_analysis()

    @classmethod
    def from_backend(cls, backend) -> "SharedWaveformBenchmark":
        """Build from `_LegacyFDIDMTransceiver` or its LinkSimulator shell."""

        obj = getattr(backend, "_backend", backend)
        with obj._lock:
            obj._prepare_matrices_locked()
            h_tf = np.asarray(obj._H_tf, dtype=np.complex128).copy()
            cfg = obj.config
            qam_order = int(obj.qam_order)
            bits_per_symbol = int(obj.bits_per_symbol)
            try:
                summary = obj.get_channel_summary()
            except Exception:
                summary = {}
        paths = list(summary.get("paths", []) or [])
        max_delay_ns = max((float(p.get("delay_ns", 0.0)) for p in paths), default=0.0)
        residual_spread = float(
            summary.get(
                "residual_doppler_spread_hz",
                summary.get("max_doppler_hz", 0.0),
            )
        )
        max_residual_doppler = float(
            summary.get("max_doppler_hz", residual_spread)
        )
        ctx = BenchmarkContext(
            m_subcarriers=int(cfg.m_subcarriers),
            n_symbols=int(cfg.n_symbols),
            subcarrier_spacing_hz=float(cfg.subcarrier_spacing_hz),
            qam_order=qam_order,
            bits_per_symbol=bits_per_symbol,
            snr_definition=str(getattr(cfg, "snr_definition", "Eb/N0")),
            path_count=max(1, len(paths)),
            maximum_delay_ns=float(max_delay_ns),
            residual_doppler_spread_hz=float(residual_spread),
            max_residual_doppler_hz=max(0.0, max_residual_doppler),
        )
        # Freeze the transform dimensions captured with H_tf.  The backend is
        # shared with the live UI and may be resized while a background search
        # is still evaluating candidates; calling its mutable bound
        # _fdit_matrices method would then return a matrix with the wrong
        # dimension for this benchmark's fixed channel snapshot.
        m_snapshot = int(ctx.m_subcarriers)
        n_snapshot = int(ctx.n_symbols)
        gamma = getattr(obj, "_gamma", None)
        if callable(gamma):
            def fdit_factory(alpha: float, beta: float):
                tx = np.kron(
                    gamma(n_snapshot, -float(beta)),
                    gamma(m_snapshot, float(alpha)),
                )
                rx = np.kron(
                    gamma(n_snapshot, float(beta)),
                    gamma(m_snapshot, -float(alpha)),
                )
                return tx, rx
        else:
            # Keep compatibility with backend-like test doubles that expose
            # only the public factory.  Validate the dimensions before the
            # matrices enter the fixed benchmark.
            mutable_factory = obj._fdit_matrices
            expected_shape = (m_snapshot * n_snapshot, m_snapshot * n_snapshot)

            def fdit_factory(alpha: float, beta: float):
                tx, rx = mutable_factory(float(alpha), float(beta))
                tx = np.asarray(tx, dtype=np.complex128)
                rx = np.asarray(rx, dtype=np.complex128)
                if tx.shape != expected_shape or rx.shape != expected_shape:
                    raise ValueError(
                        "FDIDM transform dimensions changed after benchmark snapshot"
                    )
                return tx, rx

        return cls(ctx, h_tf, fdit_factory)

    @staticmethod
    def _unitary_dft(n: int) -> np.ndarray:
        idx = np.arange(int(n), dtype=np.float64)
        return np.exp(-1j * 2.0 * np.pi * np.outer(idx, idx) / float(n)) / np.sqrt(float(n))

    def _build_otfs_isfft(self) -> np.ndarray:
        # X_TF = F_M X_DD F_N^H, column-major vectorization:
        # vec(X_TF) = (F_N^* kron F_M) vec(X_DD).
        return np.kron(self._Fn.conj(), self._Fm)

    def _build_afdm_analysis(self) -> Tuple[np.ndarray, float, float]:
        # AFDM design rule: choose c1 from the maximum normalized Doppler index;
        # c2 is a small irrational-like chirp to avoid repeated phase patterns.
        fs = max(self.context.sample_rate_hz, 1e-15)
        max_doppler = self.context.max_residual_doppler_hz
        if max_doppler is None:
            max_doppler = self.context.residual_doppler_spread_hz
        nu_max = abs(float(max_doppler)) * self.K / fs
        c1 = float((2.0 * math.ceil(nu_max) + 1.0) / (2.0 * self.K))
        c2 = float(math.sqrt(2.0) / (2.0 * self.K))
        n = np.arange(self.K, dtype=np.float64)
        lambda1 = np.diag(np.exp(-1j * 2.0 * np.pi * c1 * n * n))
        lambda2 = np.diag(np.exp(-1j * 2.0 * np.pi * c2 * n * n))
        analysis = lambda2 @ self._Fk @ lambda1
        return analysis, c1, c2

    def noise_variance(self, snr_db: float) -> float:
        lin = 10.0 ** (float(snr_db) / 10.0)
        mode = str(self.context.snr_definition).upper().replace(" ", "")
        if mode in ("ES/N0", "ESN0", "SNR"):
            return float(1.0 / max(lin, 1e-15))
        return float(1.0 / max(self.context.bits_per_symbol * lin, 1e-15))

    @staticmethod
    def _qfunc(x: np.ndarray) -> np.ndarray:
        flat = np.asarray(x, dtype=np.float64).reshape(-1)
        out = np.fromiter(
            (0.5 * math.erfc(float(v) / math.sqrt(2.0)) for v in flat),
            dtype=np.float64,
            count=flat.size,
        )
        return out.reshape(np.asarray(x).shape)

    @classmethod
    def qam_ser_from_error_variance(cls, variance: np.ndarray, order: int) -> np.ndarray:
        q = int(order)
        var = np.maximum(np.asarray(variance, dtype=np.float64), 1e-15)
        kappa = 1.0 - 1.0 / math.sqrt(float(q))
        arg = np.sqrt(3.0 / ((float(q) - 1.0) * var))
        qv = cls._qfunc(arg)
        return np.clip(4.0 * kappa * qv - 4.0 * (kappa ** 2) * (qv ** 2), 0.0, 1.0)

    def equivalent_channel(
        self,
        waveform: str,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> np.ndarray:
        name = str(waveform).upper()
        if name == "OFDM":
            return self.h_tf.copy()
        if name == "OTFS":
            T = self._otfs_tx
            return T.conj().T @ self.h_tf @ T
        if name == "AFDM":
            A = self._afdm_analysis
            return A @ self.g_time @ A.conj().T
        if name == "FDIDM":
            tx, rx = self._fdit_factory(float(alpha), float(beta))
            return rx @ self.h_tf @ tx
        raise ValueError(f"unsupported waveform: {waveform}")

    def _receiver_budget(self, waveform: str) -> int:
        if str(waveform).upper() == "OFDM":
            return 1
        # Same sparse detector budget for OTFS/AFDM/FDIDM.  It scales only with
        # physical path count, not with the waveform name.
        return int(min(self.K, max(3, 2 * int(self.context.path_count) + 1)))

    @staticmethod
    def _retain_strongest_per_row(H: np.ndarray, count: int) -> np.ndarray:
        matrix = np.asarray(H, dtype=np.complex128)
        rows, cols = matrix.shape
        keep = int(max(1, min(count, cols)))
        if keep >= cols:
            return matrix.copy()
        out = np.zeros_like(matrix)
        # Preserve each row's desired-stream coefficient.  A pure strongest-
        # coefficient truncation can delete H[i, i] when an interferer is
        # stronger, leaving the receiver model with no path for stream i.
        # Spend the remaining support budget on the strongest off-diagonal
        # coefficients.
        for i in range(rows):
            diagonal_present = i < cols and abs(matrix[i, i]) > 1e-12
            if diagonal_present:
                out[i, i] = matrix[i, i]
                remaining = keep - 1
                if remaining <= 0:
                    continue
                candidates = np.delete(np.arange(cols), i)
                row_abs = np.abs(matrix[i, candidates])
                selected = candidates[np.argpartition(row_abs, -remaining)[-remaining:]]
                out[i, selected] = matrix[i, selected]
            else:
                # A zero/degenerate diagonal does not represent a usable
                # desired path. Do not spend a support slot on it.
                idx = np.argpartition(np.abs(matrix[i]), -keep)[-keep:]
                out[i, idx] = matrix[i, idx]
        return out

    @staticmethod
    def _interference_aware_lmmse(
        H_model: np.ndarray,
        H_true: np.ndarray,
        noise_var: float,
        *,
        diagonal_covariance: bool = False,
    ) -> np.ndarray:
        """Return a complexity-limited LMMSE receiver with mismatch covariance.

        `H_model` is the subset of coefficients retained by the receiver.  The
        omitted coefficients are not assumed to vanish: their covariance is
        included in the regularization matrix.  This prevents the unphysical
        high-SNR degradation that occurs when a mismatched receiver becomes an
        increasingly aggressive pseudo-inverse of an incomplete channel model.
        """
        model = np.asarray(H_model, dtype=np.complex128)
        true = np.asarray(H_true, dtype=np.complex128)
        omitted = true - model
        rows = model.shape[0]
        eye = np.eye(rows, dtype=np.complex128)
        covariance = (
            model @ model.conj().T
            + omitted @ omitted.conj().T
            + max(float(noise_var), 1e-12) * eye
        )
        if diagonal_covariance:
            covariance = np.diag(np.diag(covariance))
        try:
            # covariance is Hermitian; (A^-1 H)^H = H^H A^-1.
            return np.linalg.solve(covariance, model).conj().T
        except np.linalg.LinAlgError:
            return model.conj().T @ np.linalg.pinv(covariance, rcond=1e-8)

    def evaluate(
        self,
        waveform: str,
        snr_db: float,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> Dict[str, float | str]:
        name = str(waveform).upper()
        H = self.equivalent_channel(name, alpha=alpha, beta=beta)
        budget = self._receiver_budget(name)
        if name == "OFDM":
            # A one-tap OFDM receiver estimates only each subcarrier's own
            # coefficient. An ICI coefficient must remain interference even
            # when it is stronger than the diagonal term.
            H_model = np.diag(np.diag(H))
        else:
            H_model = self._retain_strongest_per_row(H, budget)
        noise_var = self.noise_variance(snr_db)
        W = self._interference_aware_lmmse(
            H_model,
            H,
            noise_var,
            diagonal_covariance=(name == "OFDM"),
        )
        raw_gain = np.diag(W @ H)
        valid_gain = np.abs(raw_gain) > 1e-12
        normalization = np.zeros_like(raw_gain)
        normalization[valid_gain] = 1.0 / raw_gain[valid_gain]
        W = normalization[:, None] * W
        residual = W @ H - np.eye(self.K, dtype=np.complex128)
        error_var = (
            np.sum(np.abs(residual) ** 2, axis=1)
            + noise_var * np.sum(np.abs(W) ** 2, axis=1)
        )
        error_var[~valid_gain] = np.inf
        ser_each = self.qam_ser_from_error_variance(error_var, self.context.qam_order)
        offdiag = H - np.diag(np.diag(H))
        total_energy = float(np.linalg.norm(H, "fro") ** 2)
        diag_energy = float(np.linalg.norm(np.diag(np.diag(H)), "fro") ** 2)
        model_energy = float(np.linalg.norm(H_model, "fro") ** 2)
        omitted_energy = float(np.linalg.norm(H - H_model, "fro") ** 2)
        try:
            cond = float(np.linalg.cond(H_model))
        except Exception:
            cond = float("nan")
        return {
            "waveform": name,
            "receiver_policy": self.context.receiver_policy,
            "support_selection_source": (
                OFDM_SUPPORT_SELECTION_SOURCE
                if name == "OFDM"
                else self.context.support_selection_source
            ),
            "ser_prediction_model": "gain_normalized_gaussian_residual",
            "gain_normalization": "diagonal_of_WH",
            "residual_covariance_source": "true_channel_oracle",
            "ser": float(np.mean(ser_each)),
            "evm_percent": float(100.0 * math.sqrt(max(float(np.mean(error_var)), 0.0))),
            "mean_error_variance": float(np.mean(error_var)),
            "noise_variance": float(noise_var),
            "receiver_budget": int(budget),
            "guard_interval_budget_model": "excluded_from_uncoded_ser",
            "guard_overhead_included": False,
            "guard_samples_synthesized": False,
            "diagonal_energy_ratio": float(diag_energy / max(total_energy, 1e-15)),
            "retained_energy_ratio": float(model_energy / max(total_energy, 1e-15)),
            "omitted_energy_ratio": float(omitted_energy / max(total_energy, 1e-15)),
            "model_interference_power": float(omitted_energy / max(self.K, 1)),
            "offdiagonal_energy": float(np.linalg.norm(offdiag, "fro") ** 2),
            "condition_number_model": cond,
            "alpha": float(alpha),
            "beta": float(beta),
            "residual_doppler_spread_hz": float(self.context.residual_doppler_spread_hz),
            "max_residual_doppler_hz": float(
                self.context.max_residual_doppler_hz
                if self.context.max_residual_doppler_hz is not None
                else self.context.residual_doppler_spread_hz
            ),
            "afdm_c1": float(self.afdm_c1),
            "afdm_c2": float(self.afdm_c2),
        }

    def evaluate_all(
        self,
        snr_db: float,
        fdidm_alpha: float,
        fdidm_beta: float,
    ) -> Dict[str, Dict[str, float | str]]:
        return {
            "OFDM": self.evaluate("OFDM", snr_db),
            "OTFS": self.evaluate("OTFS", snr_db),
            "AFDM": self.evaluate("AFDM", snr_db),
            "FDIDM": self.evaluate("FDIDM", snr_db, fdidm_alpha, fdidm_beta),
        }

    @staticmethod
    def _grid(step: float) -> np.ndarray:
        s = float(np.clip(float(step), 0.05, 1.0))
        values = np.arange(0.0, 2.0 + 0.5 * s, s, dtype=np.float64)
        values = np.unique(np.round(np.clip(values, 0.0, 2.0), 9))
        return values

    @staticmethod
    def _neighbourhood(center: float, step: float, radius: int = 2) -> np.ndarray:
        vals = [center + i * step for i in range(-int(radius), int(radius) + 1)]
        vals.extend((0.0, 1.0, 2.0))
        return np.unique(np.round(np.clip(vals, 0.0, 2.0), 9))

    @classmethod
    def optimize_fdidm_over_ensemble(
        cls,
        benchmarks: Sequence["SharedWaveformBenchmark"],
        snr_db: float,
        current_alpha: float = 0.0,
        current_beta: float = 0.0,
        coarse_step: float = 0.5,
        fine_step: float = 0.1,
        decision_ser_floor: float = 1e-8,
        stop_event=None,
    ) -> Dict[str, object]:
        """Coarse/fine alpha-beta search over a slow-time channel ensemble."""

        if not benchmarks:
            raise ValueError("at least one channel benchmark is required")
        t0 = __import__("time").time()
        cache: Dict[Tuple[float, float], Dict[str, object]] = {}
        decision_floor = float(max(1e-15, decision_ser_floor))

        def stopped() -> bool:
            return bool(stop_event is not None and getattr(stop_event, "is_set", lambda: False)())

        def score(a: float, b: float) -> Dict[str, object]:
            key = (round(float(a), 9), round(float(b), 9))
            if key in cache:
                return cache[key]
            values = []
            for bench in benchmarks:
                if stopped():
                    break
                values.append(float(bench.evaluate("FDIDM", snr_db, key[0], key[1])["ser"]))
            if not values:
                values = [1.0]
            raw_safe = np.maximum(np.asarray(values, dtype=np.float64), 1e-15)
            decision_safe = np.maximum(raw_safe, decision_floor)
            decision_ser = max(float(np.mean(values)), decision_floor)
            item = {
                "alpha": float(key[0]),
                "beta": float(key[1]),
                "ser": float(np.mean(values)),
                "decision_ser": float(decision_ser),
                "geometric_ser": float(10.0 ** np.mean(np.log10(raw_safe))),
                "decision_geometric_ser": float(10.0 ** np.mean(np.log10(decision_safe))),
                "ser_samples": [float(x) for x in values],
            }
            cache[key] = item
            return item

        def cancelled_result() -> Dict[str, object]:
            return {
                "cancelled": True,
                "candidate_count": int(len(cache)),
                "search_seconds": float(__import__("time").time() - t0),
                "top_candidates": [dict(x) for x in list(cache.values())[:10]],
                "ensemble_size": int(len(benchmarks)),
                "working_snr_db": float(snr_db),
                "receiver_policy": RECEIVER_POLICY,
                "support_selection_source": SUPPORT_SELECTION_SOURCE,
                "search_mode": "shared_channel_coarse_fine_waveform_specific_lmmse",
            }

        coarse = cls._grid(coarse_step)
        for a, b in itertools.product(coarse, coarse):
            if stopped():
                break
            score(float(a), float(b))
        if stopped():
            return cancelled_result()
        if not cache:
            raise RuntimeError("FDIDM optimization stopped before any candidate")
        def ranking_key(item: Dict[str, object]):
            distance = abs(float(item["alpha"]) - float(current_alpha)) + abs(float(item["beta"]) - float(current_beta))
            return (
                float(item["decision_ser"]),
                float(distance),
                float(item["ser"]),
                float(item["geometric_ser"]),
            )

        best = min(cache.values(), key=ranking_key)

        fine_a = cls._neighbourhood(float(best["alpha"]), fine_step, radius=2)
        fine_b = cls._neighbourhood(float(best["beta"]), fine_step, radius=2)
        for a, b in itertools.product(fine_a, fine_b):
            if stopped():
                break
            score(float(a), float(b))
        if stopped():
            return cancelled_result()

        current = score(float(current_alpha), float(current_beta))
        if stopped():
            return cancelled_result()
        ranked = sorted(cache.values(), key=ranking_key)
        best = ranked[0]
        gain_db = 10.0 * math.log10(
            max(float(current["decision_ser"]), decision_floor)
            / max(float(best["decision_ser"]), decision_floor)
        )
        return {
            "cancelled": False,
            "recommended_alpha": float(best["alpha"]),
            "recommended_beta": float(best["beta"]),
            "predicted_ser_current": float(current["ser"]),
            "predicted_ser_best": float(best["ser"]),
            "predicted_geometric_ser_current": float(current["geometric_ser"]),
            "predicted_geometric_ser_best": float(best["geometric_ser"]),
            "decision_ser_floor": float(decision_floor),
            "decision_geometric_ser_current": float(current["decision_geometric_ser"]),
            "decision_geometric_ser_best": float(best["decision_geometric_ser"]),
            "decision_ser_current": float(current["decision_ser"]),
            "decision_ser_best": float(best["decision_ser"]),
            "predicted_improvement_db": float(gain_db),
            "candidate_count": int(len(cache)),
            "search_seconds": float(__import__("time").time() - t0),
            "top_candidates": [dict(x) for x in ranked[:10]],
            "ensemble_size": int(len(benchmarks)),
            "working_snr_db": float(snr_db),
            "receiver_policy": RECEIVER_POLICY,
            "support_selection_source": SUPPORT_SELECTION_SOURCE,
            "search_mode": "shared_channel_coarse_fine_waveform_specific_lmmse",
        }
