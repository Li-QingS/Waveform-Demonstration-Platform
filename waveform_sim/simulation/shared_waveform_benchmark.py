# -*- coding: utf-8 -*-
"""Shared-channel OFDM/OTFS/AFDM/FDIDM performance benchmark.

All four waveforms are evaluated on the *same* physical channel realization and
with the same QAM/SNR convention.  This replaces the former shortcut that drew
OFDM, OTFS and AFDM as three alpha/beta points inside the FDIDM theory kernel.

Receiver model
--------------
A fully dense, perfect-CSI MMSE receiver is invariant to many unitary waveform
changes and therefore hides the practical difference between waveform domains.
The benchmark instead uses a transparent complexity-limited receiver:

* OFDM: one-tap diagonal equalization; off-diagonal ICI is model mismatch.
* OTFS: strongest-coefficient sparse 2-D MMSE equalization.
* AFDM: strongest-coefficient sparse affine-domain MMSE equalization.
* FDIDM: the same sparse-MMSE budget as OTFS/AFDM, while alpha/beta select the
  fractional domain that best concentrates the current channel.

For a retained channel model H_d and the true equivalent channel H, the omitted
coupling H_o=H-H_d is treated as colored interference instead of being silently
ignored.  The complexity-limited LMMSE receiver is

    W = H_d^H (H_d H_d^H + H_o H_o^H + sigma^2 I)^-1.

The effective symbol error variance is then evaluated against the *true* channel:

    diag((W H-I)(W H-I)^H + sigma^2 W W^H).

That quantity is passed to the same square-QAM SER expression for every
waveform.  No arbitrary curve offsets are used.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BenchmarkContext:
    m_subcarriers: int
    n_symbols: int
    subcarrier_spacing_hz: float
    qam_order: int
    bits_per_symbol: int
    snr_definition: str = "Eb/N0"
    detector: str = "MMSE"
    path_count: int = 3
    maximum_delay_ns: float = 0.0
    residual_doppler_spread_hz: float = 200.0

    @property
    def k(self) -> int:
        return int(self.m_subcarriers * self.n_symbols)

    @property
    def sample_rate_hz(self) -> float:
        return float(self.m_subcarriers * self.subcarrier_spacing_hz)


class SharedWaveformBenchmark:
    """Evaluate four waveform domains on one fixed H_TF realization."""

    WAVEFORMS = ("OFDM", "OTFS", "AFDM", "FDIDM")

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
        ctx = BenchmarkContext(
            m_subcarriers=int(cfg.m_subcarriers),
            n_symbols=int(cfg.n_symbols),
            subcarrier_spacing_hz=float(cfg.subcarrier_spacing_hz),
            qam_order=int(obj.qam_order),
            bits_per_symbol=int(obj.bits_per_symbol),
            snr_definition=str(getattr(cfg, "snr_definition", "Eb/N0")),
            detector=str(getattr(cfg, "decoder", "MMSE")),
            path_count=max(1, len(paths)),
            maximum_delay_ns=float(max_delay_ns),
            residual_doppler_spread_hz=float(residual_spread),
        )
        return cls(ctx, h_tf, obj._fdit_matrices)

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
        nu_max = abs(self.context.residual_doppler_spread_hz) * self.K / fs
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

    def _cp_efficiency(self, waveform: str) -> float:
        delay_samples = int(math.ceil(self.context.maximum_delay_ns * 1e-9 * self.context.sample_rate_hz))
        cp = max(0, delay_samples)
        if cp <= 0:
            return 1.0
        name = str(waveform).upper()
        if name == "OFDM":
            # One CP per OFDM symbol.
            return float(self.M / (self.M + cp))
        # One guard interval per 2-D/affine frame in this compact comparison.
        return float(self.K / (self.K + cp))

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
        # argpartition avoids a full sort for every row.
        for i in range(rows):
            idx = np.argpartition(np.abs(matrix[i]), -keep)[-keep:]
            out[i, idx] = matrix[i, idx]
        return out

    @staticmethod
    def _interference_aware_lmmse(
        H_model: np.ndarray,
        H_true: np.ndarray,
        noise_var: float,
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
        H_model = self._retain_strongest_per_row(H, budget)
        eta = max(self._cp_efficiency(name), 1e-9)
        noise_var = self.noise_variance(snr_db) / eta
        W = self._interference_aware_lmmse(H_model, H, noise_var)
        residual = W @ H - np.eye(self.K, dtype=np.complex128)
        error_var = (
            np.sum(np.abs(residual) ** 2, axis=1)
            + noise_var * np.sum(np.abs(W) ** 2, axis=1)
        )
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
            "ser": float(np.mean(ser_each)),
            "evm_percent": float(100.0 * math.sqrt(max(float(np.mean(error_var)), 0.0))),
            "mean_error_variance": float(np.mean(error_var)),
            "noise_variance": float(noise_var),
            "receiver_budget": int(budget),
            "cp_efficiency": float(eta),
            "diagonal_energy_ratio": float(diag_energy / max(total_energy, 1e-15)),
            "retained_energy_ratio": float(model_energy / max(total_energy, 1e-15)),
            "omitted_energy_ratio": float(omitted_energy / max(total_energy, 1e-15)),
            "model_interference_power": float(omitted_energy / max(self.K, 1)),
            "offdiagonal_energy": float(np.linalg.norm(offdiag, "fro") ** 2),
            "condition_number_model": cond,
            "alpha": float(alpha),
            "beta": float(beta),
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
            item = {
                "alpha": float(key[0]),
                "beta": float(key[1]),
                "ser": float(np.mean(values)),
                "geometric_ser": float(10.0 ** np.mean(np.log10(raw_safe))),
                "decision_geometric_ser": float(10.0 ** np.mean(np.log10(decision_safe))),
                "ser_samples": [float(x) for x in values],
            }
            cache[key] = item
            return item

        coarse = cls._grid(coarse_step)
        for a, b in itertools.product(coarse, coarse):
            if stopped():
                break
            score(float(a), float(b))
        if not cache:
            raise RuntimeError("FDIDM optimization stopped before any candidate")
        def ranking_key(item: Dict[str, object]):
            distance = abs(float(item["alpha"]) - float(current_alpha)) + abs(float(item["beta"]) - float(current_beta))
            return (
                float(item["decision_geometric_ser"]),
                float(distance),
                float(item["geometric_ser"]),
                float(item["ser"]),
            )

        best = min(cache.values(), key=ranking_key)

        fine_a = cls._neighbourhood(float(best["alpha"]), fine_step, radius=2)
        fine_b = cls._neighbourhood(float(best["beta"]), fine_step, radius=2)
        for a, b in itertools.product(fine_a, fine_b):
            if stopped():
                break
            score(float(a), float(b))

        current = score(float(current_alpha), float(current_beta))
        ranked = sorted(cache.values(), key=ranking_key)
        best = ranked[0]
        gain_db = 10.0 * math.log10(
            max(float(current["decision_geometric_ser"]), decision_floor)
            / max(float(best["decision_geometric_ser"]), decision_floor)
        )
        return {
            "recommended_alpha": float(best["alpha"]),
            "recommended_beta": float(best["beta"]),
            "predicted_ser_current": float(current["ser"]),
            "predicted_ser_best": float(best["ser"]),
            "predicted_geometric_ser_current": float(current["geometric_ser"]),
            "predicted_geometric_ser_best": float(best["geometric_ser"]),
            "decision_ser_floor": float(decision_floor),
            "decision_geometric_ser_current": float(current["decision_geometric_ser"]),
            "decision_geometric_ser_best": float(best["decision_geometric_ser"]),
            "predicted_improvement_db": float(gain_db),
            "candidate_count": int(len(cache)),
            "search_seconds": float(__import__("time").time() - t0),
            "top_candidates": [dict(x) for x in ranked[:10]],
            "ensemble_size": int(len(benchmarks)),
            "working_snr_db": float(snr_db),
            "search_mode": "shared_channel_coarse_fine_sparse_mmse",
        }
