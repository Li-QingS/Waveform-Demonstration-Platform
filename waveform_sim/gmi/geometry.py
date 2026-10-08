"""Receiver-dependent local geometry and anisotropic periodic grids."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence, Tuple

import numpy as np

from waveform_sim.core.transforms import unitary_dft

from .model import ReceiverSpec


@dataclass(frozen=True)
class CurvatureAnalysis:
    curvature: np.ndarray
    leakage_upper: np.ndarray
    discarded_pairs: Tuple[Tuple[int, int], ...]
    valid: bool
    reason: str = ""

    @property
    def loewner_bounds_hold(self) -> bool:
        if not self.valid:
            return False
        lower = np.linalg.eigvalsh(self.curvature - 0.5 * self.leakage_upper)
        upper = np.linalg.eigvalsh(self.leakage_upper - self.curvature)
        return bool(np.min(lower) >= -1e-10 and np.min(upper) >= -1e-10)


@dataclass(frozen=True)
class NumericalCurvature:
    raw: np.ndarray
    projected: np.ndarray
    minimum_raw_eigenvalue: float
    projection_error: float


def fractional_generator(length: int) -> np.ndarray:
    """Hermitian generator A_L from paper equation (15)."""
    length = int(length)
    if length < 2:
        return np.zeros((length, length), dtype=np.complex128)
    U = unitary_dft(length)
    powers = [np.eye(length, dtype=np.complex128)]
    for _ in range(3):
        powers.append(powers[-1] @ U)
    generator = np.zeros_like(U)
    for k in range(4):
        projector = sum(np.exp(-1j * math.pi * k * p / 2.0) * powers[p] for p in range(4)) / 4.0
        generator += (math.pi * k / 2.0) * projector
    return 0.5 * (generator + generator.conj().T)


def fdit_generators(n_frequency: int, n_time: int) -> Tuple[np.ndarray, np.ndarray]:
    af = fractional_generator(int(n_frequency))
    at = fractional_generator(int(n_time))
    alpha = np.kron(np.eye(int(n_time)), af)
    beta = -np.kron(at, np.eye(int(n_frequency)))
    return alpha, beta


def _discarded_pairs(dimension: int, band_k: int) -> Tuple[Tuple[int, int], ...]:
    pairs = []
    for i in range(dimension):
        for j in range(i + 1, dimension):
            distance = min(abs(i - j), dimension - abs(i - j))
            if distance > band_k:
                pairs.append((i, j))
    return tuple(pairs)


def receiver_curvature(
    spectrum: Sequence[float],
    generators: Sequence[np.ndarray],
    snr: float,
    receiver: ReceiverSpec,
    charged_samples: int,
) -> CurvatureAnalysis:
    lambdas = np.asarray(spectrum, dtype=np.float64).reshape(-1)
    rho = float(snr)
    if lambdas.size < 2 or not np.all(np.isfinite(lambdas)) or np.any(lambdas <= 0.0):
        return CurvatureAnalysis(np.zeros((len(generators), len(generators))), np.zeros((len(generators), len(generators))), (), False, "spectrum must be finite, positive and nontrivial")
    if not math.isfinite(rho) or rho <= 0.0 or int(charged_samples) <= 0:
        return CurvatureAnalysis(np.zeros((len(generators), len(generators))), np.zeros((len(generators), len(generators))), (), False, "snr and charged_samples must be positive")
    dimension = lambdas.size
    receiver.validate(dimension)
    matrices = tuple(np.asarray(generator, dtype=np.complex128) for generator in generators)
    if not matrices or any(matrix.shape != (dimension, dimension) for matrix in matrices):
        return CurvatureAnalysis(np.zeros((len(matrices), len(matrices))), np.zeros((len(matrices), len(matrices))), (), False, "generator shape mismatch")
    band = receiver.effective_band_k(dimension)
    pairs = _discarded_pairs(dimension, band)
    count = len(matrices)
    Q = np.zeros((count, count), dtype=np.float64)
    U = np.zeros_like(Q)
    for i, j in pairs:
        delta = lambdas[i] - lambdas[j]
        denominator = (1.0 + rho * lambdas[i] ** 2) * (1.0 + rho * lambdas[j] ** 2)
        weight = (
            2.0
            * rho
            * delta ** 2
            / denominator
            * (1.0 + 0.5 * rho * delta ** 2 + rho ** 2 * lambdas[i] ** 2 * lambdas[j] ** 2)
        )
        # T_k=[Lambda,jA_k], with |T_ij|^2=delta^2|A_ij|^2.
        for k in range(count):
            for l in range(k, count):
                gram = float(np.real(np.conj(matrices[k][i, j]) * matrices[l][i, j]))
                Q[k, l] += weight * gram / float(charged_samples)
                # Frobenius trace contains both Hermitian entries (i,j),(j,i).
                U[k, l] += 2.0 * rho * delta ** 2 * gram / float(charged_samples)
                Q[l, k] = Q[k, l]
                U[l, k] = U[k, l]
    reason = "" if pairs else "receiver discards no generator pairs"
    return CurvatureAnalysis(Q, U, pairs, True, reason)


def leakage_curvature(
    spectrum: Sequence[float],
    generators: Sequence[np.ndarray],
    snr: float,
    receiver: ReceiverSpec,
    charged_samples: int,
) -> np.ndarray:
    return receiver_curvature(spectrum, generators, snr, receiver, charged_samples).leakage_upper


def finite_difference_curvature(
    score_field: Callable[[float, float], float],
    optimum: Tuple[float, float] = (0.0, 0.0),
    step: float = 1e-3,
) -> NumericalCurvature:
    h = float(step)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("step must be positive and finite")
    a, b = map(float, optimum)
    f0 = float(score_field(a, b))
    f_ap = float(score_field(a + h, b))
    f_am = float(score_field(a - h, b))
    f_bp = float(score_field(a, b + h))
    f_bm = float(score_field(a, b - h))
    f_pp = float(score_field(a + h, b + h))
    f_pm = float(score_field(a + h, b - h))
    f_mp = float(score_field(a - h, b + h))
    f_mm = float(score_field(a - h, b - h))
    values = np.asarray([f0, f_ap, f_am, f_bp, f_bm, f_pp, f_pm, f_mp, f_mm])
    if not np.all(np.isfinite(values)):
        raise ValueError("score_field returned non-finite values")
    q11 = (2.0 * f0 - f_ap - f_am) / (2.0 * h ** 2)
    q22 = (2.0 * f0 - f_bp - f_bm) / (2.0 * h ** 2)
    q12 = -(f_pp - f_pm - f_mp + f_mm) / (8.0 * h ** 2)
    raw = np.asarray([[q11, q12], [q12, q22]], dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(raw)
    projected_values = np.maximum(eigenvalues, 0.0)
    projected = (eigenvectors * projected_values) @ eigenvectors.T
    return NumericalCurvature(raw, projected, float(np.min(eigenvalues)), float(np.linalg.norm(projected - raw, ord="fro")))


def allocate_axis_counts(curvature: np.ndarray, capacity: int) -> Tuple[int, int]:
    matrix = np.asarray(curvature, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("curvature must be a finite 2x2 matrix")
    capacity = int(capacity)
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    q_alpha = max(0.0, float(matrix[0, 0]))
    q_beta = max(0.0, float(matrix[1, 1]))
    best = None
    for n_alpha in range(1, capacity + 1):
        for n_beta in range(1, capacity // n_alpha + 1):
            objective = q_alpha / n_alpha ** 2 + q_beta / n_beta ** 2
            # Prefer fuller use, then a deterministic alpha-first ordering.
            key = (objective, -(n_alpha * n_beta), n_alpha, n_beta)
            if best is None or key < best[0]:
                best = (key, n_alpha, n_beta)
    return int(best[1]), int(best[2])


def anisotropic_periodic_grid(
    n_alpha: int,
    n_beta: int,
    period: float = 2.0,
    *,
    lower: float = 0.0,
) -> Tuple[Tuple[float, float], ...]:
    n_alpha, n_beta = int(n_alpha), int(n_beta)
    period = float(period)
    if n_alpha <= 0 or n_beta <= 0 or not math.isfinite(period) or period <= 0.0:
        raise ValueError("grid counts and period must be positive")
    alpha = lower + (np.arange(n_alpha) + 0.5) * period / n_alpha
    beta = lower + (np.arange(n_beta) + 0.5) * period / n_beta
    return tuple((float(a), float(b)) for a in alpha for b in beta)


__all__ = [
    "CurvatureAnalysis",
    "NumericalCurvature",
    "allocate_axis_counts",
    "anisotropic_periodic_grid",
    "fdit_generators",
    "finite_difference_curvature",
    "fractional_generator",
    "leakage_curvature",
    "receiver_curvature",
]
