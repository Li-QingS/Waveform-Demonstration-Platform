"""Reproduce the finite-library example in Section VI of the GMI paper."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from waveform_sim.core.transforms import gamma_matrix, unitary_dft
from waveform_sim.gmi.geometry import allocate_axis_counts, fdit_generators, receiver_curvature
from waveform_sim.gmi.model import ReceiverSpec


SPECTRUM = np.asarray([1.9, 1.4, 1.0, 0.6], dtype=np.float64)
RADIUS = 0.04
CHARGED_SAMPLES = 18
EVALUATION_POINTS = 33


def _raw_fdit(alpha: float, beta: float) -> np.ndarray:
    V = np.kron(gamma_matrix(2, -float(beta)), gamma_matrix(2, float(alpha)))
    S0 = np.kron(np.eye(2, dtype=np.complex128), unitary_dft(2).conj().T)
    return S0 @ V


def _diagonal_rate(B: np.ndarray, rho: float) -> float:
    diagonal_energy = np.abs(np.diag(B)) ** 2
    x = float(rho) * diagonal_energy
    y = float(rho) * np.maximum(0.0, np.sum(np.abs(B) ** 2, axis=1) - diagonal_energy)

    def derivative(s: float) -> float:
        denominator = 1.0 + s * x
        numerator = s * x * (1.0 - s * (1.0 + y))
        numerator_derivative = x - 2.0 * s * x * (1.0 + y)
        return float(np.sum(x / denominator + (numerator_derivative * denominator - numerator * x) / denominator ** 2))

    lo, hi = 0.0, 1.0
    if derivative(hi) >= 0.0:
        optimum = hi
    else:
        for _ in range(48):
            middle = 0.5 * (lo + hi)
            if derivative(middle) > 0.0:
                lo = middle
            else:
                hi = middle
        optimum = 0.5 * (lo + hi)
    denominator = 1.0 + optimum * x
    objective = np.sum(np.log(denominator) + optimum * x * (1.0 - optimum * (1.0 + y)) / denominator)
    return float(objective / CHARGED_SAMPLES)


def _batch_relative_rates(delta_pairs: Sequence[Tuple[float, float]], rho: float) -> dict[Tuple[float, float], float]:
    diagonal = np.diag(SPECTRUM)
    reference = _raw_fdit(0.0, 0.0)
    x_rows = []
    y_rows = []
    keys = []
    for delta_alpha, delta_beta in delta_pairs:
        key = (round(float(delta_alpha), 12), round(float(delta_beta), 12))
        relative = _raw_fdit(key[0], key[1])
        channel = relative @ diagonal @ relative.conj().T
        B = reference.conj().T @ channel @ reference
        diagonal_energy = np.abs(np.diag(B)) ** 2
        x_rows.append(float(rho) * diagonal_energy)
        y_rows.append(float(rho) * np.maximum(0.0, np.sum(np.abs(B) ** 2, axis=1) - diagonal_energy))
        keys.append(key)
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    lo = np.zeros(x.shape[0], dtype=np.float64)
    hi = np.ones(x.shape[0], dtype=np.float64)
    for _ in range(48):
        s = 0.5 * (lo + hi)
        denominator = 1.0 + s[:, None] * x
        numerator = s[:, None] * x * (1.0 - s[:, None] * (1.0 + y))
        numerator_derivative = x - 2.0 * s[:, None] * x * (1.0 + y)
        derivative = np.sum(x / denominator + (numerator_derivative * denominator - numerator * x) / denominator ** 2, axis=1)
        positive = derivative > 0.0
        lo[positive] = s[positive]
        hi[~positive] = s[~positive]
    optimum = 0.5 * (lo + hi)
    denominator = 1.0 + optimum[:, None] * x
    objective = np.sum(np.log(denominator) + optimum[:, None] * x * (1.0 - optimum[:, None] * (1.0 + y)) / denominator, axis=1)
    return {key: float(value / CHARGED_SAMPLES) for key, value in zip(keys, objective)}


def _cell_centers(count: int) -> np.ndarray:
    count = int(count)
    return -RADIUS + (2.0 * np.arange(count) + 1.0) * RADIUS / count


def sampled_regret(n_alpha: int, n_beta: int, rho: float, evaluation_points: int = EVALUATION_POINTS) -> float:
    state_axis = np.linspace(-RADIUS, RADIUS, int(evaluation_points))
    actions = [(float(a), float(b)) for a in _cell_centers(n_alpha) for b in _cell_centers(n_beta)]
    oracle = float(np.sum(np.log(1.0 + float(rho) * SPECTRUM ** 2)) / CHARGED_SAMPLES)
    maximum = 0.0
    delta_alpha = sorted({round(float(state_value) - action_alpha, 12) for state_value in state_axis for action_alpha, _ in actions})
    delta_beta = sorted({round(float(state_value) - action_beta, 12) for state_value in state_axis for _, action_beta in actions})
    rate_cache = _batch_relative_rates([(a, b) for a in delta_alpha for b in delta_beta], rho)

    for state_alpha in state_axis:
        for state_beta in state_axis:
            best = max(rate_cache[(round(float(state_alpha) - action_alpha, 12), round(float(state_beta) - action_beta, 12))] for action_alpha, action_beta in actions)
            maximum = max(maximum, max(0.0, oracle - best))
    return float(maximum)


def reproduce_paper_example(evaluation_points: int = EVALUATION_POINTS) -> dict:
    receiver = ReceiverSpec("diagonal")
    generators = fdit_generators(2, 2)
    results = []
    for rho in (1.0, 10.0):
        analysis = receiver_curvature(SPECTRUM, generators, rho, receiver, CHARGED_SAMPLES)
        n_alpha, n_beta = allocate_axis_counts(analysis.curvature, 36)
        anisotropic = sampled_regret(n_alpha, n_beta, rho, evaluation_points)
        equal = sampled_regret(6, 6, rho, evaluation_points)
        improvement = 100.0 * (equal - anisotropic) / equal
        results.append(
            {
                "curvature_diagonal": [float(analysis.curvature[0, 0]), float(analysis.curvature[1, 1])],
                "equal_grid": [6, 6],
                "equal_sampled_regret": equal,
                "n_alpha": n_alpha,
                "n_beta": n_beta,
                "quadratic_sampled_regret": anisotropic,
                "relative_improvement_percent": improvement,
                "rho": rho,
            }
        )
    return {
        "charged_samples": CHARGED_SAMPLES,
        "evaluation_grid": [int(evaluation_points), int(evaluation_points)],
        "library_budget": 36,
        "radius": RADIUS,
        "results": results,
        "spectrum": SPECTRUM.tolist(),
        "version": 1,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-points", type=int, default=EVALUATION_POINTS)
    parser.add_argument("--json", type=Path, help="optional output JSON path")
    args = parser.parse_args(argv)
    result = reproduce_paper_example(args.evaluation_points)
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
    print(encoded)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
