"""Reproduce the finite-library example in Section VI of the GMI paper."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
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
DEFAULT_FIGURE = PROJECT_ROOT / "artifacts" / "gmi_paper_example.png"


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


def _validated_figure_rows(report: dict) -> list[dict]:
    results = report.get("results")
    if not isinstance(results, list) or len(results) != 2:
        raise ValueError("paper figure requires exactly two rho result rows")
    required = {
        "curvature_diagonal",
        "equal_grid",
        "equal_sampled_regret",
        "n_alpha",
        "n_beta",
        "quadratic_sampled_regret",
        "relative_improvement_percent",
        "rho",
    }
    rows = []
    for raw in results:
        if not isinstance(raw, dict) or not required.issubset(raw):
            missing = sorted(required.difference(raw if isinstance(raw, dict) else {}))
            raise ValueError(f"paper figure result is missing fields: {', '.join(missing)}")
        row = dict(raw)
        curvature = np.asarray(row["curvature_diagonal"], dtype=np.float64).reshape(-1)
        equal_grid = np.asarray(row["equal_grid"], dtype=np.int64).reshape(-1)
        numeric = np.asarray(
            [
                row["rho"],
                row["equal_sampled_regret"],
                row["quadratic_sampled_regret"],
                row["relative_improvement_percent"],
                row["n_alpha"],
                row["n_beta"],
            ],
            dtype=np.float64,
        )
        if curvature.size != 2 or equal_grid.size != 2:
            raise ValueError("paper figure requires two curvature axes and a two-axis equal grid")
        if not np.all(np.isfinite(curvature)) or np.any(curvature <= 0.0):
            raise ValueError("paper figure curvature must be positive and finite")
        if not np.all(np.isfinite(numeric)):
            raise ValueError("paper figure values must be finite")
        if min(float(row["equal_sampled_regret"]), float(row["quadratic_sampled_regret"])) <= 0.0:
            raise ValueError("paper figure regret must be positive")
        if min(int(row["n_alpha"]), int(row["n_beta"]), int(equal_grid[0]), int(equal_grid[1])) <= 0:
            raise ValueError("paper figure grid counts must be positive")
        row["curvature_diagonal"] = curvature
        row["equal_grid"] = equal_grid
        rows.append(row)
    return sorted(rows, key=lambda item: float(item["rho"]))


def _draw_periodic_grid(axis, n_alpha: int, n_beta: int, *, color: str, title: str) -> None:
    alpha = (2.0 * np.arange(int(n_alpha)) + 1.0) / float(n_alpha)
    beta = (2.0 * np.arange(int(n_beta)) + 1.0) / float(n_beta)
    aa, bb = np.meshgrid(alpha, beta, indexing="ij")
    axis.scatter(aa.reshape(-1), bb.reshape(-1), s=34, color=color, edgecolors="white", linewidths=0.5)
    axis.set(xlim=(0.0, 2.0), ylim=(0.0, 2.0), xlabel=r"$\alpha$", ylabel=r"$\beta$", title=title)
    axis.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    axis.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.22)


def render_paper_figure(report: dict, output_path: str | os.PathLike[str]) -> Path:
    """Render the paper reproduction summary to an atomic, headless PNG."""
    rows = _validated_figure_rows(report)
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16.0, 10.0), dpi=120)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.08))
    curvature_axis = fig.add_subplot(grid[0, 0])
    equal_axis = fig.add_subplot(grid[0, 1])
    adaptive_axis = fig.add_subplot(grid[0, 2])
    regret_axis = fig.add_subplot(grid[1, :])

    x = np.arange(len(rows), dtype=np.float64)
    width = 0.34
    alpha_values = np.asarray([row["curvature_diagonal"][0] for row in rows], dtype=np.float64)
    beta_values = np.asarray([row["curvature_diagonal"][1] for row in rows], dtype=np.float64)
    alpha_bars = curvature_axis.bar(x - width / 2.0, alpha_values, width, label=r"$Q_{\alpha\alpha}$", color="#3976af")
    beta_bars = curvature_axis.bar(x + width / 2.0, beta_values, width, label=r"$Q_{\beta\beta}$", color="#e1812c")
    curvature_axis.set_yscale("log")
    curvature_axis.set_xticks(x, [rf"$\rho={float(row['rho']):g}$" for row in rows])
    curvature_axis.set_ylabel("Receiver-dependent curvature")
    curvature_axis.set_title("Local GMI sensitivity")
    curvature_axis.grid(True, axis="y", which="both", alpha=0.22)
    curvature_axis.legend(frameon=False)
    for bars in (alpha_bars, beta_bars):
        curvature_axis.bar_label(bars, fmt="%.3g", padding=3, fontsize=9)

    equal_grid = rows[0]["equal_grid"]
    _draw_periodic_grid(
        equal_axis,
        int(equal_grid[0]),
        int(equal_grid[1]),
        color="#777777",
        title=f"Equal allocation: {int(equal_grid[0])} x {int(equal_grid[1])} = 36",
    )
    _draw_periodic_grid(
        adaptive_axis,
        int(rows[0]["n_alpha"]),
        int(rows[0]["n_beta"]),
        color="#2a9d67",
        title=f"Receiver-dependent: {int(rows[0]['n_alpha'])} x {int(rows[0]['n_beta'])} = 36",
    )

    equal_regret = np.asarray([row["equal_sampled_regret"] for row in rows], dtype=np.float64)
    adaptive_regret = np.asarray([row["quadratic_sampled_regret"] for row in rows], dtype=np.float64)
    equal_bars = regret_axis.bar(x - width / 2.0, equal_regret, width, label="Equal 6 x 6", color="#777777")
    adaptive_bars = regret_axis.bar(x + width / 2.0, adaptive_regret, width, label="Receiver-dependent 4 x 9", color="#2a9d67")
    regret_axis.set_yscale("log")
    regret_axis.set_xticks(x, [rf"$\rho={float(row['rho']):g}$" for row in rows])
    regret_axis.set_ylabel("Maximum sampled regret (nat/sample, log scale)")
    regret_axis.set_title("Same 36-token budget: lower worst-case regret")
    regret_axis.grid(True, axis="y", which="both", alpha=0.22)
    regret_axis.legend(frameon=False, loc="upper left")
    for bars in (equal_bars, adaptive_bars):
        regret_axis.bar_label(bars, labels=[f"{bar.get_height():.3e}" for bar in bars], padding=4, fontsize=9)
    for index, row in enumerate(rows):
        top = max(equal_regret[index], adaptive_regret[index])
        regret_axis.annotate(
            f"{float(row['relative_improvement_percent']):.2f}% lower",
            xy=(index, top),
            xytext=(0, 24),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#176b43",
        )

    evaluation_grid = report.get("evaluation_grid", ["?", "?"])
    spectrum = report.get("spectrum", [])
    fig.suptitle("Receiver-Dependent GMI Geometry for a Finite Transceiver Library", fontsize=18, fontweight="bold")
    fig.text(
        0.5,
        0.018,
        f"Library budget: {report.get('library_budget', '?')} | Evaluation grid: {evaluation_grid[0]} x {evaluation_grid[1]} | "
        f"Spectrum: {spectrum} | Periodic domain: [0, 2) x [0, 2)",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.085, top=0.90, hspace=0.26, wspace=0.25)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".png", dir=str(target.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        fig.savefig(temporary, format="png", dpi=120, facecolor="white")
        if temporary.stat().st_size < 1024 or temporary.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError("renderer did not produce a valid PNG")
        os.replace(temporary, target)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-points", type=int, default=EVALUATION_POINTS)
    parser.add_argument("--json", type=Path, help="optional output JSON path")
    figure_group = parser.add_mutually_exclusive_group()
    figure_group.add_argument("--figure", type=Path, help=f"PNG output path (default: {DEFAULT_FIGURE})")
    figure_group.add_argument("--no-figure", action="store_true", help="print numeric results without generating a PNG")
    args = parser.parse_args(argv)
    result = reproduce_paper_example(args.evaluation_points)
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
    print(encoded)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n", encoding="utf-8")
    if not args.no_figure:
        output = render_paper_figure(result, args.figure or DEFAULT_FIGURE)
        print(f"Figure: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
