#!/usr/bin/env python3
"""Offline validation for the hardware FDIDM alpha/beta optimizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from waveform_sim.hardware.fdidm_adaptive import FDIDMAdaptiveMixin


PERTURBATIONS = (
    "identity",
    "global_scale",
    "frequency_selective_gain",
    "time_selective_gain",
)
SER_FLOOR = 1e-15


@dataclass(frozen=True)
class ExperimentConfig:
    m: int = 4
    n: int = 4
    sample_count: int = 8
    seed: int = 20260827
    reference_step: float = 0.05
    coarse_step: float = 0.25
    fine_step: float = 0.05
    mod_order: str = "16QAM"
    equalizer: str = "MMSE"
    noise_var: float = 0.06309573444801933
    channel_kinds: Tuple[str, ...] = ("diag", "full")
    perturbations: Tuple[str, ...] = PERTURBATIONS
    global_scale_real: float = 0.8
    global_scale_imag: float = 0.3
    selective_gain_min: float = 0.5
    selective_gain_max: float = 1.5

    def validate(self) -> None:
        if self.m < 2 or self.n < 2:
            raise ValueError("m and n must both be at least 2")
        if self.m * self.n > 512:
            raise ValueError("m*n must not exceed the production default max_order=512")
        if self.sample_count < 1:
            raise ValueError("sample_count must be at least 1")
        for name, value in (
            ("reference_step", self.reference_step),
            ("coarse_step", self.coarse_step),
            ("fine_step", self.fine_step),
        ):
            if not np.isfinite(value) or value <= 0.0 or value > 2.0:
                raise ValueError(f"{name} must be in (0, 2]")
        if self.fine_step > self.coarse_step:
            raise ValueError("fine_step must not exceed coarse_step")
        if not np.isfinite(self.noise_var) or self.noise_var <= 0.0:
            raise ValueError("noise_var must be finite and positive")
        if self.mod_order.upper() not in ("QPSK", "16QAM", "64QAM"):
            raise ValueError("mod_order must be QPSK, 16QAM, or 64QAM")
        if self.equalizer.upper() not in ("ZF", "MMSE"):
            raise ValueError("equalizer must be ZF or MMSE")
        if not self.channel_kinds or any(k not in ("diag", "full") for k in self.channel_kinds):
            raise ValueError("channel_kinds must contain diag and/or full")
        unknown = sorted(set(self.perturbations) - set(PERTURBATIONS))
        if not self.perturbations or unknown:
            raise ValueError(f"unknown or empty perturbation list: {unknown}")
        scale = complex(self.global_scale_real, self.global_scale_imag)
        if abs(scale) <= 1e-12:
            raise ValueError("global scale must be nonzero")
        if not (0.0 < self.selective_gain_min <= self.selective_gain_max):
            raise ValueError("selective gain range must be positive and ordered")


class AdaptiveKernel(FDIDMAdaptiveMixin):
    """Minimal transform host for the production optimizer mixin."""

    def __init__(self) -> None:
        self._gamma_cache: Dict[Tuple[int, float], np.ndarray] = {}

    @staticmethod
    def _wrap_index(value: float) -> float:
        value = ((float(value) + 2.0) % 4.0) - 2.0
        return 2.0 if value <= -2.0 + 1e-12 else value

    @staticmethod
    def _ap_weight(power: int, eps: float) -> complex:
        delta = float(eps) - float(power)
        return (np.cos(delta * np.pi / 4.0)
                * np.cos(2.0 * delta * np.pi / 4.0)
                * np.exp(1j * 3.0 * delta * np.pi / 4.0))

    @staticmethod
    def _unitary_dft_matrix(order: int) -> np.ndarray:
        index = np.arange(int(order), dtype=np.float64)
        return (np.exp(-1j * 2.0 * np.pi * np.outer(index, index) / int(order))
                / np.sqrt(float(order)))

    def _gamma(self, order: int, eps: float) -> np.ndarray:
        n = int(order)
        wrapped = self._wrap_index(eps)
        key = (n, round(wrapped, 12))
        if key not in self._gamma_cache:
            fourier = self._unitary_dft_matrix(n)
            identity = np.eye(n, dtype=np.complex128)
            powers = (identity, fourier, fourier @ fourier, fourier.conj().T)
            gamma = np.zeros((n, n), dtype=np.complex128)
            for power in range(4):
                gamma += powers[power] * self._ap_weight(power, wrapped)
            self._gamma_cache[key] = gamma
        return self._gamma_cache[key]

    @staticmethod
    def _dft_power_apply_axis(array: np.ndarray, power: int, axis: int) -> np.ndarray:
        values = np.asarray(array, dtype=np.complex128)
        axis = int(axis) % values.ndim
        order = int(values.shape[axis])
        power = int(power) % 4
        if power == 0 or order <= 1:
            return values.copy()
        if power == 1:
            return np.fft.fft(values, axis=axis) / np.sqrt(float(order))
        if power == 3:
            return np.fft.ifft(values, axis=axis) * np.sqrt(float(order))
        reverse = np.concatenate(([0], np.arange(order - 1, 0, -1))).astype(np.int64)
        return np.take(values, reverse, axis=axis)

    def _apply_gamma_axis(self, array: np.ndarray, eps: float, axis: int) -> np.ndarray:
        values = np.asarray(array, dtype=np.complex128)
        wrapped = self._wrap_index(eps)
        result = np.zeros_like(values, dtype=np.complex128)
        for power in range(4):
            result += (self._ap_weight(power, wrapped)
                       * self._dft_power_apply_axis(values, power, axis))
        return result


def _seed_for(config: ExperimentConfig, kind: str, sample_id: int) -> int:
    kind_offset = 0 if kind == "diag" else 1_000_003
    return int((config.seed + kind_offset + 7919 * sample_id) & 0xFFFFFFFF)


def generate_channel(config: ExperimentConfig, kind: str, sample_id: int) -> np.ndarray:
    """Generate one deterministic channel without touching global RNG state."""
    rng = np.random.default_rng(_seed_for(config, kind, sample_id))
    m, n, order = config.m, config.n, config.m * config.n
    if kind == "diag":
        amplitude = np.exp(0.45 * rng.standard_normal((m, n)))
        phase = np.exp(1j * 2.0 * np.pi * rng.random((m, n)))
        return np.asarray(amplitude * phase, dtype=np.complex128)
    if kind != "full":
        raise ValueError(f"unsupported channel kind: {kind}")
    dense = ((rng.standard_normal((order, order))
              + 1j * rng.standard_normal((order, order))) / np.sqrt(2.0 * order))
    if sample_id % 2 == 0:
        dense = 0.75 * np.diag(np.diag(dense)) + 0.25 * dense
    row_power = np.mean(np.abs(dense) ** 2, axis=1)
    dense /= np.sqrt(max(float(np.mean(row_power)), 1e-15))
    return np.asarray(dense, dtype=np.complex128)


def _axis_gains(config: ExperimentConfig, kind: str) -> Tuple[np.ndarray, np.ndarray]:
    freq = np.linspace(config.selective_gain_min, config.selective_gain_max, config.m)
    time_axis = np.linspace(config.selective_gain_max, config.selective_gain_min, config.n)
    if kind == "diag":
        return freq[:, None], time_axis[None, :]
    freq_vector = np.tile(freq, config.n)
    time_vector = np.repeat(time_axis, config.m)
    return freq_vector, time_vector


def apply_perturbation(
    config: ExperimentConfig,
    channel: np.ndarray,
    kind: str,
    perturbation: str,
    noise_var: float,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    values = np.asarray(channel, dtype=np.complex128)
    if perturbation == "identity":
        return values.copy(), float(noise_var), {}
    if perturbation == "global_scale":
        scale = complex(config.global_scale_real, config.global_scale_imag)
        return values * scale, float(noise_var * abs(scale) ** 2), {
            "scale_real": float(scale.real), "scale_imag": float(scale.imag)
        }
    freq, time_axis = _axis_gains(config, kind)
    if perturbation == "frequency_selective_gain":
        gain = freq
    elif perturbation == "time_selective_gain":
        gain = time_axis
    else:
        raise ValueError(f"unsupported perturbation: {perturbation}")
    if kind == "diag":
        perturbed = values * gain
    else:
        perturbed = np.diag(np.asarray(gain).reshape(-1, order="F")) @ values
    return np.asarray(perturbed, dtype=np.complex128), float(noise_var), {
        "gain_min": float(np.min(gain)), "gain_max": float(np.max(gain))
    }


def build_snapshot(
    config: ExperimentConfig,
    channel: np.ndarray,
    kind: str,
    noise_var: float,
) -> Dict[str, Any]:
    return {
        "M": int(config.m),
        "N": int(config.n),
        "htf": np.asarray(channel, dtype=np.complex128),
        "htf_kind": str(kind),
        "htf_source": "offline_validation",
        "noise_var": float(noise_var),
        "equalizer": config.equalizer.upper(),
        "mod_order": config.mod_order.upper(),
        "alpha": 0.0,
        "beta": 0.0,
        "coarse_step": float(config.coarse_step),
        "fine_step": float(config.fine_step),
        "integer_margin_db": 0.0,
        "max_order": 512,
        "rcond": 1e-6,
        "frame_counter": 0,
        "snapshot_seq": 0,
    }


def _reference_candidates(step: float) -> List[Tuple[float, float]]:
    count = int(math.floor(2.0 / float(step) + 1e-9))
    values = [min(2.0, index * float(step)) for index in range(count + 1)]
    values.extend((0.0, 1.0, 2.0))
    unique = sorted(set(round(value, 9) for value in values if value <= 2.0 + 1e-9))
    return [(float(alpha), float(beta)) for alpha in unique for beta in unique]


def exhaustive_reference(
    kernel: AdaptiveKernel,
    snapshot: Dict[str, Any],
    candidates: Sequence[Tuple[float, float]],
) -> Dict[str, float]:
    prepared, _ = kernel._adaptive_prepare_base(snapshot)
    results = kernel._adaptive_evaluate_candidates(
        prepared,
        list(candidates),
        int(snapshot["M"]),
        int(snapshot["N"]),
        str(snapshot["mod_order"]),
    )
    current = (float(snapshot.get("alpha", 0.0)), float(snapshot.get("beta", 0.0)))
    return min(results, key=lambda item: (
        float(item["ser"]),
        (float(item["alpha"]) - current[0]) ** 2 + (float(item["beta"]) - current[1]) ** 2,
        float(item["alpha"]),
        float(item["beta"]),
    ))


def _ser_loss_db(production_ser: float, reference_ser: float) -> float:
    return float(10.0 * math.log10(
        max(float(production_ser), SER_FLOOR) / max(float(reference_ser), SER_FLOOR)
    ))


def compare_sample(
    kernel: AdaptiveKernel,
    config: ExperimentConfig,
    kind: str,
    perturbation: str,
    sample_id: int,
    candidates: Sequence[Tuple[float, float]],
) -> Dict[str, Any]:
    channel = generate_channel(config, kind, sample_id)
    perturbed, noise_var, metadata = apply_perturbation(
        config, channel, kind, perturbation, config.noise_var
    )
    snapshot = build_snapshot(config, perturbed, kind, noise_var)
    started = time.perf_counter()
    production = kernel._optimize_alpha_beta_snapshot(snapshot)
    reference = exhaustive_reference(kernel, snapshot, candidates)
    elapsed = time.perf_counter() - started
    prod_alpha = float(production["recommended_alpha"])
    prod_beta = float(production["recommended_beta"])
    ref_alpha = float(reference["alpha"])
    ref_beta = float(reference["beta"])
    return {
        "sample_id": int(sample_id),
        "seed": _seed_for(config, kind, sample_id),
        "channel_kind": kind,
        "perturbation": perturbation,
        "perturbation_metadata": metadata,
        "noise_var": float(noise_var),
        "production_alpha": prod_alpha,
        "production_beta": prod_beta,
        "production_ser": float(production["predicted_ser_best"]),
        "reference_alpha": ref_alpha,
        "reference_beta": ref_beta,
        "reference_ser": float(reference["ser"]),
        "ser_loss_db": _ser_loss_db(production["predicted_ser_best"], reference["ser"]),
        "coordinate_distance": float(math.hypot(prod_alpha - ref_alpha, prod_beta - ref_beta)),
        "candidate_count": int(production["candidate_count"]),
        "production_search_seconds": float(production["search_seconds"]),
        "comparison_seconds": float(elapsed),
    }


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def aggregate_group(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    losses = np.asarray([row["ser_loss_db"] for row in samples], dtype=np.float64)
    distances = np.asarray([row["coordinate_distance"] for row in samples], dtype=np.float64)
    ref_alpha = np.asarray([row["reference_alpha"] for row in samples], dtype=np.float64)
    ref_beta = np.asarray([row["reference_beta"] for row in samples], dtype=np.float64)
    ordered = sorted(samples, key=lambda row: int(row["sample_id"]))
    movement = np.asarray([
        math.hypot(float(right["reference_alpha"]) - float(left["reference_alpha"]),
                   float(right["reference_beta"]) - float(left["reference_beta"]))
        for left, right in zip(ordered, ordered[1:])
    ], dtype=np.float64)
    worst = max(samples, key=lambda row: float(row["ser_loss_db"]))
    return {
        "sample_count": int(len(samples)),
        "ser_loss_db": {
            "mean": float(np.mean(losses)),
            "p50": _percentile(losses, 50.0),
            "p95": _percentile(losses, 95.0),
            "max": float(np.max(losses)),
        },
        "coordinate_distance": {
            "mean": float(np.mean(distances)),
            "p95": _percentile(distances, 95.0),
            "near_0_075_rate": float(np.mean(distances <= 0.075)),
            "near_0_15_rate": float(np.mean(distances <= 0.15)),
        },
        "reference_optimum": {
            "alpha_mean": float(np.mean(ref_alpha)),
            "alpha_std": float(np.std(ref_alpha)),
            "alpha_min": float(np.min(ref_alpha)),
            "alpha_max": float(np.max(ref_alpha)),
            "beta_mean": float(np.mean(ref_beta)),
            "beta_std": float(np.std(ref_beta)),
            "beta_min": float(np.min(ref_beta)),
            "beta_max": float(np.max(ref_beta)),
            "unique_pairs": int(len(set(zip(ref_alpha.tolist(), ref_beta.tolist())))),
        },
        "adjacent_reference_movement": {
            "mean": float(np.mean(movement)) if movement.size else 0.0,
            "p95": _percentile(movement, 95.0) if movement.size else 0.0,
            "max": float(np.max(movement)) if movement.size else 0.0,
        },
        "runtime_seconds": float(sum(float(row["comparison_seconds"]) for row in samples)),
        "worst_sample": {
            "sample_id": int(worst["sample_id"]),
            "ser_loss_db": float(worst["ser_loss_db"]),
            "production": [float(worst["production_alpha"]), float(worst["production_beta"])],
            "reference": [float(worst["reference_alpha"]), float(worst["reference_beta"])],
        },
    }


def _invariance_summary(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    index: Dict[Tuple[str, int, str], Dict[str, Any]] = {
        (row["channel_kind"], int(row["sample_id"]), row["perturbation"]): row
        for row in samples
    }
    comparisons = []
    for kind, sample_id, perturbation in index:
        if perturbation != "identity":
            continue
        baseline = index[(kind, sample_id, perturbation)]
        scaled = index.get((kind, sample_id, "global_scale"))
        if scaled is None:
            continue
        comparisons.append({
            "channel_kind": kind,
            "sample_id": sample_id,
            "reference_coordinate_distance": float(math.hypot(
                float(baseline["reference_alpha"]) - float(scaled["reference_alpha"]),
                float(baseline["reference_beta"]) - float(scaled["reference_beta"]),
            )),
            "reference_ser_abs_difference": abs(
                float(baseline["reference_ser"]) - float(scaled["reference_ser"])
            ),
        })
    distances = np.asarray(
        [row["reference_coordinate_distance"] for row in comparisons], dtype=np.float64
    )
    ser_differences = np.asarray(
        [row["reference_ser_abs_difference"] for row in comparisons], dtype=np.float64
    )
    ser_equivalent = ser_differences <= 1e-12
    return {
        "comparison_count": int(len(comparisons)),
        "coordinate_match_rate": float(np.mean(distances <= 1e-9)) if distances.size else float("nan"),
        "ser_equivalent_rate": float(np.mean(ser_equivalent)) if ser_equivalent.size else float("nan"),
        "max_coordinate_distance": float(np.max(distances)) if distances.size else float("nan"),
        "max_ser_abs_difference": float(np.max(ser_differences)) if ser_differences.size else float("nan"),
        "comparisons": comparisons,
    }


def _deterministic_digest(config: ExperimentConfig, samples: Sequence[Dict[str, Any]]) -> str:
    stable_samples = []
    for row in samples:
        stable_samples.append({
            key: value for key, value in row.items()
            if key not in ("production_search_seconds", "comparison_seconds")
        })
    payload = {
        "config": _config_to_json(config),
        "samples": stable_samples,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    config.validate()
    kernel = AdaptiveKernel()
    candidates = _reference_candidates(config.reference_step)
    samples: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for kind in config.channel_kinds:
        for perturbation in config.perturbations:
            for sample_id in range(config.sample_count):
                try:
                    samples.append(compare_sample(
                        kernel, config, kind, perturbation, sample_id, candidates
                    ))
                except Exception as exc:
                    failures.append({
                        "channel_kind": kind,
                        "perturbation": perturbation,
                        "sample_id": int(sample_id),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
    groups: Dict[str, Any] = {}
    for kind in config.channel_kinds:
        for perturbation in config.perturbations:
            group_samples = [
                row for row in samples
                if row["channel_kind"] == kind and row["perturbation"] == perturbation
            ]
            if group_samples:
                groups[f"{kind}/{perturbation}"] = aggregate_group(group_samples)
    worst = max(samples, key=lambda row: float(row["ser_loss_db"])) if samples else None
    return {
        "config": _config_to_json(config),
        "summary": {
            "successful_samples": int(len(samples)),
            "failed_samples": int(len(failures)),
            "reference_candidate_count": int(len(candidates)),
            "elapsed_seconds": float(time.perf_counter() - started),
            "deterministic_digest": _deterministic_digest(config, samples),
            "worst_sample": None if worst is None else {
                "channel_kind": worst["channel_kind"],
                "perturbation": worst["perturbation"],
                "sample_id": int(worst["sample_id"]),
                "ser_loss_db": float(worst["ser_loss_db"]),
            },
            "global_scale_invariance": _invariance_summary(samples),
        },
        "groups": groups,
        "samples": samples,
        "failures": failures,
    }


def _config_to_json(config: ExperimentConfig) -> Dict[str, Any]:
    data = asdict(config)
    data["channel_kinds"] = list(config.channel_kinds)
    data["perturbations"] = list(config.perturbations)
    return data


def format_summary(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "Offline alpha/beta validation",
        (f"samples={summary['successful_samples']} failures={summary['failed_samples']} "
         f"reference_candidates={summary['reference_candidate_count']} "
         f"elapsed={summary['elapsed_seconds']:.3f}s"),
    ]
    for name, group in report["groups"].items():
        loss = group["ser_loss_db"]
        opt = group["reference_optimum"]
        lines.append(
            f"{name}: loss mean/p95/max={loss['mean']:.6f}/{loss['p95']:.6f}/"
            f"{loss['max']:.6f} dB, unique_optima={opt['unique_pairs']}, "
            f"near0.15={group['coordinate_distance']['near_0_15_rate']:.1%}"
        )
    invariance = summary["global_scale_invariance"]
    lines.append(
        "global_scale invariance: "
        f"ser_equivalent={invariance['ser_equivalent_rate']:.1%}, "
        f"coordinate_match={invariance['coordinate_match_rate']:.1%}, "
        f"max_ser_diff={invariance['max_ser_abs_difference']:.3e}"
    )
    worst = summary.get("worst_sample")
    if worst:
        lines.append(
            f"worst: {worst['channel_kind']}/{worst['perturbation']} "
            f"sample={worst['sample_id']} loss={worst['ser_loss_db']:.6f} dB"
        )
    return "\n".join(lines)


def _csv_tuple(value: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--reference-step", type=float, default=0.05)
    parser.add_argument("--coarse-step", type=float, default=0.25)
    parser.add_argument("--fine-step", type=float, default=0.05)
    parser.add_argument("--mod-order", default="16QAM")
    parser.add_argument("--equalizer", default="MMSE")
    parser.add_argument("--noise-var", type=float, default=0.06309573444801933)
    parser.add_argument("--channel-kinds", default="diag,full")
    parser.add_argument("--perturbations", default=",".join(PERTURBATIONS))
    parser.add_argument("--output", type=Path, default=Path("alpha_beta_validation.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = ExperimentConfig(
            m=args.m,
            n=args.n,
            sample_count=args.samples,
            seed=args.seed,
            reference_step=args.reference_step,
            coarse_step=args.coarse_step,
            fine_step=args.fine_step,
            mod_order=args.mod_order,
            equalizer=args.equalizer,
            noise_var=args.noise_var,
            channel_kinds=_csv_tuple(args.channel_kinds),
            perturbations=_csv_tuple(args.perturbations),
        )
        report = run_experiment(config)
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(format_summary(report))
    print(f"JSON: {args.output.resolve()}")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
