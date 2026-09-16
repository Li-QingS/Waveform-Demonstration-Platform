"""Pure evidence calculations for comparable FDIDM hardware experiments.

This module intentionally has no Qt, GNU Radio, or UHD dependencies.  It is
shared by the hardware backend, adaptive validation, and the UI observer so
that all three layers use the same measurement definitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [_json_safe(complex(item)) for item in value.reshape(-1)]
        return [_json_safe(item) for item in value.reshape(-1)]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class _Serializable:
    def as_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class TxPowerMetrics(_Serializable):
    cycle_rms: float = float("nan")
    data_rms: float = float("nan")
    peak: float = float("nan")
    papr_db: float = float("nan")
    target_rms: float = float("nan")
    safe_rms: float = float("nan")
    backoff_db: float = float("nan")
    peak_limited: bool = False
    contract_id: str = ""


@dataclass(frozen=True)
class TxPowerContract(_Serializable):
    baseline_alpha: float
    baseline_beta: float
    candidate_alpha: float
    candidate_beta: float
    requested_rms: float
    locked_rms: float
    peak_limit: float
    tolerance_db: float = 0.10
    contract_id: str = ""
    comparable: bool = True
    reason: str = ""


@dataclass(frozen=True)
class KnownSymbolMetrics(_Serializable):
    reference_symbols: int = 0
    ser_errors: int = 0
    ser_symbols: int = 0
    ser: float = float("nan")
    data_aided_evm_percent: float = float("nan")
    decision_directed_evm_percent: float = float("nan")
    residual_gain_abs: float = float("nan")
    residual_phase_deg: float = float("nan")


@dataclass(frozen=True)
class PilotFitMetrics(_Serializable):
    fit_nmse: float = float("nan")
    residual_sinr_db: float = float("nan")
    residual_power: float = float("nan")
    fitted_power: float = float("nan")


@dataclass
class EvidenceWindow(_Serializable):
    valid_frames: int = 0
    ser_errors: int = 0
    ser_symbols: int = 0
    data_aided_evm_values: list[float] = field(default_factory=list)
    crc_ok_frames: int = 0
    raw_ber_values: list[float] = field(default_factory=list)
    fec_ber_values: list[float] = field(default_factory=list)
    power_rms_values: list[float] = field(default_factory=list)
    pilot_fit_nmse_values: list[float] = field(default_factory=list)
    pilot_fitted_power_values: list[float] = field(default_factory=list)
    contract_id: str = ""
    context_key: Any = None
    dropped_overflow: int = 0
    dropped_sync: int = 0
    dropped_context: int = 0
    dropped_power: int = 0

    def add_sample(
        self,
        *,
        ser_errors: int,
        ser_symbols: int,
        data_aided_evm_percent: Any = None,
        crc_ok: bool = False,
        raw_ber: Any = None,
        fec_ber: Any = None,
        power_rms: Any = None,
        pilot_fit_nmse: Any = None,
        pilot_fitted_power: Any = None,
    ) -> None:
        self.valid_frames += 1
        self.ser_errors += max(0, int(ser_errors))
        self.ser_symbols += max(0, int(ser_symbols))
        self.crc_ok_frames += int(bool(crc_ok))
        for values, value in (
            (self.data_aided_evm_values, data_aided_evm_percent),
            (self.raw_ber_values, raw_ber),
            (self.fec_ber_values, fec_ber),
            (self.power_rms_values, power_rms),
            (self.pilot_fit_nmse_values, pilot_fit_nmse),
            (self.pilot_fitted_power_values, pilot_fitted_power),
        ):
            finite = _finite_or_none(value)
            if finite is not None:
                values.append(finite)

    def drop(self, reason: str) -> None:
        field_name = {
            "overflow": "dropped_overflow",
            "sync": "dropped_sync",
            "context": "dropped_context",
            "power": "dropped_power",
        }.get(str(reason).lower())
        if field_name is None:
            raise ValueError(f"unknown evidence drop reason: {reason}")
        setattr(self, field_name, int(getattr(self, field_name)) + 1)

    @property
    def ser(self) -> float:
        if self.ser_symbols <= 0:
            return float("nan")
        return float(self.ser_errors) / float(self.ser_symbols)

    @property
    def wilson_interval(self) -> Tuple[float, float]:
        return wilson_interval(self.ser_errors, self.ser_symbols)

    @property
    def data_aided_evm_mean(self) -> float:
        return _mean(self.data_aided_evm_values)

    @property
    def data_aided_evm_sem(self) -> float:
        values = _finite_array(self.data_aided_evm_values)
        if values.size < 2:
            return float("nan")
        return float(np.std(values, ddof=1) / math.sqrt(values.size))

    @property
    def crc_success_ratio(self) -> float:
        if self.valid_frames <= 0:
            return float("nan")
        return float(self.crc_ok_frames) / float(self.valid_frames)

    def as_dict(self) -> Dict[str, Any]:
        result = super().as_dict()
        low, high = self.wilson_interval
        result.update(
            {
                "ser": _json_safe(self.ser),
                "wilson_low": _json_safe(low),
                "wilson_high": _json_safe(high),
                "data_aided_evm_mean": _json_safe(self.data_aided_evm_mean),
                "data_aided_evm_sem": _json_safe(self.data_aided_evm_sem),
                "crc_success_ratio": _json_safe(self.crc_success_ratio),
            }
        )
        return result


@dataclass(frozen=True)
class ValidationDecision(_Serializable):
    outcome: str
    measured_ser_gain_db: float = float("nan")
    ser_gain_lower_bound_db: float = float("nan")
    baseline_ser_interval: Tuple[float, float] = (float("nan"), float("nan"))
    candidate_ser_interval: Tuple[float, float] = (float("nan"), float("nan"))
    evm_delta_pp: float = float("nan")
    reason: str = ""


def _finite_array(values: Iterable[Any]) -> np.ndarray:
    try:
        array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.empty(0, dtype=np.float64)
    return array[np.isfinite(array)]


def _mean(values: Iterable[Any]) -> float:
    array = _finite_array(values)
    return float(np.mean(array)) if array.size else float("nan")


def signal_rms(samples: Sequence[complex] | np.ndarray) -> float:
    values = np.asarray(samples).reshape(-1)
    if values.size == 0:
        return 0.0
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    if not np.all(finite):
        return float("nan")
    return float(np.sqrt(np.mean(np.abs(values) ** 2)))


def _power_metrics(
    cycle: np.ndarray,
    data_samples: np.ndarray,
    *,
    target_rms: float,
    peak_limit: float,
    safe_rms: float,
    peak_limited: bool,
    contract_id: str,
) -> TxPowerMetrics:
    cycle_rms = signal_rms(cycle)
    data_rms = signal_rms(data_samples) if data_samples.size else float("nan")
    peak = float(np.max(np.abs(cycle))) if cycle.size else 0.0
    if cycle_rms > 0.0 and peak > 0.0:
        papr_db = 20.0 * math.log10(peak / cycle_rms)
    else:
        papr_db = float("nan")
    if target_rms > 0.0 and cycle_rms > 0.0:
        backoff_db = 20.0 * math.log10(cycle_rms / target_rms)
    else:
        backoff_db = float("nan")
    return TxPowerMetrics(
        cycle_rms=cycle_rms,
        data_rms=data_rms,
        peak=peak,
        papr_db=papr_db,
        target_rms=float(target_rms),
        safe_rms=float(safe_rms),
        backoff_db=backoff_db,
        peak_limited=bool(peak_limited),
        contract_id=str(contract_id or ""),
    )


def analyze_unscaled_power(
    cycle: Sequence[complex] | np.ndarray,
    data_samples: Sequence[complex] | np.ndarray,
    *,
    target_rms: float,
    peak_limit: float,
) -> TxPowerMetrics:
    cycle_arr = np.asarray(cycle, dtype=np.complex128).reshape(-1)
    data_arr = np.asarray(data_samples, dtype=np.complex128).reshape(-1)
    base_rms = signal_rms(cycle_arr)
    peak = float(np.max(np.abs(cycle_arr))) if cycle_arr.size else 0.0
    safe_rms = base_rms * float(peak_limit) / peak if base_rms > 0.0 and peak > 0.0 else 0.0
    return _power_metrics(
        cycle_arr,
        data_arr,
        target_rms=float(target_rms),
        peak_limit=float(peak_limit),
        safe_rms=safe_rms,
        peak_limited=bool(peak > float(peak_limit)),
        contract_id="",
    )


def scale_waveform_to_rms(
    cycle: Sequence[complex] | np.ndarray,
    data_samples: Sequence[complex] | np.ndarray,
    *,
    target_rms: float,
    peak_limit: float,
    contract_id: str = "",
) -> Tuple[np.ndarray, TxPowerMetrics]:
    cycle_arr = np.asarray(cycle, dtype=np.complex128).reshape(-1)
    data_arr = np.asarray(data_samples, dtype=np.complex128).reshape(-1)
    target = max(0.0, float(target_rms))
    limit = max(0.0, float(peak_limit))
    base_rms = signal_rms(cycle_arr)
    peak = float(np.max(np.abs(cycle_arr))) if cycle_arr.size else 0.0
    safe_rms = base_rms * limit / peak if base_rms > 0.0 and peak > 0.0 else 0.0
    if cycle_arr.size == 0 or not math.isfinite(base_rms) or base_rms <= 0.0:
        scaled = np.zeros_like(cycle_arr)
        data_scaled = np.zeros_like(data_arr)
        limited = False
    else:
        requested_scale = target / base_rms
        peak_scale = limit / peak if peak > 0.0 else requested_scale
        scale = min(requested_scale, peak_scale)
        limited = bool(peak > 0.0 and peak_scale + 1e-15 < requested_scale)
        scaled = cycle_arr * scale
        data_scaled = data_arr * scale
    metrics = _power_metrics(
        scaled,
        data_scaled,
        target_rms=target,
        peak_limit=limit,
        safe_rms=safe_rms,
        peak_limited=limited,
        contract_id=contract_id,
    )
    return scaled, metrics


def create_power_contract(
    baseline_metrics: TxPowerMetrics,
    candidate_metrics: TxPowerMetrics,
    *,
    baseline_alpha: float,
    baseline_beta: float,
    candidate_alpha: float,
    candidate_beta: float,
    requested_rms: float,
    peak_limit: float,
    safety_factor: float = 0.98,
    tolerance_db: float = 0.10,
) -> TxPowerContract:
    safe_values = np.asarray(
        [requested_rms, baseline_metrics.safe_rms, candidate_metrics.safe_rms],
        dtype=np.float64,
    )
    valid = bool(
        np.all(np.isfinite(safe_values))
        and np.all(safe_values > 0.0)
        and math.isfinite(float(peak_limit))
        and float(peak_limit) > 0.0
        and 0.0 < float(safety_factor) <= 1.0
    )
    locked_rms = float(np.min(safe_values) * float(safety_factor)) if valid else 0.0
    payload = {
        "baseline": [round(float(baseline_alpha), 9), round(float(baseline_beta), 9)],
        "candidate": [round(float(candidate_alpha), 9), round(float(candidate_beta), 9)],
        "requested_rms": round(float(requested_rms), 12),
        "locked_rms": round(locked_rms, 12),
        "peak_limit": round(float(peak_limit), 12),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    contract_id = f"txp-{hashlib.sha256(encoded).hexdigest()[:12]}"
    return TxPowerContract(
        baseline_alpha=float(baseline_alpha),
        baseline_beta=float(baseline_beta),
        candidate_alpha=float(candidate_alpha),
        candidate_beta=float(candidate_beta),
        requested_rms=float(requested_rms),
        locked_rms=locked_rms,
        peak_limit=float(peak_limit),
        tolerance_db=float(tolerance_db),
        contract_id=contract_id,
        comparable=valid,
        reason="" if valid else "invalid waveform power metrics",
    )


def power_difference_db(first_rms: Any, second_rms: Any) -> float:
    first = _finite_or_none(first_rms)
    second = _finite_or_none(second_rms)
    if first is None or second is None or first <= 0.0 or second <= 0.0:
        return float("nan")
    return abs(20.0 * math.log10(second / first))


def assess_power_comparability(
    baseline_contract_id: Any,
    baseline_rms: Any,
    candidate_contract_id: Any,
    candidate_rms: Any,
    *,
    tolerance_db: float = 0.10,
) -> Tuple[bool, str, float]:
    before_id = str(baseline_contract_id or "")
    after_id = str(candidate_contract_id or "")
    if not before_id or not after_id:
        return False, "missing power contract", float("nan")
    if before_id != after_id:
        return False, "power contract changed", float("nan")
    difference = power_difference_db(baseline_rms, candidate_rms)
    if not math.isfinite(difference):
        return False, "missing actual cycle RMS", difference
    if difference > float(tolerance_db) + 1e-12:
        return False, f"actual cycle RMS differs by {difference:.3f} dB", difference
    return True, "", difference


def _nearest_indices(symbols: np.ndarray, constellation: np.ndarray) -> np.ndarray:
    distances = np.abs(symbols[:, None] - constellation[None, :]) ** 2
    return np.argmin(distances, axis=1)


def measure_known_symbols(
    rx_symbols: Sequence[complex] | np.ndarray,
    reference_symbols: Sequence[complex] | np.ndarray,
    ideal_constellation: Sequence[complex] | np.ndarray,
) -> KnownSymbolMetrics:
    rx = np.asarray(rx_symbols, dtype=np.complex128).reshape(-1)
    reference = np.asarray(reference_symbols, dtype=np.complex128).reshape(-1)
    constellation = np.asarray(ideal_constellation, dtype=np.complex128).reshape(-1)
    count = min(rx.size, reference.size)
    if count <= 0 or constellation.size <= 0:
        return KnownSymbolMetrics()
    rx = rx[:count]
    reference = reference[:count]
    finite = (
        np.isfinite(rx.real)
        & np.isfinite(rx.imag)
        & np.isfinite(reference.real)
        & np.isfinite(reference.imag)
    )
    rx = rx[finite]
    reference = reference[finite]
    count = int(rx.size)
    if count <= 0:
        return KnownSymbolMetrics()
    denominator = float(np.vdot(reference, reference).real)
    gain = np.vdot(reference, rx) / denominator if denominator > 1e-20 else 1.0 + 0.0j
    normalized = rx / gain if abs(gain) > 1e-20 else rx.copy()
    ref_power = float(np.mean(np.abs(reference) ** 2))
    data_evm = (
        100.0 * math.sqrt(float(np.mean(np.abs(normalized - reference) ** 2)) / ref_power)
        if ref_power > 1e-20
        else float("nan")
    )
    rx_indices = _nearest_indices(normalized, constellation)
    ref_indices = _nearest_indices(reference, constellation)
    decisions = constellation[rx_indices]
    decision_power = float(np.mean(np.abs(decisions) ** 2))
    decision_evm = (
        100.0 * math.sqrt(float(np.mean(np.abs(normalized - decisions) ** 2)) / decision_power)
        if decision_power > 1e-20
        else float("nan")
    )
    errors = int(np.count_nonzero(rx_indices != ref_indices))
    return KnownSymbolMetrics(
        reference_symbols=count,
        ser_errors=errors,
        ser_symbols=count,
        ser=float(errors) / float(count),
        data_aided_evm_percent=data_evm,
        decision_directed_evm_percent=decision_evm,
        residual_gain_abs=float(abs(gain)),
        residual_phase_deg=float(np.degrees(np.angle(gain))),
    )


def measure_pilot_fit(
    observed: Sequence[complex] | np.ndarray,
    fitted: Sequence[complex] | np.ndarray,
) -> PilotFitMetrics:
    raw = np.asarray(observed, dtype=np.complex128).reshape(-1)
    model = np.asarray(fitted, dtype=np.complex128).reshape(-1)
    count = min(raw.size, model.size)
    if count <= 0:
        return PilotFitMetrics()
    raw = raw[:count]
    model = model[:count]
    finite = (
        np.isfinite(raw.real)
        & np.isfinite(raw.imag)
        & np.isfinite(model.real)
        & np.isfinite(model.imag)
    )
    raw = raw[finite]
    model = model[finite]
    if raw.size == 0:
        return PilotFitMetrics()
    residual_power = float(np.mean(np.abs(raw - model) ** 2))
    fitted_power = float(np.mean(np.abs(model) ** 2))
    observed_power = float(np.mean(np.abs(raw) ** 2))
    fit_nmse = residual_power / observed_power if observed_power > 1e-20 else float("nan")
    if residual_power <= 1e-20 and fitted_power > 0.0:
        residual_sinr_db = float("inf")
    elif residual_power > 0.0 and fitted_power > 0.0:
        residual_sinr_db = 10.0 * math.log10(fitted_power / residual_power)
    else:
        residual_sinr_db = float("nan")
    return PilotFitMetrics(
        fit_nmse=fit_nmse,
        residual_sinr_db=residual_sinr_db,
        residual_power=residual_power,
        fitted_power=fitted_power,
    )


def frame_structure_metrics(
    *,
    data_samples: int,
    pilot_samples: int,
    sync_samples: int = 0,
    guard_samples: int = 0,
    other_samples: int = 0,
) -> Dict[str, Any]:
    data = max(0, int(data_samples))
    pilot = max(0, int(pilot_samples))
    sync = max(0, int(sync_samples))
    guard = max(0, int(guard_samples))
    other = max(0, int(other_samples))
    total = data + pilot + sync + guard + other
    return {
        "data_samples": data,
        "pilot_samples": pilot,
        "sync_samples": sync,
        "guard_samples": guard,
        "other_samples": other,
        "total_samples": total,
        "training_data_ratio": float(pilot) / float(data) if data else None,
        "useful_data_ratio": float(data) / float(total) if total else None,
    }


def wilson_interval(errors: int, symbols: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    n = max(0, int(symbols))
    if n <= 0:
        return float("nan"), float("nan")
    k = min(n, max(0, int(errors)))
    p = float(k) / float(n)
    z2 = float(z) ** 2
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    radius = float(z) * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def measured_ser_gain_db(baseline_ser: Any, candidate_ser: Any) -> float:
    baseline = _finite_or_none(baseline_ser)
    candidate = _finite_or_none(candidate_ser)
    # Zero observed errors does not identify a finite rate ratio.  Using an
    # arbitrary 1e-15 floor made clean candidates appear to gain >100 dB.
    if baseline is None or candidate is None or baseline <= 0.0 or candidate <= 0.0:
        return float("nan")
    return 10.0 * math.log10(baseline / candidate)


def classify_validation(
    baseline: EvidenceWindow,
    candidate: EvidenceWindow,
    *,
    comparable: bool = True,
    comparability_reason: str = "",
    min_frames: int = 24,
    max_frames: int = 64,
    min_errors_for_improvement: int = 100,
    min_ser_gain_db: float = 0.25,
    min_evm_improvement_pp: float = 1.0,
    evm_regression_tolerance_pp: float = 2.0,
    crc_regression_tolerance: float = 0.05,
) -> ValidationDecision:
    baseline_interval = baseline.wilson_interval
    candidate_interval = candidate.wilson_interval
    gain_db = measured_ser_gain_db(baseline.ser, candidate.ser)
    b_low, b_high = baseline_interval
    c_low, c_high = candidate_interval
    gain_lower_bound_db = (
        10.0 * math.log10(b_low / c_high)
        if math.isfinite(b_low) and math.isfinite(c_high) and b_low > 0.0 and c_high > 0.0
        else float("nan")
    )
    baseline_evm = baseline.data_aided_evm_mean
    candidate_evm = candidate.data_aided_evm_mean
    evm_delta = candidate_evm - baseline_evm if math.isfinite(baseline_evm) and math.isfinite(candidate_evm) else float("nan")

    def decision(outcome: str, reason: str) -> ValidationDecision:
        return ValidationDecision(
            outcome=outcome,
            measured_ser_gain_db=gain_db,
            ser_gain_lower_bound_db=gain_lower_bound_db,
            baseline_ser_interval=baseline_interval,
            candidate_ser_interval=candidate_interval,
            evm_delta_pp=evm_delta,
            reason=reason,
        )

    if not comparable:
        return decision("inconclusive", comparability_reason or "power or context is not comparable")
    if baseline.contract_id and candidate.contract_id and baseline.contract_id != candidate.contract_id:
        return decision("inconclusive", "power contract changed between windows")
    if baseline.context_key is not None and candidate.context_key is not None and baseline.context_key != candidate.context_key:
        return decision("inconclusive", "channel context changed between windows")
    if baseline.valid_frames < int(min_frames) or candidate.valid_frames < int(min_frames):
        return decision("inconclusive", "insufficient valid frames")
    if baseline.ser_symbols <= 0 or candidate.ser_symbols <= 0:
        return decision("inconclusive", "missing exact SER counts")

    # The pilot waveform is identical on both alpha/beta sides.  A large
    # change in its fit or received power indicates that sequential A/B
    # windows saw different RF conditions; data EVM/SER cannot be attributed
    # to the waveform in that case.  Older/offline evidence without pilot
    # diagnostics remains usable.
    min_pilot_samples = max(4, int(min_frames) // 2)
    if (len(baseline.pilot_fit_nmse_values) >= min_pilot_samples
            and len(candidate.pilot_fit_nmse_values) >= min_pilot_samples):
        b_pilot = float(np.median(_finite_array(baseline.pilot_fit_nmse_values)))
        c_pilot = float(np.median(_finite_array(candidate.pilot_fit_nmse_values)))
        if (abs(c_pilot - b_pilot) > 0.02
                and max(c_pilot, b_pilot) > 1.8 * max(min(c_pilot, b_pilot), 0.005)):
            return decision("inconclusive", "pilot fit changed between A/B windows; RF channel is not stationary")
    if (len(baseline.pilot_fitted_power_values) >= min_pilot_samples
            and len(candidate.pilot_fitted_power_values) >= min_pilot_samples):
        b_power = float(np.median(_finite_array(baseline.pilot_fitted_power_values)))
        c_power = float(np.median(_finite_array(candidate.pilot_fitted_power_values)))
        if b_power > 0.0 and c_power > 0.0:
            pilot_power_shift_db = abs(10.0 * math.log10(c_power / b_power))
            if pilot_power_shift_db > 3.0:
                return decision("inconclusive", f"pilot received power shifted by {pilot_power_shift_db:.2f} dB between A/B windows")

    baseline_crc = baseline.crc_success_ratio
    candidate_crc = candidate.crc_success_ratio
    if math.isfinite(evm_delta) and evm_delta > float(evm_regression_tolerance_pp):
        return decision("regressed", f"data-aided EVM regressed by {evm_delta:.3f} percentage points")
    if (
        math.isfinite(baseline_crc)
        and math.isfinite(candidate_crc)
        and candidate_crc < baseline_crc - float(crc_regression_tolerance)
    ):
        return decision("regressed", "CRC success ratio regressed")
    if math.isfinite(b_high) and math.isfinite(c_low) and b_high < c_low:
        return decision("regressed", "candidate SER is significantly worse")

    # A genuinely better candidate may have zero errors. Requiring an error
    # quota on *both* sides made the best possible result impossible to accept.
    # The baseline still needs enough events for a SER-gain claim, while the
    # Wilson interval provides the candidate-side uncertainty bound.
    enough_errors = baseline.ser_errors >= int(min_errors_for_improvement)
    intervals_show_improvement = math.isfinite(c_high) and math.isfinite(b_low) and c_high < b_low
    evm_guard_ok = not math.isfinite(evm_delta) or evm_delta <= float(evm_regression_tolerance_pp)
    supported_gain_db = gain_db if math.isfinite(gain_db) else gain_lower_bound_db
    if enough_errors and intervals_show_improvement and math.isfinite(supported_gain_db) and supported_gain_db >= float(min_ser_gain_db) and evm_guard_ok:
        return decision("improved", "candidate SER improvement is statistically supported")

    # On an already healthy link both windows can be error-free, so SER cannot
    # distinguish candidates. In that regime accept a lower-EVM candidate only
    # when the 95% confidence bands are separated, CRC is non-regressing, and
    # SER has not shown a statistically significant regression above.
    baseline_sem = baseline.data_aided_evm_sem
    candidate_sem = candidate.data_aided_evm_sem
    evm_improvement_supported = bool(
        math.isfinite(evm_delta)
        and evm_delta <= -abs(float(min_evm_improvement_pp))
        and math.isfinite(baseline_sem)
        and math.isfinite(candidate_sem)
        and candidate_evm + 1.959963984540054 * candidate_sem
            < baseline_evm - 1.959963984540054 * baseline_sem
    )
    crc_guard_ok = not (
        math.isfinite(baseline_crc)
        and math.isfinite(candidate_crc)
        and candidate_crc < baseline_crc - float(crc_regression_tolerance)
    )
    if evm_improvement_supported and crc_guard_ok:
        return decision("improved", "candidate EVM improvement is statistically supported without SER regression")

    reached_limit = baseline.valid_frames >= int(max_frames) and candidate.valid_frames >= int(max_frames)
    if not enough_errors:
        reason = "insufficient SER errors in baseline and no statistically supported EVM improvement"
    elif not intervals_show_improvement:
        reason = "SER confidence intervals overlap"
    elif not math.isfinite(supported_gain_db) or supported_gain_db < float(min_ser_gain_db):
        reason = "measured SER gain is below the configured threshold"
    elif reached_limit:
        reason = "maximum validation window reached without sufficient evidence"
    else:
        reason = "validation evidence is not yet conclusive"
    return decision("inconclusive", reason)


__all__ = [
    "EvidenceWindow",
    "KnownSymbolMetrics",
    "PilotFitMetrics",
    "TxPowerContract",
    "TxPowerMetrics",
    "ValidationDecision",
    "analyze_unscaled_power",
    "assess_power_comparability",
    "classify_validation",
    "create_power_contract",
    "frame_structure_metrics",
    "measure_known_symbols",
    "measure_pilot_fit",
    "measured_ser_gain_db",
    "power_difference_db",
    "scale_waveform_to_rms",
    "signal_rms",
    "wilson_interval",
]
