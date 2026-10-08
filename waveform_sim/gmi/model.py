"""Immutable data contracts for receiver-dependent GMI adaptation."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = 1
_TOKEN_PERIOD = 2.0


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(prefix: str, value: Any, length: int = 16) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:length]}"


def _finite_float(name: str, value: Any, *, minimum: Optional[float] = None) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _readonly_complex_matrix(name: str, value: Any) -> np.ndarray:
    matrix = np.array(value, dtype=np.complex128, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] <= 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not np.all(np.isfinite(matrix.real)) or not np.all(np.isfinite(matrix.imag)):
        raise ValueError(f"{name} must contain only finite values")
    matrix.setflags(write=False)
    return matrix


def _canonical_fractional(value: float) -> float:
    result = float(value) % _TOKEN_PERIOD
    if math.isclose(result, _TOKEN_PERIOD, abs_tol=1e-12) or math.isclose(result, 0.0, abs_tol=1e-12):
        return 0.0
    return round(result, 12)


@dataclass(frozen=True)
class ExperimentResources:
    """Common charged resources shared by every library size ``M <= cap``."""

    signal_dimension: int
    guard_samples: int = 0
    training_samples: int = 0
    control_samples: Optional[int] = None
    catalogue_cap: int = 16
    control_bits: Optional[int] = None
    control_bits_per_sample: int = 1
    sample_rate_hz: float = 1.0
    bandwidth_hz: float = 1.0
    target_rms: float = 0.25
    peak_limit: float = 0.90
    pilot_rule: str = "common_probe_v1"
    receiver_rule: str = "receiver_projection_v1"
    grid_rows: Optional[int] = None
    grid_cols: Optional[int] = None

    def __post_init__(self) -> None:
        integer_fields = ("signal_dimension", "guard_samples", "training_samples", "catalogue_cap", "control_bits_per_sample")
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.signal_dimension <= 0:
            raise ValueError("signal_dimension must be positive")
        if self.guard_samples < 0 or self.training_samples < 0:
            raise ValueError("guard_samples and training_samples must be non-negative")
        if self.catalogue_cap <= 0:
            raise ValueError("catalogue_cap must be positive")
        if self.control_bits_per_sample <= 0:
            raise ValueError("control_bits_per_sample must be positive")
        required_bits = int(math.ceil(math.log2(self.catalogue_cap))) if self.catalogue_cap > 1 else 0
        control_bits = required_bits if self.control_bits is None else int(self.control_bits)
        if control_bits < required_bits:
            raise ValueError("control_bits cannot address catalogue_cap")
        required_samples = int(math.ceil(control_bits / self.control_bits_per_sample))
        control_samples = required_samples if self.control_samples is None else int(self.control_samples)
        if control_samples < required_samples or control_samples < 0:
            raise ValueError("control_samples is smaller than the common control field")
        object.__setattr__(self, "control_bits", control_bits)
        object.__setattr__(self, "control_samples", control_samples)
        _finite_float("sample_rate_hz", self.sample_rate_hz, minimum=np.finfo(float).tiny)
        _finite_float("bandwidth_hz", self.bandwidth_hz, minimum=np.finfo(float).tiny)
        _finite_float("target_rms", self.target_rms, minimum=np.finfo(float).tiny)
        _finite_float("peak_limit", self.peak_limit, minimum=np.finfo(float).tiny)
        if self.target_rms > self.peak_limit:
            raise ValueError("target_rms cannot exceed peak_limit")
        if not str(self.pilot_rule).strip() or not str(self.receiver_rule).strip():
            raise ValueError("pilot_rule and receiver_rule are required")
        if (self.grid_rows is None) != (self.grid_cols is None):
            raise ValueError("grid_rows and grid_cols must be supplied together")
        if self.grid_rows is not None:
            if int(self.grid_rows) <= 0 or int(self.grid_cols) <= 0:
                raise ValueError("grid dimensions must be positive")
            if int(self.grid_rows) * int(self.grid_cols) != self.signal_dimension:
                raise ValueError("grid_rows * grid_cols must equal signal_dimension")

    @property
    def charged_samples(self) -> int:
        return int(self.signal_dimension + self.guard_samples + self.training_samples + int(self.control_samples or 0))

    @property
    def grid_shape(self) -> Tuple[int, int]:
        if self.grid_rows is not None and self.grid_cols is not None:
            return int(self.grid_rows), int(self.grid_cols)
        root = int(math.sqrt(self.signal_dimension))
        for rows in range(max(1, root), 0, -1):
            if self.signal_dimension % rows == 0:
                return rows, self.signal_dimension // rows
        return self.signal_dimension, 1

    def normalized_dict(self) -> dict[str, Any]:
        return {
            "bandwidth_hz": round(float(self.bandwidth_hz), 9),
            "catalogue_cap": int(self.catalogue_cap),
            "control_bits": int(self.control_bits or 0),
            "control_bits_per_sample": int(self.control_bits_per_sample),
            "control_samples": int(self.control_samples or 0),
            "grid_cols": self.grid_shape[1],
            "grid_rows": self.grid_shape[0],
            "guard_samples": int(self.guard_samples),
            "peak_limit": round(float(self.peak_limit), 12),
            "pilot_rule": str(self.pilot_rule),
            "receiver_rule": str(self.receiver_rule),
            "sample_rate_hz": round(float(self.sample_rate_hz), 9),
            "signal_dimension": int(self.signal_dimension),
            "target_rms": round(float(self.target_rms), 12),
            "training_samples": int(self.training_samples),
        }

    @property
    def resource_id(self) -> str:
        return _digest("res", self.normalized_dict())


@dataclass(frozen=True)
class ReceiverSpec:
    mode: str = "diagonal"
    band_k: Optional[int] = 0
    metric_covariance_mode: str = "receive_noise"
    exponent_interval: Tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"diagonal", "cyclic_band", "full_matched"}:
            raise ValueError(f"unsupported receiver mode: {self.mode}")
        object.__setattr__(self, "mode", mode)
        if mode == "diagonal" and self.band_k not in (0, None):
            raise ValueError("diagonal receiver requires band_k=0")
        if mode == "cyclic_band" and (self.band_k is None or int(self.band_k) < 0):
            raise ValueError("cyclic_band receiver requires a non-negative band_k")
        if not str(self.metric_covariance_mode).strip():
            raise ValueError("metric_covariance_mode is required")
        lo, hi = (float(self.exponent_interval[0]), float(self.exponent_interval[1]))
        if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 <= lo < hi):
            raise ValueError("exponent_interval must be finite, ordered and non-negative")
        object.__setattr__(self, "exponent_interval", (lo, hi))

    def effective_band_k(self, dimension: int) -> int:
        self.validate(dimension)
        if self.mode == "full_matched":
            return int(dimension) // 2
        return 0 if self.mode == "diagonal" else int(self.band_k or 0)

    def validate(self, dimension: int) -> None:
        dimension = int(dimension)
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.mode == "cyclic_band" and int(self.band_k or 0) > dimension // 2:
            raise ValueError("band_k exceeds cyclic half-bandwidth")

    def normalized_dict(self) -> dict[str, Any]:
        return {
            "band_k": None if self.mode == "full_matched" else int(self.band_k or 0),
            "exponent_interval": [float(self.exponent_interval[0]), float(self.exponent_interval[1])],
            "metric_covariance_mode": str(self.metric_covariance_mode),
            "mode": self.mode,
        }

    @property
    def receiver_id(self) -> str:
        return _digest("rx", self.normalized_dict())


@dataclass(frozen=True)
class GMIExperiment:
    B: np.ndarray
    C: np.ndarray
    Sigma: np.ndarray
    symbol_power: float
    charged_samples: int
    receiver: ReceiverSpec
    resource_id: str
    token_id: str = ""

    def __post_init__(self) -> None:
        B = _readonly_complex_matrix("B", self.B)
        C = _readonly_complex_matrix("C", self.C)
        Sigma = _readonly_complex_matrix("Sigma", self.Sigma)
        if B.shape != C.shape or B.shape != Sigma.shape:
            raise ValueError("B, C and Sigma must have the same shape")
        if not np.allclose(Sigma, Sigma.conj().T, atol=1e-10, rtol=1e-10):
            raise ValueError("Sigma must be Hermitian")
        eig = np.linalg.eigvalsh(Sigma)
        if float(np.min(eig)) <= 0.0:
            raise ValueError("Sigma must be positive definite")
        power = _finite_float("symbol_power", self.symbol_power, minimum=np.finfo(float).tiny)
        if int(self.charged_samples) <= 0:
            raise ValueError("charged_samples must be positive")
        self.receiver.validate(B.shape[0])
        if not str(self.resource_id):
            raise ValueError("resource_id is required")
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "Sigma", Sigma)
        object.__setattr__(self, "symbol_power", power)


@dataclass(frozen=True)
class GMIResult:
    valid: bool
    reason: str
    gmi_nats_per_sample: float
    gmi_bits_per_sample: float
    s_star: Optional[float]
    receiver_mode: str
    token_id: str
    resource_id: str
    covariance_min_eigenvalue: Optional[float] = None
    covariance_condition: Optional[float] = None
    solve_path: str = ""

    def __post_init__(self) -> None:
        if self.valid:
            for name in ("gmi_nats_per_sample", "gmi_bits_per_sample", "s_star"):
                value = getattr(self, name)
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(f"valid GMI result requires finite {name}")
            if self.gmi_nats_per_sample < -1e-12:
                raise ValueError("GMI cannot be negative")
        elif not str(self.reason):
            raise ValueError("invalid GMI result requires a reason")

    @classmethod
    def invalid(cls, reason: str, *, receiver_mode: str, token_id: str = "", resource_id: str = "") -> "GMIResult":
        return cls(False, str(reason), 0.0, 0.0, None, receiver_mode, token_id, resource_id, solve_path="invalid")


@dataclass(frozen=True)
class ChannelState:
    context_id: str
    epoch: int
    timestamp: float
    h_tf: Optional[np.ndarray] = None
    h_time: Optional[np.ndarray] = None
    noise_covariance: Optional[np.ndarray] = None
    noise_variance: Optional[float] = None
    symbol_power: float = 1.0
    sync_metric: float = 1.0
    effective_cfo_hz: float = 0.0
    pilot_nmse: float = 0.0
    pilot_power: float = 1.0
    continuity: bool = True
    overflow: bool = False
    state_uncertainty_nats: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.context_id):
            raise ValueError("context_id is required")
        if int(self.epoch) < 0:
            raise ValueError("epoch must be non-negative")
        if (self.h_tf is None) == (self.h_time is None):
            raise ValueError("exactly one of h_tf or h_time is required")
        channel_name = "h_tf" if self.h_tf is not None else "h_time"
        channel = _readonly_complex_matrix(channel_name, self.h_tf if self.h_tf is not None else self.h_time)
        object.__setattr__(self, channel_name, channel)
        if self.noise_covariance is not None:
            covariance = _readonly_complex_matrix("noise_covariance", self.noise_covariance)
            if covariance.shape != channel.shape:
                raise ValueError("noise_covariance shape does not match channel")
            object.__setattr__(self, "noise_covariance", covariance)
        if self.noise_variance is None and self.noise_covariance is None:
            raise ValueError("noise_variance or noise_covariance is required")
        if self.noise_variance is not None:
            _finite_float("noise_variance", self.noise_variance, minimum=np.finfo(float).tiny)
        _finite_float("symbol_power", self.symbol_power, minimum=np.finfo(float).tiny)
        _finite_float("state_uncertainty_nats", self.state_uncertainty_nats, minimum=0.0)

    @property
    def dimension(self) -> int:
        channel = self.h_tf if self.h_tf is not None else self.h_time
        return int(channel.shape[0])

    @property
    def basis(self) -> str:
        return "tf" if self.h_tf is not None else "time"

    @property
    def channel_matrix(self) -> np.ndarray:
        return self.h_tf if self.h_tf is not None else self.h_time

    def validity_reason(self) -> str:
        if self.overflow:
            return "overflow"
        if not self.continuity:
            return "probe_gap"
        diagnostics = (self.timestamp, self.sync_metric, self.effective_cfo_hz, self.pilot_nmse, self.pilot_power)
        if not all(math.isfinite(float(value)) for value in diagnostics):
            return "non_finite_acquisition_state"
        if self.sync_metric <= 0.0 or self.pilot_power <= 0.0:
            return "untrusted_acquisition_state"
        if self.noise_covariance is not None:
            if not np.allclose(self.noise_covariance, self.noise_covariance.conj().T, atol=1e-10, rtol=1e-10):
                return "noise_covariance_not_hermitian"
            if float(np.min(np.linalg.eigvalsh(self.noise_covariance))) <= 0.0:
                return "noise_covariance_not_positive_definite"
        return ""


@dataclass(frozen=True)
class WaveformToken:
    token_id: str
    alpha: float
    beta: float
    receiver_id: str
    resource_id: str
    channel_estimator: str
    equalizer: str
    transform_version: str
    implementation_mode: str
    numerical_version: str
    implementation_penalty_nats: float
    target_rms: float
    training_samples: int
    guard_samples: int
    control_samples: int

    @classmethod
    def create(
        cls,
        alpha: float,
        beta: float,
        resources: ExperimentResources,
        receiver: ReceiverSpec,
        *,
        channel_estimator: str = "diag_tf",
        equalizer: str = "receiver_projection",
        transform_version: str = "fdidm-gamma-v1",
        implementation_mode: str = "ideal",
        numerical_version: str = "gmi-v1",
        implementation_penalty_nats: float = 0.0,
    ) -> "WaveformToken":
        normalized = {
            "alpha": _canonical_fractional(alpha),
            "beta": _canonical_fractional(beta),
            "channel_estimator": str(channel_estimator),
            "control_samples": int(resources.control_samples or 0),
            "equalizer": str(equalizer),
            "guard_samples": int(resources.guard_samples),
            "implementation_mode": str(implementation_mode),
            "implementation_penalty_nats": round(float(implementation_penalty_nats), 15),
            "numerical_version": str(numerical_version),
            "receiver_id": receiver.receiver_id,
            "resource_id": resources.resource_id,
            "target_rms": round(float(resources.target_rms), 12),
            "training_samples": int(resources.training_samples),
            "transform_version": str(transform_version),
        }
        token = cls(token_id=_digest("tok", normalized), **normalized)
        token.validate()
        return token

    def validate(self) -> None:
        required = (self.token_id, self.receiver_id, self.resource_id, self.channel_estimator, self.equalizer, self.transform_version, self.implementation_mode, self.numerical_version)
        if not all(str(value).strip() for value in required):
            raise ValueError("token is missing executable context")
        if self.alpha != _canonical_fractional(self.alpha) or self.beta != _canonical_fractional(self.beta):
            raise ValueError("token alpha/beta must be canonical in [0, 2)")
        _finite_float("implementation_penalty_nats", self.implementation_penalty_nats, minimum=0.0)
        _finite_float("target_rms", self.target_rms, minimum=np.finfo(float).tiny)
        if min(int(self.training_samples), int(self.guard_samples), int(self.control_samples)) < 0:
            raise ValueError("token sample counts must be non-negative")
        payload = self.normalized_dict(include_token_id=False)
        if self.token_id != _digest("tok", payload):
            raise ValueError("token_id does not match token content")

    def normalized_dict(self, *, include_token_id: bool = True) -> dict[str, Any]:
        result = {
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "channel_estimator": self.channel_estimator,
            "control_samples": int(self.control_samples),
            "equalizer": self.equalizer,
            "guard_samples": int(self.guard_samples),
            "implementation_mode": self.implementation_mode,
            "implementation_penalty_nats": float(self.implementation_penalty_nats),
            "numerical_version": self.numerical_version,
            "receiver_id": self.receiver_id,
            "resource_id": self.resource_id,
            "target_rms": float(self.target_rms),
            "training_samples": int(self.training_samples),
            "transform_version": self.transform_version,
        }
        if include_token_id:
            result["token_id"] = self.token_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WaveformToken":
        allowed = {item.name for item in fields(cls)}
        if set(payload) != allowed:
            raise ValueError("token fields are incomplete or unknown")
        token = cls(**dict(payload))
        token.validate()
        return token


@dataclass(frozen=True)
class TransceiverLibrary:
    resources: ExperimentResources
    receiver: ReceiverSpec
    tokens: Tuple[WaveformToken, ...]
    created_from: str = "manual"
    construction: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", tuple(self.tokens))
        object.__setattr__(self, "construction", MappingProxyType(dict(self.construction)))
        self.validate()

    def validate(self) -> None:
        if int(self.schema_version) != SCHEMA_VERSION:
            raise ValueError(f"unsupported library schema_version: {self.schema_version}")
        if not self.tokens:
            raise ValueError("library must contain at least one token")
        if len(self.tokens) > self.resources.catalogue_cap:
            raise ValueError("library exceeds catalogue_cap")
        self.receiver.validate(self.resources.signal_dimension)
        ids: set[str] = set()
        coordinates: set[Tuple[float, float]] = set()
        for token in self.tokens:
            token.validate()
            if token.resource_id != self.resources.resource_id or token.receiver_id != self.receiver.receiver_id:
                raise ValueError("token context does not match library")
            if token.token_id in ids or (token.alpha, token.beta) in coordinates:
                raise ValueError("library contains duplicate tokens or periodic endpoints")
            ids.add(token.token_id)
            coordinates.add((token.alpha, token.beta))

    def _content_dict(self) -> dict[str, Any]:
        return {
            "construction": dict(self.construction),
            "created_from": str(self.created_from),
            "receiver": self.receiver.normalized_dict(),
            "resources": self.resources.normalized_dict(),
            "schema_version": int(self.schema_version),
            "tokens": [token.normalized_dict() for token in self.tokens],
        }

    @property
    def content_digest(self) -> str:
        return _digest("sha256", self._content_dict(), length=64)

    @property
    def library_id(self) -> str:
        return _digest("lib", self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        result = self._content_dict()
        result.update({"content_digest": self.content_digest, "library_id": self.library_id, "resource_id": self.resources.resource_id})
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransceiverLibrary":
        expected = {"schema_version", "created_from", "resources", "receiver", "tokens", "construction", "content_digest", "library_id", "resource_id"}
        if set(payload) != expected:
            raise ValueError("library fields are incomplete or unknown")
        resources_payload = dict(payload["resources"])
        resources = ExperimentResources(**resources_payload)
        receiver_payload = dict(payload["receiver"])
        if receiver_payload.get("mode") == "full_matched" and receiver_payload.get("band_k") is None:
            receiver_payload["band_k"] = None
        receiver_payload["exponent_interval"] = tuple(receiver_payload["exponent_interval"])
        receiver = ReceiverSpec(**receiver_payload)
        library = cls(
            resources=resources,
            receiver=receiver,
            tokens=tuple(WaveformToken.from_dict(item) for item in payload["tokens"]),
            created_from=str(payload["created_from"]),
            construction=dict(payload["construction"]),
            schema_version=int(payload["schema_version"]),
        )
        if str(payload["resource_id"]) != library.resources.resource_id:
            raise ValueError("library resource_id mismatch")
        if str(payload["library_id"]) != library.library_id or str(payload["content_digest"]) != library.content_digest:
            raise ValueError("library digest mismatch")
        return library


@dataclass(frozen=True)
class TokenScore:
    token_id: str
    valid: bool
    reason: str
    gmi_nats_per_sample: float
    uncertainty_nats: float
    implementation_penalty_nats: float
    switch_cost_nats: float
    lower_bound_nats: float
    upper_bound_nats: float

    def __post_init__(self) -> None:
        numeric = (self.gmi_nats_per_sample, self.uncertainty_nats, self.implementation_penalty_nats, self.switch_cost_nats, self.lower_bound_nats, self.upper_bound_nats)
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("TokenScore values must be finite")


@dataclass(frozen=True)
class SelectionDecision:
    current_token_id: str
    candidate_token_id: str
    gain_lower_nats: float
    threshold_nats: float
    switch_allowed: bool
    reason: str
    context_id: str
    state_epoch: int
    library_id: str
    sequence: int
    scores: Tuple[TokenScore, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.gain_lower_nats)) or not math.isfinite(float(self.threshold_nats)):
            raise ValueError("selection bounds must be finite")
        object.__setattr__(self, "scores", tuple(self.scores))


__all__ = [
    "ChannelState",
    "ExperimentResources",
    "GMIExperiment",
    "GMIResult",
    "ReceiverSpec",
    "SCHEMA_VERSION",
    "SelectionDecision",
    "TokenScore",
    "TransceiverLibrary",
    "WaveformToken",
]
