"""GMI experiments and numerical scoring from the receiver-dependent paper."""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable, Tuple

import numpy as np

from waveform_sim.core.transforms import gamma_matrix, unitary_dft

from .model import (
    ChannelState,
    ExperimentResources,
    GMIExperiment,
    GMIResult,
    ReceiverSpec,
    TransceiverLibrary,
    WaveformToken,
)


def canonical_fractional_index(value: float, period: float = 2.0) -> float:
    period = float(period)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("period must be positive and finite")
    result = float(value) % period
    if math.isclose(result, 0.0, abs_tol=1e-12) or math.isclose(result, period, abs_tol=1e-12):
        return 0.0
    return round(result, 12)


@lru_cache(maxsize=512)
def _operator_cached(m: int, n: int, alpha: float, beta: float, basis: str) -> np.ndarray:
    V = np.kron(gamma_matrix(n, -beta), gamma_matrix(m, alpha))
    if basis == "tf":
        operator = V
    elif basis == "time":
        S0 = np.kron(np.eye(n, dtype=np.complex128), unitary_dft(m).conj().T)
        operator = S0 @ V
    else:
        raise ValueError("basis must be 'tf' or 'time'")
    operator = np.asarray(operator, dtype=np.complex128)
    operator.setflags(write=False)
    return operator


def fdidm_operator(m: int, n: int, alpha: float, beta: float, basis: str = "tf") -> np.ndarray:
    return _operator_cached(int(m), int(n), canonical_fractional_index(alpha), canonical_fractional_index(beta), str(basis)).copy()


def cyclic_band_projection(matrix: np.ndarray, band_k: int) -> np.ndarray:
    source = np.asarray(matrix, dtype=np.complex128)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("matrix must be square")
    size = source.shape[0]
    band = int(band_k)
    if band < 0 or band > size // 2:
        raise ValueError("band_k is outside the cyclic range")
    indices = np.arange(size)
    distance = np.abs(indices[:, None] - indices[None, :])
    mask = np.minimum(distance, size - distance) <= band
    return np.where(mask, source, 0.0 + 0.0j)


def build_experiment(
    channel_state: ChannelState,
    token: WaveformToken,
    resources: ExperimentResources,
    receiver: ReceiverSpec,
) -> GMIExperiment:
    if channel_state.dimension != resources.signal_dimension:
        raise ValueError("channel dimension does not match resources")
    if token.resource_id != resources.resource_id or token.receiver_id != receiver.receiver_id:
        raise ValueError("token context does not match resources/receiver")
    m, n = resources.grid_shape
    F = fdidm_operator(m, n, token.alpha, token.beta, basis=channel_state.basis)
    physical_channel = channel_state.channel_matrix
    B = F.conj().T @ physical_channel @ F
    if receiver.mode == "full_matched":
        C = B.copy()
    else:
        C = cyclic_band_projection(B, receiver.effective_band_k(B.shape[0]))
    if channel_state.noise_covariance is not None:
        Sigma = F.conj().T @ channel_state.noise_covariance @ F
    else:
        Sigma = float(channel_state.noise_variance) * np.eye(B.shape[0], dtype=np.complex128)
    return GMIExperiment(
        B=B,
        C=C,
        Sigma=Sigma,
        symbol_power=float(channel_state.symbol_power),
        charged_samples=resources.charged_samples,
        receiver=receiver,
        resource_id=resources.resource_id,
        token_id=token.token_id,
    )


def _diagonal_objective_and_derivative(s: float, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    s = float(s)
    denominator = 1.0 + s * x
    numerator = s * x * (1.0 - s * (1.0 + y))
    objective = np.sum(np.log(denominator) + numerator / denominator)
    numerator_derivative = x - 2.0 * s * x * (1.0 + y)
    derivative = np.sum(x / denominator + (numerator_derivative * denominator - numerator * x) / (denominator ** 2))
    return float(objective), float(derivative)


def _score_diagonal(experiment: GMIExperiment) -> GMIResult:
    Sigma = experiment.Sigma
    diagonal = np.real(np.diag(Sigma))
    sigma2 = float(np.mean(diagonal))
    if sigma2 <= 0.0:
        return GMIResult.invalid("non-positive white-noise variance", receiver_mode=experiment.receiver.mode, token_id=experiment.token_id, resource_id=experiment.resource_id)
    white = sigma2 * np.eye(Sigma.shape[0], dtype=np.complex128)
    if not np.allclose(Sigma, white, atol=1e-9 * max(1.0, sigma2), rtol=1e-8):
        return GMIResult.invalid("diagonal closed form requires white analysis noise", receiver_mode=experiment.receiver.mode, token_id=experiment.token_id, resource_id=experiment.resource_id)
    rho = experiment.symbol_power / sigma2
    B = experiment.B
    diagonal_energy = np.abs(np.diag(B)) ** 2
    x = rho * diagonal_energy
    y = rho * np.maximum(0.0, np.sum(np.abs(B) ** 2, axis=1) - diagonal_energy)
    if not np.any(x > np.finfo(float).eps):
        return GMIResult.invalid("zero diagonal signal energy", receiver_mode=experiment.receiver.mode, token_id=experiment.token_id, resource_id=experiment.resource_id)
    lo, hi = 0.0, 1.0
    _, derivative_lo = _diagonal_objective_and_derivative(lo, x, y)
    _, derivative_hi = _diagonal_objective_and_derivative(hi, x, y)
    if derivative_lo <= 0.0:
        s_star = lo
    elif derivative_hi >= 0.0:
        s_star = hi
    else:
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            _, derivative = _diagonal_objective_and_derivative(mid, x, y)
            if derivative > 0.0:
                lo = mid
            else:
                hi = mid
        s_star = 0.5 * (lo + hi)
    objective, _ = _diagonal_objective_and_derivative(s_star, x, y)
    value = max(0.0, objective / float(experiment.charged_samples))
    return GMIResult(
        valid=True,
        reason="",
        gmi_nats_per_sample=value,
        gmi_bits_per_sample=value / math.log(2.0),
        s_star=s_star,
        receiver_mode=experiment.receiver.mode,
        token_id=experiment.token_id,
        resource_id=experiment.resource_id,
        covariance_min_eigenvalue=sigma2,
        covariance_condition=1.0,
        solve_path="diagonal_eq13_bisection48",
    )


def _score_full_matched(experiment: GMIExperiment) -> GMIResult:
    eig = np.linalg.eigvalsh(experiment.Sigma)
    minimum = float(np.min(eig))
    if minimum <= 0.0:
        return GMIResult.invalid("noise covariance is not positive definite", receiver_mode=experiment.receiver.mode, token_id=experiment.token_id, resource_id=experiment.resource_id)
    try:
        whitened = np.linalg.solve(experiment.Sigma, experiment.B)
        gram = np.eye(experiment.B.shape[1], dtype=np.complex128) + experiment.symbol_power * (experiment.B.conj().T @ whitened)
        sign, logdet = np.linalg.slogdet(gram)
    except np.linalg.LinAlgError as exc:
        return GMIResult.invalid(f"matched covariance solve failed: {exc}", receiver_mode=experiment.receiver.mode, token_id=experiment.token_id, resource_id=experiment.resource_id)
    if abs(sign - 1.0) > 1e-8 or not math.isfinite(float(np.real(logdet))):
        return GMIResult.invalid("matched log-det is not positive finite", receiver_mode=experiment.receiver.mode, token_id=experiment.token_id, resource_id=experiment.resource_id)
    value = max(0.0, float(np.real(logdet)) / float(experiment.charged_samples))
    return GMIResult(True, "", value, value / math.log(2.0), 1.0, experiment.receiver.mode, experiment.token_id, experiment.resource_id, minimum, float(np.max(eig) / minimum), "matched_logdet_solve")


def _general_objective(experiment: GMIExperiment, s: float) -> float:
    B, C, Sigma = experiment.B, experiment.C, experiment.Sigma
    power = experiment.symbol_power
    E = B - C
    try:
        sigma_inv_c = np.linalg.solve(Sigma, C)
        first = np.eye(C.shape[1], dtype=np.complex128) + s * power * C.conj().T @ sigma_inv_c
        sign, logdet = np.linalg.slogdet(first)
        if abs(sign - 1.0) > 1e-8:
            return float("-inf")
        mismatch = power * np.real(np.trace(E.conj().T @ np.linalg.solve(Sigma, E))) + B.shape[0]
        K = Sigma + s * power * C @ C.conj().T
        Rz = power * B @ B.conj().T + Sigma
        tail = np.real(np.trace(np.linalg.solve(K, Rz)))
        return float((np.real(logdet) - s * mismatch + s * tail) / experiment.charged_samples)
    except np.linalg.LinAlgError:
        return float("-inf")


def _score_cyclic_band(experiment: GMIExperiment) -> GMIResult:
    lo, hi = experiment.receiver.exponent_interval
    # Concave one-dimensional objective: deterministic golden-section search.
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = float(lo), float(hi)
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc, fd = _general_objective(experiment, c), _general_objective(experiment, d)
    for _ in range(64):
        if fc >= fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = _general_objective(experiment, c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = _general_objective(experiment, d)
    s_star = 0.5 * (left + right)
    value = _general_objective(experiment, s_star)
    if not math.isfinite(value):
        return GMIResult.invalid("cyclic-band objective is not finite", receiver_mode=experiment.receiver.mode, token_id=experiment.token_id, resource_id=experiment.resource_id)
    eig = np.linalg.eigvalsh(experiment.Sigma)
    return GMIResult(True, "", max(0.0, value), max(0.0, value) / math.log(2.0), s_star, experiment.receiver.mode, experiment.token_id, experiment.resource_id, float(np.min(eig)), float(np.max(eig) / np.min(eig)), "cyclic_band_golden_section")


def score_gmi(experiment: GMIExperiment) -> GMIResult:
    try:
        if experiment.receiver.mode == "diagonal":
            return _score_diagonal(experiment)
        if experiment.receiver.mode == "full_matched":
            return _score_full_matched(experiment)
        return _score_cyclic_band(experiment)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return GMIResult.invalid(f"scoring failed: {type(exc).__name__}: {exc}", receiver_mode=experiment.receiver.mode, token_id=experiment.token_id, resource_id=experiment.resource_id)


def score_library(channel_state: ChannelState, library: TransceiverLibrary) -> Tuple[GMIResult, ...]:
    reason = channel_state.validity_reason()
    if reason:
        return tuple(GMIResult.invalid(reason, receiver_mode=library.receiver.mode, token_id=token.token_id, resource_id=library.resources.resource_id) for token in library.tokens)
    if library.receiver.mode == "full_matched":
        first = score_gmi(build_experiment(channel_state, library.tokens[0], library.resources, library.receiver))
        return tuple(GMIResult(first.valid, first.reason, first.gmi_nats_per_sample, first.gmi_bits_per_sample, first.s_star, first.receiver_mode, token.token_id, first.resource_id, first.covariance_min_eigenvalue, first.covariance_condition, first.solve_path) for token in library.tokens)
    return tuple(score_gmi(build_experiment(channel_state, token, library.resources, library.receiver)) for token in library.tokens)


__all__ = [
    "build_experiment",
    "canonical_fractional_index",
    "cyclic_band_projection",
    "fdidm_operator",
    "score_gmi",
    "score_library",
]
