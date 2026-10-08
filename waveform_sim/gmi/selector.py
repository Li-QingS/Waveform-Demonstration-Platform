"""Fail-closed robust online selection over a finite GMI library."""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Dict, Tuple

from .model import ChannelState, SelectionDecision, TokenScore, TransceiverLibrary
from .scoring import score_library


@dataclass(frozen=True)
class RobustnessProfile:
    name: str
    uncertainty_multiplier: float
    minimum_gain_nats: float
    switch_cost_nats: float

    @classmethod
    def named(cls, name: str) -> "RobustnessProfile":
        profiles = {
            "fast": cls("fast", 0.75, 1.0e-4, 5.0e-5),
            "standard": cls("standard", 1.0, 2.5e-4, 1.0e-4),
            "conservative": cls("conservative", 1.5, 5.0e-4, 2.5e-4),
        }
        try:
            return profiles[str(name).lower()]
        except KeyError as exc:
            raise ValueError(f"unknown robustness profile: {name}") from exc


def _switch_distance(current, candidate) -> float:
    da = min(abs(candidate.alpha - current.alpha), 2.0 - abs(candidate.alpha - current.alpha))
    db = min(abs(candidate.beta - current.beta), 2.0 - abs(candidate.beta - current.beta))
    return math.hypot(da, db)


def score_tokens(
    state: ChannelState,
    library: TransceiverLibrary,
    current_token_id: str,
    profile: RobustnessProfile | str = "standard",
) -> Tuple[TokenScore, ...]:
    profile = RobustnessProfile.named(profile) if isinstance(profile, str) else profile
    by_id = {token.token_id: token for token in library.tokens}
    if current_token_id not in by_id:
        raise ValueError("current token is not in the library")
    current = by_id[current_token_id]
    results = score_library(state, library)
    scores = []
    for token, result in zip(library.tokens, results):
        uncertainty = profile.uncertainty_multiplier * float(state.state_uncertainty_nats)
        penalty = float(token.implementation_penalty_nats)
        switch_cost = 0.0 if token.token_id == current_token_id else profile.switch_cost_nats * _switch_distance(current, token)
        value = result.gmi_nats_per_sample if result.valid else 0.0
        lower = max(0.0, value - uncertainty - penalty)
        upper = value + uncertainty + penalty
        scores.append(TokenScore(token.token_id, result.valid, result.reason, value, uncertainty, penalty, switch_cost, lower, upper))
    return tuple(scores)


def select_token(
    state: ChannelState,
    library: TransceiverLibrary,
    current_token_id: str,
    profile: RobustnessProfile | str = "standard",
    *,
    sequence: int = 0,
    validation_in_progress: bool = False,
) -> SelectionDecision:
    profile = RobustnessProfile.named(profile) if isinstance(profile, str) else profile
    reason = state.validity_reason()
    scores = score_tokens(state, library, current_token_id, profile)
    current = next(score for score in scores if score.token_id == current_token_id)

    def decision(candidate_id: str, gain: float, allowed: bool, why: str) -> SelectionDecision:
        return SelectionDecision(current_token_id, candidate_id, float(gain), profile.minimum_gain_nats, allowed, why, state.context_id, int(state.epoch), library.library_id, int(sequence), scores)

    if reason:
        return decision(current_token_id, 0.0, False, reason)
    if validation_in_progress:
        return decision(current_token_id, 0.0, False, "validation_in_progress")
    if library.receiver.mode == "full_matched":
        return decision(current_token_id, 0.0, False, "reference_flat")
    valid = [score for score in scores if score.valid]
    if not current.valid or not valid:
        return decision(current_token_id, 0.0, False, "current_or_candidates_unavailable")
    token_by_id = {token.token_id: token for token in library.tokens}
    current_token = token_by_id[current_token_id]
    valid.sort(
        key=lambda score: (
            -score.lower_bound_nats,
            -score.gmi_nats_per_sample,
            _switch_distance(current_token, token_by_id[score.token_id]),
            score.token_id,
        )
    )
    candidate = valid[0]
    gain_lower = candidate.lower_bound_nats - current.upper_bound_nats - candidate.switch_cost_nats
    if candidate.token_id == current_token_id:
        return decision(current_token_id, gain_lower, False, "current_token_is_robust_best")
    if gain_lower <= profile.minimum_gain_nats:
        return decision(candidate.token_id, gain_lower, False, "gain_lower_not_above_threshold")
    return decision(candidate.token_id, gain_lower, True, "robust_gmi_gain")


class SingleFlightSelector:
    """Deduplicate state epochs and selection transactions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_key = None

    def select(self, state: ChannelState, library: TransceiverLibrary, current_token_id: str, profile: RobustnessProfile | str = "standard", *, sequence: int = 0, validation_in_progress: bool = False) -> SelectionDecision:
        key = (state.context_id, int(state.epoch), library.library_id)
        with self._lock:
            duplicate = key == self._last_key
            self._last_key = key
        decision = select_token(state, library, current_token_id, profile, sequence=sequence, validation_in_progress=validation_in_progress)
        if duplicate and decision.switch_allowed:
            return SelectionDecision(decision.current_token_id, decision.candidate_token_id, decision.gain_lower_nats, decision.threshold_nats, False, "duplicate_state_epoch", decision.context_id, decision.state_epoch, decision.library_id, decision.sequence, decision.scores)
        return decision


__all__ = ["RobustnessProfile", "SingleFlightSelector", "score_tokens", "select_token"]
