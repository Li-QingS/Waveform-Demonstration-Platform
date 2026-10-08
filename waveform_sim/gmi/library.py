"""Finite receiver-dependent transceiver-library construction."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Sequence, Tuple

import numpy as np

from .model import ChannelState, ExperimentResources, ReceiverSpec, TransceiverLibrary, WaveformToken
from .scoring import build_experiment, score_gmi


@dataclass(frozen=True)
class RegretMatrix:
    values: np.ndarray
    available: np.ndarray
    oracle_indices: Tuple[int, ...]
    oracle_scores: Tuple[float, ...]
    state_digest: str
    candidate_digest: str


@dataclass(frozen=True)
class LibrarySelection:
    selected_indices: Tuple[int, ...]
    objective: float
    covered: bool
    uncovered_states: Tuple[int, ...]
    reason: str


def _digest_rows(rows: Iterable[object]) -> str:
    import hashlib

    payload = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evaluate_regret_matrix(
    states: Sequence[ChannelState],
    candidates: Sequence[WaveformToken],
    resources: ExperimentResources,
    receiver: ReceiverSpec,
) -> RegretMatrix:
    if not states or not candidates:
        raise ValueError("states and candidates must be non-empty")
    scores = np.zeros((len(states), len(candidates)), dtype=np.float64)
    available = np.zeros_like(scores, dtype=bool)
    for row, state in enumerate(states):
        if state.validity_reason() or state.context_id == "":
            continue
        for column, token in enumerate(candidates):
            try:
                result = score_gmi(build_experiment(state, token, resources, receiver))
            except (ValueError, np.linalg.LinAlgError):
                continue
            if result.valid:
                scores[row, column] = result.gmi_nats_per_sample
                available[row, column] = True
    oracle_indices = []
    oracle_scores = []
    regrets = np.zeros_like(scores)
    for row in range(len(states)):
        valid_columns = np.flatnonzero(available[row])
        if valid_columns.size == 0:
            oracle_indices.append(-1)
            oracle_scores.append(0.0)
            continue
        best_score = float(np.max(scores[row, valid_columns]))
        best_columns = valid_columns[np.isclose(scores[row, valid_columns], best_score, atol=1e-15, rtol=1e-13)]
        oracle = int(min(best_columns, key=lambda index: candidates[int(index)].token_id))
        oracle_indices.append(oracle)
        oracle_scores.append(best_score)
        regrets[row, valid_columns] = np.maximum(0.0, best_score - scores[row, valid_columns])
    state_rows = [(state.context_id, int(state.epoch), state.basis) for state in states]
    candidate_rows = [token.token_id for token in candidates]
    return RegretMatrix(regrets, available, tuple(oracle_indices), tuple(oracle_scores), _digest_rows(state_rows), _digest_rows(candidate_rows))


def _objective(regret: RegretMatrix, selected: Sequence[int]) -> Tuple[float, Tuple[int, ...]]:
    if not selected:
        return float("inf"), tuple(range(regret.values.shape[0]))
    per_state = np.full(regret.values.shape[0], np.inf, dtype=np.float64)
    for index in selected:
        usable = regret.available[:, int(index)]
        per_state[usable] = np.minimum(per_state[usable], regret.values[usable, int(index)])
    invalid = tuple(int(index) for index in np.flatnonzero(~np.isfinite(per_state)))
    return (float(np.max(per_state)) if not invalid else float("inf")), invalid


def select_fixed_size_library(regret_matrix: RegretMatrix, capacity: int, seed: int = 0) -> LibrarySelection:
    del seed  # deterministic by design; retained in the public reproducibility contract
    capacity = int(capacity)
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    candidates = [index for index in range(regret_matrix.values.shape[1]) if np.any(regret_matrix.available[:, index])]
    if not candidates:
        return LibrarySelection((), float("inf"), False, tuple(range(regret_matrix.values.shape[0])), "no valid candidates")
    target = min(capacity, len(candidates))
    selected: list[int] = []
    while len(selected) < target:
        options = []
        for candidate in candidates:
            if candidate in selected:
                continue
            objective, invalid = _objective(regret_matrix, selected + [candidate])
            options.append((objective, len(invalid), candidate))
        _, _, chosen = min(options)
        selected.append(chosen)
    improved = True
    while improved:
        improved = False
        current_objective, _ = _objective(regret_matrix, selected)
        best = (current_objective, tuple(sorted(selected)), None)
        for old in tuple(selected):
            for new in candidates:
                if new in selected:
                    continue
                proposal = sorted(index for index in selected if index != old) + [new]
                objective, invalid = _objective(regret_matrix, proposal)
                if invalid:
                    continue
                key = (objective, tuple(sorted(proposal)), (old, new))
                if objective < best[0] - 1e-15 and key < best:
                    best = key
        if best[2] is not None:
            old, new = best[2]
            selected.remove(old)
            selected.append(new)
            selected.sort()
            improved = True
    objective, invalid = _objective(regret_matrix, selected)
    return LibrarySelection(tuple(selected), objective, not invalid, invalid, "" if not invalid else "some states have no executable candidate")


def select_cover_library(regret_matrix: RegretMatrix, epsilon: float, capacity: int) -> LibrarySelection:
    epsilon = float(epsilon)
    capacity = int(capacity)
    if epsilon < 0.0 or not math.isfinite(epsilon) or capacity <= 0:
        raise ValueError("epsilon must be finite/non-negative and capacity positive")
    coverage = regret_matrix.available & (regret_matrix.values <= epsilon + 1e-15)
    uncovered = set(range(regret_matrix.values.shape[0]))
    selected: list[int] = []
    while uncovered and len(selected) < capacity:
        options = []
        for candidate in range(regret_matrix.values.shape[1]):
            if candidate in selected:
                continue
            newly = {state for state in uncovered if coverage[state, candidate]}
            if not newly:
                continue
            worst = max(float(regret_matrix.values[state, candidate]) for state in newly)
            options.append((-len(newly), worst, candidate, newly))
        if not options:
            break
        _, _, chosen, newly = min(options, key=lambda item: item[:3])
        selected.append(chosen)
        uncovered.difference_update(newly)
    objective, invalid = _objective(regret_matrix, selected)
    uncovered.update(invalid)
    covered = not uncovered
    reason = "" if covered else ("capacity exhausted before all states were covered" if len(selected) >= capacity else "epsilon cover is infeasible")
    return LibrarySelection(tuple(selected), objective, covered, tuple(sorted(uncovered)), reason)


def build_transceiver_library(
    states: Sequence[ChannelState],
    candidates: Sequence[WaveformToken],
    resources: ExperimentResources,
    receiver: ReceiverSpec,
    *,
    capacity: int,
    method: str = "p_center",
    epsilon: float = 0.0,
    seed: int = 0,
) -> TransceiverLibrary:
    matrix = evaluate_regret_matrix(states, candidates, resources, receiver)
    if method == "p_center":
        selection = select_fixed_size_library(matrix, capacity, seed)
    elif method == "set_cover":
        selection = select_cover_library(matrix, epsilon, capacity)
    else:
        raise ValueError("method must be p_center or set_cover")
    if not selection.selected_indices:
        raise ValueError(selection.reason or "library selection failed")
    selected = tuple(candidates[index] for index in selection.selected_indices)
    construction = {
        "candidate_digest": matrix.candidate_digest,
        "capacity": int(capacity),
        "covered": bool(selection.covered),
        "epsilon": float(epsilon),
        "method": method,
        "objective": None if not math.isfinite(selection.objective) else float(selection.objective),
        "seed": int(seed),
        "state_digest": matrix.state_digest,
        "uncovered_states": list(selection.uncovered_states),
    }
    return TransceiverLibrary(resources, receiver, selected, created_from=method, construction=construction)


def save_library(path: str | os.PathLike[str], library: TransceiverLibrary) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(library.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_library(path: str | os.PathLike[str]) -> TransceiverLibrary:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return TransceiverLibrary.from_dict(payload)


__all__ = [
    "LibrarySelection",
    "RegretMatrix",
    "build_transceiver_library",
    "evaluate_regret_matrix",
    "load_library",
    "save_library",
    "select_cover_library",
    "select_fixed_size_library",
]
