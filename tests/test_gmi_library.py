import copy

import numpy as np
import pytest

from waveform_sim.gmi.geometry import anisotropic_periodic_grid
from waveform_sim.gmi.library import (
    RegretMatrix,
    build_transceiver_library,
    evaluate_regret_matrix,
    load_library,
    save_library,
    select_cover_library,
    select_fixed_size_library,
)
from waveform_sim.gmi.model import ChannelState, ExperimentResources, ReceiverSpec, WaveformToken


def fixture_data():
    resources = ExperimentResources(4, catalogue_cap=16, grid_rows=2, grid_cols=2)
    receiver = ReceiverSpec("diagonal")
    candidates = tuple(WaveformToken.create(a, b, resources, receiver) for a, b in anisotropic_periodic_grid(4, 4))
    base = np.diag([1.9, 1.4, 1.0, 0.6]).astype(complex)
    states = (
        ChannelState("ctx", 1, 1.0, h_tf=base, noise_variance=1.0),
        ChannelState("ctx", 2, 2.0, h_tf=np.roll(np.roll(base, 1, axis=0), 1, axis=1), noise_variance=1.0),
    )
    return resources, receiver, candidates, states


def test_regret_matrix_has_explicit_availability_and_oracle():
    resources, receiver, candidates, states = fixture_data()
    result = evaluate_regret_matrix(states, candidates, resources, receiver)
    assert result.values.shape == (2, 16)
    assert result.available.all()
    assert all(index >= 0 for index in result.oracle_indices)
    assert np.all(result.values >= 0.0)


def test_p_center_is_deterministic_and_capacity_safe():
    values = np.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    regret = RegretMatrix(values, np.ones_like(values, dtype=bool), (0, 2, 1), (1.0, 1.0, 1.0), "s", "c")
    first = select_fixed_size_library(regret, 2, seed=7)
    second = select_fixed_size_library(regret, 2, seed=999)
    assert first == second
    assert len(first.selected_indices) == 2
    assert first.covered


def test_set_cover_reports_feasible_and_capacity_failure():
    values = np.asarray([[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]])
    regret = RegretMatrix(values, np.ones_like(values, dtype=bool), (0, 1, 2), (1.0, 1.0, 1.0), "s", "c")
    assert select_cover_library(regret, 0.1, 3).covered
    failure = select_cover_library(regret, 0.1, 2)
    assert not failure.covered
    assert failure.uncovered_states


def test_build_save_load_is_stable(tmp_path):
    resources, receiver, candidates, states = fixture_data()
    library = build_transceiver_library(states, candidates, resources, receiver, capacity=4)
    assert len(library.tokens) <= 4
    path = tmp_path / "library.json"
    save_library(path, library)
    loaded = load_library(path)
    assert loaded.library_id == library.library_id
    assert loaded.content_digest == library.content_digest
    payload = path.read_text(encoding="utf-8").replace('"schema_version": 1', '"schema_version": 99')
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_library(path)
