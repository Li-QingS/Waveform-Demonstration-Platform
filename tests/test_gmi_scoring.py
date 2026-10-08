import math

import numpy as np
import pytest

from waveform_sim.gmi.model import ChannelState, ExperimentResources, ReceiverSpec, TransceiverLibrary, WaveformToken
from waveform_sim.gmi.scoring import (
    build_experiment,
    canonical_fractional_index,
    cyclic_band_projection,
    fdidm_operator,
    score_gmi,
    score_library,
)


def make_scoring_fixture(receiver_mode="diagonal"):
    resources = ExperimentResources(4, guard_samples=1, training_samples=1, catalogue_cap=4, grid_rows=2, grid_cols=2)
    receiver = ReceiverSpec(receiver_mode, None if receiver_mode == "full_matched" else 0)
    tokens = tuple(WaveformToken.create(a, b, resources, receiver) for a, b in [(0.0, 0.0), (0.25, 0.75), (1.0, 1.5)])
    library = TransceiverLibrary(resources, receiver, tokens)
    return resources, receiver, library


def test_canonical_period_and_operator_unitarity():
    assert canonical_fractional_index(-0.25) == 1.75
    assert canonical_fractional_index(2.0) == 0.0
    V = fdidm_operator(2, 2, 0.37, 1.21, "tf")
    assert np.allclose(V.conj().T @ V, np.eye(4), atol=1e-11)


def test_cyclic_band_projection():
    matrix = np.arange(25).reshape(5, 5).astype(complex)
    diagonal = cyclic_band_projection(matrix, 0)
    assert np.count_nonzero(diagonal - np.diag(np.diag(matrix))) == 0
    band = cyclic_band_projection(matrix, 1)
    assert band[0, 4] == matrix[0, 4]
    assert band[0, 2] == 0.0


def test_tf_and_time_basis_build_expected_effective_channel():
    resources, receiver, library = make_scoring_fixture()
    token = library.tokens[1]
    V = fdidm_operator(2, 2, token.alpha, token.beta, "tf")
    H = np.diag([2.0, 1.5, 1.0, 0.5]).astype(complex)
    state = ChannelState("ctx", 1, 1.0, h_tf=H, noise_variance=0.1)
    experiment = build_experiment(state, token, resources, receiver)
    assert np.allclose(experiment.B, V.conj().T @ H @ V)
    assert np.allclose(experiment.C, np.diag(np.diag(experiment.B)))


def test_diagonal_formula_matches_direct_grid_maximum():
    resources, receiver, library = make_scoring_fixture()
    H = np.diag([1.9, 1.4, 1.0, 0.6]).astype(complex)
    state = ChannelState("ctx", 1, 1.0, h_tf=H, noise_variance=1.0, symbol_power=10.0)
    experiment = build_experiment(state, library.tokens[1], resources, receiver)
    result = score_gmi(experiment)
    assert result.valid
    assert 0.0 <= result.s_star <= 1.0
    B = experiment.B
    x = 10.0 * np.abs(np.diag(B)) ** 2
    y = 10.0 * (np.sum(np.abs(B) ** 2, axis=1) - np.abs(np.diag(B)) ** 2)
    s = np.linspace(0.0, 1.0, 20001)
    values = np.sum(np.log(1.0 + s[:, None] * x) + s[:, None] * x * (1.0 - s[:, None] * (1.0 + y)) / (1.0 + s[:, None] * x), axis=1)
    expected = float(np.max(values)) / resources.charged_samples
    assert result.gmi_nats_per_sample == pytest.approx(expected, rel=2e-7, abs=1e-10)


def test_diagonal_rejects_colored_noise_closed_form():
    resources, receiver, library = make_scoring_fixture()
    state = ChannelState("ctx", 1, 1.0, h_tf=np.eye(4), noise_covariance=np.diag([1.0, 2.0, 1.0, 2.0]))
    result = score_gmi(build_experiment(state, library.tokens[0], resources, receiver))
    assert not result.valid
    assert "white" in result.reason


def test_shift_by_two_preserves_diagonal_gmi():
    resources, receiver, library = make_scoring_fixture()
    H = np.array([[1.0, 0.2j, 0.0, 0.1], [-0.2j, 0.8, 0.05, 0.0], [0.0, 0.05, 0.7, 0.1j], [0.1, 0.0, -0.1j, 0.6]], dtype=complex)
    state = ChannelState("ctx", 1, 1.0, h_tf=H, noise_variance=0.2)
    first = WaveformToken.create(0.37, 0.61, resources, receiver)
    shifted = WaveformToken.create(2.37, 2.61, resources, receiver)
    assert score_gmi(build_experiment(state, first, resources, receiver)).gmi_nats_per_sample == pytest.approx(score_gmi(build_experiment(state, shifted, resources, receiver)).gmi_nats_per_sample, abs=1e-12)


def test_full_matched_is_action_independent_and_uses_logdet():
    resources, receiver, library = make_scoring_fixture("full_matched")
    H = np.array([[1.0, 0.2, 0.0, 0.0], [0.1j, 0.9, 0.1, 0.0], [0.0, 0.0, 0.7, 0.2j], [0.1, 0.0, 0.0, 0.5]], dtype=complex)
    state = ChannelState("ctx", 1, 1.0, h_tf=H, noise_variance=0.3, symbol_power=2.0)
    results = score_library(state, library)
    assert all(result.valid for result in results)
    assert all(result.solve_path == "matched_logdet_solve" for result in results)
    assert max(result.gmi_nats_per_sample for result in results) - min(result.gmi_nats_per_sample for result in results) < 1e-12
    expected = np.linalg.slogdet(np.eye(4) + 2.0 / 0.3 * H.conj().T @ H)[1] / resources.charged_samples
    assert results[0].gmi_nats_per_sample == pytest.approx(expected, rel=1e-11)


def test_invalid_state_fails_closed_for_whole_library():
    _, _, library = make_scoring_fixture()
    state = ChannelState("ctx", 1, 1.0, h_tf=np.eye(4), noise_variance=1.0, overflow=True)
    results = score_library(state, library)
    assert all(not result.valid and result.reason == "overflow" for result in results)
