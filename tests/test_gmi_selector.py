from dataclasses import fields
import time

import numpy as np
import pytest

from waveform_sim.gmi.geometry import anisotropic_periodic_grid
from waveform_sim.gmi.model import ChannelState, ExperimentResources, ReceiverSpec, TransceiverLibrary, WaveformToken
from waveform_sim.gmi.selector import RobustnessProfile, SingleFlightSelector, score_tokens, select_token


def fixture_library(mode="diagonal"):
    resources = ExperimentResources(4, catalogue_cap=4, grid_rows=2, grid_cols=2)
    receiver = ReceiverSpec(mode, None if mode == "full_matched" else 0)
    tokens = tuple(WaveformToken.create(a, b, resources, receiver) for a, b in anisotropic_periodic_grid(2, 2))
    return TransceiverLibrary(resources, receiver, tokens)


def state(**overrides):
    values = dict(context_id="ctx", epoch=1, timestamp=1.0, h_tf=np.diag([1.9, 1.4, 1.0, 0.6]), noise_variance=1.0, state_uncertainty_nats=0.0)
    values.update(overrides)
    return ChannelState(**values)


def test_score_profiles_become_more_conservative():
    library = fixture_library()
    current = library.tokens[0].token_id
    uncertain = state(state_uncertainty_nats=1e-3)
    fast = score_tokens(uncertain, library, current, "fast")
    conservative = score_tokens(uncertain, library, current, "conservative")
    assert all(c.lower_bound_nats <= f.lower_bound_nats for f, c in zip(fast, conservative))


def test_threshold_is_strict_and_full_matched_is_reference():
    library = fixture_library()
    current = library.tokens[0].token_id
    decision = select_token(state(), library, current, RobustnessProfile("test", 0.0, 1e9, 0.0))
    assert not decision.switch_allowed
    full = fixture_library("full_matched")
    reference = select_token(state(), full, full.tokens[0].token_id)
    assert not reference.switch_allowed
    assert reference.reason == "reference_flat"


def test_fail_closed_on_overflow_and_probe_gap():
    library = fixture_library()
    current = library.tokens[0].token_id
    assert select_token(state(overflow=True), library, current).reason == "overflow"
    assert select_token(state(continuity=False), library, current).reason == "probe_gap"


def test_single_flight_deduplicates_switch_request():
    library = fixture_library()
    current = library.tokens[0].token_id
    selector = SingleFlightSelector()
    profile = RobustnessProfile("test", 0.0, -1.0, 0.0)
    first = selector.select(state(), library, current, profile)
    second = selector.select(state(), library, current, profile)
    if first.switch_allowed:
        assert not second.switch_allowed
        assert second.reason == "duplicate_state_epoch"


def test_no_payload_truth_in_interface():
    names = {item.name for item in fields(ChannelState)}
    assert not names.intersection({"payload", "crc", "ser", "ber", "evm"})
    library = fixture_library()
    current = library.tokens[0].token_id
    first = select_token(state(), library, current)
    external_payload = np.arange(100)  # deliberately not passed to selector
    second = select_token(state(), library, current)
    assert external_payload.size == 100
    assert first == second


def test_16_by_16_sixteen_token_scoring_is_bounded():
    resources = ExperimentResources(
        256,
        catalogue_cap=16,
        grid_rows=16,
        grid_cols=16,
        sample_rate_hz=1e6,
        bandwidth_hz=1e6,
    )
    receiver = ReceiverSpec("diagonal")
    tokens = tuple(
        WaveformToken.create(alpha, beta, resources, receiver)
        for alpha, beta in anisotropic_periodic_grid(4, 4)
    )
    library = TransceiverLibrary(resources, receiver, tokens)
    channel = ChannelState(
        "perf",
        1,
        1.0,
        h_tf=np.diag(np.linspace(0.5, 2.0, 256)),
        noise_variance=1.0,
    )
    started = time.perf_counter()
    scores = score_tokens(channel, library, tokens[0].token_id)
    elapsed = time.perf_counter() - started
    assert len(scores) == 16 and all(item.valid for item in scores)
    # The work runs off the RX thread; this generous bound catches accidental
    # exponential/library-wide regressions without making CI timing brittle.
    assert elapsed < 5.0
