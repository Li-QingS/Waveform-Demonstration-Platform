import copy
from dataclasses import fields

import numpy as np
import pytest

from waveform_sim.gmi.model import (
    ChannelState,
    ExperimentResources,
    GMIExperiment,
    GMIResult,
    ReceiverSpec,
    TransceiverLibrary,
    WaveformToken,
)


def resources(**overrides):
    values = dict(
        signal_dimension=4,
        grid_rows=2,
        grid_cols=2,
        guard_samples=4,
        training_samples=4,
        catalogue_cap=16,
        sample_rate_hz=1e6,
        bandwidth_hz=500e3,
    )
    values.update(overrides)
    return ExperimentResources(**values)


def test_resources_have_common_charged_budget_and_stable_id():
    first = resources()
    second = resources()
    assert first.control_bits == 4
    assert first.control_samples == 4
    assert first.charged_samples == 16
    assert first.resource_id == second.resource_id
    assert resources(catalogue_cap=1).control_samples == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"signal_dimension": 0},
        {"catalogue_cap": 16, "control_bits": 2},
        {"catalogue_cap": 16, "control_samples": 1},
        {"target_rms": 1.0, "peak_limit": 0.5},
        {"grid_rows": 3, "grid_cols": 2},
    ],
)
def test_resources_reject_invalid_contracts(kwargs):
    with pytest.raises(ValueError):
        resources(**kwargs)


def test_receiver_validation():
    assert ReceiverSpec("diagonal").effective_band_k(4) == 0
    assert ReceiverSpec("full_matched", None).effective_band_k(4) == 2
    assert ReceiverSpec("cyclic_band", 1).effective_band_k(4) == 1
    with pytest.raises(ValueError):
        ReceiverSpec("diagonal", 1)
    with pytest.raises(ValueError):
        ReceiverSpec("cyclic_band", 3).validate(4)


def test_experiment_validates_shape_covariance_and_finiteness():
    receiver = ReceiverSpec("diagonal")
    experiment = GMIExperiment(np.eye(2), np.eye(2), np.eye(2), 1.0, 4, receiver, "res")
    assert not experiment.B.flags.writeable
    with pytest.raises(ValueError):
        GMIExperiment(np.eye(2), np.eye(3), np.eye(2), 1.0, 4, receiver, "res")
    with pytest.raises(ValueError):
        GMIExperiment(np.eye(2), np.eye(2), np.diag([1.0, 0.0]), 1.0, 4, receiver, "res")


def test_invalid_gmi_is_explicit_and_sort_fields_are_finite():
    result = GMIResult.invalid("bad covariance", receiver_mode="diagonal")
    assert not result.valid
    assert result.gmi_nats_per_sample == 0.0
    assert result.s_star is None
    with pytest.raises(ValueError):
        GMIResult(True, "", float("nan"), 0.0, 0.5, "diagonal", "", "")


def test_channel_state_has_no_payload_truth_fields():
    names = {item.name for item in fields(ChannelState)}
    assert not names.intersection({"payload", "crc", "ser", "ber", "evm", "reference_symbols"})
    state = ChannelState("ctx", 1, 1.0, h_tf=np.eye(4), noise_variance=0.1)
    assert state.validity_reason() == ""
    assert not state.channel_matrix.flags.writeable
    with pytest.raises(TypeError):
        ChannelState("ctx", 1, 1.0, h_tf=np.eye(4), noise_variance=0.1, payload=b"truth")


def test_token_is_periodic_and_has_stable_content_id():
    common = resources()
    receiver = ReceiverSpec("diagonal")
    first = WaveformToken.create(0.0, -0.25, common, receiver)
    second = WaveformToken.create(2.0, 1.75, common, receiver)
    assert first.alpha == 0.0
    assert first.beta == 1.75
    assert first.token_id == second.token_id


def test_library_round_trip_and_strict_digest():
    common = resources()
    receiver = ReceiverSpec("diagonal")
    tokens = tuple(WaveformToken.create(alpha, beta, common, receiver) for alpha, beta in [(0.0, 0.0), (0.5, 1.0)])
    library = TransceiverLibrary(common, receiver, tokens, created_from="unit-test", construction={"seed": 7})
    encoded = library.to_dict()
    decoded = TransceiverLibrary.from_dict(encoded)
    assert decoded.library_id == library.library_id
    assert decoded.content_digest == library.content_digest
    tampered = copy.deepcopy(encoded)
    tampered["tokens"][0]["alpha"] = 0.25
    with pytest.raises(ValueError):
        TransceiverLibrary.from_dict(tampered)
    unknown = copy.deepcopy(encoded)
    unknown["schema_version"] = 99
    with pytest.raises(ValueError):
        TransceiverLibrary.from_dict(unknown)


def test_library_rejects_periodic_duplicate_and_mixed_context():
    common = resources(catalogue_cap=2)
    receiver = ReceiverSpec("diagonal")
    token = WaveformToken.create(0.0, 0.0, common, receiver)
    with pytest.raises(ValueError):
        TransceiverLibrary(common, receiver, (token, token))
    other = resources(catalogue_cap=4)
    mixed = WaveformToken.create(0.5, 0.5, other, receiver)
    with pytest.raises(ValueError):
        TransceiverLibrary(common, receiver, (token, mixed))
    with pytest.raises(ValueError):
        TransceiverLibrary(common, receiver, (token, WaveformToken.create(0.5, 0.5, common, receiver), WaveformToken.create(1.0, 1.0, common, receiver)))
