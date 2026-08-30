import json

import numpy as np
import pytest

from scripts.validate_alpha_beta_adaptation import (
    AdaptiveKernel,
    ExperimentConfig,
    _ser_loss_db,
    apply_perturbation,
    generate_channel,
    run_experiment,
)


def test_channel_generation_is_deterministic_and_shaped():
    config = ExperimentConfig(sample_count=1)
    diag_a = generate_channel(config, "diag", 0)
    diag_b = generate_channel(config, "diag", 0)
    full_a = generate_channel(config, "full", 0)
    full_b = generate_channel(config, "full", 0)
    assert diag_a.shape == (4, 4)
    assert full_a.shape == (16, 16)
    assert np.array_equal(diag_a, diag_b)
    assert np.array_equal(full_a, full_b)
    assert np.all(np.isfinite(full_a.real)) and np.all(np.isfinite(full_a.imag))


def test_global_scale_preserves_reference_ranking_for_diagonal_sample():
    config = ExperimentConfig(
        sample_count=1,
        channel_kinds=("diag",),
        perturbations=("identity", "global_scale"),
    )
    report = run_experiment(config)
    identity = report["samples"][0]
    scaled = report["samples"][1]
    assert identity["perturbation"] == "identity"
    assert scaled["perturbation"] == "global_scale"
    assert abs(identity["reference_ser"] - scaled["reference_ser"]) <= 1e-12
    assert report["summary"]["global_scale_invariance"]["ser_equivalent_rate"] == 1.0


def test_report_contains_all_groups_and_is_json_serializable():
    config = ExperimentConfig(sample_count=1)
    report = run_experiment(config)
    assert "diag/identity" in report["groups"]
    assert "full/time_selective_gain" in report["groups"]
    assert report["summary"]["reference_candidate_count"] == 1681
    json.dumps(report)


def test_numerical_report_is_reproducible_excluding_runtime():
    config = ExperimentConfig(
        sample_count=1,
        channel_kinds=("diag", "full"),
        perturbations=("identity", "global_scale"),
    )
    first = run_experiment(config)
    second = run_experiment(config)
    assert first["summary"]["deterministic_digest"] == second["summary"]["deterministic_digest"]


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="fine_step"):
        ExperimentConfig(coarse_step=0.1, fine_step=0.2).validate()
    with pytest.raises(ValueError, match="global scale"):
        ExperimentConfig(global_scale_real=0.0, global_scale_imag=0.0).validate()


def test_ser_loss_floor_is_finite():
    assert np.isfinite(_ser_loss_db(0.0, 0.0))
    assert _ser_loss_db(1.0, 0.1) > 0.0


def test_selective_perturbation_metadata_and_shape():
    config = ExperimentConfig()
    kernel = AdaptiveKernel()
    for kind in ("diag", "full"):
        channel = generate_channel(config, kind, 0)
        for perturbation in ("frequency_selective_gain", "time_selective_gain"):
            changed, noise, metadata = apply_perturbation(
                config, channel, kind, perturbation, config.noise_var
            )
            assert changed.shape == channel.shape
            assert noise == config.noise_var
            assert metadata["gain_min"] == config.selective_gain_min
            assert metadata["gain_max"] == config.selective_gain_max
