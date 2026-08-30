import numpy as np

from scripts.validate_alpha_beta_adaptation import AdaptiveKernel


def _snapshot(htf, alpha=1.0, beta=1.0):
    return {
        "M": 4,
        "N": 4,
        "htf": np.asarray(htf, dtype=np.complex128),
        "htf_kind": "diag",
        "noise_var": 0.02,
        "equalizer": "MMSE",
        "alpha": alpha,
        "beta": beta,
        "mod_order": "QPSK",
        "coarse_step": 0.25,
        "fine_step": 0.05,
        "min_improvement_db": 0.0,
        "max_order": 512,
    }


def test_local_search_is_bounded_and_moves_one_axis():
    kernel = AdaptiveKernel()
    kernel._adaptive_active_step = 0.25
    h = np.ones((4, 4), dtype=np.complex128)
    result = kernel._optimize_alpha_beta_snapshot(_snapshot(h, alpha=0.75, beta=0.75))
    assert result["candidate_count"] <= 5
    da = abs(result["recommended_alpha"] - 0.75)
    db = abs(result["recommended_beta"] - 0.75)
    assert da <= 0.25 + 1e-9
    assert db <= 0.25 + 1e-9
    assert not (da > 1e-9 and db > 1e-9)


def test_time_invariant_channel_locks_flat_beta_axis():
    kernel = AdaptiveKernel()
    kernel._adaptive_active_step = 0.25
    result = kernel._optimize_alpha_beta_snapshot(_snapshot(np.ones((4, 4), complex)))
    assert result["beta_observable"] is False
    assert result["recommended_beta"] == 1.0


def test_time_selective_channel_exposes_beta_axis():
    kernel = AdaptiveKernel()
    kernel._adaptive_active_step = 0.25
    h = np.ones((4, 4), dtype=np.complex128)
    h[:, 0] *= 0.2
    h[:, 1] *= 0.8
    h[:, 2] *= 1.4
    h[:, 3] *= 2.0
    result = kernel._optimize_alpha_beta_snapshot(_snapshot(h))
    assert result["candidate_count"] <= 5
    assert result["beta_observable"] is True


def test_failed_coarse_move_arms_fine_step():
    kernel = AdaptiveKernel()
    kernel._adaptive_active_step = 0.25
    result = kernel._optimize_alpha_beta_snapshot(_snapshot(np.ones((4, 4), complex)))
    assert result["active_step"] == 0.25
    assert result["next_active_step"] == 0.05
