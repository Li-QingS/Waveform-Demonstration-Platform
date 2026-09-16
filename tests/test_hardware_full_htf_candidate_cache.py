"""Offline regressions for full-H sync-candidate cache isolation."""

import numpy as np
import pytest

from waveform_sim.hardware.fdidm_hardtest import _LegacyFDIDMHardwareTest


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(_LegacyFDIDMHardwareTest, "_import_runtime", lambda self: None)
    monkeypatch.setattr(_LegacyFDIDMHardwareTest, "_build_top_block", lambda self: None)
    return _LegacyFDIDMHardwareTest(
        channel_estimator="full_htf",
        full_htf_once=False,
        full_htf_update_interval_frames=1,
        fdidm_m=4,
        fdidm_n=8,
        mod_order="64QAM",
        coding_scheme="none",
        tx_text="x",
        tx_min_waveform_duration_ms=0,
        tx_max_waveform_samples=65536,
    )


def _run_two_candidates(backend, monkeypatch, *, second_decodes):
    stride = backend.frame_len + backend.inter_frame_guard_len
    rx = backend._tx_waveform[:2 * stride].astype(np.complex128).copy()
    peaks = [backend.pre_guard_len, stride + backend.pre_guard_len]
    metric = np.zeros(rx.size, dtype=np.float64)
    metric[peaks] = [0.9, 0.8]
    monkeypatch.setattr(backend, "_sync_metric", lambda samples: metric)
    monkeypatch.setattr(backend, "_find_sync_peaks", lambda *args, **kwargs: peaks)
    monkeypatch.setattr(
        backend, "_refine_sync_and_cfo",
        lambda samples, coarse, search_radius: (coarse, 0.0, 1.0, 0.0),
    )

    order = backend.full_htf_order
    estimates = []

    def estimate(pilot_samples):
        scale = len(estimates) + 1
        estimates.append(scale)
        return scale * np.eye(order, dtype=np.complex128), 0.0

    monkeypatch.setattr(backend, "_estimate_htf_full_from_pilot", estimate)
    prior_htf = backend._cached_htf_full
    prior_count = backend._full_htf_estimates
    equalized = []
    real_equalize = backend._equalize_data_full_htf

    def equalize(y_tf, h_tf, noise_var, *, candidate_cache=None):
        # No candidate may publish into the cross-frame cache while sync peaks
        # are being compared, even when a previous accepted cache exists.
        assert backend._cached_htf_full is prior_htf
        assert backend._full_htf_estimates == prior_count
        assert candidate_cache is not None
        np.testing.assert_array_equal(h_tf, candidate_cache["htf"])
        equalized.append(float(h_tf[0, 0].real))
        return real_equalize(y_tf, h_tf, noise_var, candidate_cache=candidate_cache)

    monkeypatch.setattr(backend, "_equalize_data_full_htf", equalize)
    recovered = []

    def recover(rx_syms):
        candidate = len(recovered) + 1
        recovered.append(candidate)
        good = bool(second_decodes and candidate == 2)
        return (
            0.0 if good else 0.5,
            b"", b"x" if good else b"", "x" if good else "",
            1 if good else 0, good,
            np.asarray(rx_syms, dtype=np.complex64).copy(),
            5.0 if good else 90.0,
        )

    monkeypatch.setattr(backend, "_recover_payload_from_symbols", recover)
    backend._try_process_rx_window(rx, rx.size)
    return estimates, equalized, recovered


def test_second_sync_candidate_commits_only_its_own_full_htf(backend, monkeypatch):
    # A previous CSI is eligible for periodic refresh, but remains visible
    # throughout both candidate trials until the second candidate wins.
    backend._cache_full_htf(3 * np.eye(backend.full_htf_order), 0.3)
    backend._frames_decode_ok = 1
    baseline_count = backend._full_htf_estimates

    estimates, equalized, recovered = _run_two_candidates(
        backend, monkeypatch, second_decodes=True
    )

    assert estimates == equalized == recovered == [1, 2]
    assert backend._frames_decode_ok == 2
    assert backend._full_htf_estimates == baseline_count + 1
    np.testing.assert_array_equal(
        backend._cached_htf_full, 2 * np.eye(backend.full_htf_order)
    )
    assert backend._cached_htf_frame_counter == 2


@pytest.mark.parametrize("with_prior_cache", [False, True])
def test_failed_sync_candidates_leave_full_htf_cache_unchanged(
    backend, monkeypatch, with_prior_cache
):
    if with_prior_cache:
        backend._cache_full_htf(3 * np.eye(backend.full_htf_order), 0.3)
        backend._frames_decode_ok = 1
    before = backend._snapshot_channel_cache()

    estimates, equalized, recovered = _run_two_candidates(
        backend, monkeypatch, second_decodes=False
    )

    assert estimates == equalized == recovered == [1, 2]
    after = backend._snapshot_channel_cache()
    assert before.keys() == after.keys()
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(after[key], value)
        elif isinstance(value, float) and np.isnan(value):
            assert np.isnan(after[key])
        else:
            assert after[key] == value
