import numpy as np
import threading

from scripts.validate_alpha_beta_adaptation import AdaptiveKernel
from waveform_sim.hardware.evidence import TxPowerContract, TxPowerMetrics
from waveform_sim.hardware.fdidm_adaptive import FDIDMAdaptiveMixin


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


class ValidationHarness(FDIDMAdaptiveMixin):
    ALPHA_BETA_SIGNALING_MODE = "shared_memory"

    def __init__(self):
        self.alpha = 0.5
        self.beta = 1.0
        self.M = self.N = 16
        self.cp_len = 4
        self.mod_order = "QPSK"
        self.equalizer = "MMSE"
        self.channel_estimator = "diag_tf"
        self.channel_mode = "rf"
        self.sample_rate = 1e6
        self.carrier_freq = 2.4e9
        self.tx_gain = self.rx_gain = 20.0
        self.training_amplitude = 1.0
        self.tdl_rms_delay_spread_ns = 1000.0
        self.tdl_doppler_hz = self.tdl_doppler_spread_hz = 0.0
        self.tdl_snr_db = 35.0
        self.adaptive_alpha_beta_enable = True
        self.adaptive_alpha_beta_min_improvement_db = 0.5
        self.adaptive_alpha_beta_stability_evals = 2
        self.adaptive_alpha_beta_coarse_step = 0.25
        self.adaptive_alpha_beta_fine_step = 0.05
        self.adaptive_alpha_beta_interval_frames = 8
        self.adaptive_alpha_beta_cooldown_frames = 16
        self.adaptive_alpha_beta_integer_margin_db = 0.1
        self.adaptive_alpha_beta_max_order = 512
        self._frames_processed = 0
        self._adaptive_ab_lock = threading.RLock()
        self._adaptive_ab_state = "idle"
        self._adaptive_ab_recommendation = {}
        self._adaptive_ab_validation = {"state": "idle"}
        self._adaptive_ab_snapshot_seq = 0
        self._adaptive_ab_snapshot = None
        self._adaptive_ab_last_snapshot = None
        self._adaptive_ab_stable_key = None
        self._adaptive_ab_stable_count = 0
        self._adaptive_ab_last_htf_identity = None
        self._adaptive_ab_last_error = ""
        self._adaptive_ab_last_queued_frame = -10**18
        self._adaptive_ab_last_applied_frame = -10**18
        self._adaptive_ab_last_skip_reason = ""
        self._adaptive_ab_last_skip_log_wall = 0.0
        self._validation_synchronous = True
        self._running = False
        self._tx_waveform = np.ones(8, dtype=np.complex64)
        self._tx_power_metrics = TxPowerMetrics(cycle_rms=0.2, contract_id="")
        self.commits = []
        self.messages = []

    def _prepare_power_contract(self, alpha, beta):
        return TxPowerContract(
            baseline_alpha=self.alpha,
            baseline_beta=self.beta,
            candidate_alpha=float(alpha),
            candidate_beta=float(beta),
            requested_rms=0.25,
            locked_rms=0.2,
            peak_limit=0.9,
            tolerance_db=0.1,
            contract_id="test-contract",
        )

    def _commit_waveform_build(self, alpha, beta, forced_rms=None, contract_id=""):
        self.alpha = float(alpha)
        self.beta = float(beta)
        actual_rms = 0.25 if forced_rms is None else float(forced_rms)
        self._tx_power_metrics = TxPowerMetrics(
            cycle_rms=actual_rms,
            target_rms=actual_rms,
            peak=0.8,
            contract_id=contract_id,
        )
        self.commits.append((self.alpha, self.beta, forced_rms, contract_id))
        return object()

    def _alpha_beta_adaptation_context_key(self):
        return ("stable-link",)

    def _debug(self, level, message):
        self.messages.append((level, message))


def _validation_sample(harness, errors, symbols=200, evm=10.0, **overrides):
    data = {
        "ser_errors": errors,
        "ser_symbols": symbols,
        "data_aided_evm_percent": evm,
        "decision_directed_evm_percent": evm,
        "raw_bit_ber": errors / max(symbols, 1),
        "fec_bit_ber": 0.0,
        "decode_ok": True,
        "sync_valid": True,
        "overflow": False,
        "tx_power_contract_id": harness._tx_power_metrics.contract_id,
        "tx_cycle_rms": harness._tx_power_metrics.cycle_rms,
        "context_key": harness._alpha_beta_adaptation_context_key(),
    }
    data.update(overrides)
    harness._record_alpha_beta_validation_sample_locked(data)


def _settle(harness):
    for _ in range(3):
        _validation_sample(harness, 0)


def _collect(harness, count, errors, evm=10.0):
    for _ in range(count):
        _validation_sample(harness, errors, evm=evm)


def test_validation_state_collects_common_power_baseline_after_settling():
    harness = ValidationHarness()
    status = harness.apply_alpha_beta_candidate(0.75, 0.8, predicted_improvement_db=0.67)
    assert status["state"] == "baseline_settling"
    assert harness.commits[0] == (0.5, 1.0, 0.2, "test-contract")
    _settle(harness)
    assert harness._adaptive_ab_validation["state"] == "baseline_collecting"
    _collect(harness, 23, errors=20)
    assert harness._adaptive_ab_validation["state"] == "baseline_collecting"
    assert harness._adaptive_ab_validation["baseline_window"].valid_frames == 23
    _collect(harness, 1, errors=20)
    assert harness._adaptive_ab_validation["state"] == "candidate_settling"
    assert harness.commits[1] == (0.75, 0.8, 0.2, "test-contract")


def test_validation_improved_keeps_candidate_after_exact_windows():
    harness = ValidationHarness()
    harness.apply_alpha_beta_candidate(0.75, 0.8)
    _settle(harness)
    _collect(harness, 24, errors=20, evm=12.0)
    _settle(harness)
    _collect(harness, 50, errors=2, evm=8.0)
    validation = harness._adaptive_ab_validation
    assert validation["state"] == "improved"
    assert validation["outcome"] == "improved"
    assert harness.alpha == 0.75 and harness.beta == 0.8
    assert not validation["rollback_pending"]


def test_validation_inconclusive_at_max_frames_rolls_back():
    harness = ValidationHarness()
    harness.apply_alpha_beta_candidate(0.75, 0.8)
    _settle(harness)
    _collect(harness, 24, errors=20, evm=10.0)
    _settle(harness)
    _collect(harness, 64, errors=20, evm=10.0)
    validation = harness._adaptive_ab_validation
    assert validation["state"] == "inconclusive"
    assert validation["rollback_complete"]
    assert harness.alpha == 0.5 and harness.beta == 1.0
    assert "overlap" in validation["result_reason"]


def test_validation_regressed_rolls_back():
    harness = ValidationHarness()
    harness.apply_alpha_beta_candidate(0.75, 0.8)
    _settle(harness)
    _collect(harness, 24, errors=10, evm=8.0)
    _settle(harness)
    _collect(harness, 24, errors=40, evm=12.0)
    validation = harness._adaptive_ab_validation
    assert validation["state"] == "regressed"
    assert validation["rollback_complete"]
    assert harness.alpha == 0.5 and harness.beta == 1.0


def test_validation_drops_overflow_and_sync_samples():
    harness = ValidationHarness()
    harness.apply_alpha_beta_candidate(0.75, 0.8)
    _settle(harness)
    _validation_sample(harness, 10, overflow=True)
    _validation_sample(harness, 10, sync_valid=False)
    window = harness._adaptive_ab_validation["baseline_window"]
    assert window.valid_frames == 0
    assert window.dropped_overflow == 1
    assert window.dropped_sync == 1


def test_validation_context_or_power_change_is_inconclusive_and_rolls_back():
    context_harness = ValidationHarness()
    context_harness.apply_alpha_beta_candidate(0.75, 0.8)
    _settle(context_harness)
    _validation_sample(context_harness, 10, context_key=("changed",))
    assert context_harness._adaptive_ab_validation["state"] == "inconclusive"
    assert context_harness._adaptive_ab_validation["baseline_window"].dropped_context == 1

    power_harness = ValidationHarness()
    power_harness.apply_alpha_beta_candidate(0.75, 0.8)
    _settle(power_harness)
    _validation_sample(power_harness, 10, tx_cycle_rms=0.15)
    assert power_harness._adaptive_ab_validation["state"] == "inconclusive"
    assert power_harness._adaptive_ab_validation["baseline_window"].dropped_power == 1


def test_validation_invalidation_cancels_old_generation():
    harness = ValidationHarness()
    harness.apply_alpha_beta_candidate(0.75, 0.8)
    generation = harness._adaptive_ab_validation["apply_generation"]
    harness._invalidate_alpha_beta_adaptation("hardware_stop")
    assert harness._adaptive_ab_validation["state"] == "inconclusive"
    assert harness._adaptive_ab_validation["apply_generation"] > generation
    assert "hardware_stop" in harness._adaptive_ab_validation["result_reason"]


def test_validation_status_is_public_and_contains_window_progress():
    harness = ValidationHarness()
    harness.apply_alpha_beta_candidate(0.75, 0.8)
    _settle(harness)
    _collect(harness, 4, errors=10)
    status = harness.get_alpha_beta_adaptation_status()
    assert status["validation_state"] == "baseline_collecting"
    assert status["validation"]["baseline_window"]["valid_frames"] == 4
    assert status["validation"]["contract"]["contract_id"] == "test-contract"
