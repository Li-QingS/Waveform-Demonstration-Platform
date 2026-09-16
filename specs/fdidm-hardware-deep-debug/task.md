# FDIDM Hardware Deep Debug Tasks

## File list

| Operation | File | Scope |
|---|---|---|
| Modify | `waveform_sim/hardware/fdidm_hardtest.py` | stream continuity, sync/CFO, CSI/equalizer gates, evidence |
| Modify | `waveform_sim/hardware/fdidm_adaptive.py` | conservative validation and rollback evidence |
| Modify | `waveform_sim/hardware/evidence.py` | shared measured metrics and classifications |
| Modify | `scripts/diagnose_b210_rf_ports.py` | B210 diagnostic status and thresholds |
| Modify | `waveform_sim/ui/fdidm_hardware_test_tab.py` | measured RF/adaptation presentation and diagnostic action |
| Add/modify | `tests/test_fdidm_hardware_deep_debug.py` | deterministic regression coverage |

## Ordered tasks

### T1: Receive metadata and synchronization

Track monotonic probe/block counters, reject unproven continuity, expand
candidate handling beyond three peaks, and add robust preamble/CFO acceptance
gates. Verify with startup garbage, alias, weak-score, and multi-frame fixtures.

### T2: Frame metrics and equalization safety

Commit CSI only for a complete accepted frame, separate data-aided and
decision-directed EVM, preserve exact counts, and guard ill-conditioned full-H
solves. Verify finite metrics and unchanged cache after rejected candidates.

### T3: Adaptive validation

Ensure baseline/candidate share waveform power and context, drop invalid frames,
and expose explicit terminal outcome/reason. Verify insufficient evidence,
power mismatch, regression, rollback, and accepted cases.

### T4: B210 diagnostic classification

Add low-power thresholds and status classification without automatic gain
escalation. Verify classification using synthetic rows and import without UHD.

### T5: UI and integration

Wire asynchronous diagnostic output and distinguish measured/predicted/injected
fields. Verify offscreen layouts and full import/test suites.

## Execution order

`T1 -> T2 -> T3`; `T4` may proceed in parallel; `T5` follows the backend
contracts.
