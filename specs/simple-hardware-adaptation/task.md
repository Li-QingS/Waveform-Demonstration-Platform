# Simple Hardware Adaptation Tasks

## File List

| Operation | File | Responsibility |
|---|---|---|
| Modify | `waveform_sim/hardware/fdidm_adaptive.py` | Five-point local search and cached transforms |
| Modify | `waveform_sim/hardware/fdidm_hardtest.py` | CSI/CFO tracking, power normalization, live transition, measured SER |
| Modify | `waveform_sim/ui/fdidm_hardware_test_tab.py` | Preserve history and show compact adaptive status |
| New | `tests/test_simple_hardware_adaptation.py` | Deterministic algorithm and runtime-path tests |

## T1: Replace Global Search With Five-Point Local Search

**Files:** `waveform_sim/hardware/fdidm_adaptive.py`, `tests/test_simple_hardware_adaptation.py`

1. Add candidate generation for the current point and bounded alpha/beta
   neighbours.
2. Add cached diagonal Gamma power matrices keyed by dimension, axis, and step.
3. Evaluate no more than five unique candidates per snapshot.
4. Return active step, per-axis spans, observability flags, and one-step
   recommendation.
5. Preserve stale snapshot protection and existing minimum-improvement gates.

**Verification:** unit tests assert candidate count, one-axis movement, bound
handling, flat-axis locking, and finite SER output.

## T2: Add Cross-Frame CSI/CFO Tracking

**Files:** `waveform_sim/hardware/fdidm_hardtest.py`, `tests/test_simple_hardware_adaptation.py`

1. Initialize tracking fields and reset them on run/context changes.
2. Smooth accepted CFO and diagonal pilot CSI with a fixed default IIR weight.
3. Use smoothed time-average CSI for the equalizer.
4. Pass the smoothed time-resolved CSI to the adaptive snapshot.
5. Keep full synchronization as the fallback when tracking is invalid.

**Verification:** tests inject noisy static CSI and assert reduced variance;
time-selective CSI remains time-varying; lock loss selects acquisition mode.

## T3: Make Waveform Power and Live Updates Deterministic

**Files:** `waveform_sim/hardware/fdidm_hardtest.py`, `tests/test_simple_hardware_adaptation.py`

1. RMS-normalize alpha/beta data while preserving the existing peak safety
   limit.
2. Record whether peak limiting was applied in the fingerprint/status.
3. Treat alpha/beta, modulation, coding, text, and equalizer changes as
   live-safe whenever the graph geometry is unchanged.
4. Replace the vector source in place and mark a two-frame transition window.
5. Do not call the normal startup settle delay for live-safe waveform swaps.
6. Keep UI measurement history for live-safe updates.

**Verification:** mocked backend tests assert no stop/start on alpha/beta update,
vector replacement is called once, transition state expires after valid frames,
and structural changes still request restart.

## T4: Add Measured SER and Compact Diagnostics

**Files:** `waveform_sim/hardware/fdidm_hardtest.py`,
`waveform_sim/ui/fdidm_hardware_test_tab.py`

1. Compute uncoded symbol error rate from the known transmitted coded symbols
   and recovered soft symbols.
2. Publish measured SER separately from model-predicted SER, raw BER, FEC BER,
   and EVM.
3. Include active search step, axis observability, and transition state in the
   status snapshot.
4. Keep logging rate-limited and include overflow state when provided by UHD.

**Verification:** known-symbol unit tests recover exact SER; UI/status tests
verify fields are distinguishable and finite values are formatted correctly.

## T5: Integrate and Regress

**Files:** existing focused test files plus
`tests/test_simple_hardware_adaptation.py`

1. Run focused adaptive, hardware, log-export, and new simple-adaptation tests.
2. Run the full test suite without GNU Radio/UHD where supported.
3. Confirm existing offline validation output remains unchanged.
4. Inspect a synthetic runtime log for five-candidate evaluations and no fixed
   800 ms delay after live alpha/beta changes.

**Verification:** pytest results and a captured diagnostic summary satisfy the
   checklist acceptance criteria.

## Execution Order

```text
T1 -> T2 -> T3 -> T4 -> T5
```
