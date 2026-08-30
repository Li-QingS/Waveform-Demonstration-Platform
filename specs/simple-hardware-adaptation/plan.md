# Simple Hardware Adaptation Plan

## Architecture Overview

The implementation stays inside the existing hardware mixin and backend. The
receive monitor remains the only producer of valid frames; the adaptive worker
consumes a bounded snapshot and never blocks the monitor. Runtime state is
reduced to a lock state, smoothed CFO/CSI, an active search step, and one pending
waveform version.

```text
RX frame
  -> acquisition or local tracking
  -> raw pilot CSI H_frame[m,n]
  -> cross-frame IIR
      -> H_eq[m] = mean_n(H_smooth[m,n]) -> MMSE equalizer
      -> H_adapt[m,n] = H_smooth[m,n] -> five-point SER worker
  -> recommendation (one axis, one step)
  -> in-place vector-source swap
```

## Core Data Structures

### ReceiverTrackState

Fields: `locked`, `last_sync_start`, `cfo_hz`, `csi_diag`, `valid_frames`,
`transition_until`, and `transition_windows_remaining`. The state is reset on a
structural change or an explicit loss-of-lock decision.

### AdaptiveSearchState

Fields: `step`, `last_direction`, `stable_count`, `axis_observability`, and
`last_context_key`. A candidate is `(current, alpha-neighbour, beta-neighbour)`
with duplicate points removed at the bounds. The result includes the five SERs,
selected direction, and active step.

### WaveformVersion

Fields: monotonically increasing `version`, waveform fingerprint, alpha, beta,
and creation timestamp. The version is attached to the RX transition state so
old frames cannot update the new metric state.

## Module Design

### `waveform_sim/hardware/fdidm_hardtest.py`

Responsibilities:

- Maintain acquisition/tracking state in the monitor path.
- IIR smooth CFO and diagonal CSI after valid frames.
- Pass averaged CSI to the equalizer and the full time-resolved grid to the
  adaptive worker.
- Normalize each alpha/beta waveform by RMS power, then apply the existing peak
  limit and report whether peak limiting occurred.
- Replace alpha/beta waveform data using `vector_source.set_data()` while UHD is
  running. Structural changes retain the current controlled restart behavior.
- During a live swap, invalidate only the RX transition window; leave UI metric
  history intact and resume after two valid frames or the first valid frame after
  the new waveform version is observed.
- Compute measured uncoded SER from transmitted and recovered symbols using the
  existing known-symbol path, and expose overflow counters if available.

### `waveform_sim/hardware/fdidm_adaptive.py`

Responsibilities:

- Cache Gamma power matrices for `(M, step, axis)`.
- Evaluate exactly the current pair and its four axis neighbours at the active
  step. Use the existing paper SER objective and diagonal fast path.
- Mark alpha or beta unobservable when its two neighbour SERs have span below a
  relative/absolute tolerance derived from the current SER and noise floor.
- Select the best observable neighbour only when it beats the current point by
  the configured margin. Move at most one step and change from coarse to fine
  only after a coarse evaluation has no meaningful move.
- Preserve the existing asynchronous snapshot sequence and stale-result guard.

### `waveform_sim/ui/fdidm_hardware_test_tab.py`

Responsibilities:

- Keep the current apply/debounce UI contract.
- Display active step, selected direction, and measured-vs-predicted diagnostics
  when present.
- Avoid clearing plotted metric history for live-safe changes.

## Module Interaction

1. `_try_process_rx_window_impl` obtains a frame through acquisition or local
   tracking.
2. A valid pilot produces `H_frame`; the backend updates `H_smooth` once.
3. `H_eq` is passed to `_equalize_data_diag`; `H_adapt` is copied into the
   asynchronous snapshot.
4. The worker computes five SER values and publishes a one-step recommendation.
5. UI or the adaptive auto-apply path calls `configure(alpha, beta)`.
6. `configure` rebuilds only the waveform and calls `_sync_waveform_to_top_block`
   for live-safe changes; it does not call `stop()` or `start()`.

## Key Technical Decisions

| Decision | Choice | Reason |
|---|---|---|
| CSI smoothing | IIR, default weight 0.2 | One O(MN) pass and no history matrix |
| CFO smoothing | IIR, default weight 0.2 | Suppresses frame-to-frame CFO jumps |
| Search | Current + four neighbours | Bounded cost and one-step continuity |
| Candidate transform | Diagonal power fast path | Avoids MN x MN allocation |
| Equalizer CSI | Mean across time of smoothed CSI | Stable static-channel equalization |
| Adaptive CSI | Time-resolved smoothed CSI | Keeps beta observable when variation exists |
| Live transition | Two valid frames, no fixed 800 ms | Short deterministic interruption |
| Power normalization | RMS first, peak clamp second | Fair alpha/beta comparison |
| Structural changes | Existing controlled restart | Graph geometry cannot be changed in place |

## Complexity

For `K=M*N`, one frame adds `O(K)` CSI smoothing. One adaptive evaluation
performs at most five diagonal candidates, using cached Gamma-derived matrices;
it allocates no `K x K` matrix. The worker remains asynchronous and can be
discarded when a newer snapshot arrives.

## File Organization

```text
waveform_sim/hardware/fdidm_adaptive.py   # five-point optimizer
waveform_sim/hardware/fdidm_hardtest.py   # tracking, CSI, hot swap, metrics
waveform_sim/ui/fdidm_hardware_test_tab.py # status presentation only
tests/test_simple_hardware_adaptation.py  # deterministic unit tests
```
