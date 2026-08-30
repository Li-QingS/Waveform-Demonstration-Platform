# Simple Hardware Adaptation Spec

## Background

The current FDIDM hardware path has three coupled problems. Receiver quality can
change sharply after a restart even when alpha and beta are unchanged; waveform
updates cause a visible quiet period; and the adaptive controller performs a
large coarse-to-fine two-dimensional search using single-frame CSI whose time
axis has been averaged away.

The replacement must be a smaller design rather than another layer around the
existing controller: smooth the CSI already produced by the receiver, evaluate
only the current point and its four axis neighbours, move by one step, and apply
the waveform without restarting UHD.

## Goals

- Improve receiver stability by tracking CFO and channel state across frames.
- Make alpha/beta and other non-structural waveform updates visibly smooth.
- Replace the global coarse/fine search with a low-cost five-point local search.
- Preserve the paper SER objective while supplying it with time-resolved CSI.
- Keep different alpha/beta candidates on a consistent transmit-power basis.

## Functional Requirements

- F1: The receiver performs full synchronization/CFO acquisition when unlocked,
  then tracks timing, CFO, and diagonal CSI locally across valid frames. Loss of
  lock returns it to full acquisition.
- F2: The equalizer continues to use a frequency response averaged across pilot
  time slots, with lightweight cross-frame smoothing to reduce estimation noise.
- F3: The adaptive controller receives a separately smoothed, time-resolved
  diagonal CSI grid. Time averaging for equalization must not erase beta
  observability in the adaptive input.
- F4: Alpha/beta-only changes use a consistent transmit scale so that waveform
  PAPR changes do not silently change average transmit energy. The implementation
  must retain a safe peak limit.
- F5: Alpha/beta, modulation, coding, equalizer, text, and control-only settings
  are applied without stopping UHD. Structural changes to M, N, CP, sample rate,
  device, or graph topology may use a controlled restart.
- F6: A live waveform update replaces the transmit vector in place, discards only
  stale transition receive windows, and resumes after fresh valid windows rather
  than a fixed 800 ms delay.
- F7: Live updates keep existing UI history. A full structural restart may reset
  history when the metric geometry changes.
- F8: Each adaptive evaluation scores no more than the unique points among the
  current pair and its four axis neighbours at the active step size.
- F9: The active step starts at the configured coarse step. When no coarse
  neighbour gives a meaningful improvement it changes to the configured fine
  step; a channel-context reset returns it to the coarse step.
- F10: An update moves by at most one active step on one axis. Existing minimum
  improvement, repeated-recommendation, interval, and cooldown gates remain the
  switch guards.
- F11: An axis whose neighbour scores are numerically flat is reported as
  unobservable and is not changed.
- F12: Hardware evaluation reports measured uncoded SER in addition to raw BER,
  FEC BER, and EVM by reusing the known transmitted test sequence. This metric is
  diagnostic and does not add a second adaptive search loop.
- F13: UHD overflow observations are represented in exported diagnostics, and a
  receive interval known to contain an overflow is excluded from adaptive input
  when that information is available to the process.

## Non-Functional Requirements

- N1: The steady-state additions are linear in the `M x N` grid size; the new
  adaptive search must not allocate or invert an `MN x MN` matrix in `diag_tf`
  mode.
- N2: Gamma-derived power matrices for configured step values are cached and
  reused across evaluations.
- N3: The hardware receive loop remains non-blocking with respect to adaptive
  search and UI rendering.
- N4: Tests run without GNU Radio, UHD, a connected USRP, or a display unless
  explicitly marked as hardware-only.
- N5: Existing user changes and software simulation behavior outside this
  hardware path remain unchanged.

## Out Of Scope

- Future-channel prediction.
- Continuous hardware A/B exploration or machine learning.
- Exhaustive alpha/beta grid search in the live hardware controller.
- Automatic RF gain control.
- Removing the full-H or parametric TDL receivers; this change targets the
  default hardware `diag_tf` path.
- Promising an absolute EVM value independent of RF SNR, clipping, or sample
  loss.

## Acceptance Criteria

- AC1: A static synthetic channel reaches tracking mode, uses smoothed CSI, and
  returns to acquisition after an injected lock failure.
- AC2: Equalizer CSI remains time-averaged while adaptive CSI preserves an
  injected time variation and produces a non-flat beta direction.
- AC3: A static time-invariant channel reports beta as flat and never changes it
  because of numerical noise alone.
- AC4: One adaptive evaluation considers at most five unique candidates and any
  accepted recommendation differs from the current pair on only one axis by no
  more than the active step.
- AC5: Coarse-to-fine step transition, repeated-recommendation gate, cooldown,
  and context-reset behavior are covered by deterministic tests.
- AC6: Alpha/beta updates while running call the live vector replacement path,
  do not call backend stop/start, and do not arm the normal 800 ms startup delay.
- AC7: A live-safe UI apply preserves metric history; a structural apply follows
  the controlled restart path.
- AC8: Alpha/beta test waveforms have equal average energy within numerical
  tolerance when the safety peak limit is inactive; peak-limited cases are
  explicitly reported rather than silently treated as equal-power comparisons.
- AC9: Measured uncoded SER agrees with known injected symbol errors in unit
  tests and is distinct from the model-predicted SER.
- AC10: Focused hardware/adaptive/logging tests pass without requiring a USRP.
