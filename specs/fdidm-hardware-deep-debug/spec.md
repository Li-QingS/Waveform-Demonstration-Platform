# FDIDM Hardware Deep Debug Spec

## Background

The connected USRP B210 is discoverable, but the current FDIDM RF run does
not produce a reliable link. The observed baseline is approximately
`RX RMS=1.8e-4`, pilot fit `NMSE=0.94`, pilot residual `SINR=-12 dB`, SER near
`0.7`, EVM in the hundreds or thousands of percent, and zero CRC successes.
There are no RX overflows, but processing/probe gaps are present. This means
the next work must separate RF path and level faults from timing/CFO,
channel-estimation, equalization, and adaptive-control faults. A green UI
label or a model-injected SNR must never be treated as hardware evidence.

## Goals

- Make FDIDM frame acquisition and demodulation robust enough to distinguish a
  real frame from noise, startup transients, a stale probe window, or a wrong
  RF connection.
- Reduce measured EVM through trustworthy timing, CFO, pilot, CSI, noise-load,
  and equalization decisions while preserving the paper FDIDM transform.
- Make alpha/beta adaptation conservative and evidence-driven: search from
  measured CSI, apply candidates only after a stable validation window, and
  keep or roll back based on measured SER/EVM/CRC under a shared power contract.
- Provide a hardware diagnostic path for B210 antenna/frontend, gain, sample
  rate, and loopback level, with explicit safety limits and actionable failure
  reasons.
- Keep the FDIDM hardware page readable at common desktop sizes. It must show
  current link health, real-vs-predicted metrics, adaptation state, and RF
  diagnostics without squeezing plots or hiding long status text.

## Functional Requirements

- F1: At startup and after every reconfigure, discard configured settle windows
  and reject probe vectors whose absolute sample continuity cannot be proven.
  Report overflow, processing-gap, dropped-sync, and stale-window counts.
- F2: Perform coarse repeated-half timing/CFO detection followed by
  CFO-compensated full-preamble refinement. Do not accept a frame or update CSI
  when the refined preamble score is below the configured reliability gate.
  Preserve a last-good CFO only as a bounded prior, never as a hard override.
- F3: Validate frame boundaries against the configured sync, pilot, data, CP,
  and inter-frame guard lengths. A candidate must have a complete pilot and
  data region before it can affect any metric, cache, or adaptation state.
- F4: Estimate pilot CSI from the current physical frame, expose pilot fit
  NMSE and residual SINR, and keep the current-frame CSI isolated from rejected
  sync candidates. Any optional smoothing must be phase-safe and must not
  suppress the current-frame noise estimate used by MMSE loading.
- F5: Use the selected equalizer with an explicit channel-source label
  (`diag_tf`, `full_htf`, or `tdl_param`) and guard against invalid, singular,
  or ill-conditioned matrices. Equalizer warnings must identify model mismatch
  rather than silently presenting a good constellation.
- F6: Compute data-aided EVM from the known transmitted symbol sequence and
  decision-directed EVM separately. Compute exact measured SER errors and total
  symbols, raw BER, FEC BER, and CRC success; never infer exact counts from a
  rounded rate or from a random filler frame.
- F7: Keep TX waveform RMS, peak, PAPR, backoff, and contract ID identical for
  baseline and candidate windows. If actual power, context, sample count, or
  chain health is not comparable, classify the result as `inconclusive`.
- F8: Implement adaptive alpha/beta as a state machine with explicit
  `disabled`, `searching`, `candidate_applying`, `settling`, `validating`,
  `accepted`, `rollback`, and `inconclusive` states. Predicted improvement may
  trigger a candidate but may not itself produce an `improved` conclusion.
- F9: Ignore frames with overflow, processing gaps, weak sync, missing pilot,
  invalid CSI, or context changes for validation windows, while continuing to
  consume the stream and recording the exclusion reason.
- F10: Add B210 RF diagnostics that can test the configured TX frontend/RX
  antenna and gain range at low safe power before a long FDIDM run. The result
  must identify no-signal, clipping, wrong-port, and excessive-noise cases.
- F11: Make the page show real measured EVM/SER/BER/CRC, injected software-TDL
  settings, predicted alpha/beta objective values, and residual pilot metrics
  under distinct labels. Do not label injected SNR or predicted SER as measured
  RF performance.
- F12: Preserve the existing FDIDM hardware API and offline importability. New
  diagnostics and metrics must not block the UHD consumer or Qt main thread.

## Non-Functional Requirements

- N1: Keep RF output within the existing peak and RMS safety limits. No new
  automatic gain escalation may bypass the configured safety ceiling.
- N2: Do not add a second USRP, a second parallel RF chain, or a baseband-only
  success path to the hardware claim.
- N3: Keep the transform, modulation, FEC, and existing estimator contracts
  backward compatible unless a test demonstrates a correctness defect.
- N4: All new calculations must be bounded in memory and tolerant of missing
  UHD/GNU Radio/PyQt for offline tests.
- N5: Preserve user changes already in the worktree; modify only FDIDM hardware
  backend, diagnostics, adaptation, related UI, tests, and this spec workflow.
- N6: Use ASCII in new files where practical and keep exported JSON sufficient
  to reproduce every hardware conclusion.

## Out Of Scope

- Replacing FDIDM with OFDM/OTFS or changing the paper transform definition.
- Claiming lower hardware EVM from software TDL SNR, predicted SER, or a
  display-only constellation normalization.
- Increasing pilot length or adding a full MN probing sequence as a substitute
  for fixing RF level, timing, or estimator validity.
- Automatic timed A/B switching, multi-device synchronization, or a new RF
  calibration subsystem.
- Broad refactors of unrelated waveform pages or simulation behavior.

## Acceptance Criteria

- AC1: With a deterministic offline capture containing a known frame, the
  receiver finds the correct frame start, resolves a bounded CFO, decodes the
  payload, and produces finite data-aided EVM, exact SER `k/n`, and CRC fields.
- AC2: With noise, startup garbage, wrong timing, or an incomplete probe window,
  the receiver does not commit CSI, does not count the frame as valid, and
  reports a specific exclusion reason while continuing to run.
- AC3: On a clean software loopback regression, measured SER is below `1e-2`
  and data-aided EVM is below `10%`; no rejected sync candidate changes the
  cross-frame CSI cache.
- AC4: On the connected B210, the diagnostic run records device identity,
  antenna/frontend, gain, sample rate, raw IQ RMS, clipping, sync score, pilot
  NMSE/SINR, gaps, and CRC. If pilot fit remains above `0.8` NMSE or raw RMS
  remains below the configured detect floor, the page reports RF/link failure
  rather than claiming an algorithmic improvement.
- AC5: A candidate alpha/beta change enters validation only after a stable
  baseline and shared power contract. Fewer than 24 valid frames per side,
  any overflow/gap/context mismatch, or overlapping confidence intervals yields
  `inconclusive`; a statistically supported regression rolls back.
- AC6: A statistically supported measured improvement requires exact SER
  counts, 95% confidence separation or a supported EVM improvement, no CRC/EVM
  regression, and a common power contract. Predicted improvement alone can
  never yield `improved`.
- AC7: Repeated runs with the same offline capture and seed are deterministic;
  all existing tests remain green and new hardware paths remain importable
  without UHD or a display.
- AC8: At 1400x800, 1400x900, and 1200x780 offscreen sizes, the hardware page
  keeps equal 2x2 plot cells, readable result numbers, a visible adaptation/RF
  state, and no overlapping or clipped long status text.
- AC9: A hardware report exports enough fields to independently distinguish
  RF failure, synchronization failure, estimator mismatch, power
  incomparability, insufficient evidence, accepted improvement, and rollback.
