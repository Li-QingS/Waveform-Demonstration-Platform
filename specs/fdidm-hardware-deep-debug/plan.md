# FDIDM Hardware Deep Debug Plan

## Architecture

The work stays inside the existing FDIDM backend, evidence helpers, B210
diagnostic script, and Qt hardware tab. The receive monitor remains the owner
of stream consumption; decoding and diagnostics publish immutable snapshots for
the UI and adaptive worker.

## Components

1. **RX window and synchronizer**: track monotonic sample/block continuity,
   reject stale or incomplete windows, resolve repeated-half CFO aliases, and
   gate refined preamble acceptance.
2. **Frame evidence**: isolate current-frame CSI, separate pilot fit,
   data-aided EVM, decision-directed EVM, exact SER/BER/CRC, and attach an
   exclusion reason to every rejected candidate.
3. **Adaptive validator**: consume only valid evidence under one TX power and
   context contract; transition through candidate, settling, validation,
   accepted/rollback/inconclusive states.
4. **B210 diagnostics**: run low-power antenna/frontend probes and classify
   pass, no-signal, clipping, wrong-port, excessive-noise, or inconclusive.
5. **Qt presentation**: expose measured, predicted, injected, and residual
   values with distinct labels and keep the 2x2 plot area stable at target
   desktop sizes.

## Data contracts

Each accepted frame carries `sync_valid`, `cfo_hz`, `cfo_source`, pilot fit
metrics, CSI source/condition, exact symbol counts, measured EVM, CRC/FEC
status, stream continuity, and the current power contract ID. A diagnostic row
contains device/port/gain/rate settings, baseline and active RMS/tone values,
thresholds, and a classification status.

## Key decisions

- Hardware defaults use a sufficiently long repeated-half preamble and refresh
  full-H CSI periodically; stale CSI cannot silently serve a changing RF link.
- CFO refinement is applied only when phase consistency, score, jump, and
  magnitude gates pass; otherwise the frame is excluded.
- Matrix equalization uses condition/singular-value guards and reports the
  effective channel source rather than hiding model mismatch.
- Validation never treats a predicted objective as measured improvement and
  requires comparable power/context plus at least 24 valid frames per side.

## Verification

Run focused offline tests first, then the full pytest suite, then the low-power
B210 diagnostic and a short FDIDM RF run. Export JSON evidence for every RF
conclusion.
