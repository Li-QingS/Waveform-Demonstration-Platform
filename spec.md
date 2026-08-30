# Offline Alpha/Beta Adaptation Validation Spec

## Background

The hardware alpha/beta controller searches the current measured CSI using a
coarse-to-fine grid. USRP hardware is currently unavailable, so the software
model must be used to answer two questions before hardware testing:

1. Whether the current search normally reaches a point with effectively global
   minimum predicted SER.
2. Whether linear effects outside the paper model preserve or change the
   alpha/beta ranking.

## Goal

Provide a reproducible offline experiment that generates channel samples,
compares the implemented search with an exhaustive reference grid, summarizes
the distribution of optimum alpha/beta values, and measures the effect of
controlled linear channel perturbations.

## Functional Requirements

- F1: Generate deterministic diagonal and full `H_TF` channel samples for a
  configurable `M`, `N`, sample count, seed, equalizer, modulation and noise
  level.
- F2: Run the existing hardware adaptation optimizer on every sample.
- F3: Run an exhaustive reference search over a configurable fine grid,
  defaulting to step `0.05` over `[0, 2]` for both alpha and beta.
- F4: Report per-sample and aggregate search loss in dB, coordinate distance,
  exact/near basin hit rates, runtime, and worst cases.
- F5: Report the empirical distribution of exhaustive optimum alpha/beta values
  and adjacent-sample movement for controlled channel perturbation sequences.
- F6: Evaluate at least these perturbations: identity, global complex scaling
  with proportional noise scaling, frequency-selective diagonal gain, and
  time-selective diagonal gain.
- F7: Emit machine-readable results and a concise human-readable summary so
  experiments can be compared across runs.
- F8: Add automated tests covering deterministic output, scale invariance,
  search-loss calculation, and invalid configuration handling.

## Non-Functional Requirements

- N1: Experiments must not require GNU Radio, UHD, a connected USRP, or a GUI.
- N2: Results must be reproducible from the recorded seed and configuration.
- N3: The experiment must not modify the production optimizer or hardware
  runtime behavior.
- N4: The default small experiment must finish in a practical CI/runtime
  window and avoid allocations larger than the configured matrix order.
- N5: Numerical failures for one sample must be recorded as a sample failure,
  not silently omitted.

## Out Of Scope

- Future-channel prediction.
- Over-the-air signaling or real USRP measurements.
- Replacing the current coarse-to-fine search algorithm.
- Claiming hardware performance equivalence from software results alone.
- Training a machine-learning predictor for alpha/beta.

## Acceptance Criteria

- AC1: A default command produces a JSON result and a readable summary without
  hardware dependencies.
- AC2: For deterministic input, two runs produce identical aggregate metrics
  and per-sample recommendations.
- AC3: The report separates diagonal and full `H_TF` cases and includes mean,
  p95 and maximum search loss in dB.
- AC4: The report includes the exhaustive optimum alpha/beta distribution and
  a controlled-sequence movement statistic.
- AC5: Global complex scaling with proportional noise scaling preserves the
  candidate ranking within numerical tolerance for the tested samples.
- AC6: Frequency/time-selective perturbations are reported separately and may
  change the optimum; the tool must not assume invariance.
- AC7: Automated tests pass without GNU Radio, UHD, PyQt display access, or
  external services.
