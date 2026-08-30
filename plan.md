# Offline Alpha/Beta Adaptation Validation Plan

## Architecture Overview

Add one standalone experiment module under `scripts/` and one focused test
module under `tests/`. The experiment module reuses the production
`FDIDMAdaptiveMixin` objective and transform helpers through a small adapter
object, while keeping exhaustive reference search and perturbation generation
inside the experiment code. No production hardware or optimizer code is
modified.

The command flow is:

```text
configuration
  -> deterministic channel generator
  -> baseline/perturbed H_TF samples
  -> production coarse-to-fine optimizer
  -> exhaustive fine-grid reference
  -> per-sample comparison
  -> aggregate statistics
  -> JSON + text report
```

## Core Data Structures

### ExperimentConfig

Fields: `M`, `N`, `sample_count`, `seed`, `fine_step`, `coarse_step`,
`fine_refinement_step`, `mod_order`, `equalizer`, `noise_var`, `channel_kind`,
`perturbations`, and output path. Values are validated before execution.

### SampleResult

Fields: sample id, channel kind, perturbation, baseline/reported noise,
production recommendation, exhaustive optimum, production/reference SER,
loss in dB, coordinate distance, candidate count, runtime, and error text.

### AggregateReport

Separate aggregates for diagonal/full channels and each perturbation. Each
aggregate contains successful/failed counts, mean/p50/p95/max SER loss,
mean/p95 coordinate distance, near-basin hit rates, optimum alpha/beta mean,
standard deviation, range, and sequence movement statistics.

## Module Design

### `scripts/validate_alpha_beta_adaptation.py`

Responsibilities:

- Parse CLI arguments and validate configuration.
- Build a minimal adapter implementing `_gamma`, `_apply_gamma_axis`, and the
  production mixin dependencies.
- Generate deterministic diagonal/full channel samples.
- Apply named linear perturbations.
- Invoke production `_optimize_alpha_beta_snapshot`.
- Invoke exhaustive reference evaluation at the configured fine step.
- Compute per-sample and aggregate statistics.
- Write JSON and print a concise summary.

Public functions:

- `run_experiment(config) -> dict`
- `main(argv=None) -> int`

Internal functions should be pure where practical, especially channel
generation, perturbation, exhaustive evaluation, and statistics calculation.

### `tests/test_alpha_beta_validation.py`

Tests the report contract and numerical invariants without hardware.

## Search and Comparison Design

The production optimizer is called unchanged. The exhaustive reference uses
the same prepared objective and evaluates every `(alpha, beta)` point in the
fine grid. Reference ties are resolved deterministically by SER, then distance
to the current pair, then alpha/beta coordinates.

SER loss is calculated as:

```text
10 * log10(max(production_ser, floor) / max(reference_ser, floor))
```

The report also records coordinate distance, because coordinate disagreement
can be harmless when the SER surface is flat.

## Channel and Perturbation Design

- Diagonal channels: complex TF-cell gains with deterministic log-normal
  amplitude and uniform phase.
- Full channels: deterministic complex Gaussian matrices, with alternating
  diagonal-dominant and dense samples.
- `identity`: no perturbation.
- `global_scale`: multiply `H_TF` by a complex scalar and noise variance by its
  squared magnitude.
- `frequency_selective_gain`: apply nonuniform gains by subcarrier.
- `time_selective_gain`: apply nonuniform gains by symbol.

The perturbation implementation must preserve matrix shape and record the
transformation parameters in the result metadata.

## Result Format

Top-level JSON keys:

- `config`
- `summary`
- `groups`
- `samples`
- `failures`

The text summary prints one line per channel/perturbation group and highlights
the worst SER-loss sample.

## Technical Decisions

| Decision | Choice | Reason |
|---|---|---|
| Production objective | Reuse `FDIDMAdaptiveMixin` | Prevent duplicate formula implementations |
| Reference grid | Full configurable fine grid | Provides an explicit global-search baseline |
| Default matrix size | `M=N=4` | Keeps standalone and CI runs fast |
| Failure handling | Record and continue | One invalid sample must not hide aggregate results |
| Output | JSON plus stdout summary | Supports both automation and human inspection |
| Hardware dependency | None | This phase is explicitly offline |
