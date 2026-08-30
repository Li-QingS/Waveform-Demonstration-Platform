# Offline Alpha/Beta Adaptation Validation Tasks

## File List

| Operation | File | Responsibility |
|---|---|---|
| New | `scripts/validate_alpha_beta_adaptation.py` | CLI, generators, exhaustive reference, reports |
| New | `tests/test_alpha_beta_validation.py` | Determinism, invariance, report and validation tests |
| New | `checklist.md` | Acceptance checklist |

## T1: Implement Standalone Adapter and Channel Generators

**File:** `scripts/validate_alpha_beta_adaptation.py`

1. Define the adapter with production Gamma/DFT helpers.
2. Define validated configuration and deterministic diagonal/full generators.
3. Implement identity, global-scale, frequency-selective and time-selective
   perturbations.

**Verification:** import the module and generate both channel shapes for a
fixed seed; all values are finite and repeatable.

## T2: Implement Production-vs-Exhaustive Comparison

**File:** `scripts/validate_alpha_beta_adaptation.py`

1. Construct snapshots accepted by the production optimizer.
2. Call `_optimize_alpha_beta_snapshot` unchanged.
3. Evaluate every fine-grid candidate using the prepared objective.
4. Compute SER loss, coordinate distance, tie handling and failure records.

**Verification:** run one diagonal and one full sample; production and
reference SER values are finite and the report contains both recommendations.

## T3: Implement Aggregation and CLI Output

**File:** `scripts/validate_alpha_beta_adaptation.py`

1. Aggregate by channel kind and perturbation.
2. Calculate distribution and movement statistics.
3. Serialize JSON and print a concise summary.
4. Add CLI options for sample count, sizes, steps, seed, output path and
   perturbation selection.

**Verification:** default command exits zero, writes JSON, and prints group
   summaries.

## T4: Add Automated Tests

**File:** `tests/test_alpha_beta_validation.py`

1. Test deterministic generation and report shape.
2. Test global-scale invariance with proportional noise scaling.
3. Test search-loss calculation and invalid configuration handling.
4. Test that selective perturbations are represented separately.

**Verification:** run `pytest -q tests/test_alpha_beta_validation.py`.

## Execution Order

```text
T1 -> T2 -> T3 -> T4
```
