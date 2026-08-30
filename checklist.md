# Offline Alpha/Beta Adaptation Validation Checklist

## Implementation Completeness

- [x] Adapter reuses the production alpha/beta SER objective (run the script's
  single-sample path and inspect finite recommendation fields).
- [x] Diagonal and full deterministic channels are generated (run the default
  experiment and confirm both groups are present).
- [x] All four perturbations are reported independently (inspect JSON group
  keys).
- [x] Per-sample failures are recorded rather than terminating the run (run a
  deliberately invalid configuration and inspect the nonzero validation error).

## Numerical Behavior

- [x] Production and exhaustive SER values are finite for valid samples (run
  the focused comparison test).
- [x] Mean, p95 and maximum SER loss are present for diagonal and full groups
  (inspect `summary` in JSON).
- [x] Exhaustive optimum alpha/beta distributions and movement statistics are
  present (inspect group distribution fields).
- [x] Global complex scaling with proportional noise preserves SER ranking
  within tolerance; coordinate ties are reported separately (run the
  scale-invariance test).
- [x] Selective perturbations can change the optimum and are not folded into
  the identity group (inspect perturbation-specific recommendations).

## Integration and Runtime

- [x] `pytest -q tests/test_alpha_beta_validation.py` passes.
- [x] The default CLI completes without GNU Radio, UHD, USRP, or a display.
- [x] Repeating the default CLI with the same seed produces the same
  deterministic digest; real runtime fields are intentionally excluded.
- [x] The text summary identifies the worst search-loss sample.
