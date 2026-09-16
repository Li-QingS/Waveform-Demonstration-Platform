# FDIDM Hardware Deep Debug Checklist

## Implementation

- [x] Probe continuity is proven by monotonic metadata; stale/gap windows are excluded.
- [x] CFO alias/refinement has score, phase-consistency, magnitude, and jump gates.
- [x] Rejected sync candidates cannot modify CSI, metrics, or adaptation state.
- [x] Full-H solves have condition/singular-value protection and source labels.
- [x] Data-aided EVM, DD-EVM, exact SER/BER, FEC, and CRC are distinct.
- [x] Adaptive validation enforces power/context comparability and 24 valid frames.
- [x] B210 diagnostic reports a classification and evidence thresholds at low power.
- [x] UI distinguishes measured RF, predicted objective, injected TDL, and residual metrics.

## Verification

- [ ] `python -m pytest -q tests/test_fdidm_hardware_deep_debug.py` passes (legacy filename; coverage is distributed across focused tests).
- [x] `python -m pytest -q` passes with no new failures (287 passed).
- [x] B210 probe records identity, port, RMS, clipping, tone, gaps, and status.
- [x] Short FDIDM run refuses an algorithm-improvement claim when RMS/pilot fit is invalid.
- [ ] Offscreen 1400x800, 1400x900, and 1200x780 layouts have no clipping/overlap.
