# FDIDM Hardware Deep Debug Checklist

> Evidence first: every tick below refers to a command output, a log line or a
> measured hardware run.  Hardware-in-the-loop items are only ticked when the
> connected B210 was actually used.

## Implementation

- [x] Probe continuity is proven by monotonic metadata; stale/gap windows are excluded.
- [x] CFO alias/refinement has score, phase-consistency, magnitude, and jump gates.
- [x] Rejected sync candidates cannot modify CSI, metrics, or adaptation state.
- [x] Full-H solves have condition/singular-value protection and source labels.
- [x] Data-aided EVM, DD-EVM, exact SER/BER, FEC, and CRC are distinct.
- [x] Adaptive validation enforces power/context comparability and 24 valid frames.
- [x] B210 diagnostic reports a classification and evidence thresholds at low power.
- [x] UI distinguishes measured RF, predicted objective, injected TDL, and residual metrics.
- [x] Sub-sample (fractional) RX timing residual is measured against the known pilot,
      gated, removed before channel estimation, and exported in status/UI/log.
- [x] A live TX/RX gain change drops the probe vectors that still hold the previous
      link level instead of letting mixed frames reach CRC, EVM or the A/B validator.

## Verification

- [x] `python -m pytest -q` passes with no failures (316 passed).
- [x] `python -m compileall -q waveform_sim tests` and `git diff --check` pass.
- [x] B210 probe records identity, port, RMS, clipping, tone, gaps, and status.
- [x] Short FDIDM run refuses an algorithm-improvement claim when RMS/pilot fit is invalid.
- [x] Offscreen 1400x800, 1400x900, 1200x780 and 1100x700 layouts have no
      clipping/overlap (`tests/test_fdidm_hardware_comparison.py`, four sizes).
- [x] Fractional-timing unit coverage: estimator accuracy, corrector gain,
      small-residual skip, degenerate input, status fields and the post-RF TDL gate
      (`tests/test_fractional_timing.py`, 18 cases).

## Hardware-in-the-loop evidence (B210 serial 31896C6, USB 3, 2.4 GHz, 500 kS/s)

- [x] RF port probe: TX 0 dB / RX 20 dB shows no usable tone; TX 10 dB / RX 30 dB
      passes on frontend A -> A (tone +26.1 dB, no clipping); A -> B, B -> A and
      B -> B report `no_signal`.
- [x] FDIDM RF-only demodulation at TX/RX 30/30 dB: 67/67 frames decoded, SER 0,
      BER 0, sync about 0.95.
- [x] FDIDM RF-only demodulation at TX/RX 45/45 dB: best measured data-aided EVM
      10.1 %, sync 0.969, zero symbol errors over 36 frames.
- [x] Gain sweep TX30/RX30 -> TX45/RX45: EVM 26.4 % -> 16.5 % -> 10.1 %, i.e. the
      residual EVM is link-SNR limited, not ADC limited (RX 40 -> 45 dB is flat).
- [x] Ideal-TX self-check: the generated frame demodulates at EVM 0.000 % and
      pilot NMSE 3.6e-9, so the remaining EVM comes from the RF chain, not the model.
- [x] Fractional timing: residual is constant inside a run (sigma about 1e-3 samples)
      and varies between runs (0.02 .. 0.36 samples); replaying one captured frame
      through the production receiver moved EVM from 34.7 % to 25.2 % at 0.36 samples
      and from 16.1 % to 12.8 % at 0.12 samples.
- [x] Adaptive closed loop in `tdl_a_rf` mode: recommendation alpha/beta 0.75/1.0
      validated through baseline -> candidate windows, conclusion `improved`
      (EVM 14.77 % -> 13.25 %, no SER regression), `commit_complete=true`,
      `power_contract_released=true`, 66/66 frames decoded.
- [ ] Long-duration soak (more than 10 minutes) with TDL Doppler spread enabled.

## Known limits (measured, not assumed)

- The B210 TX and RX use independent local oscillators, so a common phase/frequency
  offset and phase noise remain.  They set the practical EVM floor near 10 % at the
  45 dB gain ceiling with this cable/attenuator setup; the residual is not an
  algorithm defect.
- Cross-frame complex CSI smoothing stays disabled because that phase is not
  coherent; a per-frame complex gain is estimated from the known payload instead.

## Acceptance result (2026-09-17)

- Passed 15/16; the only open item is the optional long soak, which needs a longer
  bench slot and is not required for a single debugging round.
- Commands: `python -m pytest -q` (316 passed); `python -m compileall -q waveform_sim tests`;
  `git diff --check`; `scripts/diagnose_b210_rf_ports.py`; live FDIDM runs and the
  adaptive A/B run recorded above.