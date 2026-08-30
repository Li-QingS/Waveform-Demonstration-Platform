# Simple Hardware Adaptation Checklist

## Receiver Stability

- [ ] Static noisy CSI is smoothed across frames and has lower variance than the
  raw per-frame estimate (verify with deterministic synthetic CSI test).
- [ ] Equalizer CSI is averaged across time slots while adaptive CSI retains an
  injected time variation (verify both outputs from the same pilot sequence).
- [ ] A valid tracked frame updates CFO/CSI without a full acquisition reset;
  an invalid lock falls back to acquisition (verify tracking-state test).
- [ ] Measured uncoded SER equals the injected symbol-error ratio and is reported
  separately from model SER and BER (verify known-symbol test and status fields).

## Five-Point Adaptation

- [ ] Every diagonal adaptive evaluation contains at most five unique candidates
  (inspect result candidate count in boundary and interior tests).
- [ ] A recommendation changes only alpha or beta and by no more than the active
  step (verify all deterministic recommendation cases).
- [ ] A time-invariant synthetic channel reports beta unobservable and does not
  change beta (verify flat-axis test).
- [ ] A time-selective synthetic channel can report beta observable and recommend
  a beta neighbour (verify time-selective test).
- [ ] No coarse improvement changes the active step to fine; context reset changes
  it back to coarse (verify state transition test).
- [ ] Stability, minimum-improvement, interval, and cooldown gates still prevent
  premature switching (run focused adaptive regression tests).

## Power And Switching

- [ ] Alpha/beta waveforms have equal average energy within tolerance when peak
  limiting is inactive (compare deterministic waveform builds).
- [ ] A peak-limited waveform exposes the limiting state in status/fingerprint
  (verify high-PAPR test).
- [ ] Applying alpha/beta while running updates vector data without calling
  backend stop/start (verify mocked live-update test).
- [ ] Live waveform replacement uses fresh transition windows and does not arm
  the normal 800 ms startup settle delay (verify transition-state timestamps).
- [ ] Live-safe UI apply preserves existing metric history and structural apply
  retains the controlled restart/reset behavior (verify UI/backend path test).

## Diagnostics And Integration

- [ ] Status and logs expose active step, axis observability, candidate count,
  predicted SER, and measured SER (inspect synthetic status snapshot).
- [ ] An available UHD overflow indication prevents that interval from feeding
  adaptive CSI and appears in exported diagnostics (verify injected overflow).
- [ ] Log export continues to save under the project `log/` directory (run the
  existing log-export test).
- [ ] Existing offline alpha/beta validation tests remain passing (run focused
  validation suite).
- [ ] All new tests run without GNU Radio, UHD, USRP, network, or a display.

## Commands

- [ ] `pytest -q tests/test_simple_hardware_adaptation.py`
- [ ] `pytest -q tests/test_hardware_adaptive.py tests/test_hardware_log_export.py`
- [ ] `pytest -q tests/test_alpha_beta_validation.py`
- [ ] `pytest -q`

## End-To-End Scenario

- [ ] Start from a noisy static synthetic RF snapshot, acquire and enter tracking,
  run a five-point evaluation, apply one alpha or beta step through the live path,
  consume fresh frames without the 800 ms delay, and observe distinct measured
  and predicted SER fields with no stop/start call.
