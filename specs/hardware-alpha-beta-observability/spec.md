# Hardware Alpha/Beta Observability Fix Spec

## Background

The hardware adaptive search receives `diag_tf` CSI whose frequency response
is averaged across all time slots and then repeated across the time axis. This
removes all time-axis variation from the optimizer input. Consequently the SER
objective is exactly flat in beta, and the tie policy reports the manually set
beta as the recommendation. The latest hardware log demonstrates this with
`current=best`, `gain=0.00dB`, and recommendations that follow beta changes.

Exported FDIDM logs currently default to the repository root instead of the
project `log/` directory.

## Goal

- Preserve time-resolved diagonal CSI for alpha/beta optimization without
  changing the stable averaged CSI used by the hardware equalizer.
- Distinguish an observable optimum from a flat/unidentifiable search axis.
- Export hardware logs under the project `log/` directory by default.

## Functional Requirements

- F1: The diagonal estimator provides a denoised per-time-slot CSI snapshot to
  the adaptive optimizer while retaining the existing averaged estimate for
  equalization.
- F2: The optimizer reports whether alpha and beta have a measurable objective
  span; a flat axis must be identified rather than presented as an independently
  discovered optimum.
- F3: Static channels must not trigger arbitrary beta switches merely because
  pilot noise makes one candidate microscopically better.
- F4: Time-selective diagonal CSI must allow beta recommendations to differ from
  the current beta when the predicted improvement exceeds existing gates.
- F5: UI and backend log exports default to `<project>/log/`, creating the
  directory when needed.
- F6: Search diagnostics include current point, selected point, axis
  observability, candidate count, and representative SER values.

## Non-Functional Requirements

- N1: Existing hardware equalization behavior remains unchanged.
- N2: Tests do not require GNU Radio, UHD, a connected USRP, or a display.
- N3: Existing improvement, stability, cooldown, and interval gates remain in
  force.
- N4: Per-frame diagnostic logging remains rate-limited where appropriate.

## Out Of Scope

- Future-channel prediction.
- Replacing the paper SER objective with measured BER or EVM.
- Claiming that beta is identifiable on a genuinely time-invariant channel.
- Changing the RF or software-TDL model parameters automatically.

## Acceptance Criteria

- AC1: A synthetic time-selective diagonal channel produces non-flat beta
  objective values and can recommend a beta different from the current value.
- AC2: A synthetic time-invariant diagonal channel is reported as beta-flat and
  does not cause an arbitrary beta switch.
- AC3: The equalizer continues to receive the averaged diagonal estimate.
- AC4: Restart and channel-context reset regression tests remain passing.
- AC5: Calling log export without a path creates a file under `<project>/log/`.
- AC6: The UI save dialog opens in `<project>/log/` rather than the repository
  root.
