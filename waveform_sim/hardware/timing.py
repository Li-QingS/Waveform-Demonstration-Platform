"""Sub-sample (fractional) timing estimation and correction for FDIDM RX.

A bench RF link and the USRP front end add a group delay that is not an
integer number of sample periods.  Preamble correlation removes the integer
part, but the residual fraction shifts every received frame by up to half a
sample and spreads energy into neighbouring FDIDM cells.

Measured on the connected B200/B210 at TX/RX = 45/45 dB with M = N = 16 and
500 kS/s, the residual fraction is constant inside one run (standard deviation
about 1e-3 samples) yet varies between runs (0.02 .. 0.36 samples).  It tracks
the data-aided EVM: about 10 % at 0.02 samples and about 35 % at 0.36 samples.
Removing the measured fraction took a 0.36-sample run from 34.7 % to 25.2 %
EVM and a 0.12-sample run from 16.1 % to 12.8 % EVM, while runs that were
already aligned (|delay| < 0.05 samples) were better left untouched.

The implementation is pure NumPy so it can be unit tested without GNU Radio,
UHD, Qt or a device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# Half length of the windowed-sinc interpolator.  16 taps per side is enough
# for the 500 kS/s band-limited FDIDM waveform and costs far less than one
# Wigner transform per frame.
DEFAULT_HALF_TAPS = 16
# Kaiser beta 8.6 gives roughly -60 dB sidelobes, which keeps the correction
# transparent for frames that are already aligned.
_KAISER_BETA = 8.6
# Below this residual the diagonal TF estimate already absorbs the delay and
# interpolating measurably costs more than it recovers: a 0.06-sample run got
# 1 pp worse, while 0.12 and 0.36 samples recovered 3.2 pp and 9.5 pp.
DEFAULT_MIN_ABS_DELAY = 0.10
# A candidate whose pilot correlation with the known reference is worse than
# this is noise, not a frame with a timing offset.
_MIN_CORRELATION = 0.25


@dataclass(frozen=True)
class FractionalDelayEstimate:
    """Result of one sub-sample timing measurement."""

    delay_samples: float
    integer_lag: int
    fractional_samples: float
    magnitude: float
    reliable: bool

    def correction_samples(self, minimum_abs: float = DEFAULT_MIN_ABS_DELAY) -> float:
        """Fraction of a sample to remove, or 0.0 when it is not worthwhile.

        Only the part inside one sample period is returned: the integer part is
        already chosen by preamble correlation, and re-applying it here would
        fight the frame aligner.
        """

        if not self.reliable or not np.isfinite(self.fractional_samples):
            return 0.0
        value = float(self.fractional_samples)
        return value if abs(value) >= float(max(0.0, minimum_abs)) else 0.0


def fractional_delay_kernel(delay: float, half_taps: int = DEFAULT_HALF_TAPS) -> np.ndarray:
    """Return the unit-gain FIR kernel that advances a signal by ``delay``.

    ``y[n] = x(n + delay)``: a positive ``delay`` undoes a measured positive
    delay.  The unit sum keeps the interpolator gain transparent, which
    matters because the residual complex gain is measured afterwards from the
    corrected frame.
    """

    half = int(max(1, half_taps))
    value = float(delay)
    if not np.isfinite(value):
        raise ValueError("delay must be finite")
    index = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.sinc(index + value) * np.kaiser(index.size, _KAISER_BETA)
    total = float(np.sum(kernel))
    if not np.isfinite(total) or abs(total) < 1e-12:
        raise ValueError("degenerate fractional-delay kernel")
    return kernel / total


def fractional_delay_correct(
    samples: Sequence[complex] | np.ndarray,
    delay: float,
    half_taps: int = DEFAULT_HALF_TAPS,
) -> np.ndarray:
    """Advance ``samples`` by ``delay`` samples with a windowed sinc.

    The frame is zero padded before convolution so no energy wraps from the
    end of the frame into its start.  The guard regions at both ends of the
    FDIDM frame keep the padded edge effect negligible.
    """

    arr = np.asarray(samples, dtype=np.complex128).reshape(-1)
    if arr.size == 0:
        return arr.copy()
    value = float(delay)
    if not np.isfinite(value):
        raise ValueError("delay must be finite")
    if abs(value) < 1e-9:
        return arr.copy()
    half = int(max(1, half_taps))
    kernel = fractional_delay_kernel(value, half)
    padded = np.concatenate(
        (np.zeros(half, dtype=np.complex128), arr, np.zeros(half, dtype=np.complex128))
    )
    filtered = np.convolve(padded, kernel, mode="same")
    return filtered[half:half + arr.size]


def _parabolic_fraction(left: float, centre: float, right: float) -> float:
    denominator = left - 2.0 * centre + right
    if not np.isfinite(denominator) or abs(denominator) < 1e-20:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    if not np.isfinite(offset) or abs(offset) > 0.5:
        return 0.0
    return float(offset)


def _circular(magnitude: np.ndarray, index: int) -> float:
    size = int(magnitude.size)
    if size <= 0:
        return float("nan")
    return float(magnitude[int(index) % size])


def estimate_fractional_delay(
    observed: Sequence[complex] | np.ndarray,
    reference: Sequence[complex] | np.ndarray,
    *,
    half_taps: int = DEFAULT_HALF_TAPS,
    refine: bool = True,
) -> FractionalDelayEstimate:
    """Estimate the sub-sample delay of ``observed`` relative to ``reference``.

    The fractional offset comes from parabolic interpolation of the
    cross-correlation peak.  ``refine`` then re-scores a short grid with the
    actual interpolator, which removes the bias of a pure sinc model when the
    front end also shapes the pulse.
    """

    obs = np.asarray(observed, dtype=np.complex128).reshape(-1)
    ref = np.asarray(reference, dtype=np.complex128).reshape(-1)
    if obs.size == 0 or ref.size == 0:
        return FractionalDelayEstimate(float("nan"), 0, float("nan"), float("nan"), False)
    if obs.size != ref.size:
        count = int(min(obs.size, ref.size))
        obs = obs[:count]
        ref = ref[:count]
    obs_energy = float(np.sum(np.abs(obs) ** 2))
    ref_energy = float(np.sum(np.abs(ref) ** 2))
    if obs_energy <= 0.0 or ref_energy <= 0.0:
        return FractionalDelayEstimate(float("nan"), 0, float("nan"), float("nan"), False)

    size = 1
    target = 8 * max(obs.size, ref.size)
    while size < target:
        size *= 2
    spectrum = np.fft.ifft(np.fft.fft(obs, size) * np.conj(np.fft.fft(ref, size)))
    magnitude = np.abs(spectrum)
    peak = int(np.argmax(magnitude))
    lag = peak if peak <= size // 2 else peak - size
    fraction = _parabolic_fraction(
        _circular(magnitude, peak - 1),
        _circular(magnitude, peak),
        _circular(magnitude, peak + 1),
    )
    total = float(lag) + fraction
    normaliser = float(np.sqrt(obs_energy * ref_energy))
    score = float(_circular(magnitude, peak) / normaliser) if normaliser > 0.0 else float("nan")

    fractional = total - round(total)
    if refine and abs(fractional) > 1e-9:
        grid = np.linspace(fractional - 0.06, fractional + 0.06, 7)
        best_score, best_fractional = score, fractional
        for candidate in grid:
            try:
                shifted = fractional_delay_correct(obs, float(candidate), half_taps)
            except ValueError:
                continue
            denominator = float(np.linalg.norm(shifted)) * float(np.linalg.norm(ref))
            if denominator <= 0.0:
                continue
            value = float(abs(np.vdot(ref, shifted)) / denominator)
            if np.isfinite(value) and value > best_score:
                best_score, best_fractional = value, float(candidate)
        # Re-scoring only refines the fraction; the integer alignment belongs
        # to the preamble search.
        total = float(round(total)) + float(best_fractional)
        fractional = float(best_fractional)
        score = float(best_score)

    reliable = bool(
        np.isfinite(total) and np.isfinite(score) and score >= _MIN_CORRELATION and abs(fractional) <= 0.5
    )
    return FractionalDelayEstimate(
        delay_samples=float(total),
        integer_lag=int(round(total)),
        fractional_samples=float(fractional),
        magnitude=float(score),
        reliable=reliable,
    )


def delay_needs_correction(delay: float, minimum_abs: float = DEFAULT_MIN_ABS_DELAY) -> bool:
    """Whether a measured residual is large enough to be worth interpolating."""

    if not np.isfinite(delay):
        return False
    return bool(abs(float(delay)) >= float(max(0.0, minimum_abs)))
