# -*- coding: utf-8 -*-
"""3GPP-aligned LEO/NTN channel helpers.

The project previously used one number for two physically different effects:

* predictable satellite Doppler shift caused by the satellite/user radial speed;
* residual Doppler spread that controls small-scale channel time selectivity.

This module separates them.  The large common shift is assumed to be predicted
and pre-compensated by the NTN timing/frequency assistance loop.  Only the
residual common CFO plus the selected standardized maximum Doppler frequency is
fed into the fading channel.

The NTN TDL taps and the 200/1200 Hz combinations are taken from the NR-NTN
conformance channel definitions (NTN-TDLA100 and NTN-TDLC5).  The implementation
is intentionally SISO and compact; it is a reproducible educational link model,
not a replacement for a certified 3GPP channel emulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

C_LIGHT = 299_792_458.0


@dataclass(frozen=True)
class NTNProfile:
    """One SISO NTN tapped-delay profile and its standard Doppler setting."""

    name: str
    delays_ns: Tuple[float, ...]
    powers_db: Tuple[float, ...]
    fading: Tuple[str, ...]
    maximum_doppler_hz: float
    nominal_rms_delay_ns: float

    @property
    def path_count(self) -> int:
        return len(self.delays_ns)


# TS 38.101-5 / TS 38.521-5 NTN profiles.  NTN-TDLC5 tap 1 contains a LOS
# component and a Rayleigh component at the same nominal delay, therefore the
# compact SISO representation contains two paths at delay zero.
NTN_PROFILES: Dict[str, NTNProfile] = {
    "NTN-TDLA100-200": NTNProfile(
        name="NTN-TDLA100-200",
        delays_ns=(0.0, 110.0, 285.0),
        powers_db=(0.0, -4.7, -6.5),
        fading=("Rayleigh", "Rayleigh", "Rayleigh"),
        maximum_doppler_hz=200.0,
        nominal_rms_delay_ns=100.0,
    ),
    "NTN-TDLC5-200": NTNProfile(
        name="NTN-TDLC5-200",
        delays_ns=(0.0, 0.0, 60.0),
        powers_db=(-0.6, -8.9, -21.5),
        fading=("LOS", "Rayleigh", "Rayleigh"),
        maximum_doppler_hz=200.0,
        nominal_rms_delay_ns=5.0,
    ),
    "NTN-TDLC5-1200": NTNProfile(
        name="NTN-TDLC5-1200",
        delays_ns=(0.0, 0.0, 60.0),
        powers_db=(-0.6, -8.9, -21.5),
        fading=("LOS", "Rayleigh", "Rayleigh"),
        maximum_doppler_hz=1200.0,
        nominal_rms_delay_ns=5.0,
    ),
}

# Friendly aliases accepted from old projects/combobox values.
_PROFILE_ALIASES = {
    "TDL-A": "NTN-TDLA100-200",
    "TDLA": "NTN-TDLA100-200",
    "NTN-TDLA100": "NTN-TDLA100-200",
    "TDL-C": "NTN-TDLC5-200",
    "TDLC": "NTN-TDLC5-200",
    "NTN-TDLC5": "NTN-TDLC5-200",
    "TDL-D": "NTN-TDLC5-1200",
    "CDL": "NTN-TDLC5-1200",
}


def normalize_profile_name(name: str) -> str:
    value = str(name or "NTN-TDLA100-200").upper().replace("_", "-").strip()
    if value in NTN_PROFILES:
        return value
    return _PROFILE_ALIASES.get(value, "NTN-TDLA100-200")


def get_profile(name: str) -> NTNProfile:
    return NTN_PROFILES[normalize_profile_name(name)]


def absolute_doppler_shift_hz(
    carrier_frequency_hz: float,
    velocity_kmh: float,
    radial_projection: float,
) -> float:
    """Magnitude of the predictable common satellite Doppler shift."""

    fc = max(float(carrier_frequency_hz), 0.0)
    velocity_mps = abs(float(velocity_kmh)) / 3.6
    projection = float(np.clip(float(radial_projection), 0.0, 1.0))
    return float(velocity_mps * projection * fc / C_LIGHT)


def residual_common_cfo_hz(
    absolute_shift_hz: float,
    compensation_ratio: float,
) -> float:
    """Uncompensated fraction of the common Doppler shift.

    ``compensation_ratio=1`` means perfect common-shift compensation.  This is
    deliberately separate from multipath Doppler spread.
    """

    ratio = float(np.clip(float(compensation_ratio), 0.0, 1.0))
    return float(abs(float(absolute_shift_hz)) * (1.0 - ratio))


def coherence_time_s(maximum_doppler_hz: float) -> float:
    """Return the common 0.423/f_D engineering coherence-time estimate."""

    fd = abs(float(maximum_doppler_hz))
    if fd <= 1e-12:
        return float("inf")
    return float(0.423 / fd)


def normalized_doppler(maximum_doppler_hz: float, subcarrier_spacing_hz: float) -> float:
    return float(abs(float(maximum_doppler_hz)) / max(abs(float(subcarrier_spacing_hz)), 1e-15))


def sample_path_dopplers(
    path_count: int,
    maximum_spread_hz: float,
    common_residual_hz: float,
    rng: np.random.Generator,
    deterministic: bool = False,
) -> np.ndarray:
    """Sample per-path residual Dopplers around the compensated common CFO.

    The path offsets follow a cosine angle model and stay inside
    ``common_residual ± maximum_spread``.  They must not use the large orbital
    Doppler shift directly.
    """

    count = max(1, int(path_count))
    spread = abs(float(maximum_spread_hz))
    common = float(common_residual_hz)
    if count == 1:
        offsets = np.asarray([spread], dtype=np.float64)
    elif deterministic:
        angles = np.linspace(0.0, np.pi, count, dtype=np.float64)
        offsets = spread * np.cos(angles)
    else:
        angles = rng.uniform(0.0, 2.0 * np.pi, count)
        offsets = spread * np.cos(angles)
    return np.asarray(common + offsets, dtype=np.float64)


def weighted_mean_and_rms(values: Iterable[float], weights: Iterable[float]) -> Tuple[float, float]:
    x = np.asarray(list(values), dtype=np.float64)
    w = np.asarray(list(weights), dtype=np.float64)
    if x.size == 0 or w.size != x.size:
        return float("nan"), float("nan")
    w = np.maximum(w, 0.0)
    total = float(np.sum(w))
    if total <= 1e-15:
        return float(np.mean(x)), float(np.std(x))
    w /= total
    mean = float(np.sum(w * x))
    rms = float(np.sqrt(max(np.sum(w * (x - mean) ** 2), 0.0)))
    return mean, rms


def profile_metadata(name: str) -> Dict[str, float | str | int]:
    profile = get_profile(name)
    return {
        "profile": profile.name,
        "path_count": profile.path_count,
        "maximum_doppler_hz": float(profile.maximum_doppler_hz),
        "nominal_rms_delay_ns": float(profile.nominal_rms_delay_ns),
        "maximum_excess_delay_ns": float(max(profile.delays_ns) if profile.delays_ns else 0.0),
    }
