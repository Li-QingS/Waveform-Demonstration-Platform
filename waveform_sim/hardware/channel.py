"""NTN-TDL 软件信道（从 fdidm_hardtest.py 搬移，阶段 6b）。"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


class NTNTDLChannel:
    """Small SISO NTN-TDL software channel for FDIDM bench testing.

    The tap tables follow the NTN-TDL-A/C/D profiles in 3GPP TR 38.811
    Tables 6.9.2-1, 6.9.2-3, and 6.9.2-4.  Delays in the tables are
    normalized; this class scales them to the requested RMS delay spread.

    This is intentionally a lightweight real-time GNU Radio helper:
    - fractional delays are implemented by vectorized linear interpolation;
    - Rayleigh taps use a sum-of-sinusoids approximation to the Doppler
      spectrum; LOS entries are deterministic specular components;
    - one common Doppler shift is applied to all taps, matching the NTN
      additional satellite Doppler statement in TR 38.811.
    """

    # (normalized_delay, power_dB, fading_kind)
    # LOS C/D are represented exactly as the table entries: a deterministic
    # LOS component plus a Rayleigh component at the same delay.
    TDL_PROFILES: Dict[str, List[Tuple[float, float, str]]] = {
        "tdl_a": [
            (0.0000, 0.000, "rayleigh"),
            (1.0811, -4.675, "rayleigh"),
            (2.8416, -6.482, "rayleigh"),
        ],
        "tdl_c": [
            (0.0000, -0.394, "los"),
            (0.0000, -10.618, "rayleigh"),
            (14.8124, -23.373, "rayleigh"),
        ],
        "tdl_d": [
            (0.0000, -0.284, "los"),
            (0.0000, -11.991, "rayleigh"),
            (0.5596, -9.887, "rayleigh"),
            (7.3340, -16.771, "rayleigh"),
        ],
    }

    DISPLAY_NAMES = {
        "tdl_a": "NTN-TDL-A (NLOS/Rayleigh)",
        "tdl_c": "NTN-TDL-C (LOS/Rician)",
        "tdl_d": "NTN-TDL-D (LOS/Rician)",
    }

    def __init__(
            self,
            sample_rate: float,
            model: str = "tdl_a",
            rms_delay_spread_ns: float = 1000.0,
            doppler_hz: float = 0.0,
            doppler_spread_hz: float = 0.0,
            snr_db: float = 35.0,
            seed: int = 0x38_811,
            normalize_power: bool = True,
            num_sinusoids: int = 8,
    ):
        self.sample_rate = float(max(sample_rate, 1.0))
        self.model = "tdl_a"
        self.rms_delay_spread_ns = 1000.0
        self.doppler_hz = 0.0
        self.doppler_spread_hz = 0.0
        self.snr_db = 35.0
        self.seed = int(seed) & 0xFFFFFFFF
        self.normalize_power = bool(normalize_power)
        self.num_sinusoids = int(max(4, min(int(num_sinusoids), 64)))
        self._rng = np.random.default_rng(self.seed)
        self._components: List[Dict[str, Any]] = []
        self._history = np.zeros(8, dtype=np.complex128)
        self._sample_index = 0
        self._delay_scale_s = 0.0
        self.configure(sample_rate=sample_rate, model=model,
                       rms_delay_spread_ns=rms_delay_spread_ns,
                       doppler_hz=doppler_hz,
                       doppler_spread_hz=doppler_spread_hz,
                       snr_db=snr_db,
                       seed=seed,
                       normalize_power=normalize_power,
                       num_sinusoids=num_sinusoids)

    @classmethod
    def normalize_model(cls, model: str) -> str:
        m = str(model or "tdl_a").strip().lower().replace("-", "_").replace(" ", "_")
        if m in ("a", "ntn_tdl_a", "tdla", "tdl_a"):
            return "tdl_a"
        if m in ("c", "ntn_tdl_c", "tdlc", "tdl_c"):
            return "tdl_c"
        if m in ("d", "ntn_tdl_d", "tdld", "tdl_d"):
            return "tdl_d"
        if m in cls.TDL_PROFILES:
            return m
        raise ValueError("software TDL model must be one of: tdl_a, tdl_c, tdl_d")

    @staticmethod
    def _db_to_linear(power_db: float) -> float:
        return float(10.0 ** (float(power_db) / 10.0))

    def configure(self, **kwargs: Any):
        if "sample_rate" in kwargs and kwargs["sample_rate"] is not None:
            self.sample_rate = float(max(float(kwargs["sample_rate"]), 1.0))
        if "model" in kwargs and kwargs["model"] is not None:
            self.model = self.normalize_model(kwargs["model"])
        if "rms_delay_spread_ns" in kwargs and kwargs["rms_delay_spread_ns"] is not None:
            self.rms_delay_spread_ns = float(max(0.0, float(kwargs["rms_delay_spread_ns"])))
        if "doppler_hz" in kwargs and kwargs["doppler_hz"] is not None:
            self.doppler_hz = float(kwargs["doppler_hz"])
        if "doppler_spread_hz" in kwargs and kwargs["doppler_spread_hz"] is not None:
            self.doppler_spread_hz = float(max(0.0, float(kwargs["doppler_spread_hz"])))
        if "snr_db" in kwargs and kwargs["snr_db"] is not None:
            self.snr_db = float(kwargs["snr_db"])
        if "seed" in kwargs and kwargs["seed"] is not None:
            self.seed = int(kwargs["seed"]) & 0xFFFFFFFF
        if "normalize_power" in kwargs and kwargs["normalize_power"] is not None:
            self.normalize_power = bool(kwargs["normalize_power"])
        if "num_sinusoids" in kwargs and kwargs["num_sinusoids"] is not None:
            self.num_sinusoids = int(max(4, min(int(kwargs["num_sinusoids"]), 64)))

        self._rng = np.random.default_rng(self.seed)
        raw = list(self.TDL_PROFILES[self.model])
        delays_norm = np.array([r[0] for r in raw], dtype=np.float64)
        powers = np.array([self._db_to_linear(r[1]) for r in raw], dtype=np.float64)
        if self.normalize_power:
            powers = powers / max(float(np.sum(powers)), 1e-12)
        mean_tau = float(np.sum(powers * delays_norm) / max(float(np.sum(powers)), 1e-12))
        rms_norm = float(np.sqrt(np.sum(powers * (delays_norm - mean_tau) ** 2) /
                                 max(float(np.sum(powers)), 1e-12)))
        desired_ds_s = float(self.rms_delay_spread_ns) * 1e-9
        self._delay_scale_s = desired_ds_s / max(rms_norm, 1e-12) if desired_ds_s > 0.0 else 0.0
        delays_samp = delays_norm * self._delay_scale_s * self.sample_rate
        max_delay_samp = float(np.max(delays_samp)) if delays_samp.size else 0.0
        hist_len = int(max(8, np.ceil(max_delay_samp) + 8))
        self._history = np.zeros(hist_len, dtype=np.complex128)
        self._sample_index = 0
        self._components = []
        for idx, (delay_norm, power_db, kind) in enumerate(raw):
            power_lin = self._db_to_linear(power_db)
            if self.normalize_power:
                power_lin = power_lin / max(float(np.sum([self._db_to_linear(r[1]) for r in raw])), 1e-12)
            comp: Dict[str, Any] = {
                "idx": int(idx),
                "delay_norm": float(delay_norm),
                "delay_samp": float(delay_norm) * self._delay_scale_s * self.sample_rate,
                "power_lin": float(power_lin),
                "sqrt_power": float(np.sqrt(max(power_lin, 0.0))),
                "kind": str(kind).lower(),
                "phase0": float(self._rng.uniform(0.0, 2.0 * np.pi)),
            }
            if comp["kind"] == "rayleigh":
                comp["static_rayleigh"] = ((self._rng.normal() + 1j * self._rng.normal()) / np.sqrt(2.0))
                # Deterministic-ish angle grid with a small random offset per tap.
                offset = float(self._rng.uniform(0.0, 1.0))
                comp["angles"] = 2.0 * np.pi * ((np.arange(self.num_sinusoids, dtype=np.float64) + offset)
                                                 / max(self.num_sinusoids, 1))
                comp["phases"] = self._rng.uniform(0.0, 2.0 * np.pi, size=self.num_sinusoids).astype(np.float64)
            self._components.append(comp)

    def reset(self):
        self._history[:] = 0.0
        self._sample_index = 0

    def summary(self) -> str:
        tap_desc = ", ".join(
            f"{c['kind']}@{c['delay_samp']:.3f} samp/{10*np.log10(max(c['power_lin'],1e-15)):.1f} dB"
            for c in self._components
        )
        return (f"{self.DISPLAY_NAMES.get(self.model, self.model)}, "
                f"DS={self.rms_delay_spread_ns:.1f} ns, fd={self.doppler_hz:.1f} Hz, "
                f"spread={self.doppler_spread_hz:.1f} Hz, SNR={self.snr_db:.1f} dB, taps=[{tap_desc}]")

    @staticmethod
    def _fractional_delay(ext: np.ndarray, hist_len: int, n: np.ndarray, delay_samp: float) -> np.ndarray:
        pos = float(hist_len) + n.astype(np.float64) - float(delay_samp)
        i0 = np.floor(pos).astype(np.int64)
        frac = pos - i0.astype(np.float64)
        out = np.zeros(n.size, dtype=np.complex128)
        valid = (i0 >= 0) & (i0 < ext.size)
        if np.any(valid):
            ii = i0[valid]
            ii1 = np.minimum(ii + 1, ext.size - 1)
            ff = frac[valid]
            out[valid] = (1.0 - ff) * ext[ii] + ff * ext[ii1]
        return out

    def _component_gain(self, comp: Dict[str, Any], t: np.ndarray) -> np.ndarray:
        sqrt_power = float(comp.get("sqrt_power", 0.0))
        common = float(self.doppler_hz)
        if comp.get("kind") == "los":
            return sqrt_power * np.exp(1j * (2.0 * np.pi * common * t + float(comp.get("phase0", 0.0))))
        spread = float(self.doppler_spread_hz)
        if spread <= 1e-9:
            return sqrt_power * complex(comp.get("static_rayleigh", 1.0 + 0.0j)) * np.exp(1j * 2.0 * np.pi * common * t)
        angles = np.asarray(comp.get("angles"), dtype=np.float64).reshape(-1)
        phases = np.asarray(comp.get("phases"), dtype=np.float64).reshape(-1)
        freqs = common + spread * np.cos(angles)
        # Shape: (num_sinusoids, num_samples).  The scale gives approximately
        # unit-power Rayleigh fading before multiplying by sqrt_power.
        ph = 2.0 * np.pi * freqs[:, None] * t[None, :] + phases[:, None]
        return sqrt_power * np.sum(np.exp(1j * ph), axis=0) / np.sqrt(max(freqs.size, 1))

    def process(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.complex64).reshape(-1)
        if x.size == 0:
            return np.zeros(0, dtype=np.complex64)
        # Denormal underflow from fractional-delay phasors / tiny gains and the
        # occasional overflow on a high-gain tap are numerically harmless here;
        # silence the spurious NumPy warnings around the synthesis math.
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            x128 = x.astype(np.complex128, copy=False)
            hist_len = int(self._history.size)
            ext = np.concatenate((self._history, x128))
            n = np.arange(x128.size, dtype=np.int64)
            t = (float(self._sample_index) + n.astype(np.float64)) / max(self.sample_rate, 1.0)
            y = np.zeros(x128.size, dtype=np.complex128)
            for comp in self._components:
                delayed = self._fractional_delay(ext, hist_len, n, float(comp.get("delay_samp", 0.0)))
                y += self._component_gain(comp, t) * delayed
            self._history = ext[-hist_len:].copy()
            self._sample_index += int(x128.size)
            if np.isfinite(self.snr_db) and self.snr_db < 200.0:
                sig_power = float(np.mean(np.abs(y) ** 2))
                if sig_power > 1e-18:
                    noise_power = sig_power / (10.0 ** (float(self.snr_db) / 10.0))
                    noise = np.sqrt(noise_power / 2.0) * (
                        self._rng.normal(size=y.size) + 1j * self._rng.normal(size=y.size)
                    )
                    y = y + noise
            return y.astype(np.complex64)

