"""Validated single-parameter scan plans for hardware NTN measurements."""
from __future__ import annotations

from dataclasses import dataclass
import math


NTN_SCAN_PARAMETERS = {
    "ntn_residual_cfo_hz": {
        "label": "残余频偏",
        "unit": "Hz",
        "minimum": -5000.0,
        "maximum": 5000.0,
        "start": -200.0,
        "stop": 200.0,
    },
    "ntn_doppler_spread_hz": {
        "label": "最大多普勒",
        "unit": "Hz",
        "minimum": 0.0,
        "maximum": 1200.0,
        "start": 200.0,
        "stop": 1200.0,
    },
    "ntn_delay_scale": {
        "label": "时延缩放",
        "unit": "倍",
        "minimum": 0.1,
        "maximum": 10.0,
        "start": 0.5,
        "stop": 2.0,
    },
    "ntn_velocity_kmh": {
        "label": "径向速度",
        "unit": "km/h",
        "minimum": 0.0,
        "maximum": 28800.0,
        "start": 0.0,
        "stop": 8000.0,
    },
    "ntn_radial_projection": {
        "label": "径向投影",
        "unit": "",
        "minimum": 0.0,
        "maximum": 1.0,
        "start": 0.0,
        "stop": 1.0,
    },
}


@dataclass(frozen=True)
class NTNScanPlan:
    parameter: str
    start: float
    stop: float
    points: int
    frames_per_point: int

    def __post_init__(self):
        if self.parameter not in NTN_SCAN_PARAMETERS:
            raise ValueError(f"Unsupported NTN scan parameter: {self.parameter}")
        if not math.isfinite(float(self.start)) or not math.isfinite(float(self.stop)):
            raise ValueError("Scan endpoints must be finite")
        spec = NTN_SCAN_PARAMETERS[self.parameter]
        if not spec["minimum"] <= float(self.start) <= spec["maximum"]:
            raise ValueError(f"Scan start must be within {spec['minimum']}..{spec['maximum']}")
        if not spec["minimum"] <= float(self.stop) <= spec["maximum"]:
            raise ValueError(f"Scan stop must be within {spec['minimum']}..{spec['maximum']}")
        if int(self.points) < 2 or int(self.points) > 101:
            raise ValueError("Scan point count must be between 2 and 101")
        if int(self.frames_per_point) < 1 or int(self.frames_per_point) > 10000:
            raise ValueError("Frames per point must be between 1 and 10000")
        object.__setattr__(self, "start", float(self.start))
        object.__setattr__(self, "stop", float(self.stop))
        object.__setattr__(self, "points", int(self.points))
        object.__setattr__(self, "frames_per_point", int(self.frames_per_point))

    @property
    def frame_count(self) -> int:
        return self.frames_per_point

    def values(self) -> list[float]:
        if self.points == 2:
            return [self.start, self.stop]
        step = (self.stop - self.start) / (self.points - 1)
        return [self.start + index * step for index in range(self.points - 1)] + [self.stop]

    def parameter_values(self) -> list[dict[str, float]]:
        return [{self.parameter: value} for value in self.values()]
