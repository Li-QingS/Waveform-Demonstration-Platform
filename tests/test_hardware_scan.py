import pytest

from waveform_sim.hardware.hardware_scan import NTNScanPlan


def test_scan_plan_generates_inclusive_values_for_exactly_one_ntn_parameter():
    plan = NTNScanPlan(
        parameter="ntn_doppler_spread_hz",
        start=200,
        stop=1200,
        points=3,
        frames_per_point=10,
    )

    assert plan.values() == [200.0, 700.0, 1200.0]
    assert plan.frame_count == 10
    assert plan.parameter_values() == [
        {"ntn_doppler_spread_hz": 200.0},
        {"ntn_doppler_spread_hz": 700.0},
        {"ntn_doppler_spread_hz": 1200.0},
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"parameter": "unknown", "start": 0, "stop": 1, "points": 2, "frames_per_point": 1},
        {"parameter": "ntn_doppler_spread_hz", "start": float("nan"), "stop": 1, "points": 2, "frames_per_point": 1},
        {"parameter": "ntn_doppler_spread_hz", "start": 0, "stop": 1, "points": 1, "frames_per_point": 1},
        {"parameter": "ntn_doppler_spread_hz", "start": 0, "stop": 1, "points": 102, "frames_per_point": 1},
        {"parameter": "ntn_doppler_spread_hz", "start": 0, "stop": 1, "points": 2, "frames_per_point": 0},
    ],
)
def test_scan_plan_rejects_invalid_or_unbounded_sweeps(kwargs):
    with pytest.raises(ValueError):
        NTNScanPlan(**kwargs)
