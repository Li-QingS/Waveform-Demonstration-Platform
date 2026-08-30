import math

from waveform_sim.simulation.display_metrics import LogMetricResampler


def test_display_resampler_decimates_fast_points():
    r = LogMetricResampler(interval_s=0.5, ema_alpha=0.5)
    first = r.process({"ser": 1e-1}, now_monotonic=0.0)
    assert first is not None
    assert r.process({"ser": 1e-2}, now_monotonic=0.1) is None
    second = r.process({"ser": 1e-3}, now_monotonic=0.5)
    assert second is not None
    # log10 EMA: (-1 + -3)/2 = -2.
    assert math.isclose(second["ser"], 1e-2, rel_tol=1e-12)


def test_display_resampler_does_not_invent_positive_value_for_invalid_metric():
    r = LogMetricResampler(interval_s=0.1, ema_alpha=0.3)
    out = r.process({"ser": 0.0, "evm": float("nan")}, now_monotonic=0.0)
    assert math.isnan(out["ser"])
    assert math.isnan(out["evm"])
