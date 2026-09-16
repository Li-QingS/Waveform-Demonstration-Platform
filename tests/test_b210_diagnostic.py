"""Deterministic tests for the low-power B210 RF-path diagnostic."""

import importlib
import json
import sys

import pytest


diagnostic = importlib.import_module("scripts.diagnose_b210_rf_ports")


def test_gain_spec_accepts_lists_and_inclusive_ranges():
    assert diagnostic.parse_gain_spec(
        "20, 23,27", name="TX gain", maximum_db=27.0
    ) == (20.0, 23.0, 27.0)
    assert diagnostic.parse_gain_spec(
        "20:27:2", name="TX gain", maximum_db=27.0
    ) == (20.0, 22.0, 24.0, 26.0)
    assert diagnostic.parse_gain_spec(
        "27:20:-1", name="TX gain", maximum_db=27.0
    ) == tuple(float(value) for value in range(27, 19, -1))
    assert diagnostic.parse_gain_spec(
        "20 : 22 : 1", name="TX gain", maximum_db=27.0
    ) == (20.0, 21.0, 22.0)


def test_fdidm_host_gain_ceiling_accepts_45_db_and_rejects_above_it():
    assert diagnostic.SAFE_TX_GAIN_MAX_DB == 45.0
    assert diagnostic.SAFE_RX_GAIN_MAX_DB == 45.0
    assert diagnostic.parse_gain_spec(
        "45", name="TX gain", minimum_db=diagnostic.SAFE_TX_GAIN_MIN_DB,
        maximum_db=diagnostic.SAFE_TX_GAIN_MAX_DB,
    ) == (45.0,)
    assert diagnostic.parse_gain_spec(
        "45", name="RX gain", minimum_db=diagnostic.SAFE_RX_GAIN_MIN_DB,
        maximum_db=diagnostic.SAFE_RX_GAIN_MAX_DB,
    ) == (45.0,)
    with pytest.raises(diagnostic.GainPlanError, match="outside the safe range"):
        diagnostic.parse_gain_spec(
            "45.1", name="TX gain", minimum_db=diagnostic.SAFE_TX_GAIN_MIN_DB,
            maximum_db=diagnostic.SAFE_TX_GAIN_MAX_DB,
        )
    with pytest.raises(diagnostic.GainPlanError, match="outside the safe range"):
        diagnostic.parse_gain_spec(
            "45.1", name="RX gain", minimum_db=diagnostic.SAFE_RX_GAIN_MIN_DB,
            maximum_db=diagnostic.SAFE_RX_GAIN_MAX_DB,
        )


def test_gain_spec_rejects_unsafe_or_unbounded_requests():
    with pytest.raises(diagnostic.GainPlanError, match="outside the safe range"):
        diagnostic.parse_gain_spec("28", name="TX gain", maximum_db=27.0)
    with pytest.raises(diagnostic.GainPlanError, match="step must be non-zero"):
        diagnostic.parse_gain_spec("20:27:0", name="TX gain", maximum_db=27.0)
    with pytest.raises(diagnostic.GainPlanError, match="more than 3"):
        diagnostic.parse_gain_spec(
            "0:10:1", name="TX gain", maximum_db=27.0, max_points=3
        )


def test_gain_plan_defaults_and_conflicts_are_explicit():
    assert diagnostic.resolve_gain_plan(
        None, None, name="tx", default=(0.0,), minimum_db=0.0, maximum_db=27.0
    ) == (0.0,)
    assert diagnostic.resolve_gain_plan(
        None, None, name="rx", default=(20.0,), minimum_db=0.0, maximum_db=50.0
    ) == (20.0,)
    with pytest.raises(diagnostic.GainPlanError, match="either --tx-gains"):
        diagnostic.resolve_gain_plan(
            "20,25", "20:25:5", name="tx", default=(0.0,),
            minimum_db=0.0, maximum_db=27.0,
        )


def test_gain_plan_bounds_total_hardware_runs():
    with pytest.raises(diagnostic.GainPlanError, match="combinations"):
        diagnostic.validate_gain_plan(
            (0.0, 1.0, 2.0), (10.0, 11.0), max_combinations=4
        )


def test_main_passes_each_gain_pair_to_scan(monkeypatch, capsys):
    calls = []

    def fake_scan(serial, tx_frontend, rx_antenna, **kwargs):
        calls.append((serial, tx_frontend, rx_antenna, kwargs))
        return [{
            "tx_frontend": "AB"[tx_frontend],
            "rx_frontend": "A",
            "status": "pass",
        }]

    monkeypatch.setattr(diagnostic, "scan_tx_frontend", fake_scan)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose_b210_rf_ports.py",
            "--serial", "TEST",
            "--tx-gains", "20,27",
            "--rx-gain-range", "35:45:10",
        ],
    )
    assert diagnostic.main() == 0

    # Two gain pairs are tested on each of the two TX frontends.  The
    # deterministic fake avoids importing GNU Radio while proving the
    # requested values reach the hardware call boundary.
    assert [(item[1], item[3]["tx_gain"], item[3]["rx_gain"]) for item in calls] == [
        (0, 20.0, 35.0), (1, 20.0, 35.0),
        (0, 20.0, 45.0), (1, 20.0, 45.0),
        (0, 27.0, 35.0), (1, 27.0, 35.0),
        (0, 27.0, 45.0), (1, 27.0, 45.0),
    ]
    output_rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(output_rows) == len(calls)
    assert output_rows[0]["gain_plan"] == {
        "tx_gains_db": [20.0, 27.0],
        "rx_gains_db": [35.0, 45.0],
        "max_combinations": diagnostic.MAX_GAIN_COMBINATIONS,
    }


def _row(*, rms, tone, peak=0.1, windows=4, continuity="contiguous", gap_count=0, stale_count=0):
    return {
        "rms": rms,
        "tone_amplitude": tone,
        "peak": peak,
        "windows": windows,
        "continuity": continuity,
        "gap_count": gap_count,
        "stale_count": stale_count,
    }


def test_probe_classifier_accepts_clean_tone_gain():
    result = diagnostic.classify_probe_observation(
        _row(rms=0.01, tone=1.0e-4),
        _row(rms=0.03, tone=1.0e-3, peak=0.2),
    )
    assert result.status == "pass"
    assert result.tone_delta_db == pytest.approx(20.0)


def test_probe_classifier_reports_no_signal_below_detect_floor():
    result = diagnostic.classify_probe_observation(
        _row(rms=0.01, tone=1.0e-4),
        _row(rms=1.0e-7, tone=1.0e-8),
    )
    assert result.status == "no_signal"


def test_probe_classifier_reports_clipping_before_other_failures():
    result = diagnostic.classify_probe_observation(
        _row(rms=0.01, tone=1.0e-4),
        _row(rms=0.05, tone=4.0e-4, peak=0.99),
    )
    assert result.status == "clipping"
    assert "peak" in result.reason


def test_probe_classifier_distinguishes_suppressed_wrong_port():
    result = diagnostic.classify_probe_observation(
        _row(rms=0.01, tone=1.0e-3),
        _row(rms=0.01, tone=3.0e-4),
    )
    assert result.status == "wrong_port"


def test_probe_classifier_reports_broadband_excessive_noise():
    result = diagnostic.classify_probe_observation(
        _row(rms=0.01, tone=1.0e-3),
        _row(rms=0.20, tone=2.5e-5),
    )
    assert result.status == "excessive_noise"


def test_probe_classifier_rejects_nonfinite_evidence():
    result = diagnostic.classify_probe_observation(
        _row(rms=0.01, tone=1.0e-4),
        _row(rms=0.03, tone=float("nan")),
    )
    assert result.status == "inconclusive"
    assert "non-finite" in result.reason


@pytest.mark.parametrize(
    "field, value",
    [
        ("gap_count", 1),
        ("stale_count", 1),
        ("continuity", "gap"),
        ("continuity", "stale"),
    ],
)
def test_probe_classifier_rejects_discontinuous_evidence(field, value):
    active = _row(rms=0.03, tone=1.0e-3)
    active[field] = value
    result = diagnostic.classify_probe_observation(
        _row(rms=0.01, tone=1.0e-4),
        active,
    )
    assert result.status == "inconclusive"
    assert "continuity" in result.reason


def test_probe_classifier_requires_multiple_windows():
    result = diagnostic.classify_probe_observation(
        _row(rms=0.01, tone=1.0e-4, windows=1),
        _row(rms=0.03, tone=1.0e-3),
    )
    assert result.status == "inconclusive"
    assert "windows" in result.reason
