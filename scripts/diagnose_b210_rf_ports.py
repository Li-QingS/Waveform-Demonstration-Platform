"""Short, low-power B210 antenna-path probe (no FDIDM decoder involved).

This is a diagnostic, not an RF calibration.  It compares an RX-only baseline
with a 60 kHz baseband tone transmitted at a bounded, explicitly requested
UHD gain.  The default is the lowest TX gain; optional sweeps are rejected
outside the FDIDM host-side 0--45 dB limits and the tone is always stopped
before the USRP is released.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


# The UHD B210 gain range is considerably wider than is appropriate for a
# first antenna/loopback probe.  These are deliberately explicit *host-side*
# ceilings for the FDIDM validation path.  Keeping the limits here (rather
# than relying on the daughterboard reported range) prevents an accidental
# high-power sweep when a user passes a typo or a device reports an
# unexpectedly wide range.  45 dB is a configuration ceiling, not a guarantee
# that a direct RF connection is safe: retain the waveform peak limiter and
# use an external attenuator for TX/RX loopback.
SAFE_TX_GAIN_MIN_DB = 0.0
SAFE_TX_GAIN_MAX_DB = 45.0
SAFE_RX_GAIN_MIN_DB = 0.0
SAFE_RX_GAIN_MAX_DB = 45.0
DEFAULT_TX_GAINS_DB: Tuple[float, ...] = (0.0,)
DEFAULT_RX_GAINS_DB: Tuple[float, ...] = (20.0,)
MAX_GAIN_POINTS = 32
MAX_GAIN_COMBINATIONS = 64
UHD_REOPEN_RETRIES = 2
UHD_REOPEN_DELAY_S = 0.75


class GainPlanError(ValueError):
    """Raised when a requested gain sweep is malformed or unsafe."""


def _as_decimal(value: Any, *, name: str) -> Decimal:
    """Parse one gain token without accepting NaN/Infinity."""

    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise GainPlanError(f"{name} contains a non-numeric value: {value!r}") from exc
    if not number.is_finite():
        raise GainPlanError(f"{name} contains a non-finite value: {value!r}")
    return number


def _expand_gain_range(
    start: Decimal,
    stop: Decimal,
    step: Decimal,
    *,
    name: str,
    max_points: int = MAX_GAIN_POINTS,
) -> list[float]:
    if step == 0:
        raise GainPlanError(f"{name} range step must be non-zero")
    if start < stop and step < 0:
        raise GainPlanError(f"{name} range step must be positive for an ascending range")
    if start > stop and step > 0:
        raise GainPlanError(f"{name} range step must be negative for a descending range")

    # Decimal arithmetic avoids 0.1 + 0.2 accumulation and lets us include
    # the endpoint deterministically.  The point limit is checked while
    # expanding so a typo such as ``0:27:0.001`` cannot allocate unbounded
    # memory or start an unexpectedly long hardware run.
    values: list[float] = []
    current = start
    ascending = step > 0
    while (current <= stop if ascending else current >= stop):
        values.append(float(current))
        if len(values) > int(max_points):
            raise GainPlanError(
                f"{name} range contains more than {int(max_points)} points"
            )
        current += step
    # A non-integral step can leave the endpoint just outside the loop.  It is
    # intentionally not rounded up: scans must never silently exceed the
    # requested upper bound.
    return values


_RANGE_RE = re.compile(
    r"^\s*"
    r"(?P<start>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"\s*(?P<sep>:|\.\.|-)\s*"
    r"(?P<stop>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"(?:\s*:\s*(?P<step>[+-]?(?:\d+(?:\.\d*)?|\.\d+)))?"
    r"\s*$"
)


def parse_gain_spec(
    spec: Any,
    *,
    name: str = "gain",
    minimum_db: float = 0.0,
    maximum_db: float,
    max_points: int = MAX_GAIN_POINTS,
) -> Tuple[float, ...]:
    """Parse a comma/space separated gain list or inclusive range.

    Examples accepted by the CLI are ``"20,35,45"``, ``"20:45:1"`` and
    ``"20..45:1"``.  A two-value range defaults to a 1 dB step (or -1 dB for
    a descending range).  The function is intentionally public so the Qt
    layer and deterministic tests can validate a plan without importing UHD.
    Values are returned in the order requested, with duplicates removed.
    """

    if spec is None:
        return tuple()
    if isinstance(spec, (int, float, Decimal)):
        tokens = [str(spec)]
    elif isinstance(spec, str):
        text = spec.strip()
        if not text:
            raise GainPlanError(f"{name} cannot be empty")
        # Commas/semicolons separate entries.  Preserve whitespace inside a
        # range (``20 : 27 : 1``), while still allowing ordinary whitespace-
        # separated lists such as ``20 23 27``.
        tokens = []
        for chunk in re.split(r"[,;]+", text):
            chunk = chunk.strip()
            if not chunk:
                continue
            if _RANGE_RE.match(chunk):
                tokens.append(chunk)
            else:
                tokens.extend(token for token in re.split(r"\s+", chunk) if token)
    else:
        try:
            tokens = list(spec)
        except TypeError as exc:
            raise GainPlanError(f"unsupported {name} specification: {spec!r}") from exc

    values: list[float] = []
    minimum = _as_decimal(minimum_db, name=f"{name} minimum")
    maximum = _as_decimal(maximum_db, name=f"{name} maximum")
    if minimum > maximum:
        raise GainPlanError(f"{name} safety limits are reversed")

    for token in tokens:
        if isinstance(token, (tuple, list)) and len(token) in (2, 3):
            start = _as_decimal(token[0], name=name)
            stop = _as_decimal(token[1], name=name)
            step = _as_decimal(token[2], name=name) if len(token) == 3 else (
                Decimal("1") if stop >= start else Decimal("-1")
            )
            expanded = _expand_gain_range(
                start, stop, step, name=name, max_points=max_points
            )
        else:
            raw = str(token).strip()
            match = _RANGE_RE.match(raw)
            if match:
                start = _as_decimal(match.group("start"), name=name)
                stop = _as_decimal(match.group("stop"), name=name)
                step_text = match.group("step")
                step = _as_decimal(step_text, name=name) if step_text else (
                    Decimal("1") if stop >= start else Decimal("-1")
                )
                expanded = _expand_gain_range(
                    start, stop, step, name=name, max_points=max_points
                )
            else:
                expanded = [float(_as_decimal(raw, name=name))]

        for value in expanded:
            # Validate in Decimal space where possible, then use a finite
            # float for UHD's setter.  No clamping is performed: an unsafe
            # request is rejected rather than silently changed.
            value_decimal = _as_decimal(value, name=name)
            if value_decimal < minimum or value_decimal > maximum:
                raise GainPlanError(
                    f"{name} value {float(value_decimal):g} dB is outside the "
                    f"safe range [{float(minimum):g}, {float(maximum):g}] dB"
                )
            value_float = float(value_decimal)
            if not math.isfinite(value_float):
                raise GainPlanError(f"{name} contains a non-finite value")
            if value_float not in values:
                values.append(value_float)
            if len(values) > int(max_points):
                raise GainPlanError(
                    f"{name} contains more than {int(max_points)} points"
                )

    if not values:
        raise GainPlanError(f"{name} did not contain any gain values")
    return tuple(values)


def resolve_gain_plan(
    values_spec: Any,
    range_spec: Any,
    *,
    name: str,
    default: Sequence[float],
    minimum_db: float,
    maximum_db: float,
    start: Optional[float] = None,
    stop: Optional[float] = None,
    step: Optional[float] = None,
) -> Tuple[float, ...]:
    """Resolve list/range CLI forms into one validated gain tuple."""

    if values_spec is not None and range_spec is not None:
        raise GainPlanError(f"use either --{name}-gains or --{name}-gain-range, not both")
    if any(value is not None for value in (start, stop, step)):
        if values_spec is not None or range_spec is not None:
            raise GainPlanError(
                f"use one {name} gain specification form; range start/stop cannot be combined"
            )
        if start is None or stop is None:
            raise GainPlanError(f"{name} range requires both start and stop")
        if step is None:
            step = 1.0 if float(stop) >= float(start) else -1.0
        range_spec = f"{start}:{stop}:{step}"
    if values_spec is None and range_spec is None:
        values_spec = list(default)
    return parse_gain_spec(
        range_spec if range_spec is not None else values_spec,
        name=name,
        minimum_db=minimum_db,
        maximum_db=maximum_db,
    )


def validate_gain_plan(
    tx_gains: Iterable[float],
    rx_gains: Iterable[float],
    *,
    max_combinations: int = MAX_GAIN_COMBINATIONS,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Validate a complete scan plan before opening the USRP.

    This second check protects callers that bypass the CLI and call
    :func:`scan_tx_frontend`/the scan loop directly.  It also bounds total
    runtime: each combination starts and stops a UHD flowgraph.
    """

    tx = parse_gain_spec(
        list(tx_gains), name="TX gain", minimum_db=SAFE_TX_GAIN_MIN_DB,
        maximum_db=SAFE_TX_GAIN_MAX_DB,
    )
    rx = parse_gain_spec(
        list(rx_gains), name="RX gain", minimum_db=SAFE_RX_GAIN_MIN_DB,
        maximum_db=SAFE_RX_GAIN_MAX_DB,
    )
    combinations = len(tx) * len(rx)
    if combinations > int(max_combinations):
        raise GainPlanError(
            f"gain plan contains {combinations} combinations; limit is {int(max_combinations)}"
        )
    return tx, rx


@dataclass(frozen=True)
class DiagnosticResult:
    """Classified evidence for one low-power B210 TX/RX path.

    The classifier is deliberately independent of UHD so it can be used by
    the Qt worker and deterministic tests.  ``tone_delta_db`` is only one
    piece of evidence; a large delta with a clipped receiver is not a pass.
    """

    status: str
    reason: str
    tone_delta_db: float
    baseline_rms: float
    active_rms: float
    baseline_tone: float
    active_tone: float
    baseline_peak: float
    active_peak: float
    thresholds: Dict[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def classify_probe_observation(
    baseline: Mapping[str, Any],
    active: Mapping[str, Any],
    *,
    min_active_rms: float = 1.0e-5,
    min_tone_amplitude: float = 2.0e-5,
    min_delta_db: float = 3.0,
    wrong_port_delta_db: float = -3.0,
    clipping_peak: float = 0.95,
    excessive_noise_ratio: float = 8.0,
) -> DiagnosticResult:
    """Return a conservative RF path classification.

    ``wrong_port`` is used for a reproducibly suppressed tone (rather than
    treating every weak link as ``no_signal``).  ``inconclusive`` is reserved
    for missing/invalid probe evidence so the UI cannot claim a hardware pass.
    """

    def value(row: Mapping[str, Any], key: str) -> float:
        try:
            result = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            return float("nan")
        return result

    b_rms = value(baseline, "rms")
    a_rms = value(active, "rms")
    b_tone = value(baseline, "tone_amplitude")
    a_tone = value(active, "tone_amplitude")
    b_peak = value(baseline, "peak")
    a_peak = value(active, "peak")
    def _bad_continuity(row: Mapping[str, Any]) -> bool:
        try:
            if int(row.get("gap_count", 0) or 0) > 0 or int(row.get("stale_count", 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            return True
        return str(row.get("continuity", "")).lower() in {"gap", "stale", "discontinuous", "invalid"}

    if int(baseline.get("windows", 0) or 0) < 2 or int(active.get("windows", 0) or 0) < 2:
        status, reason = "inconclusive", "insufficient probe windows for continuity evidence"
        delta = float("nan")
    elif _bad_continuity(baseline) or _bad_continuity(active):
        delta = 20.0 * math.log10(max(a_tone, 1.0e-15) / max(b_tone, 1.0e-15)) if np.isfinite(b_tone) and np.isfinite(a_tone) else float("nan")
        status, reason = "inconclusive", "probe continuity has gaps or stale windows"
    elif not all(np.isfinite(v) for v in (b_rms, a_rms, b_tone, a_tone)):
        status, reason = "inconclusive", "missing or non-finite probe evidence"
        delta = float("nan")
    else:
        delta = 20.0 * math.log10(max(a_tone, 1.0e-15) / max(b_tone, 1.0e-15))
        if np.isfinite(a_peak) and a_peak >= float(clipping_peak):
            status, reason = "clipping", f"active peak {a_peak:.3f} >= {clipping_peak:.3f}"
        elif a_rms < float(min_active_rms) or a_tone < float(min_tone_amplitude):
            status, reason = "no_signal", "active RMS or tone is below the detect floor"
        elif b_rms > 0.0 and a_rms / b_rms >= float(excessive_noise_ratio) and a_tone < float(min_tone_amplitude) * 2.0:
            status, reason = "excessive_noise", "active energy is dominated by broadband noise"
        elif delta <= float(wrong_port_delta_db):
            status, reason = "wrong_port", f"tone is suppressed by {abs(delta):.2f} dB"
        elif delta < float(min_delta_db):
            status, reason = "inconclusive", f"tone gain {delta:.2f} dB is below the pass threshold"
        else:
            status, reason = "pass", f"tone gain {delta:.2f} dB with no clipping"
    thresholds = {
        "min_active_rms": float(min_active_rms),
        "min_tone_amplitude": float(min_tone_amplitude),
        "min_delta_db": float(min_delta_db),
        "wrong_port_delta_db": float(wrong_port_delta_db),
        "clipping_peak": float(clipping_peak),
        "excessive_noise_ratio": float(excessive_noise_ratio),
    }
    return DiagnosticResult(
        status=status,
        reason=reason,
        tone_delta_db=float(delta),
        baseline_rms=float(b_rms),
        active_rms=float(a_rms),
        baseline_tone=float(b_tone),
        active_tone=float(a_tone),
        baseline_peak=float(b_peak),
        active_peak=float(a_peak),
        thresholds=thresholds,
    )


def _tone_amplitude(samples: np.ndarray, sample_rate: float, tone_hz: float) -> float:
    values = np.asarray(samples, dtype=np.complex128).reshape(-1)
    if values.size < 128:
        return float("nan")
    window = np.hanning(values.size)
    phase = np.exp(-2j * np.pi * float(tone_hz) * np.arange(values.size) / float(sample_rate))
    return float(abs(np.sum(values * window * phase)) / max(float(np.sum(window)), 1.0))


def _read_probes(probes, sample_rate: float, tone_hz: float, seconds: float):
    observations = [[] for _ in probes]
    # ``probe_signal_vc`` exposes only the newest vector.  Keep a small amount
    # of polling metadata so repeated vectors and long polling gaps cannot be
    # mistaken for independent, contiguous RF observations.
    last_fingerprint = [None for _ in probes]
    last_update = [None for _ in probes]
    gap_counts = [0 for _ in probes]
    stale_counts = [0 for _ in probes]
    max_gap_s = [0.0 for _ in probes]
    time.sleep(0.15)
    count = max(4, int(max(seconds - 0.15, 0.2) / 0.05))
    for _ in range(count):
        for channel, probe in enumerate(probes):
            values = np.asarray(probe.level(), dtype=np.complex64).reshape(-1)
            if values.size:
                now = time.monotonic()
                fingerprint = (
                    int(values.size),
                    hash(values.tobytes()),
                )
                if last_fingerprint[channel] == fingerprint:
                    stale_counts[channel] += 1
                else:
                    if last_update[channel] is not None:
                        interval = max(0.0, now - last_update[channel])
                        max_gap_s[channel] = max(max_gap_s[channel], interval)
                        # The loop targets a 50 ms poll period.  A gap above
                        # 125 ms is sufficient evidence that one or more
                        # probe windows were missed by the diagnostic reader.
                        if interval > 0.125:
                            gap_counts[channel] += 1
                    last_fingerprint[channel] = fingerprint
                    last_update[channel] = now
                    values64 = values.astype(np.complex128)
                    observations[channel].append((
                        int(values.size),
                        float(np.sqrt(np.mean(np.abs(values64) ** 2))),
                        _tone_amplitude(values, sample_rate, tone_hz),
                        float(np.max(np.abs(values64))),
                    ))
        time.sleep(0.05)
    return [
        {
            "samples": int(np.median([row[0] for row in rows])) if rows else 0,
            "rms": float(np.median([row[1] for row in rows])) if rows else float("nan"),
            "tone_amplitude": float(np.median([row[2] for row in rows])) if rows else float("nan"),
            "peak": float(np.max([row[3] for row in rows])) if rows else float("nan"),
            "windows": len(rows),
            "polls": int(count),
            "fresh_windows": len(rows),
            "gap_count": int(gap_counts[channel]),
            "stale_count": int(stale_counts[channel]),
            "max_poll_gap_s": float(max_gap_s[channel]),
            "continuity": (
                "gap" if gap_counts[channel] else
                "stale" if stale_counts[channel] else
                "contiguous" if rows else "unknown"
            ),
        }
        for channel, rows in enumerate(observations)
    ]


def scan_tx_frontend(serial: str, tx_frontend: int, rx_antenna: str,
                     *, center_hz: float, sample_rate: float, tone_hz: float,
                     tone_amplitude: float, rx_gain: float, seconds: float,
                     tx_gain: float = SAFE_TX_GAIN_MIN_DB):
    """Probe one TX frontend at one validated TX/RX gain pair.

    The public helper remains usable by existing callers that omit
    ``tx_gain``; direct callers still get the same safety validation as the
    CLI before any UHD object is constructed.
    """

    # Reject unsafe direct calls too.  The CLI validates complete plans, but
    # this function is often imported by a worker or a notebook.
    (validated_tx,), (validated_rx,) = validate_gain_plan([tx_gain], [rx_gain], max_combinations=1)
    tx_gain = validated_tx
    rx_gain = validated_rx
    from gnuradio import analog, blocks, gr, uhd

    spec = "A:A A:B"  # B210 has two frontends on its one daughterboard.
    # UHD accepts an empty address for discovery, but a serial-only address
    # must not start with a comma.  The leading-comma form is used by some
    # GNU Radio constructors when appending a full address, yet UHD 4.5 on
    # Windows rejects it for this direct diagnostic path.
    args = f"serial={serial}" if serial else ""
    tb = gr.top_block("B210 antenna-path diagnostic", catch_exceptions=True)
    source = uhd.usrp_source(
        args, uhd.stream_args(cpu_format="fc32", channels=[0, 1])
    )
    source.set_subdev_spec(spec, 0)
    source.set_samp_rate(sample_rate)
    sink = uhd.usrp_sink(
        # B210 does not support a 2 RX / 1 TX stream configuration.  Keep both
        # TX lanes active, feeding zeros to the untested lane.
        args, uhd.stream_args(cpu_format="fc32", channels=[0, 1]), ""
    )
    sink.set_subdev_spec(spec, 0)
    sink.set_samp_rate(sample_rate)

    for channel in (0, 1):
        source.set_center_freq(center_hz, channel)
        source.set_antenna(rx_antenna, channel)
        source.set_gain(rx_gain, channel)
    for channel in (0, 1):
        sink.set_center_freq(center_hz, channel)
        sink.set_antenna("TX/RX", channel)
        sink.set_gain(tx_gain, channel)

    tone = analog.sig_source_c(sample_rate, analog.GR_COS_WAVE, tone_hz, tone_amplitude)
    gate = blocks.multiply_const_cc(0.0)
    tb.connect(tone, gate, (sink, int(tx_frontend)))
    tb.connect(blocks.null_source(gr.sizeof_gr_complex), (sink, 1 - int(tx_frontend)))
    probes = []
    for channel in (0, 1):
        vectorizer = blocks.stream_to_vector(gr.sizeof_gr_complex, 4096)
        probe = blocks.probe_signal_vc(4096)
        tb.connect((source, channel), vectorizer, probe)
        probes.append(probe)

    started = False
    try:
        tb.start()
        started = True
        baseline = _read_probes(probes, sample_rate, tone_hz, seconds)
        gate.set_k(1.0)
        active = _read_probes(probes, sample_rate, tone_hz, seconds)
        gate.set_k(0.0)
    finally:
        if started:
            try:
                gate.set_k(0.0)
            finally:
                tb.stop()
                tb.wait()

    rows = []
    for channel, (before, after) in enumerate(zip(baseline, active)):
        a = float(after["tone_amplitude"])
        b = float(before["tone_amplitude"])
        delta_db = 20.0 * math.log10(max(a, 1e-15) / max(b, 1e-15))
        rows.append({
            "tx_frontend": "AB"[int(tx_frontend)],
            "rx_frontend": "AB"[channel],
            "rx_antenna": rx_antenna,
            "tx_gain_db": float(tx_gain),
            "tx_digital_amplitude": float(tone_amplitude),
            "rx_gain_db": float(rx_gain),
            "gain_limits_db": {
                "tx_min": SAFE_TX_GAIN_MIN_DB,
                "tx_max": SAFE_TX_GAIN_MAX_DB,
                "rx_min": SAFE_RX_GAIN_MIN_DB,
                "rx_max": SAFE_RX_GAIN_MAX_DB,
            },
            "tone_delta_db": float(delta_db),
            "baseline": before,
                    "active": after,
        })
        result = classify_probe_observation(before, after)
        rows[-1].update({
            "status": result.status,
            "status_reason": result.reason,
            "status_thresholds": result.thresholds,
        })
    return rows


def _scan_with_retries(serial: str, tx_frontend: int, rx_antenna: str,
                       *, center_hz: float, sample_rate: float, tone_hz: float,
                       tone_amplitude: float, rx_gain: float, tx_gain: float,
                       seconds: float, retries: int = UHD_REOPEN_RETRIES):
    """Run one probe point while allowing UHD a bounded USB reopen window.

    B210 UHD objects can keep their USB endpoint alive briefly after
    ``top_block.wait()`` returns.  A sweep must not turn that transient into a
    process-wide failure, but it also must not retry forever or hide a real
    hardware error.  The final failure is returned as one structured row so a
    partial JSON log remains useful.
    """

    attempts = max(1, int(retries) + 1)
    last_error = ""
    for attempt in range(attempts):
        if attempt:
            time.sleep(float(UHD_REOPEN_DELAY_S))
            gc.collect()
        try:
            return scan_tx_frontend(
                serial, tx_frontend, rx_antenna,
                center_hz=center_hz,
                sample_rate=sample_rate,
                tone_hz=tone_hz,
                tone_amplitude=tone_amplitude,
                rx_gain=rx_gain,
                tx_gain=tx_gain,
                seconds=seconds,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            # A failed constructor can leave a UHD handle queued for release;
            # explicitly collect before the next bounded attempt.
            gc.collect()
    return [{
        "tx_frontend": "AB"[int(tx_frontend)],
        "rx_frontend": "unknown",
        "rx_antenna": rx_antenna,
        "tx_gain_db": float(tx_gain),
        "tx_digital_amplitude": float(tone_amplitude),
        "rx_gain_db": float(rx_gain),
        "gain_limits_db": {
            "tx_min": SAFE_TX_GAIN_MIN_DB,
            "tx_max": SAFE_TX_GAIN_MAX_DB,
            "rx_min": SAFE_RX_GAIN_MIN_DB,
            "rx_max": SAFE_RX_GAIN_MAX_DB,
        },
        "status": "inconclusive",
        "status_reason": f"UHD probe failed after {attempts} attempts: {last_error}",
        "error": last_error,
        "attempts": attempts,
    }]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="31896C6")
    parser.add_argument("--center-hz", type=float, default=2.4e9)
    parser.add_argument("--sample-rate", type=float, default=500_000.0)
    parser.add_argument("--tone-hz", type=float, default=60_000.0)
    parser.add_argument("--tone-amplitude", type=float, default=0.15)
    parser.add_argument(
        "--tx-gains",
        "--tx-gain",
        dest="tx_gains",
        metavar="LIST",
        help="TX gain list in dB, e.g. 20,35,45 (default: 0; host limit: 45)",
    )
    parser.add_argument(
        "--tx-gain-range",
        "--tx-range",
        dest="tx_gain_range",
        metavar="START:STOP[:STEP]",
        help="inclusive TX gain range in dB, e.g. 20:45:5 (host limit: 45)",
    )
    parser.add_argument("--tx-gain-start", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tx-gain-stop", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tx-gain-step", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--rx-gains",
        "--rx-gain",
        dest="rx_gains",
        metavar="LIST",
        help="RX gain list in dB, e.g. 20,35,45 (default: 20; host limit: 45); --rx-gain is a legacy alias",
    )
    parser.add_argument(
        "--rx-gain-range",
        "--rx-range",
        dest="rx_gain_range",
        metavar="START:STOP[:STEP]",
        help="inclusive RX gain range in dB, e.g. 20:45:5 (host limit: 45)",
    )
    parser.add_argument("--rx-gain-start", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rx-gain-stop", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rx-gain-step", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seconds", type=float, default=0.7)
    parser.add_argument("--rx-antenna", choices=("RX2", "TX/RX"), default="RX2")
    options = parser.parse_args()
    if not (0.0 < options.tone_hz < options.sample_rate / 2.0):
        parser.error("tone must lie inside the receiver Nyquist band")
    if not (0.0 < options.tone_amplitude <= 0.25):
        parser.error("digital tone amplitude must stay at or below 0.25")
    try:
        tx_gains = resolve_gain_plan(
            options.tx_gains,
            options.tx_gain_range,
            name="tx",
            default=DEFAULT_TX_GAINS_DB,
            minimum_db=SAFE_TX_GAIN_MIN_DB,
            maximum_db=SAFE_TX_GAIN_MAX_DB,
            start=options.tx_gain_start,
            stop=options.tx_gain_stop,
            step=options.tx_gain_step,
        )
        rx_gains = resolve_gain_plan(
            options.rx_gains,
            options.rx_gain_range,
            name="rx",
            default=DEFAULT_RX_GAINS_DB,
            minimum_db=SAFE_RX_GAIN_MIN_DB,
            maximum_db=SAFE_RX_GAIN_MAX_DB,
            start=options.rx_gain_start,
            stop=options.rx_gain_stop,
            step=options.rx_gain_step,
        )
        tx_gains, rx_gains = validate_gain_plan(tx_gains, rx_gains)
    except GainPlanError as exc:
        parser.error(str(exc))

    # Print each result as soon as its flowgraph is stopped.  This keeps a
    # long sweep observable and ensures a partial run still leaves usable JSON
    # evidence if a later gain point fails.
    for tx_gain in tx_gains:
        for rx_gain in rx_gains:
            for tx in (0, 1):
                rows = _scan_with_retries(
                    options.serial, tx, options.rx_antenna,
                    center_hz=options.center_hz,
                    sample_rate=options.sample_rate,
                    tone_hz=options.tone_hz,
                    tone_amplitude=options.tone_amplitude,
                    rx_gain=rx_gain,
                    tx_gain=tx_gain,
                    seconds=options.seconds,
                )
                for row in rows:
                    row["gain_plan"] = {
                        "tx_gains_db": list(tx_gains),
                        "rx_gains_db": list(rx_gains),
                        "max_combinations": MAX_GAIN_COMBINATIONS,
                    }
                    print(json.dumps(row, ensure_ascii=False, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
