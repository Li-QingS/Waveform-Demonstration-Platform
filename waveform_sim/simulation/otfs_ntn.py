"""Rectangular CP-OTFS experiments on the same physical NTN paths as OFDM.

Payload QAM symbols occupy a delay/Doppler grid.  A unitary ISFFT maps them to
time/frequency, followed by ordinary CP-OFDM synthesis per time slot.  One
known, full-band training symbol precedes each payload grid.  Training LS holds
that estimate for the whole frame; it is intentionally a simple baseline with
no within-frame tracking.  The ideal reference knows only the TF channel's
main diagonal, not an oracle inverse of ICI or ISI.  Both receivers perform
one-tap TF equalization and a unitary SFFT back to DD.

This is an uncoded SISO teaching model with assisted timing.  It includes
fractional path delays and within-burst linear interference, not full 3GPP
fading statistics, synchronization acquisition or an advanced DD detector.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Iterable

import numpy as np

from waveform_sim.core.modem import bits_per_symbol, qam_demodulate, qam_modulate
from waveform_sim.simulation.leo_ntn_channel import NTN_PROFILES, get_profile
from waveform_sim.simulation.ofdm_ntn import (
    _linear_fractional_path, _path_realization, ber_confidence_interval,
)


_FLOAT_FIELDS = (
    "subcarrier_spacing_hz", "ebn0_db", "carrier_frequency_hz", "velocity_kmh",
    "radial_projection", "compensation_ratio", "residual_cfo_hz",
    "doppler_spread_hz", "delay_scale",
)
_AXIS_FIELDS = {
    "ebn0": "ebn0_db", "residual_cfo": "residual_cfo_hz",
    "spacing": "subcarrier_spacing_hz", "delay_scale": "delay_scale",
    "doppler": "doppler_spread_hz", "slots": "symbols_per_frame",
}
STRATEGIES = ("training_ls", "ideal", "uncompensated")


def _check_cancel(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("OTFS NTN experiment cancelled")


@dataclass(frozen=True)
class OtfsNtnConfig:
    profile: str = "NTN-TDLC5-200"
    fft_len: int = 128
    cp_len: int = 16
    subcarrier_spacing_hz: float = 30e3
    symbols_per_frame: int = 12
    modulation: str = "QPSK"
    ebn0_db: float = 10.0
    carrier_frequency_hz: float = 2e9
    velocity_kmh: float = 28080.0
    radial_projection: float = 0.5
    compensation_ratio: float = 0.999
    residual_cfo_hz: float = 0.0
    doppler_spread_hz: float = 200.0
    delay_scale: float = 1.0
    estimator: str = "training_ls"
    equalizer: str = "mmse"
    seed: int = 42

    def validated(self) -> "OtfsNtnConfig":
        """Canonicalize numeric fields and reject invalid/aliased requests."""
        numbers = {}
        for name in _FLOAT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a finite number")
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a finite number") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            numbers[name] = value
        integers = {}
        for name in ("fft_len", "cp_len", "symbols_per_frame", "seed"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            integers[name] = int(value)
        cfg = replace(
            self, profile=str(self.profile).upper(), modulation=str(self.modulation).upper(),
            estimator=str(self.estimator).lower(), equalizer=str(self.equalizer).lower(),
            **numbers, **integers,
        )
        if cfg.profile not in {"AWGN", *NTN_PROFILES}:
            raise ValueError(f"Unknown NTN profile: {cfg.profile}")
        if not 32 <= cfg.fft_len <= 2048 or cfg.fft_len & (cfg.fft_len - 1):
            raise ValueError("fft_len must be a power of two between 32 and 2048")
        if not 0 <= cfg.cp_len < cfg.fft_len:
            raise ValueError("cp_len must satisfy 0 <= cp_len < fft_len")
        if not 2 <= cfg.symbols_per_frame <= 64:
            raise ValueError("symbols_per_frame must be between 2 and 64")
        if cfg.fft_len * cfg.symbols_per_frame > 32768:
            raise ValueError("The DD grid must contain at most 32768 symbols")
        if cfg.seed < 0:
            raise ValueError("seed must be non-negative")
        if cfg.modulation not in {"QPSK", "16QAM", "64QAM"}:
            raise ValueError("modulation must be QPSK, 16QAM or 64QAM")
        if cfg.estimator not in {"training_ls", "ideal"}:
            raise ValueError("estimator must be training_ls or ideal")
        if cfg.equalizer not in {"mmse", "zf"}:
            raise ValueError("equalizer must be mmse or zf")
        if cfg.subcarrier_spacing_hz <= 0 or cfg.carrier_frequency_hz <= 0:
            raise ValueError("Subcarrier spacing and carrier frequency must be positive")
        if cfg.velocity_kmh < 0 or cfg.doppler_spread_hz < 0 or cfg.delay_scale < 0:
            raise ValueError("Velocity, Doppler spread and delay scale must be non-negative")
        if not 0 <= cfg.radial_projection <= 1 or not 0 <= cfg.compensation_ratio <= 1:
            raise ValueError("Projection and compensation ratio must lie in [0, 1]")
        if not -100 <= cfg.ebn0_db <= 160:
            raise ValueError("ebn0_db must lie between -100 and 160 dB")
        fs = cfg.fft_len * cfg.subcarrier_spacing_hz
        orbit = cfg.velocity_kmh / 3.6 * cfg.radial_projection * cfg.carrier_frequency_hz / 299_792_458.
        residual = orbit * (1. - cfg.compensation_ratio) + cfg.residual_cfo_hz
        spread = 0. if cfg.profile == "AWGN" else cfg.doppler_spread_hz
        if not math.isfinite(fs) or not math.isfinite(residual) or abs(residual) + spread >= fs / 2.:
            raise ValueError("Residual CFO plus path Doppler must remain below the sampling Nyquist frequency")
        return cfg

    def validate(self) -> "OtfsNtnConfig":
        return self.validated()


def otfs_isfft(dd: np.ndarray) -> np.ndarray:
    """Unitary ISFFT: DD (delay M, Doppler N) -> TF (time N, frequency M)."""
    dd = np.asarray(dd, dtype=np.complex128)
    if dd.ndim != 2 or min(dd.shape) < 1:
        raise ValueError("DD input must be a non-empty two-dimensional array")
    return np.fft.ifft(np.fft.fft(dd, axis=0, norm="ortho"), axis=1, norm="ortho").T


def otfs_sfft(tf: np.ndarray) -> np.ndarray:
    """Adjoint SFFT: TF (time N, frequency M) -> DD (delay M, Doppler N)."""
    tf = np.asarray(tf, dtype=np.complex128)
    if tf.ndim != 2 or min(tf.shape) < 1:
        raise ValueError("TF input must be a non-empty two-dimensional array")
    return np.fft.ifft(np.fft.fft(tf.T, axis=1, norm="ortho"), axis=0, norm="ortho")


def training_sequence(fft_len: int) -> np.ndarray:
    """A known unit-modulus full-band CAZAC training symbol (even FFT size)."""
    k = np.arange(fft_len, dtype=float)
    return np.exp(-1j * np.pi * k * k / fft_len)


def config_summary(config: OtfsNtnConfig) -> dict:
    cfg = config.validated()
    m, n, cp = cfg.fft_len, cfg.symbols_per_frame, cfg.cp_len
    rate = m * cfg.subcarrier_spacing_hz
    orbit = cfg.velocity_kmh / 3.6 * cfg.radial_projection * cfg.carrier_frequency_hz / 299_792_458.
    residual = orbit * (1. - cfg.compensation_ratio) + cfg.residual_cfo_hz
    maximum_delay = 0. if cfg.profile == "AWGN" else max(get_profile(cfg.profile).delays_ns) * 1e-9 * cfg.delay_scale
    frame_samples = (n + 1) * (m + cp) + 2 * cp
    payload_samples = n * (m + cp)
    spread = 0. if cfg.profile == "AWGN" else cfg.doppler_spread_hz
    return {
        "profile": cfg.profile, "model": "rectangular CP-OTFS, frame-start training LS / ideal diagonal CSI",
        "sample_rate_hz": float(rate), "frame_samples": int(frame_samples),
        "frame_duration_s": float(frame_samples / rate), "payload_duration_s": float(payload_samples / rate),
        "orbit_doppler_hz": float(orbit), "residual_cfo_hz": float(residual),
        "extra_residual_cfo_hz": cfg.residual_cfo_hz,
        "normalized_cfo": float(residual / cfg.subcarrier_spacing_hz),
        "normalized_doppler": float(spread / cfg.subcarrier_spacing_hz),
        "cp_duration_s": float(cp / rate), "max_delay_s": float(maximum_delay),
        "cp_margin_s": float(cp / rate - maximum_delay),
        "max_delay_samples": float(maximum_delay * rate),
        "delay_resolution_s": float(1. / rate),
        "doppler_resolution_hz": float(rate / payload_samples),
        "dd_shape": [int(m), int(n)], "training_symbols": 1,
        "n_active_subcarriers": int(m), "n_data_subcarriers": int(m),
        "n_pilot_subcarriers": 0, "n_data_symbols": int(m * n),
        "payload_bits_per_frame": int(m * n * bits_per_symbol(cfg.modulation)),
        "net_payload_efficiency": float(m * n / frame_samples),
        "bits_per_sample": float(m * n * bits_per_symbol(cfg.modulation) / frame_samples),
        "training_fraction_of_tf_symbols": float(1. / (n + 1)),
        "guard_samples": int(2 * cp), "cyclic_prefix_samples": int((n + 1) * cp),
        "fading_assumption": "固定种子路径幅度与初相；各径恒定多普勒连续旋转；辅助帧定时",
        "receiver_assumption": "每帧一个已知全带宽训练符号，LS 整帧保持；TF 单抽头均衡后 SFFT；理想参考也不消除全部 ICI/ISI",
    }


def simulate_frame(config: OtfsNtnConfig, frame_index: int = 0, stop_event=None) -> dict:
    """Generate, transmit and score one actual OTFS frame, preserving raw BER.

    Orbital residual CFO and per-path Doppler use absolute sample time.  Only
    the PDP's expected energy is normalized, never received fading power.
    Each frame is an independently guarded burst: tails interfere with later
    symbols of that burst but do not carry into the next burst.
    """
    _check_cancel(stop_event)
    cfg = config.validated()
    if isinstance(frame_index, (bool, np.bool_)) or not isinstance(frame_index, (int, np.integer)) or frame_index < 0:
        raise ValueError("frame_index must be a non-negative integer")
    frame_index = int(frame_index)
    info = config_summary(cfg)
    m, n, cp = cfg.fft_len, cfg.symbols_per_frame, cfg.cp_len
    length, rate = info["frame_samples"], info["sample_rate_hz"]
    bits_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 0, frame_index]))
    noise_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 2, frame_index]))
    tx_bits = bits_rng.integers(0, 2, m * n * bits_per_symbol(cfg.modulation), dtype=np.uint8)
    tx_dd = qam_modulate(tx_bits, cfg.modulation).reshape(m, n)
    tx_tf = otfs_isfft(tx_dd)
    training = training_sequence(m)
    grid = np.vstack((training[None, :], tx_tf))
    tx_samples = _linear_fractional_path(grid, 0., cp, cp, length)
    delays_s, gains, dopplers = _path_realization(cfg)
    delays = delays_s * rate
    times = (np.arange(length, dtype=float) + frame_index * length) / rate
    residual = info["residual_cfo_hz"]
    rx_clean = np.zeros(length, dtype=np.complex128)
    true_channel_all = np.zeros_like(grid)
    frequencies = np.fft.fftfreq(m)
    indices = cp + np.arange(n + 1)[:, None] * (m + cp) + cp + np.arange(m)[None, :]
    for delay, gain, doppler in zip(delays, gains, dopplers):
        _check_cancel(stop_event)
        phase = np.exp(2j * np.pi * float(doppler + residual) * times)
        rx_clean += gain * phase * _linear_fractional_path(grid, float(delay), cp, cp, length)
        own_symbol = cp + np.arange(m, dtype=float) - delay >= 0.
        diagonal_phase = np.mean(phase[indices] * own_symbol[None, :], axis=1)
        true_channel_all += gain * diagonal_phase[:, None] * np.exp(-2j * np.pi * frequencies[None, :] * delay)
    _check_cancel(stop_event)
    tx_energy = float(np.vdot(tx_samples, tx_samples).real)
    variance = tx_energy / (tx_bits.size * 10. ** (cfg.ebn0_db / 10.))
    unit_noise = (noise_rng.standard_normal(length) + 1j * noise_rng.standard_normal(length)) / math.sqrt(2.)
    noise = math.sqrt(variance) * unit_noise
    rx_samples = rx_clean + noise
    received_all = np.fft.fft(rx_samples[indices], axis=1, norm="ortho")
    training_estimate = received_all[0] / training
    rx_tf = received_all[1:]
    true_channel = true_channel_all[1:]
    if cfg.estimator == "ideal":
        estimate = true_channel.copy()
    else:
        estimate = np.broadcast_to(training_estimate[None, :], (n, m)).copy()
    power = np.abs(estimate) ** 2
    loading = variance if cfg.equalizer == "mmse" else 0.
    weights = np.conj(estimate) / np.maximum(power + loading, 1e-30)
    equalized_tf = rx_tf * weights
    rx_dd = otfs_sfft(equalized_tf)
    pre_eq_dd = otfs_sfft(rx_tf)
    rx_symbols = rx_dd.reshape(-1)
    rx_bits = qam_demodulate(rx_symbols, cfg.modulation)
    errors = int(np.count_nonzero(tx_bits != rx_bits))
    reference = tx_dd.reshape(-1)
    reference_energy = float(np.vdot(reference, reference).real)
    error_energy = float(np.sum(np.abs(rx_symbols - reference) ** 2))
    channel_energy = float(np.sum(np.abs(true_channel) ** 2))
    nmse = float(np.sum(np.abs(estimate - true_channel) ** 2) / max(channel_energy, 1e-30))
    _check_cancel(stop_event)
    return {
        "frame_index": frame_index, "frame_time_s": float(frame_index * info["frame_duration_s"]),
        "bit_errors": errors, "total_bits": int(tx_bits.size),
        "ber": float(errors / tx_bits.size), "frame_error": int(errors > 0),
        "evm_percent": float(100. * math.sqrt(error_energy / reference_energy)),
        "error_energy": error_energy, "reference_energy": reference_energy,
        "tx_symbols": reference, "rx_symbols": rx_symbols, "pre_eq_symbols": pre_eq_dd.reshape(-1),
        "tx_bits": tx_bits, "rx_bits": rx_bits,
        "tx_dd": tx_dd, "rx_dd": rx_dd, "pre_eq_dd": pre_eq_dd,
        "tx_tf": tx_tf, "rx_tf": rx_tf, "equalized_tf": equalized_tf,
        "training_tx_symbols": training, "training_rx_symbols": received_all[0],
        "training_channel_est": training_estimate,
        "tx_samples": tx_samples, "rx_samples": rx_samples, "rx_clean_samples": rx_clean,
        "noise_samples": noise, "unit_noise_samples": unit_noise,
        "tx_energy": tx_energy, "noise_variance": float(variance),
        "sample_rate_hz": rate, "channel_nmse": nmse,
        "channel_true": true_channel, "channel_est": estimate,
        "training_channel_true": true_channel_all[0],
        "path_delays_samples": delays.copy(), "path_gains": gains.copy(),
        "path_dopplers_hz": dopplers.copy(), "summary": info,
    }


def _point_configs(config: OtfsNtnConfig, axis: str, values: Iterable[float]):
    if axis not in _AXIS_FIELDS:
        raise ValueError("axis must be ebn0, residual_cfo, spacing, delay_scale, doppler or slots")
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError("values must be a non-empty iterable of finite numbers") from exc
    if not values:
        raise ValueError("values must not be empty")
    points = []
    seen = set()
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Scan values must be finite numbers")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Scan values must be finite numbers") from exc
        if not math.isfinite(value):
            raise ValueError("Scan values must be finite")
        if value in seen:
            raise ValueError("Scan values must not repeat")
        seen.add(value)
        if axis == "slots" and value != int(value):
            raise ValueError("slots scan values must be integers")
        point = replace(config, **{_AXIS_FIELDS[axis]: int(value) if axis == "slots" else value}).validated()
        # Validate all three variants before running *any* point, including
        # the unsuppressed orbital CFO's sampling Nyquist limit.
        variants = {
            "training_ls": replace(point, estimator="training_ls").validated(),
            "ideal": replace(point, estimator="ideal").validated(),
            "uncompensated": replace(point, estimator="training_ls", compensation_ratio=0.).validated(),
        }
        points.append((value, point, variants))
    return points


def run_study(config: OtfsNtnConfig, axis: str, values: Iterable[float], frames: int,
              stop_event=None, on_point: Callable[[dict], None] | None = None) -> list[dict]:
    """Run paired receivers on reproducible streams and publish complete points.

    Bit counters, FER and reference-energy-weighted EVM pool actual decisions;
    Wilson intervals are conditional bit-count approximations, not confidence
    intervals across independent fading trajectories.  Changing slot count or
    spacing also changes the physical duration, budget and DD resolutions.
    """
    cfg = config.validated()
    if isinstance(frames, (bool, np.bool_)) or not isinstance(frames, (int, np.integer)) or frames < 1:
        raise ValueError("frames must be a positive integer")
    prepared = _point_configs(cfg, axis, values)
    output = []
    for value, point_cfg, variants in prepared:
        counters = {name: {"bit_errors": 0, "total_bits": 0, "frame_errors": 0,
                           "frames": 0, "error_energy": 0., "reference_energy": 0.}
                    for name in STRATEGIES}
        try:
            for frame in range(int(frames)):
                _check_cancel(stop_event)
                results = {}
                for name, variant in variants.items():
                    _check_cancel(stop_event)
                    results[name] = simulate_frame(variant, frame, stop_event=stop_event)
                _check_cancel(stop_event)
                for name, result in results.items():
                    aggregate = counters[name]
                    for key in ("bit_errors", "total_bits", "error_energy", "reference_energy"):
                        aggregate[key] += result[key]
                    aggregate["frame_errors"] += result["frame_error"]
                    aggregate["frames"] += 1
            _check_cancel(stop_event)
        except InterruptedError:
            break  # Preserve only points whose entire configured budget ran.
        for aggregate in counters.values():
            aggregate["ber"] = aggregate["bit_errors"] / aggregate["total_bits"]
            aggregate["ci_low"], aggregate["ci_high"] = ber_confidence_interval(aggregate["bit_errors"], aggregate["total_bits"])
            aggregate["fer"] = aggregate["frame_errors"] / aggregate["frames"]
            aggregate["evm_percent"] = 100. * math.sqrt(aggregate["error_energy"] / aggregate["reference_energy"])
            aggregate["count"] = aggregate["frames"]
        point = {"axis": axis, "axis_value": float(value), "strategies": counters,
                 "cancelled": False, "summary": config_summary(point_cfg)}
        output.append(point)
        if on_point is not None:
            on_point(point)
    return output
