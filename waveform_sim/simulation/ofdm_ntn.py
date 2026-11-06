"""Reproducible, guarded CP-OFDM experiments for a SISO NTN link.

The waveform is evaluated on a *linear* burst time line, including its cyclic
prefixes and quiet guards.  Fractional path delays evaluate each source OFDM
symbol at its actual delayed time; they are never rounded to sample indices.
Consequently a path outside the CP introduces inter-symbol interference.

The compact fading model draws one set of LOS/Rayleigh path coefficients per
seed and rotates them at constant per-path Doppler frequencies.  It is a
continuous sum-of-rays teaching model, not a full Jakes/3GPP channel emulator.
Frame timing is assisted (known); orbital pre-compensation is configured, and
neither receiver uses payload symbols to estimate its channel.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Iterable

import numpy as np

from waveform_sim.core.modem import bits_per_symbol, qam_demodulate, qam_modulate
from waveform_sim.simulation.leo_ntn_channel import NTN_PROFILES, get_profile


@dataclass(frozen=True)
class OfdmNtnConfig:
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
    estimator: str = "pilot_ls"
    equalizer: str = "mmse"
    seed: int = 42

    def validated(self) -> "OfdmNtnConfig":
        """Return a canonical configuration, rejecting ambiguous inputs."""
        cfg = replace(
            self, profile=str(self.profile).upper(),
            modulation=str(self.modulation).upper(),
            estimator=str(self.estimator).lower(), equalizer=str(self.equalizer).lower(),
        )
        if cfg.profile not in {"AWGN", *NTN_PROFILES}:
            raise ValueError(f"Unknown NTN profile: {cfg.profile}")
        for name in ("fft_len", "cp_len", "symbols_per_frame", "seed"):
            value = getattr(cfg, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if cfg.fft_len < 32 or cfg.fft_len > 2048 or cfg.fft_len & (cfg.fft_len - 1):
            raise ValueError("fft_len must be a power of two between 32 and 2048")
        if not 0 <= cfg.cp_len < cfg.fft_len:
            raise ValueError("cp_len must satisfy 0 <= cp_len < fft_len")
        if not 1 <= cfg.symbols_per_frame <= 256:
            raise ValueError("symbols_per_frame must be between 1 and 256")
        if cfg.seed < 0:
            raise ValueError("seed must be non-negative")
        if cfg.modulation not in {"QPSK", "16QAM", "64QAM"}:
            raise ValueError("modulation must be QPSK, 16QAM or 64QAM")
        if cfg.estimator not in {"pilot_ls", "ideal"}:
            raise ValueError("estimator must be pilot_ls or ideal")
        if cfg.equalizer not in {"zf", "mmse"}:
            raise ValueError("equalizer must be zf or mmse")
        for name in (
            "subcarrier_spacing_hz", "ebn0_db", "carrier_frequency_hz", "velocity_kmh",
            "radial_projection", "compensation_ratio", "residual_cfo_hz",
            "doppler_spread_hz", "delay_scale",
        ):
            if not np.isfinite(float(getattr(cfg, name))):
                raise ValueError(f"{name} must be finite")
        if cfg.subcarrier_spacing_hz <= 0 or cfg.carrier_frequency_hz <= 0:
            raise ValueError("Subcarrier spacing and carrier frequency must be positive")
        if cfg.velocity_kmh < 0 or cfg.doppler_spread_hz < 0 or cfg.delay_scale < 0:
            raise ValueError("Velocity, Doppler spread and delay scale must be non-negative")
        if not 0 <= cfg.radial_projection <= 1 or not 0 <= cfg.compensation_ratio <= 1:
            raise ValueError("Projection and compensation ratio must lie in [0, 1]")
        if not -100 <= cfg.ebn0_db <= 160:
            raise ValueError("ebn0_db must lie between -100 and 160 dB")
        return cfg

    def validate(self) -> "OfdmNtnConfig":
        return self.validated()


def _subcarrier_plan(fft_len: int):
    # Symmetric guard bands, a nulled DC carrier, comb pilots and data tones.
    edge = 3 * fft_len // 8
    active = np.concatenate((np.arange(-edge, 0), np.arange(1, edge + 1)))
    pilot_positions = np.unique(np.r_[np.arange(0, active.size, 4), active.size - 1])
    data_positions = np.setdiff1d(np.arange(active.size), pilot_positions)
    return active.astype(int), pilot_positions.astype(int), data_positions.astype(int)


def config_summary(config: OfdmNtnConfig) -> dict:
    cfg = config.validated()
    active, pilots, data = _subcarrier_plan(cfg.fft_len)
    rate = cfg.fft_len * cfg.subcarrier_spacing_hz
    orbit = (cfg.velocity_kmh / 3.6) * cfg.radial_projection * cfg.carrier_frequency_hz / 299_792_458.0
    residual = orbit * (1.0 - cfg.compensation_ratio) + cfg.residual_cfo_hz
    maximum_delay = (0.0 if cfg.profile == "AWGN" else
                     max(get_profile(cfg.profile).delays_ns) * 1e-9 * cfg.delay_scale)
    guard = cfg.cp_len
    frame_samples = cfg.symbols_per_frame * (cfg.fft_len + cfg.cp_len) + 2 * guard
    efficiency = cfg.symbols_per_frame * data.size / frame_samples
    spread = 0.0 if cfg.profile == "AWGN" else cfg.doppler_spread_hz
    return {
        "profile": cfg.profile, "sample_rate_hz": float(rate),
        "frame_samples": int(frame_samples), "frame_duration_s": float(frame_samples / rate),
        "orbit_doppler_hz": float(orbit), "residual_cfo_hz": float(residual),
        "extra_residual_cfo_hz": float(cfg.residual_cfo_hz),
        "normalized_cfo": float(residual / cfg.subcarrier_spacing_hz),
        "normalized_doppler": float(spread / cfg.subcarrier_spacing_hz),
        "cp_duration_s": float(cfg.cp_len / rate), "max_delay_s": float(maximum_delay),
        "cp_margin_s": float(cfg.cp_len / rate - maximum_delay),
        "max_delay_samples": float(maximum_delay * rate),
        "n_active_subcarriers": int(active.size), "n_data_subcarriers": int(data.size),
        "n_pilot_subcarriers": int(pilots.size),
        "payload_bits_per_frame": int(cfg.symbols_per_frame * data.size * bits_per_symbol(cfg.modulation)),
        "net_payload_efficiency": float(efficiency),
        "bits_per_sample": float(efficiency * bits_per_symbol(cfg.modulation)),
        "fading_assumption": "固定种子路径幅度与初相；各径按恒定多普勒连续旋转；辅助帧定时",
    }


def _path_realization(cfg: OfdmNtnConfig):
    if cfg.profile == "AWGN":
        return np.array([0.0]), np.array([1.0 + 0j]), np.array([0.0])
    profile = get_profile(cfg.profile)
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 1, 0]))
    powers = 10.0 ** (np.asarray(profile.powers_db, dtype=float) / 10.0)
    powers /= powers.sum()  # Normalize the PDP, never the random channel gain.
    rayleigh = (rng.standard_normal(powers.size) + 1j * rng.standard_normal(powers.size)) / math.sqrt(2.0)
    los_phases = rng.uniform(-np.pi, np.pi, powers.size)
    gains = np.sqrt(powers) * np.asarray([
        np.exp(1j * los_phases[i]) if kind.upper() == "LOS" else rayleigh[i]
        for i, kind in enumerate(profile.fading)
    ])
    dopplers = cfg.doppler_spread_hz * np.cos(rng.uniform(-np.pi, np.pi, powers.size))
    delays_s = np.asarray(profile.delays_ns, dtype=float) * 1e-9 * cfg.delay_scale
    return delays_s, gains, dopplers


def _linear_fractional_path(grid: np.ndarray, delay: float, cp_len: int, guard: int,
                            output_length: int) -> np.ndarray:
    """Evaluate x[n-delay] with the correct source symbol at every boundary.

    Within one OFDM symbol, evaluating at a fractional time is exactly a
    frequency-domain phase ramp.  The result is then placed at the *linear*
    source-symbol boundaries, so a missing CP is not silently synthesized.
    """
    n_symbols, fft_len = grid.shape
    symbol_len = fft_len + cp_len
    integer = int(math.floor(delay))
    fractional = float(delay - integer)
    if fractional < 1e-12:
        fractional = 0.0
    elif 1.0 - fractional < 1e-12:
        integer += 1
        fractional = 0.0
    frequencies = np.fft.fftfreq(fft_len)  # signed cycles/sample, including guard tones
    shifted = np.fft.ifft(grid * np.exp(-2j * np.pi * frequencies * fractional), axis=1, norm="ortho")
    # A fractional delay moves each source symbol boundary past an integer
    # sample: its first output then corresponds to source local time 1-frac.
    start = 0 if fractional == 0.0 else 1
    local = np.arange(start, start + symbol_len)
    periodic_indices = (local - cp_len) % fft_len
    output = np.zeros(output_length, dtype=np.complex128)
    for symbol in range(n_symbols):
        destination = guard + symbol * symbol_len + integer + start
        if destination >= output_length:
            continue
        count = min(symbol_len, output_length - destination)
        output[destination:destination + count] = shifted[symbol, periodic_indices[:count]]
    return output


def simulate_frame(config: OfdmNtnConfig, frame_index: int = 0) -> dict:
    """Simulate one deterministic frame with genuine Gray-QAM bit decisions.

    Eb is the energy of the complete transmitted burst (pilots and CP
    included) divided by the number of payload bits.  Receiver fading is
    absent from this noise calculation.  Ideal CSI is only the main diagonal
    of the physical channel; it does not undo ICI or ISI with an oracle.
    """
    cfg = config.validated()
    if not isinstance(frame_index, (int, np.integer)) or isinstance(frame_index, (bool, np.bool_)) or frame_index < 0:
        raise ValueError("frame_index must be a non-negative integer")
    info = config_summary(cfg)
    n, count, cp = cfg.fft_len, cfg.symbols_per_frame, cfg.cp_len
    active, pilots, data = _subcarrier_plan(n)
    bps = bits_per_symbol(cfg.modulation)
    # The stream tag precedes the frame index.  Placing it after the index
    # would make frame-1 bits collide with the path stream because trailing
    # zero entropy words do not distinguish NumPy SeedSequence inputs.
    bits_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 0, int(frame_index)]))
    noise_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 2, int(frame_index)]))
    tx_bits = bits_rng.integers(0, 2, count * data.size * bps, dtype=np.uint8)
    tx_symbols = qam_modulate(tx_bits, cfg.modulation).reshape(count, data.size)
    pilot_values = np.where((np.arange(pilots.size)[None, :] + np.arange(count)[:, None]) % 2, -1.0, 1.0)
    grid = np.zeros((count, n), dtype=np.complex128)
    grid[:, active[data] % n] = tx_symbols
    grid[:, active[pilots] % n] = pilot_values
    guard = cp
    length = info["frame_samples"]
    tx_samples = _linear_fractional_path(grid, 0.0, cp, guard, length)
    delays_s, gains, dopplers = _path_realization(cfg)
    delays = delays_s * info["sample_rate_hz"]
    times = (np.arange(length, dtype=float) + int(frame_index) * length) / info["sample_rate_hz"]
    residual = info["residual_cfo_hz"]
    rx_clean = np.zeros(length, dtype=np.complex128)
    true_channel = np.zeros((count, active.size), dtype=np.complex128)
    for delay, gain, doppler in zip(delays, gains, dopplers):
        frequency = float(doppler + residual)
        phase = np.exp(2j * np.pi * frequency * times)
        rx_clean += gain * phase * _linear_fractional_path(grid, float(delay), cp, guard, length)
        phase_useful = np.asarray([
            phase[guard + symbol * (n + cp) + cp:guard + symbol * (n + cp) + cp + n]
            for symbol in range(count)
        ])
        # Exclude delayed contributions from preceding symbols when computing
        # this symbol's own diagonal.  Those contributions remain in rx_clean.
        own_symbol = (cp + np.arange(n, dtype=float) - delay >= 0.0)
        diagonal_phase = np.mean(phase_useful * own_symbol[None, :], axis=1)
        true_channel += gain * diagonal_phase[:, None] * np.exp(-2j * np.pi * active[None, :] * delay / n)

    tx_energy = float(np.vdot(tx_samples, tx_samples).real)
    noise_variance = tx_energy / (tx_bits.size * 10.0 ** (cfg.ebn0_db / 10.0))
    unit_noise = (noise_rng.standard_normal(length) + 1j * noise_rng.standard_normal(length)) / math.sqrt(2.0)
    noise = math.sqrt(noise_variance) * unit_noise
    rx_samples = rx_clean + noise
    useful_indices = (guard + np.arange(count)[:, None] * (n + cp) + cp + np.arange(n)[None, :])
    received_grid = np.fft.fft(rx_samples[useful_indices], axis=1, norm="ortho")
    received_active = received_grid[:, active % n]
    if cfg.estimator == "ideal":
        estimate = true_channel.copy()
    else:
        pilot_estimates = received_active[:, pilots] / pilot_values
        estimate = np.asarray([
            np.interp(active, active[pilots], row.real) + 1j * np.interp(active, active[pilots], row.imag)
            for row in pilot_estimates
        ])
    power = np.abs(estimate) ** 2
    loading = noise_variance if cfg.equalizer == "mmse" else 0.0
    weights = np.conj(estimate) / np.maximum(power + loading, 1e-15)
    equalized = received_active * weights
    rx_symbols = equalized[:, data].reshape(-1)
    rx_bits = qam_demodulate(rx_symbols, cfg.modulation)
    bit_errors = int(np.count_nonzero(tx_bits != rx_bits))
    reference = tx_symbols.reshape(-1)
    reference_energy = float(np.vdot(reference, reference).real)
    error_energy = float(np.sum(np.abs(rx_symbols - reference) ** 2))
    channel_energy = float(np.sum(np.abs(true_channel) ** 2))
    channel_nmse = float(np.sum(np.abs(estimate - true_channel) ** 2) / max(channel_energy, 1e-30))
    return {
        "frame_index": int(frame_index), "frame_time_s": float(int(frame_index) * info["frame_duration_s"]),
        "bit_errors": bit_errors, "total_bits": int(tx_bits.size),
        "ber": float(bit_errors / tx_bits.size), "frame_error": int(bit_errors > 0),
        "evm_percent": float(100.0 * math.sqrt(error_energy / max(reference_energy, 1e-30))),
        "tx_symbols": reference, "rx_symbols": rx_symbols,
        "pre_eq_symbols": received_active[:, data].reshape(-1),
        "tx_bits": tx_bits, "rx_bits": rx_bits,
        "tx_samples": tx_samples, "rx_samples": rx_samples,
        "rx_clean_samples": rx_clean, "noise_samples": noise,
        "tx_energy": tx_energy, "noise_variance": float(noise_variance),
        "sample_rate_hz": info["sample_rate_hz"], "channel_nmse": channel_nmse,
        "channel_true": true_channel, "channel_est": estimate,
        "path_delays_samples": delays.copy(), "path_gains": gains.copy(),
        "path_dopplers_hz": dopplers.copy(), "summary": info,
    }


def ber_confidence_interval(errors: int, total_bits: int, z: float = 1.959963984540054):
    """Two-sided Wilson interval; zero errors retain a finite upper bound."""
    if total_bits <= 0:
        return float("nan"), float("nan")
    probability = errors / total_bits
    denominator = 1.0 + z * z / total_bits
    center = (probability + z * z / (2.0 * total_bits)) / denominator
    radius = z * math.sqrt(probability * (1.0 - probability) / total_bits + z * z / (4.0 * total_bits ** 2)) / denominator
    low = 0.0 if errors == 0 else max(0.0, center - radius)
    high = 1.0 if errors == total_bits else min(1.0, center + radius)
    return low, high


def run_study(config: OfdmNtnConfig, axis: str, values: Iterable[float], frames: int,
              stop_event=None, on_point: Callable[[dict], None] | None = None) -> list[dict]:
    """Compare pilot LS, ideal diagonal CSI and disabled pre-compensation.

    Each point uses the same frame-indexed bits, physical paths and unit-noise
    samples for all three receivers.  Counters are pooled before computing
    BER and Wilson bounds; EVM is the RMS of per-frame EVM percentages.  Raw
    zero BER is preserved.  The caller decides how to draw censored log points.
    Cancellation is checked before every frame and each receiver variant.
    """
    cfg = config.validated()
    fields = {"ebn0": "ebn0_db", "residual_cfo": "residual_cfo_hz",
              "spacing": "subcarrier_spacing_hz", "delay_scale": "delay_scale"}
    if axis not in fields:
        raise ValueError("axis must be ebn0, residual_cfo, spacing or delay_scale")
    if not isinstance(frames, (int, np.integer)) or isinstance(frames, (bool, np.bool_)) or frames <= 0:
        raise ValueError("frames must be a positive integer")
    points = []
    stopped = lambda: bool(stop_event is not None and stop_event.is_set())
    for value in values:
        if stopped():
            break
        point_cfg = replace(cfg, **{fields[axis]: float(value)}).validated()
        variants = {
            "pilot_ls": replace(point_cfg, estimator="pilot_ls"),
            "ideal": replace(point_cfg, estimator="ideal"),
            "uncompensated": replace(point_cfg, estimator="pilot_ls", compensation_ratio=0.0),
        }
        counters = {name: {"bit_errors": 0, "total_bits": 0, "frame_errors": 0,
                           "frames": 0, "evm_square_sum": 0.0} for name in variants}
        for frame in range(int(frames)):
            if stopped():
                break
            # Commit a frame only when all three variants have completed, so
            # a cancellation never compares mismatched simulation budgets.
            frame_results = {}
            for name, variant in variants.items():
                if stopped():
                    break
                frame_results[name] = simulate_frame(variant, frame)
            if len(frame_results) != len(variants):
                break
            for name, result in frame_results.items():
                aggregate = counters[name]
                aggregate["bit_errors"] += result["bit_errors"]
                aggregate["total_bits"] += result["total_bits"]
                aggregate["frame_errors"] += result["frame_error"]
                aggregate["frames"] += 1
                aggregate["evm_square_sum"] += result["evm_percent"] ** 2
        if not counters["pilot_ls"]["frames"]:
            break
        for aggregate in counters.values():
            aggregate["ber"] = aggregate["bit_errors"] / aggregate["total_bits"]
            aggregate["ci_low"], aggregate["ci_high"] = ber_confidence_interval(aggregate["bit_errors"], aggregate["total_bits"])
            aggregate["fer"] = aggregate["frame_errors"] / aggregate["frames"]
            aggregate["evm_percent"] = math.sqrt(aggregate.pop("evm_square_sum") / aggregate["frames"])
            aggregate["count"] = aggregate["frames"]
        point = {"axis": axis, "axis_value": float(value), "strategies": counters,
                 "cancelled": stopped(), "summary": config_summary(point_cfg)}
        points.append(point)
        if on_point is not None:
            on_point(point)
        if stopped():
            break
    return points
