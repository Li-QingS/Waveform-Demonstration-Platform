"""Standard-convention DAFT/CPP AFDM on reproducible physical NTN paths.

DAFT uses exp(-j*2*pi*c*n**2), WITHOUT the legacy core's division by M.
Each burst has an ordinary-CP OFDM training block followed by AFDM blocks.
Training LS provides a static circular-channel approximation with CPP wrap
correction.  The ideal reference knows each block's complete own-block matrix,
including its time-varying paths, but does not cancel inter-block interference.

Fractional paths use Nyquist Fourier interpolation of each actual IDAFT time
block.  CPP prefix components have their shifted frequencies wrapped to the
Nyquist interval before interpolation.  In particular, sample-identical CP and
CPP have identical fractional propagation when 2*M*c1 is integral (even M).
Actual finite source intervals exclude fictitious prefixes; bursts are separate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
from typing import Callable, Iterable

import numpy as np

from waveform_sim.core.modem import bits_per_symbol, qam_demodulate, qam_modulate
from waveform_sim.simulation.leo_ntn_channel import NTN_PROFILES, get_profile
from waveform_sim.simulation.ofdm_ntn import _path_realization, ber_confidence_interval
from waveform_sim.simulation.otfs_ntn import training_sequence


STRATEGIES = ("training_ls", "ideal", "uncompensated")
_AXIS_FIELDS = {
    "ebn0": "ebn0_db", "residual_cfo": "residual_cfo_hz", "doppler": "doppler_spread_hz",
    "spacing": "subcarrier_spacing_hz", "delay_scale": "delay_scale",
    "c1": "c1", "c2": "c2", "blocks": "symbols_per_frame",
}
_FLOAT_FIELDS = (
    "subcarrier_spacing_hz", "ebn0_db", "carrier_frequency_hz", "velocity_kmh",
    "radial_projection", "compensation_ratio", "residual_cfo_hz", "doppler_spread_hz",
    "delay_scale", "c1", "c2",
)


def _check_cancel(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("AFDM NTN experiment cancelled")


@dataclass(frozen=True)
class AfdmNtnConfig:
    profile: str = "NTN-TDLC5-200"
    fft_len: int = 64
    cp_len: int = 16
    subcarrier_spacing_hz: float = 30e3
    symbols_per_frame: int = 8
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
    c1: float = 3.0 / 128.0
    c2: float = 1.0 / (4096.0 * math.sqrt(2.0))
    prefix: str = "cpp"

    def validated(self) -> "AfdmNtnConfig":
        numeric = {}
        for name in _FLOAT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be finite")
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be finite") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            numeric[name] = value
        integral = {}
        for name in ("fft_len", "cp_len", "symbols_per_frame", "seed"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            integral[name] = int(value)
        cfg = replace(self, **numeric, **integral, profile=str(self.profile).upper(),
                      modulation=str(self.modulation).upper(), estimator=str(self.estimator).lower(),
                      equalizer=str(self.equalizer).lower(), prefix=str(self.prefix).lower())
        if cfg.profile not in {"AWGN", *NTN_PROFILES}:
            raise ValueError(f"Unknown NTN profile: {cfg.profile}")
        if not 32 <= cfg.fft_len <= 256 or cfg.fft_len & (cfg.fft_len - 1):
            raise ValueError("fft_len must be a power of two between 32 and 256")
        if not 0 <= cfg.cp_len < cfg.fft_len:
            raise ValueError("cp_len must satisfy 0 <= cp_len < fft_len")
        if not 1 <= cfg.symbols_per_frame <= 32:
            raise ValueError("symbols_per_frame must be between 1 and 32")
        if cfg.seed < 0:
            raise ValueError("seed must be non-negative")
        if cfg.modulation not in {"QPSK", "16QAM", "64QAM"}:
            raise ValueError("modulation must be QPSK, 16QAM or 64QAM")
        if cfg.estimator not in {"training_ls", "ideal"}:
            raise ValueError("estimator must be training_ls or ideal")
        if cfg.equalizer not in {"zf", "mmse"}:
            raise ValueError("equalizer must be zf or mmse")
        if cfg.prefix not in {"cpp", "cp"}:
            raise ValueError("prefix must be cpp or cp")
        if not 0. <= cfg.c1 <= .5 or not 0. <= cfg.c2 <= .5:
            raise ValueError("c1 and c2 must lie in [0, 0.5] in the standard DAFT convention")
        if cfg.subcarrier_spacing_hz <= 0. or cfg.carrier_frequency_hz <= 0.:
            raise ValueError("Subcarrier spacing and carrier frequency must be positive")
        if min(cfg.velocity_kmh, cfg.doppler_spread_hz, cfg.delay_scale) < 0.:
            raise ValueError("Velocity, Doppler spread and delay scale must be non-negative")
        if not 0. <= cfg.radial_projection <= 1. or not 0. <= cfg.compensation_ratio <= 1.:
            raise ValueError("Projection and compensation ratio must lie in [0, 1]")
        if not -100. <= cfg.ebn0_db <= 160.:
            raise ValueError("ebn0_db must lie between -100 and 160 dB")
        fs = cfg.fft_len * cfg.subcarrier_spacing_hz
        orbit = cfg.velocity_kmh / 3.6 * cfg.radial_projection * cfg.carrier_frequency_hz / 299_792_458.
        residual = orbit * (1. - cfg.compensation_ratio) + cfg.residual_cfo_hz
        spread = 0. if cfg.profile == "AWGN" else cfg.doppler_spread_hz
        if not math.isfinite(fs) or not math.isfinite(residual) or abs(residual) + spread >= fs / 2.:
            raise ValueError("Residual CFO plus path Doppler must remain below the sampling Nyquist frequency")
        return cfg

    def validate(self):
        return self.validated()


def afdm_modulate(symbols, c1: float, c2: float):
    """IDAFT on the last axis, using the paper's unnormalized chirp rates."""
    symbols = np.asarray(symbols, dtype=np.complex128)
    if symbols.ndim < 1 or symbols.shape[-1] < 1:
        raise ValueError("symbols must have a non-empty final dimension")
    n = np.arange(symbols.shape[-1], dtype=float)
    return np.fft.ifft(symbols * np.exp(2j * np.pi * c2 * n**2), axis=-1, norm="ortho") * np.exp(2j * np.pi * c1 * n**2)


def afdm_demodulate(samples, c1: float, c2: float):
    """DAFT, adjoint to :func:`afdm_modulate`, along the last axis."""
    samples = np.asarray(samples, dtype=np.complex128)
    if samples.ndim < 1 or samples.shape[-1] < 1:
        raise ValueError("samples must have a non-empty final dimension")
    n = np.arange(samples.shape[-1], dtype=float)
    return np.fft.fft(samples * np.exp(-2j * np.pi * c1 * n**2), axis=-1, norm="ortho") * np.exp(-2j * np.pi * c2 * n**2)


def chirp_prefix(samples, cp_len: int, c1: float, prefix: str = "cpp"):
    """Prepend ordinary CP or the actual chirp-periodic prefix on the last axis."""
    samples = np.asarray(samples, dtype=np.complex128)
    if samples.ndim < 1 or isinstance(cp_len, (bool, np.bool_)) or not isinstance(cp_len, (int, np.integer)) or not 0 <= cp_len < samples.shape[-1]:
        raise ValueError("cp_len must be an integer smaller than the block length")
    if prefix not in {"cpp", "cp"}:
        raise ValueError("prefix must be cpp or cp")
    if not cp_len:
        return samples.copy()
    m = samples.shape[-1]
    negative = np.arange(-cp_len, 0, dtype=float)
    correction = np.exp(-2j * np.pi * c1 * (m*m + 2*m*negative)) if prefix == "cpp" else np.ones(cp_len)
    return np.concatenate((samples[..., -cp_len:] * correction, samples), axis=-1)


@lru_cache(maxsize=64)
def _transform_matrices(m, c1, c2):
    transmit = afdm_modulate(np.eye(m), c1, c2).T
    receive = transmit.conj().T.copy()
    transmit.setflags(write=False)
    receive.setflags(write=False)
    return transmit, receive


def suggested_chirp(config: AfdmNtnConfig) -> dict:
    """An integer-bound engineering suggestion, never a diversity guarantee."""
    cfg = config.validated()
    fs = cfg.fft_len * cfg.subcarrier_spacing_hz
    orbit = cfg.velocity_kmh / 3.6 * cfg.radial_projection * cfg.carrier_frequency_hz / 299_792_458.
    residual = orbit * (1. - cfg.compensation_ratio) + cfg.residual_cfo_hz
    spread = 0. if cfg.profile == "AWGN" else cfg.doppler_spread_hz
    alpha = int(math.ceil((abs(residual) + spread) / cfg.subcarrier_spacing_hz))
    maximum_delay = 0. if cfg.profile == "AWGN" else max(get_profile(cfg.profile).delays_ns) * 1e-9 * cfg.delay_scale
    delay_bound = int(math.ceil(maximum_delay * fs))
    required = (2*alpha + 1) * (delay_bound + 1)
    return {"c1": float((2*alpha + 1)/(2*cfg.fft_len)),
            "c2": float(1./(cfg.fft_len**2 * math.sqrt(2.))),
            "doppler_bin_bound": alpha, "delay_sample_bound": delay_bound,
            "integer_support_width": required, "integer_support_fits": required <= cfg.fft_len,
            "assumption": "Integer support bound only; fractional paths, detector and finite precision prevent a full-diversity guarantee."}


def config_summary(config: AfdmNtnConfig) -> dict:
    cfg = config.validated()
    m, n, cp = cfg.fft_len, cfg.symbols_per_frame, cfg.cp_len
    fs = m * cfg.subcarrier_spacing_hz
    orbit = cfg.velocity_kmh / 3.6 * cfg.radial_projection * cfg.carrier_frequency_hz / 299_792_458.
    residual = orbit * (1. - cfg.compensation_ratio) + cfg.residual_cfo_hz
    delay = 0. if cfg.profile == "AWGN" else max(get_profile(cfg.profile).delays_ns) * 1e-9 * cfg.delay_scale
    length = (n + 1) * (m + cp) + 2*cp
    spread = 0. if cfg.profile == "AWGN" else cfg.doppler_spread_hz
    correction = np.exp(-2j*np.pi*cfg.c1*(m*m + 2*m*np.arange(-cp, 0, dtype=float)))
    return {
        "profile": cfg.profile, "model": "standard DAFT AFDM with CPP/CP, training LS / ideal own-block full-matrix CSI",
        "sample_rate_hz": float(fs), "frame_samples": int(length), "frame_duration_s": float(length/fs),
        "payload_duration_s": float(n*(m+cp)/fs), "orbit_doppler_hz": float(orbit),
        "residual_cfo_hz": float(residual), "extra_residual_cfo_hz": cfg.residual_cfo_hz,
        "normalized_cfo": float(residual/cfg.subcarrier_spacing_hz), "normalized_doppler": float(spread/cfg.subcarrier_spacing_hz),
        "cp_duration_s": float(cp/fs), "cpp_duration_s": float(cp/fs), "max_delay_s": float(delay),
        "cp_margin_s": float(cp/fs-delay), "max_delay_samples": float(delay*fs),
        "n_active_subcarriers": m, "n_data_subcarriers": m, "n_data_symbols": m*n,
        "n_pilot_subcarriers": 0, "training_symbols": 1, "data_blocks": n,
        "payload_bits_per_frame": int(m*n*bits_per_symbol(cfg.modulation)),
        "net_payload_efficiency": float(m*n/length), "bits_per_sample": float(m*n*bits_per_symbol(cfg.modulation)/length),
        "guard_samples": 2*cp, "cyclic_prefix_samples": (n+1)*cp,
        "training_fraction_of_blocks": float(1./(n+1)), "prefix": cfg.prefix,
        "chirp_c1": cfg.c1, "chirp_c2": cfg.c2, "chirp_convention": "exp(-j 2 pi c n^2), no division by M",
        "cpp_matches_cp": bool(np.allclose(correction, 1., atol=1e-10, rtol=0.)),
        "cpp_matches_cp_scope": "Integer prefix identity; integral 2*M*c1 with even M also gives identical fractional propagation.",
        "fading_assumption": "固定种子路径幅度与初相；各径恒定多普勒连续旋转；辅助帧定时",
        "receiver_assumption": "训练 LS 估计静态循环信道并作 CPP 包裹修正；全矩阵 ZF/MMSE。理想 CSI 仅含本数据块，不消除跨块 ISI。",
        "fractional_extension": "Piecewise Nyquist Fourier interpolation of sampled time blocks, frequency-aliased CPP prefix, finite source intervals and independent bursts.",
        "chirp_suggestion": suggested_chirp(cfg),
    }


@lru_cache(maxsize=128)
def _interpolated_basis(times, m, c1, c2, prefix):
    """Bandlimited interpolation of actual time samples, with a CPP segment.

    Prefix chirp correction is a digital frequency shift; wrapping its combined
    carrier frequencies preserves sample identity and avoids inventing an
    unobserved analog chirp between otherwise identical discrete-time blocks.
    """
    times = np.asarray(times, dtype=float)
    freq = np.fft.fftfreq(m)
    kernel = np.exp(2j*np.pi*np.outer(times, freq)) / math.sqrt(m)
    if prefix == "cpp" and c1 != 0.:
        negative = times < 0.
        shifted = (freq-2*m*c1+.5) % 1. - .5
        kernel[negative] = (np.exp(-2j*np.pi*c1*m*m)
                            * np.exp(2j*np.pi*np.outer(times[negative], shifted)) / math.sqrt(m))
    if c1 == 0. and c2 == 0.:
        result = kernel
    else:
        transmit, _ = _transform_matrices(m, c1, c2)
        spectrum = np.fft.fft(transmit, axis=0, norm="ortho")
        result = kernel @ spectrum
    result.setflags(write=False)
    return result


def _basis_at(times: np.ndarray, m: int, c1: float, c2: float, prefix: str):
    """Map affine symbols to the declared Nyquist-interpolated waveform."""
    return _interpolated_basis(tuple(np.asarray(times, dtype=float)), m, c1, c2, prefix)


def _delayed_burst(training, data, delay, cfg, length, stop_event=None):
    """Evaluate each finite source interval at n-delay, including the real prefix."""
    m, cp = cfg.fft_len, cfg.cp_len
    start = int(math.ceil(-cp + delay - 1e-12))
    end = int(math.ceil(m + delay - 1e-12))
    local_dest = np.arange(start, end, dtype=float)
    times = local_dest - delay
    training_values = _basis_at(times, m, 0., 0., "cp") @ training
    data_values = data @ _basis_at(times, m, cfg.c1, cfg.c2, cfg.prefix).T
    values = np.vstack((training_values, data_values))
    out = np.zeros(length, dtype=np.complex128)
    for block, row in enumerate(values):
        _check_cancel(stop_event)
        destination = 2*cp + block*(m+cp) + start
        low = max(0, -destination)
        high = min(row.size, length-destination)
        if high > low:
            out[destination+low:destination+high] = row[low:high]
    return out


def _estimated_static_channel(response, cfg):
    """Static all-M-lag circular LS approximation, with actual CPP wrap phases.

    This estimated model assumes periodic extension, including noisy/fractional
    IFFT taps beyond the real prefix.  It is not the oracle propagation matrix;
    model mismatch and cross-block interference remain in measured decisions.
    """
    m = cfg.fft_len
    h = np.fft.ifft(response)
    row, col = np.indices((m, m))
    lag = (row-col) % m
    result = h[lag]
    if cfg.prefix == "cpp":
        # For wrapped input column j, the source time is j-M, independent
        # of the output row.  Using row-col instead silently corrupts all
        # non-integral chirp-rate CPP models.
        source_time = col-m
        result = result * np.where(col > row, np.exp(-2j*np.pi*cfg.c1*(m*m+2*m*source_time)), 1.)
    return result


def _detect(matrix, received, variance, equalizer):
    if equalizer == "zf":
        return np.linalg.lstsq(matrix, received, rcond=1e-10)[0]
    adjoint = matrix.conj().T
    return np.linalg.solve(adjoint @ matrix + variance*np.eye(matrix.shape[1]), adjoint @ received)


def simulate_frame(config: AfdmNtnConfig, frame_index: int = 0, stop_event=None) -> dict:
    """Generate and score actual AFDM blocks, without payload-assisted estimation."""
    _check_cancel(stop_event)
    cfg = config.validated()
    if isinstance(frame_index, (bool, np.bool_)) or not isinstance(frame_index, (int, np.integer)) or frame_index < 0:
        raise ValueError("frame_index must be a non-negative integer")
    frame_index = int(frame_index)
    info = config_summary(cfg)
    m, n, cp = cfg.fft_len, cfg.symbols_per_frame, cfg.cp_len
    length, fs = info["frame_samples"], info["sample_rate_hz"]
    bits_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 0, frame_index]))
    noise_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 2, frame_index]))
    tx_bits = bits_rng.integers(0, 2, m*n*bits_per_symbol(cfg.modulation), dtype=np.uint8)
    data = qam_modulate(tx_bits, cfg.modulation).reshape(n, m)
    training = training_sequence(m)
    tx_samples = _delayed_burst(training, data, 0., cfg, length, stop_event)
    delays_s, gains, dopplers = _path_realization(cfg)
    delays = delays_s * fs
    times = (np.arange(length, dtype=float) + frame_index*length) / fs
    indices = 2*cp + np.arange(n+1)[:, None]*(m+cp) + np.arange(m)[None, :]
    rx_clean = np.zeros(length, dtype=np.complex128)
    own_data_mapping = np.zeros((n, m, m), dtype=np.complex128)
    for delay, gain, doppler in zip(delays, gains, dopplers):
        _check_cancel(stop_event)
        phase = np.exp(2j*np.pi*(doppler+info["residual_cfo_hz"])*times)
        rx_clean += gain*phase*_delayed_burst(training, data, float(delay), cfg, length, stop_event)
        u = np.arange(m, dtype=float)-delay
        basis = _basis_at(u, m, cfg.c1, cfg.c2, cfg.prefix) * ((u >= -cp-1e-12) & (u < m))[..., None]
        own_data_mapping += gain*phase[indices[1:]][..., None] * basis[None, :, :]
    _check_cancel(stop_event)
    tx_energy = float(np.vdot(tx_samples, tx_samples).real)
    variance = tx_energy / (tx_bits.size * 10.**(cfg.ebn0_db/10.))
    unit_noise = (noise_rng.standard_normal(length)+1j*noise_rng.standard_normal(length))/math.sqrt(2.)
    noise = math.sqrt(variance)*unit_noise
    rx_samples = rx_clean+noise
    received_blocks = rx_samples[indices]
    training_fft = np.fft.fft(received_blocks[0], norm="ortho")
    training_estimate = training_fft/training
    transmit, receive = _transform_matrices(m, cfg.c1, cfg.c2)
    true_effective = receive @ own_data_mapping
    static_time_channel = _estimated_static_channel(training_estimate, cfg)
    static_effective = receive @ static_time_channel @ transmit
    estimated_effective = true_effective.copy() if cfg.estimator == "ideal" else np.broadcast_to(static_effective, true_effective.shape).copy()
    raw_affine = afdm_demodulate(received_blocks[1:], cfg.c1, cfg.c2)
    if cfg.estimator == "training_ls":
        _check_cancel(stop_event)
        detected = _detect(static_effective, raw_affine.T, variance, cfg.equalizer).T
    else:
        detected = np.empty_like(data)
        for block in range(n):
            _check_cancel(stop_event)
            detected[block] = _detect(true_effective[block], raw_affine[block], variance, cfg.equalizer)
    _check_cancel(stop_event)
    reference = data.reshape(-1)
    recovered = detected.reshape(-1)
    rx_bits = qam_demodulate(recovered, cfg.modulation)
    errors = int(np.count_nonzero(tx_bits != rx_bits))
    reference_energy = float(np.vdot(reference, reference).real)
    error_energy = float(np.sum(np.abs(recovered-reference)**2))
    channel_energy = float(np.sum(np.abs(true_effective)**2))
    return {
        "frame_index": frame_index, "frame_time_s": float(frame_index*info["frame_duration_s"]),
        "bit_errors": errors, "total_bits": int(tx_bits.size), "ber": float(errors/tx_bits.size),
        "frame_error": int(errors > 0), "evm_percent": float(100.*math.sqrt(error_energy/reference_energy)),
        "error_energy": error_energy, "reference_energy": reference_energy,
        "tx_symbols": reference, "rx_symbols": recovered, "pre_eq_symbols": raw_affine.reshape(-1),
        "data_symbols": data, "rx_affine": detected, "pre_eq_affine": raw_affine,
        "tx_bits": tx_bits, "rx_bits": rx_bits, "tx_samples": tx_samples, "rx_samples": rx_samples,
        "rx_clean_samples": rx_clean, "noise_samples": noise, "unit_noise_samples": unit_noise,
        "tx_energy": tx_energy, "noise_variance": float(variance), "sample_rate_hz": fs,
        "training_tx_symbols": training, "training_rx_symbols": training_fft,
        "training_channel_est": training_estimate,
        "channel_true": true_effective, "channel_est": estimated_effective,
        "effective_channel_true": true_effective, "effective_channel_est": estimated_effective,
        "time_channel_true": own_data_mapping @ receive,
        "time_channel_est": np.broadcast_to(static_time_channel, true_effective.shape).copy(),
        "channel_nmse": float(np.sum(np.abs(estimated_effective-true_effective)**2)/max(channel_energy, 1e-30)),
        "path_delays_samples": delays.copy(), "path_gains": gains.copy(), "path_dopplers_hz": dopplers.copy(),
        "summary": info,
    }


def _point_configs(config, axis, values):
    if axis not in _AXIS_FIELDS:
        raise ValueError("Unknown study axis; choose ebn0, residual_cfo, doppler, spacing, delay_scale, c1, c2 or blocks")
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError("values must be a non-empty iterable of finite numbers") from exc
    if not values:
        raise ValueError("values must not be empty")
    prepared, seen = [], set()
    for raw in values:
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError("Scan values must be finite numbers")
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Scan values must be finite numbers") from exc
        if not math.isfinite(value) or value in seen:
            raise ValueError("Scan values must be finite and unique")
        seen.add(value)
        if axis == "blocks" and value != int(value):
            raise ValueError("blocks scan values must be integers")
        point_cfg = replace(config, **{_AXIS_FIELDS[axis]: int(value) if axis == "blocks" else value}).validated()
        variants = {
            "training_ls": replace(point_cfg, estimator="training_ls").validated(),
            "ideal": replace(point_cfg, estimator="ideal").validated(),
            "uncompensated": replace(point_cfg, estimator="training_ls", compensation_ratio=0.).validated(),
        }
        prepared.append((value, point_cfg, variants))
    return prepared


def run_study(config: AfdmNtnConfig, axis: str, values: Iterable[float], frames: int,
              stop_event=None, on_point: Callable[[dict], None] | None = None) -> list[dict]:
    """Paired actual bit decisions; cancellation publishes only complete points."""
    cfg = config.validated()
    if isinstance(frames, (bool, np.bool_)) or not isinstance(frames, (int, np.integer)) or frames < 1:
        raise ValueError("frames must be a positive integer")
    prepared = _point_configs(cfg, axis, values)
    output = []
    for value, point_cfg, variants in prepared:
        counters = {name: {"bit_errors": 0, "total_bits": 0, "frame_errors": 0, "frames": 0,
                           "error_energy": 0., "reference_energy": 0.} for name in STRATEGIES}
        try:
            for index in range(int(frames)):
                results = {}
                for name, variant in variants.items():
                    _check_cancel(stop_event)
                    results[name] = simulate_frame(variant, index, stop_event=stop_event)
                _check_cancel(stop_event)
                for name, result in results.items():
                    aggregate = counters[name]
                    for key in ("bit_errors", "total_bits", "error_energy", "reference_energy"):
                        aggregate[key] += result[key]
                    aggregate["frame_errors"] += result["frame_error"]
                    aggregate["frames"] += 1
            _check_cancel(stop_event)
        except InterruptedError:
            break
        for aggregate in counters.values():
            aggregate["ber"] = aggregate["bit_errors"]/aggregate["total_bits"]
            aggregate["ci_low"], aggregate["ci_high"] = ber_confidence_interval(aggregate["bit_errors"], aggregate["total_bits"])
            aggregate["fer"] = aggregate["frame_errors"]/aggregate["frames"]
            aggregate["evm_percent"] = 100.*math.sqrt(aggregate["error_energy"]/aggregate["reference_energy"])
            aggregate["count"] = aggregate["frames"]
        point = {"axis": axis, "axis_value": float(value), "strategies": counters,
                 "cancelled": False, "summary": config_summary(point_cfg)}
        output.append(point)
        if on_point is not None:
            on_point(point)
    return output
