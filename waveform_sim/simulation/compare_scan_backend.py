"""Paired full-grid CP-OFDM / rectangular OTFS / DAFT-AFDM experiments.

A generic ideal-timing/CSI three-tap model, separate from standalone OFDM NTN.
All waveforms use M*T QAM resources, the same ordinary CP, identical payload
bits/channel/time-domain noise and equal complete-frame transmitted energy.
The default is full LMMSE for every waveform. Practical mode explicitly mixes
one-tap OFDM with full OTFS/AFDM and includes a receiver-complexity difference.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import math
from typing import Dict
import numpy as np
from waveform_sim.core.modem import (
    bits_per_symbol as _bits_per_symbol,
    qam_modulate as _qam_mod,
    qam_demodulate as _qam_demod,
)

MATLAB_COLORS = {"OFDM": (0, 114, 189), "OTFS": (217, 83, 25), "AFDM": (119, 172, 48)}
WAVEFORMS = ("OFDM", "OTFS", "AFDM")
RECEIVER_MODES = ("full_mmse", "practical")
CHIRP_CONVENTION = "exp(-j 2 pi c n^2), no division by FFT length"
PREFIX = "ordinary_cp"


def _check_cancel(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Waveform comparison cancelled")


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _waveforms(names):
    if isinstance(names, str):
        names = [names]
    out = tuple(dict.fromkeys(str(name).upper() for name in names))
    if not out or any(name not in WAVEFORMS for name in out):
        raise ValueError("waveforms must contain OFDM, OTFS or AFDM")
    return out


def _normalized_sparse_channel(rng, doppler_hz):
    """Normalize expected PDP energy, preserving random channel attenuation."""
    delays = np.array([0, 2, 5], dtype=np.int64)
    powers = 10.0 ** (np.array([0.0, -3.5, -7.0]) / 10.0)
    powers /= powers.sum()
    gains = np.sqrt(powers / 2.0) * (rng.standard_normal(3) + 1j * rng.standard_normal(3))
    dopplers = float(doppler_hz) * np.array([0.0, 0.6, -0.8])
    return delays, gains, dopplers


@dataclass(frozen=True)
class ScanConfig:
    fft_len: int = 64
    n_slots: int = 7
    cp_len: int = 16
    subcarrier_spacing: float = 15e3
    lmmse_diag_loading: float = 1e-8
    channel_model: str = "multipath"
    # None retains the old normalized 0.05 coefficient at every FFT length.
    # Public values use the same standard DAFT convention as standalone AFDM.
    afdm_c1: float | None = None
    afdm_c2: float | None = None

    def validated(self):
        m = _integer(self.fft_len, "fft_len", 8)
        _integer(self.n_slots, "n_slots", 1)
        cp = _integer(self.cp_len, "cp_len")
        if m > 512 or m & (m - 1):
            raise ValueError("fft_len must be a power of two between 8 and 512")
        if cp >= m:
            raise ValueError("cp_len must be smaller than fft_len")
        model = str(self.channel_model).lower()
        if model not in {"awgn", "multipath"}:
            raise ValueError("channel_model must be multipath or AWGN")
        if model == "multipath" and cp < 5:
            raise ValueError("cp_len must cover the maximum path delay (5 samples)")
        if _finite(self.subcarrier_spacing, "subcarrier_spacing") <= 0:
            raise ValueError("subcarrier_spacing must be positive")
        if _finite(self.lmmse_diag_loading, "lmmse_diag_loading") < 0:
            raise ValueError("lmmse_diag_loading must be non-negative")
        chirps = {}
        for name in ("afdm_c1", "afdm_c2"):
            value = getattr(self, name)
            if value is None:
                value = 0.05 / m
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a finite chirp coefficient")
            value = _finite(value, name)
            if not 0.0 <= value <= 0.5:
                raise ValueError(f"{name} must lie in [0, 0.5] in the standard DAFT convention")
            chirps[name] = value
        return replace(self, fft_len=m, n_slots=int(self.n_slots), cp_len=cp,
                       subcarrier_spacing=float(self.subcarrier_spacing),
                       lmmse_diag_loading=float(self.lmmse_diag_loading), channel_model=model,
                       **chirps)


def validate_parameters(snr_db, cfo_hz, doppler_hz, mod_order, frames=1, seed=42,
                        receiver_mode="full_mmse", config=None):
    """Validate a whole request before a caller changes any experiment state."""
    cfg = (config or ScanConfig()).validated()
    snr_db = _finite(snr_db, "snr_db")
    cfo_hz = _finite(cfo_hz, "cfo_hz")
    doppler_hz = _finite(doppler_hz, "doppler_hz")
    if not -100 <= snr_db <= 160:
        raise ValueError("snr_db must be between -100 and 160 dB")
    if doppler_hz < 0:
        raise ValueError("doppler_hz must be non-negative")
    maximum_path_offset = 0.0 if cfg.channel_model == "awgn" else 0.8 * doppler_hz
    if abs(cfo_hz) + maximum_path_offset >= cfg.fft_len * cfg.subcarrier_spacing / 2.0:
        raise ValueError("CFO plus maximum path Doppler must remain below the sampling Nyquist frequency")
    mod_order = str(mod_order).upper()
    _bits_per_symbol(mod_order)
    mode = str(receiver_mode).lower()
    if mode not in RECEIVER_MODES:
        raise ValueError("receiver_mode must be full_mmse or practical")
    return {"snr_db": snr_db, "cfo_hz": cfo_hz, "doppler_hz": doppler_hz,
            "mod_order": mod_order, "frames": _integer(frames, "frames", 1),
            "seed": _integer(seed, "seed"), "receiver_mode": mode}


def _wilson(errors, bits):
    if bits <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = errors / bits
    denominator = 1.0 + z * z / bits
    center = (p + z * z / (2.0 * bits)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / bits + z * z / (4.0 * bits * bits)) / denominator
    return (0.0 if errors == 0 else max(0.0, center - radius),
            1.0 if errors == bits else min(1.0, center + radius))


class WaveformScanSimulator:
    """Synchronous paired frame source shared by live comparison and scans."""

    def __init__(self, config: ScanConfig | None = None):
        self.cfg = (config or ScanConfig()).validated()
        self.M, self.T = self.cfg.fft_len, self.cfg.n_slots
        self.sample_rate = self.M * self.cfg.subcarrier_spacing
        self.n_data = self.M
        self.net_symbols = self.M * self.T
        self.frame_samples = self.T * (self.M + self.cfg.cp_len)
        self.frame_duration_s = self.frame_samples / self.sample_rate
        self.F = np.fft.fft(np.eye(self.M), axis=0, norm="ortho")
        self.Fh = self.F.conj().T
        self.A_tx, self.A_rx = self._build_afdm_mats(self.cfg.afdm_c1, self.cfg.afdm_c2)

    def _build_afdm_mats(self, c1, c2):
        # Standard DAFT coefficients, matching the independent AFDM study.
        n = np.arange(self.M)
        d1 = np.exp(-2j * np.pi * c1 * n * n)
        d2 = np.exp(-2j * np.pi * c2 * n * n)
        tx = d1.conj()[:, None] * self.Fh * d2.conj()[None, :]
        return tx, tx.conj().T

    def _build_otfs_mats(self):
        # DD=(delay M, Doppler T), column-wise vectorization.
        # ISFFT: FFT along delay, IFFT along Doppler; SFFT is its adjoint.
        ft = np.fft.fft(np.eye(self.T), axis=0, norm="ortho")
        tx = np.kron(ft.conj().T, self.F)
        return tx, tx.conj().T

    @property
    def U_dd_to_tf(self):
        return self._build_otfs_mats()[0]

    @property
    def U_tf_to_dd(self):
        return self._build_otfs_mats()[1]

    def _transmit(self, name, symbols):
        if name == "OFDM":
            return np.fft.ifft(symbols, axis=1, norm="ortho")
        if name == "AFDM":
            return symbols @ self.A_tx.T
        dd = symbols.T
        tf = np.fft.ifft(np.fft.fft(dd, axis=0, norm="ortho"), axis=1, norm="ortho")
        return np.fft.ifft(tf, axis=0, norm="ortho").T

    def _receive(self, name, time_symbols):
        if name == "OFDM":
            return np.fft.fft(time_symbols, axis=1, norm="ortho")
        if name == "AFDM":
            return time_symbols @ self.A_rx.T
        tf = np.fft.fft(time_symbols.T, axis=0, norm="ortho")
        return np.fft.fft(np.fft.ifft(tf, axis=0, norm="ortho"), axis=1, norm="ortho").T

    def _channel_matrix(self, delays, gains, fd_list, cfo_hz, slot_idx):
        row = np.arange(self.M)
        absolute = slot_idx * (self.M + self.cfg.cp_len) + self.cfg.cp_len + row
        matrix = np.zeros((self.M, self.M), dtype=np.complex128)
        for delay, gain, doppler in zip(delays, gains, fd_list):
            phase = np.exp(2j * np.pi * (doppler + cfo_hz) * absolute / self.sample_rate)
            matrix[row, (row - int(delay)) % self.M] += gain * phase
        return matrix

    def _validate_point(self, snr_db, cfo_hz, doppler_hz, mod_order, seed, frame_index, receiver_mode):
        values = validate_parameters(snr_db, cfo_hz, doppler_hz, mod_order, seed=seed,
                                      receiver_mode=receiver_mode, config=self.cfg)
        frame_index = _integer(frame_index, "frame_index")
        return (values["snr_db"], values["cfo_hz"], values["doppler_hz"], values["mod_order"],
                values["seed"], frame_index, values["receiver_mode"])

    def validate_params(self, *args, **kwargs):
        return validate_parameters(*args, **kwargs, config=self.cfg)

    def simulate_frame_set(self, waveforms, snr_db, cfo_hz, doppler_hz,
                           mod_order="QPSK", seed=42, frame_index=0,
                           receiver_mode="full_mmse", stop_event=None):
        """Return a complete pair; cancellation never publishes a partial pair."""
        names = _waveforms(waveforms)
        snr_db, cfo_hz, doppler_hz, mod_order, seed, frame_index, receiver_mode = self._validate_point(
            snr_db, cfo_hz, doppler_hz, mod_order, seed, frame_index, receiver_mode)
        _check_cancel(stop_event)
        bits_rng = np.random.default_rng(np.random.SeedSequence([seed, 0, frame_index]))
        channel_rng = np.random.default_rng(np.random.SeedSequence([seed, 1, frame_index]))
        noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 2, frame_index]))
        bps = _bits_per_symbol(mod_order)
        tx_bits = bits_rng.integers(0, 2, self.net_symbols * bps, dtype=np.uint8)
        symbols = _qam_mod(tx_bits, mod_order).reshape(self.T, self.M)
        if self.cfg.channel_model == "awgn":
            delays, gains, dopplers = np.array([0]), np.array([1.0 + 0j]), np.array([0.0])
        else:
            delays, gains, dopplers = _normalized_sparse_channel(channel_rng, doppler_hz)
        matrices = np.asarray([
            self._channel_matrix(delays, gains, dopplers, cfo_hz, frame_index * self.T + slot)
            for slot in range(self.T)
        ])
        unit_noise = (noise_rng.standard_normal(self.frame_samples)
                      + 1j * noise_rng.standard_normal(self.frame_samples)) / math.sqrt(2.0)
        # Same whole-burst energy, including the physically transmitted CP.
        common_energy = float(self.frame_samples)
        noise_variance = common_energy / (tx_bits.size * 10.0 ** (snr_db / 10.0))
        noise = math.sqrt(noise_variance) * unit_noise
        absolute_samples = frame_index * self.frame_samples + np.arange(self.frame_samples)
        cp = self.cfg.cp_len
        useful_indices = (np.arange(self.T)[:, None] * (self.M + cp) + cp + np.arange(self.M)[None, :])
        receiver_note = ("所有波形使用理想 CSI 和完整 LMMSE"
                         if receiver_mode == "full_mmse" else
                         "含接收机复杂度差异：OFDM 单抽头；OTFS/AFDM 完整 LMMSE")
        channel = {"delays": delays, "gains": gains, "dopplers_hz": dopplers, "matrices": matrices}
        # Unitary modulation gives identity input covariance.  All complete
        # receivers can therefore solve the same time-domain LMMSE problem,
        # followed by their respective unitary demodulators.  Share the
        # Hermitian decomposition rather than solving a large DD system or
        # factoring the same channel once for each waveform.
        decompositions = []
        if receiver_mode == "full_mmse" or any(name != "OFDM" for name in names):
            for h_time in matrices:
                _check_cancel(stop_event)
                eigenvalues, eigenvectors = np.linalg.eigh(h_time.conj().T @ h_time)
                decompositions.append((np.maximum(eigenvalues, 0.0), eigenvectors))
        output = {}
        for name in names:
            _check_cancel(stop_event)
            raw_time = self._transmit(name, symbols)
            prefixed = np.concatenate((raw_time[:, -cp:], raw_time), axis=1) if cp else raw_time.copy()
            raw_energy = float(np.sum(np.abs(prefixed) ** 2))
            scale = math.sqrt(common_energy / max(raw_energy, 1e-30))
            tx_samples = (scale * prefixed).reshape(-1)
            rx_clean = np.zeros(self.frame_samples, dtype=np.complex128)
            for delay, gain, doppler in zip(delays, gains, dopplers):
                delay = int(delay)
                phase = np.exp(2j * np.pi * (doppler + cfo_hz) * absolute_samples / self.sample_rate)
                if delay:
                    rx_clean[delay:] += gain * phase[delay:] * tx_samples[:-delay]
                else:
                    rx_clean += gain * phase * tx_samples
            rx_samples = rx_clean + noise
            useful = rx_samples[useful_indices]
            if receiver_mode == "practical" and name == "OFDM":
                estimates = []
                for slot, h_time in enumerate(matrices):
                    _check_cancel(stop_event)
                    diagonal = scale * np.diag(self.F @ h_time @ self.Fh)
                    received = np.fft.fft(useful[slot], norm="ortho")
                    estimates.append(received * np.conj(diagonal) /
                                     (np.abs(diagonal) ** 2 + noise_variance + self.cfg.lmmse_diag_loading))
                decoded = np.asarray(estimates)
            else:
                estimated_time = []
                for slot, h_time in enumerate(matrices):
                    _check_cancel(stop_event)
                    rhs = scale * h_time.conj().T @ useful[slot]
                    eigenvalues, eigenvectors = decompositions[slot]
                    denominator = scale * scale * eigenvalues + noise_variance + self.cfg.lmmse_diag_loading
                    estimated_time.append(eigenvectors @ ((eigenvectors.conj().T @ rhs) / denominator))
                decoded = self._receive(name, np.asarray(estimated_time))
            rx_symbols = decoded.reshape(-1)
            rx_bits = _qam_demod(rx_symbols, mod_order)
            errors = int(np.count_nonzero(tx_bits != rx_bits))
            reference = symbols.reshape(-1)
            evm = 100.0 * math.sqrt(float(np.sum(np.abs(rx_symbols - reference) ** 2)) /
                                    max(float(np.sum(np.abs(reference) ** 2)), 1e-30))
            output[name] = {
                "bit_errors": errors, "total_bits": int(tx_bits.size),
                "ber": errors / tx_bits.size, "frame_error": int(errors > 0),
                "evm_percent": evm, "rx_symbols": rx_symbols, "constellation": rx_symbols,
                "tx_symbols": reference, "tx_bits": tx_bits, "rx_bits": rx_bits,
                "noise_samples": noise, "unit_noise_samples": unit_noise,
                "noise_variance": float(noise_variance), "channel": channel,
                "tx_samples": tx_samples, "rx_samples": rx_samples, "rx_clean_samples": rx_clean,
                "tx_energy": float(np.vdot(tx_samples, tx_samples).real), "normalization_scale": scale,
                "frame_index": frame_index, "frame_time_s": frame_index * self.frame_duration_s,
                "sample_rate_hz": self.sample_rate, "frame_duration_s": self.frame_duration_s,
                "receiver_mode": receiver_mode, "receiver_note": receiver_note,
                "model": "full_grid_cp_comparison", "data_symbols": self.net_symbols,
                "pilot_symbols": 0, "cp_len": cp,
                "afdm_c1": self.cfg.afdm_c1, "afdm_c2": self.cfg.afdm_c2,
                "chirp_convention": CHIRP_CONVENTION, "prefix": PREFIX,
                "public_config": {**asdict(self.cfg), "chirp_convention": CHIRP_CONVENTION,
                                  "prefix": PREFIX},
            }
        _check_cancel(stop_event)
        return output

    def simulate_stats(self, waveforms, snr_db, cfo_hz, doppler_hz, mod_order="QPSK",
                       frames=20, seed=42, receiver_mode="full_mmse", stop_event=None):
        """Pool raw counters; cancelled work never becomes zero-error evidence."""
        names = _waveforms(waveforms)
        parameters = self.validate_params(snr_db, cfo_hz, doppler_hz, mod_order,
                                           frames, seed, receiver_mode)
        snr_db, cfo_hz, doppler_hz = (parameters[key] for key in ("snr_db", "cfo_hz", "doppler_hz"))
        mod_order, frames, seed, receiver_mode = (
            parameters[key] for key in ("mod_order", "frames", "seed", "receiver_mode"))
        counters = {name: {"bit_errors": 0, "total_bits": 0, "frame_errors": 0,
                           "frames": 0, "evm_square_sum": 0.0} for name in names}
        for frame in range(frames):
            paired = self.simulate_frame_set(names, snr_db, cfo_hz, doppler_hz,
                                             mod_order, seed, frame, receiver_mode, stop_event)
            _check_cancel(stop_event)
            for name in names:
                raw, aggregate = paired[name], counters[name]
                aggregate["bit_errors"] += raw["bit_errors"]
                aggregate["total_bits"] += raw["total_bits"]
                aggregate["frame_errors"] += raw["frame_error"]
                aggregate["frames"] += 1
                aggregate["evm_square_sum"] += raw["evm_percent"] ** 2
        for aggregate in counters.values():
            aggregate["ber"] = aggregate["bit_errors"] / aggregate["total_bits"]
            aggregate["fer"] = aggregate["frame_errors"] / aggregate["frames"]
            aggregate["evm_percent"] = math.sqrt(aggregate.pop("evm_square_sum") / aggregate["frames"])
            aggregate["ci_low"], aggregate["ci_high"] = _wilson(aggregate["bit_errors"], aggregate["total_bits"])
            aggregate["receiver_mode"] = receiver_mode
            aggregate.update(afdm_c1=self.cfg.afdm_c1, afdm_c2=self.cfg.afdm_c2,
                             chirp_convention=CHIRP_CONVENTION, prefix=PREFIX)
        return counters

    def simulate_ber(self, waveform, snr_db, cfo_hz, doppler_hz, mod_order="QPSK",
                     frames=20, seed=None, receiver_mode="full_mmse", stop_event=None):
        """Compatibility scalar API, preserving true zero-error results."""
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        name = _waveforms([waveform])[0]
        return float(self.simulate_stats([name], snr_db, cfo_hz, doppler_hz,
                                        mod_order, frames, seed, receiver_mode, stop_event)[name]["ber"])


def scan_axis(axis_name, axis_values, snr_db, mod_order, frames, fixed_cfo_hz,
              fixed_doppler_hz, seed_base=20260502, receiver_mode="full_mmse",
              stop_event=None, on_point=None, config=None) -> Dict[str, np.ndarray]:
    """Legacy array API with paired samples, raw BER and no artificial floor.

    One seed serves all waveforms and all axis points (common random numbers),
    making results independent of waveform/point iteration order. Statistics
    with Wilson bounds are available from simulate_stats or the point callback.
    """
    axis_name = str(axis_name).lower()
    if axis_name not in {"cfo", "doppler"}:
        raise ValueError("axis_name must be cfo or doppler")
    axis_values = np.asarray(axis_values, dtype=float)
    if axis_values.ndim != 1 or not axis_values.size or not np.all(np.isfinite(axis_values)):
        raise ValueError("axis_values must be a nonempty finite one-dimensional array")
    simulator = WaveformScanSimulator(config)
    # Validate the complete sweep before computing or reporting any point.
    for value in axis_values:
        simulator.validate_params(snr_db, float(value) if axis_name == "cfo" else fixed_cfo_hz,
                                  float(value) if axis_name == "doppler" else fixed_doppler_hz,
                                  mod_order, frames, seed_base, receiver_mode)
    out = {name: [] for name in WAVEFORMS}
    for value in axis_values:
        _check_cancel(stop_event)
        cfo = float(value) if axis_name == "cfo" else fixed_cfo_hz
        doppler = float(value) if axis_name == "doppler" else fixed_doppler_hz
        stats = simulator.simulate_stats(WAVEFORMS, snr_db, cfo, doppler, mod_order,
                                          frames, seed_base, receiver_mode, stop_event)
        for name in WAVEFORMS:
            out[name].append(stats[name]["ber"])
        if on_point is not None:
            on_point({"axis": axis_name, "axis_value": float(value), "strategies": stats})
    return {name: np.asarray(values, dtype=float) for name, values in out.items()}
