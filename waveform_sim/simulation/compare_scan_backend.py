"""Paired full-grid CP-OFDM / rectangular OTFS / DAFT-AFDM experiments.

A generic ideal-timing/CSI three-tap model, separate from standalone OFDM NTN.
All waveforms use M*T QAM resources, the same ordinary CP, identical payload
bits/channel/time-domain noise and equal complete-frame transmitted energy.
The default is full LMMSE for every waveform. Practical mode uses a one-tap
OFDM equalizer and a sparse iterative LMMSE equalizer for OTFS and AFDM.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import math
from statistics import NormalDist
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
PRACTICAL_CG_MAX_ITERATIONS = 2
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
                        receiver_mode="full_mmse", config=None, csi_mode="ideal"):
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
    csi_mode = str(csi_mode).lower()
    if csi_mode not in {"ideal", "pilot"}:
        raise ValueError("csi_mode must be ideal or pilot")
    if mode == "practical":
        csi_mode = "pilot"
    return {"snr_db": snr_db, "cfo_hz": cfo_hz, "doppler_hz": doppler_hz,
            "mod_order": mod_order, "frames": _integer(frames, "frames", 1),
            "seed": _integer(seed, "seed"), "receiver_mode": mode,
            "csi_mode": csi_mode}


def _sparse_pcg_mmse(h_time, observed, scale, noise_variance, diag_loading,
                     max_iterations=PRACTICAL_CG_MAX_ITERATIONS, stop_event=None):
    """Solve the sparse time-domain LMMSE system without dense factorization."""
    rows, columns = np.nonzero(h_time)
    coefficients = h_time[rows, columns]
    size = h_time.shape[0]
    regularization = float(noise_variance + diag_loading)
    scale_squared = float(scale * scale)

    def multiply_h(vector):
        result = np.zeros(size, dtype=np.complex128)
        np.add.at(result, rows, coefficients * vector[columns])
        return result

    def multiply_h_adjoint(vector):
        result = np.zeros(size, dtype=np.complex128)
        np.add.at(result, columns, coefficients.conj() * vector[rows])
        return result

    rhs = scale * multiply_h_adjoint(observed)
    if not coefficients.size or not np.any(rhs):
        return np.zeros(size, dtype=np.complex128), 0, 0

    diagonal = regularization + scale_squared * np.bincount(
        columns, weights=np.abs(coefficients) ** 2, minlength=size
    )
    solution = np.zeros(size, dtype=np.complex128)
    residual = rhs.copy()
    initial_residual_norm = float(np.linalg.norm(residual))
    preconditioned = residual / diagonal
    direction = preconditioned.copy()
    residual_product = float(np.vdot(residual, preconditioned).real)
    iterations = 0

    for _ in range(_integer(max_iterations, "max_iterations", 1)):
        _check_cancel(stop_event)
        if float(np.linalg.norm(residual)) <= max(initial_residual_norm * 1e-10, 1e-14):
            break
        projected = multiply_h(direction)
        normal_product = (
            scale_squared * multiply_h_adjoint(projected)
            + regularization * direction
        )
        denominator = float(np.vdot(direction, normal_product).real)
        if not np.isfinite(denominator) or denominator <= 0.0:
            break
        step = residual_product / denominator
        solution += step * direction
        residual -= step * normal_product
        iterations += 1
        preconditioned = residual / diagonal
        next_product = float(np.vdot(residual, preconditioned).real)
        if residual_product <= 0.0:
            break
        direction = preconditioned + (next_product / residual_product) * direction
        residual_product = next_product

    return solution, iterations, int(2 * coefficients.size * iterations)


def _wilson(errors, bits, z=1.959963984540054):
    if bits <= 0:
        return float("nan"), float("nan")
    p = errors / bits
    denominator = 1.0 + z * z / bits
    center = (p + z * z / (2.0 * bits)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / bits + z * z / (4.0 * bits * bits)) / denominator
    return (0.0 if errors == 0 else max(0.0, center - radius),
            1.0 if errors == bits else min(1.0, center + radius))


def _bootstrap_indices(frame_count: int, resamples: int = 2000, seed: int = 0):
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(frame_count), size=(int(resamples), int(frame_count)))


def _frame_cluster_bootstrap_interval(
    frame_errors, frame_bits, resamples: int = 2000, seed: int = 0,
    confidence: float = 0.95,
):
    """Percentile interval over independent frames, keeping each fade intact."""
    errors = np.asarray(frame_errors, dtype=np.int64)
    bits = np.asarray(frame_bits, dtype=np.int64)
    if errors.ndim != 1 or bits.shape != errors.shape:
        raise ValueError("frame_errors and frame_bits must be matching one-dimensional arrays")
    if not errors.size:
        return 0.0, 1.0, "insufficient_frames_worst_case"
    if np.any(bits <= 0) or np.any(errors < 0) or np.any(errors > bits):
        raise ValueError("per-frame error counts must satisfy 0 <= errors <= bits")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be between zero and one")
    if not np.any(errors):
        # With equal-sized frames, FER is a conservative upper bound on BER.
        z = NormalDist().inv_cdf(0.5 + float(confidence) / 2.0)
        upper = (
            _wilson(0, int(errors.size), z=z)[1]
            if np.all(bits == bits[0])
            else 1.0
        )
        return 0.0, float(upper), "zero_error_frame_fer_upper_bound"
    if errors.size < 2:
        return 0.0, 1.0, "insufficient_frames_worst_case"
    indices = _bootstrap_indices(errors.size, resamples=resamples, seed=seed)
    sampled_errors = errors[indices].sum(axis=1)
    sampled_bits = bits[indices].sum(axis=1)
    estimates = sampled_errors / np.maximum(sampled_bits, 1)
    tail = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    return float(low), float(high), "frame_cluster_bootstrap"


def _paired_frame_difference_interval(
    frame_errors_a, frame_bits_a, frame_errors_b, frame_bits_b,
    resamples: int = 2000, seed: int = 0,
):
    """Paired percentile interval for BER(A)-BER(B), resampling shared frames."""
    errors_a = np.asarray(frame_errors_a, dtype=np.int64)
    bits_a = np.asarray(frame_bits_a, dtype=np.int64)
    errors_b = np.asarray(frame_errors_b, dtype=np.int64)
    bits_b = np.asarray(frame_bits_b, dtype=np.int64)
    arrays = (errors_a, bits_a, errors_b, bits_b)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("paired frame counts must be nonempty one-dimensional arrays")
    if any(array.size != errors_a.size for array in arrays):
        raise ValueError("paired frame arrays must have matching lengths")
    if not errors_a.size:
        return -1.0, 1.0, "insufficient_frames_worst_case"
    if np.any(bits_a <= 0) or np.any(bits_b <= 0):
        raise ValueError("per-frame bit counts must be positive")
    if (np.any(errors_a < 0) or np.any(errors_b < 0)
            or np.any(errors_a > bits_a) or np.any(errors_b > bits_b)):
        raise ValueError("per-frame error counts must satisfy 0 <= errors <= bits")
    if errors_a.size < 2:
        return -1.0, 1.0, "insufficient_frames_worst_case"
    zero_a, zero_b = not np.any(errors_a), not np.any(errors_b)
    if zero_a or zero_b:
        def zero_upper(bits):
            return _wilson(0, int(bits.size), z=2.241402727604947)[1] if np.all(bits == bits[0]) else 1.0

        low_a, high_a, _ = (
            (0.0, zero_upper(bits_a), "zero") if zero_a
            else _frame_cluster_bootstrap_interval(
                errors_a, bits_a, resamples, seed, confidence=0.975
            )
        )
        low_b, high_b, _ = (
            (0.0, zero_upper(bits_b), "zero") if zero_b
            else _frame_cluster_bootstrap_interval(
                errors_b, bits_b, resamples, seed, confidence=0.975
            )
        )
        method = (
            "zero_error_paired_fer_bounds" if zero_a and zero_b
            else "paired_frame_cluster_bootstrap_with_zero_error_fer_bounds"
        )
        return float(low_a - high_b), float(high_a - low_b), method
    indices = _bootstrap_indices(errors_a.size, resamples=resamples, seed=seed)
    ber_a = errors_a[indices].sum(axis=1) / bits_a[indices].sum(axis=1)
    ber_b = errors_b[indices].sum(axis=1) / bits_b[indices].sum(axis=1)
    low, high = np.quantile(ber_a - ber_b, [0.025, 0.975])
    return float(low), float(high), "paired_frame_cluster_bootstrap"


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

    def _validate_point(self, snr_db, cfo_hz, doppler_hz, mod_order, seed, frame_index,
                        receiver_mode, csi_mode):
        values = validate_parameters(snr_db, cfo_hz, doppler_hz, mod_order, seed=seed,
                                      receiver_mode=receiver_mode, config=self.cfg,
                                      csi_mode=csi_mode)
        frame_index = _integer(frame_index, "frame_index")
        return (values["snr_db"], values["cfo_hz"], values["doppler_hz"], values["mod_order"],
                values["seed"], frame_index, values["receiver_mode"], values["csi_mode"])

    def validate_params(self, *args, **kwargs):
        return validate_parameters(*args, **kwargs, config=self.cfg)

    def simulate_frame_set(self, waveforms, snr_db, cfo_hz, doppler_hz,
                           mod_order="QPSK", seed=42, frame_index=0,
                           receiver_mode="full_mmse", stop_event=None, csi_mode="ideal"):
        """Return a complete pair; cancellation never publishes a partial pair."""
        names = _waveforms(waveforms)
        snr_db, cfo_hz, doppler_hz, mod_order, seed, frame_index, receiver_mode, csi_mode = self._validate_point(
            snr_db, cfo_hz, doppler_hz, mod_order, seed, frame_index, receiver_mode,
            csi_mode)
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
        slot_stride = self.M + self.cfg.cp_len
        pilot_slots = int(csi_mode == "pilot")
        data_slot_base = frame_index * (self.T + pilot_slots) + pilot_slots
        matrices = np.asarray([
            self._channel_matrix(delays, gains, dopplers, cfo_hz, data_slot_base + slot)
            for slot in range(self.T)
        ])
        unit_noise = (noise_rng.standard_normal(self.frame_samples)
                      + 1j * noise_rng.standard_normal(self.frame_samples)) / math.sqrt(2.0)
        # Keep the same total energy per information bit, including training.
        common_energy = float(self.frame_samples)
        noise_variance = common_energy / (tx_bits.size * 10.0 ** (snr_db / 10.0))
        noise = math.sqrt(noise_variance) * unit_noise
        cp = self.cfg.cp_len
        useful_indices = (np.arange(self.T)[:, None] * (self.M + cp) + cp + np.arange(self.M)[None, :])
        if csi_mode == "pilot":
            pilot_rng = np.random.default_rng(np.random.SeedSequence([seed, 3, frame_index]))
            pilot_noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 4, frame_index]))
            pilot_bits = pilot_rng.integers(0, 2, self.M * 2, dtype=np.uint8)
            pilot_symbols = _qam_mod(pilot_bits, "QPSK")
            pilot_time = np.fft.ifft(pilot_symbols, norm="ortho")
            pilot_prefixed = np.concatenate((pilot_time[-cp:], pilot_time)) if cp else pilot_time.copy()
            pilot_raw_energy = float(np.vdot(pilot_prefixed, pilot_prefixed).real)
            pilot_energy = common_energy / (self.T + 1)
            data_energy = common_energy - pilot_energy
            pilot_scale = math.sqrt(pilot_energy / max(pilot_raw_energy, 1e-30))
            pilot_unit_noise = (pilot_noise_rng.standard_normal(slot_stride)
                                + 1j * pilot_noise_rng.standard_normal(slot_stride)) / math.sqrt(2.0)
            pilot_noise = math.sqrt(noise_variance) * pilot_unit_noise
            pilot_observation_matrices = [
                self._channel_matrix(delays[index:index + 1], np.ones(1, dtype=np.complex128),
                                     dopplers[index:index + 1], cfo_hz,
                                     frame_index * (self.T + 1))
                for index in range(len(gains))
            ]
            # The estimator knows the sparse delay support but does not receive
            # the scanned residual CFO or per-path Doppler as side information.
            pilot_design_matrices = [
                self._channel_matrix(delays[index:index + 1], np.ones(1, dtype=np.complex128),
                                     np.zeros(1), 0.0, frame_index * (self.T + 1))
                for index in range(len(gains))
            ]
            pilot_useful = pilot_scale * pilot_time
            design = np.column_stack([matrix @ pilot_useful for matrix in pilot_design_matrices])
            pilot_observation = sum(
                gain * (matrix @ pilot_useful)
                for gain, matrix in zip(gains, pilot_observation_matrices)
            ) + pilot_noise[cp:]
            estimated_gains = np.linalg.lstsq(design, pilot_observation, rcond=None)[0]
            receiver_matrices = np.asarray([
                self._channel_matrix(delays, estimated_gains, np.zeros_like(dopplers), 0.0,
                                     data_slot_base + slot)
                for slot in range(self.T)
            ])
            pilot_overhead = 1.0 / (self.T + 1)
            pilot_symbols_count = self.M
        else:
            data_energy = common_energy
            pilot_energy = 0.0
            pilot_scale = 0.0
            pilot_noise = np.zeros(0, dtype=np.complex128)
            receiver_matrices = matrices.copy()
            pilot_overhead = 0.0
            pilot_symbols_count = 0
        csi_nmse = float(
            np.linalg.norm((receiver_matrices - matrices).reshape(-1)) ** 2
            / max(np.linalg.norm(matrices.reshape(-1)) ** 2, 1e-30)
        )
        data_absolute_start = data_slot_base * slot_stride
        absolute_samples = data_absolute_start + np.arange(self.frame_samples)
        receiver_note = (
            "统一完整 LMMSE 检测器家族" if receiver_mode == "full_mmse" else
            f"接收方案对照：OFDM 单抽头，OTFS/AFDM 时域稀疏 PCG-LMMSE（最多 {PRACTICAL_CG_MAX_ITERATIONS} 轮）"
        ) + (
            "；完美 CSI 理想基准" if csi_mode == "ideal" else
            "；共同 OFDM 导频 LS CSI，已知路径时延支撑，残余 CFO/多普勒未知"
        )
        channel = {"delays": delays, "gains": gains, "dopplers_hz": dopplers,
                   "matrices": matrices, "estimated_matrices": receiver_matrices}
        # Unitary modulation gives identity input covariance.  The common
        # baseline solves the same time-domain LMMSE problem for every waveform.
        decompositions = []
        if receiver_mode == "full_mmse":
            for h_time in receiver_matrices:
                _check_cancel(stop_event)
                eigenvalues, eigenvectors = np.linalg.eigh(h_time.conj().T @ h_time)
                decompositions.append((np.maximum(eigenvalues, 0.0), eigenvectors))
        output = {}
        raw_frames = {}
        scales = {}
        for name in names:
            raw_time = self._transmit(name, symbols)
            prefixed = np.concatenate((raw_time[:, -cp:], raw_time), axis=1) if cp else raw_time.copy()
            raw_frames[name] = prefixed
            raw_energy = float(np.sum(np.abs(prefixed) ** 2))
            scales[name] = math.sqrt(data_energy / max(raw_energy, 1e-30))
        for name in names:
            _check_cancel(stop_event)
            scale = scales[name]
            tx_samples = (scale * raw_frames[name]).reshape(-1)
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
                for slot, h_time in enumerate(receiver_matrices):
                    _check_cancel(stop_event)
                    diagonal = scale * np.diag(self.F @ h_time @ self.Fh)
                    received = np.fft.fft(useful[slot], norm="ortho")
                    estimates.append(received * np.conj(diagonal) /
                                     (np.abs(diagonal) ** 2 + noise_variance + self.cfg.lmmse_diag_loading))
                decoded = np.asarray(estimates)
                receiver_method = "ofdm_one_tap"
                receiver_iterations_max = 0
                receiver_complexity_proxy = self.T * self.M
                receiver_coefficients_per_row = 1
            elif receiver_mode == "practical":
                estimated_time = []
                receiver_complexity_proxy = 0
                receiver_iterations_max = 0
                for slot, h_time in enumerate(receiver_matrices):
                    _check_cancel(stop_event)
                    estimate, iterations, coefficient_applications = _sparse_pcg_mmse(
                        h_time, useful[slot], scale, noise_variance,
                        self.cfg.lmmse_diag_loading, stop_event=stop_event,
                    )
                    estimated_time.append(estimate)
                    receiver_iterations_max = max(receiver_iterations_max, iterations)
                    receiver_complexity_proxy += coefficient_applications
                decoded = self._receive(name, np.asarray(estimated_time))
                receiver_method = "sparse_pcg_lmmse"
                receiver_coefficients_per_row = int(math.ceil(
                    receiver_complexity_proxy / max(self.T * self.M, 1)
                ))
            else:
                estimated_time = []
                for slot, h_time in enumerate(receiver_matrices):
                    _check_cancel(stop_event)
                    rhs = scale * h_time.conj().T @ useful[slot]
                    eigenvalues, eigenvectors = decompositions[slot]
                    denominator = scale * scale * eigenvalues + noise_variance + self.cfg.lmmse_diag_loading
                    estimated_time.append(eigenvectors @ ((eigenvectors.conj().T @ rhs) / denominator))
                decoded = self._receive(name, np.asarray(estimated_time))
                receiver_method = "full_lmmse_time_domain"
                receiver_iterations_max = 0
                receiver_complexity_proxy = self.T * self.M ** 3
                receiver_coefficients_per_row = self.M
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
                "pilot_noise_samples": pilot_noise,
                "noise_variance": float(noise_variance), "channel": channel,
                "tx_samples": tx_samples, "rx_samples": rx_samples, "rx_clean_samples": rx_clean,
                "tx_energy": float(data_energy + pilot_energy),
                "data_tx_energy": float(np.vdot(tx_samples, tx_samples).real),
                "pilot_tx_energy": float(pilot_energy), "normalization_scale": scale,
                "frame_index": frame_index,
                "frame_time_s": frame_index * (self.T + pilot_slots) * slot_stride / self.sample_rate,
                "sample_rate_hz": self.sample_rate,
                "subcarrier_spacing_hz": self.cfg.subcarrier_spacing,
                "frame_duration_s": self.frame_duration_s + (slot_stride / self.sample_rate if csi_mode == "pilot" else 0.0),
                "receiver_mode": receiver_mode, "receiver_note": receiver_note,
                "receiver_method": receiver_method,
                "receiver_iterations_max": int(receiver_iterations_max),
                "csi_mode": csi_mode, "csi_nmse": csi_nmse,
                "pilot_ofdm_symbols": int(csi_mode == "pilot"),
                "pilot_symbols": pilot_symbols_count,
                "pilot_overhead_fraction": float(pilot_overhead),
                "payload_resource_fraction": float(1.0 - pilot_overhead),
                "receiver_coefficients_per_row": int(receiver_coefficients_per_row),
                "receiver_complexity_proxy": int(receiver_complexity_proxy),
                "receiver_complexity_scope": "equalizer only; excludes waveform transforms, vector updates and CSI estimation",
                "model": "full_grid_cp_comparison", "data_symbols": self.net_symbols,
                "cp_len": cp,
                "afdm_c1": self.cfg.afdm_c1, "afdm_c2": self.cfg.afdm_c2,
                "chirp_convention": CHIRP_CONVENTION, "prefix": PREFIX,
                "public_config": {**asdict(self.cfg), "chirp_convention": CHIRP_CONVENTION,
                                  "prefix": PREFIX},
            }
        _check_cancel(stop_event)
        return output

    def simulate_stats(self, waveforms, snr_db, cfo_hz, doppler_hz, mod_order="QPSK",
                       frames=20, seed=42, receiver_mode="full_mmse", stop_event=None,
                       csi_mode="ideal"):
        """Pool raw counters; cancelled work never becomes zero-error evidence."""
        names = _waveforms(waveforms)
        parameters = self.validate_params(snr_db, cfo_hz, doppler_hz, mod_order,
                                           frames, seed, receiver_mode, csi_mode=csi_mode)
        snr_db, cfo_hz, doppler_hz = (parameters[key] for key in ("snr_db", "cfo_hz", "doppler_hz"))
        mod_order, frames, seed, receiver_mode, csi_mode = (
            parameters[key] for key in ("mod_order", "frames", "seed", "receiver_mode", "csi_mode"))
        counters = {name: {"bit_errors": 0, "total_bits": 0, "frame_errors": 0,
                           "frames": 0, "evm_square_sum": 0.0,
                           "receiver_complexity_proxy_sum": 0,
                           "receiver_coefficients_per_row_sum": 0,
                           "receiver_iterations_max": 0,
                           "frame_error_counts": [], "frame_bit_counts": []} for name in names}
        for frame in range(frames):
            paired = self.simulate_frame_set(names, snr_db, cfo_hz, doppler_hz,
                                             mod_order, seed, frame, receiver_mode, stop_event,
                                             csi_mode=csi_mode)
            _check_cancel(stop_event)
            for name in names:
                raw, aggregate = paired[name], counters[name]
                aggregate["bit_errors"] += raw["bit_errors"]
                aggregate["total_bits"] += raw["total_bits"]
                aggregate["frame_errors"] += raw["frame_error"]
                aggregate["frames"] += 1
                aggregate["evm_square_sum"] += raw["evm_percent"] ** 2
                aggregate["receiver_complexity_proxy_sum"] += raw["receiver_complexity_proxy"]
                aggregate["receiver_coefficients_per_row_sum"] += raw["receiver_coefficients_per_row"]
                aggregate["receiver_iterations_max"] = max(
                    aggregate["receiver_iterations_max"], raw["receiver_iterations_max"]
                )
                aggregate["frame_error_counts"].append(int(raw["bit_errors"]))
                aggregate["frame_bit_counts"].append(int(raw["total_bits"]))
        for name, aggregate in counters.items():
            aggregate["ber"] = aggregate["bit_errors"] / aggregate["total_bits"]
            aggregate["fer"] = aggregate["frame_errors"] / aggregate["frames"]
            aggregate["evm_percent"] = math.sqrt(aggregate.pop("evm_square_sum") / aggregate["frames"])
            (aggregate["ci_low"], aggregate["ci_high"], aggregate["ci_method"]) = (
                _frame_cluster_bootstrap_interval(
                    aggregate["frame_error_counts"], aggregate["frame_bit_counts"]
                )
            )
            aggregate["independent_frames"] = int(aggregate["frames"])
            aggregate["ci_resamples"] = (
                2000 if aggregate["ci_method"] == "frame_cluster_bootstrap" else 0
            )
            aggregate["receiver_mode"] = receiver_mode
            aggregate["receiver_method"] = (
                "full_lmmse_time_domain" if receiver_mode == "full_mmse" else
                "ofdm_one_tap" if name == "OFDM" else "sparse_pcg_lmmse"
            )
            aggregate["receiver_iterations_max"] = int(aggregate["receiver_iterations_max"])
            aggregate["receiver_coefficients_per_row"] = (
                aggregate.pop("receiver_coefficients_per_row_sum") / aggregate["frames"]
            )
            aggregate["receiver_complexity_proxy"] = (
                aggregate.pop("receiver_complexity_proxy_sum") / aggregate["frames"]
            )
            aggregate["receiver_complexity_scope"] = (
                "equalizer only; excludes waveform transforms, vector updates and CSI estimation"
            )
            aggregate["csi_mode"] = csi_mode
            aggregate["pilot_overhead_fraction"] = float(
                1.0 / (self.T + 1) if csi_mode == "pilot" else 0.0
            )
            aggregate.update(afdm_c1=self.cfg.afdm_c1, afdm_c2=self.cfg.afdm_c2,
                             chirp_convention=CHIRP_CONVENTION, prefix=PREFIX)
        for name, aggregate in counters.items():
            paired_intervals = {}
            for other, other_aggregate in counters.items():
                if other == name:
                    continue
                low, high, method = _paired_frame_difference_interval(
                    aggregate["frame_error_counts"],
                    aggregate["frame_bit_counts"],
                    other_aggregate["frame_error_counts"],
                    other_aggregate["frame_bit_counts"],
                )
                paired_intervals[other] = {
                    "difference": float(aggregate["ber"] - other_aggregate["ber"]),
                    "ci_low": low,
                    "ci_high": high,
                    "ci_method": method,
                    "independent_frames": int(aggregate["frames"]),
                    "ci_resamples": (
                        2000 if method in {
                            "paired_frame_cluster_bootstrap",
                            "paired_frame_cluster_bootstrap_with_zero_error_fer_bounds",
                        } else 0
                    ),
                }
            aggregate["paired_ber_difference_ci"] = paired_intervals
        for aggregate in counters.values():
            aggregate.pop("frame_error_counts")
            aggregate.pop("frame_bit_counts")
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
              stop_event=None, on_point=None, config=None, csi_mode="ideal") -> Dict[str, np.ndarray]:
    """Legacy array API with paired samples, raw BER and no artificial floor.

    One seed serves all waveforms and all axis points (common random numbers),
    making results independent of waveform/point iteration order. Statistics
    with frame-cluster intervals and paired BER-difference intervals are
    available from simulate_stats or the point callback.
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
                                  mod_order, frames, seed_base, receiver_mode,
                                  csi_mode=csi_mode)
    out = {name: [] for name in WAVEFORMS}
    for value in axis_values:
        _check_cancel(stop_event)
        cfo = float(value) if axis_name == "cfo" else fixed_cfo_hz
        doppler = float(value) if axis_name == "doppler" else fixed_doppler_hz
        stats = simulator.simulate_stats(WAVEFORMS, snr_db, cfo, doppler, mod_order,
                                          frames, seed_base, receiver_mode, stop_event,
                                          csi_mode=csi_mode)
        for name in WAVEFORMS:
            out[name].append(stats[name]["ber"])
        if on_point is not None:
            on_point({"axis": axis_name, "axis_value": float(value), "strategies": stats})
    return {name: np.asarray(values, dtype=float) for name, values in out.items()}
