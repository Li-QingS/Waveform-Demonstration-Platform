# -*- coding: utf-8 -*-
"""FDIDM simulation-side predictive alpha/beta adaptive closed loop.

Ports the paper-guided channel-adaptive search mechanism from the hardware
backend (waveform_sim/hardware/fdidm_adaptive.py) into the software simulation
backend.  The search kernel uses the paper ZF/MMSE theory SER (Eq. 40-47) with
coarse + fine grid refinement and a diagonal fast path, matching the hardware
behaviour.  The closed loop is self-contained in this mixin:

    frame hook -> CSI snapshot -> background worker search -> stability / gain /
    cooldown decision -> optional auto-apply via set_indices() -> history.

The mixin is mixed into `_LegacyFDIDMTransceiver`; the compat shell
`FDIDMTransceiver` exposes the public methods through its existing __getattr__
delegation, so no shell changes are required.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_ADAPTIVE_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "auto_apply": True,
    "interval_frames": 8,
    "coarse_step": 0.25,
    "fine_step": 0.05,
    "stability_evals": 2,
    "cooldown_frames": 20,
    "min_improvement_db": 0.5,
    "max_order": 512,
    "rcond": 1e-6,
    "history_limit": 500,
}


class FDIDMSimAdaptiveMixin:
    """Predictive alpha/beta adaptation for the FDIDM software simulation."""

    # ------------------------------------------------------------ init/config
    def _init_adaptive_state_locked(self) -> None:
        """Initialize mixin state.  Must be called after backend self._lock exists."""
        if getattr(self, "_adaptive_initialized", False):
            return
        for name, default in DEFAULT_ADAPTIVE_CONFIG.items():
            attr = "adaptive_" + name
            if not hasattr(self, attr):
                setattr(self, attr, default)
        self._adaptive_lock = threading.RLock()
        self._adaptive_thread: Optional[threading.Thread] = None
        self._adaptive_stop_event = threading.Event()
        self._adaptive_event = threading.Event()
        self._adaptive_snapshot: Optional[Dict[str, Any]] = None
        self._adaptive_last_snapshot: Optional[Dict[str, Any]] = None
        self._adaptive_snapshot_seq = 0
        self._adaptive_recommendation: Dict[str, Any] = {}
        self._adaptive_stable_key: Optional[Tuple[int, int]] = None
        self._adaptive_stable_count = 0
        self._adaptive_eval_seq = 0
        self._adaptive_abs_frame = 0
        self._adaptive_state = "disabled"
        self._adaptive_last_error = ""
        self._adaptive_last_queued_frame = -10 ** 9
        self._adaptive_last_applied_frame = -10 ** 9
        self._adaptive_force_next = False
        self._adaptive_context_key: Optional[Tuple[Any, ...]] = None
        self._adaptive_history: deque = deque(
            maxlen=int(self._adaptive_cfg("history_limit", 500))
        )
        self._adaptive_initialized = True

    def _adaptive_cfg(self, name: str, default: Any = None) -> Any:
        """Read an adaptive_* config value from the host object."""
        return getattr(self, "adaptive_" + str(name), default)

    def update_adaptive_config(self, **cfg) -> None:
        """Runtime update of adaptive_* parameters (names may omit prefix)."""
        with self._adaptive_lock:
            for key, value in dict(cfg).items():
                name = str(key) if str(key).startswith("adaptive_") else "adaptive_" + str(key)
                setattr(self, name, value)
                if name == "adaptive_history_limit":
                    self._adaptive_history = deque(
                        list(self._adaptive_history), maxlen=max(1, int(value))
                    )

    # ------------------------------------------------------------ search kernel
    @staticmethod
    def _adaptive_qam_order(mod_order: str) -> int:
        return {"QPSK": 4, "16QAM": 16, "64QAM": 64}.get(str(mod_order).upper(), 4)

    @staticmethod
    def _adaptive_qfunc(values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        flat = x.reshape(-1)
        out = np.fromiter((0.5 * math.erfc(float(v) / math.sqrt(2.0)) for v in flat),
                          dtype=np.float64, count=flat.size)
        return out.reshape(x.shape)

    def _adaptive_ser_from_symbol_nsr(self, symbol_nsr: np.ndarray, mod_order: str) -> float:
        """Paper Eq. (40), averaged as in Eq. (44)/(47)."""
        xi = np.asarray(symbol_nsr, dtype=np.float64).reshape(-1)
        xi = np.where(np.isfinite(xi), np.maximum(xi, 1e-15), np.inf)
        Q = float(self._adaptive_qam_order(mod_order))
        kappa = 1.0 - 1.0 / math.sqrt(Q)
        argument = math.sqrt(3.0 / max(Q - 1.0, 1.0)) / xi
        q = self._adaptive_qfunc(argument)
        ser = 4.0 * kappa * q - 4.0 * (kappa ** 2) * (q ** 2)
        ser = np.clip(ser, 0.0, 1.0)
        return float(np.mean(ser)) if ser.size else float("inf")

    @staticmethod
    def _adaptive_grid_values(step: float) -> np.ndarray:
        step = float(max(0.01, min(float(step), 2.0)))
        count = int(math.floor(2.0 / step + 1e-9))
        vals = [min(2.0, k * step) for k in range(count + 1)]
        vals.extend([0.0, 1.0, 2.0])
        return np.asarray(sorted(set(round(float(v), 9) for v in vals if -1e-9 <= v <= 2.0 + 1e-9)),
                          dtype=np.float64)

    @staticmethod
    def _adaptive_canonical_index(value: float) -> float:
        v = ((float(value) + 2.0) % 4.0) - 2.0
        if v <= -2.0 + 1e-12:
            v = 2.0
        return float(abs(v))

    @staticmethod
    def _wrap_index(value: float) -> float:
        v = ((float(value) + 2.0) % 4.0) - 2.0
        if v <= -2.0 + 1e-12:
            v = 2.0
        return v

    @staticmethod
    def _ap_weight(p: int, eps: float) -> complex:
        d = float(eps) - float(p)
        return (np.cos(d * np.pi / 4.0)
                * np.cos(2.0 * d * np.pi / 4.0)
                * np.exp(1j * 3.0 * d * np.pi / 4.0))

    @staticmethod
    def _dft_power_apply_axis(arr: np.ndarray, power: int, axis: int) -> np.ndarray:
        """Apply the unitary DFT matrix power F^p along one matrix axis."""
        x = np.asarray(arr, dtype=np.complex128)
        axis = int(axis) % x.ndim
        n = int(x.shape[axis])
        p = int(power) % 4
        if p == 0 or n <= 1:
            return x.astype(np.complex128, copy=True)
        if p == 1:
            return (np.fft.fft(x, axis=axis) / np.sqrt(float(n))).astype(np.complex128)
        if p == 3:
            return (np.fft.ifft(x, axis=axis) * np.sqrt(float(n))).astype(np.complex128)
        idx = np.concatenate(([0], np.arange(n - 1, 0, -1))).astype(np.int64)
        return np.take(x, idx, axis=axis).astype(np.complex128)

    def _apply_gamma_axis(self, arr: np.ndarray, eps: float, axis: int) -> np.ndarray:
        """Apply Gamma^(eps) along one axis via the paper's 4-DFT sum."""
        x = np.asarray(arr, dtype=np.complex128)
        e = self._wrap_index(float(eps))
        out = np.zeros_like(x, dtype=np.complex128)
        for p in range(4):
            w = self._ap_weight(p, e)
            if abs(w) > 1e-14:
                out += w * self._dft_power_apply_axis(x, p, axis=axis)
        return out.astype(np.complex128)

    def _adaptive_prepare_base(self, snapshot: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Prepare the exact paper SER objective in a transform-efficient form.

        For H(alpha,beta)=Phi H_TF Phi^H, right multiplication by unitary Phi^H
        does not change row norms.  Eq. (41)/(46) therefore need only the left
        action of Phi; a diagonal H_TF admits an even cheaper exact form.
        """
        M = int(snapshot["M"]); N = int(snapshot["N"]); K = M * N
        htf_kind = str(snapshot.get("htf_kind", "full"))
        raw = np.asarray(snapshot["htf"], dtype=np.complex128)
        is_diag = htf_kind == "diag" or raw.shape == (M, N) or raw.ndim == 1

        noise_var = float(snapshot.get("noise_var", float("nan")))
        if is_diag:
            diag = raw.reshape(-1, order="F")[:K]
            if diag.size != K:
                raise ValueError(f"adaptive diagonal H_TF size mismatch: {diag.size} != {K}")
            if not np.all(np.isfinite(diag.real)) or not np.all(np.isfinite(diag.imag)):
                raise ValueError("adaptive diagonal H_TF contains NaN/Inf")
            signal_power = float(np.mean(np.abs(diag) ** 2))
            if not np.isfinite(noise_var) or noise_var <= 0.0:
                noise_var = max(signal_power * 1e-3, 1e-9)
            noise_var = max(noise_var, 1e-12)
            equalizer = str(snapshot.get("equalizer", "MMSE")).upper()
            if equalizer == "ZF":
                h_abs = np.abs(diag)
                nz = h_abs[h_abs > 0.0]
                med = float(np.median(nz)) if nz.size else 0.0
                floor = max(med * 1e-3, 1e-10)
                safe_abs = np.maximum(h_abs, floor)
                nsr_power = noise_var / np.maximum(safe_abs ** 2, 1e-20)
            else:
                W = np.conj(diag) / (np.abs(diag) ** 2 + noise_var)
                error = W * diag - 1.0
                nsr_power = np.abs(error) ** 2 + noise_var * np.abs(W) ** 2
            predicted_snr_db = 10.0 * math.log10(max(signal_power / noise_var, 1e-15))
            return {
                "kind": "diag_weights",
                "weights": np.asarray(nsr_power, dtype=np.float64),
            }, float(predicted_snr_db)

        Htf = raw.reshape((K, K)).astype(np.complex128, copy=False)
        if not np.all(np.isfinite(Htf.real)) or not np.all(np.isfinite(Htf.imag)):
            raise ValueError("adaptive H_TF contains NaN/Inf")
        signal_power = float(np.linalg.norm(Htf, "fro") ** 2 / max(K, 1))
        if not np.isfinite(noise_var) or noise_var <= 0.0:
            noise_var = max(signal_power * 1e-3, 1e-9)
        noise_var = max(noise_var, 1e-12)
        xi = math.sqrt(noise_var)
        equalizer = str(snapshot.get("equalizer", "MMSE")).upper()
        rcond = float(snapshot.get("rcond", 1e-6))
        I = np.eye(K, dtype=np.complex128)

        if equalizer == "ZF":
            try:
                inv_htf = np.linalg.solve(Htf, I)
            except np.linalg.LinAlgError:
                inv_htf = np.linalg.pinv(Htf, rcond=rcond)
            base = xi * inv_htf
        else:
            Hh = Htf.conj().T
            gram = Hh @ Htf
            try:
                Wtf = np.linalg.solve(gram + noise_var * I, Hh)
            except np.linalg.LinAlgError:
                load = max(noise_var, rcond * float(np.mean(np.abs(gram)) + 1e-12))
                Wtf = np.linalg.solve(gram + load * I, Hh)
            error_tf = Wtf @ Htf - I
            # Eq. (46): ||row(WH-I)||^2 + xi^2 ||row(W)||^2.
            base = np.concatenate((error_tf, xi * Wtf), axis=1)

        predicted_snr_db = 10.0 * math.log10(max(signal_power / noise_var, 1e-15))
        return {
            "kind": "matrix",
            "matrix": np.asarray(base, dtype=np.complex128),
        }, float(predicted_snr_db)

    def _adaptive_evaluate_candidates(self, prepared: Dict[str, Any], candidates: List[Tuple[float, float]],
                                      M: int, N: int, mod_order: str) -> List[Dict[str, float]]:
        """Evaluate candidate indices with exact paper ZF/MMSE row NSRs."""
        groups: Dict[float, List[float]] = {}
        for alpha, beta in candidates:
            a = round(float(alpha), 9); bb = round(float(beta), 9)
            groups.setdefault(a, []).append(bb)
        results: List[Dict[str, float]] = []

        if str(prepared.get("kind")) == "diag_weights":
            weights = np.asarray(prepared["weights"], dtype=np.float64).reshape((M, N), order="F")
            for alpha in sorted(groups):
                GM2 = np.abs(self._gamma(M, -float(alpha))) ** 2
                left = GM2 @ weights
                for beta in sorted(set(groups[alpha])):
                    GN2 = np.abs(self._gamma(N, float(beta))) ** 2
                    power = left @ GN2.T
                    symbol_nsr = np.sqrt(np.maximum(power.reshape(-1, order="F"), 0.0))
                    ser = self._adaptive_ser_from_symbol_nsr(symbol_nsr, mod_order)
                    results.append({"alpha": float(alpha), "beta": float(beta), "ser": float(ser)})
            return results

        K = int(M * N)
        base = np.asarray(prepared["matrix"], dtype=np.complex128)
        if base.shape[0] != K:
            raise ValueError(f"adaptive base rows mismatch: {base.shape[0]} != {K}")
        cube = base.reshape((M, N, base.shape[1]), order="F")
        for alpha in sorted(groups):
            after_alpha = self._apply_gamma_axis(cube, -float(alpha), axis=0)
            for beta in sorted(set(groups[alpha])):
                transformed = self._apply_gamma_axis(after_alpha, float(beta), axis=1)
                power = np.sum(transformed.real * transformed.real + transformed.imag * transformed.imag, axis=2)
                symbol_nsr = np.sqrt(np.maximum(power.reshape(-1, order="F"), 0.0))
                ser = self._adaptive_ser_from_symbol_nsr(symbol_nsr, mod_order)
                results.append({"alpha": float(alpha), "beta": float(beta), "ser": float(ser)})
        return results

    def _optimize_alpha_beta_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Coarse + fine grid search over (alpha, beta) on one CSI snapshot."""
        t0 = time.time()
        M = int(snapshot["M"]); N = int(snapshot["N"]); K = M * N
        if K > int(snapshot.get("max_order", 512)):
            raise ValueError(f"adaptive search skipped: M*N={K} exceeds max_order={snapshot.get('max_order')}")
        prepared, predicted_snr_db = self._adaptive_prepare_base(snapshot)
        diagonal_fast_path = str(prepared.get("kind")) == "diag_weights"
        coarse_step = float(snapshot.get("coarse_step", 0.25))
        fine_step = float(snapshot.get("fine_step", 0.05))
        effective_coarse = coarse_step if diagonal_fast_path else max(coarse_step, 0.50)
        coarse_vals = self._adaptive_grid_values(effective_coarse)
        coarse_candidates = [(float(a), float(bb)) for a in coarse_vals for bb in coarse_vals]
        coarse_results = self._adaptive_evaluate_candidates(prepared, coarse_candidates, M, N, snapshot["mod_order"])
        coarse_results.sort(key=lambda r: (r["ser"], r["alpha"], r["beta"]))

        refine_candidates = set()
        if diagonal_fast_path:
            radius = max(effective_coarse, 2.0 * fine_step)
            for seed in coarse_results[:3]:
                a0 = float(seed["alpha"]); b0 = float(seed["beta"])
                av = np.arange(max(0.0, a0 - radius), min(2.0, a0 + radius) + 0.5 * fine_step, fine_step)
                bv = np.arange(max(0.0, b0 - radius), min(2.0, b0 + radius) + 0.5 * fine_step, fine_step)
                for a in av:
                    for bb in bv:
                        refine_candidates.add((round(float(np.clip(a, 0.0, 2.0)), 9),
                                               round(float(np.clip(bb, 0.0, 2.0)), 9)))
        else:
            radius = max(0.5 * effective_coarse, 2.0 * fine_step)
            for seed in coarse_results[:2]:
                a0 = float(seed["alpha"]); b0 = float(seed["beta"])
                av = np.arange(max(0.0, a0 - radius), min(2.0, a0 + radius) + 0.5 * fine_step, fine_step)
                bv = np.arange(max(0.0, b0 - radius), min(2.0, b0 + radius) + 0.5 * fine_step, fine_step)
                for a in av:
                    refine_candidates.add((round(float(np.clip(a, 0.0, 2.0)), 9), round(b0, 9)))
                for bb in bv:
                    refine_candidates.add((round(a0, 9), round(float(np.clip(bb, 0.0, 2.0)), 9)))

        current_alpha = float(snapshot.get("alpha", 0.0))
        current_beta = float(snapshot.get("beta", 0.0))
        current_canonical = (round(self._adaptive_canonical_index(current_alpha), 9),
                             round(self._adaptive_canonical_index(current_beta), 9))
        refine_candidates.update({current_canonical, (0.0, 0.0), (1.0, 1.0), (0.5, 1.0)})
        fine_results = self._adaptive_evaluate_candidates(prepared, sorted(refine_candidates), M, N, snapshot["mod_order"])

        by_key: Dict[Tuple[float, float], Dict[str, float]] = {}
        for item in coarse_results + fine_results:
            key = (round(item["alpha"], 9), round(item["beta"], 9))
            old = by_key.get(key)
            if old is None or item["ser"] < old["ser"]:
                by_key[key] = item
        all_results = list(by_key.values())
        all_results.sort(key=lambda r: (r["ser"], r["alpha"], r["beta"]))
        raw_best = all_results[0]

        current_eval = self._adaptive_evaluate_candidates(
            prepared, [(current_alpha, current_beta)], M, N, snapshot["mod_order"]
        )[0]
        current_ser = float(current_eval["ser"])

        tie_limit = float(raw_best["ser"]) * (1.0 + 1e-10) + 1e-15
        tied = [r for r in all_results if r["ser"] <= tie_limit]
        best = min(tied, key=lambda r: ((r["alpha"] - current_canonical[0]) ** 2 +
                                        (r["beta"] - current_canonical[1]) ** 2,
                                        r["alpha"], r["beta"]))

        margin_db = float(snapshot.get("integer_margin_db", 0.0))
        integer_limit = float(best["ser"]) * (10.0 ** (margin_db / 10.0)) + 1e-15
        integer_results = [r for r in all_results
                           if abs(r["alpha"] - round(r["alpha"])) < 1e-9
                           and abs(r["beta"] - round(r["beta"])) < 1e-9
                           and r["ser"] <= integer_limit
                           and r["ser"] < current_ser * (1.0 - 1e-10)]
        if integer_results:
            best = min(integer_results, key=lambda r: (r["ser"],
                                                       (r["alpha"] - current_canonical[0]) ** 2 +
                                                       (r["beta"] - current_canonical[1]) ** 2))

        best_ser = float(best["ser"])
        improvement_db = 10.0 * math.log10(max(current_ser, 1e-15) / max(best_ser, 1e-15))
        ofdm = by_key.get((0.0, 0.0), {"ser": float("nan")})
        otfs = by_key.get((1.0, 1.0), {"ser": float("nan")})
        return {
            "recommended_alpha": float(best["alpha"]),
            "recommended_beta": float(best["beta"]),
            "predicted_ser_current": current_ser,
            "predicted_ser_best": best_ser,
            "predicted_ser_ofdm": float(ofdm.get("ser", float("nan"))),
            "predicted_ser_otfs": float(otfs.get("ser", float("nan"))),
            "predicted_improvement_db": float(improvement_db),
            "predicted_snr_db": float(predicted_snr_db),
            "candidate_count": int(len(all_results) + 1),
            "search_seconds": float(time.time() - t0),
            "search_mode": "diag_2d" if diagonal_fast_path else "full_coordinate",
            "htf_source": str(snapshot.get("htf_source", "unknown")),
            "htf_kind": str(snapshot.get("htf_kind", "unknown")),
            "equalizer": str(snapshot.get("equalizer", "")),
            "mod_order": str(snapshot.get("mod_order", "")),
            "frame_counter": int(snapshot.get("frame_counter", 0)),
            "snapshot_seq": int(snapshot.get("snapshot_seq", 0)),
        }

    # ------------------------------------------------------------ context/schedule
    def _adaptive_context_key_locked(self) -> Tuple[Any, ...]:
        """Context that changes the paper SER objective (excluding alpha/beta)."""
        cfg = self.config
        return (
            int(getattr(cfg, "m_subcarriers", 0)), int(getattr(cfg, "n_symbols", 0)),
            str(getattr(cfg, "mod_order", "")), str(getattr(cfg, "decoder", "")),
            str(getattr(cfg, "channel_model", "")), float(getattr(cfg, "velocity_kmh", 0.0)),
            float(getattr(cfg, "doppler_radial_factor", 0.10)),
            float(getattr(cfg, "ebn0_db", 0.0)), str(getattr(cfg, "snr_definition", "Eb/N0")),
            int(getattr(cfg, "channel_seed", 0)), bool(getattr(cfg, "dynamic_channel", False)),
            int(getattr(cfg, "channel_coherence_frames", 1)),
            str(getattr(cfg, "channel_dynamics", "fixed")),
            int(getattr(cfg, "fast_channel_coherence_symbols", 1)),
            bool(getattr(cfg, "circular_channel", True)),
            float(getattr(cfg, "tf_notch_depth_db", 0.0)),
            int(getattr(cfg, "tf_notch_count", 0)),
            str(getattr(cfg, "link_mode", "matrix")),
            str(getattr(cfg, "search_objective", "zf_theory_ser")),
        )

    def _adaptive_invalidate_locked(self, reason: str = "context_change") -> None:
        """Drop stale recommendations/stable counts (history is preserved)."""
        self._adaptive_recommendation = {}
        self._adaptive_stable_key = None
        self._adaptive_stable_count = 0
        self._adaptive_state = "waiting_channel" if self._adaptive_cfg("enabled", False) else "disabled"
        self._adaptive_last_error = ""

    def _adaptive_note_frame_processed(self) -> None:
        """Frame hook: queue a CSI snapshot when interval/context requires it."""
        if not self._adaptive_cfg("enabled", False):
            return
        self._ensure_adaptive_worker()
        with self._lock:
            if getattr(self, "_H_tf", None) is None:
                return
            htf = np.asarray(self._H_tf, dtype=np.complex128).copy()
            cfg = self.config
            M = int(cfg.m_subcarriers); N = int(cfg.n_symbols); K = M * N
            ctx = self._adaptive_context_key_locked()
            frame = int(self._adaptive_abs_frame) + 1
            mod_order = str(getattr(cfg, "mod_order", "16QAM"))
            equalizer = str(getattr(cfg, "decoder", "ZF"))
            alpha = float(getattr(cfg, "alpha", 0.0)); beta = float(getattr(cfg, "beta", 0.0))
            noise_var = float(self._noise_variance()) if hasattr(self, "_noise_variance") else float("nan")
            htf_kind = "diag" if htf.ndim == 1 or htf.shape == (M, N) else "full"

        with self._adaptive_lock:
            if not self._adaptive_cfg("enabled", False):
                return
            self._adaptive_abs_frame = int(frame)
            if ctx != self._adaptive_context_key:
                self._adaptive_invalidate_locked(reason="context_changed")
                self._adaptive_context_key = ctx
            if K > int(self._adaptive_cfg("max_order", 512)):
                self._adaptive_state = "order_limited"
                self._adaptive_last_error = f"M*N={K} exceeds adaptive max_order"
                return
            force = bool(self._adaptive_force_next)
            if not force:
                if frame - int(self._adaptive_last_applied_frame) < int(self._adaptive_cfg("cooldown_frames", 20)):
                    self._adaptive_state = "cooldown"
                    return
                if frame - int(self._adaptive_last_queued_frame) < int(self._adaptive_cfg("interval_frames", 8)):
                    return
            self._adaptive_force_next = False
            self._adaptive_snapshot_seq += 1
            snapshot = {
                "snapshot_seq": int(self._adaptive_snapshot_seq),
                "frame_counter": int(frame),
                "M": int(M), "N": int(N),
                "alpha": float(alpha), "beta": float(beta),
                "mod_order": str(mod_order), "equalizer": str(equalizer),
                "noise_var": float(noise_var),
                "htf": htf, "htf_kind": str(htf_kind), "htf_source": "simulation",
                "coarse_step": float(self._adaptive_cfg("coarse_step", 0.25)),
                "fine_step": float(self._adaptive_cfg("fine_step", 0.05)),
                "integer_margin_db": 0.0,
                "max_order": int(self._adaptive_cfg("max_order", 512)),
                "rcond": float(self._adaptive_cfg("rcond", 1e-6)),
            }
            self._adaptive_snapshot = snapshot
            self._adaptive_last_snapshot = snapshot
            self._adaptive_last_queued_frame = int(frame)
            self._adaptive_state = "queued"
            self._adaptive_last_error = ""
            self._adaptive_event.set()

    def _ensure_adaptive_worker(self) -> None:
        with self._adaptive_lock:
            thread = self._adaptive_thread
            if thread is not None and thread.is_alive():
                return
            self._adaptive_stop_event = threading.Event()
            thread = threading.Thread(
                target=self._adaptive_worker,
                name=f"fdidm-sim-adaptive-{id(self):x}",
                daemon=True,
            )
            self._adaptive_thread = thread
            thread.start()

    # ------------------------------------------------------------ worker/decision
    def _adaptive_worker(self) -> None:
        while not self._adaptive_stop_event.is_set():
            if not self._adaptive_event.wait(timeout=0.5):
                continue
            self._adaptive_event.clear()
            with self._adaptive_lock:
                snapshot = self._adaptive_snapshot
                if snapshot is None or not self._adaptive_cfg("enabled", False):
                    continue
                snapshot = dict(snapshot)
                expected_seq = int(snapshot.get("snapshot_seq", -1))
                self._adaptive_state = "optimizing"
                self._adaptive_last_error = ""
            try:
                result = self._optimize_alpha_beta_snapshot(snapshot)
            except Exception as exc:
                with self._adaptive_lock:
                    if (expected_seq == int(self._adaptive_snapshot_seq)
                            and self._adaptive_cfg("enabled", False)):
                        self._adaptive_last_error = f"{type(exc).__name__}: {exc}"
                        self._adaptive_state = "error"
                continue

            pending_apply = None
            with self._adaptive_lock:
                if expected_seq != int(self._adaptive_snapshot_seq):
                    continue
                if not self._adaptive_cfg("enabled", False):
                    continue
                fine = max(float(self._adaptive_cfg("fine_step", 0.05)), 1e-9)
                key = (int(round(float(result["recommended_alpha"]) / fine)),
                       int(round(float(result["recommended_beta"]) / fine)))
                if key == self._adaptive_stable_key:
                    self._adaptive_stable_count += 1
                else:
                    self._adaptive_stable_key = key
                    self._adaptive_stable_count = 1
                self._adaptive_eval_seq += 1
                rec = dict(result)
                rec["recommendation_seq"] = int(self._adaptive_eval_seq)
                rec["stable_count"] = int(self._adaptive_stable_count)
                rec["stable_required"] = int(self._adaptive_cfg("stability_evals", 2))
                rec["minimum_improvement_db"] = float(self._adaptive_cfg("min_improvement_db", 0.5))
                delta = abs(float(rec["recommended_alpha"]) - float(snapshot.get("alpha", 0.0))) + \
                        abs(float(rec["recommended_beta"]) - float(snapshot.get("beta", 0.0)))
                required_gain = max(float(self._adaptive_cfg("min_improvement_db", 0.5)), 1e-6)
                stable_ok = int(self._adaptive_stable_count) >= int(self._adaptive_cfg("stability_evals", 2))
                ready = (float(rec["predicted_improvement_db"]) >= required_gain
                         and stable_ok and delta >= 0.5 * fine)
                rec["ready"] = bool(ready)
                rec["pending"] = bool(ready)
                auto_apply = bool(self._adaptive_cfg("auto_apply", True))
                if ready and auto_apply:
                    action = "apply"
                elif not stable_ok:
                    action = "stable_pending"
                elif float(rec["predicted_improvement_db"]) < required_gain:
                    action = "gain_below"
                else:
                    action = "skip"
                self._adaptive_recommendation = rec
                self._adaptive_state = "ready" if ready else "tracking"
                self._adaptive_history.append({
                    "kind": "eval",
                    "seq": int(self._adaptive_eval_seq),
                    "frame": int(snapshot.get("frame_counter", 0)),
                    "alpha": float(snapshot.get("alpha", 0.0)),
                    "beta": float(snapshot.get("beta", 0.0)),
                    "rec_alpha": float(rec["recommended_alpha"]),
                    "rec_beta": float(rec["recommended_beta"]),
                    "ser_current": float(rec["predicted_ser_current"]),
                    "ser_best": float(rec["predicted_ser_best"]),
                    "ser_ofdm": float(rec["predicted_ser_ofdm"]),
                    "gain_db": float(rec["predicted_improvement_db"]),
                    "action": action,
                    "stable_count": int(self._adaptive_stable_count),
                    "candidates": int(rec["candidate_count"]),
                    "seconds": float(rec["search_seconds"]),
                    "state": str(self._adaptive_state),
                    "ts": float(time.time()),
                })
                if ready and auto_apply:
                    pending_apply = (float(rec["recommended_alpha"]), float(rec["recommended_beta"]),
                                     float(rec["predicted_improvement_db"]),
                                     int(snapshot.get("frame_counter", 0)))
                    self._adaptive_state = "applying"

            if pending_apply is not None:
                a, b, gain, frame = pending_apply
                from_alpha = float(snapshot.get("alpha", 0.0))
                from_beta = float(snapshot.get("beta", 0.0))
                try:
                    self.set_indices(a, b)
                except Exception as exc:
                    with self._adaptive_lock:
                        self._adaptive_last_error = f"apply failed: {type(exc).__name__}: {exc}"
                        self._adaptive_state = "error"
                    continue
                with self._adaptive_lock:
                    if self._adaptive_cfg("enabled", False):
                        self._adaptive_last_applied_frame = int(frame)
                        self._adaptive_state = "applied"
                        self._adaptive_history.append({
                            "kind": "switch",
                            "seq": int(self._adaptive_eval_seq),
                            "frame": int(frame),
                            "from_alpha": float(from_alpha),
                            "from_beta": float(from_beta),
                            "to_alpha": float(a),
                            "to_beta": float(b),
                            "gain_db": float(gain),
                            "reason": "stable_ready",
                            "ts": float(time.time()),
                        })

    # ------------------------------------------------------------ public API
    def start_adaptive_tuning(self, **cfg) -> bool:
        """Enable adaptation and ensure the background worker is running."""
        if cfg:
            self.update_adaptive_config(**cfg)
        if not self._adaptive_cfg("enabled", False):
            setattr(self, "adaptive_enabled", True)
        self._ensure_adaptive_worker()
        with self._adaptive_lock:
            self._adaptive_force_next = True
            self._adaptive_last_error = ""
            if self._adaptive_state == "disabled":
                self._adaptive_state = "waiting_channel"
            self._adaptive_event.set()
        return True

    def stop_adaptive_tuning(self) -> None:
        """Disable adaptation and stop the background worker."""
        with self._adaptive_lock:
            setattr(self, "adaptive_enabled", False)
            self._adaptive_state = "disabled"
            self._adaptive_force_next = False
        self._adaptive_stop_event.set()
        self._adaptive_event.set()
        thread = self._adaptive_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        with self._adaptive_lock:
            self._adaptive_thread = None

    def request_adaptive_evaluation(self) -> bool:
        """Request an immediate search using the latest CSI, or the next valid frame."""
        if not self._adaptive_cfg("enabled", False):
            return False
        self._ensure_adaptive_worker()
        with self._adaptive_lock:
            last = self._adaptive_last_snapshot
            if last is None:
                self._adaptive_force_next = True
                self._adaptive_state = "waiting_channel"
                return False
            self._adaptive_snapshot_seq += 1
            snap = dict(last)
            snap["snapshot_seq"] = int(self._adaptive_snapshot_seq)
            snap["frame_counter"] = int(self._adaptive_abs_frame)
            snap["alpha"] = float(self.config.alpha)
            snap["beta"] = float(self.config.beta)
            snap["mod_order"] = str(self.config.mod_order)
            snap["equalizer"] = str(self.config.decoder)
            snap["coarse_step"] = float(self._adaptive_cfg("coarse_step", 0.25))
            snap["fine_step"] = float(self._adaptive_cfg("fine_step", 0.05))
            snap["max_order"] = int(self._adaptive_cfg("max_order", 512))
            snap["rcond"] = float(self._adaptive_cfg("rcond", 1e-6))
            self._adaptive_snapshot = snap
            self._adaptive_state = "queued"
            self._adaptive_last_error = ""
            self._adaptive_event.set()
            return True

    def get_adaptive_status(self) -> Dict[str, Any]:
        with self._adaptive_lock:
            rec = dict(self._adaptive_recommendation or {})
            return {
                "enabled": bool(self._adaptive_cfg("enabled", False)),
                "auto_apply": bool(self._adaptive_cfg("auto_apply", True)),
                "state": str(self._adaptive_state),
                "last_error": str(self._adaptive_last_error),
                "ready": bool(rec.get("ready", False)),
                "pending": bool(rec.get("pending", False)),
                "recommendation_seq": int(rec.get("recommendation_seq", 0)),
                "recommended_alpha": float(rec.get("recommended_alpha", float("nan"))),
                "recommended_beta": float(rec.get("recommended_beta", float("nan"))),
                "predicted_ser_current": float(rec.get("predicted_ser_current", float("nan"))),
                "predicted_ser_best": float(rec.get("predicted_ser_best", float("nan"))),
                "predicted_ser_ofdm": float(rec.get("predicted_ser_ofdm", float("nan"))),
                "predicted_ser_otfs": float(rec.get("predicted_ser_otfs", float("nan"))),
                "predicted_improvement_db": float(rec.get("predicted_improvement_db", float("nan"))),
                "predicted_snr_db": float(rec.get("predicted_snr_db", float("nan"))),
                "stable_count": int(rec.get("stable_count", 0)),
                "stable_required": int(rec.get("stable_required", self._adaptive_cfg("stability_evals", 2))),
                "candidate_count": int(rec.get("candidate_count", 0)),
                "search_seconds": float(rec.get("search_seconds", float("nan"))),
                "context_key": repr(self._adaptive_context_key),
                "history_len": int(len(self._adaptive_history)),
                "coarse_step": float(self._adaptive_cfg("coarse_step", 0.25)),
                "fine_step": float(self._adaptive_cfg("fine_step", 0.05)),
                "interval_frames": int(self._adaptive_cfg("interval_frames", 8)),
                "cooldown_frames": int(self._adaptive_cfg("cooldown_frames", 20)),
                "min_improvement_db": float(self._adaptive_cfg("min_improvement_db", 0.5)),
                "max_order": int(self._adaptive_cfg("max_order", 512)),
                "history_limit": int(self._adaptive_cfg("history_limit", 500)),
            }

    def get_alpha_beta_adaptation_status(self) -> Dict[str, Any]:
        """Compatibility alias for the hardware-style status hook."""
        return self.get_adaptive_status()

    def get_adaptive_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._adaptive_lock:
            hist = list(self._adaptive_history)
        if limit is not None:
            hist = hist[-int(limit):]
        return hist
