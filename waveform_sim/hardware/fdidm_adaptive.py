"""FDIDM ????????? fdidm_hardtest.py ????? 6c??"""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FDIDMAdaptiveMixin:
    # =========================================================
    # Paper-guided channel-adaptive alpha/beta optimization
    # =========================================================
    def _alpha_beta_adaptation_context_key(self) -> Tuple[Any, ...]:
        """Context that changes the paper SER objective, excluding alpha/beta."""
        return (
            int(getattr(self, "M", 0)), int(getattr(self, "N", 0)),
            int(getattr(self, "cp_len", 0)), str(getattr(self, "mod_order", "")),
            str(getattr(self, "equalizer", "")), str(getattr(self, "channel_estimator", "")),
            str(getattr(self, "channel_mode", "")), round(float(getattr(self, "sample_rate", 0.0)), 3),
            round(float(getattr(self, "carrier_freq", 0.0)), 3),
            round(float(getattr(self, "tx_gain", 0.0)), 3),
            round(float(getattr(self, "rx_gain", 0.0)), 3),
            round(float(getattr(self, "training_amplitude", 0.0)), 6),
            round(float(getattr(self, "tdl_rms_delay_spread_ns", 0.0)), 6),
            round(float(getattr(self, "tdl_doppler_hz", 0.0)), 6),
            round(float(getattr(self, "tdl_doppler_spread_hz", 0.0)), 6),
            round(float(getattr(self, "tdl_snr_db", 0.0)), 6),
        )

    def _invalidate_alpha_beta_adaptation(self, reason: str = "context_change", cooldown: bool = False):
        lock = getattr(self, "_adaptive_ab_lock", None)
        if lock is None:
            return
        with lock:
            self._adaptive_ab_snapshot_seq = int(getattr(self, "_adaptive_ab_snapshot_seq", 0)) + 1
            self._adaptive_ab_snapshot = None
            # A snapshot from the previous channel/configuration is not valid
            # for a later manual request.  Keep the invalidation boundary
            # explicit so a restarted run cannot reuse stale CSI.
            self._adaptive_ab_last_snapshot = None
            self._adaptive_ab_recommendation = {}
            self._adaptive_ab_stable_key = None
            self._adaptive_ab_stable_count = 0
            self._adaptive_ab_last_htf_identity = None
            self._adaptive_ab_last_error = ""
            if cooldown:
                self._adaptive_ab_last_applied_frame = int(getattr(self, "_frames_processed", 0))
                self._adaptive_ab_state = "cooldown"
            else:
                # _frames_processed is reset on start() and on a new channel
                # context.  Reset both frame gates with it; otherwise the new
                # run waits until it exceeds the old run's frame number.
                self._adaptive_ab_last_queued_frame = -10**18
                self._adaptive_ab_last_applied_frame = -10**18
                self._adaptive_ab_state = "waiting_channel" if self.adaptive_alpha_beta_enable else "disabled"
            self._adaptive_ab_last_skip_reason = ""
            self._adaptive_ab_last_skip_log_wall = 0.0
        try:
            self._debug("DEBUG", f"alpha/beta adaptation invalidated: {reason}")
        except Exception:
            pass

    def _note_alpha_beta_changed_for_adaptation(self, reason: str = "alpha_beta_changed"):
        """Cancel stale recommendations after either manual or adaptive switching."""
        self._invalidate_alpha_beta_adaptation(reason=reason, cooldown=True)

    def _ensure_alpha_beta_adaptation_worker(self):
        if not bool(getattr(self, "adaptive_alpha_beta_enable", False)):
            return
        with self._adaptive_ab_lock:
            thread = getattr(self, "_adaptive_ab_thread", None)
            if thread is not None and thread.is_alive():
                return
            self._adaptive_ab_stop.clear()
            thread = threading.Thread(target=self._alpha_beta_adaptation_worker,
                                      name=f"fdidm-ab-opt-{id(self):x}", daemon=True)
            self._adaptive_ab_thread = thread
            thread.start()

    def _adaptive_ab_debug_skip(self, reason: str):
        """Rate-limit diagnostic messages for per-frame adaptation gates."""
        now = time.time()
        previous = str(getattr(self, "_adaptive_ab_last_skip_reason", ""))
        previous_wall = float(getattr(self, "_adaptive_ab_last_skip_log_wall", 0.0))
        if reason != previous or now - previous_wall >= 5.0:
            self._adaptive_ab_last_skip_reason = str(reason)
            self._adaptive_ab_last_skip_log_wall = now
            try:
                self._debug("DEBUG", f"alpha/beta adaptation skipped: {reason}")
            except Exception:
                pass

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

    def _adaptive_prepare_base(self, snapshot: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Prepare the exact paper SER objective in a transform-efficient form.

        For H(alpha,beta)=Phi H_TF Phi^H, right multiplication by unitary Phi^H
        does not change row norms.  Eq. (41) and Eq. (46) therefore require only
        the left action of Phi.  A diagonal H_TF admits an even cheaper exact
        form: row powers are |Phi|^2 times a per-TF-cell NSR vector.
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
        xi = math.sqrt(noise_var)  # normalized QAM symbols have Es=1
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
            # For diagonal H_TF, diag(Phi C Phi^H) equals
            # |Gamma_M(-alpha)|^2 W |Gamma_N(beta)|^{2T} exactly.
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
        t0 = time.time()
        M = int(snapshot["M"]); N = int(snapshot["N"]); K = M * N
        if K > int(snapshot.get("max_order", 512)):
            raise ValueError(f"adaptive search skipped: M*N={K} exceeds max_order={snapshot.get('max_order')}")

        prepared, predicted_snr_db = self._adaptive_prepare_base(snapshot)
        diagonal_fast_path = str(prepared.get("kind")) == "diag_weights"
        coarse_step = float(max(snapshot.get("coarse_step", 0.25), 0.01))
        fine_step = float(max(snapshot.get("fine_step", 0.05), 0.01))
        active_step = float(getattr(self, "_adaptive_active_step", coarse_step))
        if not np.isfinite(active_step) or active_step <= 0.0:
            active_step = coarse_step
        active_step = min(active_step, coarse_step)

        current_alpha = float(np.clip(snapshot.get("alpha", 0.0), 0.0, 2.0))
        current_beta = float(np.clip(snapshot.get("beta", 0.0), 0.0, 2.0))
        current = (round(current_alpha, 9), round(current_beta, 9))
        step = active_step
        candidates = {current}
        for da, db in ((-step, 0.0), (step, 0.0), (0.0, -step), (0.0, step)):
            candidates.add((round(float(np.clip(current_alpha + da, 0.0, 2.0)), 9),
                           round(float(np.clip(current_beta + db, 0.0, 2.0)), 9)))
        results = self._adaptive_evaluate_candidates(
            prepared, sorted(candidates), M, N, snapshot["mod_order"]
        )
        by_key = {(round(float(r["alpha"]), 9), round(float(r["beta"]), 9)): r for r in results}
        current_eval = by_key[current]
        current_ser = float(current_eval["ser"])
        # A span below this floor is measurement/numerical noise, not an
        # observable axis.  The relative part scales with the current SER while
        # the absolute part keeps near-zero SER from producing false switches.
        flat_floor = max(1e-12, abs(current_ser) * 1e-3)
        alpha_neighbours = [r for key, r in by_key.items() if key[1] == current[1] and key[0] != current[0]]
        beta_neighbours = [r for key, r in by_key.items() if key[0] == current[0] and key[1] != current[1]]
        alpha_span = (max([current_ser] + [float(r["ser"]) for r in alpha_neighbours]) -
                      min([current_ser] + [float(r["ser"]) for r in alpha_neighbours])) if alpha_neighbours else 0.0
        beta_span = (max([current_ser] + [float(r["ser"]) for r in beta_neighbours]) -
                     min([current_ser] + [float(r["ser"]) for r in beta_neighbours])) if beta_neighbours else 0.0
        alpha_observable = bool(alpha_span > flat_floor)
        beta_observable = bool(beta_span > flat_floor)

        eligible = [r for r in results if r is not current_eval]
        eligible = [r for r in eligible if not (
            (float(r["alpha"]) != current[0] and not alpha_observable) or
            (float(r["beta"]) != current[1] and not beta_observable)
        )]
        best = min([current_eval] + eligible, key=lambda r: (float(r["ser"]),
                                                               abs(float(r["alpha"]) - current[0]) +
                                                               abs(float(r["beta"]) - current[1])))
        best_ser = float(best["ser"])
        improvement_db = 10.0 * math.log10(max(current_ser, 1e-15) / max(best_ser, 1e-15))
        min_gain_db = float(snapshot.get("min_improvement_db", 0.0))
        meaningful = bool(best is not current_eval and improvement_db >= min_gain_db)

        # At the coarse step, a failed move simply arms the fine step for the
        # next snapshot.  A successful move remains one coarse step at a time.
        next_step = step if meaningful or step <= fine_step + 1e-12 else fine_step
        recommended = best if meaningful else current_eval
        direction = "none"
        if meaningful:
            if float(recommended["alpha"]) != current[0]:
                direction = "alpha"
            elif float(recommended["beta"]) != current[1]:
                direction = "beta"
        self._adaptive_active_step = float(next_step)
        return {
            "recommended_alpha": float(recommended["alpha"]),
            "recommended_beta": float(recommended["beta"]),
            "predicted_ser_current": current_ser,
            "predicted_ser_best": float(recommended["ser"]),
            "predicted_ser_ofdm": float(by_key.get((0.0, 0.0), {}).get("ser", float("nan"))),
            "predicted_ser_otfs": float(by_key.get((1.0, 1.0), {}).get("ser", float("nan"))),
            "predicted_improvement_db": float(improvement_db if meaningful else 0.0),
            "predicted_snr_db": float(predicted_snr_db),
            "candidate_count": int(len(results)),
            "search_seconds": float(time.time() - t0),
            "search_mode": "diag_five_point" if diagonal_fast_path else "full_five_point",
            "active_step": float(step),
            "next_active_step": float(next_step),
            "selected_direction": direction,
            "alpha_span": float(alpha_span),
            "beta_span": float(beta_span),
            "alpha_observable": alpha_observable,
            "beta_observable": beta_observable,
            "htf_source": str(snapshot.get("htf_source", "unknown")),
            "htf_kind": str(snapshot.get("htf_kind", "unknown")),
            "equalizer": str(snapshot.get("equalizer", "")),
            "mod_order": str(snapshot.get("mod_order", "")),
            "frame_counter": int(snapshot.get("frame_counter", 0)),
            "snapshot_seq": int(snapshot.get("snapshot_seq", 0)),
        }

    def _alpha_beta_adaptation_worker(self):
        while not self._adaptive_ab_stop.is_set():
            if not self._adaptive_ab_event.wait(timeout=0.5):
                continue
            self._adaptive_ab_event.clear()
            with self._adaptive_ab_lock:
                snapshot = self._adaptive_ab_snapshot
                if snapshot is None or not self.adaptive_alpha_beta_enable:
                    continue
                snapshot = dict(snapshot)
                expected_seq = int(snapshot.get("snapshot_seq", -1))
                self._adaptive_ab_state = "optimizing"
                self._adaptive_ab_last_error = ""
            try:
                result = self._optimize_alpha_beta_snapshot(snapshot)
            except Exception as exc:
                with self._adaptive_ab_lock:
                    if expected_seq == int(self._adaptive_ab_snapshot_seq):
                        self._adaptive_ab_last_error = f"{type(exc).__name__}: {exc}"
                        self._adaptive_ab_state = "error"
                self._debug("WARN", f"alpha/beta optimizer failed: {type(exc).__name__}: {exc}")
                continue

            with self._adaptive_ab_lock:
                if expected_seq != int(self._adaptive_ab_snapshot_seq):
                    # Alpha/beta or channel context changed during the search.
                    continue
                fine = max(float(self.adaptive_alpha_beta_fine_step), 1e-9)
                key = (int(round(float(result["recommended_alpha"]) / fine)),
                       int(round(float(result["recommended_beta"]) / fine)))
                if key == self._adaptive_ab_stable_key:
                    self._adaptive_ab_stable_count += 1
                else:
                    self._adaptive_ab_stable_key = key
                    self._adaptive_ab_stable_count = 1
                self._adaptive_ab_eval_seq += 1
                result["recommendation_seq"] = int(self._adaptive_ab_eval_seq)
                result["stable_count"] = int(self._adaptive_ab_stable_count)
                result["stable_required"] = int(self.adaptive_alpha_beta_stability_evals)
                result["minimum_improvement_db"] = float(self.adaptive_alpha_beta_min_improvement_db)
                delta = abs(float(result["recommended_alpha"]) - float(snapshot.get("alpha", 0.0))) + \
                        abs(float(result["recommended_beta"]) - float(snapshot.get("beta", 0.0)))
                required_gain = max(float(self.adaptive_alpha_beta_min_improvement_db), 1e-6)
                ready = (float(result["predicted_improvement_db"]) >= required_gain
                         and int(self._adaptive_ab_stable_count) >= int(self.adaptive_alpha_beta_stability_evals)
                         and delta >= 0.5 * fine)
                result["ready"] = bool(ready)
                result["pending"] = bool(ready)
                self._adaptive_ab_recommendation = result
                self._adaptive_ab_state = "ready" if ready else "tracking"
            level = "INFO" if result.get("ready") else "DEBUG"
            self._debug(level,
                        "alpha/beta SER search: "
                        f"current={result['predicted_ser_current']:.3e}, best={result['predicted_ser_best']:.3e}, "
                        f"gain={result['predicted_improvement_db']:.2f}dB, "
                        f"rec=({result['recommended_alpha']:.2f},{result['recommended_beta']:.2f}), "
                        f"stable={result['stable_count']}/{result['stable_required']}, "
                        f"source={result['htf_source']}, time={result['search_seconds']:.3f}s")

    def _maybe_queue_alpha_beta_adaptation(self, h_tf_est: Any, htf_kind: str,
                                           htf_source: str, noise_var: float,
                                           sync_metric: float, good_quality: bool):
        if not bool(getattr(self, "adaptive_alpha_beta_enable", False)):
            return
        self._ensure_alpha_beta_adaptation_worker()
        K = int(self.M * self.N)
        if K > int(self.adaptive_alpha_beta_max_order):
            with self._adaptive_ab_lock:
                self._adaptive_ab_state = "order_limited"
                self._adaptive_ab_last_error = f"M*N={K} > adaptive max_order={self.adaptive_alpha_beta_max_order}"
            self._adaptive_ab_debug_skip(f"order_limited(M*N={K}, max={self.adaptive_alpha_beta_max_order})")
            return
        if h_tf_est is None or not np.isfinite(float(sync_metric)):
            self._adaptive_ab_debug_skip("invalid_channel_snapshot")
            return
        if float(sync_metric) < float(self.adaptive_alpha_beta_min_sync_metric):
            self._adaptive_ab_debug_skip(
                f"sync_below_threshold({float(sync_metric):.3f} < {float(self.adaptive_alpha_beta_min_sync_metric):.3f})")
            return
        if self.adaptive_alpha_beta_require_good_frame and not bool(good_quality):
            self._adaptive_ab_debug_skip("frame_quality_gate")
            return

        frame_counter = int(getattr(self, "_frames_processed", 0))
        htf_identity = (str(htf_source), id(h_tf_est))
        with self._adaptive_ab_lock:
            force = bool(self._adaptive_ab_force_next)
            if (not force and str(htf_source) == "full_htf" and bool(getattr(self, "full_htf_once", False))
                    and htf_identity == self._adaptive_ab_last_htf_identity):
                self._adaptive_ab_debug_skip("full_htf_once_reuse")
                return
            if not force:
                if frame_counter - int(self._adaptive_ab_last_applied_frame) < int(self.adaptive_alpha_beta_cooldown_frames):
                    self._adaptive_ab_state = "cooldown"
                    self._adaptive_ab_debug_skip(
                        f"cooldown(frame={frame_counter}, applied={self._adaptive_ab_last_applied_frame}, "
                        f"need={self.adaptive_alpha_beta_cooldown_frames})")
                    return
                if frame_counter - int(self._adaptive_ab_last_queued_frame) < int(self.adaptive_alpha_beta_interval_frames):
                    self._adaptive_ab_debug_skip(
                        f"interval(frame={frame_counter}, queued={self._adaptive_ab_last_queued_frame}, "
                        f"need={self.adaptive_alpha_beta_interval_frames})")
                    return
            self._adaptive_ab_force_next = False

        raw = np.asarray(h_tf_est, dtype=np.complex128)
        if str(htf_kind) == "diag" or raw.shape == (self.M, self.N):
            htf_payload = raw.reshape(-1, order="F").copy()
            htf_kind = "diag"
        else:
            if raw.size != K * K:
                self._adaptive_ab_debug_skip(f"invalid_full_shape(size={raw.size}, expected={K * K})")
                return
            htf_payload = raw.reshape((K, K)).copy()
            htf_kind = "full"
        if not np.all(np.isfinite(htf_payload.real)) or not np.all(np.isfinite(htf_payload.imag)):
            self._adaptive_ab_debug_skip("nonfinite_channel_snapshot")
            return

        with self._adaptive_ab_lock:
            self._adaptive_ab_snapshot_seq += 1
            snapshot = {
                "snapshot_seq": int(self._adaptive_ab_snapshot_seq),
                "frame_counter": frame_counter,
                "M": int(self.M), "N": int(self.N),
                "alpha": float(self.alpha), "beta": float(self.beta),
                "mod_order": str(self.mod_order), "equalizer": str(self.equalizer),
                "noise_var": float(noise_var), "sync_metric": float(sync_metric),
                "htf": htf_payload, "htf_kind": str(htf_kind), "htf_source": str(htf_source),
                "coarse_step": float(self.adaptive_alpha_beta_coarse_step),
                "fine_step": float(self.adaptive_alpha_beta_fine_step),
                "min_improvement_db": float(self.adaptive_alpha_beta_min_improvement_db),
                "integer_margin_db": float(self.adaptive_alpha_beta_integer_margin_db),
                "max_order": int(self.adaptive_alpha_beta_max_order),
                "rcond": float(self.adaptive_alpha_beta_rcond),
            }
            self._adaptive_ab_snapshot = snapshot
            self._adaptive_ab_last_snapshot = snapshot
            self._adaptive_ab_last_queued_frame = frame_counter
            self._adaptive_ab_last_htf_identity = htf_identity
            self._adaptive_ab_state = "queued"
            self._adaptive_ab_last_error = ""
            self._adaptive_ab_event.set()

    def request_alpha_beta_adaptation(self) -> bool:
        """Request an immediate search using the latest H_TF, or the next valid frame."""
        if not bool(getattr(self, "adaptive_alpha_beta_enable", False)):
            return False
        self._ensure_alpha_beta_adaptation_worker()
        with self._adaptive_ab_lock:
            last = self._adaptive_ab_last_snapshot
            if last is None:
                self._adaptive_ab_force_next = True
                self._adaptive_ab_state = "waiting_channel"
                return False
            self._adaptive_ab_snapshot_seq += 1
            snap = dict(last)
            snap["snapshot_seq"] = int(self._adaptive_ab_snapshot_seq)
            snap["frame_counter"] = int(getattr(self, "_frames_processed", 0))
            snap["alpha"] = float(self.alpha); snap["beta"] = float(self.beta)
            snap["mod_order"] = str(self.mod_order); snap["equalizer"] = str(self.equalizer)
            snap["coarse_step"] = float(self.adaptive_alpha_beta_coarse_step)
            snap["fine_step"] = float(self.adaptive_alpha_beta_fine_step)
            snap["min_improvement_db"] = float(self.adaptive_alpha_beta_min_improvement_db)
            snap["integer_margin_db"] = float(self.adaptive_alpha_beta_integer_margin_db)
            snap["max_order"] = int(self.adaptive_alpha_beta_max_order)
            snap["rcond"] = float(self.adaptive_alpha_beta_rcond)
            self._adaptive_ab_snapshot = snap
            self._adaptive_ab_last_snapshot = snap
            self._adaptive_ab_state = "queued"
            self._adaptive_ab_event.set()
            return True

    def get_alpha_beta_adaptation_status(self) -> Dict[str, Any]:
        lock = getattr(self, "_adaptive_ab_lock", None)
        if lock is None:
            return {"enabled": False, "state": "uninitialized", "ready": False, "pending": False}
        with lock:
            rec = dict(getattr(self, "_adaptive_ab_recommendation", {}) or {})
            return {
                "enabled": bool(getattr(self, "adaptive_alpha_beta_enable", False)),
                "state": str(getattr(self, "_adaptive_ab_state", "disabled")),
                "last_error": str(getattr(self, "_adaptive_ab_last_error", "")),
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
                "active_step": float(rec.get("active_step", getattr(self, "_adaptive_active_step", getattr(self, "adaptive_alpha_beta_coarse_step", 0.25)))),
                "next_active_step": float(rec.get("next_active_step", getattr(self, "_adaptive_active_step", getattr(self, "adaptive_alpha_beta_coarse_step", 0.25)))),
                "selected_direction": str(rec.get("selected_direction", "none")),
                "alpha_span": float(rec.get("alpha_span", float("nan"))),
                "beta_span": float(rec.get("beta_span", float("nan"))),
                "alpha_observable": bool(rec.get("alpha_observable", False)),
                "beta_observable": bool(rec.get("beta_observable", False)),
                "stable_count": int(rec.get("stable_count", 0)),
                "stable_required": int(rec.get("stable_required", self.adaptive_alpha_beta_stability_evals)),
                "candidate_count": int(rec.get("candidate_count", 0)),
                "search_seconds": float(rec.get("search_seconds", float("nan"))),
                "htf_source": str(rec.get("htf_source", "")),
                "htf_kind": str(rec.get("htf_kind", "")),
                "coarse_step": float(self.adaptive_alpha_beta_coarse_step),
                "fine_step": float(self.adaptive_alpha_beta_fine_step),
                "interval_frames": int(self.adaptive_alpha_beta_interval_frames),
                "minimum_improvement_db": float(self.adaptive_alpha_beta_min_improvement_db),
                "cooldown_frames": int(self.adaptive_alpha_beta_cooldown_frames),
                "integer_margin_db": float(self.adaptive_alpha_beta_integer_margin_db),
                "max_order": int(self.adaptive_alpha_beta_max_order),
                "signaling_mode": str(getattr(self, "ALPHA_BETA_SIGNALING_MODE", "shared_memory")),
            }

    # =========================================================
    # Display helpers
    # =========================================================

    # =========================================================
    # Alpha/Beta performance surface for UI demonstration
    # =========================================================
    def _alpha_beta_surface_context_key(self) -> Tuple[Any, ...]:
        """Comparable-context key for the alpha/beta performance surface.

        The surface is meant to answer one question: under the same link,
        modulation, estimator, gain, coding and channel settings, which
        alpha/beta pair actually produced a better measured metric?  Alpha and
        beta themselves are intentionally excluded so changing them adds a new
        point; all other settings that would make points non-comparable are
        included and trigger an automatic clear.
        """
        def _r(value: Any, digits: int = 6) -> float:
            try:
                v = float(value)
                if not np.isfinite(v):
                    return float("nan")
                return round(v, int(digits))
            except Exception:
                return float("nan")

        return (
            str(getattr(self, "strict_chain_name", "")),
            str(getattr(self, "device_type", "")),
            str(getattr(self, "channel_mode", "")),
            str(getattr(self, "channel_estimator", "")),
            str(getattr(self, "requested_channel_estimator", "")),
            str(getattr(self, "mod_order", "")),
            str(getattr(self, "equalizer", "")),
            str(getattr(self, "coding_scheme", "")),
            bool(getattr(self, "coding_interleaver", False)),
            int(getattr(self, "M", 0)),
            int(getattr(self, "N", 0)),
            int(getattr(self, "cp_len", 0)),
            int(getattr(self, "max_full_htf_order", 0)),
            int(getattr(self, "tx_frame_count", 0)),
            int(getattr(self, "inter_frame_guard_len", 0)),
            int(getattr(self, "evm_average_frames", 0)),
            _r(getattr(self, "sample_rate", 0.0), 3),
            _r(getattr(self, "carrier_freq", 0.0), 3),
            _r(getattr(self, "tx_gain", 0.0), 3),
            _r(getattr(self, "rx_gain", 0.0), 3),
            _r(getattr(self, "training_amplitude", 0.0), 6),
            _r(getattr(self, "tdl_rms_delay_spread_ns", 0.0), 6),
            _r(getattr(self, "tdl_doppler_hz", 0.0), 6),
            _r(getattr(self, "tdl_doppler_spread_hz", 0.0), 6),
            _r(getattr(self, "tdl_snr_db", 0.0), 6),
            int(getattr(self, "tdl_seed", 0)),
            bool(getattr(self, "tdl_normalize_power", False)),
            int(getattr(self, "tdl_param_num_sinusoids", 0)),
            int(getattr(self, "tdl_param_max_paths", 0)),
            _r(getattr(self, "tdl_param_ridge", 0.0), 12),
            _r(getattr(self, "tdl_param_prune_db", 0.0), 6),
            int(getattr(self, "_tx_coded_bits_len", 0)),
            int(getattr(self, "_tx_uncoded_bits_len", 0)),
        )

    def _clear_alpha_beta_performance_surface_locked(self, reason: str = "context_change"):
        self._ab_metric_history = {}
        self._ab_surface_context_key = self._alpha_beta_surface_context_key()
        try:
            self._debug("INFO", f"alpha/beta performance surface cleared: {reason}")
        except Exception:
            pass

    def clear_alpha_beta_performance_surface(self, reason: str = "manual"):
        """Public UI hook: clear measured alpha/beta performance cells."""
        with self._lock:
            self._clear_alpha_beta_performance_surface_locked(reason=reason)

    def _ensure_alpha_beta_surface_context_locked(self):
        current_key = self._alpha_beta_surface_context_key()
        if current_key != getattr(self, "_ab_surface_context_key", None):
            self._clear_alpha_beta_performance_surface_locked(reason="non_alpha_beta_parameter_changed")

    @staticmethod
    def _finite_float_or_nan(value: Any) -> float:
        try:
            v = float(value)
            return v if np.isfinite(v) else float("nan")
        except Exception:
            return float("nan")

    def _alpha_beta_surface_target_samples_locked(self) -> int:
        """Number of real RX frames used before one α/β point is frozen.

        The UI asks for average metrics on the z-axis.  Therefore a point is
        not published while the current α/β value is still accumulating its
        averaging window.  Once this many processed frames have been collected,
        the cell is finalized and will never be changed again unless the user
        explicitly clears the surface or changes a non-α/β context parameter.
        """
        try:
            n = int(getattr(self, "_ab_surface_samples_per_cell", 0))
        except Exception:
            n = 0
        if n <= 0:
            try:
                n = int(getattr(self, "evm_average_frames", 1))
            except Exception:
                n = 1
        return int(max(1, min(n, 128)))

    def _finalize_alpha_beta_cell_locked(self, cell: Dict[str, Any]):
        """Freeze all z-axis metrics for one measured α/β cell.

        This is deliberately one-way: finalized cells are read-only.  It prevents
        a previously measured column from moving while the live link keeps
        processing repeated frames at the same α/β setting.
        """
        metric_store = cell.get("metrics", {}) if isinstance(cell, dict) else {}
        final_metrics: Dict[str, float] = {}
        final_counts: Dict[str, int] = {}

        for name, samples in dict(metric_store).items():
            try:
                arr = np.asarray(list(samples), dtype=np.float64).reshape(-1)
            except Exception:
                continue
            arr = arr[np.isfinite(arr)]
            if arr.size <= 0:
                continue
            if str(name) == "evm_instant_percent":
                # Keep the instantaneous diagnostic internally but do not expose it
                # in the UI selector.  It is also the mathematically correct source
                # for RMS average EVM over the finalized α/β window.
                final_metrics[str(name)] = float(arr[-1])
            elif str(name) == "evm_average_percent":
                # This value is overwritten below from evm_instant_percent when
                # available.  Keeping a fallback protects older call paths.
                final_metrics[str(name)] = float(arr[-1])
            else:
                final_metrics[str(name)] = float(np.mean(arr))
            final_counts[str(name)] = int(arr.size)

        try:
            inst = np.asarray(list(metric_store.get("evm_instant_percent", [])), dtype=np.float64).reshape(-1)
            inst = inst[np.isfinite(inst)]
        except Exception:
            inst = np.zeros(0, dtype=np.float64)
        if inst.size > 0:
            final_metrics["evm_average_percent"] = float(np.sqrt(np.mean(inst ** 2)))
            final_counts["evm_average_percent"] = int(inst.size)

        sample_count = int(max(0, cell.get("sample_count", 0)))
        decode_ok_count = int(max(0, cell.get("decode_ok_count", 0)))
        final_metrics["decode_success_ratio"] = float(decode_ok_count) / max(float(sample_count), 1.0)
        final_counts["decode_success_ratio"] = int(sample_count)

        # A frozen cell with no finite metrics is not useful; leave it partial so
        # a later valid frame under the same α/β can still complete it.
        if not final_metrics:
            return
        cell["final_metrics"] = final_metrics
        cell["final_metric_counts"] = final_counts
        cell["finalized"] = True
        cell["final_frame_counter"] = int(getattr(self, "_frames_processed", 0))
        cell["final_wall"] = float(time.time())

    def _record_alpha_beta_performance_sample_locked(self, metrics: Dict[str, Any]):
        """Record one measured frame into the alpha/beta surface.

        Caller must hold self._lock.  Values come from the same RX
        sync/equalizer/decoder path that updates BER/EVM status.  A cell is
        accumulated for a fixed averaging window and then frozen; finalized
        points are never updated by later frames, so already measured columns
        remain visually fixed while the operator moves to the next α/β setting.
        """
        self._ensure_alpha_beta_surface_context_locked()
        try:
            q_digits = int(getattr(self, "_ab_surface_quant_digits", 3))
            a = round(float(getattr(self, "alpha", 0.0)), q_digits)
            b = round(float(getattr(self, "beta", 0.0)), q_digits)
        except Exception:
            return
        key = (a, b)
        if key not in self._ab_metric_history and len(self._ab_metric_history) >= int(getattr(self, "_ab_surface_max_cells", 2048)):
            oldest = min(self._ab_metric_history.items(), key=lambda kv: float(kv[1].get("last_wall", 0.0)))[0]
            self._ab_metric_history.pop(oldest, None)

        target = self._alpha_beta_surface_target_samples_locked()
        cell = self._ab_metric_history.setdefault(key, {
            "alpha": float(a),
            "beta": float(b),
            "metrics": {},
            "sample_count": 0,
            "decode_ok_count": 0,
            "last_frame_counter": 0,
            "last_wall": 0.0,
            "target_sample_count": int(target),
            "finalized": False,
        })

        # Finalized means finalized.  Do not refresh sample_count, last_frame,
        # last_wall, or metrics; the visible bar and its metadata must remain
        # fixed for repeatability.
        if bool(cell.get("finalized", False)):
            return

        cell["target_sample_count"] = int(target)
        cell["sample_count"] = int(cell.get("sample_count", 0)) + 1
        if bool(metrics.get("decode_ok", False)):
            cell["decode_ok_count"] = int(cell.get("decode_ok_count", 0)) + 1
        cell["last_frame_counter"] = int(getattr(self, "_frames_processed", 0))
        cell["last_wall"] = float(time.time())

        metric_store = cell.setdefault("metrics", {})
        # Keep the full target window.  No rolling maxlen is used because a point
        # is frozen exactly once after target samples; losing early samples before
        # finalization would make the average ambiguous.
        for name, value in dict(metrics).items():
            if name == "decode_ok":
                value = 1.0 if bool(value) else 0.0
            if name == "evm_average_count":
                continue
            v = self._finite_float_or_nan(value)
            if not np.isfinite(v):
                continue
            if name not in metric_store:
                metric_store[name] = []
            metric_store[name].append(float(v))

        # Decode-rate is recomputed from the whole cell window.
        metric_store.setdefault("decode_success_ratio", [])
        metric_store["decode_success_ratio"].append(
            float(cell["decode_ok_count"]) / max(float(cell["sample_count"]), 1.0)
        )

        if int(cell.get("sample_count", 0)) >= int(target):
            self._finalize_alpha_beta_cell_locked(cell)
            if bool(cell.get("finalized", False)):
                try:
                    fm = cell.get("final_metrics", {})
                    evm = fm.get("evm_average_percent", float("nan"))
                    msg = (
                        f"alpha/beta point frozen: alpha={float(a):.3f}, beta={float(b):.3f}, "
                        f"samples={int(cell.get('sample_count', 0))}/{int(target)}"
                    )
                    if np.isfinite(float(evm)):
                        msg += f", EVMavg={float(evm):.3f}%"
                    self._debug("INFO", msg)
                except Exception:
                    pass

    def get_alpha_beta_performance_surface(self, metric: str = "evm_average_percent") -> Dict[str, Any]:
        """Return frozen measured alpha/beta surface points for one metric.

        Only finalized cells are returned as visible bars.  Partially measured
        cells are reported separately as progress metadata but are not drawn as
        columns, which eliminates fast-changing bars during the averaging window.
        """
        metric = str(metric or "evm_average_percent")
        with self._lock:
            self._ensure_alpha_beta_surface_context_locked()
            points: List[Dict[str, Any]] = []
            partial_count = 0
            active_key = None
            try:
                q_digits = int(getattr(self, "_ab_surface_quant_digits", 3))
                active_key = (round(float(getattr(self, "alpha", 0.0)), q_digits),
                              round(float(getattr(self, "beta", 0.0)), q_digits))
            except Exception:
                active_key = None
            active_progress = {"sample_count": 0, "target_sample_count": self._alpha_beta_surface_target_samples_locked(), "finalized": False}

            for key, cell in self._ab_metric_history.items():
                finalized = bool(cell.get("finalized", False))
                if key == active_key:
                    active_progress = {
                        "sample_count": int(cell.get("sample_count", 0)),
                        "target_sample_count": int(cell.get("target_sample_count", self._alpha_beta_surface_target_samples_locked())),
                        "finalized": finalized,
                    }
                if not finalized:
                    partial_count += 1
                    continue
                metrics_out = dict(cell.get("final_metrics", {}))
                counts_out = dict(cell.get("final_metric_counts", {}))
                if not metrics_out:
                    continue
                z = self._finite_float_or_nan(metrics_out.get(metric, float("nan")))
                points.append({
                    "alpha": float(cell.get("alpha", 0.0)),
                    "beta": float(cell.get("beta", 0.0)),
                    "z": z,
                    "metric": metric,
                    "metrics": metrics_out,
                    "metric_counts": counts_out,
                    "sample_count": int(cell.get("sample_count", 0)),
                    "decode_ok_count": int(cell.get("decode_ok_count", 0)),
                    "target_sample_count": int(cell.get("target_sample_count", self._alpha_beta_surface_target_samples_locked())),
                    "last_frame_counter": int(cell.get("final_frame_counter", cell.get("last_frame_counter", 0))),
                    "last_wall": float(cell.get("final_wall", cell.get("last_wall", 0.0))),
                    "finalized": True,
                })
            return {
                "metric": metric,
                "points": points,
                "point_count": int(len(points)),
                "partial_count": int(partial_count),
                "active_progress": active_progress,
                "current_alpha": float(getattr(self, "alpha", 0.0)),
                "current_beta": float(getattr(self, "beta", 0.0)),
                "context_key": repr(getattr(self, "_ab_surface_context_key", ())),
                "quant_digits": int(getattr(self, "_ab_surface_quant_digits", 3)),
                "samples_per_cell": int(self._alpha_beta_surface_target_samples_locked()),
            }
