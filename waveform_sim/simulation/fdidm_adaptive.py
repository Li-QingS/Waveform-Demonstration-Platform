# -*- coding: utf-8 -*-
"""Dual-timescale FDIDM alpha/beta adaptation.

Fast time scale
~~~~~~~~~~~~~~~
Every simulated frame updates the channel/receiver state and contributes one CSI
snapshot to a bounded slow-state window.  No alpha/beta grid search is executed
in the frame path.

Slow time scale
~~~~~~~~~~~~~~~
A background worker periodically evaluates a small, representative ensemble of
snapshots from the CSI window.  It performs a coarse/fine alpha-beta search,
then applies gain hysteresis, repeated-recommendation stability and a cooldown
before changing the waveform parameters.

The OFDM/OTFS/AFDM/FDIDM metrics for the live time plot are computed by a
separate display-metric thread, decimated and smoothed for display.  A long
slow-time grid search therefore never stalls the point pipeline that feeds the
real-time SER curve; the simulation itself is not slowed down to make the plot
look stable.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from waveform_sim.simulation.display_metrics import LogMetricResampler
from waveform_sim.simulation.shared_waveform_benchmark import (
    BenchmarkContext,
    SharedWaveformBenchmark,
)


DEFAULT_ADAPTIVE_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "auto_apply": True,
    # Slow optimizer cadence and state window, in simulated frames.
    # Agile defaults: react to channel changes within a few seconds while
    # keeping a small hysteresis guard against single-eval noise.
    "interval_frames": 16,
    "window_frames": 32,
    "window_stride_frames": 1,
    "ensemble_snapshots": 3,
    # Display pipeline is independent of the optimizer cadence.
    "benchmark_interval_frames": 4,
    "display_interval_s": 0.5,
    "display_ema_alpha": 0.28,
    # Search / decision controls.
    "coarse_step": 0.5,
    "fine_step": 0.1,
    "stability_evals": 2,
    "cooldown_frames": 32,
    "min_improvement_db": 0.2,
    # Below this SER the controller treats candidates as practically tied; this
    # prevents high-SNR numerical underflow from causing fake multi-dB switches.
    "decision_ser_floor": 1e-8,
    "max_order": 512,
    "history_limit": 1000,
}


class FDIDMSimAdaptiveMixin:
    """Mixin used by `_LegacyFDIDMTransceiver`."""

    # ------------------------------------------------------------------ legacy kernel helpers
    @staticmethod
    def _adaptive_qfunc(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        flat = arr.reshape(-1)
        out = np.fromiter(
            (0.5 * math.erfc(float(v) / math.sqrt(2.0)) for v in flat),
            dtype=np.float64,
            count=flat.size,
        )
        return out.reshape(arr.shape)

    @staticmethod
    def _adaptive_qam_order(mod_order: str) -> int:
        text = str(mod_order or "16QAM").upper()
        if text in ("QPSK", "4QAM", "4-QAM"):
            return 4
        digits = "".join(ch for ch in text if ch.isdigit())
        return int(digits or 16)

    @classmethod
    def _adaptive_ser_from_symbol_nsr(cls, symbol_nsr: np.ndarray, mod_order: str) -> float:
        q = cls._adaptive_qam_order(mod_order)
        xi = np.maximum(np.asarray(symbol_nsr, dtype=np.float64), 1e-15)
        kappa = 1.0 - 1.0 / math.sqrt(float(q))
        arg = np.sqrt(3.0 / (float(q) - 1.0)) / xi
        qv = cls._adaptive_qfunc(arg)
        ser = 4.0 * kappa * qv - 4.0 * (kappa ** 2) * (qv ** 2)
        return float(np.mean(np.clip(ser, 0.0, 1.0)))

    @staticmethod
    def _adaptive_grid_values(step: float) -> np.ndarray:
        s = float(np.clip(float(step), 0.05, 1.0))
        return np.unique(np.round(np.arange(0.0, 2.0 + 0.5 * s, s), 9))

    @staticmethod
    def _adaptive_canonical_index(value: float) -> float:
        # The paper/GUI uses the closed interval [0,2].  Do not modulo-wrap 2 to 0
        # because the endpoint is intentionally visible in the search result.
        return float(np.clip(float(value), 0.0, 2.0))

    # ------------------------------------------------------------------ state/config
    def _init_adaptive_state_locked(self) -> None:
        if getattr(self, "_adaptive_initialized", False):
            return
        for name, default in DEFAULT_ADAPTIVE_CONFIG.items():
            setattr(self, "adaptive_" + name, getattr(self, "adaptive_" + name, default))
        self._adaptive_lock = threading.RLock()
        self._adaptive_metric_thread: Optional[threading.Thread] = None
        self._adaptive_search_thread: Optional[threading.Thread] = None
        self._adaptive_stop_event = threading.Event()
        # Separate wake-up events so a long slow-time search can never stall
        # the display-metric pipeline that feeds the real-time SER plot.
        self._adaptive_metric_event = threading.Event()
        self._adaptive_search_event = threading.Event()
        self._adaptive_pending_metric: Optional[Dict[str, Any]] = None
        self._adaptive_pending_ensemble: Optional[List[Dict[str, Any]]] = None
        self._adaptive_force_next = False
        self._adaptive_window: deque = deque(
            maxlen=max(2, int(self._adaptive_cfg("window_frames", 32)))
        )
        self._adaptive_history: deque = deque(
            maxlen=max(1, int(self._adaptive_cfg("history_limit", 1000)))
        )
        self._adaptive_resampler = LogMetricResampler(
            interval_s=float(self._adaptive_cfg("display_interval_s", 0.5)),
            ema_alpha=float(self._adaptive_cfg("display_ema_alpha", 0.28)),
        )
        self._adaptive_recommendation: Dict[str, Any] = {}
        self._adaptive_callback = None
        self._adaptive_state = "disabled"
        self._adaptive_last_error = ""
        self._adaptive_context_key: Optional[Tuple[Any, ...]] = None
        self._adaptive_last_queued_frame = -10**9
        self._adaptive_last_metric_frame = -10**9
        self._adaptive_last_applied_frame = -10**9
        self._adaptive_stable_key: Optional[Tuple[int, int]] = None
        self._adaptive_stable_count = 0
        self._adaptive_eval_seq = 0
        # Incremented whenever the *applied* alpha/beta pair changes.  Queued
        # optimizer jobs carry this epoch so a result produced for an old pair
        # can never be applied twice after another job has already switched it.
        self._adaptive_index_epoch = int(getattr(self, "_adaptive_index_epoch", 0))
        self._adaptive_initialized = True

    def _adaptive_cfg(self, name: str, default: Any = None) -> Any:
        return getattr(self, "adaptive_" + str(name), default)

    def update_adaptive_config(self, **cfg) -> None:
        with self._adaptive_lock:
            for key, value in dict(cfg).items():
                name = str(key)
                if name.startswith("adaptive_"):
                    name = name[len("adaptive_") :]
                attr = "adaptive_" + name
                if name in ("interval_frames", "window_frames", "window_stride_frames",
                            "ensemble_snapshots", "benchmark_interval_frames", "stability_evals",
                            "cooldown_frames", "max_order", "history_limit"):
                    value = max(1 if name != "cooldown_frames" else 0, int(value))
                elif name in ("coarse_step", "fine_step", "display_interval_s",
                              "display_ema_alpha", "min_improvement_db", "decision_ser_floor"):
                    value = float(value)
                elif name in ("enabled", "auto_apply"):
                    value = bool(value)
                setattr(self, attr, value)

            new_window = max(2, int(self._adaptive_cfg("window_frames", 32)))
            if self._adaptive_window.maxlen != new_window:
                self._adaptive_window = deque(list(self._adaptive_window)[-new_window:], maxlen=new_window)
            new_history = max(1, int(self._adaptive_cfg("history_limit", 1000)))
            if self._adaptive_history.maxlen != new_history:
                self._adaptive_history = deque(list(self._adaptive_history)[-new_history:], maxlen=new_history)
            self._adaptive_resampler.configure(
                interval_s=float(self._adaptive_cfg("display_interval_s", 0.5)),
                ema_alpha=float(self._adaptive_cfg("display_ema_alpha", 0.28)),
            )

    def _adaptive_context_key_locked(self) -> Tuple[Any, ...]:
        cfg = self.config
        return (
            int(getattr(cfg, "m_subcarriers", 0)),
            int(getattr(cfg, "n_symbols", 0)),
            str(getattr(cfg, "mod_order", "")),
            str(getattr(cfg, "decoder", "")),
            str(getattr(cfg, "channel_model", "")),
            str(getattr(cfg, "ntn_profile", "")),
            float(getattr(cfg, "residual_doppler_spread_hz", 0.0)),
            float(getattr(cfg, "doppler_compensation_ratio", 1.0)),
            float(getattr(cfg, "velocity_kmh", 0.0)),
            float(getattr(cfg, "doppler_radial_factor", 0.0)),
            float(getattr(cfg, "fc_hz", 0.0)),
            float(getattr(cfg, "subcarrier_spacing_hz", 0.0)),
            float(getattr(cfg, "ebn0_db", 0.0)),
            str(getattr(cfg, "snr_definition", "Eb/N0")),
            str(getattr(cfg, "channel_dynamics", "fixed")),
            int(getattr(cfg, "channel_coherence_frames", 1)),
        )

    def _adaptive_invalidate_locked(self, context: Tuple[Any, ...]) -> None:
        self._adaptive_context_key = context
        self._adaptive_recommendation = {}
        self._adaptive_stable_key = None
        self._adaptive_stable_count = 0
        self._adaptive_window.clear()
        self._adaptive_pending_ensemble = None
        self._adaptive_state = "collecting" if self._adaptive_cfg("enabled", False) else "disabled"

    def _adaptive_note_index_change_locked(self) -> None:
        """Invalidate stale decisions after a real/user alpha-beta change.

        The CSI window remains valid because it contains physical H_TF snapshots,
        which are independent of the selected FDIDM analysis domain.  Only queued
        recommendations and their stability votes are discarded.
        """
        self._adaptive_index_epoch = int(getattr(self, "_adaptive_index_epoch", 0)) + 1
        self._adaptive_pending_ensemble = None
        self._adaptive_recommendation = {}
        self._adaptive_stable_key = None
        self._adaptive_stable_count = 0
        if bool(self._adaptive_cfg("enabled", False)):
            self._adaptive_state = "collecting"

    # ------------------------------------------------------------------ snapshot/benchmark
    def _adaptive_snapshot_locked(self) -> Dict[str, Any]:
        """Copy all data required by the worker while holding the backend lock."""
        self._prepare_matrices_locked()
        cfg = self.config
        H = np.asarray(self._H_tf, dtype=np.complex128).copy()
        table = list(self._debug.get("path_table_ns_samples_fd_abs_gain", []) or [])
        max_delay_ns = 0.0
        for row in table:
            try:
                # Supported layouts: (path, delay_ns, delay_samples, fd, fd_idx, gain)
                # and (delay_ns, delay_samples, fd, gain).
                if len(row) >= 6:
                    max_delay_ns = max(max_delay_ns, float(row[1]))
                elif len(row) >= 4:
                    max_delay_ns = max(max_delay_ns, float(row[0]))
            except Exception:
                continue
        frame = int(getattr(self, "_sim_frame", 0))
        frame_duration_s = float(cfg.n_symbols / max(cfg.subcarrier_spacing_hz, 1e-15))
        return {
            "frame": frame,
            "sim_time_s": float(frame * frame_duration_s),
            "ts": float(time.time()),
            "M": int(cfg.m_subcarriers),
            "N": int(cfg.n_symbols),
            "htf": H,
            "htf_kind": "full",
            "mod_order": str(cfg.mod_order),
            "qam_order": int(self.qam_order),
            "bits_per_symbol": int(self.bits_per_symbol),
            "snr_definition": str(getattr(cfg, "snr_definition", "Eb/N0")),
            "snr_db": float(cfg.ebn0_db),
            "equalizer": str(cfg.decoder),
            "alpha": float(cfg.alpha),
            "beta": float(cfg.beta),
            "index_epoch": int(getattr(self, "_adaptive_index_epoch", 0)),
            "subcarrier_spacing_hz": float(cfg.subcarrier_spacing_hz),
            "path_count": max(1, len(table)),
            "maximum_delay_ns": float(max_delay_ns),
            "residual_doppler_spread_hz": float(
                getattr(cfg, "residual_doppler_spread_hz", 200.0)
            ),
            "context_key": self._adaptive_context_key_locked(),
        }

    def _adaptive_benchmark_from_snapshot(self, snapshot: Dict[str, Any]) -> SharedWaveformBenchmark:
        M = int(snapshot["M"])
        N = int(snapshot["N"])
        raw = np.asarray(snapshot["htf"], dtype=np.complex128)
        if str(snapshot.get("htf_kind", "full")) == "diag" or raw.shape == (M, N) or raw.ndim == 1:
            diag = raw.reshape(-1, order="F")[: M * N]
            raw = np.diag(diag)
        ctx = BenchmarkContext(
            m_subcarriers=M,
            n_symbols=N,
            subcarrier_spacing_hz=float(snapshot.get("subcarrier_spacing_hz", getattr(self.config, "subcarrier_spacing_hz", 300e3))),
            qam_order=int(snapshot.get("qam_order", self._adaptive_qam_order(snapshot.get("mod_order", "16QAM")))),
            bits_per_symbol=int(snapshot.get("bits_per_symbol", round(math.log2(max(4, self._adaptive_qam_order(snapshot.get("mod_order", "16QAM"))))))),
            snr_definition=str(snapshot.get("snr_definition", getattr(self.config, "snr_definition", "Eb/N0"))),
            detector=str(snapshot.get("equalizer", getattr(self.config, "decoder", "MMSE"))),
            path_count=int(snapshot.get("path_count", 3)),
            maximum_delay_ns=float(snapshot.get("maximum_delay_ns", 0.0)),
            residual_doppler_spread_hz=float(snapshot.get("residual_doppler_spread_hz", 200.0)),
        )
        return SharedWaveformBenchmark(ctx, raw.reshape(M * N, M * N), self._fdit_matrices)

    @staticmethod
    def _select_ensemble(window: Sequence[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        items = list(window)
        if len(items) <= max(1, int(count)):
            return [dict(x) for x in items]
        idx = np.linspace(0, len(items) - 1, max(1, int(count)), dtype=np.int64)
        return [dict(items[int(i)]) for i in idx]

    # ------------------------------------------------------------------ compatibility optimizer
    def _optimize_alpha_beta_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        M = int(snapshot.get("M", 0)); N = int(snapshot.get("N", 0)); K = M * N
        max_order = int(snapshot.get("max_order", self._adaptive_cfg("max_order", 512)))
        if K <= 0 or K > max_order:
            raise ValueError(f"adaptive matrix order {K} exceeds max_order={max_order}")
        bench = self._adaptive_benchmark_from_snapshot(snapshot)
        result = SharedWaveformBenchmark.optimize_fdidm_over_ensemble(
            [bench],
            snr_db=float(snapshot.get("snr_db", getattr(self.config, "ebn0_db", 10.0))),
            current_alpha=float(snapshot.get("alpha", 0.0)),
            current_beta=float(snapshot.get("beta", 0.0)),
            coarse_step=float(snapshot.get("coarse_step", self._adaptive_cfg("coarse_step", 0.5))),
            fine_step=float(snapshot.get("fine_step", self._adaptive_cfg("fine_step", 0.1))),
            decision_ser_floor=float(snapshot.get(
                "decision_ser_floor", self._adaptive_cfg("decision_ser_floor", 1e-8)
            )),
        )
        refs = bench.evaluate_all(
            float(snapshot.get("snr_db", getattr(self.config, "ebn0_db", 10.0))),
            float(snapshot.get("alpha", 0.0)),
            float(snapshot.get("beta", 0.0)),
        )
        result.update({
            "predicted_ser_ofdm": float(refs["OFDM"]["ser"]),
            "predicted_ser_otfs": float(refs["OTFS"]["ser"]),
            "predicted_ser_afdm": float(refs["AFDM"]["ser"]),
            "predicted_snr_db": float(snapshot.get("snr_db", getattr(self.config, "ebn0_db", 10.0))),
            "frame_counter": int(snapshot.get("frame_counter", snapshot.get("frame", 0))),
            "snapshot_seq": int(snapshot.get("snapshot_seq", 0)),
            "search_mode": "diag_2d" if str(snapshot.get("htf_kind", "full")) == "diag" else "full_coordinate",
        })
        return result

    # ------------------------------------------------------------------ scheduling
    def _ensure_adaptive_worker(self) -> None:
        with self._adaptive_lock:
            metric_thread = self._adaptive_metric_thread
            search_thread = self._adaptive_search_thread
            need_metric = metric_thread is None or not metric_thread.is_alive()
            need_search = search_thread is None or not search_thread.is_alive()
            if not need_metric and not need_search:
                return
            self._adaptive_stop_event.clear()
            if need_metric:
                self._adaptive_metric_thread = threading.Thread(
                    target=self._adaptive_metric_loop,
                    daemon=True,
                    name="fdidm-display-metric",
                )
                self._adaptive_metric_thread.start()
            if need_search:
                self._adaptive_search_thread = threading.Thread(
                    target=self._adaptive_search_loop,
                    daemon=True,
                    name="fdidm-slow-search",
                )
                self._adaptive_search_thread.start()

    def _adaptive_note_frame_processed(self) -> None:
        if not bool(self._adaptive_cfg("enabled", False)):
            return
        self._ensure_adaptive_worker()
        with self._lock:
            snapshot = self._adaptive_snapshot_locked()
        frame = int(snapshot["frame"])
        context = tuple(snapshot["context_key"])

        with self._adaptive_lock:
            if not bool(self._adaptive_cfg("enabled", False)):
                return
            if context != self._adaptive_context_key:
                self._adaptive_invalidate_locked(context)

            stride = max(1, int(self._adaptive_cfg("window_stride_frames", 1)))
            if frame % stride == 0:
                self._adaptive_window.append(snapshot)

            metric_interval = max(1, int(self._adaptive_cfg("benchmark_interval_frames", 4)))
            if frame - self._adaptive_last_metric_frame >= metric_interval:
                self._adaptive_pending_metric = snapshot
                self._adaptive_last_metric_frame = frame
                self._adaptive_metric_event.set()

            interval = max(1, int(self._adaptive_cfg("interval_frames", 16)))
            required_window = max(2, int(self._adaptive_cfg("window_frames", 32)))
            required_manual = min(
                required_window,
                max(2, int(self._adaptive_cfg("ensemble_snapshots", 3))),
            )
            automatic_ready = (
                len(self._adaptive_window) >= required_window
                and frame - self._adaptive_last_queued_frame >= interval
            )
            manual_ready = self._adaptive_force_next and len(self._adaptive_window) >= required_manual
            should_optimize = bool(automatic_ready or manual_ready)
            if should_optimize:
                source_window_frames = int(len(self._adaptive_window))
                self._adaptive_pending_ensemble = self._select_ensemble(
                    self._adaptive_window,
                    int(self._adaptive_cfg("ensemble_snapshots", 3)),
                )
                for item in self._adaptive_pending_ensemble:
                    item["source_window_frames"] = source_window_frames
                if not self._adaptive_pending_ensemble:
                    self._adaptive_pending_ensemble = [snapshot]
                self._adaptive_last_queued_frame = frame
                self._adaptive_force_next = False
                self._adaptive_state = "queued"
            if self._adaptive_pending_ensemble is not None:
                self._adaptive_search_event.set()

    def _adaptive_metric_loop(self) -> None:
        """Dedicated thread for the decimated real-time SER display points."""
        while not self._adaptive_stop_event.is_set():
            self._adaptive_metric_event.wait(timeout=0.25)
            self._adaptive_metric_event.clear()
            if self._adaptive_stop_event.is_set():
                break
            with self._adaptive_lock:
                metric_snapshot = self._adaptive_pending_metric
                self._adaptive_pending_metric = None
                enabled = bool(self._adaptive_cfg("enabled", False))
            if not enabled or metric_snapshot is None:
                continue
            try:
                self._adaptive_process_metric(metric_snapshot)
            except Exception as exc:
                with self._adaptive_lock:
                    self._adaptive_last_error = f"metric: {type(exc).__name__}: {exc}"

    def _adaptive_search_loop(self) -> None:
        """Dedicated thread for the slow alpha/beta search and switch decision."""
        while not self._adaptive_stop_event.is_set():
            self._adaptive_search_event.wait(timeout=0.25)
            self._adaptive_search_event.clear()
            if self._adaptive_stop_event.is_set():
                break
            with self._adaptive_lock:
                ensemble = self._adaptive_pending_ensemble
                self._adaptive_pending_ensemble = None
                enabled = bool(self._adaptive_cfg("enabled", False))
            if not enabled or not ensemble:
                continue
            try:
                self._adaptive_process_ensemble(ensemble)
            except Exception as exc:
                with self._adaptive_lock:
                    self._adaptive_last_error = f"optimizer: {type(exc).__name__}: {exc}"
                    self._adaptive_state = "error"

    def _adaptive_process_metric(self, snapshot: Dict[str, Any]) -> None:
        # Skip the four-waveform evaluation when the display budget has not
        # elapsed yet; this keeps the metric thread cheap so the simulation
        # thread and the slow search keep most of the CPU.
        if not self._adaptive_resampler.ready():
            return
        bench = self._adaptive_benchmark_from_snapshot(snapshot)
        all_metrics = bench.evaluate_all(
            float(snapshot["snr_db"]),
            float(snapshot["alpha"]),
            float(snapshot["beta"]),
        )
        raw = {
            "ser_ofdm": float(all_metrics["OFDM"]["ser"]),
            "ser_otfs": float(all_metrics["OTFS"]["ser"]),
            "ser_afdm": float(all_metrics["AFDM"]["ser"]),
            "ser_fdidm": float(all_metrics["FDIDM"]["ser"]),
        }
        smoothed = self._adaptive_resampler.process(raw)
        if smoothed is None:
            return
        event = {
            "kind": "metric",
            "frame": int(snapshot["frame"]),
            "sim_time_s": float(snapshot["sim_time_s"]),
            "ts": float(snapshot["ts"]),
            "alpha": float(snapshot["alpha"]),
            "beta": float(snapshot["beta"]),
            **{k: float(v) for k, v in smoothed.items()},
            **{"raw_" + k: float(v) for k, v in raw.items()},
            "ofdm_diag_energy": float(all_metrics["OFDM"]["diagonal_energy_ratio"]),
            "otfs_retained_energy": float(all_metrics["OTFS"]["retained_energy_ratio"]),
            "afdm_retained_energy": float(all_metrics["AFDM"]["retained_energy_ratio"]),
            "fdidm_retained_energy": float(all_metrics["FDIDM"]["retained_energy_ratio"]),
        }
        with self._adaptive_lock:
            self._adaptive_history.append(event)

    @staticmethod
    def _mean_reference_metrics(
        benchmarks: Sequence[SharedWaveformBenchmark],
        snr_db: float,
        alpha: float,
        beta: float,
    ) -> Dict[str, float]:
        out = {"OFDM": [], "OTFS": [], "AFDM": [], "FDIDM": []}
        for bench in benchmarks:
            metrics = bench.evaluate_all(snr_db, alpha, beta)
            for name in out:
                out[name].append(float(metrics[name]["ser"]))
        return {name: float(np.mean(values)) for name, values in out.items()}

    def _adaptive_process_ensemble(self, snapshots: Sequence[Dict[str, Any]]) -> None:
        """Optimize one slow-time CSI ensemble without applying stale results.

        Channel snapshots do not depend on alpha/beta, so an already collected
        ensemble is still useful after a parameter change.  The *baseline* used
        for gain and the switch source, however, must be the pair that is actually
        applied when this job starts.  An index epoch is checked again before the
        decision and before the write, which prevents duplicate/stale switches.
        """
        if not snapshots:
            return
        latest = dict(snapshots[-1])
        benchmarks = [self._adaptive_benchmark_from_snapshot(s) for s in snapshots]
        with self._lock:
            current_a = float(self.config.alpha)
            current_b = float(self.config.beta)
            start_epoch = int(getattr(self, "_adaptive_index_epoch", 0))
        snr_db = float(latest["snr_db"])

        result = SharedWaveformBenchmark.optimize_fdidm_over_ensemble(
            benchmarks,
            snr_db=snr_db,
            current_alpha=current_a,
            current_beta=current_b,
            coarse_step=float(self._adaptive_cfg("coarse_step", 0.5)),
            fine_step=float(self._adaptive_cfg("fine_step", 0.1)),
            decision_ser_floor=float(self._adaptive_cfg("decision_ser_floor", 1e-8)),
            stop_event=self._adaptive_stop_event,
        )
        refs_current = self._mean_reference_metrics(
            benchmarks, snr_db, current_a, current_b
        )
        best_a = float(result["recommended_alpha"])
        best_b = float(result["recommended_beta"])
        best_values = [
            float(b.evaluate("FDIDM", snr_db, best_a, best_b)["ser"])
            for b in benchmarks
        ]
        result.update({
            "predicted_ser_ofdm": float(refs_current["OFDM"]),
            "predicted_ser_otfs": float(refs_current["OTFS"]),
            "predicted_ser_afdm": float(refs_current["AFDM"]),
            "predicted_ser_current": float(refs_current["FDIDM"]),
            "predicted_ser_best": float(np.mean(best_values)),
            "predicted_snr_db": snr_db,
            "baseline_alpha": current_a,
            "baseline_beta": current_b,
            "baseline_index_epoch": start_epoch,
        })

        # A user operation or a previous optimizer job may have changed the
        # applied pair while the matrix search was running.  Such a result is
        # diagnostic only and must not contribute a stability vote or a switch.
        with self._lock:
            actual_a = float(self.config.alpha)
            actual_b = float(self.config.beta)
            actual_epoch = int(getattr(self, "_adaptive_index_epoch", 0))
        if actual_epoch != start_epoch or (
            abs(actual_a - current_a) + abs(actual_b - current_b) > 1e-12
        ):
            with self._adaptive_lock:
                if bool(self._adaptive_cfg("enabled", False)):
                    self._adaptive_state = "collecting"
                    self._adaptive_last_error = ""
                    self._adaptive_history.append({
                        "kind": "stale_eval",
                        "frame": int(latest["frame"]),
                        "sim_time_s": float(latest["sim_time_s"]),
                        "ts": float(time.time()),
                        "baseline_alpha": current_a,
                        "baseline_beta": current_b,
                        "actual_alpha": actual_a,
                        "actual_beta": actual_b,
                        "recommended_alpha": best_a,
                        "recommended_beta": best_b,
                        "reason": "index_epoch_changed_during_search",
                    })
            return

        with self._lock:
            decision_frame = int(getattr(self, "_sim_frame", latest["frame"]))
            decision_frame_duration_s = float(
                self.config.n_symbols
                / max(self.config.subcarrier_spacing_hz, 1e-15)
            )
        decision_sim_time_s = float(decision_frame * decision_frame_duration_s)

        fine = max(float(self._adaptive_cfg("fine_step", 0.1)), 1e-6)
        key = (int(round(best_a / fine)), int(round(best_b / fine)))
        with self._adaptive_lock:
            if not bool(self._adaptive_cfg("enabled", False)):
                return
            if key == self._adaptive_stable_key:
                self._adaptive_stable_count += 1
            else:
                self._adaptive_stable_key = key
                self._adaptive_stable_count = 1
            self._adaptive_eval_seq += 1
            seq = int(self._adaptive_eval_seq)
            frame = int(decision_frame)
            gain = float(result["predicted_improvement_db"])
            stable_required = max(1, int(self._adaptive_cfg("stability_evals", 2)))
            min_gain = float(self._adaptive_cfg("min_improvement_db", 0.2))
            cooldown = max(0, int(self._adaptive_cfg("cooldown_frames", 32)))
            cooldown_ok = frame - int(self._adaptive_last_applied_frame) >= cooldown
            parameter_delta = abs(best_a - current_a) + abs(best_b - current_b)
            ready = (
                self._adaptive_stable_count >= stable_required
                and gain >= min_gain
                and parameter_delta >= 0.5 * fine
                and cooldown_ok
            )
            auto_apply = bool(self._adaptive_cfg("auto_apply", True))
            if ready and auto_apply:
                action = "apply"
            elif not cooldown_ok:
                action = "cooldown"
            elif self._adaptive_stable_count < stable_required:
                action = "stable_pending"
            elif gain < min_gain:
                action = "gain_below"
            elif parameter_delta < 0.5 * fine:
                action = "keep"
            else:
                action = "ready_manual"

            rec = dict(result)
            rec.update({
                "recommendation_seq": seq,
                "stable_count": int(self._adaptive_stable_count),
                "stable_required": stable_required,
                "minimum_improvement_db": min_gain,
                "ready": bool(ready),
                "pending": bool(ready),
                "frame_counter": frame,
                "source_window_frames": int(
                    latest.get("source_window_frames", len(snapshots))
                ),
                "ensemble_snapshots": int(len(snapshots)),
                # Backward-compatible field: number of representative snapshots.
                "window_frames": int(len(snapshots)),
            })
            self._adaptive_recommendation = rec
            self._adaptive_state = "ready" if ready else "tracking"
            self._adaptive_last_error = ""
            self._adaptive_history.append({
                "kind": "eval",
                "seq": seq,
                "frame": frame,
                "sim_time_s": decision_sim_time_s,
                "window_end_frame": int(latest["frame"]),
                "window_end_sim_time_s": float(latest["sim_time_s"]),
                "ts": float(time.time()),
                "alpha": current_a,
                "beta": current_b,
                "rec_alpha": best_a,
                "rec_beta": best_b,
                "ser_current": float(result["predicted_ser_current"]),
                "ser_best": float(result["predicted_ser_best"]),
                "ser_ofdm": float(result["predicted_ser_ofdm"]),
                "ser_otfs": float(result["predicted_ser_otfs"]),
                "ser_afdm": float(result["predicted_ser_afdm"]),
                "gain_db": gain,
                "decision_ser_floor": float(
                    result.get("decision_ser_floor", 1e-8)
                ),
                "action": action,
                "stable_count": int(self._adaptive_stable_count),
                "stable_required": stable_required,
                "candidates": int(result["candidate_count"]),
                "seconds": float(result["search_seconds"]),
                "source_window_frames": int(
                    latest.get("source_window_frames", len(snapshots))
                ),
                "ensemble_snapshots": int(len(snapshots)),
                "window_frames": int(len(snapshots)),
                "state": str(self._adaptive_state),
                "index_epoch": start_epoch,
            })
            callback = self._adaptive_callback
            callback_payload = dict(rec)

        if callback is not None:
            try:
                callback(callback_payload)
            except Exception:
                pass

        if ready and auto_apply:
            # Guard and write are one backend-lock transaction.  Without this, two
            # background jobs can both pass the epoch check before either writes,
            # then append duplicate switch events for the same recommendation.
            with self._lock:
                still_current = (
                    int(getattr(self, "_adaptive_index_epoch", 0)) == start_epoch
                    and abs(float(self.config.alpha) - current_a) <= 1e-12
                    and abs(float(self.config.beta) - current_b) <= 1e-12
                )
                if still_current:
                    # RLock makes this safe and preserves the single public path
                    # that advances the index epoch and invalidates stale votes.
                    self.set_indices(best_a, best_b)
                    applied_frame = int(getattr(self, "_sim_frame", decision_frame))
                    applied_frame_duration_s = float(
                        self.config.n_symbols
                        / max(self.config.subcarrier_spacing_hz, 1e-15)
                    )
            if not still_current:
                with self._adaptive_lock:
                    self._adaptive_state = "collecting"
                return
            applied_sim_time_s = float(applied_frame * applied_frame_duration_s)
            with self._adaptive_lock:
                if bool(self._adaptive_cfg("enabled", False)):
                    self._adaptive_last_applied_frame = int(applied_frame)
                    self._adaptive_state = "applied"
                    self._adaptive_stable_key = None
                    self._adaptive_stable_count = 0
                    self._adaptive_pending_ensemble = None
                    self._adaptive_history.append({
                        "kind": "switch",
                        "seq": seq,
                        "frame": int(applied_frame),
                        "sim_time_s": applied_sim_time_s,
                        "window_end_frame": int(latest["frame"]),
                        "window_end_sim_time_s": float(latest["sim_time_s"]),
                        "ts": float(time.time()),
                        "from_alpha": float(current_a),
                        "from_beta": float(current_b),
                        "to_alpha": float(best_a),
                        "to_beta": float(best_b),
                        "gain_db": float(result["predicted_improvement_db"]),
                        "reason": "slow_window_stable_hysteresis",
                        "source_window_frames": int(
                            latest.get("source_window_frames", len(snapshots))
                        ),
                        "ensemble_snapshots": int(len(snapshots)),
                        "window_frames": int(len(snapshots)),
                        "from_index_epoch": start_epoch,
                        "to_index_epoch": int(
                            getattr(self, "_adaptive_index_epoch", start_epoch + 1)
                        ),
                    })

    # ------------------------------------------------------------------ public API
    def start_adaptive_tuning(self, callback=None, **cfg) -> bool:
        if cfg:
            self.update_adaptive_config(**cfg)
        with self._adaptive_lock:
            self.adaptive_enabled = True
            if callback is not None:
                self._adaptive_callback = callback
            self._adaptive_state = "collecting"
            # Automatic optimization waits for a full CSI window.  Only the
            # explicit "evaluate now" action can request an early ensemble.
            self._adaptive_force_next = False
            self._adaptive_last_error = ""
        self._ensure_adaptive_worker()
        return True

    def stop_adaptive_tuning(self) -> None:
        with self._adaptive_lock:
            self.adaptive_enabled = False
            self._adaptive_state = "disabled"
            self._adaptive_force_next = False
        self._adaptive_stop_event.set()
        self._adaptive_metric_event.set()
        self._adaptive_search_event.set()
        for thread in (self._adaptive_metric_thread, self._adaptive_search_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2.0)
        with self._adaptive_lock:
            self._adaptive_metric_thread = None
            self._adaptive_search_thread = None
            self._adaptive_pending_metric = None
            self._adaptive_pending_ensemble = None

    def shutdown_adaptive_worker(self) -> None:
        self.stop_adaptive_tuning()

    def request_adaptive_evaluation(self) -> bool:
        if not bool(self._adaptive_cfg("enabled", False)):
            return False
        self._ensure_adaptive_worker()
        with self._adaptive_lock:
            self._adaptive_force_next = True
            # Manual evaluation may use a partially filled window, but never a
            # single instantaneous realization.  The next frame hook queues it
            # once at least two/ensemble snapshots are available.
            required = min(
                max(2, int(self._adaptive_cfg("window_frames", 32))),
                max(2, int(self._adaptive_cfg("ensemble_snapshots", 3))),
            )
            if len(self._adaptive_window) >= required:
                source_window_frames = int(len(self._adaptive_window))
                self._adaptive_pending_ensemble = self._select_ensemble(
                    self._adaptive_window,
                    int(self._adaptive_cfg("ensemble_snapshots", 3)),
                )
                for item in self._adaptive_pending_ensemble:
                    item["source_window_frames"] = source_window_frames
                self._adaptive_state = "queued"
                self._adaptive_force_next = False
                self._adaptive_search_event.set()
                return True
            self._adaptive_state = "waiting_window"
            return False

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
                "current_alpha": float(getattr(self.config, "alpha", float("nan"))),
                "current_beta": float(getattr(self.config, "beta", float("nan"))),
                "recommended_alpha": float(rec.get("recommended_alpha", float("nan"))),
                "recommended_beta": float(rec.get("recommended_beta", float("nan"))),
                "predicted_ser_current": float(rec.get("predicted_ser_current", float("nan"))),
                "predicted_ser_best": float(rec.get("predicted_ser_best", float("nan"))),
                "predicted_ser_ofdm": float(rec.get("predicted_ser_ofdm", float("nan"))),
                "predicted_ser_otfs": float(rec.get("predicted_ser_otfs", float("nan"))),
                "predicted_ser_afdm": float(rec.get("predicted_ser_afdm", float("nan"))),
                "predicted_improvement_db": float(rec.get("predicted_improvement_db", float("nan"))),
                "predicted_snr_db": float(rec.get("predicted_snr_db", float("nan"))),
                "stable_count": int(rec.get("stable_count", 0)),
                "stable_required": int(rec.get("stable_required", self._adaptive_cfg("stability_evals", 2))),
                "candidate_count": int(rec.get("candidate_count", 0)),
                "search_seconds": float(rec.get("search_seconds", float("nan"))),
                "context_key": repr(self._adaptive_context_key),
                "history_len": int(len(self._adaptive_history)),
                "window_fill": int(len(self._adaptive_window)),
                "window_frames": int(self._adaptive_cfg("window_frames", 32)),
                "ensemble_snapshots": int(self._adaptive_cfg("ensemble_snapshots", 3)),
                "coarse_step": float(self._adaptive_cfg("coarse_step", 0.5)),
                "fine_step": float(self._adaptive_cfg("fine_step", 0.1)),
                "interval_frames": int(self._adaptive_cfg("interval_frames", 16)),
                "cooldown_frames": int(self._adaptive_cfg("cooldown_frames", 32)),
                "min_improvement_db": float(self._adaptive_cfg("min_improvement_db", 0.2)),
                "decision_ser_floor": float(self._adaptive_cfg("decision_ser_floor", 1e-8)),
                "display_interval_s": float(self._adaptive_cfg("display_interval_s", 0.5)),
                "display_ema_alpha": float(self._adaptive_cfg("display_ema_alpha", 0.28)),
                "max_order": int(self._adaptive_cfg("max_order", 512)),
                "history_limit": int(self._adaptive_cfg("history_limit", 1000)),
            }

    def get_alpha_beta_adaptation_status(self) -> Dict[str, Any]:
        return self.get_adaptive_status()

    def get_adaptive_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._adaptive_lock:
            history = [dict(x) for x in self._adaptive_history]
        if limit is not None:
            history = history[-max(0, int(limit)) :]
        return history
