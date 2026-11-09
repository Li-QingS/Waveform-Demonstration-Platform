# -*- coding: utf-8 -*-
"""Paired live and scan workers for the common waveform comparison engine.

A complete set of waveforms is the publication unit. A cancelled frame or scan
point is never exposed as a comparison with mismatched sample budgets.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
import threading
import time

import numpy as np

from waveform_sim.simulation.compare_scan_backend import (
    CHIRP_CONVENTION, PREFIX, ScanConfig, WaveformScanSimulator,
    _frame_cluster_bootstrap_interval, _paired_frame_difference_interval,
    validate_parameters,
)


MATLAB_BLUE = (0, 114, 189)
MATLAB_ORANGE = (217, 83, 25)
MATLAB_GREEN = (119, 172, 48)
MATLAB_YELLOW = (237, 177, 32)
MATLAB_PURPLE = (126, 47, 142)
LIGHT_PLOT_BACKGROUND = (250, 250, 250)
AXIS_COLOR = (60, 60, 60)
BORDER_COLOR = (225, 225, 225)

_WAVEFORMS = ("OFDM", "OTFS", "AFDM")
_RECEIVERS = ("full_mmse", "practical")


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _run_config(waveforms, snr_db, cfo_hz, doppler_hz, mod_order, seed, receiver_mode,
                afdm_c1=None, afdm_c2=None, csi_mode="ideal"):
    if isinstance(waveforms, str):
        raise ValueError("waveforms must be a sequence of waveform names")
    names = tuple(str(name).upper().strip() for name in waveforms)
    if not names or any(name not in _WAVEFORMS for name in names):
        raise ValueError("Only OFDM, OTFS and AFDM are supported")
    config = {
        "waveforms": names,
        "snr_db": float(snr_db),
        "cfo_hz": float(cfo_hz),
        "doppler_hz": float(doppler_hz),
        "mod_order": str(mod_order).upper().strip(),
        "seed": _integer(seed, "seed"),
        "receiver_mode": str(receiver_mode).lower().strip(),
        "csi_mode": str(csi_mode).lower().strip(),
    }
    if not all(np.isfinite(config[key]) for key in ("snr_db", "cfo_hz", "doppler_hz")):
        raise ValueError("Channel parameters must be finite")
    if config["doppler_hz"] < 0:
        raise ValueError("doppler_hz must be non-negative")
    if config["mod_order"] not in ("QPSK", "16QAM", "64QAM"):
        raise ValueError("Unsupported modulation")
    if config["receiver_mode"] not in _RECEIVERS:
        raise ValueError("receiver_mode must be full_mmse or practical")
    # Use the numerical engine's bounds as well as the public worker schema,
    # so a rejected physical setup never starts a live thread.
    resources = ScanConfig(afdm_c1=afdm_c1, afdm_c2=afdm_c2).validated()
    validated = validate_parameters(
        **{key: value for key, value in config.items() if key != "waveforms"}, config=resources,
    )
    validated.pop("frames")
    config.update(validated)
    config["comparison_mode"] = (
        "unified_baseline" if config["receiver_mode"] == "full_mmse" else "practical_receiver"
    )
    config.update(afdm_c1=resources.afdm_c1, afdm_c2=resources.afdm_c2,
                  subcarrier_spacing_hz=resources.subcarrier_spacing,
                  sample_rate_hz=resources.fft_len * resources.subcarrier_spacing,
                  chirp_convention=CHIRP_CONVENTION, prefix=PREFIX)
    return config


def _simulator_config(config):
    """Reconstruct the immutable waveform settings used by both worker modes."""
    return ScanConfig(afdm_c1=config["afdm_c1"], afdm_c2=config["afdm_c2"]).validated()


def _new_counts():
    return {"bit_errors": 0, "total_bits": 0, "frames": 0, "frame_errors": 0,
            "evm_square_sum": 0.0, "csi_nmse_sum": 0.0,
            "pilot_overhead_fraction": 0.0, "receiver_coefficients_per_row_sum": 0,
            "receiver_complexity_proxy_sum": 0, "receiver_method": "not_run",
            "receiver_iterations_max": 0,
            "receiver_complexity_scope": "equalizer only; excludes waveform transforms, vector updates and CSI estimation",
            "frame_error_counts": [], "frame_bit_counts": []}


def _count_frame(counts, frame):
    errors = _integer(frame["bit_errors"], "bit_errors")
    bits = _integer(frame["total_bits"], "total_bits", minimum=1)
    evm = float(frame["evm_percent"])
    if errors > bits or not np.isfinite(evm) or evm < 0.0:
        raise ValueError("Invalid raw frame metrics")
    counts["bit_errors"] += errors
    counts["total_bits"] += bits
    counts["frames"] += 1
    counts["frame_errors"] += int(errors > 0)
    counts["evm_square_sum"] += evm * evm
    counts["csi_nmse_sum"] += float(frame.get("csi_nmse", 0.0))
    counts["pilot_overhead_fraction"] = float(frame.get("pilot_overhead_fraction", 0.0))
    counts["receiver_coefficients_per_row_sum"] += int(frame.get("receiver_coefficients_per_row", 0))
    counts["receiver_complexity_proxy_sum"] += int(frame.get("receiver_complexity_proxy", 0))
    counts["receiver_method"] = str(frame.get("receiver_method", "unknown"))
    counts["receiver_iterations_max"] = max(
        counts["receiver_iterations_max"], int(frame.get("receiver_iterations_max", 0))
    )
    counts["receiver_complexity_scope"] = str(frame.get("receiver_complexity_scope", "unknown"))
    counts["frame_error_counts"].append(errors)
    counts["frame_bit_counts"].append(bits)


def _metrics(counts):
    bits = counts["total_bits"]
    frames = counts["frames"]
    low, high, method = _frame_cluster_bootstrap_interval(
        counts["frame_error_counts"], counts["frame_bit_counts"]
    )
    return {
        "bit_errors": counts["bit_errors"], "total_bits": bits, "frames": frames,
        "frame_errors": counts["frame_errors"],
        "ber": counts["bit_errors"] / bits if bits else 0.0,
        "fer": counts["frame_errors"] / frames if frames else 0.0,
        "ci_low": low, "ci_high": high, "ci_method": method,
        "independent_frames": frames,
        "ci_resamples": 2000 if method == "frame_cluster_bootstrap" else 0,
        "evm_percent": math.sqrt(counts["evm_square_sum"] / frames) if frames else 0.0,
        "csi_nmse": counts["csi_nmse_sum"] / frames if frames else 0.0,
        "pilot_overhead_fraction": float(counts["pilot_overhead_fraction"]),
        "receiver_coefficients_per_row": (
            counts["receiver_coefficients_per_row_sum"] / frames if frames else 0.0
        ),
        "receiver_complexity_proxy": (
            counts["receiver_complexity_proxy_sum"] / frames if frames else 0.0
        ),
        "receiver_method": counts["receiver_method"],
        "receiver_iterations_max": int(counts["receiver_iterations_max"]),
        "receiver_complexity_scope": counts["receiver_complexity_scope"],
    }


def _frame_set(simulator, config, frame_index, stop_event):
    """Use the exact same call contract for live frames and scan frames."""
    if stop_event.is_set():
        raise InterruptedError("Comparison cancelled")
    result = simulator.simulate_frame_set(
        waveforms=config["waveforms"], snr_db=config["snr_db"],
        cfo_hz=config["cfo_hz"], doppler_hz=config["doppler_hz"],
        mod_order=config["mod_order"], seed=config["seed"], frame_index=frame_index,
        receiver_mode=config["receiver_mode"], stop_event=stop_event,
        csi_mode=config.get("csi_mode", "ideal"),
    )
    if stop_event.is_set():
        raise InterruptedError("Comparison cancelled")
    names = tuple(dict.fromkeys(config["waveforms"]))
    if any(name not in result for name in names):
        raise RuntimeError("Incomplete paired waveform frame")
    budgets = {_integer(result[name]["total_bits"], "total_bits", minimum=1) for name in names}
    if len(budgets) != 1:
        raise RuntimeError("Waveforms do not have equal information-bit budgets")
    return result


class _PairedWaveformRunner:
    """One thread publishes both live panels from the same complete frame set.

    stop() requests cancellation. wait() must confirm termination before a
    caller discards this runner or starts a replacement session. A timeout
    retains the live thread reference and raises instead of losing ownership.
    """

    def __init__(self, frame_interval_s=0.05, history_len=600):
        interval = float(frame_interval_s)
        if not np.isfinite(interval) or interval < 0:
            raise ValueError("frame_interval_s must be finite and non-negative")
        self.frame_interval_s = interval
        self.history_len = _integer(history_len, "history_len", minimum=1)
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()
        self._snapshot = {
            "left": self._empty_panel(), "right": self._empty_panel(),
            "error": "", "done": True, "cancelled": False, "config": {},
        }

    @staticmethod
    def _empty_panel():
        return {
            "constellation": np.zeros(0, dtype=np.complex128),
            "ber_t": np.zeros(0, dtype=float), "ber_v": np.zeros(0, dtype=float),
            "ber_upper": np.zeros(0, dtype=float), **_metrics(_new_counts()),
        }

    def is_running(self):
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self, waveforms, snr_db, cfo_hz, doppler_hz, mod_order,
              seed=42, receiver_mode="full_mmse", afdm_c1=None, afdm_c2=None,
              csi_mode="ideal"):
        config = _run_config(waveforms, snr_db, cfo_hz, doppler_hz, mod_order, seed, receiver_mode,
                             afdm_c1, afdm_c2, csi_mode)
        if len(config["waveforms"]) != 2:
            raise ValueError("The live comparison requires exactly two waveform selections")
        with self._lifecycle_lock:
            if self.is_running():
                raise RuntimeError("上次配对仿真尚未退出，请先停止并等待任务结束。")
            self._stop_event = threading.Event()
            with self._lock:
                self._snapshot = {
                    "left": self._empty_panel(), "right": self._empty_panel(),
                    "error": "", "done": False, "cancelled": False, "config": deepcopy(config),
                }
            self._thread = threading.Thread(
                target=self._run, args=(config, self._stop_event),
                name="paired-waveform-comparison", daemon=True,
            )
            try:
                self._thread.start()
            except Exception as exc:
                with self._lock:
                    self._snapshot.update(done=True, error=f"{type(exc).__name__}: {exc}")
                raise

    def stop(self):
        self._stop_event.set()

    def wait(self, timeout=1.5):
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None:
                return
            if thread.ident is None and not thread.is_alive():
                # Thread.start() can fail before an OS thread is created.
                # There is then no live resource to join or discard.
                return
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise RuntimeError("配对仿真线程尚未退出；已保留任务引用，请稍后重试。")

    def snapshot(self):
        with self._lock:
            return deepcopy(self._snapshot)

    def _run(self, config, stop_event):
        error = ""
        cancelled = False
        names = tuple(dict.fromkeys(config["waveforms"]))
        counts = {name: _new_counts() for name in names}
        histories = {name: {key: deque(maxlen=self.history_len) for key in ("t", "ber", "upper")} for name in names}
        started_at = time.monotonic()
        try:
            simulator = WaveformScanSimulator(_simulator_config(config))
            frame_index = 0
            while not stop_event.is_set():
                result = _frame_set(simulator, config, frame_index, stop_event)
                elapsed = time.monotonic() - started_at
                panels = {}
                for name in names:
                    raw = result[name]
                    _count_frame(counts[name], raw)
                    metrics = _metrics(counts[name])
                    history = histories[name]
                    history["t"].append(elapsed)
                    history["ber"].append(metrics["ber"])
                    history["upper"].append(metrics["ci_high"])
                    constellation = np.asarray(raw.get("constellation", raw.get("rx_symbols", [])), dtype=np.complex128).reshape(-1)
                    panels[name] = {
                        "constellation": constellation.copy(),
                        "ber_t": np.asarray(history["t"], dtype=float),
                        "ber_v": np.asarray(history["ber"], dtype=float),
                        "ber_upper": np.asarray(history["upper"], dtype=float), **metrics,
                    }
                # No intermediate side is visible if another waveform failed.
                with self._lock:
                    if stop_event.is_set():
                        raise InterruptedError("Comparison cancelled")
                    self._snapshot.update(
                        left=deepcopy(panels[config["waveforms"][0]]),
                        right=deepcopy(panels[config["waveforms"][1]]),
                    )
                frame_index += 1
                if stop_event.wait(self.frame_interval_s):
                    break
        except InterruptedError:
            cancelled = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._snapshot.update(error=error, done=True, cancelled=cancelled or (stop_event.is_set() and not error))


class _ScanWorker(threading.Thread):
    """Point-major, paired Monte-Carlo scan; only complete points are published."""

    def __init__(self, result_store: dict, result_lock: threading.Lock,
                 stop_event: threading.Event, params: dict):
        super().__init__(name="paired-waveform-scan", daemon=True)
        self.result_store = result_store
        self.result_lock = result_lock
        self.stop_event = stop_event
        self.params = deepcopy(params)

    def _put_point(self, family, value, counts, text):
        metrics = {name: _metrics(counts[name]) for name in _WAVEFORMS}
        for name in _WAVEFORMS:
            for other in _WAVEFORMS:
                if name == other:
                    continue
                low, high, method = _paired_frame_difference_interval(
                    counts[name]["frame_error_counts"],
                    counts[name]["frame_bit_counts"],
                    counts[other]["frame_error_counts"],
                    counts[other]["frame_bit_counts"],
                )
                metrics[name][f"ber_diff_vs_{other}"] = (
                    metrics[name]["ber"] - metrics[other]["ber"]
                )
                metrics[name][f"ber_diff_vs_{other}_ci_low"] = low
                metrics[name][f"ber_diff_vs_{other}_ci_high"] = high
                metrics[name][f"ber_diff_vs_{other}_ci_method"] = method
        with self.result_lock:
            if self.stop_event.is_set():
                raise InterruptedError("Scan cancelled")
            destination = self.result_store.setdefault(family, {})
            updated = {}
            for name in _WAVEFORMS:
                previous = destination.get(name, {})
                point = {"x": float(value), **metrics[name]}
                updated[name] = {
                    key: np.append(np.asarray(previous.get(key, []), dtype=(
                        np.int64 if key in ("bit_errors", "total_bits", "frames", "frame_errors")
                        else object if key.endswith("_method") or key.endswith("_scope")
                        else float
                    )), val)
                    for key, val in point.items()
                }
            # Replace the family atomically rather than one waveform at a time.
            self.result_store[family] = updated
            self.result_store["status"] = text

    def run(self):
        error = ""
        cancelled = False
        completed_points = 0
        with self.result_lock:
            self.result_store.update(done=False, error="", cancelled=False, config=deepcopy(self.params))
            self.result_store.setdefault("cfo", {})
            self.result_store.setdefault("doppler", {})
        try:
            config = _run_config(
                _WAVEFORMS, self.params["snr_db"], self.params["fixed_cfo"],
                self.params["fixed_doppler"], self.params["mod_order"],
                self.params.get("seed", 42), self.params.get("receiver_mode", "full_mmse"),
                self.params.get("afdm_c1"), self.params.get("afdm_c2"),
                self.params.get("csi_mode", "ideal"),
            )
            resources = _simulator_config(config)
            frames = _integer(self.params["frames"], "frames", minimum=1)
            axes = {}
            for family in ("cfo", "doppler"):
                values = np.asarray(self.params[f"{family}_axis"], dtype=float)
                if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
                    raise ValueError("扫描坐标必须是非空的一维有限数值数组。")
                if family == "doppler" and np.any(values < 0):
                    raise ValueError("多普勒扩展不能为负。")
                for value in values:
                    validate_parameters(
                        snr_db=config["snr_db"],
                        cfo_hz=float(value) if family == "cfo" else config["cfo_hz"],
                        doppler_hz=float(value) if family == "doppler" else config["doppler_hz"],
                        mod_order=config["mod_order"], frames=frames,
                        seed=config["seed"], receiver_mode=config["receiver_mode"],
                        config=resources, csi_mode=config["csi_mode"],
                    )
                axes[family] = values.copy()
            with self.result_lock:
                self.result_store["config"] = {
                    **deepcopy(self.params), **config,
                    "frames": frames, "cfo_axis": axes["cfo"].copy(), "doppler_axis": axes["doppler"].copy(),
                }
            if self.stop_event.is_set():
                raise InterruptedError("Scan cancelled")
            with self.result_lock:
                self.result_store["status"] = "正在准备三波形配对比较..."
            simulator = WaveformScanSimulator(resources)
            total_points = sum(len(values) for values in axes.values())
            for family, values in axes.items():
                for value in values:
                    point_config = dict(config)
                    point_config["cfo_hz" if family == "cfo" else "doppler_hz"] = float(value)
                    counts = {name: _new_counts() for name in _WAVEFORMS}
                    for frame_index in range(frames):
                        result = _frame_set(simulator, point_config, frame_index, self.stop_event)
                        for name in _WAVEFORMS:
                            _count_frame(counts[name], result[name])
                        with self.result_lock:
                            self.result_store["status"] = (
                                f"扫描中：{family.upper()}={float(value):g}，"
                                f"配对帧 {frame_index + 1}/{frames}；已完成 {completed_points}/{total_points} 点。"
                            )
                    self._put_point(
                        family, value, counts,
                        f"扫描中：已完成 {completed_points + 1}/{total_points} 个完整配对点，每点 {frames} 帧。",
                    )
                    completed_points += 1
            status = f"扫描完成：{completed_points} 个配对点，三波形共享比特、信道、训练条件与噪声样本。"
        except InterruptedError:
            cancelled = True
            status = f"扫描已取消，保留 {completed_points} 个完整配对点。"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            status = f"扫描失败：{error}；已保留 {completed_points} 个完整配对点。"
        finally:
            cancelled = cancelled or (self.stop_event.is_set() and not error)
            if cancelled and not error:
                status = f"扫描已取消，保留 {completed_points} 个完整配对点。"
            with self.result_lock:
                self.result_store.update(done=True, error=error, cancelled=cancelled, status=status)
