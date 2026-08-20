"""统一链路引擎（阶段 3）。

LinkSimulator 提供统一的配置、生命周期、指标、绘图与自适应入口；
FDIDM 后端当前为过渡包装（委托 waveform_sim.simulation.simple_fdidm_rx
的 _LegacyFDIDMTransceiver），后续阶段将算法逐步内聚到 core。
"""
from __future__ import annotations

import threading
from typing import Callable, Dict, Optional

from .config import AdaptiveConfig, ExperimentConfig, WaveformConfig

try:
    from waveform_sim.service.experiment_service import ExperimentService
except Exception:  # pragma: no cover - 部分安装时容错
    ExperimentService = None


_ALIASES = {
    "ebn0_db": "snr_db",
    "modulation": "mod_order",
    "decoder": "detector",
    "fc_hz": "center_freq_hz",
    "channel_seed": "seed",
    "fft_len": "fft_size",
    "doppler_spread": "doppler_spread_hz",
    "doppler_freq": "doppler_spread_hz",
    "doppler_hz": "doppler_spread_hz",
    "sample_rate": "sample_rate_hz",
}


class LinkSimulator:
    def __init__(
        self,
        config=None,
        *,
        adaptive: Optional[AdaptiveConfig] = None,
        experiment_service=None,
        auto_start_run: bool = False,
        backend=None,
        **kwargs,
    ):
        if isinstance(config, ExperimentConfig):
            self.experiment_config = config.normalized()
            self.config = self.experiment_config.waveform
            self.adaptive_config = self.experiment_config.adaptive
        else:
            if config is None:
                mapped = {_ALIASES.get(k, k): v for k, v in kwargs.items()}
                self.config = WaveformConfig(**mapped).normalized()
            elif isinstance(config, WaveformConfig):
                self.config = config.normalized()
            else:
                raise TypeError(f"Unsupported config type: {type(config)}")
            self.adaptive_config = (adaptive or AdaptiveConfig()).normalized()
            self.experiment_config = ExperimentConfig(
                waveform=self.config, adaptive=self.adaptive_config
            ).normalized()
        self.experiment_service = experiment_service
        if auto_start_run and ExperimentService is not None and self.experiment_service is None:
            self.experiment_service = ExperimentService(self.experiment_config)
            self.experiment_service.start_run()
        self._backend = backend if backend is not None else self._create_backend()

    # ------------------------------------------------------------ backend
    def _create_backend(self):
        if self.config.waveform == "FDIDM":
            # 阶段3过渡依赖：后续阶段将算法内聚进 core 后移除
            from waveform_sim.simulation.simple_fdidm_rx import _create_fdidm_backend

            return _create_fdidm_backend(**self._fdidm_legacy_kwargs())
        if self.config.waveform == "OFDM":
            from waveform_sim.simulation.simple_ofdm_rx import _create_ofdm_backend

            return _create_ofdm_backend(**self._ofdm_legacy_kwargs())
        if self.config.waveform == "OTFS":
            from waveform_sim.simulation.simple_otfs_rx import _create_otfs_backend

            return _create_otfs_backend(**self._otfs_legacy_kwargs())
        if self.config.waveform == "AFDM":
            from waveform_sim.simulation.simple_afdm_rx import _create_afdm_backend

            return _create_afdm_backend(**self._afdm_legacy_kwargs())
        raise NotImplementedError(
            f"LinkSimulator backend for {self.config.waveform} is not ready"
        )

    def _fdidm_legacy_kwargs(self) -> Dict:
        c = self.config
        return dict(
            alpha=float(c.alpha),
            beta=float(c.beta),
            m_subcarriers=int(c.m_subcarriers),
            n_symbols=int(c.n_symbols),
            subcarrier_spacing_hz=float(c.subcarrier_spacing_hz),
            mod_order=str(c.mod_order),
            channel_model=str(c.channel_model),
            velocity_kmh=float(c.velocity_kmh),
            doppler_radial_factor=float(c.doppler_radial_factor),
            decoder=str(c.detector),
            snr_db=float(c.snr_db),
            snr_definition=str(c.snr_definition),
            optimize_indices=bool(c.optimize_indices),
            search_step=float(c.search_step),
            fc_hz=float(c.center_freq_hz),
            link_mode=str(c.link_mode),
            search_objective=str(c.search_objective),
            random_channel=bool(c.random_channel),
            channel_seed=int(c.seed),
            dynamic_channel=bool(c.dynamic_channel),
            channel_coherence_frames=int(c.channel_coherence_frames),
            channel_dynamics=str(c.channel_dynamics),
            fast_channel_coherence_symbols=int(c.fast_channel_coherence_symbols),
            circular_channel=bool(c.circular_channel),
            tf_notch_depth_db=float(c.tf_notch_depth_db),
            tf_notch_count=int(c.tf_notch_count),
        )

    def _ofdm_legacy_kwargs(self) -> Dict:
        c = self.config
        return dict(
            fft_len=int(c.fft_size),
            cp_len=int(c.cp_len),
            snr_db=float(c.snr_db),
            cfo_hz=float(c.cfo_hz),
            doppler_spread_hz=float(c.doppler_spread_hz),
            delay_spread=int(c.delay_spread),
            mod_order=str(c.mod_order),
            payload_symbols=int(c.payload_symbols),
        )

    def _otfs_legacy_kwargs(self) -> Dict:
        c = self.config
        return dict(
            delay_spread=int(c.delay_spread),
            doppler_spread=float(c.doppler_spread_hz),
            snr_db=float(c.snr_db),
            mod_order=str(c.mod_order),
            cfo_hz=float(c.cfo_hz),
            n_subcarriers=int(c.n_subcarriers),
            n_symbols=int(c.n_symbols),
            sample_rate=float(c.sample_rate_hz),
            update_period=float(c.update_period),
            equalizer=str(c.equalizer),
        )

    def _afdm_legacy_kwargs(self) -> Dict:
        c = self.config
        return dict(
            c1=float(c.c1),
            c2=float(c.c2),
            snr_db=float(c.snr_db),
            mod_order=str(c.mod_order),
            doppler_freq=float(c.doppler_spread_hz),
            delay_spread=int(c.delay_spread),
            cfo_hz=float(c.cfo_hz),
            sample_rate=float(c.sample_rate_hz),
            frame_size=int(c.frame_size),
        )

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._backend.start()
        self._log_event("LINK_STARTED", {"waveform": self.config.waveform})

    def stop(self) -> None:
        self._backend.stop()
        self._log_event("LINK_STOPPED", {"waveform": self.config.waveform})

    def wait(self, timeout: Optional[float] = None) -> None:
        self._backend.wait(timeout=timeout)

    def step(self) -> None:
        self._backend.step()
        self._log_metric(self.get_last_metrics())

    # ------------------------------------------------------------ config
    def update_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            attr = _ALIASES.get(key, key)
            if hasattr(self.config, attr):
                setattr(self.config, attr, value)
        self.config.normalized()
        updater = getattr(self._backend, "update_runtime_parameters", None)
        if updater is not None:
            updater(**kwargs)

    def set_indices(self, alpha: float, beta: float) -> None:
        self.config.alpha = float(alpha)
        self.config.beta = float(beta)
        self._backend.set_indices(alpha, beta)
        self._log_event("INDICES_SET", {"alpha": float(alpha), "beta": float(beta)})

    def update_runtime_parameters(self, **kwargs) -> None:
        self.update_config(**kwargs)

    # ------------------------------------------------------------ metrics
    def get_last_metrics(self) -> Dict:
        return dict(self._backend.get_last_metrics())

    def get_ber_summary(self) -> Dict:
        summary = getattr(self._backend, "get_ber_summary", None)
        if summary is not None:
            return summary()
        m = self.get_last_metrics()
        return {
            "cumulative_ber": float(m.get("ber", 1.0)),
            "frames_processed": int(m.get("frames", 0)),
            "frames_decode_ok": int(m.get("frames_decode_ok", 0)),
            "frame_error_rate": float(m.get("fer", 1.0)),
            "bit_errors": int(m.get("bit_errors", 0)),
            "bits_total": int(m.get("total_bits", 0)),
        }

    def get_ber_estimate(self) -> float:
        m = self.get_last_metrics()
        return float(m.get("ber", m.get("ber_window", 1.0)))

    def reset_ber_stats(self) -> None:
        reset = getattr(self._backend, "reset_ber_stats", None)
        if reset is not None:
            reset()

    def get_plot_data(self) -> Dict:
        backend = self._backend
        data = {}
        for key, method in (
            ("constellation", "get_constellation"),
            ("pre_eq_constellation", "get_pre_eq_constellation"),
            ("ser", "get_ser_history"),
            ("ber", "get_ber_history"),
            ("impulse", "get_cross_domain_impulse_response"),
        ):
            fn = getattr(backend, method, None)
            if fn is not None:
                try:
                    data[key] = fn()
                except Exception:
                    pass
        return data

    # ------------------------------------------------------------ adaptive
    def start_adaptive_tuning(
        self,
        config: Optional[AdaptiveConfig] = None,
        callback: Optional[Callable] = None,
        **cfg,
    ) -> None:
        if config is not None:
            self.adaptive_config = config.normalized()
        backend = self._backend
        fn = getattr(backend, "start_adaptive_tuning", None)
        if fn is not None:
            kwargs = dict(cfg)
            if callback is not None:
                kwargs["callback"] = callback
            if kwargs:
                fn(**kwargs)
            else:
                fn()
            return
        if hasattr(backend, "search_best_indices"):
            def worker():
                result = backend.search_best_indices()
                if callback is not None:
                    callback(result)

            threading.Thread(target=worker, daemon=True, name="engine-adaptive").start()
            return
        raise NotImplementedError(
            f"adaptive tuning not available for {self.config.waveform}"
        )

    def stop_adaptive_tuning(self) -> None:
        stop = getattr(self._backend, "stop_adaptive_tuning", None)
        if stop is not None:
            stop()

    def get_adaptive_status(self) -> Dict:
        backend = self._backend
        for name in ("get_adaptive_status", "get_alpha_beta_adaptation_status"):
            fn = getattr(backend, name, None)
            if fn is not None:
                try:
                    return fn()
                except Exception:
                    continue
        return {"active": False}

    def get_adaptive_history(self, limit=None) -> list:
        fn = getattr(self._backend, "get_adaptive_history", None)
        if fn is not None:
            try:
                return fn(limit=limit)
            except Exception:
                return []
        return []

    # ------------------------------------------------------------ experiment hooks
    def _log_event(self, event: str, payload: Dict) -> None:
        svc = getattr(self, "experiment_service", None)
        if svc is not None:
            try:
                svc.log_event(event=event, module="link_simulator", payload=payload)
            except Exception:
                pass

    def _log_metric(self, metrics: Dict) -> None:
        svc = getattr(self, "experiment_service", None)
        if svc is not None:
            try:
                svc.log_metrics(metrics)
            except Exception:
                pass
