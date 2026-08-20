#!/usr/bin/env python
"""无显示环境下自检：FDIDM 引擎出帧 + 指标 + 实验 artifact。"""
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from waveform_sim.core.config import ExperimentConfig, WaveformConfig
from waveform_sim.core.engine import LinkSimulator
from waveform_sim.service.experiment_service import ExperimentService


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = ExperimentConfig(
            waveform=WaveformConfig(waveform="FDIDM", snr_db=20.0, alpha=0.0, beta=0.0, seed=7),
            runs_dir=str(tmp),
        )
        svc = ExperimentService(cfg)
        svc.start_run()
        sim = LinkSimulator(cfg, experiment_service=svc)
        sim.step()
        sim.step()
        metrics = sim.get_last_metrics()
        svc.finish_run(metrics)
        print("smoke metrics:", {k: metrics.get(k) for k in ("ber", "ser", "evm_db")})
        print("run_id:", svc.run_id)
        print("artifacts:", sorted(p.name for p in svc.writer.run_dir.iterdir()))
        assert metrics.get("total_bits", 0) > 0
        assert (svc.writer.run_dir / "config.json").exists()
    print("self test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

