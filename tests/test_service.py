"""实验记录服务与引擎挂接测试（阶段 7）。"""
import json
import time

from waveform_sim.core.config import ExperimentConfig, WaveformConfig
from waveform_sim.core.engine import LinkSimulator
from waveform_sim.service.experiment_service import ExperimentService


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_run_artifacts(tmp_path):
    cfg = ExperimentConfig(waveform=WaveformConfig(waveform="FDIDM"), runs_dir=str(tmp_path))
    svc = ExperimentService(cfg)
    run_id = svc.start_run()
    svc.log_event("TEST", module="pytest", payload={"ok": True})
    svc.log_metrics({"frame_id": 1, "ber": 0.0, "evm_db": -30.0})
    svc.finish_run({"ok": True})
    run_dir = tmp_path / run_id
    assert (run_dir / "config.json").exists()
    assert (run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "report.md").exists()
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[0]["event"] == "RUN_STARTED"
    assert events[-1]["event"] == "RUN_FINISHED"


def test_engine_experiment_hooks(tmp_path):
    cfg = ExperimentConfig(waveform=WaveformConfig(waveform="FDIDM", snr_db=12.0), runs_dir=str(tmp_path))
    svc = ExperimentService(cfg)
    sim = LinkSimulator(cfg, experiment_service=svc)
    svc.start_run()
    sim.step()
    sim.step()
    sim.start()
    try:
        ok = _wait_until(lambda: sim.get_last_metrics().get("total_bits", 0) > 0)
        assert ok
    finally:
        sim.stop()
        sim.wait(timeout=3.0)
    svc.finish_run(sim.get_last_metrics())
    events = (svc.writer.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert "LINK_STARTED" in events
    assert "LINK_STOPPED" in events
    rows = (svc.writer.metrics_path).read_text(encoding="utf-8").splitlines()
    assert len(rows) >= 3  # 表头 + 两次 step() 指标


def test_engine_auto_start_run(tmp_path):
    cfg = ExperimentConfig(waveform=WaveformConfig(waveform="FDIDM", snr_db=15.0), runs_dir=str(tmp_path))
    sim = LinkSimulator(cfg, auto_start_run=True)
    assert sim.experiment_service is not None
    assert sim.experiment_service.state.value == "RUNNING"
    sim.stop()
    sim.experiment_service.finish_run({})
    assert (sim.experiment_service.writer.run_dir / "config.json").exists()

