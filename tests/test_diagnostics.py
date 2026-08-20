"""诊断 API 测试（阶段 9）。"""
from waveform_sim.core.config import ExperimentConfig
from waveform_sim.diagnostics.health_check import format_health_check, run_health_check
from waveform_sim.diagnostics.report_exporter import export_run_report
from waveform_sim.diagnostics.snapshot import Snapshot
from waveform_sim.service.experiment_service import ExperimentService


def test_health_check_runs():
    results = run_health_check(include_usrp=False)
    assert len(results) == 8
    for r in results:
        assert r.status in ("PASS", "WARN")
    text = format_health_check(results)
    assert "NumPy" in text


def test_report_exporter(tmp_path):
    cfg = ExperimentConfig(runs_dir=str(tmp_path))
    svc = ExperimentService(cfg)
    svc.start_run()
    svc.log_metrics({"frame_id": 1, "ber": 0.0})
    svc.finish_run({})
    report = export_run_report(svc.writer.run_dir)
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert svc.run_id in text
    assert "## Config" in text


def test_snapshot():
    snap = Snapshot.capture("RUNNING", {"ber": 0.1}, {"waveform": "FDIDM"})
    d = snap.to_dict()
    assert d["state"] == "RUNNING"
    assert d["metrics"]["ber"] == 0.1
    assert d["config"]["waveform"] == "FDIDM"

