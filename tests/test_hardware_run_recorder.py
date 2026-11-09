import csv
import json

from waveform_sim.core.config import ExperimentConfig, WaveformConfig
from waveform_sim.service.hardware_run_recorder import HardwareRunRecorder


def test_hardware_run_records_sampled_frame_snapshot_and_captures_configuration(tmp_path):
    config = ExperimentConfig(
        waveform=WaveformConfig(waveform="OFDM", mod_order="16QAM"),
        runs_dir=str(tmp_path),
    )
    recorder = HardwareRunRecorder(
        config,
        metadata={
            "channel_injection": {"enabled": True, "profile": "NTN-TDLC5-200"},
            "rf": {"center_freq_hz": 2.4e9, "sample_rate_hz": 960_000},
        },
    )

    run_id = recorder.start()
    status = {
        "frame_id": 7,
        "frame_bit_errors": 2,
        "frame_bits_total": 128,
        "frame_ber": 2 / 128,
        "bit_errors": 5,
        "bits_total": 512,
        "cumulative_ber": 5 / 512,
        "decode_ok": False,
    }
    assert recorder.record_status(status) is True
    assert recorder.record_status(status) is False
    recorder.finish({"frames_processed": 7})

    run_dir = tmp_path / run_id
    config_data = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config_data["waveform"]["waveform"] == "OFDM"
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    configured = next(event for event in events if event["event"] == "HARDWARE_CONFIGURED")
    assert configured["payload"]["channel_injection"]["profile"] == "NTN-TDLC5-200"

    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["frame_id"] == "7"
    assert rows[0]["recording_type"] == "polling_snapshot"
    assert rows[0]["skipped_frames_since_last"] == "6"
    assert rows[0]["frame_bit_errors"] == "2"
    assert rows[0]["frame_bits_total"] == "128"
    assert rows[0]["bit_errors"] == "5"
    assert rows[0]["bits_total"] == "512"
    assert float(rows[0]["ber"]) == 5 / 512
    assert rows[0]["decode_ok"] == "False"


def test_recorder_skips_empty_frame_zero_and_marks_polling_frame_gaps(tmp_path):
    config = ExperimentConfig(
        waveform=WaveformConfig(waveform="OFDM"),
        runs_dir=str(tmp_path),
    )
    recorder = HardwareRunRecorder(config)
    run_id = recorder.start()

    assert recorder.record_status({"frame_id": 0, "bits_total": 0}) is False
    assert recorder.record_status({"frame_id": 2, "bits_total": 200, "bit_errors": 1}) is True
    assert recorder.record_status({"frame_id": 5, "bits_total": 500, "bit_errors": 3}) is True
    recorder.finish()

    with (tmp_path / run_id / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["frame_id"] for row in rows] == ["2", "5"]
    assert [row["recording_type"] for row in rows] == ["polling_snapshot"] * 2
    assert [row["skipped_frames_since_last"] for row in rows] == ["1", "2"]
