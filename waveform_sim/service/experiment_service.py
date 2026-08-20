"""实验编排：run 生命周期与 artifact 记录。"""
from __future__ import annotations

from typing import Dict, Optional

from ..core.config import ExperimentConfig
from .artifact_writer import RunArtifactWriter
from .event_logger import EventLogger
from .run_state import RunState


class ExperimentService:
    def __init__(self, config: ExperimentConfig | None = None):
        self.config = (config or ExperimentConfig()).normalized()
        self.writer = RunArtifactWriter(self.config)
        self.logger: EventLogger | None = None
        self.state = RunState.CONFIGURED

    @property
    def run_id(self) -> str:
        return self.writer.run_id

    @property
    def run_dir(self):
        return self.writer.run_dir

    def start_run(self) -> str:
        self.state = RunState.STARTING
        self.writer.create()
        self.logger = EventLogger(self.writer.events_path, self.writer.run_id)
        self.log_event("RUN_STARTED", module="experiment_service", payload=self.config.to_dict())
        self.state = RunState.RUNNING
        return self.writer.run_id

    def log_event(
        self,
        event: str,
        module: str = "platform",
        payload: Optional[Dict] = None,
        **context,
    ) -> None:
        if self.logger is not None:
            self.logger.log(event=event, module=module, payload=payload or {}, **context)

    def log_metrics(self, metrics: Dict) -> None:
        self.writer.write_metric(metrics)

    def finish_run(self, summary: Optional[Dict] = None) -> None:
        self.state = RunState.STOPPING
        self.log_event("RUN_FINISHED", module="experiment_service", payload=summary or {})
        self.writer.write_report(summary or {})
        self.state = RunState.STOPPED

    def fail_run(self, error: str) -> None:
        self.state = RunState.ERROR
        self.log_event("RUN_FAILED", module="experiment_service", payload={"error": str(error)})
        self.writer.write_report({"error": str(error)})

