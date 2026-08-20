"""状态快照 dataclass。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict


@dataclass
class Snapshot:
    timestamp: str
    state: str
    metrics: Dict
    config: Dict

    @classmethod
    def capture(cls, state: str, metrics: Dict, config: Dict) -> "Snapshot":
        return cls(datetime.now(timezone.utc).isoformat(), state, dict(metrics), dict(config))

    def to_dict(self):
        return asdict(self)

