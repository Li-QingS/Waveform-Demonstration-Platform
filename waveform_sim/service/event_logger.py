"""追加式 JSONL 事件日志。"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class EventLogger:
    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self._lock = threading.Lock()
        self._seq = 0

    def log(
        self,
        event: str,
        module: str = "platform",
        payload: Optional[Dict[str, Any]] = None,
        **context,
    ) -> Dict[str, Any]:
        with self._lock:
            self._seq += 1
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                "seq": self._seq,
                "module": module,
                "event": event,
                "payload": payload or {},
            }
            record.update({k: v for k, v in context.items() if v is not None})
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            return record

