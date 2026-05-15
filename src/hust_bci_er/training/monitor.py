"""JSONL training monitor for long-running experiments."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping


class TrainingMonitor:
    def __init__(self, run_dir: Path, *, route_id: str, run_id: str) -> None:
        self.run_dir = run_dir
        self.route_id = route_id
        self.run_id = run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "training_events.jsonl"
        self.heartbeat_path = self.run_dir / "heartbeat.json"

    def log(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        record = {
            "time": time.time(),
            "route_id": self.route_id,
            "run_id": self.run_id,
            "event": event,
            "payload": dict(payload or {}),
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.heartbeat_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def epoch(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        self.log("epoch", {"epoch": int(epoch), "metrics": dict(metrics)})

    def finish(self, status: str = "completed") -> None:
        self.log("finish", {"status": status})
