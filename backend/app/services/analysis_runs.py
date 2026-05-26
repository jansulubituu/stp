import json
import threading
import time
import uuid
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalysisRun:
    events: list[dict[str, Any]] = field(default_factory=list)
    complete: bool = False
    condition: threading.Condition = field(default_factory=threading.Condition)
    created_at: float = field(default_factory=time.time)


class AnalysisRunRegistry:
    """In-memory SSE event registry for local/single-process execution."""

    def __init__(self) -> None:
        self._runs: dict[str, AnalysisRun] = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        run_id = uuid.uuid4().hex
        with self._lock:
            if len(self._runs) >= 100:
                completed = sorted(
                    ((key, run) for key, run in self._runs.items() if run.complete),
                    key=lambda item: item[1].created_at,
                )
                for key, _run in completed[: max(1, len(self._runs) - 99)]:
                    self._runs.pop(key, None)
            self._runs[run_id] = AnalysisRun()
        return run_id

    def publish(self, run_id: str, event_type: str, data: dict[str, Any], *, complete: bool = False) -> None:
        run = self.get(run_id)
        with run.condition:
            run.events.append({"type": event_type, "data": data})
            run.complete = run.complete or complete
            run.condition.notify_all()

    def get(self, run_id: str) -> AnalysisRun:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        return run

    def events(self, run_id: str) -> Generator[str, None, None]:
        run = self.get(run_id)
        cursor = 0
        while True:
            with run.condition:
                if cursor >= len(run.events) and not run.complete:
                    run.condition.wait(timeout=15)
                pending = run.events[cursor:]
                finished = run.complete and cursor + len(pending) >= len(run.events)
            if not pending:
                if finished:
                    return
                yield ": ping\n\n"
                continue
            for event in pending:
                cursor += 1
                payload = json.dumps(event["data"], ensure_ascii=False)
                yield f"event: {event['type']}\ndata: {payload}\n\n"
            if finished:
                return


analysis_runs = AnalysisRunRegistry()
