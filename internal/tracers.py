"""Tracer sinks: Console (one line per event), Jsonl (replayable record).

    agent = Agent(llm, tracers=[ConsoleTracer()])
    agent = Agent(llm, tracers=[JsonlTracer("run.jsonl")])

Both satisfy core's ``Tracer`` protocol; write your own by matching it.
The isolation rule lives in the loop, not here: a tracer raising never
kills the run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from core import StepEvent


class ConsoleTracer:
    """Compact one-line-per-event tracing to a stream (default stderr)."""

    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stderr

    async def on_event(self, event: StepEvent) -> None:
        p = event.payload
        brief = {k: p[k] for k in ("name", "point", "hook", "status", "text") if k in p}
        print(f"[{event.seq:>3}] {event.kind.value}/{event.phase.value:<5} "
              f"{event.step_id} {brief}", file=self.stream)


class JsonlTracer:
    """Append every event as one JSON line — a replayable run record."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def on_event(self, event: StepEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), default=str, ensure_ascii=False) + "\n")
