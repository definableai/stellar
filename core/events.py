"""The event grammar. One shape, two axes: kind x phase.

Every observable thing the agent does is a *step*. Every step emits
``start`` and ``end`` exactly once, and zero or more ``delta`` events
in between. That is the entire grammar.

kinds: run (the whole run) / text (one LLM generation) /
tool (one execution) / hook (one invocation).
Payload contracts by (kind, phase) — see README for the full table.
``seq`` is a per-run monotonic counter: it is the total order of the
run and the resume cursor for reconnecting consumers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepKind(str, Enum):
    RUN = "run"
    TEXT = "text"
    TOOL = "tool"
    HOOK = "hook"


class StepPhase(str, Enum):
    START = "start"
    DELTA = "delta"
    END = "end"


@dataclass
class StepEvent:
    run_id: str
    step_id: str
    kind: StepKind
    phase: StepPhase
    seq: int
    ts: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """e.g. 'step_start' — the three event categories on the wire."""
        return f"step_{self.phase.value}"

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "step_id": self.step_id,
                "kind": self.kind.value, "phase": self.phase.value,
                "seq": self.seq, "ts": self.ts, "payload": self.payload}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)
