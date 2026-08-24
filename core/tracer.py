"""The tracer seam: a sink receiving every StepEvent in seq order,
including everything through run/end on a stop — the durable record.
Isolation rule: a tracer raising NEVER kills the run (the opposite of
hooks). Console/Jsonl implementations live in ``core/tracers.py``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .events import StepEvent


@runtime_checkable
class Tracer(Protocol):
    async def on_event(self, event: StepEvent) -> None: ...
