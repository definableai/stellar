"""Run machinery: RunContext, RunResult, RunHandle.

``agent.run()`` returns a RunHandle immediately. The handle:

    * owns the event stream — every event is buffered with a monotonic
      ``seq``, so any number of consumers can attach at any time and
      ``events(after_seq=n)`` replays the past then follows live
      (your durable stream / reconnect story);
    * owns cancellation — ``stop()`` is graceful: current chunk finishes,
      text/end (partial), tool/end (cancelled), run/end (status="stopped")
      still reach every tracer and subscriber;
    * owns completion — ``await handle.result()`` yields the RunResult.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

from .events import StepEvent, StepKind, StepPhase
from .types import ErrorInfo, Message, Usage, new_id

RunStatus = Literal["running", "completed", "stopped", "error"]

_DONE = object()  # stream sentinel


@dataclass
class RunResult:
    run_id: str
    status: RunStatus
    messages: list[Message]        # full transcript, including new turns
    output: str | None             # final assistant text (may be partial on stop)
    usage: Usage
    error: ErrorInfo | None = None
    stop_reason: str | None = None  # e.g. "user", "max_steps"


@dataclass
class RunContext:
    """Passed to hooks and tools. ``state`` is user scratch space."""

    run_id: str
    handle: "RunHandle"
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def stop_requested(self) -> bool:
        return self.handle.stop_requested

    def stop(self, reason: str = "context") -> None:
        """Hooks/tools may request a graceful stop."""
        self.handle.stop(reason)


class RunHandle:
    def __init__(self, run_id: str | None, tracers: list[Any]) -> None:
        self.run_id = run_id or new_id("run")
        self._tracers = tracers
        self._buffer: list[StepEvent] = []
        self._subs: list[asyncio.Queue] = []
        self._seq = 0
        self._stop = asyncio.Event()
        self._stop_reason: str | None = None
        self._result_fut: asyncio.Future[RunResult] = (
            asyncio.get_running_loop().create_future()
        )
        self._finished = False
        self._inbox: list[Message] = []         # mid-run steering queue
        self._task: asyncio.Task | None = None  # set by Agent.run()

    # ---- stream side -------------------------------------------------

    async def emit(
        self,
        kind: StepKind,
        phase: StepPhase,
        payload: dict[str, Any],
        step_id: str,
    ) -> StepEvent:
        """Internal: called by the loop for every event."""
        self._seq += 1
        event = StepEvent(
            run_id=self.run_id,
            step_id=step_id,
            kind=kind,
            phase=phase,
            seq=self._seq,
            payload=payload,
        )
        self._buffer.append(event)
        for q in list(self._subs):
            q.put_nowait(event)
        for tracer in self._tracers:
            try:
                await tracer.on_event(event)
            except Exception:
                pass  # tracer isolation: observability never kills the run
        return event

    async def events(self, after_seq: int = -1) -> AsyncIterator[StepEvent]:
        """Replay buffered events past ``after_seq``, then follow live.

        Safe to call multiple times, from multiple consumers, and after
        the run has finished (pure replay).
        """
        q: asyncio.Queue = asyncio.Queue()
        self._subs.append(q)
        try:
            last = after_seq
            for event in list(self._buffer):
                if event.seq > last:
                    yield event
                    last = event.seq
            if self._finished:
                return
            while True:
                item = await q.get()
                if item is _DONE:
                    return
                if item.seq > last:
                    yield item
                    last = item.seq
        finally:
            self._subs.remove(q)

    def __aiter__(self) -> AsyncIterator[StepEvent]:
        return self.events()

    # ---- control side ------------------------------------------------

    def stop(self, reason: str = "user") -> None:
        """Request a graceful stop. Idempotent. Sync — call it anywhere."""
        if not self._finished and not self._stop.is_set():
            self._stop_reason = reason
            self._stop.set()

    def send(self, input: "str | Message") -> None:
        """Queue a user message mid-run; the loop picks it up before its
        next LLM step. Ignored after the run finishes."""
        if not self._finished:
            self._inbox.append(Message(role="user", content=input)
                               if isinstance(input, str) else input)

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    @property
    def status(self) -> RunStatus:
        if not self._finished:
            return "running"
        return self._result_fut.result().status

    async def result(self) -> RunResult:
        return await asyncio.shield(self._result_fut)

    def __await__(self):
        """``await handle`` == ``await handle.result()``."""
        return self.result().__await__()

    # ---- lifecycle (loop-internal) ------------------------------------

    def _finish(self, result: RunResult) -> None:
        self._finished = True
        if not self._result_fut.done():
            self._result_fut.set_result(result)
        for q in list(self._subs):
            q.put_nowait(_DONE)
