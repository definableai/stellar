"""Run machinery: RunContext, RunResult, RunHandle.

``agent.run()`` returns a RunHandle immediately. The handle owns the
event stream (every event buffered with monotonic ``seq``; any number
of consumers attach any time — ``events(after_seq=n)`` replays the past
then follows live), cancellation (``stop()`` is graceful: text/end
partial, tool/end cancelled, run/end stopped still reach every tracer
and subscriber), and completion (``await handle`` -> RunResult).
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

from .events import StepEvent, StepKind, StepPhase
from .types import ErrorInfo, Message, Usage, new_id

RunStatus = Literal["running", "completed", "truncated", "stopped", "error"]

_DONE = object()  # stream sentinel


@dataclass
class RunResult:
    run_id: str
    status: RunStatus
    messages: list[Message]  # final request view + new turns (the log has all)
    output: str | None             # final assistant text (may be partial on stop)
    usage: Usage
    error: ErrorInfo | None = None
    stop_reason: str | None = None  # e.g. "user", "max_steps"


@dataclass
class RunContext:
    """Passed to hooks and tools. ``state`` is user scratch space;
    ``usage`` is the run's cumulative total (updated after each LLM step)."""

    run_id: str
    handle: "RunHandle"
    agent: Any = None   # the running Agent — subagents derive from it
    state: dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    last_usage: Usage = field(default_factory=Usage)  # most recent LLM step
    # (its input_tokens == context size the provider just saw; compaction keys on it)

    @property
    def stop_requested(self) -> bool:
        return self.handle.stop_requested

    def stop(self, reason: str = "context") -> None:
        """Hooks/tools may request a graceful stop."""
        self.handle.stop(reason)


class RunHandle:
    # ponytail: fixed caps as class attrs — override on the class or a
    # subclass. Hours-long runs must not grow memory without bound:
    # events(after_seq=n) replays only what the ring still holds (gaps
    # visible via seq); a slow subscriber sheds its OLDEST queued events,
    # run/end + done sentinel always arrive (needs max_queue >= 2).
    max_buffer = 10_000     # replay ring (events)
    max_queue = 1_000       # per-subscriber queue (events)

    def __init__(self, run_id: str | None, tracers: list[Any]) -> None:
        self.run_id = run_id or new_id("run")
        self._tracers = tracers
        self._buffer: deque[StepEvent] = deque(maxlen=self.max_buffer)
        self._subs: list[asyncio.Queue] = []
        self._seq = 0
        self._stop = asyncio.Event()
        self._stop_reason: str | None = None
        self._result_fut: asyncio.Future[RunResult] = (
            asyncio.get_running_loop().create_future()
        )
        self._finished = False
        self._inbox: list[Message] = []         # mid-run steering queue
        self._inbox_closed = False              # set past the final drain
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
            self._offer(q, event)
        for tracer in self._tracers:
            try:
                await tracer.on_event(event)
            except Exception:
                pass  # tracer isolation: observability never kills the run
        return event

    async def events(self, after_seq: int = -1) -> AsyncIterator[StepEvent]:
        """Replay buffered events past ``after_seq``, then follow live.
        Safe for many consumers, any time, incl. after finish (pure replay)."""
        q: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue)
        self._subs.append(q)
        try:
            last = after_seq
            for event in list(self._buffer):
                if event.seq > last:
                    yield event
                    last = event.seq
            while True:
                # the run may have finished while we replayed the snapshot:
                # drain the queued live tail (incl. run/end) before leaving
                if self._finished and q.empty():
                    return
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

    def send(self, input: "str | Message") -> bool:
        """Queue a user message mid-run; the loop picks it up before its
        next LLM step (or its final drain). False = NOT accepted (run
        finished / past the drain) — requeue elsewhere, never drop."""
        if self._finished or self._inbox_closed:
            return False
        self._inbox.append(Message(role="user", content=input)
                           if isinstance(input, str) else input)
        return True

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    async def wait_stop(self) -> None:
        """Await until a stop is requested (hooks race this vs. user input)."""
        await self._stop.wait()

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
            self._offer(q, _DONE)

    @staticmethod
    def _offer(q: asyncio.Queue, item: Any) -> None:
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:      # slow consumer: shed its oldest event
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            q.put_nowait(item)         # space just freed; single-threaded
