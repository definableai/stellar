"""The data plane: what happened, in order, for anyone who wants to watch.

emit() is sync and never fails — a listener that raises is logged, and the
next one still hears the event. Async consumers take stream() instead: the
log replayed from wherever they left off, then live, no gap and no doubles.

Names are free-form, but the loop emits the eight stage names, stamped with
the run's id; drivers/ turns this bus into SSE or WebSocket frames.
"""

import asyncio
import inspect
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from core.contracts import ContractError

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """One thing that happened: numbered, named, stamped."""

    seq: int                             # per-bus, monotonic, starts at 0
    name: str
    data: Any = None
    source: str | None = None            # who said it: "loop", a tool name, you
    run_id: str | None = None
    ts: float = field(default_factory=time.time)


def _hears(name: str, prefix: str) -> bool:
    """A prefix stops at a dot: "tool" hears "tool.pre", never "toolbox.x"."""
    return not prefix or name == prefix or name.startswith(prefix + ".")


class Events:
    """Emit into it, listen to it, stream it. Emitting never blocks."""

    def __init__(self) -> None:
        self.seq = 0
        self.log: list[Event] = []       # ponytail: unbounded, cap it later
        # one listener is (fn, prefix it hears, the one run_id it wants or None)
        self.listeners: list[tuple[Callable[[Event], Any], str, str | None]] = []

    def emit(self, name: str, data: Any = None, source: str | None = None,
             run_id: str | None = None) -> Event:
        """Stamp it, keep it, tell everyone. A listener's sins stay its own."""
        event = Event(self.seq, name, data, source, run_id, time.time())
        self.seq += 1
        self.log.append(event)
        for fn, prefix, wanted in list(self.listeners):   # one may attach mid-emit
            if wanted in (None, run_id) and _hears(name, prefix):
                try:
                    fn(event)
                except Exception:
                    logger.exception("listener %r broke on %s", fn, name)
        return event

    def listen(self, fn: Callable[[Event], Any], prefix: str = "",
               run_id: str | None = None) -> Callable[[Event], Any]:
        """Hear every matching event, synchronously. Hands fn back, so it decorates.

        prefix "" hears everything, "tool" hears tool.*; run_id pins one run.
        An async def is refused: emit() never awaits.
        """
        if inspect.iscoroutinefunction(fn):
            raise ContractError(
                "listeners are sync — emit() never awaits. Write def, not "
                "async def; an async consumer takes stream() instead")
        self.listeners.append((fn, prefix, run_id))
        return fn

    def detach(self, fn: Callable[[Event], Any]) -> None:
        """Stop telling fn anything. Never was listening? Then nothing happens."""
        self.listeners = [heard for heard in self.listeners if heard[0] != fn]

    async def stream(self, run_id: str | None = None, prefix: str = "",
                     since: int = 0) -> AsyncIterator[Event]:
        """The log from `since`, then live — no gap, no doubles, in seq order.

        `since` trims the replay, never the live tail. The live listener goes
        on before the log is read, so an event landing mid-replay waits in the
        queue instead of falling between the two — and it hears the whole run,
        so a run_id stream ends on that run's run.post, prefix or no prefix.
        """
        queue: asyncio.Queue = asyncio.Queue()    # ponytail: unbounded, a slow
        push = queue.put_nowait                   # consumer is memory
        self.listen(push, run_id=run_id)          # live first, then the log
        old = deque(event for event in self.log if event.seq >= since)
        seen = -1                                 # the highest seq handed over
        try:
            while True:
                event = old.popleft() if old else await queue.get()
                if event.seq <= seen or run_id not in (None, event.run_id):
                    continue                      # replayed already, or not ours
                seen = event.seq
                if _hears(event.name, prefix):
                    yield event
                if run_id is not None and event.name == "run.post":
                    return                        # one run, one stream
        finally:
            self.detach(push)
