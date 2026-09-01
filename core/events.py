"""The data plane: what happened, in order, for anyone who wants to watch.

emit() is sync and never fails — a listener that raises is logged, and the
next one still hears the event. Async consumers take stream() instead: the
log replayed from wherever they left off, then live, no gap and no doubles.

Names are free-form, but the loop emits the eight stage names, stamped with
the run's id; drivers/ turns this bus into SSE or WebSocket frames.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Annotated, Any, AsyncIterator, Callable

from core.contracts import ContractError

__all__ = ["Event", "Events"]

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """One thing that happened: numbered, named, stamped."""

    seq: Annotated[int, "per-bus, monotonic, starts at 0"]
    name: Annotated[str, "the dotted name — the loop's eight, or anything you emit"]
    data: Annotated[Any, "whatever rode along; stored as-is"] = None
    source: Annotated[str | None, "who said it: 'loop', a tool name, you"] = None
    run_id: Annotated[str | None, "the run it belongs to, or None for bus-wide"] = None
    ts: Annotated[
        float, "when it was stamped, as time.time()"] = field(default_factory=time.time)


class Events:
    """Emit into it, listen to it, stream it. Emitting never blocks."""

    def __init__(self) -> None:
        self.seq = 0
        self.log: list[Event] = []       # ponytail: unbounded, cap it later
        # one listener is (fn, prefix it hears, the one run_id it wants or None)
        self.listeners: list[tuple[Callable[[Event], Any], str, str | None]] = []

    def emit(
        self,
        name: Annotated[
            str, "dotted event name — the loop uses the eight, you use anything"],
        data: Annotated[Any, "whatever should ride along; stored as-is"] = None,
        source: Annotated[
            str | None, "who is speaking — 'loop', a tool's name, or yours"] = None,
        run_id: Annotated[
            str | None, "pin the event to one run, or None for bus-wide"] = None,
    ) -> Event:
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

    def listen(
        self,
        fn: Annotated[
            Callable[[Event], Any],
            "any sync callable of one Event; what it returns is dropped"],
        prefix: Annotated[str, "'' hears everything, 'tool' hears tool.*"] = "",
        run_id: Annotated[
            str | None, "the one run it wants, or None to hear the whole bus"] = None,
    ) -> Callable[[Event], Any]:
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

    def detach(
        self,
        fn: Annotated[
            Callable[[Event], Any],
            "matched by equality, and unfiled from every prefix and run it had"],
    ) -> None:
        """Stop telling fn anything. Never was listening? Then nothing happens."""
        self.listeners = [heard for heard in self.listeners if heard[0] != fn]

    async def stream(
        self,
        run_id: Annotated[
            str | None,
            "None watches the whole bus; an id watches that one run only"] = None,
        prefix: Annotated[
            str, "'' hears everything, 'tool' hears tool.* — replay and live"] = "",
        since: Annotated[int, "the first seq worth replaying; 0 is the whole log"] = 0,
    ) -> AsyncIterator[Event]:
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


def _hears(name: str, prefix: str) -> bool:
    """A prefix stops at a dot: "tool" hears "tool.pre", never "toolbox.x"."""
    return not prefix or name == prefix or name.startswith(prefix + ".")
