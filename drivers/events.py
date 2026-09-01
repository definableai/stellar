"""The event stream: one run's bells, fanned out to whoever is listening.

The eight event names ARE the eight hook names — the grammar is the hook
vocabulary, nothing more. One stream is one run: the run_post event closes
it, and every subscriber runs out of events.
"""

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

BUFFER = 1000            # events kept for replay, and per subscriber
DONE = object()          # the sentinel that ends a subscriber


@dataclass
class StepEvent:
    """One bell: which one it was, when, in what order, with what on it."""

    event: str
    seq: int
    ts: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)


def offer(queue, item) -> None:
    """Room or not, the newest event gets in — the oldest is the one that goes."""
    if queue.full():
        queue.get_nowait()
    queue.put_nowait(item)


class EventStream:
    """Emit into it, subscribe to follow along. Emitting never blocks."""

    def __init__(self) -> None:
        self.seq = 0
        self.past: deque[StepEvent] = deque(maxlen=BUFFER)
        self.queues: list[asyncio.Queue] = []
        self.closed = False

    def emit(self, event: str, payload: dict) -> None:
        """Stamp one event, remember it, hand it to everyone listening."""
        self.seq += 1
        stamped = StepEvent(event, self.seq, payload=payload)
        self.past.append(stamped)
        for queue in self.queues:
            offer(queue, stamped)
        if event == "run_post":              # one stream, one run
            self.close()

    def close(self) -> None:
        """The stream is over: every subscriber reaches the end of its events."""
        self.closed = True
        for queue in self.queues:
            offer(queue, DONE)

    async def subscribe(self) -> AsyncIterator[StepEvent]:
        """Everything the buffer still holds, then whatever happens next."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=BUFFER)
        self.queues.append(queue)            # before the snapshot: no event twice
        try:
            for event in list(self.past):
                yield event
            while not (self.closed and queue.empty()):
                event = await queue.get()
                if event is DONE:
                    return
                yield event
        finally:
            self.queues.remove(queue)
