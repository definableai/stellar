"""Worker: the long-lived agent. One agent + one durable session; each
input gets its own turn (turn == run), processed strictly one at a time.
Idle is just ``serve()`` parked on an empty queue — hours are fine.

    worker = Worker(agent, Session("w.jsonl"))
    task = asyncio.create_task(worker.serve())
    worker.send("do the thing")            # queues a turn
    worker.steer("also consider X")        # joins the active turn, else queues
    ...
    await worker.close()                   # finish current turn, drain, exit
    await task

Crash recovery is the whole point of the session:

    worker = Worker(agent, Session.load("w.jsonl"))   # repairs, continues

Stream the active turn via ``worker.current.events()``. Inputs queued
but never run at close() are appended to the session unanswered — the
next resume's turn sees them. ponytail: single asyncio loop, no
threads; one Worker per session file (the session's flock enforces it).

``run_kwargs`` are reused for every turn: a seeded ``state=`` dict is
shared by identity, i.e. worker-lifetime scratch space, not per-turn.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from .agent import Agent
from .run import RunHandle
from .session import Session
from .types import Message

_CLOSE = object()


class Worker:
    def __init__(self, agent: Agent, session: Session, **run_kwargs: Any):
        self.agent = agent
        self.session = session
        self.run_kwargs = run_kwargs      # per-turn run() extras (params=...)
        self.current: RunHandle | None = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._closed = False

    def send(self, input: str | Message) -> None:
        """Queue input for its own turn. Raises after close()."""
        if self._closed:
            raise RuntimeError("worker closed")
        self._queue.put_nowait(input)

    def steer(self, input: str | Message) -> None:
        """Inject into the active turn; falls back to a queued turn.
        send() reports acceptance, so a steer landing in the turn's
        final-drain window requeues instead of vanishing."""
        cur = self.current
        if cur is None or cur.status != "running" or not cur.send(input):
            self.send(input)

    def stop_turn(self, reason: str = "user") -> None:
        """Gracefully stop the active turn (the worker keeps serving)."""
        if self.current is not None:
            self.current.stop(reason)

    async def serve(self) -> None:
        """Process turns until close(). Run errors land in each turn's
        RunResult and never kill the loop; agent.run() itself raising
        (closed session, bad input) is fatal and propagates."""
        while True:
            item = await self._queue.get()
            try:
                if item is _CLOSE:
                    return
                if self._closed:  # closing: keep it durable, skip the turn
                    self.session.append(Message(role="user", content=item)
                                        if isinstance(item, str) else item)
                    continue
                self.current = self.agent.run(item, session=self.session,
                                              **self.run_kwargs)
                try:
                    await self.current
                except asyncio.CancelledError:
                    # a killed worker must not leave an orphan run writing
                    # to the session: stop it and let it close out first
                    self.current.stop("worker cancelled")
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(self.current.result())
                    raise
            finally:
                self._queue.task_done()

    async def idle(self) -> None:
        """Await until every queued turn has been processed."""
        await self._queue.join()

    async def close(self, *, stop_current: bool = False) -> None:
        """No new turns; the active turn finishes (or is stopped), then
        pending inputs are drained into the session and serve() returns."""
        self._closed = True
        if stop_current:
            self.stop_turn("worker closed")
        self._queue.put_nowait(_CLOSE)
