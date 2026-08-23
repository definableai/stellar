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

Self-check: uv run python -m internal.worker
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from core import Agent, Message, RunHandle, Session

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


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from core import LLMReply

    class EchoLLM:
        async def stream(self, messages, tools, **params):
            yield LLMReply(message=Message(
                role="assistant", content=f"saw {len(messages)}"))

    class SlowLLM:
        async def stream(self, messages, tools, **params):
            await asyncio.sleep(0.05)
            yield LLMReply(message=Message(role="assistant", content="slow done"))

    async def _selfcheck() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "w.jsonl"

            # two turns, then resume with full history
            w = Worker(Agent(EchoLLM()), Session(path))
            task = asyncio.create_task(w.serve())
            w.send("one")
            w.send("two")
            await w.idle()        # both turns actually run before shutdown
            await w.close()
            await task
            w.session.close()
            logged = Session.load(path).messages()
            assert [m.role for m in logged] == ["user", "assistant"] * 2
            assert logged[1].content == "saw 1"
            assert logged[3].content == "saw 3"     # turn 2 saw turn 1's pair

            # steer joins the active turn; its message lands durably
            with Session.load(path) as s2:
                w2 = Worker(Agent(SlowLLM()), s2)
                task2 = asyncio.create_task(w2.serve())
                w2.send("turn")
                await asyncio.sleep(0.01)           # turn is mid-LLM now
                w2.steer("psst")
                await w2.close()
                await task2
            roles = [m.role for m in Session.load(path).messages()]
            assert roles.count("user") == 4 and roles.count("assistant") == 3

            # close() drains never-run inputs into the log, unanswered
            with Session.load(path) as s3:
                w3 = Worker(Agent(EchoLLM()), s3)
                w3.send("late one")
                w3.send("late two")
                await w3.close()
                await w3.serve()                    # processes the drain + _CLOSE
            tail = Session.load(path).messages()[-2:]
            assert [m.content for m in tail] == ["late one", "late two"]

            # closed worker refuses new input
            try:
                w3.send("nope")
                raise AssertionError("send after close must raise")
            except RuntimeError:
                pass

            # steer in the run/end window is requeued, never lost
            path_b = Path(td) / "b.jsonl"

            class BoundarySteer:
                worker: Worker | None = None
                fired = False

                async def on_event(self, e) -> None:
                    if (not self.fired and e.kind.value == "run"
                            and e.phase.value == "end"):
                        self.fired = True
                        assert self.worker is not None
                        self.worker.steer("BOUNDARY")

            tracer = BoundarySteer()
            wb = Worker(Agent(EchoLLM(), tracers=[tracer]), Session(path_b))
            tracer.worker = wb
            tb = asyncio.create_task(wb.serve())
            wb.send("first")
            await wb.idle()
            await wb.close()
            await tb
            wb.session.close()
            contents = [m.content for m in Session.load(path_b).messages()]
            assert "BOUNDARY" in contents      # requeued as its own turn
            assert len(contents) == 4          # two full turns

            # cancelling serve() stops the active turn — no orphan writer
            path_c = Path(td) / "c.jsonl"
            wc = Worker(Agent(SlowLLM()), Session(path_c))
            tc = asyncio.create_task(wc.serve())
            wc.send("doomed")
            await asyncio.sleep(0.01)          # mid-LLM
            tc.cancel()
            try:
                await tc
                raise AssertionError("serve must propagate cancellation")
            except asyncio.CancelledError:
                pass
            assert wc.current is not None and wc.current.status == "stopped"
            wc.session.close()
            assert [m.role for m in Session.load(path_c).messages()] == \
                ["user", "assistant"]

        print("worker self-check ok")

    asyncio.run(_selfcheck())
