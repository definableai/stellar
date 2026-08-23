"""The hours-long agent: one Worker, one durable session file.

The five things a long-lived agent has to get right, told as a story:

    send()       one input, one turn — queued, run strictly one at a time
    steer()      an input that joins the turn *already running*
    stop_turn()  killing a runaway turn without killing the worker
    close()      inputs queued but never run still land in the log
    resume       process dies -> Worker(agent, Session.load(path)) continues

Fake LLMs (scripted, parked, never-ending): no keys, no network. Each
phase asserts on what a fresh reader finds in the session file — the
only thing that survives between phases, which is the whole point.

    uv run python -m examples.long_lived_worker
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
from pathlib import Path

from core import Agent, LLMDelta, LLMReply, Message, Session, ToolCall, tool
from internal.worker import Worker


class ScriptedLLM:
    """One step, reporting how much transcript it was handed — which is how
    the asserts prove history survived a restart."""
    async def stream(self, messages, tools, **params):
        yield LLMReply(message=Message(role="assistant",
                                       content=f"ok (saw {len(messages)})"))


class ParkedLLM:
    """Two steps around a tool that parks: a real window to steer into.
    ponytail: steering is drained at the top of the *next* step, so a turn
    only ever hears it if it has one — hence the tool call."""
    async def stream(self, messages, tools, **params):
        if not any(m.role == "tool" for m in messages):
            yield LLMReply(message=Message(role="assistant", tool_calls=[
                ToolCall(id="c1", name="work")]))
            return
        heard = [m.content for m in messages if m.role == "user"][-1]
        yield LLMReply(message=Message(role="assistant", content=f"heard {heard}"))


class NeverEndingLLM:
    """Streams until told to stop. The loop checks between chunks, so
    stop_turn() lands within one chunk and the partial text is persisted;
    ``runaway = False`` makes the next turn behave."""
    def __init__(self) -> None:
        self.started, self.runaway = asyncio.Event(), True

    async def stream(self, messages, tools, **params):
        while self.runaway:
            yield LLMDelta(text="thinking... ")
            self.started.set()
            await asyncio.sleep(0.005)   # streaming pace — and yields the loop
        yield LLMReply(message=Message(role="assistant", content="dropped it"))


in_tool, steered = asyncio.Event(), asyncio.Event()


@tool()
async def work(ctx):
    """Slow work — parks here so the demo can steer a running turn."""
    in_tool.set()
    await steered.wait()
    return "worked"


@contextlib.asynccontextmanager
async def serving(agent: Agent, session: Session):
    """Host a worker for the block, then shut it down cleanly. The raw
    form is the four awaits below — see the crash phase for it in the open."""
    w = Worker(agent, session)
    task = asyncio.create_task(w.serve())
    try:
        yield w
    finally:
        await w.idle()       # every queued turn has run
        await w.close()      # no new turns, drain, serve() returns
        await task
        session.close()


def log(path: Path) -> list[Message]:
    """What a fresh process reads off disk (and only one may, at a time)."""
    with Session.load(path) as s:
        return s.messages()


async def main() -> None:
    path = Path(tempfile.mkdtemp()) / "worker.jsonl"
    print(f"session: {path}")

    # -- one input, one turn, one at a time --------------------------------
    print("turn 1, turn 2...")
    async with serving(Agent(ScriptedLLM()), Session(path)) as w:
        w.send("summarize the inbox")
        w.send("now draft a reply")
    msgs = log(path)
    assert [m.role for m in msgs] == ["user", "assistant"] * 2
    assert msgs[3].content == "ok (saw 3)"         # turn 2 saw turn 1's pair

    # -- steering joins the turn already in flight -------------------------
    print("steering...")
    async with serving(Agent(ParkedLLM(), tools=[work]), Session.load(path)) as w:
        w.send("draft the release notes")
        await in_tool.wait()             # the turn is parked mid-tool now
        w.steer("...and mention the deadline")
        steered.set()
    msgs = log(path)
    assert [m.role for m in msgs[-5:]] == [        # the steer landed *inside*
        "user", "assistant", "tool", "user", "assistant"]   # the running turn
    assert msgs[-1].content == "heard ...and mention the deadline"

    # -- stop_turn kills the turn, not the worker --------------------------
    print("stopping a runaway turn...")
    llm = NeverEndingLLM()
    async with serving(Agent(llm), Session.load(path)) as w:
        w.send("think about it forever")
        await llm.started.wait()
        w.stop_turn("enough")
        await w.idle()
        assert w.current.status == "stopped"
        llm.runaway = False
        w.send("ok, drop it")            # same worker, still serving
    msgs = log(path)                     # partial reply persisted, then turn 2
    assert msgs[-3].meta["interrupted"] and msgs[-1].content == "dropped it"

    # -- close() drains what never ran, durably ----------------------------
    print("closing with work still queued...")
    with Session.load(path) as s:
        w = Worker(Agent(ScriptedLLM()), s)
        w.send("check the deploy")       # nothing is serving it yet
        await w.close()
        await w.serve()                  # drains the queue, then exits
    orphan = log(path)[-1]
    assert orphan.role == "user" and orphan.content == "check the deploy"

    # -- crash: the process dies mid-turn ----------------------------------
    print("crashing mid-turn...")
    llm, s = NeverEndingLLM(), Session.load(path)
    w = Worker(Agent(llm), s)
    task = asyncio.create_task(w.serve())
    w.send("a turn nobody will finish")
    await llm.started.wait()
    task.cancel()                        # kill -9, near enough
    with contextlib.suppress(asyncio.CancelledError):
        await task
    s.close()   # ponytail: the OS does this on process death, flock is per-process

    # -- resume: same file, new worker, nothing lost -----------------------
    print("resuming...")
    async with serving(Agent(ScriptedLLM()), Session.load(path)) as w:
        w.send("where were we?")
    msgs = log(path)
    assert msgs[-1].content == f"ok (saw {len(msgs) - 1})"   # the whole history
    assert "check the deploy" in [m.content for m in msgs]   # orphan still there

    print("long lived worker demo ok")


if __name__ == "__main__":
    asyncio.run(main())
