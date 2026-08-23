"""Agent loop self-checks with a fake LLM: session wiring, resume,
truncation, stop-mid-stream.

Run: uv run python tests/test_agent.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (  # noqa: E402
    Agent, LLMDelta, LLMReply, Message, RunHandle, Session, StepKind,
    StepPhase, ToolCall, tool,
)


@tool(parameters={"type": "object", "properties": {"x": {"type": "string"}}})
def echo(ctx, x: str = "") -> str:
    """Echo back."""
    return f"echo:{x}"


class ScriptedLLM:
    """Plays back canned replies; records every request's messages."""

    def __init__(self, replies: list[LLMReply]) -> None:
        self.replies = list(replies)
        self.seen: list[list[Message]] = []

    async def stream(self, messages, tools, **params):
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        if reply.message.content:
            yield LLMDelta(text=reply.message.content)
        yield reply


class NeverEndingLLM:
    """Streams deltas forever; only a stop() ends the step."""

    async def stream(self, messages, tools, **params):
        i = 0
        while True:
            yield LLMDelta(text=f"chunk{i} ")
            i += 1
            await asyncio.sleep(0)


def tool_reply(call_id: str, x: str) -> LLMReply:
    return LLMReply(
        message=Message(role="assistant",
                        tool_calls=[ToolCall(call_id, "echo", {"x": x})]),
        stop_reason="tool_use")


def text_reply(text: str) -> LLMReply:
    return LLMReply(message=Message(role="assistant", content=text))


async def test_session_run_and_resume(dir: Path) -> None:
    path = dir / "s.jsonl"
    llm = ScriptedLLM([tool_reply("c1", "hi"), text_reply("done")])
    agent = Agent(llm, tools=[echo], system="be brief")

    with Session(path) as s:
        result = await agent.run("go", session=s)
    assert result.status == "completed"
    assert result.output == "done"

    # durable transcript: user, assistant+call, tool result, assistant
    logged = Session.load(path).messages()
    assert [m.role for m in logged] == ["user", "assistant", "tool", "assistant"]
    assert logged[2].tool_result.content == "echo:hi"
    assert all(m.role != "system" for m in logged)  # system is config, not log

    # request derived: system first, then the log
    assert llm.seen[0][0].role == "system"
    assert llm.seen[0][1].content == "go"

    # resume: second turn sees full history
    llm2 = ScriptedLLM([text_reply("again")])
    agent2 = Agent(llm2, tools=[echo], system="be brief")
    with Session.load(path) as s2:
        result2 = await agent2.run("more", session=s2)
    assert result2.status == "completed"
    assert [m.role for m in llm2.seen[0]] == [
        "system", "user", "assistant", "tool", "assistant", "user"]
    assert len(Session.load(path)) == 6  # + "more", + "again"


async def test_session_and_history_exclusive() -> None:
    agent = Agent(ScriptedLLM([text_reply("x")]))
    try:
        agent.run("hi", session=Session(), history=[])
        raise AssertionError("session+history must raise")
    except ValueError:
        pass


async def test_truncated() -> None:
    replies = [tool_reply(f"c{i}", str(i)) for i in range(9)]
    agent = Agent(ScriptedLLM(replies), tools=[echo], max_steps=2)
    s = Session()
    result = await agent.run("go", session=s)
    assert result.status == "truncated"
    assert result.stop_reason == "max_steps"
    assert len(s) == 1 + 2 * 2  # input + 2x (assistant + tool result)


async def test_stop_mid_stream(dir: Path) -> None:
    path = dir / "stop.jsonl"
    agent = Agent(NeverEndingLLM())
    with Session(path) as s:
        handle = agent.run("go", session=s)
        async for e in handle.events():
            if e.kind is StepKind.TEXT and e.phase is StepPhase.DELTA:
                handle.stop("test")
                break
        result = await handle
    assert result.status == "stopped"
    assert result.stop_reason == "test"

    last = Session.load(path).messages()[-1]
    assert last.role == "assistant"
    assert last.meta.get("interrupted") is True
    assert last.content and last.content.startswith("chunk0")


async def test_parallel_hook_failure_settles_siblings() -> None:
    done: list[str] = []

    @tool(parameters={"type": "object", "properties": {}})
    async def slow(ctx) -> str:
        """Slow tool."""
        await asyncio.sleep(0.02)
        done.append("slow")
        return "slow-ok"

    def bad_hook(hctx) -> None:
        if hctx.call.name == "echo":
            raise RuntimeError("boom")

    reply = LLMReply(message=Message(role="assistant", tool_calls=[
        ToolCall("c1", "echo", {"x": "a"}), ToolCall("c2", "slow", {})]),
        stop_reason="tool_use")
    agent = Agent(ScriptedLLM([reply]), tools=[echo, slow],
                  parallel_tools=True, hooks={"after_tool": [bad_hook]})
    s = Session()
    result = await agent.run("go", session=s)
    assert result.status == "error"
    assert result.error and result.error.message == "boom"
    assert done == ["slow"]  # sibling settled, not orphaned
    results = [m.tool_result.content for m in s.messages() if m.role == "tool"]
    assert results == ["slow-ok"]  # sibling's real result stays in the log


async def test_failing_late_append_never_hangs(dir: Path) -> None:
    @tool(parameters={"type": "object", "properties": {}})
    def sneaky(ctx) -> str:
        """Sends an unserializable late message."""
        ctx.run.handle.send(Message(role="user", content=[{"bad": object()}]))
        return "ok"

    reply = LLMReply(message=Message(
        role="assistant", tool_calls=[ToolCall("c1", "sneaky", {})]),
        stop_reason="tool_use")
    agent = Agent(ScriptedLLM([reply]), tools=[sneaky], max_steps=1)
    s = Session(dir / "hang.jsonl")  # file-backed: appends actually serialize
    # post-loop inbox drain fails to serialize; handle must still finish
    result = await asyncio.wait_for(agent.run("go", session=s), timeout=2)
    assert result.status == "error"
    assert result.error is not None


async def test_input_batch_atomic() -> None:
    agent = Agent(ScriptedLLM([text_reply("x")]))
    s = Session()
    try:
        agent.run([Message(role="user", content="good"),
                   Message(role="user", content=[{"bad": object()}])], session=s)
        raise AssertionError("bad input batch must raise")
    except TypeError:
        pass
    assert len(s) == 0  # no orphan user turn in the log


async def test_before_llm_rebind_honored() -> None:
    def compact(hctx) -> None:
        hctx.messages = [m for m in hctx.messages if m.role != "user"] + [
            Message(role="user", content="[summary]")]

    llm = ScriptedLLM([text_reply("ok")])
    agent = Agent(llm, system="sys", hooks={"before_llm": [compact]})
    result = await agent.run("original")
    assert result.status == "completed"
    seen = [(m.role, m.content) for m in llm.seen[0]]
    assert ("user", "original") not in seen  # rebound view reached the LLM
    assert ("user", "[summary]") in seen


async def test_partial_survives_tiny_buffer() -> None:
    old = RunHandle.max_buffer
    RunHandle.max_buffer = 8   # ring far smaller than the streamed deltas
    try:
        agent = Agent(NeverEndingLLM())
        handle = agent.run("go")
        n = 0
        async for e in handle.events():
            if e.kind is StepKind.TEXT and e.phase is StepPhase.DELTA:
                n += 1
                if n >= 20:
                    handle.stop("test")
                    break
        result = await handle
    finally:
        RunHandle.max_buffer = old
    assert result.status == "stopped"
    # synthesis no longer reads the (capped) buffer: full partial text kept
    assert result.output.startswith("chunk0 ")
    assert "chunk19 " in result.output


async def test_slow_subscriber_sheds_old_keeps_end() -> None:
    old = RunHandle.max_queue
    RunHandle.max_queue = 4
    try:
        agent = Agent(NeverEndingLLM())
        handle = agent.run("go")
        gen = handle.events()
        events = [await anext(gen)]     # attach, then stall while it floods
        for _ in range(200):
            await asyncio.sleep(0)
        handle.stop("test")
        await handle
        async for e in gen:             # drain what survived
            events.append(e)
    finally:
        RunHandle.max_queue = old
    seqs = [e.seq for e in events]
    assert seqs != list(range(seqs[0], seqs[-1] + 1))   # old events shed
    assert any(e.kind is StepKind.RUN and e.phase is StepPhase.END
               for e in events)                          # run/end survived


async def test_reconnect_drains_tail() -> None:
    agent = Agent(ScriptedLLM([text_reply("hello world")]))
    handle = agent.run("go")
    while handle._seq < 3:          # let events accumulate in the ring
        await asyncio.sleep(0)
    gen = handle.events()           # attach mid-run: non-empty snapshot
    events = [await anext(gen)]     # start replaying...
    await handle                    # ...run finishes while we're paused
    async for e in gen:             # must drain the live tail, not drop it
        events.append(e)
    assert any(e.kind is StepKind.RUN and e.phase is StepPhase.END
               for e in events)
    seqs = [e.seq for e in events]
    assert seqs == sorted(set(seqs))


async def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        await test_session_run_and_resume(d)
        await test_session_and_history_exclusive()
        await test_truncated()
        await test_stop_mid_stream(d)
        await test_parallel_hook_failure_settles_siblings()
        await test_failing_late_append_never_hangs(d)
        await test_input_batch_atomic()
        await test_before_llm_rebind_honored()
        await test_partial_survives_tiny_buffer()
        await test_slow_subscriber_sheds_old_keeps_end()
        await test_reconnect_drains_tail()
    print("test_agent: all ok")


if __name__ == "__main__":
    asyncio.run(main())
