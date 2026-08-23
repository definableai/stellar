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
    Agent, LLMDelta, LLMReply, Message, ReplyBuilder, RunHandle, Session,
    StepKind, StepPhase, ToolCall, file_block, hook, image_block, text_block,
    tool, validate_args,
)
from internal.subagent import as_tool  # noqa: E402


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


def sub_call(call_id: str, name: str, prompt: str) -> LLMReply:
    return LLMReply(
        message=Message(role="assistant",
                        tool_calls=[ToolCall(call_id, name, {"prompt": prompt})]),
        stop_reason="tool_use")


async def test_subagent_forwarding() -> None:
    child = Agent(ScriptedLLM([text_reply("leaf answer")]))
    parent = Agent(
        ScriptedLLM([sub_call("c1", "helper", "dig"), text_reply("synth")]),
        tools=[as_tool(child, name="helper", description="delegate")])
    handle = parent.run("go")
    events = [e async for e in handle.events()]
    result = await handle
    assert result.status == "completed" and result.output == "synth"
    # child's stream is visible through the parent handle
    child_kinds = [e.payload["child"]["kind"] for e in events
                   if e.kind is StepKind.TOOL and e.phase is StepPhase.DELTA
                   and "child" in e.payload]
    assert "text" in child_kinds and "run" in child_kinds
    # the child's final text became the tool result
    tool_end = next(e for e in events
                    if e.kind is StepKind.TOOL and e.phase is StepPhase.END)
    assert tool_end.payload["result"] == "leaf answer"


async def test_subagent_depth_limit() -> None:
    grand = Agent(ScriptedLLM([text_reply("leaf")]))
    mid = Agent(
        ScriptedLLM([sub_call("m1", "sub2", "deeper"), text_reply("mid done")]),
        tools=[as_tool(grand, name="sub2", description="d", max_depth=1)])
    parent = Agent(
        ScriptedLLM([sub_call("p1", "sub1", "deep"), text_reply("parent done")]),
        tools=[as_tool(mid, name="sub1", description="d", max_depth=1)])
    handle = parent.run("go")
    events = [e async for e in handle.events()]
    result = await handle
    assert result.status == "completed" and result.output == "parent done"
    # mid's attempt to go deeper failed as a readable tool error
    nested_ends = [e.payload["child"] for e in events
                   if e.kind is StepKind.TOOL and e.phase is StepPhase.DELTA
                   and e.payload.get("child", {}).get("kind") == "tool"
                   and e.payload["child"]["phase"] == "end"]
    assert any(c["payload"]["is_error"] and "depth limit" in str(c["payload"]["result"])
               for c in nested_ends)


async def test_subagent_stop_propagates() -> None:
    child = Agent(NeverEndingLLM())
    parent = Agent(ScriptedLLM([sub_call("c1", "sub", "spin")]),
                   tools=[as_tool(child, name="sub", description="d")])
    handle = parent.run("go")
    async for e in handle.events():
        if (e.kind is StepKind.TOOL and e.phase is StepPhase.DELTA
                and e.payload.get("child", {}).get("kind") == "text"
                and e.payload["child"]["phase"] == "delta"):
            handle.stop("user")   # parent stop mid-child-stream
            break
    result = await asyncio.wait_for(handle.result(), timeout=2)  # no hang
    assert result.status == "stopped"


async def test_send_after_finish_rejected() -> None:
    agent = Agent(ScriptedLLM([text_reply("x")]))
    handle = agent.run("go")
    await handle
    assert handle.send("late") is False   # caller knows to requeue


async def test_state_shared_by_identity() -> None:
    @tool(parameters={"type": "object", "properties": {}})
    def bump(ctx) -> int:
        """Count runs via shared state."""
        ctx.run.state["n"] = ctx.run.state.get("n", 0) + 1
        return ctx.run.state["n"]

    shared: dict = {}                      # empty dict must still be shared
    for _ in range(2):
        agent = Agent(ScriptedLLM([
            LLMReply(message=Message(role="assistant",
                                     tool_calls=[ToolCall("c", "bump", {})]),
                     stop_reason="tool_use"),
            text_reply("done")]), tools=[bump])
        await agent.run("go", state=shared)
    assert shared["n"] == 2


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


def test_replybuilder() -> None:
    b = ReplyBuilder()
    assert b.text("Hel").text == "Hel"
    b.text("lo")
    b.tool_call(0, id="c1", name="ad")
    b.tool_call(0, name="d")                  # name fragments concatenate
    b.tool_args(0, '{"a": ')
    b.tool_args(0, "2}")
    b.tool_call(1, name="bad")
    b.tool_args(1, '{"x": tru')               # truncated JSON
    b.usage(input_tokens=7, output_tokens=3)
    b.finish("tool_use")
    r = b.reply()
    assert r.message.content == "Hello"
    assert r.message.tool_calls[0] == ToolCall("c1", "add", {"a": 2})
    bad = r.message.tool_calls[1]
    assert bad.arguments == {} and bad.name == "bad"
    assert r.message.meta["invalid_tool_args"] == {bad.id: '{"x": tru'}
    assert (r.usage.input_tokens, r.usage.output_tokens) == (7, 3)
    assert r.stop_reason == "tool_use"


def test_validate_args() -> None:
    schema = {"type": "object",
              "properties": {"path": {"type": "string"},
                             "limit": {"type": "integer"},
                             "mode": {"enum": ["r", "w"]},
                             "tags": {"type": "array",
                                      "items": {"type": "string"}}},
              "required": ["path"]}
    assert validate_args(schema, {"path": "/x", "limit": 3, "mode": "r",
                                  "tags": ["a"]}) == []
    problems = validate_args(schema, {"limit": True, "mode": "x",
                                      "tags": ["a", 1], "junk": 0})
    text = "; ".join(problems)
    assert "missing required argument: 'path'" in text
    assert "limit: expected integer, got bool" in text
    assert "mode: expected one of" in text
    assert "tags: [1]: expected string" in text
    assert "unknown argument: 'junk'" in text


def test_blocks() -> None:
    assert text_block("hi") == {"type": "text", "text": "hi"}
    assert image_block(url="http://x/i.png") == {"type": "image",
                                                 "url": "http://x/i.png"}
    assert image_block(data="QUJD")["data"] == "QUJD"
    f = file_block(data="QUJD", name="doc.pdf")
    assert f["type"] == "file" and f["media_type"] == "application/pdf"
    assert f["name"] == "doc.pdf"


async def test_invalid_args_reach_model_readably() -> None:
    @tool(parameters={"type": "object",
                      "properties": {"path": {"type": "string"}},
                      "required": ["path"]})
    def read(ctx, path: str = "") -> str:
        """Read."""
        raise AssertionError("handler must not run on invalid args")

    # bad-typed args from the model
    r1 = await Agent(ScriptedLLM([
        LLMReply(message=Message(role="assistant", tool_calls=[
            ToolCall("c1", "read", {"path": 7})]), stop_reason="tool_use"),
        text_reply("done")]), tools=[read]).run("go")
    tr = next(m.tool_result for m in r1.messages if m.role == "tool")
    assert tr.is_error and "expected string" in tr.content

    # adapter-flagged malformed JSON (ReplyBuilder policy)
    r2 = await Agent(ScriptedLLM([
        LLMReply(message=Message(role="assistant", tool_calls=[
            ToolCall("c2", "read", {})],
            meta={"invalid_tool_args": {"c2": '{"path": "/tm'}}),
            stop_reason="tool_use"),
        text_reply("done")]), tools=[read]).run("go")
    tr2 = next(m.tool_result for m in r2.messages if m.role == "tool")
    assert tr2.is_error and "Malformed" in tr2.content


async def test_hook_decorator_dx() -> None:
    @hook("before_llm")
    def inject(ctx) -> None:
        ctx.messages.append(Message(role="user", content="[injected]"))

    llm = ScriptedLLM([text_reply("ok")])
    result = await Agent(llm, hooks=[inject]).run("go")
    assert result.status == "completed"
    assert llm.seen[0][-1].content == "[injected]"


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
        await test_subagent_forwarding()
        await test_subagent_depth_limit()
        await test_subagent_stop_propagates()
        await test_send_after_finish_rejected()
        await test_state_shared_by_identity()
        test_replybuilder()
        test_validate_args()
        test_blocks()
        await test_invalid_args_reach_model_readably()
        await test_hook_decorator_dx()
    print("test_agent: all ok")


if __name__ == "__main__":
    asyncio.run(main())
