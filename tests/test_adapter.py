"""Adapter self-checks: mount/unwind, LIFO restore, shadowing, partial
failure, and live self-composition mid-run (the loop late-binds).

Run: uv run python tests/test_adapter.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hypothesis import given, settings, strategies as st  # noqa: E402

from core import (  # noqa: E402
    Agent, Hook, LLMDelta, LLMReply, Message, Tool, ToolCall, ToolSpec, tool,
)


@tool(parameters={"type": "object", "properties": {"x": {"type": "string"}}})
def echo(ctx, x: str = "") -> str:
    """Echo back."""
    return f"echo:{x}"


def named_tool(name: str) -> Tool:
    return Tool(spec=ToolSpec(name=name), handler=lambda ctx: name)


class ScriptedLLM:
    """Canned replies; records the messages AND tool specs of each request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen: list[list[Message]] = []
        self.seen_tools: list[list[str]] = []

    async def stream(self, messages, tools, **params):
        self.seen.append(list(messages))
        self.seen_tools.append([t.name for t in tools])
        reply = self.replies.pop(0)
        if reply.message.content:
            yield LLMDelta(text=reply.message.content)
        yield reply


def call_reply(cid: str, name: str, args: dict) -> LLMReply:
    return LLMReply(message=Message(role="assistant",
                                    tool_calls=[ToolCall(cid, name, args)]),
                    stop_reason="tool_use")


def text_reply(text: str) -> LLMReply:
    return LLMReply(message=Message(role="assistant", content=text))


# ---- mount / drop ------------------------------------------------------

def test_mount_drop_restores_everything() -> None:
    agent = Agent("llm0", tools=[echo], system="s0")
    before = dict(agent.tools)
    order: list[str] = []

    def pack(ctx):
        assert ctx.config == {"k": 1}
        ctx.tool(named_tool("t1"))
        ctx.hook("before_tool", guard)
        ctx.llm("llm1")
        ctx.effect(lambda: order.append("custom"))

    def guard(hctx):  # never fired here
        pass

    scope = agent.use(pack, k=1)
    assert scope.name == "pack" and scope.source is None
    assert "tool:t1" in scope.notes and "llm:str" in scope.notes
    assert any(n.startswith("hook:before_tool:") and n.endswith("guard")
               for n in scope.notes)
    assert "t1" in agent.tools and agent.llm == "llm1"
    assert agent.hooks.get("before_tool") == [guard]

    assert agent.drop("pack") == []
    assert agent.tools == before and agent.llm == "llm0"
    assert agent.hooks.get("before_tool") == [] and order == ["custom"]
    assert agent.adapters == {}


def test_lifo_nesting_and_shadowing() -> None:
    agent = Agent("L0", tools=[named_tool("t")])
    v0 = agent.tools["t"]
    agent.use(lambda c: (c.llm("L1"), c.tool(named_tool("t"))), name="a")
    v1 = agent.tools["t"]
    agent.use(lambda c: (c.llm("L2"), c.tool(named_tool("t"))), name="b")
    assert agent.llm == "L2" and agent.tools["t"] is not v1
    agent.drop("b")
    assert agent.llm == "L1" and agent.tools["t"] is v1   # shadow restored
    agent.drop("a")
    assert agent.llm == "L0" and agent.tools["t"] is v0


def test_partial_failure_mounts_nothing() -> None:
    agent = Agent(None)
    undone: list[str] = []

    def bad(ctx):
        ctx.tool(named_tool("t1"))
        ctx.effect(lambda: undone.append("e1"))
        raise RuntimeError("boom")

    try:
        agent.use(bad)
        raise AssertionError("must raise")
    except RuntimeError:
        pass
    assert "t1" not in agent.tools and undone == ["e1"]
    assert agent.adapters == {}   # nothing mounted


def test_loud_errors() -> None:
    agent = Agent(None)
    agent.use(lambda c: None, name="x")
    try:
        agent.use(lambda c: None, name="x")
        raise AssertionError("duplicate must raise")
    except ValueError:
        pass
    try:
        agent.drop("nope")
        raise AssertionError("unknown drop must raise")
    except KeyError:
        pass
    try:
        agent.hooks.remove("before_tool", lambda c: None)
        raise AssertionError("removing unattached hook must raise")
    except ValueError:
        pass


def test_dispose_collects_inverse_errors() -> None:
    agent = Agent(None)
    ran: list[str] = []

    def pack(ctx):
        ctx.effect(lambda: ran.append("first"))
        ctx.effect(lambda: 1 / 0)          # runs before "first" (LIFO)

    agent.use(pack)
    errors = agent.drop("pack")
    assert [e.type for e in errors] == ["ZeroDivisionError"]
    assert ran == ["first"]               # a failing inverse strands nothing


def test_out_of_order_drop_relinks() -> None:
    # dropping A first must not resurrect A's tool/llm when B drops later:
    # drop(A) unwinds B, drops A, re-runs B's setup on the new base
    agent = Agent("L0", tools=[named_tool("x")])
    v0 = agent.tools["x"]
    agent.use(lambda c: (c.llm("L1"), c.tool(named_tool("x"))), name="a")
    agent.use(lambda c: (c.llm("L2"), c.tool(named_tool("x"))), name="b")
    agent.drop("a")
    assert agent.llm == "L2" and list(agent.adapters) == ["b"]
    assert agent.tools["x"].handler(None) == "x"      # b's tool, live
    agent.drop("b")
    assert agent.llm == "L0" and agent.tools["x"] is v0
    assert agent.adapters == {}


def test_hook_object_form() -> None:
    agent = Agent(None)
    fn = lambda hctx: None  # noqa: E731
    agent.use(lambda c: c.hook(Hook("after_tool", fn)), name="h")
    assert agent.hooks.get("after_tool") == [fn]
    agent.drop("h")
    assert agent.hooks.get("after_tool") == []


@given(st.lists(st.sampled_from("abc"), max_size=8))
@settings(max_examples=50, deadline=None)
def prop_reverse_drops_restore(names: list[str]) -> None:
    agent = Agent(None, tools=[echo])
    original = dict(agent.tools)
    for i, n in enumerate(names):
        t = named_tool(n)
        agent.use(lambda c, t=t: c.tool(t), name=f"ad{i}")
    for i in reversed(range(len(names))):
        assert agent.drop(f"ad{i}") == []
    assert agent.tools == original and agent.adapters == {}


# ---- live self-composition: the loop late-binds ------------------------

async def test_agent_extends_itself_mid_run() -> None:
    @tool(parameters={"type": "object", "properties": {}})
    def extend(ctx):
        """Mount a calc tool into the running agent."""
        ctx.run.agent.use(lambda c: c.tool(named_tool("calc")), name="calcpack")
        return "mounted"

    llm = ScriptedLLM([call_reply("c1", "extend", {}),
                       call_reply("c2", "calc", {}),
                       text_reply("done")])
    agent = Agent(llm, tools=[extend])
    result = await agent.run("go")
    assert result.status == "completed" and result.output == "done"
    assert "calc" in llm.seen_tools[1]      # NEXT step saw the new tool
    tools_used = [m.tool_result for m in result.messages if m.role == "tool"]
    assert tools_used[1].content == "calc" and not tools_used[1].is_error
    assert "calc" in agent.tools            # survives the run


async def test_agent_swaps_its_own_llm_mid_run() -> None:
    llm_b = ScriptedLLM([text_reply("from-b")])

    @tool(parameters={"type": "object", "properties": {}})
    def become_b(ctx):
        """Swap this agent's LLM."""
        ctx.run.agent.use(lambda c: c.llm(llm_b), name="swap")
        return "swapped"

    llm_a = ScriptedLLM([call_reply("c1", "become_b", {})])
    agent = Agent(llm_a, tools=[become_b])
    result = await agent.run("go")
    assert result.status == "completed" and result.output == "from-b"
    assert len(llm_a.seen) == 1 and len(llm_b.seen) == 1   # step 2 ran on B
    assert len(llm_b.seen[0]) >= 3   # ...with the full transcript intact


async def test_hook_dropping_itself_skips_no_sibling() -> None:
    # _fire snapshots the hook list: a's self-drop must not skip b
    fired: list[str] = []
    llm = ScriptedLLM([call_reply("c1", "echo", {"x": "hi"}), text_reply("ok")])
    agent = Agent(llm, tools=[echo])

    def a_hook(hctx):
        fired.append("a")
        agent.drop("a")

    def b_hook(hctx):
        fired.append("b")

    agent.use(lambda c: c.hook("before_tool", a_hook), name="a")
    agent.use(lambda c: c.hook("before_tool", b_hook), name="b")
    result = await agent.run("go")
    assert result.status == "completed"
    assert fired == ["a", "b"]
    assert "a" not in agent.adapters and "b" in agent.adapters


async def test_self_mounted_hook_guards_itself() -> None:
    def deny_echo(hctx):
        from core import ToolResult
        if hctx.call.name == "echo":
            hctx.result = ToolResult(hctx.call.id, "echo", "denied", is_error=True)

    @tool(parameters={"type": "object", "properties": {}})
    def guard_me(ctx):
        """Mount a hook that blocks echo."""
        ctx.run.agent.use(lambda c: c.hook("before_tool", deny_echo), name="g")
        return "guarded"

    llm = ScriptedLLM([call_reply("c1", "guard_me", {}),
                       call_reply("c2", "echo", {"x": "hi"}),
                       text_reply("ok")])
    agent = Agent(llm, tools=[echo, guard_me])
    result = await agent.run("go")
    tools_used = [m.tool_result for m in result.messages if m.role == "tool"]
    assert tools_used[1].content == "denied" and tools_used[1].is_error


def main() -> None:
    test_mount_drop_restores_everything()
    test_lifo_nesting_and_shadowing()
    test_partial_failure_mounts_nothing()
    test_loud_errors()
    test_dispose_collects_inverse_errors()
    test_out_of_order_drop_relinks()
    test_hook_object_form()
    prop_reverse_drops_restore()
    asyncio.run(test_agent_extends_itself_mid_run())
    asyncio.run(test_agent_swaps_its_own_llm_mid_run())
    asyncio.run(test_hook_dropping_itself_skips_no_sibling())
    asyncio.run(test_self_mounted_hook_guards_itself())
    print("test_adapter: all ok")


if __name__ == "__main__":
    main()
