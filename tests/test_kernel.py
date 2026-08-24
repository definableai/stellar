"""Kernel self-checks: boot mounts, the self-composition loop, the path
jail, edit-reload, atomic reload, double-load, self-lockdown.

Run: uv run python tests/test_kernel.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (  # noqa: E402
    Agent, LLMDelta, LLMReply, Message, ToolCall, boot, kernel,
)

CALC_V1 = '''
from core import Tool, ToolSpec

def setup(ctx):
    ctx.tool(Tool(ToolSpec("calc", "Add a and b.",
                           {"type": "object", "required": ["a", "b"],
                            "properties": {"a": {"type": "number"},
                                           "b": {"type": "number"}}}),
                  handler=lambda cctx, a, b: a + b))
'''


class Scripted:
    def __init__(self, replies):
        self.replies = list(replies)

    async def stream(self, messages, tools, **params):
        reply = self.replies.pop(0)
        if reply.message.content:
            yield LLMDelta(text=reply.message.content)
        yield reply


def call(cid, name, args):
    return LLMReply(message=Message(
        role="assistant", tool_calls=[ToolCall(cid, name, args)]),
        stop_reason="tool_use")


def text(t):
    return LLMReply(message=Message(role="assistant", content=t))


async def test_kernel_self_composition() -> None:
    with tempfile.TemporaryDirectory() as d:
        ws = Path(d)
        (ws / "calc.py").write_text(CALC_V1)
        (ws / "tool").mkdir()
        (ws / "tool" / "probe.py").write_text(CALC_V1.replace('"calc"', '"probe"'))
        (ws / "_wip.py").write_text("syntax error(")        # skipped: leading _
        (ws / "__pycache__").mkdir()
        (ws / "__pycache__" / "calc.py").write_text("syntax error(")

        # a workspace is the developer's call: no default, no silent guess
        try:
            Agent(Scripted([])).use(kernel)
            raise AssertionError("kernel without workspace must raise")
        except ValueError as ex:
            assert "workspace=" in str(ex)

        # boot: kernel + the workspace files (subdirectories included)
        agent = Agent(Scripted([]), max_steps=8)
        scopes = boot(agent, ws)
        assert [s.name for s in scopes] == ["kernel", "calc", "tool/probe"]
        assert "adapter_load" in agent.tools and "calc" in agent.tools
        assert "probe" in agent.tools

        # the full loop: agent inspects itself, unloads calc, reloads it,
        # calls it — every self-change is an ordinary, logged tool step
        agent.llm = Scripted([
            call("c1", "adapter_list", {}),
            call("c2", "adapter_unload", {"name": "calc"}),
            call("c3", "adapter_load", {"path": "calc.py"}),
            call("c4", "calc", {"a": 2, "b": 3}),
            text("done"),
        ])
        result = await agent.run("evolve")
        assert result.status == "completed"
        results = [m.tool_result for m in result.messages if m.role == "tool"]
        assert not any(r.is_error for r in results), results
        assert results[3].content == 5

        # jail: escapes and non-.py fail as readable tool errors
        agent.llm = Scripted([call("c5", "adapter_load",
                                   {"path": "../evil.py"}), text("ok")])
        result = await agent.run("try escape")
        bad = [m.tool_result for m in result.messages if m.role == "tool"][0]
        assert bad.is_error and "load only from" in bad.content

        # reload picks up an edit
        (ws / "calc.py").write_text(CALC_V1.replace("a + b", "a * b"))
        agent.llm = Scripted([call("c6", "adapter_reload", {"name": "calc"}),
                              call("c7", "calc", {"a": 2, "b": 3}), text("ok")])
        result = await agent.run("reload")
        results = [m.tool_result for m in result.messages if m.role == "tool"]
        assert results[1].content == 6

        # a broken edit must not cost the working adapter (atomic reload)
        (ws / "calc.py").write_text("def setup(ctx:\n")
        agent.llm = Scripted([call("c8", "adapter_reload", {"name": "calc"}),
                              call("c9", "calc", {"a": 1, "b": 1}), text("ok")])
        result = await agent.run("bad edit")
        rs = [m.tool_result for m in result.messages if m.role == "tool"]
        assert rs[0].is_error and "SyntaxError" in rs[0].content
        assert rs[1].content == 1        # the a*b version is still mounted

        # double-load: rejected BEFORE the module body would run again
        agent.llm = Scripted([call("c10", "adapter_load", {"path": "calc.py"}),
                              text("ok")])
        result = await agent.run("dup")
        dup = [m.tool_result for m in result.messages if m.role == "tool"][0]
        assert dup.is_error and "already mounted" in dup.content

        # self-lockdown: dropping the kernel removes the four tools
        # (calc mounted after it remounts on the new base and survives)
        agent.drop("kernel")
        assert "adapter_load" not in agent.tools and "calc" in agent.tools


async def main() -> None:
    await test_kernel_self_composition()
    print("test_kernel: all ok")


if __name__ == "__main__":
    asyncio.run(main())
