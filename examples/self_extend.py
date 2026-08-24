"""Self-extension: the agent writes a tool for itself, mounts it, uses
it — and the tool survives a restart, because the self is files.

Act 1: an agent with no calculator is asked to add. It writes
``calc.py`` into its adapter workspace with an ordinary file tool, loads
it with ``adapter_load`` (a tool call like any other — gated by the same
hooks, recorded in the same log), and answers with its own creation.

Act 2: a NEW process boots from the same workspace. The calculator is
just there — ``boot()`` mounts every workspace file. No migration step,
no registry: the directory is the manifest.

    uv run python -m examples.self_extend       # offline: scripted LLM + asserts
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from core import Agent, LLMDelta, LLMReply, Message, Tool, ToolCall, tool
from internal.kernel import boot

# What the "model" decides to write for itself. An adapter is one file:
# setup(ctx) registering whatever it contributes.
CALC_SOURCE = '''
from core import tool

def setup(ctx):
    @ctx.tool
    @tool(parameters={"type": "object", "required": ["a", "b"],
                      "properties": {"a": {"type": "number"},
                                     "b": {"type": "number"}}})
    def calc(cctx, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
'''


def make_write_file(ws: Path) -> Tool:
    @tool(parameters={"type": "object", "required": ["path", "content"],
                      "properties": {"path": {"type": "string"},
                                     "content": {"type": "string"}}},
          parallel_safe=False)
    def write_file(cctx, path: str, content: str) -> str:
        """Write a file into the adapter workspace."""
        p = (ws / path).resolve()
        if not p.is_relative_to(ws):
            raise ValueError(f"writes stay inside the workspace: {path!r}")
        p.write_text(content)
        return f"wrote {p.name} ({len(content)} bytes)"
    return write_file


class ScriptedLLM:
    """Canned replies, in order — the offline stand-in for a provider."""

    def __init__(self, replies: list[LLMReply]) -> None:
        self.replies = list(replies)

    async def stream(self, messages, tools, **params):
        reply = self.replies.pop(0)
        if reply.message.content:
            yield LLMDelta(text=reply.message.content)
        yield reply


def call(cid: str, name: str, args: dict) -> LLMReply:
    return LLMReply(message=Message(role="assistant",
                                    tool_calls=[ToolCall(cid, name, args)]),
                    stop_reason="tool_use")


def text(t: str) -> LLMReply:
    return LLMReply(message=Message(role="assistant", content=t))


async def demo() -> None:
    with tempfile.TemporaryDirectory() as d:
        ws = Path(d).resolve()

        # ---- act 1: no calculator? write one, mount it, use it --------
        agent = Agent(ScriptedLLM([
            call("c1", "write_file", {"path": "calc.py", "content": CALC_SOURCE}),
            call("c2", "adapter_load", {"path": "calc.py"}),
            call("c3", "calc", {"a": 12, "b": 34}),
            text("12 + 34 = 46"),
        ]), tools=[make_write_file(ws)])
        boot(agent, ws)                       # kernel only: workspace is empty
        assert "calc" not in agent.tools

        result = await agent.run("add 12 and 34 (you have no calculator... yet)")
        assert result.status == "completed" and result.output == "12 + 34 = 46"
        results = [m.tool_result for m in result.messages if m.role == "tool"]
        assert not any(r.is_error for r in results), results
        assert results[2].content == 46
        assert "calc" in agent.tools          # the agent grew a tool
        # every act of self-change is an ordinary logged tool step
        assert [r.name for r in results] == ["write_file", "adapter_load", "calc"]

        # ---- act 2: restart — the self survives as files ---------------
        reborn = Agent(ScriptedLLM([call("c4", "calc", {"a": 2, "b": 3}),
                                    text("5")]))
        scopes = boot(reborn, ws)             # directory IS the manifest
        assert [s.name for s in scopes] == ["kernel", "calc"]
        result2 = await reborn.run("add 2 and 3")
        assert result2.output == "5"
        assert [m.tool_result.content for m in result2.messages
                if m.role == "tool"] == [5]

    print("self_extend demo ok")


if __name__ == "__main__":
    asyncio.run(demo())
