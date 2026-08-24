"""Self-guard: the agent writes a before_tool hook restricting its own
bash, mounts it, and is then demonstrably blocked by its own guardrail.

Hooks mounted by an adapter bind the agent that mounted them — there is
no privileged self that bypasses them. The same short-circuit rule that
powers HITL approval (``ctx.result`` set in before_tool skips
execution) is what the agent uses on itself.

    uv run python -m examples.self_guard       # offline: scripted LLM + asserts
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from core import Agent, LLMDelta, LLMReply, Message, Tool, ToolCall, tool
from internal.kernel import boot

# The guardrail the agent writes for itself: bash may only run git.
GUARD_SOURCE = '''
from core import ToolResult

def setup(ctx):
    def only_git(hctx):
        cmd = hctx.call.arguments.get("cmd", "")
        if hctx.call.name == "bash" and not cmd.startswith("git "):
            hctx.result = ToolResult(hctx.call.id, "bash",
                                     f"blocked by self-guard: {cmd!r}",
                                     is_error=True)
    ctx.hook("before_tool", only_git)
'''


@tool(parameters={"type": "object", "required": ["cmd"],
                  "properties": {"cmd": {"type": "string"}}},
      parallel_safe=False)
def bash(cctx, cmd: str) -> str:
    """Pretend shell — the demo is about the hook, not the subprocess."""
    return f"ran: {cmd}"


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
        return f"wrote {p.name}"
    return write_file


class ScriptedLLM:
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
        agent = Agent(ScriptedLLM([
            call("c1", "bash", {"cmd": "rm -rf scratch/"}),     # unguarded: runs
            call("c2", "write_file", {"path": "guard.py",
                                      "content": GUARD_SOURCE}),
            call("c3", "adapter_load", {"path": "guard.py"}),
            call("c4", "bash", {"cmd": "rm -rf scratch/"}),     # now: blocked
            call("c5", "bash", {"cmd": "git status"}),          # allowed
            text("guarded myself"),
        ]), tools=[bash, make_write_file(ws)])
        boot(agent, ws)

        result = await agent.run("restrict your own bash to git, then prove it")
        assert result.status == "completed"
        r = [m.tool_result for m in result.messages if m.role == "tool"]

        assert r[0].content == "ran: rm -rf scratch/"          # before: anything
        assert r[3].is_error and "blocked by self-guard" in r[3].content
        assert r[4].content == "ran: git status"               # the carve-out
        assert agent.hooks.get("before_tool") != []            # binds the mounter

    print("self_guard demo ok")


if __name__ == "__main__":
    asyncio.run(demo())
