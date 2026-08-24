"""Self-swap: the agent replaces its own LLM mid-session; the very next
step streams from the new provider with the transcript intact.

The agent (on "deepseek" here — a scripted stand-in) is told it is
slow. It writes an llm adapter file — ``setup(ctx)`` calling
``ctx.llm(...)`` — loads it, and the loop's late binding does the rest:
``self.llm`` is resolved at every step, so step 3 simply streams from
"moonshot". Nothing restarted, nothing re-sent; dropping the adapter
would restore deepseek (inverses unwind LIFO).

    uv run python -m examples.self_swap        # offline: scripted LLMs + asserts
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from core import Agent, LLMDelta, LLMReply, Message, Tool, ToolCall, tool
from internal.kernel import boot

# The llm adapter the agent writes for itself. Offline stand-in: a real
# one would construct e.g. internal.llm.moonshot.MoonshotLLM(...) instead.
MOONSHOT_SOURCE = '''
from core import LLMDelta, LLMReply, Message

class MoonshotStandIn:
    async def stream(self, messages, tools, **params):
        text = f"moonshot here — continuing a transcript of {len(messages)} messages"
        yield LLMDelta(text=text)
        yield LLMReply(message=Message(role="assistant", content=text))

def setup(ctx):
    ctx.llm(MoonshotStandIn())
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
        return f"wrote {p.name}"
    return write_file


class DeepseekStandIn:
    """The 'slow' provider the agent starts on; counts its steps."""

    def __init__(self, replies: list[LLMReply]) -> None:
        self.replies = list(replies)
        self.steps = 0

    async def stream(self, messages, tools, **params):
        self.steps += 1
        reply = self.replies.pop(0)
        if reply.message.content:
            yield LLMDelta(text=reply.message.content)
        yield reply


def call(cid: str, name: str, args: dict) -> LLMReply:
    return LLMReply(message=Message(role="assistant",
                                    tool_calls=[ToolCall(cid, name, args)]),
                    stop_reason="tool_use")


async def demo() -> None:
    with tempfile.TemporaryDirectory() as d:
        ws = Path(d).resolve()
        deepseek = DeepseekStandIn([
            call("c1", "write_file", {"path": "moonshot.py",
                                      "content": MOONSHOT_SOURCE}),
            call("c2", "adapter_load", {"path": "moonshot.py"}),
        ])
        agent = Agent(deepseek, tools=[make_write_file(ws)])
        boot(agent, ws)

        result = await agent.run("you are slow — switch yourself to moonshot")
        assert result.status == "completed"
        assert result.output.startswith("moonshot here")
        assert deepseek.steps == 2            # steps 1-2 on deepseek...
        assert "of 5 messages" in result.output   # ...step 3 on moonshot, with
        # the full transcript: user + 2x(assistant+tool), nothing lost

        assert type(agent.llm).__name__ == "MoonshotStandIn"
        assert agent.drop("moonshot") == []   # inverse restores the prior llm
        assert agent.llm is deepseek

    print("self_swap demo ok")


if __name__ == "__main__":
    asyncio.run(demo())
