"""The smallest real agent: one JSON-Schema tool, the whole event stream.

Shows a tool that streams progress through ``ctx.emit_delta``, the
run/text/tool event grammar rendered to a console, and the RunResult.

    uv run python -m examples.quickstart        # offline: scripted LLM + asserts
    uv run python -m examples.quickstart live   # real: needs OPENAI_API_KEY
"""

from __future__ import annotations

import asyncio
import json
import operator
import sys

from core import (Agent, LLMDelta, LLMReply, Message, RunResult, StepEvent,
                  StepKind, StepPhase, ToolCall, Usage, tool)

OPS = {"add": operator.add, "mul": operator.mul, "pow": operator.pow}


@tool(parameters={"type": "object",
                  "properties": {"a": {"type": "number"},
                                 "b": {"type": "number"},
                                 "op": {"enum": list(OPS)}},
                  "required": ["a", "b", "op"]})
async def calc(ctx, a: float, b: float, op: str) -> str:
    """Apply one arithmetic operation to two numbers."""
    await ctx.emit_delta({"progress": f"{a} {op} {b}"})   # -> a tool/delta event
    return str(OPS[op](a, b))


# ---- the console: the event grammar in five branches ------------------


def show(e: StepEvent) -> None:
    """Text deltas inline, one line per tool start/progress/end."""
    kind, phase, p = e.kind, e.phase, e.payload
    if kind is StepKind.TEXT and phase is StepPhase.DELTA and p["channel"] == "text":
        print(p["text"], end="", flush=True)
    elif kind is StepKind.TEXT and phase is StepPhase.END and p["text"]:
        print()
    elif kind is StepKind.TOOL and phase is StepPhase.START:
        print(f"  . {p['name']}({json.dumps(p['arguments'])})")
    elif kind is StepKind.TOOL and phase is StepPhase.DELTA:
        print(f"    ... {p.get('progress', p)}")   # payload is tool-defined
    elif kind is StepKind.TOOL and phase is StepPhase.END:
        print(f"    -> {p['result']}")


async def drive(agent: Agent, prompt: str) -> tuple[RunResult, list[StepEvent]]:
    """Stream one run to the console, then take its result."""
    handle = agent.run(prompt)
    events: list[StepEvent] = []
    async for e in handle:          # stream first...
        show(e)
        events.append(e)
    result = await handle           # ...then take the result
    print(f"[{result.status}] {result.output}\nusage: {result.usage.to_dict()}")
    return result, events


# ---- offline mode -----------------------------------------------------


class ScriptedLLM:
    """Canned replies, in order — the offline stand-in for a provider."""

    def __init__(self, replies: list[LLMReply]) -> None:
        self.replies = list(replies)

    async def stream(self, messages, tools, **params):
        reply = self.replies.pop(0)
        if reply.message.content:
            yield LLMDelta(text=reply.message.content)
        yield reply


SCRIPT = [
    LLMReply(message=Message(role="assistant", content="Multiplying.",
                             tool_calls=[ToolCall("c1", "calc",
                                                  {"a": 6, "b": 7, "op": "mul"})]),
             usage=Usage(input_tokens=20, output_tokens=8), stop_reason="tool_use"),
    LLMReply(message=Message(role="assistant", content="6 x 7 = 42"),
             usage=Usage(input_tokens=34, output_tokens=6)),
]


async def demo() -> None:
    agent = Agent(ScriptedLLM(SCRIPT), tools=[calc], system="Use the calc tool.")
    result, events = await drive(agent, "what is 6 times 7?")

    steps = [(e.kind.value, e.phase.value) for e in events]
    assert steps[0] == ("run", "start") and steps[-1] == ("run", "end")
    assert steps.count(("tool", "delta")) == 1          # the streamed progress
    assert [m.role for m in result.messages] == [
        "system", "user", "assistant", "tool", "assistant"]
    assert next(m.tool_result.content
                for m in result.messages if m.role == "tool") == "42"
    assert result.status == "completed" and result.output == "6 x 7 = 42"
    assert result.usage == Usage(input_tokens=54, output_tokens=14)
    print("quickstart demo ok")


async def live(prompt: str) -> None:
    from internal.llm.openai import OpenAILLM   # deferred: offline mode needs
                                                # nothing outside core/

    agent = Agent(OpenAILLM(model="gpt-4o-mini"), tools=[calc],
                  system="Use the calc tool for arithmetic, then state the answer.")
    await drive(agent, prompt)


if __name__ == "__main__":
    if sys.argv[1:2] == ["live"]:
        asyncio.run(live(" ".join(sys.argv[2:]) or "what is 2 to the power of 10?"))
    else:
        asyncio.run(demo())
