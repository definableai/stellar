"""Multi-agent delegation: one lead, two specialist subagents.

Shows ``subagent()`` deriving each child from the *running parent* — own
``system=``, everything else inherited — the researcher calling a tool it
inherited from the lead's toolset, and the child's whole event stream
forwarding through the parent handle as ``tool/delta`` events carrying
``payload["child"]``. The tree render turns that back into visible
structure: parent events flush left, child events indented one level.

Offline by default: one shared ScriptedLLM drives every step, parent and
child alike (children inherit the LLM, so replies are consumed in call
order — lead, researcher, lookup-turn, lead, writer, lead).
LIVE: swap one line —
``from internal.llm.anthropic import AnthropicLLM; llm = AnthropicLLM()``.

    uv run python -m examples.multi_agent
"""

from __future__ import annotations

import asyncio
from typing import Any

from core import (Agent, LLMDelta, LLMReply, Message, StepKind, StepPhase,
                  ToolCall, tool)
from internal.subagent import subagent

FACTS = {"raccoons": "nocturnal, dexterous, ~20 distinct vocalizations"}


@tool(parameters={"type": "object",
                  "properties": {"topic": {"type": "string"}},
                  "required": ["topic"]})
def lookup(ctx, topic: str = "") -> str:
    """Look a topic up in the knowledge base."""
    return FACTS.get(topic, "no entry")


class ScriptedLLM:
    """Canned replies popped in call order — the offline seam."""

    def __init__(self, replies: list[LLMReply]) -> None:
        self.replies = list(replies)

    async def stream(self, messages, tools, **params):
        reply = self.replies.pop(0)
        if reply.message.content:
            yield LLMDelta(text=reply.message.content)
        yield reply


def says(text: str) -> LLMReply:
    return LLMReply(message=Message(role="assistant", content=text))


def calls(name: str, **args: Any) -> LLMReply:
    return LLMReply(message=Message(role="assistant", tool_calls=[
        ToolCall(f"c-{name}", name, args)]), stop_reason="tool_use")


ANSWER = "Raccoons: dexterous night-shift bandits with ~20 words."
SCRIPT = [
    calls("researcher", prompt="what do we know about raccoons?"),  # lead
    calls("lookup", topic="raccoons"),                              # researcher
    says("Facts: nocturnal, dexterous, ~20 vocalizations."),        # researcher
    calls("writer", prompt="turn those facts into one line"),       # lead
    says(ANSWER),                                                   # writer
    says(f"Report. {ANSWER}"),                                      # lead
]


def _short(v: Any, n: int = 64) -> str:
    s = v if isinstance(v, str) else ", ".join(f"{k}={w}" for k, w in v.items())
    return s if len(s) <= n else s[:n - 1] + "..."


def _line(kind: str, phase: str, p: dict[str, Any]) -> str:
    """One line per interesting event. ponytail: text deltas dropped —
    the tree is about structure, not tokens."""
    if kind == "run":
        return f"[run {phase}]" + (f" {p['status']}" if phase == "end" else "")
    if kind == "tool" and phase == "start":
        return f"-> {p['name']}({_short(p['arguments'])})"
    if kind == "tool" and phase == "end":
        return f"   = {_short(p['result'])}"
    if kind == "text" and phase == "end" and p.get("text"):
        return f'   " {_short(p["text"])}'
    return ""


def render(e: dict[str, Any], depth: int = 0) -> None:
    """Print an event; a forwarded child event recurses one level in."""
    child = e["payload"].get("child")
    if child is not None:
        return render(child, depth + 1)
    line = _line(e["kind"], e["phase"], e["payload"])
    if line:
        print("  " * depth + line)


async def main() -> None:
    llm = ScriptedLLM(SCRIPT)
    lead = Agent(
        llm,
        system="You are the lead. Delegate, then report.",
        tools=[
            lookup,   # the researcher inherits this from the lead's toolset
            subagent(name="researcher", description="Dig up facts on a topic.",
                     system="You are the researcher. Use lookup; report facts."),
            subagent(name="writer", description="Turn facts into one line.",
                     system="You are the writer. One vivid line, no lists."),
        ],
    )
    print("event tree (indent = delegation depth):")
    handle = lead.run("write me a line about raccoons")
    events = []
    async for e in handle.events():
        events.append(e)
        render(e.to_dict())
    result = await handle

    assert result.status == "completed", result.status
    assert result.output == f"Report. {ANSWER}", result.output
    assert not llm.replies, "script desynced"        # every step consumed

    kids = [e.payload["child"] for e in events
            if e.kind is StepKind.TOOL and e.phase is StepPhase.DELTA
            and "child" in e.payload]
    ends = [c for c in kids if c["kind"] == "run" and c["phase"] == "end"]
    assert len(ends) == 2, ends                      # both children ran nested
    assert all(c["payload"]["status"] == "completed" for c in ends)
    assert any(c["kind"] == "tool" and c["payload"].get("name") == "lookup"
               for c in kids)                        # inherited tool, in a child
    print("multi agent demo ok")


if __name__ == "__main__":
    asyncio.run(main())
