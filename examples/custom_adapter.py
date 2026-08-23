"""Write your own LLM adapter — the whole job, in one file.

A toy provider streams events in a dialect nobody else speaks;
``ReplyBuilder`` translates them into the core contract (deltas, then
exactly one reply) and a real ``Agent`` runs on top with a real tool.
The second run is the stream every adapter eventually meets: tool-call
JSON cut mid-flight — the builder flags it, and the loop hands the model
a readable error instead of silently dropping the call. Sections follow
the recipe in internal/llm/common.py; no network, no key.

    uv run python -m examples.custom_adapter
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Sequence

from core import (Agent, LLMDelta, LLMReply, Message, ReplyBuilder, StepKind,
                  StepPhase, ToolSpec, tool)
from internal.llm.common import data_url, map_blocks

# ---- the toy provider (pretend this is someone else's SDK) ------------
# Takes a prompt string, streams dicts in its own shape:
#   {"kind": "think"|"say", "chunk": str}                incremental output
#   {"kind": "call", "slot": int, "tool": str, "id": str}      a call opens
#   {"kind": "call", "slot": int, "args_piece": str}      ...its JSON, piecewise
#   {"kind": "done", "reason": "call"|"stop"|"cut", "spent": int}

SEEN: list[str] = []          # every prompt the toy actually received


async def toy_call(prompt: str,
                   script: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    SEEN.append(prompt)
    for ev in script:
        await asyncio.sleep(0)      # a real one would await the socket here
        yield ev


# ---- request: core -> wire (transformations 1 and 2) ------------------


def _blocks(content: Any) -> str:
    """2. content blocks -> the toy's shapes. It only speaks flat text,
    so images and files degrade to a URL mention."""
    parts = map_blocks(content,
                       text=lambda b: b.get("text", ""),
                       image=lambda b: f"<img {data_url(b, 'image/png')}>",
                       file=lambda b: f"<doc {data_url(b, 'application/pdf')}>")
    return parts if isinstance(parts, str) else " ".join(parts)


def _to_toy(messages: Sequence[Message]) -> str:
    """1. core Message[] -> the toy's wire: one flat script. It has no
    roles and no tool protocol, so calls and results become prose lines."""
    lines: list[str] = []
    for m in messages:
        if m.tool_result:
            lines.append(f"result: {m.tool_result.content}")
            continue
        if text := _blocks(m.content):
            lines.append(f"{m.role}: {text}")
        lines += [f"{m.role} called {c.name}{c.arguments}" for c in m.tool_calls]
    return "\n".join(lines)


# ---- adapter ----------------------------------------------------------


class ToyLLM:
    """One canned stream per LLM step. Transformation 3 is the loop
    below: every provider event maps to exactly one builder call."""

    def __init__(self, *turns: list[dict[str, Any]]) -> None:
        self.turns = list(turns)

    async def stream(self, messages: Sequence[Message], tools: Sequence[ToolSpec],
                     **params: Any) -> AsyncIterator[LLMDelta | LLMReply]:
        # a real adapter would put `tools`/`params` in the request payload
        b = ReplyBuilder()
        async for ev in toy_call(_to_toy(messages), self.turns.pop(0)):
            kind, slot = ev["kind"], ev.get("slot", 0)
            if kind == "say":
                yield b.text(ev["chunk"])
            elif kind == "think":
                yield b.reasoning(ev["chunk"])      # never lands in the reply
            elif kind == "call":
                b.tool_call(slot, id=ev.get("id", ""), name=ev.get("tool", ""))
                if "args_piece" in ev:
                    yield b.tool_args(slot, ev["args_piece"])
            elif kind == "done":
                b.usage(output_tokens=ev.get("spent", 0))
                b.finish({"call": "tool_use",            # unmapped -> "end"
                          "cut": "length"}.get(ev["reason"]))
        yield b.reply()             # exactly one reply, always last


# ---- self-check -------------------------------------------------------


RAN: list[tuple[float, float]] = []


@tool(parameters={"type": "object", "required": ["a", "b"],
                  "properties": {"a": {"type": "number"},
                                 "b": {"type": "number"}}})
def add(ctx, a: float, b: float) -> float:
    """Add two numbers."""
    RAN.append((a, b))
    return a + b


CALL = [{"kind": "think", "chunk": "they want a sum"},
        {"kind": "say", "chunk": "on it"},
        {"kind": "call", "slot": 0, "tool": "add", "id": "c1"},
        {"kind": "call", "slot": 0, "args_piece": '{"a": 2,'},   # JSON arrives
        {"kind": "call", "slot": 0, "args_piece": ' "b": 3}'},   # in fragments
        {"kind": "done", "reason": "call", "spent": 12}]
CUT = [{"kind": "call", "slot": 0, "tool": "add", "id": "c9"},
       {"kind": "call", "slot": 0, "args_piece": '{"a": 2, "b"'},  # truncated
       {"kind": "done", "reason": "cut"}]
SAY = [{"kind": "say", "chunk": "2 + 3 = 5"}, {"kind": "done", "reason": "stop"}]


async def main() -> None:
    handle = Agent(ToyLLM(CALL, SAY), tools=[add]).run("add 2 and 3")
    channels = [e.payload["channel"] async for e in handle
                if e.kind is StepKind.TEXT and e.phase is StepPhase.DELTA]
    r = await handle                                 # both steps, in order
    assert channels == ["reasoning", "text", "tool_args", "tool_args",
                        "text"], channels
    assert r.messages[1].tool_calls[0].arguments == {"a": 2, "b": 3}  # reassembled
    assert r.messages[1].content == "on it"          # reasoning stayed out
    assert (RAN, r.messages[2].tool_result.content) == ([(2, 3)], 5)
    assert "result: 5" in SEEN[-1]                   # the result went back out
    assert (r.output, r.usage.output_tokens) == ("2 + 3 = 5", 12)

    RAN.clear()
    r = await Agent(ToyLLM(CUT, CALL, SAY), tools=[add]).run("add 2 and 3")
    assert r.messages[1].meta["invalid_tool_args"] == {"c9": '{"a": 2, "b"'}
    bad = r.messages[2].tool_result                  # what the model is told
    assert bad.is_error and str(bad.content).startswith("Malformed tool arguments")
    assert RAN == [(2, 3)]        # the cut call never ran; the re-issue did
    assert r.output == "2 + 3 = 5"
    print("custom adapter demo ok")


if __name__ == "__main__":
    asyncio.run(main())
