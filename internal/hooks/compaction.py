"""Context compaction — a before_llm hook that keeps the transcript small.

    hooks.add("before_llm", compaction(max_chars=400_000, keep_last=8,
                                       summarizer=OpenAILLM(model="gpt-4o-mini")))

When the serialized transcript exceeds ``max_chars`` (chars ~ 4x tokens —
deliberately tokenizer-free), everything between the leading system
messages and the last ``keep_last`` messages is replaced by one summary
message. With a ``summarizer`` LLM the middle is summarized (one-shot
call); with ``summarizer=None`` it is dropped with a marker. The cut
never orphans tool results — the kept window is walked back so it never
starts inside a tool-call/result pair.

Self-check: uv run python -m internal.hooks.compaction
"""

from __future__ import annotations

import json
from typing import Any

from core import LLMReply, Message, Usage


def _size(messages: list[Message]) -> int:
    return len(json.dumps([m.to_dict() for m in messages], default=str))


def _text(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        if m.role == "tool" and m.tool_result:
            lines.append(f"tool[{m.tool_result.name}] -> {m.tool_result.content}")
        else:
            body = m.content if isinstance(m.content, str) else json.dumps(m.content)
            calls = "".join(f" [called {c.name}({json.dumps(c.arguments)})]"
                            for c in m.tool_calls)
            lines.append(f"{m.role}: {body or ''}{calls}")
    return "\n".join(lines)


def compaction(max_chars: int = 400_000, keep_last: int = 8,
               summarizer: Any = None, max_tokens: int | None = None):
    """-> before_llm hook.

    With ``max_tokens``, triggers on the previous step's *real* input
    size (``ctx.run.last_usage.input_tokens`` — exact, tokenizer-free),
    falling back to chars/4 before the first measurement (fresh run,
    resumed session). Without it: ``max_chars`` on the serialized
    transcript (chars ~ 4x tokens)."""

    async def hook(ctx: Any) -> None:
        msgs = ctx.messages
        if max_tokens is not None:
            measured = getattr(ctx.run, "last_usage", Usage()).input_tokens
            if (measured or _size(msgs) // 4) <= max_tokens:
                return
        elif _size(msgs) <= max_chars:
            return
        head = 0
        while head < len(msgs) and msgs[head].role == "system":
            head += 1
        cut = max(head, len(msgs) - keep_last)
        while cut > head and msgs[cut].role == "tool":   # keep pairs together
            cut -= 1
        middle = msgs[head:cut]
        if len(middle) < 3:
            return   # fails OPEN: nothing worth compacting — oversized keep
                     # window goes to the provider rather than aborting the run
        if summarizer is not None:
            prompt = ("Summarize this conversation segment concisely. Preserve "
                      "facts, decisions, file paths, numbers, and unresolved "
                      "tasks:\n\n" + _text(middle))
            reply = None
            async for item in summarizer.stream(
                    [Message(role="user", content=prompt)], []):
                if isinstance(item, LLMReply):
                    reply = item
            summary = (reply and reply.message.content) or "(summary unavailable)"
        else:
            summary = f"({len(middle)} earlier messages dropped)"
        msgs[head:cut] = [Message(role="user",
                                  content=f"[Conversation so far]\n{summary}",
                                  meta={"compacted": len(middle)})]

    return hook


if __name__ == "__main__":
    import asyncio

    from core import Agent, Hooks, LLMDelta, Message, ToolCall, ToolResult

    async def _selfcheck() -> None:
        def transcript(n: int) -> list[Message]:
            msgs = [Message(role="system", content="be terse")]
            for i in range(n):
                msgs += [
                    Message(role="user", content=f"question {i} " + "x" * 200),
                    Message(role="assistant", tool_calls=[
                        ToolCall(id=f"c{i}", name="look", arguments={"i": i})]),
                    Message(role="tool", tool_result=ToolResult(f"c{i}", "look", i)),
                    Message(role="assistant", content=f"answer {i} " + "y" * 200),
                ]
            return msgs

        class Run:      # minimal RunContext stand-in
            def __init__(self, tokens: int = 0):
                self.last_usage = Usage(input_tokens=tokens)

        class Ctx:      # minimal LLMHookContext stand-in
            def __init__(self, messages, run: Run | None = None):
                self.messages = messages
                self.run = run or Run()

        # under threshold: untouched
        msgs = transcript(2)
        before = list(msgs)
        await compaction(max_chars=1_000_000)(Ctx(msgs))
        assert msgs == before

        # over threshold, no summarizer: dropped with marker
        msgs = transcript(20)
        await compaction(max_chars=5_000, keep_last=6)(Ctx(msgs))
        assert msgs[0].role == "system"
        assert "[Conversation so far]" in msgs[1].content
        assert msgs[1].meta["compacted"] > 0
        assert 2 + 6 <= len(msgs) <= 2 + 8   # walk-back may widen the window
        assert msgs[2].role != "tool"          # kept window never starts on a result
        for i, m in enumerate(msgs):           # every tool result's call is present
            if m.role == "tool":
                prev = msgs[i - 1]
                assert any(c.id == m.tool_result.call_id for c in prev.tool_calls)
        assert _size(msgs) < 5_000 + 2_000

        # with summarizer: one-shot LLM call produces the summary
        class Summarizer:
            async def stream(self, messages, tools, **params):
                assert "question 0" in messages[0].content
                yield LLMDelta(text="s")
                yield LLMReply(message=Message(role="assistant",
                                               content="user asked 0..n; answered"))

        msgs = transcript(20)
        await compaction(max_chars=5_000, keep_last=4,
                         summarizer=Summarizer())(Ctx(msgs))
        assert "user asked 0..n; answered" in msgs[1].content

        # token threshold: measured usage decides, char size ignored
        msgs = transcript(20)
        await compaction(max_tokens=1_000, keep_last=4)(Ctx(msgs, Run(50_000)))
        assert msgs[1].meta.get("compacted")
        msgs = transcript(20)
        await compaction(max_tokens=10**9, keep_last=4)(Ctx(msgs, Run(500)))
        assert not any(m.meta.get("compacted") for m in msgs)
        # unmeasured (first step): falls back to chars/4 estimate
        msgs = transcript(20)
        await compaction(max_tokens=1_000, keep_last=4)(Ctx(msgs, Run(0)))
        assert msgs[1].meta.get("compacted")

        # end to end: hook wired into a real Agent run
        class Echo:
            async def stream(self, messages, tools, **params):
                yield LLMReply(message=Message(
                    role="assistant", content=f"saw {len(messages)} msgs"))

        hooks = Hooks()
        hooks.add("before_llm", compaction(max_chars=3_000, keep_last=2))
        h = Agent(Echo(), hooks=hooks).run("hi", history=transcript(20)[1:])
        r = await h.result()
        assert r.status == "completed" and int(r.output.split()[1]) <= 5

        print("compaction hook self-check ok")

    asyncio.run(_selfcheck())
