"""Context compaction — a before_llm hook that keeps the transcript small.

    agent = Agent(llm, hooks=[compaction(max_tokens=100_000, keep_last=8,
                                         summarizer=OpenAILLM(model="gpt-4o-mini"))])

When the serialized transcript exceeds ``max_chars`` (chars ~ 4x tokens —
deliberately tokenizer-free), everything between the leading system
messages and the last ``keep_last`` messages is replaced by one summary
message. With a ``summarizer`` LLM the middle is summarized (one-shot
call); with ``summarizer=None`` it is dropped with a marker. The cut
never orphans tool results — the kept window is walked back so it never
starts inside a tool-call/result pair.
"""

from __future__ import annotations

import json
from typing import Any

from core import Hook, LLMReply, Message, Usage


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

    return Hook("before_llm", hook)


def setup(ctx: Any) -> None:
    """Adapter shape (core/adapter.py); config = compaction kwargs."""
    ctx.hook(compaction(**ctx.config))
