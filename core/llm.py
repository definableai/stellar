"""The LLM layer contract: an adapter is anything with a ``stream()``
method that yields zero or more ``LLMDelta`` (incremental text), then
exactly one ``LLMReply`` (the complete assistant message, tool calls
fully assembled) and stops. That single rule is the whole contract:
provider mess (partial tool-call JSON, event formats, retries, stop
reasons) lives inside the adapter; the loop sees only ``types.py``
data. Adapters may honor cancellation; the loop stops on stop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Protocol, Sequence, runtime_checkable

from .tools import ToolSpec
from .types import Message, ToolCall, Usage, new_id

Channel = Literal["text", "reasoning", "tool_args"]


class LLMError(RuntimeError):
    """What adapters raise on a failed provider call. Provider-agnostic:
    handlers branch on status (retry 429/5xx, die on 401), never on vendor."""

    def __init__(self, status: int, body: str):
        self.status, self.body = status, body
        super().__init__(f"LLM HTTP {status}: {body}")

    @property
    def retryable(self) -> bool:
        return self.status in (408, 429) or self.status >= 500


@dataclass
class LLMDelta:
    """An incremental chunk of assistant output. ``channel``: "text",
    "reasoning", or "tool_args" (partial JSON of call #``index``, display
    only — the loop executes from the assembled LLMReply). Reasoning never
    lands in the reply; adapters stash raw blocks in ``message.meta``."""

    text: str = ""
    channel: Channel = "text"
    index: int = 0


@dataclass
class LLMReply:
    """The final, complete result of one LLM call."""

    message: Message                      # role="assistant"; content and/or tool_calls
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end"              # "end" | "tool_use" | "length" | adapter-defined
    raw: Any = None                       # optional escape hatch: the provider response


class ReplyBuilder:
    """The adapter skeleton. Parse your provider's stream, feed it here,
    yield what comes back, finish with the assembled reply:

        b = ReplyBuilder()
        async for ev in provider_stream:
            if is_text(ev):       yield b.text(ev.chunk)
            elif is_think(ev):    yield b.reasoning(ev.chunk)
            elif call_opens(ev):  b.tool_call(ev.index, id=ev.id, name=ev.name)
            elif is_args(ev):     yield b.tool_args(ev.index, ev.fragment)
            elif is_usage(ev):    b.usage(input_tokens=..., output_tokens=...)
            elif is_stop(ev):     b.finish(ev.reason)
        yield b.reply()

    The builder owns the assembly rules so adapters cannot get them
    wrong: text joins in stream order, argument fragments accumulate
    per index, and unparseable argument JSON (length-stop truncation,
    provider bugs) yields the call with ``arguments={}`` plus its raw
    fragment in ``message.meta["invalid_tool_args"]`` — the loop turns
    that into a readable model-facing error instead of a silent drop.
    Provider extras go in ``.meta``."""

    def __init__(self) -> None:
        self._text: list[str] = []
        self._calls: dict[int, dict[str, str]] = {}
        self._usage = Usage()
        self._stop = "end"
        self.meta: dict[str, Any] = {}     # lands on message.meta verbatim

    def text(self, chunk: str) -> LLMDelta:
        self._text.append(chunk)
        return LLMDelta(text=chunk, channel="text")

    def reasoning(self, chunk: str) -> LLMDelta:
        return LLMDelta(text=chunk, channel="reasoning")

    def tool_call(self, index: int, *, id: str = "", name: str = "") -> None:
        """Open (or extend) call #index. ``name`` fragments concatenate
        (providers stream names in pieces); first non-empty ``id`` wins."""
        slot = self._calls.setdefault(index, {"id": "", "name": "", "json": ""})
        slot["id"] = slot["id"] or id
        slot["name"] += name

    def tool_args(self, index: int, fragment: str) -> LLMDelta:
        self.tool_call(index)
        self._calls[index]["json"] += fragment
        return LLMDelta(text=fragment, channel="tool_args", index=index)

    def usage(self, *, input_tokens: int | None = None,
              output_tokens: int | None = None,
              reasoning_tokens: int | None = None) -> None:
        if input_tokens is not None:
            self._usage.input_tokens = input_tokens
        if output_tokens is not None:
            self._usage.output_tokens = output_tokens
        if reasoning_tokens is not None:
            self._usage.reasoning_tokens = reasoning_tokens

    def finish(self, stop_reason: str | None) -> None:
        self._stop = stop_reason or self._stop

    def reply(self) -> LLMReply:
        calls: list[ToolCall] = []
        invalid: dict[str, str] = {}
        for i in sorted(self._calls):
            c = self._calls[i]
            cid = c["id"] or new_id("call")
            try:
                args = json.loads(c["json"] or "{}")
                if not isinstance(args, dict):
                    raise ValueError
            except ValueError:
                args, invalid[cid] = {}, c["json"]
            calls.append(ToolCall(id=cid, name=c["name"], arguments=args))
        meta = dict(self.meta)
        if invalid:
            meta["invalid_tool_args"] = invalid
        return LLMReply(
            message=Message(role="assistant",
                            content="".join(self._text) or None,
                            tool_calls=calls, meta=meta),
            usage=self._usage, stop_reason=self._stop)


@runtime_checkable
class LLM(Protocol):
    """Structural protocol — implement the shape, no inheritance needed."""

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        **params: Any,
    ) -> AsyncIterator[LLMDelta | LLMReply]:
        """Yield LLMDelta chunks, then exactly one LLMReply."""
        ...
