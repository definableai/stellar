"""The LLM layer contract.

An LLM adapter is anything with a ``stream()`` method that:

    1. yields zero or more ``LLMDelta`` (incremental text), then
    2. yields exactly one ``LLMReply`` (the complete assistant message,
       with any tool calls fully assembled) and stops.

That single rule — "deltas, then exactly one reply" — is the whole
contract. All provider-specific mess (partial tool-call JSON
accumulation, event formats, retries, stop reasons) lives inside the
adapter. The loop only ever sees generic ``Message``/``ToolCall`` data,
which is what keeps the core hackable and provider-agnostic.

Adapters may honor ``asyncio`` cancellation; the loop also stops
consuming between chunks when a stop is requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, Sequence, runtime_checkable

from .tools import ToolSpec
from .types import Message, Usage


@dataclass
class LLMDelta:
    """An incremental chunk of assistant text."""

    text: str = ""


@dataclass
class LLMReply:
    """The final, complete result of one LLM call."""

    message: Message                      # role="assistant"; content and/or tool_calls
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end"              # "end" | "tool_use" | "length" | adapter-defined
    raw: Any = None                       # optional escape hatch: the provider response


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
