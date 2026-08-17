"""Generic data types. The loop speaks only this vocabulary.

Provider adapters translate these to/from provider-specific formats.
Nothing in this file knows about OpenAI, Anthropic, or anyone else.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class ToolCall:
    """The LLM asking for a tool to be executed."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    """The outcome of a tool execution, fed back to the LLM."""

    call_id: str
    name: str
    content: Any = None
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorInfo:
    """The one error shape: any ``"error"`` key in an event payload or
    RunResult holds this, serialized. Model-facing channels (ToolResult
    content) stay plain strings — models read prose, consumers parse this."""

    type: str                  # exception class or code: "UnknownTool", "Cancelled"
    message: str
    source: str                # "run" | "tool" | "hook"
    detail: str | None = None  # traceback / response body; consumer-only

    @classmethod
    def from_exc(cls, ex: BaseException, source: str, detail: str | None = None) -> "ErrorInfo":
        return cls(type(ex).__name__, str(ex), source, detail)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Message:
    """One turn in the conversation. Generic across all providers.

    - assistant messages may carry ``tool_calls``
    - tool messages carry exactly one ``tool_result``
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_result: ToolResult | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [c.to_dict() for c in self.tool_calls]
        if self.tool_result:
            d["tool_result"] = self.tool_result.to_dict()
        if self.meta:
            d["meta"] = self.meta
        return d


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0  # subset of output_tokens on reasoning models

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
