"""Generic data types — the loop speaks only this vocabulary; adapters
translate to/from provider formats. Nothing here knows any provider."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, NotRequired, TypedDict

Role = Literal["system", "user", "assistant", "tool"]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---- content blocks: what a Message.content list may hold. Plain dicts
# on the wire (serializable, hackable); TypedDicts give the full picture.

class TextBlock(TypedDict):
    type: Literal["text"]
    text: str


class ImageBlock(TypedDict):
    type: Literal["image"]
    url: NotRequired[str]          # either a URL...
    media_type: NotRequired[str]   # ...or base64 data + its media type
    data: NotRequired[str]


class FileBlock(TypedDict):
    type: Literal["file"]          # PDFs, docs — provider willing
    media_type: str
    url: NotRequired[str]
    data: NotRequired[str]         # base64
    name: NotRequired[str]


Block = TextBlock | ImageBlock | FileBlock


def text_block(text: str) -> TextBlock:
    return {"type": "text", "text": text}


def image_block(*, url: str | None = None, data: str | None = None,
                media_type: str = "image/png") -> ImageBlock:
    if url:
        return {"type": "image", "url": url}
    return {"type": "image", "media_type": media_type, "data": data or ""}


def file_block(*, url: str | None = None, data: str | None = None,
               media_type: str = "application/pdf",
               name: str | None = None) -> FileBlock:
    b: FileBlock = {"type": "file", "media_type": media_type}
    if url:
        b["url"] = url
    else:
        b["data"] = data or ""
    if name:
        b["name"] = name
    return b


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
    """The one error shape: any ``"error"`` key in event payloads or
    RunResult holds this, serialized. Model-facing channels (ToolResult
    content) stay plain prose strings; consumers parse this."""

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
    """One turn in the conversation, generic across providers. Assistant
    messages may carry ``tool_calls``; tool messages carry exactly one
    ``tool_result``; ``content`` is a string or a list of Blocks
    (multimodal — build with text_block / image_block / file_block)."""

    role: Role
    content: str | list[Block] | list[dict[str, Any]] | None = None
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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(role=d["role"], content=d.get("content"),
                   tool_calls=[ToolCall(**c) for c in d.get("tool_calls") or []],
                   tool_result=ToolResult(**d["tool_result"])
                   if d.get("tool_result") else None,
                   meta=d.get("meta") or {})


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
