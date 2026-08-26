"""The three data nouns: Part, ToolCall, Message.

Plain dataclasses with no methods. To checkpoint one, use
dataclasses.asdict; to load it back, call the class with the dict.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Part:
    """One piece of a message. `type` names the piece.

    Core never opens `data`; model adapters do. So adapters agree, the two
    shared spellings are "text" (data: the str) and "image" (data: {"url": …}
    or {"media_type": …, "data": base64 str}). Other types are adapter-defined.
    """

    type: str
    data: Any = None


@dataclass
class ToolCall:
    """The model asking for one tool to run."""

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """One line in the agent's notebook."""

    role: Role
    content: str | list[Part] = ""
    tool_calls: list[ToolCall] = field(default_factory=list)   # assistant only
    tool_call_id: str | None = None                            # tool role only
    meta: dict[str, Any] = field(default_factory=dict)         # provider junk;
    # adapters put usage here as {"input_tokens": …, "output_tokens": …}
