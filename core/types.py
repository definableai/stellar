"""The three data nouns: Part, ToolCall, Message.

Plain dataclasses. dataclasses.asdict checkpoints one; loading it back is
yours, including the ToolCalls and Parts nested in a Message.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, cast

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Part:
    """One piece of a message. `type` names the piece.

    Three types are reserved words core itself folds into a Message: "text",
    "tool_call" and "meta". One more spelling is shared so adapters agree:
    "image" (data: {"url": …} or {"media_type": …, "data": base64 str}).
    Every other type is adapter-defined and passes through untouched.
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
    """One line in the agent's notebook.

    content is always list[Part]: pass a str for convenience and it becomes
    one text part at construction ("" becomes []). The .text property joins
    the text parts back — the branch nobody has to write anymore.
    """

    role: Role
    content: str | list[Part] = ""
    tool_calls: list[ToolCall] = field(default_factory=list)   # assistant only
    tool_call_id: str | None = None                            # tool role only
    meta: dict[str, Any] = field(default_factory=dict)         # provider junk;
    # adapters put usage here as {"input_tokens": …, "output_tokens": …}

    def __post_init__(self) -> None:
        if isinstance(self.content, str):
            self.content = [Part("text", self.content)] if self.content else []

    @property
    def text(self) -> str:
        """Every text part, joined."""
        parts = cast(list[Part], self.content)  # always parts after construction
        return "".join(p.data for p in parts if p.type == "text")
