"""The three data nouns: Part, ToolCall, Message.

Plain dataclasses. dataclasses.asdict checkpoints one; loading it back is
yours, including the ToolCalls and Parts nested in a Message.

The floor of the stack: nothing here imports the rest of core, and every
other module — loop, hooks, adapters, drivers — speaks in these three.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, cast

__all__ = ["Message", "Part", "ToolCall"]

Role = Literal["system", "user", "assistant", "tool"]   # who a Message is from


@dataclass
class Part:
    """One piece of a message. `type` names the piece.

    Three types are reserved words core itself folds into a Message: "text",
    "tool_call" and "meta". One more spelling is shared so adapters agree:
    "image" (data: {"url": …} or {"media_type": …, "data": base64 str}).
    Every other type is adapter-defined and passes through untouched.
    """

    type: Annotated[str, "which piece this is — the word fold() and adapters read"]
    data: Annotated[Any, "the payload; its shape is whatever `type` promises"] = None


@dataclass
class ToolCall:
    """The model asking for one tool to run."""

    id: Annotated[str, "the provider's id for this ask; the tool message answers it"]
    name: Annotated[str, "which tool to run — the key it is filed under"]
    args: Annotated[
        dict[str, Any],
        "the call's keyword arguments, already parsed"] = field(default_factory=dict)


@dataclass
class Message:
    """One line in the agent's notebook.

    content is always list[Part]: pass a str for convenience and it becomes
    one text part at construction ("" becomes []). The .text property joins
    the text parts back — the branch nobody has to write anymore.
    """

    role: Annotated[Role, "who the line is from — system, user, assistant or tool"]
    content: Annotated[
        str | list[Part], "the parts; a str becomes one text part at construction"] = ""
    tool_calls: Annotated[
        list[ToolCall], "assistant only"] = field(default_factory=list)
    tool_call_id: Annotated[str | None, "tool role only"] = None
    meta: Annotated[
        dict[str, Any],
        "provider junk; adapters put usage as {'input_tokens': …, 'output_tokens': …}",
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.content, str):
            self.content = [Part("text", self.content)] if self.content else []

    @property
    def text(self) -> str:
        """Every text part, joined."""
        parts = cast(list[Part], self.content)  # always parts after construction
        return "".join(p.data for p in parts if p.type == "text")
