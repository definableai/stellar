"""The agent: one bag holding everything, and one method to run it."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from core.contracts import Hook, Model, Tool
from core.loop import check, run
from core.types import Message, ToolCall


@dataclass
class Agent:
    """A robot with one backpack. Swap any part by assignment, even mid-run."""

    # harness — what it brought.
    model: Model
    tools: Mapping[str, Tool] = field(default_factory=dict)
    hooks: list[Hook] = field(default_factory=list)
    # memory — what happened. This is what you checkpoint.
    messages: list[Message] = field(default_factory=list)
    step: int = 0
    # in flight — hooks read these and may replace them.
    response: Message | None = None
    call: ToolCall | None = None
    result: Message | None = None
    # the spare pocket. Namespaced keys: extra["budget.tokens"].
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        check(self)

    async def run(self) -> "Agent":
        """Ask, act, repeat. The only method; the work is in core/loop.py."""
        return await run(self)               # the loop's run, not this one
