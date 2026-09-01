"""The agent: one bag holding everything, and one method to run it."""

from dataclasses import dataclass, field

from core.contracts import Hook, Model, Tool
from core.loop import check, run
from core.types import Message, Part, ToolCall


@dataclass
class Agent:
    """A robot with one backpack. Swap any part by assignment, even mid-run."""

    # harness — what it brought.
    model: Model
    tools: list[Tool] = field(default_factory=list)
    hooks: list[Hook] = field(default_factory=list)
    # memory — what happened. This is what you checkpoint.
    messages: list[Message] = field(default_factory=list)
    step: int = 0
    # in flight — hooks read these and may replace them.
    response: Message | None = None
    call: ToolCall | None = None
    result: Message | None = None
    delta: Part | None = None            # the Part a delta bell is about
    # the spare pocket. Namespaced keys: extra["budget.tokens"].
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        check(self)

    async def run(self, input: str | Message | None = None) -> Message | None:
        """Ask, act, repeat. The only method; the work is in core/loop.py.

        A str becomes a user message; a Message is appended as-is; None runs
        on the messages already there. Gives back the last message.
        """
        if isinstance(input, str):
            input = Message("user", input)
        if input is not None:
            self.messages.append(input)
        await run(self)                      # the loop's run, not this one
        return self.messages[-1] if self.messages else None
