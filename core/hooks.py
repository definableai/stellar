"""The hook layer. Exactly four points, mutation semantics.

    before_llm   sees (messages, tools) — may edit/extend them
    after_llm    additionally sees ``reply`` — may edit it
    before_tool  sees the ToolCall — may edit arguments, or set
                 ``ctx.result`` to SHORT-CIRCUIT the tool (it will not
                 execute). This is your permission/HITL gate.
    after_tool   sees ``ctx.result`` — may replace/redact it

Rules:
    * Hooks mutate their context in place; return value is ignored.
    * Hooks may be sync or async.
    * A hook raising ABORTS the run (fail closed — a guardrail that
      fails must not fail silently). Wrap best-effort hooks yourself.
    * Every hook invocation is itself a step: hook/start + hook/end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Mapping

from .tools import ToolSpec
from .types import Message, ToolCall, ToolResult

if TYPE_CHECKING:
    from .run import RunContext

HookPoint = Literal["before_llm", "after_llm", "before_tool", "after_tool"]
HOOK_POINTS: tuple[HookPoint, ...] = (
    "before_llm",
    "after_llm",
    "before_tool",
    "after_tool",
)


@dataclass
class LLMHookContext:
    """Context for before_llm / after_llm."""

    run: "RunContext"
    messages: list[Message]        # mutable — edit to shape what the LLM sees
    tools: list[ToolSpec]          # mutable — edit to shape what the LLM may call
    reply: Message | None = None   # set on after_llm — mutable


@dataclass
class ToolHookContext:
    """Context for before_tool / after_tool."""

    run: "RunContext"
    call: ToolCall                 # mutable — edit arguments before execution
    result: ToolResult | None = None
    # before_tool: setting ``result`` skips execution (short-circuit).
    # after_tool:  ``result`` holds the outcome — replace to override.


HookFn = Callable[[Any], Any]  # (LLMHookContext | ToolHookContext) -> None


class Hooks:
    """Ordered registry of hook functions per point."""

    def __init__(
        self, initial: Mapping[HookPoint, Iterable[HookFn]] | None = None
    ) -> None:
        self._hooks: dict[HookPoint, list[HookFn]] = {p: [] for p in HOOK_POINTS}
        if initial:
            for point, fns in initial.items():
                for fn in fns:
                    self.add(point, fn)

    def add(self, point: HookPoint, fn: HookFn) -> HookFn:
        if point not in self._hooks:
            raise ValueError(f"Unknown hook point: {point!r}. Use one of {HOOK_POINTS}.")
        self._hooks[point].append(fn)
        return fn

    def on(self, point: HookPoint) -> Callable[[HookFn], HookFn]:
        """Decorator form:  @hooks.on("before_tool")"""
        return lambda fn: self.add(point, fn)

    def get(self, point: HookPoint) -> list[HookFn]:
        return self._hooks[point]
