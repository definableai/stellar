"""The hook layer. Exactly four points, mutation semantics.
(See the ``@hook`` decorator below for the attach recipe.)

    before_llm   sees (messages, tools) — may edit/extend them
    after_llm    additionally sees ``reply`` — may edit it
    before_tool  sees the ToolCall — may edit arguments, or set
                 ``ctx.result`` to SHORT-CIRCUIT the tool (it will not
                 execute). This is your permission/HITL gate.
    after_tool   sees ``ctx.result`` — may replace/redact it

Rules:
    * Hooks mutate their context in place (return ignored); sync or async.
    * A hook raising ABORTS the run (fail closed — a guardrail that
      fails must not fail silently). Wrap best-effort hooks yourself.
    * Every hook invocation is itself a step: hook/start + hook/end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable, Literal, Mapping

from .tools import ToolSpec
from .types import Message, ToolCall, ToolResult

if TYPE_CHECKING:
    from .run import RunContext

HookPoint = Literal["before_llm", "after_llm", "before_tool", "after_tool"]
HOOK_POINTS: tuple[HookPoint, ...] = (
    "before_llm", "after_llm", "before_tool", "after_tool")


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
    emit_delta: Callable[[dict[str, Any]], Awaitable[Any]] | None = None  # -> tool/delta
    # before_tool: setting ``result`` skips execution (short-circuit).
    # after_tool:  ``result`` holds the outcome — replace to override.


HookFn = Callable[["LLMHookContext | ToolHookContext"], Any]  # return ignored


@dataclass
class Hook:
    """A hook function bound to its point — what ``@hook(point)`` makes.
    Attach a flat list: ``Agent(hooks=[redact, compaction(...), gate])``."""

    point: HookPoint
    fn: HookFn


def hook(point: HookPoint) -> Callable[[HookFn], Hook]:
    """Decorator: bind a (ctx) -> None function to a hook point.

        @hook("before_llm")
        async def redact(ctx: LLMHookContext):
            ctx.messages[:] = scrub(ctx.messages)

        agent = Agent(llm, hooks=[redact])
    """
    if point not in HOOK_POINTS:
        raise ValueError(f"Unknown hook point: {point!r}. Use one of {HOOK_POINTS}.")
    return lambda fn: Hook(point, fn)


class Hooks:
    """Ordered registry, per point. Rarely hand-built: Agent(hooks=[...])
    takes a flat list of ``Hook``s (or a {point: [fns]} mapping)."""

    def __init__(
        self,
        initial: Iterable[Hook] | Mapping[HookPoint, Iterable[HookFn]] | None = None,
    ) -> None:
        self._hooks: dict[HookPoint, list[HookFn]] = {p: [] for p in HOOK_POINTS}
        if isinstance(initial, Mapping):
            for point, fns in initial.items():
                for fn in fns:
                    self.add(point, fn)
        elif initial:
            for h in initial:
                self.add(h.point, h.fn)

    def add(self, point: HookPoint, fn: HookFn | Hook) -> HookFn:
        if isinstance(fn, Hook):        # a Hook knows its own point
            return self.add(fn.point, fn.fn)
        if point not in self._hooks:
            raise ValueError(f"Unknown hook point: {point!r}. Use one of {HOOK_POINTS}.")
        self._hooks[point].append(fn)
        return fn

    def remove(self, point: HookPoint, fn: HookFn) -> None:
        """Detach ``fn`` from a point. ValueError if not attached (loud)."""
        self._hooks[point].remove(fn)

    def get(self, point: HookPoint) -> list[HookFn]:
        return self._hooks[point]
